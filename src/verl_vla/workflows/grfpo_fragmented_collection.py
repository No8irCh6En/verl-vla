# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0

"""Independent one-GPU candidate collection for Group-Relative FPO."""

from __future__ import annotations

import fcntl
import hashlib
import json
import multiprocessing as mp
import os
import time
import traceback
from multiprocessing.connection import wait as wait_for_connections
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from verl import DataProto

from verl_vla.envs.action_executor import ActionExecutionConfig
from verl_vla.envs.robodojo.config import RoboDojoSimulatorConfig
from verl_vla.recorder.config import RecorderConfig
from verl_vla.teleop.config import TeleopConfig
from verl_vla.trainer.grfpo.collection_store import (
    CandidateLease,
    CollectionWindowSpec,
    CollectionWindowStore,
    PolicyIdentity,
    candidate_assignment,
)
from verl_vla.trainer.grfpo.group_validation import (
    RolloutStateIntegrityError,
    inspect_rollout_state_integrity,
    validate_same_condition_group,
)
from verl_vla.trainer.grfpo.rollout_batch import concatenate_rollout_groups_with_padding
from verl_vla.utils.data import stack_dataproto_with_padding
from verl_vla.workers.env.config import EnvWorkerConfig, SimulatorConfig

from .native_fastwam_robodojo_eval import (
    _ipc_observation,
    infer_shared_model_outputs_in_microbatches,
    load_shared_fastwam_model,
    merge_ipc_observations,
    robodojo_observation_batch,
)


def _configured_task_names(config) -> list[str]:
    task_names = [str(task) for task in config.task_names]
    configured_task_set = set(task_names)
    included_text = str(config.get("included_task_names", ""))
    included = [
        value.strip()
        for value in included_text.replace(",", ":").split(":")
        if value.strip()
    ]
    unknown_included = set(included).difference(task_names)
    if unknown_included:
        raise ValueError(f"Unknown included_task_names: {sorted(unknown_included)}")
    if included:
        included_set = set(included)
        task_names = [task for task in task_names if task in included_set]
    excluded = {
        value.strip()
        for value in str(config.get("excluded_task_names", "")).replace(",", ":").split(":")
        if value.strip()
    }
    # Validate exclusions against the canonical configured task vocabulary,
    # not the already narrowed include list.  Resume jobs persist the selected
    # tasks in their manifest and may also inherit the original exclusion; the
    # two filters should therefore be idempotent rather than rejecting a task
    # that was already removed by ``included_task_names``.
    unknown = excluded.difference(configured_task_set)
    if unknown:
        raise ValueError(f"Unknown excluded_task_names: {sorted(unknown)}")
    selected = [task for task in task_names if task not in excluded]
    if not selected:
        raise ValueError("excluded_task_names removed every configured task.")
    return selected


def collection_window_spec(config) -> CollectionWindowSpec:
    selected_tasks = _configured_task_names(config)
    return CollectionWindowSpec(
        run_id=str(config.run_id),
        intended_update=int(config.intended_update),
        group_size=int(config.group_size),
        target_accepted_groups=int(config.target_accepted_groups),
        minimum_accepted_groups=int(config.minimum_accepted_groups),
        max_candidate_groups=int(config.max_candidate_groups),
        complete_round_after_target=bool(config.complete_round_after_target),
        candidate_round_size=(
            int(config.collector_count) if bool(config.complete_round_after_target) else 0
        ),
        drain_candidate_budget=bool(config.drain_candidate_budget),
        task_names=tuple(selected_tasks),
        suite_ids=tuple(int(suite) for suite in config.suite_ids),
        condition_manifest_sha256=str(config.condition_manifest_sha256),
        group_reward_source=str(config.group_reward_source),
        informative_group_criterion=str(config.informative_group_criterion),
        accepted_group_reduction=str(config.accepted_group_reduction),
        policy_seed_namespace=str(config.get("policy_seed_namespace", "")),
        raw_state_abs_limit=float(config.get("raw_state_abs_limit", 100.0)),
        actor_objective=str(config.get("actor_objective", "fpo")),
        flow_grpo_noise_level=float(config.get("flow_grpo_noise_level", 0.01)),
        policy=PolicyIdentity(
            immutable_base_path=str(Path(str(config.model_root)).expanduser().resolve()),
            immutable_base_sha256=str(config.checkpoint_sha256),
            trainer_checkpoint_path=(
                None
                if config.verl_actor_checkpoint in (None, "")
                else str(Path(str(config.verl_actor_checkpoint)).expanduser().resolve())
            ),
            rollout_policy_version=int(config.rollout_policy_version),
        ),
    )


def _feedback_batch(
    rewards,
    terminations,
    truncations,
    successes,
    extras: list[Any],
) -> DataProto:
    tensors = {
        "reward": torch.as_tensor(rewards).cpu(),
        "terminated": torch.as_tensor(terminations).cpu(),
        "truncated": torch.as_tensor(truncations).cpu(),
        "success": torch.as_tensor(successes).cpu(),
    }
    if extras and isinstance(extras[0], dict) and "score" in extras[0]:
        tensors["score"] = torch.as_tensor(extras[0]["score"], dtype=torch.float32).cpu()
    return DataProto.from_dict(tensors=tensors)


def collate_candidate_trajectory(
    slots: list[tuple[DataProto, DataProto, DataProto]],
    final_obs: DataProto,
) -> tuple[DataProto, DataProto]:
    """Collate B=G fixed-lane slots into the same contract as ``EnvLoop``."""

    if not slots:
        raise ValueError("A candidate group must contain at least one policy interaction.")
    batch_dict: dict[str, Any] = {}
    for prefix, index in (("obs", 0), ("action", 1), ("next", 2)):
        batch_dict.update(stack_dataproto_with_padding([slot[index] for slot in slots], prefix))
    rollout = DataProto.from_single_dict(batch_dict, meta_info={})
    return rollout, final_obs


def _candidate_schedule(config, spec: CollectionWindowSpec):
    task_names = _configured_task_names(config)
    suite_ids = [int(suite) for suite in config.suite_ids]
    manifest_path = Path(str(config.condition_manifest)).expanduser().resolve()
    manifest_bytes = manifest_path.read_bytes()
    actual_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if actual_sha256 != str(config.condition_manifest_sha256):
        raise RuntimeError(
            f"Condition manifest SHA-256 mismatch for {manifest_path}: "
            f"expected={config.condition_manifest_sha256}, actual={actual_sha256}."
        )
    manifest = json.loads(manifest_bytes)
    training_layouts: dict[tuple[str, int], list[int]] = {}
    for task_name in task_names:
        task = manifest["tasks"].get(task_name)
        if task is None:
            raise KeyError(f"Condition manifest does not define task {task_name!r}.")
        for suite_id in suite_ids:
            suite = task["suites"].get(str(suite_id))
            if suite is None:
                raise KeyError(f"Condition manifest does not define ({task_name!r}, suite={suite_id}).")
            train_layouts = [int(layout) for layout in suite["train_layout_ids"]]
            heldout_layouts = {int(case["layout_id"]) for case in suite["heldout_test"]}
            overlap = sorted(set(train_layouts) & heldout_layouts)
            if overlap:
                raise RuntimeError(
                    f"Condition manifest leaks held-out layouts into training for "
                    f"({task_name!r}, suite={suite_id}): {overlap}."
                )
            training_layouts[(task_name, suite_id)] = train_layouts
    return [
        candidate_assignment(
            spec=spec,
            candidate_index=candidate_index,
            task_names=task_names,
            suite_ids=suite_ids,
            training_layouts=training_layouts,
            run_seed=int(config.run_seed),
        )
        for candidate_index in range(spec.max_candidate_groups)
    ]


def _collector_assignments(config, spec: CollectionWindowSpec):
    """Return the legacy fixed-lane slice of the predeclared schedule."""

    task_names = _configured_task_names(config)
    suite_ids = [int(suite) for suite in config.suite_ids]
    collector_count = int(config.collector_count)
    collector_slot = int(config.collector_slot)
    expected_collectors = len(task_names) * len(suite_ids)
    if collector_count != expected_collectors:
        raise ValueError(
            "Persistent fragmented collection requires one collector lane per (task,suite): "
            f"collector_count={collector_count}, expected={expected_collectors}."
        )
    if not 0 <= collector_slot < collector_count:
        raise ValueError(f"collector_slot must be in [0,{collector_count}), got {collector_slot}.")
    return [
        assignment
        for assignment in _candidate_schedule(config, spec)
        if assignment.candidate_index % collector_count == collector_slot
    ]


def _collect_one_group(
    *,
    config,
    model,
    env,
    assignment,
    spec,
    store,
    reset_eval: bool,
    lease: CandidateLease | None = None,
    infer_outputs=None,
):
    torch.cuda.set_device(int(config.simulator_device_id))
    group_start = time.perf_counter()
    last_lease_heartbeat = group_start

    def renew_lease_if_due() -> None:
        nonlocal lease, last_lease_heartbeat
        now = time.perf_counter()
        if lease is None or now - last_lease_heartbeat < float(config.candidate_lease_heartbeat_seconds):
            return
        lease = store.renew_candidate_lease(
            lease,
            lease_timeout_seconds=float(config.candidate_lease_timeout_seconds),
        )
        last_lease_heartbeat = now

    physical_batch_size = int(env.num_envs)
    if physical_batch_size <= 0 or spec.group_size % physical_batch_size != 0:
        raise ValueError(
            "Physical candidate batch must tile the complete GRPO group: "
            f"physical_batch_size={physical_batch_size}, group_size={spec.group_size}."
        )

    rollout_waves: list[DataProto] = []
    rollout_end_waves: list[DataProto] = []
    wave_policy_calls: list[int] = []
    for wave_start in range(0, spec.group_size, physical_batch_size):
        wave_stop = wave_start + physical_batch_size
        env.set_runtime_eval_cases(
            layout_ids=[assignment.layout_id] * physical_batch_size,
            policy_seeds=list(assignment.policy_seeds[wave_start:wave_stop]),
        )
        if infer_outputs is None:
            if model is None:
                raise ValueError("Local candidate collection requires a model instance.")
            model.reset()
        obs, _ = env.reset(
            options={
                # Runtime cases are replaced for every physical wave, so each
                # wave begins from its first explicitly assigned case.
                "reset_eval": True,
                "extra": {"mode": "eval", "progress_callback": renew_lease_if_due},
            }
        )
        # Keep provenance unique across waves even though the native evaluator
        # restarts its local case cursor at zero for each runtime schedule.
        episode_ids = np.arange(wave_start, wave_stop, dtype=np.int64)
        if "eval_episode_id" in obs:
            obs["eval_episode_id"] = episode_ids.copy()
        if hasattr(env, "_episode_ids"):
            env._episode_ids[:] = episode_ids

        slots: list[tuple[DataProto, DataProto, DataProto]] = []
        done = torch.zeros(physical_batch_size, dtype=torch.bool)
        for _ in range(int(config.max_policy_calls)):
            renew_lease_if_due()
            observation = robodojo_observation_batch(obs)
            if infer_outputs is None:
                actions = infer_shared_model_outputs_in_microbatches(
                    model,
                    obs,
                    int(config.policy_microbatch_size),
                    model_device_id=int(config.model_device_id),
                    eval=spec.actor_objective != "flow_grpo",
                )
            else:
                actions = infer_outputs(obs)
            torch.cuda.set_device(int(config.simulator_device_id))
            next_obs, rewards, terminated, truncated, successes, *extras = env.step(actions.batch["action"])
            feedback = _feedback_batch(rewards, terminated, truncated, successes, extras)
            slots.append((observation, actions, feedback))
            done |= (
                (feedback.batch["terminated"].bool() | feedback.batch["truncated"].bool())
                .flatten(start_dim=1)
                .any(dim=1)
            )
            obs = next_obs
            if bool(done.all()):
                break

        wave_rollout, wave_end_obs = collate_candidate_trajectory(
            slots, robodojo_observation_batch(obs)
        )
        rollout_waves.append(wave_rollout)
        rollout_end_waves.append(wave_end_obs)
        wave_policy_calls.append(len(slots))

    if len(rollout_waves) == 1:
        rollout, rollout_end_obs = rollout_waves[0], rollout_end_waves[0]
    else:
        rollout, rollout_end_obs = concatenate_rollout_groups_with_padding(
            rollout_waves, rollout_end_waves
        )
    state_integrity = inspect_rollout_state_integrity(
        rollout,
        rollout_end_obs,
        raw_state_abs_limit=spec.raw_state_abs_limit,
    )
    if not state_integrity["valid"]:
        failure_path = store.record_integrity_failure(
            assignment=assignment,
            diagnostics=state_integrity,
            lease=lease,
        )
        print(
            json.dumps(
                {
                    "status": "candidate_state_integrity_failure",
                    "candidate_id": assignment.candidate_id,
                    "diagnostics": state_integrity,
                    "failure_record": str(failure_path) if failure_path is not None else None,
                    "action": "release_lease_and_restart_simulator",
                },
                sort_keys=True,
                allow_nan=False,
            ),
            flush=True,
        )
        raise RolloutStateIntegrityError(state_integrity)
    if spec.actor_objective == "flow_grpo":
        required_trace = {
            "action.flow_grpo.latents",
            "action.flow_grpo.old_log_probs",
            "action.flow_grpo.sigmas",
            "action.flow_grpo.deltas",
        }
        missing_trace = sorted(required_trace.difference(rollout.batch.keys()))
        if missing_trace:
            raise KeyError(
                "Flow-GRPO candidate is missing behavior-policy SDE trace fields: "
                f"{missing_trace}"
            )
    group_record = validate_same_condition_group(
        rollout,
        group_size=spec.group_size,
        group_id=assignment.candidate_id,
        rollout_policy_version=spec.policy.rollout_policy_version,
        reward_source=spec.group_reward_source,
        informative_group_criterion=spec.informative_group_criterion,
        min_partial_trajectories_for_score_only_group=int(config.min_partial_trajectories_for_score_only_group),
        partial_score_threshold=float(config.partial_score_threshold),
        expected_suite_id=assignment.suite_id,
        allowed_layout_ids={assignment.layout_id},
    )
    group_record.update(
        {
            "candidate_attempt": assignment.candidate_index,
            "intended_update": spec.intended_update,
            "task_name": assignment.task_name,
            "suite_id": assignment.suite_id,
            "layout_id": assignment.layout_id,
            "collection_wall_s": time.perf_counter() - group_start,
            "policy_calls": max(wave_policy_calls),
            "physical_rollout_batch_size": physical_batch_size,
            "physical_rollout_waves": len(rollout_waves),
            "state_integrity": state_integrity,
            "actor_objective": spec.actor_objective,
            "flow_grpo_noise_level": (
                spec.flow_grpo_noise_level if spec.actor_objective == "flow_grpo" else None
            ),
        }
    )
    commit_result = store.commit_group(
        assignment=assignment,
        rollout=rollout,
        rollout_end_obs=rollout_end_obs,
        group_record=group_record,
        lease=lease,
    )
    closed = store.close_if_ready()
    result = {
        "status": commit_result,
        "candidate_id": assignment.candidate_id,
        "group_record": group_record,
        "window": {
            "accepted": len(store.scan()["accepted"]),
            "target": spec.target_accepted_groups,
            "closed": closed,
        },
    }
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return result


def _candidate_physical_batch_size(config, spec: CollectionWindowSpec, assignment) -> int:
    """Resolve an optional per-candidate simulator-width recovery override."""

    physical_batch_size = int(config.get("physical_rollout_batch_size", 0))
    override_path = (
        CollectionWindowStore(config.collection_root, spec).window_dir
        / "PHYSICAL_BATCH_OVERRIDES.json"
    )
    if physical_batch_size == 0 and override_path.is_file():
        overrides = json.loads(override_path.read_text(encoding="utf-8"))
        physical_batch_size = int(overrides.get("candidates", {}).get(assignment.candidate_id, 0))
    if physical_batch_size == 0:
        physical_batch_size = spec.group_size
    if physical_batch_size <= 0 or spec.group_size % physical_batch_size != 0:
        raise ValueError(
            "physical_rollout_batch_size must be a positive divisor of group_size: "
            f"physical_rollout_batch_size={physical_batch_size}, group_size={spec.group_size}."
        )
    return physical_batch_size


def _build_candidate_env(
    config,
    spec: CollectionWindowSpec,
    assignment,
    *,
    rank: int = 0,
    world_size: int = 1,
):
    """Build one persistent simulator for a single task/suite condition."""

    physical_batch_size = _candidate_physical_batch_size(config, spec, assignment)

    simulator_cfg = RoboDojoSimulatorConfig(
        robodojo_root=str(config.robodojo_root),
        task_name=assignment.task_name,
        task_id=assignment.task_id,
        seed=assignment.suite_id,
        env_cfg_type=str(config.env_cfg_type),
        device_id=int(config.simulator_device_id),
        headless=bool(config.headless),
        layout_ids=[assignment.layout_id] * physical_batch_size,
        eval_policy_seed_schedule=list(assignment.policy_seeds[:physical_batch_size]),
        # The frozen manifest, not the legacy 6-9 convention, defines which
        # layouts are held out independently for every task/suite.
        allow_held_out_eval_layouts=True,
        eval_num=spec.group_size,
        reset_max_attempts=int(config.reset_max_attempts),
        defer_chunk_observations=False,
    )
    env_cfg = EnvWorkerConfig(
        auto_reset=False,
        action_execution=ActionExecutionConfig(mode="serial"),
        modes=["eval"],
        num_envs=physical_batch_size,
        simulator=SimulatorConfig(simulator_type="robodojo", robodojo=simulator_cfg),
        teleop=TeleopConfig(enable=False),
        recorder=RecorderConfig(enable=False),
        device="cuda",
    )

    # Isaac must own the CUDA application before Fast-WAM is imported/loaded.
    from verl_vla.envs.robodojo.robodojo_env import RoboDojoEnv

    return _construct_robodojo_env_with_node_startup_lock(
        RoboDojoEnv,
        env_cfg,
        rank=rank,
        world_size=world_size,
    )


def _construct_robodojo_env_with_node_startup_lock(
    env_class,
    env_cfg,
    *,
    rank: int = 0,
    world_size: int = 1,
):
    """Serialize only fragile Isaac renderer startup on each physical node.

    Independent Slurm jobs can be placed on different GPUs of one DGX. Their
    imports start at different times but frequently converge at ``Simulation
    App Starting``, where concurrent Vulkan initialization has produced
    repeatable ``ERROR_DEVICE_LOST`` failures. A node-local advisory lock avoids
    that collision; it is released as soon as the constructor returns, so all
    environment stepping and Fast-WAM rollout remain fully parallel.
    """

    lock_path = Path(os.environ.get("GRFPO_ISAAC_STARTUP_LOCK", "/tmp/verl-vla-isaac-startup.lock"))
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    wait_started = time.monotonic()
    with lock_path.open("a+", encoding="utf-8") as startup_lock:
        fcntl.flock(startup_lock.fileno(), fcntl.LOCK_EX)
        wait_s = time.monotonic() - wait_started
        print(
            json.dumps(
                {
                    "status": "isaac_node_startup_lock_acquired",
                    "hostname": os.uname().nodename,
                    "wait_s": wait_s,
                    "lock_path": str(lock_path),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return env_class(
            OmegaConf.structured(env_cfg),
            rank=rank,
            world_size=world_size,
            only_eval=True,
        )


def _select_candidate_on_existing_env(env, assignment) -> None:
    """Install the next layout/seeds without rebuilding a same-task simulator."""

    physical_batch_size = int(env.num_envs)
    env.set_runtime_eval_cases(
        layout_ids=[assignment.layout_id] * physical_batch_size,
        policy_seeds=list(assignment.policy_seeds[:physical_batch_size]),
    )


def _collector_owner_id(config) -> str:
    array_job = os.environ.get("SLURM_ARRAY_JOB_ID", os.environ.get("SLURM_JOB_ID"))
    array_task = os.environ.get("SLURM_ARRAY_TASK_ID", str(config.collector_slot))
    if array_job is None:
        return f"local:{array_task}:{os.getpid()}"
    return f"{array_job}:{array_task}"


def _remote_action_dataproto(payload: dict[str, Any]) -> DataProto:
    """Reconstruct one canonical action batch returned by the model broker."""

    tensors = {key: torch.as_tensor(np.asarray(value)) for key, value in payload.items()}
    required = {"action", "full_action"}
    missing = required.difference(tensors)
    if missing:
        raise KeyError(f"Shared-model response is missing tensors: {sorted(missing)}")
    return DataProto.from_dict(tensors=tensors)


def _claim_shared_worker_candidate(
    *,
    config,
    spec: CollectionWindowSpec,
    store: CollectionWindowStore,
    assignments,
    preferred_condition: tuple[str, int] | None,
    owner_id: str,
) -> CandidateLease | None:
    """Claim a candidate, waiting only for steal-any work that is in flight."""

    lease_timeout_seconds = float(config.candidate_lease_timeout_seconds)
    lease = store.claim_candidate(
        assignments=assignments,
        owner_id=owner_id,
        lease_timeout_seconds=lease_timeout_seconds,
        preferred_condition=preferred_condition,
    )
    if lease is not None or preferred_condition is not None:
        return lease
    idle_log_at = time.monotonic()
    while lease is None:
        closed = store.close_if_ready()
        state = store.scan()
        if closed is not None or state["closed"]:
            return None
        lease = store.claim_candidate(
            assignments=assignments,
            owner_id=owner_id,
            lease_timeout_seconds=lease_timeout_seconds,
        )
        if lease is not None:
            return lease
        now = time.monotonic()
        if now - idle_log_at >= 60:
            print(
                json.dumps(
                    {
                        "status": "shared_simulator_queue_wait",
                        "completed_candidate_groups": state["completed_candidate_count"],
                        "max_candidate_groups": spec.max_candidate_groups,
                        "owner_id": owner_id,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            idle_log_at = now
        time.sleep(float(config.shared_queue_idle_poll_seconds))
    return lease


def _shared_model_simulator_worker(
    worker_id: int,
    worker_count: int,
    config_values: dict[str, Any],
    model_connection,
) -> None:
    """Own one vec8 RoboDojo process while the parent owns Fast-WAM.

    Every committed artifact remains one complete same-condition G=8 group.
    Workers never combine group rewards or trajectories; only inference model
    parameters are shared in the parent process.
    """

    runtime_root = Path(os.environ.get("TMPDIR", "/tmp")) / f"shared-grfpo-sim-{worker_id}"
    runtime_root.mkdir(parents=True, exist_ok=True)
    for variable, suffix in (
        ("TMPDIR", "tmp"),
        ("XDG_CACHE_HOME", "cache"),
        ("XDG_CONFIG_HOME", "config"),
        ("XDG_DATA_HOME", "data"),
    ):
        path = runtime_root / suffix
        path.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(path)

    config = OmegaConf.create(config_values)
    spec = collection_window_spec(config)
    store = CollectionWindowStore(config.collection_root, spec)
    store.initialize()
    assignments = _candidate_schedule(config, spec)
    task_names = _configured_task_names(config)
    suite_ids = [int(suite) for suite in config.suite_ids]
    steal_any = bool(config.shared_queue_steal_any) or worker_id > 0
    preferred_condition = None
    if not steal_any:
        collector_slot = int(config.collector_slot)
        preferred_condition = (
            task_names[collector_slot // len(suite_ids)],
            suite_ids[collector_slot % len(suite_ids)],
        )
    # CollectionStore treats the same owner reappearing as a restarted worker
    # and may immediately reclaim its lease.  Simulator children must therefore
    # have distinct owners even though they share one Slurm collector process.
    owner_id = f"{_collector_owner_id(config)}:sim{worker_id}"
    lease: CandidateLease | None = None
    env = None
    groups_processed = 0
    request_index = 0
    try:
        lease = _claim_shared_worker_candidate(
            config=config,
            spec=spec,
            store=store,
            assignments=assignments,
            preferred_condition=preferred_condition,
            owner_id=owner_id,
        )
        if lease is None:
            model_connection.send(
                {
                    "kind": "done",
                    "worker_id": worker_id,
                    "result": {"status": "condition_drained", "groups_processed": 0},
                }
            )
            return
        condition = (lease.assignment.task_name, lease.assignment.suite_id)
        env = _build_candidate_env(
            config,
            spec,
            lease.assignment,
            rank=worker_id,
            world_size=worker_count,
        )

        def infer_outputs(obs) -> DataProto:
            nonlocal request_index
            request_id = request_index
            request_index += 1
            model_connection.send(
                {
                    "kind": "infer",
                    "worker_id": worker_id,
                    "request_id": request_id,
                    "observation": _ipc_observation(obs),
                }
            )
            if not model_connection.poll(600):
                raise TimeoutError(
                    f"Shared model did not answer worker={worker_id} request={request_id} "
                    "within 600 seconds."
                )
            response = model_connection.recv()
            if response.get("kind") == "error":
                raise RuntimeError(str(response["error"]))
            if response.get("kind") != "actions" or int(response["request_id"]) != request_id:
                raise RuntimeError(
                    f"Invalid shared-model response for worker={worker_id} request={request_id}: {response}"
                )
            return _remote_action_dataproto(response["tensors"])

        while lease is not None:
            assignment = lease.assignment
            if (assignment.task_name, assignment.suite_id) != condition:
                raise RuntimeError("A shared RoboDojo child cannot change task/suite while warm.")
            _select_candidate_on_existing_env(env, assignment)
            result = _collect_one_group(
                config=config,
                model=None,
                env=env,
                assignment=assignment,
                spec=spec,
                store=store,
                reset_eval=True,
                lease=lease,
                infer_outputs=infer_outputs,
            )
            groups_processed += 1
            lease = None
            if result["window"]["closed"] is not None:
                model_connection.send(
                    {
                        "kind": "done",
                        "worker_id": worker_id,
                        "result": {
                            "status": "window_closed",
                            "condition": list(condition),
                            "groups_processed": groups_processed,
                        },
                    }
                )
                return
            lease = store.claim_candidate(
                assignments=assignments,
                owner_id=owner_id,
                lease_timeout_seconds=float(config.candidate_lease_timeout_seconds),
                preferred_condition=condition,
            )
        model_connection.send(
            {
                "kind": "done",
                "worker_id": worker_id,
                "result": {
                    "status": "condition_drained",
                    "condition": list(condition),
                    "groups_processed": groups_processed,
                },
            }
        )
    except BaseException:
        if lease is not None:
            store.release_candidate_lease(lease)
        try:
            model_connection.send(
                {
                    "kind": "error",
                    "worker_id": worker_id,
                    "error": traceback.format_exc(),
                }
            )
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if env is not None:
            try:
                env.close()
            except BaseException:
                try:
                    model_connection.send(
                        {
                            "kind": "error",
                            "worker_id": worker_id,
                            "error": traceback.format_exc(),
                        }
                    )
                except (BrokenPipeError, EOFError, OSError):
                    pass
        model_connection.close()


def _shared_model_multiprocess_collector(
    config,
    spec: CollectionWindowSpec,
    store: CollectionWindowStore,
) -> dict[str, Any]:
    """Serve multiple independent vec8 simulators from one Fast-WAM replica."""

    simulator_count = int(config.simulators_per_gpu)
    if simulator_count < 2:
        raise ValueError("Shared-model collection requires simulators_per_gpu >= 2.")
    if str(config.rollout_seed_mode) != "episode":
        raise ValueError(
            "Shared-model collection currently requires rollout_seed_mode=episode so inference "
            "is stateless across interleaved candidate groups."
        )
    if int(config.physical_rollout_batch_size) not in {0, spec.group_size}:
        raise ValueError("Shared-model collection currently requires one complete vec8 wave per group.")

    context = mp.get_context("spawn")
    # A dedicated duplex Pipe per simulator exactly matches the protocol:
    # one outstanding inference request per child, followed by one response.
    # Pipes avoid POSIX named semaphores, which are fragile under Slurm/Isaac
    # process startup and caused spawn-time SemLock failures with mp.Queue.
    pipe_pairs = [context.Pipe(duplex=True) for _ in range(simulator_count)]
    parent_connections = [pair[0] for pair in pipe_pairs]
    child_connections = [pair[1] for pair in pipe_pairs]
    config_values = OmegaConf.to_container(config, resolve=True)
    workers = []
    startup_stagger_seconds = float(config.shared_simulator_startup_stagger_seconds)
    for worker_id in range(simulator_count):
        process = context.Process(
            target=_shared_model_simulator_worker,
            args=(
                worker_id,
                simulator_count,
                config_values,
                child_connections[worker_id],
            ),
            name=f"grfpo-shared-model-vec{spec.group_size}-sim-{worker_id}",
        )
        process.start()
        child_connections[worker_id].close()
        workers.append(process)
        if worker_id + 1 < simulator_count and startup_stagger_seconds > 0:
            time.sleep(startup_stagger_seconds)

    model = None
    results: list[dict[str, Any]] = []
    active = set(range(simulator_count))
    coalesce_seconds = float(config.shared_inference_coalesce_ms) / 1000.0

    def accept_terminal(message: dict[str, Any]) -> None:
        worker_id = int(message["worker_id"])
        if message["kind"] == "error":
            raise RuntimeError(
                f"Shared-model RoboDojo worker {worker_id} failed:\n{message['error']}"
            )
        results.append(dict(message["result"]))
        active.discard(worker_id)

    try:
        model, loaded_checkpoint = load_shared_fastwam_model(config)
        model.reset()
        print(
            json.dumps(
                {
                    "status": "shared_rollout_model_ready",
                    "model_replicas": 1,
                    "simulator_processes": simulator_count,
                    "group_size_per_simulator": spec.group_size,
                    "logical_concurrent_trajectories": simulator_count * spec.group_size,
                    "loaded_checkpoint": loaded_checkpoint,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        last_progress = time.monotonic()
        while active:
            active_connections = [parent_connections[index] for index in active]
            ready = wait_for_connections(active_connections, timeout=30)
            if not ready:
                failed = [
                    (index, process.exitcode)
                    for index, process in enumerate(workers)
                    if index in active and process.exitcode not in (None, 0)
                ]
                if failed:
                    raise RuntimeError(
                        f"Shared-model simulator process exited: {failed}"
                    ) from None
                if time.monotonic() - last_progress > 600:
                    raise TimeoutError(
                        "No progress from shared-model GRFPO simulators for 600 seconds."
                    ) from None
                continue
            last_progress = time.monotonic()
            pending_messages = [connection.recv() for connection in ready]
            inference_messages = []
            for message in pending_messages:
                if message["kind"] == "infer":
                    inference_messages.append(message)
                else:
                    accept_terminal(message)
            deadline = time.monotonic() + coalesce_seconds
            requested_workers = {
                int(message["worker_id"]) for message in inference_messages
            }
            while len(requested_workers) < len(active):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                eligible = [
                    parent_connections[index]
                    for index in active
                    if index not in requested_workers
                ]
                ready = wait_for_connections(eligible, timeout=remaining)
                if not ready:
                    break
                for connection in ready:
                    message = connection.recv()
                    if message["kind"] == "infer":
                        inference_messages.append(message)
                        requested_workers.add(int(message["worker_id"]))
                    else:
                        accept_terminal(message)
            if not inference_messages:
                continue
            merged, slices = merge_ipc_observations(
                [message["observation"] for message in inference_messages]
            )
            outputs = infer_shared_model_outputs_in_microbatches(
                model,
                merged,
                int(config.policy_microbatch_size),
                model_device_id=int(config.model_device_id),
                eval=spec.actor_objective != "flow_grpo",
            ).to("cpu")
            for message, output_slice in zip(inference_messages, slices, strict=True):
                tensors = {
                    key: value[output_slice].detach().cpu().numpy()
                    for key, value in outputs.batch.items()
                }
                worker_id = int(message["worker_id"])
                parent_connections[worker_id].send(
                    {
                        "kind": "actions",
                        "request_id": int(message["request_id"]),
                        "tensors": tensors,
                    }
                )
        status = (
            "window_closed"
            if any(result["status"] == "window_closed" for result in results)
            else "condition_drained"
        )
        return {
            "status": status,
            "model_replicas": 1,
            "simulator_processes": simulator_count,
            "groups_processed": sum(int(result.get("groups_processed", 0)) for result in results),
            "workers": results,
        }
    finally:
        del model
        for worker_id, connection in enumerate(parent_connections):
            try:
                if worker_id in active:
                    connection.send({"kind": "error", "error": "model broker stopped"})
            except (BrokenPipeError, EOFError, OSError):
                pass
            connection.close()
        for process in workers:
            process.join(timeout=60)
            if process.is_alive():
                process.terminate()
                process.join(timeout=30)


def _shared_queue_collector(config, spec: CollectionWindowSpec, store: CollectionWindowStore) -> dict[str, Any]:
    """Collect one condition at a time from an atomic cross-GPU queue.

    A process stays on one task/suite so the expensive Isaac environment and
    Fast-WAM replica remain warm. Once that condition has no claimable work,
    the Slurm supervisor starts a fresh process in steal-any mode; that is the
    only point at which a GPU changes task/suite.
    """

    assignments = _candidate_schedule(config, spec)
    collector_slot = int(config.collector_slot)
    collector_count = int(config.collector_count)
    if not 0 <= collector_slot < collector_count:
        raise ValueError(f"collector_slot must be in [0,{collector_count}), got {collector_slot}.")
    task_names = _configured_task_names(config)
    suite_ids = [int(suite) for suite in config.suite_ids]
    expected_collectors = len(task_names) * len(suite_ids)
    steal_any = bool(config.shared_queue_steal_any)
    if not steal_any and collector_count != expected_collectors:
        raise ValueError(
            "Shared queue warm-start requires one initial collector per (task,suite): "
            f"collector_count={collector_count}, expected={expected_collectors}."
        )

    preferred_condition = None
    if not steal_any:
        preferred_condition = (
            task_names[collector_slot // len(suite_ids)],
            suite_ids[collector_slot % len(suite_ids)],
        )
    owner_id = _collector_owner_id(config)
    lease_timeout_seconds = float(config.candidate_lease_timeout_seconds)

    lease = store.claim_candidate(
        assignments=assignments,
        owner_id=owner_id,
        lease_timeout_seconds=lease_timeout_seconds,
        preferred_condition=preferred_condition,
    )
    if lease is None:
        if preferred_condition is not None:
            return {
                "status": "condition_drained",
                "condition": list(preferred_condition),
                "groups_processed": 0,
            }
        idle_log_at = time.monotonic()
        while lease is None:
            closed = store.close_if_ready()
            state = store.scan()
            if closed is not None or state["closed"]:
                return {"status": "window_closed", "groups_processed": 0}
            lease = store.claim_candidate(
                assignments=assignments,
                owner_id=owner_id,
                lease_timeout_seconds=lease_timeout_seconds,
            )
            if lease is None:
                now = time.monotonic()
                if now - idle_log_at >= 60:
                    print(
                        json.dumps(
                            {
                                "status": "shared_queue_wait",
                                "completed_candidate_groups": state["completed_candidate_count"],
                                "max_candidate_groups": spec.max_candidate_groups,
                                "owner_id": owner_id,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    idle_log_at = now
                time.sleep(float(config.shared_queue_idle_poll_seconds))

    condition = (lease.assignment.task_name, lease.assignment.suite_id)
    env = _build_candidate_env(config, spec, lease.assignment)
    results = []
    try:
        model, _loaded_rl_checkpoint = load_shared_fastwam_model(config)
        while lease is not None:
            assignment = lease.assignment
            if (assignment.task_name, assignment.suite_id) != condition:
                raise RuntimeError("A warm RoboDojo process cannot change task/suite.")
            _select_candidate_on_existing_env(env, assignment)
            try:
                result = _collect_one_group(
                    config=config,
                    model=model,
                    env=env,
                    assignment=assignment,
                    spec=spec,
                    store=store,
                    reset_eval=True,
                    lease=lease,
                )
            except BaseException:
                store.release_candidate_lease(lease)
                raise
            results.append(result)
            if result["window"]["closed"] is not None:
                return {
                    "status": "window_closed",
                    "condition": list(condition),
                    "groups_processed": len(results),
                    "results": results,
                }
            lease = store.claim_candidate(
                assignments=assignments,
                owner_id=owner_id,
                lease_timeout_seconds=lease_timeout_seconds,
                preferred_condition=condition,
            )
        return {
            "status": "condition_drained",
            "condition": list(condition),
            "groups_processed": len(results),
            "results": results,
        }
    except BaseException:
        if lease is not None:
            store.release_candidate_lease(lease)
        print("collector_python_base_exception_before_close", flush=True)
        traceback.print_exc()
        raise
    finally:
        torch.cuda.set_device(int(config.simulator_device_id))
        try:
            env.close()
        except BaseException:
            print("collector_python_base_exception_during_close", flush=True)
            traceback.print_exc()
            raise


def run_grfpo_collector(config) -> dict[str, Any]:
    """Run one persistent task-specific collector, committing after every group."""

    spec = collection_window_spec(config)
    store = CollectionWindowStore(config.collection_root, spec)
    store.initialize()
    state = store.scan()
    if state["closed"]:
        return {"status": "window_closed"}
    if int(config.simulators_per_gpu) > 1:
        if not bool(config.shared_candidate_queue):
            raise ValueError("simulators_per_gpu > 1 requires the atomic shared candidate queue.")
        return _shared_model_multiprocess_collector(config, spec, store)
    if bool(config.shared_candidate_queue):
        return _shared_queue_collector(config, spec, store)
    assignments = [
        assignment
        for assignment in _collector_assignments(config, spec)
        if not store.candidate_complete(assignment.candidate_id)
    ]
    if not assignments:
        return {"status": "collector_complete"}
    conditions = {(assignment.task_name, assignment.suite_id) for assignment in assignments}
    if len(conditions) != 1:
        raise RuntimeError(
            f"One persistent RoboDojo collector cannot change task/suite: {sorted(conditions)}"
        )

    group_size = spec.group_size
    simulator_cfg = RoboDojoSimulatorConfig(
        robodojo_root=str(config.robodojo_root),
        task_name=assignments[0].task_name,
        task_id=assignments[0].task_id,
        seed=assignments[0].suite_id,
        env_cfg_type=str(config.env_cfg_type),
        device_id=int(config.simulator_device_id),
        headless=bool(config.headless),
        layout_ids=[assignment.layout_id for assignment in assignments for _ in range(group_size)],
        eval_policy_seed_schedule=[seed for assignment in assignments for seed in assignment.policy_seeds],
        # The frozen manifest, not the legacy 6-9 convention, defines which
        # layouts are held out independently for every task/suite.
        allow_held_out_eval_layouts=True,
        eval_num=len(assignments) * group_size,
        reset_max_attempts=int(config.reset_max_attempts),
        defer_chunk_observations=False,
    )
    env_cfg = EnvWorkerConfig(
        auto_reset=False,
        action_execution=ActionExecutionConfig(mode="serial"),
        modes=["eval"],
        num_envs=group_size,
        simulator=SimulatorConfig(simulator_type="robodojo", robodojo=simulator_cfg),
        teleop=TeleopConfig(enable=False),
        recorder=RecorderConfig(enable=False),
        device="cuda",
    )

    # Isaac must own the CUDA application before Fast-WAM is imported/loaded.
    from verl_vla.envs.robodojo.robodojo_env import RoboDojoEnv

    env = _construct_robodojo_env_with_node_startup_lock(RoboDojoEnv, env_cfg)
    try:
        model, _loaded_rl_checkpoint = load_shared_fastwam_model(config)
        results = []
        for assignment_index, assignment in enumerate(assignments):
            while not store.round_ready_to_start(assignment.candidate_index):
                if store.scan()["closed"]:
                    return {
                        "status": "window_closed_at_round_boundary",
                        "groups_processed": len(results),
                        "results": results,
                    }
                time.sleep(float(config.round_barrier_poll_seconds))
            if store.scan()["closed"]:
                break
            if store.candidate_complete(assignment.candidate_id):
                continue
            results.append(
                _collect_one_group(
                    config=config,
                    model=model,
                    env=env,
                    assignment=assignment,
                    spec=spec,
                    store=store,
                    reset_eval=assignment_index == 0,
                )
            )
            if results[-1]["window"]["closed"] is not None:
                break
        return {"status": "collector_finished", "groups_processed": len(results), "results": results}
    except BaseException:
        # Isaac/RoboDojo has a few paths that raise SystemExit(0), which would
        # otherwise look like a healthy process exit and hide the call site
        # from the shell supervisor.  Preserve the traceback before simulator
        # teardown; this is diagnostic only and does not alter retry/commit
        # semantics.
        print("collector_python_base_exception_before_close", flush=True)
        traceback.print_exc()
        raise
    finally:
        torch.cuda.set_device(int(config.simulator_device_id))
        try:
            env.close()
        except BaseException:
            print("collector_python_base_exception_during_close", flush=True)
            traceback.print_exc()
            raise


def run_grfpo_candidate(config) -> dict[str, Any]:
    """Compatibility name for the persistent fragmented collector workflow."""

    return run_grfpo_collector(config)


def run_grfpo_collection_status(config) -> dict[str, Any]:
    """Initialize/inspect/close a collection window without allocating a GPU."""

    store = CollectionWindowStore(config.collection_root, collection_window_spec(config))
    store.initialize()
    closed = store.close_if_ready()
    state = store.scan()
    result = {
        "window_dir": str(store.window_dir),
        "accepted_groups": len(state["accepted"]),
        "rejected_groups": len(state["rejected"]),
        "completed_candidate_groups": state["completed_candidate_count"],
        "closed": closed,
    }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)
    return result
