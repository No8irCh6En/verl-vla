# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Single-process, single-GPU native Fast-WAM evaluation in vector RoboDojo."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import queue
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from verl import DataProto

from verl_vla.envs.action_executor import ActionExecutionConfig
from verl_vla.envs.robodojo.config import RoboDojoSimulatorConfig
from verl_vla.models.fastwam.native_checkpoint import (
    EXPORT_MANIFEST_FILENAME,
    NATIVE_WEIGHTS_RELATIVE_PATH,
    load_verl_fastwam_actor_into_native_policy,
    sha256_file,
)
from verl_vla.recorder.config import RecorderConfig, VideoRecorderConfig
from verl_vla.teleop.config import TeleopConfig
from verl_vla.workers.env.config import EnvWorkerConfig, SimulatorConfig


def robodojo_observation_batch(obs: dict[str, Any], indices: range | None = None) -> DataProto:
    """Convert native RoboDojo observations to the canonical model batch."""

    if indices is None:
        indices = range(len(obs["observation"]))
    rows = [obs["observation"][index] for index in indices]
    tensors = {key: torch.as_tensor(np.stack([row[key] for row in rows])) for key in rows[0]}
    non_tensors = {"task": np.asarray([obs["task"][index] for index in indices], dtype=object)}
    for field in (
        "task_id",
        "suite_id",
        "eval_episode_id",
        "layout_id",
        "environment_seed",
        "policy_seed",
    ):
        if field in obs:
            non_tensors[field] = np.asarray([obs[field][index] for index in indices], dtype=np.int64)
    return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)


def infer_shared_model_in_microbatches(
    model,
    obs: dict[str, Any],
    microbatch_size: int,
    *,
    model_device_id: int = 0,
) -> torch.Tensor:
    """Run one shared Fast-WAM replica over a vector observation batch."""

    if torch.cuda.is_available():
        torch.cuda.set_device(model_device_id)
    batch_size = len(obs["observation"])
    chunks = []
    for start in range(0, batch_size, microbatch_size):
        stop = min(start + microbatch_size, batch_size)
        output = model.sac_sample_actions(robodojo_observation_batch(obs, range(start, stop)), eval=True)
        chunks.append(output.action.cpu())
    actions = torch.cat(chunks, dim=0)
    if actions.shape[:2] != (batch_size, 24):
        raise RuntimeError(f"Expected executed actions [B,24,14], got {tuple(actions.shape)}")
    return actions


def infer_shared_model_outputs_in_microbatches(
    model,
    obs: dict[str, Any],
    microbatch_size: int,
    *,
    model_device_id: int = 0,
    eval: bool = True,
) -> DataProto:
    """Return canonical action/trace data while keeping one model replica.

    Native evaluation uses ``eval=True``.  The fragmented Flow-GRPO collector
    passes ``eval=False`` so Fast-WAM records the stochastic behavior-policy
    SDE trace; this is independent of ``torch.nn.Module.eval()``.
    """

    if torch.cuda.is_available():
        torch.cuda.set_device(model_device_id)
    batch_size = len(obs["observation"])
    chunks = []
    for start in range(0, batch_size, microbatch_size):
        stop = min(start + microbatch_size, batch_size)
        output = model.sac_sample_actions(
            robodojo_observation_batch(obs, range(start, stop)), eval=eval
        ).to_data_proto()
        retained = ["action", "full_action"]
        retained.extend(
            key for key in output.batch.keys() if str(key).startswith("flow_grpo.")
        )
        output.batch = output.batch.select(*retained)
        chunks.append(output.to("cpu"))
    actions = DataProto.concat(chunks)
    if tuple(actions.batch["action"].shape) != (batch_size, 24, 14):
        raise RuntimeError(f"Expected executed actions [B,24,14], got {tuple(actions.batch['action'].shape)}")
    if tuple(actions.batch["full_action"].shape) != (batch_size, 32, 14):
        raise RuntimeError(f"Expected full actions [B,32,14], got {tuple(actions.batch['full_action'].shape)}")
    return actions


def terminal_or_latest_scores(
    score_steps: torch.Tensor,
    terminated_steps: torch.Tensor,
    truncated_steps: torch.Tensor,
    already_done: np.ndarray,
) -> np.ndarray:
    """Return each active episode's terminal score, or its latest live score.

    Serial action execution pads the remainder of a chunk with zeros after an
    environment terminates.  Selecting the last element would therefore erase
    a terminal score, while taking a running maximum can credit a transient
    state that was lost before termination.  Select the first done transition
    when present and otherwise the last executed transition.
    """

    scores = torch.as_tensor(score_steps).float()
    dones = torch.as_tensor(terminated_steps).bool() | torch.as_tensor(truncated_steps).bool()
    if scores.ndim != 2 or dones.shape != scores.shape:
        raise ValueError(
            "score/terminated/truncated tensors must have matching [B,K] shapes, "
            f"got score={tuple(scores.shape)} done={tuple(dones.shape)}."
        )
    already_done = np.asarray(already_done, dtype=bool).reshape(-1)
    if len(already_done) != scores.shape[0]:
        raise ValueError("already_done must contain one flag per vector environment.")
    result = np.zeros(scores.shape[0], dtype=np.float32)
    for index, (score_row, done_row) in enumerate(zip(scores, dones, strict=True)):
        if already_done[index]:
            continue
        done_indices = torch.nonzero(done_row, as_tuple=False).flatten()
        position = int(done_indices[0]) if done_indices.numel() else int(score_row.numel()) - 1
        result[index] = float(score_row[position])
    return result


def capture_first_policy_call(
    capture_dir: Path,
    *,
    wave: int,
    obs: dict[str, Any],
    actions: torch.Tensor,
) -> None:
    """Persist raw canonical inputs and executed actions for evaluator parity debugging."""

    capture_dir.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(obs["observation"]):
        stem = capture_dir / f"wave_{wave:04d}_env_{index:04d}"
        temporary = stem.with_suffix(".npz.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                head_camera=np.asarray(row["observation.images.cam_head"], dtype=np.uint8),
                left_camera=np.asarray(row["observation.images.cam_left_wrist"], dtype=np.uint8),
                right_camera=np.asarray(row["observation.images.cam_right_wrist"], dtype=np.uint8),
                proprio=np.asarray(row["observation.state"], dtype=np.float32),
                action=np.asarray(actions[index].cpu(), dtype=np.float32),
            )
        temporary.replace(stem.with_suffix(".npz"))
        metadata = {
            "wave": int(wave),
            "env_index": int(index),
            "instruction": str(obs["task"][index]),
            "layout_id": int(obs["layout_id"][index]),
            "environment_seed": int(obs["environment_seed"][index]),
            "policy_seed": int(obs["policy_seed"][index]),
        }
        stem.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")


def load_shared_fastwam_model(config):
    """Load one native Fast-WAM replica, optionally applying a verl actor in memory."""

    model_root = Path(str(config.model_root)).expanduser().resolve()
    model_device_id = int(config.model_device_id)
    from verl_vla.models.fastwam.trainable_model import FastWAMTrainableModel

    torch.cuda.set_device(model_device_id)
    actor_objective = str(config.get("actor_objective", "fpo"))
    if actor_objective not in {"fpo", "flow_grpo"}:
        raise ValueError(f"Unsupported actor_objective={actor_objective!r}.")
    model = FastWAMTrainableModel.from_pretrained(
        str(model_root),
        adapter_config={
            "policy_root": str(config.policy_root),
            "checkpoint_sha256": _checkpoint_sha(model_root, config.checkpoint_sha256),
            "weights_relative_path": str(NATIVE_WEIGHTS_RELATIVE_PATH),
            "dataset_stats_relative_path": "dataset_stats.json",
            "action_horizon": 32,
            "action_chunk_size": 24,
            "num_inference_steps": int(config.num_inference_steps),
            "rollout_action_latent_scale": float(config.rollout_action_latent_scale),
            "rollout_seed_mode": str(config.rollout_seed_mode),
            "sigma_shift": config.sigma_shift,
            "seed": int(config.seed),
            "text_cfg_scale": float(config.text_cfg_scale),
            "negative_prompt": str(config.negative_prompt),
            "rand_device": str(config.rand_device),
            "tiled": bool(config.tiled),
            "sim_cfg_name": str(config.sim_cfg_name),
            "sim_task": str(config.sim_task),
            "fpo": {"enabled": False, "value_enabled": False},
            "flow_grpo": {
                "enabled": actor_objective == "flow_grpo",
                "noise_level": float(config.get("flow_grpo_noise_level", 0.01)),
                "transition_batch_size": int(
                    config.get("flow_grpo_transition_batch_size", 3)
                ),
            },
        },
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    in_memory_rl_checkpoint = None
    if config.verl_actor_checkpoint not in (None, ""):
        print(f"loading_in_memory_rl_checkpoint={config.verl_actor_checkpoint}", flush=True)
        in_memory_rl_checkpoint = load_verl_fastwam_actor_into_native_policy(
            source=str(config.verl_actor_checkpoint),
            native_policy=model.policy,
        )
        print(f"loaded_in_memory_rl_checkpoint={in_memory_rl_checkpoint}", flush=True)
    return model, in_memory_rl_checkpoint


def _checkpoint_sha(model_root: Path, configured_sha256: str | None = None) -> str:
    if configured_sha256 not in (None, ""):
        value = str(configured_sha256).lower()
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("checkpoint_sha256 must be exactly 64 lowercase hexadecimal characters.")
        return value
    manifest_path = model_root / EXPORT_MANIFEST_FILENAME
    if manifest_path.is_file():
        with manifest_path.open(encoding="utf-8") as stream:
            manifest = json.load(stream)
        value = str(manifest["weights_sha256"])
        if len(value) != 64:
            raise ValueError(f"Invalid weights_sha256 in {manifest_path}")
        return value
    return sha256_file(model_root / NATIVE_WEIGHTS_RELATIVE_PATH)


def _ipc_observation(obs: dict[str, Any]) -> dict[str, Any]:
    """Copy only policy inputs/identities into a spawn-safe IPC payload."""

    result: dict[str, Any] = {
        "observation": [
            {key: np.asarray(value) for key, value in row.items()}
            for row in obs["observation"]
        ],
        "task": [str(value) for value in obs["task"]],
    }
    for field in (
        "task_id",
        "suite_id",
        "eval_episode_id",
        "layout_id",
        "environment_seed",
        "policy_seed",
    ):
        if field in obs:
            result[field] = np.asarray(obs[field])
    return result


def merge_ipc_observations(observations: list[dict[str, Any]]) -> tuple[dict[str, Any], list[slice]]:
    """Merge independent simulator requests for one shared model forward path.

    The returned slices map model outputs back to their originating simulator.
    This helper deliberately preserves request and row order; per-trajectory
    policy seeds therefore remain attached to exactly the same observations.
    """

    if not observations:
        raise ValueError("At least one observation request is required.")
    merged: dict[str, Any] = {"observation": [], "task": []}
    identity_fields = (
        "task_id",
        "suite_id",
        "eval_episode_id",
        "layout_id",
        "environment_seed",
        "policy_seed",
    )
    present_fields = {field for field in identity_fields if field in observations[0]}
    for index, observation in enumerate(observations[1:], start=1):
        actual_fields = {field for field in identity_fields if field in observation}
        if actual_fields != present_fields:
            raise ValueError(
                "Shared-model observation identity fields differ across requests: "
                f"request0={sorted(present_fields)} request{index}={sorted(actual_fields)}"
            )

    slices: list[slice] = []
    offset = 0
    field_rows: dict[str, list[np.ndarray]] = {field: [] for field in present_fields}
    for observation in observations:
        rows = list(observation["observation"])
        tasks = list(observation["task"])
        if not rows or len(rows) != len(tasks):
            raise ValueError("Every shared-model request must contain equally sized nonempty rows/tasks.")
        stop = offset + len(rows)
        slices.append(slice(offset, stop))
        offset = stop
        merged["observation"].extend(rows)
        merged["task"].extend(tasks)
        for field in present_fields:
            values = np.asarray(observation[field])
            if values.reshape(-1).shape[0] != len(rows):
                raise ValueError(f"Identity field {field!r} does not match request batch size.")
            field_rows[field].append(values.reshape(-1))
    for field, rows in field_rows.items():
        merged[field] = np.concatenate(rows, axis=0)
    return merged, slices


def _shared_model_env_worker(
    worker_id: int,
    worker_count: int,
    config_values: dict[str, Any],
    case_layouts: list[int],
    case_policy_seeds: list[int],
    request_queue,
    response_queue,
) -> None:
    """Own one vecN RoboDojo simulator and request actions from its parent."""

    runtime_root = Path(os.environ.get("TMPDIR", "/tmp")) / f"shared-sim-{worker_id}"
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

    env = None
    try:
        num_envs = len(case_layouts)
        simulator_device_id = int(config_values["simulator_device_id"])
        robodojo_cfg = RoboDojoSimulatorConfig(
            robodojo_root=str(config_values["robodojo_root"]),
            task_name=str(config_values["task_name"]),
            task_id=int(config_values["task_id"]),
            env_cfg_type=str(config_values["env_cfg_type"]),
            seed=int(config_values["seed"]),
            device_id=simulator_device_id,
            headless=bool(config_values["headless"]),
            layout_ids=case_layouts,
            eval_policy_seed_schedule=case_policy_seeds,
            allow_held_out_eval_layouts=bool(config_values["allow_held_out_eval_layouts"]),
            eval_num=num_envs,
            reset_max_attempts=int(config_values["reset_max_attempts"]),
            defer_chunk_observations=False,
        )
        recorder = RecorderConfig(
            enable=bool(config_values["record_video"]),
            recorders=("video",) if bool(config_values["record_video"]) else (),
            video=VideoRecorderConfig(
                enable=bool(config_values["record_video"]),
                root=str(Path(str(config_values["output_dir"])) / "videos"),
                fps=int(config_values["video_fps"]),
            ),
        )
        env_cfg = EnvWorkerConfig(
            auto_reset=False,
            action_execution=ActionExecutionConfig(mode="serial"),
            modes=["eval"],
            num_envs=num_envs,
            simulator=SimulatorConfig(simulator_type="robodojo", robodojo=robodojo_cfg),
            teleop=TeleopConfig(enable=False),
            recorder=recorder,
            device="cuda",
        )
        # AppLauncher must run in the child before importing app-dependent
        # RoboDojo modules.  No Fast-WAM model is loaded in this process.
        from verl_vla.envs.robodojo.robodojo_env import RoboDojoEnv

        env = RoboDojoEnv(
            OmegaConf.structured(env_cfg), rank=worker_id, world_size=worker_count, only_eval=True
        )
        torch.cuda.set_device(simulator_device_id)
        obs, _ = env.reset(options={"reset_eval": True, "extra": {"mode": "eval"}})
        identity = [
            {
                "eval_episode_id": int(obs["eval_episode_id"][index]),
                "layout_id": int(obs["layout_id"][index]),
                "environment_seed": int(obs["environment_seed"][index]),
                "policy_seed": int(obs["policy_seed"][index]),
            }
            for index in range(num_envs)
        ]
        success = np.zeros(num_envs, dtype=bool)
        done = np.zeros(num_envs, dtype=bool)
        score = np.zeros(num_envs, dtype=np.float32)
        calls = np.zeros(num_envs, dtype=np.int64)
        process_score_reduction = str(config_values["process_score_reduction"])
        for policy_call_index in range(int(config_values["max_policy_calls"])):
            calls[~done] += 1
            request_queue.put(
                {
                    "kind": "infer",
                    "worker_id": worker_id,
                    "policy_call_index": policy_call_index,
                    "observation": _ipc_observation(obs),
                }
            )
            response = response_queue.get(timeout=600)
            if response.get("kind") == "error":
                raise RuntimeError(str(response["error"]))
            actions = torch.as_tensor(response["actions"], dtype=torch.float32)
            torch.cuda.set_device(simulator_device_id)
            step_result = env.step(actions)
            obs, _, terminated, truncated, successes, *extras = step_result
            terminated_steps = torch.as_tensor(terminated).bool()
            truncated_steps = torch.as_tensor(truncated).bool()
            success |= torch.as_tensor(successes).bool().any(dim=1).cpu().numpy()
            if extras and "score" in extras[0]:
                if process_score_reduction == "trajectory_max":
                    score = np.maximum(
                        score,
                        torch.as_tensor(extras[0]["score"]).amax(dim=1).cpu().numpy(),
                    )
                else:
                    active_scores = terminal_or_latest_scores(
                        extras[0]["score"], terminated_steps, truncated_steps, done
                    )
                    score[~done] = active_scores[~done]
            done |= (
                terminated_steps.any(dim=1).cpu().numpy()
                | truncated_steps.any(dim=1).cpu().numpy()
            )
            if done.all():
                break
        cases = [
            {
                **case,
                "success": bool(success[index]),
                "score": float(score[index]),
                "policy_calls": int(calls[index]),
                "done": bool(done[index]),
                "simulator_shard": worker_id,
            }
            for index, case in enumerate(identity)
        ]
        request_queue.put({"kind": "done", "worker_id": worker_id, "cases": cases})
    except BaseException:
        request_queue.put(
            {
                "kind": "error",
                "worker_id": worker_id,
                "error": traceback.format_exc(),
            }
        )
    finally:
        if env is not None:
            env.close()


def run_shared_model_sharded_robodojo_eval(
    config, case_layouts: list[int], case_policy_seeds: list[int]
) -> dict[str, Any]:
    """Run independent simulator processes against one parent Fast-WAM replica."""

    simulator_shards = int(config.simulator_shards)
    num_envs = int(config.num_envs)
    if simulator_shards < 2 or num_envs % simulator_shards:
        raise ValueError("simulator_shards must be >=2 and evenly divide num_envs.")
    shard_size = num_envs // simulator_shards
    if len(case_layouts) != num_envs or len(case_policy_seeds) != num_envs:
        raise ValueError("Shared-model sharded eval currently requires exactly one vector wave.")

    config_values = {
        "robodojo_root": str(config.robodojo_root),
        "task_name": str(config.task_name),
        "task_id": int(config.task_id),
        "env_cfg_type": str(config.env_cfg_type),
        "seed": int(config.seed),
        "simulator_device_id": int(config.simulator_device_id),
        "headless": bool(config.headless),
        "allow_held_out_eval_layouts": bool(config.allow_held_out_eval_layouts),
        "reset_max_attempts": int(config.reset_max_attempts),
        "record_video": bool(config.record_video),
        "video_fps": int(config.video_fps),
        "output_dir": str(config.output_dir),
        "process_score_reduction": str(config.process_score_reduction),
        "max_policy_calls": int(config.max_policy_calls),
    }
    context = mp.get_context("spawn")
    request_queue = context.Queue(maxsize=simulator_shards * 2)
    response_queues = [context.Queue(maxsize=1) for _ in range(simulator_shards)]
    workers = []
    # Starting several Isaac/Kit renderers on the same GPU at exactly the same
    # instant can make Vulkan return DEVICE_LOST even though steady-state VRAM
    # is sufficient.  Stagger only renderer initialization; all shards still
    # execute concurrently after startup and share the same policy replica.
    shard_startup_stagger_seconds = float(
        os.environ.get("VERL_VLA_SIMULATOR_SHARD_STARTUP_STAGGER_SECONDS", "8")
    )
    for worker_id in range(simulator_shards):
        start = worker_id * shard_size
        stop = start + shard_size
        process = context.Process(
            target=_shared_model_env_worker,
            args=(
                worker_id,
                simulator_shards,
                config_values,
                case_layouts[start:stop],
                case_policy_seeds[start:stop],
                request_queue,
                response_queues[worker_id],
            ),
            name=f"robodojo-vec{shard_size}-shard-{worker_id}",
        )
        process.start()
        workers.append(process)
        if worker_id + 1 < simulator_shards and shard_startup_stagger_seconds > 0:
            time.sleep(shard_startup_stagger_seconds)

    model = None
    try:
        model, in_memory_rl_checkpoint = load_shared_fastwam_model(config)
        active = set(range(simulator_shards))
        all_cases: list[dict[str, Any]] = []
        last_progress = time.monotonic()
        while active:
            try:
                message = request_queue.get(timeout=30)
            except queue.Empty:
                failed = [
                    (index, process.exitcode)
                    for index, process in enumerate(workers)
                    if index in active and process.exitcode not in (None, 0)
                ]
                if failed:
                    raise RuntimeError(
                        f"RoboDojo simulator shard exited unexpectedly: {failed}"
                    ) from None
                if time.monotonic() - last_progress > 600:
                    raise TimeoutError(
                        "No progress from shared-model simulator shards for 600 seconds."
                    ) from None
                continue
            last_progress = time.monotonic()
            worker_id = int(message["worker_id"])
            kind = str(message["kind"])
            if kind == "infer":
                actions = infer_shared_model_in_microbatches(
                    model,
                    message["observation"],
                    int(config.policy_microbatch_size),
                    model_device_id=int(config.model_device_id),
                )
                response_queues[worker_id].put(
                    {"kind": "actions", "actions": actions.numpy()}
                )
            elif kind == "done":
                all_cases.extend(message["cases"])
                active.discard(worker_id)
                print(
                    f"shared_simulator_complete={worker_id + 1}/{simulator_shards} "
                    f"cases={len(message['cases'])} total_cases={len(all_cases)}",
                    flush=True,
                )
            elif kind == "error":
                raise RuntimeError(
                    f"RoboDojo simulator shard {worker_id} failed:\n{message['error']}"
                )
            else:
                raise ValueError(f"Unknown shared evaluator message: {kind!r}")

        output_dir = Path(str(config.output_dir)).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        all_cases.sort(key=lambda row: (int(row["layout_id"]), int(row["policy_seed"])))
        summary = {
            "task_name": str(config.task_name),
            "model_root": str(Path(str(config.model_root)).expanduser().resolve()),
            "in_memory_rl_checkpoint": in_memory_rl_checkpoint,
            "num_cases": len(all_cases),
            "robodojo_eval_seed": int(config.seed),
            "num_envs": num_envs,
            "simulator_shards": simulator_shards,
            "simulator_vec_size": shard_size,
            "reset_max_attempts": int(config.reset_max_attempts),
            "policy_microbatch_size": int(config.policy_microbatch_size),
            "process_score_reduction": str(config.process_score_reduction),
            "action_chunk_size": 24,
            "max_policy_calls": int(config.max_policy_calls),
            "simulator_device_id": int(config.simulator_device_id),
            "model_device_id": int(config.model_device_id),
            "model_replicas": 1,
            "successes": sum(int(row["success"]) for row in all_cases),
            "success_rate": float(np.mean([row["success"] for row in all_cases])),
            "mean_score": float(np.mean([row["score"] for row in all_cases])),
            "all_cases_done": all(bool(row["done"]) for row in all_cases),
            "incomplete_case_count": sum(not bool(row["done"]) for row in all_cases),
            "cases": all_cases,
        }
        temporary = output_dir / "metrics.json.tmp"
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2, sort_keys=True)
            stream.write("\n")
        temporary.replace(output_dir / "metrics.json")
        print(f"metrics_persisted_before_isaac_close={output_dir / 'metrics.json'}", flush=True)
        return summary
    finally:
        del model
        for response_queue in response_queues:
            try:
                response_queue.put_nowait({"kind": "error", "error": "parent evaluator stopped"})
            except queue.Full:
                pass
        for process in workers:
            process.join(timeout=60)
            if process.is_alive():
                process.terminate()
                process.join(timeout=30)


def run_native_fastwam_robodojo_eval(config) -> dict[str, Any]:
    """Evaluate without Ray or a separate rollout GPU.

    One process owns one vectorized RoboDojo backend and exactly one Fast-WAM
    model. ``policy_microbatch_size`` limits inference activation memory; it
    never creates another model replica.
    """

    model_root = Path(str(config.model_root)).expanduser().resolve()
    case_layouts = [int(value) for value in config.layout_ids]
    case_policy_seeds = [int(value) for value in config.policy_seeds]
    if len(case_layouts) != len(case_policy_seeds):
        raise ValueError("layout_ids and policy_seeds must define the same number of cases.")
    num_envs = int(config.num_envs)
    if not case_layouts or len(case_layouts) % num_envs:
        raise ValueError("The number of fixed cases must be a nonzero multiple of num_envs.")
    microbatch_size = int(config.policy_microbatch_size)
    if not 1 <= microbatch_size <= num_envs:
        raise ValueError("policy_microbatch_size must be in [1,num_envs].")
    simulator_device_id = int(config.simulator_device_id)
    model_device_id = int(config.model_device_id)
    process_score_reduction = str(config.process_score_reduction)
    if process_score_reduction not in {"terminal", "trajectory_max"}:
        raise ValueError("process_score_reduction must be 'terminal' or 'trajectory_max'.")
    simulator_shards = int(config.simulator_shards)
    if simulator_shards < 1:
        raise ValueError("simulator_shards must be positive.")
    if simulator_shards > 1:
        return run_shared_model_sharded_robodojo_eval(
            config,
            case_layouts,
            case_policy_seeds,
        )

    robodojo_cfg = RoboDojoSimulatorConfig(
        robodojo_root=str(config.robodojo_root),
        task_name=str(config.task_name),
        task_id=int(config.task_id),
        env_cfg_type=str(config.env_cfg_type),
        # RoboDojo's public benchmark defines three distinct layout suites via
        # eval seed 0/1/2.  Do not silently leave the simulator on the dataclass
        # default when Hydra selects another suite.  Policy seeds remain
        # independently controlled by eval_policy_seed_schedule below.
        seed=int(config.seed),
        device_id=simulator_device_id,
        headless=bool(config.headless),
        layout_ids=case_layouts,
        eval_policy_seed_schedule=case_policy_seeds,
        allow_held_out_eval_layouts=bool(config.allow_held_out_eval_layouts),
        eval_num=len(case_layouts),
        reset_max_attempts=int(config.reset_max_attempts),
        # Evaluation must match RoboDojo's official per-step render/sensor
        # cadence even when video recording is disabled.  The deferred fast
        # path is not yet closed-loop-equivalent on manipulation tasks.
        defer_chunk_observations=False,
    )
    recorder = RecorderConfig(
        enable=bool(config.record_video),
        recorders=("video",) if bool(config.record_video) else (),
        video=VideoRecorderConfig(
            enable=bool(config.record_video),
            root=str(Path(str(config.output_dir)) / "videos"),
            fps=int(config.video_fps),
        ),
    )
    env_cfg = EnvWorkerConfig(
        auto_reset=False,
        action_execution=ActionExecutionConfig(mode="serial"),
        modes=["eval"],
        num_envs=num_envs,
        simulator=SimulatorConfig(simulator_type="robodojo", robodojo=robodojo_cfg),
        teleop=TeleopConfig(enable=False),
        recorder=recorder,
        device="cuda",
    )

    # RoboDojo must launch Isaac before app-dependent modules are imported.
    from verl_vla.envs.robodojo.robodojo_env import RoboDojoEnv

    env = RoboDojoEnv(OmegaConf.structured(env_cfg), rank=0, world_size=1, only_eval=True)
    summary: dict[str, Any] | None = None
    try:
        # Do not initialize CUDA through torch before Isaac's AppLauncher. Once
        # Isaac owns the simulator GPU, validate both roles and explicitly
        # select the appropriate device at every simulator/model boundary.
        visible_gpu_count = torch.cuda.device_count()
        for role, device_id in (
            ("simulator", simulator_device_id),
            ("model", model_device_id),
        ):
            if not 0 <= device_id < visible_gpu_count:
                raise ValueError(
                    f"{role}_device_id={device_id} is outside the {visible_gpu_count} visible CUDA devices."
                )
        model, in_memory_rl_checkpoint = load_shared_fastwam_model(config)
        results: list[dict[str, Any]] = []
        first_call_capture_dir = (
            None
            if config.first_call_capture_dir in (None, "")
            else Path(str(config.first_call_capture_dir)).expanduser().resolve()
        )
        wave_count = len(case_layouts) // num_envs
        for wave in range(wave_count):
            torch.cuda.set_device(simulator_device_id)
            obs, _ = env.reset(
                options={
                    "reset_eval": wave == 0,
                    "extra": {"mode": "eval"},
                }
            )
            identity = [
                {
                    "eval_episode_id": int(obs["eval_episode_id"][index]),
                    "layout_id": int(obs["layout_id"][index]),
                    "environment_seed": int(obs["environment_seed"][index]),
                    "policy_seed": int(obs["policy_seed"][index]),
                }
                for index in range(num_envs)
            ]
            success = np.zeros(num_envs, dtype=bool)
            done = np.zeros(num_envs, dtype=bool)
            score = np.zeros(num_envs, dtype=np.float32)
            calls = np.zeros(num_envs, dtype=np.int64)
            for policy_call_index in range(int(config.max_policy_calls)):
                calls[~done] += 1
                actions = infer_shared_model_in_microbatches(
                    model,
                    obs,
                    microbatch_size,
                    model_device_id=model_device_id,
                )
                if policy_call_index == 0 and first_call_capture_dir is not None:
                    capture_first_policy_call(
                        first_call_capture_dir,
                        wave=wave,
                        obs=obs,
                        actions=actions,
                    )
                torch.cuda.set_device(simulator_device_id)
                step_result = env.step(actions)
                obs, _, terminated, truncated, successes, *extras = step_result
                terminated_steps = torch.as_tensor(terminated).bool()
                truncated_steps = torch.as_tensor(truncated).bool()
                success |= torch.as_tensor(successes).bool().any(dim=1).cpu().numpy()
                if extras and "score" in extras[0]:
                    if process_score_reduction == "trajectory_max":
                        score = np.maximum(
                            score,
                            torch.as_tensor(extras[0]["score"]).amax(dim=1).cpu().numpy(),
                        )
                    else:
                        active_scores = terminal_or_latest_scores(
                            extras[0]["score"], terminated_steps, truncated_steps, done
                        )
                        score[~done] = active_scores[~done]
                terminated = terminated_steps.any(dim=1).cpu().numpy()
                truncated = truncated_steps.any(dim=1).cpu().numpy()
                done |= terminated | truncated
                if done.all():
                    break
            for index, case in enumerate(identity):
                results.append(
                    {
                        **case,
                        "success": bool(success[index]),
                        "score": float(score[index]),
                        "policy_calls": int(calls[index]),
                        "done": bool(done[index]),
                    }
                )
            print(
                f"wave={wave + 1}/{wave_count} success={int(success.sum())}/{num_envs} "
                f"cumulative={sum(int(row['success']) for row in results)}/{len(results)}",
                flush=True,
            )

        # Some Isaac/Kit builds exit nonzero from SimulationApp.close() even
        # after every rollout wave completed. Persist atomically before close
        # so a shutdown failure cannot silently discard a valid evaluation.
        output_dir = Path(str(config.output_dir)).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "task_name": str(config.task_name),
            "model_root": str(model_root),
            "in_memory_rl_checkpoint": in_memory_rl_checkpoint,
            "num_cases": len(results),
            "robodojo_eval_seed": int(config.seed),
            "num_envs": num_envs,
            "simulator_shards": 1,
            "simulator_vec_size": num_envs,
            "reset_max_attempts": int(config.reset_max_attempts),
            "policy_microbatch_size": microbatch_size,
            "process_score_reduction": process_score_reduction,
            "action_chunk_size": 24,
            "max_policy_calls": int(config.max_policy_calls),
            "simulator_device_id": simulator_device_id,
            "model_device_id": model_device_id,
            "model_replicas": 1,
            "successes": sum(int(row["success"]) for row in results),
            "success_rate": float(np.mean([row["success"] for row in results])),
            "mean_score": float(np.mean([row["score"] for row in results])),
            "all_cases_done": all(bool(row["done"]) for row in results),
            "incomplete_case_count": sum(not bool(row["done"]) for row in results),
            "cases": results,
        }
        metrics_path = output_dir / "metrics.json"
        temporary_metrics_path = output_dir / "metrics.json.tmp"
        with temporary_metrics_path.open("w", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2, sort_keys=True)
            stream.write("\n")
        temporary_metrics_path.replace(metrics_path)
        print(f"metrics_persisted_before_isaac_close={metrics_path}", flush=True)
    finally:
        torch.cuda.set_device(simulator_device_id)
        env.close()

    assert summary is not None, "Evaluation completed without constructing its summary."
    return summary
