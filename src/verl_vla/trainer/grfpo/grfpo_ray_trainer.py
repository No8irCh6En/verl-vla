# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
from collections import Counter
from pathlib import Path
from pprint import pprint
from typing import Any

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm
from verl import DataProto
from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage
from verl.utils.metric import reduce_metrics

from verl_vla.train_cluster import TrainCluster
from verl_vla.utils.keys import OBS_KEY
from verl_vla.utils.rlpd import pad_dataproto_to_divisor_with_valid_mask

from .config import GRFPOTrainerConfig
from .group_validation import _terminal_trajectory_scores, validate_same_condition_group

module_logger = logging.getLogger(__name__)


def _bounded_candidate_batch_size(
    *,
    configured_concurrency: int,
    remaining_candidate_budget: int,
    remaining_accepted_slots: int,
) -> int:
    values = {
        "configured_concurrency": configured_concurrency,
        "remaining_candidate_budget": remaining_candidate_budget,
        "remaining_accepted_slots": remaining_accepted_slots,
    }
    if any(value <= 0 for value in values.values()):
        raise ValueError(f"Candidate collection batch bounds must all be positive, got {values}.")
    return min(values.values())


def _quantized_candidate_batch_size(
    *,
    configured_concurrency: int,
    remaining_candidate_budget: int,
    remaining_accepted_slots: int,
    groups_per_stage: int,
) -> int:
    """Choose a batch that consists of complete all-worker pipeline stages.

    Worker-local grouping produces ``groups_per_stage`` candidate groups at
    once because a stage is dispatched to every EnvWorker rank. Near the M
    target we may therefore collect a final sibling group that is informative
    but not selected for training. Returning zero means the remaining candidate
    budget cannot fund one atomic stage.
    """

    if groups_per_stage <= 0:
        raise ValueError(f"groups_per_stage must be positive, got {groups_per_stage}.")
    if groups_per_stage == 1:
        return _bounded_candidate_batch_size(
            configured_concurrency=configured_concurrency,
            remaining_candidate_budget=remaining_candidate_budget,
            remaining_accepted_slots=remaining_accepted_slots,
        )
    if configured_concurrency <= 0 or remaining_candidate_budget <= 0 or remaining_accepted_slots <= 0:
        raise ValueError(
            "Candidate collection batch bounds must all be positive, got "
            f"configured_concurrency={configured_concurrency}, "
            f"remaining_candidate_budget={remaining_candidate_budget}, "
            f"remaining_accepted_slots={remaining_accepted_slots}."
        )
    upper_bound = min(configured_concurrency, remaining_candidate_budget)
    if upper_bound < groups_per_stage:
        return 0
    desired = min(upper_bound, max(remaining_accepted_slots, groups_per_stage))
    return (desired // groups_per_stage) * groups_per_stage


def _configured_env_worker_count(config: Any) -> int:
    resource = config.cluster.resource.env
    per_node = int(resource.workers_per_node) if str(resource.device) == "cpu" else int(resource.gpus_per_node)
    return int(resource.nnodes) * per_node


def _candidate_groups_per_stage(config: Any, trainer_config: Any) -> int:
    partition = str(getattr(trainer_config, "candidate_group_partition", "stage"))
    if partition == "stage":
        return 1
    if partition == "worker_stage":
        group_size = int(trainer_config.group_size)
        envs_per_worker = int(config.cluster.env.env_worker.num_envs)
        if envs_per_worker < group_size or envs_per_worker % group_size != 0:
            raise ValueError(
                "worker_stage grouping requires num_envs to be a positive multiple of "
                f"group_size: num_envs={envs_per_worker}, group_size={group_size}."
            )
        return _configured_env_worker_count(config) * (envs_per_worker // group_size)
    raise ValueError(f"Unknown candidate_group_partition={partition!r}.")


def _candidate_task_assignments(
    config: Any,
    trainer_config: Any,
    *,
    group_count: int,
) -> list[tuple[str, int]]:
    """Return task name/id in the stage-major candidate collation order."""

    robodojo = config.cluster.env.env_worker.simulator.robodojo
    worker_count = _configured_env_worker_count(config)
    stage_count = int(config.cluster.env.env_loop.pipeline_stage_num)
    process_schedule = list(robodojo.worker_stage_task_schedule)
    if process_schedule:
        expected_processes = worker_count * stage_count
        if len(process_schedule) != expected_processes:
            raise ValueError(
                "worker_stage_task_schedule must align with the configured EnvWorker topology: "
                f"expected={expected_processes}, got={len(process_schedule)}."
            )
        task_names = list(robodojo.task_names)
        task_to_id = {task_name: task_id for task_id, task_name in enumerate(task_names)}
    else:
        process_schedule = [str(robodojo.task_name)] * (worker_count * stage_count)
        task_to_id = {str(robodojo.task_name): int(robodojo.task_id)}

    partition = str(trainer_config.candidate_group_partition)
    if partition == "stage":
        assignments = []
        for stage_id in range(stage_count):
            stage_tasks = process_schedule[stage_id * worker_count : (stage_id + 1) * worker_count]
            if len(set(stage_tasks)) != 1:
                raise ValueError(
                    "stage grouping combines all EnvWorker ranks into one GRPO group, so every "
                    f"worker in stage {stage_id} must own the same task: {stage_tasks!r}."
                )
            task_name = stage_tasks[0]
            assignments.append((task_name, task_to_id[task_name]))
    elif partition == "worker_stage":
        group_size = int(trainer_config.group_size)
        envs_per_worker = int(config.cluster.env.env_worker.num_envs)
        groups_per_process = envs_per_worker // group_size
        assignments = [
            (task_name, task_to_id[task_name])
            for task_name in process_schedule
            for _ in range(groups_per_process)
        ]
    else:
        raise ValueError(f"Unknown candidate_group_partition={partition!r}.")
    if group_count > len(assignments):
        raise ValueError(
            f"Requested {group_count} task assignments from only {len(assignments)} worker-stage group slots."
        )
    return assignments[:group_count]


def _optional_refill_is_unnecessary(
    *,
    candidate_groups: int,
    accepted_groups: int,
    initial_candidate_groups: int,
    accepted_groups_before_refill: int,
) -> bool:
    """Stop after the initial candidate window when it is informative enough."""

    if initial_candidate_groups <= 0 or accepted_groups_before_refill <= 0:
        return False
    return candidate_groups >= initial_candidate_groups and accepted_groups >= accepted_groups_before_refill


def _candidate_schedule_slot_after_fixed_eval(current_slot: int, pipeline_stage_num: int) -> int:
    """Account for the post-eval replacement of one prefetched train reset."""

    if current_slot < 0 or pipeline_stage_num <= 0:
        raise ValueError(
            "Candidate schedule state must be non-negative with positive pipeline width: "
            f"current_slot={current_slot}, pipeline_stage_num={pipeline_stage_num}."
        )
    return current_slot + pipeline_stage_num


def _validate_fastwam_rollout_temperature(config: Any) -> None:
    model = config.cluster.actor_rollout_ref.model
    if str(getattr(model, "native_architecture", "")) != "fastwam":
        return
    rollout = config.cluster.actor_rollout_ref.rollout
    temperature = float(getattr(rollout, "temperature", 1.0))
    if not np.isclose(temperature, 1.0):
        raise ValueError(
            "Fast-WAM does not consume veRL's token-logit rollout temperature. "
            "Keep rollout.temperature=1.0; flow exploration requires an explicit, "
            "separately validated action-latent sampling treatment."
        )


def _fastwam_action_latent_scale(config: Any) -> float:
    return float(
        OmegaConf.select(
            config,
            "cluster.actor_rollout_ref.model.adapter.rollout_action_latent_scale",
            default=1.0,
        )
    )


def _fastwam_rollout_seed_mode(config: Any) -> str:
    return str(
        OmegaConf.select(
            config,
            "cluster.actor_rollout_ref.model.adapter.rollout_seed_mode",
            default="episode",
        )
    )


def apply_grpo_outcome_advantages(
    actor_input: DataProto,
    *,
    group_record: dict[str, Any] | None = None,
    group_records: list[dict[str, Any]] | None = None,
    group_size: int,
    reward_source: str = "binary_success",
    accepted_group_reduction: str = "group_equal",
) -> dict[str, Any]:
    """Normalize outcomes per group and construct the hierarchical actor weights.

    ``group_equal`` computes mean_g(mean_i(mean_valid_chunk(loss))).
    ``task_equal`` computes mean_t(mean_{g in t}(mean_i(mean_valid_chunk(loss)))).
    In both modes every trajectory in a group and every valid chunk in a
    trajectory retain equal hierarchical weight.
    """

    if (group_record is None) == (group_records is None):
        raise ValueError("Provide exactly one of group_record or group_records.")
    records = [group_record] if group_record is not None else list(group_records or [])
    if accepted_group_reduction not in {"group_equal", "task_equal"}:
        raise ValueError(
            "accepted_group_reduction must be 'group_equal' or 'task_equal', "
            f"got {accepted_group_reduction!r}."
        )
    expected_trajectories = len(records) * group_size
    if len(actor_input) != expected_trajectories:
        raise ValueError(
            f"Expected {expected_trajectories} trajectories from {len(records)} G={group_size} groups "
            f"before chunk flattening, got {len(actor_input)}."
        )
    policy_versions = {int(record["rollout_policy_version"]) for record in records}
    if len(policy_versions) != 1:
        raise ValueError(f"One online update must use exactly one theta_old, got versions={policy_versions}.")
    group_ids = np.asarray([record["group_id"] for record in records for _ in range(group_size)], dtype=object)
    if "obs.group_id" in actor_input.non_tensor_batch:
        observed_group_ids = np.asarray(actor_input.non_tensor_batch["obs.group_id"])
        observed_group_ids = observed_group_ids[:, 0] if observed_group_ids.ndim > 1 else observed_group_ids
        if not np.array_equal(observed_group_ids, group_ids):
            raise ValueError("Actor-input group ids do not match the accepted-group assembly order.")
    reward_key = "info.trajectory_outcome" if reward_source == "binary_success" else "info.trajectory_process_score"
    if reward_key not in actor_input.batch:
        raise KeyError(f"GRPO reward source {reward_source!r} requires {reward_key!r}.")
    outcomes = actor_input.batch[reward_key].float()
    if tuple(outcomes.shape) != (expected_trajectories,):
        raise ValueError(f"trajectory outcomes must have shape [{expected_trajectories}], got {tuple(outcomes.shape)}.")
    scalar_mask = torch.ones_like(outcomes).unsqueeze(-1)
    scalar_advantages, _ = compute_grpo_outcome_advantage(
        token_level_rewards=outcomes.unsqueeze(-1),
        response_mask=scalar_mask,
        index=group_ids,
        norm_adv_by_std_in_grpo=True,
    )
    trajectory_advantages = scalar_advantages[:, 0]
    if not torch.isfinite(trajectory_advantages).all():
        raise FloatingPointError("GRPO trajectory advantages contain NaN or Inf.")
    per_group_diagnostics = []
    for group_index, record in enumerate(records):
        group_slice = slice(group_index * group_size, (group_index + 1) * group_size)
        group_outcomes = outcomes[group_slice]
        group_advantages = trajectory_advantages[group_slice]
        recorded_rewards = torch.as_tensor(
            record.get("reward_vector", group_outcomes.tolist()),
            device=group_outcomes.device,
            dtype=group_outcomes.dtype,
        )
        if recorded_rewards.shape != group_outcomes.shape or not torch.allclose(
            recorded_rewards, group_outcomes, atol=1e-7, rtol=0
        ):
            raise RuntimeError(f"Collected and actor-input GRPO rewards disagree for group {record['group_id']}.")
        if reward_source == "binary_success" and bool(record["mixed"]):
            positive = group_outcomes.bool()
            if not bool((group_advantages[positive] > 0).all()):
                raise RuntimeError("Successful GRPO trajectories must have positive advantages.")
            if not bool((group_advantages[~positive] < 0).all()):
                raise RuntimeError("Failed GRPO trajectories must have negative advantages.")
            if not torch.isclose(group_advantages.mean(), group_advantages.new_zeros(()), atol=1e-6):
                raise RuntimeError(f"GRPO advantages must normalize independently within group {record['group_id']}.")
        group_reward_std = group_outcomes.std()
        per_group_diagnostics.append(
            {
                "group_id": record["group_id"],
                "group_key": record.get("group_key", {}),
                "layout": int(record.get("group_key", {}).get("layout_id", -1)),
                "reward_source": reward_source,
                "reward_vector": [float(value) for value in group_outcomes.tolist()],
                "successes": int(record.get("successes", 0)),
                "partial_only_count": int(record.get("partial_only_count", 0)),
                "informative_reason": record.get("informative_reason"),
                "advantage_vector": [float(value) for value in group_advantages.tolist()],
                "reward_mean": float(group_outcomes.mean()),
                "reward_std": float(group_reward_std),
                "advantage_mean": float(group_advantages.mean()),
                "advantage_std": float(group_advantages.std()),
            }
        )

    valids = actor_input.batch["info.valids"].float()
    valid_chunks = valids.sum(dim=1)
    if bool((valid_chunks <= 0).any()):
        raise ValueError("Every GRPO trajectory must contain at least one valid policy chunk.")
    # First make each trajectory contribute one unit independently of length.
    trajectory_balanced_weights = valids / valid_chunks.unsqueeze(-1)
    if not torch.allclose(
        trajectory_balanced_weights.sum(dim=1), torch.ones_like(valid_chunks), atol=1e-6
    ):
        raise RuntimeError("Every trajectory must contribute exactly unit total loss weight.")

    group_scales = torch.ones(len(records), dtype=valids.dtype, device=valids.device)
    task_group_counts: dict[str, int] = {}
    if accepted_group_reduction == "task_equal":
        task_names = []
        for record in records:
            if not record.get("task_name"):
                raise KeyError("task_equal reduction requires task_name on every accepted group record.")
            task_names.append(str(record["task_name"]))
        task_group_counts = dict(Counter(task_names))
        group_scales = torch.tensor(
            [1.0 / task_group_counts[task_name] for task_name in task_names],
            dtype=valids.dtype,
            device=valids.device,
        )

    trajectory_scales = group_scales.repeat_interleave(group_size)
    loss_weights = trajectory_balanced_weights * trajectory_scales.unsqueeze(-1)
    group_weight_sums = loss_weights.sum(dim=1).reshape(len(records), group_size).sum(dim=1)
    expected_group_weight_sums = group_scales * float(group_size)
    if not torch.allclose(group_weight_sums, expected_group_weight_sums, atol=1e-6):
        raise RuntimeError("Accepted-group loss weights disagree with the configured hierarchical reduction.")
    if accepted_group_reduction == "task_equal":
        task_weight_sums = {
            task_name: group_weight_sums[
                torch.tensor(
                    [str(record["task_name"]) == task_name for record in records],
                    device=group_weight_sums.device,
                    dtype=torch.bool,
                )
            ].sum()
            for task_name in task_group_counts
        }
        for task_name, weight_sum in task_weight_sums.items():
            # The weights are stored in float32.  With many accepted groups,
            # summing repeated 1 / M_t scales can accumulate a few ulps (for
            # example, 12 fold groups produce 7.99999857 instead of 8).  This
            # assertion guards reduction semantics, not bitwise summation.
            if not torch.isclose(
                weight_sum, weight_sum.new_tensor(float(group_size)), atol=1e-5, rtol=0
            ):
                raise RuntimeError(
                    f"Task-equal reduction assigned task {task_name!r} total weight {float(weight_sum)}, "
                    f"expected G={group_size}."
                )
    actor_input.batch["fpo.trajectory_group_advantage"] = trajectory_advantages
    actor_input.batch["fpo.advantages"] = trajectory_advantages.unsqueeze(-1) * valids
    actor_input.batch["fpo.loss_weights"] = loss_weights
    actor_input.meta_info.update(
        {
            "advantage_estimator": "grpo_outcome",
            "group_reward_source": reward_source,
            "accepted_group_reduction": accepted_group_reduction,
            "critic_enabled": False,
            "rollout_policy_version": int(next(iter(policy_versions))),
            "accepted_groups": len(records),
            "accepted_training_trajectories": expected_trajectories,
            "reduction": (
                "mean_task(mean_group_within_task(mean_trajectory(mean_valid_chunk(loss))))"
                if accepted_group_reduction == "task_equal"
                else "mean_group(mean_trajectory(mean_valid_chunk(loss)))"
            ),
            "task_group_counts": task_group_counts,
        }
    )
    group_reward_stds = torch.tensor([item["reward_std"] for item in per_group_diagnostics], dtype=torch.float32)
    diagnostics = {
        "group/reward_mean": float(outcomes.mean()),
        "group/reward_std": float(group_reward_stds.mean()),
        "group/advantage_mean": float(trajectory_advantages.mean()),
        "group/advantage_std": float(trajectory_advantages.std()),
        "group/fraction_nonzero_advantage_samples": float((trajectory_advantages != 0).float().mean()),
        "group/fraction_zero_variance_groups": float((group_reward_stds == 0).float().mean()),
        "group/fraction_all_success_groups": float(
            sum(item["successes"] == group_size for item in per_group_diagnostics) / len(records)
        ),
        "group/fraction_all_failure_groups": float(
            sum(item["successes"] == 0 for item in per_group_diagnostics) / len(records)
        ),
        "trajectory_advantages": [float(value) for value in trajectory_advantages.tolist()],
        "groups": per_group_diagnostics,
    }
    return diagnostics


def first_rollout_observation_and_action(rollout: DataProto) -> tuple[DataProto, torch.Tensor]:
    """Extract the first fixed observation and its on-policy native action."""

    obs_prefix = f"{OBS_KEY}."
    tensors = {
        str(key)[len(obs_prefix) :]: value[:, 0].clone()
        for key, value in rollout.batch.items()
        if str(key).startswith(obs_prefix)
    }
    non_tensors = {
        str(key)[len(obs_prefix) :]: np.asarray(value)[:, 0].copy()
        for key, value in rollout.non_tensor_batch.items()
        if str(key).startswith(obs_prefix)
    }
    if not tensors:
        raise ValueError("Fixed-observation diagnostic requires tensor observation fields.")
    action = rollout.batch["action.full_action"][:, 0].detach().cpu().clone()
    return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors), action


def _tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def summarize_dataproto_contract(data: DataProto) -> dict[str, Any]:
    """Return a compact, JSON-safe field/shape/dtype/finiteness audit."""

    tensor_fields: dict[str, Any] = {}
    all_finite = True
    if data.batch is not None:
        for key in sorted(data.batch.keys()):
            value = data.batch[key]
            finite = True
            if value.is_floating_point() or value.is_complex():
                finite = bool(torch.isfinite(value).all().item())
            tensor_fields[str(key)] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "finite": finite,
            }
            all_finite &= finite

    non_tensor_fields = {
        str(key): {"shape": list(np.asarray(value).shape), "dtype": str(np.asarray(value).dtype)}
        for key, value in sorted(data.non_tensor_batch.items())
    }
    return {
        "batch_size": len(data),
        "tensor_fields": tensor_fields,
        "non_tensor_fields": non_tensor_fields,
        "all_floating_tensors_finite": all_finite,
    }


def _unique_integer_values(data: DataProto, key: str) -> list[int]:
    if key not in data.non_tensor_batch:
        return []
    values = np.asarray(data.non_tensor_batch[key]).reshape(-1)
    return sorted({int(value) for value in values})


def summarize_rollout_trajectories(rollout: DataProto) -> list[dict[str, Any]]:
    """Build one auditable record per fixed, non-auto-reset environment lane."""

    terminated = rollout.batch["next.terminated"].bool()
    truncated = rollout.batch["next.truncated"].bool()
    success = rollout.batch["next.success"].bool()
    rewards = rollout.batch["next.reward"].float()
    batch_size, policy_slots, action_steps = terminated.shape[:3]
    records = []
    for lane in range(batch_size):
        flat_done = (terminated[lane] | truncated[lane]).reshape(-1)
        done_indices = torch.nonzero(flat_done, as_tuple=False).flatten()
        final_step = int(done_indices[0].item()) if len(done_indices) else int(flat_done.numel() - 1)
        record: dict[str, Any] = {
            "lane": lane,
            "policy_calls": min(policy_slots, final_step // action_steps + 1),
            "action_steps": final_step + 1,
            "return": float(rewards[lane].reshape(-1)[: final_step + 1].sum().item()),
            "success": bool(success[lane].reshape(-1)[: final_step + 1].any().item()),
            "terminated": bool(terminated[lane].reshape(-1)[: final_step + 1].any().item()),
            "truncated": bool(truncated[lane].reshape(-1)[: final_step + 1].any().item()),
        }
        for field in ("task_id", "eval_episode_id", "layout_id", "environment_seed", "policy_seed"):
            key = f"obs.{field}"
            if key in rollout.non_tensor_batch:
                record[field] = int(np.asarray(rollout.non_tensor_batch[key])[lane, 0])
        records.append(record)
    return records


def build_fpo_trajectory_audit(
    *,
    raw_contract: dict[str, Any],
    raw_semantics: dict[str, Any],
    actor_input: DataProto,
) -> dict[str, Any]:
    """Combine the env-loop and prepared-FPO views of one trajectory batch."""

    valids = actor_input.batch["info.valids"]
    action_valids = actor_input.batch["info.action_valids"]
    durations = action_valids.sum(dim=-1)
    valid_slots = int(valids.sum().item())
    total_slots = int(valids.numel())
    prepared_contract = summarize_dataproto_contract(actor_input)
    return {
        "raw_rollout": raw_contract,
        "prepared_actor_input": prepared_contract,
        "semantics": {
            **raw_semantics,
            "valid_policy_slots": valid_slots,
            "idle_policy_slots": total_slots - valid_slots,
            "total_policy_slots": total_slots,
            "valid_action_substeps": int(action_valids.sum().item()),
            "duration_min_valid": float(durations[valids.bool()].min().item()) if valid_slots else 0.0,
            "duration_max_valid": float(durations[valids.bool()].max().item()) if valid_slots else 0.0,
            "duration_sum_valid": float(durations[valids.bool()].sum().item()) if valid_slots else 0.0,
            "prepared_all_finite": prepared_contract["all_floating_tensors_finite"],
        },
    }


def prepare_fpo_actor_input(
    rollout: DataProto,
    rollout_end_obs: DataProto,
    *,
    trainer_config: GRFPOTrainerConfig,
    global_steps: int,
) -> DataProto:
    """Convert an env-loop window into chunk-level vanilla-FPO transitions.

    The env loop returns ``S`` aligned observation/action/feedback slots plus
    the observation after the final action as a separate ``rollout_end_obs``.
    This function builds the matching next observations and folds each chunk's
    low-level feedback into the rewards, discounts, and masks consumed by the
    FPO worker.
    """
    obs_prefix = f"{OBS_KEY}."
    next_obs_prefix = "next_obs."
    rollout_steps = int(rollout.batch["action.action"].shape[1])
    batch_size = len(rollout)

    # Align every tensor observation with its exact successor:
    #   obs      = [s0, s1, ..., s(S-1)]
    #   next_obs = [s1, s2, ..., sS]
    # Intermediate successors come from the next rollout slot; only sS needs
    # the separate final observation returned by EnvLoop.
    tensor_obs_keys = [key for key in rollout.batch.keys() if key.startswith(obs_prefix)]
    profile_obs_keys = [key for key in tensor_obs_keys if key[len(obs_prefix) :].startswith("profile.")]
    if profile_obs_keys:
        # Profiling fields describe collection infrastructure, not policy state.
        # Keep this defensive boundary even though EnvLoop strips them: batches
        # persisted by older workers or composed through TensorDict views must
        # not be aligned as next observations or reach the actor.
        rollout.batch = rollout.batch.exclude(*profile_obs_keys)
        tensor_obs_keys = [key for key in tensor_obs_keys if key not in profile_obs_keys]

    for obs_key in tensor_obs_keys:
        field = obs_key[len(obs_prefix) :]
        obs = rollout.batch[obs_key]
        end_obs = rollout_end_obs.batch[field]
        if obs.shape[:2] != (batch_size, rollout_steps) or end_obs.shape != (batch_size, *obs.shape[2:]):
            raise ValueError(
                f"FPO rollout observations must have shape [B, S, ...] and the final observation [B, ...], "
                f"got {obs_key}={tuple(obs.shape)} and {field}={tuple(end_obs.shape)}."
            )
        rollout.batch[f"{next_obs_prefix}{field}"] = torch.cat([obs[:, 1:], end_obs.unsqueeze(1)], dim=1)

    # Tasks, task ids, and other object-valued observations follow the same
    # temporal alignment as tensor observations.
    trajectory_metadata_fields = {
        "eval_episode_id",
        "layout_id",
        "environment_seed",
        "policy_seed",
        "group_id",
        "trajectory_id",
        "group_key",
        "rollout_policy_version",
    }
    for obs_key in [key for key in rollout.non_tensor_batch if key.startswith(obs_prefix)]:
        field = obs_key[len(obs_prefix) :]
        # These identify a whole trajectory rather than model inputs at the
        # successor state. Keep them on ``obs`` for audit/grouping and do not
        # manufacture a temporal next-observation field.
        if field in trajectory_metadata_fields:
            continue
        obs = rollout.non_tensor_batch[obs_key]
        end_obs = rollout_end_obs.non_tensor_batch[field]
        if obs.shape[:2] != (batch_size, rollout_steps) or end_obs.shape != (batch_size, *obs.shape[2:]):
            raise ValueError(
                f"FPO rollout observations must have shape [B, S, ...] and the final observation [B, ...], "
                f"got {obs_key}={obs.shape} and {field}={end_obs.shape}."
            )
        rollout.non_tensor_batch[f"{next_obs_prefix}{field}"] = np.concatenate(
            [obs[:, 1:], np.expand_dims(end_obs, axis=1)],
            axis=1,
        )

    # Environment feedback is emitted per executed low-level action, whereas
    # FPO treats one policy action chunk as one semi-MDP transition.
    reward_source = str(getattr(trainer_config, "group_reward_source", "binary_success"))
    if reward_source == "process_score" and "next.score" not in rollout.batch:
        raise KeyError("group_reward_source='process_score' requires rollout field 'next.score'.")
    trajectory_process_score = (
        _terminal_trajectory_scores(rollout)
        if "next.score" in rollout.batch
        else rollout.batch["next.success"].bool().flatten(start_dim=1).any(dim=1).float()
    )
    terminated_substeps = rollout.batch.pop("next.terminated").bool()
    truncated_substeps = rollout.batch.pop("next.truncated").bool()
    reward_substeps = rollout.batch.pop("next.reward").float()
    success_substeps = rollout.batch.pop("next.success").bool()
    rollout.batch.pop("next.score", None)
    done_substeps = terminated_substeps | truncated_substeps

    action_substeps = rollout.batch["action.action"].shape[2]
    native_action_horizon = rollout.batch["action.full_action"].shape[2]
    if reward_substeps.shape[2] != action_substeps:
        raise ValueError(
            f"Reward chunk length {reward_substeps.shape[2]} must match executed action length {action_substeps}."
        )
    if native_action_horizon < action_substeps:
        raise ValueError(
            f"Native action horizon {native_action_horizon} cannot be shorter than executed length {action_substeps}."
        )

    # Include the substep and policy chunk that end the episode, but mask all
    # feedback after the first termination/truncation.  ``auto_reset=False``
    # rollouts retain their fixed episode until the next explicit reset, so
    # this mask must carry across later policy chunks as well as across the
    # low-level actions inside the terminal chunk.
    chunk_dones = done_substeps.any(dim=2)
    valid_chunks = (chunk_dones.cumsum(dim=1) - chunk_dones.long()) == 0
    valid_action_substeps = (done_substeps.cumsum(dim=2) - done_substeps.long()) == 0
    valid_action_substeps &= valid_chunks.unsqueeze(-1)

    # Discount and sum low-level rewards within each action chunk. The step
    # penalty is charged once per policy decision rather than once per substep.
    substep_indices = torch.arange(action_substeps, device=reward_substeps.device, dtype=torch.float32)
    reward_discounts = float(trainer_config.gamma) ** substep_indices
    rewards = (reward_substeps * valid_action_substeps * reward_discounts).sum(dim=2)
    rewards -= float(trainer_config.step_penalty) * valid_chunks.float()
    dones = chunk_dones.float() * valid_chunks.float()

    # A chunk may stop early, so its semi-MDP duration is the number of actions
    # actually executed. Terminal chunks must not bootstrap V(next_obs), and
    # their GAE recursion must not cross into an auto-reset episode.
    executed_steps = valid_action_substeps.sum(dim=2)
    discounts = float(trainer_config.gamma) ** executed_steps
    gae_discounts = (float(trainer_config.gamma) * float(trainer_config.gae_lambda)) ** executed_steps
    discounts *= (1.0 - dones) * valid_chunks.float()
    gae_discounts *= (1.0 - dones) * valid_chunks.float()

    # Pi0 predicts its full native action horizon H, while the environment may
    # execute only K <= H actions. Preserve the H dimension so the worker can
    # exclude unexecuted CFM-loss positions when constructing the log ratio.
    action_valids = torch.zeros(
        *valid_action_substeps.shape[:2],
        native_action_horizon,
        device=valid_action_substeps.device,
        dtype=torch.float32,
    )
    action_valids[:, :, :action_substeps] = valid_action_substeps.float()

    rollout.batch["info.rewards"] = rewards
    rollout.batch["info.dones"] = dones
    rollout.batch["info.durations"] = executed_steps.to(torch.float32)
    rollout.batch["info.discounts"] = discounts
    rollout.batch["info.gae_discounts"] = gae_discounts
    # Slots after a fixed episode terminates are rollout padding, just like
    # padding later appended for distributed divisibility.  Neither kind may
    # contribute to value loss, advantage normalization, or PPO loss.
    rollout.batch["info.valids"] = valid_chunks.float()
    rollout.batch["info.action_valids"] = action_valids
    trajectory_outcome = success_substeps.flatten(start_dim=1).any(dim=1).float()
    trajectory_process_score = trajectory_process_score.to(device=trajectory_outcome.device, dtype=torch.float32)
    trajectory_process_score[trajectory_outcome.bool()] = 1.0
    rollout.batch["info.trajectory_outcome"] = trajectory_outcome
    rollout.batch["info.trajectory_process_score"] = trajectory_process_score
    rollout.meta_info.update(
        {
            "global_steps": global_steps,
            "gamma": float(trainer_config.gamma),
            "gae_lambda": float(trainer_config.gae_lambda),
            "global_token_num": [0] * len(rollout),
        }
    )
    return rollout


def concatenate_rollout_groups_with_padding(
    rollouts: list[DataProto],
    rollout_end_observations: list[DataProto],
) -> tuple[DataProto, DataProto]:
    """Concatenate complete groups whose policy-call counts differ.

    EnvLoop pads trajectories within one vectorized group to that group's
    longest episode. Independently collected groups can still have different
    ``S`` dimensions. Observation padding uses the exact terminal observation
    so the last real transition keeps its true successor. Action and feedback
    padding is zero and is masked by ``prepare_fpo_actor_input``.
    """

    if not rollouts or len(rollouts) != len(rollout_end_observations):
        raise ValueError(
            "Rollout groups and end observations must be non-empty and aligned: "
            f"rollouts={len(rollouts)}, end_observations={len(rollout_end_observations)}."
        )

    # A collection window can straddle a metadata-schema repair.  In
    # particular, older RoboDojo groups omitted ``suite_id`` from the final
    # observation even though ``obs.suite_id`` was present and constant in the
    # rollout.  DataProto.concat requires an identical field contract.  Fill a
    # missing terminal field only from that same group's last rollout
    # observation; never invent a value or copy one across groups.
    end_tensor_keys = {
        key
        for end_obs in rollout_end_observations
        for key in (end_obs.batch.keys() if end_obs.batch is not None else [])
    }
    end_non_tensor_keys = {
        key for end_obs in rollout_end_observations for key in end_obs.non_tensor_batch
    }
    normalized_end_observations: list[DataProto] = []
    for group_index, (rollout, end_obs) in enumerate(
        zip(rollouts, rollout_end_observations, strict=True)
    ):
        tensors = {key: end_obs.batch[key] for key in end_obs.batch.keys()}
        non_tensors = {key: np.asarray(value) for key, value in end_obs.non_tensor_batch.items()}
        for key in end_tensor_keys - set(tensors):
            rollout_key = f"{OBS_KEY}.{key}"
            if rollout_key not in rollout.batch:
                raise ValueError(
                    f"Terminal tensor field {key!r} is missing from group {group_index} and "
                    f"cannot be recovered from {rollout_key!r}."
                )
            tensors[key] = rollout.batch[rollout_key][:, -1].clone()
        for key in end_non_tensor_keys - set(non_tensors):
            rollout_key = f"{OBS_KEY}.{key}"
            if rollout_key not in rollout.non_tensor_batch:
                raise ValueError(
                    f"Terminal non-tensor field {key!r} is missing from group {group_index} and "
                    f"cannot be recovered from {rollout_key!r}."
                )
            non_tensors[key] = np.asarray(rollout.non_tensor_batch[rollout_key])[:, -1].copy()
        normalized_end_observations.append(
            DataProto.from_dict(
                tensors=tensors,
                non_tensors=non_tensors,
                meta_info=dict(end_obs.meta_info),
            )
        )

    rollout_end_observations = normalized_end_observations

    tensor_keys = set(rollouts[0].batch.keys())
    non_tensor_keys = set(rollouts[0].non_tensor_batch)
    max_steps = max(int(rollout.batch["action.action"].shape[1]) for rollout in rollouts)
    padded_rollouts: list[DataProto] = []

    for group_index, (rollout, end_obs) in enumerate(
        zip(rollouts, rollout_end_observations, strict=True)
    ):
        if set(rollout.batch.keys()) != tensor_keys or set(rollout.non_tensor_batch) != non_tensor_keys:
            raise ValueError(f"Rollout group {group_index} has a different field contract.")
        steps = int(rollout.batch["action.action"].shape[1])
        if steps <= 0:
            raise ValueError(f"Rollout group {group_index} has no policy-call slots.")
        batch_size = len(rollout)
        pad_steps = max_steps - steps

        tensors: dict[str, torch.Tensor] = {}
        for key in tensor_keys:
            value = rollout.batch[key]
            if value.shape[:2] != (batch_size, steps):
                raise ValueError(
                    f"Rollout tensor {key!r} in group {group_index} must start with [B, S]="
                    f"[{batch_size}, {steps}], got {tuple(value.shape)}."
                )
            if pad_steps == 0:
                tensors[key] = value
                continue
            if key.startswith(f"{OBS_KEY}."):
                field = key[len(OBS_KEY) + 1 :]
                terminal = end_obs.batch.get(field)
                if terminal is None:
                    # Profiling fields have no terminal value and are removed
                    # before the actor consumes the batch.
                    terminal = value[:, -1]
                if terminal.shape != (batch_size, *value.shape[2:]):
                    raise ValueError(
                        f"Terminal observation {field!r} for group {group_index} has shape "
                        f"{tuple(terminal.shape)}, expected {(batch_size, *value.shape[2:])}."
                    )
                padding = terminal.unsqueeze(1).expand(batch_size, pad_steps, *value.shape[2:])
            else:
                padding = torch.zeros(
                    (batch_size, pad_steps, *value.shape[2:]),
                    dtype=value.dtype,
                    device=value.device,
                )
            tensors[key] = torch.cat([value, padding], dim=1)

        non_tensors: dict[str, np.ndarray] = {}
        for key in non_tensor_keys:
            value = np.asarray(rollout.non_tensor_batch[key])
            if value.shape[:2] != (batch_size, steps):
                raise ValueError(
                    f"Rollout non-tensor {key!r} in group {group_index} must start with [B, S]="
                    f"[{batch_size}, {steps}], got {value.shape}."
                )
            if pad_steps == 0:
                non_tensors[key] = value
                continue
            terminal = None
            if key.startswith(f"{OBS_KEY}."):
                field = key[len(OBS_KEY) + 1 :]
                terminal = end_obs.non_tensor_batch.get(field)
            if terminal is None:
                terminal = value[:, -1]
            terminal = np.asarray(terminal)
            if terminal.shape != (batch_size, *value.shape[2:]):
                raise ValueError(
                    f"Terminal non-tensor observation for {key!r} in group {group_index} has "
                    f"shape {terminal.shape}, expected {(batch_size, *value.shape[2:])}."
                )
            padding = np.repeat(np.expand_dims(terminal, axis=1), pad_steps, axis=1)
            non_tensors[key] = np.concatenate([value, padding], axis=1)

        padded_rollouts.append(
            DataProto.from_dict(
                tensors=tensors,
                non_tensors=non_tensors,
                meta_info=dict(rollout.meta_info),
            )
        )

    return DataProto.concat(padded_rollouts), DataProto.concat(rollout_end_observations)


class GRFPORayTrainer:
    def __init__(self, trainer_config, cluster: TrainCluster, tracking_config: dict[str, Any]):
        self.cluster = cluster
        self.trainer_config: GRFPOTrainerConfig = instantiate(trainer_config)
        self.config = OmegaConf.create(tracking_config)
        _validate_fastwam_rollout_temperature(self.config)
        actor_config = self.config.cluster.actor_rollout_ref.actor
        model_fpo_config = self.config.cluster.actor_rollout_ref.model.adapter.fpo
        critic_enabled = bool(actor_config.value.enabled)
        model_value_enabled = bool(getattr(model_fpo_config, "value_enabled", False))
        if str(getattr(self.config.cluster.actor_rollout_ref.model, "native_architecture", "")) == "fastwam":
            if critic_enabled or model_value_enabled:
                raise ValueError("Fast-WAM FPO is critic-free; Value Head, value optimizer, and GAE are unsupported.")
        if critic_enabled != model_value_enabled:
            raise ValueError(
                "FPO critic configuration mismatch: actor.value.enabled and model.adapter.fpo.value_enabled must agree."
            )
        if self.trainer_config.advantage_estimator == "gae" and not critic_enabled:
            raise ValueError("GAE advantage estimation requires the FPO value head.")
        if self.trainer_config.advantage_estimator == "grpo_outcome" and critic_enabled:
            raise ValueError("grpo_outcome must run with the FPO critic/value head disabled.")
        if self.trainer_config.advantage_estimator == "grpo_outcome":
            if bool(actor_config.normalize_advantages):
                raise ValueError("grpo_outcome forbids second-stage minibatch advantage normalization.")
            if int(actor_config.value_only_updates) != 0 or float(actor_config.vf_coef) != 0:
                raise ValueError("grpo_outcome requires value_only_updates=0 and vf_coef=0.")
            if not bool(self.trainer_config.informative_group_sampling):
                raise ValueError("Production grpo_outcome requires informative_group_sampling=true.")
        self.critic_enabled = critic_enabled
        self.candidate_group_count = 0
        self.run_id = str(time.time_ns())

        if self.trainer_config.advantage_estimator == "grpo_outcome":
            env_worker = self.config.cluster.env.env_worker
            pipeline_stage_num = int(self.config.cluster.env.env_loop.pipeline_stage_num)
            concurrent_candidate_groups = int(self.trainer_config.concurrent_candidate_groups)
            groups_per_stage = _candidate_groups_per_stage(self.config, self.trainer_config)
            if concurrent_candidate_groups > pipeline_stage_num * groups_per_stage:
                raise ValueError(
                    "concurrent_candidate_groups cannot exceed the configured worker-stage slots, "
                    f"got {concurrent_candidate_groups} > "
                    f"{pipeline_stage_num}*{groups_per_stage}."
                )
            if str(getattr(self.trainer_config, "candidate_group_partition", "stage")) == "worker_stage":
                envs_per_worker = int(env_worker.num_envs)
                group_size = int(self.trainer_config.group_size)
                if envs_per_worker < group_size or envs_per_worker % group_size != 0:
                    raise ValueError(
                        "worker_stage grouping requires every vectorized simulator to own one "
                        "or more complete groups: "
                        f"num_envs={envs_per_worker}, group_size={group_size}."
                    )
                if concurrent_candidate_groups % groups_per_stage != 0:
                    raise ValueError(
                        "worker_stage concurrent_candidate_groups must contain complete all-worker "
                        f"stages: concurrency={concurrent_candidate_groups}, "
                        f"groups_per_stage={groups_per_stage}."
                    )
                if not bool(getattr(self.config.cluster.env.env_loop, "rollout_partition_by_env_worker", False)):
                    raise ValueError(
                        "worker_stage grouping requires env_loop.rollout_partition_by_env_worker=true "
                        "to keep Fast-WAM inference at B=G per request."
                    )
                rollout_partition_size = int(getattr(self.config.cluster.env.env_loop, "rollout_partition_size", 0))
                if envs_per_worker > group_size and rollout_partition_size != group_size:
                    raise ValueError(
                        "A worker-local vector containing multiple groups requires "
                        "env_loop.rollout_partition_size=group_size so Fast-WAM still receives "
                        f"one group per request: num_envs={envs_per_worker}, "
                        f"partition_size={rollout_partition_size}, group_size={group_size}."
                    )
            action_execution = env_worker.action_execution
            if bool(env_worker.auto_reset):
                raise ValueError("Phase 6D grpo_outcome requires auto_reset=false.")
            if str(action_execution.mode) != "serial":
                raise ValueError("Phase 6D grpo_outcome requires serial action execution.")
            if bool(action_execution.interpolation.enable) or bool(action_execution.serial_smoothing.enable):
                raise ValueError("Phase 6D grpo_outcome forbids interpolation and serial smoothing.")

    @staticmethod
    def _compact_group_rollout_metrics(
        wave_metrics: list[dict[str, float]],
        *,
        group_size: int,
        wall_seconds: float,
        aggregate_share: float = 1.0,
    ) -> dict[str, float]:
        if not 0 < aggregate_share <= 1:
            raise ValueError(f"aggregate_share must be in (0, 1], got {aggregate_share}.")
        sum_keys = {
            "count/executed_low_level_steps",
            "count/idle_policy_lane_slots",
            "count/policy_lane_slots",
            "count/env_loop_effective_steps",
            "count/env_simulator_restarts",
            "timing_s/env_loop_stage_wall_max",
            "timing_s/env_loop_total",
            "timing_s/env_loop_run",
            "timing_s/env_loop_finish_rollout",
            "timing_s/env_loop_collate_trajectories",
            "timing_s/env_loop_collate_last_obs",
            "timing_s/env_loop_reset_wait",
            "timing_s/env_simulator_restart",
            "timing_s/fastwam_inference_sum",
            "timing_s/switch_to_rollout",
            "timing_s/switch_to_train",
            "timing_s/rollout_once_total",
            "timing_s/rollout_generate_sequences",
            "timing_s/rollout_reset_dispatch",
            "timing_s/rollout_trajectory_records",
            "timing_s/rollout_dataset_collection",
            "timing_s/rollout_unaccounted",
        }
        maximum_keys = {
            "count/env_worker_ranks",
            "count/rollout_worker_ranks",
            "memory_bytes/fastwam_gpu_peak_memory_allocated_bytes",
            "memory_bytes/fastwam_gpu_peak_memory_reserved_bytes",
        }
        compact: dict[str, float] = {}
        for key in sum_keys:
            compact[key] = float(sum(float(metrics.get(key, 0.0)) for metrics in wave_metrics) * aggregate_share)
        for key in maximum_keys:
            compact[key] = float(max((float(metrics.get(key, 0.0)) for metrics in wave_metrics), default=0.0))
        compact["timing_s/group_rollout_wall"] = float(wall_seconds)
        compact["throughput/candidate_trajectories_per_hour"] = group_size * 3600.0 / max(wall_seconds, 1e-12)
        compact["fraction/idle_policy_lane_slots"] = compact["count/idle_policy_lane_slots"] / max(
            compact["count/policy_lane_slots"], 1.0
        )
        return compact

    def _collect_candidate_group(self, *, intended_update: int, candidate_attempt: int):
        group_size = int(self.trainer_config.group_size)
        rollout_parts: list[DataProto] = []
        end_obs_parts: list[DataProto] = []
        wave_metrics: list[dict[str, float]] = []
        collected = 0
        group_start = time.perf_counter()
        while collected < group_size:
            rollout_output, rollout_end_obs, _datasets, metrics = self.cluster.rollout(async_rollout=False)
            if len(rollout_output) <= 0 or collected + len(rollout_output) > group_size:
                raise ValueError(
                    f"Physical rollout batch cannot tile G={group_size}: collected={collected}, "
                    f"next_batch={len(rollout_output)}."
                )
            rollout_parts.append(rollout_output)
            end_obs_parts.append(rollout_end_obs)
            wave_metrics.append(metrics)
            collected += len(rollout_output)

        self.candidate_group_count += 1
        group_id = f"run-{self.run_id}/policy-{self.global_steps:04d}/candidate-{self.candidate_group_count:04d}"
        rollout = DataProto.concat(rollout_parts)
        rollout_end_obs = DataProto.concat(end_obs_parts)
        group_record = validate_same_condition_group(
            rollout,
            group_size=group_size,
            group_id=group_id,
            rollout_policy_version=self.global_steps,
            reward_source=self.trainer_config.group_reward_source,
            informative_group_criterion=self.trainer_config.informative_group_criterion,
            min_partial_trajectories_for_score_only_group=(
                self.trainer_config.min_partial_trajectories_for_score_only_group
            ),
            partial_score_threshold=self.trainer_config.partial_score_threshold,
        )
        group_record.update(
            {
                "intended_update": int(intended_update),
                "candidate_attempt": int(candidate_attempt),
                "wave_count": len(rollout_parts),
                "physical_batch_size": len(rollout_parts[0]),
                "rollout_action_latent_scale": _fastwam_action_latent_scale(self.config),
                "rollout_seed_mode": _fastwam_rollout_seed_mode(self.config),
                "trajectories": summarize_rollout_trajectories(rollout),
                "rollout_metrics": self._compact_group_rollout_metrics(
                    wave_metrics,
                    group_size=group_size,
                    wall_seconds=time.perf_counter() - group_start,
                ),
                "pipeline_schedule_slots_reserved": 1,
                "unused_pipeline_stage_count": 0,
                "next_policy_seed_cursor": max(group_record["policy_seeds"]) + 1,
            }
        )
        return rollout, rollout_end_obs, group_record

    def _collect_candidate_groups(
        self,
        *,
        intended_update: int,
        first_candidate_attempt: int,
        group_count: int,
    ) -> list[tuple[DataProto, DataProto, dict[str, Any]]]:
        """Collect complete candidate groups in one framework-native stage batch.

        Each active pipeline stage owns one or more same-condition G-sized
        groups, partitioned contiguously within each EnvWorker vector. One
        TrainCluster rollout call therefore freezes/synchronizes theta_old once,
        executes all requested stages, and returns them in stage-major order.
        """

        group_size = int(self.trainer_config.group_size)
        pipeline_stage_num = int(self.config.cluster.env.env_loop.pipeline_stage_num)
        groups_per_stage = _candidate_groups_per_stage(self.config, self.trainer_config)
        if group_count % groups_per_stage != 0:
            raise ValueError(
                "Candidate group count must consist of complete worker-stage rows: "
                f"group_count={group_count}, groups_per_stage={groups_per_stage}."
            )
        active_pipeline_stages = group_count // groups_per_stage
        if not 1 <= active_pipeline_stages <= pipeline_stage_num:
            raise ValueError(
                "Requested candidate groups exceed the configured pipeline: "
                f"groups={group_count}, active_stages={active_pipeline_stages}, "
                f"pipeline_stages={pipeline_stage_num}."
            )

        collection_batch_start = time.perf_counter()
        rollout_output, rollout_end_obs, _datasets, metrics = self.cluster.rollout(
            async_rollout=False,
            active_pipeline_stages=active_pipeline_stages,
        )
        collection_batch_wall = time.perf_counter() - collection_batch_start
        expected_trajectories = group_count * group_size
        if len(rollout_output) != expected_trajectories or len(rollout_end_obs) != expected_trajectories:
            raise ValueError(
                "Concurrent candidate collection must return one complete group per active stage: "
                f"expected={expected_trajectories}, rollout={len(rollout_output)}, "
                f"end_obs={len(rollout_end_obs)}."
            )
        rollout_groups = rollout_output.chunk(group_count)
        end_obs_groups = rollout_end_obs.chunk(group_count)
        collection_batch_id = (
            f"run-{self.run_id}/policy-{self.global_steps:04d}/candidate-batch-{self.candidate_group_count + 1:04d}"
        )
        per_group_metrics = self._compact_group_rollout_metrics(
            [metrics],
            group_size=group_size,
            wall_seconds=collection_batch_wall / group_count,
            aggregate_share=1.0 / group_count,
        )
        per_group_metrics.update(
            {
                "timing_s/concurrent_batch_rollout_wall": float(collection_batch_wall),
                "throughput/concurrent_candidate_groups_per_hour": group_count
                * 3600.0
                / max(collection_batch_wall, 1e-12),
                "count/concurrent_candidate_groups": float(group_count),
            }
        )

        results: list[tuple[DataProto, DataProto, dict[str, Any]]] = []
        for batch_index, (rollout, rollout_end) in enumerate(zip(rollout_groups, end_obs_groups, strict=True)):
            candidate_attempt = first_candidate_attempt + batch_index
            self.candidate_group_count += 1
            group_id = f"run-{self.run_id}/policy-{self.global_steps:04d}/candidate-{self.candidate_group_count:04d}"
            group_record = validate_same_condition_group(
                rollout,
                group_size=group_size,
                group_id=group_id,
                rollout_policy_version=self.global_steps,
                reward_source=self.trainer_config.group_reward_source,
                informative_group_criterion=self.trainer_config.informative_group_criterion,
                min_partial_trajectories_for_score_only_group=(
                    self.trainer_config.min_partial_trajectories_for_score_only_group
                ),
                partial_score_threshold=self.trainer_config.partial_score_threshold,
            )
            group_record.update(
                {
                    "intended_update": int(intended_update),
                    "candidate_attempt": int(candidate_attempt),
                    "wave_count": 1,
                    "physical_batch_size": len(rollout),
                    "rollout_action_latent_scale": _fastwam_action_latent_scale(self.config),
                    "rollout_seed_mode": _fastwam_rollout_seed_mode(self.config),
                    "concurrent_collection_batch_id": collection_batch_id,
                    "concurrent_collection_batch_index": int(batch_index),
                    "concurrent_candidate_groups": int(group_count),
                    "active_pipeline_stages": int(active_pipeline_stages),
                    "candidate_group_partition": str(
                        getattr(self.trainer_config, "candidate_group_partition", "stage")
                    ),
                    "groups_per_pipeline_stage": int(groups_per_stage),
                    "trajectories": summarize_rollout_trajectories(rollout),
                    "rollout_metrics": dict(per_group_metrics),
                }
            )
            results.append((rollout, rollout_end, group_record))

        batch_policy_seeds = [
            int(seed) for _rollout, _rollout_end, group_record in results for seed in group_record["policy_seeds"]
        ]
        first_policy_seed = min(batch_policy_seeds)
        expected_policy_seeds = list(range(first_policy_seed, first_policy_seed + group_count * group_size))
        if sorted(batch_policy_seeds) != expected_policy_seeds:
            raise RuntimeError(
                "Concurrent stage-major groups must occupy contiguous G-sized policy-seed blocks: "
                f"expected={expected_policy_seeds}, got={sorted(batch_policy_seeds)}."
            )
        # EnvWorker reset initializes every configured simulator stage, even
        # when the bounded collector activates only a prefix. Reserve those
        # unrolled schedule slots explicitly so a process restart resumes from
        # the same condition/seed cursor as an uninterrupted run.
        reserved_group_slots = pipeline_stage_num * groups_per_stage
        next_policy_seed_cursor = first_policy_seed + reserved_group_slots * group_size
        for _rollout, _rollout_end, group_record in results:
            group_record.update(
                {
                    "pipeline_schedule_slots_reserved": int(reserved_group_slots),
                    "unused_pipeline_stage_count": int(pipeline_stage_num - active_pipeline_stages),
                    "unused_candidate_group_slot_count": int(reserved_group_slots - group_count),
                    "next_policy_seed_cursor": int(next_policy_seed_cursor),
                }
            )
        return results

    def _prepare_actor_input(
        self,
        rollout_output: DataProto,
        rollout_end_obs: DataProto,
        *,
        group_record: dict[str, Any] | None = None,
        group_records: list[dict[str, Any]] | None = None,
    ) -> DataProto:
        self.last_training_trajectories = summarize_rollout_trajectories(rollout_output)
        raw_contract = summarize_dataproto_contract(rollout_output)
        raw_semantics = {
            "action_shape": list(rollout_output.batch["action.action"].shape),
            "full_action_shape": list(rollout_output.batch["action.full_action"].shape),
            "reward_shape": list(rollout_output.batch["next.reward"].shape),
            "terminated_shape": list(rollout_output.batch["next.terminated"].shape),
            "truncated_shape": list(rollout_output.batch["next.truncated"].shape),
            "success_shape": list(rollout_output.batch["next.success"].shape),
            "task_ids": _unique_integer_values(rollout_output, "obs.task_id"),
            "eval_episode_ids": _unique_integer_values(rollout_output, "obs.eval_episode_id"),
            "terminated_substeps": int(rollout_output.batch["next.terminated"].sum().item()),
            "truncated_substeps": int(rollout_output.batch["next.truncated"].sum().item()),
            "successful_substeps": int(rollout_output.batch["next.success"].sum().item()),
        }
        actor_input = prepare_fpo_actor_input(
            rollout_output,
            rollout_end_obs,
            trainer_config=self.trainer_config,
            global_steps=self.global_steps,
        )
        if self.trainer_config.advantage_estimator == "grpo_outcome":
            if (group_record is None) == (group_records is None):
                raise ValueError("grpo_outcome requires exactly one accepted group record or record list.")
            self.last_group_diagnostics = apply_grpo_outcome_advantages(
                actor_input,
                group_record=group_record,
                group_records=group_records,
                group_size=int(self.trainer_config.group_size),
                reward_source=self.trainer_config.group_reward_source,
                accepted_group_reduction=self.trainer_config.accepted_group_reduction,
            )
        elif group_record is not None or group_records is not None:
            raise ValueError("GAE mode must not receive GRPO group metadata.")
        self.last_trajectory_audit = build_fpo_trajectory_audit(
            raw_contract=raw_contract,
            raw_semantics=raw_semantics,
            actor_input=actor_input,
        )
        module_logger.warning(
            "FPO trajectory contract audit step=%d %s",
            self.global_steps,
            json.dumps(self.last_trajectory_audit, sort_keys=True),
        )
        padded = pad_dataproto_to_divisor_with_valid_mask(
            actor_input,
            int(self.cluster.actor_worker_group.world_size),
            valid_key="info.valids",
        )
        if self.trainer_config.advantage_estimator == "grpo_outcome":
            padded.batch["fpo.advantages"] *= padded.batch["info.valids"]
            padded.batch["fpo.loss_weights"] *= padded.batch["info.valids"]
        return padded

    def _policy_identity(self, global_step: int) -> dict[str, Any]:
        model = self.config.cluster.actor_rollout_ref.model
        return {
            "immutable_base_path": str(model.path),
            "immutable_base_sha256": str(model.adapter.checkpoint_sha256),
            "trainer_checkpoint_step": int(global_step),
        }

    def _append_evidence(self, record: dict[str, Any], *, candidate: bool = False) -> None:
        path_value = (
            getattr(self.trainer_config, "candidate_evidence_jsonl", None)
            if candidate
            else getattr(self.trainer_config, "evidence_jsonl", None)
        )
        if candidate and not path_value:
            path_value = getattr(self.trainer_config, "evidence_jsonl", None)
        if not path_value:
            return
        path = Path(path_value)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")

    def _read_evidence(self, *, candidate: bool = False) -> list[dict[str, Any]]:
        path_value = (
            getattr(self.trainer_config, "candidate_evidence_jsonl", None)
            if candidate
            else getattr(self.trainer_config, "evidence_jsonl", None)
        )
        if candidate and not path_value:
            path_value = getattr(self.trainer_config, "evidence_jsonl", None)
        if not path_value:
            return []
        path = Path(path_value)
        if not path.exists():
            return []
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise TypeError(f"Evidence line {line_number} must be a JSON object.")
                records.append(record)
        return records

    def _candidate_spool_root(self) -> Path | None:
        value = getattr(self.trainer_config, "candidate_spool_dir", None)
        return Path(value).expanduser().resolve() if value else None

    def _candidate_resume_state_path(self) -> Path | None:
        value = getattr(self.trainer_config, "candidate_resume_state_path", None)
        return Path(value).expanduser().resolve() if value else None

    def _persist_accepted_candidate(
        self,
        rollout: DataProto,
        rollout_end_obs: DataProto,
        group_record: dict[str, Any],
    ) -> None:
        """Atomically spool one accepted group before advertising it in JSONL."""

        root = self._candidate_spool_root()
        if root is None:
            return
        intended_update = int(group_record["intended_update"])
        policy_version = int(group_record["rollout_policy_version"])
        attempt = int(group_record["candidate_attempt"])
        digest = hashlib.sha256(str(group_record["group_id"]).encode("utf-8")).hexdigest()[:12]
        window = root / f"update_{intended_update:04d}_theta_{policy_version:04d}"
        final_dir = window / f"candidate_{attempt:04d}_{digest}"
        tmp_dir = window / f".{final_dir.name}.tmp-{os.getpid()}-{time.time_ns()}"
        window.mkdir(parents=True, exist_ok=True)
        if final_dir.exists():
            raise FileExistsError(f"Candidate spool destination already exists: {final_dir}")
        tmp_dir.mkdir()
        try:
            rollout.save_to_disk(tmp_dir / "rollout.pkl")
            rollout_end_obs.save_to_disk(tmp_dir / "rollout_end_obs.pkl")
            metadata = {
                "group_id": str(group_record["group_id"]),
                "intended_update": intended_update,
                "rollout_policy_version": policy_version,
                "candidate_attempt": attempt,
                "group_size": int(self.trainer_config.group_size),
                "immutable_base_path": str(self.config.cluster.actor_rollout_ref.model.path),
                "immutable_base_sha256": str(
                    self.config.cluster.actor_rollout_ref.model.adapter.checkpoint_sha256
                ),
            }
            (tmp_dir / "metadata.json").write_text(
                json.dumps(metadata, sort_keys=True, indent=2) + "\n", encoding="utf-8"
            )
            (tmp_dir / "COMMITTED").touch()
            os.replace(tmp_dir, final_dir)
        except BaseException:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise
        group_record["candidate_spool_path"] = str(final_dir)

    def _load_spooled_candidate(
        self, group_record: dict[str, Any]
    ) -> tuple[DataProto, DataProto, dict[str, Any]]:
        path_value = group_record.get("candidate_spool_path")
        if not path_value:
            raise RuntimeError(
                f"Accepted resumed group {group_record.get('group_id')} has no candidate_spool_path."
            )
        root = self._candidate_spool_root()
        if root is None:
            raise RuntimeError("Cannot load accepted candidate because candidate spooling is disabled.")
        spool_dir = Path(path_value).expanduser().resolve()
        if root != spool_dir and root not in spool_dir.parents:
            raise RuntimeError(f"Candidate spool path escapes configured root: {spool_dir}")
        if not (spool_dir / "COMMITTED").is_file():
            raise RuntimeError(f"Candidate spool is not atomically committed: {spool_dir}")
        metadata = json.loads((spool_dir / "metadata.json").read_text(encoding="utf-8"))
        expected = {
            "group_id": str(group_record["group_id"]),
            "intended_update": int(group_record["intended_update"]),
            "rollout_policy_version": int(group_record["rollout_policy_version"]),
            "candidate_attempt": int(group_record["candidate_attempt"]),
            "group_size": int(self.trainer_config.group_size),
            "immutable_base_path": str(self.config.cluster.actor_rollout_ref.model.path),
            "immutable_base_sha256": str(
                self.config.cluster.actor_rollout_ref.model.adapter.checkpoint_sha256
            ),
        }
        if metadata != expected:
            raise RuntimeError(
                "Candidate spool policy/group identity mismatch: "
                f"expected={expected}, stored={metadata}."
            )
        rollout = DataProto.load_from_disk(spool_dir / "rollout.pkl")
        rollout_end_obs = DataProto.load_from_disk(spool_dir / "rollout_end_obs.pkl")
        if len(rollout) != expected["group_size"] or len(rollout_end_obs) != expected["group_size"]:
            raise RuntimeError(
                f"Candidate spool must contain exactly G={expected['group_size']} rows: "
                f"rollout={len(rollout)}, end_obs={len(rollout_end_obs)}."
            )
        return rollout, rollout_end_obs, group_record

    def _write_candidate_resume_state(
        self,
        *,
        rollout_policy_version: int,
        intended_update: int,
        next_candidate_attempt: int,
        candidate_schedule_slot: int,
    ) -> None:
        path = self._candidate_resume_state_path()
        if path is None:
            return
        groups_per_wave = int(self.config.cluster.env.env_loop.pipeline_stage_num) * _candidate_groups_per_stage(
            self.config, self.trainer_config
        )
        if candidate_schedule_slot % groups_per_wave != 0:
            raise RuntimeError(
                "Candidate schedule cursor must end on a complete pipeline wave: "
                f"slot={candidate_schedule_slot}, groups_per_wave={groups_per_wave}."
            )
        envs_per_worker = int(self.config.cluster.env.env_worker.num_envs)
        payload = {
            "rollout_policy_version": int(rollout_policy_version),
            "intended_update": int(intended_update),
            "next_candidate_attempt": int(next_candidate_attempt),
            "candidate_schedule_slot": int(candidate_schedule_slot),
            "train_case_cursor_start": int(candidate_schedule_slot // groups_per_wave * envs_per_worker),
            "immutable_base_path": str(self.config.cluster.actor_rollout_ref.model.path),
            "immutable_base_sha256": str(self.config.cluster.actor_rollout_ref.model.adapter.checkpoint_sha256),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
        tmp_path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp_path, path)

    def _read_candidate_resume_state(self) -> dict[str, Any] | None:
        path = self._candidate_resume_state_path()
        if path is None or not path.exists():
            return None
        state = json.loads(path.read_text(encoding="utf-8"))
        expected_path = str(self.config.cluster.actor_rollout_ref.model.path)
        expected_sha = str(self.config.cluster.actor_rollout_ref.model.adapter.checkpoint_sha256)
        if state.get("immutable_base_path") != expected_path or state.get("immutable_base_sha256") != expected_sha:
            raise RuntimeError("Candidate resume state belongs to a different immutable base policy.")
        return state

    def _cleanup_candidate_spool_window(self, *, intended_update: int, rollout_policy_version: int) -> None:
        root = self._candidate_spool_root()
        if root is None:
            return
        window = (root / f"update_{intended_update:04d}_theta_{rollout_policy_version:04d}").resolve()
        if root != window and root not in window.parents:
            raise RuntimeError(f"Refusing to clean candidate spool outside configured root: {window}")
        if window.is_dir():
            shutil.rmtree(window)

    def _resume_candidate_records(self, *, intended_update: int) -> list[dict[str, Any]]:
        """Recover the unfinished bounded candidate window after a restart."""

        records: list[dict[str, Any]] = []
        for record in self._read_evidence(candidate=True):
            same_window = (
                int(record.get("intended_update", -1)) == intended_update
                and int(record.get("rollout_policy_version", -1)) == self.global_steps
            )
            if not same_window:
                continue
            if record.get("record_type") in {"no_informative_group", "collection_window_closed"}:
                # That six-candidate sampling iteration closed without an
                # optimizer step. A subsequent iteration at the same theta
                # receives a fresh bounded window.
                records = []
            elif record.get("record_type") == "candidate_group":
                records.append(record)
        return records

    def _fit_group_relative(self, logger) -> None:
        total_steps = int(self.trainer_config.total_training_steps)
        max_candidates = int(self.trainer_config.max_candidate_groups_per_update)
        target_groups = int(self.trainer_config.accepted_groups_per_update)
        min_partial_groups = int(self.trainer_config.min_accepted_groups_for_partial_update)
        initial_candidates = int(self.trainer_config.initial_candidate_groups_per_update)
        refill_threshold = int(self.trainer_config.accepted_groups_before_optional_refill)
        progress = tqdm(total=total_steps, initial=self.global_steps, desc="Group-Relative FPO training")
        sampling_iteration = 0
        resume_state = self._read_candidate_resume_state()
        candidate_schedule_slot = int(resume_state["candidate_schedule_slot"]) if resume_state else 0
        if resume_state and int(resume_state["rollout_policy_version"]) > self.global_steps:
            raise RuntimeError(
                "Candidate resume state is newer than the loaded actor checkpoint: "
                f"state_theta={resume_state['rollout_policy_version']}, loaded_theta={self.global_steps}."
            )
        configured_train_cursor = int(
            getattr(self.config.cluster.env.env_worker.simulator.robodojo, "train_case_cursor_start", 0)
        )
        if resume_state and configured_train_cursor != int(resume_state["train_case_cursor_start"]):
            raise RuntimeError(
                "Launcher did not restore the RoboDojo train cursor from candidate resume state: "
                f"configured={configured_train_cursor}, expected={resume_state['train_case_cursor_start']}."
            )
        historical_candidates = [
            record for record in self._read_evidence(candidate=True) if record.get("record_type") == "candidate_group"
        ]
        if historical_candidates and self._candidate_spool_root() is not None and resume_state is None:
            raise RuntimeError("Candidate evidence exists without its required durable resume-state file.")
        self.candidate_group_count = len(historical_candidates)
        cumulative_candidate_groups = len(historical_candidates)
        cumulative_collected_trajectories = cumulative_candidate_groups * int(self.trainer_config.group_size)
        cumulative_all_failure_groups = sum(
            int(record.get("rejection_reason") == "all_failure") for record in historical_candidates
        )
        cumulative_all_success_groups = sum(
            int(record.get("rejection_reason") == "all_success") for record in historical_candidates
        )
        cumulative_mixed_groups = sum(int(bool(record.get("mixed"))) for record in historical_candidates)
        cumulative_informative_groups = sum(
            int(bool(record.get("informative", record.get("mixed", False)))) for record in historical_candidates
        )
        module_logger.warning(
            "Group-Relative FPO startup: advantage_source=grpo_outcome critic_enabled=false "
            "group_size=%d accepted_groups_per_update=%d min_partial_groups=%d "
            "initial_candidate_groups=%d optional_refill_threshold=%d candidate_budget=%d "
            "concurrent_candidate_groups=%d pipeline_stage_num=%d candidate_group_partition=%s "
            "informative_group_sampling=true reward_source=%s informative_criterion=%s "
            "min_partial_score_only=%d fpo_estimator=fpo_uniform_unweighted "
            "ratio_type=%s trust_region=%s async_rollout=false",
            int(self.trainer_config.group_size),
            target_groups,
            min_partial_groups,
            initial_candidates,
            refill_threshold,
            max_candidates,
            int(getattr(self.trainer_config, "concurrent_candidate_groups", 1)),
            int(self.config.cluster.env.env_loop.pipeline_stage_num),
            str(getattr(self.trainer_config, "candidate_group_partition", "stage")),
            self.trainer_config.group_reward_source,
            self.trainer_config.informative_group_criterion,
            self.trainer_config.min_partial_trajectories_for_score_only_group,
            str(self.config.cluster.actor_rollout_ref.actor.get("fpo_variant", "vanilla")),
            str(self.config.cluster.actor_rollout_ref.actor.get("trust_region_mode", "ppo")),
        )

        try:
            while self.global_steps < total_steps:
                sampling_iteration += 1
                intended_update = self.global_steps + 1
                collection_attempt_id = (
                    f"run-{self.run_id}/policy-{self.global_steps:04d}/collection-{sampling_iteration:04d}"
                )
                resumed_records = self._resume_candidate_records(intended_update=intended_update)
                candidate_records: list[dict[str, Any]] = list(resumed_records)
                accepted: list[tuple[DataProto, DataProto, dict[str, Any]]] = [
                    self._load_spooled_candidate(record)
                    for record in resumed_records
                    if bool(record.get("selected_for_training", record.get("accepted", False)))
                ]
                sampling_start = time.perf_counter()
                candidate_attempt = max(
                    (int(record["candidate_attempt"]) for record in resumed_records), default=0
                ) + 1
                if resumed_records:
                    module_logger.warning(
                        "Resuming preempted candidate window: theta_old=%d intended_update=%d "
                        "candidates=%d accepted_spooled=%d next_attempt=%d schedule_slot=%d",
                        self.global_steps,
                        intended_update,
                        len(resumed_records),
                        len(accepted),
                        candidate_attempt,
                        candidate_schedule_slot,
                    )
                collection_lifecycle_metrics = self.cluster.begin_rollout_collection()
                try:
                    while candidate_attempt <= max_candidates and len(accepted) < target_groups:
                        schedule = list(self.config.cluster.env.env_worker.simulator.robodojo.train_layout_schedule)
                        remaining_candidate_budget = max_candidates - candidate_attempt + 1
                        remaining_accepted_slots = target_groups - len(accepted)
                        configured_concurrency = int(getattr(self.trainer_config, "concurrent_candidate_groups", 1))
                        groups_per_stage = _candidate_groups_per_stage(self.config, self.trainer_config)
                        collection_group_count = _quantized_candidate_batch_size(
                            configured_concurrency=configured_concurrency,
                            remaining_candidate_budget=remaining_candidate_budget,
                            remaining_accepted_slots=remaining_accepted_slots,
                            groups_per_stage=groups_per_stage,
                        )
                        if collection_group_count == 0:
                            module_logger.warning(
                                "Remaining candidate budget cannot fund one atomic worker-stage "
                                "batch: remaining_budget=%d groups_per_stage=%d; stopping collection.",
                                remaining_candidate_budget,
                                groups_per_stage,
                            )
                            break
                        planned_layouts = [
                            int(schedule[(candidate_schedule_slot + offset) % len(schedule)])
                            for offset in range(collection_group_count)
                        ]
                        planned_tasks = _candidate_task_assignments(
                            self.config,
                            self.trainer_config,
                            group_count=collection_group_count,
                        )
                        module_logger.warning(
                            "Starting candidate group batch: planned_tasks=%s planned_layouts=%s groups=%d "
                            "trajectories_per_group=%d physical_lanes_per_worker=%d G=%d theta_old=%d "
                            "candidates=%d-%d/%d accepted=%d/%d",
                            [task_name for task_name, _task_id in planned_tasks],
                            planned_layouts,
                            collection_group_count,
                            int(self.trainer_config.group_size),
                            int(self.config.cluster.env.env_worker.num_envs),
                            int(self.trainer_config.group_size),
                            self.global_steps,
                            candidate_attempt,
                            candidate_attempt + collection_group_count - 1,
                            max_candidates,
                            len(accepted),
                            target_groups,
                        )
                        pipeline_stage_num = int(self.config.cluster.env.env_loop.pipeline_stage_num)
                        if pipeline_stage_num > 1 or groups_per_stage > 1:
                            collected_groups = self._collect_candidate_groups(
                                intended_update=intended_update,
                                first_candidate_attempt=candidate_attempt,
                                group_count=collection_group_count,
                            )
                        else:
                            if collection_group_count != 1:
                                raise RuntimeError("Concurrent candidate collection requires pipeline_stage_num > 1.")
                            collected_groups = [
                                self._collect_candidate_group(
                                    intended_update=intended_update,
                                    candidate_attempt=candidate_attempt,
                                )
                            ]

                        if len(collected_groups) != collection_group_count:
                            raise RuntimeError(
                                f"Requested {collection_group_count} candidate groups, got {len(collected_groups)}."
                            )
                        for batch_index, (rollout, rollout_end_obs, group_record) in enumerate(collected_groups):
                            expected_layout = planned_layouts[batch_index]
                            actual_layout = int(group_record["group_key"]["layout_id"])
                            if actual_layout != expected_layout:
                                raise RuntimeError(
                                    "Candidate condition schedule diverged from its preregistered stage slot: "
                                    f"schedule_slot={candidate_schedule_slot + batch_index}, "
                                    f"expected_layout={expected_layout}, actual_layout={actual_layout}."
                                )
                            expected_task_name, expected_task_id = planned_tasks[batch_index]
                            actual_task_id = int(group_record["group_key"]["task_id"])
                            if actual_task_id != expected_task_id:
                                raise RuntimeError(
                                    "Candidate task assignment diverged from its persistent worker-stage slot: "
                                    f"batch_index={batch_index}, expected_task={expected_task_name!r}, "
                                    f"expected_task_id={expected_task_id}, actual_task_id={actual_task_id}."
                                )
                            group_record["task_name"] = expected_task_name
                            group_record["candidate_schedule_slot"] = int(candidate_schedule_slot + batch_index)
                            cumulative_candidate_groups += 1
                            cumulative_collected_trajectories += int(self.trainer_config.group_size)
                            group_record["collection_attempt_id"] = collection_attempt_id
                            group_record["policy_identity"] = self._policy_identity(self.global_steps)
                            selected_for_training = bool(group_record["informative"] and len(accepted) < target_groups)
                            group_record["accepted"] = selected_for_training
                            group_record["selected_for_training"] = selected_for_training
                            if group_record["mixed"]:
                                cumulative_mixed_groups += 1
                            if group_record["informative"]:
                                cumulative_informative_groups += 1
                                if not selected_for_training:
                                    group_record["selection_reason"] = "target_filled_by_concurrent_sibling"
                            elif group_record["successes"] == 0:
                                cumulative_all_failure_groups += 1
                                group_record["rejection_reason"] = group_record["informative_reason"]
                            else:
                                cumulative_all_success_groups += 1
                                group_record["rejection_reason"] = "all_success"
                            if selected_for_training:
                                self._persist_accepted_candidate(rollout, rollout_end_obs, group_record)
                            candidate_records.append(group_record)
                            self._append_evidence({"record_type": "candidate_group", **group_record}, candidate=True)
                            module_logger.warning(
                                "Candidate group complete: group_id=%s task=%s layout=%d reward_vector=%s "
                                "accepted=%s concurrent_batch=%s",
                                group_record["group_id"],
                                group_record["task_name"],
                                group_record["group_key"]["layout_id"],
                                group_record["reward_vector"],
                                group_record["accepted"],
                                group_record.get("concurrent_collection_batch_id", "serial"),
                            )
                            if selected_for_training:
                                accepted.append((rollout, rollout_end_obs, group_record))
                        candidate_schedule_slot += pipeline_stage_num * groups_per_stage
                        candidate_attempt += collection_group_count
                        self._write_candidate_resume_state(
                            rollout_policy_version=self.global_steps,
                            intended_update=intended_update,
                            next_candidate_attempt=candidate_attempt,
                            candidate_schedule_slot=candidate_schedule_slot,
                        )
                        if _optional_refill_is_unnecessary(
                            candidate_groups=len(candidate_records),
                            accepted_groups=len(accepted),
                            initial_candidate_groups=initial_candidates,
                            accepted_groups_before_refill=refill_threshold,
                        ):
                            module_logger.warning(
                                "Initial candidate window is informative enough; skipping optional "
                                "refill: candidates=%d accepted=%d threshold=%d.",
                                len(candidate_records),
                                len(accepted),
                                refill_threshold,
                            )
                            break
                finally:
                    collection_lifecycle_metrics.update(self.cluster.end_rollout_collection())

                sampling_wall_seconds = time.perf_counter() - sampling_start
                if self.trainer_config.collection_only:
                    record = {
                        "record_type": "collection_only_summary",
                        "collection_attempt_id": collection_attempt_id,
                        "sampling_iteration": sampling_iteration,
                        "intended_update": intended_update,
                        "rollout_policy_version": self.global_steps,
                        "candidate_groups_attempted": len(candidate_records),
                        "candidate_group_ids": [item["group_id"] for item in candidate_records],
                        "accepted_groups": len(accepted),
                        "sampling_wall_seconds": sampling_wall_seconds,
                    }
                    self._append_evidence(record)
                    self._append_evidence(
                        {
                            "record_type": "collection_window_closed",
                            "reason": "collection_only",
                            "intended_update": intended_update,
                            "rollout_policy_version": self.global_steps,
                        },
                        candidate=True,
                    )
                    module_logger.warning(
                        "Collection-only candidate window complete: theta_old=%d candidates=%d "
                        "accepted=%d wall_seconds=%.3f; exiting before FPO preparation/training.",
                        self.global_steps,
                        len(candidate_records),
                        len(accepted),
                        sampling_wall_seconds,
                    )
                    return
                if len(accepted) < min_partial_groups:
                    record = {
                        "record_type": "insufficient_informative_groups",
                        "collection_attempt_id": collection_attempt_id,
                        "sampling_iteration": sampling_iteration,
                        "intended_update": intended_update,
                        "rollout_policy_version": self.global_steps,
                        "candidate_groups_attempted": len(candidate_records),
                        "candidate_group_ids": [item["group_id"] for item in candidate_records],
                        "accepted_groups": len(accepted),
                        "minimum_groups_for_update": min_partial_groups,
                        "sampling_wall_seconds": sampling_wall_seconds,
                    }
                    self._append_evidence(record)
                    self._append_evidence(
                        {
                            "record_type": "collection_window_closed",
                            "reason": "insufficient_informative_groups",
                            "intended_update": intended_update,
                            "rollout_policy_version": self.global_steps,
                        },
                        candidate=True,
                    )
                    self._cleanup_candidate_spool_window(
                        intended_update=intended_update,
                        rollout_policy_version=self.global_steps,
                    )
                    self._write_candidate_resume_state(
                        rollout_policy_version=self.global_steps,
                        intended_update=intended_update,
                        next_candidate_attempt=1,
                        candidate_schedule_slot=candidate_schedule_slot,
                    )
                    logger.log(
                        data={
                            "sampling/insufficient_informative_groups": 1.0,
                            "sampling/candidate_groups_attempted": float(len(candidate_records)),
                            "sampling/accepted_groups": float(len(accepted)),
                            "sampling/collected_trajectories": float(
                                len(candidate_records) * int(self.trainer_config.group_size)
                            ),
                        },
                        step=self.global_steps,
                    )
                    continue

                accepted_rollouts = [item[0] for item in accepted]
                accepted_end_obs = [item[1] for item in accepted]
                accepted_groups = [item[2] for item in accepted]
                policy_versions = {int(item["rollout_policy_version"]) for item in accepted_groups}
                if policy_versions != {self.global_steps}:
                    raise RuntimeError(
                        f"Accepted groups must all use theta_old={self.global_steps}, got {policy_versions}."
                    )
                if any(len(rollout) != int(self.trainer_config.group_size) for rollout in accepted_rollouts):
                    raise RuntimeError("Every accepted group must retain all G=8 trajectories.")
                rollout_output, rollout_end_obs = concatenate_rollout_groups_with_padding(
                    accepted_rollouts,
                    accepted_end_obs,
                )
                actor_input = self._prepare_actor_input(
                    rollout_output,
                    rollout_end_obs,
                    group_records=accepted_groups,
                )
                advantage_groups = self.last_group_diagnostics["groups"]
                for accepted_group, group_diagnostics in zip(accepted_groups, advantage_groups, strict=True):
                    advantages = group_diagnostics["advantage_vector"]
                    accepted_group["advantage_vector"] = advantages
                    for trajectory, advantage in zip(accepted_group["trajectories"], advantages, strict=True):
                        trajectory["group_id"] = accepted_group["group_id"]
                        trajectory["trajectory_id"] = f"{accepted_group['group_id']}/trajectory-{trajectory['lane']}"
                        trajectory["group_advantage"] = advantage
                        trajectory["rollout_policy_version"] = accepted_group["rollout_policy_version"]

                valid_policy_chunks = int(actor_input.batch["info.valids"].sum().item())
                module_logger.warning(
                    "Starting multi-group FPO training: theta_old=%d accepted_groups=%d "
                    "logical_trajectories=%d valid_policy_chunks=%d micro_batch_size=%d update_epochs=%d "
                    "rollout_action_latent_scale=%.6f",
                    next(iter(policy_versions)),
                    len(accepted_groups),
                    len(actor_input),
                    valid_policy_chunks,
                    int(self.config.cluster.actor_rollout_ref.actor.micro_batch_size),
                    int(self.config.cluster.actor_rollout_ref.actor.update_epochs),
                    _fastwam_action_latent_scale(self.config),
                )
                update_start = time.perf_counter()
                actor_output = self.cluster.train(actor_input, async_update=False)
                update_wall_seconds = time.perf_counter() - update_start
                actor_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                if not all(np.isfinite(value) for value in actor_metrics.values() if np.isscalar(value)):
                    raise FloatingPointError("Group-Relative FPO actor metrics contain NaN or Inf.")

                self.global_steps = intended_update
                if self.trainer_config.save_freq > 0 and (
                    self.global_steps == total_steps or self.global_steps % self.trainer_config.save_freq == 0
                ):
                    self.cluster.save_checkpoint(self.global_steps)

                layouts = {int(item["group_key"]["layout_id"]) for item in accepted_groups}
                tasks = {
                    str(item.get("task_name", f"task_id={item['group_key']['task_id']}"))
                    for item in accepted_groups
                }
                candidate_groups_by_task = Counter(str(item["task_name"]) for item in candidate_records)
                accepted_groups_by_task = Counter(str(item["task_name"]) for item in accepted_groups)
                group_keys = {
                    json.dumps(item["group_key"], sort_keys=True, ensure_ascii=False) for item in accepted_groups
                }
                rejected_all_failure = sum(
                    int(not item["accepted"] and int(item["successes"]) == 0) for item in candidate_records
                )
                rejected_all_success = sum(
                    int(not item["accepted"] and int(item["successes"]) == int(self.trainer_config.group_size))
                    for item in candidate_records
                )
                mixed_groups = sum(int(bool(item["mixed"])) for item in candidate_records)
                informative_groups = sum(int(bool(item["informative"])) for item in candidate_records)
                accepted_partial_only_groups = sum(
                    int(item.get("informative_reason") == "score_variance_partial_only") for item in accepted_groups
                )
                accepted_trajectories = len(accepted_groups) * int(self.trainer_config.group_size)
                metrics: dict[str, Any] = {
                    **actor_metrics,
                    **{
                        key: value
                        for key, value in self.last_group_diagnostics.items()
                        if key not in {"trajectory_advantages", "groups"}
                    },
                    "training/global_step": float(self.global_steps),
                    "training/rollout_policy_version": float(next(iter(policy_versions))),
                    "sampling/rollout_action_latent_scale": _fastwam_action_latent_scale(self.config),
                    "sampling/candidate_groups_attempted": float(len(candidate_records)),
                    "sampling/accepted_groups": float(len(accepted_groups)),
                    "sampling/target_accepted_groups": float(target_groups),
                    "sampling/initial_candidate_groups": float(initial_candidates),
                    "sampling/optional_refill_threshold": float(refill_threshold),
                    "sampling/optional_refill_used": float(
                        initial_candidates > 0 and len(candidate_records) > initial_candidates
                    ),
                    "sampling/partial_large_batch_update": float(len(accepted_groups) < target_groups),
                    "sampling/rejected_all_failure_groups": float(rejected_all_failure),
                    "sampling/rejected_all_success_groups": float(rejected_all_success),
                    "sampling/mixed_group_acceptance_rate": mixed_groups / max(len(candidate_records), 1),
                    "sampling/informative_group_acceptance_rate": informative_groups / max(len(candidate_records), 1),
                    "sampling/accepted_partial_only_groups": float(accepted_partial_only_groups),
                    "sampling/candidate_groups_per_accepted_group": len(candidate_records) / len(accepted_groups),
                    "sampling/collected_trajectories": float(
                        len(candidate_records) * int(self.trainer_config.group_size)
                    ),
                    "sampling/trained_trajectories": float(accepted_trajectories),
                    "sampling/unique_tasks": float(len(tasks)),
                    "sampling/unique_layouts": float(len(layouts)),
                    "sampling/unique_group_keys": float(len(group_keys)),
                    "throughput/accepted_groups_per_hour": len(accepted_groups)
                    * 3600.0
                    / max(sampling_wall_seconds, 1e-12),
                    "throughput/raw_trajectories_per_hour": len(candidate_records)
                    * int(self.trainer_config.group_size)
                    * 3600.0
                    / max(sampling_wall_seconds, 1e-12),
                    "sampling/cumulative_candidate_groups": float(cumulative_candidate_groups),
                    "sampling/cumulative_collected_trajectories": float(cumulative_collected_trajectories),
                    "sampling/cumulative_mixed_groups": float(cumulative_mixed_groups),
                    "sampling/cumulative_informative_groups": float(cumulative_informative_groups),
                    "sampling/cumulative_all_failure_groups": float(cumulative_all_failure_groups),
                    "sampling/cumulative_all_success_groups": float(cumulative_all_success_groups),
                    "sampling/cumulative_acceptance_rate": cumulative_mixed_groups
                    / max(cumulative_candidate_groups, 1),
                    "sampling/cumulative_informative_acceptance_rate": cumulative_informative_groups
                    / max(cumulative_candidate_groups, 1),
                    "timing_s/group_sampling": sampling_wall_seconds,
                    "timing_s/fpo_training": update_wall_seconds,
                    **collection_lifecycle_metrics,
                }
                for task_name in sorted(candidate_groups_by_task):
                    candidate_count = candidate_groups_by_task[task_name]
                    accepted_count = accepted_groups_by_task[task_name]
                    metrics[f"sampling/task/{task_name}/candidate_groups"] = float(candidate_count)
                    metrics[f"sampling/task/{task_name}/accepted_groups"] = float(accepted_count)
                    metrics[f"sampling/task/{task_name}/acceptance_rate"] = accepted_count / candidate_count
                logger.log(data=metrics, step=self.global_steps)
                self._append_evidence(
                    {
                        "record_type": "group_relative_update",
                        "global_step": self.global_steps,
                        "collection_attempt_id": collection_attempt_id,
                        "policy_before_update": self._policy_identity(next(iter(policy_versions))),
                        "policy_after_update": self._policy_identity(self.global_steps),
                        "sampling_iteration": sampling_iteration,
                        "logical_batch": {
                            "group_size": int(self.trainer_config.group_size),
                            "accepted_groups": len(accepted_groups),
                            "accepted_trajectories": accepted_trajectories,
                            "target_accepted_groups": target_groups,
                            "partial": len(accepted_groups) < target_groups,
                        },
                        "candidate_group_ids": [item["group_id"] for item in candidate_records],
                        "accepted_groups": accepted_groups,
                        "candidate_groups_by_task": dict(sorted(candidate_groups_by_task.items())),
                        "accepted_groups_by_task": dict(sorted(accepted_groups_by_task.items())),
                        "unique_tasks": sorted(tasks),
                        "unique_layouts": sorted(layouts),
                        "unique_group_keys": sorted(group_keys),
                        "metrics": metrics,
                        "trajectory_audit": self.last_trajectory_audit,
                    }
                )
                self._append_evidence(
                    {
                        "record_type": "collection_window_closed",
                        "reason": "optimizer_update_complete",
                        "intended_update": intended_update,
                        "rollout_policy_version": next(iter(policy_versions)),
                        "global_step_after_update": self.global_steps,
                    },
                    candidate=True,
                )
                self._cleanup_candidate_spool_window(
                    intended_update=intended_update,
                    rollout_policy_version=next(iter(policy_versions)),
                )
                self._write_candidate_resume_state(
                    rollout_policy_version=self.global_steps,
                    intended_update=self.global_steps + 1,
                    next_candidate_attempt=1,
                    candidate_schedule_slot=candidate_schedule_slot,
                )
                progress.update(1)

                if self.trainer_config.test_freq > 0 and (
                    self.global_steps == total_steps or self.global_steps % self.trainer_config.test_freq == 0
                ):
                    eval_episodes = self.trainer_config.eval_episodes
                    val_metrics = self.cluster.eval(max_episodes=eval_episodes if eval_episodes > 0 else None)
                    # TrainCluster.rollout() has already prefetched the next
                    # all-stage training reset. Evaluation mutates those same
                    # simulators, so TrainCluster.eval() deliberately discards
                    # that stale observation and performs one fresh all-stage
                    # training reset at the end of evaluation. Both resets
                    # advance RoboDojo's train cursor, whereas the local
                    # preregistered schedule currently points at the discarded
                    # prefetch. Skip exactly one pipeline wave so the next
                    # assertion compares against the fresh post-eval reset.
                    pipeline_stage_num = int(self.config.cluster.env.env_loop.pipeline_stage_num)
                    groups_per_stage = _candidate_groups_per_stage(self.config, self.trainer_config)
                    candidate_schedule_slot = _candidate_schedule_slot_after_fixed_eval(
                        candidate_schedule_slot,
                        pipeline_stage_num * groups_per_stage,
                    )
                    module_logger.warning(
                        "Fixed evaluation invalidated one prefetched train reset; "
                        "advanced candidate schedule by %d slots to %d.",
                        pipeline_stage_num * groups_per_stage,
                        candidate_schedule_slot,
                    )
                    logger.log(data=val_metrics, step=self.global_steps)
                    self._append_evidence(
                        {
                            "record_type": "fixed_eval",
                            "global_step": self.global_steps,
                            "policy_identity": self._policy_identity(self.global_steps),
                            "metrics": val_metrics,
                            "trajectories": getattr(self.cluster, "last_eval_records", []),
                        }
                    )
        finally:
            progress.close()

    def fit(self) -> None:
        """Run only the GRFPO lifecycle; generic FPO/GAE has a separate trainer."""

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.trainer_config.project_name,
            experiment_name=self.trainer_config.experiment_name,
            default_backend=self.trainer_config.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        self.global_steps = 0
        checkpoint_state = self.cluster.load_checkpoint()
        if checkpoint_state is not None:
            self.global_steps, _checkpoint_dir = checkpoint_state

        if self.trainer_config.val_only:
            eval_episodes = self.trainer_config.eval_episodes
            val_metrics = self.cluster.eval(max_episodes=eval_episodes if eval_episodes > 0 else None)
            pprint(f"GRFPO checkpoint evaluation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.trainer_config.evidence_jsonl:
                self._append_evidence(
                    {
                        "record_type": "fixed_eval",
                        "global_step": self.global_steps,
                        "policy_identity": self._policy_identity(self.global_steps),
                        "metrics": val_metrics,
                        "trajectories": getattr(self.cluster, "last_eval_records", []),
                    }
                )
            return
        if self.trainer_config.val_before_train:
            raise ValueError("GRFPO requires a predeclared base evaluation; set val_before_train=false.")
        self._fit_group_relative(logger)
