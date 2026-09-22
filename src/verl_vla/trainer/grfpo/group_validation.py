"""Lightweight same-condition group validation for GRFPO collection.

This module deliberately avoids importing the Ray/FSDP trainer.  Fragmented
collectors need these checks before committing a candidate group, but do not
need any optimizer or distributed-training dependencies.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import torch
from verl import DataProto


class RolloutStateIntegrityError(RuntimeError):
    """A candidate contains robot state that must never enter actor training."""

    def __init__(self, diagnostics: dict[str, Any]) -> None:
        self.diagnostics = diagnostics
        super().__init__(
            "GRFPO candidate contains invalid raw proprio state: "
            f"absmax={diagnostics['raw_state_absmax']}, "
            f"limit={diagnostics['raw_state_abs_limit']}, "
            f"nonfinite={diagnostics['nonfinite_count']}, "
            f"trajectories={diagnostics['offending_trajectory_indices']}"
        )


def inspect_rollout_state_integrity(
    rollout: DataProto,
    rollout_end_obs: DataProto | None = None,
    *,
    raw_state_abs_limit: float,
) -> dict[str, Any]:
    """Inspect raw proprio before normalization and return JSON-safe diagnostics."""

    if not np.isfinite(raw_state_abs_limit) or raw_state_abs_limit <= 0:
        raise ValueError("raw_state_abs_limit must be finite and positive.")
    rollout_state_key = next(
        (key for key in ("obs.observation.state", "obs.state") if key in rollout.batch),
        None,
    )
    if rollout_state_key is None:
        raise KeyError(
            "GRFPO rollout is missing raw proprio tensor "
            "'obs.observation.state' (or prepared alias 'obs.state')."
        )

    tensors: list[tuple[str, torch.Tensor]] = [
        (rollout_state_key, rollout.batch[rollout_state_key])
    ]
    if rollout_end_obs is not None:
        for end_key in ("state", "observation.state"):
            if end_key in rollout_end_obs.batch:
                tensors.append((f"rollout_end_obs.{end_key}", rollout_end_obs.batch[end_key]))
                break

    batch_size = len(rollout)
    offending = torch.zeros(batch_size, dtype=torch.bool)
    nonfinite_count = 0
    raw_state_absmax = 0.0
    tensor_diagnostics: dict[str, Any] = {}
    for name, value in tensors:
        state = value.detach().to(device="cpu", dtype=torch.float32)
        if state.ndim < 2 or state.shape[0] != batch_size:
            raise ValueError(
                f"{name} must have leading trajectory dimension {batch_size}, got {tuple(state.shape)}."
            )
        finite = torch.isfinite(state)
        flat_finite = finite.reshape(batch_size, -1)
        flat_abs = state.abs().reshape(batch_size, -1)
        safe_abs = torch.where(flat_finite, flat_abs, torch.full_like(flat_abs, float("inf")))
        per_trajectory_absmax = safe_abs.amax(dim=1)
        offending |= (~flat_finite).any(dim=1) | (per_trajectory_absmax > raw_state_abs_limit)
        tensor_nonfinite = int((~finite).sum().item())
        nonfinite_count += tensor_nonfinite
        finite_values = flat_abs[flat_finite]
        tensor_absmax = float(finite_values.max().item()) if finite_values.numel() else None
        if tensor_absmax is not None:
            raw_state_absmax = max(raw_state_absmax, tensor_absmax)
        tensor_diagnostics[name] = {
            "shape": list(state.shape),
            "raw_state_absmax": tensor_absmax,
            "nonfinite_count": tensor_nonfinite,
        }

    return {
        "valid": not bool(offending.any()),
        "raw_state_absmax": raw_state_absmax,
        "raw_state_abs_limit": float(raw_state_abs_limit),
        "nonfinite_count": nonfinite_count,
        "offending_trajectory_indices": torch.nonzero(offending, as_tuple=False).flatten().tolist(),
        "tensors": tensor_diagnostics,
    }


def _normalized_instruction(value: Any) -> str:
    return " ".join(str(value).split()).casefold()


def _constant_trajectory_values(rollout: DataProto, key: str) -> list[Any]:
    if key not in rollout.non_tensor_batch:
        raise KeyError(f"Same-condition GRPO rollout is missing {key!r}.")
    values = np.asarray(rollout.non_tensor_batch[key])
    if values.shape[0] != len(rollout):
        raise ValueError(f"{key} must have one leading row per trajectory, got {values.shape}.")
    if values.ndim == 1:
        return values.tolist()
    result = []
    for trajectory_index, row in enumerate(values):
        flat = np.asarray(row).reshape(-1)
        first = flat[0]
        if any(value != first for value in flat[1:]):
            raise ValueError(f"{key} changed within trajectory {trajectory_index}.")
        result.append(first)
    return result


def _terminal_trajectory_scores(rollout: DataProto) -> torch.Tensor:
    """Return the score at the first terminal/truncated step per trajectory."""

    if "next.score" not in rollout.batch:
        raise KeyError("group_reward_source='process_score' requires rollout field 'next.score'.")
    scores = rollout.batch["next.score"].float().flatten(start_dim=1)
    dones = (rollout.batch["next.terminated"].bool() | rollout.batch["next.truncated"].bool()).flatten(start_dim=1)
    if scores.shape != dones.shape:
        raise ValueError(
            "RoboDojo score and done tensors must have matching flattened shapes, "
            f"got score={tuple(scores.shape)} done={tuple(dones.shape)}."
        )
    if not torch.isfinite(scores).all():
        raise FloatingPointError("RoboDojo process scores contain NaN or Inf.")
    terminal_scores = []
    for score_row, done_row in zip(scores, dones, strict=True):
        done_indices = torch.nonzero(done_row, as_tuple=False).flatten()
        terminal_index = int(done_indices[0]) if done_indices.numel() else int(score_row.numel()) - 1
        terminal_scores.append(score_row[terminal_index])
    return torch.stack(terminal_scores)


def validate_same_condition_group(
    rollout: DataProto,
    *,
    group_size: int,
    group_id: str,
    rollout_policy_version: int,
    reward_source: str = "binary_success",
    informative_group_criterion: str = "binary_success",
    min_partial_trajectories_for_score_only_group: int = 1,
    partial_score_threshold: float = 0.15,
    expected_suite_id: int | None = None,
    allowed_layout_ids: set[int] | None = None,
) -> dict[str, Any]:
    """Validate and annotate one complete same-condition trajectory group."""

    if len(rollout) != group_size:
        raise ValueError(f"GRPO group {group_id} must contain exactly {group_size} trajectories, got {len(rollout)}.")
    task_ids = [int(value) for value in _constant_trajectory_values(rollout, "obs.task_id")]
    suite_ids = [int(value) for value in _constant_trajectory_values(rollout, "obs.suite_id")]
    instructions = [_normalized_instruction(value) for value in _constant_trajectory_values(rollout, "obs.task")]
    layout_ids = [int(value) for value in _constant_trajectory_values(rollout, "obs.layout_id")]
    environment_seeds = [int(value) for value in _constant_trajectory_values(rollout, "obs.environment_seed")]
    policy_seeds = [int(value) for value in _constant_trajectory_values(rollout, "obs.policy_seed")]
    group_keys = list(zip(task_ids, instructions, suite_ids, layout_ids, environment_seeds, strict=True))
    if len(set(group_keys)) != 1:
        raise ValueError(f"GRPO group {group_id} mixes environment conditions: {group_keys}.")
    if len(set(policy_seeds)) != group_size:
        raise ValueError(f"GRPO group {group_id} requires {group_size} distinct policy seeds, got {policy_seeds}.")
    if expected_suite_id is not None and any(suite_id != expected_suite_id for suite_id in suite_ids):
        raise ValueError(f"GRPO group {group_id} used suite ids {suite_ids}, expected {expected_suite_id}.")
    if allowed_layout_ids is not None and any(layout_id not in allowed_layout_ids for layout_id in layout_ids):
        raise ValueError(
            "GRPO training group used a layout outside its pre-registered pool: "
            f"layouts={layout_ids}, allowed={sorted(allowed_layout_ids)}."
        )

    if reward_source not in {"binary_success", "process_score"}:
        raise ValueError(f"Unsupported GRPO reward_source={reward_source!r}.")
    if informative_group_criterion not in {"binary_success", "full_success_mixed", "reward_variance"}:
        raise ValueError(f"Unsupported informative_group_criterion={informative_group_criterion!r}.")
    if informative_group_criterion == "reward_variance" and reward_source != "process_score":
        raise ValueError("reward_variance filtering requires process_score rewards.")

    successes = rollout.batch["next.success"].bool().flatten(start_dim=1).any(dim=1)
    binary_success_vector = successes.float()
    if reward_source == "process_score":
        process_scores = _terminal_trajectory_scores(rollout)
    else:
        process_scores = (
            _terminal_trajectory_scores(rollout) if "next.score" in rollout.batch else binary_success_vector.clone()
        )
    process_scores = process_scores.clone()
    process_scores[successes] = 1.0
    training_rewards = binary_success_vector if reward_source == "binary_success" else process_scores
    full_success_count = int(successes.sum().item())
    binary_mixed = 0 < full_success_count < group_size
    partial_only = (~successes) & (process_scores >= float(partial_score_threshold))
    partial_only_count = int(partial_only.sum().item())
    reward_varies = not bool(
        torch.allclose(training_rewards, training_rewards[:1].expand_as(training_rewards), atol=1e-7, rtol=0)
    )
    if informative_group_criterion in {"binary_success", "full_success_mixed"}:
        informative = binary_mixed
        informative_reason = "mixed_full_success" if informative else "not_mixed_full_success"
    else:
        partial_minimum_met = full_success_count > 0 or (
            partial_only_count >= int(min_partial_trajectories_for_score_only_group)
        )
        informative = reward_varies and partial_minimum_met
        if informative and full_success_count > 0:
            informative_reason = "score_variance_with_full_success"
        elif informative:
            informative_reason = "score_variance_partial_only"
        elif reward_varies:
            informative_reason = "partial_only_below_minimum"
        else:
            informative_reason = "zero_training_reward_variance"

    group_key = group_keys[0]
    group_key_text = json.dumps(group_key, ensure_ascii=False, separators=(",", ":"))
    policy_slots = int(rollout.batch["next.success"].shape[1])
    rollout.non_tensor_batch["obs.group_id"] = np.full((group_size, policy_slots), group_id, dtype=object)
    rollout.non_tensor_batch["obs.trajectory_id"] = np.asarray(
        [[f"{group_id}/trajectory-{index}"] * policy_slots for index in range(group_size)], dtype=object
    )
    rollout.non_tensor_batch["obs.group_key"] = np.full((group_size, policy_slots), group_key_text, dtype=object)
    rollout.non_tensor_batch["obs.rollout_policy_version"] = np.full(
        (group_size, policy_slots), int(rollout_policy_version), dtype=np.int64
    )
    return {
        "group_id": group_id,
        "group_key": {
            "task_id": group_key[0],
            "normalized_instruction": group_key[1],
            "suite_id": group_key[2],
            "layout_id": group_key[3],
            "environment_seed": group_key[4],
        },
        "rollout_policy_version": int(rollout_policy_version),
        "policy_seeds": policy_seeds,
        "reward_source": reward_source,
        "reward_vector": [float(value) for value in training_rewards.tolist()],
        "binary_success_vector": [int(value) for value in binary_success_vector.tolist()],
        "process_score_vector": [float(value) for value in process_scores.tolist()],
        "successes": full_success_count,
        "full_success_count": full_success_count,
        "partial_only_count": partial_only_count,
        "partial_or_better_count": int((process_scores >= float(partial_score_threshold)).sum().item()),
        "binary_mixed": binary_mixed,
        "mixed": binary_mixed,
        "training_reward_varies": reward_varies,
        "informative": informative,
        "informative_reason": informative_reason,
    }
