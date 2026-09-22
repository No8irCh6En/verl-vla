# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0

"""Lightweight rollout-batch assembly shared by collection and training."""

from __future__ import annotations

import numpy as np
import torch
from verl import DataProto

from verl_vla.utils.data import flatten_trajectories
from verl_vla.utils.keys import OBS_KEY


def compact_valid_policy_chunks_for_fpo(
    data: DataProto,
    *,
    actor_world_size: int,
    micro_batch_size: int,
) -> DataProto:
    """Remove trajectory padding before fixed-MC FPO replay.

    The actor objective is already weighted by ``fpo.loss_weights`` and
    ``info.valids``. A zero-valid slot therefore has exactly zero objective
    contribution and need not run Fast-WAM. Compact globally before veRL
    dispatches the batch so multiple FSDP ranks receive equal numbers of
    physical microbatches. At most ``world_size * micro_batch_size - 1``
    zero-weight synchronization dummies are appended for divisibility.
    """

    if actor_world_size <= 0 or micro_batch_size <= 0:
        raise ValueError(
            "actor_world_size and micro_batch_size must be positive, got "
            f"{actor_world_size} and {micro_batch_size}."
        )
    if data.batch["info.valids"].ndim != 2:
        raise ValueError("FPO compaction expects trajectory-shaped info.valids [B,S].")
    if data.batch["fpo.loss_weights"].shape != data.batch["info.valids"].shape:
        raise ValueError("fpo.loss_weights must match trajectory-shaped info.valids.")

    flat = flatten_trajectories(data, reference_key="action.action")
    valids = flat.batch["info.valids"].bool()
    weights = flat.batch["fpo.loss_weights"]
    if torch.any(valids & (~torch.isfinite(weights) | (weights <= 0))):
        raise ValueError("Every valid FPO policy chunk must have a finite positive loss weight.")
    if torch.any((~valids) & (weights != 0)):
        raise ValueError("Invalid/padded FPO policy chunks must have zero loss weight.")

    valid_indices = torch.nonzero(valids, as_tuple=False).flatten()
    if valid_indices.numel() == 0:
        raise ValueError("FPO logical batch contains no valid policy chunks.")
    compact = flat.select_idxs(valid_indices)
    valid_count = len(compact)
    original_slots = len(flat)

    divisor = actor_world_size * micro_batch_size
    sync_dummy_count = (-valid_count) % divisor
    if sync_dummy_count:
        dummy_indices = torch.zeros(sync_dummy_count, dtype=torch.long)
        compact = DataProto.concat([compact, compact.select_idxs(dummy_indices)])
        dummy_slice = slice(len(compact) - sync_dummy_count, len(compact))
        compact.batch["info.valids"][dummy_slice] = 0
        compact.batch["fpo.loss_weights"][dummy_slice] = 0
        compact.batch["fpo.advantages"][dummy_slice] = 0
        compact.batch["info.action_valids"][dummy_slice] = 0

    if len(compact) % divisor:
        raise AssertionError("Compacted FPO batch is not divisible across FSDP microbatches.")
    if int(compact.batch["info.valids"].sum().item()) != valid_count:
        raise AssertionError("FPO compaction changed the number of valid policy chunks.")
    if not torch.equal(
        compact.batch["fpo.loss_weights"] > 0,
        compact.batch["info.valids"].bool(),
    ):
        raise AssertionError("FPO compacted valid mask and loss weights disagree.")

    compact.meta_info = {
        **compact.meta_info,
        "fpo_preflattened_valid_chunks": True,
        "fpo_original_physical_slots": original_slots,
        "fpo_valid_policy_chunks": valid_count,
        "fpo_sync_dummy_slots": sync_dummy_count,
        "fpo_compacted_physical_slots": len(compact),
    }
    return compact


def concatenate_rollout_groups_with_padding(
    rollouts: list[DataProto],
    rollout_end_observations: list[DataProto],
) -> tuple[DataProto, DataProto]:
    """Concatenate rollout batches whose policy-call counts differ.

    Observation padding uses each trajectory's terminal observation. Action
    and feedback padding is zero and is ignored by the downstream valid mask.
    The helper intentionally has no Ray/trainer imports so fragmented rollout
    collectors can reuse it without loading the training stack.
    """

    if not rollouts or len(rollouts) != len(rollout_end_observations):
        raise ValueError(
            "Rollout groups and end observations must be non-empty and aligned: "
            f"rollouts={len(rollouts)}, end_observations={len(rollout_end_observations)}."
        )

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
