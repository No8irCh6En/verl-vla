# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

import logging
import math
import os
import time
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

import torch
from tqdm import tqdm
from verl import DataProto
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils.device import get_device_id, get_device_name
from verl.workers.config import TrainingWorkerConfig
from verl.workers.engine_workers import TrainingWorker

from verl_vla.utils.data import flatten_trajectories, get_dataproto_from_prefix
from verl_vla.workers.config import FPOActorConfig

logger = logging.getLogger(__name__)

_PRE_OPTIMIZER_IDENTITY_ATOL = 1.0e-5


def _diagnostic_identity_atol() -> float:
    """Return an optional tolerance used only by the read-only replay diagnostic.

    The production pre-optimizer gate deliberately continues to use the fixed
    ``_PRE_OPTIMIZER_IDENTITY_ATOL`` above.  Batched BF16 kernels can introduce
    a small old/current replay delta even with identical weights and fixed MC
    inputs, so a stress test may opt into a slightly wider, explicitly logged
    diagnostic tolerance without weakening live training safety.
    """

    value = float(os.environ.get("GRFPO_DIAGNOSTIC_IDENTITY_ATOL", _PRE_OPTIMIZER_IDENTITY_ATOL))
    if not math.isfinite(value) or value < _PRE_OPTIMIZER_IDENTITY_ATOL or value > 1.0e-2:
        raise ValueError(
            "GRFPO_DIAGNOSTIC_IDENTITY_ATOL must be finite and in "
            f"[{_PRE_OPTIMIZER_IDENTITY_ATOL}, 0.01], got {value}."
        )
    return value


def _synchronize_for_timing() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _require_finite(name: str, value: torch.Tensor) -> None:
    finite = torch.isfinite(value).all()
    if value.is_cuda and hasattr(torch, "_assert_async"):
        # Preserve fail-closed finite checks without forcing one host/device
        # synchronization for every loss tensor in every micro-batch.
        torch._assert_async(finite, f"FPO {name} contains NaN or Inf.")
    elif not bool(finite):
        raise FloatingPointError(f"FPO {name} contains NaN or Inf.")


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    discounts: torch.Tensor,
    gae_discounts: torch.Tensor,
    valids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE over rollout chunks without crossing padded or terminal slots."""
    advantages = torch.zeros_like(values)
    last_advantage = torch.zeros_like(values[:, 0])
    for step in reversed(range(values.shape[1])):
        active = valids[:, step]
        delta = rewards[:, step] + discounts[:, step] * next_values[:, step] - values[:, step]
        last_advantage = (delta + gae_discounts[:, step] * last_advantage) * active
        advantages[:, step] = last_advantage
    return advantages, advantages + values


def clipped_policy_loss(
    log_ratio: torch.Tensor,
    advantages: torch.Tensor,
    valids: torch.Tensor,
    clip_coef: float,
    *,
    loss_weights: torch.Tensor | None = None,
    normalization: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Vanilla-FPO's PPO surrogate, where CFM loss differences are log ratios."""
    ratio = log_ratio.exp()
    unclipped = -advantages * ratio
    clipped = -advantages * ratio.clamp(1.0 - clip_coef, 1.0 + clip_coef)
    weights = valids if loss_weights is None else loss_weights * valids
    denominator = weights.sum().clamp_min(1.0) if normalization is None else normalization.clamp_min(1.0)
    loss = (torch.maximum(unclipped, clipped) * weights).sum() / denominator
    with torch.no_grad():
        approx_kl = (((ratio - 1.0) - log_ratio) * weights).sum() / denominator
        clip_fraction = (((ratio - 1.0).abs() > clip_coef).float() * weights).sum() / denominator
    return loss, {"ratio": ratio.detach(), "approx_kl": approx_kl, "clip_fraction": clip_fraction}


def compute_cfm_log_ratio(
    old_cfm_loss: torch.Tensor,
    current_cfm_loss: torch.Tensor,
    action_valids: torch.Tensor,
) -> torch.Tensor:
    """Aggregate fixed-MC CFM differences over executed action steps only."""
    if old_cfm_loss.shape != current_cfm_loss.shape or old_cfm_loss.ndim != 3:
        raise ValueError("old/current CFM losses must match with shape [B,H,N].")
    if action_valids.shape != old_cfm_loss.shape[:2]:
        raise ValueError("action_valids must have shape [B,H] matching CFM losses.")
    mask = action_valids.to(device=old_cfm_loss.device, dtype=old_cfm_loss.dtype).unsqueeze(-1)
    return ((old_cfm_loss - current_cfm_loss) * mask).sum(dim=1).mean(dim=-1)


def compute_cfm_per_sample_log_ratio(
    old_cfm_loss: torch.Tensor,
    current_cfm_loss: torch.Tensor,
    action_valids: torch.Tensor,
    *,
    clamp_max: float | None,
) -> torch.Tensor:
    """FPO++ log ratios, one independently clipped ratio per fixed MC draw.

    The executed action positions H are summed exactly as in the existing
    chunk-level vanilla estimator. Unlike vanilla FPO, N is retained rather
    than averaged before exponentiation, yielding ``[B,N]``.
    """

    if old_cfm_loss.shape != current_cfm_loss.shape or old_cfm_loss.ndim != 3:
        raise ValueError("old/current CFM losses must match with shape [B,H,N].")
    if action_valids.shape != old_cfm_loss.shape[:2]:
        raise ValueError("action_valids must have shape [B,H] matching CFM losses.")
    mask = action_valids.to(device=old_cfm_loss.device, dtype=old_cfm_loss.dtype).unsqueeze(-1)
    raw_log_ratio = ((old_cfm_loss - current_cfm_loss) * mask).sum(dim=1)
    if clamp_max is None:
        return raw_log_ratio
    if clamp_max <= 0:
        raise ValueError("FPO++ clamp_max must be positive.")
    return raw_log_ratio + (raw_log_ratio.clamp(-clamp_max, clamp_max) - raw_log_ratio).detach()


def fpo_plus_plus_policy_loss(
    log_ratio: torch.Tensor,
    advantages: torch.Tensor,
    valids: torch.Tensor,
    clip_coef: float,
    *,
    trust_region_mode: str = "ppo",
    spo_clip_coef: float = 0.01,
    loss_weights: torch.Tensor | None = None,
    normalization: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Per-MC FPO++ surrogate with PPO or opt-in asymmetric SPO clipping."""

    if log_ratio.ndim != 2:
        raise ValueError("FPO++ log_ratio must have shape [B,N].")
    if advantages.shape != log_ratio.shape[:1] or valids.shape != log_ratio.shape[:1]:
        raise ValueError("FPO++ advantages/valids must have shape [B].")
    if trust_region_mode not in {"ppo", "aspo"}:
        raise ValueError(f"Unsupported FPO++ trust_region_mode={trust_region_mode!r}.")
    if spo_clip_coef <= 0:
        raise ValueError("FPO++ spo_clip_coef must be positive.")

    ratio = log_ratio.exp()
    advantages_mc = advantages.unsqueeze(-1)
    ppo_unclipped = -advantages_mc * ratio
    ppo_clipped = -advantages_mc * ratio.clamp(1.0 - clip_coef, 1.0 + clip_coef)
    ppo_loss = torch.maximum(ppo_unclipped, ppo_clipped)
    if trust_region_mode == "aspo":
        # Official ASPO: PPO for positive advantages and the SPO quadratic
        # pullback for negative advantages.
        spo_objective = advantages_mc * ratio - advantages_mc.abs() / (2.0 * spo_clip_coef) * (ratio - 1.0).square()
        element_loss = torch.where(advantages_mc > 0, ppo_loss, -spo_objective)
    else:
        element_loss = ppo_loss

    chunk_weights = valids if loss_weights is None else loss_weights * valids
    mc_weights = chunk_weights.unsqueeze(-1) / float(log_ratio.shape[1])
    denominator = chunk_weights.sum().clamp_min(1.0) if normalization is None else normalization.clamp_min(1.0)
    loss = (element_loss * mc_weights).sum() / denominator
    with torch.no_grad():
        approx_kl = (((ratio - 1.0) - log_ratio) * mc_weights).sum() / denominator
        clip_fraction = (((ratio - 1.0).abs() > clip_coef).float() * mc_weights).sum() / denominator
    return loss, {"ratio": ratio.detach(), "approx_kl": approx_kl, "clip_fraction": clip_fraction}


def distributed_explained_variance(
    returns: torch.Tensor,
    predictions: torch.Tensor,
    valids: torch.Tensor,
) -> torch.Tensor:
    """Compute explained variance from rollout-wide sufficient statistics."""
    valid_returns = returns * valids
    residuals = (returns - predictions) * valids
    stats = torch.stack(
        [
            valid_returns.sum(),
            valid_returns.square().sum(),
            residuals.sum(),
            residuals.square().sum(),
            valids.sum(),
        ]
    )
    torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
    count = stats[4].clamp_min(1.0)
    return_mean = stats[0] / count
    return_variance = stats[1] / count - return_mean.square()
    residual_mean = stats[2] / count
    residual_variance = stats[3] / count - residual_mean.square()
    return 1.0 - residual_variance / return_variance.clamp_min(1e-8)


def distributed_masked_stats(values: torch.Tensor, valids: torch.Tensor) -> dict[str, torch.Tensor]:
    """Compute rollout-wide finite mean/std/min/max without changing training tensors."""

    values = values.float()
    mask = valids.to(device=values.device, dtype=torch.bool)
    selected = values[mask]
    if selected.numel():
        sums = torch.stack([selected.double().sum(), selected.double().square().sum(), mask.sum().double()])
        extrema = torch.stack([selected.min(), selected.max()])
    else:
        sums = torch.zeros(3, dtype=torch.float64, device=values.device)
        extrema = torch.tensor([float("inf"), float("-inf")], device=values.device)
    torch.distributed.all_reduce(sums, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(extrema[:1], op=torch.distributed.ReduceOp.MIN)
    torch.distributed.all_reduce(extrema[1:], op=torch.distributed.ReduceOp.MAX)
    count = sums[2].clamp_min(1.0)
    mean = sums[0] / count
    variance = (sums[1] / count - mean.square()).clamp_min(0.0)
    return {"mean": mean.float(), "std": variance.sqrt().float(), "min": extrema[0], "max": extrema[1]}


def distributed_gradient_mean_denominator(local_weights: torch.Tensor) -> torch.Tensor:
    """Return the per-rank loss denominator for data-parallel averaged gradients.

    FSDP/DDP averages gradients across data-parallel ranks.  Each rank must
    therefore divide its local numerator by ``global_weight / world_size``;
    dividing by ``global_weight`` here would make the subsequent gradient
    average introduce an unintended extra factor of ``1 / world_size``.
    """

    denominator = local_weights.sum()
    torch.distributed.all_reduce(denominator, op=torch.distributed.ReduceOp.SUM)
    denominator /= torch.distributed.get_world_size()
    return denominator


def _gradient_mapping_dot(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
) -> float:
    """Exact CPU dot product for two identically named gradient shards."""

    if left.keys() != right.keys():
        raise RuntimeError("Task-gradient parameter sets do not match.")
    return math.fsum(float(torch.dot(left[name].reshape(-1), right[name].reshape(-1))) for name in left)


def _gradient_mapping_linear_combination_stats(
    gradients: dict[str, dict[str, torch.Tensor]],
    coefficients: dict[str, float],
) -> tuple[float, dict[str, float]]:
    """Return exact norm and projections without materializing another model-sized vector."""

    names = sorted(gradients)
    if set(names) != set(coefficients):
        raise ValueError("Gradient and coefficient task sets must match.")
    gram = {task: {} for task in names}
    for left_index, left in enumerate(names):
        for right in names[left_index:]:
            value = _gradient_mapping_dot(gradients[left], gradients[right])
            gram[left][right] = value
            gram[right][left] = value
    squared_norm = math.fsum(
        coefficients[left] * coefficients[right] * gram[left][right]
        for left in names
        for right in names
    )
    norm = math.sqrt(max(squared_norm, 0.0))
    projections = {
        task: math.fsum(coefficients[other] * gram[task][other] for other in names)
        for task in names
    }
    return norm, projections


class FPOTrainingWorker(TrainingWorker):
    """On-policy vanilla Flow Policy Optimization worker."""

    def __init__(self, config: TrainingWorkerConfig, actor_config: FPOActorConfig, tokenizer=None):
        super().__init__(config=config)
        self.actor_config = actor_config
        self.tokenizer = tokenizer or self.model_config.tokenizer
        self.local_mini_batch_size = self._global_to_local_batch_size(actor_config.mini_batch_size)
        self._fpo_initialized = False
        self._identity_gate_has_passed = False

    @staticmethod
    def _global_to_local_batch_size(global_batch_size: int) -> int:
        world_size = torch.distributed.get_world_size()
        if global_batch_size % world_size:
            raise ValueError(f"FPO mini_batch_size={global_batch_size} must be divisible by world_size={world_size}.")
        return global_batch_size // world_size

    def _ensure_fpo_initialized(self) -> None:
        if self._fpo_initialized:
            return
        self.engine.module.fpo_init()
        self.value_parameters: list[torch.nn.Parameter] = []
        self.value_optimizer: torch.optim.Optimizer | None = None
        self.value_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        if self.actor_config.value.enabled:
            self.value_parameters = self.engine.module.fpo_get_value_parameters()
            self.value_optimizer = torch.optim.Adam(
                self.value_parameters,
                lr=self.actor_config.value.lr,
                weight_decay=self.actor_config.value.weight_decay,
            )
            self.value_scheduler = torch.optim.lr_scheduler.ConstantLR(self.value_optimizer, factor=1.0)
        self._fpo_initialized = True

    def _value_optimizer_checkpoint_path(self, local_path: str) -> Path:
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        return Path(local_path) / f"fpo_value_optim_world_size_{world_size}_rank_{rank}.pt"

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        self._ensure_fpo_initialized()
        super().save_checkpoint(local_path, hdfs_path, global_step, max_ckpt_to_keep)
        if not self.actor_config.value.enabled:
            return
        assert self.value_optimizer is not None and self.value_scheduler is not None
        checkpoint_path = self._value_optimizer_checkpoint_path(local_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "optimizer": self.value_optimizer.state_dict(),
                "scheduler": self.value_scheduler.state_dict(),
            },
            checkpoint_path,
        )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        self._ensure_fpo_initialized()
        if not self.actor_config.value.enabled:
            return super().load_checkpoint(local_path, hdfs_path, del_local_after_load)
        assert self.value_optimizer is not None and self.value_scheduler is not None
        checkpoint_path = self._value_optimizer_checkpoint_path(local_path)
        if checkpoint_path.exists():
            state = torch.load(
                checkpoint_path,
                map_location=torch.device(get_device_name(), get_device_id()),
                weights_only=True,
            )
            self.value_optimizer.load_state_dict(state["optimizer"])
            self.value_scheduler.load_state_dict(state["scheduler"])
        else:
            logger.warning(
                "Checkpoint %s predates FPO value-optimizer persistence; resuming value optimization "
                "with fresh Adam state.",
                checkpoint_path,
            )
        return super().load_checkpoint(local_path, hdfs_path, del_local_after_load)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def export_portable_checkpoint(self, local_path):
        """Export model and optimizer state for a later FSDP world-size change."""

        self._ensure_fpo_initialized()
        return self.engine.checkpoint_manager.export_portable_checkpoint(local_path)

    @staticmethod
    def _global_normalize(values: torch.Tensor, valids: torch.Tensor) -> torch.Tensor:
        valids = valids.to(device=values.device, dtype=values.dtype)
        valid_values = values * valids
        stats = torch.stack([(valid_values).sum(), (valid_values.square()).sum(), valids.sum()])
        torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
        mean = stats[0] / stats[2].clamp_min(1.0)
        variance = stats[1] / stats[2].clamp_min(1.0) - mean.square()
        return (values - mean) / variance.clamp_min(1e-8).sqrt()

    def _forward_old_statistics(self, data: DataProto) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
        preflattened = bool(data.meta_info.get("fpo_preflattened_valid_chunks", False))
        flat = data if preflattened else flatten_trajectories(data, reference_key="action.action")
        old_values: list[torch.Tensor] = []
        next_values: list[torch.Tensor] = []
        old_cfm_losses = []
        for micro_batch in flat.split(self.actor_config.micro_batch_size):
            micro_batch = micro_batch.to(get_device_id())
            obs = get_dataproto_from_prefix(micro_batch, "obs.")
            with torch.no_grad(), torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
                if self.actor_config.value.enabled:
                    next_obs = get_dataproto_from_prefix(micro_batch, "next_obs.")
                    old_values.append(self.engine.module.fpo_forward_value(obs, self.tokenizer))
                    next_values.append(self.engine.module.fpo_forward_value(next_obs, self.tokenizer))
                old_cfm_losses.append(
                    self.engine.module.fpo_cfm_loss(
                        obs,
                        self.tokenizer,
                        micro_batch.batch["action.full_action"],
                        micro_batch.batch["fpo.timesteps"],
                        micro_batch.batch["fpo.noise"],
                        loss_kernel=getattr(self.actor_config, "cfm_loss_kernel", "mse"),
                        huber_delta=float(getattr(self.actor_config, "huber_delta", 1.0)),
                    )
                )
        if preflattened:
            return (
                torch.cat(old_values) if old_values else None,
                torch.cat(next_values) if next_values else None,
                torch.cat(old_cfm_losses),
            )
        batch_size, rollout_steps = data.batch["info.valids"].shape
        return (
            torch.cat(old_values).reshape(batch_size, rollout_steps) if old_values else None,
            torch.cat(next_values).reshape(batch_size, rollout_steps) if next_values else None,
            torch.cat(old_cfm_losses).reshape(batch_size, rollout_steps, *old_cfm_losses[0].shape[1:]),
        )

    def _prepare_training_batch(self, data: DataProto) -> DataProto:
        full_actions = data.batch["action.full_action"]
        sample_count = self.actor_config.n_action_samples
        device = get_device_id()
        mc_seed = getattr(self.actor_config, "fpo_mc_seed", None)
        mc_generator = None
        if mc_seed is not None:
            # A fixed world-size/topology is part of the counterfactual
            # experiment contract. Offset rank streams so every rank remains
            # deterministic without drawing identical MC samples.
            generator_device = torch.device(get_device_name(), get_device_id())
            mc_generator = torch.Generator(device=generator_device)
            mc_generator.manual_seed(int(mc_seed) + 1_000_003 * torch.distributed.get_rank())
        preflattened = bool(data.meta_info.get("fpo_preflattened_valid_chunks", False))
        if preflattened:
            if full_actions.ndim != 3 or data.batch["info.valids"].ndim != 1:
                raise ValueError("Preflattened FPO input must use action [V,H,D] and valids [V].")
            slot_count, action_horizon, action_dim = full_actions.shape
            data.batch["fpo.timesteps"] = torch.rand(
                slot_count,
                sample_count,
                device=device,
                dtype=torch.float32,
                generator=mc_generator,
            )
            data.batch["fpo.noise"] = torch.randn(
                slot_count,
                sample_count,
                action_horizon,
                action_dim,
                device=device,
                dtype=full_actions.dtype,
                generator=mc_generator,
            )
        else:
            batch_size, rollout_steps, action_horizon, action_dim = full_actions.shape
            data.batch["fpo.timesteps"] = torch.rand(
                batch_size,
                rollout_steps,
                sample_count,
                device=device,
                dtype=torch.float32,
                generator=mc_generator,
            )
            data.batch["fpo.noise"] = torch.randn(
                batch_size,
                rollout_steps,
                sample_count,
                action_horizon,
                action_dim,
                device=device,
                dtype=full_actions.dtype,
                generator=mc_generator,
            )

        old_values, next_values, old_cfm_loss = self._forward_old_statistics(data)
        _require_finite("old CFM scores", old_cfm_loss)
        valids = data.batch["info.valids"].to(device=device, dtype=torch.float32)
        if self.actor_config.value.enabled:
            assert old_values is not None and next_values is not None
            _require_finite("old value predictions", old_values)
            _require_finite("next value predictions", next_values)
            advantages, returns = compute_gae(
                rewards=data.batch["info.rewards"].to(device),
                values=old_values,
                next_values=next_values,
                discounts=data.batch["info.discounts"].to(device),
                gae_discounts=data.batch["info.gae_discounts"].to(device),
                valids=valids,
            )
            _require_finite("GAE advantages", advantages)
            _require_finite("GAE returns", returns)
            data.batch["fpo.old_values"] = old_values
            data.batch["fpo.advantages"] = advantages
            data.batch["fpo.returns"] = returns
        else:
            if data.meta_info.get("advantage_estimator") != "grpo_outcome":
                raise ValueError("Critic-free FPO requires precomputed grpo_outcome advantages.")
            if "fpo.advantages" not in data.batch or "fpo.loss_weights" not in data.batch:
                raise KeyError("Critic-free FPO requires fpo.advantages and fpo.loss_weights.")
            _require_finite("GRPO advantages", data.batch["fpo.advantages"])
            _require_finite("trajectory-balanced loss weights", data.batch["fpo.loss_weights"])
            data.batch["fpo.advantages"] = data.batch["fpo.advantages"].to(device)
            data.batch["fpo.loss_weights"] = data.batch["fpo.loss_weights"].to(device)
        data.batch["fpo.old_cfm_loss"] = old_cfm_loss
        return data if preflattened else flatten_trajectories(data, reference_key="action.action")

    def _compute_policy_log_ratio(
        self,
        micro_batch: DataProto,
        obs: DataProto,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return log ratio plus elementwise identity delta/mask for vanilla FPO."""

        current_cfm_loss = self.engine.module.fpo_cfm_loss(
            obs,
            self.tokenizer,
            micro_batch.batch["action.full_action"],
            micro_batch.batch["fpo.timesteps"],
            micro_batch.batch["fpo.noise"],
            loss_kernel=getattr(self.actor_config, "cfm_loss_kernel", "mse"),
            huber_delta=float(getattr(self.actor_config, "huber_delta", 1.0)),
        )
        old_cfm_loss = micro_batch.batch["fpo.old_cfm_loss"]
        if getattr(self.actor_config, "fpo_variant", "vanilla") == "fpo_plus_plus":
            log_ratio = compute_cfm_per_sample_log_ratio(
                old_cfm_loss,
                current_cfm_loss,
                micro_batch.batch["info.action_valids"],
                clamp_max=getattr(self.actor_config, "log_ratio_clamp", None),
            )
        else:
            log_ratio = compute_cfm_log_ratio(
                old_cfm_loss,
                current_cfm_loss,
                micro_batch.batch["info.action_valids"],
            )
        return (
            log_ratio,
            old_cfm_loss - current_cfm_loss.detach(),
            micro_batch.batch["info.action_valids"].bool().unsqueeze(-1),
        )

    def _compute_clipped_policy_loss(
        self,
        log_ratio: torch.Tensor,
        advantages: torch.Tensor,
        valids: torch.Tensor,
        *,
        loss_weights: torch.Tensor | None,
        normalization: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if getattr(self.actor_config, "fpo_variant", "vanilla") == "fpo_plus_plus":
            return fpo_plus_plus_policy_loss(
                log_ratio,
                advantages,
                valids,
                self.actor_config.clip_coef,
                trust_region_mode=getattr(self.actor_config, "trust_region_mode", "ppo"),
                spo_clip_coef=getattr(self.actor_config, "spo_clip_coef", 0.01),
                loss_weights=loss_weights,
                normalization=normalization,
            )
        return clipped_policy_loss(
            log_ratio,
            advantages,
            valids,
            self.actor_config.clip_coef,
            loss_weights=loss_weights,
            normalization=normalization,
        )

    def _algorithm_metrics(self) -> dict[str, float]:
        plus_plus = getattr(self.actor_config, "fpo_variant", "vanilla") == "fpo_plus_plus"
        return {
            "algorithm/fpo_uniform_unweighted": 1.0,
            "algorithm/vanilla_fpo_ratio": float(not plus_plus),
            "algorithm/fpo_plus_plus_per_mc_ratio": float(plus_plus),
            "algorithm/fpo_plus_plus_aspo": float(
                plus_plus and getattr(self.actor_config, "trust_region_mode", "ppo") == "aspo"
            ),
            "algorithm/fpo_plus_plus_huber": float(
                plus_plus and getattr(self.actor_config, "cfm_loss_kernel", "mse") == "huber"
            ),
            "algorithm/fpo_plus_plus_gradient_preserving_clamp": float(
                plus_plus and getattr(self.actor_config, "log_ratio_clamp", None) is not None
            ),
            "algorithm/flow_grpo_ratio": 0.0,
        }

    def _diagnostic_backward(
        self,
        data: DataProto,
        *,
        normalization: torch.Tensor,
    ) -> tuple[float, float]:
        """Backward one fixed-MC subset without stepping or changing optimizer state."""

        micro_batches = data.split(self.actor_config.micro_batch_size)
        self.engine.optimizer_zero_grad()
        max_abs_identity_log_ratio = 0.0
        total_policy_loss = 0.0
        try:
            for index, micro_batch in enumerate(micro_batches):
                sync_gradients = index == len(micro_batches) - 1
                self.engine.module.set_requires_gradient_sync(sync_gradients)
                self.engine.module.set_is_last_backward(sync_gradients)
                micro_batch = micro_batch.to(get_device_id())
                obs = get_dataproto_from_prefix(micro_batch, "obs.")
                valids = micro_batch.batch["info.valids"].float()
                with torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
                    log_ratio, _, _ = self._compute_policy_log_ratio(micro_batch, obs)
                    policy_loss, _ = self._compute_clipped_policy_loss(
                        log_ratio,
                        micro_batch.batch["fpo.advantages"],
                        valids,
                        loss_weights=micro_batch.batch["fpo.loss_weights"],
                        normalization=normalization,
                    )
                _require_finite("task-gradient diagnostic loss", policy_loss)
                total_policy_loss += float(policy_loss.detach())
                valid_log_ratio = log_ratio.detach()[valids.bool()]
                if valid_log_ratio.numel():
                    max_abs_identity_log_ratio = max(
                        max_abs_identity_log_ratio,
                        float(valid_log_ratio.abs().max()),
                    )
                policy_loss.backward()
        finally:
            self.engine.module.set_requires_gradient_sync(True)
            self.engine.module.set_is_last_backward(True)
        diagnostic_atol = _diagnostic_identity_atol()
        if max_abs_identity_log_ratio > diagnostic_atol:
            raise RuntimeError(
                "Fixed-MC gradient diagnostic must run at theta_old before any optimizer step: "
                f"max_abs_log_ratio={max_abs_identity_log_ratio:.9g} "
                f"diagnostic_atol={diagnostic_atol:.9g}."
            )
        return max_abs_identity_log_ratio, total_policy_loss

    @torch.no_grad()
    def _diagnostic_forward_loss(
        self,
        data: DataProto,
        *,
        normalization: torch.Tensor,
    ) -> dict[str, float]:
        """Evaluate one fixed-old/advantage/MC subset without backward or step."""

        totals = {
            "surrogate_loss": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
            "ratio_sum": 0.0,
            "ratio_square_sum": 0.0,
            "ratio_count": 0.0,
            "ratio_min": float("inf"),
            "ratio_max": float("-inf"),
        }
        for micro_batch in data.split(self.actor_config.micro_batch_size):
            micro_batch = micro_batch.to(get_device_id())
            obs = get_dataproto_from_prefix(micro_batch, "obs.")
            valids = micro_batch.batch["info.valids"].float()
            with torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
                log_ratio, _, _ = self._compute_policy_log_ratio(micro_batch, obs)
                policy_loss, policy_metrics = self._compute_clipped_policy_loss(
                    log_ratio,
                    micro_batch.batch["fpo.advantages"],
                    valids,
                    loss_weights=micro_batch.batch["fpo.loss_weights"],
                    normalization=normalization,
                )
            _require_finite("task-gradient diagnostic evaluation loss", policy_loss)
            ratios = policy_metrics["ratio"]
            valid_ratios = ratios[valids.bool()]
            totals["surrogate_loss"] += float(policy_loss)
            totals["approx_kl"] += float(policy_metrics["approx_kl"])
            totals["clip_fraction"] += float(policy_metrics["clip_fraction"])
            if valid_ratios.numel():
                values = valid_ratios.float()
                totals["ratio_sum"] += float(values.sum())
                totals["ratio_square_sum"] += float(values.square().sum())
                totals["ratio_count"] += float(values.numel())
                totals["ratio_min"] = min(totals["ratio_min"], float(values.min()))
                totals["ratio_max"] = max(totals["ratio_max"], float(values.max()))
        count = max(totals.pop("ratio_count"), 1.0)
        mean = totals.pop("ratio_sum") / count
        square_mean = totals.pop("ratio_square_sum") / count
        totals["ratio_mean"] = mean
        totals["ratio_std"] = math.sqrt(max(square_mean - mean * mean, 0.0))
        if not math.isfinite(totals["ratio_min"]):
            totals["ratio_min"] = 1.0
            totals["ratio_max"] = 1.0
        return totals

    @staticmethod
    def _diagnostic_local_tensor(value: torch.Tensor, *, name: str) -> torch.Tensor:
        if hasattr(value, "to_local"):
            local = value.to_local()
            if tuple(local.shape) != tuple(value.shape):
                raise ValueError(
                    f"Diagnostic target tensor {name!r} is sharded: "
                    f"global={tuple(value.shape)} local={tuple(local.shape)}. "
                    "Use a portable/world-size-one full actor state."
                )
            value = local
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Diagnostic target {name!r} is {type(value).__name__}, not Tensor.")
        return value.detach()

    @staticmethod
    def _diagnostic_scope(name: str) -> str:
        if ".action_expert." in name or ".mixtures.action." in name:
            return "action_expert"
        if ".video_expert." in name or ".mixtures.video." in name:
            return "video_expert"
        if ".proprio_encoder." in name:
            return "proprio_encoder"
        return name.split(".", 2)[1] if "." in name else name

    @staticmethod
    def _resolve_diagnostic_actor_state(source: str | Path) -> Path:
        source = Path(source).expanduser().resolve()
        if source.is_file():
            return source
        actor = source if source.name == "actor" else source / "actor"
        model = actor / "model_world_size_1_rank_0.pt"
        if not model.is_file():
            raise FileNotFoundError(
                f"Task diagnostic requires a portable/world-size-one actor state: {model}"
            )
        return model

    def _apply_diagnostic_target(
        self,
        target_state: Mapping[str, torch.Tensor],
        *,
        parameter_names: set[str],
    ) -> None:
        with torch.no_grad():
            seen: set[str] = set()
            for name, parameter in self.engine.module.named_parameters():
                if name not in parameter_names:
                    continue
                if name not in target_state:
                    raise KeyError(f"Diagnostic target is missing trainable parameter {name!r}.")
                destination = parameter.to_local() if hasattr(parameter, "to_local") else parameter
                source = self._diagnostic_local_tensor(target_state[name], name=name)
                if tuple(source.shape) != tuple(destination.shape):
                    raise ValueError(
                        f"Diagnostic target shape mismatch for {name!r}: "
                        f"{tuple(source.shape)} != {tuple(destination.shape)}"
                    )
                destination.copy_(source.to(device=destination.device, dtype=destination.dtype))
                seen.add(name)
            missing = parameter_names - seen
            if missing:
                raise KeyError(f"Model did not expose diagnostic trainable parameters: {sorted(missing)[:10]}")

    def _capture_gradient_mapping_cpu(self) -> dict[str, torch.Tensor]:
        gradients: dict[str, torch.Tensor] = {}
        for name, parameter in self.engine.module.named_parameters():
            if not parameter.requires_grad:
                continue
            if parameter.grad is None:
                # A trainable parameter may be outside the action path used by
                # this objective. It must be absent consistently across task
                # branches; _gradient_mapping_dot checks that invariant.
                continue
            gradient = parameter.grad
            # FSDP2 retains a DTensor wrapper even for a world-size-one mesh.
            # Consolidated diagnostic targets and the captured theta0 snapshot
            # are plain tensors, so keeping this wrapper would make the exact
            # g dot (theta_target - theta0) calculation dispatch a mixed
            # Tensor/DTensor operator.  The one-rank precondition above makes
            # the local tensor the complete gradient, not a shard.
            if hasattr(gradient, "to_local"):
                gradient = gradient.to_local()
            gradients[name] = gradient.detach().float().cpu().clone()
        if not gradients:
            raise RuntimeError("Task-gradient diagnostic found no trainable actor parameters.")
        return gradients

    def diagnose_task_gradients(self, data: DataProto) -> dict[str, object]:
        """Measure fixed-MC task/group gradients at theta_old without optimizer.step.

        This intentionally supports one actor rank first.  Exact multi-rank FSDP
        reconstruction needs a distributed-shard artifact format; silently
        treating rank-local norms as global norms would be mathematically wrong.
        """

        if torch.distributed.get_world_size() != 1:
            raise RuntimeError("Exact task-gradient diagnostics currently require one actor rank.")
        if self.actor_config.value.enabled:
            raise RuntimeError("Task-gradient diagnostics are defined for critic-free GRPO only.")
        required = {"fpo.task_index", "fpo.group_index", "fpo.loss_weights", "fpo.advantages"}
        missing = sorted(required - set(data.batch.keys()))
        if missing:
            raise KeyError(f"Task-gradient diagnostic input is missing {missing}.")
        task_names = [str(x) for x in data.meta_info.get("fpo_task_names", [])]
        if not task_names:
            raise ValueError("Task-gradient diagnostic requires fpo_task_names metadata.")
        include_group_diagnostics = bool(data.meta_info.get("task_gradient_include_groups", True))
        raw_targets = data.meta_info.get("task_gradient_target_checkpoints", {})
        if not isinstance(raw_targets, Mapping):
            raise TypeError("task_gradient_target_checkpoints must be a label -> checkpoint mapping.")
        target_paths = {str(label): self._resolve_diagnostic_actor_state(path) for label, path in raw_targets.items()}
        target_states = {
            label: torch.load(path, map_location="cpu", mmap=True, weights_only=False)
            for label, path in target_paths.items()
        }
        for label, state in target_states.items():
            if not isinstance(state, Mapping):
                raise TypeError(f"Diagnostic target {label!r} did not contain a state-dict mapping.")

        # Sample timesteps/noise exactly once, then reuse the materialized old
        # scores and MC tensors for every task and group branch below.
        flat = self._prepare_training_batch(data)
        valid = flat.batch["info.valids"].bool()
        flat = flat.select_idxs(torch.where(valid)[0])
        task_index = flat.batch["fpo.task_index"].long()
        group_index = flat.batch["fpo.group_index"].long()
        groups_by_task: dict[int, list[int]] = {}
        for index in range(len(task_names)):
            groups_by_task[index] = sorted({int(x) for x in group_index[task_index == index].tolist()})
            if not groups_by_task[index]:
                raise RuntimeError(f"Task {task_names[index]!r} has no valid accepted group rows.")

        task_gradients: dict[str, dict[str, torch.Tensor]] = {}
        task_norms: dict[str, float] = {}
        task_surrogate_before: dict[str, float] = {}
        task_delta_dot: dict[str, dict[str, float]] = {label: {} for label in target_states}
        identity_errors: list[float] = []
        named_parameters = dict(self.engine.module.named_parameters())
        trainable_theta0 = {
            name: (
                parameter.to_local() if hasattr(parameter, "to_local") else parameter
            ).detach().cpu().clone()
            for name, parameter in named_parameters.items()
            if parameter.requires_grad
        }
        if not trainable_theta0:
            raise RuntimeError("Task-gradient diagnostic found no trainable theta0 parameters.")
        for index, task_name in enumerate(task_names):
            subset = flat.select_idxs(torch.where(task_index == index)[0])
            normalization = subset.batch["fpo.loss_weights"].sum().to(get_device_id()).clamp_min(1.0)
            identity_error, surrogate_before = self._diagnostic_backward(
                subset,
                normalization=normalization,
            )
            identity_errors.append(identity_error)
            task_surrogate_before[task_name] = surrogate_before
            gradient = self._capture_gradient_mapping_cpu()
            task_gradients[task_name] = gradient
            task_norms[task_name] = math.sqrt(max(_gradient_mapping_dot(gradient, gradient), 0.0))
            for label, target_state in target_states.items():
                dot = 0.0
                for name, grad in gradient.items():
                    if name not in target_state or name not in trainable_theta0:
                        raise KeyError(f"Diagnostic target {label!r} is missing trainable parameter {name!r}.")
                    target = self._diagnostic_local_tensor(target_state[name], name=name).float()
                    theta0 = trainable_theta0[name].float()
                    dot += float((grad * (target - theta0)).sum(dtype=torch.float64))
                task_delta_dot[label][task_name] = dot
            logger.warning(
                "Fixed-MC task gradient task=%s groups=%d norm=%.9g",
                task_name,
                len(groups_by_task[index]),
                task_norms[task_name],
            )

        # Compute the expensive model-sized dot products once. All
        # counterfactual combinations below are then scalar Gram algebra.
        gram = {task: {} for task in task_names}
        for left_index, left in enumerate(task_names):
            for right in task_names[left_index:]:
                value = _gradient_mapping_dot(task_gradients[left], task_gradients[right])
                gram[left][right] = value
                gram[right][left] = value
        pairwise_cosine = {
            left: {
                right: gram[left][right] / max(task_norms[left] * task_norms[right], 1.0e-30)
                for right in task_names
            }
            for left in task_names
        }

        group_norms: dict[str, list[float]] = {task: [] for task in task_names}
        coherence: dict[str, float] = {}
        if include_group_diagnostics:
            for task_id, task_name in enumerate(task_names):
                for group_id in groups_by_task[task_id]:
                    mask = group_index == group_id
                    subset = flat.select_idxs(torch.where(mask)[0])
                    normalization = subset.batch["fpo.loss_weights"].sum().to(get_device_id()).clamp_min(1.0)
                    identity_error, _ = self._diagnostic_backward(subset, normalization=normalization)
                    identity_errors.append(identity_error)
                    squared_norm = 0.0
                    for parameter in self.engine.module.parameters():
                        if parameter.requires_grad and parameter.grad is not None:
                            gradient = parameter.grad
                            if hasattr(gradient, "to_local"):
                                gradient = gradient.to_local()
                            squared_norm += float(gradient.detach().float().square().sum())
                    group_norms[task_name].append(math.sqrt(max(squared_norm, 0.0)))

            coherence = {
                task_names[index]: (
                    task_norms[task_names[index]] * len(groups_by_task[index])
                    / max(math.fsum(group_norms[task_names[index]]), 1.0e-30)
                )
                for index in range(len(task_names))
            }
        group_counts = {task_names[i]: len(groups_by_task[i]) for i in range(len(task_names))}
        total_groups = sum(group_counts.values())
        current_coefficients = {task: group_counts[task] / total_groups for task in task_names}
        task_equal_coefficients = {task: 1.0 / len(task_names) for task in task_names}
        def combination_norm(coefficients: dict[str, float]) -> float:
            squared = math.fsum(
                coefficients[left] * coefficients[right] * gram[left][right]
                for left in task_names
                for right in task_names
            )
            return math.sqrt(max(squared, 0.0))

        current_norm = combination_norm(current_coefficients)
        task_equal_norm = combination_norm(task_equal_coefficients)
        current_dot_task_equal = math.fsum(
            current_coefficients[left]
            * task_equal_coefficients[right]
            * gram[left][right]
            for left in task_names
            for right in task_names
        )
        current_task_equal_cosine = current_dot_task_equal / max(current_norm * task_equal_norm, 1.0e-30)
        target_delta_stats: dict[str, dict[str, object]] = {}
        for label, target_state in target_states.items():
            scope_squares: dict[str, dict[str, float | int]] = {}
            for name, theta0 in trainable_theta0.items():
                if name not in target_state:
                    raise KeyError(f"Diagnostic target {label!r} is missing trainable parameter {name!r}.")
                target = self._diagnostic_local_tensor(target_state[name], name=name)
                if tuple(target.shape) != tuple(theta0.shape):
                    raise ValueError(f"Diagnostic target shape mismatch for {label}:{name}.")
                delta_norm = float(torch.linalg.vector_norm(target.float() - theta0.float()))
                reference_norm = float(torch.linalg.vector_norm(theta0.float()))
                scope = self._diagnostic_scope(name)
                bucket = scope_squares.setdefault(
                    scope,
                    {"parameter_count": 0, "reference_sq": 0.0, "delta_sq": 0.0},
                )
                bucket["parameter_count"] = int(bucket["parameter_count"]) + theta0.numel()
                bucket["reference_sq"] = float(bucket["reference_sq"]) + reference_norm * reference_norm
                bucket["delta_sq"] = float(bucket["delta_sq"]) + delta_norm * delta_norm
            scopes = {}
            for scope, bucket in scope_squares.items():
                reference_norm = math.sqrt(float(bucket["reference_sq"]))
                delta_norm = math.sqrt(float(bucket["delta_sq"]))
                scopes[scope] = {
                    "parameter_count": int(bucket["parameter_count"]),
                    "reference_norm": reference_norm,
                    "delta_norm": delta_norm,
                    "relative_delta_norm": delta_norm / max(reference_norm, 1.0e-30),
                }
            global_reference_sq = math.fsum(item["reference_norm"] ** 2 for item in scopes.values())
            global_delta_sq = math.fsum(item["delta_norm"] ** 2 for item in scopes.values())
            target_delta_stats[label] = {
                "checkpoint": str(target_paths[label]),
                "scope": "trainable_parameters",
                "global_reference_norm": math.sqrt(global_reference_sq),
                "global_delta_norm": math.sqrt(global_delta_sq),
                "global_relative_delta_norm": math.sqrt(global_delta_sq) / max(math.sqrt(global_reference_sq), 1.0e-30),
                "scopes": scopes,
            }

        task_surrogate_after: dict[str, dict[str, dict[str, float]]] = {}
        trainable_names = set(trainable_theta0)
        for label, target_state in target_states.items():
            self._apply_diagnostic_target(target_state, parameter_names=trainable_names)
            per_task = {}
            for index, task_name in enumerate(task_names):
                subset = flat.select_idxs(torch.where(task_index == index)[0])
                normalization = subset.batch["fpo.loss_weights"].sum().to(get_device_id()).clamp_min(1.0)
                per_task[task_name] = self._diagnostic_forward_loss(subset, normalization=normalization)
            task_surrogate_after[label] = per_task
        self.engine.optimizer_zero_grad()
        return {
            "schema_version": 2,
            "fixed_mc_reused_across_all_branches": True,
            "optimizer_step_performed": False,
            "actor_world_size": 1,
            "group_level_diagnostics_included": include_group_diagnostics,
            "task_names": task_names,
            "task_group_counts": group_counts,
            "task_gradient_norms": task_norms,
            "task_surrogate_loss_before": task_surrogate_before,
            "task_gradient_dot_parameter_delta": task_delta_dot,
            "target_parameter_delta": target_delta_stats,
            "task_surrogate_after": task_surrogate_after,
            "pairwise_task_gradient_cosine": pairwise_cosine,
            "per_group_gradient_norms": group_norms,
            "within_task_gradient_coherence": coherence,
            "current_group_weighted_gradient_norm": current_norm,
            "task_equal_gradient_norm": task_equal_norm,
            "current_vs_task_equal_gradient_cosine": current_task_equal_cosine,
            "max_abs_identity_log_ratio": max(identity_errors, default=0.0),
            "diagnostic_identity_atol": _diagnostic_identity_atol(),
        }

    def _update_fpo_policy(self, data: DataProto) -> dict[str, float]:
        _synchronize_for_timing()
        total_start = time.perf_counter()
        lr_override = getattr(self.actor_config, "post_resume_lr_override", None)
        if lr_override is not None:
            lr_override = float(lr_override)
            for param_group in self.engine.optimizer.param_groups:
                param_group["lr"] = lr_override
                if "initial_lr" in param_group:
                    param_group["initial_lr"] = lr_override
            scheduler = getattr(self.engine, "lr_scheduler", None)
            if scheduler is not None:
                if hasattr(scheduler, "base_lrs"):
                    scheduler.base_lrs = [lr_override for _ in scheduler.base_lrs]
                if hasattr(scheduler, "_last_lr"):
                    scheduler._last_lr = [lr_override for _ in scheduler._last_lr]
            actual_lrs = {float(group["lr"]) for group in self.engine.optimizer.param_groups}
            if actual_lrs != {lr_override}:
                raise RuntimeError(
                    "post-resume actor LR override did not reach every optimizer param group: "
                    f"expected={lr_override} actual={sorted(actual_lrs)}"
                )
        detailed_sync_timing = bool(getattr(self.actor_config, "profile_cuda_sync_timing", False))
        memory_device = torch.device(get_device_name(), get_device_id())
        if memory_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(memory_device)
            memory_start_stats = torch.tensor(
                [
                    torch.cuda.memory_allocated(memory_device),
                    torch.cuda.memory_reserved(memory_device),
                ],
                device=memory_device,
                dtype=torch.float64,
            )
        else:
            memory_start_stats = torch.zeros(2, device=memory_device, dtype=torch.float64)

        def sync_detailed_timing() -> None:
            if detailed_sync_timing:
                _synchronize_for_timing()

        timing_totals = {
            "fpo_old_statistics_forward": 0.0,
            "fpo_update_forward": 0.0,
            "fpo_backward_gradient_sync": 0.0,
            "fpo_optimizer": 0.0,
            "fpo_distributed_metrics": 0.0,
        }

        phase_start = time.perf_counter()
        data = self._prepare_training_batch(data)
        sync_detailed_timing()
        timing_totals["fpo_old_statistics_forward"] += time.perf_counter() - phase_start
        local_mini_batch_size = (
            len(data)
            if getattr(self.actor_config, "full_logical_batch_gradient_accumulation", False)
            else self.local_mini_batch_size
        )
        if len(data) < local_mini_batch_size:
            raise ValueError(
                f"Each FPO rank needs at least {local_mini_batch_size} valid/padded rollout slots, got {len(data)}."
            )

        critic_enabled = bool(self.actor_config.value.enabled)
        value_only = critic_enabled and int(data.meta_info["global_steps"]) <= self.actor_config.value_only_updates
        if not critic_enabled and data.meta_info.get("advantage_estimator") != "grpo_outcome":
            raise ValueError("Critic-free FPO update received a non-GRPO advantage source.")
        accepted_group_reduction = str(data.meta_info.get("accepted_group_reduction", "group_equal"))
        if not critic_enabled and accepted_group_reduction not in {"group_equal", "task_equal"}:
            raise ValueError(
                "Critic-free FPO received an invalid accepted-group reduction: "
                f"{accepted_group_reduction!r}."
            )
        advantage_stats = distributed_masked_stats(data.batch["fpo.advantages"], data.batch["info.valids"])
        metric_device = torch.device(get_device_name(), get_device_id())
        ratio_sums = torch.zeros(3, dtype=torch.float64, device=metric_device)
        ratio_extrema = torch.tensor([float("inf"), float("-inf")], device=metric_device, dtype=torch.float32)
        metric_lists: dict[str, list[float | torch.Tensor]] = defaultdict(list)
        epochs_run = 0
        optimizer_steps_applied = 0
        kl_early_stop_triggered = False
        identity_gate_checked = False
        self._active_update_epoch = -1
        mini_batches_per_epoch = (len(data) + local_mini_batch_size - 1) // local_mini_batch_size
        update_progress = tqdm(
            total=self.actor_config.update_epochs * mini_batches_per_epoch,
            desc="FPO update",
            disable=torch.distributed.get_rank() != 0,
            leave=False,
        )

        for _epoch in range(self.actor_config.update_epochs):
            self._active_update_epoch = _epoch
            shuffle_seed = getattr(self.actor_config, "fpo_shuffle_seed", None)
            if shuffle_seed is None:
                permutation = torch.randperm(len(data))
            else:
                shuffle_generator = torch.Generator(device="cpu")
                shuffle_generator.manual_seed(
                    int(shuffle_seed) + 1_000_003 * torch.distributed.get_rank() + _epoch
                )
                permutation = torch.randperm(len(data), generator=shuffle_generator)
            epoch_kls = []
            for mini_batch_index, start in enumerate(range(0, len(data), local_mini_batch_size)):
                update_progress.set_postfix(
                    epoch=f"{_epoch + 1}/{self.actor_config.update_epochs}",
                    minibatch=f"{mini_batch_index + 1}/{mini_batches_per_epoch}",
                    value_only=value_only,
                )
                indices = permutation[start : start + local_mini_batch_size]
                mini_batch = data.select_idxs(indices)
                # Match vanilla FPO/PPO: normalize over the current global
                # minibatch after shuffling, not once over the whole rollout.
                # Each rank contributes its local shard to these statistics.
                if self.actor_config.normalize_advantages:
                    if not critic_enabled:
                        raise RuntimeError("GRPO advantages must not be normalized again after grouping.")
                    sync_detailed_timing()
                    phase_start = time.perf_counter()
                    mini_batch.batch["fpo.advantages"] = self._global_normalize(
                        mini_batch.batch["fpo.advantages"],
                        mini_batch.batch["info.valids"].float(),
                    )
                    sync_detailed_timing()
                    timing_totals["fpo_distributed_metrics"] += time.perf_counter() - phase_start
                micro_batches = mini_batch.split(self.actor_config.micro_batch_size)
                grad_accum_steps = len(micro_batches)
                hierarchical_normalization = None
                if not critic_enabled:
                    hierarchical_normalization = distributed_gradient_mean_denominator(
                        mini_batch.batch["fpo.loss_weights"].to(metric_device)
                    )
                    if not torch.isfinite(hierarchical_normalization) or hierarchical_normalization <= 0:
                        raise RuntimeError("GRPO trajectory-balanced loss has no finite positive weight.")
                self.engine.optimizer_zero_grad()
                if self.value_optimizer is not None:
                    self.value_optimizer.zero_grad()

                hierarchical_policy_loss = torch.zeros((), device=metric_device)
                hierarchical_approx_kl = torch.zeros((), device=metric_device)
                hierarchical_clip_fraction = torch.zeros((), device=metric_device)

                try:
                    for micro_batch_index, micro_batch in enumerate(micro_batches):
                        sync_gradients = micro_batch_index == grad_accum_steps - 1
                        self.engine.module.set_requires_gradient_sync(sync_gradients)
                        self.engine.module.set_is_last_backward(sync_gradients)

                        micro_batch = micro_batch.to(get_device_id())
                        obs = get_dataproto_from_prefix(micro_batch, "obs.")
                        valids = micro_batch.batch["info.valids"].float()
                        sync_detailed_timing()
                        phase_start = time.perf_counter()
                        with torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
                            if critic_enabled:
                                values = self.engine.module.fpo_forward_value(obs, self.tokenizer)
                                value_loss = (
                                    0.5
                                    * ((values - micro_batch.batch["fpo.returns"]).square() * valids).sum()
                                    / valids.sum().clamp_min(1.0)
                                )
                            else:
                                values = valids
                                value_loss = valids.new_zeros(())

                            if value_only:
                                policy_loss = values.new_zeros(())
                                policy_metrics = {
                                    "ratio": torch.ones_like(values),
                                    "approx_kl": values.new_zeros(()),
                                    "clip_fraction": values.new_zeros(()),
                                }
                            else:
                                log_ratio, identity_score_delta, identity_score_mask = self._compute_policy_log_ratio(
                                    micro_batch, obs
                                )
                                identity_gate_required = critic_enabled or not self._identity_gate_has_passed
                                if not identity_gate_checked and identity_gate_required:
                                    # The first actor-loss evaluation must use
                                    # exactly the same policy, MC timesteps,
                                    # noise, and executed-action mask as the
                                    # frozen old statistic.  Check this before
                                    # any actor optimizer step so a broken
                                    # old/current pairing cannot silently enter
                                    # PPO clipping.
                                    identity_valids = valids.bool()
                                    identity_error = log_ratio.detach().new_zeros((), dtype=torch.float32)
                                    identity_cfm_error = log_ratio.detach().new_zeros((), dtype=torch.float32)
                                    identity_ratio_min = log_ratio.detach().new_tensor(
                                        float("inf"), dtype=torch.float32
                                    )
                                    identity_ratio_max = log_ratio.detach().new_tensor(
                                        float("-inf"), dtype=torch.float32
                                    )
                                    identity_ratio_sums = log_ratio.detach().new_zeros(4, dtype=torch.float64)
                                    if identity_valids.any():
                                        valid_log_ratio = log_ratio.detach()[identity_valids].float()
                                        valid_ratio = valid_log_ratio.exp()
                                        identity_error = valid_log_ratio.abs().max()
                                        valid_action_mask = identity_score_mask[identity_valids]
                                        if valid_action_mask.any():
                                            score_delta = identity_score_delta[identity_valids]
                                            identity_cfm_error = (
                                                score_delta[valid_action_mask.expand_as(score_delta)]
                                                .abs()
                                                .max()
                                                .float()
                                            )
                                        identity_ratio_min = valid_ratio.min()
                                        identity_ratio_max = valid_ratio.max()
                                        identity_ratio_sums = torch.stack(
                                            [
                                                valid_ratio.double().sum(),
                                                valid_ratio.double().square().sum(),
                                                torch.tensor(
                                                    float(valid_ratio.numel()),
                                                    dtype=torch.float64,
                                                    device=valid_ratio.device,
                                                ),
                                                ((valid_ratio - 1.0).abs() > self.actor_config.clip_coef)
                                                .double()
                                                .sum(),
                                            ]
                                        )
                                    torch.distributed.all_reduce(identity_error, op=torch.distributed.ReduceOp.MAX)
                                    torch.distributed.all_reduce(identity_cfm_error, op=torch.distributed.ReduceOp.MAX)
                                    torch.distributed.all_reduce(identity_ratio_sums, op=torch.distributed.ReduceOp.SUM)
                                    torch.distributed.all_reduce(identity_ratio_min, op=torch.distributed.ReduceOp.MIN)
                                    torch.distributed.all_reduce(identity_ratio_max, op=torch.distributed.ReduceOp.MAX)
                                    identity_valid_count = identity_ratio_sums[2]
                                    # A shuffled physical microbatch can contain only padding from
                                    # an early-terminated trajectory. Keep searching for the first
                                    # globally valid microbatch; the logical-batch guard below still
                                    # fails before optimizer.step if none exists anywhere.
                                    if identity_valid_count > 0:
                                        identity_ratio_mean = identity_ratio_sums[0] / identity_valid_count
                                        identity_ratio_variance = (
                                            identity_ratio_sums[1] / identity_valid_count - identity_ratio_mean.square()
                                        ).clamp_min(0.0)
                                        identity_ratio_std = identity_ratio_variance.sqrt()
                                        identity_clip_fraction = identity_ratio_sums[3] / identity_valid_count
                                        if (
                                            not torch.isfinite(identity_error)
                                            or identity_error > _PRE_OPTIMIZER_IDENTITY_ATOL
                                            or not torch.isfinite(identity_cfm_error)
                                            or identity_cfm_error > _PRE_OPTIMIZER_IDENTITY_ATOL
                                            or not torch.isfinite(identity_ratio_sums).all()
                                            or not torch.isfinite(identity_ratio_min)
                                            or not torch.isfinite(identity_ratio_max)
                                        ):
                                            raise RuntimeError(
                                                "FPO pre-optimizer identity gate failed: "
                                                f"max_abs_log_ratio={float(identity_error):.9g} "
                                                f"max_abs_cfm_error={float(identity_cfm_error):.9g} "
                                                f"atol={_PRE_OPTIMIZER_IDENTITY_ATOL:.9g}."
                                            )
                                        identity_gate_checked = True
                                        self._identity_gate_has_passed = True
                                        metric_lists["fpo/pre_optimizer_identity_max_abs_log_ratio"].append(
                                            float(identity_error)
                                        )
                                        metric_lists["fpo/pre_optimizer_identity_max_abs_cfm_error"].append(
                                            float(identity_cfm_error)
                                        )
                                        metric_lists["fpo/pre_optimizer_ratio_mean"].append(float(identity_ratio_mean))
                                        metric_lists["fpo/pre_optimizer_ratio_min"].append(float(identity_ratio_min))
                                        metric_lists["fpo/pre_optimizer_ratio_max"].append(float(identity_ratio_max))
                                        metric_lists["fpo/pre_optimizer_ratio_std"].append(float(identity_ratio_std))
                                        metric_lists["fpo/pre_optimizer_clip_fraction"].append(
                                            float(identity_clip_fraction)
                                        )
                                        logger.warning(
                                            "FPO pre-optimizer identity gate passed: max_abs_log_ratio=%.9g "
                                            "max_abs_cfm_error=%.9g ratio_mean=%.9g ratio_min=%.9g "
                                            "ratio_max=%.9g ratio_std=%.9g clip_fraction=%.9g atol=%.9g",
                                            float(identity_error),
                                            float(identity_cfm_error),
                                            float(identity_ratio_mean),
                                            float(identity_ratio_min),
                                            float(identity_ratio_max),
                                            float(identity_ratio_std),
                                            float(identity_clip_fraction),
                                            _PRE_OPTIMIZER_IDENTITY_ATOL,
                                        )
                                policy_loss, policy_metrics = self._compute_clipped_policy_loss(
                                    log_ratio,
                                    micro_batch.batch["fpo.advantages"],
                                    valids,
                                    loss_weights=(None if critic_enabled else micro_batch.batch["fpo.loss_weights"]),
                                    normalization=hierarchical_normalization,
                                )
                                valid_ratios = policy_metrics["ratio"][valids.bool()].float()
                                if valid_ratios.numel():
                                    ratio_sums += torch.stack(
                                        [
                                            valid_ratios.double().sum(),
                                            valid_ratios.double().square().sum(),
                                            torch.tensor(
                                                float(valid_ratios.numel()),
                                                dtype=torch.float64,
                                                device=valid_ratios.device,
                                            ),
                                        ]
                                    )
                                    ratio_extrema[0] = torch.minimum(ratio_extrema[0], valid_ratios.min())
                                    ratio_extrema[1] = torch.maximum(ratio_extrema[1], valid_ratios.max())

                            loss = (
                                policy_loss + self.actor_config.vf_coef * value_loss if critic_enabled else policy_loss
                            )
                        _require_finite("actor loss", policy_loss)
                        _require_finite("value loss", value_loss)
                        _require_finite("combined loss", loss)
                        sync_detailed_timing()
                        timing_totals["fpo_update_forward"] += time.perf_counter() - phase_start
                        phase_start = time.perf_counter()
                        # In GRPO mode each micro-batch numerator is already
                        # divided by the full distributed group weight. Summing
                        # micro-batch gradients is the exact hierarchical loss.
                        backward_loss = loss / grad_accum_steps if critic_enabled else loss
                        backward_loss.backward()
                        sync_detailed_timing()
                        timing_totals["fpo_backward_gradient_sync"] += time.perf_counter() - phase_start

                        if critic_enabled:
                            metric_lists["actor/loss"].append(float(policy_loss.detach()))
                            metric_lists["value/loss"].append(float(value_loss.detach()))
                            metric_lists["fpo/approx_kl"].append(float(policy_metrics["approx_kl"]))
                            metric_lists["fpo/clip_fraction"].append(float(policy_metrics["clip_fraction"]))
                        else:
                            hierarchical_policy_loss += policy_loss.detach()
                            hierarchical_approx_kl += policy_metrics["approx_kl"].detach()
                            hierarchical_clip_fraction += policy_metrics["clip_fraction"].detach()
                        metric_lists["fpo/ratio_mean"].append(policy_metrics["ratio"].mean().detach())
                        if critic_enabled:
                            epoch_kls.append(float(policy_metrics["approx_kl"]))
                        progress_interval = max(grad_accum_steps // 8, 1)
                        if torch.distributed.get_rank() == 0 and (
                            micro_batch_index == 0 or (micro_batch_index + 1) % progress_interval == 0 or sync_gradients
                        ):
                            update_progress.set_postfix(
                                epoch=f"{_epoch + 1}/{self.actor_config.update_epochs}",
                                minibatch=f"{mini_batch_index + 1}/{mini_batches_per_epoch}",
                                microbatch=f"{micro_batch_index + 1}/{grad_accum_steps}",
                                actor_loss=f"{float(hierarchical_policy_loss):.5g}",
                                ratio_mean=f"{float(policy_metrics['ratio'].mean()):.5g}",
                                approx_kl=f"{float(hierarchical_approx_kl):.5g}",
                                clip_fraction=f"{float(hierarchical_clip_fraction):.5g}",
                            )
                finally:
                    self.engine.module.set_requires_gradient_sync(True)
                    self.engine.module.set_is_last_backward(True)

                identity_gate_required = critic_enabled or not self._identity_gate_has_passed
                if not value_only and identity_gate_required and not identity_gate_checked:
                    raise RuntimeError(
                        "FPO pre-optimizer identity gate found no valid rollout slots in the complete optimizer batch."
                    )

                if not critic_enabled:
                    metric_lists["actor/loss"].append(hierarchical_policy_loss.detach())
                    metric_lists["fpo/approx_kl"].append(hierarchical_approx_kl.detach())
                    metric_lists["fpo/clip_fraction"].append(hierarchical_clip_fraction.detach())
                    # Preserve the per-epoch values as well as the historical
                    # across-epoch averages. Epoch 1 starts at theta_old and
                    # therefore has KL=0; averaging it with epoch 2 previously
                    # hid a factor of two in update-magnitude investigations.
                    metric_lists[f"actor/loss_epoch_{_epoch + 1}"].append(
                        hierarchical_policy_loss.detach()
                    )
                    metric_lists[f"fpo/approx_kl_epoch_{_epoch + 1}"].append(
                        hierarchical_approx_kl.detach()
                    )
                    metric_lists[f"fpo/clip_fraction_epoch_{_epoch + 1}"].append(
                        hierarchical_clip_fraction.detach()
                    )
                    epoch_kls.append(float(hierarchical_approx_kl))

                pre_step_kl = hierarchical_approx_kl.detach().clone()
                if not critic_enabled:
                    # With the group-weight denominator corrected for DDP's
                    # gradient averaging, each rank holds W * local/global.
                    # Averaging the rank scalars reconstructs the exact global
                    # logical-batch KL used for the gate.
                    torch.distributed.all_reduce(pre_step_kl, op=torch.distributed.ReduceOp.SUM)
                    pre_step_kl /= torch.distributed.get_world_size()
                metric_lists["fpo/pre_step_old_policy_kl"].append(pre_step_kl)
                metric_lists[f"fpo/pre_step_old_policy_kl_step_{_epoch + 1}"].append(pre_step_kl)
                skip_actor_step = bool(
                    not value_only
                    and getattr(self.actor_config, "kl_early_stop_mode", "post_epoch") == "pre_optimizer"
                    and _epoch > 0
                    and self.actor_config.target_kl is not None
                    and float(pre_step_kl) > float(self.actor_config.target_kl)
                )

                sync_detailed_timing()
                phase_start = time.perf_counter()
                if critic_enabled:
                    assert self.value_optimizer is not None and self.value_scheduler is not None
                    value_grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.value_parameters, max_norm=self.actor_config.value.clip_grad
                    )
                    self.value_optimizer.step()
                    self.value_scheduler.step()
                    for parameter in self.value_parameters:
                        parameter.grad = None

                if skip_actor_step:
                    # Gradients for this epoch have already been accumulated
                    # (so the gate uses the full logical batch), but no model,
                    # optimizer, or scheduler state is advanced.
                    self.engine.optimizer_zero_grad()
                    kl_early_stop_triggered = True
                elif not value_only:
                    actor_grad_norm = self.engine.optimizer_step()
                    self.engine.lr_scheduler_step()
                    metric_lists["actor/grad_norm"].append(float(actor_grad_norm))
                    metric_lists[f"actor/grad_norm_step_{_epoch + 1}"].append(float(actor_grad_norm))
                    optimizer_steps_applied += 1
                sync_detailed_timing()
                timing_totals["fpo_optimizer"] += time.perf_counter() - phase_start
                if critic_enabled:
                    metric_lists["value/grad_norm"].append(float(value_grad_norm))
                update_progress.update(1)

            epochs_run += 1
            mean_epoch_kl = sum(epoch_kls) / max(len(epoch_kls), 1)
            if kl_early_stop_triggered:
                break
            if (
                getattr(self.actor_config, "kl_early_stop_mode", "post_epoch") == "post_epoch"
                and self.actor_config.target_kl is not None
                and mean_epoch_kl > self.actor_config.target_kl
            ):
                break
        update_progress.close()
        self._active_update_epoch = -1

        metric_device = data.batch["fpo.advantages"].device
        valids = data.batch["info.valids"].to(metric_device).float()
        rewards = data.batch["info.rewards"].to(metric_device).float()
        explained_variance = None
        if critic_enabled:
            returns = data.batch["fpo.returns"].float()
            old_values = data.batch["fpo.old_values"].to(metric_device).float()
            sync_detailed_timing()
            phase_start = time.perf_counter()
            explained_variance = distributed_explained_variance(returns, old_values, valids)
            sync_detailed_timing()
            timing_totals["fpo_distributed_metrics"] += time.perf_counter() - phase_start
        valid_count = valids.sum().clamp_min(1.0)

        metrics = {}
        for key, values in metric_lists.items():
            if not values:
                continue
            tensor_values = [
                value.detach().to(device=metric_device, dtype=torch.float32).reshape(())
                if isinstance(value, torch.Tensor)
                else torch.tensor(value, device=metric_device, dtype=torch.float32)
                for value in values
            ]
            metrics[key] = float(torch.stack(tensor_values).mean())
        torch.distributed.all_reduce(ratio_sums, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(ratio_extrema[:1], op=torch.distributed.ReduceOp.MIN)
        torch.distributed.all_reduce(ratio_extrema[1:], op=torch.distributed.ReduceOp.MAX)
        if ratio_sums[2] == 0:
            ratio_sums[0] = 1.0
            ratio_sums[1] = 1.0
            ratio_sums[2] = 1.0
            ratio_extrema.fill_(1.0)
        ratio_count = ratio_sums[2].clamp_min(1.0)
        ratio_mean = ratio_sums[0] / ratio_count
        ratio_std = (ratio_sums[1] / ratio_count - ratio_mean.square()).clamp_min(0.0).sqrt()
        metrics.update(
            {
                "data/reward_mean": float((rewards * valids).sum() / valid_count),
                "data/valid_ratio": float(valids.mean()),
                "actor/lr": float(self.engine.optimizer.param_groups[0]["lr"]),
                "algorithm/critic_enabled": float(critic_enabled),
                "algorithm/grpo_outcome": float(not critic_enabled),
                "algorithm/task_equal_reduction": float(
                    not critic_enabled and accepted_group_reduction == "task_equal"
                ),
                "algorithm/full_logical_batch_gradient_accumulation": float(
                    getattr(self.actor_config, "full_logical_batch_gradient_accumulation", False)
                ),
                "fpo/value_only": float(value_only),
                "fpo/epochs_run": float(epochs_run),
                "fpo/optimizer_steps_applied": float(optimizer_steps_applied),
                "fpo/kl_early_stop_triggered": float(kl_early_stop_triggered),
                "fpo/kl_early_stop_pre_optimizer": float(
                    getattr(self.actor_config, "kl_early_stop_mode", "post_epoch") == "pre_optimizer"
                ),
                "fpo/kl_early_stop_threshold": float(
                    self.actor_config.target_kl if self.actor_config.target_kl is not None else -1.0
                ),
                "fpo/fixed_mc_seed": float(
                    getattr(self.actor_config, "fpo_mc_seed", -1)
                    if getattr(self.actor_config, "fpo_mc_seed", None) is not None
                    else -1
                ),
                "fpo/fixed_shuffle_seed": float(
                    getattr(self.actor_config, "fpo_shuffle_seed", -1)
                    if getattr(self.actor_config, "fpo_shuffle_seed", None) is not None
                    else -1
                ),
                "fpo/advantage_mean": float(advantage_stats["mean"]),
                "fpo/advantage_std": float(advantage_stats["std"]),
                "fpo/advantage_min": float(advantage_stats["min"]),
                "fpo/advantage_max": float(advantage_stats["max"]),
                "fpo/ratio_mean": float(ratio_mean),
                "fpo/ratio_std": float(ratio_std),
                "fpo/ratio_min": float(ratio_extrema[0]),
                "fpo/ratio_max": float(ratio_extrema[1]),
            }
        )
        metrics.update(self._algorithm_metrics())
        metrics["timing_s/per_microbatch_cuda_sync_enabled"] = float(detailed_sync_timing)
        if critic_enabled:
            assert explained_variance is not None and self.value_optimizer is not None
            metrics["value/explained_variance"] = float(explained_variance)
            metrics["value/lr"] = float(self.value_optimizer.param_groups[0]["lr"])
        _synchronize_for_timing()
        if memory_device.type == "cuda":
            memory_peak_stats = torch.tensor(
                [
                    torch.cuda.max_memory_allocated(memory_device),
                    torch.cuda.max_memory_reserved(memory_device),
                ],
                device=memory_device,
                dtype=torch.float64,
            )
        else:
            memory_peak_stats = torch.zeros(2, device=memory_device, dtype=torch.float64)
        torch.distributed.all_reduce(memory_start_stats, op=torch.distributed.ReduceOp.MAX)
        torch.distributed.all_reduce(memory_peak_stats, op=torch.distributed.ReduceOp.MAX)
        gib = float(1024**3)
        metrics["memory/fpo_start_allocated_gib"] = float(memory_start_stats[0] / gib)
        metrics["memory/fpo_start_reserved_gib"] = float(memory_start_stats[1] / gib)
        metrics["memory/fpo_peak_allocated_gib"] = float(memory_peak_stats[0] / gib)
        metrics["memory/fpo_peak_reserved_gib"] = float(memory_peak_stats[1] / gib)
        metrics.update({f"timing_s/{name}": value for name, value in timing_totals.items()})
        metrics["timing_s/fpo_update_total"] = time.perf_counter() - total_start
        return metrics

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="train"), blocking=False)
    def train_mini_batch(self, data: DataProto) -> DataProto:
        self._ensure_fpo_initialized()
        with self.engine.train_mode():
            metrics = self._update_fpo_policy(data)
        return DataProto(meta_info={"metrics": metrics})
