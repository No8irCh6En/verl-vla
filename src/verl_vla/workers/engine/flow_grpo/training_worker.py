"""Flow-GRPO transition-likelihood training worker.

The environment/group orchestration is shared with the current on-policy
trainer, while this worker owns the method-specific rollout trace contract and
PPO ratio. It intentionally does not add Flow-GRPO branches to vanilla FPO.
"""

from __future__ import annotations

import logging

import torch
from verl import DataProto
from verl.utils.device import get_device_id

from verl_vla.utils.data import flatten_trajectories
from verl_vla.workers.config import FlowGRPOActorConfig
from verl_vla.workers.engine.fpo.training_worker import FPOTrainingWorker, _require_finite

logger = logging.getLogger(__name__)


def training_transition_count(total: int, fraction: float) -> int:
    """Match official Flow-GRPO's ``int(num_steps * timestep_fraction)``."""

    if total <= 0:
        raise ValueError("Flow-GRPO rollout must contain at least one transition.")
    if not 0 < fraction <= 1:
        raise ValueError("Flow-GRPO transition fraction must lie in (0, 1].")
    return max(1, int(total * fraction))


def compute_flow_grpo_log_ratio(
    old_log_probs: torch.Tensor,
    current_log_probs: torch.Tensor,
    action_valids: torch.Tensor,
) -> torch.Tensor:
    """Return ``log p_current - log p_old`` for each stochastic flow step.

    The model already averages over action dimension D. H is averaged only
    over actions actually executed by RoboDojo, yielding ``[B,T]``.
    """

    if old_log_probs.shape != current_log_probs.shape or old_log_probs.ndim != 3:
        raise ValueError("old/current Flow-GRPO log probabilities must match with shape [B,T,H].")
    if action_valids.shape != (old_log_probs.shape[0], old_log_probs.shape[2]):
        raise ValueError("action_valids must have shape [B,H] matching Flow-GRPO log probabilities.")
    mask = action_valids.to(device=old_log_probs.device, dtype=old_log_probs.dtype).unsqueeze(1)
    denominator = mask.sum(dim=-1).clamp_min(1.0)
    return ((current_log_probs - old_log_probs) * mask).sum(dim=-1) / denominator


def compute_replay_calibrated_flow_grpo_log_ratio(
    rollout_old_log_probs: torch.Tensor,
    replay_old_log_probs: torch.Tensor,
    current_log_probs: torch.Tensor,
    action_valids: torch.Tensor,
) -> torch.Tensor:
    """Return the official current/old ratio with a frozen replay correction.

    Some flow backbones are numerically batch dependent.  The rollout path uses
    a vectorized group batch while actor replay may require a smaller physical
    microbatch.  ``replay_old_log_probs`` is therefore materialized once under
    theta_old, before any optimizer step, and remains frozen for the complete
    update.  Its difference from the recorded rollout likelihood is a frozen
    path correction, not a moving current-policy denominator.

    Algebraically the corrected ratio equals ``current - replay_old`` while
    retaining the recorded rollout likelihood as the behavior-policy anchor.
    """

    if rollout_old_log_probs.shape != replay_old_log_probs.shape:
        raise ValueError("rollout/replay theta_old Flow-GRPO log probabilities must match.")
    replay_correction = (rollout_old_log_probs - replay_old_log_probs).detach()
    corrected_current = current_log_probs + replay_correction
    return compute_flow_grpo_log_ratio(
        rollout_old_log_probs,
        corrected_current,
        action_valids,
    )


def flow_grpo_clipped_policy_loss(
    log_ratio: torch.Tensor,
    advantages: torch.Tensor,
    valids: torch.Tensor,
    clip_coef: float,
    *,
    loss_weights: torch.Tensor | None = None,
    normalization: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Average per-transition PPO losses while preserving one chunk weight."""

    if log_ratio.ndim != 2:
        raise ValueError("Flow-GRPO log_ratio must have shape [B,T].")
    if advantages.shape != log_ratio.shape[:1] or valids.shape != log_ratio.shape[:1]:
        raise ValueError("Flow-GRPO advantages/valids must have shape [B].")
    ratio = log_ratio.exp()
    advantages_t = advantages.unsqueeze(-1)
    unclipped = -advantages_t * ratio
    clipped = -advantages_t * ratio.clamp(1.0 - clip_coef, 1.0 + clip_coef)
    chunk_weights = valids if loss_weights is None else loss_weights * valids
    transition_weights = chunk_weights.unsqueeze(-1) / float(log_ratio.shape[1])
    denominator = chunk_weights.sum().clamp_min(1.0) if normalization is None else normalization.clamp_min(1.0)
    loss = (torch.maximum(unclipped, clipped) * transition_weights).sum() / denominator
    with torch.no_grad():
        # Diagnostic used by the official Flow-GRPO trainer.
        approx_kl = (0.5 * log_ratio.square() * transition_weights).sum() / denominator
        clip_fraction = (((ratio - 1.0).abs() > clip_coef).float() * transition_weights).sum() / denominator
    return loss, {"ratio": ratio.detach(), "approx_kl": approx_kl, "clip_fraction": clip_fraction}


class FlowGRPOTrainingWorker(FPOTrainingWorker):
    """Critic-free Flow-GRPO worker using recorded Fast-WAM SDE transitions."""

    def __init__(self, config, actor_config: FlowGRPOActorConfig, tokenizer=None):
        super().__init__(config=config, actor_config=actor_config, tokenizer=tokenizer)
        self._frozen_replay_old_log_probs: torch.Tensor | None = None
        self._frozen_reference_initialized: torch.Tensor | None = None
        self._rollout_replay_error_stats: torch.Tensor | None = None
        self._rollout_replay_error_max: torch.Tensor | None = None
        self._frozen_reference_validated = False

    def _ensure_fpo_initialized(self) -> None:
        """Initialize Flow-GRPO without invoking the vanilla-FPO CFM contract."""

        if self._fpo_initialized:
            return
        self.engine.module.flow_grpo_init()
        self.value_parameters = []
        self.value_optimizer = None
        self.value_scheduler = None
        self._fpo_initialized = True

    def _prepare_training_batch(self, data: DataProto) -> DataProto:
        full_actions = data.batch["action.full_action"]
        preflattened = bool(data.meta_info.get("fpo_preflattened_valid_chunks", False))
        if preflattened:
            if full_actions.ndim != 3 or data.batch["info.valids"].ndim != 1:
                raise ValueError("Preflattened Flow-GRPO input must use action [V,H,D] and valids [V].")
            slot_count, action_horizon, action_dim = full_actions.shape
        else:
            if full_actions.ndim != 4 or data.batch["info.valids"].ndim != 2:
                raise ValueError("Trajectory-shaped Flow-GRPO input must use action [B,S,H,D] and valids [B,S].")
            batch_size, rollout_steps, action_horizon, action_dim = full_actions.shape
        required = {
            "action.flow_grpo.latents",
            "action.flow_grpo.old_log_probs",
            "action.flow_grpo.sigmas",
            "action.flow_grpo.deltas",
        }
        missing = sorted(required - set(data.batch.keys()))
        if missing:
            raise KeyError(f"Flow-GRPO rollout is missing recorded transition fields: {missing}")

        latents = data.batch["action.flow_grpo.latents"]
        old_log_probs = data.batch["action.flow_grpo.old_log_probs"]
        sigmas = data.batch["action.flow_grpo.sigmas"]
        deltas = data.batch["action.flow_grpo.deltas"]
        if preflattened:
            if (
                latents.ndim != 4
                or latents.shape[:1] != (slot_count,)
                or latents.shape[2:]
                != (
                    action_horizon,
                    action_dim,
                )
            ):
                raise ValueError("Preflattened Flow-GRPO latents must have shape [V,T+1,H,D].")
            transition_count = latents.shape[1] - 1
            if old_log_probs.shape != (slot_count, transition_count, action_horizon):
                raise ValueError("Preflattened Flow-GRPO old log probabilities must have shape [V,T,H].")
            if sigmas.shape != (slot_count, transition_count):
                raise ValueError("Preflattened Flow-GRPO sigmas must have shape [V,T].")
        else:
            if latents.shape[:2] != (batch_size, rollout_steps) or latents.shape[3:] != (
                action_horizon,
                action_dim,
            ):
                raise ValueError("Flow-GRPO latents must have shape [B,S,T+1,H,D].")
            transition_count = latents.shape[2] - 1
            if old_log_probs.shape != (batch_size, rollout_steps, transition_count, action_horizon):
                raise ValueError("Flow-GRPO old log probabilities must have shape [B,S,T,H].")
            if sigmas.shape != (batch_size, rollout_steps, transition_count):
                raise ValueError("Flow-GRPO sigmas must have shape [B,S,T].")
        if deltas.shape != sigmas.shape:
            raise ValueError("Flow-GRPO deltas must match sigma shape [B,S,T].")
        for name, value in (
            ("latents", latents),
            ("old log probabilities", old_log_probs),
            ("sigmas", sigmas),
            ("deltas", deltas),
        ):
            _require_finite(f"Flow-GRPO {name}", value)

        if data.meta_info.get("advantage_estimator") != "grpo_outcome":
            raise ValueError("Flow-GRPO requires precomputed same-group outcome advantages.")
        if "fpo.advantages" not in data.batch or "fpo.loss_weights" not in data.batch:
            raise KeyError("Flow-GRPO requires group advantages and hierarchical loss weights.")
        device = get_device_id()
        _require_finite("Flow-GRPO advantages", data.batch["fpo.advantages"])
        _require_finite("Flow-GRPO trajectory-balanced loss weights", data.batch["fpo.loss_weights"])
        data.batch["fpo.advantages"] = data.batch["fpo.advantages"].to(device)
        data.batch["fpo.loss_weights"] = data.batch["fpo.loss_weights"].to(device)

        # The durable large-batch trainer removes all zero-weight trajectory
        # padding before dispatch. Preserve that representation so Flow-GRPO
        # receives the same padding-compute elimination as FPO.
        flat = data if preflattened else flatten_trajectories(data, reference_key="action.action")
        trained_transition_count = training_transition_count(
            transition_count,
            self.actor_config.train_transition_fraction,
        )
        flat.batch["flow_grpo.reference_index"] = torch.arange(
            len(flat),
            device=device,
            dtype=torch.long,
        )
        # A frozen theta_old replay reference is small compared with the model
        # activations: [flattened chunks, trained transitions, action horizon].
        # Keeping it on the actor device avoids one CPU synchronization per
        # physical microbatch.
        self._frozen_replay_old_log_probs = torch.empty(
            len(flat),
            trained_transition_count,
            action_horizon,
            device=device,
            dtype=torch.float32,
        )
        self._frozen_reference_initialized = torch.zeros(
            len(flat),
            device=device,
            dtype=torch.bool,
        )
        self._rollout_replay_error_stats = torch.zeros(3, device=device, dtype=torch.float64)
        self._rollout_replay_error_max = torch.zeros((), device=device, dtype=torch.float64)
        self._frozen_reference_validated = False
        # Validate every freshly collected theta_old batch, not just the first
        # update handled by this long-lived actor worker.
        self._identity_gate_has_passed = False
        return flat

    def _compute_policy_log_ratio(
        self,
        micro_batch: DataProto,
        obs: DataProto,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        total_transitions = micro_batch.batch["action.flow_grpo.sigmas"].shape[1]
        transition_count = training_transition_count(
            total_transitions,
            self.actor_config.train_transition_fraction,
        )
        current = self.engine.module.flow_grpo_log_probs(
            obs,
            self.tokenizer,
            micro_batch.batch["action.flow_grpo.latents"][:, : transition_count + 1],
            micro_batch.batch["action.flow_grpo.sigmas"][:, :transition_count],
            micro_batch.batch["action.flow_grpo.deltas"][:, :transition_count],
        )
        rollout_old = micro_batch.batch["action.flow_grpo.old_log_probs"][:, :transition_count]
        if (
            self._frozen_replay_old_log_probs is None
            or self._frozen_reference_initialized is None
            or self._rollout_replay_error_stats is None
            or self._rollout_replay_error_max is None
        ):
            raise RuntimeError("Flow-GRPO theta_old reference was not initialized for this rollout batch.")

        reference_indices = micro_batch.batch["flow_grpo.reference_index"].long()
        update_epoch = int(getattr(self, "_active_update_epoch", -1))
        if update_epoch < 0:
            raise RuntimeError("Flow-GRPO log-prob evaluation occurred outside an active update epoch.")
        if update_epoch > 0:
            if not self._frozen_reference_validated:
                if not bool(self._frozen_reference_initialized.all()):
                    raise RuntimeError(
                        "Flow-GRPO attempted a post-update replay before every sample had a frozen theta_old reference."
                    )
                self._frozen_reference_validated = True
            replay_old = self._frozen_replay_old_log_probs[reference_indices]
        else:
            # The first full-batch pass happens entirely before optimizer.step.
            # Materialize that exact training-path likelihood once; later
            # passes must reuse it after theta changes.
            replay_old = current.detach().float()
            self._frozen_replay_old_log_probs[reference_indices] = replay_old
            self._frozen_reference_initialized[reference_indices] = True

            valid_mask = micro_batch.batch["info.action_valids"].bool().unsqueeze(1).expand_as(
                replay_old
            ) & micro_batch.batch["info.valids"].bool().reshape(-1, 1, 1)
            replay_error = (rollout_old.detach().float() - replay_old).abs()[valid_mask]
            if replay_error.numel():
                self._rollout_replay_error_stats += torch.stack(
                    (
                        replay_error.double().sum(),
                        replay_error.double().square().sum(),
                        replay_error.new_tensor(float(replay_error.numel()), dtype=torch.float64),
                    )
                )
                self._rollout_replay_error_max = torch.maximum(
                    self._rollout_replay_error_max,
                    replay_error.max().double(),
                )

        replay_correction = (rollout_old - replay_old).detach()
        corrected_current = current + replay_correction
        return (
            compute_replay_calibrated_flow_grpo_log_ratio(
                rollout_old,
                replay_old,
                current,
                micro_batch.batch["info.action_valids"],
            ),
            rollout_old - corrected_current.detach(),
            micro_batch.batch["info.action_valids"].bool().unsqueeze(1),
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
        return flow_grpo_clipped_policy_loss(
            log_ratio,
            advantages.clamp(
                -self.actor_config.advantage_clip_max,
                self.actor_config.advantage_clip_max,
            ),
            valids,
            self.actor_config.clip_coef,
            loss_weights=loss_weights,
            normalization=normalization,
        )

    def _algorithm_metrics(self) -> dict[str, float]:
        metrics = {
            "algorithm/fpo_uniform_unweighted": 0.0,
            "algorithm/vanilla_fpo_ratio": 0.0,
            "algorithm/fpo_plus_plus_per_mc_ratio": 0.0,
            "algorithm/fpo_plus_plus_aspo": 0.0,
            "algorithm/flow_grpo_ratio": 1.0,
            "algorithm/flow_grpo_frozen_old_reference": 1.0,
            "algorithm/flow_grpo_replay_calibrated_reference": 1.0,
        }
        if self._frozen_reference_initialized is None or not bool(self._frozen_reference_initialized.all()):
            raise RuntimeError("Flow-GRPO update ended without materializing every theta_old reference.")
        if self._rollout_replay_error_stats is None or self._rollout_replay_error_max is None:
            raise RuntimeError("Flow-GRPO rollout/replay diagnostics were not initialized.")
        stats = self._rollout_replay_error_stats.clone()
        maximum = self._rollout_replay_error_max.clone()
        torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(maximum, op=torch.distributed.ReduceOp.MAX)
        count = stats[2].clamp_min(1.0)
        mean = stats[0] / count
        variance = (stats[1] / count - mean.square()).clamp_min(0.0)
        metrics.update(
            {
                "flow_grpo/rollout_replay_logprob_delta_mean": float(mean),
                "flow_grpo/rollout_replay_logprob_delta_std": float(variance.sqrt()),
                "flow_grpo/rollout_replay_logprob_delta_max": float(maximum),
                "flow_grpo/frozen_old_reference_fraction": float(self._frozen_reference_initialized.float().mean()),
            }
        )
        return metrics
