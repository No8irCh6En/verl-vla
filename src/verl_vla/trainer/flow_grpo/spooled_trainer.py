"""Preempt-safe Flow-GRPO actor update over the shared durable group store."""

from __future__ import annotations

from typing import Any

from verl_vla.trainer.grfpo.spooled_trainer import SpooledGRFPOTrainer


class SpooledFlowGRPOTrainer(SpooledGRFPOTrainer):
    """Reuse collection/reduction/streaming while keeping a Flow-only actor."""

    actor_objective = "flow_grpo"

    def _validate_actor_contract(self, *, actor: dict[str, Any], model: dict[str, Any]) -> None:
        if actor.get("_target_") != "verl_vla.workers.config.FlowGRPOActorConfig":
            raise ValueError("Spooled Flow-GRPO requires FlowGRPOActorConfig.")
        if bool(actor["value"]["enabled"]) or float(actor["vf_coef"]) != 0:
            raise ValueError("Spooled Flow-GRPO is critic-free.")
        if bool(actor["normalize_advantages"]):
            raise ValueError("Spooled Flow-GRPO forbids minibatch advantage normalization.")
        adapter = model["adapter"]
        if bool(adapter.get("fpo", {}).get("enabled", False)):
            raise ValueError("Spooled Flow-GRPO must not enable the FPO CFM objective.")
        if not bool(adapter.get("flow_grpo", {}).get("enabled", False)):
            raise ValueError("Spooled Flow-GRPO requires model.adapter.flow_grpo.enabled=true.")
        configured_noise = float(adapter["flow_grpo"]["noise_level"])
        if configured_noise != float(self.store.spec.flow_grpo_noise_level):
            raise ValueError(
                "Flow-GRPO rollout/training noise-level mismatch: "
                f"window={self.store.spec.flow_grpo_noise_level}, actor={configured_noise}."
            )

    def run_task_gradient_diagnostic(self, output_path):
        del output_path
        raise NotImplementedError(
            "The fixed-CFM task-gradient diagnostic is FPO-specific and cannot be used for Flow-GRPO."
        )


__all__ = ["SpooledFlowGRPOTrainer"]
