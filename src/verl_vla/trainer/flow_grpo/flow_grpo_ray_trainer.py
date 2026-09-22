from __future__ import annotations

from verl_vla.trainer.grfpo.grfpo_ray_trainer import GRFPORayTrainer


class FlowGRPORayTrainer(GRFPORayTrainer):
    """Flow-GRPO objective with the common same-condition group lifecycle.

    The inherited code ends collection before training and computes per-group
    outcome advantages.  Flow-GRPO's distinct transition-likelihood objective
    lives in ``workers.engine.flow_grpo`` and is selected by its actor config.
    """

    def __init__(self, trainer_config, cluster, tracking_config):
        actor = tracking_config["cluster"]["actor_rollout_ref"]["actor"]
        adapter_flow = tracking_config["cluster"]["actor_rollout_ref"]["model"]["adapter"]["flow_grpo"]
        if actor.get("_target_") != "verl_vla.workers.config.FlowGRPOActorConfig":
            raise ValueError("Flow-GRPO requires FlowGRPOActorConfig/FlowGRPOTrainingWorker.")
        if not adapter_flow.get("enabled", False):
            raise ValueError("Flow-GRPO requires Fast-WAM rollout SDE trace collection.")
        super().__init__(trainer_config, cluster, tracking_config)


__all__ = ["FlowGRPORayTrainer"]
