from __future__ import annotations

from dataclasses import dataclass

from verl_vla.trainer.grfpo.config import GRFPOTrainerConfig


@dataclass
class FlowGRPOTrainerConfig(GRFPOTrainerConfig):
    """Same-condition collection settings shared with GRFPO.

    The actor worker and likelihood objective remain Flow-GRPO-specific; this
    dataclass reuses only the group/on-policy collection contract.
    """

    _target_: str = "verl_vla.trainer.flow_grpo.config.FlowGRPOTrainerConfig"
    advantage_estimator: str = "grpo_outcome"
    informative_group_sampling: bool = True

    def __post_init__(self):
        super().__post_init__()
        if self.advantage_estimator != "grpo_outcome":
            raise ValueError("Flow-GRPO requires same-condition group outcome advantages.")


__all__ = ["FlowGRPOTrainerConfig"]
