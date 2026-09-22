# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0

from .collection_store import (
    CandidateLease,
    CollectionWindowSpec,
    CollectionWindowStore,
    PolicyIdentity,
    abandon_owner_leases,
    skip_window_candidates,
)
from .config import GRFPOTrainerConfig
from .group_validation import validate_same_condition_group

__all__ = [
    "CollectionWindowSpec",
    "CollectionWindowStore",
    "CandidateLease",
    "PolicyIdentity",
    "abandon_owner_leases",
    "skip_window_candidates",
    "GRFPOTrainerConfig",
    "GRFPORayTrainer",
    "SpooledGRFPOTrainer",
    "apply_grpo_outcome_advantages",
    "concatenate_rollout_groups_with_padding",
    "prepare_fpo_actor_input",
    "validate_same_condition_group",
]


def __getattr__(name: str):
    """Keep lightweight collection utilities independent of Ray/FSDP extras."""

    trainer_names = {
        "GRFPORayTrainer",
        "apply_grpo_outcome_advantages",
        "concatenate_rollout_groups_with_padding",
        "prepare_fpo_actor_input",
    }
    if name in trainer_names:
        from . import grfpo_ray_trainer

        return getattr(grfpo_ray_trainer, name)
    if name == "SpooledGRFPOTrainer":
        from .spooled_trainer import SpooledGRFPOTrainer

        return SpooledGRFPOTrainer
    raise AttributeError(name)
