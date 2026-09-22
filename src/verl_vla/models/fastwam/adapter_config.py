# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Framework-owned configuration for the native Fast-WAM rollout adapter."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FastWAMFPOConfig:
    """Critic-free FPO additions to the native Fast-WAM model."""

    enabled: bool = False
    value_enabled: bool = False
    # Parse-only compatibility for archived Phase 6B/6C Hydra configs. These
    # fields are never used to construct parameters in the Fast-WAM adapter.
    value_input_dim: int | None = None
    value_hidden_dims: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "value_hidden_dims", tuple(int(dim) for dim in self.value_hidden_dims))
        if self.value_enabled:
            raise ValueError(
                "Fast-WAM FPO is critic-free: value_enabled=true, Value Head, and GAE are unsupported."
            )


@dataclass(frozen=True)
class FastWAMFlowGRPOConfig:
    """Model-side stochastic rollout trace required by Flow-GRPO."""

    enabled: bool = False
    # Fast-WAM's normalized action flow is more sensitive than both image
    # diffusion and the other action policies currently shipped by verl-vla.
    # Online calibration on the fixed c2000 policy gave 72/72 failures at both
    # 0.7 (job 547360) and 0.08 (job 547379); use a conservative non-zero SDE
    # scale that remains close to the pretrained ODE policy.
    noise_level: float = 0.01
    # Number of recorded SDE transitions evaluated in one action-expert/MoT
    # forward. This changes only physical batching: transition likelihoods,
    # reduction, and the frozen-old objective are identical to batch size 1.
    transition_batch_size: int = 1

    def __post_init__(self) -> None:
        if not math.isfinite(self.noise_level) or self.noise_level <= 0:
            raise ValueError("Flow-GRPO noise_level must be finite and positive.")
        if self.transition_batch_size <= 0:
            raise ValueError("Flow-GRPO transition_batch_size must be positive.")


@dataclass(frozen=True)
class FastWAMAdapterConfig:
    """Paths and deployment semantics not stored in the native `.pt` file."""

    policy_root: str
    checkpoint_sha256: str
    weights_relative_path: str = "checkpoints/weights/step_020000.pt"
    dataset_stats_relative_path: str = "dataset_stats.json"
    action_horizon: int = 32
    action_chunk_size: int = 24
    num_inference_steps: int = 10
    rollout_action_latent_scale: float = 1.0
    rollout_seed_mode: str = "episode"
    sigma_shift: float | None = None
    seed: int | None = 0
    text_cfg_scale: float = 1.0
    negative_prompt: str = ""
    rand_device: str = "cpu"
    tiled: bool = False
    sim_cfg_name: str = "sim_robotwin.yaml"
    sim_task: str = "robotwin_uncond_3cam_384_1e-4"
    fpo: FastWAMFPOConfig | Mapping[str, Any] = field(default_factory=FastWAMFPOConfig)
    flow_grpo: FastWAMFlowGRPOConfig | Mapping[str, Any] = field(
        default_factory=FastWAMFlowGRPOConfig
    )

    def __post_init__(self) -> None:
        if isinstance(self.fpo, Mapping):
            object.__setattr__(self, "fpo", FastWAMFPOConfig(**dict(self.fpo)))
        elif not isinstance(self.fpo, FastWAMFPOConfig):
            raise TypeError(f"fpo must be FastWAMFPOConfig or Mapping, got {type(self.fpo).__name__}.")
        if isinstance(self.flow_grpo, Mapping):
            object.__setattr__(
                self,
                "flow_grpo",
                FastWAMFlowGRPOConfig(**dict(self.flow_grpo)),
            )
        elif not isinstance(self.flow_grpo, FastWAMFlowGRPOConfig):
            raise TypeError(
                "flow_grpo must be FastWAMFlowGRPOConfig or Mapping, "
                f"got {type(self.flow_grpo).__name__}."
            )
        if self.action_horizon != 32:
            raise ValueError(f"Fast-WAM native action_horizon must remain 32, got {self.action_horizon}.")
        if self.action_chunk_size != 24:
            raise ValueError(f"RoboDojo Fast-WAM action_chunk_size must remain 24, got {self.action_chunk_size}.")
        if self.num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive.")
        if not math.isfinite(self.rollout_action_latent_scale) or self.rollout_action_latent_scale <= 0:
            raise ValueError("rollout_action_latent_scale must be finite and positive.")
        if self.rollout_seed_mode not in {"episode", "per_policy_call"}:
            raise ValueError("rollout_seed_mode must be 'episode' or 'per_policy_call'.")
        if len(self.checkpoint_sha256) != 64:
            raise ValueError("checkpoint_sha256 must be a 64-character SHA256 digest.")

    def save_pretrained(self, save_directory: str) -> None:
        """The full verl checkpoint owns adapter state; Fast-WAM owns native files."""

        del save_directory


__all__ = ["FastWAMAdapterConfig", "FastWAMFPOConfig", "FastWAMFlowGRPOConfig"]
