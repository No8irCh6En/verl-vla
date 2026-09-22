# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field

from verl.base_config import BaseConfig
from verl.trainer.config import CheckpointConfig
from verl.utils.profiler.config import ProfilerConfig
from verl.workers.config.model import HFModelConfig

from .engine import FSDPEngineConfig
from .optimizer import FSDPOptimizerConfig

__all__ = [
    "SACConfig",
    "SACCriticConfig",
    "SACReplayConfig",
    "SACTD3Config",
    "SACCQLConfig",
    "ACPConfig",
    "FPOValueConfig",
    "ActorDataKeysConfig",
    "BaseVLAActorConfig",
    "ActorConfig",
    "FPOActorConfig",
    "FlowGRPOActorConfig",
    "SFTActorConfig",
]


@dataclass
class SACConfig(BaseConfig):
    """Configuration for Soft Actor-Critic specific training behavior."""

    initial_alpha: float = 0.0
    auto_entropy: bool = False
    alpha_type: str = "exp"
    alpha_lr: float = 3e-4
    target_entropy: float = -64.0
    backup_entropy: bool = True

    def __post_init__(self):
        valid_alpha_types = ["exp", "softplus"]
        if self.alpha_type not in valid_alpha_types:
            raise ValueError(f"Invalid alpha_type: {self.alpha_type}. Must be one of {valid_alpha_types}")
        if self.auto_entropy and self.initial_alpha <= 0:
            raise ValueError(f"initial_alpha must be positive when auto_entropy is enabled, got {self.initial_alpha}")


@dataclass
class SACTD3Config(BaseConfig):
    """Configuration for optional TD3+BC actor loss."""

    enabled: bool = False
    bc_alpha: float = 2.5

    def __post_init__(self):
        if self.bc_alpha <= 0:
            raise ValueError(f"td3 bc_alpha must be positive, got {self.bc_alpha}")


@dataclass
class SACCQLConfig(BaseConfig):
    """Configuration for optional Conservative Q-Learning critic regularization."""

    enabled: bool = False
    alpha: float = 1.0
    temperature: float = 1.0
    noise_scale: float | None = None

    def __post_init__(self):
        if self.alpha < 0:
            raise ValueError(f"cql alpha must be non-negative, got {self.alpha}")
        if self.temperature <= 0:
            raise ValueError(f"cql temperature must be positive, got {self.temperature}")
        if self.noise_scale is not None and self.noise_scale < 0:
            raise ValueError(f"cql noise_scale must be non-negative when provided, got {self.noise_scale}")


@dataclass
class SACCriticConfig(BaseConfig):
    """Configuration for SAC critic optimizer and update schedule."""

    gamma: float = 0.99
    tau: float = 0.25
    force_target_tau_one_in_warmup: bool = True
    skip_update_when_actor_update: bool = False
    lr: float = 1e-4
    weight_decay: float = 0.0
    grad_clip: float | None = None
    warmup_steps: int = 0
    only_steps_after_rollout: int = 0
    resample_target_action: bool = True

    def __post_init__(self):
        if self.gamma <= 0:
            raise ValueError(f"critic gamma must be positive, got {self.gamma}")
        if self.tau <= 0:
            raise ValueError(f"critic tau must be positive, got {self.tau}")
        if self.lr <= 0:
            raise ValueError(f"critic lr must be positive, got {self.lr}")
        if self.grad_clip is not None and self.grad_clip <= 0:
            raise ValueError(f"critic grad_clip must be positive when provided, got {self.grad_clip}")
        if self.warmup_steps < 0:
            raise ValueError(f"critic warmup_steps must be non-negative, got {self.warmup_steps}")
        if self.only_steps_after_rollout < 0:
            raise ValueError(
                f"critic only_steps_after_rollout must be non-negative, got {self.only_steps_after_rollout}"
            )


@dataclass
class SACReplayConfig(BaseConfig):
    """Configuration for SAC replay sampling, batching, and persistence."""

    critic_positive_sample_ratio: float = 0.5
    actor_positive_sample_ratio: float = 0.5
    online_sample_batch_size: int | None = None
    offline_sample_batch_size: int | None = None
    save_interval: int = 500
    online_single_size: int = 1000
    offline_single_size: int = 1000
    save_dir: str = "/tmp/replay_pools"

    def __post_init__(self):
        if not 0 <= self.critic_positive_sample_ratio <= 1:
            raise ValueError(
                f"replay critic_positive_sample_ratio must be in [0, 1], got {self.critic_positive_sample_ratio}"
            )
        if not 0 <= self.actor_positive_sample_ratio <= 1:
            raise ValueError(
                f"replay actor_positive_sample_ratio must be in [0, 1], got {self.actor_positive_sample_ratio}"
            )
        if self.online_sample_batch_size is not None and self.online_sample_batch_size < 0:
            raise ValueError(
                "replay online_sample_batch_size must be non-negative when provided, "
                f"got {self.online_sample_batch_size}"
            )
        if self.offline_sample_batch_size is not None and self.offline_sample_batch_size < 0:
            raise ValueError(
                "replay offline_sample_batch_size must be non-negative when provided, "
                f"got {self.offline_sample_batch_size}"
            )
        if self.save_interval <= 0:
            raise ValueError(f"replay save_interval must be positive, got {self.save_interval}")
        if self.online_single_size <= 0:
            raise ValueError(f"replay online_single_size must be positive, got {self.online_single_size}")
        if self.offline_single_size <= 0:
            raise ValueError(f"replay offline_single_size must be positive, got {self.offline_single_size}")


@dataclass
class ACPConfig(BaseConfig):
    """Configuration for advantage-conditioned prompt tagging."""

    enable: bool = False
    indicator_dropout_prob: float = 0.0
    positive_tag: str = "Advantage: positive"
    negative_tag: str = "Advantage: negative"

    def __post_init__(self):
        if not 0 <= self.indicator_dropout_prob <= 1:
            raise ValueError(f"ACP indicator_dropout_prob must be in [0, 1], got {self.indicator_dropout_prob}")


@dataclass
class FPOValueConfig(BaseConfig):
    """Optimizer settings for the vanilla-FPO state-value head."""

    enabled: bool = True
    lr: float = 1e-4
    weight_decay: float = 0.0
    clip_grad: float = 25.0

    def __post_init__(self):
        if self.enabled and self.lr <= 0:
            raise ValueError(f"fpo value lr must be positive, got {self.lr}")
        if self.weight_decay < 0:
            raise ValueError(f"fpo value weight_decay must be non-negative, got {self.weight_decay}")
        if self.clip_grad <= 0:
            raise ValueError(f"fpo value clip_grad must be positive, got {self.clip_grad}")


@dataclass
class ActorDataKeysConfig(BaseConfig):
    """Batch field names shared by actor training and rollout."""

    task: str = "task"
    action: str = "action"
    action_mask: str | None = "action_is_pad"
    indicator: str | None = None
    target_value: str | None = None


@dataclass
class BaseVLAActorConfig(BaseConfig):
    """Shared actor config used by algorithm-specific VLA actor configs."""

    _mutable_fields = BaseConfig._mutable_fields | {
        "engine",
        "data_keys",
        "model_config",
    }

    strategy: str = "fsdp"

    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    optim: FSDPOptimizerConfig = field(default_factory=FSDPOptimizerConfig)
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    fsdp_config: FSDPEngineConfig = field(default_factory=FSDPEngineConfig)
    data_keys: ActorDataKeysConfig = field(default_factory=ActorDataKeysConfig)
    engine: BaseConfig = field(default_factory=BaseConfig)
    model_config: HFModelConfig | None = None

    def __post_init__(self):
        if self.strategy not in {"fsdp", "fsdp2"}:
            raise ValueError(f"Unsupported actor strategy: {self.strategy}")
        self.engine = self.fsdp_config


@dataclass
class ActorConfig(BaseVLAActorConfig):
    """SAC actor config with local FSDP/optimizer config types."""

    _target_: str = "verl_vla.workers.config.ActorConfig"

    sac: SACConfig = field(default_factory=SACConfig)
    td3: SACTD3Config = field(default_factory=SACTD3Config)
    cql: SACCQLConfig = field(default_factory=SACCQLConfig)
    critic: SACCriticConfig = field(default_factory=SACCriticConfig)
    replay: SACReplayConfig = field(default_factory=SACReplayConfig)

    actor_update_interval: int = 1
    ema_decay: float | None = None
    mini_batch_size: int = 256
    micro_batch_size: int = 16

    def __post_init__(self):
        super().__post_init__()
        if self.actor_update_interval <= 0:
            raise ValueError(f"actor_update_interval must be positive, got {self.actor_update_interval}")
        if self.ema_decay is not None and not 0 < self.ema_decay < 1:
            raise ValueError(f"ema_decay must be in (0, 1) when provided, got {self.ema_decay}")
        if self.mini_batch_size <= 0:
            raise ValueError(f"mini_batch_size must be positive, got {self.mini_batch_size}")
        if self.micro_batch_size <= 0:
            raise ValueError(f"micro_batch_size must be positive, got {self.micro_batch_size}")


@dataclass
class FPOActorConfig(BaseVLAActorConfig):
    """Vanilla Flow Policy Optimization actor update configuration."""

    _target_: str = "verl_vla.workers.config.FPOActorConfig"

    value: FPOValueConfig = field(default_factory=FPOValueConfig)
    mini_batch_size: int = 128
    micro_batch_size: int = 4
    update_epochs: int = 2
    n_action_samples: int = 4
    clip_coef: float = 0.01
    # ``vanilla`` averages fixed-MC CFM losses before exponentiation.  The
    # manipulation form of FPO++ keeps one ratio per MC sample and clips each
    # sample independently.
    fpo_variant: str = "vanilla"
    # Official manipulation FPO++ uses PPO; ASPO remains an explicit ablation
    # because the authors report that it can hurt fine-tuning performance.
    trust_region_mode: str = "ppo"
    spo_clip_coef: float = 0.01
    cfm_loss_kernel: str = "mse"
    huber_delta: float = 1.0
    log_ratio_clamp: float | None = None
    vf_coef: float = 1.0
    normalize_advantages: bool = True
    value_only_updates: int = 1
    target_kl: float | None = 0.1
    # ``post_epoch`` preserves the historical behavior: the optimizer step is
    # applied before target_kl is inspected. ``pre_optimizer`` is an explicit
    # trust-region ablation that may discard a fully accumulated later epoch
    # before its optimizer step. It is deliberately opt-in.
    kl_early_stop_mode: str = "post_epoch"
    # Optional deterministic streams for fixed-MC counterfactual replays.
    # They are independent so data-order and CFM-noise effects can be held
    # fixed while changing exactly one optimizer/trust-region variable.
    fpo_mc_seed: int | None = None
    fpo_shuffle_seed: int | None = None
    # Loading a resumable checkpoint also restores the optimizer param-group
    # LR. This explicit post-resume override prevents a half-LR ablation from
    # silently continuing with the checkpoint's original LR.
    post_resume_lr_override: float | None = None
    # Stream every local shard through micro-batches and take one optimizer
    # step per update epoch. This keeps a variable-size multi-group rollout as
    # one logical gradient batch, including an allowed partial M>=4 update.
    full_logical_batch_gradient_accumulation: bool = False
    # Exact per-section wall-time profiling inserts CUDA synchronization around
    # every physical micro-batch. Keep it opt-in for production training;
    # total update time is synchronized and accurate in either mode.
    profile_cuda_sync_timing: bool = False

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.value, FPOValueConfig):
            from hydra.utils import instantiate

            object.__setattr__(self, "value", instantiate(self.value))
        if self.mini_batch_size <= 0:
            raise ValueError(f"fpo mini_batch_size must be positive, got {self.mini_batch_size}")
        if self.micro_batch_size <= 0:
            raise ValueError(f"fpo micro_batch_size must be positive, got {self.micro_batch_size}")
        if self.update_epochs <= 0:
            raise ValueError(f"fpo update_epochs must be positive, got {self.update_epochs}")
        if self.n_action_samples <= 0:
            raise ValueError(f"fpo n_action_samples must be positive, got {self.n_action_samples}")
        if not 0 < self.clip_coef < 1:
            raise ValueError(f"fpo clip_coef must be in (0, 1), got {self.clip_coef}")
        if self.fpo_variant not in {"vanilla", "fpo_plus_plus"}:
            raise ValueError(f"fpo_variant must be 'vanilla' or 'fpo_plus_plus', got {self.fpo_variant!r}.")
        if self.trust_region_mode not in {"ppo", "aspo"}:
            raise ValueError(f"trust_region_mode must be 'ppo' or 'aspo', got {self.trust_region_mode!r}.")
        if self.trust_region_mode == "aspo" and self.fpo_variant != "fpo_plus_plus":
            raise ValueError("ASPO is only defined for fpo_variant='fpo_plus_plus'.")
        if self.spo_clip_coef <= 0:
            raise ValueError("spo_clip_coef must be positive.")
        if self.cfm_loss_kernel not in {"mse", "huber"}:
            raise ValueError("cfm_loss_kernel must be 'mse' or 'huber'.")
        if self.huber_delta <= 0:
            raise ValueError("huber_delta must be positive.")
        if self.log_ratio_clamp is not None and self.log_ratio_clamp <= 0:
            raise ValueError("log_ratio_clamp must be positive when configured.")
        if self.fpo_variant == "fpo_plus_plus":
            if self.cfm_loss_kernel != "huber":
                raise ValueError("Full FPO++ requires cfm_loss_kernel='huber'.")
        if self.vf_coef < 0:
            raise ValueError(f"fpo vf_coef must be non-negative, got {self.vf_coef}")
        if self.value_only_updates < 0:
            raise ValueError(f"fpo value_only_updates must be non-negative, got {self.value_only_updates}")
        if not self.value.enabled:
            if self.vf_coef != 0:
                raise ValueError("critic-free FPO requires vf_coef=0.")
            if self.value_only_updates != 0:
                raise ValueError("critic-free FPO requires value_only_updates=0.")
            if self.normalize_advantages:
                raise ValueError("critic-free group-relative FPO requires normalize_advantages=false.")
        if self.target_kl is not None and self.target_kl <= 0:
            raise ValueError(f"fpo target_kl must be positive when provided, got {self.target_kl}")
        if self.kl_early_stop_mode not in {"post_epoch", "pre_optimizer"}:
            raise ValueError(
                "fpo kl_early_stop_mode must be 'post_epoch' or 'pre_optimizer', "
                f"got {self.kl_early_stop_mode!r}."
            )
        if self.kl_early_stop_mode == "pre_optimizer":
            if self.value.enabled:
                raise ValueError("pre_optimizer KL stopping is currently defined only for critic-free FPO.")
            if not self.full_logical_batch_gradient_accumulation:
                raise ValueError(
                    "pre_optimizer KL stopping requires full_logical_batch_gradient_accumulation=true."
                )
            if self.target_kl is None:
                raise ValueError("pre_optimizer KL stopping requires a finite target_kl.")
        for name, seed in (("fpo_mc_seed", self.fpo_mc_seed), ("fpo_shuffle_seed", self.fpo_shuffle_seed)):
            if seed is not None and seed < 0:
                raise ValueError(f"{name} must be non-negative when provided, got {seed}.")
        if self.post_resume_lr_override is not None and self.post_resume_lr_override <= 0:
            raise ValueError(
                "post_resume_lr_override must be positive when provided, "
                f"got {self.post_resume_lr_override}."
            )


@dataclass
class FlowGRPOActorConfig(FPOActorConfig):
    """Flow-GRPO transition-likelihood actor update configuration."""

    _target_: str = "verl_vla.workers.config.FlowGRPOActorConfig"
    train_transition_fraction: float = 0.99
    advantage_clip_max: float = 5.0

    def __post_init__(self):
        super().__post_init__()
        if self.fpo_variant != "vanilla" or self.trust_region_mode != "ppo":
            raise ValueError("Flow-GRPO has its own SDE likelihood ratio and cannot enable FPO++/ASPO.")
        if self.value.enabled or self.vf_coef != 0 or self.value_only_updates != 0:
            raise ValueError("Flow-GRPO requires critic-free group-relative advantages.")
        if self.normalize_advantages:
            raise ValueError("Flow-GRPO advantages are normalized once within each rollout group.")
        if not 0 < self.train_transition_fraction <= 1:
            raise ValueError("Flow-GRPO train_transition_fraction must lie in (0, 1].")
        if self.advantage_clip_max <= 0:
            raise ValueError("Flow-GRPO advantage_clip_max must be positive.")
        # The first full logical-batch pass materializes theta_old before its
        # optimizer step. At least one subsequent pass is required for the
        # official current/frozen-old ratio and clipping to constrain an
        # already-updated actor.
        if self.update_epochs < 2:
            raise ValueError("Flow-GRPO requires update_epochs>=2 for a non-degenerate frozen-old ratio.")
        if not self.full_logical_batch_gradient_accumulation:
            raise ValueError("Fast-WAM Flow-GRPO requires full logical-batch gradient accumulation per update epoch.")


@dataclass
class SFTActorConfig(BaseVLAActorConfig):
    """SFT actor config kept separate from SAC-specific fields."""

    _target_: str = "verl_vla.workers.config.SFTActorConfig"

    acp: ACPConfig = field(default_factory=ACPConfig)

    ema_decay: float | None = None
    mini_batch_size: int = 256
    micro_batch_size: int | None = None

    def __post_init__(self):
        super().__post_init__()

        if self.ema_decay is not None and not 0 < self.ema_decay < 1:
            raise ValueError(f"ema_decay must be in (0, 1) when provided, got {self.ema_decay}")

        if self.mini_batch_size <= 0:
            raise ValueError(f"mini_batch_size must be positive, got {self.mini_batch_size}")

        if self.micro_batch_size is not None and self.micro_batch_size <= 0:
            raise ValueError(f"micro_batch_size must be positive when provided, got {self.micro_batch_size}")
