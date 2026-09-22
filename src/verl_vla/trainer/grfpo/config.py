# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0

from dataclasses import dataclass, field

from verl.base_config import BaseConfig

__all__ = ["GRFPOTrainerConfig"]


@dataclass
class GRFPOTrainerConfig(BaseConfig):
    """Same-condition group-relative advantages with an FPO actor objective."""

    _target_: str = "verl_vla.trainer.grfpo.config.GRFPOTrainerConfig"

    project_name: str = "vla-grfpo"
    experiment_name: str = "robodojo"
    logger: list[str] = field(default_factory=lambda: ["console"])
    total_training_steps: int = 1000
    gamma: float = 0.995
    gae_lambda: float = 0.99
    step_penalty: float = 0.0
    advantage_estimator: str = "grpo_outcome"
    group_size: int = 8
    accepted_groups_per_update: int = 1
    concurrent_candidate_groups: int = 1
    candidate_group_partition: str = "stage"
    min_accepted_groups_for_partial_update: int = 1
    initial_candidate_groups_per_update: int = 0
    accepted_groups_before_optional_refill: int = 0
    max_candidate_groups_per_update: int = 6
    informative_group_sampling: bool = False
    # Scalar used for per-group GRPO normalization. ``process_score`` keeps
    # strict binary success separate while allowing RoboDojo's rule-based
    # partial-completion score (for stack_bowls: 0 / 0.15 / 1).
    group_reward_source: str = "binary_success"
    # Top-level reduction over accepted groups. ``group_equal`` implements
    # mean_g L_g. ``task_equal`` implements mean_t(mean_{g in t} L_g), so a
    # task that happens to yield more informative groups does not receive more
    # optimizer weight merely because mixed-group acceptance was higher.
    accepted_group_reduction: str = "group_equal"
    # ``full_success_mixed`` is the explicit name for SimpleVLA-style
    # 0<K_success<G filtering. ``binary_success`` is retained as a backwards-
    # compatible alias. ``reward_variance`` additionally permits partial-only
    # groups whose selected training rewards are not all equal.
    informative_group_criterion: str = "binary_success"
    # Only applies to score-varying groups with zero full successes. It permits
    # a direct one-versus-two partial-trajectory ablation without conflating a
    # partial completion with task success.
    min_partial_trajectories_for_score_only_group: int = 1
    partial_score_threshold: float = 0.15
    async_rollout: bool = False
    save_freq: int = 50
    test_freq: int = 25
    eval_episodes: int = 50
    val_before_train: bool = True
    val_only: bool = False
    collection_only: bool = False
    post_update_fixed_observation_diagnostic: bool = False
    evidence_jsonl: str | None = None
    candidate_evidence_jsonl: str | None = None
    # Optional durable spool for accepted rollout tensors.  When both paths
    # are configured, a preempted collection window can resume without
    # discarding already accepted on-policy groups.
    candidate_spool_dir: str | None = None
    candidate_resume_state_path: str | None = None
    esi_redundant_time: int = 0

    def __post_init__(self):
        if self.total_training_steps <= 0:
            raise ValueError(f"total_training_steps must be positive, got {self.total_training_steps}")
        if not 0 < self.gamma <= 1:
            raise ValueError(f"gamma must be in (0, 1], got {self.gamma}")
        if not 0 <= self.gae_lambda <= 1:
            raise ValueError(f"gae_lambda must be in [0, 1], got {self.gae_lambda}")
        if self.advantage_estimator != "grpo_outcome":
            raise ValueError("GRFPO always uses per-group outcome advantages; GAE belongs to trainer/fpo.")
        if self.group_reward_source not in {"binary_success", "process_score"}:
            raise ValueError(
                f"group_reward_source must be 'binary_success' or 'process_score', got {self.group_reward_source!r}."
            )
        if self.accepted_group_reduction not in {"group_equal", "task_equal"}:
            raise ValueError(
                "accepted_group_reduction must be 'group_equal' or 'task_equal', "
                f"got {self.accepted_group_reduction!r}."
            )
        if self.informative_group_criterion not in {
            "binary_success",
            "full_success_mixed",
            "reward_variance",
        }:
            raise ValueError(
                "informative_group_criterion must be 'binary_success', "
                "'full_success_mixed', or 'reward_variance', "
                f"got {self.informative_group_criterion!r}."
            )
        if self.min_partial_trajectories_for_score_only_group <= 0:
            raise ValueError("min_partial_trajectories_for_score_only_group must be positive.")
        if not 0 < self.partial_score_threshold < 1:
            raise ValueError("partial_score_threshold must lie strictly between zero and one.")
        if self.informative_group_criterion == "reward_variance" and self.group_reward_source != "process_score":
            raise ValueError(
                "informative_group_criterion='reward_variance' requires group_reward_source='process_score'."
            )
        if self.group_size <= 1:
            raise ValueError(f"group_size must be greater than one, got {self.group_size}")
        if self.accepted_groups_per_update <= 0:
            raise ValueError(f"accepted_groups_per_update must be positive, got {self.accepted_groups_per_update}")
        if self.concurrent_candidate_groups <= 0:
            raise ValueError(f"concurrent_candidate_groups must be positive, got {self.concurrent_candidate_groups}")
        if self.concurrent_candidate_groups > self.accepted_groups_per_update:
            raise ValueError(
                "concurrent_candidate_groups cannot exceed accepted_groups_per_update, "
                f"got {self.concurrent_candidate_groups} > {self.accepted_groups_per_update}."
            )
        if self.candidate_group_partition not in {"stage", "worker_stage"}:
            raise ValueError(
                f"candidate_group_partition must be 'stage' or 'worker_stage', got {self.candidate_group_partition!r}."
            )
        if not 1 <= self.min_accepted_groups_for_partial_update <= self.accepted_groups_per_update:
            raise ValueError(
                "min_accepted_groups_for_partial_update must be in "
                f"[1, accepted_groups_per_update], got {self.min_accepted_groups_for_partial_update}"
            )
        refill_values = (
            self.initial_candidate_groups_per_update,
            self.accepted_groups_before_optional_refill,
        )
        if any(value < 0 for value in refill_values):
            raise ValueError("Optional-refill settings must be non-negative.")
        if bool(self.initial_candidate_groups_per_update) != bool(self.accepted_groups_before_optional_refill):
            raise ValueError(
                "initial_candidate_groups_per_update and "
                "accepted_groups_before_optional_refill must both be zero or both be positive."
            )
        if self.accepted_groups_before_optional_refill > self.accepted_groups_per_update:
            raise ValueError("accepted_groups_before_optional_refill cannot exceed accepted_groups_per_update.")
        if self.max_candidate_groups_per_update <= 0:
            raise ValueError(
                f"max_candidate_groups_per_update must be positive, got {self.max_candidate_groups_per_update}"
            )
        if self.group_size != 8:
            raise ValueError(f"Group-Relative FPO requires group_size=8, got {self.group_size}")
        if self.async_rollout:
            raise ValueError("Group-Relative FPO requires async_rollout=false.")
        if self.max_candidate_groups_per_update < self.accepted_groups_per_update:
            raise ValueError("max_candidate_groups_per_update must be at least accepted_groups_per_update.")
        if self.initial_candidate_groups_per_update > self.max_candidate_groups_per_update:
            raise ValueError("initial_candidate_groups_per_update cannot exceed max_candidate_groups_per_update.")
        if self.min_partial_trajectories_for_score_only_group > self.group_size - 1:
            raise ValueError("min_partial_trajectories_for_score_only_group must be smaller than group_size.")
        if self.eval_episodes == 0:
            raise ValueError("eval_episodes must be positive or negative to use the benchmark default")
        if bool(self.candidate_spool_dir) != bool(self.candidate_resume_state_path):
            raise ValueError(
                "candidate_spool_dir and candidate_resume_state_path must be configured together."
            )
        if self.candidate_spool_dir and self.save_freq != 1:
            raise ValueError("Preempt-safe candidate spooling requires save_freq=1.")
