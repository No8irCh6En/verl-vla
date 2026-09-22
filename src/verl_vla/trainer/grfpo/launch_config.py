"""Single source of truth for fragmented GRFPO rollout-and-train launches."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

_CANONICAL_TASK_NAMES = (
    "stack_bowls",
    "fold_clothes",
    "put_bottles_into_dustbin",
    "match_and_pick_from_conveyor",
    "classify_objects",
)
_SUITE_COUNT = 3


@dataclass(frozen=True)
class GRFPOChainLaunchConfig:
    """Validated experiment protocol plus its exact runtime environment.

    Collection stopping, reward selection, and top-level loss reduction live
    together here so an ablation cannot accidentally change one by omitting a
    shell variable from a copied launcher.
    """

    run_id: str
    algorithm: str = "grfpo"
    start_theta: int = 0
    updates: int = 8
    minimum_accepted_groups: int = 32
    candidate_groups: int = 96
    early_stop: bool = False
    process_score: bool = False
    task_equal: bool = False
    collector_count: int = 12
    collector_concurrency: int = 8
    opportunistic_preempt_collectors: int = 4
    simulators_per_gpu: int = 1
    excluded_task_names: str = "classify_objects"
    policy_seed_namespace: str = ""
    train_gpus_per_node: int = 1
    train_policy_microbatch: int = 1
    train_fsdp_size: int = -1
    actor_lr: float = 1.0e-5
    kl_early_stop_mode: str = "post_epoch"
    target_kl: float = 0.1
    flow_grpo_noise_level: float = 0.01
    flow_grpo_transition_batch_size: int = 3
    continuous: bool = False

    def __post_init__(self) -> None:
        if not self.run_id or "/" in self.run_id or "," in self.run_id:
            raise ValueError("run_id must be a non-empty path component without commas.")
        if self.algorithm not in {"grfpo", "flow_grpo"}:
            raise ValueError("algorithm must be 'grfpo' or 'flow_grpo'.")
        if self.start_theta < 0 or self.updates <= 0:
            raise ValueError("Require start_theta >= 0 and updates > 0.")
        if self.minimum_accepted_groups <= 0:
            raise ValueError("minimum_accepted_groups must be positive.")
        if self.candidate_groups < self.minimum_accepted_groups:
            raise ValueError("candidate_groups must be at least minimum_accepted_groups.")
        if self.collector_count <= 0:
            raise ValueError("collector_count must be positive.")
        excluded_tasks = {
            value.strip() for value in self.excluded_task_names.replace(",", ":").split(":") if value.strip()
        }
        unknown_exclusions = excluded_tasks.difference(_CANONICAL_TASK_NAMES)
        if unknown_exclusions:
            raise ValueError(f"Unknown excluded_task_names: {sorted(unknown_exclusions)}")
        expected_collectors = _SUITE_COUNT * (len(_CANONICAL_TASK_NAMES) - len(excluded_tasks))
        if self.collector_count != expected_collectors:
            raise ValueError(
                "collector_count must provide one warm-start lane per (task,suite): "
                f"got {self.collector_count}, expected {expected_collectors}."
            )
        if not 1 <= self.collector_concurrency <= self.collector_count:
            raise ValueError("collector_concurrency must be in [1, collector_count].")
        if not 0 <= self.opportunistic_preempt_collectors <= self.collector_count:
            raise ValueError("opportunistic_preempt_collectors must be in [0, collector_count].")
        if self.simulators_per_gpu not in {1, 2, 3}:
            raise ValueError(
                "simulators_per_gpu must be 1 (established), 2 (shared-model 2xvec8), or 3 (shared-model 3xvec8)."
            )
        if self.early_stop and self.candidate_groups % self.collector_count:
            raise ValueError(
                "early-stop mode closes only at a complete collector round, so "
                "candidate_groups must be divisible by collector_count."
            )
        if self.train_gpus_per_node not in {1, 2}:
            raise ValueError("train_gpus_per_node must be 1 or 2 for the current spooled trainer.")
        if self.train_policy_microbatch <= 0:
            raise ValueError("train_policy_microbatch must be positive.")
        if self.train_fsdp_size != -1 and (
            self.train_fsdp_size <= 0
            or self.train_fsdp_size > self.train_gpus_per_node
            or self.train_gpus_per_node % self.train_fsdp_size
        ):
            raise ValueError("train_fsdp_size must be -1 or a positive divisor of train_gpus_per_node.")
        if not math.isfinite(self.actor_lr) or self.actor_lr <= 0:
            raise ValueError("actor_lr must be finite and positive.")
        if self.kl_early_stop_mode not in {"post_epoch", "pre_optimizer"}:
            raise ValueError("kl_early_stop_mode must be either 'post_epoch' or 'pre_optimizer'.")
        if not math.isfinite(self.target_kl) or self.target_kl <= 0:
            raise ValueError("target_kl must be finite and positive.")
        if not math.isfinite(self.flow_grpo_noise_level) or self.flow_grpo_noise_level <= 0:
            raise ValueError("flow_grpo_noise_level must be finite and positive.")
        if self.flow_grpo_transition_batch_size <= 0:
            raise ValueError("flow_grpo_transition_batch_size must be positive.")
        for name, value in (
            ("excluded_task_names", self.excluded_task_names),
            ("policy_seed_namespace", self.policy_seed_namespace),
        ):
            if "," in value:
                raise ValueError(f"{name} must not contain a comma because Slurm --export uses commas.")

    @property
    def final_theta(self) -> int:
        return self.start_theta + self.updates

    @property
    def group_reward_source(self) -> str:
        return "process_score" if self.process_score else "binary_success"

    @property
    def accepted_group_reduction(self) -> str:
        return "task_equal" if self.task_equal else "group_equal"

    @property
    def actor_objective(self) -> str:
        return "flow_grpo" if self.algorithm == "flow_grpo" else "fpo"

    @property
    def output_namespace(self) -> str:
        return "flow_grpo" if self.algorithm == "flow_grpo" else "grfpo"

    @property
    def complete_round_after_target(self) -> bool:
        return self.early_stop

    @property
    def drain_candidate_budget(self) -> bool:
        return not self.early_stop

    def runtime_environment(self) -> dict[str, str]:
        """Return every experiment-defining variable consumed by the chain."""

        values: dict[str, object] = {
            "MULTITASK_GRFPO_RUN_ID": self.run_id,
            "VVLA_GROUP_RL_METHOD": self.algorithm,
            "VVLA_GROUP_RL_OUTPUT_NAMESPACE": self.output_namespace,
            "GRFPO_ACTOR_OBJECTIVE": self.actor_objective,
            "GRFPO_FLOW_GRPO_NOISE_LEVEL": self.flow_grpo_noise_level,
            "GRFPO_FLOW_GRPO_TRANSITION_BATCH_SIZE": self.flow_grpo_transition_batch_size,
            "GRFPO_CHAIN_START_THETA": self.start_theta,
            "GRFPO_CHAIN_FINAL_THETA": self.final_theta,
            "GRFPO_CHAIN_CONTINUOUS": int(self.continuous),
            # A single public minimum controls both the trainability gate and
            # the early-stop lower bound. In fixed-budget mode it never
            # truncates the 96-candidate schedule.
            "GRFPO_TARGET_GROUPS": self.minimum_accepted_groups,
            "GRFPO_MIN_GROUPS": self.minimum_accepted_groups,
            "GRFPO_MAX_CANDIDATES": self.candidate_groups,
            "GRFPO_COMPLETE_ROUND_AFTER_TARGET": int(self.complete_round_after_target),
            "GRFPO_DRAIN_CANDIDATE_BUDGET": int(self.drain_candidate_budget),
            "GRFPO_COLLECTOR_COUNT": self.collector_count,
            "GRFPO_COLLECTOR_CONCURRENCY": self.collector_concurrency,
            "GRFPO_SHARED_CANDIDATE_QUEUE": 1,
            "GRFPO_OPPORTUNISTIC_PREEMPT_COLLECTORS": self.opportunistic_preempt_collectors,
            "GRFPO_SIMULATORS_PER_GPU": self.simulators_per_gpu,
            "GRFPO_EXCLUDED_TASK_NAMES": self.excluded_task_names,
            "GRFPO_GROUP_REWARD_SOURCE": self.group_reward_source,
            "GRFPO_INFORMATIVE_GROUP_CRITERION": "full_success_mixed",
            "GRFPO_ACCEPTED_GROUP_REDUCTION": self.accepted_group_reduction,
            "GRFPO_POLICY_SEED_NAMESPACE": self.policy_seed_namespace,
            "GRFPO_TRAIN_GPUS_PER_NODE": self.train_gpus_per_node,
            "GRFPO_TRAIN_POLICY_MICROBATCH": self.train_policy_microbatch,
            "GRFPO_FSDP_SIZE": self.train_fsdp_size,
            "GRFPO_ACTOR_LR": self.actor_lr,
            "GRFPO_KL_EARLY_STOP_MODE": self.kl_early_stop_mode,
            "GRFPO_TARGET_KL": self.target_kl,
        }
        return {key: str(value) for key, value in values.items()}

    def manifest(self, *, selection: dict[str, Any]) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "algorithm": ("flow_grpo" if self.algorithm == "flow_grpo" else "grfpo_plus_plus"),
            "launch_config": asdict(self),
            "derived_protocol": {
                "final_theta": self.final_theta,
                "group_size": 8,
                "target_accepted_groups": self.minimum_accepted_groups,
                "minimum_accepted_groups": self.minimum_accepted_groups,
                "max_candidate_groups": self.candidate_groups,
                "complete_round_after_target": self.complete_round_after_target,
                "drain_candidate_budget": self.drain_candidate_budget,
                "use_all_accepted_groups": True,
                "group_reward_source": self.group_reward_source,
                "informative_group_criterion": "full_success_mixed",
                "accepted_group_reduction": self.accepted_group_reduction,
                "actor_objective": self.actor_objective,
                "flow_grpo_noise_level": (self.flow_grpo_noise_level if self.algorithm == "flow_grpo" else None),
                "actor_lr": self.actor_lr,
                "kl_early_stop_mode": self.kl_early_stop_mode,
                "target_kl": self.target_kl,
            },
            "initial_checkpoint_selection": selection,
            "runtime_environment": self.runtime_environment(),
        }
