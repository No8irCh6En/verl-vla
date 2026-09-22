# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

from dataclasses import dataclass, field

from verl.base_config import BaseConfig


@dataclass
class RoboDojoSimulatorConfig(BaseConfig):
    """Native RoboDojo simulator configuration for a verl-vla EnvWorker."""

    simulator_type: str = "robodojo"
    robodojo_root: str = ""
    task_name: str = "stack_bowls"
    # Stable multi-task manifest. Its order defines task_id. An empty manifest
    # keeps the original single-task task_name/task_id behavior.
    task_names: list[str] = field(default_factory=list)
    task_id: int = 0
    # Explicit stage-major assignment for the persistent RoboDojo simulator
    # processes: stage 0 / all worker ranks, then stage 1 / all ranks, etc.
    # The length must equal env-worker world_size * pipeline_stage_num.
    # Tasks are bound once at simulator construction; resets never rebuild a
    # scene merely to sample another task.
    worker_stage_task_schedule: list[str] = field(default_factory=list)
    env_cfg_type: str = "arx_x5"
    action_type: str = "joint"
    device_id: int = 0
    headless: bool = True
    layout_ids: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
    train_layout_schedule: list[int] = field(default_factory=list)
    train_policy_seed_schedule: list[int] = field(default_factory=list)
    train_layout_repeat: int = 1
    train_policy_seed_start: int | None = None
    # Local per-process cursor restored by a preempt-safe launcher.  Every
    # worker/stage receives the same cursor because _next_cases applies its
    # own stage-major rank offset.
    train_case_cursor_start: int = 0
    eval_policy_seed_schedule: list[int] = field(default_factory=list)
    # Evaluation may deliberately target locked/unseen layouts. This flag
    # never relaxes the training-schedule guard below.
    allow_held_out_eval_layouts: bool = False
    eval_num: int = 1
    seed: int = 0
    reset_max_attempts: int = 3
    reset_rpc_timeout_s: float = 240.0
    reset_process_max_restarts: int = 2
    restart_between_rollouts: bool = False
    restart_every_rollouts: int = 1
    parallel_restart_stages: bool = False
    # Rollout-only fast path: execute every low-level control and reward check,
    # but materialize cameras/proprio once at the serial policy-chunk boundary.
    defer_chunk_observations: bool = False

    def __post_init__(self) -> None:
        if not self.robodojo_root:
            raise ValueError("robodojo_root is required.")
        for field_name, task_names in (
            ("task_name", [self.task_name]),
            ("task_names", self.task_names),
            ("worker_stage_task_schedule", self.worker_stage_task_schedule),
        ):
            invalid = [
                task
                for task in task_names
                if not task or any(char in task for char in ("/", "\\", "."))
            ]
            if invalid:
                raise ValueError(
                    f"{field_name} must contain non-empty RoboDojo task identifiers: {invalid!r}"
                )
        if len(set(self.task_names)) != len(self.task_names):
            raise ValueError(f"task_names must define unique stable task ids: {self.task_names!r}")
        if self.task_names and self.task_name not in self.task_names:
            raise ValueError(
                f"single-task fallback task_name={self.task_name!r} must occur in task_names={self.task_names!r}"
            )
        unknown_scheduled_tasks = sorted(set(self.worker_stage_task_schedule) - set(self.task_names))
        if unknown_scheduled_tasks:
            raise ValueError(
                "worker_stage_task_schedule contains tasks absent from task_names: "
                f"{unknown_scheduled_tasks!r}"
            )
        if self.task_id < 0:
            raise ValueError(f"task_id must be non-negative, got {self.task_id}")
        if self.action_type != "joint":
            raise ValueError("Phase 6A Fast-WAM integration requires joint actions.")
        if not self.layout_ids:
            raise ValueError("layout_ids must not be empty.")
        held_out_train = sorted(set(int(value) for value in self.train_layout_schedule) & {6, 7, 8, 9})
        if held_out_train:
            raise ValueError(f"Training must not use held-out layouts 6-9: {held_out_train}")
        held_out_eval = sorted(set(int(value) for value in self.layout_ids) & {6, 7, 8, 9})
        if held_out_eval and not self.allow_held_out_eval_layouts:
            raise ValueError(
                "Evaluation layouts include held-out layouts 6-9 but "
                f"allow_held_out_eval_layouts=false: {held_out_eval}"
            )
        if self.train_policy_seed_schedule and len(self.train_policy_seed_schedule) != len(
            self.train_layout_schedule
        ):
            raise ValueError("train_policy_seed_schedule must align one-to-one with train_layout_schedule.")
        if self.train_layout_repeat <= 0:
            raise ValueError("train_layout_repeat must be positive.")
        if self.train_layout_repeat != 1 and self.train_policy_seed_schedule:
            raise ValueError("Repeated train layouts require generated policy seeds, not an explicit seed schedule.")
        if self.train_policy_seed_start is not None and self.train_policy_seed_schedule:
            raise ValueError("train_policy_seed_start and train_policy_seed_schedule are mutually exclusive.")
        if self.train_case_cursor_start < 0:
            raise ValueError("train_case_cursor_start must be non-negative.")
        if self.eval_policy_seed_schedule and len(self.eval_policy_seed_schedule) != len(self.layout_ids):
            raise ValueError("eval_policy_seed_schedule must align one-to-one with layout_ids.")
        if self.reset_max_attempts < 1:
            raise ValueError("reset_max_attempts must be at least 1.")
        if self.reset_rpc_timeout_s <= 0:
            raise ValueError("reset_rpc_timeout_s must be positive.")
        if self.reset_process_max_restarts < 0:
            raise ValueError("reset_process_max_restarts must be non-negative.")
        if self.restart_every_rollouts < 1:
            raise ValueError("restart_every_rollouts must be at least 1.")

    def task_assignment(
        self,
        *,
        worker_rank: int,
        worker_world_size: int,
        stage_id: int,
        stage_num: int,
    ) -> tuple[str, int]:
        """Resolve the persistent task owned by one worker-stage simulator."""

        if not 0 <= worker_rank < worker_world_size:
            raise ValueError(
                f"worker_rank must be in [0, {worker_world_size}), got {worker_rank}."
            )
        if not 0 <= stage_id < stage_num:
            raise ValueError(f"stage_id must be in [0, {stage_num}), got {stage_id}.")
        if not self.worker_stage_task_schedule:
            return self.task_name, int(self.task_id)

        expected_slots = worker_world_size * stage_num
        if len(self.worker_stage_task_schedule) != expected_slots:
            raise ValueError(
                "worker_stage_task_schedule must provide exactly one persistent task per "
                f"worker-stage simulator: expected={expected_slots}, "
                f"got={len(self.worker_stage_task_schedule)}."
            )
        slot = stage_id * worker_world_size + worker_rank
        task_name = self.worker_stage_task_schedule[slot]
        return task_name, self.task_names.index(task_name)
