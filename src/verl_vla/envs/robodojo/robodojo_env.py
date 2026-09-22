# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""RoboDojo's native vector simulator exposed through the BaseEnv contract."""

from __future__ import annotations

import importlib
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from omegaconf import OmegaConf
from typing_extensions import override

from verl_vla.envs.base import BaseEnv

from .config import RoboDojoSimulatorConfig

logger = logging.getLogger(__name__)


def _standardize_rgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB image, got {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.shape[:2] != (240, 320):
        image = cv2.resize(image, (320, 240), interpolation=cv2.INTER_AREA)
    if image.shape != (240, 320, 3):
        raise ValueError(f"Expected canonical RGB [240,320,3], got {image.shape}")
    return image


class RoboDojoEnv(BaseEnv):
    """One native RoboDojo/Isaac instance hosting `num_envs` vector slots."""

    env_type = "robodojo"

    def __init__(
        self,
        cfg,
        rank: int,
        world_size: int,
        stage_id: int = 0,
        stage_num: int = 1,
        only_eval: bool = False,
    ) -> None:
        del only_eval
        self._schedule_stage_num = int(stage_num)
        self.robodojo_cfg = OmegaConf.to_object(cfg.simulator.robodojo)
        if not isinstance(self.robodojo_cfg, RoboDojoSimulatorConfig):
            self.robodojo_cfg = RoboDojoSimulatorConfig(**dict(self.robodojo_cfg))
        self.robodojo_root = Path(self.robodojo_cfg.robodojo_root).expanduser().resolve()
        if not self.robodojo_root.is_dir():
            raise FileNotFoundError(f"RoboDojo root not found: {self.robodojo_root}")
        root_text = str(self.robodojo_root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)

        # IsaacLab requires AppLauncher before importing app-dependent RoboDojo modules.
        from isaaclab.app import AppLauncher

        # Slurm cgroups expose the allocated CUDA GPU to PyTorch as logical
        # cuda:0, while Vulkan/Kit's activeGpu index is physical-node scoped.
        # Keep RoboDojo tensors on logical ``device_id`` but tell AppLauncher
        # which physical renderer GPU Slurm assigned to this single-GPU job.
        physical_gpu_text = os.environ.get("VERL_VLA_PHYSICAL_GPU_ID", "").split(",", 1)[0]
        physical_gpu_id = (
            int(physical_gpu_text)
            if physical_gpu_text.isdigit()
            else int(self.robodojo_cfg.device_id)
        )
        self.app = AppLauncher(
            headless=bool(self.robodojo_cfg.headless),
            enable_cameras=True,
            device=f"cuda:{physical_gpu_id}",
        ).app
        self.backend = None
        self._train_case_cursor = int(self.robodojo_cfg.train_case_cursor_start)
        self._eval_case_cursor = 0
        self._episode_ids = np.full(int(cfg.num_envs), -1, dtype=np.int64)
        self._layout_ids = np.full(int(cfg.num_envs), -1, dtype=np.int64)
        self._policy_seeds = np.full(int(cfg.num_envs), int(self.robodojo_cfg.seed), dtype=np.int64)
        self._runtime_eval_layout_ids: list[int] | None = None
        self._runtime_eval_policy_seeds: list[int] | None = None
        super().__init__(cfg, rank, world_size, stage_id=stage_id)
        if self.robodojo_cfg.defer_chunk_observations:
            if str(cfg.action_execution.mode) != "serial":
                raise ValueError("defer_chunk_observations requires serial action execution")
            if self.auto_reset_enabled:
                raise ValueError("defer_chunk_observations requires auto_reset=false")
            if self.teleops:
                raise ValueError("defer_chunk_observations is incompatible with teleop")
            if self.recorder is not None:
                raise ValueError("defer_chunk_observations is incompatible with trajectory recording")
        self._deferred_observation_steps = 0
        self._chunk_observation_captures = 0

    @override
    def env_init(self) -> None:
        from env.global_configs import BENCHMARK, ENV_CONFIG_PATH, ROOT_DIR
        from utils.load_file import load_yaml
        from utils.pipeline_utils import process_config, process_randomization

        task_registry = importlib.import_module(f"task.{BENCHMARK}.task_registry")
        eval_cfg = load_yaml(os.path.join(ENV_CONFIG_PATH, f"{self.robodojo_cfg.env_cfg_type}.yml"))
        eval_cfg.update(
            {
                "task_name": self.robodojo_cfg.task_name,
                "num_envs": self.num_envs,
                "device_id": self.robodojo_cfg.device_id,
                "eval_batch": True,
                "policy_name": "verl_vla_fastwam",
                "additional_info": "phase6a",
                "seed": self.robodojo_cfg.seed,
                "physx_monitor_enabled": False,
                "eval_num": self.robodojo_cfg.eval_num,
            }
        )
        benchmark_path = os.path.join(ROOT_DIR, "task", BENCHMARK)
        env_cfg = OmegaConf.create(
            {
                "sim": load_yaml(os.path.join(ENV_CONFIG_PATH, "sim", eval_cfg["config"]["sim"] + ".yml")),
                "scene": load_yaml(os.path.join(ENV_CONFIG_PATH, "scene", eval_cfg["config"]["scene"] + ".yml")),
                "camera": load_yaml(os.path.join(ENV_CONFIG_PATH, "camera", eval_cfg["config"]["camera"] + ".yml")),
                "robot": load_yaml(os.path.join(ENV_CONFIG_PATH, "robot", eval_cfg["config"]["robot"] + ".yml")),
                "task_env": load_yaml(
                    task_registry.task_config_path(os.path.join(benchmark_path, "config"), self.robodojo_cfg.task_name)
                ),
                "eval_cfg": eval_cfg,
                "deploy_cfg": {
                    "policy_name": "verl_vla_fastwam",
                    "external_policy_control": True,
                },
            }
        )
        OmegaConf.update(env_cfg, "sim.scene.num_envs", self.num_envs, force_add=True)
        OmegaConf.update(env_cfg, "eval_cfg.num_envs", self.num_envs, force_add=True)
        env_cfg = process_randomization(env_cfg)
        env_cfg, _ = process_config(env_cfg, task_name=self.robodojo_cfg.task_name)
        OmegaConf.update(
            env_cfg,
            "camera.default_frequency",
            eval_cfg["observation"].get("collect_freq", 0),
            force_add=True,
        )
        env_cfg.sim.seed = [0 for _ in range(self.num_envs)]

        from src.eval_client.eval_env import create_eval_env
        from XPolicyLab.utils.process_data import get_robot_action_dim_info

        self.robot_action_dim_info = get_robot_action_dim_info(self.robodojo_cfg.env_cfg_type)
        self.backend = create_eval_env(env_cfg, self.app)

    def _next_cases(self, mode: str) -> tuple[list[int], list[int], list[int]]:
        if mode == "eval":
            runtime_layouts = getattr(self, "_runtime_eval_layout_ids", None)
            runtime_seeds = getattr(self, "_runtime_eval_policy_seeds", None)
            layouts = list(runtime_layouts if runtime_layouts is not None else self.robodojo_cfg.layout_ids)
            seeds = list(
                runtime_seeds
                if runtime_seeds is not None
                else self.robodojo_cfg.eval_policy_seed_schedule
            ) or [self.robodojo_cfg.seed] * len(layouts)
            cursor_name = "_eval_case_cursor"
        else:
            layouts = list(self.robodojo_cfg.train_layout_schedule) or list(self.robodojo_cfg.layout_ids)
            seeds = list(self.robodojo_cfg.train_policy_seed_schedule) or [self.robodojo_cfg.seed] * len(layouts)
            cursor_name = "_train_case_cursor"
        cursor = int(getattr(self, cursor_name))
        if mode == "eval":
            # Each Ray worker owns an independent cursor, so shard a reset wave
            # by worker/stage rank. Without this offset every worker evaluates
            # the same fixed cases and TrainCluster must discard the duplicates.
            world_size = int(getattr(self, "world_size", 1))
            rank = int(getattr(self, "rank", 0))
            stage_id = int(getattr(self, "stage_id", 0))
            stage_num = int(getattr(self, "_schedule_stage_num", 1))
            reset_wave = cursor // self.num_envs
            parallel_width = world_size * stage_num * self.num_envs
            # EnvLoop concatenates one logical rollout batch stage-major:
            # all worker ranks for stage 0, then all ranks for stage 1. Keep
            # the deterministic case stream in that same order so each stage
            # forms an independent same-condition group.
            local_offset = (stage_id * world_size + rank) * self.num_envs
            absolute_indices = [
                reset_wave * parallel_width + local_offset + offset
                for offset in range(self.num_envs)
            ]
            indices = [index % len(layouts) for index in absolute_indices]
            setattr(self, cursor_name, cursor + self.num_envs)
            return (
                [int(layouts[index]) for index in indices],
                [int(seeds[index]) for index in indices],
                [int(index) for index in indices],
            )
        if mode == "train" and (
            int(self.robodojo_cfg.train_layout_repeat) != 1
            or self.robodojo_cfg.train_policy_seed_start is not None
        ):
            repeat = int(self.robodojo_cfg.train_layout_repeat)
            expanded_size = len(layouts) * repeat
            # Every Ray EnvWorker/stage owns an independent simulator and
            # cursor. Map each local reset wave into one stage-major global
            # deterministic case stream. Consequently each active pipeline
            # stage is one complete G=8 condition with eight distinct policy
            # seeds, matching EnvLoop's stage-major collation order.
            world_size = int(getattr(self, "world_size", 1))
            rank = int(getattr(self, "rank", 0))
            stage_id = int(getattr(self, "stage_id", 0))
            stage_num = int(getattr(self, "_schedule_stage_num", 1))
            reset_wave = cursor // self.num_envs
            parallel_width = world_size * stage_num * self.num_envs
            local_offset = (stage_id * world_size + rank) * self.num_envs
            absolute_indices = [
                reset_wave * parallel_width + local_offset + offset
                for offset in range(self.num_envs)
            ]
            expanded_indices = [index % expanded_size for index in absolute_indices]
            selected_layouts = [int(layouts[index // repeat]) for index in expanded_indices]
            seed_start = (
                int(self.robodojo_cfg.train_policy_seed_start)
                if self.robodojo_cfg.train_policy_seed_start is not None
                else int(self.robodojo_cfg.seed)
            )
            setattr(self, cursor_name, cursor + self.num_envs)
            return (
                selected_layouts,
                [seed_start + index for index in absolute_indices],
                [int(index) for index in expanded_indices],
            )
        indices = [(cursor + offset) % len(layouts) for offset in range(self.num_envs)]
        setattr(self, cursor_name, (cursor + self.num_envs) % len(layouts))
        return (
            [int(layouts[index]) for index in indices],
            [int(seeds[index]) for index in indices],
            [int(index) for index in indices],
        )

    def set_runtime_eval_cases(self, *, layout_ids: list[int], policy_seeds: list[int]) -> None:
        """Replace one eval reset wave without mutating the frozen config."""

        if len(layout_ids) != self.num_envs or len(policy_seeds) != self.num_envs:
            raise ValueError(
                "Runtime eval cases must provide one layout and policy seed per vector env: "
                f"layouts={len(layout_ids)}, seeds={len(policy_seeds)}, num_envs={self.num_envs}."
            )
        self._runtime_eval_layout_ids = [int(value) for value in layout_ids]
        self._runtime_eval_policy_seeds = [int(value) for value in policy_seeds]
        self._eval_case_cursor = 0

    def get_state(self) -> dict[str, int]:
        """Return the schedule state needed across a fresh Isaac subprocess."""

        return {
            "train_case_cursor": int(self._train_case_cursor),
            "eval_case_cursor": int(self._eval_case_cursor),
        }

    def load_state(self, state: dict[str, int]) -> None:
        """Restore deterministic case cursors after simulator recreation."""

        self._train_case_cursor = int(state["train_case_cursor"])
        self._eval_case_cursor = int(state["eval_case_cursor"])

    def _canonical_observation(self, raw: dict[str, Any]) -> dict[str, np.ndarray]:
        from XPolicyLab.utils.process_data import pack_robot_state

        vision = raw["vision"]
        return {
            "observation.images.cam_head": _standardize_rgb(vision["cam_head"]["color"]),
            "observation.images.cam_left_wrist": _standardize_rgb(vision["cam_left_wrist"]["color"]),
            "observation.images.cam_right_wrist": _standardize_rgb(vision["cam_right_wrist"]["color"]),
            "observation.state": pack_robot_state(
                raw,
                self.robodojo_cfg.action_type,
                self.robot_action_dim_info,
                source_type="obs",
                state_type="state",
            ).astype(np.float32),
        }

    def _instruction(self, raw: dict[str, Any]) -> str:
        value = raw.get("task_instruction", raw.get("instruction", raw.get("instructions")))
        if isinstance(value, (list, tuple)):
            value = value[0] if value else ""
        value = str(value or "").strip()
        if not value:
            raise RuntimeError(
                "RoboDojo observation is missing its task instruction; refusing to use a "
                f"stack_bowls fallback for task={self.robodojo_cfg.task_name!r}."
            )
        return value

    def _reset_backend_stably(
        self,
        layouts: list[int],
        progress_callback: Callable[[], None] | None = None,
    ) -> None:
        """Reset the full vector to the same cases until every scene is stable."""

        last_error: Exception | None = None
        max_attempts = int(self.robodojo_cfg.reset_max_attempts)
        for attempt in range(1, max_attempts + 1):
            # A native reset can take several minutes.  Fragmented online
            # collection passes a lease-renewal callback here so a healthy
            # long reset is not mistaken for a preempted collector and stolen
            # by another shared-queue worker.  Other callers leave it unset.
            if progress_callback is not None:
                progress_callback()
            try:
                self.backend.reset(seed=layouts)
            except Exception as exc:
                is_unstable = exc.__class__.__name__ == "UnStableError" or "All scene Unstable" in str(exc)
                if not is_unstable:
                    raise
                last_error = exc
                unstable_envs = sorted(int(item) for item in getattr(self.backend, "unstable_envs", ()))
            else:
                unstable_envs = sorted(int(item) for item in getattr(self.backend, "unstable_envs", ()))
                active_episode_count = int(getattr(self.backend, "episode_nums", self.num_envs))
                if not unstable_envs and active_episode_count == self.num_envs:
                    if progress_callback is not None:
                        progress_callback()
                    return
                last_error = RuntimeError(
                    "RoboDojo returned a partial unstable vector: "
                    f"unstable_envs={unstable_envs}, active_episode_count={active_episode_count}, "
                    f"expected={self.num_envs}"
                )
            if progress_callback is not None:
                progress_callback()
            logger.warning(
                "RoboDojo stable-reset retry attempt=%d/%d layouts=%s unstable_envs=%s error=%s",
                attempt,
                max_attempts,
                layouts,
                unstable_envs,
                last_error,
            )
        raise RuntimeError(
            f"RoboDojo could not stably reset all vector slots after {max_attempts} attempts "
            f"for fixed layouts {layouts}."
        ) from last_error

    def _observations(self, env_ids: np.ndarray) -> dict[str, Any]:
        raw_rows = self.backend.get_obs_batch(env_idx_list=env_ids.tolist())
        return {
            "observation": [self._canonical_observation(raw) for raw in raw_rows],
            "task": [self._instruction(raw) for raw in raw_rows],
            "task_id": np.full(len(env_ids), int(self.robodojo_cfg.task_id), dtype=np.int64),
            "suite_id": np.full(len(env_ids), int(self.robodojo_cfg.seed), dtype=np.int64),
            "eval_episode_id": self._episode_ids[env_ids].copy(),
            "layout_id": self._layout_ids[env_ids].copy(),
            # RoboDojo's reset seed is the immutable saved-layout/case id.
            "environment_seed": self._layout_ids[env_ids].copy(),
            "policy_seed": self._policy_seeds[env_ids].copy(),
        }

    @override
    def env_reset(self, *, env_ids, reset_eval: bool = False, extra: dict[str, Any] | None = None):
        extra = extra or {}
        mode = str(extra.get("mode", "eval" if reset_eval else "train"))
        progress_callback = extra.get("progress_callback")
        if progress_callback is not None and not callable(progress_callback):
            raise TypeError("RoboDojo reset progress_callback must be callable when provided.")
        if mode not in {"train", "eval"}:
            raise ValueError(f"Unsupported RoboDojo reset mode: {mode}")
        env_ids = np.asarray(env_ids, dtype=np.int64).reshape(-1)
        all_ids = np.arange(self.num_envs, dtype=np.int64)
        if not np.array_equal(env_ids, all_ids):
            raise NotImplementedError(
                "Native RoboDojo reset is global; partial reset is rejected to avoid resetting unrelated live envs. "
                "Use auto_reset=false and full-vector episode resets."
            )
        if reset_eval:
            self._eval_case_cursor = 0
        layouts, policy_seeds, case_ids = self._next_cases(mode)
        # RoboDojo can transiently reject a saved scene as physically unstable.
        # Retry the exact same vector cases as one atomic batch; never substitute
        # another layout/seed or allow a partially stable vector to reach step().
        self._reset_backend_stably(layouts, progress_callback=progress_callback)
        self.backend.run_reward()
        # RoboDojo registers process-score predicates separately from its
        # binary task reward. Match the official evaluator lifecycle.
        if hasattr(self.backend, "get_score"):
            self.backend.get_score()
        # Fixed-case evaluation ids are benchmark indices, not globally
        # increasing episode counters.  Keep them aligned with the rotating
        # layout queue and restart from zero when TrainCluster begins a fresh
        # evaluation pass.
        self._episode_ids[:] = case_ids
        self._layout_ids[:] = layouts
        self._policy_seeds[:] = policy_seeds
        return self._observations(all_ids)

    @override
    def env_step(self, action, *, env_ids):
        from XPolicyLab.utils.process_data import unpack_robot_state

        env_ids = np.asarray(env_ids, dtype=np.int64).reshape(-1)
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (len(env_ids), 14):
            raise ValueError(f"RoboDojo action must have shape [{len(env_ids)},14], got {action.shape}")
        action_rows = unpack_robot_state(
            action,
            self.robodojo_cfg.action_type,
            self.robot_action_dim_info,
            source_type="obs",
        )
        self.backend.take_action_batch(action_rows, env_idx_list=env_ids.tolist())
        if self.robodojo_cfg.defer_chunk_observations:
            # The serial executor exposes one observation after the action
            # chunk. Reuse the last one here while physics/reward/end checks
            # still execute, then take one fresh read in the finalizer below.
            if self._latest_obs is None:
                raise RuntimeError("deferred RoboDojo stepping requires an observation from reset()")
            obs = self._slice_latest_obs(env_ids)
            self._deferred_observation_steps += len(env_ids)
        else:
            obs = self._observations(env_ids)
        ended = np.asarray([self.backend.end_flag[int(env_id)] for env_id in env_ids], dtype=bool)
        success = ended & np.asarray([self.backend.success[int(env_id)] for env_id in env_ids], dtype=bool)
        reward_manager = getattr(self.backend, "reward_manager", None)
        process_score = None
        if reward_manager is not None and hasattr(reward_manager, "get_score"):
            # Match RoboDojo's official evaluator: a failed trajectory must not
            # receive the terminal 100-point tier merely because it transiently
            # satisfied the process predicate.  Earlier partial tiers remain
            # available, while environment success is authoritative below.
            gated_scores = reward_manager.get_score(
                reward_lst=np.asarray(self.backend.success, dtype=np.float32)
            )
            process_score = np.asarray(gated_scores, dtype=np.float32)[env_ids] / 100.0
            process_score[success] = 1.0
        obs.update(
            {
                "next.reward": success.astype(np.float32),
                "next.terminated": success,
                "next.truncated": ended & ~success,
                "next.success": success,
                # Recorder-only metadata. These fields are not part of the
                # policy observation; they make videos self-identifying and
                # preserve RoboDojo's dense process score in visual audits.
                "extra": [
                    {
                        "info.eval_episode_id": int(self._episode_ids[int(env_id)]),
                        "info.layout_id": int(self._layout_ids[int(env_id)]),
                        "info.environment_seed": int(self._layout_ids[int(env_id)]),
                        "info.policy_seed": int(self._policy_seeds[int(env_id)]),
                        "next.score": float(process_score[local_id]) if process_score is not None else float(success[local_id]),
                    }
                    for local_id, env_id in enumerate(env_ids)
                ],
            }
        )
        if process_score is not None:
            obs["next.score"] = process_score
        return obs

    @override
    def finalize_execution_observation(self, merged_step_result, *, env_ids):
        if not self.robodojo_cfg.defer_chunk_observations:
            return merged_step_result
        if merged_step_result is None:
            raise RuntimeError("cannot finalize a deferred RoboDojo observation before any control step")

        env_ids = np.asarray(env_ids, dtype=np.int64).reshape(-1)
        fresh = self._observations(env_ids)
        for local_id, env_id in enumerate(env_ids):
            env_id = int(env_id)
            merged_step_result["observation"][env_id] = fresh["observation"][local_id]
            merged_step_result["task"][env_id] = fresh["task"][local_id]
            merged_step_result["task_id"][env_id] = fresh["task_id"][local_id]
            for field in ("suite_id", "eval_episode_id", "layout_id", "environment_seed", "policy_seed"):
                if field in fresh and field in merged_step_result:
                    merged_step_result[field][env_id] = fresh[field][local_id]
        self._chunk_observation_captures += 1
        return merged_step_result

    @override
    def env_close(self) -> None:
        if self.backend is not None:
            self.backend.close()
            self.backend = None
        if self.app is not None:
            self.app.close()
            self.app = None

    @override
    def env_benchmark_size(self) -> int:
        return len(self.robodojo_cfg.layout_ids)

    @override
    def get_recorder_strategy_kwargs(self) -> dict[str, Any]:
        return {
            "camera_names": ("cam_head", "cam_left_wrist", "cam_right_wrist"),
            "image_shape": (240, 320, 3),
            "state_dim": 14,
            "action_dim": 14,
            "fps": int(self.cfg.recorder.video.fps),
            "robot_type": self.robodojo_cfg.env_cfg_type,
        }
