# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""RoboDojo-to-LeRobot frame conversion helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
from typing_extensions import override

from verl_vla.recorder.strategies.base import BaseLeRobotStrategy


class RoboDojoLeRobotStrategy(BaseLeRobotStrategy):
    """Recording schema for the three-camera ARX-X5 RoboDojo tasks."""

    def __init__(
        self,
        *,
        camera_names: tuple[str, ...] = ("cam_head", "cam_left_wrist", "cam_right_wrist"),
        image_shape: tuple[int, int, int] = (240, 320, 3),
        state_dim: int = 14,
        action_dim: int = 14,
        fps: int = 30,
        robot_type: str = "arx_x5",
    ) -> None:
        self.camera_names = tuple(camera_names)
        self.image_shape = tuple(image_shape)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self._fps = int(fps)
        self._robot_type = str(robot_type)

    @property
    @override
    def fps(self) -> int:
        return self._fps

    @property
    @override
    def robot_type(self) -> str:
        return self._robot_type

    @override
    def features(self) -> dict[str, dict[str, Any]]:
        features: dict[str, dict[str, Any]] = {
            f"observation.images.{name}": {
                "dtype": "video",
                "shape": self.image_shape,
                "names": ["height", "width", "channel"],
            }
            for name in self.camera_names
        }
        features.update(
            {
                "observation.state": {
                    "dtype": "float32",
                    "shape": (self.state_dim,),
                    "names": ["state"],
                },
                "action": {
                    "dtype": "float32",
                    "shape": (self.action_dim,),
                    "names": ["action"],
                },
                "next.reward": {"dtype": "float32", "shape": (1,), "names": None},
                "next.score": {"dtype": "float32", "shape": (1,), "names": None},
                "next.terminated": {"dtype": "bool", "shape": (1,), "names": None},
                "next.truncated": {"dtype": "bool", "shape": (1,), "names": None},
                "next.success": {"dtype": "bool", "shape": (1,), "names": None},
                "info.is_intervention": {"dtype": "bool", "shape": (1,), "names": None},
                "info.eval_episode_id": {"dtype": "int64", "shape": (1,), "names": None},
                "info.layout_id": {"dtype": "int64", "shape": (1,), "names": None},
                "info.environment_seed": {"dtype": "int64", "shape": (1,), "names": None},
                "info.policy_seed": {"dtype": "int64", "shape": (1,), "names": None},
            }
        )
        return features

    @override
    def make_frame(
        self,
        *,
        observation: dict[str, Any],
        action: Any,
        task: str,
        next_reward: Any = 0.0,
        next_terminated: Any = False,
        next_truncated: Any = False,
        next_success: Any = False,
        is_intervention: Any = False,
    ) -> dict[str, Any]:
        frame: dict[str, Any] = {
            "observation.state": np.asarray(observation["observation.state"], dtype=np.float32),
            "action": np.asarray(action, dtype=np.float32),
            "next.reward": np.asarray(next_reward, dtype=np.float32).reshape(1),
            "next.score": np.asarray(observation.get("next.score", next_reward), dtype=np.float32).reshape(1),
            "next.terminated": np.asarray(next_terminated, dtype=bool).reshape(1),
            "next.truncated": np.asarray(next_truncated, dtype=bool).reshape(1),
            "next.success": np.asarray(next_success, dtype=bool).reshape(1),
            "info.is_intervention": np.asarray(is_intervention, dtype=bool).reshape(1),
            "task": str(task),
        }
        for name in self.camera_names:
            key = f"observation.images.{name}"
            frame[key] = np.ascontiguousarray(observation[key])
        for name in ("eval_episode_id", "layout_id", "environment_seed", "policy_seed"):
            frame[f"info.{name}"] = np.asarray(observation.get(f"info.{name}", -1), dtype=np.int64).reshape(1)
        return frame
