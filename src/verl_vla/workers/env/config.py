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

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hydra.utils import instantiate
from verl.base_config import BaseConfig

from verl_vla.envs.action_executor import ActionExecutionConfig
from verl_vla.envs.arena.config import ArenaSimulatorConfig
from verl_vla.envs.libero.config import LiberoSimulatorConfig
from verl_vla.envs.piper.config import PiperConfig
from verl_vla.envs.robodojo.config import RoboDojoSimulatorConfig
from verl_vla.recorder.config import RecorderConfig
from verl_vla.teleop.config import TeleopConfig

__all__ = ["EnvWorkerConfig", "SimulatorConfig"]


@dataclass
class SimulatorConfig(BaseConfig):
    """Simulator config consumed by environment workers."""

    simulator_type: str = "libero"
    libero: LiberoSimulatorConfig = field(default_factory=LiberoSimulatorConfig)
    arena: ArenaSimulatorConfig = field(default_factory=ArenaSimulatorConfig)
    piper: PiperConfig = field(default_factory=PiperConfig)
    robodojo: RoboDojoSimulatorConfig | None = None

    def __post_init__(self):
        if not isinstance(self.libero, LiberoSimulatorConfig):
            object.__setattr__(self, "libero", instantiate(self.libero))
        if not isinstance(self.arena, ArenaSimulatorConfig):
            object.__setattr__(self, "arena", instantiate(self.arena))
        if not isinstance(self.piper, PiperConfig):
            object.__setattr__(self, "piper", instantiate(self.piper))
        if self.robodojo is not None and not isinstance(self.robodojo, RoboDojoSimulatorConfig):
            object.__setattr__(self, "robodojo", instantiate(self.robodojo))
        if self.simulator_type not in {"libero", "arena", "piper", "robodojo"}:
            raise ValueError(f"Unsupported simulator_type: {self.simulator_type}")
        if self.simulator_type == "robodojo" and self.robodojo is None:
            raise ValueError("simulator.robodojo config is required when simulator_type=robodojo")


@dataclass
class EnvWorkerConfig(BaseConfig):
    """Configuration for environment workers."""

    auto_reset: bool = False
    confirm_before_record: bool = False
    log_step_latency: bool = False
    target_step_hz: float | None = None
    action_execution: ActionExecutionConfig = field(default_factory=ActionExecutionConfig)
    modes: list[str] = field(default_factory=lambda: ["train"])
    num_envs: int = 1
    simulator: SimulatorConfig = field(default_factory=SimulatorConfig)
    teleop: TeleopConfig = field(default_factory=TeleopConfig)
    recorder: RecorderConfig = field(default_factory=RecorderConfig)
    device: str | None = None
    profiler: Any | None = None
    simulator_start_timeout_s: int = 180
    # Delay initial simulator creation by rank * this value.  Some Isaac/RTX
    # builds crash in native renderer startup when several independent GPU
    # workers initialize Kit at exactly the same time on one host.  This only
    # staggers cold startup; steady-state environment stepping remains fully
    # concurrent.
    initial_start_stagger_s: float = 0.0
    # Delay the first reset after each simulator-process start by this value
    # times the global worker-stage slot.  RoboDojo performs much of its heavy
    # USD/material initialization during the first reset rather than during
    # EnvManager.start_simulator(), so initial_start_stagger_s alone does not
    # prevent many cold Isaac resets from contending at once.
    cold_reset_stagger_s: float = 0.0
    # Ray actor RPC concurrency. Values >1 allow independent pipeline-stage
    # simulator subprocesses owned by one EnvWorker rank to step concurrently.
    # The TrainCluster lifecycle prevents reset/restart from overlapping a
    # rollout collection window.
    ray_max_concurrency: int = 1

    def __post_init__(self):
        if not isinstance(self.action_execution, ActionExecutionConfig):
            action_execution = instantiate(self.action_execution)
            if not isinstance(action_execution, ActionExecutionConfig):
                raise TypeError(
                    "action execution config must instantiate to ActionExecutionConfig, "
                    f"got {type(action_execution).__name__}"
                )
            object.__setattr__(self, "action_execution", action_execution)
        if not isinstance(self.simulator, SimulatorConfig):
            simulator = instantiate(self.simulator)
            if not isinstance(simulator, SimulatorConfig):
                raise TypeError(f"simulator config must instantiate to SimulatorConfig, got {type(simulator).__name__}")
            object.__setattr__(self, "simulator", simulator)
        if not isinstance(self.teleop, TeleopConfig):
            object.__setattr__(self, "teleop", instantiate(self.teleop))
        if not isinstance(self.recorder, RecorderConfig):
            object.__setattr__(self, "recorder", instantiate(self.recorder))
        if self.num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {self.num_envs}")
        if self.target_step_hz is not None and self.target_step_hz <= 0:
            raise ValueError(f"target_step_hz must be positive when set, got {self.target_step_hz}")
        if self.simulator_start_timeout_s <= 0:
            raise ValueError(f"simulator_start_timeout_s must be positive, got {self.simulator_start_timeout_s}")
        if self.initial_start_stagger_s < 0:
            raise ValueError(f"initial_start_stagger_s must be non-negative, got {self.initial_start_stagger_s}")
        if self.cold_reset_stagger_s < 0:
            raise ValueError(f"cold_reset_stagger_s must be non-negative, got {self.cold_reset_stagger_s}")
        if self.ray_max_concurrency <= 0:
            raise ValueError(f"ray_max_concurrency must be positive, got {self.ray_max_concurrency}")
        if not set(self.modes).issubset({"train", "eval"}):
            raise ValueError(f"Unsupported env worker modes: {self.modes}")
