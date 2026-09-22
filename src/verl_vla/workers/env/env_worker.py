# Copyright 2025 The RLinf Authors.
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.distributed.device_mesh import init_device_mesh
from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import (
    Dispatch,
    collect_lazy_compute_data_proto,
    dispatch_lazy_compute_data_proto,
    register,
)
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import (
    get_device_name,
)
from verl.utils.distributed import initialize_global_process_group_ray
from verl.utils.profiler import DistProfiler, DistProfilerExtension, ProfilerConfig

from verl_vla.recorder import merge_lerobot_datasets
from verl_vla.workers.env.config import EnvWorkerConfig

from .env_manager import EnvManager, SimulatorCommandTimeoutError

logger = logging.getLogger(__name__)


def _reset_simulator_with_recovery(
    simulator: EnvManager,
    *,
    options: dict,
    timeout_s: float,
    max_process_restarts: int,
):
    """Retry a timed-out reset from the latest completed reset boundary.

    Snapshotting immediately after every successful reset is essential for a
    simulator shared by train and evaluation.  Otherwise a timeout during a
    later reset can restore the cursor captured by an old periodic restart and
    silently put different worker/stage processes on different policy-seed
    waves.
    """

    process_restarts = 0
    while True:
        try:
            result = simulator.call("reset", options=options, timeout_s=timeout_s)
            simulator.snapshot_state(timeout_s=min(float(timeout_s), 30.0))
            return result, process_restarts
        except SimulatorCommandTimeoutError:
            if process_restarts >= max_process_restarts:
                raise
            process_restarts += 1
            logger.error(
                "RoboDojo reset/state-snapshot RPC timed out; force-restarting simulator "
                "(retry=%s/%s, rank=%s, stage_id=%s).",
                process_restarts,
                max_process_restarts,
                simulator.rank,
                simulator.stage_id,
            )
            simulator.force_stop_simulator()
            simulator.start_simulator()


def _restart_simulators_with_parallel_start(simulators: list[EnvManager]) -> None:
    """Restart independent stage subprocesses without serializing Isaac startup.

    All old children are stopped before any replacement is started.  This keeps
    the number of live simulator processes bounded by the configured stage
    count while allowing the expensive, independent Isaac/Kit startups to
    overlap inside one EnvWorker.
    """

    if not simulators:
        return
    for simulator in simulators:
        simulator.stop_simulator()
    if len(simulators) == 1:
        simulators[0].start_simulator()
        return
    with ThreadPoolExecutor(max_workers=len(simulators)) as executor:
        list(executor.map(lambda simulator: simulator.start_simulator(), simulators))


def dispatch_reset_env(worker_group, *args, **kwargs):
    mode = kwargs.pop("mode", "train")
    reset_eval = kwargs.pop("reset_eval", False)
    reset_args = DataProto.from_dict(
        meta_info={
            "mode": mode,
            "reset_eval": reset_eval,
        }
    )
    return dispatch_lazy_compute_data_proto("env", worker_group, reset_args, **kwargs)


def dispatch_env_interact_step(worker_group, *args, **kwargs):
    mode = kwargs.pop("mode", "train")
    all_args, all_kwargs = dispatch_lazy_compute_data_proto("env", worker_group, *args, **kwargs)
    all_kwargs["mode"] = [mode] * worker_group.world_size
    return all_args, all_kwargs


def collect_reset_env(worker_group, *args, **kwargs):
    return collect_lazy_compute_data_proto("env", worker_group, *args, **kwargs)


def put_tensor_cpu(data_dict):
    for key, value in data_dict.items():
        if isinstance(value, dict):
            data_dict[key] = put_tensor_cpu(value)
        if isinstance(value, torch.Tensor):
            data_dict[key] = value.cpu().contiguous()
    return data_dict


def create_env_batch_dataproto(obs, rewards, terminations, truncations, successes, scores=None, meta=None):
    step_result = {
        "observation": obs["observation"],
        "task": obs["task"],
        "task_id": obs.get("task_id"),
        "eval_episode_id": obs.get("eval_episode_id"),
        "layout_id": obs.get("layout_id"),
        "environment_seed": obs.get("environment_seed"),
        "policy_seed": obs.get("policy_seed"),
        "next.reward": rewards,
        "next.terminated": terminations,
        "next.truncated": truncations,
        "next.success": successes,
    }
    if meta is not None:
        step_result["meta"] = meta

    step_result = put_tensor_cpu(step_result)
    obs_tensor_batch = {}
    observations = step_result["observation"]
    if observations:
        for key in observations[0]:
            obs_tensor_batch[f"obs.{key}"] = torch.as_tensor(
                np.stack([observation[key] for observation in observations])
            )
    tensor_batch = {
        **obs_tensor_batch,
        "next.reward": step_result["next.reward"],
        "next.terminated": step_result["next.terminated"],
        "next.truncated": step_result["next.truncated"],
        "next.success": step_result["next.success"],
    }
    if scores is not None:
        tensor_batch["next.score"] = torch.as_tensor(scores, dtype=torch.float32)
    non_tensor_batch = {"obs.task": step_result["task"]}
    non_tensor_batch["obs.task_id"] = np.asarray(step_result["task_id"], dtype=np.int64)
    if step_result["eval_episode_id"] is not None:
        non_tensor_batch["obs.eval_episode_id"] = np.asarray(step_result["eval_episode_id"], dtype=np.int64)
    for field in ("layout_id", "environment_seed", "policy_seed"):
        if step_result[field] is not None:
            non_tensor_batch[f"obs.{field}"] = np.asarray(step_result[field], dtype=np.int64)
    output = DataProto.from_dict(tensors=tensor_batch, non_tensors=non_tensor_batch)

    return output


class EnvWorker(Worker, DistProfilerExtension):
    def __init__(self, config: DictConfig, role=None):
        Worker.__init__(self)
        self.cfg = config
        self.role = role
        self.env_worker_cfg: EnvWorkerConfig = config.env_worker
        self.simulator_type = self.env_worker_cfg.simulator.simulator_type
        self.simulator_cfg = OmegaConf.structured(self.env_worker_cfg)
        self.train_video_cnt = 0
        self.eval_video_cnt = 0

        self.simulator_list = []
        self.last_obs_list = []
        self.last_dones_list = []
        self.eval_simulator_list = []

        self.stage_num = self.cfg.env_loop.pipeline_stage_num
        device_name = self.env_worker_cfg.device or get_device_name()
        if device_name == "cpu":
            # CPU env workers do not need torch distributed collectives; only Ray dispatch metadata is required.
            self._register_dispatch_collect_info("env", dp_rank=self.rank, is_collect=True)
        else:
            initialize_global_process_group_ray(timeout_second=None)
            env_device_mesh = init_device_mesh(
                device_name, mesh_shape=(self.world_size, 1), mesh_dim_names=["dp", "tp"]
            )
            self._register_dispatch_collect_info("env", dp_rank=env_device_mesh["dp"].get_local_rank(), is_collect=True)

        # Initialize profiler
        omega_profiler_config = self.env_worker_cfg.profiler or {}
        profiler_config = omega_conf_to_dataclass(omega_profiler_config, dataclass_type=ProfilerConfig)
        if omega_profiler_config.get("tool", None) in ["npu", "nsys", "torch", "torch_memory"]:
            tool_config = omega_conf_to_dataclass(
                omega_profiler_config.get("tool_config", {}).get(omega_profiler_config.get("tool"))
            )
        else:
            tool_config = None
        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=profiler_config, tool_config=tool_config)
        )

    def _robodojo_simulator_cfg_for_stage(self, stage_id: int):
        """Bind one persistent RoboDojo task to this worker-stage process."""

        robodojo_cfg = self.env_worker_cfg.simulator.robodojo
        if robodojo_cfg is None:
            raise ValueError("RoboDojo simulator requires simulator.robodojo config.")
        task_name, task_id = robodojo_cfg.task_assignment(
            worker_rank=int(self.rank),
            worker_world_size=int(self.world_size),
            stage_id=int(stage_id),
            stage_num=int(self.stage_num),
        )
        stage_cfg = copy.deepcopy(self.simulator_cfg)
        OmegaConf.update(stage_cfg, "simulator.robodojo.task_name", task_name)
        OmegaConf.update(stage_cfg, "simulator.robodojo.task_id", task_id)
        logger.warning(
            "RoboDojo persistent task assignment: worker_rank=%d stage_id=%d task=%s task_id=%d",
            int(self.rank),
            int(stage_id),
            task_name,
            task_id,
        )
        return stage_cfg

    def _make_eval_env_cfg(self):
        eval_cfg = copy.deepcopy(self.simulator_cfg)
        OmegaConf.set_readonly(eval_cfg, False)
        OmegaConf.set_struct(eval_cfg, False)
        if "teleop" in eval_cfg:
            OmegaConf.set_readonly(eval_cfg.teleop, False)
            eval_cfg.teleop.enable = False
        return eval_cfg

    def _simulators(self, mode: str):
        if mode == "eval":
            if self.eval_simulator_list:
                return self.eval_simulator_list
            # Arena / Isaac / LeRobot run a single simulator instance that serves
            # both train and eval (eval is selected via the reset_eval option at
            # reset time), so reuse the shared simulator list instead of a
            # dedicated eval one.
            if self.simulator_type in ("arena", "isaac", "lerobot", "piper", "robodojo") and self.simulator_list:
                return self.simulator_list
            raise RuntimeError("Eval simulator is not initialized. Add 'eval' to env.env_worker.modes.")
        return self.simulator_list

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    @DistProfiler.annotate(color="green", role="env_init")
    def init_worker(self):
        logger.warning(
            "Environment worker placement: simulator=%s rank=%s cuda_visible_devices=%s",
            self.simulator_type,
            self.rank,
            os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        )
        start_delay_s = float(self.env_worker_cfg.initial_start_stagger_s) * int(self.rank)
        if start_delay_s > 0:
            logger.warning(
                "Staggering initial simulator startup by %.1fs for env rank %s",
                start_delay_s,
                self.rank,
            )
            time.sleep(start_delay_s)
        if self.simulator_type == "libero":
            from verl_vla.envs.libero.libero_env import LiberoEnv

            modes = list(self.env_worker_cfg.modes)
            if not set(modes).issubset({"train", "eval"}):
                raise ValueError(f"Unsupported LIBERO env modes: {modes}")
            eval_cfg = self._make_eval_env_cfg() if "eval" in modes else None
            for stage_id in range(self.stage_num):
                if "train" in modes:
                    self.simulator_list.append(
                        EnvManager(
                            self.simulator_cfg,
                            rank=self._rank,
                            world_size=self._world_size,
                            env_cls=LiberoEnv,
                            stage_id=stage_id,
                            stage_num=self.stage_num,
                            start_timeout_s=self.env_worker_cfg.simulator_start_timeout_s,
                        )
                    )
                if eval_cfg is not None:
                    self.eval_simulator_list.append(
                        EnvManager(
                            eval_cfg,
                            rank=self._rank,
                            world_size=self._world_size,
                            env_cls=LiberoEnv,
                            stage_id=stage_id,
                            stage_num=self.stage_num,
                            only_eval=True,
                            start_timeout_s=self.env_worker_cfg.simulator_start_timeout_s,
                        )
                    )

        elif self.simulator_type == "isaac":
            from verl_vla.envs.isaac.isaac_env import IsaacEnv

            for stage_id in range(self.stage_num):
                self.simulator_list.append(
                    EnvManager(
                        self.simulator_cfg,
                        rank=self._rank,
                        world_size=self._world_size,
                        env_cls=IsaacEnv,
                        stage_id=stage_id,
                        stage_num=self.stage_num,
                        start_timeout_s=self.env_worker_cfg.simulator_start_timeout_s,
                    )
                )
        elif self.simulator_type == "lerobot":
            from verl_vla.envs.lerobot.lerobot_env import LeRobotEnv

            for stage_id in range(self.stage_num):
                self.simulator_list.append(
                    EnvManager(
                        self.simulator_cfg,
                        rank=self._rank,
                        world_size=self._world_size,
                        env_cls=LeRobotEnv,
                        stage_id=stage_id,
                        stage_num=self.stage_num,
                        start_timeout_s=self.env_worker_cfg.simulator_start_timeout_s,
                    )
                )
        elif self.simulator_type == "arena":
            from verl_vla.envs.arena.arena_env import IsaacLabArenaEnv

            for stage_id in range(self.stage_num):
                self.simulator_list.append(
                    EnvManager(
                        self.simulator_cfg,
                        rank=self._rank,
                        world_size=self._world_size,
                        env_cls=IsaacLabArenaEnv,
                        stage_id=stage_id,
                        stage_num=self.stage_num,
                        start_timeout_s=self.env_worker_cfg.simulator_start_timeout_s,
                    )
                )
        elif self.simulator_type == "piper":
            from verl_vla.envs.piper.piper_env import PiperEnv

            for stage_id in range(self.stage_num):
                self.simulator_list.append(
                    EnvManager(
                        self.simulator_cfg,
                        rank=self._rank,
                        world_size=self._world_size,
                        env_cls=PiperEnv,
                        stage_id=stage_id,
                        stage_num=self.stage_num,
                        start_timeout_s=self.env_worker_cfg.simulator_start_timeout_s,
                    )
                )
        elif self.simulator_type == "robodojo":
            from verl_vla.envs.robodojo.robodojo_env import RoboDojoEnv

            for stage_id in range(self.stage_num):
                self.simulator_list.append(
                    EnvManager(
                        self._robodojo_simulator_cfg_for_stage(stage_id),
                        rank=self._rank,
                        world_size=self._world_size,
                        env_cls=RoboDojoEnv,
                        stage_id=stage_id,
                        stage_num=self.stage_num,
                        start_timeout_s=self.env_worker_cfg.simulator_start_timeout_s,
                    )
                )
        else:
            raise NotImplementedError(f"Simulator type {self.simulator_type} not implemented")

        for simulator in self.simulator_list:
            simulator.start_simulator()
        for simulator in self.eval_simulator_list:
            simulator.start_simulator()

    @register(
        dispatch_mode={
            "dispatch_fn": dispatch_env_interact_step,
            "collect_fn": collect_reset_env,
        },
        blocking=False,
    )
    @DistProfiler.annotate(color="red", role="env_interact_step")
    def env_interact_step(self, data: DataProto, mode: str = "train") -> dict:
        """
        This function is used to interact with the environment.
        """
        if data.batch is not None and "action" in data.batch.keys():
            chunk_actions: torch.Tensor = data.batch["action"]
            chunk_values = data.batch.get("critic_value")
        else:
            chunk_actions = torch.as_tensor(data.non_tensor_batch["action"])
            chunk_values = (
                torch.as_tensor(data.non_tensor_batch["critic_value"])
                if "critic_value" in data.non_tensor_batch
                else None
            )
        stage_id: int = data.meta_info["stage_id"]

        simulators = self._simulators(mode)
        step_start = time.perf_counter()
        step_output = simulators[stage_id].step(chunk_actions, chunk_values=chunk_values)
        step_seconds = time.perf_counter() - step_start
        if len(step_output) == 4:
            extracted_obs, chunk_rewards, chunk_terminations, chunk_truncations = step_output
            chunk_successes = chunk_terminations
            chunk_scores = None
        elif len(step_output) == 5:
            extracted_obs, chunk_rewards, chunk_terminations, chunk_truncations, chunk_successes = step_output
            chunk_scores = None
        else:
            extracted_obs, chunk_rewards, chunk_terminations, chunk_truncations, chunk_successes, infos = step_output
            chunk_scores = infos.get("score") if isinstance(infos, dict) else None

        env_batch = create_env_batch_dataproto(
            obs=extracted_obs,
            rewards=chunk_rewards,
            terminations=chunk_terminations,
            truncations=chunk_truncations,
            successes=chunk_successes,
            scores=chunk_scores,
        )
        env_batch.batch["profile.env_step_seconds"] = torch.full(
            (len(env_batch),), step_seconds, dtype=torch.float64
        )
        return env_batch

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_eval_benchmark_size(self):
        """Get the number of episodes in the eval benchmark."""
        simulator = self.eval_simulator_list[0] if self.eval_simulator_list else self.simulator_list[0]
        return int(simulator.env_benchmark_size())

    @register(
        dispatch_mode={
            "dispatch_fn": dispatch_reset_env,
            "collect_fn": collect_reset_env,
        },
        blocking=False,
    )
    @DistProfiler.annotate(color="blue", role="env_reset_env")
    def reset_env(self, _data: DataProto):
        mode = _data.meta_info.get("mode", "train")
        reset_eval = bool(_data.meta_info.get("reset_eval", False))
        simulators = self._simulators(mode)

        robodojo_cfg = self.env_worker_cfg.simulator.robodojo if self.simulator_type == "robodojo" else None

        def _reset_stage(stage_id: int):
            options = {
                "env_idx": list(range(self.env_worker_cfg.num_envs)),
                "mode": mode,
                "extra": {"mode": mode},
            }
            if reset_eval:
                options["reset_eval"] = True
            reset_start = time.perf_counter()
            if robodojo_cfg is None:
                result = simulators[stage_id].reset(options=options)
                process_restarts = 0
            else:
                cold_stagger_s = float(self.env_worker_cfg.cold_reset_stagger_s)
                if simulators[stage_id].cold_start_pending and cold_stagger_s > 0:
                    # task_assignment() is stage-major, so use the same global
                    # slot order to spread cold USD/material loads across all
                    # EnvWorkers on the node.  A recovered single child does
                    # not affect the on-policy/group boundary.
                    global_stage_slot = int(stage_id) * int(self.world_size) + int(self.rank)
                    delay_s = cold_stagger_s * global_stage_slot
                    if delay_s > 0:
                        logger.warning(
                            "Staggering cold RoboDojo reset by %.1fs "
                            "(rank=%s stage_id=%s global_stage_slot=%s).",
                            delay_s,
                            self.rank,
                            stage_id,
                            global_stage_slot,
                        )
                        time.sleep(delay_s)
                result, process_restarts = _reset_simulator_with_recovery(
                    simulators[stage_id],
                    options=options,
                    timeout_s=float(robodojo_cfg.reset_rpc_timeout_s),
                    max_process_restarts=int(robodojo_cfg.reset_process_max_restarts),
                )
            return stage_id, result, time.perf_counter() - reset_start, process_restarts

        if self.stage_num > 1:
            # Every stage owns an independent EnvManager subprocess.  Reset
            # their already-initialized RoboDojo instances concurrently, as
            # the rollout step path already does, while restoring stage-major
            # order before DataProto assembly.
            with ThreadPoolExecutor(max_workers=self.stage_num) as executor:
                stage_results = list(executor.map(_reset_stage, range(self.stage_num)))
        else:
            stage_results = [_reset_stage(0)]
        stage_results.sort(key=lambda item: item[0])
        result_list = [item[1] for item in stage_results]
        reset_durations = [item[2] for item in stage_results]
        reset_process_restarts = [item[3] for item in stage_results]
        output_tensor_dict = {}
        output_non_tensor_dict = {}

        observations = [observation for obs, _info in result_list for observation in obs["observation"]]
        if observations:
            for key in observations[0]:
                output_tensor_dict[key] = torch.as_tensor(np.stack([observation[key] for observation in observations]))
        output_non_tensor_dict["task"] = [task for obs, _info in result_list for task in obs["task"]]
        task_ids = [task_id for obs, _info in result_list for task_id in obs.get("task_id", [])]
        output_non_tensor_dict["task_id"] = np.asarray(task_ids, dtype=np.int64)
        eval_episode_ids = [
            eval_episode_id for obs, _info in result_list for eval_episode_id in obs.get("eval_episode_id", [])
        ]
        if eval_episode_ids:
            output_non_tensor_dict["eval_episode_id"] = np.asarray(eval_episode_ids, dtype=np.int64)
        for field in ("layout_id", "environment_seed", "policy_seed"):
            values = [value for obs, _info in result_list for value in obs.get(field, [])]
            if values:
                output_non_tensor_dict[field] = np.asarray(values, dtype=np.int64)

        output_tensor_dict["profile.reset_seconds"] = torch.as_tensor(
            [
                reset_seconds
                for (obs, _info), reset_seconds in zip(result_list, reset_durations, strict=True)
                for _ in obs["observation"]
            ],
            dtype=torch.float64,
        )
        output_tensor_dict["profile.reset_process_restarts"] = torch.as_tensor(
            [
                process_restarts
                for (obs, _info), process_restarts in zip(result_list, reset_process_restarts, strict=True)
                for _ in obs["observation"]
            ],
            dtype=torch.int64,
        )

        output = DataProto.from_dict(tensors=output_tensor_dict, non_tensors=output_non_tensor_dict)
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def restart_simulators(self, mode: str = "train") -> None:
        """Recreate simulator subprocesses while preserving adapter-owned state."""

        simulators = self._simulators(mode)
        robodojo_cfg = self.env_worker_cfg.simulator.robodojo if self.simulator_type == "robodojo" else None
        if robodojo_cfg is not None and bool(robodojo_cfg.parallel_restart_stages):
            _restart_simulators_with_parallel_start(simulators)
        else:
            for simulator in simulators:
                simulator.stop_simulator()
                simulator.start_simulator()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    @DistProfiler.annotate(color="blue", role="env_record")
    def record(self):
        assert self.stage_num == 1
        self.simulator_list[0].record()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def replay_episode(self, episode):
        assert self.stage_num == 1
        return self.simulator_list[0].replay_episode(**episode)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    @DistProfiler.annotate(color="gray", role="env_finish_rollout")
    def finish_rollout(self):
        for simulator in self.simulator_list:
            simulator.finish_rollout()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    @DistProfiler.annotate(color="gray", role="env_pop_lerobot_dataset")
    def pop_lerobot_dataset(self):
        recorder_cfg = self.env_worker_cfg.recorder
        if not recorder_cfg.enable or not recorder_cfg.lerobot.enable:
            return None

        datasets = []
        for simulator in self.simulator_list:
            dataset = simulator.pop_completed_dataset()
            if dataset is not None:
                datasets.append(dataset)
        if not datasets:
            return None

        root = Path(recorder_cfg.lerobot.root)
        repo_id = f"{recorder_cfg.lerobot.repo_id}_rank_{self._rank}"
        return merge_lerobot_datasets(
            roots=[dataset["root"] for dataset in datasets],
            output_root=root / repo_id,
            repo_id=repo_id,
            repo_ids=[dataset["repo_id"] for dataset in datasets],
            append=True,
        )
