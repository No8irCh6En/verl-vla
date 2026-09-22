# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import asyncio
import logging
import os
import time

import torch
from tqdm import tqdm
from verl import DataProto
from verl.single_controller.ray import RayWorkerGroup

from verl_vla.train_cluster.config import EnvLoopConfig
from verl_vla.utils.data import (
    get_dataproto_from_prefix,
    stack_dataproto_with_padding,
    update_progress_trajectory_counts,
)
from verl_vla.utils.keys import ACTION_KEY, FEEDBACK_KEY, OBS_KEY

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class EnvLoop:
    """An env loop manages interactions between models and vectorized environments."""

    def __init__(
        self,
        env_wg: RayWorkerGroup,
        rollout_wg: RayWorkerGroup,
        config: EnvLoopConfig,
        switch_actor_rollout_mode: bool,
    ):
        self.env_wg = env_wg
        self.rollout_wg = rollout_wg

        self.stage_num = config.pipeline_stage_num
        self.max_interactions = config.max_interactions
        self.rollout_partition_by_env_worker = bool(
            getattr(config, "rollout_partition_by_env_worker", False)
        )
        self.rollout_partition_size = int(getattr(config, "rollout_partition_size", 0))
        self.switch_actor_rollout_mode = switch_actor_rollout_mode
        self.rollout_collection_active = False

    def begin_rollout_collection(self) -> dict[str, float]:
        """Hold a colocated actor in rollout mode for one on-policy collection window."""

        if self.rollout_collection_active:
            raise RuntimeError("A rollout collection window is already active.")
        switch_s = 0.0
        if self.switch_actor_rollout_mode:
            switch_start_t = time.perf_counter()
            self.rollout_wg.switch_to_rollout()
            switch_s = time.perf_counter() - switch_start_t
        self.rollout_collection_active = True
        return {"timing_s/collection_switch_to_rollout": switch_s}

    def end_rollout_collection(self) -> dict[str, float]:
        """Return a colocated actor to train mode after candidate collection."""

        if not self.rollout_collection_active:
            raise RuntimeError("No rollout collection window is active.")
        switch_s = 0.0
        try:
            if self.switch_actor_rollout_mode:
                switch_start_t = time.perf_counter()
                self.rollout_wg.switch_to_train()
                switch_s = time.perf_counter() - switch_start_t
        finally:
            self.rollout_collection_active = False
        return {"timing_s/collection_switch_to_train": switch_s}

    def _strip_meta_info(self, data: DataProto) -> DataProto:
        return DataProto(
            batch=data.batch,
            non_tensor_batch=data.non_tensor_batch,
            meta_info={},
        )

    def generate_sequences(
        self,
        reset_future: asyncio.Future,
        *,
        eval: bool = False,
        active_stage_count: int | None = None,
    ) -> tuple[DataProto, DataProto]:
        if active_stage_count is None:
            active_stage_count = self.stage_num
        if not 1 <= active_stage_count <= self.stage_num:
            raise ValueError(
                f"active_stage_count must be in [1, {self.stage_num}], got {active_stage_count}."
            )
        active_stage_ids = tuple(range(active_stage_count))
        total_start_t = time.perf_counter()
        reset_wait_start_t = time.perf_counter()
        reset_results = reset_future.get()
        reset_wait_s = time.perf_counter() - reset_wait_start_t
        reset_worker_seconds = None
        reset_process_restarts = None
        if reset_results.batch is not None:
            reset_profile_fields = [
                key
                for key in ("profile.reset_seconds", "profile.reset_process_restarts")
                if key in reset_results.batch.keys()
            ]
            if "profile.reset_seconds" in reset_profile_fields:
                reset_worker_seconds = reset_results.batch["profile.reset_seconds"].to(torch.float64)
            if "profile.reset_process_restarts" in reset_profile_fields:
                reset_process_restarts = reset_results.batch["profile.reset_process_restarts"].to(torch.int64)

            # Ray dispatch may return a TensorDict view whose ``pop`` removes a
            # key from the view but leaves it visible after a later chunk/cat.
            # Materialize a profile-free TensorDict before observations are
            # restructured so instrumentation can never become ``obs.*`` model
            # input (for example ``obs.profile.reset_process_restarts``).
            if reset_profile_fields:
                reset_results = DataProto(
                    batch=reset_results.batch.exclude(*reset_profile_fields),
                    non_tensor_batch=reset_results.non_tensor_batch,
                    meta_info=reset_results.meta_info,
                )
        rollout_meta_info = {"eval": True} if eval else {}
        env_mode = "eval" if eval else "train"

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        manage_actor_mode = self.switch_actor_rollout_mode and not self.rollout_collection_active
        if manage_actor_mode:
            switch_start_t = time.perf_counter()
            self.rollout_wg.switch_to_rollout()
            switch_to_rollout_s = time.perf_counter() - switch_start_t
            run_start_t = time.perf_counter()
            output, last_obs, run_metrics = loop.run_until_complete(
                self.run(
                    reset_results,
                    rollout_meta_info,
                    env_mode=env_mode,
                    active_stage_ids=active_stage_ids,
                )
            )
            run_s = time.perf_counter() - run_start_t
            switch_start_t = time.perf_counter()
            self.rollout_wg.switch_to_train()
            switch_to_train_s = time.perf_counter() - switch_start_t
        else:
            switch_to_rollout_s = 0.0
            switch_to_train_s = 0.0
            run_start_t = time.perf_counter()
            output, last_obs, run_metrics = loop.run_until_complete(
                self.run(
                    reset_results,
                    rollout_meta_info,
                    env_mode=env_mode,
                    active_stage_ids=active_stage_ids,
                )
            )
            run_s = time.perf_counter() - run_start_t

        total_s = time.perf_counter() - total_start_t
        metrics = dict(output.meta_info.get("metrics", {}))
        metrics.update(
            {
                "timing_s/env_loop_total": total_s,
                "timing_s/env_loop_reset_wait": reset_wait_s,
                "timing_s/env_loop_run": run_s,
                "timing_s/switch_to_rollout": switch_to_rollout_s,
                "timing_s/switch_to_train": switch_to_train_s,
                "count/rollout_collection_window_active": float(self.rollout_collection_active),
            }
        )
        metrics.update(run_metrics)
        if reset_worker_seconds is not None and reset_worker_seconds.numel() > 0:
            metrics.update(
                {
                    "timing_s/env_reset_worker_mean": float(reset_worker_seconds.mean().item()),
                    "timing_s/env_reset_worker_max": float(reset_worker_seconds.max().item()),
                }
            )
        if reset_process_restarts is not None and reset_process_restarts.numel() > 0:
            metrics["count/env_reset_process_restarts"] = float(reset_process_restarts.sum().item())
        output.meta_info["metrics"] = metrics
        return output, last_obs

    async def run(
        self,
        reset_results: DataProto,
        rollout_meta_info: dict,
        *,
        env_mode: str,
        active_stage_ids: tuple[int, ...] | None = None,
    ) -> tuple[DataProto, DataProto, dict[str, float]]:
        if active_stage_ids is None:
            active_stage_ids = tuple(range(self.stage_num))
        if not active_stage_ids:
            raise ValueError("At least one pipeline stage must be active.")
        if len(set(active_stage_ids)) != len(active_stage_ids) or any(
            stage_id < 0 or stage_id >= self.stage_num for stage_id in active_stage_ids
        ):
            raise ValueError(
                f"active_stage_ids must be unique values in [0, {self.stage_num}), got {active_stage_ids}."
            )

        trajectories = {stage_id: [] for stage_id in active_stage_ids}
        last_obs_by_stage: dict[int, DataProto] = {}

        staged_obs = self._restructure_obs_data(reset_results)
        for stage_id in active_stage_ids:
            trajectories[stage_id].append({OBS_KEY: self._strip_meta_info(staged_obs[stage_id])})

        def _submit_rollout(data: DataProto):
            if not self.rollout_partition_by_env_worker:
                return [self.rollout_wg.generate_sequences(data)]
            worker_count = int(self.env_wg.world_size)
            if len(data) % worker_count != 0:
                raise ValueError(
                    "Worker-local rollout partition requires an equal batch per EnvWorker: "
                    f"batch={len(data)}, env_workers={worker_count}."
                )
            futures = []
            for worker_data in data.chunk(worker_count):
                partition_size = self.rollout_partition_size
                if partition_size:
                    if len(worker_data) % partition_size != 0:
                        raise ValueError(
                            "Worker-local rollout batch must be divisible by the configured "
                            f"partition size: batch={len(worker_data)}, "
                            f"partition_size={partition_size}."
                        )
                    worker_parts = worker_data.chunk(len(worker_data) // partition_size)
                else:
                    worker_parts = [worker_data]
                for worker_part in worker_parts:
                    worker_part.meta_info = dict(rollout_meta_info)
                    futures.append(self.rollout_wg.generate_sequences(worker_part))
            return futures

        async def _resolve_rollout(futures) -> DataProto:
            parts = await asyncio.gather(
                *[asyncio.to_thread(future.get) for future in futures]
            )
            return parts[0] if len(parts) == 1 else DataProto.concat(parts)

        rollout_futures = {}
        for stage_id in active_stage_ids:
            vla_input = staged_obs[stage_id]
            vla_input.meta_info = rollout_meta_info
            rollout_futures[stage_id] = _submit_rollout(vla_input)

        stage_timing = {
            stage_id: {
                "stage_wall_s": 0.0,
                "rollout_wait_s": 0.0,
                "env_wait_s": 0.0,
                "rollout_wait_calls": 0.0,
                "env_wait_calls": 0.0,
                "effective_steps": 0.0,
                "model_inference_s": 0.0,
                "model_inference_calls": 0.0,
                "env_step_worker_s": 0.0,
                "env_step_worker_calls": 0.0,
                "policy_lane_slots": 0.0,
                "idle_policy_lane_slots": 0.0,
                "executed_low_level_steps": 0.0,
                "gpu_memory_allocated_bytes": 0.0,
                "gpu_memory_reserved_bytes": 0.0,
                "gpu_peak_memory_allocated_bytes": 0.0,
                "gpu_peak_memory_reserved_bytes": 0.0,
            }
            for stage_id in active_stage_ids
        }
        done_lanes_by_stage: dict[int, torch.Tensor] = {}

        progress_bar = tqdm(total=self.max_interactions, desc="Rollout Progress", leave=True)
        progress_counts = {"done_eps": 0, "succ_eps": 0}
        progress_lane_state: dict[int, dict[str, torch.Tensor]] = {}

        async def _stage_loop(stage_id: int):
            stage_start_t = time.perf_counter()
            step_idx = 0
            while step_idx < self.max_interactions:
                rollout_wait_start_t = time.perf_counter()
                action_result: DataProto = await _resolve_rollout(rollout_futures[stage_id])
                stage_timing[stage_id]["rollout_wait_s"] += time.perf_counter() - rollout_wait_start_t
                stage_timing[stage_id]["rollout_wait_calls"] += 1.0

                if action_result.batch is not None:
                    inference_key = "profile.inference_seconds"
                    if inference_key in action_result.batch.keys():
                        inference_seconds = action_result.batch.pop(inference_key).to(torch.float64)
                        stage_timing[stage_id]["model_inference_s"] += float(inference_seconds.mean().item())
                        stage_timing[stage_id]["model_inference_calls"] += 1.0
                    for metric_name in (
                        "gpu_memory_allocated_bytes",
                        "gpu_memory_reserved_bytes",
                        "gpu_peak_memory_allocated_bytes",
                        "gpu_peak_memory_reserved_bytes",
                    ):
                        profile_key = f"profile.{metric_name}"
                        if profile_key in action_result.batch.keys():
                            metric_value = float(action_result.batch.pop(profile_key).max().item())
                            stage_timing[stage_id][metric_name] = max(
                                stage_timing[stage_id][metric_name], metric_value
                            )

                action_batch_size = len(action_result)
                stage_timing[stage_id]["policy_lane_slots"] += float(action_batch_size)
                if stage_id in done_lanes_by_stage:
                    stage_timing[stage_id]["idle_policy_lane_slots"] += float(
                        done_lanes_by_stage[stage_id].sum().item()
                    )

                trajectories[stage_id][-1][ACTION_KEY] = self._strip_meta_info(action_result)
                action_result.meta_info["stage_id"] = stage_id
                env_ref = self.env_wg.env_interact_step(action_result, mode=env_mode)

                env_wait_start_t = time.perf_counter()
                env_result: DataProto = await asyncio.to_thread(env_ref.get)
                stage_timing[stage_id]["env_wait_s"] += time.perf_counter() - env_wait_start_t
                stage_timing[stage_id]["env_wait_calls"] += 1.0
                env_step_key = "profile.env_step_seconds"
                if env_result.batch is not None and env_step_key in env_result.batch.keys():
                    env_step_seconds = env_result.batch.pop(env_step_key).to(torch.float64)
                    stage_timing[stage_id]["env_step_worker_s"] += float(env_step_seconds.mean().item())
                    stage_timing[stage_id]["env_step_worker_calls"] += 1.0
                chunk_done = env_result.batch["next.terminated"].bool() | env_result.batch["next.truncated"].bool()
                episode_done = chunk_done.flatten(start_dim=1).any(dim=1)
                if stage_id not in done_lanes_by_stage:
                    done_lanes_by_stage[stage_id] = torch.zeros_like(episode_done)
                valid_low_level_steps = (chunk_done.cumsum(dim=1) - chunk_done.long()) == 0
                valid_low_level_steps &= ~done_lanes_by_stage[stage_id].unsqueeze(1)
                stage_timing[stage_id]["executed_low_level_steps"] += float(
                    valid_low_level_steps.sum().item()
                )
                done_lanes_by_stage[stage_id] |= episode_done
                update_progress_trajectory_counts(
                    env_result,
                    stage_id=stage_id,
                    progress_counts=progress_counts,
                    progress_lane_state=progress_lane_state,
                )
                progress_bar.set_postfix(progress_counts, refresh=False)

                next_step = self._strip_meta_info(get_dataproto_from_prefix(env_result, FEEDBACK_KEY, "."))
                next_obs = self._strip_meta_info(get_dataproto_from_prefix(env_result, OBS_KEY, "."))

                current_slot = trajectories[stage_id].pop()
                current_slot[FEEDBACK_KEY] = next_step
                trajectories[stage_id].append(current_slot)
                last_obs_by_stage[stage_id] = next_obs

                stage_timing[stage_id]["effective_steps"] += 1.0
                step_idx += 1
                if stage_id == 0:
                    progress_bar.update(1)
                if step_idx < self.max_interactions:
                    trajectories[stage_id].append({OBS_KEY: next_obs})

                if step_idx < self.max_interactions:
                    vla_input = next_obs
                    vla_input.meta_info = rollout_meta_info
                    rollout_futures[stage_id] = _submit_rollout(vla_input)

            stage_timing[stage_id]["stage_wall_s"] = time.perf_counter() - stage_start_t

        try:
            await asyncio.gather(*[asyncio.create_task(_stage_loop(sid)) for sid in active_stage_ids])
        finally:
            progress_bar.close()
        success_count = progress_counts["succ_eps"]
        trajectory_count = progress_counts["done_eps"]
        print(
            f"Rollout collected trajectories: success={success_count}, "
            f"failed={trajectory_count - success_count}, total={trajectory_count}"
        )
        finish_start_t = time.perf_counter()
        self.env_wg.finish_rollout()
        finish_rollout_s = time.perf_counter() - finish_start_t
        collate_start_t = time.perf_counter()
        collated_meta_info = dict(rollout_meta_info)
        output = self._collate_trajectories(trajectories, meta_info=collated_meta_info)
        collate_trajectories_s = time.perf_counter() - collate_start_t
        last_obs_start_t = time.perf_counter()
        last_obs = DataProto.concat(
            [self._strip_meta_info(last_obs_by_stage[stage_id]) for stage_id in active_stage_ids]
        )
        last_obs_collate_s = time.perf_counter() - last_obs_start_t
        stage_wall_max_s = max(stage_timing[sid]["stage_wall_s"] for sid in active_stage_ids)
        rollout_wait_sum_s = sum(stage_timing[sid]["rollout_wait_s"] for sid in active_stage_ids)
        env_wait_sum_s = sum(stage_timing[sid]["env_wait_s"] for sid in active_stage_ids)
        rollout_wait_calls = sum(stage_timing[sid]["rollout_wait_calls"] for sid in active_stage_ids)
        env_wait_calls = sum(stage_timing[sid]["env_wait_calls"] for sid in active_stage_ids)
        effective_steps = sum(stage_timing[sid]["effective_steps"] for sid in active_stage_ids)
        model_inference_sum_s = sum(stage_timing[sid]["model_inference_s"] for sid in active_stage_ids)
        model_inference_calls = sum(stage_timing[sid]["model_inference_calls"] for sid in active_stage_ids)
        env_step_worker_sum_s = sum(stage_timing[sid]["env_step_worker_s"] for sid in active_stage_ids)
        env_step_worker_calls = sum(stage_timing[sid]["env_step_worker_calls"] for sid in active_stage_ids)
        policy_lane_slots = sum(stage_timing[sid]["policy_lane_slots"] for sid in active_stage_ids)
        idle_policy_lane_slots = sum(stage_timing[sid]["idle_policy_lane_slots"] for sid in active_stage_ids)
        executed_low_level_steps = sum(
            stage_timing[sid]["executed_low_level_steps"] for sid in active_stage_ids
        )

        run_metrics = {
            "timing_s/env_loop_stage_wall_max": stage_wall_max_s,
            "timing_s/env_loop_finish_rollout": finish_rollout_s,
            "timing_s/env_loop_collate_trajectories": collate_trajectories_s,
            "timing_s/env_loop_collate_last_obs": last_obs_collate_s,
            "timing_s/env_loop_rollout_wait_sum": rollout_wait_sum_s,
            "timing_s/env_loop_env_wait_sum": env_wait_sum_s,
            "timing_s/env_loop_rollout_wait_avg": rollout_wait_sum_s / max(1.0, rollout_wait_calls),
            "timing_s/env_loop_env_wait_avg": env_wait_sum_s / max(1.0, env_wait_calls),
            "timing_s/fastwam_inference_sum": model_inference_sum_s,
            "timing_s/fastwam_inference_avg": model_inference_sum_s / max(1.0, model_inference_calls),
            "timing_s/robodojo_vector_step_sum": env_step_worker_sum_s,
            "timing_s/robodojo_vector_step_avg": env_step_worker_sum_s / max(1.0, env_step_worker_calls),
            "timing_s/ray_dataproto_rollout_overhead_estimate": max(
                0.0, rollout_wait_sum_s - model_inference_sum_s
            ),
            "timing_s/ray_dataproto_env_overhead_estimate": max(0.0, env_wait_sum_s - env_step_worker_sum_s),
            "throughput/env_loop_effective_steps_per_s": effective_steps / max(1e-12, stage_wall_max_s),
            "throughput/env_loop_env_rpc_per_s": env_wait_calls / max(1e-12, stage_wall_max_s),
            "throughput/policy_calls_per_s": rollout_wait_calls / max(1e-12, stage_wall_max_s),
            "throughput/policy_chunks_per_s": (policy_lane_slots - idle_policy_lane_slots)
            / max(1e-12, stage_wall_max_s),
            "throughput/env_transitions_per_s": executed_low_level_steps / max(1e-12, stage_wall_max_s),
            "throughput/trajectories_per_hour": trajectory_count * 3600.0 / max(1e-12, stage_wall_max_s),
            "count/env_loop_effective_steps": effective_steps,
            "count/env_loop_env_rpc_calls": env_wait_calls,
            "count/env_loop_rollout_wait_calls": rollout_wait_calls,
            "count/policy_lane_slots": policy_lane_slots,
            "count/idle_policy_lane_slots": idle_policy_lane_slots,
            "count/executed_low_level_steps": executed_low_level_steps,
            "fraction/idle_policy_lane_slots": idle_policy_lane_slots / max(1.0, policy_lane_slots),
            "count/rollout_worker_ranks": float(self.rollout_wg.world_size),
            "count/env_worker_ranks": float(self.env_wg.world_size),
            "count/active_pipeline_stages": float(len(active_stage_ids)),
        }
        for metric_name in (
            "gpu_memory_allocated_bytes",
            "gpu_memory_reserved_bytes",
            "gpu_peak_memory_allocated_bytes",
            "gpu_peak_memory_reserved_bytes",
        ):
            run_metrics[f"memory_bytes/fastwam_{metric_name}"] = max(
                stage_timing[sid][metric_name] for sid in active_stage_ids
            )
        return output, last_obs, run_metrics

    def _restructure_obs_data(self, data_proto: DataProto) -> list[DataProto]:
        num_workers = self.env_wg.world_size
        staged_data = [[] for _ in range(self.stage_num)]
        chunks = data_proto.chunk(num_workers)
        for worker_chunk in chunks:
            stage_chunks = worker_chunk.chunk(self.stage_num)
            for stage_id, data in enumerate(stage_chunks):
                staged_data[stage_id].append(data)
        return [DataProto.concat(data_list) for data_list in staged_data]

    def _collate_trajectories(self, trajectories: dict, meta_info) -> DataProto:
        stage_ids = tuple(sorted(trajectories))
        if not stage_ids:
            raise ValueError("Cannot collate an empty pipeline-stage set.")
        trajectory_lengths = {len(trajectories[stage_id]) for stage_id in stage_ids}
        if len(trajectory_lengths) != 1:
            raise ValueError(
                f"All active pipeline stages must have equal trajectory lengths, got {trajectory_lengths}."
            )
        # Collate each stage directly to [B_stage, S, ...], then concatenate
        # stages once along B.  The former step-first implementation repeatedly
        # concatenated multi-camera observations at every step before stacking,
        # creating O(stage_count^2) copies for four-stage candidate collection.
        collated_stages = []
        for stage_id in stage_ids:
            batch_dict = {}
            stage_trajectory = trajectories[stage_id]
            for field_key in (OBS_KEY, ACTION_KEY, FEEDBACK_KEY):
                batch_dict.update(
                    stack_dataproto_with_padding(
                        [step[field_key] for step in stage_trajectory],
                        field_key,
                    )
                )
            collated_stages.append(DataProto.from_single_dict(batch_dict, meta_info={}))

        output = DataProto.concat(collated_stages)
        output.meta_info = meta_info
        return output
