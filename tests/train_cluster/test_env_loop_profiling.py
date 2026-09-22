from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from verl import DataProto

from verl_vla.train_cluster.env_loop import EnvLoop


class _Ref:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


def _action(batch_size: int, inference_seconds: float, peak_bytes: float) -> DataProto:
    return DataProto.from_dict(
        tensors={
            "action": torch.zeros(batch_size, 24, 14),
            "full_action": torch.zeros(batch_size, 32, 14),
            "profile.inference_seconds": torch.full((batch_size,), inference_seconds, dtype=torch.float64),
            "profile.gpu_memory_allocated_bytes": torch.full((batch_size,), peak_bytes / 2, dtype=torch.float64),
            "profile.gpu_memory_reserved_bytes": torch.full((batch_size,), peak_bytes, dtype=torch.float64),
            "profile.gpu_peak_memory_allocated_bytes": torch.full(
                (batch_size,), peak_bytes * 0.75, dtype=torch.float64
            ),
            "profile.gpu_peak_memory_reserved_bytes": torch.full((batch_size,), peak_bytes, dtype=torch.float64),
        }
    )


def _env_step(done: list[bool], *, step_seconds: float) -> DataProto:
    batch_size = len(done)
    terminated = torch.zeros(batch_size, 24, dtype=torch.bool)
    for row, is_done in enumerate(done):
        if is_done:
            terminated[row] = True
    return DataProto.from_dict(
        tensors={
            "obs.state": torch.zeros(batch_size, 14),
            "next.reward": torch.zeros(batch_size, 24),
            "next.terminated": terminated,
            "next.truncated": torch.zeros_like(terminated),
            "next.success": terminated.clone(),
            "profile.env_step_seconds": torch.full((batch_size,), step_seconds, dtype=torch.float64),
        },
        non_tensors={
            "obs.task": np.asarray(["task"] * batch_size, dtype=object),
            "obs.task_id": np.arange(batch_size, dtype=np.int64),
        },
    )


class _RolloutGroup:
    world_size = 2

    def __init__(self):
        self.actions = iter([_action(2, 0.1, 1000), _action(2, 0.2, 2000)])

    def generate_sequences(self, _obs):
        return _Ref(next(self.actions))


class _EnvGroup:
    world_size = 1

    def __init__(self):
        self.steps = iter([_env_step([True, False], step_seconds=0.3), _env_step([True, True], step_seconds=0.4)])
        self.finished = False

    def env_interact_step(self, _action, *, mode):
        assert mode == "train"
        return _Ref(next(self.steps))

    def finish_rollout(self):
        self.finished = True


def test_env_loop_reports_worker_timings_memory_throughput_and_idle_slots():
    env_group = _EnvGroup()
    loop = EnvLoop(
        env_wg=env_group,
        rollout_wg=_RolloutGroup(),
        config=SimpleNamespace(pipeline_stage_num=1, max_interactions=2),
        switch_actor_rollout_mode=False,
    )
    reset = DataProto.from_dict(
        tensors={
            "state": torch.zeros(2, 14),
            "profile.reset_seconds": torch.full((2,), 0.5, dtype=torch.float64),
            "profile.reset_process_restarts": torch.tensor([1, 0], dtype=torch.int64),
        },
        non_tensors={"task": np.asarray(["task", "task"], dtype=object), "task_id": np.asarray([0, 1])},
    )

    output, _last_obs = loop.generate_sequences(_Ref(reset))
    metrics = output.meta_info["metrics"]

    assert env_group.finished is True
    assert metrics["timing_s/env_reset_worker_mean"] == 0.5
    assert metrics["count/env_reset_process_restarts"] == 1
    assert metrics["timing_s/fastwam_inference_sum"] == pytest.approx(0.3)
    assert metrics["timing_s/robodojo_vector_step_sum"] == pytest.approx(0.7)
    assert metrics["count/policy_lane_slots"] == 4
    assert metrics["count/idle_policy_lane_slots"] == 1
    assert metrics["count/executed_low_level_steps"] == 26
    assert metrics["fraction/idle_policy_lane_slots"] == 0.25
    assert metrics["count/rollout_worker_ranks"] == 2
    assert metrics["memory_bytes/fastwam_gpu_peak_memory_reserved_bytes"] == 2000
    assert metrics["throughput/policy_calls_per_s"] > 0
    assert metrics["throughput/policy_chunks_per_s"] > 0
    assert metrics["throughput/env_transitions_per_s"] > 0
    assert metrics["throughput/trajectories_per_hour"] > 0
    assert "action.profile.inference_seconds" not in output.batch.keys()
    assert "profile.env_step_seconds" not in output.batch.keys()
    assert "obs.profile.reset_process_restarts" not in output.batch.keys()


def test_env_loop_can_run_only_requested_stage_from_stage_major_reset_batch():
    generated_states: list[list[float]] = []

    class RolloutGroup:
        world_size = 1

        def generate_sequences(self, obs):
            generated_states.append(obs.batch["state"][:, 0].tolist())
            return _Ref(_action(len(obs), 0.1, 1000))

    class EnvGroup:
        world_size = 2

        def env_interact_step(self, action, *, mode):
            assert mode == "train"
            assert action.meta_info["stage_id"] == 0
            return _Ref(_env_step([True] * len(action), step_seconds=0.2))

        def finish_rollout(self):
            pass

    loop = EnvLoop(
        env_wg=EnvGroup(),
        rollout_wg=RolloutGroup(),
        config=SimpleNamespace(pipeline_stage_num=2, max_interactions=1),
        switch_actor_rollout_mode=False,
    )
    # Ray collection is worker-major: [w0s0, w0s1, w1s0, w1s1].
    reset = DataProto.from_dict(
        tensors={
            "state": torch.tensor([[0.0], [10.0], [1.0], [11.0]]).repeat(1, 14),
        },
        non_tensors={
            "task": np.asarray(["w0s0", "w0s1", "w1s0", "w1s1"], dtype=object),
            "task_id": np.arange(4, dtype=np.int64),
        },
    )

    output, last_obs = loop.generate_sequences(_Ref(reset), active_stage_count=1)

    assert generated_states == [[0.0, 1.0]]
    assert len(output) == len(last_obs) == 2
    assert output.meta_info["metrics"]["count/active_pipeline_stages"] == 1.0


def test_env_loop_keeps_worker_local_grpo_groups_as_b8_actor_requests():
    actor_batch_sizes = []

    class RolloutGroup:
        world_size = 1

        def generate_sequences(self, obs):
            actor_batch_sizes.append(len(obs))
            return _Ref(_action(len(obs), 0.1, 1000))

    class EnvGroup:
        world_size = 2

        def env_interact_step(self, action, *, mode):
            assert mode == "train"
            assert len(action) == 16
            return _Ref(_env_step([True] * 16, step_seconds=0.2))

        def finish_rollout(self):
            pass

    loop = EnvLoop(
        env_wg=EnvGroup(),
        rollout_wg=RolloutGroup(),
        config=SimpleNamespace(
            pipeline_stage_num=1,
            max_interactions=1,
            rollout_partition_by_env_worker=True,
        ),
        switch_actor_rollout_mode=False,
    )
    reset = DataProto.from_dict(
        tensors={"state": torch.arange(16, dtype=torch.float32).reshape(16, 1).repeat(1, 14)},
        non_tensors={
            "task": np.asarray(["task"] * 16, dtype=object),
            "task_id": np.zeros(16, dtype=np.int64),
        },
    )

    output, last_obs = loop.generate_sequences(_Ref(reset))

    assert actor_batch_sizes == [8, 8]
    assert len(output) == len(last_obs) == 16


def test_env_loop_splits_one_vector24_simulator_into_three_b8_actor_requests():
    actor_batch_sizes = []

    class RolloutGroup:
        world_size = 1

        def generate_sequences(self, obs):
            actor_batch_sizes.append(len(obs))
            return _Ref(_action(len(obs), 0.1, 1000))

    class EnvGroup:
        world_size = 1

        def env_interact_step(self, action, *, mode):
            assert mode == "train"
            assert len(action) == 24
            return _Ref(_env_step([True] * 24, step_seconds=0.2))

        def finish_rollout(self):
            pass

    loop = EnvLoop(
        env_wg=EnvGroup(),
        rollout_wg=RolloutGroup(),
        config=SimpleNamespace(
            pipeline_stage_num=1,
            max_interactions=1,
            rollout_partition_by_env_worker=True,
            rollout_partition_size=8,
        ),
        switch_actor_rollout_mode=False,
    )
    reset = DataProto.from_dict(
        tensors={"state": torch.arange(24, dtype=torch.float32).reshape(24, 1).repeat(1, 14)},
        non_tensors={
            "task": np.asarray(["task"] * 24, dtype=object),
            "task_id": np.zeros(24, dtype=np.int64),
        },
    )

    output, last_obs = loop.generate_sequences(_Ref(reset))

    assert actor_batch_sizes == [8, 8, 8]
    assert len(output) == len(last_obs) == 24


def test_env_loop_rejects_invalid_active_stage_count():
    loop = EnvLoop(
        env_wg=SimpleNamespace(world_size=1),
        rollout_wg=SimpleNamespace(world_size=1),
        config=SimpleNamespace(pipeline_stage_num=2, max_interactions=1),
        switch_actor_rollout_mode=False,
    )
    with pytest.raises(ValueError, match="active_stage_count"):
        loop.generate_sequences(_Ref(DataProto()), active_stage_count=3)


def test_multistage_trajectory_collation_preserves_stage_major_order():
    loop = EnvLoop(
        env_wg=SimpleNamespace(world_size=2),
        rollout_wg=SimpleNamespace(world_size=1),
        config=SimpleNamespace(pipeline_stage_num=2, max_interactions=2),
        switch_actor_rollout_mode=False,
    )

    def step(stage: int, time_step: int):
        values = torch.tensor([[stage * 100 + time_step * 10], [stage * 100 + time_step * 10 + 1]])
        non_tensors = np.asarray([f"s{stage}l0", f"s{stage}l1"], dtype=object)
        return {
            "obs": DataProto.from_dict(tensors={"state": values}, non_tensors={"task": non_tensors}),
            "action": DataProto.from_dict(tensors={"action": values + 1000}),
            "next": DataProto.from_dict(tensors={"reward": values + 2000}),
        }

    output = loop._collate_trajectories(
        {0: [step(0, 0), step(0, 1)], 1: [step(1, 0), step(1, 1)]},
        meta_info={"source": "test"},
    )

    assert output.batch["obs.state"].squeeze(-1).tolist() == [
        [0, 10],
        [1, 11],
        [100, 110],
        [101, 111],
    ]
    assert output.non_tensor_batch["obs.task"].tolist() == [
        ["s0l0", "s0l0"],
        ["s0l1", "s0l1"],
        ["s1l0", "s1l0"],
        ["s1l1", "s1l1"],
    ]
    assert output.meta_info == {"source": "test"}
