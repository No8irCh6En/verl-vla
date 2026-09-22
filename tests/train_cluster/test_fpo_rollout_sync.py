# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from verl import DataProto

from verl_vla.train_cluster.cluster import RolloutState, TrainCluster
from verl_vla.workers.env.env_worker import _restart_simulators_with_parallel_start


def test_restart_stops_all_simulators_then_starts_stages_concurrently():
    import threading

    events: list[str] = []
    event_lock = threading.Lock()
    start_barrier = threading.Barrier(3, timeout=2.0)

    class Simulator:
        def __init__(self, simulator_id: int):
            self.simulator_id = simulator_id

        def stop_simulator(self):
            with event_lock:
                events.append(f"stop-{self.simulator_id}")

        def start_simulator(self):
            start_barrier.wait()
            with event_lock:
                events.append(f"start-{self.simulator_id}")

    _restart_simulators_with_parallel_start([Simulator(0), Simulator(1), Simulator(2)])

    assert events[:3] == ["stop-0", "stop-1", "stop-2"]
    assert set(events[3:]) == {"start-0", "start-1", "start-2"}


def test_synchronous_separate_rollout_syncs_weights_before_collection():
    events: list[str] = []
    cluster = object.__new__(TrainCluster)
    cluster.cluster_type = "env_loop"
    cluster.env_loop = object()
    cluster.config = SimpleNamespace(
        resource=SimpleNamespace(separate_rollout_model=SimpleNamespace(enabled=True))
    )
    cluster.rollout_state = RolloutState()
    cluster._rollout_collection_active = False
    cluster.update_weights = lambda: events.append("update_weights")

    def fake_rollout_once(env_loop, *, config, state, active_pipeline_stages=None):
        del env_loop, config
        events.append(f"collect_fresh_rollout_stages_{active_pipeline_stages}")
        return DataProto(), DataProto(), {}, {}, state

    cluster._rollout_once = fake_rollout_once
    cluster.rollout(async_rollout=False, active_pipeline_stages=2)

    assert events == ["update_weights", "collect_fresh_rollout_stages_2"]


def test_collection_window_syncs_and_switches_once_for_multiple_rollouts():
    events: list[str] = []

    class EnvLoop:
        def begin_rollout_collection(self):
            events.append("switch_to_rollout")
            return {"timing_s/collection_switch_to_rollout": 1.0}

        def end_rollout_collection(self):
            events.append("switch_to_train")
            return {"timing_s/collection_switch_to_train": 2.0}

    cluster = object.__new__(TrainCluster)
    cluster.cluster_type = "env_loop"
    cluster.env_loop = EnvLoop()
    cluster.config = SimpleNamespace(
        resource=SimpleNamespace(separate_rollout_model=SimpleNamespace(enabled=True))
    )
    cluster.rollout_state = RolloutState()
    cluster._pending_rollout_ref = None
    cluster._rollout_collection_active = False
    cluster.update_weights = lambda: events.append("update_weights")

    def fake_rollout_once(env_loop, *, config, state, active_pipeline_stages=None):
        del env_loop, config, active_pipeline_stages
        events.append("rollout")
        return DataProto(), DataProto(), {}, {}, state

    cluster._rollout_once = fake_rollout_once

    metrics = cluster.begin_rollout_collection()
    cluster.rollout(async_rollout=False, active_pipeline_stages=4)
    cluster.rollout(async_rollout=False, active_pipeline_stages=4)
    metrics.update(cluster.end_rollout_collection())

    assert events == [
        "update_weights",
        "switch_to_rollout",
        "rollout",
        "rollout",
        "switch_to_train",
    ]
    assert metrics == {
        "timing_s/collection_weight_sync": pytest.approx(0.0, abs=0.1),
        "timing_s/collection_switch_to_rollout": 1.0,
        "timing_s/collection_switch_to_train": 2.0,
    }


def test_rollout_once_forwards_active_pipeline_stage_count(monkeypatch):
    seen: list[int] = []

    class EnvWorkerGroup:
        def reset_env(self):
            return "reset"

        def pop_lerobot_dataset(self):
            return []

    env_loop = SimpleNamespace(
        env_wg=EnvWorkerGroup(),
        generate_sequences=lambda reset_future, *, active_stage_count: (
            seen.append(active_stage_count) or DataProto(meta_info={"metrics": {}}),
            DataProto(),
        ),
    )
    config = SimpleNamespace(
        env=SimpleNamespace(
            env_worker=SimpleNamespace(
                auto_reset=False,
                recorder=SimpleNamespace(enable=False),
                simulator=SimpleNamespace(simulator_type="dummy"),
            )
        )
    )
    monkeypatch.setattr(TrainCluster, "_collect_trajectory_records", staticmethod(lambda *args, **kwargs: []))

    TrainCluster._rollout_once(
        env_loop,
        config=config,
        state=RolloutState(),
        active_pipeline_stages=2,
    )

    assert seen == [2]


def test_rollout_restarts_robodojo_before_prefetch_when_configured(monkeypatch):
    events: list[str] = []
    output = DataProto(meta_info={"metrics": {}})

    class EnvWorkerGroup:
        def reset_env(self):
            events.append("reset")
            return f"reset-{events.count('reset')}"

        def restart_simulators(self, *, mode):
            events.append(f"restart-{mode}")

        def pop_lerobot_dataset(self):
            return []

    env_wg = EnvWorkerGroup()
    env_loop = SimpleNamespace(
        env_wg=env_wg,
        generate_sequences=lambda reset_future: (
            events.append(f"generate-{reset_future}") or output,
            DataProto(),
        ),
    )
    config = SimpleNamespace(
        env=SimpleNamespace(
            env_worker=SimpleNamespace(
                auto_reset=False,
                recorder=SimpleNamespace(enable=False),
                simulator=SimpleNamespace(
                    simulator_type="robodojo",
                    robodojo=SimpleNamespace(restart_between_rollouts=True),
                ),
            )
        )
    )
    state = RolloutState()
    monkeypatch.setattr(TrainCluster, "_collect_trajectory_records", staticmethod(lambda *args, **kwargs: []))

    _output, _last_obs, _datasets, metrics, state = TrainCluster._rollout_once(
        env_loop,
        config=config,
        state=state,
    )

    assert events == ["reset", "generate-reset-1", "restart-train", "reset"]
    assert state.reset_future == "reset-2"
    assert metrics["timing_s/env_simulator_restart"] >= 0.0
    assert metrics["count/env_simulator_restarts"] == 1.0


def test_rollout_restarts_robodojo_once_after_two_physical_waves(monkeypatch):
    events: list[str] = []

    class EnvWorkerGroup:
        def reset_env(self):
            events.append("reset")
            return f"reset-{events.count('reset')}"

        def restart_simulators(self, *, mode):
            events.append(f"restart-{mode}")

        def pop_lerobot_dataset(self):
            return []

    env_wg = EnvWorkerGroup()

    def generate_sequences(reset_future):
        events.append(f"generate-{reset_future}")
        return DataProto(meta_info={"metrics": {}}), DataProto()

    env_loop = SimpleNamespace(env_wg=env_wg, generate_sequences=generate_sequences)
    config = SimpleNamespace(
        env=SimpleNamespace(
            env_worker=SimpleNamespace(
                auto_reset=False,
                recorder=SimpleNamespace(enable=False),
                simulator=SimpleNamespace(
                    simulator_type="robodojo",
                    robodojo=SimpleNamespace(
                        restart_between_rollouts=True,
                        restart_every_rollouts=2,
                    ),
                ),
            )
        )
    )
    state = RolloutState()
    monkeypatch.setattr(TrainCluster, "_collect_trajectory_records", staticmethod(lambda *args, **kwargs: []))

    *_unused, first_metrics, state = TrainCluster._rollout_once(env_loop, config=config, state=state)
    *_unused, second_metrics, state = TrainCluster._rollout_once(env_loop, config=config, state=state)

    assert events == [
        "reset",
        "generate-reset-1",
        "reset",
        "generate-reset-2",
        "restart-train",
        "reset",
    ]
    assert first_metrics["count/env_simulator_restarts"] == 0.0
    assert second_metrics["count/env_simulator_restarts"] == 1.0
    assert state.rollouts_since_simulator_restart == 0


def test_fixed_observation_actions_use_rollout_mode_lifecycle():
    events: list[str] = []
    expected = DataProto.from_dict(tensors={"full_action": torch.ones(1, 32, 14)})

    class Ref:
        def get(self):
            events.append("get")
            return expected

    class RolloutGroup:
        def switch_to_rollout(self):
            events.append("switch_to_rollout")

        def generate_sequences(self, observations):
            events.append(f"generate_eval_{observations.meta_info['eval']}")
            return Ref()

        def switch_to_train(self):
            events.append("switch_to_train")

    cluster = object.__new__(TrainCluster)
    cluster.cluster_type = "env_loop"
    cluster.env_loop = SimpleNamespace(rollout_wg=RolloutGroup(), switch_actor_rollout_mode=True)

    result = cluster.generate_actions(DataProto.from_dict(tensors={"state": torch.zeros(1, 14)}), eval=True)

    assert result is expected
    assert events == ["switch_to_rollout", "generate_eval_True", "get", "switch_to_train"]


def test_eval_refreshes_prefetched_training_reset_after_shared_simulator_mutation():
    reset_calls: list[tuple[str, bool]] = []

    class FakeEnvWorkerGroup:
        def get_eval_benchmark_size(self):
            return [1]

        def reset_env(self, *, mode="train", reset_eval=False):
            reset_calls.append((mode, reset_eval))
            return f"{mode}-reset-{len(reset_calls)}"

    terminal_eval_output = DataProto.from_dict(
        tensors={
            "next.terminated": torch.ones(1, 1, 1, dtype=torch.bool),
            "next.truncated": torch.zeros(1, 1, 1, dtype=torch.bool),
            "next.success": torch.zeros(1, 1, 1, dtype=torch.bool),
            "next.reward": torch.zeros(1, 1, 1),
        },
        non_tensors={
            "obs.task_id": np.zeros((1, 1), dtype=np.int64),
            "obs.eval_episode_id": np.zeros((1, 1), dtype=np.int64),
        },
        meta_info={"metrics": {}},
    )

    cluster = object.__new__(TrainCluster)
    cluster.cluster_type = "env_loop"
    cluster.config = SimpleNamespace(env=SimpleNamespace(env_worker=SimpleNamespace(auto_reset=False)))
    cluster.worker_groups = {"env": FakeEnvWorkerGroup()}
    cluster.env_loop = SimpleNamespace(
        generate_sequences=lambda reset_future, eval: (terminal_eval_output, DataProto())
    )
    cluster.rollout_state = RolloutState(
        reset_future="stale-prefetched-train-reset",
        carry_state={"length": np.array([17]), "reward": np.array([3.0])},
    )
    cluster._pending_rollout_ref = None
    cluster._ready_rollout_result = None
    cluster._rollout_collection_active = False
    cluster.update_weights = lambda: None

    metrics = cluster.eval(max_episodes=1)

    assert metrics["val/trajectory_count"] == 1.0
    assert reset_calls == [("eval", True), ("train", False)]
    assert cluster.rollout_state.reset_future == "train-reset-2"
    assert cluster.rollout_state.carry_state == {"length": None, "reward": None}
