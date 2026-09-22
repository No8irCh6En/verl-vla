# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from verl import DataProto
from verl.utils import tracking as tracking_module

from verl_vla.trainer.grfpo import grfpo_ray_trainer as trainer_module
from verl_vla.trainer.fpo import fpo_ray_trainer as fpo_trainer_module
from verl_vla.trainer.fpo.fpo_ray_trainer import FPORayTrainer
from verl_vla.trainer.grfpo.config import GRFPOTrainerConfig
from verl_vla.trainer.grfpo.grfpo_ray_trainer import (
    GRFPORayTrainer,
    _bounded_candidate_batch_size,
    _candidate_schedule_slot_after_fixed_eval,
    _quantized_candidate_batch_size,
    _validate_fastwam_rollout_temperature,
)


class FakeTracking:
    records: list[tuple[int, dict]] = []

    def __init__(self, **kwargs) -> None:
        del kwargs

    def log(self, *, data, step) -> None:
        self.records.append((step, dict(data)))


class FakeLifecycleCluster:
    def __init__(self) -> None:
        self.policy_version = 0
        self.events: list[str] = []

    def load_checkpoint(self):
        return None

    def rollout(self, *, async_rollout):
        assert async_rollout is False
        self.events.append(f"rollout_theta_{self.policy_version}")
        return DataProto(), DataProto(), {}, {}

    def train(self, data, *, async_update):
        del data
        assert async_update is False
        self.policy_version += 1
        self.events.append(f"train_to_theta_{self.policy_version}")
        return DataProto(meta_info={"metrics": {}})

    def eval(self, *, max_episodes):
        assert max_episodes == 1
        self.events.append(f"fresh_eval_theta_{self.policy_version}")
        return {"eval/policy_version": float(self.policy_version)}

    def save_checkpoint(self, global_step):
        self.events.append(f"save_theta_{self.policy_version}_step_{global_step}")


def test_candidate_batch_size_never_overshoots_budget_or_remaining_accepted_slots():
    assert _bounded_candidate_batch_size(
        configured_concurrency=2,
        remaining_candidate_budget=11,
        remaining_accepted_slots=8,
    ) == 2
    assert _bounded_candidate_batch_size(
        configured_concurrency=2,
        remaining_candidate_budget=1,
        remaining_accepted_slots=8,
    ) == 1


def test_worker_local_candidate_batch_uses_complete_all_worker_stages():
    assert _quantized_candidate_batch_size(
        configured_concurrency=4,
        remaining_candidate_budget=16,
        remaining_accepted_slots=8,
        groups_per_stage=2,
    ) == 4
    assert _quantized_candidate_batch_size(
        configured_concurrency=4,
        remaining_candidate_budget=16,
        remaining_accepted_slots=3,
        groups_per_stage=2,
    ) == 2
    assert _quantized_candidate_batch_size(
        configured_concurrency=4,
        remaining_candidate_budget=16,
        remaining_accepted_slots=1,
        groups_per_stage=2,
    ) == 2
    assert _quantized_candidate_batch_size(
        configured_concurrency=4,
        remaining_candidate_budget=1,
        remaining_accepted_slots=1,
        groups_per_stage=2,
    ) == 0


def test_fixed_eval_skips_the_discarded_prefetched_pipeline_wave():
    assert _candidate_schedule_slot_after_fixed_eval(76, 4) == 80
    with pytest.raises(ValueError, match="positive pipeline width"):
        _candidate_schedule_slot_after_fixed_eval(76, 0)
    assert _bounded_candidate_batch_size(
        configured_concurrency=2,
        remaining_candidate_budget=11,
        remaining_accepted_slots=1,
    ) == 1


def test_fastwam_rejects_generic_token_temperature_as_a_silent_noop():
    config = OmegaConf.create(
        {
            "cluster": {
                "actor_rollout_ref": {
                    "model": {"native_architecture": "fastwam"},
                    "rollout": {"temperature": 1.6},
                }
            }
        }
    )
    with pytest.raises(ValueError, match="does not consume"):
        _validate_fastwam_rollout_temperature(config)


def test_official_fpo_trainer_evaluates_fresh_theta1_after_actor_update(monkeypatch):
    FakeTracking.records = []
    monkeypatch.setattr(tracking_module, "Tracking", FakeTracking)
    monkeypatch.setattr(fpo_trainer_module, "should_save_ckpt_esi", lambda **kwargs: False)

    cluster = FakeLifecycleCluster()
    trainer = object.__new__(FPORayTrainer)
    trainer.cluster = cluster
    trainer.config = OmegaConf.create({})
    trainer.trainer_config = SimpleNamespace(
        project_name="test",
        experiment_name="fpo-lifecycle",
        logger=["console"],
        eval_episodes=1,
        val_before_train=False,
        val_only=False,
        total_training_steps=2,
        async_rollout=False,
        test_freq=1,
        save_freq=1,
        esi_redundant_time=0,
    )
    trainer._prepare_actor_input = lambda rollout, end_obs: DataProto()

    trainer.fit()

    assert cluster.events == [
        "rollout_theta_0",
        "train_to_theta_1",
        "fresh_eval_theta_1",
        "save_theta_1_step_1",
        "rollout_theta_1",
        "train_to_theta_2",
        "fresh_eval_theta_2",
        "save_theta_2_step_2",
    ]
    assert FakeTracking.records[-1][0] == 2
    assert FakeTracking.records[-1][1]["eval/policy_version"] == 2.0


def test_group_relative_val_only_evaluates_loaded_policy_without_training(monkeypatch):
    FakeTracking.records = []
    monkeypatch.setattr(tracking_module, "Tracking", FakeTracking)

    cluster = FakeLifecycleCluster()
    cluster.policy_version = 4
    cluster.load_checkpoint = lambda: (4, "/tmp/checkpoints/global_step_4")
    trainer = object.__new__(GRFPORayTrainer)
    trainer.cluster = cluster
    trainer.config = OmegaConf.create({})
    trainer.trainer_config = GRFPOTrainerConfig(
        project_name="test",
        experiment_name="grpo-eval",
        logger=["console"],
        eval_episodes=1,
        val_before_train=False,
        val_only=True,
        evidence_jsonl=None,
    )
    trainer._fit_group_relative = lambda logger: (_ for _ in ()).throw(
        AssertionError("val_only must not enter the Group-Relative FPO training loop")
    )

    trainer.fit()

    assert cluster.events == ["fresh_eval_theta_4"]
    assert FakeTracking.records == [(4, {"eval/policy_version": 4.0})]


def test_multi_group_collection_finishes_before_one_actor_update():
    events = []

    class Cluster:
        def begin_rollout_collection(self):
            events.append(("begin_collection", 0))
            return {"timing_s/collection_weight_sync": 0.0}

        def end_rollout_collection(self):
            events.append(("end_collection", 0))
            return {"timing_s/collection_switch_to_train": 0.0}

        def train(self, data, *, async_update):
            assert async_update is False
            events.append(("train", len(data)))
            return DataProto(meta_info={"metrics": {"actor/loss": [0.0]}})

        def save_checkpoint(self, global_step):
            events.append(("save", global_step))

    trainer = object.__new__(GRFPORayTrainer)
    trainer.cluster = Cluster()
    trainer.global_steps = 0
    trainer.candidate_group_count = 0
    trainer.run_id = "test"
    trainer.last_trajectory_audit = {}
    trainer.trainer_config = GRFPOTrainerConfig(
        total_training_steps=1,
        group_size=8,
        accepted_groups_per_update=4,
        min_accepted_groups_for_partial_update=4,
        max_candidate_groups_per_update=4,
        concurrent_candidate_groups=2,
        initial_candidate_groups_per_update=0,
        accepted_groups_before_optional_refill=0,
        evidence_jsonl=None,
        candidate_evidence_jsonl=None,
        collection_only=False,
        save_freq=0,
        test_freq=0,
        val_before_train=False,
    )
    trainer.config = OmegaConf.create(
        {
            "cluster": {
                "resource": {
                    "env": {
                        "device": "cuda",
                        "gpus_per_node": 1,
                        "workers_per_node": 1,
                        "nnodes": 1,
                    }
                },
                "env": {
                    "env_loop": {"pipeline_stage_num": 2},
                    "env_worker": {
                        "num_envs": 2,
                        "simulator": {
                            "robodojo": {
                                "task_name": "stack_bowls",
                                "task_id": 0,
                                "task_names": ["stack_bowls"],
                                "worker_stage_task_schedule": [],
                                "train_layout_schedule": [0, 1, 2, 3, 4, 5],
                            }
                        },
                    }
                },
                    "actor_rollout_ref": {
                        "model": {"path": "/immutable", "adapter": {"checkpoint_sha256": "sha"}},
                        "actor": {"micro_batch_size": 1, "update_epochs": 1},
                    },
            }
        }
    )

    def collect_one(*, intended_update, candidate_attempt):
        del intended_update
        rollout = DataProto.from_dict(
            tensors={
                "x": torch.zeros(8, 1),
                "action.action": torch.zeros(8, 1, 24, 14),
            }
        )
        end_obs = DataProto.from_dict(tensors={"x": torch.zeros(8, 1)})
        group_id = f"g{candidate_attempt}"
        record = {
            "group_id": group_id,
            "group_key": {
                "task_id": 0,
                "normalized_instruction": "task",
                "layout_id": candidate_attempt - 1,
                "environment_seed": candidate_attempt - 1,
            },
            "rollout_policy_version": trainer.global_steps,
            "policy_seeds": list(range(candidate_attempt * 8, candidate_attempt * 8 + 8)),
            "reward_vector": [1] + [0] * 7,
            "successes": 1,
            "mixed": True,
            "informative": True,
            "informative_reason": "mixed_full_success",
            "trajectories": [{"lane": lane} for lane in range(8)],
            "rollout_metrics": {},
        }
        return rollout, end_obs, record

    def collect_batch(*, intended_update, first_candidate_attempt, group_count):
        events.append(
            ("collect_batch", first_candidate_attempt, group_count, trainer.global_steps)
        )
        return [
            collect_one(
                intended_update=intended_update,
                candidate_attempt=first_candidate_attempt + offset,
            )
            for offset in range(group_count)
        ]

    def prepare(rollout, end_obs, *, group_records):
        assert len(rollout) == len(end_obs) == 32
        assert len(group_records) == 4
        trainer.last_group_diagnostics = {
            "trajectory_advantages": [0.0] * 32,
            "groups": [
                {"advantage_vector": [1.0] + [-1.0 / 7] * 7} for _ in group_records
            ],
            "group/advantage_mean": 0.0,
        }
        return DataProto.from_dict(
            tensors={
                "x": torch.zeros(32, 1),
                "info.valids": torch.ones(32, 1),
            }
        )

    trainer._collect_candidate_groups = collect_batch
    trainer._prepare_actor_input = prepare

    class Logger:
        def log(self, *, data, step):
            del data, step

    trainer._fit_group_relative(Logger())

    assert events[:3] == [
        ("begin_collection", 0),
        ("collect_batch", 1, 2, 0),
        ("collect_batch", 3, 2, 0),
    ]
    assert events[3:] == [("end_collection", 0), ("train", 32)]
    assert trainer.global_steps == 1


def test_concurrent_candidate_collection_splits_stage_major_complete_groups():
    batch_size = 16
    success = torch.zeros(batch_size, 1, 1, dtype=torch.bool)
    success[0] = True
    success[8] = True
    rollout = DataProto.from_dict(
        tensors={
            "next.success": success,
            "next.terminated": success.clone(),
            "next.truncated": torch.zeros_like(success),
            "next.reward": success.float(),
        },
        non_tensors={
            "obs.task_id": np.zeros((batch_size, 1), dtype=np.int64),
            "obs.suite_id": np.zeros((batch_size, 1), dtype=np.int64),
            "obs.task": np.full((batch_size, 1), "stack bowls", dtype=object),
            "obs.layout_id": np.asarray([[0]] * 8 + [[1]] * 8, dtype=np.int64),
            "obs.environment_seed": np.asarray([[0]] * 8 + [[1]] * 8, dtype=np.int64),
            "obs.policy_seed": np.arange(batch_size, dtype=np.int64).reshape(batch_size, 1),
            "obs.eval_episode_id": np.arange(batch_size, dtype=np.int64).reshape(batch_size, 1),
        },
    )
    end_obs = DataProto.from_dict(tensors={"state": torch.zeros(batch_size, 14)})
    rollout_calls: list[tuple[bool, int]] = []

    class Cluster:
        def rollout(self, *, async_rollout, active_pipeline_stages):
            rollout_calls.append((async_rollout, active_pipeline_stages))
            active_trajectories = active_pipeline_stages * 8
            return (
                rollout[:active_trajectories],
                end_obs[:active_trajectories],
                {},
                {"count/policy_lane_slots": float(active_trajectories)},
            )

    trainer = object.__new__(GRFPORayTrainer)
    trainer.cluster = Cluster()
    trainer.trainer_config = GRFPOTrainerConfig()
    trainer.config = OmegaConf.create(
        {"cluster": {"env": {"env_loop": {"pipeline_stage_num": 2}}}}
    )
    trainer.global_steps = 3
    trainer.candidate_group_count = 0
    trainer.run_id = "test"

    groups = trainer._collect_candidate_groups(
        intended_update=4,
        first_candidate_attempt=1,
        group_count=2,
    )

    assert rollout_calls == [(False, 2)]
    assert [len(group_rollout) for group_rollout, _end, _record in groups] == [8, 8]
    assert [record["group_key"]["layout_id"] for _rollout, _end, record in groups] == [0, 1]
    assert {record["rollout_policy_version"] for _rollout, _end, record in groups} == {3}
    assert all(record["concurrent_candidate_groups"] == 2 for _rollout, _end, record in groups)
    assert all(record["rollout_action_latent_scale"] == 1.0 for _rollout, _end, record in groups)
    assert all(record["rollout_seed_mode"] == "episode" for _rollout, _end, record in groups)
    assert all(record["next_policy_seed_cursor"] == 16 for _rollout, _end, record in groups)

    final_group = trainer._collect_candidate_groups(
        intended_update=4,
        first_candidate_attempt=3,
        group_count=1,
    )
    final_record = final_group[0][2]
    assert final_record["unused_pipeline_stage_count"] == 1
    assert final_record["next_policy_seed_cursor"] == 16


def test_worker_local_candidate_collection_splits_each_stage_by_env_worker():
    group_size = 8
    group_count = 4
    batch_size = group_size * group_count
    success = torch.zeros(batch_size, 1, 1, dtype=torch.bool)
    success[::group_size] = True
    rollout = DataProto.from_dict(
        tensors={
            "next.success": success,
            "next.terminated": success.clone(),
            "next.truncated": torch.zeros_like(success),
            "next.reward": success.float(),
        },
        non_tensors={
            "obs.task_id": np.zeros((batch_size, 1), dtype=np.int64),
            "obs.suite_id": np.zeros((batch_size, 1), dtype=np.int64),
            "obs.task": np.full((batch_size, 1), "stack bowls", dtype=object),
            "obs.layout_id": np.repeat(np.arange(group_count), group_size).reshape(batch_size, 1),
            "obs.environment_seed": np.repeat(np.arange(group_count), group_size).reshape(
                batch_size, 1
            ),
            "obs.policy_seed": np.arange(batch_size, dtype=np.int64).reshape(batch_size, 1),
            "obs.eval_episode_id": np.arange(batch_size, dtype=np.int64).reshape(batch_size, 1),
        },
    )
    end_obs = DataProto.from_dict(tensors={"state": torch.zeros(batch_size, 14)})
    rollout_calls = []

    class Cluster:
        def rollout(self, *, async_rollout, active_pipeline_stages):
            rollout_calls.append((async_rollout, active_pipeline_stages))
            active_trajectories = active_pipeline_stages * 2 * group_size
            return (
                rollout[:active_trajectories],
                end_obs[:active_trajectories],
                {},
                {"count/policy_lane_slots": float(active_trajectories)},
            )

    trainer = object.__new__(GRFPORayTrainer)
    trainer.cluster = Cluster()
    trainer.trainer_config = GRFPOTrainerConfig(candidate_group_partition="worker_stage")
    trainer.config = OmegaConf.create(
        {
            "cluster": {
                "resource": {
                    "env": {
                        "device": "cuda",
                        "gpus_per_node": 2,
                        "workers_per_node": 1,
                        "nnodes": 1,
                    }
                },
                "env": {
                    "env_loop": {"pipeline_stage_num": 2},
                    "env_worker": {"num_envs": group_size},
                },
            }
        }
    )
    trainer.global_steps = 4
    trainer.candidate_group_count = 0
    trainer.run_id = "worker-local"

    groups = trainer._collect_candidate_groups(
        intended_update=5,
        first_candidate_attempt=1,
        group_count=4,
    )

    assert rollout_calls == [(False, 2)]
    assert [len(group_rollout) for group_rollout, _end, _record in groups] == [8] * 4
    assert [record["group_key"]["layout_id"] for _rollout, _end, record in groups] == [0, 1, 2, 3]
    assert all(record["candidate_group_partition"] == "worker_stage" for _r, _e, record in groups)
    assert all(record["groups_per_pipeline_stage"] == 2 for _r, _e, record in groups)
    assert all(record["next_policy_seed_cursor"] == 32 for _r, _e, record in groups)


def test_worker_local_candidate_collection_splits_one_vector24_into_three_groups():
    group_size = 8
    group_count = 3
    batch_size = group_size * group_count
    success = torch.zeros(batch_size, 1, 1, dtype=torch.bool)
    success[::group_size] = True
    rollout = DataProto.from_dict(
        tensors={
            "next.success": success,
            "next.terminated": success.clone(),
            "next.truncated": torch.zeros_like(success),
            "next.reward": success.float(),
        },
        non_tensors={
            "obs.task_id": np.zeros((batch_size, 1), dtype=np.int64),
            "obs.suite_id": np.zeros((batch_size, 1), dtype=np.int64),
            "obs.task": np.full((batch_size, 1), "stack bowls", dtype=object),
            "obs.layout_id": np.repeat(np.arange(group_count), group_size).reshape(batch_size, 1),
            "obs.environment_seed": np.repeat(np.arange(group_count), group_size).reshape(
                batch_size, 1
            ),
            "obs.policy_seed": np.arange(batch_size, dtype=np.int64).reshape(batch_size, 1),
            "obs.eval_episode_id": np.arange(batch_size, dtype=np.int64).reshape(batch_size, 1),
        },
    )
    end_obs = DataProto.from_dict(tensors={"state": torch.zeros(batch_size, 14)})
    rollout_calls = []

    class Cluster:
        def rollout(self, *, async_rollout, active_pipeline_stages):
            rollout_calls.append((async_rollout, active_pipeline_stages))
            assert active_pipeline_stages == 1
            return (
                rollout,
                end_obs,
                {},
                {"count/policy_lane_slots": float(batch_size)},
            )

    trainer = object.__new__(GRFPORayTrainer)
    trainer.cluster = Cluster()
    trainer.trainer_config = GRFPOTrainerConfig(candidate_group_partition="worker_stage")
    trainer.config = OmegaConf.create(
        {
            "cluster": {
                "resource": {
                    "env": {
                        "device": "cuda",
                        "gpus_per_node": 1,
                        "workers_per_node": 1,
                        "nnodes": 1,
                    }
                },
                "env": {
                    "env_loop": {"pipeline_stage_num": 1},
                    "env_worker": {"num_envs": batch_size},
                },
            }
        }
    )
    trainer.global_steps = 4
    trainer.candidate_group_count = 0
    trainer.run_id = "vector24"

    groups = trainer._collect_candidate_groups(
        intended_update=5,
        first_candidate_attempt=1,
        group_count=3,
    )

    assert rollout_calls == [(False, 1)]
    assert [len(group_rollout) for group_rollout, _end, _record in groups] == [8, 8, 8]
    assert [record["group_key"]["layout_id"] for _rollout, _end, record in groups] == [0, 1, 2]
    assert all(record["groups_per_pipeline_stage"] == 3 for _r, _e, record in groups)
    assert all(record["next_policy_seed_cursor"] == 24 for _r, _e, record in groups)
