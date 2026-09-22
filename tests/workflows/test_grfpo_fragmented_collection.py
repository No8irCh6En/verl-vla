import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf
from verl import DataProto

from verl_vla.trainer.grfpo.collection_store import (
    CandidateAssignment,
    CollectionWindowSpec,
    CollectionWindowStore,
    PolicyIdentity,
)
from verl_vla.workflows.grfpo_fragmented_collection import (
    _candidate_physical_batch_size,
    _collector_assignments,
    _configured_task_names,
    _construct_robodojo_env_with_node_startup_lock,
    collate_candidate_trajectory,
    collection_window_spec,
    concatenate_rollout_groups_with_padding,
)


def _obs(offset: int) -> DataProto:
    return DataProto.from_dict(
        tensors={"observation.state": torch.full((8, 14), offset, dtype=torch.float32)},
        non_tensors={
            "task": np.asarray(["stack bowls"] * 8, dtype=object),
            "task_id": np.zeros(8, dtype=np.int64),
            "suite_id": np.zeros(8, dtype=np.int64),
            "layout_id": np.ones(8, dtype=np.int64),
            "environment_seed": np.ones(8, dtype=np.int64),
            "policy_seed": np.arange(8, dtype=np.int64),
        },
    )


def test_candidate_collation_matches_env_loop_tensor_contract():
    slots = []
    for step in range(3):
        action = DataProto.from_dict(
            tensors={
                "action": torch.zeros(8, 24, 14),
                "full_action": torch.zeros(8, 32, 14),
            }
        )
        feedback = DataProto.from_dict(
            tensors={
                "reward": torch.zeros(8, 24),
                "terminated": torch.zeros(8, 24, dtype=torch.bool),
                "truncated": torch.zeros(8, 24, dtype=torch.bool),
                "success": torch.zeros(8, 24, dtype=torch.bool),
            }
        )
        slots.append((_obs(step), action, feedback))
    rollout, end_obs = collate_candidate_trajectory(slots, _obs(3))
    assert rollout.batch["obs.observation.state"].shape == (8, 3, 14)
    assert rollout.batch["action.action"].shape == (8, 3, 24, 14)
    assert rollout.batch["action.full_action"].shape == (8, 3, 32, 14)
    assert rollout.batch["next.success"].shape == (8, 3, 24)
    assert end_obs.batch["observation.state"].shape == (8, 14)


def test_persistent_collector_lanes_cover_budget_once_and_never_change_task_or_suite(tmp_path):
    tasks = ["a", "b", "c", "d", "e"]
    manifest = {
        "tasks": {
            task: {
                "suites": {
                    str(suite): {
                        "train_layout_ids": [0, 1, 2, 3, 4, 5, 10],
                        "heldout_test": [{"layout_id": 6}],
                    }
                    for suite in (0, 1, 2)
                }
            }
            for task in tasks
        }
    }
    manifest_path = tmp_path / "conditions.json"
    manifest_bytes = json.dumps(manifest, sort_keys=True).encode()
    manifest_path.write_bytes(manifest_bytes)
    spec = CollectionWindowSpec(
        run_id="run",
        intended_update=1,
        group_size=8,
        target_accepted_groups=16,
        minimum_accepted_groups=8,
        max_candidate_groups=23,
        policy=PolicyIdentity("/base", "a" * 64, None, 0),
    )
    all_indices = []
    for slot in range(15):
        config = OmegaConf.create(
            {
                "task_names": tasks,
                "suite_ids": [0, 1, 2],
                "collector_count": 15,
                "collector_slot": slot,
                "condition_manifest": str(manifest_path),
                "condition_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                "run_seed": 7865,
            }
        )
        assignments = _collector_assignments(config, spec)
        assert {(assignment.task_name, assignment.suite_id) for assignment in assignments} == {
            (tasks[slot // 3], slot % 3)
        }
        all_indices.extend(assignment.candidate_index for assignment in assignments)
    assert sorted(all_indices) == list(range(23))


def test_excluded_task_rebalances_full_candidate_budget_over_remaining_conditions(tmp_path):
    tasks = ["a", "b", "c", "d", "e"]
    manifest = {
        "tasks": {
            task: {
                "suites": {
                    str(suite): {
                        "train_layout_ids": list(range(12)),
                        "heldout_test": [{"layout_id": 20}],
                    }
                    for suite in (0, 1, 2)
                }
            }
            for task in tasks
        }
    }
    manifest_path = tmp_path / "conditions.json"
    manifest_bytes = json.dumps(manifest, sort_keys=True).encode()
    manifest_path.write_bytes(manifest_bytes)
    spec = CollectionWindowSpec(
        run_id="run",
        intended_update=1,
        group_size=8,
        target_accepted_groups=32,
        minimum_accepted_groups=16,
        max_candidate_groups=96,
        policy=PolicyIdentity("/base", "a" * 64, None, 0),
    )
    counts = {}
    all_indices = []
    for slot in range(12):
        config = OmegaConf.create(
            {
                "task_names": tasks,
                "included_task_names": "",
                "excluded_task_names": "e",
                "suite_ids": [0, 1, 2],
                "collector_count": 12,
                "collector_slot": slot,
                "condition_manifest": str(manifest_path),
                "condition_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                "run_seed": 7865,
            }
        )
        assert _configured_task_names(config) == tasks[:-1]
        assignments = _collector_assignments(config, spec)
        condition = (tasks[slot // 3], slot % 3)
        assert {(assignment.task_name, assignment.suite_id) for assignment in assignments} == {
            condition
        }
        counts[condition] = len(assignments)
        all_indices.extend(assignment.candidate_index for assignment in assignments)
    assert sorted(all_indices) == list(range(96))
    assert set(counts.values()) == {8}


def test_manifest_include_and_original_exclusion_are_idempotent():
    config = OmegaConf.create(
        {
            "task_names": ["a", "b", "c", "d", "e"],
            "included_task_names": "a:b:c:d",
            "excluded_task_names": "e",
        }
    )
    assert _configured_task_names(config) == ["a", "b", "c", "d"]


def test_spooled_training_config_reconstructs_exact_manifest_condition_identity():
    tasks = [
        "stack_bowls",
        "fold_clothes",
        "put_bottles_into_dustbin",
        "match_and_pick_from_conveyor",
    ]
    digest = "a" * 64
    with initialize_config_module(config_module="verl_vla.workflows.config", version_base=None):
        config = compose(
            config_name="train/grfpo_spooled",
            overrides=[
                f"collection.task_names=[{','.join(tasks)}]",
                "collection.suite_ids=[0,1,2]",
                "collection.collector_count=12",
                f"collection.condition_manifest_sha256={digest}",
                "collection.model_root=/immutable/base",
                "collection.checkpoint_sha256=base-sha",
            ],
        )

    spec = collection_window_spec(config.collection)
    assert spec.task_names == tuple(tasks)
    assert spec.suite_ids == (0, 1, 2)
    assert spec.condition_manifest_sha256 == digest
    assert spec.candidate_round_size == 0


def test_node_startup_lock_serializes_only_env_construction(tmp_path, monkeypatch):
    monkeypatch.setenv("GRFPO_ISAAC_STARTUP_LOCK", str(tmp_path / "isaac.lock"))
    state_lock = threading.Lock()
    active = 0
    max_active = 0

    class FakeEnv:
        def __init__(self, _config, *, rank, world_size, only_eval):
            nonlocal active, max_active
            assert (rank, world_size, only_eval) == (0, 1, True)
            with state_lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.05)
            with state_lock:
                active -= 1

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(_construct_robodojo_env_with_node_startup_lock, FakeEnv, {"device": "cuda"})
            for _ in range(2)
        ]
        for future in futures:
            assert isinstance(future.result(), FakeEnv)
    assert max_active == 1


def test_candidate_physical_batch_override_preserves_default_and_requires_group_divisor(tmp_path):
    spec = CollectionWindowSpec(
        run_id="run",
        intended_update=8,
        group_size=8,
        target_accepted_groups=4,
        minimum_accepted_groups=4,
        max_candidate_groups=8,
        policy=PolicyIdentity("/base", "a" * 64, None, 7),
    )
    assignment = CandidateAssignment(3, "task", 0, 2, 1, tuple(range(8)))
    config = OmegaConf.create({"collection_root": str(tmp_path), "physical_rollout_batch_size": 0})
    assert _candidate_physical_batch_size(config, spec, assignment) == 8

    window = CollectionWindowStore(tmp_path, spec).window_dir
    window.mkdir(parents=True)
    (window / "PHYSICAL_BATCH_OVERRIDES.json").write_text(
        json.dumps({"schema_version": 1, "candidates": {"candidate_0003": 2}}),
        encoding="utf-8",
    )
    assert _candidate_physical_batch_size(config, spec, assignment) == 2

    (window / "PHYSICAL_BATCH_OVERRIDES.json").write_text(
        json.dumps({"schema_version": 1, "candidates": {"candidate_0003": 3}}),
        encoding="utf-8",
    )
    with np.testing.assert_raises_regex(ValueError, "positive divisor"):
        _candidate_physical_batch_size(config, spec, assignment)


def test_physical_candidate_waves_are_padded_and_concatenated_without_reweighting():
    def wave(batch_size: int, steps: int, marker: float):
        rollout = DataProto.from_dict(
            tensors={
                "obs.observation.state": torch.full((batch_size, steps, 14), marker),
                "action.action": torch.full((batch_size, steps, 24, 14), marker),
                "next.success": torch.zeros(batch_size, steps, 24, dtype=torch.bool),
            },
            non_tensors={"obs.task": np.full((batch_size, steps), f"task-{marker}", dtype=object)},
        )
        end_obs = DataProto.from_dict(
            tensors={"observation.state": torch.full((batch_size, 14), marker + 10)},
            non_tensors={"task": np.full(batch_size, f"task-{marker}", dtype=object)},
        )
        return rollout, end_obs

    first, first_end = wave(2, 3, 1.0)
    second, second_end = wave(2, 1, 2.0)
    combined, combined_end = concatenate_rollout_groups_with_padding(
        [first, second], [first_end, second_end]
    )
    assert len(combined) == len(combined_end) == 4
    assert combined.batch["action.action"].shape == (4, 3, 24, 14)
    assert torch.count_nonzero(combined.batch["action.action"][2:, 1:]) == 0
    assert torch.all(combined.batch["obs.observation.state"][2:, 1:] == 12.0)
