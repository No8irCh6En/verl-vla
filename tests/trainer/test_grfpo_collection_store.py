from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from verl import DataProto

from verl_vla.trainer.grfpo.collection_store import (
    CandidateAssignment,
    CollectionWindowSpec,
    CollectionWindowStore,
    PolicyIdentity,
    abandon_owner_leases,
    candidate_assignment,
    skip_window_candidates,
)


def _spec() -> CollectionWindowSpec:
    return CollectionWindowSpec(
        run_id="test-run",
        intended_update=1,
        group_size=8,
        target_accepted_groups=2,
        minimum_accepted_groups=1,
        max_candidate_groups=3,
        policy=PolicyIdentity(
            immutable_base_path="/models/base",
            immutable_base_sha256="a" * 64,
            trainer_checkpoint_path=None,
            rollout_policy_version=0,
        ),
    )


def _group(candidate_index: int, *, informative: bool):
    rollout = DataProto.from_dict(
        tensors={"value": torch.arange(8)},
        non_tensors={"name": np.asarray([str(index) for index in range(8)], dtype=object)},
    )
    end_obs = DataProto.from_dict(tensors={"value": torch.arange(8) + 1})
    record = {
        "group_id": f"candidate_{candidate_index:04d}",
        "rollout_policy_version": 0,
        "informative": informative,
        "reward_vector": [1.0, 0.0] + [0.0] * 6 if informative else [0.0] * 8,
    }
    assignment = CandidateAssignment(
        candidate_index=candidate_index,
        task_name="stack_bowls",
        task_id=0,
        suite_id=0,
        layout_id=candidate_index,
        policy_seeds=tuple(range(candidate_index * 8, candidate_index * 8 + 8)),
    )
    return assignment, rollout, end_obs, record


def test_committed_groups_survive_reopen_and_close_selects_only_whole_groups(tmp_path: Path):
    store = CollectionWindowStore(tmp_path, _spec())
    store.initialize()
    assignment, rollout, end_obs, record = _group(0, informative=True)
    assert (
        store.commit_group(assignment=assignment, rollout=rollout, rollout_end_obs=end_obs, group_record=record)
        == "accepted"
    )

    reopened = CollectionWindowStore(tmp_path, _spec())
    reopened.initialize()
    assert len(reopened.scan()["accepted"]) == 1
    assert reopened.close_if_ready() is None

    assignment, rollout, end_obs, record = _group(1, informative=False)
    assert (
        reopened.commit_group(assignment=assignment, rollout=rollout, rollout_end_obs=end_obs, group_record=record)
        == "rejected"
    )
    assignment, rollout, end_obs, record = _group(2, informative=True)
    assert (
        reopened.commit_group(assignment=assignment, rollout=rollout, rollout_end_obs=end_obs, group_record=record)
        == "accepted"
    )

    closed = reopened.close_if_ready()
    assert closed is not None and closed["outcome"] == "ready"
    assert len(closed["selected_group_paths"]) == 2
    loaded = reopened.load_selected_groups()
    assert len(loaded) == 2
    assert all(len(group_rollout) == 8 for group_rollout, _end_obs, _record in loaded)


def test_closed_window_rejects_late_or_duplicate_publication(tmp_path: Path):
    store = CollectionWindowStore(tmp_path, _spec())
    store.initialize()
    for candidate_index in (0, 1):
        assignment, rollout, end_obs, record = _group(candidate_index, informative=True)
        assert (
            store.commit_group(assignment=assignment, rollout=rollout, rollout_end_obs=end_obs, group_record=record)
            == "accepted"
        )
    assert store.close_if_ready()["outcome"] == "ready"

    assignment, rollout, end_obs, record = _group(0, informative=True)
    assert (
        store.commit_group(assignment=assignment, rollout=rollout, rollout_end_obs=end_obs, group_record=record)
        == "duplicate"
    )
    assignment, rollout, end_obs, record = _group(2, informative=True)
    assert (
        store.commit_group(assignment=assignment, rollout=rollout, rollout_end_obs=end_obs, group_record=record)
        == "late"
    )


def test_collection_manifest_pins_method_specific_rollout_trace(tmp_path: Path):
    base = _spec()
    flow_spec = CollectionWindowSpec(
        **{
            **base.__dict__,
            "actor_objective": "flow_grpo",
            "flow_grpo_noise_level": 0.02,
        }
    )
    store = CollectionWindowStore(tmp_path, flow_spec)
    store.initialize()
    manifest = json.loads((store.window_dir / "manifest.json").read_text())
    assert manifest["actor_objective"] == "flow_grpo"
    assert manifest["flow_grpo_noise_level"] == 0.02

    fpo_store = CollectionWindowStore(tmp_path, base)
    with pytest.raises(RuntimeError, match="manifest mismatch"):
        fpo_store.initialize()


def test_round_balanced_close_drains_round_and_uses_all_mixed_groups(tmp_path: Path):
    spec = CollectionWindowSpec(
        run_id="round-balanced",
        intended_update=1,
        group_size=8,
        target_accepted_groups=2,
        minimum_accepted_groups=1,
        max_candidate_groups=6,
        policy=PolicyIdentity("/base", "a" * 64, None, 0),
        complete_round_after_target=True,
        candidate_round_size=3,
    )
    store = CollectionWindowStore(tmp_path, spec)
    store.initialize()
    for candidate_index in (0, 1):
        assignment, rollout, end_obs, record = _group(candidate_index, informative=True)
        assert store.commit_group(
            assignment=assignment,
            rollout=rollout,
            rollout_end_obs=end_obs,
            group_record=record,
        ) == "accepted"
    # The target is a lower bound: the window cannot close until the third
    # lane completes the same candidate round.
    assert store.close_if_ready() is None
    assert not store.round_ready_to_start(3)

    assignment, rollout, end_obs, record = _group(2, informative=True)
    assert store.commit_group(
        assignment=assignment,
        rollout=rollout,
        rollout_end_obs=end_obs,
        group_record=record,
    ) == "accepted"
    assert store.round_ready_to_start(3)
    closed = store.close_if_ready()
    assert closed["outcome"] == "ready"
    assert closed["accepted_groups"] == 3
    assert len(closed["selected_group_paths"]) == 3
    assert len(store.load_selected_groups()) == 3


def test_drain_candidate_budget_keeps_target_as_lower_bound_and_uses_all_groups(tmp_path: Path):
    spec = CollectionWindowSpec(
        run_id="drain-budget",
        intended_update=1,
        group_size=8,
        target_accepted_groups=2,
        minimum_accepted_groups=1,
        max_candidate_groups=4,
        policy=PolicyIdentity("/base", "a" * 64, None, 0),
        drain_candidate_budget=True,
    )
    store = CollectionWindowStore(tmp_path, spec)
    store.initialize()
    for candidate_index in range(3):
        assignment, rollout, end_obs, record = _group(candidate_index, informative=True)
        assert store.commit_group(
            assignment=assignment,
            rollout=rollout,
            rollout_end_obs=end_obs,
            group_record=record,
        ) == "accepted"
    # Exceeding M=2 does not close or truncate this predeclared budget.
    assert store.close_if_ready() is None

    assignment, rollout, end_obs, record = _group(3, informative=False)
    assert store.commit_group(
        assignment=assignment,
        rollout=rollout,
        rollout_end_obs=end_obs,
        group_record=record,
    ) == "rejected"
    closed = store.close_if_ready()
    assert closed["outcome"] == "ready"
    assert closed["accepted_groups"] == 3
    assert closed["drain_candidate_budget"] is True
    assert len(closed["selected_group_paths"]) == 3


def test_assignment_is_reproducible_and_group_policy_seeds_are_distinct():
    spec = _spec()
    kwargs = {
        "spec": spec,
        "candidate_index": 2,
        "task_names": ["stack_bowls", "fold_clothes"],
        "suite_ids": [0, 1, 2],
        "training_layouts": {
            (task, suite): [1, 0, 3, 5, 2, 4]
            for task in ("stack_bowls", "fold_clothes")
            for suite in (0, 1, 2)
        },
        "run_seed": 7865,
    }
    first = candidate_assignment(**kwargs)
    second = candidate_assignment(**kwargs)
    assert first == second
    assert len(first.policy_seeds) == 8
    assert len(set(first.policy_seeds)) == 8
    assert first.suite_id == 2
    assert first.layout_id in {0, 1, 2, 3, 4, 5}


def test_policy_seed_namespace_pairs_distinct_ablation_run_ids():
    common = dict(
        intended_update=1,
        group_size=8,
        target_accepted_groups=2,
        minimum_accepted_groups=1,
        max_candidate_groups=3,
        policy=PolicyIdentity("/base", "a" * 64, None, 0),
        policy_seed_namespace="paired-ablation",
    )
    training_layouts = {("stack_bowls", 0): [0, 1, 2]}
    left = candidate_assignment(
        spec=CollectionWindowSpec(run_id="binary", **common),
        candidate_index=1,
        task_names=["stack_bowls"],
        suite_ids=[0],
        training_layouts=training_layouts,
        run_seed=7865,
    )
    right = candidate_assignment(
        spec=CollectionWindowSpec(
            run_id="terminal-score",
            group_reward_source="process_score",
            **common,
        ),
        candidate_index=1,
        task_names=["stack_bowls"],
        suite_ids=[0],
        training_layouts=training_layouts,
        run_seed=7865,
    )

    assert left.layout_id == right.layout_id
    assert left.policy_seeds == right.policy_seeds


def test_training_transaction_recovers_and_becomes_idempotent_after_receipt(tmp_path: Path):
    store = CollectionWindowStore(tmp_path, _spec())
    store.initialize()
    with store.training_transaction() as receipt:
        assert receipt is None
    # Exiting without a receipt models a preempted/failed actor update. The
    # next process can acquire the lock and retry from the retained groups.
    with store.training_transaction() as receipt:
        assert receipt is None
        store.write_training_receipt({"new_policy_version": 1, "checkpoint_dir": "/ckpt/theta1"})
    with store.training_transaction() as receipt:
        assert receipt["new_policy_version"] == 1


def test_counterfactual_training_receipt_does_not_touch_source_receipt(tmp_path: Path):
    store = CollectionWindowStore(tmp_path / "source", _spec())
    store.initialize()
    store.write_training_receipt({"new_policy_version": 1, "checkpoint_dir": "/original"})
    replay_receipt = tmp_path / "branch-a" / "REPLAY_TRAINED.json"

    with store.training_transaction(replay_receipt) as receipt:
        assert receipt is None
        store.write_training_receipt(
            {"new_policy_version": 1, "checkpoint_dir": "/counterfactual"},
            replay_receipt,
        )

    with store.training_transaction(replay_receipt) as receipt:
        assert receipt["checkpoint_dir"] == "/counterfactual"
    with store.training_transaction() as receipt:
        assert receipt["checkpoint_dir"] == "/original"


def test_shared_candidate_queue_claims_unique_work_and_releases_on_commit(tmp_path: Path):
    store = CollectionWindowStore(tmp_path, _spec())
    store.initialize()
    assignments = [_group(index, informative=True)[0] for index in range(3)]

    first = store.claim_candidate(
        assignments=assignments,
        owner_id="worker-a",
        lease_timeout_seconds=60,
        preferred_condition=("stack_bowls", 0),
    )
    second = store.claim_candidate(
        assignments=assignments,
        owner_id="worker-b",
        lease_timeout_seconds=60,
        preferred_condition=("stack_bowls", 0),
    )
    assert first is not None and first.assignment.candidate_index == 0
    assert second is not None and second.assignment.candidate_index == 1
    assert first.token != second.token

    assignment, rollout, end_obs, record = _group(0, informative=True)
    assert store.commit_group(
        assignment=assignment,
        rollout=rollout,
        rollout_end_obs=end_obs,
        group_record=record,
        lease=first,
    ) == "accepted"
    assert not (store.leases_dir / "candidate_0000.json").exists()

    assert store.release_candidate_lease(second)
    replacement = store.claim_candidate(
        assignments=assignments,
        owner_id="worker-c",
        lease_timeout_seconds=60,
    )
    assert replacement is not None and replacement.assignment.candidate_index == 1


def test_expired_candidate_lease_is_reclaimed_and_stale_commit_is_rejected(tmp_path: Path):
    store = CollectionWindowStore(tmp_path, _spec())
    store.initialize()
    assignment, rollout, end_obs, record = _group(0, informative=True)
    first = store.claim_candidate(
        assignments=[assignment],
        owner_id="preempted-worker",
        lease_timeout_seconds=0.01,
    )
    assert first is not None
    time.sleep(0.02)
    replacement = store.claim_candidate(
        assignments=[assignment],
        owner_id="replacement-worker",
        lease_timeout_seconds=60,
    )
    assert replacement is not None and replacement.token != first.token
    assert store.commit_group(
        assignment=assignment,
        rollout=rollout,
        rollout_end_obs=end_obs,
        group_record=record,
        lease=first,
    ) == "stale_lease"
    assert store.commit_group(
        assignment=assignment,
        rollout=rollout,
        rollout_end_obs=end_obs,
        group_record=record,
        lease=replacement,
    ) == "accepted"
    events = [json.loads(line) for line in (store.window_dir / "LEASE_EVENTS.jsonl").read_text().splitlines()]
    assert [event["event"] for event in events] == ["claim", "reclaim"]


def test_authoritative_worker_death_releases_only_that_owners_leases(tmp_path: Path):
    store = CollectionWindowStore(tmp_path, _spec())
    store.initialize()
    assignments = [_group(index, informative=True)[0] for index in range(3)]
    dead = store.claim_candidate(
        assignments=assignments,
        owner_id="dead-job:3",
        lease_timeout_seconds=900,
    )
    live = store.claim_candidate(
        assignments=assignments,
        owner_id="live-job:4",
        lease_timeout_seconds=900,
    )
    assert dead is not None and live is not None

    assert abandon_owner_leases(store.window_dir, "dead-job:3") == [dead.assignment.candidate_id]
    assert not (store.leases_dir / f"{dead.assignment.candidate_id}.json").exists()
    assert (store.leases_dir / f"{live.assignment.candidate_id}.json").exists()

    replacement = store.claim_candidate(
        assignments=assignments,
        owner_id="replacement-job:3",
        lease_timeout_seconds=900,
    )
    assert replacement is not None
    assert replacement.assignment == dead.assignment
    assignment, rollout, end_obs, record = _group(dead.assignment.candidate_index, informative=True)
    assert store.commit_group(
        assignment=assignment,
        rollout=rollout,
        rollout_end_obs=end_obs,
        group_record=record,
        lease=dead,
    ) == "stale_lease"


def test_operator_skip_is_terminal_but_not_a_rejected_rollout(tmp_path: Path):
    store = CollectionWindowStore(tmp_path, _spec())
    store.initialize()
    skipped = skip_window_candidates(
        store.window_dir,
        ["candidate_0001"],
        reason="operator_cancelled_low_value_condition",
    )
    assert skipped == ["candidate_0001"]
    assert store.candidate_complete("candidate_0001")
    state = store.scan()
    assert len(state["skipped"]) == 1
    assert len(state["rejected"]) == 0
    assert state["completed_candidate_count"] == 1


def test_integrity_failure_is_durable_but_candidate_remains_retriable(tmp_path: Path):
    store = CollectionWindowStore(tmp_path, _spec())
    store.initialize()
    assignment, _rollout, _end_obs, _record = _group(0, informative=True)
    lease = store.claim_candidate(
        assignments=[assignment],
        owner_id="corrupt-simulator",
        lease_timeout_seconds=60,
    )
    assert lease is not None

    failure = store.record_integrity_failure(
        assignment=assignment,
        diagnostics={
            "valid": False,
            "raw_state_absmax": 454_608.6875,
            "raw_state_abs_limit": 100.0,
            "nonfinite_count": 0,
            "offending_trajectory_indices": [2, 3, 5],
        },
        lease=lease,
    )
    assert failure is not None and failure.is_file()
    assert not store.candidate_complete(assignment.candidate_id)
    assert store.scan()["completed_candidate_count"] == 0
    assert store.release_candidate_lease(lease)

    replacement = store.claim_candidate(
        assignments=[assignment],
        owner_id="fresh-simulator",
        lease_timeout_seconds=60,
    )
    assert replacement is not None
    assert replacement.assignment == assignment
