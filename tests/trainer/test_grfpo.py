# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from verl import DataProto

from verl_vla.trainer.grfpo import GRFPOTrainerConfig, prepare_fpo_actor_input
from verl_vla.trainer.grfpo.group_validation import inspect_rollout_state_integrity
from verl_vla.trainer.grfpo.grfpo_ray_trainer import (
    GRFPORayTrainer,
    _candidate_task_assignments,
    _optional_refill_is_unnecessary,
    apply_grpo_outcome_advantages,
    build_fpo_trajectory_audit,
    concatenate_rollout_groups_with_padding,
    first_rollout_observation_and_action,
    summarize_dataproto_contract,
    summarize_rollout_trajectories,
    validate_same_condition_group,
)
from verl_vla.trainer.grfpo.rollout_batch import compact_valid_policy_chunks_for_fpo
from verl_vla.workers.config import FPOActorConfig, FPOValueConfig
from verl_vla.workers.engine.fpo import training_worker as training_worker_module
from verl_vla.workers.engine.fpo.training_worker import (
    FPOTrainingWorker,
    _gradient_mapping_dot,
    _gradient_mapping_linear_combination_stats,
    clipped_policy_loss,
    compute_cfm_log_ratio,
    compute_cfm_per_sample_log_ratio,
    compute_gae,
    distributed_gradient_mean_denominator,
    fpo_plus_plus_policy_loss,
)


@pytest.mark.parametrize(
    ("candidates", "accepted", "expected"),
    [(26, 20, False), (27, 11, False), (27, 12, True), (36, 12, True)],
)
def test_optional_refill_boundary_retains_initial_mixed_groups(candidates, accepted, expected):
    assert (
        _optional_refill_is_unnecessary(
            candidate_groups=candidates,
            accepted_groups=accepted,
            initial_candidate_groups=27,
            accepted_groups_before_refill=12,
        )
        is expected
    )


def test_optional_refill_boundary_is_disabled_by_default():
    assert not _optional_refill_is_unnecessary(
        candidate_groups=27,
        accepted_groups=12,
        initial_candidate_groups=0,
        accepted_groups_before_refill=0,
    )


def test_multitask_candidate_assignments_follow_stage_major_worker_order():
    config = SimpleNamespace(
        cluster=SimpleNamespace(
            resource=SimpleNamespace(
                env=SimpleNamespace(device="cuda", nnodes=1, gpus_per_node=3, workers_per_node=1)
            ),
            env=SimpleNamespace(
                env_loop=SimpleNamespace(pipeline_stage_num=3),
                env_worker=SimpleNamespace(
                    num_envs=8,
                    simulator=SimpleNamespace(
                        robodojo=SimpleNamespace(
                            task_name="stack_bowls",
                            task_id=0,
                            task_names=[
                                "stack_bowls",
                                "push_T",
                                "cover_blocks",
                                "plug_in_charger",
                                "fill_egg_holder",
                            ],
                            worker_stage_task_schedule=[
                                "stack_bowls",
                                "push_T",
                                "cover_blocks",
                                "plug_in_charger",
                                "fill_egg_holder",
                                "stack_bowls",
                                "push_T",
                                "cover_blocks",
                                "plug_in_charger",
                            ],
                        )
                    ),
                ),
            ),
        )
    )
    trainer = SimpleNamespace(group_size=8, candidate_group_partition="worker_stage")

    assignments = _candidate_task_assignments(config, trainer, group_count=9)

    assert [task_name for task_name, _task_id in assignments] == [
        "stack_bowls",
        "push_T",
        "cover_blocks",
        "plug_in_charger",
        "fill_egg_holder",
        "stack_bowls",
        "push_T",
        "cover_blocks",
        "plug_in_charger",
    ]
    assert [task_id for _task_name, task_id in assignments] == [0, 1, 2, 3, 4, 0, 1, 2, 3]


def test_prepare_fpo_actor_input_preserves_temporal_contract_and_masks_terminal_chunk():
    obs_states = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    rollout_end_states = torch.tensor([[100, 101, 102, 103], [200, 201, 202, 203]], dtype=torch.float32)
    rollout = DataProto.from_dict(
        tensors={
            "obs.state": obs_states,
            # Reset/collection profiling is not a model observation and has no
            # corresponding final observation. This reproduces the Phase 6E
            # M=8 preparation failure and must be stripped defensively.
            "obs.profile.reset_process_restarts": torch.ones(2, 3, dtype=torch.int64),
            "action.action": torch.zeros(2, 3, 3, 2),
            "action.full_action": torch.zeros(2, 3, 5, 4),
            "next.reward": torch.tensor(
                [
                    [[1.0, 2.0, 99.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
                    [[0.0, 1.0, 2.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
                ]
            ),
            "next.terminated": torch.tensor(
                [
                    [[False, True, True], [False, False, False], [False, False, False]],
                    [[False, False, False], [False, False, False], [False, False, False]],
                ]
            ),
            "next.truncated": torch.zeros(2, 3, 3, dtype=torch.bool),
            "next.success": torch.zeros(2, 3, 3, dtype=torch.bool),
        },
        non_tensors={
            "obs.task": np.array([["a0", "a1", "a2"], ["b0", "b1", "b2"]], dtype=object),
            "obs.layout_id": np.asarray([[0, 0, 0], [3, 3, 3]], dtype=np.int64),
            "obs.environment_seed": np.asarray([[0, 0, 0], [3, 3, 3]], dtype=np.int64),
            "obs.policy_seed": np.asarray([[10, 10, 10], [11, 11, 11]], dtype=np.int64),
        },
    )

    output = prepare_fpo_actor_input(
        rollout,
        DataProto.from_dict(
            tensors={"state": rollout_end_states},
            non_tensors={"task": np.array(["a3", "b3"], dtype=object)},
        ),
        trainer_config=SimpleNamespace(step_penalty=0.25, gamma=0.995, gae_lambda=0.99),
        global_steps=7,
    )

    assert output.batch["info.valids"].shape == (2, 3)
    assert output.batch["info.action_valids"].shape == (2, 3, 5)
    assert output.batch["info.durations"].shape == (2, 3)
    torch.testing.assert_close(
        output.batch["next_obs.state"],
        torch.cat([obs_states[:, 1:], rollout_end_states.unsqueeze(1)], dim=1),
    )
    np.testing.assert_array_equal(
        output.non_tensor_batch["next_obs.task"],
        np.array([["a1", "a2", "a3"], ["b1", "b2", "b3"]], dtype=object),
    )
    assert "next_obs.layout_id" not in output.non_tensor_batch
    assert "next_obs.environment_seed" not in output.non_tensor_batch
    assert "next_obs.policy_seed" not in output.non_tensor_batch
    assert "obs.profile.reset_process_restarts" not in output.batch.keys()
    assert "next_obs.profile.reset_process_restarts" not in output.batch.keys()
    np.testing.assert_array_equal(output.non_tensor_batch["obs.layout_id"][:, 0], [0, 3])
    gamma = 0.995
    torch.testing.assert_close(
        output.batch["info.rewards"][0],
        torch.tensor([1 + gamma * 2 - 0.25, 0.0, 0.0]),
    )
    torch.testing.assert_close(output.batch["info.dones"][0], torch.tensor([1.0, 0.0, 0.0]))
    torch.testing.assert_close(output.batch["info.discounts"][0], torch.tensor([0.0, 0.0, 0.0]))
    torch.testing.assert_close(output.batch["info.valids"][0], torch.tensor([1.0, 0.0, 0.0]))
    torch.testing.assert_close(output.batch["info.action_valids"][0, 0], torch.tensor([1, 1, 0, 0, 0]).float())
    torch.testing.assert_close(output.batch["info.durations"][0], torch.tensor([2.0, 0.0, 0.0]))
    assert not output.batch["info.action_valids"][0, 1:].any()
    assert output.meta_info["global_steps"] == 7
    assert output.meta_info["gamma"] == 0.995


def test_variable_length_group_concat_preserves_terminal_successor_and_masks_padding():
    def make_group(steps: int, offset: float, *, omit_terminal_suite: bool = False) -> tuple[DataProto, DataProto]:
        batch_size = 2
        state = torch.arange(batch_size * steps * 2, dtype=torch.float32).reshape(batch_size, steps, 2)
        state += offset
        terminal_state = torch.tensor([[900.0, 901.0], [910.0, 911.0]]) + offset
        terminated = torch.zeros(batch_size, steps, 2, dtype=torch.bool)
        terminated[:, -1, -1] = True
        rollout = DataProto.from_dict(
            tensors={
                "obs.state": state,
                "action.action": torch.ones(batch_size, steps, 2, 1),
                "action.full_action": torch.ones(batch_size, steps, 3, 1),
                "next.reward": torch.ones(batch_size, steps, 2),
                "next.terminated": terminated,
                "next.truncated": torch.zeros_like(terminated),
                "next.success": terminated.clone(),
            },
            non_tensors={
                "obs.task": np.full((batch_size, steps), f"task-{offset}", dtype=object),
                "obs.suite_id": np.full((batch_size, steps), int(offset) + 1, dtype=np.int64),
                "obs.layout_id": np.full((batch_size, steps), int(offset), dtype=np.int64),
            },
        )
        terminal_non_tensors = {
            "task": np.full(batch_size, f"terminal-{offset}", dtype=object),
            "suite_id": np.full(batch_size, int(offset) + 1, dtype=np.int64),
        }
        if omit_terminal_suite:
            terminal_non_tensors.pop("suite_id")
        end_obs = DataProto.from_dict(
            tensors={"state": terminal_state},
            non_tensors=terminal_non_tensors,
        )
        return rollout, end_obs

    # Regress the mixed-schema window produced when suite_id preservation was
    # repaired after suite-0 groups had already committed.
    short_rollout, short_end = make_group(2, 0.0, omit_terminal_suite=True)
    long_rollout, long_end = make_group(3, 10.0)
    rollout, end_obs = concatenate_rollout_groups_with_padding(
        [short_rollout, long_rollout],
        [short_end, long_end],
    )

    assert rollout.batch["action.action"].shape == (4, 3, 2, 1)
    torch.testing.assert_close(rollout.batch["obs.state"][:2, 2], short_end.batch["state"])
    assert not rollout.batch["action.action"][:2, 2].any()
    assert not rollout.batch["next.reward"][:2, 2].any()
    assert end_obs.non_tensor_batch["suite_id"].tolist() == [1, 1, 11, 11]

    actor_input = prepare_fpo_actor_input(
        rollout,
        end_obs,
        trainer_config=SimpleNamespace(step_penalty=0.0, gamma=0.995, gae_lambda=0.99),
        global_steps=0,
    )
    torch.testing.assert_close(actor_input.batch["next_obs.state"][:2, 1], short_end.batch["state"])
    torch.testing.assert_close(
        actor_input.batch["info.valids"][:2],
        torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]]),
    )
    assert not actor_input.batch["info.action_valids"][:2, 2].any()


@pytest.mark.parametrize(("world_size", "micro_batch_size"), [(1, 1), (1, 4), (2, 2)])
def test_fpo_valid_chunk_compaction_preserves_weighted_loss_and_gradient(
    world_size, micro_batch_size
):
    valids = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    loss_weights = torch.tensor([[0.5, 0.5, 0.0], [1.0, 0.0, 0.0]])
    features = torch.tensor([[1.0, 2.0, 1000.0], [3.0, 1000.0, 1000.0]])
    data = DataProto.from_dict(
        tensors={
            "action.action": torch.zeros(2, 3, 1, 1),
            "info.valids": valids,
            "info.action_valids": valids.unsqueeze(-1).expand(-1, -1, 2).clone(),
            "fpo.loss_weights": loss_weights,
            "fpo.advantages": valids.clone(),
            "probe.features": features,
        },
        non_tensors={
            "obs.task": np.full((2, 3), "task", dtype=object),
        },
    )

    reference_parameter = torch.nn.Parameter(torch.tensor(0.25))
    reference_loss = (
        reference_parameter * features * loss_weights
    ).sum() / loss_weights.sum()
    reference_loss.backward()
    reference_gradient = reference_parameter.grad.detach().clone()

    compact = compact_valid_policy_chunks_for_fpo(
        data,
        actor_world_size=world_size,
        micro_batch_size=micro_batch_size,
    )
    parameter = torch.nn.Parameter(torch.tensor(0.25))
    numerator = torch.zeros(())
    for physical_micro_batch in compact.split(micro_batch_size):
        numerator = numerator + (
            parameter
            * physical_micro_batch.batch["probe.features"]
            * physical_micro_batch.batch["fpo.loss_weights"]
        ).sum()
    loss = numerator / compact.batch["fpo.loss_weights"].sum()
    loss.backward()

    assert compact.meta_info["fpo_original_physical_slots"] == 6
    assert compact.meta_info["fpo_valid_policy_chunks"] == 3
    assert compact.meta_info["fpo_sync_dummy_slots"] == (-3) % (world_size * micro_batch_size)
    assert len(compact) % (world_size * micro_batch_size) == 0
    assert int(compact.batch["info.valids"].sum()) == 3
    torch.testing.assert_close(loss, reference_loss.detach())
    torch.testing.assert_close(parameter.grad, reference_gradient)


def test_compute_gae_stops_at_terminal_and_padding_boundaries():
    rewards = torch.tensor([[1.0, 2.0, 100.0, 100.0]])
    values = torch.zeros_like(rewards)
    next_values = torch.tensor([[10.0, 10.0, 10.0, 10.0]])
    discounts = torch.tensor([[0.5, 0.0, 0.5, 0.5]])
    gae_discounts = torch.tensor([[0.5, 0.0, 0.5, 0.5]])
    valids = torch.tensor([[1.0, 1.0, 0.0, 0.0]])

    advantages, returns = compute_gae(
        rewards,
        values,
        next_values,
        discounts,
        gae_discounts,
        valids,
    )

    torch.testing.assert_close(advantages, torch.tensor([[7.0, 2.0, 0.0, 0.0]]))
    torch.testing.assert_close(returns, advantages)
    assert torch.isfinite(advantages).all()
    assert torch.isfinite(returns).all()


def test_fpo_trajectory_audit_records_contract_finiteness_and_idle_slots():
    rollout = DataProto.from_dict(
        tensors={
            "obs.state": torch.zeros(1, 2, 14),
            "action.action": torch.zeros(1, 2, 24, 14),
            "action.full_action": torch.zeros(1, 2, 32, 14),
            "next.reward": torch.zeros(1, 2, 24),
            "next.terminated": torch.zeros(1, 2, 24, dtype=torch.bool),
            "next.truncated": torch.zeros(1, 2, 24, dtype=torch.bool),
            "next.success": torch.zeros(1, 2, 24, dtype=torch.bool),
        },
        non_tensors={
            "obs.task": np.asarray([["task", "task"]], dtype=object),
            "obs.task_id": np.asarray([[5, 5]], dtype=np.int64),
        },
    )
    raw_contract = summarize_dataproto_contract(rollout)
    actor_input = prepare_fpo_actor_input(
        rollout,
        DataProto.from_dict(
            tensors={"state": torch.zeros(1, 14)},
            non_tensors={"task": np.asarray(["task"], dtype=object), "task_id": np.asarray([5])},
        ),
        trainer_config=SimpleNamespace(step_penalty=0.0, gamma=0.995, gae_lambda=0.99),
        global_steps=1,
    )
    actor_input.batch["info.valids"][0, 1] = 0
    actor_input.batch["info.action_valids"][0, 1] = 0
    audit = build_fpo_trajectory_audit(
        raw_contract=raw_contract,
        raw_semantics={"task_ids": [5]},
        actor_input=actor_input,
    )

    assert audit["raw_rollout"]["tensor_fields"]["action.full_action"]["shape"] == [1, 2, 32, 14]
    assert audit["prepared_actor_input"]["all_floating_tensors_finite"] is True
    assert audit["semantics"]["valid_policy_slots"] == 1
    assert audit["semantics"]["idle_policy_slots"] == 1
    assert audit["semantics"]["duration_min_valid"] == 24
    assert audit["semantics"]["duration_max_valid"] == 24


def test_fixed_observation_diagnostic_extracts_first_slot_without_aliasing():
    rollout = DataProto.from_dict(
        tensors={
            "obs.state": torch.arange(2 * 3 * 14, dtype=torch.float32).reshape(2, 3, 14),
            "action.full_action": torch.arange(2 * 3 * 32 * 14, dtype=torch.float32).reshape(2, 3, 32, 14),
        },
        non_tensors={
            "obs.task": np.asarray([["a0", "a1", "a2"], ["b0", "b1", "b2"]], dtype=object),
            "obs.task_id": np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64),
        },
    )

    observation, action = first_rollout_observation_and_action(rollout)

    torch.testing.assert_close(observation.batch["state"], rollout.batch["obs.state"][:, 0])
    torch.testing.assert_close(action, rollout.batch["action.full_action"][:, 0])
    np.testing.assert_array_equal(observation.non_tensor_batch["task"], np.asarray(["a0", "b0"]))
    np.testing.assert_array_equal(observation.non_tensor_batch["task_id"], np.asarray([0, 3]))
    observation.batch["state"].zero_()
    assert rollout.batch["obs.state"].count_nonzero() > 0


def test_rollout_trajectory_summary_keeps_case_seed_and_terminal_counts():
    rollout = DataProto.from_dict(
        tensors={
            "next.reward": torch.tensor([[[0.0, 1.0, 0.0]], [[0.0, 0.0, 0.0]]]),
            "next.terminated": torch.tensor([[[False, True, True]], [[False, False, False]]]),
            "next.truncated": torch.tensor([[[False, False, False]], [[False, False, True]]]),
            "next.success": torch.tensor([[[False, True, True]], [[False, False, False]]]),
        },
        non_tensors={
            "obs.task_id": np.asarray([[0], [0]]),
            "obs.layout_id": np.asarray([[3], [3]]),
            "obs.environment_seed": np.asarray([[3], [3]]),
            "obs.policy_seed": np.asarray([[101], [102]]),
        },
    )

    records = summarize_rollout_trajectories(rollout)

    assert records[0] == {
        "lane": 0,
        "policy_calls": 1,
        "action_steps": 2,
        "return": 1.0,
        "success": True,
        "terminated": True,
        "truncated": False,
        "task_id": 0,
        "layout_id": 3,
        "environment_seed": 3,
        "policy_seed": 101,
    }
    assert records[1]["action_steps"] == 3
    assert records[1]["success"] is False


def test_clipped_policy_loss_uses_cfm_loss_difference_as_log_ratio():
    log_ratio = torch.log(torch.tensor([1.2, 0.8, 4.0]))
    advantages = torch.tensor([1.0, -1.0, 100.0])
    valids = torch.tensor([1.0, 1.0, 0.0])

    loss, metrics = clipped_policy_loss(log_ratio, advantages, valids, clip_coef=0.1)

    # positive advantage clips 1.2 to 1.1; negative advantage clips 0.8 to 0.9.
    torch.testing.assert_close(loss, torch.tensor((-1.1 + 0.9) / 2))
    assert torch.isfinite(loss)
    torch.testing.assert_close(metrics["clip_fraction"], torch.tensor(1.0))


def test_cfm_log_ratio_ignores_unexecuted_32_step_suffix():
    old = torch.zeros(2, 32, 4)
    current = torch.zeros_like(old)
    current[:, 24:] = 1000.0
    mask = torch.zeros(2, 32)
    mask[:, :24] = 1.0

    torch.testing.assert_close(compute_cfm_log_ratio(old, current, mask), torch.zeros(2))


def test_cfm_log_ratio_honors_early_terminal_prefix():
    old = torch.zeros(1, 32, 4)
    current = torch.ones_like(old)
    current[:, 2:] = 999.0
    mask = torch.zeros(1, 32)
    mask[:, :2] = 1.0

    # Two executed steps, each with mean fixed-MC loss difference -1.
    torch.testing.assert_close(compute_cfm_log_ratio(old, current, mask), torch.tensor([-2.0]))


def _group_rollout(*, layout_ids=None, policy_seeds=None, rewards=None, scores=None):
    group_size, policy_slots, action_steps = 8, 2, 3
    layout_ids = layout_ids or [3] * group_size
    policy_seeds = policy_seeds or list(range(3000, 3008))
    rewards = rewards or [0, 0, 0, 1, 0, 0, 0, 0]
    scores = rewards if scores is None else scores
    successes = torch.zeros(group_size, policy_slots, action_steps, dtype=torch.bool)
    process_scores = torch.zeros(group_size, policy_slots, action_steps, dtype=torch.float32)
    for index, reward in enumerate(rewards):
        successes[index, 0, 1] = bool(reward)
        process_scores[index].fill_(float(scores[index]))
    return DataProto.from_dict(
        tensors={
            "next.success": successes,
            "next.score": process_scores,
            "next.reward": successes.float(),
            "next.terminated": successes.clone(),
            "next.truncated": ~successes.any(dim=(1, 2))[:, None, None].expand_as(successes).clone(),
        },
        non_tensors={
            "obs.task": np.full((group_size, policy_slots), "  Stack   bowls ", dtype=object),
            "obs.task_id": np.zeros((group_size, policy_slots), dtype=np.int64),
            "obs.suite_id": np.zeros((group_size, policy_slots), dtype=np.int64),
            "obs.layout_id": np.repeat(np.asarray(layout_ids)[:, None], policy_slots, axis=1),
            "obs.environment_seed": np.repeat(np.asarray(layout_ids)[:, None], policy_slots, axis=1),
            "obs.policy_seed": np.repeat(np.asarray(policy_seeds)[:, None], policy_slots, axis=1),
        },
    )


def test_g8_same_condition_group_contract_and_distinct_seeds():
    rollout = _group_rollout()
    record = validate_same_condition_group(
        rollout,
        group_size=8,
        group_id="group-1",
        rollout_policy_version=4,
    )

    assert record["group_key"] == {
        "task_id": 0,
        "normalized_instruction": "stack bowls",
        "suite_id": 0,
        "layout_id": 3,
        "environment_seed": 3,
    }
    assert record["policy_seeds"] == list(range(3000, 3008))
    assert record["reward_vector"] == [0, 0, 0, 1, 0, 0, 0, 0]
    assert record["mixed"] is True
    assert np.unique(rollout.non_tensor_batch["obs.group_id"]).tolist() == ["group-1"]

    with pytest.raises(ValueError, match="mixes environment conditions"):
        validate_same_condition_group(
            _group_rollout(layout_ids=[3] * 7 + [2]),
            group_size=8,
            group_id="bad-condition",
            rollout_policy_version=4,
        )
    with pytest.raises(ValueError, match="distinct policy seeds"):
        validate_same_condition_group(
            _group_rollout(policy_seeds=[3000] * 8),
            group_size=8,
            group_id="bad-seeds",
            rollout_policy_version=4,
        )


def test_rollout_state_integrity_rejects_one_corrupt_trajectory_before_group_filtering():
    rollout = _group_rollout()
    rollout.batch["obs.observation.state"] = torch.zeros(8, 2, 14)
    end_obs = DataProto.from_dict(tensors={"state": torch.zeros(8, 14)})

    healthy = inspect_rollout_state_integrity(
        rollout,
        end_obs,
        raw_state_abs_limit=100.0,
    )
    assert healthy["valid"] is True
    assert healthy["offending_trajectory_indices"] == []

    rollout.batch["obs.observation.state"][3, 0, 0] = 454_608.6875
    corrupt = inspect_rollout_state_integrity(
        rollout,
        end_obs,
        raw_state_abs_limit=100.0,
    )
    assert corrupt["valid"] is False
    assert corrupt["raw_state_absmax"] == pytest.approx(454_608.6875)
    assert corrupt["offending_trajectory_indices"] == [3]


def test_rollout_state_integrity_rejects_nonfinite_terminal_state():
    rollout = _group_rollout()
    rollout.batch["obs.observation.state"] = torch.zeros(8, 2, 14)
    end_obs = DataProto.from_dict(tensors={"state": torch.zeros(8, 14)})
    end_obs.batch["state"][5, 2] = float("nan")

    result = inspect_rollout_state_integrity(
        rollout,
        end_obs,
        raw_state_abs_limit=100.0,
    )
    assert result["valid"] is False
    assert result["nonfinite_count"] == 1
    assert result["offending_trajectory_indices"] == [5]


def test_process_score_group_accepts_one_partial_without_counting_it_as_success():
    rollout = _group_rollout(rewards=[0] * 8, scores=[0.15] + [0.0] * 7)
    record = validate_same_condition_group(
        rollout,
        group_size=8,
        group_id="partial-one",
        rollout_policy_version=4,
        reward_source="process_score",
        informative_group_criterion="reward_variance",
        min_partial_trajectories_for_score_only_group=1,
    )

    assert record["successes"] == 0
    assert record["binary_mixed"] is False
    assert record["partial_only_count"] == 1
    assert record["partial_or_better_count"] == 1
    assert record["reward_vector"] == pytest.approx([0.15] + [0.0] * 7)
    assert record["informative"] is True
    assert record["informative_reason"] == "score_variance_partial_only"

    requires_two = validate_same_condition_group(
        rollout,
        group_size=8,
        group_id="partial-two-required",
        rollout_policy_version=4,
        reward_source="process_score",
        informative_group_criterion="reward_variance",
        min_partial_trajectories_for_score_only_group=2,
    )
    assert requires_two["successes"] == 0
    assert requires_two["informative"] is False
    assert requires_two["informative_reason"] == "partial_only_below_minimum"


def test_process_score_full_success_mixed_rejects_partial_only_and_uses_partial_ranking_when_anchored():
    partial_only = _group_rollout(rewards=[0] * 8, scores=[0.15, 0.15] + [0.0] * 6)
    rejected = validate_same_condition_group(
        partial_only,
        group_size=8,
        group_id="partial-only-rejected",
        rollout_policy_version=4,
        reward_source="process_score",
        informative_group_criterion="full_success_mixed",
    )
    assert rejected["full_success_count"] == 0
    assert rejected["training_reward_varies"] is True
    assert rejected["informative"] is False
    assert rejected["informative_reason"] == "not_mixed_full_success"

    anchored = _group_rollout(
        rewards=[1] + [0] * 7,
        scores=[1.0, 0.15] + [0.0] * 6,
    )
    accepted = validate_same_condition_group(
        anchored,
        group_size=8,
        group_id="full-success-anchored",
        rollout_policy_version=4,
        reward_source="process_score",
        informative_group_criterion="full_success_mixed",
    )
    assert accepted["full_success_count"] == 1
    assert accepted["partial_only_count"] == 1
    assert accepted["reward_vector"] == pytest.approx([1.0, 0.15] + [0.0] * 6)
    assert accepted["informative"] is True
    assert accepted["informative_reason"] == "mixed_full_success"


def test_process_score_advantage_uses_partial_reward_but_preserves_binary_success_field():
    valids = torch.ones(8, 1)
    scores = torch.tensor([0.15] + [0.0] * 7)
    actor_input = DataProto.from_dict(
        tensors={
            "info.trajectory_outcome": torch.zeros(8),
            "info.trajectory_process_score": scores,
            "info.valids": valids,
        }
    )
    record = {
        "group_id": "partial",
        "rollout_policy_version": 0,
        "mixed": False,
        "informative": True,
        "successes": 0,
        "partial_only_count": 1,
        "informative_reason": "score_variance_partial_only",
    }

    diagnostics = apply_grpo_outcome_advantages(
        actor_input,
        group_record=record,
        group_size=8,
        reward_source="process_score",
    )

    advantages = actor_input.batch["fpo.trajectory_group_advantage"]
    assert advantages[0] > 0
    assert (advantages[1:] < 0).all()
    assert diagnostics["groups"][0]["successes"] == 0
    assert diagnostics["groups"][0]["partial_only_count"] == 1


@pytest.mark.parametrize(
    ("rewards", "expect_nonzero"),
    [
        ([0] * 8, False),
        ([1] * 8, False),
        ([1] + [0] * 7, True),
        ([1] * 4 + [0] * 4, True),
        ([1] * 7 + [0], True),
    ],
)
def test_grpo_outcome_advantage_zero_variance_signs_and_balanced_broadcast(rewards, expect_nonzero):
    valids = torch.zeros(8, 4)
    for index in range(8):
        valids[index, : 1 + index % 4] = 1
    actor_input = DataProto.from_dict(
        tensors={
            "info.trajectory_outcome": torch.tensor(rewards, dtype=torch.float32),
            "info.valids": valids,
        }
    )
    record = {"group_id": "g", "rollout_policy_version": 0, "mixed": 0 < sum(rewards) < 8}

    diagnostics = apply_grpo_outcome_advantages(actor_input, group_record=record, group_size=8)
    trajectory_advantages = actor_input.batch["fpo.trajectory_group_advantage"]

    assert torch.isfinite(trajectory_advantages).all()
    torch.testing.assert_close(actor_input.batch["fpo.loss_weights"].sum(dim=1), torch.ones(8))
    torch.testing.assert_close(actor_input.batch["fpo.advantages"], trajectory_advantages[:, None] * valids)
    assert bool((trajectory_advantages != 0).any()) is expect_nonzero
    if expect_nonzero:
        reward_mask = torch.tensor(rewards, dtype=torch.bool)
        assert (trajectory_advantages[reward_mask] > 0).all()
        assert (trajectory_advantages[~reward_mask] < 0).all()
        torch.testing.assert_close(trajectory_advantages.mean(), torch.tensor(0.0), atol=1e-6, rtol=0)
    else:
        torch.testing.assert_close(trajectory_advantages, torch.zeros(8))
        assert diagnostics["group/fraction_zero_variance_groups"] == 1.0


def test_trajectory_balanced_clipped_loss_weights_short_and_long_episodes_equally():
    log_ratio = torch.log(torch.tensor([1.02, 0.98, 1.00, 1.01]))
    advantages = torch.ones(4)
    valids = torch.ones(4)
    weights = torch.tensor([1.0, 1 / 3, 1 / 3, 1 / 3])

    loss, _metrics = clipped_policy_loss(
        log_ratio,
        advantages,
        valids,
        clip_coef=0.5,
        loss_weights=weights,
        normalization=weights.sum(),
    )

    expected = -(torch.tensor(1.02) + torch.tensor([0.98, 1.00, 1.01]).mean()) / 2
    torch.testing.assert_close(loss, expected)


def test_hierarchical_loss_and_gradient_are_invariant_to_physical_microbatch_boundaries():
    # Two G=8 groups with deliberately different trajectory lengths. Each
    # trajectory contributes total weight one, so each group contributes eight
    # regardless of how its chunks are split across physical micro-batches.
    chunk_counts = [1, 2, 3, 4, 1, 3, 2, 4, 4, 2, 1, 3, 2, 4, 3, 1]
    advantages = []
    features = []
    weights = []
    for trajectory_index, chunk_count in enumerate(chunk_counts):
        group_advantage = 1.0 if trajectory_index % 3 else -0.5
        for chunk_index in range(chunk_count):
            advantages.append(group_advantage)
            features.append(0.01 * (1 + trajectory_index + chunk_index))
            weights.append(1.0 / chunk_count)

    advantages = torch.tensor(advantages)
    features = torch.tensor(features)
    weights = torch.tensor(weights)
    valids = torch.ones_like(weights)
    normalization = weights.sum()
    torch.testing.assert_close(normalization, torch.tensor(16.0))

    unsplit_parameter = torch.nn.Parameter(torch.tensor(0.2))
    unsplit_loss, _ = clipped_policy_loss(
        unsplit_parameter * features,
        advantages,
        valids,
        clip_coef=0.5,
        loss_weights=weights,
        normalization=normalization,
    )
    unsplit_loss.backward()

    split_parameter = torch.nn.Parameter(torch.tensor(0.2))
    split_loss = torch.zeros(())
    # Boundaries intentionally cut through trajectories and groups.
    boundaries = [0, 3, 11, 19, len(weights)]
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        micro_loss, _ = clipped_policy_loss(
            split_parameter * features[start:end],
            advantages[start:end],
            valids[start:end],
            clip_coef=0.5,
            loss_weights=weights[start:end],
            normalization=normalization,
        )
        split_loss = split_loss + micro_loss.detach()
        micro_loss.backward()

    torch.testing.assert_close(split_loss, unsplit_loss.detach(), atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(split_parameter.grad, unsplit_parameter.grad, atol=1e-7, rtol=1e-6)


def test_multi_group_grpo_normalizes_independently_and_balances_groups():
    valids = torch.zeros(16, 4)
    for index in range(16):
        valids[index, : 1 + index % 4] = 1
    rewards = [1] + [0] * 7 + [1] * 7 + [0]
    group_ids = np.asarray([["g0"] * 4 for _ in range(8)] + [["g1"] * 4 for _ in range(8)], dtype=object)
    actor_input = DataProto.from_dict(
        tensors={
            "info.trajectory_outcome": torch.tensor(rewards, dtype=torch.float32),
            "info.valids": valids,
        },
        non_tensors={"obs.group_id": group_ids},
    )
    records = [
        {
            "group_id": "g0",
            "group_key": {"layout_id": 0},
            "rollout_policy_version": 3,
            "mixed": True,
        },
        {
            "group_id": "g1",
            "group_key": {"layout_id": 1},
            "rollout_policy_version": 3,
            "mixed": True,
        },
    ]

    diagnostics = apply_grpo_outcome_advantages(actor_input, group_records=records, group_size=8)

    advantages = actor_input.batch["fpo.trajectory_group_advantage"].reshape(2, 8)
    torch.testing.assert_close(advantages.mean(dim=1), torch.zeros(2), atol=1e-6, rtol=0)
    torch.testing.assert_close(
        actor_input.batch["fpo.loss_weights"].sum(dim=1).reshape(2, 8).sum(dim=1),
        torch.tensor([8.0, 8.0]),
    )
    assert diagnostics["groups"][0]["advantage_vector"] != diagnostics["groups"][1]["advantage_vector"]
    assert actor_input.meta_info["accepted_groups"] == 2
    assert actor_input.meta_info["rollout_policy_version"] == 3

    records[1]["rollout_policy_version"] = 4
    with pytest.raises(ValueError, match="exactly one theta_old"):
        apply_grpo_outcome_advantages(actor_input, group_records=records, group_size=8)


def test_task_equal_reduction_equalizes_tasks_without_changing_per_group_advantages():
    # Task A contributes one accepted group; task B contributes two. The
    # top-level task-equal reduction must assign total weight G=8 to each task,
    # hence group weights [8, 4, 4], while GRPO normalization remains per group.
    group_size = 8
    records = [
        {
            "group_id": "a0",
            "task_name": "task_a",
            "rollout_policy_version": 0,
            "mixed": True,
        },
        {
            "group_id": "b0",
            "task_name": "task_b",
            "rollout_policy_version": 0,
            "mixed": True,
        },
        {
            "group_id": "b1",
            "task_name": "task_b",
            "rollout_policy_version": 0,
            "mixed": True,
        },
    ]
    trajectory_count = len(records) * group_size
    valids = torch.zeros(trajectory_count, 4)
    for index in range(trajectory_count):
        valids[index, : 1 + index % 4] = 1
    rewards = torch.tensor(([1] + [0] * 7) * len(records), dtype=torch.float32)
    group_ids = np.asarray(
        [[record["group_id"]] * 4 for record in records for _ in range(group_size)],
        dtype=object,
    )
    actor_input = DataProto.from_dict(
        tensors={"info.trajectory_outcome": rewards, "info.valids": valids},
        non_tensors={"obs.group_id": group_ids},
    )

    apply_grpo_outcome_advantages(
        actor_input,
        group_records=records,
        group_size=group_size,
        accepted_group_reduction="task_equal",
    )

    trajectory_weight = actor_input.batch["fpo.loss_weights"].sum(dim=1)
    group_weight = trajectory_weight.reshape(3, group_size).sum(dim=1)
    torch.testing.assert_close(group_weight, torch.tensor([8.0, 4.0, 4.0]))
    torch.testing.assert_close(group_weight[0], group_weight[1:].sum())
    assert actor_input.meta_info["accepted_group_reduction"] == "task_equal"
    assert actor_input.meta_info["task_group_counts"] == {"task_a": 1, "task_b": 2}
    assert actor_input.meta_info["reduction"].startswith("mean_task")


def test_task_equal_reduction_tolerates_float32_accumulation_for_uneven_large_batch():
    # Mirrors the first fixed-96 mainline batch: 49 accepted groups split
    # unevenly across four tasks.  Repeated float32 1/M_t scales need not sum
    # bitwise to G=8, but every task must retain equal top-level weight.
    group_size = 8
    task_counts = {"stack": 8, "fold": 12, "bottles": 19, "conveyor": 10}
    records = [
        {
            "group_id": f"{task_name}_{group_index}",
            "task_name": task_name,
            "rollout_policy_version": 1,
            "mixed": True,
        }
        for task_name, count in task_counts.items()
        for group_index in range(count)
    ]
    trajectory_count = len(records) * group_size
    valids = torch.ones(trajectory_count, 1)
    rewards = torch.tensor(([1] + [0] * 7) * len(records), dtype=torch.float32)
    group_ids = np.asarray(
        [[record["group_id"]] for record in records for _ in range(group_size)],
        dtype=object,
    )
    actor_input = DataProto.from_dict(
        tensors={"info.trajectory_outcome": rewards, "info.valids": valids},
        non_tensors={"obs.group_id": group_ids},
    )

    apply_grpo_outcome_advantages(
        actor_input,
        group_records=records,
        group_size=group_size,
        accepted_group_reduction="task_equal",
    )

    group_weights = actor_input.batch["fpo.loss_weights"].sum(dim=1).reshape(-1, group_size).sum(dim=1)
    offset = 0
    for count in task_counts.values():
        torch.testing.assert_close(
            group_weights[offset : offset + count].sum(),
            torch.tensor(float(group_size)),
            atol=1e-5,
            rtol=0,
        )
        offset += count


def test_distributed_gradient_mean_denominator_accounts_for_fsdp_gradient_average(monkeypatch):
    def fake_all_reduce(value, op):
        assert op == torch.distributed.ReduceOp.SUM
        # This rank owns weight 3 and the other rank owns weight 5.
        value.fill_(8.0)

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    denominator = distributed_gradient_mean_denominator(torch.tensor([1.0, 2.0]))

    # FSDP averages the two rank gradients, so each local numerator is divided
    # by global_weight/world_size=4, yielding (numerator_0+numerator_1)/8.
    torch.testing.assert_close(denominator, torch.tensor(4.0))


def test_task_gradient_gram_and_counterfactual_combination_are_exact():
    gradients = {
        "a": {"p0": torch.tensor([3.0, 0.0]), "p1": torch.tensor([0.0])},
        "b": {"p0": torch.tensor([0.0, 4.0]), "p1": torch.tensor([0.0])},
    }
    assert _gradient_mapping_dot(gradients["a"], gradients["b"]) == 0.0
    norm, projections = _gradient_mapping_linear_combination_stats(
        gradients,
        {"a": 0.5, "b": 0.5},
    )
    assert norm == pytest.approx(2.5)
    assert projections == pytest.approx({"a": 4.5, "b": 8.0})


def test_group_relative_fpo_surrogate_backward_is_finite():
    current_loss = torch.nn.Parameter(torch.tensor([0.0, 0.0]))
    old_loss = torch.zeros(2, 1, 1)
    mask = torch.ones(2, 1)
    log_ratio = compute_cfm_log_ratio(old_loss, current_loss[:, None, None], mask)
    loss, _metrics = clipped_policy_loss(
        log_ratio,
        torch.tensor([1.0, -1.0]),
        torch.ones(2),
        clip_coef=0.1,
        loss_weights=torch.ones(2),
        normalization=torch.tensor(2.0),
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert current_loss.grad is not None
    assert torch.isfinite(current_loss.grad).all()


def test_fpo_plus_plus_retains_mc_dimension_and_masks_unexecuted_actions():
    old = torch.zeros(1, 4, 2)
    current = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [100.0, 100.0], [100.0, 100.0]]])
    action_valids = torch.tensor([[1.0, 1.0, 0.0, 0.0]])

    log_ratio = compute_cfm_per_sample_log_ratio(old, current, action_valids, clamp_max=10.0)

    torch.testing.assert_close(log_ratio, torch.tensor([[-4.0, -6.0]]))


def test_fpo_plus_plus_log_ratio_clamp_preserves_gradient():
    old = torch.zeros(1, 1, 1)
    current = torch.nn.Parameter(torch.tensor([[[20.0]]]))

    log_ratio = compute_cfm_per_sample_log_ratio(
        old,
        current,
        torch.ones(1, 1),
        clamp_max=5.0,
    )
    torch.testing.assert_close(log_ratio, torch.tensor([[-5.0]]))
    log_ratio.sum().backward()
    torch.testing.assert_close(current.grad, torch.tensor([[[-1.0]]]))


def test_fpo_plus_plus_log_ratio_clamp_can_be_disabled():
    old = torch.zeros(1, 1, 1)
    current = torch.tensor([[[20.0]]])

    log_ratio = compute_cfm_per_sample_log_ratio(
        old,
        current,
        torch.ones(1, 1),
        clamp_max=None,
    )

    torch.testing.assert_close(log_ratio, torch.tensor([[-20.0]]))


def test_fpo_plus_plus_ppo_averages_per_mc_surrogates_without_changing_chunk_weight():
    log_ratio = torch.log(torch.tensor([[1.2, 0.8], [1.0, 1.0]]))
    advantages = torch.tensor([1.0, -1.0])
    valids = torch.ones(2)

    loss, metrics = fpo_plus_plus_policy_loss(
        log_ratio,
        advantages,
        valids,
        clip_coef=0.1,
        trust_region_mode="ppo",
    )

    # First chunk: mean(-1.1, -0.8); second: mean(1, 1). Then mean chunks.
    torch.testing.assert_close(loss, torch.tensor(0.025))
    assert metrics["ratio"].shape == (2, 2)
    assert torch.isfinite(metrics["approx_kl"])


def test_fpo_plus_plus_aspo_keeps_negative_advantage_pullback_gradient():
    log_ratio = torch.nn.Parameter(torch.log(torch.tensor([[1.5]])))
    loss, _ = fpo_plus_plus_policy_loss(
        log_ratio,
        torch.tensor([-1.0]),
        torch.ones(1),
        clip_coef=0.1,
        trust_region_mode="aspo",
        spo_clip_coef=0.1,
    )
    loss.backward()

    assert log_ratio.grad is not None
    assert torch.isfinite(log_ratio.grad).all()
    assert not torch.allclose(log_ratio.grad, torch.zeros_like(log_ratio.grad))


def test_candidate_window_attempts_resume_across_process_restart(tmp_path):
    evidence = tmp_path / "metrics.jsonl"
    records = [
        {
            "record_type": "candidate_group",
            "intended_update": 2,
            "rollout_policy_version": 1,
            "candidate_attempt": 1,
            "accepted": False,
        }
    ]
    evidence.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    trainer = object.__new__(GRFPORayTrainer)
    trainer.global_steps = 1
    trainer.trainer_config = SimpleNamespace(evidence_jsonl=str(evidence))

    resumed = trainer._resume_candidate_records(intended_update=2)

    assert [record["candidate_attempt"] for record in resumed] == [1]

    records.append(
        {
            "record_type": "no_informative_group",
            "intended_update": 2,
            "rollout_policy_version": 1,
        }
    )
    evidence.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    assert trainer._resume_candidate_records(intended_update=2) == []


def test_accepted_candidate_spool_roundtrip_and_resume_cursor(tmp_path):
    trainer = object.__new__(GRFPORayTrainer)
    trainer.trainer_config = SimpleNamespace(
        candidate_spool_dir=str(tmp_path / "spool"),
        candidate_resume_state_path=str(tmp_path / "resume.json"),
        candidate_group_partition="stage",
        group_size=2,
    )
    trainer.config = SimpleNamespace(
        cluster=SimpleNamespace(
            actor_rollout_ref=SimpleNamespace(
                model=SimpleNamespace(
                    path="/immutable/base",
                    adapter=SimpleNamespace(checkpoint_sha256="abc123"),
                )
            ),
            env=SimpleNamespace(
                env_loop=SimpleNamespace(pipeline_stage_num=3),
                env_worker=SimpleNamespace(num_envs=2),
            ),
        )
    )
    rollout = DataProto.from_dict(tensors={"x": torch.arange(4).reshape(2, 2)})
    end_obs = DataProto.from_dict(tensors={"x": torch.arange(2).reshape(2, 1)})
    record = {
        "group_id": "run-a/policy-0000/candidate-0001",
        "intended_update": 1,
        "rollout_policy_version": 0,
        "candidate_attempt": 1,
    }

    trainer._persist_accepted_candidate(rollout, end_obs, record)
    restored_rollout, restored_end_obs, restored_record = trainer._load_spooled_candidate(record)

    torch.testing.assert_close(restored_rollout.batch["x"], rollout.batch["x"])
    torch.testing.assert_close(restored_end_obs.batch["x"], end_obs.batch["x"])
    assert restored_record["candidate_spool_path"] == record["candidate_spool_path"]

    trainer._write_candidate_resume_state(
        rollout_policy_version=0,
        intended_update=1,
        next_candidate_attempt=7,
        candidate_schedule_slot=6,
    )
    state = trainer._read_candidate_resume_state()
    assert state["next_candidate_attempt"] == 7
    assert state["candidate_schedule_slot"] == 6
    # Two complete reset waves * two physical lanes per worker.
    assert state["train_case_cursor_start"] == 4

    trainer._cleanup_candidate_spool_window(intended_update=1, rollout_policy_version=0)
    assert not (tmp_path / "spool" / "update_0001_theta_0000").exists()


def test_critic_free_worker_scores_old_cfm_without_value_or_gae(monkeypatch):
    class Module:
        value_calls = 0

        def fpo_forward_value(self, *args, **kwargs):
            self.value_calls += 1
            raise AssertionError("critic path must not execute")

        def fpo_cfm_loss(self, obs, tokenizer, actions, timesteps, noise, **kwargs):
            del kwargs
            del obs, tokenizer, noise
            return torch.zeros(actions.shape[0], actions.shape[1], timesteps.shape[1])

    worker = object.__new__(FPOTrainingWorker)
    worker.actor_config = SimpleNamespace(
        value=SimpleNamespace(enabled=False),
        n_action_samples=2,
        micro_batch_size=1,
    )
    worker.tokenizer = None
    module = Module()
    worker.engine = SimpleNamespace(module=module)
    monkeypatch.setattr(training_worker_module, "get_device_id", lambda: "cpu")
    monkeypatch.setattr(training_worker_module, "get_device_name", lambda: "cpu")
    data = DataProto.from_dict(
        tensors={
            "obs.state": torch.zeros(2, 1, 14),
            "action.action": torch.zeros(2, 1, 24, 14),
            "action.full_action": torch.zeros(2, 1, 32, 14),
            "info.valids": torch.ones(2, 1),
            "fpo.advantages": torch.tensor([[1.0], [-1.0]]),
            "fpo.loss_weights": torch.ones(2, 1),
        },
        meta_info={"advantage_estimator": "grpo_outcome"},
    )

    prepared = worker._prepare_training_batch(data)

    assert module.value_calls == 0
    assert "fpo.old_values" not in prepared.batch
    assert "fpo.returns" not in prepared.batch
    assert prepared.batch["fpo.old_cfm_loss"].shape == (2, 32, 2)


def test_identity_gate_skips_leading_padding_microbatch(monkeypatch):
    class Module(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.delta = torch.nn.Parameter(torch.tensor(0.0))

        def fpo_cfm_loss(self, obs, tokenizer, actions, timesteps, noise, **kwargs):
            del kwargs
            del obs, tokenizer, noise
            return self.delta * torch.ones(actions.shape[0], actions.shape[1], timesteps.shape[1])

        def set_requires_gradient_sync(self, enabled):
            del enabled

        def set_is_last_backward(self, enabled):
            del enabled

    class Engine:
        def __init__(self):
            self.module = Module()
            self.optimizer = SimpleNamespace(param_groups=[{"lr": 1.0e-5}])

        def optimizer_zero_grad(self):
            self.module.delta.grad = None

        def optimizer_step(self):
            return torch.tensor(0.0)

        def lr_scheduler_step(self):
            pass

    worker = object.__new__(FPOTrainingWorker)
    worker.actor_config = SimpleNamespace(
        value=SimpleNamespace(enabled=False),
        n_action_samples=1,
        micro_batch_size=1,
        full_logical_batch_gradient_accumulation=True,
        update_epochs=1,
        normalize_advantages=False,
        clip_coef=0.1,
        target_kl=None,
        vf_coef=0.0,
        value_only_updates=0,
    )
    worker.tokenizer = None
    worker.engine = Engine()
    worker.value_optimizer = None
    worker.value_scheduler = None
    worker._identity_gate_has_passed = False

    monkeypatch.setattr(training_worker_module, "get_device_id", lambda: None)
    monkeypatch.setattr(training_worker_module, "get_device_name", lambda: "cpu")
    monkeypatch.setattr(training_worker_module, "_synchronize_for_timing", lambda: None)
    monkeypatch.setattr(
        training_worker_module,
        "distributed_masked_stats",
        lambda values, valids: {"mean": 1.0, "std": 0.0, "min": 1.0, "max": 1.0},
    )
    monkeypatch.setattr(
        training_worker_module,
        "distributed_gradient_mean_denominator",
        lambda weights: weights.sum(),
    )
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda tensor, op=None: None)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(torch, "randperm", lambda size: torch.tensor([1, 0]))

    valids = torch.tensor([[1.0, 0.0]])
    action_valids = torch.zeros(1, 2, 32)
    action_valids[:, 0, :24] = 1.0
    data = DataProto.from_dict(
        tensors={
            "obs.state": torch.zeros(1, 2, 14),
            "action.action": torch.zeros(1, 2, 24, 14),
            "action.full_action": torch.zeros(1, 2, 32, 14),
            "info.valids": valids,
            "info.action_valids": action_valids,
            "info.rewards": torch.tensor([[1.0, 0.0]]),
            "fpo.advantages": torch.tensor([[1.0, 0.0]]),
            "fpo.loss_weights": valids.clone(),
        },
        meta_info={"advantage_estimator": "grpo_outcome", "global_steps": 5},
    )

    metrics = worker._update_fpo_policy(data)

    assert worker._identity_gate_has_passed is True
    assert metrics["fpo/pre_optimizer_identity_max_abs_log_ratio"] == 0.0
    assert metrics["fpo/pre_optimizer_identity_max_abs_cfm_error"] == 0.0


def test_critic_free_actor_config_rejects_second_advantage_normalization():
    with pytest.raises(ValueError, match="normalize_advantages=false"):
        FPOActorConfig(
            value=FPOValueConfig(enabled=False),
            vf_coef=0.0,
            value_only_updates=0,
            normalize_advantages=True,
        )


def test_fpo_plus_plus_config_is_explicit_and_aspo_cannot_leak_into_vanilla():
    config = FPOActorConfig(
        value=FPOValueConfig(enabled=False),
        vf_coef=0.0,
        value_only_updates=0,
        normalize_advantages=False,
        fpo_variant="fpo_plus_plus",
        trust_region_mode="ppo",
        cfm_loss_kernel="huber",
        log_ratio_clamp=5.0,
    )
    assert config.fpo_variant == "fpo_plus_plus"
    assert config.trust_region_mode == "ppo"

    with pytest.raises(ValueError, match="ASPO is only defined"):
        FPOActorConfig(fpo_variant="vanilla", trust_region_mode="aspo")


def test_pre_optimizer_kl_config_is_explicit_and_critic_free():
    config = FPOActorConfig(
        value=FPOValueConfig(enabled=False),
        vf_coef=0.0,
        value_only_updates=0,
        normalize_advantages=False,
        full_logical_batch_gradient_accumulation=True,
        target_kl=5e-4,
        kl_early_stop_mode="pre_optimizer",
        fpo_mc_seed=17001,
        fpo_shuffle_seed=17002,
        post_resume_lr_override=5e-6,
    )
    assert config.kl_early_stop_mode == "pre_optimizer"
    assert config.fpo_mc_seed == 17001
    assert config.fpo_shuffle_seed == 17002
    assert config.post_resume_lr_override == 5e-6

    with pytest.raises(ValueError, match="critic-free"):
        FPOActorConfig(kl_early_stop_mode="pre_optimizer", target_kl=5e-4)
    with pytest.raises(ValueError, match="full_logical_batch_gradient_accumulation"):
        FPOActorConfig(
            value=FPOValueConfig(enabled=False),
            vf_coef=0.0,
            value_only_updates=0,
            normalize_advantages=False,
            kl_early_stop_mode="pre_optimizer",
            target_kl=5e-4,
        )


def test_score_reward_variance_group_config_keeps_partial_threshold_explicit():
    config = GRFPOTrainerConfig(
        advantage_estimator="grpo_outcome",
        informative_group_sampling=True,
        group_reward_source="process_score",
        informative_group_criterion="reward_variance",
        min_partial_trajectories_for_score_only_group=1,
        partial_score_threshold=0.15,
    )
    assert config.group_reward_source == "process_score"
    assert config.informative_group_criterion == "reward_variance"
    assert config.min_partial_trajectories_for_score_only_group == 1

    with pytest.raises(ValueError, match="requires group_reward_source='process_score'"):
        GRFPOTrainerConfig(
            advantage_estimator="grpo_outcome",
            informative_group_criterion="reward_variance",
        )


def test_process_score_full_success_mixed_config_is_explicit():
    config = GRFPOTrainerConfig(
        advantage_estimator="grpo_outcome",
        informative_group_sampling=True,
        group_reward_source="process_score",
        informative_group_criterion="full_success_mixed",
    )
    assert config.group_reward_source == "process_score"
    assert config.informative_group_criterion == "full_success_mixed"
