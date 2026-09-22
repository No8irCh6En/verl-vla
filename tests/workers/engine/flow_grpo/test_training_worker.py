from __future__ import annotations

from types import SimpleNamespace

import torch
from verl import DataProto

from verl_vla.workers.engine.flow_grpo.training_worker import (
    FlowGRPOTrainingWorker,
    compute_flow_grpo_log_ratio,
    compute_replay_calibrated_flow_grpo_log_ratio,
    flow_grpo_clipped_policy_loss,
    training_transition_count,
)


def test_log_ratio_uses_current_minus_old_and_only_executed_actions():
    old = torch.zeros(2, 3, 5)
    current = torch.zeros_like(old)
    current[0, :, :2] = 0.4
    current[0, :, 2:] = 100.0
    current[1, :, :4] = -0.25
    current[1, :, 4:] = 100.0
    action_valids = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 1, 0]], dtype=torch.float32)

    ratio = compute_flow_grpo_log_ratio(old, current, action_valids)

    torch.testing.assert_close(ratio[0], torch.full((3,), 0.4))
    torch.testing.assert_close(ratio[1], torch.full((3,), -0.25))


def test_transition_average_preserves_one_weight_per_chunk():
    log_ratio = torch.zeros(2, 4, requires_grad=True)
    advantages = torch.tensor([1.0, -1.0])
    valids = torch.ones(2)
    trajectory_balanced_weights = torch.tensor([0.25, 0.75])

    loss, metrics = flow_grpo_clipped_policy_loss(
        log_ratio,
        advantages,
        valids,
        0.1,
        loss_weights=trajectory_balanced_weights,
        normalization=trajectory_balanced_weights.sum(),
    )

    # Four flow transitions do not multiply either chunk's top-level weight.
    torch.testing.assert_close(loss, torch.tensor(0.5))
    torch.testing.assert_close(metrics["ratio"], torch.ones(2, 4))
    loss.backward()
    assert log_ratio.grad is not None
    assert torch.isfinite(log_ratio.grad).all()


def test_official_transition_fraction_uses_first_nine_of_ten_steps():
    assert training_transition_count(10, 0.99) == 9
    assert training_transition_count(10, 1.0) == 10


def test_replay_calibration_freezes_theta_old_and_preserves_policy_gradient():
    rollout_old = torch.tensor(
        [[[0.0, 0.1], [0.2, -0.2]]],
    )
    replay_old = torch.tensor(
        [[[0.2, -0.1], [0.4, 0.3]]],
    )
    current = replay_old.clone().requires_grad_(True)
    action_valids = torch.tensor([[1.0, 0.0]])

    identity_log_ratio = compute_replay_calibrated_flow_grpo_log_ratio(
        rollout_old,
        replay_old,
        current,
        action_valids,
    )

    torch.testing.assert_close(identity_log_ratio, torch.zeros_like(identity_log_ratio))
    identity_log_ratio.sum().backward()
    assert current.grad is not None
    torch.testing.assert_close(current.grad[..., 0], torch.ones_like(current.grad[..., 0]))
    torch.testing.assert_close(current.grad[..., 1], torch.zeros_like(current.grad[..., 1]))

    updated_current = torch.tensor(
        [[[0.25, -0.1], [0.45, 0.3]]],
        requires_grad=True,
    )
    updated_log_ratio = compute_replay_calibrated_flow_grpo_log_ratio(
        rollout_old,
        replay_old,
        updated_current,
        action_valids,
    )
    torch.testing.assert_close(updated_log_ratio, torch.full_like(updated_log_ratio, 0.05))

    loss, metrics = flow_grpo_clipped_policy_loss(
        updated_log_ratio,
        advantages=torch.ones(1),
        valids=torch.ones(1),
        clip_coef=1.0e-3,
    )
    torch.testing.assert_close(loss, torch.tensor(-1.001), rtol=1.0e-5, atol=1.0e-6)
    torch.testing.assert_close(metrics["clip_fraction"], torch.tensor(1.0))


def test_precompacted_valid_chunks_are_not_flattened_a_second_time():
    valid_chunks, transitions, horizon, action_dim = 3, 2, 32, 14
    data = DataProto.from_dict(
        tensors={
            "action.action": torch.zeros(valid_chunks, 24, action_dim),
            "action.full_action": torch.zeros(valid_chunks, horizon, action_dim),
            "action.flow_grpo.latents": torch.zeros(valid_chunks, transitions + 1, horizon, action_dim),
            "action.flow_grpo.old_log_probs": torch.zeros(valid_chunks, transitions, horizon),
            "action.flow_grpo.sigmas": torch.zeros(valid_chunks, transitions),
            "action.flow_grpo.deltas": torch.ones(valid_chunks, transitions),
            "info.valids": torch.ones(valid_chunks, dtype=torch.bool),
            "fpo.advantages": torch.tensor([1.0, -1.0, 0.5]),
            "fpo.loss_weights": torch.tensor([0.25, 0.25, 0.5]),
        },
        meta_info={
            "advantage_estimator": "grpo_outcome",
            "fpo_preflattened_valid_chunks": True,
        },
    )
    worker = object.__new__(FlowGRPOTrainingWorker)
    worker.actor_config = SimpleNamespace(train_transition_fraction=0.99)

    flat = worker._prepare_training_batch(data)

    assert len(flat) == valid_chunks
    assert flat.batch["action.flow_grpo.latents"].shape == (
        valid_chunks,
        transitions + 1,
        horizon,
        action_dim,
    )
    torch.testing.assert_close(flat.batch["flow_grpo.reference_index"], torch.arange(valid_chunks))
