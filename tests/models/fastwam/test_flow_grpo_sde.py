from __future__ import annotations

import math

import torch

from fastwam.models.wan22.schedulers.flow_grpo_sde import flow_grpo_sde_step_with_logprob


def test_flow_grpo_sde_matches_official_equations_and_is_replayable():
    torch.manual_seed(7)
    sample = torch.randn(2, 3, 4)
    velocity = torch.randn_like(sample, requires_grad=True)
    noise = torch.randn_like(sample)
    sigma = torch.tensor([1.0, 0.6])
    sigma_next = torch.tensor([0.8, 0.3])
    noise_level = 0.7

    sampled, sampled_log_prob, sampled_mean = flow_grpo_sde_step_with_logprob(
        model_output=velocity,
        sigma=sigma,
        sigma_next=sigma_next,
        sample=sample,
        noise_level=noise_level,
        variance_noise=noise,
    )
    replayed, replayed_log_prob, replayed_mean = flow_grpo_sde_step_with_logprob(
        model_output=velocity,
        sigma=sigma,
        sigma_next=sigma_next,
        sample=sample,
        noise_level=noise_level,
        next_sample=sampled.detach(),
    )

    torch.testing.assert_close(replayed, sampled)
    torch.testing.assert_close(replayed_mean, sampled_mean)
    torch.testing.assert_close(replayed_log_prob, sampled_log_prob)

    # Independent transcription of the official first-row equation, including
    # its sigma=1 denominator substitution with the next schedule sigma.
    dt = sigma_next[0] - sigma[0]
    std_dev = torch.sqrt(sigma[0] / (1.0 - sigma_next[0])) * noise_level
    expected_mean = sample[0] * (1.0 + std_dev**2 / (2.0 * sigma[0]) * dt)
    expected_mean += velocity[0] * (
        1.0 + std_dev**2 * (1.0 - sigma[0]) / (2.0 * sigma[0])
    ) * dt
    transition_std = std_dev * torch.sqrt(-dt)
    expected_log_prob = (
        -((sampled[0].detach() - expected_mean) ** 2) / (2.0 * transition_std**2)
        - torch.log(transition_std)
        - 0.5 * math.log(2.0 * math.pi)
    )
    torch.testing.assert_close(sampled_mean[0], expected_mean)
    torch.testing.assert_close(sampled_log_prob[0], expected_log_prob)

    replayed_log_prob.mean().backward()
    assert velocity.grad is not None
    assert torch.isfinite(velocity.grad).all()


def test_flow_grpo_sde_rejects_non_decreasing_schedule():
    value = torch.zeros(1, 2, 3)
    try:
        flow_grpo_sde_step_with_logprob(
            model_output=value,
            sigma=torch.tensor([0.5]),
            sigma_next=torch.tensor([0.5]),
            sample=value,
            noise_level=0.7,
        )
    except ValueError as error:
        assert "sigma_next" in str(error)
    else:
        raise AssertionError("non-decreasing Flow-GRPO schedule was accepted")


def test_flow_grpo_sde_converges_to_native_ode_step_as_noise_vanishes():
    sample = torch.tensor([[[0.2, -0.4]]])
    velocity = torch.tensor([[[0.3, 0.1]]])
    sigma = torch.tensor([0.8])
    sigma_next = torch.tensor([0.7])
    delta = sigma_next - sigma
    native_ode_next = sample + velocity * delta

    _next, _log_prob, transition_mean = flow_grpo_sde_step_with_logprob(
        model_output=velocity,
        sigma=sigma,
        sigma_next=sigma_next,
        sample=sample,
        noise_level=1.0e-6,
        variance_noise=torch.zeros_like(sample),
    )

    torch.testing.assert_close(transition_mean, native_ode_next, atol=1.0e-6, rtol=1.0e-6)
