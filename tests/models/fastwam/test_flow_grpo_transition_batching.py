from __future__ import annotations

from types import SimpleNamespace

import torch
from verl import DataProto

from verl_vla.models.fastwam import FastWAMAdapterConfig, FastWAMTrainableModel


class _FakeActionExpert(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.7))

    def pre_dit(self, *, action_tokens, timestep, context, context_mask):
        del context_mask
        tokens = action_tokens * self.scale + timestep[:, None, None] * 1.0e-3
        return {
            "tokens": tokens,
            "freqs": None,
            "t_mod": None,
            "context": context,
            "context_mask": torch.ones(context.shape[:2], dtype=torch.bool),
        }

    def post_dit(self, action_tokens, action_pre):
        del action_pre
        return action_tokens


class _FakeMoT(torch.nn.Module):
    def __init__(self, action_expert: torch.nn.Module) -> None:
        super().__init__()
        self.mixtures = torch.nn.ModuleDict({"action": action_expert})
        self.forward_calls = 0

    def forward_action_with_video_cache(
        self,
        *,
        action_tokens,
        action_freqs,
        action_t_mod,
        action_context_payload,
        video_kv_cache,
        attention_mask,
        video_seq_len,
    ):
        self.forward_calls += 1
        del action_freqs, action_t_mod, attention_mask, video_seq_len
        context = action_context_payload["context"][:, :1, :1]
        cache = video_kv_cache[0]["k"][:, :1, :1]
        return action_tokens + context + cache


class _FakePolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.action_expert = _FakeActionExpert()
        self.mot = _FakeMoT(self.action_expert)
        self.infer_action_scheduler = SimpleNamespace(num_train_timesteps=1000)
        self.torch_dtype = torch.float32
        self.device = torch.device("cpu")


class _TransitionBatchingModel(FastWAMTrainableModel):
    def __init__(self, transition_batch_size: int) -> None:
        policy = _FakePolicy()
        runtime = SimpleNamespace(model=policy)
        config = FastWAMAdapterConfig(
            policy_root="/unused",
            checkpoint_sha256="0" * 64,
            flow_grpo={
                "enabled": True,
                "noise_level": 0.01,
                "transition_batch_size": transition_batch_size,
            },
        )
        super().__init__(runtime_policy=runtime, config=config, model_path="/unused")

    def _fpo_observation_conditioning(self, obs):
        batch_size = int(obs.batch.batch_size[0])
        context = torch.arange(batch_size, dtype=torch.float32).reshape(batch_size, 1, 1)
        return (
            torch.zeros(batch_size, 1),
            torch.zeros(batch_size, 1),
            context,
            torch.ones(batch_size, 1, dtype=torch.bool),
        )

    def _fpo_video_cache(self, images, context, context_mask):
        del images, context_mask
        cache = [{"k": context + 0.25, "v": context - 0.25}]
        return cache, torch.ones(1, 1, dtype=torch.bool), 1


def _flow_inputs(batch_size: int = 2, transition_count: int = 5):
    generator = torch.Generator().manual_seed(20260905)
    latents = torch.randn(
        batch_size,
        transition_count + 1,
        32,
        14,
        generator=generator,
    )
    schedule = torch.linspace(1.0, 0.1, transition_count + 1)
    sigmas = schedule[:-1].expand(batch_size, -1).clone()
    deltas = (schedule[1:] - schedule[:-1]).expand(batch_size, -1).clone()
    obs = DataProto.from_dict(tensors={"dummy": torch.zeros(batch_size, 1)})
    return obs, latents, sigmas, deltas


def test_transition_batching_matches_serial_log_probs_and_gradients():
    serial = _TransitionBatchingModel(transition_batch_size=1)
    batched = _TransitionBatchingModel(transition_batch_size=3)
    batched.load_state_dict(serial.state_dict())
    obs, latents, sigmas, deltas = _flow_inputs()

    serial_log_probs = serial.flow_grpo_log_probs(obs, None, latents, sigmas, deltas)
    batched_log_probs = batched.flow_grpo_log_probs(obs, None, latents, sigmas, deltas)

    assert serial_log_probs.shape == (2, 5, 32)
    torch.testing.assert_close(batched_log_probs, serial_log_probs, rtol=1.0e-5, atol=1.0e-5)
    assert serial.policy.mot.forward_calls == 5
    assert batched.policy.mot.forward_calls == 2

    serial_log_probs.sum().backward()
    batched_log_probs.sum().backward()
    torch.testing.assert_close(
        batched.policy.action_expert.scale.grad,
        serial.policy.action_expert.scale.grad,
        rtol=1.0e-5,
        atol=1.0e-5,
    )


def test_transition_batching_config_rejects_non_positive_values():
    try:
        _TransitionBatchingModel(transition_batch_size=0)
    except ValueError as error:
        assert "transition_batch_size" in str(error)
    else:
        raise AssertionError("transition_batch_size=0 must be rejected")
