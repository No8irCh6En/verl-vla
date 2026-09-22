from __future__ import annotations

import asyncio

import numpy as np
import pytest
import torch
from verl import DataProto

from verl_vla.models.base import SupportFPOTraining
from verl_vla.models.fastwam import FastWAMAdapterConfig, FastWAMOutput, FastWAMTrainableModel
from verl_vla.workers.rollout.hf_rollout import HFRollout


class FakeNativePolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
        self.batch_calls = 0
        self.last_images = None
        self.last_proprio = None
        self.last_seeds = None
        self.last_action_latent_scale = None
        self.last_flow_grpo_noise_level = None
        self.last_return_flow_grpo_trace = None
        self.action_expert = torch.nn.Identity()
        self.mot = type("FakeMoTAlias", (), {"mixtures": {"action": self.action_expert}})()

    def infer_action_batch(self, *, input_image, proprio, **kwargs):
        self.last_seeds = kwargs.get("seeds")
        self.last_action_latent_scale = kwargs.get("action_latent_scale")
        self.last_flow_grpo_noise_level = kwargs.get("flow_grpo_noise_level")
        self.last_return_flow_grpo_trace = kwargs.get("return_flow_grpo_trace")
        self.batch_calls += 1
        self.last_images = input_image.clone()
        self.last_proprio = proprio.clone()
        batch = input_image.shape[0]
        action = torch.arange(batch * 32 * 14, dtype=torch.float32).reshape(batch, 32, 14) + self.anchor
        result = {"action": action}
        if self.last_return_flow_grpo_trace:
            transition_count = 10
            result["flow_grpo"] = {
                "latents": torch.zeros(batch, transition_count + 1, 32, 14),
                "old_log_probs": torch.zeros(batch, transition_count, 32),
                "sigmas": torch.ones(batch, transition_count),
                "deltas": -torch.full((batch, transition_count), 0.1),
            }
        return result


class FakeRuntime:
    def __init__(self):
        self.model = FakeNativePolicy()
        self.legacy_calls = 0

    def _build_robotwin_image_tensor(self, row):
        head = float(row["observation"]["head_camera"]["rgb"][0, 0, 0])
        left = float(row["observation"]["left_camera"]["rgb"][0, 0, 0])
        right = float(row["observation"]["right_camera"]["rgb"][0, 0, 0])
        return torch.tensor([head, left, right], dtype=torch.float32).reshape(1, 3, 1, 1)

    def _normalize_state(self, state):
        return torch.from_numpy(np.asarray(state, dtype=np.float32)).unsqueeze(0) + 1.0

    def _denormalize_action(self, action):
        return action.detach().cpu().float().numpy() + 10.0

    def _infer_action_chunk(self, row, task):
        del row, task
        self.legacy_calls += 1
        return np.arange(32 * 14, dtype=np.float32).reshape(32, 14)

    def reset(self):
        pass


def _config(**overrides) -> FastWAMAdapterConfig:
    return FastWAMAdapterConfig(
        policy_root="/unused",
        checkpoint_sha256="0" * 64,
        **overrides,
    )


def _model() -> tuple[FastWAMTrainableModel, FakeRuntime]:
    runtime = FakeRuntime()
    model = FastWAMTrainableModel(runtime_policy=runtime, config=_config(), model_path="/unused")
    return model, runtime


def test_fastwam_declares_non_generative_checkpoint_contract():
    model, _runtime = _model()

    assert model.can_generate() is False


def _obs(batch_size: int) -> DataProto:
    images = []
    for camera_value in (11, 22, 33):
        image = torch.full((batch_size, 240, 320, 3), camera_value, dtype=torch.uint8)
        images.append(image)
    states = torch.arange(batch_size * 14, dtype=torch.float32).reshape(batch_size, 14)
    return DataProto.from_dict(
        tensors={
            "observation.images.cam_head": images[0],
            "observation.images.cam_left_wrist": images[1],
            "observation.images.cam_right_wrist": images[2],
            "observation.state": states,
        },
        non_tensors={"task": np.asarray([f"task-{idx}" for idx in range(batch_size)], dtype=object)},
    )


def _obs_with_policy_seeds(seeds: list[int]) -> DataProto:
    obs = _obs(len(seeds))
    obs.non_tensor_batch["policy_seed"] = np.asarray(seeds, dtype=np.int64)
    return obs


def test_b1_uses_legacy_oracle_and_preserves_full_horizon():
    model, runtime = _model()
    result = model.sac_sample_actions(_obs(1))
    expected = torch.arange(32 * 14, dtype=torch.float32).reshape(1, 32, 14)

    assert runtime.legacy_calls == 1
    assert runtime.model.batch_calls == 0
    assert result.full_action.shape == (1, 32, 14)
    assert result.action.shape == (1, 24, 14)
    torch.testing.assert_close(result.full_action, expected, rtol=0, atol=0)
    torch.testing.assert_close(result.action, expected[:, :24], rtol=0, atol=0)
    assert torch.isfinite(result.full_action).all()
    output_keys = set(result.to_data_proto().batch.keys())
    assert {"action", "full_action", "profile.inference_seconds"}.issubset(output_keys)
    assert result.inference_seconds is not None
    assert result.inference_seconds >= 0


@pytest.mark.parametrize("batch_size", [2, 4])
def test_true_batch_calls_native_flow_once(batch_size):
    model, runtime = _model()
    result = model.sac_sample_actions(_obs(batch_size))

    assert runtime.legacy_calls == 0
    assert runtime.model.batch_calls == 1
    assert model.loaded_policy_replicas == 1
    assert result.full_action.shape == (batch_size, 32, 14)
    assert result.action.shape == (batch_size, 24, 14)
    assert torch.isfinite(result.full_action).all()
    assert not torch.equal(result.full_action[0], result.full_action[1])
    # Fake normalizer proves the batched denormalization boundary ran once.
    assert float(result.full_action.min()) == 10.0
    torch.testing.assert_close(
        runtime.model.last_proprio,
        _obs(batch_size).batch["observation.state"] + 1.0,
    )


def test_batch_rollout_forwards_one_explicit_policy_seed_per_lane():
    model, runtime = _model()
    model.sac_sample_actions(_obs_with_policy_seeds([101, 202]))

    assert runtime.model.last_seeds == [101, 202]


def test_rollout_action_latent_scale_is_explicit_and_forwards_to_native_flow():
    runtime = FakeRuntime()
    model = FastWAMTrainableModel(
        runtime_policy=runtime,
        config=_config(rollout_action_latent_scale=1.35),
        model_path="/unused",
    )
    model.sac_sample_actions(_obs_with_policy_seeds([101, 202]))

    assert runtime.model.last_action_latent_scale == pytest.approx(1.35)
    assert runtime.model.last_seeds == [101, 202]


def test_flow_grpo_sde_trace_is_training_only_and_eval_uses_deterministic_flow():
    runtime = FakeRuntime()
    model = FastWAMTrainableModel(
        runtime_policy=runtime,
        config=_config(flow_grpo={"enabled": True, "noise_level": 0.7}),
        model_path="/unused",
    )
    obs = _obs_with_policy_seeds([101, 202])

    train_output = model.sac_sample_actions(obs, eval=False)
    assert runtime.model.last_flow_grpo_noise_level == pytest.approx(0.7)
    assert runtime.model.last_return_flow_grpo_trace is True
    assert train_output.flow_grpo_latents.shape == (2, 11, 32, 14)
    assert "flow_grpo.old_log_probs" in train_output.to_data_proto().batch

    eval_output = model.sac_sample_actions(obs, eval=True)
    assert runtime.model.last_flow_grpo_noise_level is None
    assert runtime.model.last_return_flow_grpo_trace is False
    assert eval_output.flow_grpo_latents is None


@pytest.mark.parametrize("scale", [0.0, -1.0, float("inf"), float("nan")])
def test_rollout_action_latent_scale_must_be_finite_and_positive(scale):
    with pytest.raises(ValueError, match="rollout_action_latent_scale"):
        _config(rollout_action_latent_scale=scale)


def test_per_policy_call_seed_mode_is_fresh_and_reproducible():
    def sample_two_calls():
        runtime = FakeRuntime()
        model = FastWAMTrainableModel(
            runtime_policy=runtime,
            config=_config(rollout_seed_mode="per_policy_call"),
            model_path="/unused",
        )
        obs = _obs_with_policy_seeds([101, 202])
        model.sac_sample_actions(obs)
        first = list(runtime.model.last_seeds)
        model.sac_sample_actions(obs)
        second = list(runtime.model.last_seeds)
        return first, second

    first, second = sample_two_calls()
    assert first != second
    assert (first, second) == sample_two_calls()


def test_rollout_seed_mode_rejects_unknown_modes():
    with pytest.raises(ValueError, match="rollout_seed_mode"):
        _config(rollout_seed_mode="temperature")


def test_camera_and_proprio_order_are_exact():
    model, _runtime = _model()
    rows, tasks = model._canonical_rows(_obs(2))

    assert tasks == ["task-0", "task-1"]
    assert rows[0]["observation"]["head_camera"]["rgb"][0, 0, 0] == 11
    assert rows[0]["observation"]["left_camera"]["rgb"][0, 0, 0] == 22
    assert rows[0]["observation"]["right_camera"]["rgb"][0, 0, 0] == 33
    np.testing.assert_array_equal(rows[1]["joint_action"]["vector"], np.arange(14, 28, dtype=np.float32))


def test_output_rejects_32_step_execution_regression():
    full_action = torch.zeros((1, 32, 14))
    with pytest.raises(ValueError, match=r"\[B,24,14\]"):
        FastWAMOutput(action=full_action, full_action=full_action)


def test_adapter_config_locks_32_to_24_semantics():
    base = dict(policy_root="/unused", checkpoint_sha256="0" * 64)
    with pytest.raises(ValueError, match="must remain 32"):
        FastWAMAdapterConfig(**base, action_horizon=31)
    with pytest.raises(ValueError, match="must remain 24"):
        FastWAMAdapterConfig(**base, action_chunk_size=32)


def test_adapter_config_accepts_hydra_nested_mapping():
    from omegaconf import OmegaConf

    config = FastWAMAdapterConfig(
        policy_root="/unused",
        checkpoint_sha256="0" * 64,
        fpo=OmegaConf.create({"enabled": True, "value_enabled": False}),
    )

    assert config.fpo.enabled is True
    assert config.fpo.value_enabled is False


def test_fastwam_rejects_legacy_value_head_configuration():
    with pytest.raises(ValueError, match="critic-free"):
        FastWAMAdapterConfig(
            policy_root="/unused",
            checkpoint_sha256="0" * 64,
            fpo={"enabled": True, "value_enabled": True},
        )


class FakeNormalizer:
    def forward(self, tensor):
        return tensor * 0.5


class FakeActionExpert(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.75))

    def pre_dit(self, *, action_tokens, timestep, context, context_mask):
        del timestep
        return {
            "tokens": action_tokens * self.scale,
            "freqs": torch.zeros(1),
            "t_mod": torch.zeros(1),
            "context": context,
            "context_mask": context_mask,
        }

    def post_dit(self, tokens, pre_state):
        del pre_state
        return tokens


class FakeVideoExpert(torch.nn.Module):
    fuse_vae_embedding_in_latents = False

    def __init__(self):
        super().__init__()
        self.frozen_scale = torch.nn.Parameter(torch.tensor(2.0))

    def pre_dit(self, *, x, timestep, context, context_mask, action, fuse_vae_embedding_in_latents):
        del timestep, action, fuse_vae_embedding_in_latents
        tokens = x.mean(dim=tuple(range(1, x.ndim)), keepdim=False)[:, None, None] * self.frozen_scale
        return {
            "tokens": tokens,
            "freqs": torch.zeros(1),
            "t_mod": torch.zeros(1),
            "context": context,
            "context_mask": context_mask,
            "meta": {"tokens_per_frame": 1},
        }


class FakeMoT(torch.nn.Module):
    def __init__(self, action_expert):
        super().__init__()
        self.mixtures = torch.nn.ModuleDict({"action": action_expert})

    def prefill_video_cache(self, *, video_tokens, **kwargs):
        del kwargs
        return [{"k": video_tokens, "v": video_tokens}]

    def forward_action_with_video_cache(self, *, action_tokens, video_kv_cache, **kwargs):
        del kwargs
        # Preserve the cached-prefix dependency without allowing its frozen branch to train.
        return action_tokens + video_kv_cache[0]["k"].sum() * 0.0


class FakeFPOPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu")
        self.torch_dtype = torch.float32
        self.action_expert = FakeActionExpert()
        self.video_expert = FakeVideoExpert()
        self.mot = FakeMoT(self.action_expert)
        self.text_anchor = torch.nn.Parameter(torch.tensor(1.0))
        self.train_action_scheduler = type("Scheduler", (), {"num_train_timesteps": 1000})()

    def encode_prompt(self, prompts):
        batch = len(prompts)
        context = torch.arange(batch * 2 * 4, dtype=torch.float32).reshape(batch, 2, 4)
        return context * self.text_anchor, torch.ones(batch, 2, dtype=torch.bool)

    def _append_proprio_to_context(self, *, context, context_mask, proprio):
        del proprio
        return context, context_mask

    def _encode_input_image_latents_batch_tensor(self, *, input_image, tiled):
        del tiled
        return input_image

    @staticmethod
    def _build_mot_attention_mask(*, video_seq_len, action_seq_len, video_tokens_per_frame, device):
        del video_tokens_per_frame
        total_steps = video_seq_len + action_seq_len
        return torch.ones(total_steps, total_steps, dtype=torch.bool, device=device)


class FakeProcessor:
    shape_meta = {"action": [{"key": "joint"}]}

    def __init__(self):
        self.normalizer = type(
            "NormalizerGroup",
            (),
            {"normalizers": {"action": {"joint": FakeNormalizer()}}},
        )()


class FakeFPORuntime(FakeRuntime):
    def __init__(self):
        self.model = FakeFPOPolicy()
        self.processor = FakeProcessor()
        self.legacy_calls = 0


def _fpo_model() -> FastWAMTrainableModel:
    config = FastWAMAdapterConfig(
        policy_root="/unused",
        checkpoint_sha256="0" * 64,
        fpo={
            "enabled": True,
            "value_enabled": False,
        },
    )
    return FastWAMTrainableModel(runtime_policy=FakeFPORuntime(), config=config, model_path="/unused")


def test_fastwam_implements_official_fpo_contract_and_freeze_boundary():
    model = _fpo_model()

    assert isinstance(model, SupportFPOTraining)
    assert model.policy.action_expert.scale.requires_grad
    assert not model.policy.video_expert.frozen_scale.requires_grad
    assert not model.policy.text_anchor.requires_grad


def test_fastwam_rejects_a_split_action_expert_and_mot_train_scope():
    runtime = FakeFPORuntime()
    runtime.model.mot.mixtures["action"] = FakeActionExpert()
    config = FastWAMAdapterConfig(
        policy_root="/unused",
        checkpoint_sha256="0" * 64,
        fpo={"enabled": True, "value_enabled": False},
    )

    with pytest.raises(TypeError, match="alias the same module"):
        FastWAMTrainableModel(runtime_policy=runtime, config=config, model_path="/unused")


def test_fastwam_fpo_cfm_capability_does_not_require_value_head():
    model = _fpo_model()
    model.fpo_init()

    loss = model.fpo_cfm_loss(
        _obs(1),
        None,
        torch.zeros(1, 32, 14),
        torch.full((1, 2), 0.5),
        torch.ones(1, 2, 32, 14),
    )
    assert loss.shape == (1, 32, 2)
    assert torch.isfinite(loss).all()


def test_fpo_cfm_fixed_mc_identity_shape_and_action_expert_gradients():
    model = _fpo_model()
    obs = _obs(2)
    actions = torch.linspace(-1, 1, 2 * 32 * 14).reshape(2, 32, 14)
    timesteps = torch.tensor([[0.1, 0.3, 0.5, 0.9], [0.2, 0.4, 0.6, 0.8]])
    generator = torch.Generator().manual_seed(606)
    noise = torch.randn((2, 4, 32, 14), generator=generator)

    old_loss = model.fpo_cfm_loss(obs, None, actions, timesteps, noise)
    current_loss = model.fpo_cfm_loss(obs, None, actions, timesteps, noise)
    assert old_loss.shape == (2, 32, 4)
    assert torch.isfinite(old_loss).all()
    torch.testing.assert_close(old_loss, current_loss, rtol=0, atol=0)
    log_ratio = (old_loss - current_loss).sum(dim=1).mean(dim=-1)
    torch.testing.assert_close(log_ratio, torch.zeros(2), rtol=0, atol=0)
    actor_loss = -(log_ratio.exp() * torch.ones_like(log_ratio)).mean()
    assert torch.isfinite(actor_loss)

    actor_loss.backward()
    assert model.policy.action_expert.scale.grad is not None
    assert torch.isfinite(model.policy.action_expert.scale.grad)
    assert model.policy.video_expert.frozen_scale.grad is None
    assert model.policy.text_anchor.grad is None


def test_fpo_state_dict_round_trip_preserves_actor_outputs():
    model = _fpo_model()
    obs = _obs(1)
    actions = torch.zeros(1, 32, 14)
    timesteps = torch.full((1, 4), 0.5)
    noise = torch.ones(1, 4, 32, 14)
    before_cfm = model.fpo_cfm_loss(obs, None, actions, timesteps, noise).detach()

    restored = _fpo_model()
    restored.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(restored.fpo_cfm_loss(obs, None, actions, timesteps, noise), before_cfm, rtol=0, atol=0)


def test_official_rollout_weight_sync_changes_fresh_fastwam_rollout():
    source, _source_runtime = _model()
    target, _target_runtime = _model()
    with torch.no_grad():
        source.policy.anchor.fill_(2.0)
        target.policy.anchor.zero_()

    obs = _obs(2)
    before = target.sac_sample_actions(obs).full_action.clone()

    async def iter_actor_weights():
        for name, parameter in source.state_dict().items():
            yield f"_fsdp_wrapped_module.{name}", parameter.clone()

    rollout = object.__new__(HFRollout)
    rollout.module = target
    rollout.engine = None
    asyncio.run(rollout.update_weights(iter_actor_weights()))

    fresh = target.sac_sample_actions(obs).full_action
    torch.testing.assert_close(fresh - before, torch.full_like(fresh, 2.0), rtol=0, atol=0)
    torch.testing.assert_close(target.policy.anchor, source.policy.anchor, rtol=0, atol=0)
