# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Thin verl-vla rollout wrapper around the native Fast-WAM implementation."""

from __future__ import annotations

import hashlib
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from verl import DataProto

from verl_vla.models.base import (
    ModelOutput,
    SupportFlowGRPOTraining,
    SupportFPOTraining,
    TrainableVLAModelBase,
)

from .adapter_config import FastWAMAdapterConfig

logger = logging.getLogger(__name__)

CAMERA_KEYS = (
    "observation.images.cam_head",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)
STATE_KEY = "observation.state"
ACTION_DIM = 14


def robust_cfm_element_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    delta: float,
) -> torch.Tensor:
    """Elementwise Huber kernel on the MSE scale used by FPO++."""

    if prediction.shape != target.shape:
        raise ValueError("CFM prediction and target must have identical shapes.")
    if delta <= 0:
        raise ValueError("FPO++ Huber delta must be positive.")
    error = prediction.float() - target.float()
    abs_error = error.abs()
    return torch.where(
        abs_error <= delta,
        error.square(),
        2.0 * delta * abs_error - delta**2,
    )


class FastWAMOutput(ModelOutput):
    """Executed prefix plus the complete native Fast-WAM action sample."""

    def __init__(
        self,
        *,
        action: torch.Tensor,
        full_action: torch.Tensor,
        flow_grpo_latents: torch.Tensor | None = None,
        flow_grpo_old_log_probs: torch.Tensor | None = None,
        flow_grpo_sigmas: torch.Tensor | None = None,
        flow_grpo_deltas: torch.Tensor | None = None,
        inference_seconds: float | None = None,
        gpu_memory_allocated_bytes: int | None = None,
        gpu_memory_reserved_bytes: int | None = None,
        gpu_peak_memory_allocated_bytes: int | None = None,
        gpu_peak_memory_reserved_bytes: int | None = None,
    ) -> None:
        if full_action.ndim != 3 or tuple(full_action.shape[1:]) != (32, ACTION_DIM):
            raise ValueError(f"full_action must have shape [B,32,14], got {tuple(full_action.shape)}")
        if action.ndim != 3 or tuple(action.shape[1:]) != (24, ACTION_DIM):
            raise ValueError(f"action must have shape [B,24,14], got {tuple(action.shape)}")
        if action.shape[0] != full_action.shape[0]:
            raise ValueError("action and full_action batch dimensions must match.")
        self.action = action
        self.full_action = full_action
        self.flow_grpo_latents = flow_grpo_latents
        self.flow_grpo_old_log_probs = flow_grpo_old_log_probs
        self.flow_grpo_sigmas = flow_grpo_sigmas
        self.flow_grpo_deltas = flow_grpo_deltas
        trace_values = (
            flow_grpo_latents,
            flow_grpo_old_log_probs,
            flow_grpo_sigmas,
            flow_grpo_deltas,
        )
        if any(value is not None for value in trace_values):
            if any(value is None for value in trace_values):
                raise ValueError("Flow-GRPO rollout trace fields must be provided together.")
            assert flow_grpo_latents is not None
            assert flow_grpo_old_log_probs is not None
            assert flow_grpo_sigmas is not None
            assert flow_grpo_deltas is not None
            batch_size, transition_count, horizon = flow_grpo_old_log_probs.shape
            if batch_size != action.shape[0] or horizon != full_action.shape[1]:
                raise ValueError("Flow-GRPO old log probabilities must have shape [B,T,H].")
            if flow_grpo_latents.shape != (
                batch_size,
                transition_count + 1,
                horizon,
                full_action.shape[2],
            ):
                raise ValueError("Flow-GRPO latents must have shape [B,T+1,H,D].")
            if flow_grpo_sigmas.shape != (batch_size, transition_count):
                raise ValueError("Flow-GRPO sigmas must have shape [B,T].")
            if flow_grpo_deltas.shape != (batch_size, transition_count):
                raise ValueError("Flow-GRPO deltas must have shape [B,T].")
        self.inference_seconds = inference_seconds
        self.gpu_memory_allocated_bytes = gpu_memory_allocated_bytes
        self.gpu_memory_reserved_bytes = gpu_memory_reserved_bytes
        self.gpu_peak_memory_allocated_bytes = gpu_peak_memory_allocated_bytes
        self.gpu_peak_memory_reserved_bytes = gpu_peak_memory_reserved_bytes

    def to_data_proto(self) -> DataProto:
        tensors = {"action": self.action, "full_action": self.full_action}
        if self.flow_grpo_latents is not None:
            tensors.update(
                {
                    "flow_grpo.latents": self.flow_grpo_latents,
                    "flow_grpo.old_log_probs": self.flow_grpo_old_log_probs,
                    "flow_grpo.sigmas": self.flow_grpo_sigmas,
                    "flow_grpo.deltas": self.flow_grpo_deltas,
                }
            )
        batch_size = self.action.shape[0]
        device = self.action.device
        optional_metrics = {
            "profile.inference_seconds": self.inference_seconds,
            "profile.gpu_memory_allocated_bytes": self.gpu_memory_allocated_bytes,
            "profile.gpu_memory_reserved_bytes": self.gpu_memory_reserved_bytes,
            "profile.gpu_peak_memory_allocated_bytes": self.gpu_peak_memory_allocated_bytes,
            "profile.gpu_peak_memory_reserved_bytes": self.gpu_peak_memory_reserved_bytes,
        }
        for key, value in optional_metrics.items():
            if value is not None:
                tensors[key] = torch.full((batch_size,), float(value), dtype=torch.float64, device=device)
        return DataProto.from_dict(tensors=tensors)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FastWAMTrainableModel(TrainableVLAModelBase, SupportFPOTraining, SupportFlowGRPOTraining):
    """Native Fast-WAM rollout plus the official vanilla-FPO model contract."""

    def __init__(self, *, runtime_policy: Any, config: FastWAMAdapterConfig, model_path: str) -> None:
        super().__init__(policy=runtime_policy.model)
        self.runtime_policy = runtime_policy
        self.config = config
        self.model_path = str(model_path)
        self.rollout_call_count = 0
        self.loaded_policy_replicas = 1
        self._policy_seed_call_counts: dict[int, int] = {}
        if config.fpo.enabled or config.flow_grpo.enabled:
            self._configure_action_expert_parameters()

    def _configure_action_expert_parameters(self) -> None:
        if not hasattr(self.policy, "action_expert"):
            raise TypeError("Fast-WAM policy-gradient training requires native policy.action_expert.")
        mot = getattr(self.policy, "mot", None)
        mixtures = getattr(mot, "mixtures", None)
        mot_action_expert = mixtures["action"] if mixtures is not None and "action" in mixtures else None
        if mot_action_expert is not self.policy.action_expert:
            raise TypeError(
                "Fast-WAM train scope requires policy.action_expert and "
                "policy.mot.mixtures['action'] to alias the same module."
            )
        self.policy.requires_grad_(False)
        self.policy.eval()
        self.policy.action_expert.requires_grad_(True)
        self.policy.action_expert.train()

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        *,
        adapter_config: dict[str, Any],
        torch_dtype: torch.dtype,
    ) -> FastWAMTrainableModel:
        config = FastWAMAdapterConfig(**adapter_config)
        model_root = Path(model_path).expanduser().resolve()
        checkpoint_path = model_root / config.weights_relative_path
        dataset_stats_path = model_root / config.dataset_stats_relative_path
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Fast-WAM checkpoint not found: {checkpoint_path}")
        if not dataset_stats_path.is_file():
            raise FileNotFoundError(f"Fast-WAM dataset statistics not found: {dataset_stats_path}")
        actual_sha256 = _sha256(checkpoint_path)
        if actual_sha256 != config.checkpoint_sha256:
            raise ValueError(
                f"Fast-WAM checkpoint SHA256 mismatch: expected {config.checkpoint_sha256}, "
                f"got {actual_sha256} for {checkpoint_path}"
            )

        policy_root = Path(config.policy_root).expanduser().resolve()
        upstream_root = policy_root / "FastWAM"
        for import_path in (upstream_root, upstream_root / "src"):
            import_text = str(import_path)
            if import_text not in sys.path:
                sys.path.insert(0, import_text)

        from experiments.robotwin.fastwam_policy.deploy_policy import get_model

        if torch_dtype == torch.bfloat16:
            mixed_precision = "bf16"
        elif torch_dtype == torch.float16:
            mixed_precision = "fp16"
        else:
            mixed_precision = "no"
        runtime_policy = get_model(
            {
                "ckpt_setting": str(checkpoint_path),
                "dataset_stats_path": str(dataset_stats_path),
                "sim_cfg_name": config.sim_cfg_name,
                "sim_task": config.sim_task,
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "mixed_precision": mixed_precision,
                "action_horizon": config.action_horizon,
                "replan_steps": config.action_chunk_size,
                "num_inference_steps": config.num_inference_steps,
                "sigma_shift": config.sigma_shift,
                "seed": config.seed,
                "text_cfg_scale": config.text_cfg_scale,
                "negative_prompt": config.negative_prompt,
                "rand_device": config.rand_device,
                "tiled": config.tiled,
                "timing_enabled": False,
            }
        )
        if int(runtime_policy.action_horizon) != 32 or int(runtime_policy.replan_steps) != 24:
            raise RuntimeError(
                "Native runtime changed Fast-WAM deployment semantics: "
                f"horizon={runtime_policy.action_horizon}, replan={runtime_policy.replan_steps}."
            )
        logger.warning(
            "Fast-WAM replica loaded once: replica_id=%s checkpoint=%s sha256=%s "
            "cuda_visible_devices=%s",
            id(runtime_policy.model),
            checkpoint_path,
            actual_sha256,
            os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        )
        return cls(runtime_policy=runtime_policy, config=config, model_path=str(model_root))

    def train(self, mode: bool = True) -> FastWAMTrainableModel:
        super().train(mode)
        if self.config.fpo.enabled or self.config.flow_grpo.enabled:
            # Frozen conditioning must remain deterministic even during an actor update.
            self.policy.eval()
            self.policy.action_expert.train(mode)
        return self

    def can_generate(self) -> bool:
        """Declare that this native action policy is not an HF text generator.

        verl's generic FSDP checkpoint manager queries this Transformers-style
        capability even when ``hf_model`` export is disabled.  Native Fast-WAM
        checkpoints use the regular FSDP model/optimizer/extra contents, so the
        correct answer is explicitly false.
        """

        return False

    def sac_init(self) -> None:
        """Register rollout inference without claiming SAC training support."""

    @staticmethod
    def _image_to_hwc_uint8(image: torch.Tensor, *, key: str) -> np.ndarray:
        image = image.detach().cpu()
        if image.ndim != 3:
            raise ValueError(f"{key} must be CHW or HWC, got {tuple(image.shape)}")
        if image.shape[0] == 3:
            image = image.permute(1, 2, 0)
        elif image.shape[-1] != 3:
            raise ValueError(f"{key} must be CHW or HWC RGB, got {tuple(image.shape)}")
        if tuple(image.shape) != (240, 320, 3):
            raise ValueError(f"{key} must be canonical [240,320,3], got {tuple(image.shape)}")
        if image.dtype != torch.uint8:
            if image.is_floating_point() and float(image.max()) <= 1.0:
                image = image * 255.0
            image = image.clamp(0, 255).to(torch.uint8)
        return image.numpy()

    def _canonical_rows(self, obs: DataProto) -> tuple[list[dict[str, Any]], list[str]]:
        missing = [key for key in (*CAMERA_KEYS, STATE_KEY) if key not in obs.batch]
        if missing:
            raise KeyError(f"Fast-WAM canonical observation is missing keys: {missing}")
        state = obs.batch[STATE_KEY]
        if state.ndim != 2 or state.shape[1] != ACTION_DIM:
            raise ValueError(f"{STATE_KEY} must have shape [B,14], got {tuple(state.shape)}")
        tasks = list(obs.non_tensor_batch["task"])
        if len(tasks) != state.shape[0]:
            raise ValueError(f"task batch length {len(tasks)} does not match state batch {state.shape[0]}")
        rows = []
        for batch_idx in range(state.shape[0]):
            rows.append(
                {
                    "observation": {
                        "head_camera": {
                            "rgb": self._image_to_hwc_uint8(obs.batch[CAMERA_KEYS[0]][batch_idx], key=CAMERA_KEYS[0])
                        },
                        "left_camera": {
                            "rgb": self._image_to_hwc_uint8(obs.batch[CAMERA_KEYS[1]][batch_idx], key=CAMERA_KEYS[1])
                        },
                        "right_camera": {
                            "rgb": self._image_to_hwc_uint8(obs.batch[CAMERA_KEYS[2]][batch_idx], key=CAMERA_KEYS[2])
                        },
                    },
                    "joint_action": {
                        "vector": state[batch_idx].detach().cpu().to(torch.float32).numpy()
                    },
                }
            )
        return rows, [str(task) for task in tasks]

    def _infer_batch(
        self,
        rows: list[dict[str, Any]],
        tasks: list[str],
        *,
        sample_seeds: list[int] | None,
        flow_grpo_rollout: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        from fastwam.datasets.lerobot.constants import DEFAULT_PROMPT

        images = torch.cat([self.runtime_policy._build_robotwin_image_tensor(row) for row in rows], dim=0)
        proprio = torch.cat(
            [self.runtime_policy._normalize_state(row["joint_action"]["vector"]) for row in rows],
            dim=0,
        )
        prompts = [DEFAULT_PROMPT.format(task=task) for task in tasks]
        native_result = self.policy.infer_action_batch(
            prompt=prompts,
            input_image=images,
            action_horizon=self.config.action_horizon,
            proprio=proprio,
            negative_prompt=self.config.negative_prompt,
            text_cfg_scale=self.config.text_cfg_scale,
            num_inference_steps=self.config.num_inference_steps,
            sigma_shift=self.config.sigma_shift,
            seeds=sample_seeds,
            rand_device=self.config.rand_device,
            action_latent_scale=self.config.rollout_action_latent_scale,
            tiled=self.config.tiled,
            flow_grpo_noise_level=(
                self.config.flow_grpo.noise_level
                if flow_grpo_rollout
                else None
            ),
            return_flow_grpo_trace=flow_grpo_rollout,
        )
        pred = native_result["action"]
        full_action = torch.from_numpy(self.runtime_policy._denormalize_action(pred))
        return full_action, native_result.get("flow_grpo")

    @staticmethod
    def _derive_policy_call_seed(base_seed: int, call_index: int) -> int:
        payload = f"fastwam-policy-call:{int(base_seed)}:{int(call_index)}".encode()
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)

    def _rollout_sample_seeds(self, base_seeds: list[int]) -> list[int]:
        if self.config.rollout_seed_mode == "episode":
            return base_seeds
        result = []
        for base_seed in base_seeds:
            call_index = self._policy_seed_call_counts.get(base_seed, 0)
            result.append(self._derive_policy_call_seed(base_seed, call_index))
            self._policy_seed_call_counts[base_seed] = call_index + 1
        return result

    @torch.no_grad()
    def sac_sample_actions(
        self,
        obs: DataProto,
        tokenizer: torch.nn.Module | None = None,
        eval: bool = False,
    ) -> FastWAMOutput:
        del tokenizer
        rows, tasks = self._canonical_rows(obs)
        flow_grpo_rollout = self.config.flow_grpo.enabled and not eval
        policy_seed_values = obs.non_tensor_batch.get("policy_seed")
        if policy_seed_values is None:
            sample_seeds = [int(self.config.seed)] * len(rows) if self.config.seed is not None else None
        else:
            base_seeds = [int(value) for value in np.asarray(policy_seed_values).reshape(-1)]
            if len(base_seeds) != len(rows):
                raise ValueError("policy_seed must contain exactly one seed per Fast-WAM observation row.")
            sample_seeds = self._rollout_sample_seeds(base_seeds)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        inference_start = time.perf_counter()
        flow_trace = None
        if (
            len(rows) == 1
            and policy_seed_values is None
            and self.config.rollout_action_latent_scale == 1.0
            and not flow_grpo_rollout
        ):
            # This is intentionally the established B=1 oracle path.
            full_action_np = self.runtime_policy._infer_action_chunk(rows[0], tasks[0])[None, ...]
            full_action = torch.from_numpy(np.asarray(full_action_np, dtype=np.float32))
        else:
            full_action, flow_trace = self._infer_batch(
                rows,
                tasks,
                sample_seeds=sample_seeds,
                flow_grpo_rollout=flow_grpo_rollout,
            )
            full_action = full_action.to(dtype=torch.float32)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - inference_start

        target_device = obs.batch[STATE_KEY].device
        full_action = full_action.to(device=target_device)
        flow_output: dict[str, torch.Tensor | None] = {
            "flow_grpo_latents": None,
            "flow_grpo_old_log_probs": None,
            "flow_grpo_sigmas": None,
            "flow_grpo_deltas": None,
        }
        if flow_trace is not None:
            flow_output = {
                "flow_grpo_latents": flow_trace["latents"].to(device=target_device),
                "flow_grpo_old_log_probs": flow_trace["old_log_probs"].to(device=target_device),
                "flow_grpo_sigmas": flow_trace["sigmas"].to(device=target_device),
                "flow_grpo_deltas": flow_trace["deltas"].to(device=target_device),
            }
        if tuple(full_action.shape[1:]) != (self.config.action_horizon, ACTION_DIM):
            raise RuntimeError(f"Native Fast-WAM returned unexpected action shape {tuple(full_action.shape)}")
        if not torch.isfinite(full_action).all():
            raise FloatingPointError("Native Fast-WAM returned NaN or Inf actions.")

        self.rollout_call_count += 1
        logger.warning(
            "Fast-WAM batched rollout call=%d batch=%d replica_id=%s full_shape=%s executed_steps=%d "
            "action_latent_scale=%.6f rollout_seed_mode=%s inference_seconds=%.6f",
            self.rollout_call_count,
            len(rows),
            id(self.policy),
            tuple(full_action.shape),
            self.config.action_chunk_size,
            self.config.rollout_action_latent_scale,
            self.config.rollout_seed_mode,
            inference_seconds,
        )
        cuda_metrics: dict[str, int | None] = {
            "gpu_memory_allocated_bytes": None,
            "gpu_memory_reserved_bytes": None,
            "gpu_peak_memory_allocated_bytes": None,
            "gpu_peak_memory_reserved_bytes": None,
        }
        if torch.cuda.is_available():
            cuda_metrics.update(
                {
                    "gpu_memory_allocated_bytes": torch.cuda.memory_allocated(),
                    "gpu_memory_reserved_bytes": torch.cuda.memory_reserved(),
                    "gpu_peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "gpu_peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
                }
            )
        return FastWAMOutput(
            action=full_action[:, : self.config.action_chunk_size],
            full_action=full_action,
            **flow_output,
            inference_seconds=inference_seconds,
            **cuda_metrics,
        )

    def reset(self) -> None:
        self._policy_seed_call_counts.clear()
        self.runtime_policy.reset()

    # --- Official vanilla-FPO model contract ---

    def fpo_init(self) -> None:
        if not self.config.fpo.enabled:
            raise RuntimeError("FPO requires adapter.fpo.enabled=true.")
        try:
            from torch.distributed.fsdp import register_fsdp_forward_method
        except ImportError:
            return
        register_fsdp_forward_method(self, "fpo_cfm_loss")

    def flow_grpo_init(self) -> None:
        if not self.config.flow_grpo.enabled:
            raise RuntimeError("Flow-GRPO requires its model-side rollout trace configuration.")
        try:
            from torch.distributed.fsdp import register_fsdp_forward_method
        except ImportError:
            return
        register_fsdp_forward_method(self, "flow_grpo_log_probs")

    def flow_grpo_log_probs(
        self,
        obs: DataProto,
        tokenizer: torch.nn.Module,
        latents: torch.Tensor,
        sigmas: torch.Tensor,
        deltas: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate recorded Flow-GRPO transitions under the current actor.

        The returned shape is ``[B,T,H]``: the Gaussian log probability is
        averaged over action dimensions D exactly as in the official code,
        while H is retained for the environment executed-action mask.
        """

        del tokenizer
        if not self.config.flow_grpo.enabled:
            raise RuntimeError("flow_grpo_log_probs requires flow_grpo.enabled=true.")
        if latents.ndim != 4:
            raise ValueError("Flow-GRPO latents must have shape [B,T+1,H,D].")
        batch_size, latent_count, horizon, action_dim = latents.shape
        transition_count = latent_count - 1
        if (horizon, action_dim) != (self.config.action_horizon, ACTION_DIM):
            raise ValueError("Flow-GRPO latent action shape must be [H,D]=[32,14].")
        if sigmas.shape != (batch_size, transition_count) or deltas.shape != (
            batch_size,
            transition_count,
        ):
            raise ValueError("Flow-GRPO sigmas/deltas must have shape [B,T].")
        if not torch.isfinite(latents).all() or not torch.isfinite(sigmas).all() or not torch.isfinite(deltas).all():
            raise FloatingPointError("Flow-GRPO rollout trace contains NaN or Inf.")

        from fastwam.models.wan22.schedulers.flow_grpo_sde import flow_grpo_sde_step_with_logprob

        images, _proprio, context, context_mask = self._fpo_observation_conditioning(obs)
        video_kv_cache, attention_mask, video_seq_len = self._fpo_video_cache(images, context, context_mask)
        latents = latents.to(device=self.policy.device, dtype=self.policy.torch_dtype)
        sigmas = sigmas.to(device=self.policy.device, dtype=torch.float32)
        deltas = deltas.to(device=self.policy.device, dtype=torch.float32)
        # The historical implementation issued one large action-expert/MoT
        # forward per transition.  Batch adjacent transitions so the exact same
        # [B,T,H] likelihood is computed with fewer, better-utilized forwards.
        # Keeping this bounded (rather than always flattening all T) controls
        # the repeated video-KV memory footprint.
        transition_batch_size = min(
            self.config.flow_grpo.transition_batch_size,
            transition_count,
        )
        log_prob_blocks = []
        for transition_start in range(0, transition_count, transition_batch_size):
            transition_end = min(transition_start + transition_batch_size, transition_count)
            block_size = transition_end - transition_start
            x_t = latents[:, transition_start:transition_end].reshape(
                batch_size * block_size,
                horizon,
                action_dim,
            )
            next_x = latents[:, transition_start + 1 : transition_end + 1].reshape(
                batch_size * block_size,
                horizon,
                action_dim,
            )
            sigma = sigmas[:, transition_start:transition_end].reshape(-1)
            sigma_next = sigma + deltas[:, transition_start:transition_end].reshape(-1)
            native_timestep = sigma.to(dtype=x_t.dtype) * float(
                self.policy.infer_action_scheduler.num_train_timesteps
            )
            repeated_context = self._repeat_fpo_samples(context, block_size)
            repeated_context_mask = self._repeat_fpo_samples(context_mask, block_size)
            repeated_cache = [
                {name: self._repeat_fpo_samples(value, block_size) for name, value in layer.items()}
                for layer in video_kv_cache
            ]
            action_pre = self.policy.action_expert.pre_dit(
                action_tokens=x_t,
                timestep=native_timestep,
                context=repeated_context,
                context_mask=repeated_context_mask,
            )
            action_tokens = self.policy.mot.forward_action_with_video_cache(
                action_tokens=action_pre["tokens"],
                action_freqs=action_pre["freqs"],
                action_t_mod=action_pre["t_mod"],
                action_context_payload={
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
                video_kv_cache=repeated_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
            )
            predicted_velocity = self.policy.action_expert.post_dit(action_tokens, action_pre)
            _next, element_log_prob, _mean = flow_grpo_sde_step_with_logprob(
                model_output=predicted_velocity,
                sigma=sigma,
                sigma_next=sigma_next,
                sample=x_t,
                noise_level=self.config.flow_grpo.noise_level,
                next_sample=next_x,
            )
            log_prob_blocks.append(
                element_log_prob.mean(dim=-1).reshape(batch_size, block_size, horizon)
            )
        result = torch.cat(log_prob_blocks, dim=1)
        if not torch.isfinite(result).all():
            raise FloatingPointError("Current Flow-GRPO log probabilities contain NaN or Inf.")
        return result

    def _fpo_observation_conditioning(
        self,
        obs: DataProto,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return image, normalized proprio, frozen context, and context mask."""
        from fastwam.datasets.lerobot.constants import DEFAULT_PROMPT

        rows, tasks = self._canonical_rows(obs)
        with torch.no_grad():
            images = torch.cat([self.runtime_policy._build_robotwin_image_tensor(row) for row in rows], dim=0)
            proprio = torch.cat(
                [self.runtime_policy._normalize_state(row["joint_action"]["vector"]) for row in rows],
                dim=0,
            ).to(device=self.policy.device, dtype=self.policy.torch_dtype)
            prompts = [DEFAULT_PROMPT.format(task=task) for task in tasks]
            context, context_mask = self.policy.encode_prompt(prompts)
            context, context_mask = self.policy._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )
        return images.detach(), proprio.detach(), context.detach(), context_mask.detach()

    def _fpo_video_cache(
        self,
        images: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> tuple[list[dict[str, torch.Tensor]], torch.Tensor, int]:
        """Build the frozen first-frame MoT prefix once per observation."""
        with torch.no_grad():
            first_frame_latents = self.policy._encode_input_image_latents_batch_tensor(
                input_image=images.to(device=self.policy.device, dtype=self.policy.torch_dtype),
                tiled=self.config.tiled,
            )
            fuse_flag = bool(getattr(self.policy.video_expert, "fuse_vae_embedding_in_latents", False))
            timestep_video = torch.zeros(
                (first_frame_latents.shape[0],),
                dtype=first_frame_latents.dtype,
                device=first_frame_latents.device,
            )
            video_pre = self.policy.video_expert.pre_dit(
                x=first_frame_latents,
                timestep=timestep_video,
                context=context,
                context_mask=context_mask,
                action=None,
                fuse_vae_embedding_in_latents=fuse_flag,
            )
            video_seq_len = int(video_pre["tokens"].shape[1])
            attention_mask = self.policy._build_mot_attention_mask(
                video_seq_len=video_seq_len,
                action_seq_len=self.config.action_horizon,
                video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
                device=video_pre["tokens"].device,
            )
            video_kv_cache = self.policy.mot.prefill_video_cache(
                video_tokens=video_pre["tokens"],
                video_freqs=video_pre["freqs"],
                video_t_mod=video_pre["t_mod"],
                video_context_payload={
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
            )
        return (
            [{name: value.detach() for name, value in layer.items()} for layer in video_kv_cache],
            attention_mask.detach(),
            video_seq_len,
        )

    def _normalize_fpo_actions(self, actions: torch.Tensor) -> torch.Tensor:
        action_meta = self.runtime_policy.processor.shape_meta["action"]
        if len(action_meta) != 1:
            raise ValueError("Expected exactly one merged Fast-WAM action key.")
        action_key = action_meta[0]["key"]
        normalizer = self.runtime_policy.processor.normalizer.normalizers["action"][action_key]
        normalized = normalizer.forward(actions.detach().to(device="cpu", dtype=torch.float32))
        return normalized.to(device=self.policy.device, dtype=self.policy.torch_dtype)

    @staticmethod
    def _repeat_fpo_samples(tensor: torch.Tensor, n_action_samples: int) -> torch.Tensor:
        batch_size = tensor.shape[0]
        return (
            tensor.unsqueeze(1)
            .expand(-1, n_action_samples, *tensor.shape[1:])
            .reshape(batch_size * n_action_samples, *tensor.shape[1:])
        )

    def fpo_cfm_loss(
        self,
        obs: DataProto,
        tokenizer: torch.nn.Module,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
        *,
        loss_kernel: str = "mse",
        huber_delta: float = 1.0,
    ) -> torch.Tensor:
        del tokenizer
        if not self.config.fpo.enabled:
            raise RuntimeError("FPO requires adapter.fpo.enabled=true.")
        if timesteps.ndim != 2:
            raise ValueError(f"FPO timesteps must be [B,N], got {tuple(timesteps.shape)}.")
        if noise.ndim != 4:
            raise ValueError(f"FPO noise must be [B,N,32,14], got {tuple(noise.shape)}.")
        batch_size, n_action_samples = timesteps.shape
        expected_actions = (batch_size, self.config.action_horizon, ACTION_DIM)
        expected_noise = (batch_size, n_action_samples, self.config.action_horizon, ACTION_DIM)
        if tuple(actions.shape) != expected_actions:
            raise ValueError(f"FPO requires full native actions {expected_actions}, got {tuple(actions.shape)}.")
        if tuple(noise.shape) != expected_noise:
            raise ValueError(f"FPO noise must have shape {expected_noise}, got {tuple(noise.shape)}.")
        if not torch.isfinite(actions).all() or not torch.isfinite(timesteps).all() or not torch.isfinite(noise).all():
            raise FloatingPointError("FPO actions, timesteps, and noise must be finite.")
        if bool(((timesteps < 0) | (timesteps > 1)).any()):
            raise ValueError("FPO timesteps are continuous sigmas and must lie in [0,1].")
        if loss_kernel not in {"mse", "huber"}:
            raise ValueError("FPO loss_kernel must be 'mse' or 'huber'.")
        if huber_delta <= 0:
            raise ValueError("FPO huber_delta must be positive.")

        action_tensor = self._normalize_fpo_actions(actions)
        images, _proprio, context, context_mask = self._fpo_observation_conditioning(obs)
        video_kv_cache, attention_mask, video_seq_len = self._fpo_video_cache(images, context, context_mask)

        repeated_actions = self._repeat_fpo_samples(action_tensor, n_action_samples)
        repeated_noise = noise.reshape(batch_size * n_action_samples, *noise.shape[2:]).to(
            device=self.policy.device,
            dtype=repeated_actions.dtype,
        )
        sigmas = timesteps.reshape(-1).to(device=self.policy.device, dtype=repeated_actions.dtype)
        sigma_expanded = sigmas[:, None, None]
        noisy_action = (1.0 - sigma_expanded) * repeated_actions + sigma_expanded * repeated_noise
        target_velocity = repeated_noise - repeated_actions
        native_timestep = sigmas * float(self.policy.train_action_scheduler.num_train_timesteps)

        repeated_context = self._repeat_fpo_samples(context, n_action_samples)
        repeated_context_mask = self._repeat_fpo_samples(context_mask, n_action_samples)
        repeated_cache = [
            {name: self._repeat_fpo_samples(value, n_action_samples) for name, value in layer.items()}
            for layer in video_kv_cache
        ]
        action_pre = self.policy.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=native_timestep,
            context=repeated_context,
            context_mask=repeated_context_mask,
        )
        action_tokens = self.policy.mot.forward_action_with_video_cache(
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            video_kv_cache=repeated_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
        )
        predicted_velocity = self.policy.action_expert.post_dit(action_tokens, action_pre)
        if loss_kernel == "huber":
            # Match manipulation FPO++ exactly: this is twice PyTorch's Huber
            # loss, keeping the quadratic region on the same scale as MSE.
            element_loss = robust_cfm_element_loss(
                predicted_velocity,
                target_velocity,
                delta=float(huber_delta),
            )
        else:
            element_loss = F.mse_loss(
                predicted_velocity.float(),
                target_velocity.float(),
                reduction="none",
            )
        token_loss = element_loss.mean(dim=-1)
        return token_loss.reshape(batch_size, n_action_samples, self.config.action_horizon).permute(0, 2, 1)
