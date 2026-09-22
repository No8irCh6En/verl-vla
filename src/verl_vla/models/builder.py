# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Explicit VLA model construction without Transformers AutoClass registration."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import torch


def _apply_overrides(config, overrides: Mapping) -> None:
    for name, value in overrides.items():
        setattr(config, name, value)


def build_vla_model(model_config, *, torch_dtype: torch.dtype):
    architecture = model_config.native_architecture
    path = model_config.local_path
    overrides = dict(model_config.override_config)
    if "model_config" in overrides:
        overrides = dict(overrides["model_config"])

    if architecture == "pi0":
        from .pi0_torch import PI0TrainableModel

        return PI0TrainableModel.from_pretrained(
            path,
            adapter_config=dict(model_config.adapter),
            policy_config_overrides=overrides,
            torch_dtype=torch_dtype,
        )

    if architecture == "fastwam":
        from .fastwam import FastWAMTrainableModel

        if overrides:
            raise ValueError("Fast-WAM architecture is checkpoint-owned; model.override_config must be empty")
        return FastWAMTrainableModel.from_pretrained(
            path,
            adapter_config=dict(model_config.adapter),
            torch_dtype=torch_dtype,
        )

    if architecture == "act":
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.pretrained import SAFETENSORS_SINGLE_FILE

        from .act_torch import ACTTrainableModel
        from .act_torch.processor import load_act_processors

        if overrides:
            raise ValueError("ACT architecture is checkpoint-owned; model.override_config must be empty")
        model_path = Path(path)
        weights_path = model_path / SAFETENSORS_SINGLE_FILE
        initialization_path = model_path / "initialization.json"
        if weights_path.is_file():
            policy = ACTPolicy.from_pretrained(path)
        else:
            if not initialization_path.is_file():
                raise FileNotFoundError(
                    f"Native ACT weights are missing at {weights_path}. Config-only initialization requires "
                    f"an explicit {initialization_path.name} sidecar."
                )
            with initialization_path.open(encoding="utf-8") as file:
                initialization = json.load(file)
            if initialization != {"type": "act_config"}:
                raise ValueError(f"Unsupported ACT initialization metadata in {initialization_path}")
            config = PreTrainedConfig.from_pretrained(path)
            if not isinstance(config, ACTConfig):
                raise TypeError(f"Expected a native ACT config at {path}, got {type(config).__name__}")
            policy = ACTPolicy(config)
        adapter_config = dict(model_config.adapter)
        processor_dataset_root = adapter_config.pop("processor_dataset_root", None)
        preprocessor, postprocessor = load_act_processors(
            policy.config,
            model_path,
            dataset_root=processor_dataset_root,
        )
        policy.to(dtype=torch_dtype)
        return ACTTrainableModel(
            policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            adapter_config=adapter_config,
            model_path=path,
        )

    if architecture == "gaussian_actor":
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.pretrained import SAFETENSORS_SINGLE_FILE

        from .gaussian_actor import (
            GaussianActorConfig,
            GaussianActorPolicy,
            GaussianActorTrainableModel,
            load_gaussian_actor_processors,
        )

        if overrides:
            raise ValueError("Gaussian actor architecture is checkpoint-owned; model.override_config must be empty")
        model_path = Path(path)
        weights_path = model_path / SAFETENSORS_SINGLE_FILE
        initialization_path = model_path / "initialization.json"
        if weights_path.is_file():
            policy = GaussianActorPolicy.from_pretrained(path, strict=True)
        else:
            if not initialization_path.is_file():
                raise FileNotFoundError(
                    f"Native Gaussian actor weights are missing at {weights_path}. Config-only initialization "
                    f"requires an explicit {initialization_path.name} sidecar."
                )
            with initialization_path.open(encoding="utf-8") as file:
                initialization = json.load(file)
            if initialization != {"type": "gaussian_actor_config"}:
                raise ValueError(f"Unsupported Gaussian actor initialization metadata in {initialization_path}")
            config = PreTrainedConfig.from_pretrained(path)
            if not isinstance(config, GaussianActorConfig):
                raise TypeError(f"Expected GaussianActorConfig in {model_path}, got {type(config).__name__}")
            policy = GaussianActorPolicy(config)
        adapter_config = dict(model_config.adapter)
        processor_dataset_root = adapter_config.pop("processor_dataset_root", None)
        preprocessor, postprocessor = load_gaussian_actor_processors(
            policy.config,
            model_path,
            dataset_root=processor_dataset_root,
        )
        policy.to(dtype=torch_dtype)
        return GaussianActorTrainableModel(
            policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            adapter_config=adapter_config,
        )

    if architecture == "gr00t_n1d6":
        from gr00t.configs.model.gr00t_n1d6 import Gr00tN1d6Config

        from .gr00t_n1d6.adapter_config import Gr00tAdapterConfig
        from .gr00t_n1d6.trainable_model import Gr00tN1d6TrainableModel, load_gr00t_n1d6_policy

        config = Gr00tN1d6Config.from_pretrained(path)
        _apply_overrides(config, overrides)
        adapter_config = Gr00tAdapterConfig(model_path=path, **dict(model_config.adapter))

        policy = load_gr00t_n1d6_policy(path, config=config, torch_dtype=torch_dtype)
        return Gr00tN1d6TrainableModel(policy, adapter_config=adapter_config)

    if architecture == "openvla_oft":
        from .openvla_oft.configuration_prismatic import OpenVLAConfig
        from .openvla_oft.modeling_prismatic import OpenVLAForActionPrediction
        from .openvla_oft.trainable_model import OpenVLATrainableModel

        config = OpenVLAConfig.from_pretrained(path)
        _apply_overrides(config, overrides)
        policy = OpenVLAForActionPrediction.from_pretrained(path, config=config, torch_dtype=torch_dtype)
        statistics_path = Path(path) / "dataset_statistics.json"
        if statistics_path.is_file():
            with statistics_path.open(encoding="utf-8") as file:
                policy.norm_stats = json.load(file)
        processor = model_config.processor
        if processor is None:
            raise ValueError("OpenVLA requires its native PrismaticProcessor")
        return OpenVLATrainableModel(
            policy,
            processor=processor,
            adapter_config=model_config.adapter,
            artifact_source_dir=path,
        )

    if architecture == "recap_value_critic":
        from .recap_value_critic import ReCapValueCriticConfig, ReCapValueCriticTrainableModel

        config = ReCapValueCriticConfig.from_pretrained(path)
        _apply_overrides(config, overrides)
        return ReCapValueCriticTrainableModel.from_pretrained(path, config=config, torch_dtype=torch_dtype)

    if architecture == "recap_resnet18_value_critic":
        from .recap_resnet18_value_critic import (
            ReCapResNet18ValueCriticConfig,
            ReCapResNet18ValueCriticTrainableModel,
        )

        config = ReCapResNet18ValueCriticConfig.from_pretrained(path)
        _apply_overrides(config, overrides)
        return ReCapResNet18ValueCriticTrainableModel.from_pretrained(path, config=config, torch_dtype=torch_dtype)

    raise ValueError(f"Unsupported VLA architecture: {architecture!r}")


__all__ = ["build_vla_model"]
