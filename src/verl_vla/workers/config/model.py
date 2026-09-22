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

"""Model configuration that preserves native policy checkpoint formats."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from types import SimpleNamespace

from huggingface_hub import snapshot_download
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.import_utils import import_external_libs
from verl.workers.config.model import HFModelConfig

__all__ = ["VLAModelConfig"]


def resolve_model_path(path: str, *, use_shm: bool) -> str:
    """Resolve a local, HDFS, or Hugging Face model path to a local directory."""

    expanded_path = os.path.expanduser(path)
    if os.path.exists(expanded_path):
        return copy_to_local(expanded_path, use_shm=use_shm)
    if path.startswith("hdfs://"):
        return copy_to_local(path, use_shm=use_shm)

    local_path = snapshot_download(repo_id=path)
    if use_shm:
        return copy_to_local(local_path, use_shm=True)
    return local_path


@dataclass
class VLAModelConfig(HFModelConfig):
    """HF-compatible config with a native Diffusers PI0 detection path."""

    _mutable_fields = HFModelConfig._mutable_fields | {
        "native_architecture",
        "share_embeddings_and_output_weights",
        "adapter",
        "lora_rank",
        "lora_alpha",
        "target_modules",
        "target_parameters",
        "exclude_modules",
        "lora_adapter_path",
    }

    native_architecture: str | None = None
    adapter: dict = field(default_factory=dict)
    lora: dict = field(
        default_factory=lambda: {
            "rank": 0,
            "alpha": 16,
            "target_modules": "all-linear",
            "target_parameters": None,
            "exclude_modules": None,
            "adapter_path": None,
            "merge": True,
        }
    )

    def __post_init__(self):
        requested_native_architecture = self.native_architecture
        # Adapt the canonical nested VLA config to verl's flat FSDP LoRA contract.
        self.lora_rank = int(self.lora["rank"])
        self.lora_alpha = int(self.lora["alpha"])
        self.target_modules = self.lora["target_modules"]
        self.target_parameters = self.lora["target_parameters"]
        self.exclude_modules = self.lora["exclude_modules"]
        self.lora_adapter_path = self.lora["adapter_path"]

        if self.lora_adapter_path is not None and self.lora_rank <= 0:
            raise ValueError("lora.rank must be positive when lora.adapter_path is set")

        import_external_libs(self.external_lib)
        self.local_path = resolve_model_path(self.path, use_shm=self.use_shm)
        config_path = os.path.join(self.local_path, "config.json")
        try:
            with open(config_path, encoding="utf-8") as file:
                checkpoint_config = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            checkpoint_config = {}

        class_name = str(checkpoint_config.get("_class_name", ""))
        model_type = str(checkpoint_config.get("model_type", ""))
        policy_type = str(checkpoint_config.get("type", ""))
        architectures = " ".join(checkpoint_config.get("architectures", []))
        identity = f"{class_name} {model_type} {architectures}".lower()
        if requested_native_architecture is not None:
            architecture = str(requested_native_architecture)
        elif class_name == "PI0Policy":
            architecture = "pi0"
        elif model_type == "openvla":
            architecture = "openvla_oft"
        elif policy_type == "act":
            architecture = "act"
        elif policy_type == "gaussian_actor":
            architecture = "gaussian_actor"
        elif model_type == "recap_value_critic":
            architecture = "recap_value_critic"
        elif model_type == "recap_resnet18_value_critic":
            architecture = "recap_resnet18_value_critic"
        elif "gr00tn1d6" in identity or "gr00t_n1d6" in identity:
            architecture = "gr00t_n1d6"
        else:
            architecture = None

        if architecture is None:
            raise ValueError(
                f"Unsupported VLA checkpoint metadata in {config_path}. Add an explicit model builder instead of "
                "registering the model with a Transformers AutoClass."
            )

        self.native_architecture = architecture
        self.model_type = "vla_native"
        if self.tokenizer_path is None:
            self.tokenizer_path = self.path
        if self.load_tokenizer:
            self.local_tokenizer_path = resolve_model_path(self.tokenizer_path, use_shm=self.use_shm)
            if architecture == "openvla_oft":
                from transformers import AutoTokenizer

                from verl_vla.models.openvla_oft.processing_prismatic import (
                    PrismaticImageProcessor,
                    PrismaticProcessor,
                )

                image_processor = PrismaticImageProcessor.from_pretrained(self.local_tokenizer_path)
                tokenizer = AutoTokenizer.from_pretrained(
                    self.local_tokenizer_path,
                    trust_remote_code=self.trust_remote_code,
                )
                self.processor = PrismaticProcessor(image_processor=image_processor, tokenizer=tokenizer)
                self.tokenizer = self.processor.tokenizer
            else:
                self.tokenizer = hf_tokenizer(self.local_tokenizer_path, trust_remote_code=self.trust_remote_code)
                self.processor = None

        # verl's generic worker only uses this object for optional MFU
        # accounting. Model construction never consumes it.
        self.hf_config = SimpleNamespace(model_type="vla_native")
        self.generation_config = None
        self.share_embeddings_and_output_weights = False
