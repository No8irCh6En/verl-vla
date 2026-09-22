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

"""FSDP checkpoint manager with native policy export support."""

from __future__ import annotations

import os
import json
from pathlib import Path

import torch
import torch.distributed
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.tensor import DTensor, distribute_tensor
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.fsdp_utils import (
    fsdp_version,
    get_fsdp_full_state_dict,
    merged_lora_context,
    normalize_peft_param_name,
)


def _contains_peft_model(model: torch.nn.Module) -> bool:
    from peft import PeftModel

    return any(isinstance(module, PeftModel) for module in model.modules())


def _unwrap_trainable_model(model: torch.nn.Module) -> torch.nn.Module:
    return model._fsdp_wrapped_module if fsdp_version(model) == 1 else model


def _save_lora_adapter(
    model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
    output_dir: str | Path,
) -> None:
    """Save the nested native policy adapter through PEFT's standard API."""

    from peft import PeftModel

    trainable_model = _unwrap_trainable_model(model)
    policy = trainable_model.policy
    if not isinstance(policy, PeftModel):
        raise TypeError(f"Expected a PEFT policy, got {type(policy).__name__}")

    policy_state_dict = trainable_model.extract_policy_state_dict(state_dict)
    policy.save_pretrained(
        output_dir,
        state_dict=policy_state_dict,
        safe_serialization=True,
    )


class NativePolicyFSDPCheckpointManager(FSDPCheckpointManager):
    """Delegate ``hf_model`` export to the VLA adapter instead of AutoModel."""

    _PORTABLE_MODEL = "portable_full_model.pt"
    _PORTABLE_OPTIMIZER = "portable_full_optimizer.pt"
    _PORTABLE_METADATA = "portable_checkpoint.json"

    @staticmethod
    def _atomic_torch_save(value, path: Path) -> None:
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(value, temporary_path)
        os.replace(temporary_path, path)

    def export_portable_checkpoint(self, local_path: str) -> dict[str, object]:
        """Export a world-size-independent full model/optimizer checkpoint.

        veRL's regular FSDP2 checkpoint contains DTensors tied to the mesh that
        created it.  The official distributed-checkpoint state-dict API emits
        canonical parameter names and full CPU tensors, allowing a later run
        with a different FSDP world size to reshard both model and Adam state.
        """

        if self.optimizer is None:
            raise RuntimeError("Portable checkpoint export requires an optimizer.")
        options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        model_state, optimizer_state = get_state_dict(
            self.model,
            self.optimizer,
            options=options,
        )
        output_dir = Path(local_path)
        if self.rank == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
            self._atomic_torch_save(model_state, output_dir / self._PORTABLE_MODEL)
            self._atomic_torch_save(optimizer_state, output_dir / self._PORTABLE_OPTIMIZER)
            metadata = {
                "format": "torch.distributed.checkpoint.full_state_dict",
                "format_version": 1,
                "source_world_size": self.world_size,
                "model_file": self._PORTABLE_MODEL,
                "optimizer_file": self._PORTABLE_OPTIMIZER,
            }
            metadata_path = output_dir / self._PORTABLE_METADATA
            temporary_path = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
            temporary_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            os.replace(temporary_path, metadata_path)
        torch.distributed.barrier()
        return {
            "portable_checkpoint": str(output_dir),
            "source_world_size": self.world_size,
            "rank": self.rank,
        }

    def _load_portable_checkpoint(self, local_path: str) -> None:
        if self.optimizer is None:
            raise RuntimeError("Portable checkpoint load requires an optimizer.")
        checkpoint_dir = Path(local_path)
        model_path = checkpoint_dir / self._PORTABLE_MODEL
        optimizer_path = checkpoint_dir / self._PORTABLE_OPTIMIZER
        if not model_path.is_file() or not optimizer_path.is_file():
            raise FileNotFoundError(
                "Checkpoint world size differs from the current FSDP mesh, but portable "
                f"state is incomplete under {checkpoint_dir}."
            )
        print(
            f"[Rank {self.rank}] Loading portable full model/optimizer for local reshard",
            flush=True,
        )
        # Rank-0 broadcast of a 46-GB custom FSDP2 state can stall inside
        # torch.distributed.checkpoint before launching any NCCL work.  The
        # checkpoint resides on the node-visible filesystem and host memory is
        # explicitly sized for all ranks, so let each rank mmap the same full
        # state and independently derive its local shard instead.
        model_state = torch.load(model_path, map_location="cpu", weights_only=False, mmap=True)
        print(f"[Rank {self.rank}] Portable model mapped; applying explicit DTensor reshard", flush=True)
        target_state = self.model.state_dict()
        if model_state.keys() != target_state.keys():
            missing = sorted(target_state.keys() - model_state.keys())
            unexpected = sorted(model_state.keys() - target_state.keys())
            raise RuntimeError(
                "Portable model keys do not match the current model: "
                f"missing={missing[:10]}, unexpected={unexpected[:10]}"
            )
        sharded_model_state = {}
        for name, target in target_state.items():
            source = model_state[name]
            if isinstance(target, DTensor):
                if not isinstance(source, torch.Tensor) or isinstance(source, DTensor):
                    raise TypeError(f"Portable model tensor {name!r} has invalid type {type(source).__name__}.")
                if source.shape != target.shape:
                    raise ValueError(
                        f"Portable model tensor {name!r} shape mismatch: {source.shape} != {target.shape}."
                    )
                sharded_model_state[name] = distribute_tensor(
                    source.to(device=target.device),
                    device_mesh=target.device_mesh,
                    placements=target.placements,
                    src_data_rank=None,
                )
            elif isinstance(target, torch.Tensor):
                sharded_model_state[name] = source.to(device=target.device, dtype=target.dtype)
            else:
                sharded_model_state[name] = source
        incompatible = self.model.load_state_dict(sharded_model_state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "Portable checkpoint load was not strict: "
                f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
            )
        del sharded_model_state, target_state, model_state
        torch.cuda.empty_cache()
        print(f"[Rank {self.rank}] Portable model reshard applied; loading optimizer", flush=True)

        optimizer_state = torch.load(optimizer_path, map_location="cpu", weights_only=False, mmap=True)
        set_optimizer_state_dict(
            self.model,
            self.optimizer,
            optimizer_state,
            options=StateDictOptions(
                full_state_dict=True,
                cpu_offload=False,
                broadcast_from_rank0=False,
                strict=True,
            ),
        )
        del optimizer_state
        print(f"[Rank {self.rank}] Portable optimizer reshard applied", flush=True)

        # Scheduler state is identical across ranks. RNG state is restored from
        # the source rank-0 file; subsequent rank-local data partitioning keeps
        # stochastic FPO inputs distinct and deterministic.
        if self.should_load_extra:
            source_extra_path = checkpoint_dir / "extra_state_world_size_1_rank_0.pt"
            if not source_extra_path.is_file():
                raise FileNotFoundError(f"Missing portable extra state source: {source_extra_path}")
            extra_state = torch.load(source_extra_path, map_location="cpu", weights_only=False)
            if "rng" in extra_state:
                self.load_rng_state(extra_state["rng"])
            scheduler_state = extra_state.get("lr_scheduler")
            if scheduler_state is not None and self.lr_scheduler is not None:
                self.lr_scheduler.load_state_dict(scheduler_state)
        torch.distributed.barrier()

    def load_checkpoint(self, local_path: str, hdfs_path: str = None, del_local_after_load=False):
        if local_path is None:
            return None
        fsdp_config_path = Path(local_path) / "fsdp_config.json"
        if fsdp_config_path.is_file():
            saved_world_size = int(json.loads(fsdp_config_path.read_text(encoding="utf-8"))["world_size"])
            if saved_world_size != self.world_size:
                self._load_portable_checkpoint(local_path)
                return None
        return super().load_checkpoint(local_path, hdfs_path, del_local_after_load)

    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        should_export = self.should_save_hf_model
        original_contents = self.checkpoint_save_contents
        adapter = _unwrap_trainable_model(self.model)
        model_config = adapter.config
        auto_map = getattr(model_config, "auto_map", None)
        if should_export:
            self.checkpoint_save_contents = [item for item in original_contents if item != "hf_model"]
            # verl's generic checkpoint manager sees the verl-vla wrapper and
            # cannot copy native Transformers custom code from it. The adapter
            # owns that export and restores the native config below.
            if auto_map is not None:
                del model_config.auto_map
        try:
            super().save_checkpoint(
                local_path,
                hdfs_path=hdfs_path,
                global_step=global_step,
                max_ckpt_to_keep=max_ckpt_to_keep,
            )
        finally:
            self.checkpoint_save_contents = original_contents
            if auto_map is not None:
                model_config.auto_map = auto_map

        if not should_export:
            return

        has_lora = _contains_peft_model(self.model)
        if has_lora:
            state_dict = get_fsdp_full_state_dict(self.model, offload_to_cpu=True, rank0_only=True)
            if self.rank == 0:
                _save_lora_adapter(
                    self.model,
                    state_dict,
                    Path(local_path) / "lora_adapter",
                )
                del state_dict
            torch.distributed.barrier()

            with merged_lora_context(self.model, backup_adapters=True):
                state_dict = get_fsdp_full_state_dict(self.model, offload_to_cpu=True, rank0_only=True)
                state_dict = {name: tensor.clone() for name, tensor in state_dict.items()}
            state_dict = normalize_peft_param_name(state_dict)
        else:
            state_dict = get_fsdp_full_state_dict(self.model, offload_to_cpu=True, rank0_only=True)
        if self.rank == 0:
            export_policy = getattr(adapter, "export_policy", None)
            output_dir = os.path.join(local_path, "huggingface")
            if callable(export_policy):
                export_policy(output_dir, state_dict=state_dict)
            else:
                save_pretrained = getattr(adapter, "save_pretrained", None)
                if not callable(save_pretrained):
                    raise TypeError(
                        f"{type(adapter).__name__} implements neither export_policy() nor save_pretrained()"
                    )
                save_pretrained(output_dir, state_dict=state_dict)
            del state_dict
        torch.distributed.barrier()
