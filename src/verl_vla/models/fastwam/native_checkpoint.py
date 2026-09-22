# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Export a full verl-vla Fast-WAM actor checkpoint in upstream-native form."""

from __future__ import annotations

import hashlib
import importlib
import json
import re
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

NATIVE_WEIGHTS_RELATIVE_PATH = Path("checkpoints/weights/step_020000.pt")
DATASET_STATS_FILENAME = "dataset_stats.json"
EXPORT_MANIFEST_FILENAME = "export_manifest.json"


def _load_verl_actor_state(model_path: Path) -> Mapping[str, torch.Tensor]:
    """Load a trusted verl actor state with PyTorch 2.6 DTensor support."""

    # PyTorch 2.6's weights-only unpickler deliberately refuses DTensor globals
    # until the DTensor package has registered them.  Actor checkpoints written
    # by the official FSDP manager contain DTensor metadata even at world size
    # one, so registration is required by both in-memory evaluation and export.
    importlib.import_module("torch.distributed.tensor")
    try:
        wrapper_state = torch.load(model_path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:  # PyTorch before mmap/weights_only support.
        wrapper_state = torch.load(model_path, map_location="cpu")
    if not isinstance(wrapper_state, Mapping):
        raise TypeError(f"Expected actor state_dict mapping, got {type(wrapper_state).__name__}")
    return wrapper_state


def _native_local_tensor(value: torch.Tensor, *, name: str) -> torch.Tensor:
    """Unwrap a DTensor only when this rank owns the complete global tensor.

    A replicated-data HSDP mesh can have total mesh size greater than one
    while its FSDP axis has size one.  Such checkpoints are named as
    world-size-N artifacts, but every rank-local tensor is complete and is
    therefore safe to consume independently.  A genuinely sharded FSDP
    tensor remains rejected by the local/global shape check below.
    """

    dtensor_module = importlib.import_module("torch.distributed.tensor")
    dtensor_type = dtensor_module.DTensor
    if not isinstance(value, dtensor_type):
        return value
    local = value.to_local()
    if tuple(local.shape) != tuple(value.shape):
        raise ValueError(
            f"Cannot export sharded DTensor {name!r} from its rank-local value: "
            f"global={tuple(value.shape)} local={tuple(local.shape)} mesh={value.device_mesh} "
            f"placements={value.placements}. A live full-state gather is required."
        )
    return local


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_native_fastwam_payload(
    wrapper_state_dict: Mapping[str, torch.Tensor],
    *,
    step: int,
    torch_dtype: str = "torch.bfloat16",
) -> dict[str, Any]:
    """Map the verl wrapper state to Fast-WAM's ``FastWAM.save_checkpoint`` schema.

    The full actor checkpoint contains both ``policy.mot`` and the alias
    ``policy.dit``.  The upstream checkpoint owns ``mot`` only, so exporting
    both aliases would duplicate several gigabytes and produce a schema the
    upstream loader does not understand.
    """

    mot_prefix = "policy.mot."
    proprio_prefix = "policy.proprio_encoder."
    mot = {
        name.removeprefix(mot_prefix): _native_local_tensor(tensor, name=name)
        for name, tensor in wrapper_state_dict.items()
        if name.startswith(mot_prefix)
    }
    proprio = {
        name.removeprefix(proprio_prefix): _native_local_tensor(tensor, name=name)
        for name, tensor in wrapper_state_dict.items()
        if name.startswith(proprio_prefix)
    }
    if not mot:
        raise ValueError("Actor checkpoint contains no policy.mot.* tensors.")
    required_mot_scopes = ("mixtures.video.", "mixtures.action.")
    missing = [scope for scope in required_mot_scopes if not any(name.startswith(scope) for name in mot)]
    if missing:
        raise ValueError(f"Actor checkpoint is missing native Fast-WAM MoT scopes: {missing}")
    if not proprio:
        raise ValueError("Actor checkpoint contains no policy.proprio_encoder.* tensors.")

    return {
        "mot": mot,
        "proprio_encoder": proprio,
        "step": int(step),
        "torch_dtype": str(torch_dtype),
    }


def resolve_actor_checkpoint(source: str | Path) -> tuple[Path, Path, int]:
    """Resolve a global-step directory, actor directory, or explicit model file.

    A multi-rank directory resolves to rank zero, but this does *not* imply
    that an arbitrary FSDP shard is accepted.  ``build_native_fastwam_payload``
    subsequently unwraps every requested DTensor through
    ``_native_local_tensor`` and rejects it unless its local shape equals its
    global shape.  Replicated-data HSDP checkpoints therefore work directly,
    while genuinely sharded FSDP checkpoints still fail closed.
    """

    source = Path(source).expanduser().resolve()
    explicit_model_file = source.is_file()
    if explicit_model_file:
        model_path = source
        actor_dir = source.parent
    elif source.name == "actor":
        actor_dir = source
        model_path = actor_dir / "model_world_size_1_rank_0.pt"
    else:
        actor_dir = source / "actor"
        model_path = actor_dir / "model_world_size_1_rank_0.pt"

    model_shards = sorted(actor_dir.glob("model_world_size_*_rank_*.pt"))
    if not model_path.is_file():
        if model_shards:
            parsed = []
            for shard in model_shards:
                match = re.fullmatch(r"model_world_size_(\d+)_rank_(\d+)\.pt", shard.name)
                if match is None:
                    raise ValueError(f"Unrecognized actor checkpoint shard name: {shard}")
                parsed.append((int(match.group(1)), int(match.group(2)), shard))
            world_sizes = {world_size for world_size, _, _ in parsed}
            if len(world_sizes) != 1:
                raise ValueError(
                    f"Actor directory mixes checkpoint world sizes {sorted(world_sizes)}: {actor_dir}"
                )
            saved_world_size = next(iter(world_sizes))
            ranks = {rank for _, rank, _ in parsed}
            if ranks != set(range(saved_world_size)):
                raise ValueError(
                    f"Incomplete actor checkpoint ranks for world_size={saved_world_size}: "
                    f"found={sorted(ranks)} directory={actor_dir}"
                )
            model_path = next(path for _, rank, path in parsed if rank == 0)
        else:
            raise FileNotFoundError(f"Fast-WAM actor state not found: {model_path}")

    match = re.search(r"global_step_(\d+)", str(source))
    step = int(match.group(1)) if match else 0
    return model_path, actor_dir, step


def load_verl_fastwam_actor_into_native_policy(
    *,
    source: str | Path,
    native_policy: torch.nn.Module,
) -> dict[str, Any]:
    """Apply a world-size-1 verl actor checkpoint directly in memory.

    This is the no-export path for evaluation: the native base model is loaded
    normally (which also constructs its processor and normalization state),
    then the RL-trained MoT/proprio tensors replace its weights. No second
    native checkpoint is materialized on disk.
    """

    model_path, _, inferred_step = resolve_actor_checkpoint(source)
    wrapper_state = _load_verl_actor_state(model_path)
    payload = build_native_fastwam_payload(wrapper_state, step=inferred_step)
    mot = getattr(native_policy, "mot", None)
    proprio_encoder = getattr(native_policy, "proprio_encoder", None)
    if not isinstance(mot, torch.nn.Module) or not isinstance(proprio_encoder, torch.nn.Module):
        raise TypeError("Native Fast-WAM policy must expose torch modules 'mot' and 'proprio_encoder'.")
    mot.load_state_dict(payload["mot"], strict=True)
    proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
    del wrapper_state, payload
    return {
        "source_verl_actor_state": str(model_path),
        "source_verl_global_step": inferred_step,
        "conversion": "in_memory_policy.mot_and_proprio_encoder",
    }


def export_verl_fastwam_checkpoint(
    *,
    source: str | Path,
    base_model_root: str | Path,
    output_root: str | Path,
    step: int | None = None,
    torch_dtype: str = "torch.bfloat16",
) -> dict[str, Any]:
    """Create a portable checkpoint loadable by native Fast-WAM/RoboDojo."""

    model_path, _, inferred_step = resolve_actor_checkpoint(source)
    base_model_root = Path(base_model_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    stats_source = base_model_root / DATASET_STATS_FILENAME
    if not stats_source.is_file():
        raise FileNotFoundError(f"Base Fast-WAM dataset statistics not found: {stats_source}")
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite native checkpoint directory: {output_root}")

    wrapper_state = _load_verl_actor_state(model_path)
    exported_step = inferred_step if step is None else int(step)
    payload = build_native_fastwam_payload(wrapper_state, step=exported_step, torch_dtype=torch_dtype)

    weights_path = output_root / NATIVE_WEIGHTS_RELATIVE_PATH
    weights_path.parent.mkdir(parents=True, exist_ok=False)
    torch.save(payload, weights_path)
    shutil.copy2(stats_source, output_root / DATASET_STATS_FILENAME)
    manifest = {
        "format": "fastwam_native_checkpoint",
        "source_verl_actor_state": str(model_path),
        "source_verl_global_step": exported_step,
        "base_model_root": str(base_model_root),
        "weights_relative_path": str(NATIVE_WEIGHTS_RELATIVE_PATH),
        "weights_size_bytes": weights_path.stat().st_size,
        "weights_sha256": sha256_file(weights_path),
        "dataset_stats_sha256": sha256_file(output_root / DATASET_STATS_FILENAME),
        "torch_dtype": str(torch_dtype),
    }
    with (output_root / EXPORT_MANIFEST_FILENAME).open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return manifest
