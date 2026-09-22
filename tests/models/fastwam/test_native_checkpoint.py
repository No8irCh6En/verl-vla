from pathlib import Path

import pytest
import torch

from verl_vla.models.fastwam.native_checkpoint import (
    build_native_fastwam_payload,
    load_verl_fastwam_actor_into_native_policy,
    resolve_actor_checkpoint,
)


def _wrapper_state():
    return {
        "policy.mot.mixtures.video.block.weight": torch.tensor([1.0]),
        "policy.mot.mixtures.action.block.weight": torch.tensor([2.0]),
        "policy.dit.mixtures.video.block.weight": torch.tensor([1.0]),
        "policy.dit.mixtures.action.block.weight": torch.tensor([2.0]),
        "policy.proprio_encoder.weight": torch.tensor([3.0]),
        "policy.text_encoder.weight": torch.tensor([4.0]),
    }


def test_native_payload_uses_mot_once_and_preserves_proprio():
    payload = build_native_fastwam_payload(_wrapper_state(), step=7)

    assert payload["step"] == 7
    assert set(payload["mot"]) == {
        "mixtures.video.block.weight",
        "mixtures.action.block.weight",
    }
    assert set(payload["proprio_encoder"]) == {"weight"}
    assert "dit" not in payload


def test_resolve_actor_checkpoint_selects_rank_zero_for_multirank_directory(tmp_path: Path):
    actor_dir = tmp_path / "global_step_3" / "actor"
    actor_dir.mkdir(parents=True)
    rank_files = []
    for rank in range(2):
        path = actor_dir / f"model_world_size_2_rank_{rank}.pt"
        path.touch()
        rank_files.append(path)

    model_path, resolved_actor_dir, step = resolve_actor_checkpoint(actor_dir.parent)

    assert model_path == rank_files[0].resolve()
    assert resolved_actor_dir == actor_dir.resolve()
    assert step == 3


def test_resolve_actor_checkpoint_allows_explicit_replica_rank_file(tmp_path: Path):
    actor_dir = tmp_path / "global_step_3" / "actor"
    actor_dir.mkdir(parents=True)
    rank_files = []
    for rank in range(2):
        path = actor_dir / f"model_world_size_2_rank_{rank}.pt"
        path.touch()
        rank_files.append(path)

    model_path, resolved_actor_dir, step = resolve_actor_checkpoint(rank_files[0])

    assert model_path == rank_files[0].resolve()
    assert resolved_actor_dir == actor_dir.resolve()
    assert step == 3


def test_rl_actor_can_be_applied_to_native_policy_without_export(tmp_path: Path):
    class _Mixtures(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.video = torch.nn.Linear(1, 1, bias=False)
            self.action = torch.nn.Linear(1, 1, bias=False)

    class _NativePolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mot = torch.nn.Module()
            self.mot.mixtures = _Mixtures()
            self.proprio_encoder = torch.nn.Linear(1, 1, bias=False)

    actor_dir = tmp_path / "global_step_9" / "actor"
    actor_dir.mkdir(parents=True)
    state = {
        "policy.mot.mixtures.video.weight": torch.tensor([[1.0]]),
        "policy.mot.mixtures.action.weight": torch.tensor([[2.0]]),
        "policy.proprio_encoder.weight": torch.tensor([[3.0]]),
    }
    torch.save(state, actor_dir / "model_world_size_1_rank_0.pt")
    policy = _NativePolicy()

    metadata = load_verl_fastwam_actor_into_native_policy(source=actor_dir.parent, native_policy=policy)

    assert metadata["source_verl_global_step"] == 9
    torch.testing.assert_close(policy.mot.mixtures.video.weight, torch.tensor([[1.0]]))
    torch.testing.assert_close(policy.mot.mixtures.action.weight, torch.tensor([[2.0]]))
    torch.testing.assert_close(policy.proprio_encoder.weight, torch.tensor([[3.0]]))
