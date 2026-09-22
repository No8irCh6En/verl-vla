# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from verl.trainer.config import CheckpointConfig
from verl.workers.engine_workers import TrainingWorker

from verl_vla.train_cluster.checkpoint import CheckpointHelper
from verl_vla.train_cluster.config import TrainClusterCheckpointConfig
from verl_vla.workers.engine.fpo import training_worker as fpo_training_worker
from verl_vla.workers.engine.fpo.training_worker import FPOTrainingWorker


class FakeActorWorkerGroup:
    def __init__(self) -> None:
        self.saved: list[tuple] = []
        self.loaded: list[tuple] = []

    def save_checkpoint(self, local_path, remote_path, global_step, max_ckpt_to_keep):
        Path(local_path).mkdir(parents=True, exist_ok=True)
        self.saved.append((local_path, remote_path, global_step, max_ckpt_to_keep))

    def load_checkpoint(self, local_path, del_local_after_load=False):
        self.loaded.append((local_path, del_local_after_load))


def _populated_optimizer() -> tuple[torch.nn.Linear, torch.optim.Adam, torch.optim.lr_scheduler.ConstantLR]:
    module = torch.nn.Linear(3, 1)
    optimizer = torch.optim.Adam(module.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    module(torch.ones(2, 3)).square().mean().backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    return module, optimizer, scheduler


def test_official_checkpoint_helper_save_and_auto_reload(tmp_path):
    worker_group = FakeActorWorkerGroup()
    config = TrainClusterCheckpointConfig(
        resume_mode="auto",
        default_local_dir=str(tmp_path / "checkpoints"),
        max_actor_ckpt_to_keep=2,
    )
    actor_config = SimpleNamespace(checkpoint=CheckpointConfig(async_save=False))
    helper = CheckpointHelper(config, actor_config, worker_group)

    helper.save(global_step=2)
    marker = tmp_path / "checkpoints" / "latest_checkpointed_iteration.txt"
    assert marker.read_text() == "2"
    assert worker_group.saved == [
        (str(tmp_path / "checkpoints" / "global_step_2" / "actor"), None, 2, 2)
    ]

    assert helper.load() == (2, str(tmp_path / "checkpoints" / "global_step_2"))
    assert worker_group.loaded == [
        (str(tmp_path / "checkpoints" / "global_step_2" / "actor"), False)
    ]


def test_fpo_value_optimizer_sidecar_save_and_reload(tmp_path, monkeypatch):
    _module, optimizer, scheduler = _populated_optimizer()
    original_state = optimizer.state_dict()

    monkeypatch.setattr(TrainingWorker, "save_checkpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(TrainingWorker, "load_checkpoint", lambda *args, **kwargs: "base-loaded")
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(fpo_training_worker, "get_device_name", lambda: "cpu")
    monkeypatch.setattr(fpo_training_worker, "get_device_id", lambda: None)

    saver = object.__new__(FPOTrainingWorker)
    saver._fpo_initialized = True
    saver.actor_config = SimpleNamespace(value=SimpleNamespace(enabled=True))
    saver.value_optimizer = optimizer
    saver.value_scheduler = scheduler
    FPOTrainingWorker.save_checkpoint.__wrapped__(saver, str(tmp_path), global_step=2)

    sidecar = tmp_path / "fpo_value_optim_world_size_1_rank_0.pt"
    assert sidecar.is_file()

    _restored_module, restored_optimizer, restored_scheduler = _populated_optimizer()
    restored_optimizer.state.clear()
    loader = object.__new__(FPOTrainingWorker)
    loader._fpo_initialized = True
    loader.actor_config = SimpleNamespace(value=SimpleNamespace(enabled=True))
    loader.value_optimizer = restored_optimizer
    loader.value_scheduler = restored_scheduler
    result = FPOTrainingWorker.load_checkpoint.__wrapped__(loader, str(tmp_path))

    assert result == "base-loaded"
    restored_state = restored_optimizer.state_dict()
    assert restored_state["param_groups"] == original_state["param_groups"]
    for parameter_id, state in original_state["state"].items():
        for name, expected in state.items():
            torch.testing.assert_close(restored_state["state"][parameter_id][name], expected)


def test_critic_free_fpo_checkpoint_has_no_value_optimizer_sidecar(tmp_path, monkeypatch):
    monkeypatch.setattr(TrainingWorker, "save_checkpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(TrainingWorker, "load_checkpoint", lambda *args, **kwargs: "base-loaded")

    worker = object.__new__(FPOTrainingWorker)
    worker._fpo_initialized = True
    worker.actor_config = SimpleNamespace(value=SimpleNamespace(enabled=False))

    FPOTrainingWorker.save_checkpoint.__wrapped__(worker, str(tmp_path), global_step=1)
    assert not list(tmp_path.glob("fpo_value_optim_*.pt"))
    assert FPOTrainingWorker.load_checkpoint.__wrapped__(worker, str(tmp_path)) == "base-loaded"
