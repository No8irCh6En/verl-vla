# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0

"""Shared actor-only update driver for durable group-RL collection windows."""

from __future__ import annotations

import json
from pathlib import Path
from pprint import pprint
from typing import Any, cast

import ray
from hydra.utils import instantiate
from omegaconf import OmegaConf

from verl_vla.train_cluster import TrainCluster
from verl_vla.trainer.grfpo.collection_store import CollectionWindowStore
from verl_vla.utils.ray_utils import ensure_ray_initialized, get_controller_remote_options
from verl_vla.workflows.grfpo_fragmented_collection import collection_window_spec


def run_spooled_group_rl(config, *, actor_objective: str):
    """Train one closed window with its matching method-specific actor."""

    if actor_objective not in {"fpo", "flow_grpo"}:
        raise ValueError(f"Unsupported actor_objective={actor_objective!r}.")
    ensure_ray_initialized(config)
    remote_options = get_controller_remote_options(config)
    return ray.get(_run_spooled_group_rl_remote.options(**remote_options).remote(config, actor_objective))


@ray.remote
def _run_spooled_group_rl_remote(config, actor_objective: str):
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.set_struct(config, False)
    OmegaConf.resolve(config)
    # Validate the closed-window identity before allocating an actor rank or
    # loading Fast-WAM.  This catches a method/trace mismatch on CPU.
    store = CollectionWindowStore(
        config.collection.collection_root,
        collection_window_spec(config.collection),
    )
    store.initialize()
    if store.spec.actor_objective != actor_objective:
        raise ValueError(
            "Requested trainer does not match the immutable collection trace: "
            f"window={store.spec.actor_objective!r}, trainer={actor_objective!r}."
        )

    if actor_objective == "fpo":
        from verl_vla.trainer.grfpo.spooled_trainer import SpooledGRFPOTrainer

        trainer_class = SpooledGRFPOTrainer
    else:
        from verl_vla.trainer.flow_grpo.spooled_trainer import SpooledFlowGRPOTrainer

        trainer_class = SpooledFlowGRPOTrainer

    cluster = TrainCluster(instantiate(config.cluster, _recursive_=False))
    cluster.start()
    try:
        if bool(config.get("checkpoint_portable_export_only", False)):
            checkpoint_state = cluster.load_checkpoint()
            export_result = cluster.actor_worker_group.export_portable_checkpoint(
                str(Path(config.cluster.checkpoint.resume_from_path) / "actor")
            )
            result = {
                "checkpoint_portable_export_only": True,
                "checkpoint_state": checkpoint_state,
                "export_result": export_result,
                "train_world_size": cluster.train_world_size,
            }
            print(result, flush=True)
            return result
        if bool(config.get("checkpoint_load_smoke_only", False)):
            checkpoint_state = cluster.load_checkpoint()
            result = {
                "checkpoint_load_smoke_only": True,
                "checkpoint_state": checkpoint_state,
                "train_world_size": cluster.train_world_size,
            }
            print(result, flush=True)
            return result
        trainer = trainer_class(
            trainer_config=config.trainer,
            cluster=cluster,
            store=store,
            tracking_config=cast(dict[str, Any], OmegaConf.to_container(config, resolve=True)),
            counterfactual_replay_receipt_path=config.get("counterfactual_replay_receipt_path"),
        )
        if bool(config.get("task_gradient_diagnostic_only", False)):
            output_path = config.get("task_gradient_diagnostic_output")
            if not output_path:
                raise ValueError("task_gradient_diagnostic_only requires task_gradient_diagnostic_output.")
            targets_path = config.get("task_gradient_diagnostic_targets_json")
            targets: dict[str, str] = {}
            if targets_path not in (None, ""):
                targets_record = json.loads(Path(str(targets_path)).read_text(encoding="utf-8"))
                if not isinstance(targets_record, dict) or not all(
                    isinstance(label, str) and isinstance(path, str)
                    for label, path in targets_record.items()
                ):
                    raise TypeError(
                        "task_gradient_diagnostic_targets_json must contain a JSON object "
                        "mapping labels to checkpoint paths."
                    )
                targets = targets_record
            return trainer.run_task_gradient_diagnostic(
                str(output_path),
                target_checkpoints=targets,
                include_group_diagnostics=bool(
                    config.get("task_gradient_diagnostic_include_groups", True)
                ),
            )
        return trainer.fit_one_update()
    finally:
        cluster.shutdown()


__all__ = ["run_spooled_group_rl"]
