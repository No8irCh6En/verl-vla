# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0

"""Train one GRFPO update from a closed durable collection window."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
from hydra.utils import instantiate
from verl import DataProto
from verl.utils.metric import reduce_metrics

from verl_vla.train_cluster import TrainCluster

from .collection_store import CollectionWindowStore
from .config import GRFPOTrainerConfig
from .grfpo_ray_trainer import (
    apply_grpo_outcome_advantages,
    concatenate_rollout_groups_with_padding,
    prepare_fpo_actor_input,
)
from .rollout_batch import compact_valid_policy_chunks_for_fpo


class SpooledGRFPOTrainer:
    """Actor-only, one-update trainer over previously collected raw groups."""

    actor_objective = "fpo"

    def __init__(
        self,
        *,
        trainer_config,
        cluster: TrainCluster,
        store: CollectionWindowStore,
        tracking_config: dict[str, Any],
        counterfactual_replay_receipt_path: str | Path | None = None,
    ) -> None:
        self.trainer_config: GRFPOTrainerConfig = instantiate(trainer_config)
        self.cluster = cluster
        self.store = store
        self.tracking_config = tracking_config
        self.counterfactual_replay_receipt_path = (
            None
            if counterfactual_replay_receipt_path in (None, "")
            else Path(counterfactual_replay_receipt_path).expanduser().resolve()
        )
        actor = tracking_config["cluster"]["actor_rollout_ref"]["actor"]
        model = tracking_config["cluster"]["actor_rollout_ref"]["model"]
        self.actor_micro_batch_size = int(actor["micro_batch_size"])
        if self.actor_micro_batch_size <= 0:
            raise ValueError("GRFPO actor micro_batch_size must be positive.")
        if self.store.spec.actor_objective != self.actor_objective:
            raise ValueError(
                "Collection/trainer actor-objective mismatch: "
                f"window={self.store.spec.actor_objective!r}, trainer={self.actor_objective!r}."
            )
        self._validate_actor_contract(actor=actor, model=model)

    def _validate_actor_contract(self, *, actor: dict[str, Any], model: dict[str, Any]) -> None:
        if bool(actor["value"]["enabled"]) or float(actor["vf_coef"]) != 0:
            raise ValueError("Spooled Group-Relative FPO is critic-free.")
        if bool(actor["normalize_advantages"]):
            raise ValueError("Spooled Group-Relative FPO forbids minibatch advantage normalization.")
        if not bool(model["adapter"]["fpo"]["enabled"]):
            raise ValueError("Spooled Group-Relative FPO requires model.adapter.fpo.enabled=true.")
        if bool(model["adapter"]["fpo"]["value_enabled"]):
            raise ValueError("Spooled Group-Relative FPO does not use fpo_value_head.")
        if bool(model["adapter"].get("flow_grpo", {}).get("enabled", False)):
            raise ValueError("Spooled Group-Relative FPO must not enable the Flow-GRPO rollout objective.")

    def _load_actor_input(self) -> tuple[DataProto, list[dict[str, Any]]]:
        groups = self.store.load_selected_groups()
        if not groups:
            raise RuntimeError("Closed GRFPO window selected no groups.")
        rollouts, end_observations, records = zip(*groups, strict=True)
        rollout, end_obs = concatenate_rollout_groups_with_padding(
            list(rollouts),
            list(end_observations),
        )
        actor_input = prepare_fpo_actor_input(
            rollout,
            end_obs,
            trainer_config=self.trainer_config,
            global_steps=self.store.spec.intended_update,
        )
        apply_grpo_outcome_advantages(
            actor_input,
            group_records=list(records),
            group_size=self.store.spec.group_size,
            reward_source=self.trainer_config.group_reward_source,
            accepted_group_reduction=self.trainer_config.accepted_group_reduction,
        )
        # Stable integer labels let an explicit diagnostic replay partition the
        # same logical batch without parsing prompts or weakening group
        # boundaries. They are inert during the normal actor update.
        group_size = int(self.store.spec.group_size)
        rollout_steps = int(actor_input.batch["info.valids"].shape[1])
        task_names = sorted({str(record["task_name"]) for record in records})
        task_to_index = {task: index for index, task in enumerate(task_names)}
        trajectory_groups = torch.arange(len(records), dtype=torch.long).repeat_interleave(group_size)
        trajectory_tasks = torch.tensor(
            [task_to_index[str(record["task_name"])] for record in records for _ in range(group_size)],
            dtype=torch.long,
        )
        actor_input.batch["fpo.group_index"] = trajectory_groups.unsqueeze(1).expand(-1, rollout_steps).clone()
        actor_input.batch["fpo.task_index"] = trajectory_tasks.unsqueeze(1).expand(-1, rollout_steps).clone()
        actor_input.meta_info["fpo_task_names"] = task_names
        actor_input = compact_valid_policy_chunks_for_fpo(
            actor_input,
            actor_world_size=int(self.cluster.actor_worker_group.world_size),
            micro_batch_size=self.actor_micro_batch_size,
        )
        return actor_input, list(records)

    def fit_one_update(self) -> dict[str, Any]:
        spec = self.store.spec
        if spec.intended_update != spec.policy.rollout_policy_version + 1:
            raise ValueError(
                "A spooled trainer performs exactly one update: "
                f"theta_old={spec.policy.rollout_policy_version}, intended_update={spec.intended_update}."
            )
        with self.store.training_transaction(self.counterfactual_replay_receipt_path) as existing_receipt:
            if existing_receipt is not None:
                print(f"training_already_complete={json.dumps(existing_receipt, sort_keys=True)}", flush=True)
                return existing_receipt

            checkpoint_state = self.cluster.load_checkpoint()
            expected_checkpoint = spec.policy.trainer_checkpoint_path
            if expected_checkpoint is None:
                if checkpoint_state is not None or spec.policy.rollout_policy_version != 0:
                    raise RuntimeError("theta0 must start from the immutable native model without a verl checkpoint.")
            else:
                if checkpoint_state is None:
                    raise RuntimeError(f"Expected theta_old checkpoint {expected_checkpoint}, but none was loaded.")
                loaded_step, loaded_path = checkpoint_state
                if int(loaded_step) != spec.policy.rollout_policy_version:
                    raise RuntimeError(
                        f"Loaded checkpoint step {loaded_step} does not match theta_old "
                        f"{spec.policy.rollout_policy_version}."
                    )
                if Path(loaded_path).resolve() != Path(expected_checkpoint).resolve():
                    raise RuntimeError(
                        f"Loaded checkpoint path {loaded_path} does not match collected policy {expected_checkpoint}."
                    )

            actor_input, records = self._load_actor_input()
            compaction = {
                key: int(actor_input.meta_info[key])
                for key in (
                    "fpo_original_physical_slots",
                    "fpo_valid_policy_chunks",
                    "fpo_sync_dummy_slots",
                    "fpo_compacted_physical_slots",
                )
            }
            print(f"fpo_actor_compaction={json.dumps(compaction, sort_keys=True)}", flush=True)
            start = time.perf_counter()
            update_output = self.cluster.train(actor_input, async_update=False)
            train_wall_s = time.perf_counter() - start
            metrics = reduce_metrics(update_output.meta_info.get("metrics", {}))
            epochs_run = float(metrics.get("fpo/epochs_run", 0.0))
            if train_wall_s > 0 and epochs_run > 0:
                metrics["throughput/fpo_valid_chunks_per_s"] = (
                    compaction["fpo_valid_policy_chunks"] * epochs_run / train_wall_s
                )
            # Checkpoint first. If preemption occurs before this finishes, the
            # retained spools allow the entire deterministic update to rerun.
            self.cluster.save_checkpoint(spec.intended_update)
            assert self.cluster.checkpoint_helper is not None
            checkpoint_dir = str(Path(self.cluster.checkpoint_helper.step_dir(spec.intended_update)).resolve())
            receipt = {
                "actor_objective": self.actor_objective,
                "rollout_policy_version": spec.policy.rollout_policy_version,
                "new_policy_version": spec.intended_update,
                "selected_groups": len(records),
                "accepted_training_trajectories": len(records) * spec.group_size,
                "actor_world_size": int(self.cluster.actor_worker_group.world_size),
                "actor_micro_batch_size": self.actor_micro_batch_size,
                "accepted_group_reduction": self.trainer_config.accepted_group_reduction,
                "actor_batch_compaction": compaction,
                "checkpoint_dir": checkpoint_dir,
                "train_wall_s": train_wall_s,
                "metrics": metrics,
                "completed_at_unix_s": time.time(),
            }
            self.store.write_training_receipt(receipt, self.counterfactual_replay_receipt_path)
            print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False), flush=True)
            return receipt

    def run_task_gradient_diagnostic(
        self,
        output_path: str | Path,
        *,
        target_checkpoints: dict[str, str] | None = None,
        include_group_diagnostics: bool = True,
    ) -> dict[str, Any]:
        """Replay a closed batch at theta_old without changing model/optimizer state."""

        spec = self.store.spec
        checkpoint_state = self.cluster.load_checkpoint()
        expected_checkpoint = spec.policy.trainer_checkpoint_path
        if expected_checkpoint is None:
            if checkpoint_state is not None or spec.policy.rollout_policy_version != 0:
                raise RuntimeError("theta0 diagnostics must start from the immutable native model.")
        else:
            if checkpoint_state is None:
                raise RuntimeError(f"Expected theta_old checkpoint {expected_checkpoint}, but none was loaded.")
            loaded_step, loaded_path = checkpoint_state
            if int(loaded_step) != spec.policy.rollout_policy_version:
                raise RuntimeError(
                    f"Loaded checkpoint step {loaded_step} does not match theta_old "
                    f"{spec.policy.rollout_policy_version}."
                )
            if Path(loaded_path).resolve() != Path(expected_checkpoint).resolve():
                raise RuntimeError(
                    f"Loaded checkpoint path {loaded_path} does not match collected policy {expected_checkpoint}."
                )

        actor_input, records = self._load_actor_input()
        actor_input.meta_info["task_gradient_target_checkpoints"] = dict(target_checkpoints or {})
        actor_input.meta_info["task_gradient_include_groups"] = bool(include_group_diagnostics)
        result = self.cluster.diagnose_fpo_task_gradients(actor_input)
        diagnostics = dict(result.meta_info["task_gradient_diagnostics"])
        diagnostics.update(
            {
                "window": str(self.store.root.resolve()),
                "rollout_policy_version": spec.policy.rollout_policy_version,
                "intended_update": spec.intended_update,
                "selected_groups": len(records),
                "accepted_training_trajectories": len(records) * spec.group_size,
            }
        )
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(diagnostics, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(diagnostics, indent=2, sort_keys=True, allow_nan=False), flush=True)
        return diagnostics
