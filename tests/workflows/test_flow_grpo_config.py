from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
from verl.utils.config import omega_conf_to_dataclass

from verl_vla.models.fastwam import FastWAMAdapterConfig
from verl_vla.train_cluster import TrainCluster
from verl_vla.trainer.flow_grpo import FlowGRPOTrainerConfig
from verl_vla.workers.config import FlowGRPOActorConfig


def test_flow_grpo_config_composes_as_a_separate_method():
    config_dir = Path(__file__).resolve().parents[2] / "configs" / "flow_grpo"
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(config_name="fastwam_robodojo_3gpu")

    trainer = instantiate(config.trainer)
    actor = omega_conf_to_dataclass(
        config.cluster.actor_rollout_ref.actor,
        dataclass_type=FlowGRPOActorConfig,
    )
    adapter = FastWAMAdapterConfig(
        **OmegaConf.to_container(config.cluster.actor_rollout_ref.model.adapter, resolve=True)
    )

    assert isinstance(trainer, FlowGRPOTrainerConfig)
    assert isinstance(actor, FlowGRPOActorConfig)
    assert actor.value.enabled is False
    assert adapter.fpo.enabled is False
    assert adapter.flow_grpo.enabled is True
    assert adapter.flow_grpo.noise_level == pytest.approx(0.01)
    assert adapter.flow_grpo.transition_batch_size == 3
    assert actor.train_transition_fraction == pytest.approx(0.99)
    assert actor.advantage_clip_max == pytest.approx(5.0)
    assert actor.update_epochs == 2
    assert actor.clip_coef == pytest.approx(1.0e-3)
    assert trainer.group_size == 8
    assert trainer.concurrent_candidate_groups == 6
    assert config.cluster.env.env_worker.num_envs == 8
    assert config.cluster.resource.env.gpus_per_node == 2
    assert config.cluster.resource.model.gpus_per_node == 1
    assert config.cluster.checkpoint.resume_mode == "disable"

    cluster = TrainCluster(instantiate(config.cluster, _recursive_=False))
    cluster._build_resource_pool_plan()
    assert cluster.resource_pool_spec == {"env_gpu_pool": [2], "train_rollout_pool": [1]}


def test_flow_grpo_four_gpu_topology_and_hard_group_gate():
    config_dir = Path(__file__).resolve().parents[2] / "configs" / "flow_grpo"
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(config_name="fastwam_robodojo_4gpu")

    trainer = instantiate(config.trainer)
    actor = omega_conf_to_dataclass(
        config.cluster.actor_rollout_ref.actor,
        dataclass_type=FlowGRPOActorConfig,
    )
    adapter = FastWAMAdapterConfig(
        **OmegaConf.to_container(config.cluster.actor_rollout_ref.model.adapter, resolve=True)
    )

    assert isinstance(trainer, FlowGRPOTrainerConfig)
    assert isinstance(actor, FlowGRPOActorConfig)
    assert actor.value.enabled is False
    assert adapter.fpo.enabled is False
    assert adapter.flow_grpo.enabled is True
    assert adapter.flow_grpo.transition_batch_size == 3
    assert trainer.group_size == 8
    assert trainer.concurrent_candidate_groups == 9
    assert trainer.initial_candidate_groups_per_update == 27
    assert trainer.accepted_groups_before_optional_refill == 12
    assert trainer.max_candidate_groups_per_update == 36
    assert trainer.min_accepted_groups_for_partial_update == 12
    assert trainer.total_training_steps == 8
    assert trainer.save_freq == 1
    assert config.cluster.env.env_worker.num_envs == 8
    assert config.cluster.env.env_loop.pipeline_stage_num == 3
    assert config.cluster.resource.env.gpus_per_node == 3
    assert config.cluster.resource.model.gpus_per_node == 1
    assert config.cluster.checkpoint.resume_mode == "disable"
    assert config.cluster.checkpoint.max_actor_ckpt_to_keep == 8

    cluster = TrainCluster(instantiate(config.cluster, _recursive_=False))
    cluster._build_resource_pool_plan()
    assert cluster.resource_pool_spec == {"env_gpu_pool": [3], "train_rollout_pool": [1]}


def test_flow_grpo_rejects_single_pass_moving_reference_regression():
    config_dir = Path(__file__).resolve().parents[2] / "configs" / "flow_grpo"
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(
            config_name="fastwam_robodojo_3gpu",
            overrides=["cluster.actor_rollout_ref.actor.update_epochs=1"],
        )

    with pytest.raises(ValueError, match="update_epochs>=2"):
        omega_conf_to_dataclass(
            config.cluster.actor_rollout_ref.actor,
            dataclass_type=FlowGRPOActorConfig,
        )


def test_flow_grpo_four_gpu_training_optimized_topology():
    config_dir = Path(__file__).resolve().parents[2] / "configs" / "flow_grpo"
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(
            config_name="fastwam_robodojo_4gpu",
            overrides=[
                "cluster.resource.env.gpus_per_node=2",
                "cluster.resource.model.gpus_per_node=2",
                "trainer.concurrent_candidate_groups=6",
                "trainer.initial_candidate_groups_per_update=24",
                "trainer.experiment_name=fastwam-robodojo-flow-grpo-4gpu-2actor",
            ],
        )

    adapter = FastWAMAdapterConfig(
        **OmegaConf.to_container(config.cluster.actor_rollout_ref.model.adapter, resolve=True)
    )
    assert adapter.flow_grpo.transition_batch_size == 3
    assert config.cluster.resource.env.gpus_per_node == 2
    assert config.cluster.resource.model.gpus_per_node == 2
    assert config.trainer.concurrent_candidate_groups == 6
    assert config.trainer.initial_candidate_groups_per_update == 24
    assert config.trainer.accepted_groups_per_update == 27
    assert config.trainer.min_accepted_groups_for_partial_update == 12

    cluster = TrainCluster(instantiate(config.cluster, _recursive_=False))
    cluster._build_resource_pool_plan()
    assert cluster.resource_pool_spec == {"env_gpu_pool": [2], "train_rollout_pool": [2]}


def test_spooled_flow_grpo_config_uses_shared_group_contract_and_flow_actor(monkeypatch):
    monkeypatch.setenv("GRFPO_OUTPUT_ROOT", "/tmp/flow-output")
    monkeypatch.setenv("GRFPO_COLLECTION_ROOT", "/tmp/flow-collection")
    monkeypatch.setenv("FASTWAM_POLICY_ROOT", "/tmp/fastwam-policy")
    config_dir = Path(__file__).resolve().parents[2] / "src/verl_vla/workflows/config"
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(
            config_name="train/flow_grpo_spooled",
            overrides=[
                "collection.condition_manifest_sha256=" + "a" * 64,
                "collection.model_root=/tmp/model",
                "collection.checkpoint_sha256=" + "b" * 64,
            ],
        )

    trainer = instantiate(config.trainer)
    actor = omega_conf_to_dataclass(
        config.cluster.actor_rollout_ref.actor,
        dataclass_type=FlowGRPOActorConfig,
    )
    adapter = FastWAMAdapterConfig(
        **OmegaConf.to_container(config.cluster.actor_rollout_ref.model.adapter, resolve=True)
    )
    assert isinstance(trainer, FlowGRPOTrainerConfig)
    assert isinstance(actor, FlowGRPOActorConfig)
    assert config.collection.actor_objective == "flow_grpo"
    assert config.collection.group_reward_source == "binary_success"
    assert config.collection.accepted_group_reduction == "group_equal"
    assert adapter.fpo.enabled is False
    assert adapter.flow_grpo.enabled is True
    assert adapter.flow_grpo.noise_level == pytest.approx(0.01)
    assert adapter.flow_grpo.transition_batch_size == 3
