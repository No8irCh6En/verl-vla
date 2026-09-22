import pytest

from verl_vla.trainer.grfpo.launch_config import GRFPOChainLaunchConfig


def test_fixed_budget_is_default_and_switches_are_orthogonal():
    config = GRFPOChainLaunchConfig(run_id="fixed96", process_score=True, task_equal=True)
    env = config.runtime_environment()
    assert env["GRFPO_MIN_GROUPS"] == "32"
    assert env["GRFPO_TARGET_GROUPS"] == "32"
    assert env["GRFPO_MAX_CANDIDATES"] == "96"
    assert env["GRFPO_COMPLETE_ROUND_AFTER_TARGET"] == "0"
    assert env["GRFPO_DRAIN_CANDIDATE_BUDGET"] == "1"
    assert env["GRFPO_GROUP_REWARD_SOURCE"] == "process_score"
    assert env["GRFPO_ACCEPTED_GROUP_REDUCTION"] == "task_equal"
    assert env["GRFPO_SIMULATORS_PER_GPU"] == "1"
    assert env["GRFPO_FSDP_SIZE"] == "-1"
    assert env["GRFPO_ACTOR_LR"] == "1e-05"
    assert env["GRFPO_KL_EARLY_STOP_MODE"] == "post_epoch"
    assert env["GRFPO_TARGET_KL"] == "0.1"
    assert env["GRFPO_ACTOR_OBJECTIVE"] == "fpo"
    assert env["VVLA_GROUP_RL_OUTPUT_NAMESPACE"] == "grfpo"


def test_early_stop_is_explicit_and_mutually_exclusive_with_drain():
    config = GRFPOChainLaunchConfig(run_id="early", early_stop=True)
    env = config.runtime_environment()
    assert env["GRFPO_COMPLETE_ROUND_AFTER_TARGET"] == "1"
    assert env["GRFPO_DRAIN_CANDIDATE_BUDGET"] == "0"


def test_invalid_collection_protocol_fails_before_sbatch():
    with pytest.raises(ValueError, match="at least minimum"):
        GRFPOChainLaunchConfig(
            run_id="invalid",
            minimum_accepted_groups=32,
            candidate_groups=16,
        )


def test_warm_start_lane_count_fails_before_sbatch():
    with pytest.raises(ValueError, match="one warm-start lane per"):
        GRFPOChainLaunchConfig(run_id="too-few-collectors", collector_count=2)


@pytest.mark.parametrize("simulator_count", [2, 3])
def test_shared_model_multiple_vec8_simulators_are_explicit_and_manifested(simulator_count):
    config = GRFPOChainLaunchConfig(run_id="shared", simulators_per_gpu=simulator_count)
    assert config.runtime_environment()["GRFPO_SIMULATORS_PER_GPU"] == str(simulator_count)
    assert config.manifest(selection={})["launch_config"]["simulators_per_gpu"] == simulator_count


def test_unvalidated_shared_simulator_count_is_rejected():
    with pytest.raises(ValueError, match="simulators_per_gpu"):
        GRFPOChainLaunchConfig(run_id="too-many-simulators", simulators_per_gpu=4)


def test_flow_grpo_reuses_collection_switches_and_changes_only_actor_contract():
    config = GRFPOChainLaunchConfig(
        run_id="flow-shared",
        algorithm="flow_grpo",
        process_score=True,
        task_equal=True,
        simulators_per_gpu=3,
        flow_grpo_noise_level=0.02,
        flow_grpo_transition_batch_size=4,
    )
    env = config.runtime_environment()
    assert env["GRFPO_ACTOR_OBJECTIVE"] == "flow_grpo"
    assert env["VVLA_GROUP_RL_OUTPUT_NAMESPACE"] == "flow_grpo"
    assert env["GRFPO_GROUP_REWARD_SOURCE"] == "process_score"
    assert env["GRFPO_ACCEPTED_GROUP_REDUCTION"] == "task_equal"
    assert env["GRFPO_SIMULATORS_PER_GPU"] == "3"
    assert env["GRFPO_FLOW_GRPO_NOISE_LEVEL"] == "0.02"
    assert env["GRFPO_FLOW_GRPO_TRANSITION_BATCH_SIZE"] == "4"
    manifest = config.manifest(selection={})
    assert manifest["algorithm"] == "flow_grpo"
    assert manifest["derived_protocol"]["actor_objective"] == "flow_grpo"


def test_unknown_group_rl_algorithm_is_rejected():
    with pytest.raises(ValueError, match="algorithm"):
        GRFPOChainLaunchConfig(run_id="unknown", algorithm="not-a-method")


def test_two_rank_hsdp_topology_is_explicit_and_manifested():
    config = GRFPOChainLaunchConfig(run_id="hsdp", train_gpus_per_node=2, train_fsdp_size=1)
    assert config.runtime_environment()["GRFPO_FSDP_SIZE"] == "1"
    assert config.manifest(selection={})["launch_config"]["train_fsdp_size"] == 1


def test_invalid_fsdp_size_is_rejected():
    with pytest.raises(ValueError, match="train_fsdp_size"):
        GRFPOChainLaunchConfig(run_id="bad-hsdp", train_gpus_per_node=2, train_fsdp_size=3)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("actor_lr", 0.0, "actor_lr"),
        ("target_kl", float("nan"), "target_kl"),
        ("kl_early_stop_mode", "unknown", "kl_early_stop_mode"),
    ],
)
def test_invalid_training_guard_fails_before_sbatch(field, value, message):
    kwargs = {field: value}
    with pytest.raises(ValueError, match=message):
        GRFPOChainLaunchConfig(run_id="invalid-training-guard", **kwargs)
