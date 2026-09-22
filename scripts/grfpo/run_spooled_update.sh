#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../.." && pwd)
source "$repo_root/scripts/lib/runtime_env.sh"
vvla_require_fastwam_runtime
python=$VVLA_PYTHON
job_id=${SLURM_JOB_ID:?Must run inside Slurm.}
# Large drained windows contain enough heterogeneous, long trajectories that
# allocator growth/fragmentation can reach the 80-GB ceiling late in epoch 1
# even with a physical batch of two.  Stream one chunk at a time by default;
# the logical group/trajectory-balanced batch and optimizer semantics remain
# unchanged.
policy_microbatch=${GRFPO_POLICY_MICROBATCH:-1}
train_gpus_per_node=${GRFPO_TRAIN_GPUS_PER_NODE:-1}
# ``-1`` shards parameters across every actor rank (the historical default).
# Setting this to ``1`` with two actor ranks creates a 2 x 1 HSDP mesh: two
# data-parallel replicas with no cross-rank parameter sharding.  This avoids a
# parameter all-gather/reshard for every physical microbatch and is useful for
# Fast-WAM's microbatch-one, full-logical-batch accumulation path.
fsdp_size=${GRFPO_FSDP_SIZE:--1}
checkpoint_load_smoke_only=${GRFPO_CHECKPOINT_LOAD_SMOKE_ONLY:-0}
checkpoint_portable_export_only=${GRFPO_CHECKPOINT_PORTABLE_EXPORT_ONLY:-0}
task_gradient_diagnostic_only=${GRFPO_TASK_GRADIENT_DIAGNOSTIC_ONLY:-0}
task_gradient_diagnostic_output=${GRFPO_TASK_GRADIENT_DIAGNOSTIC_OUTPUT:-null}
task_gradient_diagnostic_targets_json=${GRFPO_TASK_GRADIENT_DIAGNOSTIC_TARGETS_JSON:-null}
task_gradient_diagnostic_include_groups=${GRFPO_TASK_GRADIENT_DIAGNOSTIC_INCLUDE_GROUPS:-1}
task_names_colon=${GRFPO_TASK_NAMES_COLON:?GRFPO_TASK_NAMES_COLON must come from the window manifest}
suite_ids_colon=${GRFPO_SUITE_IDS_COLON:?GRFPO_SUITE_IDS_COLON must come from the window manifest}
condition_manifest_sha256=${GRFPO_CONDITION_MANIFEST_SHA256:?GRFPO_CONDITION_MANIFEST_SHA256 must come from the window manifest}
group_reward_source=${GRFPO_GROUP_REWARD_SOURCE:?GRFPO_GROUP_REWARD_SOURCE must come from the window manifest}
informative_group_criterion=${GRFPO_INFORMATIVE_GROUP_CRITERION:?GRFPO_INFORMATIVE_GROUP_CRITERION must come from the window manifest}
accepted_group_reduction=${GRFPO_ACCEPTED_GROUP_REDUCTION:?GRFPO_ACCEPTED_GROUP_REDUCTION must come from the window manifest}
policy_seed_namespace=${GRFPO_POLICY_SEED_NAMESPACE:-}
actor_lr=${GRFPO_ACTOR_LR:-1.0e-5}
kl_early_stop_mode=${GRFPO_KL_EARLY_STOP_MODE:-post_epoch}
target_kl=${GRFPO_TARGET_KL:-0.1}
fixed_mc_seed=${GRFPO_FIXED_MC_SEED:-}
fixed_shuffle_seed=${GRFPO_FIXED_SHUFFLE_SEED:-}
collector_count=${GRFPO_COLLECTOR_COUNT:?GRFPO_COLLECTOR_COUNT must come from the window manifest}
actor_objective=${GRFPO_ACTOR_OBJECTIVE:-fpo}
flow_grpo_noise_level=${GRFPO_FLOW_GRPO_NOISE_LEVEL:-0.01}
flow_grpo_transition_batch_size=${GRFPO_FLOW_GRPO_TRANSITION_BATCH_SIZE:-3}
case "$actor_objective" in
    fpo) train_entrypoint=verl_vla.entrypoints.train.grfpo_spooled ;;
    flow_grpo) train_entrypoint=verl_vla.entrypoints.train.flow_grpo_spooled ;;
    *) echo "Invalid GRFPO_ACTOR_OBJECTIVE: $actor_objective" >&2; exit 2 ;;
esac
if [[ ! "$task_names_colon" =~ ^[A-Za-z0-9_]+(:[A-Za-z0-9_]+)*$ ]]; then
    echo "Invalid task-name manifest transport: $task_names_colon" >&2
    exit 2
fi
if [[ ! "$suite_ids_colon" =~ ^[0-9]+(:[0-9]+)*$ ]]; then
    echo "Invalid suite-id manifest transport: $suite_ids_colon" >&2
    exit 2
fi
if [[ "$accepted_group_reduction" != group_equal && "$accepted_group_reduction" != task_equal ]]; then
    echo "Invalid accepted-group reduction: $accepted_group_reduction" >&2
    exit 2
fi
if [[ "$kl_early_stop_mode" != post_epoch && "$kl_early_stop_mode" != pre_optimizer ]]; then
    echo "Invalid GRFPO_KL_EARLY_STOP_MODE: $kl_early_stop_mode" >&2
    exit 2
fi
for seed_name in fixed_mc_seed fixed_shuffle_seed; do
    seed_value=${!seed_name}
    if [[ -n "$seed_value" && ! "$seed_value" =~ ^[0-9]+$ ]]; then
        echo "$seed_name must be empty or a non-negative integer, got $seed_value" >&2
        exit 2
    fi
done
task_names_hydra="[${task_names_colon//:/,}]"
suite_ids_hydra="[${suite_ids_colon//:/,}]"
if (( policy_microbatch <= 0 )); then
    echo "GRFPO_POLICY_MICROBATCH must be positive, got $policy_microbatch" >&2
    exit 2
fi
if (( train_gpus_per_node <= 0 )); then
    echo "GRFPO_TRAIN_GPUS_PER_NODE must be positive, got $train_gpus_per_node" >&2
    exit 2
fi
if (( fsdp_size != -1 )); then
    if (( fsdp_size <= 0 || fsdp_size > train_gpus_per_node || train_gpus_per_node % fsdp_size != 0 )); then
        echo "GRFPO_FSDP_SIZE must be -1 or a positive divisor of GRFPO_TRAIN_GPUS_PER_NODE; got fsdp_size=$fsdp_size gpus=$train_gpus_per_node" >&2
        exit 2
    fi
fi
runtime_tmp="/tmp/vvla-grfpo-spooled-train-${job_id}"
ray_tmp="/tmp/vvla-grfpo-spooled-ray-${job_id}"
mkdir -p "$runtime_tmp/cache" "$ray_tmp" "$GRFPO_OUTPUT_ROOT" "$GRFPO_LOG_ROOT"

cleanup_runtime() {
    case "$runtime_tmp" in /tmp/vvla-grfpo-spooled-train-[0-9]*) rm -rf -- "$runtime_tmp" ;; esac
    case "$ray_tmp" in /tmp/vvla-grfpo-spooled-ray-[0-9]*) rm -rf -- "$ray_tmp" ;; esac
}
trap cleanup_runtime EXIT

resume_args=(cluster.checkpoint.resume_mode=disable)
if [[ -n "$GRFPO_ACTOR_CHECKPOINT" ]]; then
    resume_args=(
        cluster.checkpoint.resume_mode=resume_path
        "cluster.checkpoint.resume_from_path=$GRFPO_ACTOR_CHECKPOINT"
    )
fi

fixed_seed_args=()
if [[ -n "$fixed_mc_seed" ]]; then
    fixed_seed_args+=("cluster.actor_rollout_ref.actor.fpo_mc_seed=$fixed_mc_seed")
fi
if [[ -n "$fixed_shuffle_seed" ]]; then
    fixed_seed_args+=("cluster.actor_rollout_ref.actor.fpo_shuffle_seed=$fixed_shuffle_seed")
fi

cd "$repo_root"
env TMPDIR="$runtime_tmp" \
    XDG_CACHE_HOME="$runtime_tmp/cache" \
    PYTHONNOUSERSITE=1 \
    PYTHONPATH="$(vvla_runtime_pythonpath)" \
    HF_HOME="$HF_HOME" \
    DIFFSYNTH_MODEL_BASE_PATH="$FASTWAM_ROOT/checkpoints" \
    DIFFSYNTH_SKIP_DOWNLOAD=true \
    "$python" -m "$train_entrypoint" \
    "output_dir=$GRFPO_OUTPUT_ROOT" \
    "collection.collection_root=$GRFPO_COLLECTION_ROOT" \
    "collection.run_id=$GRFPO_RUN_ID" \
    "collection.intended_update=$GRFPO_INTENDED_UPDATE" \
    "collection.rollout_policy_version=$GRFPO_THETA_OLD" \
    "collection.task_names=$task_names_hydra" \
    "collection.included_task_names=" \
    "collection.excluded_task_names=" \
    "collection.suite_ids=$suite_ids_hydra" \
    "collection.condition_manifest_sha256=$condition_manifest_sha256" \
    "collection.group_reward_source=$group_reward_source" \
    "collection.informative_group_criterion=$informative_group_criterion" \
    "collection.accepted_group_reduction=$accepted_group_reduction" \
    "collection.policy_seed_namespace=$policy_seed_namespace" \
    "collection.actor_objective=$actor_objective" \
    "collection.flow_grpo_noise_level=$flow_grpo_noise_level" \
    "collection.collector_count=$collector_count" \
    "collection.target_accepted_groups=$GRFPO_TARGET_GROUPS" \
    "collection.minimum_accepted_groups=$GRFPO_MIN_GROUPS" \
    "collection.max_candidate_groups=$GRFPO_MAX_CANDIDATES" \
    "collection.complete_round_after_target=${GRFPO_COMPLETE_ROUND_AFTER_TARGET:-0}" \
    "collection.drain_candidate_budget=${GRFPO_DRAIN_CANDIDATE_BUDGET:-0}" \
    "collection.model_root=$GRFPO_MODEL_ROOT" \
    "collection.checkpoint_sha256=$GRFPO_MODEL_SHA256" \
    "collection.verl_actor_checkpoint=$GRFPO_ACTOR_CHECKPOINT" \
    "cluster.actor_rollout_ref.model.adapter.policy_root=$FASTWAM_POLICY_ROOT" \
    "cluster.actor_rollout_ref.model.adapter.flow_grpo.noise_level=$flow_grpo_noise_level" \
    "cluster.actor_rollout_ref.model.adapter.flow_grpo.transition_batch_size=$flow_grpo_transition_batch_size" \
    "cluster.checkpoint.default_local_dir=$GRFPO_OUTPUT_ROOT/checkpoints" \
    "cluster.actor_rollout_ref.actor.micro_batch_size=$policy_microbatch" \
    "cluster.actor_rollout_ref.actor.fsdp_config.fsdp_size=$fsdp_size" \
    "cluster.actor_rollout_ref.actor.optim.lr=$actor_lr" \
    "cluster.actor_rollout_ref.actor.post_resume_lr_override=$actor_lr" \
    "cluster.actor_rollout_ref.actor.kl_early_stop_mode=$kl_early_stop_mode" \
    "cluster.actor_rollout_ref.actor.target_kl=$target_kl" \
    "trainer.group_reward_source=$group_reward_source" \
    "trainer.informative_group_criterion=$informative_group_criterion" \
    "trainer.accepted_group_reduction=$accepted_group_reduction" \
    "cluster.resource.model.gpus_per_node=$train_gpus_per_node" \
    "checkpoint_load_smoke_only=$checkpoint_load_smoke_only" \
    "checkpoint_portable_export_only=$checkpoint_portable_export_only" \
    "task_gradient_diagnostic_only=$task_gradient_diagnostic_only" \
    "task_gradient_diagnostic_output=$task_gradient_diagnostic_output" \
    "task_gradient_diagnostic_targets_json=$task_gradient_diagnostic_targets_json" \
    "task_gradient_diagnostic_include_groups=$task_gradient_diagnostic_include_groups" \
    "ray_kwargs.ray_init._temp_dir=$ray_tmp" \
    "${fixed_seed_args[@]}" \
    "${resume_args[@]}"
