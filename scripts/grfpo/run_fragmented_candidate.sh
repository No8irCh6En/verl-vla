#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../.." && pwd)
source "$repo_root/scripts/lib/runtime_env.sh"
vvla_require_fastwam_runtime
python=$VVLA_PYTHON
job_id=${SLURM_JOB_ID:?Must run inside Slurm.}
collector_slot=${SLURM_ARRAY_TASK_ID:?Must run as a collector array task.}
task_uid="${job_id}_${collector_slot}"
runtime_tmp="/tmp/vvla-grfpo-collector-${task_uid}"
cache_tmp="$runtime_tmp/cache"
mkdir -p "$runtime_tmp" "$cache_tmp" "$GRFPO_LOG_ROOT"

cleanup_runtime() {
    case "$runtime_tmp" in /tmp/vvla-grfpo-collector-[0-9]*_[0-9]*) rm -rf -- "$runtime_tmp" ;; esac
}
trap cleanup_runtime EXIT

cd "$repo_root"
env TMPDIR="$runtime_tmp" \
    XDG_CACHE_HOME="$cache_tmp" \
    PYTHONNOUSERSITE=1 \
    PYTHONPATH="$(vvla_runtime_pythonpath)" \
    HF_HOME="$HF_HOME" \
    DIFFSYNTH_MODEL_BASE_PATH="$FASTWAM_ROOT/checkpoints" \
    DIFFSYNTH_SKIP_DOWNLOAD=true \
    "$python" -m verl_vla.entrypoints.grfpo_collect \
    "collection_root=$GRFPO_COLLECTION_ROOT" \
    "run_id=$GRFPO_RUN_ID" \
    "intended_update=$GRFPO_INTENDED_UPDATE" \
    "rollout_policy_version=$GRFPO_THETA_OLD" \
    "policy_seed_namespace=${GRFPO_POLICY_SEED_NAMESPACE:-}" \
    "collector_slot=$collector_slot" \
    "collector_count=$GRFPO_COLLECTOR_COUNT" \
    "target_accepted_groups=$GRFPO_TARGET_GROUPS" \
    "minimum_accepted_groups=$GRFPO_MIN_GROUPS" \
    "max_candidate_groups=$GRFPO_MAX_CANDIDATES" \
    "drain_candidate_budget=${GRFPO_DRAIN_CANDIDATE_BUDGET:-0}" \
    "model_root=$GRFPO_MODEL_ROOT" \
    "checkpoint_sha256=$GRFPO_MODEL_SHA256" \
    "verl_actor_checkpoint=$GRFPO_ACTOR_CHECKPOINT" \
    "policy_root=$FASTWAM_POLICY_ROOT" \
    "robodojo_root=$ROBODOJO_ROOT" \
    "condition_manifest=${VVLA_CONDITION_MANIFEST:?VVLA_CONDITION_MANIFEST is required}" \
    "group_reward_source=${GRFPO_GROUP_REWARD_SOURCE:-binary_success}" \
    "informative_group_criterion=${GRFPO_INFORMATIVE_GROUP_CRITERION:-full_success_mixed}" \
    "accepted_group_reduction=${GRFPO_ACCEPTED_GROUP_REDUCTION:-group_equal}" \
    "actor_objective=${GRFPO_ACTOR_OBJECTIVE:-fpo}" \
    "flow_grpo_noise_level=${GRFPO_FLOW_GRPO_NOISE_LEVEL:-0.01}" \
    "flow_grpo_transition_batch_size=${GRFPO_FLOW_GRPO_TRANSITION_BATCH_SIZE:-3}" \
    "simulators_per_gpu=${GRFPO_SIMULATORS_PER_GPU:-1}" \
    "shared_simulator_startup_stagger_seconds=${GRFPO_SHARED_SIMULATOR_STARTUP_STAGGER_SECONDS:-8}" \
    "shared_inference_coalesce_ms=${GRFPO_SHARED_INFERENCE_COALESCE_MS:-5}" \
    "policy_microbatch_size=$GRFPO_POLICY_MICROBATCH"
