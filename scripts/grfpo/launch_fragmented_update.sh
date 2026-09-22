#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../.." && pwd)
source "$repo_root/scripts/lib/runtime_env.sh"
vvla_require_fastwam_runtime
selection=${MULTITASK_SFT_SELECTION_JSON:?MULTITASK_SFT_SELECTION_JSON must be exported by the launcher}
python=$VVLA_PYTHON
theta_old=${1:-0}
intended_update=$((theta_old + 1))
target_groups=${GRFPO_TARGET_GROUPS:-32}
minimum_groups=${GRFPO_MIN_GROUPS:-32}
# Fixed candidate-budget collection is the safe default. Target-based
# complete-round stopping must be requested explicitly by the unified launcher.
complete_round_after_target=${GRFPO_COMPLETE_ROUND_AFTER_TARGET:-0}
drain_candidate_budget=${GRFPO_DRAIN_CANDIDATE_BUDGET:-1}
# Current mainline excludes classify_objects: four tasks x three suites form
# twelve balanced condition lanes, with eight candidates per lane.
max_candidates=${GRFPO_MAX_CANDIDATES:-96}
collector_count=${GRFPO_COLLECTOR_COUNT:-12}
collector_concurrency=${GRFPO_COLLECTOR_CONCURRENCY:-8}
shared_candidate_queue=${GRFPO_SHARED_CANDIDATE_QUEUE:-1}
excluded_task_names=${GRFPO_EXCLUDED_TASK_NAMES:-classify_objects}
group_reward_source=${GRFPO_GROUP_REWARD_SOURCE:-binary_success}
informative_group_criterion=${GRFPO_INFORMATIVE_GROUP_CRITERION:-full_success_mixed}
accepted_group_reduction=${GRFPO_ACCEPTED_GROUP_REDUCTION:-group_equal}
policy_seed_namespace=${GRFPO_POLICY_SEED_NAMESPACE:-}
actor_objective=${GRFPO_ACTOR_OBJECTIVE:-fpo}
flow_grpo_noise_level=${GRFPO_FLOW_GRPO_NOISE_LEVEL:-0.01}
flow_grpo_transition_batch_size=${GRFPO_FLOW_GRPO_TRANSITION_BATCH_SIZE:-3}
output_namespace=${VVLA_GROUP_RL_OUTPUT_NAMESPACE:-grfpo}
opportunistic_preempt_collectors=${GRFPO_OPPORTUNISTIC_PREEMPT_COLLECTORS:-4}
if [[ "$actor_objective" != fpo && "$actor_objective" != flow_grpo ]]; then
    echo "GRFPO_ACTOR_OBJECTIVE must be fpo or flow_grpo" >&2
    exit 2
fi
if [[ "$output_namespace" != grfpo && "$output_namespace" != flow_grpo ]]; then
    echo "VVLA_GROUP_RL_OUTPUT_NAMESPACE must be grfpo or flow_grpo" >&2
    exit 2
fi
if [[ "$group_reward_source" != binary_success && "$group_reward_source" != process_score ]]; then
    echo "GRFPO_GROUP_REWARD_SOURCE must be binary_success or process_score" >&2
    exit 2
fi
if [[ "$informative_group_criterion" != binary_success \
      && "$informative_group_criterion" != full_success_mixed \
      && "$informative_group_criterion" != reward_variance ]]; then
    echo "Unsupported GRFPO_INFORMATIVE_GROUP_CRITERION=$informative_group_criterion" >&2
    exit 2
fi
if [[ "$informative_group_criterion" == reward_variance && "$group_reward_source" != process_score ]]; then
    echo "reward_variance requires process_score rewards" >&2
    exit 2
fi
if [[ "$accepted_group_reduction" != group_equal && "$accepted_group_reduction" != task_equal ]]; then
    echo "GRFPO_ACCEPTED_GROUP_REDUCTION must be group_equal or task_equal" >&2
    exit 2
fi
if (( collector_concurrency > collector_count )); then
    collector_concurrency=$collector_count
fi
if [[ ! "$opportunistic_preempt_collectors" =~ ^[0-9]+$ ]] \
    || (( opportunistic_preempt_collectors > collector_count )); then
    echo "GRFPO_OPPORTUNISTIC_PREEMPT_COLLECTORS must be in [0,$collector_count], got $opportunistic_preempt_collectors" >&2
    exit 2
fi

readarray -t selected < <("$python" - "$selection" <<'PY'
import json
import sys
record = json.load(open(sys.argv[1], encoding="utf-8"))
print(record["eval_checkpoint_root"])
print(record["checkpoint_sha256"])
print(record["selected"]["step"])
PY
)
run_id=${MULTITASK_GRFPO_RUN_ID:-"multitask5_sft_c${selected[2]}_grfpo_pp_allsuite_v2_m32_v1"}
output_root="$VVLA_OUTPUT_ROOT/$output_namespace/$run_id"
collection_root="$output_root/candidate_spool"
log_root=$VVLA_LOG_ROOT
actor_checkpoint=""
if (( theta_old > 0 )); then
    actor_checkpoint="$output_root/checkpoints/global_step_$theta_old"
    test -d "$actor_checkpoint/actor"
fi
mkdir -p "$output_root" "$collection_root" "$log_root"

# A long-lived chain cannot have its inherited environment changed after
# submission.  This marker permits a validated shared-model rollout topology
# to take effect at the next, not-yet-submitted theta_old window without
# mutating or interrupting the currently open on-policy window.
rollout_topology_file="$output_root/ROLLOUT_TOPOLOGY.json"
marker_simulators_per_gpu=""
if [[ -s "$rollout_topology_file" ]]; then
    marker_simulators_per_gpu=$("$python" - "$rollout_topology_file" <<'PY'
import json
import sys

record = json.load(open(sys.argv[1], encoding="utf-8"))
value = int(record["simulators_per_gpu"])
if value not in (1, 2, 3):
    raise ValueError(f"simulators_per_gpu must be 1, 2, or 3, got {value}")
print(value)
PY
    )
fi
simulators_per_gpu=${GRFPO_SIMULATORS_PER_GPU:-${marker_simulators_per_gpu:-1}}
if [[ "$simulators_per_gpu" != 1 && "$simulators_per_gpu" != 2 && "$simulators_per_gpu" != 3 ]]; then
    echo "GRFPO_SIMULATORS_PER_GPU must be 1, 2, or 3, got $simulators_per_gpu" >&2
    exit 2
fi
orchestration_root="$output_root/orchestration"
mkdir -p "$orchestration_root"
submission_record="$orchestration_root/$(printf 'update_%04d_theta_%04d.submitted.json' "$intended_update" "$theta_old")"
exec 9>"$orchestration_root/$(printf '.update_%04d_theta_%04d.launch.lock' "$intended_update" "$theta_old")"
if ! flock -n 9; then
    echo "Another launcher owns theta_old=$theta_old update=$intended_update" >&2
    exit 2
fi
if [[ -s "$submission_record" ]]; then
    cat "$submission_record"
    exit 0
fi

submitted_jobs=()
launch_committed=0
cleanup_partial_launch() {
    if (( launch_committed == 0 )); then
        for job in "${submitted_jobs[@]}"; do
            [[ -n "$job" ]] && scancel "$job" 2>/dev/null || true
        done
    fi
}
trap cleanup_partial_launch EXIT

submit_with_retry() {
    local attempt output
    for attempt in 1 2 3 4 5; do
        # Only stdout is the job id. Slurm may print transient controller
        # diagnostics on stderr before the same invocation succeeds.
        if output=$(sbatch --parsable "$@"); then
            printf '%s\n' "$output"
            return 0
        fi
        echo "launch_sbatch_transient_failure attempt=$attempt/5" >&2
        sleep $((attempt * 10))
    done
    return 1
}

common_export="ALL,GRFPO_COLLECTION_ROOT=$collection_root,GRFPO_RUN_ID=$run_id,GRFPO_INTENDED_UPDATE=$intended_update,GRFPO_THETA_OLD=$theta_old,GRFPO_TARGET_GROUPS=$target_groups,GRFPO_MIN_GROUPS=$minimum_groups,GRFPO_MAX_CANDIDATES=$max_candidates,GRFPO_COMPLETE_ROUND_AFTER_TARGET=$complete_round_after_target,GRFPO_DRAIN_CANDIDATE_BUDGET=$drain_candidate_budget,GRFPO_MODEL_ROOT=${selected[0]},GRFPO_MODEL_SHA256=${selected[1]},GRFPO_ACTOR_CHECKPOINT=$actor_checkpoint,GRFPO_OUTPUT_ROOT=$output_root,GRFPO_LOG_ROOT=$log_root,GRFPO_POLICY_MICROBATCH=4,GRFPO_COLLECTOR_COUNT=$collector_count,GRFPO_COLLECTOR_CONCURRENCY=$collector_concurrency,GRFPO_SHARED_CANDIDATE_QUEUE=$shared_candidate_queue,GRFPO_SIMULATORS_PER_GPU=$simulators_per_gpu,GRFPO_EXCLUDED_TASK_NAMES=$excluded_task_names,GRFPO_GROUP_REWARD_SOURCE=$group_reward_source,GRFPO_INFORMATIVE_GROUP_CRITERION=$informative_group_criterion,GRFPO_ACCEPTED_GROUP_REDUCTION=$accepted_group_reduction,GRFPO_POLICY_SEED_NAMESPACE=$policy_seed_namespace,GRFPO_ACTOR_OBJECTIVE=$actor_objective,GRFPO_FLOW_GRPO_NOISE_LEVEL=$flow_grpo_noise_level,GRFPO_FLOW_GRPO_TRANSITION_BATCH_SIZE=$flow_grpo_transition_batch_size,VVLA_GROUP_RL_OUTPUT_NAMESPACE=$output_namespace"
vvla_sbatch_site_args normal "$log_root/fragmented_collect_%A_%a.out"
normal_site_args=("${VVLA_SBATCH_SITE_ARGS[@]}")
vvla_sbatch_site_args preempt "$log_root/fragmented_collect_%A_%a.out"
preempt_site_args=("${VVLA_SBATCH_SITE_ARGS[@]}")
vvla_sbatch_site_args cpu "$log_root/fragmented_watch_%j.out"
cpu_site_args=("${VVLA_SBATCH_SITE_ARGS[@]}")
normal_lane_count=$collector_concurrency
collector_job=$(submit_with_retry \
    "${normal_site_args[@]}" \
    --array="0-$((normal_lane_count - 1))%$normal_lane_count" \
    --export="$common_export" \
    "$repo_root/scripts/grfpo/submit_fragmented_collector_normal.sbatch")
submitted_jobs+=("$collector_job")
preempt_collector_job=""
if (( normal_lane_count < collector_count )); then
    preempt_lane_count=$((collector_count - normal_lane_count))
    preempt_collector_job=$(submit_with_retry \
        "${preempt_site_args[@]}" \
        --array="$normal_lane_count-$((collector_count - 1))%$preempt_lane_count" \
        --export="$common_export" \
        "$repo_root/scripts/grfpo/submit_fragmented_collector_preempt.sbatch")
    submitted_jobs+=("$preempt_collector_job")
fi
opportunistic_collector_job=""
if [[ "$shared_candidate_queue" == 1 ]] && (( opportunistic_preempt_collectors > 0 )); then
    opportunistic_export="$common_export,GRFPO_SHARED_QUEUE_STEAL_ANY=1"
    if opportunistic_collector_job=$(submit_with_retry \
        "${preempt_site_args[@]}" \
        --array="0-$((opportunistic_preempt_collectors - 1))%$opportunistic_preempt_collectors" \
        --export="$opportunistic_export" \
        "$repo_root/scripts/grfpo/submit_fragmented_collector_preempt.sbatch"); then
        submitted_jobs+=("$opportunistic_collector_job")
    else
        # Elastic capacity is best-effort. The twelve condition-pinned lanes
        # remain a complete valid topology when the preempt submit quota is
        # temporarily exhausted.
        opportunistic_collector_job=""
        echo "warning: opportunistic collectors unavailable; continuing with base lanes" >&2
    fi
fi

window_dir="$collection_root/$run_id/$(printf 'update_%04d_theta_%04d' "$intended_update" "$theta_old")"
mkdir -p "$window_dir"
collector_job_file="$window_dir/COLLECTOR_JOBS.tsv"
temporary="$collector_job_file.tmp-$$"
: > "$temporary"
for ((lane = 0; lane < normal_lane_count; lane++)); do
    printf '%s\t%s\tnormal\n' "$lane" "$collector_job" >> "$temporary"
done
for ((lane = normal_lane_count; lane < collector_count; lane++)); do
    printf '%s\t%s\tpreempt\n' "$lane" "$preempt_collector_job" >> "$temporary"
done
mv "$temporary" "$collector_job_file"
if [[ -n "$opportunistic_collector_job" ]]; then
    printf '%s\t%s\t%s\t%s\t%s\n' \
        "$opportunistic_collector_job" "0-$((opportunistic_preempt_collectors - 1))" \
        preempt scheduler "$(date --iso-8601=seconds)" \
        >> "$window_dir/OPPORTUNISTIC_COLLECTOR_JOBS.tsv"
fi
watch_export="$common_export,GRFPO_COLLECT_JOB_ID=$collector_job,GRFPO_INITIAL_COLLECTOR_PARTITION=normal"
watch_job=$(submit_with_retry "${cpu_site_args[@]}" --export="$watch_export" \
    "$repo_root/scripts/grfpo/monitor_fragmented_window.sbatch")
submitted_jobs+=("$watch_job")
opportunistic_watch_job=""
if [[ -n "$opportunistic_collector_job" ]]; then
    if opportunistic_watch_job=$(sbatch --parsable \
        "${cpu_site_args[@]}" \
        --export="ALL,GRFPO_WINDOW_DIR=$window_dir,GRFPO_LOG_ROOT=$log_root" \
        "$repo_root/scripts/grfpo/monitor_opportunistic_collectors.sbatch"); then
        submitted_jobs+=("$opportunistic_watch_job")
        printf '%s\n' "$opportunistic_watch_job" > "$window_dir/OPPORTUNISTIC_WATCH_JOB_ID"
    else
        # The main watcher and base collector lanes remain sufficient for a
        # correct update.  This sidecar only shortens recovery of best-effort
        # opportunistic capacity.
        echo "warning: opportunistic collector monitor unavailable" >&2
        opportunistic_watch_job=""
    fi
fi
"$python" - "$submission_record" "$collector_job" "$preempt_collector_job" "$opportunistic_collector_job" "$watch_job" "$opportunistic_watch_job" "$theta_old" "$intended_update" "$shared_candidate_queue" "$simulators_per_gpu" <<'PY'
import json
import os
import sys
import time
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "collector_job": sys.argv[2],
    "normal_collector_job": sys.argv[2],
    "preempt_collector_job": sys.argv[3] or None,
    "opportunistic_collector_job": sys.argv[4] or None,
    "watcher_job": sys.argv[5],
    "opportunistic_watcher_job": sys.argv[6] or None,
    "theta_old": int(sys.argv[7]),
    "intended_update": int(sys.argv[8]),
    "shared_candidate_queue": bool(int(sys.argv[9])),
    "simulators_per_gpu": int(sys.argv[10]),
    "initial_collector_partitions": ["normal", "preempt"],
    "submitted_at_unix_s": time.time(),
}
temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, path)
PY
launch_committed=1
printf 'normal_collector_job=%s preempt_collector_job=%s opportunistic_collector_job=%s watcher_job=%s opportunistic_watcher_job=%s theta_old=%s intended_update=%s lanes=%s shared_queue=%s simulators_per_gpu=%s\n' \
    "$collector_job" "$preempt_collector_job" "$opportunistic_collector_job" "$watch_job" "$opportunistic_watch_job" "$theta_old" "$intended_update" "$collector_count" "$shared_candidate_queue" "$simulators_per_gpu"
