#!/usr/bin/env bash
set -euo pipefail

window_dir=${1:?usage: resume_fragmented_window.sh COLLECTION_WINDOW}
window_dir=$(realpath "$window_dir")
manifest=$window_dir/manifest.json
test -s "$manifest"
if [[ -s "$window_dir/CLOSED.json" || -s "$window_dir/TRAINED.json" ]]; then
    echo "window_not_collecting=$window_dir"
    exit 0
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../.." && pwd)
source "$repo_root/scripts/lib/runtime_env.sh"
python=$VVLA_PYTHON
collector_count=${GRFPO_COLLECTOR_COUNT:-}
collector_concurrency=${GRFPO_COLLECTOR_CONCURRENCY:-8}
shared_candidate_queue=${GRFPO_SHARED_CANDIDATE_QUEUE:-1}
shared_queue_steal_any=${GRFPO_SHARED_QUEUE_STEAL_ANY:-0}
excluded_task_names=${GRFPO_EXCLUDED_TASK_NAMES:-}

exec 9>"$window_dir/.resume-submit.lock"
flock 9

# Do not create two live jobs for a lane. Completed candidates are durable,
# but concurrent duplicates would still waste a GPU until atomic commit.
collector_job_file=$window_dir/COLLECTOR_JOBS.tsv
if [[ -s "$collector_job_file" ]]; then
    while IFS=$'\t' read -r lane job _partition; do
        [[ "$lane" =~ ^[0-9]+$ && -n "$job" ]] || continue
        # Slurm controller queries have occasionally stalled for minutes on
        # this cluster.  A resume must not hold .resume-submit.lock forever;
        # an inconclusive query falls through to the normal atomic-commit-safe
        # resubmission path.
        if timeout 15s squeue -h -j "${job}_${lane}" 2>/dev/null | grep -q .; then
            echo "collector_already_active lane=$lane job=${job}_${lane}"
            exit 0
        fi
    done < "$collector_job_file"
fi

submitted_jobs=()
resume_committed=0
cleanup_partial_resume() {
    if (( resume_committed == 0 )); then
        for job in "${submitted_jobs[@]}"; do
            [[ -n "$job" ]] && scancel "$job" 2>/dev/null || true
        done
    fi
}
trap cleanup_partial_resume EXIT

submit_with_retry() {
    local attempt output
    for attempt in 1 2 3 4 5; do
        if output=$(sbatch --parsable "$@"); then
            printf '%s\n' "$output"
            return 0
        fi
        echo "resume_sbatch_transient_failure attempt=$attempt/5" >&2
        sleep $((attempt * 10))
    done
    return 1
}

readarray -t fields < <("$python" - "$manifest" <<'PY'
import json
import sys

record = json.load(open(sys.argv[1], encoding="utf-8"))
policy = record["policy"]
print(record["run_id"])
print(record["intended_update"])
print(policy["rollout_policy_version"])
print(record["target_accepted_groups"])
print(record["minimum_accepted_groups"])
print(record["max_candidate_groups"])
print(int(bool(record.get("complete_round_after_target", False))))
print(int(bool(record.get("drain_candidate_budget", False))))
print(policy["immutable_base_path"])
print(policy["immutable_base_sha256"])
print(policy.get("trainer_checkpoint_path") or "")
print(":".join(record.get("task_names", [])))
print(len(record.get("task_names", [])) * len(record.get("suite_ids", [])))
print(record.get("group_reward_source", "binary_success"))
print(record.get("informative_group_criterion", "full_success_mixed"))
print(record.get("accepted_group_reduction", "group_equal"))
print(record.get("policy_seed_namespace", ""))
PY
)

run_id=${fields[0]}
intended_update=${fields[1]}
theta_old=${fields[2]}
target_groups=${fields[3]}
minimum_groups=${fields[4]}
max_candidates=${fields[5]}
complete_round_after_target=${fields[6]}
drain_candidate_budget=${fields[7]}
model_root=${fields[8]}
model_sha256=${fields[9]}
actor_checkpoint=${fields[10]}
included_task_names=${fields[11]}
manifest_collector_count=${fields[12]}
group_reward_source=${fields[13]}
informative_group_criterion=${fields[14]}
accepted_group_reduction=${fields[15]}
policy_seed_namespace=${fields[16]}
collector_count=${collector_count:-$manifest_collector_count}
if (( collector_count <= 0 )); then
    echo "Cannot derive collector_count from manifest" >&2
    exit 2
fi
if (( collector_concurrency > collector_count )); then
    collector_concurrency=$collector_count
fi
collection_root=$(dirname "$(dirname "$window_dir")")
output_root=$(dirname "$collection_root")
log_root=${GRFPO_LOG_ROOT:-$VVLA_LOG_ROOT}
mkdir -p "$log_root"

common_export="ALL,GRFPO_COLLECTION_ROOT=$collection_root,GRFPO_RUN_ID=$run_id,GRFPO_INTENDED_UPDATE=$intended_update,GRFPO_THETA_OLD=$theta_old,GRFPO_TARGET_GROUPS=$target_groups,GRFPO_MIN_GROUPS=$minimum_groups,GRFPO_MAX_CANDIDATES=$max_candidates,GRFPO_COMPLETE_ROUND_AFTER_TARGET=$complete_round_after_target,GRFPO_DRAIN_CANDIDATE_BUDGET=$drain_candidate_budget,GRFPO_MODEL_ROOT=$model_root,GRFPO_MODEL_SHA256=$model_sha256,GRFPO_ACTOR_CHECKPOINT=$actor_checkpoint,GRFPO_OUTPUT_ROOT=$output_root,GRFPO_LOG_ROOT=$log_root,GRFPO_POLICY_MICROBATCH=4,GRFPO_COLLECTOR_COUNT=$collector_count,GRFPO_COLLECTOR_CONCURRENCY=$collector_concurrency,GRFPO_SHARED_CANDIDATE_QUEUE=$shared_candidate_queue,GRFPO_SHARED_QUEUE_STEAL_ANY=$shared_queue_steal_any,GRFPO_INCLUDED_TASK_NAMES=$included_task_names,GRFPO_EXCLUDED_TASK_NAMES=$excluded_task_names,GRFPO_GROUP_REWARD_SOURCE=$group_reward_source,GRFPO_INFORMATIVE_GROUP_CRITERION=$informative_group_criterion,GRFPO_ACCEPTED_GROUP_REDUCTION=$accepted_group_reduction,GRFPO_POLICY_SEED_NAMESPACE=$policy_seed_namespace"

# Keep the same split topology used by the primary launcher.  Submitting all
# fifteen lanes as one normal array can exceed QOSMaxSubmitJobPerUserLimit even
# when only eight array elements are allowed to run concurrently.
normal_lane_count=$collector_concurrency
vvla_sbatch_site_args normal "$log_root/fragmented_collect_%A_%a.out"
normal_site_args=("${VVLA_SBATCH_SITE_ARGS[@]}")
vvla_sbatch_site_args preempt "$log_root/fragmented_collect_%A_%a.out"
preempt_site_args=("${VVLA_SBATCH_SITE_ARGS[@]}")
vvla_sbatch_site_args cpu "$log_root/fragmented_watch_%j.out"
cpu_site_args=("${VVLA_SBATCH_SITE_ARGS[@]}")
collector_job=$(submit_with_retry \
    "${normal_site_args[@]}" \
    --array="0-$((normal_lane_count - 1))%$normal_lane_count" \
    --export="$common_export" \
    "$repo_root/scripts/grfpo/submit_fragmented_collector_normal.sbatch")
submitted_jobs+=("$collector_job")
preempt_collector_job=""
if (( normal_lane_count < collector_count )); then
    preempt_lane_count=$((collector_count - normal_lane_count))
    if ! preempt_collector_job=$(submit_with_retry \
        "${preempt_site_args[@]}" \
        --array="$normal_lane_count-$((collector_count - 1))%$preempt_lane_count" \
        --export="$common_export" \
        "$repo_root/scripts/grfpo/submit_fragmented_collector_preempt.sbatch"); then
        exit 3
    fi
    submitted_jobs+=("$preempt_collector_job")
fi

temporary=$collector_job_file.tmp-$$
: > "$temporary"
for ((lane = 0; lane < normal_lane_count; lane++)); do
    printf '%s\t%s\tnormal\n' "$lane" "$collector_job" >> "$temporary"
done
for ((lane = normal_lane_count; lane < collector_count; lane++)); do
    printf '%s\t%s\tpreempt\n' "$lane" "$preempt_collector_job" >> "$temporary"
done
mv "$temporary" "$collector_job_file"

watch_export="$common_export,GRFPO_COLLECT_JOB_ID=$collector_job,GRFPO_INITIAL_COLLECTOR_PARTITION=normal"
watch_job=$(submit_with_retry "${cpu_site_args[@]}" --export="$watch_export" \
    "$repo_root/scripts/grfpo/monitor_fragmented_window.sbatch")
submitted_jobs+=("$watch_job")

"$python" - "$window_dir/RESUME_SUBMISSIONS.jsonl" "$collector_job" "$preempt_collector_job" "$watch_job" "$shared_candidate_queue" "$shared_queue_steal_any" <<'PY'
import json
import sys
import time

record = {
    "collector_job": sys.argv[2],
    "normal_collector_job": sys.argv[2],
    "preempt_collector_job": sys.argv[3] or None,
    "watcher_job": sys.argv[4],
    "initial_collector_partitions": ["normal", "preempt"],
    "shared_candidate_queue": bool(int(sys.argv[5])),
    "shared_queue_steal_any": bool(int(sys.argv[6])),
    "submitted_at_unix_s": time.time(),
}
with open(sys.argv[1], "a", encoding="utf-8") as stream:
    stream.write(json.dumps(record, sort_keys=True) + "\n")
PY

resume_committed=1

echo "resumed_window=$window_dir normal_collector_job=$collector_job preempt_collector_job=$preempt_collector_job watcher_job=$watch_job"
