#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../.." && pwd)
source "$repo_root/scripts/lib/runtime_env.sh"

window_dir=${1:?usage: add_opportunistic_collectors.sh WINDOW COUNT [normal|preempt] [NODELIST]}
count=${2:?usage: add_opportunistic_collectors.sh WINDOW COUNT [normal|preempt] [NODELIST]}
partition=${3:-preempt}
nodelist=${4:-}
window_dir=$(realpath "$window_dir")
manifest="$window_dir/manifest.json"
test -s "$manifest"
if [[ -s "$window_dir/CLOSED.json" || -s "$window_dir/TRAINED.json" ]]; then
    echo "window_not_collecting=$window_dir" >&2
    exit 2
fi
if [[ ! "$count" =~ ^[1-9][0-9]*$ ]]; then
    echo "COUNT must be a positive integer, got $count" >&2
    exit 2
fi

python=$VVLA_PYTHON
case "$partition" in
    normal) collector_sbatch="$repo_root/scripts/grfpo/submit_fragmented_collector_normal.sbatch" ;;
    preempt) collector_sbatch="$repo_root/scripts/grfpo/submit_fragmented_collector_preempt.sbatch" ;;
    *) echo "partition must be normal or preempt, got $partition" >&2; exit 2 ;;
esac

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
print(":".join(record["task_names"]))
print(len(record["task_names"]) * len(record["suite_ids"]))
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
collector_count=${fields[12]}
group_reward_source=${fields[13]}
informative_group_criterion=${fields[14]}
accepted_group_reduction=${fields[15]}
policy_seed_namespace=${fields[16]}
if (( count > collector_count )); then
    echo "COUNT=$count exceeds condition count=$collector_count" >&2
    exit 2
fi

collection_root=$(dirname "$(dirname "$window_dir")")
output_root=$(dirname "$collection_root")
log_root=${GRFPO_LOG_ROOT:-$VVLA_LOG_ROOT}
mkdir -p "$log_root"
common_export="ALL,GRFPO_COLLECTION_ROOT=$collection_root,GRFPO_RUN_ID=$run_id,GRFPO_INTENDED_UPDATE=$intended_update,GRFPO_THETA_OLD=$theta_old,GRFPO_TARGET_GROUPS=$target_groups,GRFPO_MIN_GROUPS=$minimum_groups,GRFPO_MAX_CANDIDATES=$max_candidates,GRFPO_COMPLETE_ROUND_AFTER_TARGET=$complete_round_after_target,GRFPO_DRAIN_CANDIDATE_BUDGET=$drain_candidate_budget,GRFPO_MODEL_ROOT=$model_root,GRFPO_MODEL_SHA256=$model_sha256,GRFPO_ACTOR_CHECKPOINT=$actor_checkpoint,GRFPO_OUTPUT_ROOT=$output_root,GRFPO_LOG_ROOT=$log_root,GRFPO_POLICY_MICROBATCH=${GRFPO_POLICY_MICROBATCH:-4},GRFPO_COLLECTOR_COUNT=$collector_count,GRFPO_COLLECTOR_CONCURRENCY=$collector_count,GRFPO_SHARED_CANDIDATE_QUEUE=1,GRFPO_SHARED_QUEUE_STEAL_ANY=1,GRFPO_INCLUDED_TASK_NAMES=$included_task_names,GRFPO_EXCLUDED_TASK_NAMES=,GRFPO_GROUP_REWARD_SOURCE=$group_reward_source,GRFPO_INFORMATIVE_GROUP_CRITERION=$informative_group_criterion,GRFPO_ACCEPTED_GROUP_REDUCTION=$accepted_group_reduction,GRFPO_POLICY_SEED_NAMESPACE=$policy_seed_namespace"

vvla_sbatch_site_args "$partition" "$log_root/fragmented_collect_%A_%a.out"
submit_args=(--parsable "${VVLA_SBATCH_SITE_ARGS[@]}" --array="0-$((count - 1))%$count" --export="$common_export")
if [[ -n "$nodelist" ]]; then
    submit_args+=(--nodelist="$nodelist")
fi

exec 9>"$window_dir/.opportunistic-submit.lock"
flock 9
job=$(sbatch "${submit_args[@]}" "$collector_sbatch")
printf '%s\t%s\t%s\t%s\t%s\n' \
    "$job" "0-$((count - 1))" "$partition" "${nodelist:-scheduler}" "$(date --iso-8601=seconds)" \
    >> "$window_dir/OPPORTUNISTIC_COLLECTOR_JOBS.tsv"
printf 'opportunistic_collector_job=%s count=%s partition=%s nodelist=%s theta_old=%s window=%s\n' \
    "$job" "$count" "$partition" "${nodelist:-scheduler}" "$theta_old" "$window_dir"
