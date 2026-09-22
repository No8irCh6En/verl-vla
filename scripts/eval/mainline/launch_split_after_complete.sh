#!/usr/bin/env bash
set -euo pipefail

if (( $# < 3 || $# > 8 )); then
    echo "usage: $0 WAIT_FOR_EVAL_COMPLETE RUN_MANIFEST OUTPUT_ROOT [NORMAL_GPU_LANES] [PREEMPT_GPU_LANES] [TIME] [EVALUATORS_PER_GPU] [COALESCED_ITEMS_PER_EVALUATOR]" >&2
    exit 2
fi

wait_marker=$(realpath -m "$1")
run_manifest=$(realpath -m "$2")
output_root=$(realpath -m "$3")
normal_lanes=${4:-6}
preempt_lanes=${5:-6}
time_limit=${6:-06:00:00}
evaluators_per_gpu=${7:-1}
coalesced_items_per_evaluator=${8:-1}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
derived_repo_root=$(cd -- "$script_dir/../../.." && pwd)
source "$derived_repo_root/scripts/lib/runtime_env.sh"
bootstrap_python=$VVLA_PYTHON
runtime_record=$("$bootstrap_python" - "$run_manifest" <<'PY'
import json
import sys

runtime = json.load(open(sys.argv[1], encoding="utf-8"))["runtime"]
slurm = runtime["slurm"]
values = (
    runtime["repo_root"], runtime["python"], runtime.get("native_lib_dir") or "",
    slurm.get("module") or "", slurm.get("exclude") or "", slurm.get("account") or "",
    slurm.get("normal_partition") or "normal", slurm.get("preempt_partition") or "preempt",
    slurm.get("cpu_partition") or "cpu", slurm.get("normal_qos") or "",
    slurm.get("preempt_qos") or "", slurm.get("cpu_qos") or "",
)
separator = "\x1f"
for value in values:
    if "\n" in str(value) or separator in str(value):
        raise ValueError("runtime manifest values contain an unsupported control character")
print(separator.join(str(value) for value in values))
PY
)
IFS=$'\x1f' read -r \
    repo_root python conda_lib slurm_module exclude account \
    normal_partition preempt_partition cpu_partition \
    normal_qos preempt_qos cpu_qos <<< "$runtime_record"
if [[ -z "$repo_root" || -z "$python" ]]; then
    echo "RUN_MANIFEST runtime must define non-empty repo_root and python" >&2
    exit 2
fi
runner=$repo_root/scripts/eval/mainline/eval.py
module_prefix=""
[[ -n "$slurm_module" ]] && module_prefix="module load $slurm_module; "
native_prefix=""
[[ -n "$conda_lib" ]] && native_prefix="export LD_LIBRARY_PATH=$conda_lib:\${LD_LIBRARY_PATH:-}; "

if (( normal_lanes < 0 || preempt_lanes < 0 || normal_lanes + preempt_lanes < 1 || evaluators_per_gpu < 1 || coalesced_items_per_evaluator < 1 )); then
    echo "lane counts must be non-negative with a positive total" >&2
    exit 2
fi
if (( evaluators_per_gpu > 1 && coalesced_items_per_evaluator > 1 )); then
    echo "packed evaluators and shared-model coalescing are mutually exclusive" >&2
    exit 2
fi
test -s "$run_manifest"
mkdir -p "$output_root/logs"

while [[ ! -s "$wait_marker" ]]; do
    printf '%s waiting_for=%s\n' "$(date --iso-8601=seconds)" "$wait_marker"
    sleep 60
done
if [[ -s "$output_root/EVAL_COMPLETE.json" ]]; then
    echo "eval_already_complete=$output_root"
    exit 0
fi

physical_lane_count=$((normal_lanes + preempt_lanes))
logical_lane_count=$((physical_lane_count * evaluators_per_gpu))
cpus_per_task=32
memory=220G
if (( evaluators_per_gpu == 3 )); then
    cpus_per_task=57
    memory=300G
fi
startup_stagger_seconds=0
if (( evaluators_per_gpu > 1 )); then
    startup_stagger_seconds=30
fi
if (( coalesced_items_per_evaluator > 1 )); then
    normal_wrap="set -Eeuo pipefail; ${module_prefix}export VERL_VLA_PHYSICAL_GPU_ID=0; ${native_prefix}cd $repo_root; physical_lane=\$SLURM_ARRAY_TASK_ID; exec env -u SLURM_ARRAY_TASK_ID $python $runner coalesced-lane --run-manifest $run_manifest --lane-count $physical_lane_count --lane=\$physical_lane --items-per-evaluator $coalesced_items_per_evaluator"
    preempt_wrap="set -Eeuo pipefail; ${module_prefix}export VERL_VLA_PHYSICAL_GPU_ID=0; ${native_prefix}cd $repo_root; physical_lane=\$((SLURM_ARRAY_TASK_ID + ${normal_lanes})); exec env -u SLURM_ARRAY_TASK_ID $python $runner coalesced-lane --run-manifest $run_manifest --lane-count $physical_lane_count --lane=\$physical_lane --items-per-evaluator $coalesced_items_per_evaluator"
else
    normal_wrap="set -Eeuo pipefail; ${module_prefix}export VERL_VLA_PHYSICAL_GPU_ID=0; ${native_prefix}cd $repo_root; physical_lane=\$SLURM_ARRAY_TASK_ID; exec env -u SLURM_ARRAY_TASK_ID $python $runner packed-lane --run-manifest $run_manifest --physical-lane-count $physical_lane_count --physical-lane=\$physical_lane --evaluators-per-gpu $evaluators_per_gpu --startup-stagger-seconds $startup_stagger_seconds"
    preempt_wrap="set -Eeuo pipefail; ${module_prefix}export VERL_VLA_PHYSICAL_GPU_ID=0; ${native_prefix}cd $repo_root; physical_lane=\$((SLURM_ARRAY_TASK_ID + ${normal_lanes})); exec env -u SLURM_ARRAY_TASK_ID $python $runner packed-lane --run-manifest $run_manifest --physical-lane-count $physical_lane_count --physical-lane=\$physical_lane --evaluators-per-gpu $evaluators_per_gpu --startup-stagger-seconds $startup_stagger_seconds"
fi
normal_job=
preempt_job=
if (( normal_lanes > 0 )); then
    normal_args=(--parsable "--partition=$normal_partition" --nodes=1 --gpus-per-node=1)
    [[ -n "$account" ]] && normal_args+=("--account=$account")
    [[ -n "$normal_qos" ]] && normal_args+=("--qos=$normal_qos")
    [[ -n "$exclude" ]] && normal_args+=("--exclude=$exclude")
    normal_job=$(sbatch "${normal_args[@]}" \
        --cpus-per-task="$cpus_per_task" --mem="$memory" --time="$time_limit" --requeue \
        --array="0-$((normal_lanes - 1))%${normal_lanes}" \
        --job-name=review-split-n \
        --output="$output_root/logs/split_normal_%A_%a.out" \
        --wrap="$normal_wrap")
fi
if (( preempt_lanes > 0 )); then
    preempt_args=(--parsable "--partition=$preempt_partition" --nodes=1 --gpus-per-node=1)
    [[ -n "$account" ]] && preempt_args+=("--account=$account")
    [[ -n "$preempt_qos" ]] && preempt_args+=("--qos=$preempt_qos")
    [[ -n "$exclude" ]] && preempt_args+=("--exclude=$exclude")
    preempt_job=$(sbatch "${preempt_args[@]}" \
        --cpus-per-task="$cpus_per_task" --mem="$memory" --time="$time_limit" --requeue \
        --array="0-$((preempt_lanes - 1))%${preempt_lanes}" \
        --job-name=review-split-p \
        --output="$output_root/logs/split_preempt_%A_%a.out" \
        --wrap="$preempt_wrap")
fi
printf 'split_eval_submitted normal_job=%s preempt_job=%s physical_lanes=%d evaluators_per_gpu=%d coalesced_items=%d logical_lanes=%d manifest=%s\n' \
    "$normal_job" "$preempt_job" "$physical_lane_count" "$evaluators_per_gpu" "$coalesced_items_per_evaluator" "$logical_lane_count" "$run_manifest"

monitor=$repo_root/scripts/eval/mainline/monitor_split_eval.py
normal_monitor_arg=
preempt_monitor_arg=
if [[ -n "$normal_job" ]]; then
    normal_monitor_arg="--normal-array-job $normal_job"
fi
if [[ -n "$preempt_job" ]]; then
    preempt_monitor_arg="--preempt-array-job $preempt_job"
fi
monitor_args=(--parsable "--partition=$cpu_partition" --nodes=1 --cpus-per-task=1 --mem=2G)
[[ -n "$account" ]] && monitor_args+=("--account=$account")
[[ -n "$cpu_qos" ]] && monitor_args+=("--qos=$cpu_qos")
monitor_job=$(sbatch "${monitor_args[@]}" \
    --time=12:00:00 --requeue --job-name=eval-lane-watch \
    --output="$output_root/logs/split_monitor_%j.out" \
    --wrap="exec $python $monitor --run-manifest $run_manifest --normal-lanes $normal_lanes --preempt-lanes $preempt_lanes --evaluators-per-gpu $evaluators_per_gpu --coalesced-items-per-evaluator $coalesced_items_per_evaluator $normal_monitor_arg $preempt_monitor_arg --time $time_limit")
printf 'split_eval_monitor_submitted job=%s output_root=%s\n' "$monitor_job" "$output_root"
