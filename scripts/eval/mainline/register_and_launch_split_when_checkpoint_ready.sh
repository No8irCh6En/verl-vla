#!/usr/bin/env bash
set -euo pipefail

if (( $# < 3 || $# > 8 )); then
    echo "usage: $0 CHECKPOINT_DIR SPEC OUTPUT_ROOT [NORMAL_GPU_LANES] [PREEMPT_GPU_LANES] [TIME] [EVALUATORS_PER_GPU] [COALESCED_ITEMS_PER_EVALUATOR]" >&2
    exit 2
fi

checkpoint_dir=$(realpath -m "$1")
spec=$(realpath -m "$2")
output_root=$(realpath -m "$3")
normal_lanes=${4:-4}
preempt_lanes=${5:-4}
time_limit=${6:-12:00:00}
evaluators_per_gpu=${7:-1}
coalesced_items_per_evaluator=${8:-1}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../.." && pwd)
source "$repo_root/scripts/lib/runtime_env.sh"
vvla_require_fastwam_runtime
python=$VVLA_PYTHON
checkpoint_marker="$checkpoint_dir/actor/model_world_size_1_rank_0.pt"
run_manifest="$output_root/RUN_MANIFEST.json"

test -s "$spec"
mkdir -p "$output_root/logs"

while [[ ! -s "$checkpoint_marker" ]]; do
    printf '%s checkpoint_pending=%s\n' "$(date --iso-8601=seconds)" "$checkpoint_dir"
    sleep 300
done

if [[ -s "$output_root/EVAL_COMPLETE.json" ]]; then
    echo "eval_already_complete=$output_root"
    exit 0
fi

if [[ ! -s "$run_manifest" ]]; then
    total_lanes=$((normal_lanes + preempt_lanes))
    "$python" "$repo_root/scripts/eval/mainline/eval.py" launch \
        --spec "$spec" \
        --output-root "$output_root" \
        --policy-root "$FASTWAM_POLICY_ROOT" \
        --robodojo-root "$ROBODOJO_ROOT" \
        --hf-home "$HF_HOME" \
        --python "$python" \
        --partition normal \
        --lanes "$total_lanes" \
        --evaluators-per-gpu "$evaluators_per_gpu" \
        --time "$time_limit" \
        --dry-run
fi

exec "$repo_root/scripts/eval/mainline/launch_split_after_complete.sh" \
    "$checkpoint_marker" "$run_manifest" "$output_root" \
    "$normal_lanes" "$preempt_lanes" "$time_limit" "$evaluators_per_gpu" \
    "$coalesced_items_per_evaluator"
