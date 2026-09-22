#!/usr/bin/env bash
set -euo pipefail

checkpoint_dir=${1:?usage: launch_when_checkpoint_ready.sh CHECKPOINT_DIR SPEC OUTPUT_ROOT [PARTITION] [LANES] [TIME] [POLICY_LABEL]}
spec=${2:?usage: launch_when_checkpoint_ready.sh CHECKPOINT_DIR SPEC OUTPUT_ROOT [PARTITION] [LANES] [TIME] [POLICY_LABEL]}
output_root=${3:?usage: launch_when_checkpoint_ready.sh CHECKPOINT_DIR SPEC OUTPUT_ROOT [PARTITION] [LANES] [TIME] [POLICY_LABEL]}
partition=${4:-preempt}
lanes=${5:-4}
time_limit=${6:-12:00:00}
policy_label=${7:-}

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../.." && pwd)
source "$repo_root/scripts/lib/runtime_env.sh"
vvla_require_fastwam_runtime
python=$VVLA_PYTHON
checkpoint_dir=$(realpath -m "$checkpoint_dir")
spec=$(realpath -m "$spec")
output_root=$(realpath -m "$output_root")

if [[ "$partition" != normal && "$partition" != preempt ]]; then
    echo "partition must be normal or preempt, got $partition" >&2
    exit 2
fi
if [[ ! "$lanes" =~ ^[1-9][0-9]*$ ]]; then
    echo "lanes must be a positive integer, got $lanes" >&2
    exit 2
fi
test -s "$spec"
mkdir -p "$output_root/logs"

while true; do
    if [[ -s "$output_root/LAUNCH.json" || -s "$output_root/EVAL_COMPLETE.json" ]]; then
        echo "eval_already_launched_or_complete=$output_root"
        exit 0
    fi
    if [[ -s "$checkpoint_dir/actor/model_world_size_1_rank_0.pt" ]]; then
        echo "checkpoint_ready=$checkpoint_dir"
        cd "$repo_root"
        launch_args=(
            scripts/eval/mainline/eval.py launch
            --spec "$spec" \
            --output-root "$output_root" \
            --partition "$partition" \
            --lanes "$lanes" \
            --time "$time_limit"
        )
        if [[ -n "$policy_label" ]]; then
            launch_args+=(--policy-label "$policy_label")
        fi
        exec "$python" "${launch_args[@]}"
    fi
    printf '%s checkpoint_pending=%s\n' "$(date --iso-8601=seconds)" "$checkpoint_dir"
    sleep 300
done
