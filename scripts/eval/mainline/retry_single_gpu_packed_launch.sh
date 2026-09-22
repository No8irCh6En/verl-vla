#!/usr/bin/env bash
set -euo pipefail

if (( $# < 3 || $# > 6 )); then
    echo "usage: $0 WAIT_MARKER RUN_MANIFEST OUTPUT_ROOT [TIME] [EVALUATORS_PER_GPU] [POLL_SECONDS]" >&2
    exit 2
fi

wait_marker=$(realpath -m "$1")
run_manifest=$(realpath -m "$2")
output_root=$(realpath -m "$3")
time_limit=${4:-02:00:00}
evaluators_per_gpu=${5:-3}
poll_seconds=${6:-60}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
launcher="$script_dir/launch_split_after_complete.sh"

test -s "$wait_marker"
test -s "$run_manifest"
mkdir -p "$output_root/logs"

attempt=0
while [[ ! -s "$output_root/EVAL_COMPLETE.json" ]]; do
    attempt=$((attempt + 1))
    for topology in normal preempt; do
        normal_lanes=0
        preempt_lanes=0
        if [[ "$topology" == normal ]]; then
            normal_lanes=1
        else
            preempt_lanes=1
        fi
        echo "$(date --iso-8601=seconds) launch_attempt=$attempt partition=$topology evaluators_per_gpu=$evaluators_per_gpu"
        set +e
        "$launcher" "$wait_marker" "$run_manifest" "$output_root" \
            "$normal_lanes" "$preempt_lanes" "$time_limit" "$evaluators_per_gpu"
        status=$?
        set -e
        if (( status == 0 )); then
            echo "$(date --iso-8601=seconds) packed_eval_submitted partition=$topology"
            exit 0
        fi
        echo "$(date --iso-8601=seconds) launch_deferred partition=$topology status=$status"
    done
    sleep "$poll_seconds"
done

echo "$(date --iso-8601=seconds) eval_already_complete=$output_root"
