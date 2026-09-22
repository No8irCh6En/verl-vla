#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../.." && pwd)
source "$repo_root/scripts/lib/runtime_env.sh"
vvla_require_python
python=$VVLA_PYTHON

usage() {
    cat <<'EOF'
usage:
  manage.sh status RUN_ID
  manage.sh replace-slow RUN_ID LANE [REASON]
  manage.sh replace-node RUN_ID NODE [REASON]
  manage.sh add-collectors RUN_ID COUNT [normal|preempt] [NODELIST]
  manage.sh resume RUN_ID

LANE is the stable primary collector lane shown by `status`. replace-slow is
handled by the durable CPU watcher: it cancels that lane, releases its one
uncommitted lease, avoids the current node, and submits a replacement.
replace-node creates the same request for every primary lane currently running
on NODE.
EOF
}

command=${1:-}
run_id=${2:-}
[[ -n "$command" && -n "$run_id" ]] || { usage >&2; exit 2; }
vvla_output_root=$VVLA_OUTPUT_ROOT
vvla_output_root=$(cd -- "$vvla_output_root" && pwd)
output_root="$vvla_output_root/grfpo/$run_id"
state_path="$output_root/CHAIN_STATE.json"
[[ -s "$state_path" ]] || { echo "missing chain state: $state_path" >&2; exit 2; }

theta=$($python - "$state_path" <<'PY'
import json
import sys
print(int(json.load(open(sys.argv[1], encoding="utf-8"))["current_theta"]))
PY
)
intended_update=$((theta + 1))
window="$output_root/candidate_spool/$run_id/$(printf 'update_%04d_theta_%04d' "$intended_update" "$theta")"

request_replacement() {
    local lane=$1 reason=$2 request_dir
    [[ "$lane" =~ ^[0-9]+$ ]] || { echo "invalid lane: $lane" >&2; return 2; }
    [[ -s "$window/manifest.json" && ! -s "$window/CLOSED.json" ]] \
        || { echo "window is not open: $window" >&2; return 2; }
    request_dir="$window/REPLACE_REQUESTS"
    mkdir -p "$request_dir"
    "$python" - "$request_dir/lane_${lane}.json" "$lane" "$reason" <<'PY'
import json
import os
import sys
import time
from pathlib import Path

path = Path(sys.argv[1])
payload = {"lane": int(sys.argv[2]), "reason": sys.argv[3], "requested_at_unix_s": time.time()}
tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(tmp, path)
PY
}

case "$command" in
    status)
        cat "$state_path"
        echo "window=$window"
        if [[ -s "$window/manifest.json" ]]; then
            "$repo_root/scripts/grfpo/summarize_policy_window.py" "$window" --quiet || true
        fi
        if [[ -s "$window/COLLECTOR_JOBS.tsv" ]]; then
            printf '%-6s %-16s %-10s %-10s %-12s %s\n' LANE ELEMENT PARTITION STATE ELAPSED NODE
            while IFS=$'\t' read -r lane job partition; do
                [[ "$lane" =~ ^[0-9]+$ && -n "$job" ]] || continue
                element="${job}_${lane}"
                read -r state elapsed node < <(squeue -h -j "$element" -o '%T %M %N' 2>/dev/null | head -n 1)
                printf '%-6s %-16s %-10s %-10s %-12s %s\n' \
                    "$lane" "$element" "$partition" "${state:-MISSING}" "${elapsed:--}" "${node:--}"
            done < "$window/COLLECTOR_JOBS.tsv"
        fi
        ;;
    replace-slow)
        lane=${3:-}
        reason=${4:-operator_observed_slow_collector}
        request_replacement "$lane" "$reason"
        echo "replacement_requested lane=$lane window=$window watcher_poll_seconds<=60"
        ;;
    replace-node)
        node=${3:-}
        reason=${4:-operator_observed_slow_node}
        [[ -n "$node" && -s "$window/COLLECTOR_JOBS.tsv" ]] || { usage >&2; exit 2; }
        matched=0
        while IFS=$'\t' read -r lane job _partition; do
            [[ "$lane" =~ ^[0-9]+$ && -n "$job" ]] || continue
            element="${job}_${lane}"
            current_node=$(squeue -h -j "$element" -o '%N' 2>/dev/null | head -n 1 || true)
            if [[ "$current_node" == "$node" ]]; then
                request_replacement "$lane" "$reason:$node"
                echo "replacement_requested lane=$lane element=$element node=$node"
                matched=$((matched + 1))
            fi
        done < "$window/COLLECTOR_JOBS.tsv"
        (( matched > 0 )) || { echo "no running primary collector found on node=$node" >&2; exit 3; }
        echo "replacement_requests=$matched node=$node watcher_poll_seconds<=60"
        ;;
    add-collectors)
        count=${3:-}
        partition=${4:-preempt}
        nodelist=${5:-}
        [[ "$count" =~ ^[1-9][0-9]*$ ]] || { usage >&2; exit 2; }
        "$repo_root/scripts/grfpo/add_opportunistic_collectors.sh" \
            "$window" "$count" "$partition" "$nodelist"
        ;;
    resume)
        "$repo_root/scripts/grfpo/resume_fragmented_window.sh" "$window"
        ;;
    *) usage >&2; exit 2 ;;
esac
