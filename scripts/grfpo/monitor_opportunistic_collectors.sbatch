#!/usr/bin/env bash
#SBATCH --job-name=grfpo-opp-watch
# Site partition/account/QoS are supplied by the controller.
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
repo_root=${VVLA_REPO_ROOT:?VVLA_REPO_ROOT must be exported by the public launcher}
source "$repo_root/scripts/lib/runtime_env.sh"

window_dir=${GRFPO_WINDOW_DIR:?GRFPO_WINDOW_DIR must identify one collection window}
poll_seconds=${GRFPO_OPPORTUNISTIC_MONITOR_POLL_SECONDS:-60}
job_file="$window_dir/OPPORTUNISTIC_COLLECTOR_JOBS.tsv"
event_file="$window_dir/OPPORTUNISTIC_FATAL_EVENTS.tsv"
log_root=${GRFPO_LOG_ROOT:-$VVLA_LOG_ROOT}
normal_log_root=$VVLA_LOG_ROOT
python=$VVLA_PYTHON

# The sidecar can start a few seconds before the first collector creates the
# manifest.  Wait for that normal launch race, but never watch an unvalidated
# path indefinitely.
for _attempt in $(seq 1 60); do
    [[ -d "$window_dir" && -f "$window_dir/manifest.json" ]] && break
    sleep 5
done
if [[ ! -d "$window_dir" || ! -f "$window_dir/manifest.json" ]]; then
    echo "invalid collection window after startup grace: $window_dir" >&2
    exit 2
fi

release_owner() {
    local owner=$1
    PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}" \
        "$python" -m verl_vla.entrypoints.grfpo_release_owner_leases \
        "$window_dir" "$owner"
}

# OPPORTUNISTIC_COLLECTOR_JOBS.tsv stores the compact array range emitted by
# launch_fragmented_update.sh/add_opportunistic_collectors.sh.  Current
# launchers use either N or A-B; reject unexpected syntax instead of silently
# monitoring the wrong elements.
expand_array_range() {
    local spec=$1 start end lane
    spec=${spec%%%*}
    if [[ "$spec" =~ ^[0-9]+$ ]]; then
        printf '%s\n' "$spec"
    elif [[ "$spec" =~ ^([0-9]+)-([0-9]+)$ ]]; then
        start=${BASH_REMATCH[1]}
        end=${BASH_REMATCH[2]}
        for ((lane = start; lane <= end; lane++)); do
            printf '%s\n' "$lane"
        done
    else
        echo "unsupported opportunistic array range: $spec" >&2
        return 2
    fi
}

fatal_after_latest_supervisor() {
    local stdout_log=$1 stderr_log=$2 partition=$3 supervisor_line fatal_line
    [[ -f "$stdout_log" ]] || return 1
    supervisor_line=$(grep -an 'collector_supervisor .*attempt=' "$stdout_log" 2>/dev/null \
        | tail -n 1 | cut -d: -f1 || true)
    supervisor_line=${supervisor_line:-0}
    if [[ "$partition" == preempt ]]; then
        fatal_line=$(grep -anE '\[Fatal\] \[carb\.crashreporter|ERROR_OUT_OF_DEVICE_MEMORY' \
            "$stdout_log" 2>/dev/null | tail -n 1 | cut -d: -f1 || true)
        fatal_line=${fatal_line:-0}
        (( fatal_line > supervisor_line ))
        return
    fi
    [[ -f "$stderr_log" ]] || return 1
    grep -aqE '\[Fatal\] \[carb\.crashreporter|ERROR_OUT_OF_DEVICE_MEMORY' "$stderr_log" \
        || return 1
    # Normal collectors split stdout and stderr, so compare mtimes rather than
    # unrelated line numbers.  A healthy restarted attempt advances stdout.
    (( $(stat -c '%Y' "$stderr_log") > $(stat -c '%Y' "$stdout_log") ))
}

declare -A handled
while [[ -e "$window_dir/OPEN" && ! -e "$window_dir/CLOSED.json" ]]; do
    if [[ -s "$job_file" ]]; then
        while IFS=$'\t' read -r job array_range partition _placement _submitted_at; do
            [[ "$job" =~ ^[0-9]+$ ]] || continue
            while IFS= read -r lane; do
                element="${job}_${lane}"
                [[ -z "${handled[$element]:-}" ]] || continue
                state=$(squeue -h -j "$element" -o '%T' 2>/dev/null | head -n 1 || true)
                [[ "$state" == RUNNING ]] || continue
                node=$(squeue -h -j "$element" -o '%N' 2>/dev/null | head -n 1 || true)
                if [[ "$partition" == preempt ]]; then
                    stdout_log="$log_root/fragmented_collect_${job}_${lane}.out"
                    stderr_log=$stdout_log
                else
                    stdout_log="$normal_log_root/grfpo_fragmented_collect_${job}_${lane}.out"
                    stderr_log="$normal_log_root/grfpo_fragmented_collect_${job}_${lane}.err"
                fi
                if fatal_after_latest_supervisor "$stdout_log" "$stderr_log" "$partition"; then
                    handled[$element]=1
                    printf '%s\t%s\t%s\t%s\n' \
                        "$(date --iso-8601=seconds)" "$element" "${node:-unknown}" "$partition" \
                        >> "$event_file"
                    echo "opportunistic_collector_fatal element=$element node=${node:-unknown} partition=$partition"
                    scancel "$element" 2>/dev/null || true
                    release_owner "${job}:${lane}" \
                        || echo "warning: failed to release opportunistic owner=${job}:${lane}" >&2
                fi
            done < <(expand_array_range "$array_range")
        done < "$job_file"
    fi
    sleep "$poll_seconds"
done

echo "opportunistic_monitor_complete window=$window_dir"
