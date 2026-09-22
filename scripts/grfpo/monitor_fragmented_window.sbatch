#!/usr/bin/env bash
#SBATCH --job-name=grfpo-window-watch
# Site partition/account/QoS are supplied by the controller.
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
repo_root=${VVLA_REPO_ROOT:?VVLA_REPO_ROOT must be exported by the public launcher}
script_dir="$repo_root/scripts/grfpo"
source "$repo_root/scripts/lib/runtime_env.sh"
window_dir="$GRFPO_COLLECTION_ROOT/$GRFPO_RUN_ID/$(printf 'update_%04d_theta_%04d' "$GRFPO_INTENDED_UPDATE" "$GRFPO_THETA_OLD")"
mkdir -p "$GRFPO_LOG_ROOT" "$window_dir/accepted" "$window_dir/rejected"
collector_count=${GRFPO_COLLECTOR_COUNT:-15}
collector_concurrency=${GRFPO_COLLECTOR_CONCURRENCY:-8}
shared_candidate_queue=${GRFPO_SHARED_CANDIDATE_QUEUE:-0}
# Slurm's --export itself uses commas as separators, so callers may pass this
# list colon-delimited.  Normalize only after the job environment is restored.
replacement_exclude_nodes=${GRFPO_REPLACEMENT_EXCLUDE_NODES:-}
replacement_exclude_nodes=${replacement_exclude_nodes//:/,}
# A command-line --exclude overrides the corresponding #SBATCH directive.
# Keep the launcher's validated denylist in the watcher as well, then add
# nodes that fatally crash during this window. This prevents replacement jobs
# from repeatedly consuming startup time on the same unhealthy/overloaded
# host without permanently changing cluster-wide placement policy.
# dgx-01 repeatedly crashed Isaac's gpu.foundation renderer during the theta7
# window.  Keep it out of collector replacements while leaving pure training
# placement unchanged.
validated_exclude_nodes=${SLURM_EXCLUDE:-}
if [[ -n "$replacement_exclude_nodes" ]]; then
    replacement_exclude_nodes="$validated_exclude_nodes,$replacement_exclude_nodes"
else
    replacement_exclude_nodes=$validated_exclude_nodes
fi
normal_collector_sbatch="$script_dir/submit_fragmented_collector_normal.sbatch"
preempt_collector_sbatch="$script_dir/submit_fragmented_collector_preempt.sbatch"
watcher_sbatch="$script_dir/monitor_fragmented_window.sbatch"
summary_script="$script_dir/summarize_policy_window.py"
python=$VVLA_PYTHON
collector_job_file="$window_dir/COLLECTOR_JOBS.tsv"
initial_collector_partition=${GRFPO_INITIAL_COLLECTOR_PARTITION:-normal}
poll_seconds=${GRFPO_MONITOR_POLL_SECONDS:-60}
normal_pending_grace_seconds=${GRFPO_NORMAL_PENDING_GRACE_SECONDS:-600}
heartbeat_timeout_seconds=${GRFPO_COLLECTOR_HEARTBEAT_TIMEOUT_SECONDS:-900}
node_quarantine_fatal_threshold=${GRFPO_NODE_QUARANTINE_FATAL_THRESHOLD:-2}
max_fatal_replacements_per_slot=${GRFPO_MAX_FATAL_REPLACEMENTS_PER_SLOT:-4}
max_circuit_rescues_per_slot=${GRFPO_MAX_CIRCUIT_RESCUES_PER_SLOT:-1}
normal_log_root=$VVLA_LOG_ROOT
node_fatal_event_file="$window_dir/NODE_FATAL_EVENTS.tsv"
circuit_rescue_event_file="$window_dir/CIRCUIT_RESCUE_EVENTS.tsv"
replacement_request_dir="$window_dir/REPLACE_REQUESTS"
replacement_event_file="$window_dir/OPERATOR_REPLACEMENTS.tsv"
mkdir -p "$replacement_request_dir"
declare -a collector_jobs
declare -a collector_partitions
declare -a missing_polls
declare -a normal_pending_polls
declare -a fatal_replacements
declare -a circuit_rescues
declare -a disabled_slots
declare -A node_fatal_counts
declare -A seen_fatal_elements
for ((slot = 0; slot < collector_count; slot++)); do
    collector_jobs[$slot]=$GRFPO_COLLECT_JOB_ID
    collector_partitions[$slot]=$initial_collector_partition
    missing_polls[$slot]=0
    normal_pending_polls[$slot]=0
    fatal_replacements[$slot]=0
    circuit_rescues[$slot]=0
    disabled_slots[$slot]=0
done

# Preserve health/circuit evidence across watcher renewal or manual repair.
# Without this, restarting only the CPU watcher forgets bad placements and can
# spend several more GPU allocations rediscovering the same renderer failures.
if [[ -s "$node_fatal_event_file" ]]; then
    while IFS=$'\t' read -r _timestamp event_lane event_element event_node; do
        [[ "$event_lane" =~ ^[0-9]+$ && -n "$event_element" && -n "$event_node" ]] || continue
        [[ -n "${seen_fatal_elements[$event_element]:-}" ]] && continue
        seen_fatal_elements[$event_element]=1
        if [[ "$event_node" != unknown ]]; then
            node_fatal_counts[$event_node]=$(( ${node_fatal_counts[$event_node]:-0} + 1 ))
        fi
        if (( event_lane < collector_count )); then
            fatal_replacements[$event_lane]=$((fatal_replacements[$event_lane] + 1))
        fi
    done < "$node_fatal_event_file"
fi
if [[ -s "$circuit_rescue_event_file" ]]; then
    while IFS=$'\t' read -r _timestamp event_lane _event_job; do
        [[ "$event_lane" =~ ^[0-9]+$ ]] || continue
        if (( event_lane < collector_count )); then
            circuit_rescues[$event_lane]=$((circuit_rescues[$event_lane] + 1))
        fi
    done < "$circuit_rescue_event_file"
fi
for failed_node in "${!node_fatal_counts[@]}"; do
    if (( node_fatal_counts[$failed_node] >= node_quarantine_fatal_threshold )); then
        case ",$replacement_exclude_nodes," in
            *",$failed_node,"*) ;;
            *) replacement_exclude_nodes="$replacement_exclude_nodes,$failed_node" ;;
        esac
    fi
done
replacement_exclude_nodes=${replacement_exclude_nodes#,}
export GRFPO_REPLACEMENT_EXCLUDE_NODES=${replacement_exclude_nodes//,/:}

# A renewed watcher inherits the latest per-lane replacement IDs rather than
# reverting to the original array job ID.
if [[ -s "$collector_job_file" ]]; then
    while IFS=$'\t' read -r lane job partition; do
        if [[ "$lane" =~ ^[0-9]+$ ]] && (( lane < collector_count )) && [[ -n "$job" ]]; then
            collector_jobs[$lane]=$job
            if [[ "$partition" == normal || "$partition" == preempt ]]; then
                collector_partitions[$lane]=$partition
            elif [[ "$partition" == disabled ]]; then
                collector_partitions[$lane]=disabled
                disabled_slots[$lane]=1
            fi
        fi
    done < "$collector_job_file"
fi

renew_watcher() {
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        vvla_sbatch_site_args cpu "$GRFPO_LOG_ROOT/fragmented_watch_%j.out"
        successor=$(sbatch --parsable \
            "${VVLA_SBATCH_SITE_ARGS[@]}" \
            --dependency="afterany:$SLURM_JOB_ID" \
            --export=ALL \
            "$watcher_sbatch")
        printf '%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "$SLURM_JOB_ID" "$successor" \
            >> "$window_dir/WATCH_RENEWALS.tsv"
        echo "watcher_renewal successor=$successor"
        exit 0
    fi
    exit 5
}
trap renew_watcher USR1

write_collector_jobs() {
    local temporary="$collector_job_file.tmp-$$"
    local lane
    : > "$temporary"
    for ((lane = 0; lane < collector_count; lane++)); do
        printf '%s\t%s\t%s\n' \
            "$lane" "${collector_jobs[$lane]}" "${collector_partitions[$lane]}" \
            >> "$temporary"
    done
    mv "$temporary" "$collector_job_file"
}

submit_collector_lane() {
    local partition=$1
    local lane=$2
    local avoid_node=${3:-}
    local sbatch_script submit_exclude
    case "$partition" in
        normal) sbatch_script=$normal_collector_sbatch ;;
        preempt) sbatch_script=$preempt_collector_sbatch ;;
        *) echo "unsupported collector partition=$partition" >&2; return 2 ;;
    esac
    vvla_sbatch_site_args "$partition" "$GRFPO_LOG_ROOT/fragmented_collect_%A_%a.out"
    local replacement_args=(--parsable "${VVLA_SBATCH_SITE_ARGS[@]}" --array="$lane" --export=ALL)
    submit_exclude=$replacement_exclude_nodes
    if [[ -n "$avoid_node" ]]; then
        case ",$submit_exclude," in
            *",$avoid_node,"*) ;;
            *) submit_exclude="$submit_exclude,$avoid_node" ;;
        esac
    fi
    submit_exclude=${submit_exclude#,}
    if [[ -n "$submit_exclude" ]]; then
        replacement_args+=(--exclude="$submit_exclude")
    fi
    # QOSMaxSubmitJobPerUserLimit is transient while other collector/eval
    # arrays drain.  A failed replacement submission must not terminate the
    # only durable window watcher: that leaves live lanes unmanaged and a
    # complete window can never transition to training.  Keep the watcher
    # alive and retry the exact same lane/partition until Slurm accepts it.
    local attempt=0 output
    while true; do
        attempt=$((attempt + 1))
        if output=$(sbatch "${replacement_args[@]}" "$sbatch_script"); then
            printf '%s\n' "$output"
            return 0
        fi
        echo "collector_submit_retry lane=$lane partition=$partition attempt=$attempt" >&2
        sleep 30
    done
}

quarantine_collector_node() {
    local element=$1 lane=$2 failed_node count
    last_failed_node=""
    fatal_event_new=0
    failed_node=$(squeue -h -j "$element" -o '%N' 2>/dev/null | head -n 1 || true)
    if [[ -z "$failed_node" || "$failed_node" == "(null)" ]]; then
        failed_node=unknown
    else
        last_failed_node=$failed_node
    fi
    if [[ -z "${seen_fatal_elements[$element]:-}" ]]; then
        printf '%s\t%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "$lane" "$element" "$failed_node" \
            >> "$node_fatal_event_file"
        seen_fatal_elements[$element]=1
        fatal_event_new=1
    fi
    if [[ "$failed_node" == unknown ]]; then
        return 0
    fi
    case ",$replacement_exclude_nodes," in
        *",$failed_node,"*) return 0 ;;
    esac
    count=$(( ${node_fatal_counts[$failed_node]:-0} + 1 ))
    node_fatal_counts[$failed_node]=$count
    if (( count < node_quarantine_fatal_threshold )); then
        echo "collector_node_fatal_observed element=$element node=$failed_node count=$count/$node_quarantine_fatal_threshold action=retain"
        return 0
    fi
    case ",$replacement_exclude_nodes," in
        *",$failed_node,"*) ;;
        *)
            replacement_exclude_nodes="$replacement_exclude_nodes,$failed_node"
            export GRFPO_REPLACEMENT_EXCLUDE_NODES=${replacement_exclude_nodes//,/:}
            echo "collector_node_quarantined element=$element node=$failed_node count=$count exclude=$replacement_exclude_nodes"
            ;;
    esac
}

abandon_collector_leases() {
    local job=$1 lane=$2
    if [[ "$shared_candidate_queue" != 1 ]]; then
        return 0
    fi
    PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}" \
        "$python" -m verl_vla.entrypoints.grfpo_release_owner_leases \
        "$window_dir" "${job}:${lane}" \
        || echo "warning: failed to release leases owner=${job}:${lane}" >&2
}

lane_complete() {
    local slot=$1 candidate candidate_id
    if [[ "$shared_candidate_queue" == 1 ]]; then
        # A shared worker has no permanent candidate slice. It may exit only
        # after the entire predeclared queue is durable; otherwise replacing it
        # preserves available collection parallelism and stale leases will be
        # reclaimed after their bounded timeout.
        for ((candidate = 0; candidate < GRFPO_MAX_CANDIDATES; candidate++)); do
            printf -v candidate_id 'candidate_%04d' "$candidate"
            if [[ ! -s "$window_dir/rejected/$candidate_id.json" && \
                  ! -s "$window_dir/skipped/$candidate_id.json" && \
                  ! -f "$window_dir/accepted/$candidate_id/COMMITTED" ]]; then
                return 1
            fi
        done
        return 0
    fi
    for ((candidate = slot; candidate < GRFPO_MAX_CANDIDATES; candidate += collector_count)); do
        printf -v candidate_id 'candidate_%04d' "$candidate"
        if [[ ! -s "$window_dir/rejected/$candidate_id.json" && \
              ! -s "$window_dir/skipped/$candidate_id.json" && \
              ! -f "$window_dir/accepted/$candidate_id/COMMITTED" ]]; then
            return 1
        fi
    done
    return 0
}

write_collector_jobs

process_operator_replacement_requests() {
    local request lane old_job old_element partition avoid_node replacement reason processed
    shopt -s nullglob
    for request in "$replacement_request_dir"/lane_*.json; do
        readarray -t request_fields < <("$python" - "$request" <<'PY'
import json
import sys
record = json.load(open(sys.argv[1], encoding="utf-8"))
print(int(record["lane"]))
print(str(record.get("reason", "operator_requested")))
PY
        )
        lane=${request_fields[0]}
        reason=${request_fields[1]}
        if (( lane < 0 || lane >= collector_count )); then
            echo "invalid replacement request lane=$lane file=$request" >&2
            mv "$request" "$request.invalid"
            continue
        fi
        old_job=${collector_jobs[$lane]}
        old_element="${old_job}_${lane}"
        partition=${collector_partitions[$lane]}
        [[ "$partition" == normal || "$partition" == preempt ]] || partition=preempt
        avoid_node=$(squeue -h -j "$old_element" -o '%N' 2>/dev/null | head -n 1 || true)
        [[ "$avoid_node" == "(null)" ]] && avoid_node=""
        scancel "$old_element" 2>/dev/null || true
        abandon_collector_leases "$old_job" "$lane"
        if replacement=$(submit_collector_lane "$partition" "$lane" "$avoid_node"); then
            collector_jobs[$lane]=$replacement
            collector_partitions[$lane]=$partition
            missing_polls[$lane]=0
            normal_pending_polls[$lane]=0
            write_collector_jobs
            printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
                "$(date --iso-8601=seconds)" "$lane" "$old_element" "${avoid_node:-unknown}" \
                "${replacement}_${lane}" "$reason" >> "$replacement_event_file"
            processed="$request.processed-$(date +%s)"
            mv "$request" "$processed"
            echo "collector_operator_replaced lane=$lane old=$old_element new=${replacement}_${lane} avoid=${avoid_node:-none} reason=$reason"
        else
            echo "collector_operator_replacement_submit_failed lane=$lane old=$old_element" >&2
        fi
    done
    shopt -u nullglob
}

while [[ ! -s "$window_dir/CLOSED.json" ]]; do
    process_operator_replacement_requests
    accepted=$(find "$window_dir/accepted" -mindepth 2 -maxdepth 2 -name COMMITTED 2>/dev/null | wc -l)
    rejected=$(find "$window_dir/rejected" -mindepth 1 -maxdepth 1 -name 'candidate_*.json' 2>/dev/null | wc -l)
    printf '%s theta_old=%s accepted=%s/%s rejected=%s collector_job=%s\n' \
        "$(date --iso-8601=seconds)" "$GRFPO_THETA_OLD" "$accepted" "$GRFPO_TARGET_GROUPS" \
        "$rejected" "$(tr '\n' ',' < "$collector_job_file")"
    if [[ -s "$window_dir/manifest.json" ]]; then
        "$summary_script" "$window_dir" \
            --json "$window_dir/ONLINE_POLICY_METRICS.json" \
            --markdown "$window_dir/ONLINE_POLICY_METRICS.md" \
            --run-json "$window_dir/../ONLINE_CHECKPOINT_COMPARISON.json" \
            --run-markdown "$window_dir/../ONLINE_CHECKPOINT_COMPARISON.md" \
            --quiet || echo "warning: live policy summary failed" >&2
    fi
    for ((lane = 0; lane < collector_count; lane++)); do
        if (( disabled_slots[$lane] == 1 )); then
            continue
        fi
        element="${collector_jobs[$lane]}_$lane"
        job_state=$(squeue -h -j "$element" -o '%T' 2>/dev/null | head -n 1 || true)
        if [[ -n "$job_state" ]]; then
            missing_polls[$lane]=0
            if [[ "$job_state" == RUNNING ]]; then
                if [[ "${collector_partitions[$lane]}" == normal ]]; then
                    heartbeat_log="$normal_log_root/grfpo_fragmented_collect_${collector_jobs[$lane]}_${lane}.out"
                    fatal_log="$normal_log_root/grfpo_fragmented_collect_${collector_jobs[$lane]}_${lane}.err"
                else
                    heartbeat_log="$GRFPO_LOG_ROOT/fragmented_collect_${collector_jobs[$lane]}_${lane}.out"
                    # The preempt launcher does not split stderr, so its fatal
                    # diagnostics and supervisor heartbeat share one file.
                    fatal_log="$heartbeat_log"
                fi
                if [[ -f "$heartbeat_log" ]]; then
                    # Isaac's crash reporter can keep appending a backtrace for
                    # many minutes after the renderer process has fatally
                    # crashed.  A pure mtime heartbeat therefore mistakes a
                    # dead collector for useful progress.  Treat a fatal line
                    # written after the latest supervisor attempt as a dead
                    # attempt; fatal text from an older, already-restarted
                    # attempt is deliberately ignored.
                    fatal_detected=0
                    if [[ "$fatal_log" == "$heartbeat_log" ]]; then
                        supervisor_line=$(grep -an 'collector_supervisor .*attempt=' "$heartbeat_log" 2>/dev/null \
                            | tail -n 1 | cut -d: -f1 || true)
                        fatal_line=$(grep -anE '\[Fatal\] \[carb\.crashreporter|ERROR_OUT_OF_DEVICE_MEMORY' "$fatal_log" 2>/dev/null \
                            | tail -n 1 | cut -d: -f1 || true)
                        supervisor_line=${supervisor_line:-0}
                        fatal_line=${fatal_line:-0}
                        (( fatal_line > supervisor_line )) && fatal_detected=1
                    elif grep -aqE '\[Fatal\] \[carb\.crashreporter|ERROR_OUT_OF_DEVICE_MEMORY' "$fatal_log" 2>/dev/null; then
                        # Normal jobs split stdout/stderr.  Line numbers from
                        # separate files are not comparable.  A restarted,
                        # healthy attempt keeps advancing stdout beyond an old
                        # stderr fatal; an active crash reporter instead keeps
                        # stderr newer than the stopped rollout heartbeat.
                        fatal_mtime=$(stat -c '%Y' "$fatal_log" 2>/dev/null || echo 0)
                        heartbeat_mtime=$(stat -c '%Y' "$heartbeat_log" 2>/dev/null || echo 0)
                        (( fatal_mtime > heartbeat_mtime )) && fatal_detected=1
                    fi
                    if (( fatal_detected )); then
                        echo "collector_crash_reporter_fatal slot=$lane element=$element heartbeat_log=$heartbeat_log fatal_log=$fatal_log"
                        quarantine_collector_node "$element" "$lane"
                        # A reporter fatal is authoritative: the Isaac process is
                        # no longer capable of committing its group even when
                        # Slurm still reports RUNNING while Breakpad unwinds.
                        # Release its shared-queue lease before requesting a
                        # replacement so a healthy collector can steal the work
                        # even if sbatch is temporarily unavailable.
                        scancel "$element" 2>/dev/null || true
                        abandon_collector_leases "${collector_jobs[$lane]}" "$lane"
                        if (( fatal_event_new == 1 )); then
                            fatal_replacements[$lane]=$((fatal_replacements[$lane] + 1))
                        fi
                        if [[ "$shared_candidate_queue" == 1 ]] \
                            && (( fatal_replacements[$lane] >= max_fatal_replacements_per_slot )); then
                            # One last placement on preempt can use a currently
                            # idle healthy node even when repeated normal
                            # replacements landed on bad/overloaded hosts.  It
                            # is explicitly bounded so a broken lane cannot
                            # churn forever.  Candidate ownership was already
                            # released above, so healthy warm workers remain
                            # free to steal the same backlog meanwhile.
                            if (( circuit_rescues[$lane] < max_circuit_rescues_per_slot )) \
                                && replacement=$(submit_collector_lane preempt "$lane" "$last_failed_node"); then
                                circuit_rescues[$lane]=$((circuit_rescues[$lane] + 1))
                                printf '%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "$lane" "$replacement" \
                                    >> "$circuit_rescue_event_file"
                                collector_jobs[$lane]=$replacement
                                collector_partitions[$lane]=preempt
                                missing_polls[$lane]=0
                                normal_pending_polls[$lane]=0
                                write_collector_jobs
                                echo "collector_slot_circuit_rescue slot=$lane job=$replacement rescue=${circuit_rescues[$lane]}/$max_circuit_rescues_per_slot fatal_replacements=${fatal_replacements[$lane]} partition=preempt"
                                continue
                            fi
                            collector_jobs[$lane]=disabled
                            collector_partitions[$lane]=disabled
                            disabled_slots[$lane]=1
                            write_collector_jobs
                            echo "collector_slot_circuit_open slot=$lane fatal_replacements=${fatal_replacements[$lane]} circuit_rescues=${circuit_rescues[$lane]} unfinished_candidates_remain_in_shared_queue=1"
                            continue
                        fi
                        if replacement=$(submit_collector_lane normal "$lane" "$last_failed_node"); then
                            collector_jobs[$lane]=$replacement
                            collector_partitions[$lane]=normal
                            missing_polls[$lane]=0
                            normal_pending_polls[$lane]=0
                            write_collector_jobs
                            echo "resubmitted_crashed_collector slot=$lane job=$replacement partition=normal prior_element=$element"
                        else
                            echo "crashed_collector_resubmit_failed slot=$lane; dead element canceled and lease released" >&2
                        fi
                        continue
                    fi
                    heartbeat_mtime=$(stat -c '%Y' "$heartbeat_log" 2>/dev/null || date +%s)
                    heartbeat_age=$(( $(date +%s) - heartbeat_mtime ))
                    if (( heartbeat_age >= heartbeat_timeout_seconds )); then
                        echo "collector_heartbeat_timeout slot=$lane element=$element age_s=$heartbeat_age log=$heartbeat_log"
                        # Submit first: if the scheduler rejects the replacement,
                        # preserve the old allocation for manual diagnosis.
                        if replacement=$(submit_collector_lane normal "$lane"); then
                            scancel "$element" 2>/dev/null || true
                            abandon_collector_leases "${collector_jobs[$lane]}" "$lane"
                            collector_jobs[$lane]=$replacement
                            collector_partitions[$lane]=normal
                            missing_polls[$lane]=0
                            normal_pending_polls[$lane]=0
                            write_collector_jobs
                            echo "resubmitted_stale_collector slot=$lane job=$replacement partition=normal prior_element=$element"
                        else
                            echo "stale_collector_resubmit_failed slot=$lane; retaining element=$element" >&2
                        fi
                        continue
                    fi
                fi
            fi
            if [[ "${collector_partitions[$lane]}" == normal && "$job_state" == PENDING ]]; then
                normal_pending_polls[$lane]=$((normal_pending_polls[$lane] + 1))
                pending_seconds=$((normal_pending_polls[$lane] * poll_seconds))
                if (( pending_seconds >= normal_pending_grace_seconds )); then
                    echo "normal_pending_timeout slot=$lane element=$element seconds=$pending_seconds; fallback=preempt"
                    # Submit first so a transient preempt-QoS limit never
                    # destroys a valid pending normal request.
                    if replacement=$(submit_collector_lane preempt "$lane"); then
                        scancel "$element" 2>/dev/null || true
                        abandon_collector_leases "${collector_jobs[$lane]}" "$lane"
                        collector_jobs[$lane]=$replacement
                        collector_partitions[$lane]=preempt
                        normal_pending_polls[$lane]=0
                        write_collector_jobs
                        echo "resubmitted_collector slot=$lane job=$replacement partition=preempt prior_element=$element"
                    else
                        normal_pending_polls[$lane]=0
                        echo "preempt_fallback_submit_failed slot=$lane; retaining normal element=$element" >&2
                    fi
                fi
            else
                normal_pending_polls[$lane]=0
            fi
            continue
        fi
        if lane_complete "$lane"; then
            missing_polls[$lane]=0
            continue
        fi
        missing_polls[$lane]=$((missing_polls[$lane] + 1))
        # Two absent polls avoid racing Slurm's own --requeue transition.
        if (( missing_polls[$lane] < 2 )); then
            echo "collector_missing_once slot=$lane prior_element=$element"
            continue
        fi
        # A missing lane always retries normal first. If it remains pending,
        # the branch above moves only that lane to preempt after the grace
        # period. This avoids duplicate live collectors for one candidate lane.
        replacement=$(submit_collector_lane normal "$lane")
        abandon_collector_leases "${collector_jobs[$lane]}" "$lane"
        collector_jobs[$lane]=$replacement
        collector_partitions[$lane]=normal
        missing_polls[$lane]=0
        normal_pending_polls[$lane]=0
        write_collector_jobs
        echo "resubmitted_collector slot=$lane job=$replacement partition=normal prior_element=$element"
    done
    sleep "$poll_seconds"
done

cat "$window_dir/CLOSED.json"
"$summary_script" "$window_dir" \
    --json "$window_dir/ONLINE_POLICY_METRICS.json" \
    --markdown "$window_dir/ONLINE_POLICY_METRICS.md" \
    --run-json "$window_dir/../ONLINE_CHECKPOINT_COMPARISON.json" \
    --run-markdown "$window_dir/../ONLINE_CHECKPOINT_COMPARISON.md"
for job in "${collector_jobs[@]}"; do
    scancel "$job" 2>/dev/null || true
done
if [[ -s "$window_dir/OPPORTUNISTIC_COLLECTOR_JOBS.tsv" ]]; then
    while IFS=$'\t' read -r job _array_range _partition _placement _submitted_at; do
        [[ "$job" =~ ^[0-9]+$ ]] && scancel "$job" 2>/dev/null || true
    done < "$window_dir/OPPORTUNISTIC_COLLECTOR_JOBS.tsv"
fi
outcome=$("$python" - "$window_dir/CLOSED.json" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["outcome"])
PY
)
if [[ "$outcome" != ready ]]; then
    echo "Collection window closed without a trainable batch: $outcome" >&2
    exit 3
fi

"$script_dir/submit_spooled_window.sh" "$window_dir"
