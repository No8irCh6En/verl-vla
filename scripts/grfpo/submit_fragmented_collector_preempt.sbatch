#!/usr/bin/env bash
#SBATCH --job-name=grfpo-g8-collect
# Site partition/account/QoS are supplied by the controller.
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
# The H800 nodes expose 224 schedulable CPUs. Reserving 57 per collector makes
# four colocated jobs impossible (4*57 > 224) while retaining the validated
# three-collector / 3*vec8 topology. Memory alone did not enforce this on the
# 2-TiB nodes: four or five colocated Isaac renderers repeatedly crashed.
#SBATCH --cpus-per-task=57
#SBATCH --mem=300G
#SBATCH --time=04:00:00
#SBATCH --requeue
# Node exclusions are supplied by the site contract at submission time.
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail
repo_root=${VVLA_REPO_ROOT:?VVLA_REPO_ROOT must be exported by the public launcher}
script_dir="$repo_root/scripts/grfpo"
source "$repo_root/scripts/lib/runtime_env.sh"
vvla_load_site_module

log_root=${GRFPO_LOG_ROOT:?GRFPO_LOG_ROOT must be set}
python=$VVLA_PYTHON
array_task_id=${SLURM_ARRAY_TASK_ID:?This job must be submitted as an array task}
completion_marker="$log_root/.collector_${SLURM_ARRAY_JOB_ID}_${array_task_id}.completed.json"
rm -f -- "$completion_marker"

# Actual Slurm preemption is handled by --requeue.  This inner loop handles
# transient Isaac/RoboDojo startup failures, including failure paths that
# incorrectly terminate Python with exit status zero.  Committed groups live in
# the shared collection window and are skipped automatically after a restart.
attempt=0
max_attempts=${GRFPO_COLLECTOR_STARTUP_ATTEMPTS:-8}
shared_queue=${GRFPO_SHARED_CANDIDATE_QUEUE:-0}
steal_any=${GRFPO_SHARED_QUEUE_STEAL_ANY:-0}
# The Python workflow now holds a node-local flock only while constructing the
# Isaac application, so jobs on different nodes can start immediately while
# renderers on one DGX cannot collide in Vulkan initialization. Retain this as
# an optional scheduler-wide backoff for diagnosis, disabled by default.
initial_stagger_s=${GRFPO_COLLECTOR_INITIAL_STAGGER_S:-0}
if (( ${SLURM_RESTART_COUNT:-0} == 0 && array_task_id > 0 && initial_stagger_s > 0 )); then
    delay_s=$((array_task_id * initial_stagger_s))
    echo "collector_initial_stagger slot=$array_task_id delay_s=$delay_s"
    sleep "$delay_s"
fi
while (( attempt < max_attempts )); do
    attempt=$((attempt + 1))
    rm -f -- "$completion_marker"
    echo "collector_supervisor slot=$array_task_id attempt=$attempt/$max_attempts restart_count=${SLURM_RESTART_COUNT:-0}"
    set +e
    GRFPO_COLLECTOR_COMPLETION_MARKER="$completion_marker" \
        GRFPO_SHARED_QUEUE_STEAL_ANY="$steal_any" \
        "$script_dir/run_fragmented_candidate.sh"
    status=$?
    set -e
    if [[ -s "$completion_marker" ]]; then
        if [[ "$shared_queue" == 1 ]]; then
            completion_status=$($python - "$completion_marker" <<'PY'
import json
import sys

print(json.load(open(sys.argv[1], encoding="utf-8"))["status"])
PY
            )
            if [[ "$completion_status" == condition_drained ]]; then
                # The expensive warm process exhausted one task/suite. Keep
                # this GPU allocation and start a clean Isaac subprocess that
                # can atomically steal any remaining condition.
                echo "collector_condition_drained slot=$array_task_id switching_to_shared_queue_steal_any=1"
                steal_any=1
                attempt=0
                rm -f -- "$completion_marker"
                continue
            fi
        fi
        echo "collector_completed_normally slot=$array_task_id marker=$completion_marker"
        exit 0
    fi
    echo "collector_exited_without_completion slot=$array_task_id status=$status attempt=$attempt" >&2
    sleep $((15 * attempt))
done

echo "collector exhausted $max_attempts local recovery attempts" >&2
exit 42
