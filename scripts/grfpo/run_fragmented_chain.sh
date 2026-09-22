#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../.." && pwd)
source "$repo_root/scripts/lib/runtime_env.sh"
vvla_require_fastwam_runtime
selection=${MULTITASK_SFT_SELECTION_JSON:?MULTITASK_SFT_SELECTION_JSON must be exported by the launcher}
python=$VVLA_PYTHON
start_theta=${GRFPO_CHAIN_START_THETA:-0}
final_theta=${GRFPO_CHAIN_FINAL_THETA:-8}
continuous=${GRFPO_CHAIN_CONTINUOUS:-0}
poll_seconds=${GRFPO_CHAIN_POLL_SECONDS:-60}
minimum_free_gb=${GRFPO_CHAIN_MIN_FREE_GB:-100}

if [[ "$continuous" != 0 && "$continuous" != 1 ]]; then
    echo "GRFPO_CHAIN_CONTINUOUS must be 0 or 1, got $continuous" >&2
    exit 2
fi
if (( start_theta < 0 )) || { [[ "$continuous" == 0 ]] && (( final_theta <= start_theta )); }; then
    echo "Require 0 <= start_theta < final_theta, got $start_theta -> $final_theta" >&2
    exit 2
fi

selected_step=$($python - "$selection" <<'PY'
import json
import sys

print(json.load(open(sys.argv[1], encoding="utf-8"))["selected"]["step"])
PY
)
run_id=${MULTITASK_GRFPO_RUN_ID:-"multitask5_sft_c${selected_step}_grfpo_pp_allsuite_v2_m32_v1"}
output_namespace=${VVLA_GROUP_RL_OUTPUT_NAMESPACE:-grfpo}
if [[ "$output_namespace" != grfpo && "$output_namespace" != flow_grpo ]]; then
    echo "VVLA_GROUP_RL_OUTPUT_NAMESPACE must be grfpo or flow_grpo" >&2
    exit 2
fi
output_root="$VVLA_OUTPUT_ROOT/$output_namespace/$run_id"
collection_root="$output_root/candidate_spool/$run_id"
state_path="$output_root/CHAIN_STATE.json"
stop_path="$output_root/STOP_CHAIN"
mkdir -p "$output_root"

exec 9>"$output_root/.fragmented_chain.lock"
if ! flock -n 9; then
    echo "A fragmented GRFPO chain controller is already running for $run_id" >&2
    exit 2
fi

write_state() {
    local status=$1 theta=$2 detail=$3
    "$python" - "$state_path" "$status" "$theta" "$final_theta" "$detail" <<'PY'
import json
import os
import sys
import time
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "status": sys.argv[2],
    "current_theta": int(sys.argv[3]),
    "final_theta": int(sys.argv[4]),
    "detail": sys.argv[5],
    "updated_at_unix_s": time.time(),
}
temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, path)
PY
}

renew_controller() {
    # CPU jobs have a 12-hour wall-time.  A successor starts only after this
    # process exits and replays durable TRAINED receipts before resuming its
    # wait, so renewal cannot duplicate an optimizer update.
    # Fixed-length chains need the same renewal while the requested final
    # theta has not yet been reached.  Otherwise a slow collection window can
    # silently end an otherwise healthy theta_0 -> theta_N run at 12 hours.
    if { [[ "$continuous" == 1 ]] || (( theta < final_theta )); } \
        && [[ -n "${SLURM_JOB_ID:-}" ]]; then
        vvla_sbatch_site_args cpu "$VVLA_LOG_ROOT/fragmented_chain_%j.out"
        successor=$(sbatch --parsable \
            "${VVLA_SBATCH_SITE_ARGS[@]}" \
            --dependency="afterany:$SLURM_JOB_ID" \
            --export=ALL \
            "$repo_root/scripts/grfpo/submit_fragmented_chain_cpu.sbatch")
        write_state renewing "$theta" "controller wall-time renewal job=$successor"
        echo "$(date --iso-8601=seconds) controller_renewal successor=$successor theta=$theta"
        exit 0
    fi
    write_state stopped "$theta" "controller received wall-time signal without continuous mode"
    exit 5
}
trap renew_controller USR1

check_storage_before_launch() {
    local available_gb
    available_gb=$(df -BG --output=avail "$output_root" | tail -1 | tr -dc '0-9')
    if [[ -z "$available_gb" ]] || (( available_gb < minimum_free_gb )); then
        write_state stopped "$theta" \
            "storage guard: available=${available_gb:-unknown}GiB minimum=${minimum_free_gb}GiB"
        echo "Refusing to launch another update: available=${available_gb:-unknown}GiB " \
            "minimum=${minimum_free_gb}GiB" >&2
        exit 6
    fi
}

launch_update_with_retry() {
    local theta_old=$1
    local intended_update=$((theta_old + 1))
    while ! "$repo_root/scripts/grfpo/launch_fragmented_update.sh" "$theta_old"; do
        # QOSMaxSubmitJobPerUserLimit is common while an evaluation array is
        # draining. No collection window has been committed in that case, so
        # retain the CPU controller and retry without requiring an operator.
        write_state waiting "$theta_old" \
            "collector submission unavailable; retrying theta_${theta_old}->theta_${intended_update}"
        echo "$(date --iso-8601=seconds) collector_submit_retry theta_old=$theta_old" >&2
        sleep "$poll_seconds"
    done
}

theta=$start_theta
write_state waiting "$theta" "waiting for theta_$((theta + 1))"
while [[ "$continuous" == 1 ]] || (( theta < final_theta )); do
    if [[ -e "$stop_path" ]]; then
        write_state stopped "$theta" "operator stop marker present: $stop_path"
        echo "$(date --iso-8601=seconds) chain_stopped marker=$stop_path theta=$theta"
        exit 0
    fi
    intended_update=$((theta + 1))
    window="$collection_root/$(printf 'update_%04d_theta_%04d' "$intended_update" "$theta")"
    receipt="$window/TRAINED.json"
    if [[ ! -s "$receipt" && ! -s "$window/manifest.json" ]]; then
        check_storage_before_launch
        write_state launching "$theta" "launching theta_$theta -> theta_$intended_update"
        launch_update_with_retry "$theta"
        write_state waiting "$theta" "waiting for theta_$intended_update"
    fi
    echo "$(date --iso-8601=seconds) waiting theta_old=$theta -> theta_new=$intended_update receipt=$receipt"

    while [[ ! -s "$receipt" ]]; do
        if [[ -s "$window/CLOSED.json" ]]; then
            outcome=$($python - "$window/CLOSED.json" <<'PY'
import json
import sys

print(json.load(open(sys.argv[1], encoding="utf-8"))["outcome"])
PY
)
            if [[ "$outcome" != ready ]]; then
                write_state stopped "$theta" "collection closed with outcome=$outcome"
                echo "Cannot continue: $window closed with outcome=$outcome" >&2
                exit 3
            fi
            if ! "$repo_root/scripts/grfpo/submit_spooled_window.sh" "$window"; then
                # Slurm controller pressure is transient and the complete
                # collection window is durable. Keep the chain alive and retry
                # instead of turning a submission outage into a stopped run.
                write_state waiting "$theta" "training submission retry for theta_$intended_update"
                echo "$(date --iso-8601=seconds) training_submit_retry window=$window" >&2
                sleep "$poll_seconds"
                continue
            fi
        fi
        sleep "$poll_seconds"
    done

    checkpoint=$($python - "$receipt" "$intended_update" <<'PY'
import json
import sys

receipt = json.load(open(sys.argv[1], encoding="utf-8"))
expected = int(sys.argv[2])
if int(receipt["new_policy_version"]) != expected:
    raise SystemExit(
        f"TRAINED receipt version mismatch: {receipt['new_policy_version']} != {expected}"
    )
print(receipt["checkpoint_dir"])
PY
)
    if [[ ! -d "$checkpoint/actor" ]]; then
        write_state stopped "$theta" "missing durable actor checkpoint: $checkpoint/actor"
        echo "TRAINED exists but actor checkpoint is missing: $checkpoint/actor" >&2
        exit 4
    fi

    theta=$intended_update
    write_state trained "$theta" "durable checkpoint=$checkpoint"
    echo "$(date --iso-8601=seconds) theta=$theta is durable at $checkpoint"
    if [[ "$continuous" == 0 ]] && (( theta >= final_theta )); then
        break
    fi

    if [[ -e "$stop_path" ]]; then
        write_state stopped "$theta" "operator stop marker present: $stop_path"
        echo "$(date --iso-8601=seconds) chain_stopped marker=$stop_path theta=$theta"
        exit 0
    fi

    check_storage_before_launch
    write_state launching "$theta" "launching theta_$theta -> theta_$((theta + 1))"
    launch_update_with_retry "$theta"
    write_state waiting "$theta" "waiting for theta_$((theta + 1))"
done

write_state complete "$theta" "requested chain completed"
echo "$(date --iso-8601=seconds) chain_complete final_theta=$theta"
