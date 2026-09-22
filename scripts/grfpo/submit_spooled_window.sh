#!/usr/bin/env bash
set -euo pipefail

window_dir=${1:?usage: submit_spooled_window.sh COLLECTION_WINDOW}
window_dir=$(realpath "$window_dir")
manifest="$window_dir/manifest.json"
test -s "$manifest"

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../.." && pwd)
source "$repo_root/scripts/lib/runtime_env.sh"
python=$VVLA_PYTHON
normal_train_sbatch="$repo_root/scripts/grfpo/submit_spooled_update_normal.sbatch"
preempt_train_sbatch="$repo_root/scripts/grfpo/submit_spooled_update_preempt.sbatch"
normal_train_2gpu_sbatch="$repo_root/scripts/grfpo/submit_spooled_update_2gpu_normal.sbatch"
preempt_train_2gpu_sbatch="$repo_root/scripts/grfpo/submit_spooled_update_2gpu_preempt.sbatch"
max_attempts=${GRFPO_MAX_TRAIN_ATTEMPTS:-12}
# A full drained-window actor update takes roughly two hours and cannot yet
# resume within an epoch.  Two preempt attempts were killed late in epoch 2,
# losing more than three hours of compute.  Prefer a stable normal slot for
# actor training; preempt remains useful for resumable rollout/eval work.
# Explicit launchers may still override this grace period.
normal_grace_seconds=${GRFPO_NORMAL_PENDING_GRACE_SECONDS:-21600}

exec 9>"$window_dir/.train-submit.lock"
flock 9
if [[ -s "$window_dir/TRAINED.json" ]]; then
    echo "training_complete=$window_dir/TRAINED.json"
    exit 0
fi

prior_job=""
prior_attempt=0
prior_partition=""
prior_submitted_at=0
submit_partition=normal
if [[ -s "$window_dir/TRAIN_SUBMITTED.json" ]]; then
    readarray -t prior < <("$python" - "$window_dir/TRAIN_SUBMITTED.json" <<'PY'
import json
import sys

record = json.load(open(sys.argv[1], encoding="utf-8"))
print(record.get("job_id", ""))
print(record.get("attempt", 1))
print(record.get("partition", "preempt"))
print(record.get("submitted_at_unix_s", 0))
PY
    )
    prior_job=${prior[0]}
    prior_attempt=${prior[1]}
    prior_partition=${prior[2]}
    prior_submitted_at=${prior[3]}
    if [[ -n "$prior_job" ]]; then
        prior_state=$(squeue -h -j "$prior_job" -o '%T' 2>/dev/null | head -n 1 || true)
        if [[ -n "$prior_state" ]]; then
            if [[ "$prior_state" == PENDING && "$prior_partition" == normal ]]; then
                pending_seconds=$($python - "$prior_submitted_at" <<'PY'
import sys
import time
print(max(0, int(time.time() - float(sys.argv[1]))))
PY
                )
                if (( pending_seconds >= normal_grace_seconds )); then
                    echo "normal training job=$prior_job pending ${pending_seconds}s; fallback=preempt"
                    scancel "$prior_job" 2>/dev/null || true
                    submit_partition=preempt
                else
                    echo "training_active_job=$prior_job partition=normal state=$prior_state attempt=$prior_attempt"
                    exit 0
                fi
            else
                echo "training_active_job=$prior_job partition=$prior_partition state=$prior_state attempt=$prior_attempt"
                exit 0
            fi
        elif [[ "$prior_partition" == normal ]]; then
            submit_partition=preempt
        else
            # A preempted/failed attempt is followed by a fresh normal-first try.
            submit_partition=normal
        fi
    fi
fi

attempt=$((prior_attempt + 1))
if (( attempt > max_attempts )); then
    echo "Training retry budget exhausted: attempt=$attempt max=$max_attempts window=$window_dir" >&2
    exit 4
fi

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
# Slurm --export uses commas as field separators.  Use a colon transport
# encoding here and reconstruct Hydra list syntax inside the batch job.
print(":".join(record["task_names"]))
print(":".join(str(value) for value in record["suite_ids"]))
print(record["condition_manifest_sha256"])
print(len(record["task_names"]) * len(record["suite_ids"]))
print(record.get("group_reward_source", "binary_success"))
print(record.get("informative_group_criterion", "full_success_mixed"))
print(record.get("accepted_group_reduction", "group_equal"))
print(record.get("policy_seed_namespace", ""))
print(record.get("actor_objective", "fpo"))
print(record.get("flow_grpo_noise_level", 0.01))
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
task_names_colon=${fields[11]}
suite_ids_colon=${fields[12]}
condition_manifest_sha256=${fields[13]}
collector_count=${fields[14]}
group_reward_source=${fields[15]}
informative_group_criterion=${fields[16]}
accepted_group_reduction=${fields[17]}
policy_seed_namespace=${fields[18]}
actor_objective=${fields[19]}
flow_grpo_noise_level=${fields[20]}
collection_root=$(dirname "$(dirname "$window_dir")")
output_root=$(dirname "$collection_root")
log_root=${GRFPO_LOG_ROOT:-$VVLA_LOG_ROOT}
mkdir -p "$log_root"

# The controller is intentionally long-lived, so its inherited environment
# cannot be changed after submission.  A validated topology marker lets a
# running collection/training chain adopt a faster actor topology at the next
# closed-window boundary without restarting collection or changing theta_old.
topology_file="$output_root/TRAINING_TOPOLOGY.json"
marker_train_gpus=""
marker_policy_microbatch=""
marker_fsdp_size=""
if [[ -s "$topology_file" ]]; then
    readarray -t topology_fields < <("$python" - "$topology_file" <<'PY'
import json
import sys

record = json.load(open(sys.argv[1], encoding="utf-8"))
gpus = int(record["gpus_per_node"])
microbatch = int(record["policy_microbatch"])
fsdp_size = int(record.get("fsdp_size", -1))
if gpus not in (1, 2):
    raise ValueError(f"gpus_per_node must be 1 or 2, got {gpus}")
if microbatch <= 0:
    raise ValueError(f"policy_microbatch must be positive, got {microbatch}")
if fsdp_size != -1 and (fsdp_size <= 0 or fsdp_size > gpus or gpus % fsdp_size):
    raise ValueError(f"fsdp_size must be -1 or divide gpus_per_node, got {fsdp_size}")
print(gpus)
print(microbatch)
print(fsdp_size)
PY
    )
    marker_train_gpus=${topology_fields[0]}
    marker_policy_microbatch=${topology_fields[1]}
    marker_fsdp_size=${topology_fields[2]}
fi
train_gpus_per_node=${GRFPO_TRAIN_GPUS_PER_NODE:-${marker_train_gpus:-1}}
train_policy_microbatch=${GRFPO_TRAIN_POLICY_MICROBATCH:-${marker_policy_microbatch:-1}}
train_fsdp_size=${GRFPO_FSDP_SIZE:-${marker_fsdp_size:--1}}
if [[ "$train_gpus_per_node" != 1 && "$train_gpus_per_node" != 2 ]]; then
    echo "GRFPO_TRAIN_GPUS_PER_NODE must be 1 or 2, got $train_gpus_per_node" >&2
    exit 6
fi
if [[ ! "$train_policy_microbatch" =~ ^[1-9][0-9]*$ ]]; then
    echo "GRFPO_TRAIN_POLICY_MICROBATCH must be positive, got $train_policy_microbatch" >&2
    exit 6
fi
if [[ ! "$train_fsdp_size" =~ ^-?[0-9]+$ ]] || (( train_fsdp_size == 0 || train_fsdp_size < -1 )); then
    echo "GRFPO_FSDP_SIZE must be -1 or a positive integer, got $train_fsdp_size" >&2
    exit 6
fi
if (( train_fsdp_size > 0 )) && (( train_fsdp_size > train_gpus_per_node || train_gpus_per_node % train_fsdp_size != 0 )); then
    echo "GRFPO_FSDP_SIZE=$train_fsdp_size must divide gpus_per_node=$train_gpus_per_node" >&2
    exit 6
fi

# Trust-region settings can be selected after a frozen checkpoint evaluation
# without changing the already-collected theta_old batch.  The marker is
# durable and is read only at a closed-window training boundary.  Environment
# overrides remain available for an explicitly launched one-off job.
trust_region_file="$output_root/TRAINING_TRUST_REGION.json"
marker_actor_lr=""
marker_kl_mode=""
marker_target_kl=""
if [[ -s "$trust_region_file" ]]; then
    readarray -t trust_region_fields < <("$python" - "$trust_region_file" <<'PY'
import json
import math
import sys

record = json.load(open(sys.argv[1], encoding="utf-8"))
actor_lr = float(record["actor_lr"])
target_kl = float(record["target_kl"])
mode = str(record["kl_early_stop_mode"])
if not math.isfinite(actor_lr) or actor_lr <= 0:
    raise ValueError(f"actor_lr must be finite and positive, got {actor_lr}")
if not math.isfinite(target_kl) or target_kl <= 0:
    raise ValueError(f"target_kl must be finite and positive, got {target_kl}")
if mode not in {"post_epoch", "pre_optimizer"}:
    raise ValueError(f"unsupported kl_early_stop_mode={mode!r}")
print(actor_lr)
print(mode)
print(target_kl)
PY
    )
    marker_actor_lr=${trust_region_fields[0]}
    marker_kl_mode=${trust_region_fields[1]}
    marker_target_kl=${trust_region_fields[2]}
fi
actor_lr=${GRFPO_ACTOR_LR:-${marker_actor_lr:-1.0e-5}}
kl_early_stop_mode=${GRFPO_KL_EARLY_STOP_MODE:-${marker_kl_mode:-post_epoch}}
target_kl=${GRFPO_TARGET_KL:-${marker_target_kl:-0.1}}
fixed_mc_seed=${GRFPO_FIXED_MC_SEED:-}
fixed_shuffle_seed=${GRFPO_FIXED_SHUFFLE_SEED:-}
for seed_name in fixed_mc_seed fixed_shuffle_seed; do
    seed_value=${!seed_name}
    if [[ -n "$seed_value" && ! "$seed_value" =~ ^[0-9]+$ ]]; then
        echo "$seed_name must be empty or a non-negative integer, got $seed_value" >&2
        exit 6
    fi
done

submit_with_retry() {
    local attempt output
    for attempt in 1 2 3 4 5; do
        # Keep Slurm's transient diagnostics on stderr. Only stdout belongs in
        # the durable job-id field; sbatch may warn and still ultimately
        # succeed after its own controller retry.
        if output=$(sbatch --parsable "$@"); then
            printf '%s\n' "$output"
            return 0
        fi
        echo "training_sbatch_transient_failure attempt=$attempt/5 error=$output" >&2
        sleep $((attempt * 10))
    done
    return 1
}

case "$submit_partition:$train_gpus_per_node" in
    normal:1) train_sbatch=$normal_train_sbatch ;;
    preempt:1) train_sbatch=$preempt_train_sbatch ;;
    normal:2) train_sbatch=$normal_train_2gpu_sbatch ;;
    preempt:2) train_sbatch=$preempt_train_2gpu_sbatch ;;
    *) echo "unsupported training partition=$submit_partition" >&2; exit 5 ;;
esac

vvla_sbatch_site_args "$submit_partition" "$log_root/fragmented_train_%j.out"

train_job=$(submit_with_retry \
    "${VVLA_SBATCH_SITE_ARGS[@]}" \
    --export=ALL,GRFPO_COLLECTION_ROOT="$collection_root",GRFPO_RUN_ID="$run_id",GRFPO_INTENDED_UPDATE="$intended_update",GRFPO_THETA_OLD="$theta_old",GRFPO_TARGET_GROUPS="$target_groups",GRFPO_MIN_GROUPS="$minimum_groups",GRFPO_MAX_CANDIDATES="$max_candidates",GRFPO_COMPLETE_ROUND_AFTER_TARGET="$complete_round_after_target",GRFPO_DRAIN_CANDIDATE_BUDGET="$drain_candidate_budget",GRFPO_MODEL_ROOT="$model_root",GRFPO_MODEL_SHA256="$model_sha256",GRFPO_ACTOR_CHECKPOINT="$actor_checkpoint",GRFPO_OUTPUT_ROOT="$output_root",GRFPO_LOG_ROOT="$log_root",GRFPO_TASK_NAMES_COLON="$task_names_colon",GRFPO_SUITE_IDS_COLON="$suite_ids_colon",GRFPO_CONDITION_MANIFEST_SHA256="$condition_manifest_sha256",GRFPO_COLLECTOR_COUNT="$collector_count",GRFPO_POLICY_MICROBATCH="$train_policy_microbatch",GRFPO_TRAIN_GPUS_PER_NODE="$train_gpus_per_node",GRFPO_FSDP_SIZE="$train_fsdp_size",GRFPO_GROUP_REWARD_SOURCE="$group_reward_source",GRFPO_INFORMATIVE_GROUP_CRITERION="$informative_group_criterion",GRFPO_ACCEPTED_GROUP_REDUCTION="$accepted_group_reduction",GRFPO_POLICY_SEED_NAMESPACE="$policy_seed_namespace",GRFPO_ACTOR_OBJECTIVE="$actor_objective",GRFPO_FLOW_GRPO_NOISE_LEVEL="$flow_grpo_noise_level",GRFPO_FLOW_GRPO_TRANSITION_BATCH_SIZE="${GRFPO_FLOW_GRPO_TRANSITION_BATCH_SIZE:-3}",GRFPO_ACTOR_LR="$actor_lr",GRFPO_KL_EARLY_STOP_MODE="$kl_early_stop_mode",GRFPO_TARGET_KL="$target_kl",GRFPO_FIXED_MC_SEED="$fixed_mc_seed",GRFPO_FIXED_SHUFFLE_SEED="$fixed_shuffle_seed" \
    "$train_sbatch")

"$python" - "$window_dir/TRAIN_SUBMITTED.json" "$train_job" "$attempt" "$prior_job" "$submit_partition" "$train_gpus_per_node" "$train_policy_microbatch" "$train_fsdp_size" "$actor_lr" "$kl_early_stop_mode" "$target_kl" "$fixed_mc_seed" "$fixed_shuffle_seed" <<'PY'
import json
import os
import sys
import time
from pathlib import Path

path = Path(sys.argv[1])
record = {
    "job_id": sys.argv[2],
    "attempt": int(sys.argv[3]),
    "prior_job_id": sys.argv[4] or None,
    "partition": sys.argv[5],
    "gpus_per_node": int(sys.argv[6]),
    "policy_microbatch": int(sys.argv[7]),
    "fsdp_size": int(sys.argv[8]),
    "actor_lr": float(sys.argv[9]),
    "kl_early_stop_mode": sys.argv[10],
    "target_kl": float(sys.argv[11]),
    "fpo_mc_seed": int(sys.argv[12]) if sys.argv[12] else None,
    "fpo_shuffle_seed": int(sys.argv[13]) if sys.argv[13] else None,
    "submitted_at_unix_s": time.time(),
}
temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, path)
PY
echo "submitted_train_job=$train_job partition=$submit_partition gpus=$train_gpus_per_node fsdp_size=$train_fsdp_size microbatch=$train_policy_microbatch actor_lr=$actor_lr kl_mode=$kl_early_stop_mode target_kl=$target_kl mc_seed=${fixed_mc_seed:-random} shuffle_seed=${fixed_shuffle_seed:-random} attempt=$attempt prior_job=$prior_job"
