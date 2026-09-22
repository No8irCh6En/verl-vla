#!/usr/bin/env bash
# Shared runtime contract for supported verl-vla launchers.
#
# Repository-local paths are derived from this file. External repositories,
# models, data, and cluster settings are supplied by a site file or the caller.
# This file may be sourced repeatedly by controllers and Slurm workers.

# Environment variables cross process and Slurm job boundaries; shell
# functions do not.  An inherited marker alone therefore cannot prove that
# this shell has loaded the runtime helpers.  Guard on a function that is
# defined by this file instead.
if declare -F vvla_require_fastwam_runtime >/dev/null 2>&1; then
    return 0 2>/dev/null || exit 0
fi
export _VVLA_RUNTIME_ENV_LOADED=1

_vvla_lib_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export VVLA_REPO_ROOT="${VVLA_REPO_ROOT:-$(cd -- "$_vvla_lib_dir/../.." && pwd)}"
export VVLA_PYTHON="${VVLA_PYTHON:-$(command -v python3 || true)}"
export VVLA_OUTPUT_ROOT="${VVLA_OUTPUT_ROOT:-${VVLA_REPO_ROOT}/outputs}"
export VVLA_LOG_ROOT="${VVLA_LOG_ROOT:-${VVLA_OUTPUT_ROOT}/logs}"
export HF_HOME="${HF_HOME:-${XDG_CACHE_HOME:-${HOME}/.cache}/huggingface}"
export VVLA_PYTHON_EXTRA_PATHS="${VVLA_PYTHON_EXTRA_PATHS:-}"

if [[ -n "${FASTWAM_POLICY_ROOT:-}" ]]; then
    export FASTWAM_ROOT="${FASTWAM_ROOT:-${FASTWAM_POLICY_ROOT}/FastWAM}"
fi
if [[ -n "${VVLA_NATIVE_LIB_DIR:-}" ]]; then
    export LD_LIBRARY_PATH="${VVLA_NATIVE_LIB_DIR}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# Generic Slurm defaults. A cluster-specific site file should override these.
export VVLA_SLURM_NORMAL_PARTITION="${VVLA_SLURM_NORMAL_PARTITION:-normal}"
export VVLA_SLURM_PREEMPT_PARTITION="${VVLA_SLURM_PREEMPT_PARTITION:-preempt}"
export VVLA_SLURM_CPU_PARTITION="${VVLA_SLURM_CPU_PARTITION:-cpu}"
export VVLA_SLURM_NORMAL_QOS="${VVLA_SLURM_NORMAL_QOS:-}"
export VVLA_SLURM_PREEMPT_QOS="${VVLA_SLURM_PREEMPT_QOS:-}"
export VVLA_SLURM_CPU_QOS="${VVLA_SLURM_CPU_QOS:-}"
export VVLA_SLURM_MODULE="${VVLA_SLURM_MODULE:-}"
export SLURM_ACCOUNT="${SLURM_ACCOUNT:-}"
export SLURM_EXCLUDE="${SLURM_EXCLUDE:-}"

vvla_require_value() {
    local name=$1 value=${!1:-}
    if [[ -z "$value" ]]; then
        echo "Required runtime value is unset: $name" >&2
        return 2
    fi
}

vvla_require_dir() {
    local name=$1 value=${!1:-}
    vvla_require_value "$name" || return
    if [[ ! -d "$value" ]]; then
        echo "Required directory does not exist: $name=$value" >&2
        return 2
    fi
}

vvla_require_file() {
    local name=$1 value=${!1:-}
    vvla_require_value "$name" || return
    if [[ ! -f "$value" ]]; then
        echo "Required file does not exist: $name=$value" >&2
        return 2
    fi
}

vvla_require_python() {
    vvla_require_value VVLA_PYTHON || return
    if [[ ! -x "$VVLA_PYTHON" ]]; then
        echo "VVLA_PYTHON is not executable: $VVLA_PYTHON" >&2
        return 2
    fi
}

vvla_require_fastwam_runtime() {
    vvla_require_python || return
    vvla_require_dir ROBODOJO_ROOT || return
    vvla_require_dir FASTWAM_POLICY_ROOT || return
    vvla_require_dir FASTWAM_ROOT || return
    mkdir -p "$VVLA_OUTPUT_ROOT" "$VVLA_LOG_ROOT" "$HF_HOME"
}

vvla_check_fastwam_python() {
    vvla_require_fastwam_runtime || return
    "$VVLA_PYTHON" "$VVLA_REPO_ROOT/scripts/lib/check_runtime.py" \
        --repo-root "$VVLA_REPO_ROOT" \
        --robodojo-root "$ROBODOJO_ROOT" \
        --policy-root "$FASTWAM_POLICY_ROOT"
}

vvla_load_site_module() {
    if [[ -n "$VVLA_SLURM_MODULE" ]]; then
        if ! command -v module >/dev/null 2>&1; then
            echo "VVLA_SLURM_MODULE is set but the module command is unavailable" >&2
            return 2
        fi
        module load "$VVLA_SLURM_MODULE"
    fi
}

vvla_runtime_pythonpath() {
    # runtime_site appends optional pure-Python fallbacks after the primary
    # interpreter paths. Putting Python-3.10 site-packages directly in
    # PYTHONPATH would prepend their binary numpy/torch wheels and break the
    # Python-3.11 RoboDojo runtime.
    local value="$VVLA_REPO_ROOT/scripts/lib/runtime_site:$VVLA_REPO_ROOT/src:$ROBODOJO_ROOT:$FASTWAM_ROOT/src"
    printf '%s\n' "$value"
}

vvla_sbatch_site_args() {
    local role=$1 output=${2:-} partition qos
    case "$role" in
        normal) partition=$VVLA_SLURM_NORMAL_PARTITION; qos=$VVLA_SLURM_NORMAL_QOS ;;
        preempt) partition=$VVLA_SLURM_PREEMPT_PARTITION; qos=$VVLA_SLURM_PREEMPT_QOS ;;
        cpu) partition=$VVLA_SLURM_CPU_PARTITION; qos=$VVLA_SLURM_CPU_QOS ;;
        *) echo "Unknown Slurm role: $role" >&2; return 2 ;;
    esac
    VVLA_SBATCH_SITE_ARGS=("--partition=$partition")
    [[ -n "$SLURM_ACCOUNT" ]] && VVLA_SBATCH_SITE_ARGS+=("--account=$SLURM_ACCOUNT")
    [[ -n "$qos" ]] && VVLA_SBATCH_SITE_ARGS+=("--qos=$qos")
    [[ -n "$SLURM_EXCLUDE" && "$role" != cpu ]] && VVLA_SBATCH_SITE_ARGS+=("--exclude=$SLURM_EXCLUDE")
    [[ -n "$output" ]] && VVLA_SBATCH_SITE_ARGS+=("--output=$output")
}
