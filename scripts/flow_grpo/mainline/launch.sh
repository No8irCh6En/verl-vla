#!/usr/bin/env bash
set -euo pipefail

# Flow-GRPO uses the same durable group-RL collection/training chain as GRFPO.
# This wrapper fixes only the actor objective; every collection/reward/reduction
# switch is accepted by the common launcher.
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../.." && pwd)
source "$repo_root/scripts/lib/runtime_env.sh"
vvla_require_fastwam_runtime
python=$VVLA_PYTHON
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$python" -m verl_vla.entrypoints.launch_grfpo_chain \
    "$@" \
    --algorithm flow_grpo
