#!/usr/bin/env bash
set -euo pipefail

# Sole user-facing GRFPO++ rollout+train launcher.  Repository-local paths are
# derived from this file so the checkout can live anywhere.  VVLA_PYTHON is an
# explicit override for clusters that do not activate the desired environment
# before invoking the launcher.
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../.." && pwd)
source "$repo_root/scripts/lib/runtime_env.sh"
vvla_require_fastwam_runtime
python=$VVLA_PYTHON
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$python" -m verl_vla.entrypoints.launch_grfpo_chain "$@"
