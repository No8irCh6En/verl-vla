#!/usr/bin/env bash
#SBATCH --job-name=grfpo-spooled-train2-n
# Site partition/account/QoS are supplied by the controller.
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=48
#SBATCH --mem=360G
#SBATCH --time=06:00:00
#SBATCH --requeue
# Node exclusions are supplied by the site contract at submission time.
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
repo_root=${VVLA_REPO_ROOT:?VVLA_REPO_ROOT must be exported by the public launcher}
script_dir="$repo_root/scripts/grfpo"
source "$repo_root/scripts/lib/runtime_env.sh"
vvla_load_site_module
export GRFPO_TRAIN_GPUS_PER_NODE=2
exec "$script_dir/run_spooled_update.sh"
