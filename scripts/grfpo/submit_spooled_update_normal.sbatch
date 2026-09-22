#!/usr/bin/env bash
#SBATCH --job-name=grfpo-spooled-train-n
# Site partition/account/QoS are supplied by the controller.
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=240G
#SBATCH --time=06:00:00
#SBATCH --requeue
# Node exclusions are supplied by the site contract at submission time.
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

# Keep the optimizer path identical across partitions; only placement differs.
repo_root=${VVLA_REPO_ROOT:?VVLA_REPO_ROOT must be exported by the public launcher}
exec bash "$repo_root/scripts/grfpo/submit_spooled_update_preempt.sbatch"
