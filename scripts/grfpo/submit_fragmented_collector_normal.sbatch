#!/bin/bash
#SBATCH --job-name=grfpo-frag-n
# Site partition/account/QoS are supplied by the controller.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
# Match the preempt launcher: 57 CPUs makes four collectors impossible on a
# 224-CPU DGX while retaining the validated three-collector / 3*vec8 topology.
#SBATCH --cpus-per-task=57
#SBATCH --mem=300G
#SBATCH --time=06:00:00
#SBATCH --requeue
# Node exclusions are supplied by the site contract at submission time.
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail

# Placement differs, but collection/retry/atomic-commit behavior is shared
# with the preempt-safe runtime below.
repo_root=${VVLA_REPO_ROOT:?VVLA_REPO_ROOT must be exported by the public launcher}
exec bash "$repo_root/scripts/grfpo/submit_fragmented_collector_preempt.sbatch"
