#!/usr/bin/env bash
#SBATCH --job-name=grfpo-chain
# Site partition/account/QoS are supplied by the public launcher.
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
repo_root=${VVLA_REPO_ROOT:?VVLA_REPO_ROOT must be exported by the public launcher}
exec "$repo_root/scripts/grfpo/run_fragmented_chain.sh"
