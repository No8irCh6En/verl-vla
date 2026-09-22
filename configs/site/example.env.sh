#!/usr/bin/env bash
# Copy this file outside the repository, edit it for the target machine, then:
#   source /path/to/site.env.sh
#   /path/to/verl-vla/scripts/grfpo/mainline/launch.sh ...

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "Source this file; do not execute it." >&2
    exit 2
fi

# External source trees.
export ROBODOJO_ROOT=/path/to/RoboDojo
export FASTWAM_POLICY_ROOT="$ROBODOJO_ROOT/XPolicyLab/policy/FastWAM"
export FASTWAM_ROOT="$FASTWAM_POLICY_ROOT/FastWAM"

# The environment must contain verl-vla, RoboDojo/Isaac, and Fast-WAM runtime
# dependencies. Preserve the environment entrypoint rather than resolving its
# symlink to a base Python interpreter.
export VVLA_PYTHON=/path/to/python-environment/bin/python
export VVLA_NATIVE_LIB_DIR=/path/to/python-environment/lib

# Writable storage.
export VVLA_OUTPUT_ROOT=/path/to/artifacts/verl-vla
export VVLA_LOG_ROOT="$VVLA_OUTPUT_ROOT/logs"
export HF_HOME=/path/to/cache/huggingface

# Optional experiment defaults. The public launcher can also receive these as
# command-line arguments.
export VVLA_MULTITASK_SFT_SELECTION=/path/to/sft_selection.json

# Slurm deployment. Leave QoS/module/exclude empty when the target site does
# not use them.
export SLURM_ACCOUNT=your_account
export VVLA_SLURM_NORMAL_PARTITION=normal
export VVLA_SLURM_PREEMPT_PARTITION=preempt
export VVLA_SLURM_CPU_PARTITION=cpu
export VVLA_SLURM_NORMAL_QOS=
export VVLA_SLURM_PREEMPT_QOS=
export VVLA_SLURM_CPU_QOS=
export VVLA_SLURM_MODULE=
export SLURM_EXCLUDE=
