import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).parents[2]
SUPPORTED_RUNTIME_FILES = (
    "scripts/lib/runtime_env.sh",
    "scripts/lib/check_runtime.py",
    "scripts/grfpo/mainline/launch.sh",
    "scripts/grfpo/mainline/manage.sh",
    "scripts/flow_grpo/mainline/launch.sh",
    "scripts/grfpo/run_fragmented_chain.sh",
    "scripts/grfpo/launch_fragmented_update.sh",
    "scripts/grfpo/run_fragmented_candidate.sh",
    "scripts/grfpo/run_spooled_update.sh",
    "scripts/grfpo/submit_fragmented_chain_cpu.sbatch",
    "scripts/grfpo/submit_fragmented_collector_normal.sbatch",
    "scripts/grfpo/submit_fragmented_collector_preempt.sbatch",
    "scripts/grfpo/monitor_fragmented_window.sbatch",
    "scripts/grfpo/monitor_opportunistic_collectors.sbatch",
    "scripts/grfpo/add_opportunistic_collectors.sh",
    "scripts/grfpo/resume_fragmented_window.sh",
    "scripts/grfpo/submit_spooled_window.sh",
    "scripts/grfpo/submit_spooled_update_normal.sbatch",
    "scripts/grfpo/submit_spooled_update_preempt.sbatch",
    "scripts/grfpo/submit_spooled_update_2gpu_normal.sbatch",
    "scripts/grfpo/submit_spooled_update_2gpu_preempt.sbatch",
    "scripts/eval/mainline/eval.py",
    "scripts/eval/mainline/monitor_split_eval.py",
    "scripts/eval/mainline/launch_split_after_complete.sh",
    "scripts/eval/mainline/launch_when_checkpoint_ready.sh",
    "scripts/eval/mainline/register_and_launch_split_when_checkpoint_ready.sh",
    "scripts/eval/mainline/retry_single_gpu_packed_launch.sh",
    "src/verl_vla/entrypoints/launch_grfpo_chain.py",
    "src/verl_vla/workflows/config/grfpo_fragmented_collection.yaml",
    "src/verl_vla/workflows/config/train/grfpo_spooled.yaml",
)

SLURM_WORKER_FILES = tuple(
    relative for relative in SUPPORTED_RUNTIME_FILES if relative.endswith(".sbatch")
)


def test_supported_rollout_train_eval_runtime_has_no_personal_absolute_paths():
    forbidden = (
        "/project/peilab/cliang",
        "/home/qshou",
        "progress/phase6b/runtime_site",
    )
    violations = []
    for relative in SUPPORTED_RUNTIME_FILES:
        text = (REPO_ROOT / relative).read_text(encoding="utf-8")
        for token in forbidden:
            if token in text:
                violations.append((relative, token))
    assert violations == []


def test_slurm_workers_use_exported_repo_root_not_spooled_script_location():
    violations = []
    for relative in SLURM_WORKER_FILES:
        text = (REPO_ROOT / relative).read_text(encoding="utf-8")
        if "BASH_SOURCE" in text or "VVLA_REPO_ROOT" not in text:
            violations.append(relative)
    assert violations == []


def test_runtime_helpers_load_when_parent_marker_is_inherited():
    runtime = REPO_ROOT / "scripts/lib/runtime_env.sh"
    environment = os.environ.copy()
    environment["_VVLA_RUNTIME_ENV_LOADED"] = "1"
    result = subprocess.run(
        [
            "bash",
            "-c",
            f'source "{runtime}" && declare -F vvla_require_fastwam_runtime >/dev/null',
        ],
        env=environment,
        check=False,
    )
    assert result.returncode == 0
