"""Submit one validated fragmented GRFPO rollout-and-train chain."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from verl_vla.trainer.grfpo.launch_config import GRFPOChainLaunchConfig

REPO_ROOT = Path(__file__).resolve().parents[3]


def _required_actor_files(actor_dir: Path) -> tuple[str, ...]:
    """Return the complete rank-local file set declared by a checkpoint.

    HSDP runs use an FSDP mesh of size one replicated over multiple actor
    ranks.  Those checkpoints are valid continuation points and must not be
    rejected merely because their filenames contain ``world_size_2``.
    """

    config_path = actor_dir / "fsdp_config.json"
    if not config_path.is_file():
        return ("fsdp_config.json",)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    world_size = int(config["world_size"])
    if world_size <= 0:
        raise ValueError(f"Invalid checkpoint world_size={world_size}: {config_path}")
    rank_files = tuple(
        f"{kind}_world_size_{world_size}_rank_{rank}.pt"
        for rank in range(world_size)
        for kind in ("model", "optim", "extra_state")
    )
    return ("fsdp_config.json", *rank_files)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Unified durable GRFPO++/Flow-GRPO collection and training launcher. "
            "Fixed-budget collection is the default; --early-stop is the only "
            "way to enable target-based stopping."
        )
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--algorithm",
        choices=("grfpo", "flow_grpo"),
        default="grfpo",
        help=(
            "Actor objective only. Collection, group rewards, task reduction, "
            "queueing, and simulator topology remain shared."
        ),
    )
    parser.add_argument("--start-theta", type=int, default=0)
    parser.add_argument(
        "--start-checkpoint",
        type=Path,
        help=(
            "Full trainer checkpoint named global_step_<start-theta>. Required when "
            "starting a new chain from theta > 0; it is linked, not copied."
        ),
    )
    parser.add_argument("--updates", type=int, default=8)
    parser.add_argument("--minimum-accepted-groups", type=int, default=32)
    parser.add_argument("--candidate-groups", type=int, default=96)
    parser.add_argument("--early-stop", action="store_true")
    parser.add_argument("--process-score", action="store_true")
    parser.add_argument("--task-equal", action="store_true")
    parser.add_argument("--collector-count", type=int, default=12)
    parser.add_argument("--collector-concurrency", type=int, default=8)
    parser.add_argument("--opportunistic-preempt-collectors", type=int, default=4)
    parser.add_argument(
        "--simulators-per-gpu",
        type=int,
        choices=(1, 2, 3),
        default=1,
        help=(
            "Independent vec8 RoboDojo processes served by one shared Fast-WAM replica. "
            "Values 2 and 3 require their cluster smoke tests; group boundaries remain G=8."
        ),
    )
    parser.add_argument("--excluded-task-names", default="classify_objects")
    parser.add_argument("--policy-seed-namespace", default="")
    parser.add_argument("--train-gpus-per-node", type=int, default=1)
    parser.add_argument("--train-policy-microbatch", type=int, default=1)
    parser.add_argument(
        "--train-fsdp-size",
        type=int,
        default=-1,
        help="FSDP shard group size; use 1 for two-rank HSDP checkpoints.",
    )
    parser.add_argument("--actor-lr", type=float, default=1.0e-5)
    parser.add_argument(
        "--kl-early-stop-mode",
        choices=("post_epoch", "pre_optimizer"),
        default="post_epoch",
    )
    parser.add_argument("--target-kl", type=float, default=0.1)
    parser.add_argument("--flow-grpo-noise-level", type=float, default=0.01)
    parser.add_argument("--flow-grpo-transition-batch-size", type=int, default=3)
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument(
        "--selection",
        type=Path,
        default=os.environ.get("VVLA_MULTITASK_SFT_SELECTION"),
        help="SFT checkpoint-selection receipt; relative paths use the launch cwd.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=os.environ.get("VVLA_OUTPUT_ROOT"),
        help=(
            "Artifact root containing the grfpo/ or flow_grpo/ namespace. "
            "Defaults to VVLA_OUTPUT_ROOT."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--robodojo-root",
        type=Path,
        default=os.environ.get("ROBODOJO_ROOT"),
        help="RoboDojo checkout (or ROBODOJO_ROOT).",
    )
    parser.add_argument(
        "--policy-root",
        type=Path,
        default=os.environ.get("FASTWAM_POLICY_ROOT"),
        help="Fast-WAM integration root (or FASTWAM_POLICY_ROOT).",
    )
    parser.add_argument(
        "--hf-home",
        type=Path,
        default=os.environ.get("HF_HOME"),
        help="Hugging Face cache root (or HF_HOME).",
    )
    parser.add_argument(
        "--native-lib-dir",
        type=Path,
        default=os.environ.get("VVLA_NATIVE_LIB_DIR"),
        help="Optional native-library directory prepended by worker launchers.",
    )
    parser.add_argument(
        "--condition-manifest",
        type=Path,
        default=REPO_ROOT / "configs/eval/fastwam/robodojo_condition_split_v2_expansion.json",
        help="Predeclared RoboDojo condition schedule.",
    )
    return parser


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _check_runtime_python(robodojo_root: Path, policy_root: Path) -> None:
    """Fail before Slurm submission when the selected overlay is incomplete."""

    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/lib/check_runtime.py"),
            "--repo-root",
            str(REPO_ROOT),
            "--robodojo-root",
            str(robodojo_root),
            "--policy-root",
            str(policy_root),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )


def _checkpoint_provenance(start_theta: int, checkpoint: Path | None) -> dict | None:
    if start_theta == 0:
        if checkpoint is not None:
            raise ValueError("--start-checkpoint is invalid when --start-theta=0.")
        return None
    if checkpoint is None:
        raise ValueError("--start-checkpoint is required when --start-theta > 0.")

    resolved = checkpoint.expanduser().resolve(strict=True)
    expected_name = f"global_step_{start_theta}"
    if resolved.name != expected_name:
        raise ValueError(f"Start checkpoint must be named {expected_name!r}, got {resolved.name!r}.")
    actor_dir = resolved / "actor"
    required_actor_files = _required_actor_files(actor_dir)
    missing = [name for name in required_actor_files if not (actor_dir / name).is_file()]
    if missing:
        raise ValueError(f"Start checkpoint is incomplete; missing actor files: {missing}")
    return {
        "policy_version": start_theta,
        "source_path": str(resolved),
        "transfer": "symbolic_link",
        "actor_files": {name: {"size_bytes": (actor_dir / name).stat().st_size} for name in required_actor_files},
    }


def _stage_start_checkpoint(output_root: Path, provenance: dict | None) -> None:
    if provenance is None:
        return
    destination = output_root / "checkpoints" / f"global_step_{provenance['policy_version']}"
    source = Path(provenance["source_path"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve(strict=True) != source:
            raise RuntimeError(f"Existing start-checkpoint link points elsewhere: {destination}")
        return
    if destination.exists():
        raise RuntimeError(f"Refusing to replace existing start checkpoint: {destination}")
    destination.symlink_to(source, target_is_directory=True)


def main() -> None:
    args = _parser().parse_args()
    if args.output_root is None:
        raise ValueError("--output-root or VVLA_OUTPUT_ROOT is required.")
    if args.selection is None:
        raise ValueError("--selection or VVLA_MULTITASK_SFT_SELECTION is required.")
    for name in ("robodojo_root", "policy_root", "hf_home"):
        if getattr(args, name) is None:
            raise ValueError(f"--{name.replace('_', '-')} or its site environment variable is required.")
    selection_path = args.selection.expanduser().resolve(strict=True)
    output_base = args.output_root.expanduser().resolve()
    robodojo_root = args.robodojo_root.expanduser().resolve(strict=True)
    policy_root = args.policy_root.expanduser().resolve(strict=True)
    hf_home = args.hf_home.expanduser().resolve(strict=True)
    condition_manifest = args.condition_manifest.expanduser().resolve(strict=True)
    native_lib_dir = (
        args.native_lib_dir.expanduser().resolve(strict=True) if args.native_lib_dir else None
    )
    _check_runtime_python(robodojo_root, policy_root)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    checkpoint_provenance = _checkpoint_provenance(args.start_theta, args.start_checkpoint)
    config = GRFPOChainLaunchConfig(
        run_id=args.run_id,
        algorithm=args.algorithm,
        start_theta=args.start_theta,
        updates=args.updates,
        minimum_accepted_groups=args.minimum_accepted_groups,
        candidate_groups=args.candidate_groups,
        early_stop=args.early_stop,
        process_score=args.process_score,
        task_equal=args.task_equal,
        collector_count=args.collector_count,
        collector_concurrency=args.collector_concurrency,
        opportunistic_preempt_collectors=args.opportunistic_preempt_collectors,
        simulators_per_gpu=args.simulators_per_gpu,
        excluded_task_names=args.excluded_task_names,
        policy_seed_namespace=args.policy_seed_namespace,
        train_gpus_per_node=args.train_gpus_per_node,
        train_policy_microbatch=args.train_policy_microbatch,
        train_fsdp_size=args.train_fsdp_size,
        actor_lr=args.actor_lr,
        kl_early_stop_mode=args.kl_early_stop_mode,
        target_kl=args.target_kl,
        flow_grpo_noise_level=args.flow_grpo_noise_level,
        flow_grpo_transition_batch_size=args.flow_grpo_transition_batch_size,
        continuous=args.continuous,
    )
    manifest = config.manifest(selection=selection)
    manifest["selection_receipt"] = str(selection_path)
    manifest["initial_trainer_checkpoint"] = checkpoint_provenance
    output_root = output_base / config.output_namespace / config.run_id
    manifest_path = output_root / "EXPERIMENT_MANIFEST.json"
    state_path = output_root / "CHAIN_STATE.json"
    if state_path.exists():
        raise SystemExit(f"Refusing to reuse an initialized run: {state_path}")

    env_values = config.runtime_environment()
    env_values.update(
        {
            "VVLA_REPO_ROOT": str(REPO_ROOT),
            # Preserve the environment entrypoint. Resolving an overlay
            # Python symlink can silently select its base environment.
            "VVLA_PYTHON": os.path.abspath(sys.executable),
            "VVLA_OUTPUT_ROOT": str(output_base),
            "VVLA_LOG_ROOT": str(output_root / "logs"),
            "MULTITASK_SFT_SELECTION_JSON": str(selection_path),
            "ROBODOJO_ROOT": str(robodojo_root),
            "FASTWAM_POLICY_ROOT": str(policy_root),
            "FASTWAM_ROOT": str((policy_root / "FastWAM").resolve(strict=True)),
            "HF_HOME": str(hf_home),
            "VVLA_CONDITION_MANIFEST": str(condition_manifest),
            "VVLA_NATIVE_LIB_DIR": str(native_lib_dir) if native_lib_dir else "",
            "VVLA_PYTHON_EXTRA_PATHS": os.environ.get("VVLA_PYTHON_EXTRA_PATHS", ""),
            "SLURM_ACCOUNT": os.environ.get("SLURM_ACCOUNT", ""),
            "VVLA_SLURM_NORMAL_PARTITION": os.environ.get(
                "VVLA_SLURM_NORMAL_PARTITION", "normal"
            ),
            "VVLA_SLURM_PREEMPT_PARTITION": os.environ.get(
                "VVLA_SLURM_PREEMPT_PARTITION", "preempt"
            ),
            "VVLA_SLURM_CPU_PARTITION": os.environ.get("VVLA_SLURM_CPU_PARTITION", "cpu"),
            "VVLA_SLURM_NORMAL_QOS": os.environ.get("VVLA_SLURM_NORMAL_QOS", ""),
            "VVLA_SLURM_PREEMPT_QOS": os.environ.get("VVLA_SLURM_PREEMPT_QOS", ""),
            "VVLA_SLURM_CPU_QOS": os.environ.get("VVLA_SLURM_CPU_QOS", ""),
            "VVLA_SLURM_MODULE": os.environ.get("VVLA_SLURM_MODULE", ""),
        }
    )
    invalid_export_values = {key: value for key, value in env_values.items() if "," in value}
    if invalid_export_values:
        raise ValueError(f"Slurm --export values cannot contain commas: {invalid_export_values}")
    export_spec = "ALL," + ",".join(f"{key}={value}" for key, value in env_values.items())
    manifest["resolved_runtime"] = {
        "repo_root": str(REPO_ROOT),
        "python_entrypoint": os.path.abspath(sys.executable),
        "output_root": str(output_base),
        "robodojo_root": str(robodojo_root),
        "fastwam_policy_root": str(policy_root),
        "hf_home": str(hf_home),
        "native_lib_dir": str(native_lib_dir) if native_lib_dir else None,
        "condition_manifest": str(condition_manifest),
        "slurm": {
            "account": env_values["SLURM_ACCOUNT"],
            "normal_partition": env_values["VVLA_SLURM_NORMAL_PARTITION"],
            "preempt_partition": env_values["VVLA_SLURM_PREEMPT_PARTITION"],
            "cpu_partition": env_values["VVLA_SLURM_CPU_PARTITION"],
            "normal_qos": env_values["VVLA_SLURM_NORMAL_QOS"],
            "preempt_qos": env_values["VVLA_SLURM_PREEMPT_QOS"],
            "cpu_qos": env_values["VVLA_SLURM_CPU_QOS"],
            "module": env_values["VVLA_SLURM_MODULE"],
            "exclude": os.environ.get("SLURM_EXCLUDE", ""),
        },
    }
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise SystemExit(f"Existing manifest differs from requested launch: {manifest_path}")
    command = ["sbatch", "--parsable"]
    cpu_partition = env_values["VVLA_SLURM_CPU_PARTITION"]
    command.append(f"--partition={cpu_partition}")
    if env_values["SLURM_ACCOUNT"]:
        command.append(f"--account={env_values['SLURM_ACCOUNT']}")
    if env_values["VVLA_SLURM_CPU_QOS"]:
        command.append(f"--qos={env_values['VVLA_SLURM_CPU_QOS']}")
    command.extend(
        [
            f"--output={output_root / 'logs' / 'fragmented_chain_%j.out'}",
            f"--export={export_spec}",
            str(REPO_ROOT / "scripts/grfpo/submit_fragmented_chain_cpu.sbatch"),
        ]
    )
    preview = {
        "manifest": manifest,
        "output_root": str(output_root),
        "submit_command": command,
    }
    if args.dry_run:
        print(json.dumps(preview, indent=2, sort_keys=True))
        return

    _atomic_json(manifest_path, manifest)
    (output_root / "logs").mkdir(parents=True, exist_ok=True)
    _stage_start_checkpoint(output_root, checkpoint_provenance)
    _atomic_json(
        output_root / "TRAINING_TRUST_REGION.json",
        {
            "schema_version": 1,
            "actor_lr": config.actor_lr,
            "kl_early_stop_mode": config.kl_early_stop_mode,
            "target_kl": config.target_kl,
            "written_by": "launch_grfpo_chain",
        },
    )
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    job_id = result.stdout.strip().split(";")[0]
    submission = {
        "job_id": job_id,
        "submitted_at_unix_s": time.time(),
        "argv": sys.argv,
        "manifest_path": str(manifest_path),
    }
    _atomic_json(output_root / "LAUNCH_SUBMISSION.json", submission)
    print(json.dumps(submission, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
