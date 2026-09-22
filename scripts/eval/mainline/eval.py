#!/usr/bin/env python3
"""One public launcher for fixed-topology native Fast-WAM + RoboDojo evals.

The experiment spec owns policy identities, fixed conditions, repeats, and the
evaluator topology.  ``launch`` expands that spec into immutable work items and
submits a small number of durable Slurm lanes.  Lanes resume at item granularity
and invoke the existing native evaluator in a fresh process for every
policy/task/layout/repeat tuple.  ``summarize`` reports strict success at both
trajectory and layout level; individual flip labels are never promoted to a
checkpoint-level conclusion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from math import ceil
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
LABEL_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
TASK_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")


def _resolve_input_path(value: str | Path | None, name: str, *, directory: bool) -> Path:
    if value is None or not str(value).strip():
        raise ValueError(f"{name} is required.")
    path = Path(value).expanduser().resolve(strict=True)
    if directory and not path.is_dir():
        raise NotADirectoryError(f"{name} is not a directory: {path}")
    if not directory and not path.is_file():
        raise FileNotFoundError(f"{name} is not a file: {path}")
    return path


def _resolve_python_entrypoint(value: str | Path | None) -> Path:
    """Validate a Python executable without dereferencing its environment shim.

    The project uses an overlay virtual environment whose ``bin/python`` is a
    symlink to the RoboDojo base interpreter.  Python selects the overlay's
    site-packages from the symlink entrypoint.  Calling ``Path.resolve()`` here
    changes that entrypoint to the base interpreter and silently drops packages
    such as ``verl`` on Slurm workers.
    """

    if value is None or not str(value).strip():
        raise ValueError("--python is required.")
    path = Path(os.path.abspath(os.path.expanduser(str(value))))
    if not path.is_file():
        raise FileNotFoundError(f"--python is not a file: {path}")
    if not os.access(path, os.X_OK):
        raise PermissionError(f"--python is not executable: {path}")
    return path


def _resolve_spec_paths(spec: dict[str, Any], spec_dir: Path) -> dict[str, Any]:
    """Resolve user-owned policy paths exactly once at the spec boundary."""

    normalized = json.loads(json.dumps(spec))
    for policy in normalized.get("policies", []):
        for key in ("model_root", "verl_actor_checkpoint"):
            raw = policy.get(key)
            if raw in (None, ""):
                continue
            path = Path(str(raw)).expanduser()
            if not path.is_absolute():
                path = spec_dir / path
            policy[key] = str(path.resolve(strict=True))
    return normalized


def _runtime_from_args(args: argparse.Namespace) -> dict[str, Any]:
    policy_root = _resolve_input_path(args.policy_root, "--policy-root/FASTWAM_POLICY_ROOT", directory=True)
    robodojo_root = _resolve_input_path(args.robodojo_root, "--robodojo-root/ROBODOJO_ROOT", directory=True)
    python = _resolve_python_entrypoint(args.python)
    hf_home = _resolve_input_path(args.hf_home, "--hf-home/HF_HOME", directory=True)
    fastwam_src = (policy_root / "FastWAM/src").resolve(strict=True)
    native_lib_dir = None
    if args.native_lib_dir:
        native_lib_dir = str(_resolve_input_path(args.native_lib_dir, "--native-lib-dir", directory=True))
    return {
        "repo_root": str(REPO_ROOT),
        "python": str(python),
        "policy_root": str(policy_root),
        "robodojo_root": str(robodojo_root),
        "fastwam_src": str(fastwam_src),
        "hf_home": str(hf_home),
        "native_lib_dir": native_lib_dir,
        "slurm": {
            "account": args.account,
            "qos": args.qos,
            "module": args.module,
            "exclude": args.exclude,
            "normal_partition": os.environ.get("VVLA_SLURM_NORMAL_PARTITION", "normal"),
            "preempt_partition": os.environ.get("VVLA_SLURM_PREEMPT_PARTITION", "preempt"),
            "cpu_partition": os.environ.get("VVLA_SLURM_CPU_PARTITION", "cpu"),
            "normal_qos": os.environ.get("VVLA_SLURM_NORMAL_QOS", ""),
            "preempt_qos": os.environ.get("VVLA_SLURM_PREEMPT_QOS", ""),
            "cpu_qos": os.environ.get("VVLA_SLURM_CPU_QOS", ""),
        },
    }


def _check_runtime_python(runtime: dict[str, Any]) -> None:
    """Validate the exact interpreter before reserving GPU resources."""

    subprocess.run(
        [
            str(runtime["python"]),
            str(REPO_ROOT / "scripts/lib/check_runtime.py"),
            "--repo-root",
            str(runtime["repo_root"]),
            "--robodojo-root",
            str(runtime["robodojo_root"]),
            "--policy-root",
            str(runtime["policy_root"]),
        ],
        check=True,
    )


def _runtime(run_manifest: dict[str, Any]) -> dict[str, Any]:
    runtime = run_manifest.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError("Run manifest is missing the resolved runtime path contract.")
    return runtime


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    atomic_write(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _require_int(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}.")
    return value


def validate_spec(spec: dict[str, Any], *, check_paths: bool) -> None:
    if int(spec.get("schema_version", -1)) != 1:
        raise ValueError("Fixed eval spec schema_version must be 1.")
    name = str(spec.get("name", ""))
    if not name or not LABEL_PATTERN.fullmatch(name):
        raise ValueError(f"Invalid experiment name: {name!r}.")
    repeats = _require_int(spec.get("repeats"), "repeats", 1)
    policies = spec.get("policies")
    targets = spec.get("targets")
    topology = spec.get("topology")
    if not isinstance(policies, list) or len(policies) < 1:
        raise ValueError("policies must be a non-empty list.")
    if not isinstance(targets, list) or len(targets) < 1:
        raise ValueError("targets must be a non-empty list.")
    if not isinstance(topology, dict):
        raise ValueError("topology must be an object.")

    labels: set[str] = set()
    for index, policy in enumerate(policies):
        if not isinstance(policy, dict):
            raise TypeError(f"policies[{index}] must be an object.")
        label = str(policy.get("label", ""))
        if not label or not LABEL_PATTERN.fullmatch(label) or label in labels:
            raise ValueError(f"Invalid or duplicate policy label: {label!r}.")
        labels.add(label)
        model_root = Path(str(policy.get("model_root", ""))).expanduser()
        if not model_root.is_absolute():
            raise ValueError(f"Policy {label} model_root must be absolute.")
        expected_step = policy.get("expected_source_verl_global_step")
        actor_checkpoint = policy.get("verl_actor_checkpoint")
        if expected_step is None:
            if actor_checkpoint not in (None, ""):
                raise ValueError(f"Native policy {label} must not set verl_actor_checkpoint.")
        else:
            _require_int(expected_step, f"{label}.expected_source_verl_global_step", 1)
            if not actor_checkpoint:
                raise ValueError(f"RL policy {label} requires verl_actor_checkpoint.")
            actor_checkpoint = Path(str(actor_checkpoint)).expanduser()
            if not actor_checkpoint.is_absolute():
                raise ValueError(f"Policy {label} verl_actor_checkpoint must be absolute.")
            if check_paths and not (actor_checkpoint / "actor/model_world_size_1_rank_0.pt").is_file():
                raise FileNotFoundError(f"Missing actor state for {label}: {actor_checkpoint}")
        checkpoint_sha = str(policy.get("checkpoint_sha256", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", checkpoint_sha):
            raise ValueError(f"Policy {label} has invalid checkpoint_sha256.")
        if check_paths:
            native_weight = model_root / "checkpoints/weights/step_020000.pt"
            if not native_weight.is_file():
                raise FileNotFoundError(f"Missing native weight for {label}: {native_weight}")
            if not (model_root / "dataset_stats.json").is_file():
                raise FileNotFoundError(f"Missing dataset_stats.json for {label}: {model_root}")

    num_envs = _require_int(topology.get("num_envs"), "topology.num_envs", 1)
    microbatch = _require_int(topology.get("policy_microbatch_size"), "topology.policy_microbatch_size", 1)
    if microbatch > num_envs:
        raise ValueError("policy_microbatch_size cannot exceed num_envs.")
    if topology.get("full_observation") is not True:
        raise ValueError("The canonical evaluator requires full_observation=true.")
    if topology.get("record_video") is not True:
        raise ValueError("The targeted replay protocol requires record_video=true.")
    if str(topology.get("process_score_reduction")) != "terminal":
        raise ValueError("New canonical evals must use terminal process score.")
    _require_int(topology.get("max_policy_calls"), "topology.max_policy_calls", 1)
    _require_int(topology.get("reset_max_attempts"), "topology.reset_max_attempts", 1)

    target_keys: set[tuple[str, int, int]] = set()
    for index, target in enumerate(targets):
        if not isinstance(target, dict):
            raise TypeError(f"targets[{index}] must be an object.")
        task = str(target.get("task_name", ""))
        if not task or not TASK_PATTERN.fullmatch(task):
            raise ValueError(f"Invalid target task_name: {task!r}.")
        _require_int(target.get("task_id"), f"targets[{index}].task_id")
        suite = _require_int(target.get("suite_id"), f"targets[{index}].suite_id")
        layout = _require_int(target.get("layout_id"), f"targets[{index}].layout_id")
        key = (task, suite, layout)
        if key in target_keys:
            raise ValueError(f"Duplicate task/suite/layout target: {key}.")
        target_keys.add(key)
        seeds = target.get("policy_seeds")
        if not isinstance(seeds, list) or len(seeds) != num_envs:
            raise ValueError(f"Target {key} must provide exactly num_envs={num_envs} policy seeds.")
        normalized_seeds = [_require_int(seed, f"{key}.policy_seed") for seed in seeds]
        if len(set(normalized_seeds)) != len(normalized_seeds):
            raise ValueError(f"Target {key} contains duplicate policy seeds.")
    if repeats * num_envs < 6 and spec.get("purpose") == "targeted_replay":
        raise ValueError("targeted_replay requires at least six trials per layout.")


def build_work_items(spec: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for policy in spec["policies"]:
        for target in spec["targets"]:
            for repeat in range(int(spec["repeats"])):
                item_id = (
                    f"{policy['label']}__{target['task_name']}__s{int(target['suite_id']):02d}"
                    f"__l{int(target['layout_id']):03d}__r{repeat:02d}"
                )
                items.append(
                    {
                        "item_id": item_id,
                        "policy_label": str(policy["label"]),
                        "task_name": str(target["task_name"]),
                        "task_id": int(target["task_id"]),
                        "suite_id": int(target["suite_id"]),
                        "layout_id": int(target["layout_id"]),
                        "policy_seeds": [int(value) for value in target["policy_seeds"]],
                        "repeat": repeat,
                    }
                )
    if len({item["item_id"] for item in items}) != len(items):
        raise AssertionError("Generated duplicate fixed-eval work item ids.")
    return items


def item_root(run_manifest: dict[str, Any], item: dict[str, Any]) -> Path:
    return Path(run_manifest["output_root"]) / "items" / item["item_id"]


def complete_path(run_manifest: dict[str, Any], item: dict[str, Any]) -> Path:
    return item_root(run_manifest, item) / "COMPLETE.json"


def policy_by_label(spec: dict[str, Any], label: str) -> dict[str, Any]:
    matches = [policy for policy in spec["policies"] if policy["label"] == label]
    if len(matches) != 1:
        raise RuntimeError(f"Policy label {label!r} is not unique in the run manifest.")
    return matches[0]


def validate_metrics(
    metrics_path: Path,
    *,
    run_manifest: dict[str, Any],
    item: dict[str, Any],
    require_video: bool,
) -> dict[str, Any]:
    metrics = read_json(metrics_path)
    spec = run_manifest["spec"]
    topology = spec["topology"]
    policy = policy_by_label(spec, item["policy_label"])
    seeds = [int(value) for value in item["policy_seeds"]]
    cases = metrics.get("cases")
    if metrics.get("task_name") != item["task_name"] or not isinstance(cases, list):
        raise ValueError(f"Task/case mismatch in {metrics_path}.")
    if int(metrics.get("num_cases", -1)) != len(seeds) or len(cases) != len(seeds):
        raise ValueError(f"Case-count mismatch in {metrics_path}.")
    actual_keys = sorted((int(case["layout_id"]), int(case["policy_seed"])) for case in cases)
    expected_keys = sorted((int(item["layout_id"]), seed) for seed in seeds)
    if actual_keys != expected_keys:
        raise ValueError(f"Fixed condition/seed mismatch in {metrics_path}.")
    expected_scalars = {
        "robodojo_eval_seed": int(item["suite_id"]),
        "num_envs": int(topology["num_envs"]),
        "policy_microbatch_size": int(topology["policy_microbatch_size"]),
        "max_policy_calls": int(topology["max_policy_calls"]),
        "action_chunk_size": 24,
    }
    for key, expected in expected_scalars.items():
        if int(metrics.get(key, -1)) != expected:
            raise ValueError(f"Topology mismatch {key} in {metrics_path}: expected {expected}.")
    if metrics.get("process_score_reduction") != topology["process_score_reduction"]:
        raise ValueError(f"Process-score reduction mismatch in {metrics_path}.")
    expected_step = policy.get("expected_source_verl_global_step")
    identity = metrics.get("in_memory_rl_checkpoint")
    if expected_step is None:
        if identity is not None:
            raise ValueError(f"Native policy unexpectedly loaded an RL checkpoint in {metrics_path}.")
    else:
        if int((identity or {}).get("source_verl_global_step", -1)) != int(expected_step):
            raise ValueError(f"RL checkpoint step mismatch in {metrics_path}: {identity!r}.")
        expected_actor_state = (
            Path(str(policy["verl_actor_checkpoint"])) / "actor/model_world_size_1_rank_0.pt"
        ).resolve()
        actual_actor_state = Path(str((identity or {}).get("source_verl_actor_state", ""))).resolve()
        if actual_actor_state != expected_actor_state:
            raise ValueError(
                f"RL checkpoint path mismatch in {metrics_path}: expected {expected_actor_state}, "
                f"got {actual_actor_state}."
            )
    if not bool(metrics.get("all_cases_done")):
        raise ValueError(f"Incomplete trajectories in {metrics_path}.")
    video_root = metrics_path.parent / "videos"
    shared_video_root = metrics.get("shared_video_root")
    if shared_video_root:
        video_root = Path(run_manifest["output_root"]) / str(shared_video_root)
    if require_video and not any(video_root.rglob("*.mp4")):
        raise ValueError(f"No video was produced for {metrics_path} (video_root={video_root}).")
    return metrics


def read_complete(
    run_manifest: dict[str, Any], item: dict[str, Any], *, require_video: bool = True
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    marker_path = complete_path(run_manifest, item)
    if not marker_path.is_file():
        return None
    marker = read_json(marker_path)
    if marker.get("item_id") != item["item_id"]:
        raise ValueError(f"Completion marker identity mismatch: {marker_path}")
    metrics_path = Path(run_manifest["output_root"]) / str(marker["metrics_relative_path"])
    metrics = validate_metrics(
        metrics_path,
        run_manifest=run_manifest,
        item=item,
        require_video=require_video,
    )
    return marker, metrics


def _json_list(values: list[int]) -> str:
    return json.dumps(values, separators=(",", ":"))


def run_item(run_manifest: dict[str, Any], item: dict[str, Any]) -> None:
    existing = read_complete(run_manifest, item)
    if existing is not None:
        print(f"EVAL_SKIP_COMPLETE item={item['item_id']}", flush=True)
        return
    spec = run_manifest["spec"]
    runtime = _runtime(run_manifest)
    topology = spec["topology"]
    policy = policy_by_label(spec, item["policy_label"])
    root = item_root(run_manifest, item)
    root.mkdir(parents=True, exist_ok=True)
    max_attempts = int(spec.get("attempts_per_item", 3))
    job_token = os.environ.get("SLURM_JOB_ID", f"manual-{os.getpid()}")
    last_status = 1
    for attempt in range(1, max_attempts + 1):
        attempt_dir = root / "attempts" / f"attempt_{attempt:02d}"
        metrics_path = attempt_dir / "metrics.json"
        if metrics_path.is_file():
            try:
                validate_metrics(
                    metrics_path,
                    run_manifest=run_manifest,
                    item=item,
                    require_video=True,
                )
            except Exception as error:
                raise RuntimeError(
                    f"Existing attempt is invalid and will not be overwritten: {metrics_path}: {error}"
                ) from error
            marker = {
                "schema_version": 1,
                "item_id": item["item_id"],
                "attempt": attempt,
                "metrics_relative_path": str(metrics_path.relative_to(Path(run_manifest["output_root"]))),
                "completed_at_unix_s": time.time(),
                "recovered_existing_metrics": True,
            }
            atomic_write_json(complete_path(run_manifest, item), marker)
            return
        attempt_dir.mkdir(parents=True, exist_ok=True)
        runtime_tmp = Path(f"/tmp/vvla-fixed-eval-{job_token}-{item['item_id']}-{attempt}")
        runtime_tmp.mkdir(parents=True, exist_ok=True)
        command = [
            str(runtime["python"]),
            "-m",
            "verl_vla.entrypoints.native_fastwam_robodojo_eval",
            f"model_root={policy['model_root']}",
            f"checkpoint_sha256={policy['checkpoint_sha256']}",
            f"verl_actor_checkpoint={policy.get('verl_actor_checkpoint') or 'null'}",
            f"policy_root={runtime['policy_root']}",
            f"robodojo_root={runtime['robodojo_root']}",
            f"task_name={item['task_name']}",
            f"task_id={item['task_id']}",
            f"seed={item['suite_id']}",
            f"layout_ids={_json_list([int(item['layout_id'])] * len(item['policy_seeds']))}",
            f"policy_seeds={_json_list(item['policy_seeds'])}",
            "allow_held_out_eval_layouts=true",
            f"num_envs={topology['num_envs']}",
            f"policy_microbatch_size={topology['policy_microbatch_size']}",
            f"reset_max_attempts={topology['reset_max_attempts']}",
            "simulator_device_id=0",
            "model_device_id=0",
            f"max_policy_calls={topology['max_policy_calls']}",
            f"process_score_reduction={topology['process_score_reduction']}",
            f"record_video={str(bool(topology['record_video'])).lower()}",
            f"output_dir={attempt_dir}",
        ]
        environment = dict(os.environ)
        environment.update(
            {
                "TMPDIR": str(runtime_tmp),
                "XDG_CACHE_HOME": str(runtime_tmp / "cache"),
                "XDG_CONFIG_HOME": str(runtime_tmp / "config"),
                "XDG_DATA_HOME": str(runtime_tmp / "data"),
                "PYTHONNOUSERSITE": "1",
                "PYTHONPATH": ":".join(
                    [
                        str(REPO_ROOT / "src"),
                        str(runtime["robodojo_root"]),
                        str(runtime["fastwam_src"]),
                    ]
                ),
                "HF_HOME": str(runtime["hf_home"]),
                "DIFFSYNTH_MODEL_BASE_PATH": str(Path(runtime["fastwam_src"]).parent / "checkpoints"),
                "DIFFSYNTH_SKIP_DOWNLOAD": "true",
                "HYDRA_FULL_ERROR": "1",
            }
        )
        for path in (runtime_tmp / "cache", runtime_tmp / "config", runtime_tmp / "data"):
            path.mkdir(parents=True, exist_ok=True)
        log_path = attempt_dir / "eval.log"
        print(
            f"EVAL_ITEM_START item={item['item_id']} attempt={attempt}/{max_attempts} "
            f"topology=vec{topology['num_envs']}/mb{topology['policy_microbatch_size']}",
            flush=True,
        )
        with log_path.open("a", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=runtime["repo_root"],
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
                log.flush()
            last_status = process.wait()
        try:
            if metrics_path.is_file():
                validate_metrics(
                    metrics_path,
                    run_manifest=run_manifest,
                    item=item,
                    require_video=True,
                )
                marker = {
                    "schema_version": 1,
                    "item_id": item["item_id"],
                    "attempt": attempt,
                    "evaluator_exit_status": last_status,
                    "metrics_relative_path": str(metrics_path.relative_to(Path(run_manifest["output_root"]))),
                    "completed_at_unix_s": time.time(),
                }
                atomic_write_json(complete_path(run_manifest, item), marker)
                print(
                    f"EVAL_ITEM_COMPLETE item={item['item_id']} attempt={attempt} evaluator_status={last_status}",
                    flush=True,
                )
                return
        finally:
            if str(runtime_tmp).startswith("/tmp/vvla-fixed-eval-"):
                shutil.rmtree(runtime_tmp, ignore_errors=True)
        print(
            f"EVAL_ITEM_RETRY item={item['item_id']} attempt={attempt} status={last_status}",
            flush=True,
        )
        time.sleep(15 * attempt)
    raise RuntimeError(f"Evaluation item failed after {max_attempts} attempts: {item['item_id']}")


def collect_rows(run_manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for item in run_manifest["work_items"]:
        complete = read_complete(run_manifest, item)
        if complete is None:
            missing.append(item["item_id"])
            continue
        marker, metrics = complete
        for case in metrics["cases"]:
            rows.append(
                {
                    "policy": item["policy_label"],
                    "task": item["task_name"],
                    "suite_id": int(item["suite_id"]),
                    "layout_id": int(item["layout_id"]),
                    "policy_seed": int(case["policy_seed"]),
                    "repeat": int(item["repeat"]),
                    "success": bool(case["success"]),
                    "terminal_process_score": float(case.get("score", 0.0)),
                    "policy_calls": int(case["policy_calls"]),
                    "video_root": str(
                        (
                            Path(run_manifest["output_root"]) / Path(marker["metrics_relative_path"]).parent / "videos"
                        ).resolve()
                    ),
                }
            )
    return rows, missing


def summarize_run(run_manifest: dict[str, Any], *, require_complete: bool) -> dict[str, Any]:
    rows, missing = collect_rows(run_manifest)
    if require_complete and missing:
        raise RuntimeError(f"Evaluation is incomplete: {len(missing)} work items remain.")
    spec = run_manifest["spec"]
    policy_rows: list[dict[str, Any]] = []
    for policy in [entry["label"] for entry in spec["policies"]]:
        selected = [row for row in rows if row["policy"] == policy]
        layout_groups: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
        for row in selected:
            layout_groups[(row["task"], row["suite_id"], row["layout_id"])].append(row)
        layouts = []
        for (task, suite, layout), group in sorted(layout_groups.items()):
            layouts.append(
                {
                    "task": task,
                    "suite_id": suite,
                    "layout_id": layout,
                    "trials": len(group),
                    "successes": sum(int(row["success"]) for row in group),
                    "success_rate": statistics.fmean(float(row["success"]) for row in group),
                    "mean_terminal_process_score": statistics.fmean(row["terminal_process_score"] for row in group),
                }
            )
        task_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for layout in layouts:
            task_groups[layout["task"]].append(layout)
        tasks = [
            {
                "task": task,
                "layouts": len(group),
                "trials": sum(layout["trials"] for layout in group),
                "successes": sum(layout["successes"] for layout in group),
                "layout_macro_success_rate": statistics.fmean(layout["success_rate"] for layout in group),
            }
            for task, group in sorted(task_groups.items())
        ]
        policy_rows.append(
            {
                "policy": policy,
                "completed_trials": len(selected),
                "successes": sum(int(row["success"]) for row in selected),
                "trajectory_success_rate": (
                    statistics.fmean(float(row["success"]) for row in selected) if selected else None
                ),
                "layout_macro_success_rate": (
                    statistics.fmean(layout["success_rate"] for layout in layouts) if layouts else None
                ),
                "tasks": tasks,
                "layouts": layouts,
            }
        )

    comparisons: list[dict[str, Any]] = []
    labels = [entry["label"] for entry in spec["policies"]]
    for left_index, left in enumerate(labels):
        for right in labels[left_index + 1 :]:
            left_rows = {
                (row["task"], row["suite_id"], row["layout_id"], row["policy_seed"], row["repeat"]): row
                for row in rows
                if row["policy"] == left
            }
            right_rows = {
                (row["task"], row["suite_id"], row["layout_id"], row["policy_seed"], row["repeat"]): row
                for row in rows
                if row["policy"] == right
            }
            overlap = sorted(set(left_rows) & set(right_rows))
            layout_deltas: dict[tuple[str, int, int], list[float]] = defaultdict(list)
            success_to_fail = 0
            fail_to_success = 0
            for key in overlap:
                left_success = bool(left_rows[key]["success"])
                right_success = bool(right_rows[key]["success"])
                success_to_fail += int(left_success and not right_success)
                fail_to_success += int(not left_success and right_success)
                layout_deltas[key[:3]].append(float(right_success) - float(left_success))
            per_layout = [
                {
                    "task": key[0],
                    "suite_id": key[1],
                    "layout_id": key[2],
                    "right_minus_left_success_rate": statistics.fmean(values),
                }
                for key, values in sorted(layout_deltas.items())
            ]
            comparisons.append(
                {
                    "left": left,
                    "right": right,
                    "paired_trials": len(overlap),
                    "success_to_fail": success_to_fail,
                    "fail_to_success": fail_to_success,
                    "net_successes": fail_to_success - success_to_fail,
                    "layouts_improved": sum(item["right_minus_left_success_rate"] > 0 for item in per_layout),
                    "layouts_regressed": sum(item["right_minus_left_success_rate"] < 0 for item in per_layout),
                    "layouts_tied": sum(item["right_minus_left_success_rate"] == 0 for item in per_layout),
                    "per_layout": per_layout,
                }
            )
    return {
        "schema_version": 1,
        "protocol": str(spec["name"]),
        "fixed_topology": spec["topology"],
        "repeats": int(spec["repeats"]),
        "expected_work_items": len(run_manifest["work_items"]),
        "completed_work_items": len(run_manifest["work_items"]) - len(missing),
        "complete": not missing,
        "missing_work_items": missing,
        "policies": policy_rows,
        "comparisons": comparisons,
        "generated_at_unix_s": time.time(),
    }


def summary_markdown(summary: dict[str, Any]) -> str:
    topology = summary["fixed_topology"]
    lines = [
        f"# {summary['protocol']}",
        "",
        f"Status: **{summary['completed_work_items']}/{summary['expected_work_items']} work items**; "
        f"complete={str(summary['complete']).lower()}.",
        "",
        "Fixed execution signature: "
        f"`vec{topology['num_envs']}`, policy microbatch `{topology['policy_microbatch_size']}`, "
        f"full observation, terminal process score, video enabled, {summary['repeats']} repeats.",
        "",
        "Strict RoboDojo success is the primary label. Process score never converts a strict failure into success.",
        "",
        "## Policy aggregate",
        "",
        "| Policy | Trials | Strict success | Trajectory SR | Layout-macro SR |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for policy in summary["policies"]:
        trajectory_rate = policy["trajectory_success_rate"]
        layout_rate = policy["layout_macro_success_rate"]
        lines.append(
            f"| {policy['policy']} | {policy['completed_trials']} | {policy['successes']} | "
            f"{100 * trajectory_rate:.2f}% | {100 * layout_rate:.2f}% |"
            if trajectory_rate is not None and layout_rate is not None
            else f"| {policy['policy']} | 0 | 0 | n/a | n/a |"
        )
    lines.extend(["", "## Layout-level aggregate", ""])
    for policy in summary["policies"]:
        lines.extend(
            [
                f"### {policy['policy']}",
                "",
                "| Task | Suite | Layout | Success | SR | Mean terminal score |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for layout in policy["layouts"]:
            lines.append(
                f"| {layout['task']} | {layout['suite_id']} | {layout['layout_id']} | "
                f"{layout['successes']}/{layout['trials']} | {100 * layout['success_rate']:.2f}% | "
                f"{layout['mean_terminal_process_score']:.4f} |"
            )
        lines.append("")
    if summary["comparisons"]:
        lines.extend(
            [
                "## Paired checkpoint comparison",
                "",
                "| Left → right | Paired trials | S→F | F→S | Net | Layouts + / - / tied |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for comparison in summary["comparisons"]:
            lines.append(
                f"| {comparison['left']} → {comparison['right']} | {comparison['paired_trials']} | "
                f"{comparison['success_to_fail']} | {comparison['fail_to_success']} | "
                f"{comparison['net_successes']:+d} | {comparison['layouts_improved']} / "
                f"{comparison['layouts_regressed']} / {comparison['layouts_tied']} |"
            )
    if summary["missing_work_items"]:
        lines.extend(["", "## Missing work items", "", *[f"- `{item}`" for item in summary["missing_work_items"]]])
    return "\n".join(lines) + "\n"


def write_summary(run_manifest: dict[str, Any], *, require_complete: bool) -> dict[str, Any]:
    summary = summarize_run(run_manifest, require_complete=require_complete)
    output_root = Path(run_manifest["output_root"])
    atomic_write_json(output_root / "summary.json", summary)
    atomic_write(output_root / "summary.md", summary_markdown(summary))
    if summary["complete"]:
        atomic_write_json(
            output_root / "EVAL_COMPLETE.json",
            {
                "schema_version": 1,
                "protocol": summary["protocol"],
                "completed_work_items": summary["completed_work_items"],
                "completed_at_unix_s": time.time(),
            },
        )
    return summary


def load_run_manifest(path: Path) -> dict[str, Any]:
    manifest = read_json(path.resolve())
    if int(manifest.get("schema_version", -1)) != 1:
        raise ValueError(f"Unsupported run manifest: {path}")
    validate_spec(manifest["spec"], check_paths=False)
    expected = build_work_items(manifest["spec"])
    if manifest.get("work_items") != expected:
        raise ValueError("Run manifest work items do not match its immutable spec.")
    _runtime(manifest)
    return manifest


def command_launch(args: argparse.Namespace) -> None:
    spec_path = args.spec.expanduser().resolve(strict=True)
    spec = _resolve_spec_paths(read_json(spec_path), spec_path.parent)
    requested_policy_labels = list(args.policy_label or [])
    if requested_policy_labels:
        if len(set(requested_policy_labels)) != len(requested_policy_labels):
            raise ValueError(f"Duplicate --policy-label values: {requested_policy_labels}")
        available = {str(policy.get("label", "")) for policy in spec.get("policies", [])}
        unknown = sorted(set(requested_policy_labels) - available)
        if unknown:
            raise ValueError(f"Unknown --policy-label values: {unknown}; available={sorted(available)}")
        requested = set(requested_policy_labels)
        spec = {
            **spec,
            "policies": [policy for policy in spec["policies"] if policy["label"] in requested],
        }
    validate_spec(spec, check_paths=True)
    if args.coalesced_items_per_evaluator > 1 and args.evaluators_per_gpu != 1:
        raise ValueError(
            "--coalesced-items-per-evaluator and multiple independent --evaluators-per-gpu are mutually exclusive."
        )
    output_root = args.output_root.expanduser().resolve()
    runtime = _runtime_from_args(args)
    _check_runtime_python(runtime)
    work_items = build_work_items(spec)
    run_manifest = {
        "schema_version": 1,
        "spec_path": str(spec_path),
        "spec_sha256": sha256(spec_path),
        "output_root": str(output_root),
        "runtime": runtime,
        "spec": spec,
        "work_items": work_items,
    }
    manifest_path = output_root / "RUN_MANIFEST.json"
    if manifest_path.exists():
        existing = read_json(manifest_path)
        if existing != run_manifest:
            raise RuntimeError(f"Refusing to change an existing eval run manifest: {manifest_path}")
    else:
        atomic_write_json(manifest_path, run_manifest)
    (output_root / "logs").mkdir(parents=True, exist_ok=True)
    summary = write_summary(run_manifest, require_complete=False)
    print(
        json.dumps(
            {
                "run_manifest": str(manifest_path),
                "work_items": len(work_items),
                "trajectories": len(work_items) * int(spec["topology"]["num_envs"]),
                "completed_work_items": summary["completed_work_items"],
                "gpu_lanes": args.lanes,
                "evaluators_per_gpu": args.evaluators_per_gpu,
                "coalesced_items_per_evaluator": args.coalesced_items_per_evaluator,
                "maximum_logical_lanes": args.lanes * args.evaluators_per_gpu,
                "partition": args.partition,
                "dry_run": args.dry_run,
                "policy_labels": [policy["label"] for policy in spec["policies"]],
            },
            indent=2,
        )
    )
    if args.dry_run:
        return
    slurm = runtime["slurm"]
    if not slurm["account"]:
        raise ValueError("--account or SLURM_ACCOUNT is required before job submission.")
    launcher_prefix = "set -Eeuo pipefail; "
    if slurm["module"]:
        launcher_prefix += f"module load {shlex.quote(str(slurm['module']))}; "
    if runtime["native_lib_dir"]:
        launcher_prefix += (
            f"export LD_LIBRARY_PATH={shlex.quote(str(runtime['native_lib_dir']))}:${{LD_LIBRARY_PATH:-}}; "
        )
    # Slurm presents the one allocated GPU as process-local cuda:0 even when
    # its node-global physical index is different.
    launcher_prefix += "export VERL_VLA_PHYSICAL_GPU_ID=0; "
    launcher_prefix += f"cd {shlex.quote(str(runtime['repo_root']))}; "
    if args.coalesced_items_per_evaluator > 1:
        bundles = coalesced_item_bundles(run_manifest, args.coalesced_items_per_evaluator)
        gpu_lane_count = min(args.lanes, len(bundles))
        logical_lane_count = gpu_lane_count
        wrap = (
            launcher_prefix + f"exec {shlex.quote(str(runtime['python']))} "
            f"{shlex.quote(str(Path(__file__).resolve()))} coalesced-lane "
            f"--run-manifest {shlex.quote(str(manifest_path))} --lane-count {gpu_lane_count} "
            f"--items-per-evaluator {args.coalesced_items_per_evaluator}"
        )
    else:
        gpu_lane_count = min(args.lanes, ceil(len(work_items) / args.evaluators_per_gpu))
        logical_lane_count = gpu_lane_count * args.evaluators_per_gpu
        wrap = (
            launcher_prefix + f"exec {shlex.quote(str(runtime['python']))} "
            f"{shlex.quote(str(Path(__file__).resolve()))} packed-lane "
            f"--run-manifest {shlex.quote(str(manifest_path))} "
            f"--physical-lane-count {gpu_lane_count} "
            f"--evaluators-per-gpu {args.evaluators_per_gpu} "
            f"--startup-stagger-seconds {args.startup_stagger_seconds}"
        )
    cpus_per_task = args.cpus_per_task or (57 if args.evaluators_per_gpu == 3 else 32)
    memory = args.memory or ("300G" if args.evaluators_per_gpu == 3 else "220G")
    partition_name = slurm.get(f"{args.partition}_partition", args.partition)
    partition_qos = slurm.get(f"{args.partition}_qos") or slurm["qos"]
    command = [
        "sbatch",
        "--parsable",
        f"--partition={partition_name}",
        f"--account={slurm['account']}",
        "--nodes=1",
        "--gpus-per-node=1",
        f"--cpus-per-task={cpus_per_task}",
        f"--mem={memory}",
        f"--time={args.time}",
        "--requeue",
        f"--array=0-{gpu_lane_count - 1}%{gpu_lane_count}",
        f"--job-name={spec['name'][:40]}",
        f"--output={output_root}/logs/lane_%A_%a.out",
        f"--wrap={wrap}",
    ]
    if slurm["exclude"]:
        command.insert(-2, f"--exclude={slurm['exclude']}")
    if partition_qos:
        command.insert(4, f"--qos={partition_qos}")
    job_id = subprocess.check_output(command, text=True).strip()
    launch_record = {
        "schema_version": 1,
        "job_id": job_id,
        "partition": args.partition,
        "gpu_lanes": gpu_lane_count,
        "evaluators_per_gpu": args.evaluators_per_gpu,
        "coalesced_items_per_evaluator": args.coalesced_items_per_evaluator,
        "logical_lanes": logical_lane_count,
        "cpus_per_task": cpus_per_task,
        "memory": memory,
        "time": args.time,
        "submitted_at_unix_s": time.time(),
        "run_manifest": str(manifest_path),
    }
    atomic_write_json(output_root / "LAUNCH.json", launch_record)
    print(f"submitted_job={job_id} run_manifest={manifest_path}")


def command_lane(args: argparse.Namespace) -> None:
    run_manifest = load_run_manifest(args.run_manifest)
    lane = int(os.environ.get("SLURM_ARRAY_TASK_ID", args.lane if args.lane is not None else -1))
    if not 0 <= lane < args.lane_count:
        raise ValueError(f"Invalid lane={lane} for lane_count={args.lane_count}.")
    for index, item in enumerate(run_manifest["work_items"]):
        if index % args.lane_count == lane:
            run_item(run_manifest, item)
            write_summary(run_manifest, require_complete=False)
    summary = write_summary(run_manifest, require_complete=False)
    print(
        f"EVAL_LANE_COMPLETE lane={lane}/{args.lane_count} "
        f"complete={summary['completed_work_items']}/{summary['expected_work_items']}",
        flush=True,
    )


def packed_logical_lanes(physical_lane: int, physical_lane_count: int, evaluators_per_gpu: int) -> tuple[int, ...]:
    if physical_lane_count < 1:
        raise ValueError("physical_lane_count must be positive.")
    if evaluators_per_gpu < 1:
        raise ValueError("evaluators_per_gpu must be positive.")
    if not 0 <= physical_lane < physical_lane_count:
        raise ValueError(f"Invalid physical_lane={physical_lane} for physical_lane_count={physical_lane_count}.")
    first = physical_lane * evaluators_per_gpu
    return tuple(first + offset for offset in range(evaluators_per_gpu))


def coalesced_item_bundles(run_manifest: dict[str, Any], items_per_evaluator: int) -> list[list[dict[str, Any]]]:
    """Group distinct layouts into one larger native vector evaluator.

    Items may only share a native process when policy, task, RoboDojo suite,
    and repeat are identical.  Grouping by repeat is important: putting two
    repeats of the same ``(layout, policy_seed)`` in one vector would make the
    returned cases ambiguous.  Within each compatible group, adjacent layouts
    are packed into vectors such as ``3 layouts * vec6 = vec18``.
    """

    if items_per_evaluator < 1:
        raise ValueError("items_per_evaluator must be positive.")
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    order: list[tuple[Any, ...]] = []
    for item in run_manifest["work_items"]:
        key = (
            item["policy_label"],
            item["task_name"],
            int(item["task_id"]),
            int(item["suite_id"]),
            int(item["repeat"]),
        )
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(item)
    bundles: list[list[dict[str, Any]]] = []
    for key in order:
        items = groups[key]
        for start in range(0, len(items), items_per_evaluator):
            bundle = items[start : start + items_per_evaluator]
            condition_keys = [(int(item["layout_id"]), int(seed)) for item in bundle for seed in item["policy_seeds"]]
            if len(condition_keys) != len(set(condition_keys)):
                raise ValueError(
                    "Coalesced evaluator bundle contains duplicate layout/policy-seed cases: "
                    f"{[item['item_id'] for item in bundle]}"
                )
            bundles.append(bundle)
    return bundles


def _native_eval_environment(run_manifest: dict[str, Any], runtime_tmp: Path) -> dict[str, str]:
    runtime = _runtime(run_manifest)
    environment = dict(os.environ)
    environment.update(
        {
            "TMPDIR": str(runtime_tmp),
            "XDG_CACHE_HOME": str(runtime_tmp / "cache"),
            "XDG_CONFIG_HOME": str(runtime_tmp / "config"),
            "XDG_DATA_HOME": str(runtime_tmp / "data"),
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": ":".join(
                [
                    str(REPO_ROOT / "src"),
                    str(runtime["robodojo_root"]),
                    str(runtime["fastwam_src"]),
                ]
            ),
            "HF_HOME": str(runtime["hf_home"]),
            "DIFFSYNTH_MODEL_BASE_PATH": str(Path(runtime["fastwam_src"]).parent / "checkpoints"),
            "DIFFSYNTH_SKIP_DOWNLOAD": "true",
            "HYDRA_FULL_ERROR": "1",
        }
    )
    for path in (runtime_tmp / "cache", runtime_tmp / "config", runtime_tmp / "data"):
        path.mkdir(parents=True, exist_ok=True)
    return environment


def run_coalesced_bundle(run_manifest: dict[str, Any], bundle: list[dict[str, Any]]) -> None:
    """Evaluate several layout items in one mixed-layout native vector."""

    missing = [item for item in bundle if read_complete(run_manifest, item) is None]
    if not missing:
        print(
            f"COALESCED_EVAL_SKIP_COMPLETE items={[item['item_id'] for item in bundle]}",
            flush=True,
        )
        return
    if len(missing) == 1:
        run_item(run_manifest, missing[0])
        return

    first = missing[0]
    compatibility = (
        first["policy_label"],
        first["task_name"],
        int(first["task_id"]),
        int(first["suite_id"]),
        int(first["repeat"]),
    )
    for item in missing[1:]:
        actual = (
            item["policy_label"],
            item["task_name"],
            int(item["task_id"]),
            int(item["suite_id"]),
            int(item["repeat"]),
        )
        if actual != compatibility:
            raise ValueError(f"Incompatible coalesced items: {compatibility!r} != {actual!r}.")

    spec = run_manifest["spec"]
    runtime = _runtime(run_manifest)
    topology = spec["topology"]
    policy = policy_by_label(spec, first["policy_label"])
    logical_num_envs = int(topology["num_envs"])
    layout_ids = [int(item["layout_id"]) for item in missing for _ in item["policy_seeds"]]
    policy_seeds = [int(seed) for item in missing for seed in item["policy_seeds"]]
    physical_num_envs = len(policy_seeds)
    if physical_num_envs != logical_num_envs * len(missing):
        raise AssertionError("Coalesced vector size does not match its logical item count.")

    bundle_digest = hashlib.sha256("\n".join(item["item_id"] for item in missing).encode("utf-8")).hexdigest()[:16]
    bundle_root = Path(run_manifest["output_root"]) / "coalesced" / bundle_digest
    max_attempts = int(spec.get("attempts_per_item", 3))
    job_token = os.environ.get("SLURM_JOB_ID", f"manual-{os.getpid()}")
    last_status = 1
    for attempt in range(1, max_attempts + 1):
        attempt_dir = bundle_root / "attempts" / f"attempt_{attempt:02d}"
        metrics_path = attempt_dir / "metrics.json"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        runtime_tmp = Path(f"/tmp/vvla-fixed-eval-{job_token}-coalesced-{bundle_digest}-{attempt}")
        runtime_tmp.mkdir(parents=True, exist_ok=True)
        command = [
            str(runtime["python"]),
            "-m",
            "verl_vla.entrypoints.native_fastwam_robodojo_eval",
            f"model_root={policy['model_root']}",
            f"checkpoint_sha256={policy['checkpoint_sha256']}",
            f"verl_actor_checkpoint={policy.get('verl_actor_checkpoint') or 'null'}",
            f"policy_root={runtime['policy_root']}",
            f"robodojo_root={runtime['robodojo_root']}",
            f"task_name={first['task_name']}",
            f"task_id={first['task_id']}",
            f"seed={first['suite_id']}",
            f"layout_ids={_json_list(layout_ids)}",
            f"policy_seeds={_json_list(policy_seeds)}",
            "allow_held_out_eval_layouts=true",
            f"num_envs={physical_num_envs}",
            f"simulator_shards={len(missing)}",
            f"policy_microbatch_size={topology['policy_microbatch_size']}",
            f"reset_max_attempts={topology['reset_max_attempts']}",
            "simulator_device_id=0",
            "model_device_id=0",
            f"max_policy_calls={topology['max_policy_calls']}",
            f"process_score_reduction={topology['process_score_reduction']}",
            f"record_video={str(bool(topology['record_video'])).lower()}",
            f"output_dir={attempt_dir}",
        ]
        print(
            f"COALESCED_EVAL_START items={[item['item_id'] for item in missing]} "
            f"physical_vec={physical_num_envs} logical_vec={logical_num_envs} "
            f"attempt={attempt}/{max_attempts}",
            flush=True,
        )
        environment = _native_eval_environment(run_manifest, runtime_tmp)
        log_path = attempt_dir / "eval.log"
        with log_path.open("a", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=runtime["repo_root"],
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
                log.flush()
            last_status = process.wait()
        try:
            if not metrics_path.is_file():
                continue
            combined = read_json(metrics_path)
            combined_cases = combined.get("cases")
            if combined.get("task_name") != first["task_name"] or not isinstance(combined_cases, list):
                raise ValueError(f"Invalid coalesced metrics: {metrics_path}")
            expected_keys = sorted(zip(layout_ids, policy_seeds, strict=True))
            actual_keys = sorted((int(case["layout_id"]), int(case["policy_seed"])) for case in combined_cases)
            if actual_keys != expected_keys:
                raise ValueError(
                    f"Coalesced condition mismatch in {metrics_path}: expected={expected_keys}, actual={actual_keys}"
                )
            if int(combined.get("num_envs", -1)) != physical_num_envs:
                raise ValueError(f"Coalesced num_envs mismatch in {metrics_path}.")
            video_root = attempt_dir / "videos"
            if bool(topology["record_video"]) and not any(video_root.rglob("*.mp4")):
                raise ValueError(f"No coalesced videos were produced in {video_root}.")

            shared_video_root = str(video_root.relative_to(Path(run_manifest["output_root"])))
            case_index = {(int(case["layout_id"]), int(case["policy_seed"])): case for case in combined_cases}
            for item in missing:
                item_cases = [case_index[(int(item["layout_id"]), int(seed))] for seed in item["policy_seeds"]]
                split = {
                    **combined,
                    "num_cases": len(item_cases),
                    "num_envs": logical_num_envs,
                    "physical_num_envs": physical_num_envs,
                    "coalesced_item_count": len(missing),
                    "coalesced_bundle_id": bundle_digest,
                    "shared_video_root": shared_video_root,
                    "successes": sum(int(case["success"]) for case in item_cases),
                    "success_rate": float(sum(int(case["success"]) for case in item_cases) / len(item_cases)),
                    "mean_score": float(sum(float(case["score"]) for case in item_cases) / len(item_cases)),
                    "all_cases_done": all(bool(case["done"]) for case in item_cases),
                    "incomplete_case_count": sum(not bool(case["done"]) for case in item_cases),
                    "cases": item_cases,
                }
                item_attempt = item_root(run_manifest, item) / "attempts" / f"attempt_{attempt:02d}"
                split_metrics_path = item_attempt / "metrics.json"
                atomic_write_json(split_metrics_path, split)
                validate_metrics(
                    split_metrics_path,
                    run_manifest=run_manifest,
                    item=item,
                    require_video=True,
                )
                atomic_write_json(
                    complete_path(run_manifest, item),
                    {
                        "schema_version": 1,
                        "item_id": item["item_id"],
                        "attempt": attempt,
                        "evaluator_exit_status": last_status,
                        "metrics_relative_path": str(split_metrics_path.relative_to(Path(run_manifest["output_root"]))),
                        "coalesced_bundle_id": bundle_digest,
                        "completed_at_unix_s": time.time(),
                    },
                )
            print(
                f"COALESCED_EVAL_COMPLETE bundle={bundle_digest} "
                f"items={[item['item_id'] for item in missing]} evaluator_status={last_status}",
                flush=True,
            )
            return
        finally:
            if str(runtime_tmp).startswith("/tmp/vvla-fixed-eval-"):
                shutil.rmtree(runtime_tmp, ignore_errors=True)
        time.sleep(15 * attempt)
    raise RuntimeError(
        f"Coalesced evaluation failed after {max_attempts} attempts: "
        f"{[item['item_id'] for item in missing]}, status={last_status}"
    )


def command_coalesced_lane(args: argparse.Namespace) -> None:
    run_manifest = load_run_manifest(args.run_manifest)
    lane = int(os.environ.get("SLURM_ARRAY_TASK_ID", args.lane if args.lane is not None else -1))
    if not 0 <= lane < args.lane_count:
        raise ValueError(f"Invalid lane={lane} for lane_count={args.lane_count}.")
    bundles = coalesced_item_bundles(run_manifest, args.items_per_evaluator)
    for index, bundle in enumerate(bundles):
        if index % args.lane_count == lane:
            run_coalesced_bundle(run_manifest, bundle)
            write_summary(run_manifest, require_complete=False)
    summary = write_summary(run_manifest, require_complete=False)
    print(
        f"COALESCED_LANE_COMPLETE lane={lane}/{args.lane_count} "
        f"complete={summary['completed_work_items']}/{summary['expected_work_items']}",
        flush=True,
    )


def command_coalesced_plan_lane(args: argparse.Namespace) -> None:
    """Run an immutable recovery plan containing explicit work-item bundles."""

    run_manifest = load_run_manifest(args.run_manifest)
    plan = read_json(args.plan.resolve())
    raw_bundles = plan.get("bundles")
    if not isinstance(raw_bundles, list) or not raw_bundles:
        raise ValueError(f"Recovery plan has no bundles: {args.plan}")
    item_by_id = {item["item_id"]: item for item in run_manifest["work_items"]}
    bundles: list[list[dict[str, Any]]] = []
    seen: set[str] = set()
    for raw_bundle in raw_bundles:
        if not isinstance(raw_bundle, list) or not raw_bundle:
            raise ValueError(f"Invalid recovery bundle in {args.plan}: {raw_bundle!r}")
        bundle: list[dict[str, Any]] = []
        for item_id in raw_bundle:
            if item_id in seen:
                raise ValueError(f"Duplicate recovery item {item_id!r} in {args.plan}")
            try:
                item = item_by_id[item_id]
            except KeyError as error:
                raise ValueError(f"Unknown recovery item {item_id!r} in {args.plan}") from error
            seen.add(item_id)
            bundle.append(item)
        bundles.append(bundle)
    lane = int(os.environ.get("SLURM_ARRAY_TASK_ID", args.lane if args.lane is not None else -1))
    if not 0 <= lane < args.lane_count:
        raise ValueError(f"Invalid lane={lane} for lane_count={args.lane_count}.")
    for index, bundle in enumerate(bundles):
        if index % args.lane_count == lane:
            run_coalesced_bundle(run_manifest, bundle)
            write_summary(run_manifest, require_complete=False)
    print(
        f"COALESCED_PLAN_LANE_COMPLETE lane={lane}/{args.lane_count} plan={args.plan} bundles={len(bundles)}",
        flush=True,
    )


def command_freeze_missing_plan(args: argparse.Namespace) -> None:
    """Freeze currently incomplete work items into stable compatible bundles."""

    run_manifest = load_run_manifest(args.run_manifest)
    missing_manifest = {
        **run_manifest,
        "work_items": [item for item in run_manifest["work_items"] if read_complete(run_manifest, item) is None],
    }
    bundles = coalesced_item_bundles(missing_manifest, args.items_per_evaluator)
    plan = {
        "schema_version": 1,
        "run_manifest": str(args.run_manifest.resolve()),
        "created_at_unix_s": time.time(),
        "items_per_evaluator": args.items_per_evaluator,
        "item_count": sum(len(bundle) for bundle in bundles),
        "bundles": [[item["item_id"] for item in bundle] for bundle in bundles],
    }
    atomic_write_json(args.plan.resolve(), plan)
    print(
        f"frozen_missing_plan={args.plan.resolve()} items={plan['item_count']} bundles={len(bundles)}",
        flush=True,
    )


def command_packed_lane(args: argparse.Namespace) -> None:
    """Run independent fixed-eval lanes concurrently on one allocated GPU."""

    run_manifest = load_run_manifest(args.run_manifest)
    runtime = _runtime(run_manifest)
    physical_lane = int(
        os.environ.get(
            "SLURM_ARRAY_TASK_ID",
            args.physical_lane if args.physical_lane is not None else -1,
        )
    )
    logical_lanes = packed_logical_lanes(
        physical_lane,
        args.physical_lane_count,
        args.evaluators_per_gpu,
    )
    logical_lane_count = args.physical_lane_count * args.evaluators_per_gpu
    child_environment = dict(os.environ)
    # Each child must honor its explicit logical lane rather than inheriting
    # the physical Slurm array index.
    child_environment.pop("SLURM_ARRAY_TASK_ID", None)
    allocated_cpus = int(child_environment.get("SLURM_CPUS_PER_TASK", "32"))
    threads_per_evaluator = max(1, allocated_cpus // args.evaluators_per_gpu)
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        child_environment[variable] = str(threads_per_evaluator)
    processes: list[tuple[int, subprocess.Popen[str]]] = []
    for local_index, logical_lane in enumerate(logical_lanes):
        if local_index and args.startup_stagger_seconds:
            time.sleep(args.startup_stagger_seconds)
        command = [
            str(runtime["python"]),
            str(Path(__file__).resolve()),
            "lane",
            "--run-manifest",
            str(args.run_manifest.resolve()),
            "--lane-count",
            str(logical_lane_count),
            "--lane",
            str(logical_lane),
        ]
        print(
            f"PACKED_EVAL_START physical_lane={physical_lane}/{args.physical_lane_count} "
            f"local_evaluator={local_index}/{args.evaluators_per_gpu} "
            f"logical_lane={logical_lane}/{logical_lane_count}",
            flush=True,
        )
        processes.append(
            (
                logical_lane,
                subprocess.Popen(command, cwd=runtime["repo_root"], env=child_environment),
            )
        )
    running = dict(processes)
    while running:
        for logical_lane, process in tuple(running.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            del running[logical_lane]
            if return_code:
                # Release the physical GPU promptly. The durable monitor will
                # resubmit this packed lane elsewhere, and every completed item
                # from healthy siblings will be skipped on restart.
                for sibling in running.values():
                    sibling.terminate()
                for sibling in running.values():
                    try:
                        sibling.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        sibling.kill()
                        sibling.wait()
                raise RuntimeError(
                    f"Packed evaluator failure on physical lane {physical_lane}: "
                    f"logical_lane={logical_lane}, return_code={return_code}"
                )
        if running:
            time.sleep(1)
    # Child lanes update the shared summary opportunistically. Refresh once
    # after all three have exited so concurrent atomic replacements cannot
    # leave a stale-but-valid partial summary behind.
    run_manifest = load_run_manifest(args.run_manifest)
    summary = write_summary(run_manifest, require_complete=False)
    print(
        f"PACKED_EVAL_COMPLETE physical_lane={physical_lane}/{args.physical_lane_count} "
        f"logical_lanes={list(logical_lanes)} "
        f"complete={summary['completed_work_items']}/{summary['expected_work_items']}",
        flush=True,
    )


def command_status(args: argparse.Namespace) -> None:
    run_manifest = load_run_manifest(args.run_manifest)
    summary = write_summary(run_manifest, require_complete=False)
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "protocol",
                    "expected_work_items",
                    "completed_work_items",
                    "complete",
                    "missing_work_items",
                )
            },
            indent=2,
        )
    )


def command_summarize(args: argparse.Namespace) -> None:
    run_manifest = load_run_manifest(args.run_manifest)
    summary = write_summary(run_manifest, require_complete=not args.allow_incomplete)
    print(json.dumps(summary, indent=2, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)

    launch = subparsers.add_parser("launch", help="Freeze a run manifest and submit durable lanes.")
    launch.add_argument("--spec", type=Path, required=True)
    launch.add_argument("--output-root", type=Path, required=True)
    launch.add_argument(
        "--policy-root",
        type=Path,
        default=os.environ.get("FASTWAM_POLICY_ROOT"),
        help="Fast-WAM integration root (or FASTWAM_POLICY_ROOT).",
    )
    launch.add_argument(
        "--robodojo-root",
        type=Path,
        default=os.environ.get("ROBODOJO_ROOT"),
        help="RoboDojo checkout (or ROBODOJO_ROOT).",
    )
    launch.add_argument(
        "--hf-home",
        type=Path,
        default=os.environ.get("HF_HOME"),
        help="Hugging Face cache root (or HF_HOME).",
    )
    launch.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="Interpreter propagated to Slurm workers; defaults to the current interpreter.",
    )
    launch.add_argument(
        "--native-lib-dir",
        type=Path,
        default=os.environ.get("VVLA_NATIVE_LIB_DIR"),
        help="Optional directory prepended to LD_LIBRARY_PATH on workers.",
    )
    launch.add_argument("--partition", choices=("normal", "preempt"), default="preempt")
    launch.add_argument("--account", default=os.environ.get("SLURM_ACCOUNT"))
    launch.add_argument("--qos", default=os.environ.get("SLURM_QOS"))
    launch.add_argument("--module", default=os.environ.get("VVLA_SLURM_MODULE"))
    launch.add_argument("--lanes", type=int, default=4, help="Number of physical GPU lanes.")
    launch.add_argument(
        "--evaluators-per-gpu",
        type=int,
        default=1,
        help="Independent native RoboDojo evaluator processes colocated on each GPU.",
    )
    launch.add_argument(
        "--coalesced-items-per-evaluator",
        type=int,
        default=1,
        help=(
            "Run this many compatible layout items in independent RoboDojo simulator "
            "shards served by one shared Fast-WAM replica (for example, 3 * vec6)."
        ),
    )
    launch.add_argument(
        "--startup-stagger-seconds",
        type=int,
        default=30,
        help="Delay between starting colocated evaluator processes.",
    )
    launch.add_argument("--cpus-per-task", type=int)
    launch.add_argument("--memory", help="Slurm memory request, for example 300G.")
    launch.add_argument(
        "--policy-label",
        action="append",
        default=None,
        help="Launch only the named policy from a multi-policy spec; may be repeated.",
    )
    launch.add_argument("--time", default="12:00:00")
    launch.add_argument("--exclude", default=os.environ.get("SLURM_EXCLUDE", ""))
    launch.add_argument("--dry-run", action="store_true")
    launch.set_defaults(function=command_launch)

    lane = subparsers.add_parser("lane", help=argparse.SUPPRESS)
    lane.add_argument("--run-manifest", type=Path, required=True)
    lane.add_argument("--lane-count", type=int, required=True)
    lane.add_argument("--lane", type=int)
    lane.set_defaults(function=command_lane)

    packed_lane = subparsers.add_parser("packed-lane", help=argparse.SUPPRESS)
    packed_lane.add_argument("--run-manifest", type=Path, required=True)
    packed_lane.add_argument("--physical-lane-count", type=int, required=True)
    packed_lane.add_argument("--physical-lane", type=int)
    packed_lane.add_argument("--evaluators-per-gpu", type=int, required=True)
    packed_lane.add_argument("--startup-stagger-seconds", type=int, default=0)
    packed_lane.set_defaults(function=command_packed_lane)

    coalesced_lane = subparsers.add_parser("coalesced-lane", help=argparse.SUPPRESS)
    coalesced_lane.add_argument("--run-manifest", type=Path, required=True)
    coalesced_lane.add_argument("--lane-count", type=int, required=True)
    coalesced_lane.add_argument("--lane", type=int)
    coalesced_lane.add_argument("--items-per-evaluator", type=int, required=True)
    coalesced_lane.set_defaults(function=command_coalesced_lane)

    coalesced_plan_lane = subparsers.add_parser("coalesced-plan-lane", help=argparse.SUPPRESS)
    coalesced_plan_lane.add_argument("--run-manifest", type=Path, required=True)
    coalesced_plan_lane.add_argument("--plan", type=Path, required=True)
    coalesced_plan_lane.add_argument("--lane-count", type=int, required=True)
    coalesced_plan_lane.add_argument("--lane", type=int)
    coalesced_plan_lane.set_defaults(function=command_coalesced_plan_lane)

    freeze_missing_plan = subparsers.add_parser("freeze-missing-plan", help=argparse.SUPPRESS)
    freeze_missing_plan.add_argument("--run-manifest", type=Path, required=True)
    freeze_missing_plan.add_argument("--plan", type=Path, required=True)
    freeze_missing_plan.add_argument("--items-per-evaluator", type=int, required=True)
    freeze_missing_plan.set_defaults(function=command_freeze_missing_plan)

    status = subparsers.add_parser("status", help="Refresh and print durable completion status.")
    status.add_argument("--run-manifest", type=Path, required=True)
    status.set_defaults(function=command_status)

    summarize = subparsers.add_parser("summarize", help="Write JSON/Markdown layout aggregates.")
    summarize.add_argument("--run-manifest", type=Path, required=True)
    summarize.add_argument("--allow-incomplete", action="store_true")
    summarize.set_defaults(function=command_summarize)
    return result


def main() -> None:
    args = parser().parse_args()
    if hasattr(args, "lanes") and args.lanes <= 0:
        raise ValueError("--lanes must be positive.")
    if hasattr(args, "evaluators_per_gpu") and args.evaluators_per_gpu <= 0:
        raise ValueError("--evaluators-per-gpu must be positive.")
    if hasattr(args, "coalesced_items_per_evaluator") and args.coalesced_items_per_evaluator <= 0:
        raise ValueError("--coalesced-items-per-evaluator must be positive.")
    if hasattr(args, "items_per_evaluator") and args.items_per_evaluator <= 0:
        raise ValueError("--items-per-evaluator must be positive.")
    if hasattr(args, "startup_stagger_seconds") and args.startup_stagger_seconds < 0:
        raise ValueError("--startup-stagger-seconds must be non-negative.")
    args.function(args)


if __name__ == "__main__":
    main()
