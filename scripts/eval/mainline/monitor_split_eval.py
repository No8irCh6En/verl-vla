#!/usr/bin/env python3
"""Keep fixed-eval split lanes alive until every durable item is complete.

Each evaluator lane owns work-item indices modulo the global lane count.  This
monitor only resubmits a lane when that lane still owns missing work and its
latest Slurm job is no longer pending/running.  Completed items are durable and
are skipped by ``eval.py lane`` on restart.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
PYTHON = Path(sys.executable)
RUNNER = REPO_ROOT / "scripts/eval/mainline/eval.py"
ACTIVE_STATES = {
    "CONFIGURING",
    "COMPLETING",
    "PENDING",
    "REQUEUED",
    "RESIZING",
    "RUNNING",
    "SUSPENDED",
}


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def slurm_state(job_ref: str) -> str | None:
    result = subprocess.run(
        ["squeue", "-h", "-j", job_ref, "-o", "%T"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    states = [line.strip().upper() for line in result.stdout.splitlines() if line.strip()]
    if any(state in ACTIVE_STATES for state in states):
        return next(state for state in states if state in ACTIVE_STATES)
    return states[0] if states else None


def submit_physical_lane(
    *, manifest: Path, output_root: Path, physical_lane: int,
    physical_lane_count: int, evaluators_per_gpu: int,
    coalesced_items_per_evaluator: int,
    coalesced_plan: Path | None,
    preferred_partition: str, time_limit: str,
) -> tuple[str, str]:
    run_manifest = json.loads(manifest.read_text(encoding="utf-8"))
    runtime = run_manifest["runtime"]
    slurm = runtime["slurm"]
    repo_root = Path(runtime["repo_root"])
    python = Path(runtime["python"])
    runner = repo_root / "scripts/eval/mainline/eval.py"
    native_lib_dir = runtime.get("native_lib_dir")
    module_prefix = ""
    if slurm.get("module"):
        module_prefix = f"module load {slurm['module']}; "
    native_prefix = ""
    if native_lib_dir:
        native_prefix = f"export LD_LIBRARY_PATH={native_lib_dir}:${{LD_LIBRARY_PATH:-}}; "
    partitions = [preferred_partition]
    alternate = "preempt" if preferred_partition == "normal" else "normal"
    if alternate not in partitions:
        partitions.append(alternate)
    # Slurm exposes a single allocated device through CUDA_VISIBLE_DEVICES, so
    # CUDA/PhysX/Kit must use its process-local ordinal rather than the node's
    # physical GPU number (which may be 1..7 and is invalid in the cgroup).
    gpu_binding = "export VERL_VLA_PHYSICAL_GPU_ID=0; "
    if coalesced_plan is not None:
        wrap = (
            f"set -Eeuo pipefail; {module_prefix}"
            f"{gpu_binding}"
            f"{native_prefix}"
            f"cd {repo_root}; exec {python} {runner} coalesced-plan-lane "
            f"--run-manifest {manifest} --plan {coalesced_plan} "
            f"--lane-count {physical_lane_count} --lane={physical_lane}"
        )
        cpus_per_task = 32
        memory = "220G"
    elif coalesced_items_per_evaluator > 1:
        wrap = (
            f"set -Eeuo pipefail; {module_prefix}"
            f"{gpu_binding}"
            f"{native_prefix}"
            f"cd {repo_root}; exec {python} {runner} coalesced-lane "
            f"--run-manifest {manifest} --lane-count {physical_lane_count} "
            f"--lane={physical_lane} "
            f"--items-per-evaluator {coalesced_items_per_evaluator}"
        )
        cpus_per_task = 32
        memory = "220G"
    else:
        wrap = (
            f"set -Eeuo pipefail; {module_prefix}"
            f"{gpu_binding}"
            f"{native_prefix}"
            f"cd {repo_root}; exec {python} {runner} packed-lane "
            f"--run-manifest {manifest} --physical-lane-count {physical_lane_count} "
            f"--physical-lane={physical_lane} --evaluators-per-gpu {evaluators_per_gpu} "
            f"--startup-stagger-seconds {30 if evaluators_per_gpu > 1 else 0}"
        )
        cpus_per_task = 57 if evaluators_per_gpu == 3 else 32
        memory = "300G" if evaluators_per_gpu == 3 else "220G"
    errors: list[str] = []
    for partition_role in partitions:
        partition = slurm.get(f"{partition_role}_partition", partition_role)
        qos = slurm.get(f"{partition_role}_qos") or (
            slurm.get("qos") if partition_role == preferred_partition else ""
        )
        command = [
            "sbatch", "--parsable", f"--partition={partition}",
            "--nodes=1", "--gpus-per-node=1", f"--cpus-per-task={cpus_per_task}",
            f"--mem={memory}", f"--time={time_limit}", "--requeue",
            f"--job-name=eval-repair-{physical_lane:02d}",
            f"--output={output_root}/logs/repair_lane{physical_lane:02d}_%j.out",
            f"--wrap={wrap}",
        ]
        if slurm.get("account"):
            command.insert(3, f"--account={slurm['account']}")
        if qos:
            command.insert(4, f"--qos={qos}")
        if slurm.get("exclude"):
            command.insert(-2, f"--exclude={slurm['exclude']}")
        result = subprocess.run(command, text=True, capture_output=True)
        if result.returncode == 0:
            return result.stdout.strip().split(";", 1)[0], partition_role
        errors.append(f"{partition_role}/{partition}: {result.stderr.strip()}")
    raise RuntimeError("; ".join(errors))


def missing_physical_lanes(
    manifest: dict[str, Any], output_root: Path,
    physical_lane_count: int, evaluators_per_gpu: int,
    coalesced_items_per_evaluator: int = 1,
    coalesced_plan: dict[str, Any] | None = None,
) -> dict[int, int]:
    counts = {lane: 0 for lane in range(physical_lane_count)}
    if coalesced_plan is not None:
        item_by_id = {item["item_id"]: item for item in manifest["work_items"]}
        for bundle_index, bundle in enumerate(coalesced_plan["bundles"]):
            missing = sum(
                not (output_root / "items" / item_by_id[item_id]["item_id"] / "COMPLETE.json").is_file()
                for item_id in bundle
            )
            if missing:
                counts[bundle_index % physical_lane_count] += missing
        return counts
    if coalesced_items_per_evaluator > 1:
        groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        order: list[tuple[Any, ...]] = []
        for item in manifest["work_items"]:
            key = (
                item["policy_label"], item["task_name"], int(item["task_id"]),
                int(item["suite_id"]), int(item["repeat"]),
            )
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(item)
        bundle_index = 0
        for key in order:
            items = groups[key]
            for start in range(0, len(items), coalesced_items_per_evaluator):
                bundle = items[start : start + coalesced_items_per_evaluator]
                missing = sum(
                    not (output_root / "items" / item["item_id"] / "COMPLETE.json").is_file()
                    for item in bundle
                )
                if missing:
                    counts[bundle_index % physical_lane_count] += missing
                bundle_index += 1
        return counts
    logical_lane_count = physical_lane_count * evaluators_per_gpu
    for index, item in enumerate(manifest["work_items"]):
        marker = output_root / "items" / item["item_id"] / "COMPLETE.json"
        if not marker.is_file():
            logical_lane = index % logical_lane_count
            counts[logical_lane // evaluators_per_gpu] += 1
    return counts


def register_array_launches(
    state: dict[str, Any],
    *,
    normal_lanes: int,
    preempt_lanes: int,
    normal_array_job: str | None,
    preempt_array_job: str | None,
) -> bool:
    """Make a newly launched array the latest owner of each physical lane.

    A split evaluation can be relaunched into the same durable output after an
    environment failure.  Keeping the old array as the latest state makes the
    monitor immediately submit duplicate repair jobs alongside the new array.
    """

    changed = False
    launches = (
        (0, normal_lanes, normal_array_job, "normal"),
        (normal_lanes, preempt_lanes, preempt_array_job, "preempt"),
    )
    for offset, lane_count, array_job, partition in launches:
        if not array_job:
            continue
        for local_lane in range(lane_count):
            lane = offset + local_lane
            record = {"job_ref": f"{array_job}_{local_lane}", "partition": partition}
            history = state["lane_jobs"].setdefault(str(lane), [])
            if not history or history[-1] != record:
                history.append(record)
                changed = True
    return changed


def parse_lane_job(value: str) -> tuple[int, str, str]:
    """Parse ``LANE=JOB_REF:PARTITION_ROLE`` from a recovery launch."""

    try:
        lane_text, job_and_partition = value.split("=", 1)
        job_ref, partition = job_and_partition.rsplit(":", 1)
        lane = int(lane_text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "--lane-job must have the form LANE=JOB_REF:normal|preempt"
        ) from exc
    if lane < 0 or not job_ref or partition not in {"normal", "preempt"}:
        raise argparse.ArgumentTypeError(
            "--lane-job must have the form LANE=JOB_REF:normal|preempt"
        )
    return lane, job_ref, partition


def register_explicit_lane_jobs(
    state: dict[str, Any],
    lane_jobs: list[tuple[int, str, str]],
    *,
    physical_lane_count: int,
) -> bool:
    """Register individually submitted recovery jobs as current lane owners."""

    changed = False
    seen: set[int] = set()
    for lane, job_ref, partition in lane_jobs:
        if lane >= physical_lane_count:
            raise ValueError(
                f"Explicit lane {lane} is outside [0, {physical_lane_count})."
            )
        if lane in seen:
            raise ValueError(f"Duplicate explicit --lane-job for lane {lane}.")
        seen.add(lane)
        record = {"job_ref": job_ref, "partition": partition}
        history = state["lane_jobs"].setdefault(str(lane), [])
        if not history or history[-1] != record:
            history.append(record)
            changed = True
    return changed


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--run-manifest", type=Path, required=True)
    result.add_argument("--normal-lanes", type=int, required=True)
    result.add_argument("--preempt-lanes", type=int, required=True)
    result.add_argument("--evaluators-per-gpu", type=int, default=1)
    result.add_argument("--coalesced-items-per-evaluator", type=int, default=1)
    result.add_argument("--coalesced-plan", type=Path)
    result.add_argument("--normal-array-job")
    result.add_argument("--preempt-array-job")
    result.add_argument(
        "--lane-job",
        action="append",
        default=[],
        type=parse_lane_job,
        help="Existing recovery job ownership as LANE=JOB_REF:normal|preempt; repeatable.",
    )
    result.add_argument("--time", default="06:00:00")
    result.add_argument("--poll-seconds", type=int, default=60)
    return result


def main() -> None:
    args = parser().parse_args()
    manifest_path = args.run_manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    coalesced_plan_path = args.coalesced_plan.resolve() if args.coalesced_plan else None
    coalesced_plan = (
        json.loads(coalesced_plan_path.read_text(encoding="utf-8"))
        if coalesced_plan_path is not None
        else None
    )
    output_root = Path(manifest["output_root"]).resolve()
    physical_lane_count = args.normal_lanes + args.preempt_lanes
    if (
        physical_lane_count < 1
        or args.evaluators_per_gpu < 1
        or args.coalesced_items_per_evaluator < 1
    ):
        raise ValueError("At least one split lane is required.")
    if args.evaluators_per_gpu > 1 and args.coalesced_items_per_evaluator > 1:
        raise ValueError("Packed and coalesced evaluator modes are mutually exclusive.")
    if coalesced_plan is not None and (
        args.evaluators_per_gpu > 1 or args.coalesced_items_per_evaluator > 1
    ):
        raise ValueError("A frozen coalesced plan cannot be combined with other packed modes.")
    output_root.joinpath("logs").mkdir(parents=True, exist_ok=True)
    lock_stream = output_root.joinpath(".split_eval_monitor.lock").open("w")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"monitor_already_running={output_root}", flush=True)
        return

    state_path = output_root / "SPLIT_MONITOR_STATE.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        expected_topology = {
            "physical_lane_count": physical_lane_count,
            "evaluators_per_gpu": args.evaluators_per_gpu,
            "coalesced_items_per_evaluator": args.coalesced_items_per_evaluator,
            "coalesced_plan": str(coalesced_plan_path) if coalesced_plan_path else None,
            "normal_lanes": args.normal_lanes,
            "preempt_lanes": args.preempt_lanes,
        }
        mismatches = {
            key: (state.get(key), value)
            for key, value in expected_topology.items()
            if state.get(key) != value
        }
        if state.get("run_manifest") != str(manifest_path) or mismatches:
            raise ValueError(
                "Existing split-monitor state does not match this launch: "
                f"run_manifest={state.get('run_manifest')!r}, topology={mismatches}"
            )
        if register_array_launches(
            state,
            normal_lanes=args.normal_lanes,
            preempt_lanes=args.preempt_lanes,
            normal_array_job=args.normal_array_job,
            preempt_array_job=args.preempt_array_job,
        ) | register_explicit_lane_jobs(
            state,
            args.lane_job,
            physical_lane_count=physical_lane_count,
        ):
            atomic_json(state_path, state)
    else:
        lane_jobs: dict[str, list[dict[str, str]]] = {
            str(lane): [] for lane in range(physical_lane_count)
        }
        if args.normal_array_job:
            for lane in range(args.normal_lanes):
                lane_jobs[str(lane)].append(
                    {"job_ref": f"{args.normal_array_job}_{lane}", "partition": "normal"}
                )
        if args.preempt_array_job:
            for local_lane in range(args.preempt_lanes):
                lane = args.normal_lanes + local_lane
                lane_jobs[str(lane)].append(
                    {"job_ref": f"{args.preempt_array_job}_{local_lane}", "partition": "preempt"}
                )
        state = {
            "schema_version": 1,
            "run_manifest": str(manifest_path),
            "physical_lane_count": physical_lane_count,
            "evaluators_per_gpu": args.evaluators_per_gpu,
            "coalesced_items_per_evaluator": args.coalesced_items_per_evaluator,
            "coalesced_plan": str(coalesced_plan_path) if coalesced_plan_path else None,
            "normal_lanes": args.normal_lanes,
            "preempt_lanes": args.preempt_lanes,
            "lane_jobs": lane_jobs,
            "resubmissions": [],
        }
        register_array_launches(
            state,
            normal_lanes=args.normal_lanes,
            preempt_lanes=args.preempt_lanes,
            normal_array_job=args.normal_array_job,
            preempt_array_job=args.preempt_array_job,
        )
        register_explicit_lane_jobs(
            state,
            args.lane_job,
            physical_lane_count=physical_lane_count,
        )
        atomic_json(state_path, state)

    while True:
        missing = missing_physical_lanes(
            manifest,
            output_root,
            physical_lane_count,
            args.evaluators_per_gpu,
            args.coalesced_items_per_evaluator,
            coalesced_plan,
        )
        total_missing = sum(missing.values())
        print(
            f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} missing={total_missing} "
            f"per_lane={','.join(f'{lane}:{count}' for lane, count in missing.items())}",
            flush=True,
        )
        if total_missing == 0:
            runtime = manifest["runtime"]
            subprocess.run(
                [
                    str(runtime["python"]),
                    str(Path(runtime["repo_root"]) / "scripts/eval/mainline/eval.py"),
                    "summarize",
                    "--run-manifest",
                    str(manifest_path),
                ],
                check=True,
            )
            state["completed_at_unix_s"] = time.time()
            atomic_json(state_path, state)
            return

        for lane, count in missing.items():
            if count == 0:
                continue
            history = state["lane_jobs"].setdefault(str(lane), [])
            latest = history[-1] if history else None
            current_state = slurm_state(latest["job_ref"]) if latest else None
            if current_state in ACTIVE_STATES:
                continue
            preferred = "normal" if lane < args.normal_lanes else "preempt"
            try:
                job_id, partition = submit_physical_lane(
                    manifest=manifest_path,
                    output_root=output_root,
                    physical_lane=lane,
                    physical_lane_count=physical_lane_count,
                    evaluators_per_gpu=args.evaluators_per_gpu,
                    coalesced_items_per_evaluator=args.coalesced_items_per_evaluator,
                    coalesced_plan=coalesced_plan_path,
                    preferred_partition=preferred,
                    time_limit=args.time,
                )
            except RuntimeError as error:
                print(f"lane_resubmit_deferred lane={lane} reason={error}", flush=True)
                continue
            record = {
                "job_ref": job_id,
                "partition": partition,
                "submitted_at_unix_s": time.time(),
                "missing_items": count,
            }
            history.append(record)
            state["resubmissions"].append({"lane": lane, **record})
            atomic_json(state_path, state)
            print(
                f"lane_resubmitted lane={lane}/{physical_lane_count} job={job_id} "
                f"partition={partition} missing={count}",
                flush=True,
            )
        atomic_json(state_path, state)
        time.sleep(max(15, args.poll_seconds))


if __name__ == "__main__":
    main()
