#!/usr/bin/env python3
"""Summarize checkpoint performance from every durable candidate group.

This intentionally includes both accepted and rejected groups.  Reporting only
accepted mixed groups would condition on the training filter and overestimate
the rollout policy's actual success rate.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _records(window: Path) -> list[dict[str, Any]]:
    paths = list((window / "accepted").glob("candidate_*/metadata.json"))
    paths.extend((window / "rejected").glob("candidate_*.json"))
    records = []
    for path in sorted(paths):
        record = _read(path)
        # Files become visible only at the durable commit boundary. Their mtime
        # is therefore a stable completion timestamp for legacy records that
        # predate explicit queue completion events.
        record["_durable_completed_at_unix_s"] = path.stat().st_mtime
        records.append(record)
    candidate_ids = [int(record["candidate"]["candidate_index"]) for record in records]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise RuntimeError(f"Duplicate candidate indices in {window}")
    return records


def _aggregate(records: list[dict[str, Any]], *, window: Path) -> dict[str, Any]:
    manifest = _read(window / "manifest.json")
    expected_theta = int(manifest["policy"]["rollout_policy_version"])
    expected_group_size = int(manifest["group_size"])
    task_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    successes: list[int] = []
    scores: list[float] = []
    group_classes = {"all_failure": 0, "mixed": 0, "all_success": 0}

    for record in records:
        group = record["group_record"]
        theta = int(group["rollout_policy_version"])
        if theta != expected_theta or int(record["policy"]["rollout_policy_version"]) != expected_theta:
            raise RuntimeError(f"Mixed theta_old identities in {window}: expected {expected_theta}, found {theta}")
        binary = [int(value) for value in group["binary_success_vector"]]
        process = [float(value) for value in group["process_score_vector"]]
        if len(binary) != expected_group_size or len(process) != expected_group_size:
            raise RuntimeError(
                f"Candidate {group['group_id']} is not G={expected_group_size}: "
                f"success={len(binary)}, score={len(process)}"
            )
        if any(value not in {0, 1} for value in binary) or any(not math.isfinite(value) for value in process):
            raise RuntimeError(f"Invalid success/score vector in {group['group_id']}")
        k = sum(binary)
        classification = "all_failure" if k == 0 else "all_success" if k == expected_group_size else "mixed"
        group_classes[classification] += 1
        row = {"binary": binary, "scores": process, "classification": classification}
        task_rows[str(group["task_name"])].append(row)
        successes.extend(binary)
        scores.extend(process)

    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        row_binary = [value for row in rows for value in row["binary"]]
        row_scores = [value for row in rows for value in row["scores"]]
        mixed = sum(row["classification"] == "mixed" for row in rows)
        all_failure = sum(row["classification"] == "all_failure" for row in rows)
        all_success = sum(row["classification"] == "all_success" for row in rows)
        return {
            "candidate_groups": len(rows),
            "trajectories": len(row_binary),
            "full_successes": sum(row_binary),
            "full_success_rate": sum(row_binary) / len(row_binary) if row_binary else None,
            "mean_process_score": statistics.fmean(row_scores) if row_scores else None,
            "all_failure_groups": all_failure,
            "mixed_groups": mixed,
            "all_success_groups": all_success,
            "mixed_group_rate": mixed / len(rows) if rows else None,
        }

    accepted_count = sum(bool(record["group_record"]["informative"]) for record in records)
    completed = len(records)
    skipped_count = len(list((window / "skipped").glob("candidate_*.json")))
    lease_events_path = window / "LEASE_EVENTS.jsonl"
    lease_events = []
    if lease_events_path.exists():
        lease_events = [json.loads(line) for line in lease_events_path.read_text(encoding="utf-8").splitlines()]
    queue_candidate_indices = {
        int(event["candidate_index"])
        for event in lease_events
        if event.get("event") in {"claim", "reclaim"} and event.get("candidate_index") is not None
    }
    queue_records = [
        record
        for record in records
        if int(record["candidate"]["candidate_index"]) in queue_candidate_indices
    ]
    first_claim_at = min(
        (
            float(event["at_unix_s"])
            for event in lease_events
            if event.get("event") in {"claim", "reclaim"}
        ),
        default=None,
    )
    last_queue_completion_at = max(
        (float(record["_durable_completed_at_unix_s"]) for record in queue_records),
        default=None,
    )
    collection_elapsed_s = (
        max(0.0, last_queue_completion_at - first_claim_at)
        if first_claim_at is not None and last_queue_completion_at is not None
        else None
    )
    queue_completion_times = sorted(
        float(record["_durable_completed_at_unix_s"])
        for record in queue_records
    )

    def recent_rate(period_s: float) -> float | None:
        if first_claim_at is None or last_queue_completion_at is None:
            return None
        interval_start = max(float(first_claim_at), last_queue_completion_at - period_s)
        interval_s = last_queue_completion_at - interval_start
        if interval_s <= 0:
            return None
        recent_count = sum(timestamp > interval_start for timestamp in queue_completion_times)
        return recent_count * 3600.0 / interval_s

    queue_metrics = {
        "enabled": bool(lease_events),
        "claim_events": sum(event["event"] == "claim" for event in lease_events),
        "reclaim_events": sum(event["event"] == "reclaim" for event in lease_events),
        "unique_worker_owners": len({event["owner_id"] for event in lease_events}),
        "completed_claimed_candidate_groups": len(queue_records),
        "collection_elapsed_s": collection_elapsed_s,
        "candidate_groups_per_hour": (
            len(queue_records) * 3600.0 / collection_elapsed_s
            if collection_elapsed_s and collection_elapsed_s > 0
            else None
        ),
        # Rolling rates expose steady-state collection separately from model
        # and Isaac cold-start overhead.  Anchor them at the latest durable
        # completion so a live in-progress group does not create artificial
        # throughput decay between watcher polls.
        "candidate_groups_per_hour_last_15m": recent_rate(15 * 60),
        "candidate_groups_per_hour_last_30m": recent_rate(30 * 60),
    }
    return {
        "schema_version": 1,
        "generated_at_unix_s": time.time(),
        "window": str(window.resolve()),
        "intended_update": int(manifest["intended_update"]),
        "rollout_policy_version": expected_theta,
        "checkpoint_kind": "trainer_checkpoint" if expected_theta > 0 else "immutable_sft_initialization",
        "estimator_scope": "all_durable_candidate_groups_including_training_rejections",
        "is_fixed_evaluation": False,
        "candidate_groups": completed,
        "skipped_candidate_groups": skipped_count,
        "accepted_mixed_groups": accepted_count,
        "rejected_groups": completed - accepted_count,
        "raw_trajectories": len(successes),
        "full_successes": sum(successes),
        "full_success_rate": sum(successes) / len(successes) if successes else None,
        "mean_process_score": statistics.fmean(scores) if scores else None,
        "group_outcomes": group_classes,
        "group_outcome_fractions": {
            key: value / completed if completed else None for key, value in group_classes.items()
        },
        "shared_queue": queue_metrics,
        "per_task": {task: summarize(rows) for task, rows in sorted(task_rows.items())},
    }


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.2f}%"


def _markdown(summary: dict[str, Any]) -> str:
    queue_throughput = summary["shared_queue"]["candidate_groups_per_hour"]
    queue_throughput_15m = summary["shared_queue"]["candidate_groups_per_hour_last_15m"]
    queue_throughput_30m = summary["shared_queue"]["candidate_groups_per_hour_last_30m"]
    lines = [
        f"# Online rollout estimate: theta{summary['rollout_policy_version']}",
        "",
        "> This is an online estimate over every completed candidate group (accepted and rejected), "
        "not a fixed evaluation. Candidate conditions follow the training proposal schedule.",
        "",
        f"- Intended update: {summary['intended_update']}",
        f"- Candidate groups: {summary['candidate_groups']}",
        f"- Operator-skipped candidates (not rolled/trained): {summary['skipped_candidate_groups']}",
        f"- Raw trajectories: {summary['raw_trajectories']}",
        f"- Full success: {summary['full_successes']}/{summary['raw_trajectories']} "
        f"({_percent(summary['full_success_rate'])})",
        f"- Mean process score: {summary['mean_process_score'] if summary['mean_process_score'] is not None else 'n/a'}",
        f"- Mixed-group yield: {_percent(summary['group_outcome_fractions']['mixed'])}",
        f"- Shared candidate queue: {summary['shared_queue']['enabled']}",
        f"- Candidate groups/hour: {queue_throughput if queue_throughput is not None else 'n/a'}",
        f"- Candidate groups/hour (latest 15m): "
        f"{queue_throughput_15m if queue_throughput_15m is not None else 'n/a'}",
        f"- Candidate groups/hour (latest 30m): "
        f"{queue_throughput_30m if queue_throughput_30m is not None else 'n/a'}",
        f"- Queue reclaims: {summary['shared_queue']['reclaim_events']}",
        "",
        "| Task | Groups | All fail | Mixed | All success | Trajectories | Full success | Mean score | Mixed yield |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for task, row in summary["per_task"].items():
        score = "n/a" if row["mean_process_score"] is None else f"{row['mean_process_score']:.4f}"
        lines.append(
            f"| {task} | {row['candidate_groups']} | {row['all_failure_groups']} | "
            f"{row['mixed_groups']} | {row['all_success_groups']} | {row['trajectories']} | "
            f"{row['full_successes']}/{row['trajectories']} ({_percent(row['full_success_rate'])}) | "
            f"{score} | {_percent(row['mixed_group_rate'])} |"
        )
    return "\n".join(lines) + "\n"


def _run_comparison(collection_run_root: Path) -> tuple[dict[str, Any], str]:
    rows = []
    for path in sorted(collection_run_root.glob("update_*/ONLINE_POLICY_METRICS.json")):
        row = _read(path)
        if int(row.get("candidate_groups", 0)) > 0:
            rows.append(row)
    payload = {
        "schema_version": 1,
        "generated_at_unix_s": time.time(),
        "estimator_scope": "all_durable_candidate_groups_including_training_rejections",
        "is_fixed_evaluation": False,
        "policies": rows,
    }
    lines = [
        "# Online checkpoint comparison",
        "",
        "> These are continuously updated training-schedule rollout estimates, not a fixed paired evaluation.",
        "",
        "| Policy | Candidate groups | Raw trajectories | Full success | Mean score | Mixed yield |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        score = "n/a" if row["mean_process_score"] is None else f"{row['mean_process_score']:.4f}"
        lines.append(
            f"| theta{row['rollout_policy_version']} | {row['candidate_groups']} | {row['raw_trajectories']} | "
            f"{row['full_successes']}/{row['raw_trajectories']} ({_percent(row['full_success_rate'])}) | "
            f"{score} | {_percent(row['group_outcome_fractions']['mixed'])} |"
        )
    return payload, "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("window", type=Path)
    parser.add_argument("--json", dest="json_path", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--run-json", type=Path)
    parser.add_argument("--run-markdown", type=Path)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    summary = _aggregate(_records(args.window), window=args.window)
    json_text = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.json_path:
        _atomic_write(args.json_path, json_text)
    if args.markdown:
        _atomic_write(args.markdown, _markdown(summary))
    if args.run_json or args.run_markdown:
        run_payload, run_markdown = _run_comparison(args.window.parent)
        if args.run_json:
            _atomic_write(args.run_json, json.dumps(run_payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
        if args.run_markdown:
            _atomic_write(args.run_markdown, run_markdown)
    if not args.quiet:
        print(json_text, end="")


if __name__ == "__main__":
    main()
