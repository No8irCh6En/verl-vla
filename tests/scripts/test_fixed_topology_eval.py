import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

SCRIPT = Path(__file__).parents[2] / "scripts/eval/mainline/eval.py"
SPEC = importlib.util.spec_from_file_location("fixed_topology_eval", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_python_environment_entrypoint_symlink_is_not_dereferenced(tmp_path):
    entrypoint = tmp_path / "overlay-python"
    entrypoint.symlink_to(Path(sys.executable))

    resolved = MODULE._resolve_python_entrypoint(entrypoint)

    assert resolved == Path(os.path.abspath(entrypoint))
    assert resolved.is_symlink()


def _experiment_spec(tmp_path: Path):
    return {
        "schema_version": 1,
        "name": "unit_targeted_replay",
        "purpose": "targeted_replay",
        "repeats": 3,
        "policies": [
            {
                "label": "left",
                "model_root": str(tmp_path / "model"),
                "checkpoint_sha256": "0" * 64,
                "verl_actor_checkpoint": str(tmp_path / "left"),
                "expected_source_verl_global_step": 1,
            },
            {
                "label": "right",
                "model_root": str(tmp_path / "model"),
                "checkpoint_sha256": "0" * 64,
                "verl_actor_checkpoint": str(tmp_path / "right"),
                "expected_source_verl_global_step": 2,
            },
        ],
        "targets": [
            {
                "task_name": "task_a",
                "task_id": 0,
                "suite_id": 0,
                "layout_id": 10,
                "policy_seeds": [100, 101],
            },
            {
                "task_name": "task_a",
                "task_id": 0,
                "suite_id": 0,
                "layout_id": 11,
                "policy_seeds": [110, 111],
            },
        ],
        "topology": {
            "num_envs": 2,
            "policy_microbatch_size": 2,
            "full_observation": True,
            "record_video": True,
            "max_policy_calls": 34,
            "reset_max_attempts": 10,
            "process_score_reduction": "terminal",
        },
    }


def test_spec_relative_paths_are_resolved_from_the_spec_directory(tmp_path):
    spec_dir = tmp_path / "experiment"
    model_root = spec_dir / "model"
    actor_root = spec_dir / "checkpoints/global_step_1"
    model_root.mkdir(parents=True)
    actor_root.mkdir(parents=True)
    spec = _experiment_spec(tmp_path)
    spec["policies"] = [
        {
            **spec["policies"][0],
            "model_root": "model",
            "verl_actor_checkpoint": "checkpoints/global_step_1",
        }
    ]

    normalized = MODULE._resolve_spec_paths(spec, spec_dir)

    assert normalized["policies"][0]["model_root"] == str(model_root.resolve())
    assert normalized["policies"][0]["verl_actor_checkpoint"] == str(
        actor_root.resolve()
    )
    assert spec["policies"][0]["model_root"] == "model"


def test_fixed_replay_plan_and_layout_aggregate(tmp_path):
    spec = _experiment_spec(tmp_path)
    MODULE.validate_spec(spec, check_paths=False)
    items = MODULE.build_work_items(spec)
    assert len(items) == 2 * 2 * 3
    output_root = tmp_path / "output"
    run_manifest = {
        "schema_version": 1,
        "spec_path": str(tmp_path / "spec.json"),
        "spec_sha256": "1" * 64,
        "output_root": str(output_root),
        "spec": spec,
        "work_items": items,
    }

    # Left succeeds on both seeds in layout 10 and one seed in layout 11.
    # Right loses layout 10 but gains the second seed in layout 11. Repeating
    # three times must preserve layout as the top-level reporting unit.
    success_seeds = {
        ("left", 10): {100, 101},
        ("left", 11): {110},
        ("right", 10): {100},
        ("right", 11): {110, 111},
    }
    for item in items:
        attempt_dir = MODULE.item_root(run_manifest, item) / "attempts/attempt_01"
        videos = attempt_dir / "videos"
        videos.mkdir(parents=True)
        (videos / "case.mp4").write_bytes(b"video")
        cases = []
        for seed in item["policy_seeds"]:
            success = seed in success_seeds[(item["policy_label"], item["layout_id"])]
            cases.append(
                {
                    "layout_id": item["layout_id"],
                    "policy_seed": seed,
                    "success": success,
                    "score": 1.0 if success else 0.2,
                    "policy_calls": 10,
                    "done": True,
                }
            )
        metrics = {
            "task_name": item["task_name"],
            "in_memory_rl_checkpoint": {
                "source_verl_global_step": 1 if item["policy_label"] == "left" else 2,
                "source_verl_actor_state": str(
                    (
                        Path(spec["policies"][0 if item["policy_label"] == "left" else 1]["verl_actor_checkpoint"])
                        / "actor/model_world_size_1_rank_0.pt"
                    ).resolve()
                ),
            },
            "num_cases": 2,
            "robodojo_eval_seed": 0,
            "num_envs": 2,
            "policy_microbatch_size": 2,
            "process_score_reduction": "terminal",
            "action_chunk_size": 24,
            "max_policy_calls": 34,
            "all_cases_done": True,
            "cases": cases,
        }
        metrics_path = attempt_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
        marker = {
            "schema_version": 1,
            "item_id": item["item_id"],
            "attempt": 1,
            "metrics_relative_path": str(metrics_path.relative_to(output_root)),
        }
        marker_path = MODULE.complete_path(run_manifest, item)
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        marker_path.write_text(json.dumps(marker), encoding="utf-8")

    summary = MODULE.summarize_run(run_manifest, require_complete=True)
    assert summary["complete"] is True
    assert summary["completed_work_items"] == 12
    assert [row["completed_trials"] for row in summary["policies"]] == [12, 12]
    assert [row["successes"] for row in summary["policies"]] == [9, 9]
    comparison = summary["comparisons"][0]
    assert comparison["paired_trials"] == 12
    assert comparison["success_to_fail"] == 3
    assert comparison["fail_to_success"] == 3
    assert comparison["layouts_regressed"] == 1
    assert comparison["layouts_improved"] == 1


def test_three_vec6_evaluators_per_gpu_have_disjoint_lane_ownership(tmp_path):
    spec = _experiment_spec(tmp_path)
    spec["repeats"] = 1
    spec["topology"]["num_envs"] = 6
    spec["topology"]["policy_microbatch_size"] = 2
    for target in spec["targets"]:
        seed = target["policy_seeds"][0]
        target["policy_seeds"] = list(range(seed, seed + 6))
    MODULE.validate_spec(spec, check_paths=False)
    assert len(MODULE.build_work_items(spec)) == 4

    ownership = [MODULE.packed_logical_lanes(physical_lane, 4, 3) for physical_lane in range(4)]
    assert ownership == [
        (0, 1, 2),
        (3, 4, 5),
        (6, 7, 8),
        (9, 10, 11),
    ]
    assert sorted(lane for lanes in ownership for lane in lanes) == list(range(12))


def test_packed_gpu_lane_starts_three_independent_logical_lanes(tmp_path, monkeypatch):
    launched = []

    class FinishedProcess:
        def __init__(self, command, **kwargs):
            launched.append((command, kwargs))

        def poll(self):
            return 0

    monkeypatch.setattr(MODULE.subprocess, "Popen", FinishedProcess)
    monkeypatch.setattr(
        MODULE,
        "load_run_manifest",
        lambda _: {
            "runtime": {
                "python": "/usr/bin/python3",
                "repo_root": str(Path(__file__).parents[2]),
            }
        },
    )
    monkeypatch.setattr(
        MODULE,
        "write_summary",
        lambda *_args, **_kwargs: {"completed_work_items": 3, "expected_work_items": 3},
    )
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "1")
    MODULE.command_packed_lane(
        SimpleNamespace(
            run_manifest=tmp_path / "RUN_MANIFEST.json",
            physical_lane_count=4,
            physical_lane=None,
            evaluators_per_gpu=3,
            startup_stagger_seconds=0,
        )
    )
    assert len(launched) == 3
    commands = [entry[0] for entry in launched]
    assert [command[command.index("--lane") + 1] for command in commands] == ["3", "4", "5"]
    assert all(command[command.index("--lane-count") + 1] == "12" for command in commands)
    assert all("SLURM_ARRAY_TASK_ID" not in entry[1]["env"] for entry in launched)


def test_coalesced_bundles_pack_distinct_layouts_within_repeat(tmp_path):
    spec = _experiment_spec(tmp_path)
    # Add a third layout so every repeat becomes exactly one 3 * vec2 bundle.
    spec["targets"].append(
        {
            "task_name": "task_a",
            "task_id": 0,
            "suite_id": 0,
            "layout_id": 12,
            "policy_seeds": [120, 121],
        }
    )
    items = MODULE.build_work_items(spec)
    run_manifest = {"spec": spec, "work_items": items}
    bundles = MODULE.coalesced_item_bundles(run_manifest, 3)

    # 2 policies * 3 repeats, with one mixed-layout vec6 bundle per pair.
    assert len(bundles) == 6
    for bundle in bundles:
        assert len(bundle) == 3
        assert len({item["repeat"] for item in bundle}) == 1
        assert {item["layout_id"] for item in bundle} == {10, 11, 12}
        condition_keys = {(item["layout_id"], seed) for item in bundle for seed in item["policy_seeds"]}
        assert len(condition_keys) == 6
