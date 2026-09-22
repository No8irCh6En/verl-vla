import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parents[2] / "scripts/eval/mainline/monitor_split_eval.py"
SPEC = importlib.util.spec_from_file_location("split_eval_monitor", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_relaunch_arrays_replace_stale_repairs_as_latest_lane_jobs():
    state = {
        "lane_jobs": {
            "0": [
                {"job_ref": "100_0", "partition": "normal"},
                {"job_ref": "101", "partition": "normal", "missing_items": 6},
            ],
            "1": [{"job_ref": "100_1", "partition": "normal"}],
            "2": [{"job_ref": "200_0", "partition": "preempt"}],
        }
    }

    changed = MODULE.register_array_launches(
        state,
        normal_lanes=2,
        preempt_lanes=1,
        normal_array_job="300",
        preempt_array_job="400",
    )

    assert changed is True
    assert state["lane_jobs"]["0"][-1] == {"job_ref": "300_0", "partition": "normal"}
    assert state["lane_jobs"]["1"][-1] == {"job_ref": "300_1", "partition": "normal"}
    assert state["lane_jobs"]["2"][-1] == {"job_ref": "400_0", "partition": "preempt"}


def test_registering_same_array_twice_is_idempotent():
    state = {"lane_jobs": {"0": [{"job_ref": "300_0", "partition": "normal"}]}}

    changed = MODULE.register_array_launches(
        state,
        normal_lanes=1,
        preempt_lanes=0,
        normal_array_job="300",
        preempt_array_job=None,
    )

    assert changed is False
    assert len(state["lane_jobs"]["0"]) == 1


def test_explicit_recovery_jobs_become_latest_lane_owners():
    state = {
        "lane_jobs": {
            "0": [{"job_ref": "300_0", "partition": "normal"}],
            "1": [{"job_ref": "300_1", "partition": "normal"}],
        }
    }

    changed = MODULE.register_explicit_lane_jobs(
        state,
        [MODULE.parse_lane_job("0=441:normal"), MODULE.parse_lane_job("1=901_2:preempt")],
        physical_lane_count=2,
    )

    assert changed is True
    assert state["lane_jobs"]["0"][-1] == {"job_ref": "441", "partition": "normal"}
    assert state["lane_jobs"]["1"][-1] == {"job_ref": "901_2", "partition": "preempt"}
