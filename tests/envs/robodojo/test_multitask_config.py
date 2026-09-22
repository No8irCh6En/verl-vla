from pathlib import Path

import pytest

from verl_vla.envs.robodojo.config import RoboDojoSimulatorConfig


@pytest.mark.parametrize(
    "task_name",
    ["stack_bowls", "push_T", "cover_blocks", "plug_in_charger", "fill_egg_holder"],
)
def test_selected_multitask_sft_tasks_are_valid_robodojo_identifiers(task_name: str) -> None:
    cfg = RoboDojoSimulatorConfig(robodojo_root=str(Path("/tmp/robodojo")), task_name=task_name)
    assert cfg.task_name == task_name


@pytest.mark.parametrize("task_name", ["", "../stack_bowls", "task/name", "task.name"])
def test_unsafe_task_identifiers_are_rejected(task_name: str) -> None:
    with pytest.raises(ValueError, match="task_name"):
        RoboDojoSimulatorConfig(robodojo_root=str(Path("/tmp/robodojo")), task_name=task_name)


def test_worker_stage_task_schedule_is_explicit_and_stage_major() -> None:
    tasks = ["stack_bowls", "push_T", "cover_blocks", "plug_in_charger", "fill_egg_holder"]
    schedule = [
        "stack_bowls",
        "push_T",
        "cover_blocks",
        "plug_in_charger",
        "fill_egg_holder",
        "stack_bowls",
        "push_T",
        "cover_blocks",
        "plug_in_charger",
    ]
    cfg = RoboDojoSimulatorConfig(
        robodojo_root=str(Path("/tmp/robodojo")),
        task_name="stack_bowls",
        task_names=tasks,
        worker_stage_task_schedule=schedule,
    )

    assignments = [
        cfg.task_assignment(worker_rank=rank, worker_world_size=3, stage_id=stage, stage_num=3)
        for stage in range(3)
        for rank in range(3)
    ]

    assert [task for task, _task_id in assignments] == schedule
    assert [task_id for _task, task_id in assignments] == [0, 1, 2, 3, 4, 0, 1, 2, 3]


def test_worker_stage_task_schedule_must_match_runtime_topology() -> None:
    cfg = RoboDojoSimulatorConfig(
        robodojo_root=str(Path("/tmp/robodojo")),
        task_name="stack_bowls",
        task_names=["stack_bowls", "push_T"],
        worker_stage_task_schedule=["stack_bowls", "push_T"],
    )

    with pytest.raises(ValueError, match="exactly one persistent task"):
        cfg.task_assignment(worker_rank=0, worker_world_size=3, stage_id=0, stage_num=1)
