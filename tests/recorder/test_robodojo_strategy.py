"""Focused tests for RoboDojo video/LeRobot frame conversion."""

from __future__ import annotations

import numpy as np

from verl_vla.recorder.strategies import get_lerobot_strategy


def test_robodojo_strategy_preserves_three_cameras_action_and_case_identity():
    strategy = get_lerobot_strategy("robodojo", fps=25, robot_type="arx_x5")
    observation = {
        "observation.images.cam_head": np.full((240, 320, 3), 11, dtype=np.uint8),
        "observation.images.cam_left_wrist": np.full((240, 320, 3), 22, dtype=np.uint8),
        "observation.images.cam_right_wrist": np.full((240, 320, 3), 33, dtype=np.uint8),
        "observation.state": np.arange(14, dtype=np.float32),
        "info.eval_episode_id": 2,
        "info.layout_id": 7,
        "info.environment_seed": 7,
        "info.policy_seed": 2022,
        "next.score": 0.15,
    }

    frame = strategy.make_frame(
        observation=observation,
        action=np.arange(14, dtype=np.float32) / 10,
        task="stack the bowls",
        next_reward=0.0,
        next_truncated=False,
        next_terminated=False,
        next_success=False,
    )

    assert strategy.fps == 25
    assert strategy.robot_type == "arx_x5"
    assert frame["observation.images.cam_head"][0, 0, 0] == 11
    assert frame["observation.images.cam_left_wrist"][0, 0, 0] == 22
    assert frame["observation.images.cam_right_wrist"][0, 0, 0] == 33
    np.testing.assert_array_equal(frame["observation.state"], np.arange(14, dtype=np.float32))
    assert frame["action"].shape == (14,)
    assert frame["info.layout_id"].item() == 7
    assert frame["info.policy_seed"].item() == 2022
    assert np.isclose(frame["next.score"].item(), 0.15)


def test_robodojo_strategy_schema_matches_frame_fields():
    strategy = get_lerobot_strategy("robodojo")
    frame = strategy.make_frame(
        observation={
            "observation.images.cam_head": np.zeros((240, 320, 3), dtype=np.uint8),
            "observation.images.cam_left_wrist": np.zeros((240, 320, 3), dtype=np.uint8),
            "observation.images.cam_right_wrist": np.zeros((240, 320, 3), dtype=np.uint8),
            "observation.state": np.zeros(14, dtype=np.float32),
        },
        action=np.zeros(14, dtype=np.float32),
        task="task",
    )
    # `task` is LeRobot's standard episode task string, not a dataset feature.
    assert set(frame) - {"task"} == set(strategy.features())
