from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from verl_vla.envs.robodojo.robodojo_env import RoboDojoEnv, _standardize_rgb


def _install_process_data_stub(monkeypatch):
    process_data = ModuleType("XPolicyLab.utils.process_data")

    def pack_robot_state(raw, action_type, robot_action_dim_info, source_type, state_type):
        assert action_type == "joint"
        assert robot_action_dim_info == {"arm_dim": [6, 6], "ee_dim": [1, 1]}
        assert source_type == "obs"
        assert state_type == "state"
        return np.asarray(raw["packed_state"], dtype=np.float32)

    def unpack_robot_state(action, action_type, robot_action_dim_info, source_type):
        assert action_type == "joint"
        assert robot_action_dim_info == {"arm_dim": [6, 6], "ee_dim": [1, 1]}
        assert source_type == "obs"
        return [{"packed": row.copy()} for row in action]

    process_data.pack_robot_state = pack_robot_state
    process_data.unpack_robot_state = unpack_robot_state
    monkeypatch.setitem(sys.modules, "XPolicyLab", ModuleType("XPolicyLab"))
    monkeypatch.setitem(sys.modules, "XPolicyLab.utils", ModuleType("XPolicyLab.utils"))
    monkeypatch.setitem(sys.modules, "XPolicyLab.utils.process_data", process_data)


def _env() -> RoboDojoEnv:
    env = object.__new__(RoboDojoEnv)
    env.num_envs = 2
    env.robodojo_cfg = SimpleNamespace(
        task_name="stack_bowls",
        task_id=0,
        action_type="joint",
        layout_ids=[0, 1, 2, 3, 4, 5],
        train_layout_schedule=[],
        train_policy_seed_schedule=[],
        train_layout_repeat=1,
        train_policy_seed_start=None,
        eval_policy_seed_schedule=[],
        seed=0,
        reset_max_attempts=3,
        defer_chunk_observations=False,
    )
    env.robot_action_dim_info = {"arm_dim": [6, 6], "ee_dim": [1, 1]}
    env._episode_ids = np.asarray([4, 5], dtype=np.int64)
    env._layout_ids = np.asarray([4, 5], dtype=np.int64)
    env._policy_seeds = np.asarray([0, 0], dtype=np.int64)
    env._train_case_cursor = 0
    env._eval_case_cursor = 0
    return env


def test_runtime_eval_cases_override_frozen_config_without_mutation():
    env = _env()
    env.num_envs = 2
    env._eval_case_cursor = 17
    original_layouts = list(env.robodojo_cfg.layout_ids)
    original_seeds = list(env.robodojo_cfg.eval_policy_seed_schedule)

    env.set_runtime_eval_cases(layout_ids=[41, 41], policy_seeds=[101, 102])
    layouts, seeds, case_ids = env._next_cases("eval")

    assert layouts == [41, 41]
    assert seeds == [101, 102]
    assert case_ids == [0, 1]
    assert list(env.robodojo_cfg.layout_ids) == original_layouts
    assert list(env.robodojo_cfg.eval_policy_seed_schedule) == original_seeds


def test_observation_emits_bound_task_id(monkeypatch):
    _install_process_data_stub(monkeypatch)
    env = _env()
    env.robodojo_cfg.task_id = 4
    env.backend = SimpleNamespace(
        get_obs_batch=lambda env_idx_list: [
            {
                "vision": {
                    "cam_head": {"color": np.zeros((240, 320, 3), dtype=np.uint8)},
                    "cam_left_wrist": {"color": np.zeros((240, 320, 3), dtype=np.uint8)},
                    "cam_right_wrist": {"color": np.zeros((240, 320, 3), dtype=np.uint8)},
                },
                "packed_state": np.zeros(14, dtype=np.float32),
                "task_instruction": "Place the four eggs into the holder.",
            }
            for _ in env_idx_list
        ]
    )

    observation = env._observations(np.asarray([0, 1]))

    np.testing.assert_array_equal(observation["task_id"], np.asarray([4, 4]))


def test_canonical_observation_preserves_camera_and_proprio_order(monkeypatch):
    _install_process_data_stub(monkeypatch)
    env = _env()
    raw = {
        "vision": {
            "cam_head": {"color": np.full((240, 320, 3), 11, dtype=np.uint8)},
            "cam_left_wrist": {"color": np.full((240, 320, 3), 22, dtype=np.uint8)},
            "cam_right_wrist": {"color": np.full((240, 320, 3), 33, dtype=np.uint8)},
        },
        "packed_state": np.arange(14, dtype=np.float32),
    }
    result = env._canonical_observation(raw)

    assert result["observation.images.cam_head"][0, 0, 0] == 11
    assert result["observation.images.cam_left_wrist"][0, 0, 0] == 22
    assert result["observation.images.cam_right_wrist"][0, 0, 0] == 33
    np.testing.assert_array_equal(result["observation.state"], np.arange(14, dtype=np.float32))


def test_rgb_standardization_is_uint8_240x320():
    raw = np.full((480, 640, 3), 260.0, dtype=np.float32)
    result = _standardize_rgb(raw)
    assert result.shape == (240, 320, 3)
    assert result.dtype == np.uint8
    assert result.min() == 255 and result.max() == 255


def test_simulator_restart_state_preserves_case_cursors():
    before = _env()
    before._train_case_cursor = 24
    before._eval_case_cursor = 4

    state = before.get_state()
    after = _env()
    after.load_state(state)

    assert after._train_case_cursor == 24
    assert after._eval_case_cursor == 4


def test_env_step_maps_reward_done_success_and_global_id_order(monkeypatch):
    _install_process_data_stub(monkeypatch)
    env = _env()

    class Backend:
        end_flag = [False, True]
        success = [True, True]

        def take_action_batch(self, rows, env_idx_list):
            self.rows = rows
            self.env_idx_list = env_idx_list

    env.backend = Backend()
    env._observations = lambda env_ids: {
        "observation": [{"observation.state": np.full(14, env_id)} for env_id in env_ids],
        "task": [f"task-{env_id}" for env_id in env_ids],
        "task_id": np.zeros(len(env_ids), dtype=np.int64),
        "eval_episode_id": env._episode_ids[env_ids],
    }
    action = np.stack([np.full(14, 101), np.full(14, 202)]).astype(np.float32)
    result = env.env_step(action, env_ids=np.asarray([1, 0]))

    assert env.backend.env_idx_list == [1, 0]
    np.testing.assert_array_equal(env.backend.rows[0]["packed"], action[0])
    np.testing.assert_array_equal(result["next.reward"], [1.0, 0.0])
    np.testing.assert_array_equal(result["next.terminated"], [True, False])
    np.testing.assert_array_equal(result["next.truncated"], [False, False])
    np.testing.assert_array_equal(result["next.success"], [True, False])
    assert result["task"] == ["task-1", "task-0"]


def test_failure_boundary_maps_to_truncation(monkeypatch):
    _install_process_data_stub(monkeypatch)
    env = _env()
    env.backend = SimpleNamespace(
        end_flag=[True, False],
        success=[False, True],
        take_action_batch=lambda rows, env_idx_list: None,
    )
    env._observations = lambda env_ids: {
        "observation": [{} for _ in env_ids],
        "task": ["task" for _ in env_ids],
        "task_id": np.zeros(len(env_ids), dtype=np.int64),
    }
    result = env.env_step(np.zeros((1, 14), dtype=np.float32), env_ids=np.asarray([0]))
    np.testing.assert_array_equal(result["next.terminated"], [False])
    np.testing.assert_array_equal(result["next.truncated"], [True])
    np.testing.assert_array_equal(result["next.success"], [False])


def test_env_step_uses_official_reward_gated_process_score(monkeypatch):
    _install_process_data_stub(monkeypatch)
    env = _env()

    class RewardManager:
        def get_score(self, *, reward_lst):
            np.testing.assert_array_equal(reward_lst, [0.0, 1.0])
            return [15.0, 100.0]

    env.backend = SimpleNamespace(
        end_flag=[True, True],
        success=[False, True],
        reward_manager=RewardManager(),
        take_action_batch=lambda rows, env_idx_list: None,
    )
    env._observations = lambda env_ids: {
        "observation": [{} for _ in env_ids],
        "task": ["task" for _ in env_ids],
        "task_id": np.zeros(len(env_ids), dtype=np.int64),
    }

    result = env.env_step(np.zeros((2, 14), dtype=np.float32), env_ids=np.asarray([0, 1]))

    np.testing.assert_allclose(result["next.score"], [0.15, 1.0])


def test_deferred_chunk_observation_executes_control_without_camera_read(monkeypatch):
    _install_process_data_stub(monkeypatch)
    env = _env()
    env.robodojo_cfg.defer_chunk_observations = True
    env._latest_obs = {
        "observation": [{"observation.state": np.full(14, 10)}, {"observation.state": np.full(14, 20)}],
        "task": ["old-0", "old-1"],
        "task_id": np.asarray([0, 0]),
        "eval_episode_id": env._episode_ids.copy(),
        "layout_id": env._layout_ids.copy(),
        "environment_seed": env._layout_ids.copy(),
        "policy_seed": env._policy_seeds.copy(),
    }
    env._deferred_observation_steps = 0
    env._chunk_observation_captures = 0

    class Backend:
        end_flag = [False, True]
        success = [False, True]

        def take_action_batch(self, rows, env_idx_list):
            self.rows = rows
            self.env_idx_list = env_idx_list

    env.backend = Backend()
    env._observations = lambda env_ids: pytest.fail("intermediate control tick must not read cameras")

    result = env.env_step(np.zeros((2, 14), dtype=np.float32), env_ids=np.asarray([0, 1]))

    assert env.backend.env_idx_list == [0, 1]
    assert result["task"] == ["old-0", "old-1"]
    np.testing.assert_array_equal(result["next.success"], [False, True])
    assert env._deferred_observation_steps == 2


def test_deferred_chunk_finalizer_replaces_only_observation_payload():
    env = _env()
    env.robodojo_cfg.defer_chunk_observations = True
    env._chunk_observation_captures = 0
    merged = {
        "observation": ["old-0", "old-1"],
        "task": ["old-task-0", "old-task-1"],
        "task_id": np.asarray([0, 0]),
        "eval_episode_id": np.asarray([4, 5]),
        "layout_id": np.asarray([4, 5]),
        "environment_seed": np.asarray([4, 5]),
        "policy_seed": np.asarray([10, 11]),
        "next.reward": np.asarray([0.0, 1.0]),
        "next.terminated": np.asarray([False, True]),
        "next.truncated": np.asarray([False, False]),
        "next.success": np.asarray([False, True]),
    }
    env._observations = lambda env_ids: {
        "observation": [f"fresh-{idx}" for idx in env_ids],
        "task": [f"task-{idx}" for idx in env_ids],
        "task_id": np.asarray([9 for _ in env_ids]),
        "eval_episode_id": env._episode_ids[env_ids].copy(),
        "layout_id": env._layout_ids[env_ids].copy(),
        "environment_seed": env._layout_ids[env_ids].copy(),
        "policy_seed": env._policy_seeds[env_ids].copy(),
    }

    result = env.finalize_execution_observation(merged, env_ids=np.asarray([0, 1]))

    assert result["observation"] == ["fresh-0", "fresh-1"]
    np.testing.assert_array_equal(result["next.reward"], [0.0, 1.0])
    np.testing.assert_array_equal(result["next.success"], [False, True])
    assert env._chunk_observation_captures == 1


def test_partial_reset_is_rejected_without_touching_backend():
    env = _env()
    env.backend = SimpleNamespace(reset=lambda seed: pytest.fail("backend reset must not be called"))
    with pytest.raises(NotImplementedError, match="partial reset is rejected"):
        env.env_reset(env_ids=np.asarray([1]))


def test_eval_reset_restarts_fixed_case_ids_with_layout_queue():
    env = _env()
    env._eval_case_cursor = 4
    reset_seeds = []
    env.backend = SimpleNamespace(
        reset=lambda seed: reset_seeds.append(list(seed)),
        run_reward=lambda: None,
    )
    env._observations = lambda env_ids: {
        "observation": [{} for _ in env_ids],
        "task": ["task" for _ in env_ids],
        "task_id": np.zeros(len(env_ids), dtype=np.int64),
        "eval_episode_id": env._episode_ids[env_ids].copy(),
    }

    first = env.env_reset(env_ids=np.asarray([0, 1]), reset_eval=True, extra={"mode": "eval"})
    second = env.env_reset(env_ids=np.asarray([0, 1]), reset_eval=False, extra={"mode": "eval"})

    assert reset_seeds == [[0, 1], [2, 3]]
    np.testing.assert_array_equal(first["eval_episode_id"], [0, 1])
    np.testing.assert_array_equal(second["eval_episode_id"], [2, 3])


def test_reset_retries_exact_same_cases_after_all_unstable():
    env = _env()
    calls = []

    class UnStableError(Exception):
        pass

    class Backend:
        unstable_envs = set()

        def reset(self, seed):
            calls.append(list(seed))
            if len(calls) == 1:
                self.unstable_envs = {0, 1}
                raise UnStableError("All scene Unstable Error!")
            self.unstable_envs = set()

        def run_reward(self):
            pass

    env.backend = Backend()
    env._observations = lambda env_ids: {"eval_episode_id": env._episode_ids[env_ids].copy()}

    result = env.env_reset(env_ids=np.asarray([0, 1]), reset_eval=True, extra={"mode": "eval"})

    assert calls == [[0, 1], [0, 1]]
    np.testing.assert_array_equal(result["eval_episode_id"], [0, 1])
    assert env._eval_case_cursor == 2


def test_reset_rejects_partial_unstable_vector_and_retries_same_cases():
    env = _env()
    calls = []

    class Backend:
        unstable_envs = set()
        episode_nums = 2

        def reset(self, seed):
            calls.append(list(seed))
            self.unstable_envs = {0} if len(calls) == 1 else set()

        def run_reward(self):
            pass

    env.backend = Backend()
    env._observations = lambda env_ids: {"eval_episode_id": env._episode_ids[env_ids].copy()}

    env.env_reset(env_ids=np.asarray([0, 1]), reset_eval=True, extra={"mode": "eval"})

    assert calls == [[0, 1], [0, 1]]


def test_reset_detects_partial_batch_from_episode_count_when_ids_are_not_exposed():
    env = _env()
    calls = []

    class Backend:
        unstable_envs = set()
        episode_nums = 2

        def reset(self, seed):
            calls.append(list(seed))
            self.episode_nums = 1 if len(calls) == 1 else 2

        def run_reward(self):
            pass

    env.backend = Backend()
    env._observations = lambda env_ids: {"eval_episode_id": env._episode_ids[env_ids].copy()}

    env.env_reset(env_ids=np.asarray([0, 1]), reset_eval=True, extra={"mode": "eval"})

    assert calls == [[0, 1], [0, 1]]


def test_reset_stops_after_bounded_attempts_without_case_substitution():
    env = _env()
    env.robodojo_cfg.reset_max_attempts = 2
    calls = []

    class Backend:
        unstable_envs = {0}

        def reset(self, seed):
            calls.append(list(seed))

    env.backend = Backend()

    with pytest.raises(RuntimeError, match=r"after 2 attempts.*fixed layouts \[0, 1\]"):
        env.env_reset(env_ids=np.asarray([0, 1]), reset_eval=True, extra={"mode": "eval"})

    assert calls == [[0, 1], [0, 1]]


def test_train_and_eval_case_cursors_are_independent_and_carry_policy_seed():
    env = _env()
    env.robodojo_cfg.train_layout_schedule = [0, 0, 3, 3]
    env.robodojo_cfg.train_policy_seed_schedule = [10, 11, 12, 13]
    env.robodojo_cfg.layout_ids = [0, 0, 1, 1]
    env.robodojo_cfg.eval_policy_seed_schedule = [20, 21, 22, 23]
    reset_seeds = []
    env.backend = SimpleNamespace(
        reset=lambda seed: reset_seeds.append(list(seed)),
        run_reward=lambda: None,
    )
    env._observations = RoboDojoEnv._observations.__get__(env)
    env.backend.get_obs_batch = lambda env_idx_list: [
        {
            "task_instruction": "Stack the three bowls together.",
            "vision": {
                "cam_head": {"color": np.zeros((240, 320, 3), dtype=np.uint8)},
                "cam_left_wrist": {"color": np.zeros((240, 320, 3), dtype=np.uint8)},
                "cam_right_wrist": {"color": np.zeros((240, 320, 3), dtype=np.uint8)},
            },
            "packed_state": np.zeros(14, dtype=np.float32),
        }
        for _ in env_idx_list
    ]

    # Avoid requiring the process-data stub: this test targets scheduling metadata.
    env._canonical_observation = lambda raw: {"observation.state": raw["packed_state"]}
    train0 = env.env_reset(env_ids=np.asarray([0, 1]), extra={"mode": "train"})
    eval0 = env.env_reset(env_ids=np.asarray([0, 1]), reset_eval=True, extra={"mode": "eval"})
    train1 = env.env_reset(env_ids=np.asarray([0, 1]), extra={"mode": "train"})

    assert reset_seeds == [[0, 0], [0, 0], [3, 3]]
    np.testing.assert_array_equal(train0["policy_seed"], [10, 11])
    np.testing.assert_array_equal(eval0["policy_seed"], [20, 21])
    np.testing.assert_array_equal(train1["policy_seed"], [12, 13])


def test_group_schedule_repeats_one_condition_for_g8_with_distinct_monotonic_policy_seeds():
    env = _env()
    env.robodojo_cfg.train_layout_schedule = [0, 1, 2, 3, 4, 5]
    env.robodojo_cfg.train_layout_repeat = 8
    env.robodojo_cfg.train_policy_seed_start = 3000

    waves = [env._next_cases("train") for _ in range(5)]

    assert [layouts for layouts, _seeds, _ids in waves[:4]] == [[0, 0]] * 4
    assert [seed for _layouts, seeds, _ids in waves[:4] for seed in seeds] == list(range(3000, 3008))
    assert waves[4][0] == [1, 1]
    assert waves[4][1] == [3008, 3009]


def test_group_schedule_shards_one_g8_condition_across_four_env_workers():
    workers = []
    for rank in range(4):
        env = _env()
        env.rank = rank
        env.world_size = 4
        env.stage_id = 0
        env._schedule_stage_num = 1
        env.robodojo_cfg.train_layout_schedule = [0, 1, 2, 3, 4, 5]
        env.robodojo_cfg.train_layout_repeat = 8
        env.robodojo_cfg.train_policy_seed_start = 3000
        workers.append(env)

    first_group = [worker._next_cases("train") for worker in workers]
    second_group = [worker._next_cases("train") for worker in workers]

    assert [layout for layouts, _seeds, _ids in first_group for layout in layouts] == [0] * 8
    assert [seed for _layouts, seeds, _ids in first_group for seed in seeds] == list(range(3000, 3008))
    assert [layout for layouts, _seeds, _ids in second_group for layout in layouts] == [1] * 8
    assert [seed for _layouts, seeds, _ids in second_group for seed in seeds] == list(range(3008, 3016))


def test_group_schedule_tiles_one_g8_condition_in_two_waves_with_two_env_workers():
    workers = []
    for rank in range(2):
        env = _env()
        env.rank = rank
        env.world_size = 2
        env.stage_id = 0
        env._schedule_stage_num = 1
        env.robodojo_cfg.train_layout_schedule = [0, 1, 2, 3, 4, 5]
        env.robodojo_cfg.train_layout_repeat = 8
        env.robodojo_cfg.train_policy_seed_start = 3000
        workers.append(env)

    first_group = [worker._next_cases("train") for _wave in range(2) for worker in workers]
    second_group = [worker._next_cases("train") for _wave in range(2) for worker in workers]

    assert [layout for layouts, _seeds, _ids in first_group for layout in layouts] == [0] * 8
    assert [seed for _layouts, seeds, _ids in first_group for seed in seeds] == list(range(3000, 3008))
    assert [layout for layouts, _seeds, _ids in second_group for layout in layouts] == [1] * 8
    assert [seed for _layouts, seeds, _ids in second_group for seed in seeds] == list(range(3008, 3016))


def test_group_schedule_shards_one_g8_condition_across_two_four_lane_env_workers():
    workers = []
    for rank in range(2):
        env = _env()
        env.num_envs = 4
        env.rank = rank
        env.world_size = 2
        env.stage_id = 0
        env._schedule_stage_num = 1
        env.robodojo_cfg.train_layout_schedule = [0, 1, 2, 3, 4, 5]
        env.robodojo_cfg.train_layout_repeat = 8
        env.robodojo_cfg.train_policy_seed_start = 3000
        workers.append(env)

    first_group = [worker._next_cases("train") for worker in workers]
    second_group = [worker._next_cases("train") for worker in workers]

    assert [layout for layouts, _seeds, _ids in first_group for layout in layouts] == [0] * 8
    assert [seed for _layouts, seeds, _ids in first_group for seed in seeds] == list(range(3000, 3008))
    assert [layout for layouts, _seeds, _ids in second_group for layout in layouts] == [1] * 8
    assert [seed for _layouts, seeds, _ids in second_group for seed in seeds] == list(range(3008, 3016))


def test_group_schedule_is_stage_major_for_two_concurrent_g8_groups():
    workers_by_stage = []
    for stage_id in range(2):
        stage_workers = []
        for rank in range(2):
            env = _env()
            env.num_envs = 4
            env.rank = rank
            env.world_size = 2
            env.stage_id = stage_id
            env._schedule_stage_num = 2
            env.robodojo_cfg.train_layout_schedule = [0, 1, 2, 3, 4, 5]
            env.robodojo_cfg.train_layout_repeat = 8
            env.robodojo_cfg.train_policy_seed_start = 3000
            stage_workers.append(env)
        workers_by_stage.append(stage_workers)

    groups = [
        [worker._next_cases("train") for worker in stage_workers]
        for stage_workers in workers_by_stage
    ]

    assert [layout for layouts, _seeds, _ids in groups[0] for layout in layouts] == [0] * 8
    assert [seed for _layouts, seeds, _ids in groups[0] for seed in seeds] == list(range(3000, 3008))
    assert [layout for layouts, _seeds, _ids in groups[1] for layout in layouts] == [1] * 8
    assert [seed for _layouts, seeds, _ids in groups[1] for seed in seeds] == list(range(3008, 3016))


def test_worker_local_schedule_assigns_one_complete_g8_condition_per_worker_stage():
    groups = []
    for stage_id in range(2):
        for rank in range(2):
            env = _env()
            env.num_envs = 8
            env.rank = rank
            env.world_size = 2
            env.stage_id = stage_id
            env._schedule_stage_num = 2
            env.robodojo_cfg.train_layout_schedule = [0, 1, 2, 3, 4, 5]
            env.robodojo_cfg.train_layout_repeat = 8
            env.robodojo_cfg.train_policy_seed_start = 3000
            groups.append(env._next_cases("train"))

    assert [layouts for layouts, _seeds, _ids in groups] == [
        [0] * 8,
        [1] * 8,
        [2] * 8,
        [3] * 8,
    ]
    assert [seed for _layouts, seeds, _ids in groups for seed in seeds] == list(
        range(3000, 3032)
    )


def test_one_vector24_simulator_assigns_three_contiguous_g8_conditions():
    env = _env()
    env.num_envs = 24
    env.rank = 0
    env.world_size = 1
    env.stage_id = 0
    env._schedule_stage_num = 1
    env.robodojo_cfg.train_layout_schedule = [0, 1, 2, 3, 4, 5]
    env.robodojo_cfg.train_layout_repeat = 8
    env.robodojo_cfg.train_policy_seed_start = 3000

    layouts, policy_seeds, _case_ids = env._next_cases("train")

    assert layouts == [0] * 8 + [1] * 8 + [2] * 8
    assert policy_seeds == list(range(3000, 3024))


def test_fixed_eval_schedule_shards_twelve_cases_across_two_four_lane_workers():
    workers = []
    layouts = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    policy_seeds = [2000, 2001, 2010, 2011, 2020, 2021, 2030, 2031, 2040, 2041, 2050, 2051]
    for rank in range(2):
        env = _env()
        env.num_envs = 4
        env.rank = rank
        env.world_size = 2
        env.stage_id = 0
        env._schedule_stage_num = 1
        env.robodojo_cfg.layout_ids = layouts
        env.robodojo_cfg.eval_policy_seed_schedule = policy_seeds
        workers.append(env)

    first_wave = [worker._next_cases("eval") for worker in workers]
    second_wave = [worker._next_cases("eval") for worker in workers]

    assert [case_id for _layouts, _seeds, case_ids in first_wave for case_id in case_ids] == list(range(8))
    assert [case_id for _layouts, _seeds, case_ids in second_wave for case_id in case_ids] == [
        8,
        9,
        10,
        11,
        0,
        1,
        2,
        3,
    ]
    first_twelve = [
        (layout, seed, case_id)
        for wave in (first_wave, second_wave)
        for worker_layouts, worker_seeds, worker_case_ids in wave
        for layout, seed, case_id in zip(worker_layouts, worker_seeds, worker_case_ids, strict=True)
    ][:12]
    assert first_twelve == list(zip(layouts, policy_seeds, range(12), strict=True))
