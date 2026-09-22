from __future__ import annotations

import queue
from types import SimpleNamespace

import pytest

from verl_vla.workers.env.env_manager import EnvManager, SimulatorCommandTimeoutError
from verl_vla.workers.env.env_worker import _reset_simulator_with_recovery


class _AliveProcess:
    pid = 1234
    exitcode = None

    @staticmethod
    def is_alive() -> bool:
        return True


class _CommandQueue:
    def __init__(self) -> None:
        self.items = []

    def put(self, item) -> None:
        self.items.append(item)


class _EmptyResultQueue:
    @staticmethod
    def get(*, timeout):
        raise queue.Empty


class _SuccessfulResultQueue:
    @staticmethod
    def get(*, timeout):
        del timeout
        return {"status": "success", "data": {"observation": "ready"}}


def test_bounded_simulator_rpc_raises_specific_timeout(monkeypatch) -> None:
    manager = object.__new__(EnvManager)
    manager.rank = 2
    manager.stage_id = 0
    manager.process = _AliveProcess()
    manager.command_queue = _CommandQueue()
    manager.result_queue = _EmptyResultQueue()
    clock = iter((1.0, 2.0))
    monkeypatch.setattr("verl_vla.workers.env.env_manager.time.monotonic", lambda: next(clock))

    with pytest.raises(SimulatorCommandTimeoutError, match="reset.*timed out"):
        manager.call("reset", options={}, timeout_s=0.01)

    assert manager.command_queue.items == [{"method": "reset", "args": (), "kwargs": {"options": {}}}]


def test_successful_reset_clears_cold_start_marker() -> None:
    manager = object.__new__(EnvManager)
    manager.rank = 2
    manager.stage_id = 1
    manager.process = _AliveProcess()
    manager.command_queue = _CommandQueue()
    manager.result_queue = _SuccessfulResultQueue()
    manager.cold_start_pending = True

    result = manager.call("reset", options={}, timeout_s=1.0)

    assert result == {"observation": "ready"}
    assert manager.cold_start_pending is False


def test_reset_timeout_restarts_child_and_retries_same_options() -> None:
    class Simulator:
        rank = 0
        stage_id = 0

        def __init__(self) -> None:
            self.calls = []
            self.force_stops = 0
            self.starts = 0
            self.snapshots = 0

        def call(self, name, *args, timeout_s=None, **kwargs):
            self.calls.append((name, args, timeout_s, kwargs))
            if len(self.calls) == 1:
                raise SimulatorCommandTimeoutError("hung")
            return SimpleNamespace(observation="same-case")

        def force_stop_simulator(self) -> None:
            self.force_stops += 1

        def start_simulator(self) -> None:
            self.starts += 1

        def snapshot_state(self, *, timeout_s: float) -> None:
            assert timeout_s == 30.0
            self.snapshots += 1

    simulator = Simulator()
    options = {"env_idx": [0, 1, 2, 3], "mode": "train"}
    result, restarts = _reset_simulator_with_recovery(
        simulator,
        options=options,
        timeout_s=240,
        max_process_restarts=2,
    )

    assert result.observation == "same-case"
    assert restarts == 1
    assert simulator.force_stops == 1
    assert simulator.starts == 1
    assert simulator.snapshots == 1
    assert [call[3]["options"] for call in simulator.calls] == [options, options]


def test_successful_resets_refresh_recovery_cursor_before_a_later_timeout() -> None:
    class Simulator:
        rank = 0
        stage_id = 1

        def __init__(self) -> None:
            self.cursor = 0
            self.state_buffer = 0
            self.timeout_next_reset = False
            self.force_stops = 0
            self.starts = 0

        def call(self, name, *args, timeout_s=None, **kwargs):
            del args, timeout_s, kwargs
            assert name == "reset"
            if self.timeout_next_reset:
                self.timeout_next_reset = False
                # Model a reset that advanced in the doomed child but never
                # returned a usable observation to the parent.
                self.cursor += 1
                raise SimulatorCommandTimeoutError("hung")
            current = self.cursor
            self.cursor += 1
            return SimpleNamespace(case=current)

        def snapshot_state(self, *, timeout_s: float) -> None:
            assert timeout_s == 30.0
            self.state_buffer = self.cursor

        def force_stop_simulator(self) -> None:
            self.force_stops += 1

        def start_simulator(self) -> None:
            self.starts += 1
            self.cursor = self.state_buffer

    simulator = Simulator()
    first, first_restarts = _reset_simulator_with_recovery(
        simulator,
        options={"mode": "eval"},
        timeout_s=240,
        max_process_restarts=2,
    )
    assert first.case == 0
    assert first_restarts == 0
    assert simulator.state_buffer == 1

    simulator.timeout_next_reset = True
    second, second_restarts = _reset_simulator_with_recovery(
        simulator,
        options={"mode": "train"},
        timeout_s=240,
        max_process_restarts=2,
    )
    assert second.case == 1
    assert second_restarts == 1
    assert simulator.state_buffer == 2
    assert simulator.force_stops == simulator.starts == 1
