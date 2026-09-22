"""Recorder durability-barrier tests."""

from __future__ import annotations

import time

from verl_vla.recorder.async_recorder import AsyncRecorder
from verl_vla.recorder.base import BaseRecorder


class _SlowRecorder(BaseRecorder):
    def __init__(self):
        self.events = []

    def record_once(self, **kwargs):
        self.events.append(("record", kwargs["env_id"]))

    def save_episode(self, env_id=0):
        time.sleep(0.02)
        self.events.append(("save", env_id))

    def clear_episode(self, env_id=0):
        self.events.append(("clear", env_id))

    def flush(self):
        self.events.append(("flush", None))

    def finalize(self):
        self.events.append(("finalize", None))


def test_async_flush_waits_for_all_earlier_episode_saves():
    inner = _SlowRecorder()
    recorder = AsyncRecorder(inner, queue_size=8)
    recorder.save_episode(0)
    recorder.save_episode(1)

    recorder.flush()

    assert inner.events == [("save", 0), ("save", 1), ("flush", None)]
    recorder.finalize()
