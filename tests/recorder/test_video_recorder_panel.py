"""Focused layout tests for the human-readable video overlay."""

from __future__ import annotations

import numpy as np

from verl_vla.recorder.config import VideoRecorderConfig
from verl_vla.recorder.impl.video import VideoRecorder


def test_panel_size_is_constant_when_action_text_length_changes(tmp_path):
    recorder = VideoRecorder(
        cfg=VideoRecorderConfig(root=str(tmp_path), fps=30, font_size=14),
        env_type="robodojo",
        num_envs=1,
    )
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    short = recorder._append_text_panel(image, ["action: [0]", "info.policy_seed: [2022]"])
    long = recorder._append_text_panel(image, [("action: long-value " * 80), "info.policy_seed: [2022]"])

    assert long.shape == short.shape
    assert long.shape[0] % 16 == 0
    assert long.shape[1:] == (320, 3)


def test_fit_line_never_exceeds_video_width(tmp_path):
    recorder = VideoRecorder(
        cfg=VideoRecorderConfig(root=str(tmp_path), fps=30, font_size=14),
        env_type="robodojo",
        num_envs=1,
    )
    from PIL import ImageFont

    font = ImageFont.load_default(size=14)
    line = recorder._fit_line("action: " + "1.2345 " * 100, font, 300)
    assert font.getlength(line) <= 300
    assert line.endswith(" ...")
