"""Tests of hmb/video.py: frame sampling on the analysis grid, output image size and the
saved frame plan."""

import numpy as np
import pytest

from hmb.video import FramePlan, output_size, sample_indices


def test_all_frames_by_default():
    assert sample_indices(10, 24.0, None).tolist() == list(range(10))
    assert sample_indices(10, 24.0, 30.0).tolist() == list(range(10))  # never upsample


def test_60_to_24_keeps_exact_source_times():
    idx = sample_indices(121, 60.0, 24.0)
    times = idx / 60.0
    # nearest-frame selection: every selected time within half a source frame of the grid
    grid = np.arange(len(idx)) / 24.0
    assert np.max(np.abs(times - grid)) <= 0.5 / 60.0 + 1e-12
    assert idx[0] == 0 and idx[-1] == 120 and np.all(np.diff(idx) > 0)


def test_30_to_24_and_decimation():
    idx = sample_indices(150, 30.0, 24.0)
    assert set(np.diff(idx)) <= {1, 2}
    assert sample_indices(120, 60.0, 30.0).tolist() == list(range(0, 120, 2))


def test_output_size_even_and_bounded():
    assert output_size(2560, 1440, 1280) == (1280, 720)
    assert output_size(1366, 768, None) == (1366, 768)
    w, h = output_size(1366, 768, 1280)
    assert max(w, h) <= 1280 and w % 2 == 0 and h % 2 == 0


def test_plan_roundtrip(tmp_path):
    plan = FramePlan(video="v.mp4", source_fps=60.0, num_source_frames=5, width=8, height=6,
                     sample_fps=30.0, indices=[0, 2, 4], timestamps=[0.0, 2 / 60, 4 / 60])
    path = plan.save(tmp_path)
    again = FramePlan.load(path)
    assert again == plan and again.num_frames == 3
