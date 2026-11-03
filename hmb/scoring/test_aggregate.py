"""Tests of the shared frequency / severity / persistence aggregation (hmb.scoring.aggregate)."""
import numpy as np
import pytest

from hmb.scoring.aggregate import aggregate, longest_run, ramp, run_bounds


def test_clean_clip_scores_100():
    out = aggregate(np.zeros(50), threshold=0.05)
    assert out.score == 100.0 and out.num_flagged == 0 and out.coverage == 1.0


def test_paper_formula_on_flagged_frames():
    m = np.zeros(10)
    m[2:5] = 1.0
    out = aggregate(m, threshold=0.05)
    # r = 0.3, s = 1, p = 0.3 -> D = 0.15 + 0.3 + 0.06
    assert out.D == pytest.approx(0.51)
    assert out.score == pytest.approx(49.0)


def test_subthreshold_noise_does_not_accumulate_with_length():
    short = aggregate(np.full(100, 0.04), threshold=0.05)
    long = aggregate(np.full(1000, 0.04), threshold=0.05)
    assert short.score == long.score == 100.0


def test_invalid_frames_are_excluded_and_reported():
    m = np.array([1.0, 0.0, 0.0, 0.0])
    valid = np.array([True, True, False, False])
    out = aggregate(m, threshold=0.05, valid=valid)
    assert out.num_valid == 2 and out.coverage == 0.5 and out.r == 0.5
    assert np.isnan(aggregate(m, valid=np.zeros(4, bool)).score)


def test_runs():
    flags = np.array([1, 0, 1, 1, 0, 1, 1, 1], bool)
    starts, stops = run_bounds(flags)
    assert starts.tolist() == [0, 2, 5] and stops.tolist() == [1, 4, 8]
    assert longest_run(flags) == 3
    assert longest_run(np.zeros(4, bool)) == 0


def test_helpers():
    assert ramp([0.0, 0.02, 0.05], 0.01, 0.03).tolist() == pytest.approx([0.0, 0.5, 1.0])
