"""Regression tests for collision aggregation and CLI result handling."""

import json
import sys
from types import SimpleNamespace

import pytest

from metrics.self_collision_bvh.scoring import aggregate_collision_score
from metrics.self_collision_bvh import main as collision_cli


def test_mild_collisions_contribute_severity_and_persistence():
    result = aggregate_collision_score([0.02, 0.02])
    assert result["r_severe"] == 0
    assert result["severity_mean"] == pytest.approx(0.5)
    assert result["continuity"] == 1
    assert result["score"] == pytest.approx(65)


def test_mixed_collisions_match_paper_definition():
    result = aggregate_collision_score([0.02, 0.03, 0])
    assert result["r_severe"] == pytest.approx(1 / 3)
    assert result["severity_mean"] == pytest.approx(0.75)
    assert result["continuity"] == pytest.approx(2 / 3)
    assert result["score"] == pytest.approx(47.5)


def test_empty_collision_sequence_is_not_a_perfect_video():
    with pytest.raises(ValueError, match="empty"):
        aggregate_collision_score([])


def cli_arguments(monkeypatch, tmp_path):
    source = tmp_path / "motion.pt"
    source.touch()
    output = tmp_path / "scores"
    monkeypatch.setattr(sys, "argv", ["self-collision", "--input_path", str(source), "--output_dir", str(output)])
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    return source, output


def test_cli_evaluates_input_and_writes_result(monkeypatch, tmp_path):
    source, output = cli_arguments(monkeypatch, tmp_path)
    calls = []

    def evaluate(**kwargs):
        calls.append(kwargs["result_path"])
        return {"sequence": "motion", "self_collision_bvh_score": 65.0,
                "penetration_frame_rate": 0, "r_mild": 1, "r_severe": 0,
                "severity_mean": 0.5, "continuity": 1, "avg_M_t": 0.02,
                "face_collision_ratio": 0.02, "avg_pairs_per_frame": 10,
                "avg_face_ratio_per_frame": 0.02}

    monkeypatch.setattr(collision_cli, "evaluate_sequence", evaluate)
    collision_cli.main()
    assert calls == [str(source)]
    result = json.loads((output / "motion_self_collision_bvh.json").read_text())
    assert result["self_collision_bvh_score"] == 65


def test_cli_fails_when_evaluation_fails(monkeypatch, tmp_path):
    _, output = cli_arguments(monkeypatch, tmp_path)

    def fail(**kwargs):
        raise RuntimeError("missing model")

    monkeypatch.setattr(collision_cli, "evaluate_sequence", fail)
    with pytest.raises(RuntimeError, match="failed for: motion"):
        collision_cli.main()
    result = json.loads((output / "all_self_collision_bvh_results.json").read_text())
    assert result["motion"]["error"] == "missing model"
