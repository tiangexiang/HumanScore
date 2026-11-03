"""Regression tests for the confidence-only extra-limb definition."""

import importlib
import json
import os
import sys
from types import SimpleNamespace

import pytest

compute_score = importlib.import_module("metrics.multi-limbs.eval_extra_limbs").compute_extra_limbs_score


def write_detections(tmp_path, detections):
    path = tmp_path / "output.json"
    path.write_text(json.dumps(detections))
    return str(path)


def test_small_box_does_not_suppress_confident_extra_limb(tmp_path):
    path = write_detections(tmp_path, {
        "frame_000000.jpg": {"bbox": [[0, 0, 1, 1], [0, 0, 1000, 1000]],
                             "score": [0.03, 0.9],
                             "class": ["human with extra hand", "normal human"]},
    })
    assert compute_score(path)["score"] == pytest.approx(0.0)


def test_score_matches_paper_triplet(tmp_path):
    # One moderate frame (m=.5), one clear frame, then a normal frame.
    path = write_detections(tmp_path, {
        "frame_000000.jpg": {"score": [0.0175], "class": ["human with extra arm"]},
        "frame_000001.jpg": {"score": [0.03], "class": ["human with extra leg"]},
        "frame_000002.jpg": {"score": [], "class": []},
    })
    result = compute_score(path)
    assert result["r"] == pytest.approx(2 / 3)
    assert result["severity_mean"] == pytest.approx(0.75)
    assert result["continuity"] == pytest.approx(2 / 3)
    assert result["score"] == pytest.approx(100 * (1 - 0.5 * 2 / 3 - 0.3 * 0.75 - 0.2 * 2 / 3))


def test_confidence_at_mild_threshold_is_not_violation(tmp_path):
    path = write_detections(tmp_path, {
        "frame_000000.jpg": {"score": [0.005], "class": ["human with extra hand"]},
    })
    assert compute_score(path)["score"] == 100


def test_empty_inference_is_not_a_perfect_video(tmp_path):
    with pytest.raises(ValueError, match="at least one frame"):
        compute_score(write_detections(tmp_path, {}))


def test_explicit_cached_cli_preserves_empty_frames(monkeypatch, tmp_path):
    source = write_detections(tmp_path, {
        "frame_000000.jpg": {"score": [0.03], "class": ["human with extra hand"]},
        "frame_000001.jpg": {"score": [], "class": []},
    })
    output = tmp_path / "scores"
    monkeypatch.setattr(sys, "argv", ["extra-limbs", "--video_path", "clip.mp4",
                                      "--detections-json", source, "--output_dir", str(output)])
    module = importlib.import_module("metrics.multi-limbs.eval_extra_limbs")
    module.main()
    result = json.loads((output / "clip_extra_limbs.json").read_text())
    assert result["num_frames"] == 2
    assert result["r"] == 0.5
    assert result["inference"] == "cached"
    assert result["source_output_json"] == source


def test_hadm_compatibility_settings_are_scoped_to_child(monkeypatch, tmp_path):
    module = importlib.import_module("metrics.multi-limbs.eval_extra_limbs")
    root = tmp_path / "HADM"
    for relative in ["tools/lazyconfig_train_net.py", "config.py", "model.pth"]:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    monkeypatch.delenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", raising=False)
    monkeypatch.setenv("HMB_HADM_UPSTREAM", "1")
    monkeypatch.setenv("PYTHONPATH", "/isolated/overlay")
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", run)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    module.run_hadm_inference(str(root), str(tmp_path / "frames"), str(tmp_path / "results"),
                              config_file="config.py", checkpoint_path="model.pth")
    command, kwargs = calls[0]
    assert kwargs["env"]["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] == "1"
    assert "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD" not in os.environ
    assert kwargs["env"]["PYTHONPATH"].startswith("/isolated/overlay:")
    assert "model.backbone.net.xattn=False" in command
