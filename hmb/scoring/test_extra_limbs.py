"""Tests of the extra-limbs metric (hmb.scoring.extra_limbs): graded r/s/p on HADM evidence."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from hmb.scoring import extra_limbs as el

REPO = Path(__file__).resolve().parents[2]
HAND, ARM, LEG, FEET = el.EXTRA_LIMB_CLASSES


def make_output(evidence, cls=HAND, extra=None, name="frame_{:06d}.jpg"):
    """HADM output.json mapping with one box per frame carrying ``evidence``."""
    out = {}
    for i, value in enumerate(evidence):
        rec = {"bbox": [], "score": [], "class": []}
        if value > 0:
            rec["bbox"].append([10.0, 20.0, 110.0, 220.0])
            rec["score"].append(float(value))
            rec["class"].append(cls)
        for label, score in (extra or {}).get(i, []):
            rec["bbox"].append([0.0, 0.0, 50.0, 50.0])
            rec["score"].append(float(score))
            rec["class"].append(label)
        out[name.format(i)] = rec
    return out


def _graded_reference(M, tau=0.001, tau_severe=0.03):
    """Frame-by-frame reference of the graded r/s/p formula (independent of the vectorised code)."""
    m = [min(max((x - tau) / (tau_severe - tau), 0.0), 1.0) if x > tau else 0.0 for x in M]
    T, mass = len(m), sum(m)
    r = mass / T
    s = sum(v * v for v in m) / mass if mass > 0 else 0.0
    best = run = 0.0
    for v in m:
        run = run + v if v > 0 else 0.0
        best = max(best, run)
    p = best / T
    return 100.0 * (1.0 - min(max(0.5 * r + 0.3 * s + 0.2 * p, 0.0), 1.0))


def test_graded_score_matches_the_reference_formula():
    rng = np.random.default_rng(0)
    for _ in range(50):
        M = rng.choice([0.0, 0.0, 0.002, 0.008, 0.02, 0.5, 0.9], size=48)
        assert el.graded_rsp(M)["score"] == pytest.approx(_graded_reference(M.tolist()))


def test_full_severity_frames_reproduce_the_paper_rsp():
    M = np.zeros(24)
    M[[3, 4, 5, 10]] = 0.6
    out = el.graded_rsp(M)
    assert out["r"] == pytest.approx(4 / 24) and out["s"] == pytest.approx(1.0) and out["p"] == pytest.approx(3 / 24)
    assert el.graded_rsp(np.zeros(24))["score"] == 100.0


def test_weak_evidence_costs_less_than_strong_evidence():
    assert el.graded_rsp(np.full(24, 0.004))["score"] > el.graded_rsp(np.full(24, 0.02))["score"]
    at_floor = np.full(24, el.TAU_FLOOR)                # at (not above) the detector floor
    assert el.graded_rsp(at_floor)["score"] == 100.0


def test_evidence_uses_only_the_extra_limb_classes_in_file_name_order(tmp_path):
    m = [0.0, 0.3, 0.0, 0.02]
    extra = {0: [("human missing hand", 0.9), ("human with extra face", 0.9)], 2: [(LEG, 0.7), (ARM, 0.1)]}
    path = tmp_path / "output.json"
    path.write_text(json.dumps(make_output(m, extra=extra)))
    assert el.frame_evidence(path).tolist() == pytest.approx([0.0, 0.3, 0.7, 0.02])


def _paper_module():
    path = REPO / "metrics" / "multi-limbs" / "eval_extra_limbs.py"
    spec = importlib.util.spec_from_file_location("paper_extra_limbs_for_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_score_equals_the_paper_code_when_every_flag_is_at_full_severity(tmp_path):
    """Binary evidence (0 or >= tau_severe) makes graded and hard flags identical."""
    paper = _paper_module()
    rng = np.random.default_rng(1)
    for trial in range(10):
        T = int(rng.integers(20, 200))
        m = rng.choice([0.0, 0.04, 0.6], size=T, p=[0.8, 0.1, 0.1])
        path = tmp_path / f"o{trial}.json"
        path.write_text(json.dumps(make_output(m, cls=[HAND, ARM, LEG, FEET][trial % 4])))
        expected = paper.compute_extra_limbs_score(str(path))["score"]
        assert el.graded_rsp(el.frame_evidence(path))["score"] == pytest.approx(expected, abs=1e-9)


def test_session_layouts(tmp_path):
    session = tmp_path / "S"
    results = session / "cache" / "multi_limbs" / "clip" / "hadm_results"
    results.mkdir(parents=True)
    m = np.zeros(48)
    m[10:12] = 0.4
    (results / "output.json").write_text(json.dumps(make_output(m)))
    res = el.score_session(session)
    assert res["extra_limbs_score"] == pytest.approx(el.graded_rsp(m)["score"])
    assert res["hadm_output"] == str(results / "output.json")
    # the original HADM runner writes hadm_results/<timestamp>/output.json
    other = tmp_path / "T" / "cache" / "multi_limbs" / "walk" / "hadm_results" / "2000-01-01_00-00-00"
    other.mkdir(parents=True)
    (other / "output.json").write_text(json.dumps(make_output(m)))
    assert el.score_session(tmp_path / "T", "walk")["score"] == pytest.approx(res["score"])
    with pytest.raises(FileNotFoundError):
        el.score_session(tmp_path / "empty")


def test_cli_on_output_json(tmp_path, capsys):
    path = tmp_path / "output.json"
    m = np.zeros(48)
    m[10:12] = 0.4
    path.write_text(json.dumps(make_output(m)))
    assert el.main(["--output-json", str(path), "--out", str(tmp_path / "r.json")]) == 0
    res = json.loads((tmp_path / "r.json").read_text())
    assert res["extra_limbs_score"] == pytest.approx(el.graded_rsp(m)["score"])


def test_invalid_inputs_raise(tmp_path):
    with pytest.raises(ValueError):
        el.graded_rsp([])
    bad = tmp_path / "bad.json"
    for payload in ({}, {"f_0.jpg": {"bbox": [[0, 0, 1, 1]], "score": [1.5], "class": [HAND]}},
                    {"f_0.jpg": {"bbox": [], "score": [0.1], "class": []}}):
        bad.write_text(json.dumps(payload))
        with pytest.raises(ValueError):
            el.frame_evidence(bad)
