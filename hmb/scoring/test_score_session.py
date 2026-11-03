"""Tests of the session scorer (hmb.scoring.score_session): routing, report, paper scores, reasons."""
import json

import pytest

from hmb.scoring import score_session as ss


def fake_paper(calls, score=50.0):
    def run(session, stem, *, metrics, write, use_cuda):
        calls.append(tuple(metrics))
        return {"metrics": {m: {"score": score, "status": "ok"} for m in metrics}}
    return run


def test_score_session_routes_each_metric(tmp_path, monkeypatch):
    from hmb.scoring import paper_scores

    calls = {"paper": [], "own": []}
    monkeypatch.setattr(paper_scores, "score_session", fake_paper(calls["paper"]))
    for m in ("extra_limbs", "bone_length", "joint_range", "self_collision"):
        monkeypatch.setitem(ss.SCORERS, m, lambda s, st, o, m=m: calls["own"].append(m) or {"score": 90.0, "details": {}})
    rep = ss.score_session(tmp_path, "clip", write=True)
    assert calls["paper"] == [("kinematic_extremes", "motion_smoothness")]   # one paper call per session
    assert calls["own"] == ["extra_limbs", "bone_length", "joint_range", "self_collision"]
    assert rep["scores"]["extra_limbs"] == 90.0 and rep["scores"]["motion_smoothness"] == 50.0
    assert rep["metrics"]["motion_smoothness"]["details"]["source"] == "metrics/motion_smoothness (paper implementation)"
    assert rep["overall"] == pytest.approx((4 * 90.0 + 2 * 50.0) / 6)
    assert rep["tiers"] == {"anatomy": 90.0, "kinematic": 90.0, "kinetic": 50.0}
    assert rep["status"] == {m: "ok" for m in ss.METRICS} and rep["reasons"] == {}
    assert "paper_scores" not in rep
    written = json.loads((tmp_path / ss.OUTPUT_NAME).read_text())
    assert ss.OUTPUT_NAME == "scores.json" and written["overall"] == rep["overall"]
    assert set(written) == {"session", "stem", "scores", "tiers", "overall", "status", "reasons", "metrics"}


def test_paper_scores_add_all_six_paper_metrics_in_one_call(tmp_path, monkeypatch):
    from hmb.scoring import paper_scores

    calls = []
    monkeypatch.setattr(paper_scores, "score_session", fake_paper(calls, score=40.0))
    for m in ("extra_limbs", "bone_length", "joint_range", "self_collision"):
        monkeypatch.setitem(ss.SCORERS, m, lambda s, st, o: {"score": 90.0, "details": {}})
    rep = ss.score_session(tmp_path, "clip", write=False, paper_scores=True)
    assert calls == [ss.METRICS]                     # the kinetic entries are shared with the paper section
    paper = rep["paper_scores"]
    assert paper["scores"] == {m: 40.0 for m in ss.METRICS} and paper["overall"] == pytest.approx(40.0)
    assert paper["status"] == {m: "ok" for m in ss.METRICS} and paper["reasons"] == {}
    assert rep["scores"]["motion_smoothness"] == 40.0 and rep["scores"]["bone_length"] == 90.0
    # a subset without the kinetic metrics still gets the full paper section
    calls.clear()
    rep = ss.score_session(tmp_path, "clip", write=False, paper_scores=True, metrics=("bone_length",))
    assert calls == [ss.METRICS] and set(rep["scores"]) == {"bone_length"}


def test_paper_scores_report_unsupported_and_missing_metrics(tmp_path, monkeypatch):
    from hmb.scoring import paper_scores

    def partial(session, stem, *, metrics, write, use_cuda):
        out = {m: {"score": 70.0, "status": "ok"} for m in metrics}
        out.get("bone_length", {}).update(score=None, status="missing_input", error="/S/bone_fit/clip_bone_length.npz")
        out.get("self_collision", {}).update(score=None, status="unsupported")
        return {"metrics": out}

    monkeypatch.setattr(paper_scores, "score_session", partial)
    for m in ("extra_limbs", "bone_length", "joint_range", "self_collision"):
        monkeypatch.setitem(ss.SCORERS, m, lambda s, st, o: {"score": 90.0, "details": {}})
    paper = ss.score_session(tmp_path, "clip", write=False, paper_scores=True)["paper_scores"]
    assert paper["overall"] is None and paper["tiers"]["kinetic"] == pytest.approx(70.0)
    assert paper["status"]["bone_length"] == "missing_input" and paper["status"]["self_collision"] == "unsupported"
    assert paper["reasons"] == {"bone_length": "/S/bone_fit/clip_bone_length.npz",
                                "self_collision": "the paper implementation returned no finite score"}


def test_paper_missing_input_is_reported(tmp_path, monkeypatch):
    from hmb.scoring import paper_scores

    monkeypatch.setattr(paper_scores, "score_session", lambda *a, metrics, **k: {
        "metrics": {m: {"score": None, "status": "missing_input", "error": "no motion"} for m in metrics}})
    rep = ss.score_session(tmp_path, "clip", write=False, metrics=("motion_smoothness",))
    assert rep["metrics"]["motion_smoothness"]["status"] == "missing_input" and rep["scores"]["motion_smoothness"] is None
    rep = ss.score_session(tmp_path, "clip", write=False, metrics=("extra_limbs",))
    assert rep["metrics"]["extra_limbs"]["status"] == "missing_input"          # no HADM output.json
    assert rep["reasons"]["extra_limbs"].startswith("no HADM output.json")


def test_unsupported_and_missing_metrics_carry_a_one_line_reason(tmp_path, monkeypatch):
    from hmb.scoring import bone_length as bl

    def unsupported_bone_length(session, stem, opts):
        return {"score": float("nan"), "details": {"coverage": 0.1777, "supported": False, "per_segment": {"x": {}},
                                                    "config": bl.DEFAULT_CONFIG.as_dict()}}

    def missing_mesh(session, stem, opts):
        raise FileNotFoundError(tmp_path / "gvhmr" / "clip" / "hmr4d_results.pt")

    def broken(session, stem, opts):
        raise RuntimeError("first line\nsecond line")

    monkeypatch.setitem(ss.SCORERS, "bone_length", unsupported_bone_length)
    monkeypatch.setitem(ss.SCORERS, "self_collision", missing_mesh)
    monkeypatch.setitem(ss.SCORERS, "joint_range", broken)
    rep = ss.score_session(tmp_path, "clip", write=False, metrics=("bone_length", "joint_range", "self_collision"))
    bone, joint, mesh = (rep["metrics"][m] for m in ("bone_length", "joint_range", "self_collision"))
    assert bone["status"] == "unsupported" and bone["reason"] == "keypoint coverage 0.18 < 0.25 of the analysed frames"
    assert mesh["status"] == "missing_input"
    assert mesh["reason"] == f"missing input {tmp_path / 'gvhmr' / 'clip' / 'hmr4d_results.pt'}"
    assert joint["status"] == "error" and joint["reason"] == "RuntimeError: first line"
    assert rep["reasons"] == {"bone_length": bone["reason"], "joint_range": joint["reason"],
                              "self_collision": mesh["reason"]}


def test_status_reason_from_scorer_details():
    from hmb.scoring import bone_length as bl

    config = bl.DEFAULT_CONFIG.as_dict()

    def reason(metric, details, **entry):
        return ss.status_reason(metric, {"status": "unsupported", "details": details, **entry})

    assert ss.status_reason("bone_length", {"status": "ok", "score": 90.0}) is None
    # numbers that round alike get more decimals
    assert reason("bone_length", {"coverage": 0.2499, "config": config}) == \
        "keypoint coverage 0.2499 < 0.2500 of the analysed frames"
    assert reason("bone_length", {"coverage": 0.6, "per_segment": {}, "config": config}) == \
        "no body segment observed in enough frames (at least 5)"
    assert reason("joint_range", {"num_frames": 40, "num_valid": 0}) == \
        "no frame with a detected person (0 of 40 analysed frames)"
    assert reason("self_collision", {"num_frames": 0}) == "the body mesh has no frames"
    assert reason("motion_smoothness", {"aggregate": {"num_valid": 0}, "num_frames": 12}) == \
        "no usable frame (0 of 12 analysed frames)"
    assert reason("kinematic_extremes", {"source": "metrics/kinematic_extreme (paper implementation)"}) == \
        "the paper implementation returned no finite score"
    assert reason("extra_limbs", {"reason": "custom\nmore"}) == "custom"
    assert reason("extra_limbs", None) == "the scorer returned no finite score"
    assert ss.status_reason("extra_limbs", {"status": "missing_input", "error": "no HADM output.json"}) == \
        "no HADM output.json"
    assert ss.status_reason("extra_limbs", {"status": "missing_input", "error": "x", "reason": "kept"}) == "kept"
    assert len(ss.status_reason("extra_limbs", {"status": "error", "error": "y" * 2000})) == ss.REASON_MAX_CHARS


def test_cli_prints_the_summary_and_paper_scores(tmp_path, monkeypatch, capsys):
    from hmb.scoring import paper_scores

    monkeypatch.setattr(paper_scores, "score_session", fake_paper([], score=60.0))
    for m in ("extra_limbs", "bone_length", "joint_range", "self_collision"):
        monkeypatch.setitem(ss.SCORERS, m, lambda s, st, o: {"score": 80.0, "details": {}})
    assert ss.main(["--session-dir", str(tmp_path), "--stem", "clip", "--paper-scores"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["overall"] == pytest.approx((4 * 80 + 2 * 60) / 6)
    assert printed["paper_scores"]["overall"] == pytest.approx(60.0)
    assert (tmp_path / "scores.json").is_file()
