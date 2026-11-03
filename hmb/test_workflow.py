"""Tests of the single-video CLI (hmb.cli): engine options, artifact reuse and invalidation, scoring and its cache."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from hmb import cli
from hmb.layout import SessionLayout
from hmb.performance import RunTimings
from hmb.scoring.score_session import METRICS, tiers_and_overall
from hmb.test_batch import FakeEngines


def cli_args(tmp_path, *extra, command="prepare"):
    """``hmb.cli`` arguments for the fake video ``tmp_path/source/walk.mp4`` and session ``tmp_path/session``."""
    video = tmp_path / "source" / "walk.mp4"
    if not video.is_file():
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(b"fake video")
    return cli.build_parser().parse_args([command, "--video", str(video), "--session-dir", str(tmp_path / "session"),
                                          *extra])


def fake_report(stem, scores, paper=None):
    tiers, overall = tiers_and_overall(scores)
    report = {"stem": stem, "scores": scores, "tiers": tiers, "overall": overall,
              "status": {m: "ok" if scores[m] is not None else "missing_input" for m in METRICS},
              "metrics": {m: {"score": scores[m], "status": "ok" if scores[m] is not None else "missing_input",
                              "seconds": 0.0, **({} if scores[m] is not None else {"error": f"no input for {m}",
                                                                                     "reason": f"no input for {m}"})}
                          for m in METRICS}}
    if paper is not None:
        p_tiers, p_overall = tiers_and_overall(paper)
        report["paper_scores"] = {"scores": paper, "tiers": p_tiers, "overall": p_overall,
                                  "status": {m: "ok" for m in METRICS}, "reasons": {},
                                  "metrics": {m: {"score": paper[m], "status": "ok"} for m in METRICS}}
    return report


def fake_scorer(calls, scores=None, paper=None):
    """Stand-in for :func:`hmb.cli.score_session_subprocess` that writes ``<S>/scores.json``."""
    def run(session_dir, stem, paper_scores=False):
        calls.append({"stem": stem, "paper_scores": paper_scores})
        values = scores or {m: 50.0 + i for i, m in enumerate(METRICS)}
        paper_values = (paper or {m: 40.0 for m in METRICS}) if paper_scores else None
        (Path(session_dir) / "scores.json").write_text(json.dumps(fake_report(stem, values, paper_values)))
    return run


# ----------------------------------------------------------------------------- prepare
def test_prepare_maps_cli_options_to_engines(tmp_path, monkeypatch):
    engines = FakeEngines(tmp_path, monkeypatch)
    camera = tmp_path / "camera.json"
    camera.write_text('{"fx": 1000, "fy": 1000, "cx": 32, "cy": 24}')
    args = cli_args(tmp_path, "--max-iters", "7", "--camera-calibration", str(camera), "--sample-fps", "10",
                    "--max-side", "32", "--person", "track", "--gvhmr-f-mm", "24",
                    "--metrabs-model", str(tmp_path / "metrabs"), "--hadm-root", str(tmp_path / "hadm"))
    session = cli.prepare_session(args)
    assert sorted(engines.engines()) == ["detect", "fit", "hadm", "mesh"]
    fit = engines.last("fit")["options"]
    assert fit["max_iters"] == 7 and fit["batch"] == 1 and fit["camera_calibration"] == str(camera)
    assert fit["steps_per_call"] == 7  # one compiled scan length
    assert fit["modes"] == ["rigid"] and fit["unconstrained_joints"] is True  # the variable fit is paper-only
    detect = engines.last("detect")["options"]
    assert detect["person"] == "track" and detect["camera_calibration"] == str(camera)
    assert detect["model_path"] == str(tmp_path / "metrabs")
    mesh = engines.last("mesh")
    assert mesh["options"] == {"camera": "static", "f_mm": 24.0}
    assert mesh["extra"][mesh["extra"].index("--gvhmr-root") + 1] == str((cli.REPO_ROOT / "GVHMR").resolve())
    assert engines.last("hadm")["options"] == {"hadm_root": str((tmp_path / "hadm").resolve())}

    manifest = json.loads((session / "manifest.json").read_text())
    assert manifest["preparation_status"] == "ready" and manifest["video_name"] == "walk"
    config = manifest["config"]
    assert config["frame_plan"] == str(session / "input/frames.json")
    assert config["requested_fit_options"]["sample_fps"] == 10.0 and config["requested_fit_options"]["max_side"] == 32
    assert set(config["stage_fingerprints"]) == {"detection", "rigid_fit", "variable_fit", "mesh"}
    assert manifest["provenance"]["frame_sampling"]["frames"] == 5
    identity = json.loads((session / "cache/multi_limbs/inference_identity.json").read_text())
    assert identity["fingerprint"] == config["hadm_inference_fingerprint"]


def test_defaults_are_the_humanscore_protocol():
    args = cli.build_parser().parse_args(["run-all", "--video", "clip.mp4"])
    cli.validate_args(args)
    assert (args.sample_fps, args.max_side, args.person, args.static_cam) == (24.0, 1280, "track", True)
    assert not args.paper_scores
    assert cli.session_dir_from_args(args) == (Path("humanscore_output") / "clip").resolve()


def test_fit_steps_per_call_avoids_a_second_compilation():
    assert cli.fused_steps_per_call(10000) == 200
    assert cli.fused_steps_per_call(300) == 150
    assert cli.fused_steps_per_call(7) == 7
    assert cli.fused_steps_per_call(10001) == 137  # 73 x 137
    assert cli.fused_steps_per_call(10007) == 200  # prime: keep the default and its remainder call


def test_reconstruction_needs_follow_the_requested_scores():
    parse = cli.build_parser().parse_args
    assert cli.reconstruction_needs(parse(["run-all", "--video", "clip.mp4"])) == {"detect", "rigid", "mesh", "hadm"}
    assert cli.reconstruction_needs(parse(["run-all", "--video", "clip.mp4", "--paper-scores"])) == {
        "detect", "rigid", "variable", "mesh", "hadm"}
    assert cli.reconstruction_needs(parse(["prepare", "--video", "clip.mp4", "--skip-gvhmr"])) == {
        "detect", "rigid", "hadm"}


def test_option_validation(monkeypatch):
    parse = cli.build_parser().parse_args
    with pytest.raises(ValueError, match="positive"):
        cli.validate_args(parse(["prepare", "--video", "v.mp4", "--sample-fps", "-1"]))
    with pytest.raises(ValueError, match="engine-options"):
        cli.validate_args(parse(["prepare", "--video", "v.mp4", "--engine-options", '{"gpu": {}}']))
    args = parse(["prepare", "--video", "v.mp4", "--sample-fps", "0", "--max-side", "0",
                  "--engine-options", '{"hadm": {"precision": "fp32"}}'])
    cli.validate_args(args)
    assert args.sample_fps is None and args.max_side is None and args.engine_options == {"hadm": {"precision": "fp32"}}
    # A moving camera needs SimpleVO (pycolmap) for the mesh: fail clearly before any work.
    monkeypatch.setattr(cli, "module_available", lambda name: False)
    with pytest.raises(ValueError, match="pycolmap"):
        cli.validate_args(parse(["run-all", "--video", "v.mp4", "--moving-cam"]))
    cli.validate_args(parse(["run-all", "--video", "v.mp4", "--moving-cam", "--skip-gvhmr"]))
    monkeypatch.setattr(cli, "module_available", lambda name: True)
    args = parse(["run-all", "--video", "v.mp4", "--moving-cam"])
    cli.validate_args(args)
    assert cli.engine_options(args, {"hadm_root": "h"}, {"mesh"})["mesh"]["camera"] == "simplevo"


def test_options_are_matched_by_their_full_name_only():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["run-all", "--video", "v.mp4", "--engine", "{}"])   # not --engine-options


def test_prepare_reuses_and_invalidates_by_fingerprint(tmp_path, monkeypatch):
    engines = FakeEngines(tmp_path, monkeypatch)
    session = cli.prepare_session(cli_args(tmp_path, "--max-iters", "5"))
    engines.clear()
    cli.prepare_session(cli_args(tmp_path, "--max-iters", "5"))
    assert engines.runs() == []  # everything verified and reused

    keypoints = session / "baseline/walk_keypoints.npz"
    keypoints.write_text("same detection")  # content marker: must survive a fit-only change
    cli.prepare_session(cli_args(tmp_path, "--max-iters", "5"))  # keypoint content changed -> the fit
    assert engines.engines() == ["fit"] and engines.last("fit")["options"]["modes"] == ["rigid"]
    engines.clear()
    cli.prepare_session(cli_args(tmp_path, "--max-iters", "6"))
    assert engines.engines() == ["fit"] and keypoints.read_text() == "same detection"
    engines.clear()
    hadm = '{"hadm": {"precision": "fp32"}}'
    cli.prepare_session(cli_args(tmp_path, "--max-iters", "6", "--engine-options", hadm))
    assert engines.engines() == ["hadm"]
    engines.clear()
    cli.prepare_session(cli_args(tmp_path, "--max-iters", "6", "--engine-options", hadm, "--sample-fps", "10"))
    assert sorted(engines.engines()) == ["detect", "fit", "hadm", "mesh"]
    assert keypoints.read_text() == "keypoints"

    # A changed rigid-fit fingerprint (e.g. an edited hmb/fit_engine.py) refits only that fit.
    engines.clear()
    manifest = json.loads((session / "manifest.json").read_text())
    manifest["config"]["stage_fingerprints"]["rigid_fit"] = "stale"
    (session / "manifest.json").write_text(json.dumps(manifest))
    cli.prepare_session(cli_args(tmp_path, "--max-iters", "6", "--engine-options", hadm, "--sample-fps", "10"))
    assert engines.engines() == ["fit"] and engines.last("fit")["options"]["modes"] == ["rigid"]


def test_paper_scores_after_a_default_run_add_only_the_variable_fit(tmp_path, monkeypatch):
    engines = FakeEngines(tmp_path, monkeypatch)
    session = cli.prepare_session(cli_args(tmp_path))
    rigid = session / "baseline/walk_fitted_model.npz"
    before = rigid.stat().st_mtime_ns
    assert not (session / "bone_fit/walk_bone_length.npz").exists()
    engines.clear()
    cli.prepare_session(cli_args(tmp_path, "--paper-scores"))
    assert engines.engines() == ["fit"] and engines.last("fit")["options"]["modes"] == ["variable"]
    assert rigid.stat().st_mtime_ns == before
    assert (session / "bone_fit/walk_bone_length.npz").is_file()
    engines.clear()
    cli.prepare_session(cli_args(tmp_path))  # a later default run keeps both fits
    assert engines.runs() == []


def test_session_rejects_a_different_input(tmp_path, monkeypatch):
    FakeEngines(tmp_path, monkeypatch)
    cli.prepare_session(cli_args(tmp_path))
    another = tmp_path / "source" / "other.mp4"
    another.write_bytes(b"other")
    args = cli_args(tmp_path)
    args.video = str(another)
    with pytest.raises(ValueError, match="different video"):
        cli.prepare_session(args)
    assert (tmp_path / "session/input/walk.mp4").resolve() == (tmp_path / "source/walk.mp4").resolve()


def test_changed_inputs_are_detected_before_scoring(tmp_path, monkeypatch):
    FakeEngines(tmp_path, monkeypatch)
    camera = tmp_path / "camera.json"
    camera.write_text('{"fx": 1000, "fy": 1000, "cx": 32, "cy": 24}')
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "SMPLX_NEUTRAL.npz").write_bytes(b"model")
    session = cli.prepare_session(cli_args(tmp_path, "--camera-calibration", str(camera),
                                           "--smplx-model-dir", str(model_dir)))
    manifest = json.loads((session / "manifest.json").read_text())
    cli.validate_source(manifest)
    camera.write_text('{"fx": 900, "fy": 900, "cx": 32, "cy": 24}')
    with pytest.raises(ValueError, match="Camera calibration changed"):
        cli.validate_source(manifest)
    camera.write_text('{"fx": 1000, "fy": 1000, "cx": 32, "cy": 24}')
    (model_dir / "SMPLX_NEUTRAL.npz").write_bytes(b"updated model")
    with pytest.raises(ValueError, match="SMPL-X asset changed"):
        cli.validate_source(manifest)
    video = tmp_path / "source/walk.mp4"
    video.write_bytes(b"changed input")
    with pytest.raises(ValueError, match="changed since preparation"):
        cli.validate_source(manifest)
    with pytest.raises(ValueError, match="changed since preparation"):
        cli.prepare_session(cli_args(tmp_path, "--camera-calibration", str(camera),
                                     "--smplx-model-dir", str(model_dir)))


def test_force_failure_removes_old_artifacts_and_caches(tmp_path, monkeypatch):
    engines = FakeEngines(tmp_path, monkeypatch)
    session = cli.prepare_session(cli_args(tmp_path))
    (session / "cache/stale_detection.json").write_text("{}")
    (session / "scores.json").write_text("{}")
    engines.set("FAKE_ENGINE_FAIL", "detect,mesh,hadm,fit")
    with pytest.raises(RuntimeError, match="engine"):
        cli.prepare_session(cli_args(tmp_path, "--force"))
    assert not list((session / "baseline").iterdir())
    assert not (session / "cache/stale_detection.json").exists()
    assert not (session / "scores.json").exists()
    assert json.loads((session / "manifest.json").read_text())["preparation_status"] == "preparing"


def test_changed_detector_root_clears_inference_cache(tmp_path, monkeypatch):
    engines = FakeEngines(tmp_path, monkeypatch)
    session = cli.prepare_session(cli_args(tmp_path, "--hadm-root", str(tmp_path / "hadm-a")))
    cached = session / "cache/multi_limbs/walk/hadm_results/output.json"
    cached.write_text('{"cached": true}')
    engines.clear()
    cli.prepare_session(cli_args(tmp_path, "--hadm-root", str(tmp_path / "hadm-a")))
    assert engines.runs() == [] and cached.read_text() == '{"cached": true}'
    cli.prepare_session(cli_args(tmp_path, "--hadm-root", str(tmp_path / "hadm-b")))
    assert engines.engines() == ["hadm"] and cached.read_text() == "{}"


def test_sampling_options_that_select_the_same_frames_rebuild_nothing(tmp_path, monkeypatch):
    engines = FakeEngines(tmp_path, monkeypatch)  # probe: 30 fps, 12 frames, 64x48
    session = cli.prepare_session(cli_args(tmp_path, "--sample-fps", "0", "--max-side", "0"))
    plan = (session / "input/frames.json").read_text()
    fingerprints = json.loads((session / "manifest.json").read_text())["config"]["stage_fingerprints"]
    for extra in (("--sample-fps", "30", "--max-side", "0"), ("--sample-fps", "60", "--max-side", "64"),
                  ("--sample-fps", "0", "--max-side", "1280")):
        engines.clear()
        cli.prepare_session(cli_args(tmp_path, *extra))
        assert engines.runs() == [], extra
        manifest = json.loads((session / "manifest.json").read_text())
        assert (session / "input/frames.json").read_text() == plan
        assert manifest["config"]["stage_fingerprints"] == fingerprints
        assert manifest["provenance"]["frame_sampling"]["frames"] == 12
    # Fewer frames is a new plan: every product is rebuilt.
    engines.clear()
    cli.prepare_session(cli_args(tmp_path, "--sample-fps", "24"))
    assert sorted(engines.engines()) == ["detect", "fit", "hadm", "mesh"]
    assert json.loads((session / "input/frames.json").read_text())["sample_fps"] == 24.0


# ----------------------------------------------------------------------------- score
def prepared_session(tmp_path, monkeypatch, *extra):
    FakeEngines(tmp_path, monkeypatch)
    session = cli.prepare_session(cli_args(tmp_path, *extra))
    layout = SessionLayout.create(session)
    return layout, json.loads(layout.manifest_path.read_text())


def test_scores_are_reused_while_verified(tmp_path, monkeypatch):
    layout, manifest = prepared_session(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(cli, "score_session_subprocess", fake_scorer(calls))
    summary, ok = cli.run_scores(manifest, layout, RunTimings(layout.session_dir, "t"))
    assert ok and calls == [{"stem": "walk", "paper_scores": False}]
    assert summary["overall"] == pytest.approx(52.5) and summary["status"] == {m: "ok" for m in METRICS}
    assert "paper_scores" not in summary and (layout.cache_dir / "score_cache.json").is_file()
    assert cli.run_scores(manifest, layout, RunTimings(layout.session_dir, "t"))[1] and len(calls) == 1
    # an input the scorer reads changed (e.g. a refitted motion export): rescored
    (layout.baseline_dir / "walk_opensim_skeleton.mot").write_text("refit")
    assert cli.run_scores(manifest, layout, RunTimings(layout.session_dir, "t"))[1] and len(calls) == 2
    # a tampered report is never reused
    (layout.session_dir / "scores.json").write_text(json.dumps(fake_report("walk", {m: 1.0 for m in METRICS})))
    summary, _ = cli.run_scores(manifest, layout, RunTimings(layout.session_dir, "t"))
    assert len(calls) == 3 and summary["overall"] == pytest.approx(52.5)
    cli.run_scores(manifest, layout, RunTimings(layout.session_dir, "t"), rescore=True)
    assert len(calls) == 4


def test_paper_scores_are_computed_on_request_and_a_superset_report_is_reused(tmp_path, monkeypatch):
    layout, manifest = prepared_session(tmp_path, monkeypatch, "--paper-scores")
    calls = []
    monkeypatch.setattr(cli, "score_session_subprocess", fake_scorer(calls))
    cli.run_scores(manifest, layout, RunTimings(layout.session_dir, "t"))
    summary, ok = cli.run_scores(manifest, layout, RunTimings(layout.session_dir, "t"), paper_scores=True)
    assert ok and [c["paper_scores"] for c in calls] == [False, True]  # without paper scores: recomputed
    assert summary["paper_scores"]["overall"] == pytest.approx(40.0)
    assert set(summary["paper_scores"]) == {"scores", "tiers", "overall", "status", "reasons"}
    summary, ok = cli.run_scores(manifest, layout, RunTimings(layout.session_dir, "t"))
    assert ok and len(calls) == 2 and "paper_scores" not in summary  # the paper report serves a default request


def test_failed_scoring_reports_every_metric(tmp_path, monkeypatch):
    layout, manifest = prepared_session(tmp_path, monkeypatch)

    def crash(session_dir, stem, paper_scores=False):
        raise RuntimeError("scorer crashed")

    monkeypatch.setattr(cli, "score_session_subprocess", crash)
    summary, ok = cli.run_scores(manifest, layout, RunTimings(layout.session_dir, "t"))
    assert not ok and summary["overall"] is None
    assert summary["status"] == {m: "error" for m in METRICS}
    assert summary["reasons"]["bone_length"] == "scorer crashed"

    def changes_input(session_dir, stem, paper_scores=False):
        fake_scorer([])(session_dir, stem, paper_scores)
        (layout.baseline_dir / "walk_keypoints.npz").write_text("changed while scoring")

    monkeypatch.setattr(cli, "score_session_subprocess", changes_input)
    summary, ok = cli.run_scores(manifest, layout, RunTimings(layout.session_dir, "t"))
    assert not ok and "changed while computing" in summary["reasons"]["extra_limbs"]
    assert not (layout.cache_dir / "score_cache.json").exists()


def test_missing_metric_makes_the_run_incomplete(tmp_path, monkeypatch):
    layout, manifest = prepared_session(tmp_path, monkeypatch)
    scores = {m: 80.0 for m in METRICS}
    scores["self_collision"] = None
    monkeypatch.setattr(cli, "score_session_subprocess", fake_scorer([], scores=scores))
    summary, ok = cli.run_scores(manifest, layout, RunTimings(layout.session_dir, "t"))
    assert not ok and summary["overall"] is None and summary["tiers"]["anatomy"] == 80.0
    assert summary["status"]["self_collision"] == "missing_input"
    assert summary["reasons"] == {"self_collision": "no input for self_collision"}


def test_engine_failure_is_isolated_in_run_all(tmp_path, monkeypatch):
    engines = FakeEngines(tmp_path, monkeypatch)
    engines.set("FAKE_ENGINE_FAIL", "mesh")
    calls = []
    monkeypatch.setattr(cli, "score_session_subprocess", fake_scorer(calls))
    args = cli_args(tmp_path, command="run-all")
    timings = RunTimings(tmp_path / "session", "run-all")
    summary, ok = cli.execute_video(args, timings)
    assert ok is False and len(calls) == 1  # the available artifacts are still scored
    manifest = json.loads((tmp_path / "session/manifest.json").read_text())
    assert manifest["preparation_status"] == "preparing" and "mesh: mesh exploded" in manifest["preparation_error"]
    stages = {entry["stage"]: entry for entry in timings.payload["stages"]}
    assert stages["engine:mesh"]["status"] == "error" and stages["engine:detect"]["status"] == "ok"
    assert stages["reconstruction_engines"]["status"] == "error" and stages["frame_plan"]["status"] == "ok"
    assert stages["scoring"]["status"] == "ok"
    # The next run only retries the failed engine.
    engines.clear()
    monkeypatch.delenv("FAKE_ENGINE_FAIL")
    summary, ok = cli.execute_video(args, RunTimings(tmp_path / "session", "run-all"))
    assert ok and engines.engines() == ["mesh"]
    assert json.loads((tmp_path / "session/manifest.json").read_text())["preparation_status"] == "ready"


def test_score_command_prints_the_table_and_sets_the_exit_status(tmp_path, monkeypatch, capsys):
    layout, _ = prepared_session(tmp_path, monkeypatch)
    monkeypatch.setattr(cli, "score_session_subprocess", fake_scorer([]))
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["score", "--session-dir", str(layout.session_dir), "--paper-scores"])
    assert exit_info.value.code == 0
    printed = capsys.readouterr().out
    assert "Scores for walk" in printed and "OVERALL" in printed and "paper" in printed
    assert "  52.5" in printed and "  40.0" in printed
    monkeypatch.setattr(cli, "score_session_subprocess",
                        fake_scorer([], scores={**{m: 70.0 for m in METRICS}, "bone_length": None}))
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["score", "--session-dir", str(layout.session_dir), "--rescore"])
    assert exit_info.value.code == 1


def test_scorer_runs_with_the_sessions_model_directory(tmp_path, monkeypatch):
    layout = SessionLayout.create(tmp_path / "session")
    selected = tmp_path / "selected"
    selected.mkdir()
    (layout.manifest_path).write_text(json.dumps({"config": {"smplx_model_dir": str(selected)}}))
    monkeypatch.setenv("HMB_SMPLX_MODEL", str(tmp_path / "unrelated/SMPLX_NEUTRAL.npz"))
    calls = []

    def fake_run(cmd, cwd=None, env=None):
        calls.append((cmd, env))
        (layout.session_dir / "scores.json").write_text("{}")

    monkeypatch.setattr(cli.process_groups, "run", fake_run)
    cli.score_session_subprocess(layout.session_dir, "clip")
    cmd, env = calls[-1]
    assert cmd[1:3] == ["-m", "hmb.scoring.score_session"] and "--paper-scores" not in cmd
    # no model in the configured folder: the caller's environment and the default search are kept
    assert env["HMB_SMPLX_MODEL"] == str(tmp_path / "unrelated/SMPLX_NEUTRAL.npz")
    (selected / "SMPLX_NEUTRAL.npz").write_bytes(b"model")
    cli.score_session_subprocess(layout.session_dir, "clip", paper_scores=True)
    cmd, env = calls[-1]
    assert "--paper-scores" in cmd and env["HMB_SMPLX_MODEL"] == "" and env["HMB_SMPLX_MODEL_DIR"] == str(selected)
    assert cli.session_smplx_dir(tmp_path / "no_session") is None


def test_cli_help_runs_from_a_subprocess():
    for command in (["--help"], ["prepare", "--help"], ["score", "--help"], ["run-all", "--help"]):
        out = subprocess.run([sys.executable, "-m", "hmb.cli", *command], cwd=cli.REPO_ROOT, capture_output=True,
                             text=True, env={**os.environ, "PYTHONPATH": str(cli.REPO_ROOT)})
        assert out.returncode == 0, out.stderr
