"""hmb.pipeline.score_video and scripts/humanscore_single.py on CPU.

The four reconstruction engines are the fake engine subprocesses of
``hmb.test_batch.FakeEngines``; the scorer subprocess is replaced by a fake that
writes ``<S>/scores.json``, so the real ``hmb.cli`` orchestration (plan,
fingerprints, caches, summaries) runs in between.
"""
import importlib.util
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from hmb import cli, pipeline
from hmb.report import is_complete
from hmb.scoring.score_session import METRICS, TIERS, tiers_and_overall
from hmb.test_batch import FakeEngines

VALUES = {name: 60.0 + 5 * i for i, name in enumerate(METRICS)}   # 60, 65, ..., 85
PAPER_VALUES = {name: 40.0 + 2 * i for i, name in enumerate(METRICS)}
CONTRACT_KEYS = {"video", "scores", "tiers", "overall", "status", "reasons", "person", "files", "timings"}


class Fakes:
    def __init__(self, tmp_path, monkeypatch):
        self.engines = FakeEngines(tmp_path, monkeypatch)
        self.calls = []
        monkeypatch.setattr(cli, "score_session_subprocess", self.score)

    def entries(self, session: Path, stem: str, values: dict) -> dict:
        has_mesh = (session / "gvhmr" / stem / "hmr4d_results.pt").is_file()
        entries = {}
        for name in METRICS:
            if name == "self_collision" and not has_mesh:
                entries[name] = {"score": None, "status": "missing_input", "error": "no mesh", "reason": "no mesh",
                                 "seconds": 0.0}
            else:
                entries[name] = {"score": values[name], "status": "ok", "seconds": 0.0}
        return entries

    def score(self, session_dir, stem, paper_scores=False):
        self.calls.append({"stem": stem, "paper_scores": paper_scores})
        print(f"[fake scorer] scoring {stem}")
        # a child process writing to the inherited stdout (as the real scorer subprocess does)
        subprocess.run([sys.executable, "-c", f"print('[fake scorer child] {stem}')"], check=True)
        session = Path(session_dir)
        entries = self.entries(session, stem, VALUES)
        scores = {m: entries[m]["score"] for m in METRICS}
        tiers, overall = tiers_and_overall(scores)
        report = {"stem": stem, "scores": scores, "tiers": tiers, "overall": overall, "metrics": entries,
                  "status": {m: e["status"] for m, e in entries.items()},
                  "reasons": {m: e["reason"] for m, e in entries.items() if e.get("reason")}}
        if paper_scores:
            paper = self.entries(session, stem, PAPER_VALUES)
            p_scores = {m: paper[m]["score"] for m in METRICS}
            p_tiers, p_overall = tiers_and_overall(p_scores)
            report["paper_scores"] = {"scores": p_scores, "tiers": p_tiers, "overall": p_overall,
                                      "status": {m: e["status"] for m, e in paper.items()},
                                      "reasons": {m: e["reason"] for m, e in paper.items() if e.get("reason")},
                                      "metrics": paper}
        (session / "scores.json").write_text(json.dumps(report))


@pytest.fixture
def fakes(tmp_path, monkeypatch):
    return Fakes(tmp_path, monkeypatch)


def make_video(tmp_path, name="walk clip.mp4"):
    video = tmp_path / "videos" / name
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"fake video")
    return video


def test_score_video_runs_the_cli_pipeline_and_writes_the_contract(tmp_path, fakes):
    video = make_video(tmp_path)
    out = tmp_path / "out"
    result = pipeline.score_video(video, out)

    assert set(result) == CONTRACT_KEYS
    assert result["video"] == str(video.resolve()) and result["files"]["session"] == str(out.resolve())
    assert result["files"]["report"] == str(out.resolve() / "scores.json")
    assert result["scores"] == VALUES and result["status"] == {m: "ok" for m in METRICS} and result["reasons"] == {}
    assert is_complete(result)
    assert result["tiers"] == {"anatomy": 62.5, "kinematic": 72.5, "kinetic": 82.5}
    assert result["overall"] == pytest.approx(72.5)
    assert set(result["timings"]) == {"total_seconds", "reconstruction_seconds", "scoring_seconds"}
    assert result["timings"]["total_seconds"] > 0 and result["timings"]["reconstruction_seconds"] is not None
    assert fakes.calls == [{"stem": "walk_clip", "paper_scores": False}]

    # the HumanScore protocol: 24 fps, 1280 px, the tracked person, a static camera; no variable fit
    detect, mesh = fakes.engines.last("detect")["options"], fakes.engines.last("mesh")["options"]
    assert detect["person"] == "track" and "person_boxes" not in detect
    assert mesh["camera"] == "static" and fakes.engines.last("fit")["options"]["modes"] == ["rigid"]
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["config"]["requested_fit_options"]["sample_fps"] == 24.0
    assert manifest["config"]["requested_fit_options"]["max_side"] == 1280

    on_disk = json.loads((out / "humanscore.json").read_text())
    assert on_disk == json.loads(json.dumps(result))
    summary = (out / "summary.txt").read_text()
    assert "HumanScore: walk clip.mp4" in summary and "OVERALL" in summary and "72.5" in summary
    assert "self-collision" in summary and "incomplete" not in summary and "paper" not in summary

    # a second call reuses every verified artifact and score
    fakes.engines.clear()
    again = pipeline.score_video(video, out)
    assert fakes.engines.runs() == [] and len(fakes.calls) == 1
    assert again["scores"] == result["scores"]


def test_paper_scores_are_added_on_request(tmp_path, fakes):
    video = make_video(tmp_path)
    result = pipeline.score_video(video, tmp_path / "out", paper_scores=True)
    assert set(result) == CONTRACT_KEYS | {"paper_scores"}
    paper = result["paper_scores"]
    assert set(paper) == {"scores", "tiers", "overall", "status", "reasons"}
    assert paper["scores"] == PAPER_VALUES and paper["overall"] == pytest.approx(sum(PAPER_VALUES.values()) / 6)
    assert fakes.engines.last("fit")["options"]["modes"] == ["rigid", "variable"]
    summary = (tmp_path / "out" / "summary.txt").read_text()
    header = [line for line in summary.splitlines() if "metric" in line and "status" in line][0]
    assert header.split()[-3:] == ["score", "paper", "status"]
    # the default call afterwards reuses the report but shows no paper scores
    again = pipeline.score_video(video, tmp_path / "out")
    assert "paper_scores" not in again and len(fakes.calls) == 1


def test_failed_engine_gives_an_incomplete_result_with_the_other_scores(tmp_path, fakes):
    fakes.engines.set("FAKE_ENGINE_FAIL", "mesh")
    result = pipeline.score_video(make_video(tmp_path), tmp_path / "out", paper_scores=True)
    assert not is_complete(result) and result["status"]["self_collision"] == "missing_input"
    assert result["scores"]["self_collision"] is None and result["scores"]["bone_length"] == VALUES["bone_length"]
    assert result["tiers"]["kinematic"] is None and result["tiers"]["anatomy"] == 62.5 and result["overall"] is None
    assert result["paper_scores"]["status"]["self_collision"] == "missing_input"
    assert result["paper_scores"]["overall"] is None
    summary = (tmp_path / "out" / "summary.txt").read_text()
    assert "incomplete: self_collision missing_input" in summary
    assert result["reasons"] == {"self_collision": "no mesh"} and "    self_collision missing_input: no mesh" in summary
    assert "paper self_collision missing_input: no mesh" in summary


def test_unsupported_metric_reason_from_the_scorer_details(tmp_path, fakes, monkeypatch):
    original = fakes.score

    def bone_length_unsupported(session_dir, stem, paper_scores=False):
        original(session_dir, stem, paper_scores)
        path = Path(session_dir) / "scores.json"
        report = json.loads(path.read_text())
        # as hmb.scoring.score_session writes it for a person seen in too few frames (no "reason": derived here)
        report["metrics"]["bone_length"] = {"score": None, "status": "unsupported",
                                            "details": {"coverage": 48 / 270, "supported": False,
                                                        "config": {"min_coverage": 0.25}}}
        report["scores"]["bone_length"] = None
        path.write_text(json.dumps(report))

    monkeypatch.setattr(cli, "score_session_subprocess", bone_length_unsupported)
    result = pipeline.score_video(make_video(tmp_path), tmp_path / "out")
    assert result["status"]["bone_length"] == "unsupported" and not is_complete(result)
    assert result["reasons"] == {"bone_length": "keypoint coverage 0.18 < 0.25 of the analysed frames"}
    on_disk = json.loads((tmp_path / "out" / "humanscore.json").read_text())
    assert on_disk["reasons"] == result["reasons"]
    summary = (tmp_path / "out" / "summary.txt").read_text()
    assert "bone_length unsupported: keypoint coverage 0.18 < 0.25 of the analysed frames" in summary


def test_summary_does_not_count_other_people_when_following_person_boxes():
    result = {"video": "/x/person_3.mp4", "scores": {m: 80.0 for m in METRICS},
              "status": {m: "ok" for m in METRICS}, "tiers": {t: 80.0 for t in TIERS},
              "overall": 80.0, "reasons": {}, "files": {"session": "/x"},
              "person": {"selection": "boxes", "frames_analysed": 270, "frames_found": 48,
                         "frames_with_several_people": 80}}
    table = pipeline.format_table(result)
    assert "person found in 48/270 frames (selection: follows the given person boxes)" in table
    assert "several people" not in table
    result["person"]["selection"] = "track"
    assert "(selection: track); several people visible in 80 frames" in pipeline.format_table(result)


def test_person_boxes_drive_the_detection_engine_and_invalidate_on_change(tmp_path, fakes):
    video = make_video(tmp_path)
    boxes = tmp_path / "person_boxes.json"
    boxes.write_text(json.dumps({"timestamps": [0.0, 1 / 30], "boxes_xyxy": [[1, 2, 30, 40], None]}))
    pipeline.score_video(video, tmp_path / "out", person_boxes=boxes)
    detect = fakes.engines.last("detect")["options"]
    assert detect["person"] == "boxes" and detect["person_boxes"] == str(boxes.resolve())
    assert len(detect["person_boxes_sha256"]) == 64
    mesh = fakes.engines.last("mesh")["options"]   # the body mesh follows the same person
    assert (mesh["person_boxes"], mesh["person_boxes_sha256"]) == (detect["person_boxes"],
                                                                   detect["person_boxes_sha256"])
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert manifest["config"]["requested_fit_options"]["person"] == "boxes"

    fakes.engines.clear()
    pipeline.score_video(video, tmp_path / "out", person_boxes=boxes)
    assert fakes.engines.runs() == []                     # unchanged track: everything reused
    boxes.write_text(json.dumps({"timestamps": [0.0, 1 / 30], "boxes_xyxy": [[1, 2, 30, 40], [2, 2, 31, 40]]}))
    pipeline.score_video(video, tmp_path / "out", person_boxes=boxes)
    assert sorted(fakes.engines.engines()) == ["detect", "fit", "mesh"]  # new prior: keypoints, fit and mesh


def test_quiet_sends_all_output_to_the_log(tmp_path, fakes, capfd):
    out = tmp_path / "out"
    pipeline.score_video(make_video(tmp_path), out, quiet=True)
    console = capfd.readouterr()
    lines = [line for line in console.out.splitlines() if line.strip()]
    assert lines and all(line.startswith("[humanscore]") for line in lines), console.out
    assert any("3-D keypoints (MeTRAbs): done" in line for line in lines)
    assert any(line.strip().endswith("scoring: six HumanScore metrics") for line in lines) and "done in" in lines[-1]
    assert console.err == ""
    log = (out / "logs" / "humanscore.log").read_text()
    assert "[fake scorer] scoring walk_clip" in log and "[fake scorer child] walk_clip" in log
    assert "[HMB] Report:" in log
    # the console is restored afterwards
    print("after")
    assert capfd.readouterr().out == "after\n"


def test_invalid_inputs_fail_before_any_work(tmp_path, fakes, monkeypatch):
    video = make_video(tmp_path)
    out = tmp_path / "out"
    with pytest.raises(FileNotFoundError):
        pipeline.score_video(tmp_path / "missing.mp4", out)
    with pytest.raises(ValueError, match="person"):
        pipeline.score_video(video, out, person="largest")
    with pytest.raises(ValueError, match="sample_fps"):
        pipeline.score_video(video, out, sample_fps=-1)
    bad = tmp_path / "bad_boxes.json"
    bad.write_text(json.dumps({"timestamps": [0.0], "boxes_xyxy": [[1, 2, 3]]}))
    with pytest.raises(ValueError, match="person-boxes"):
        pipeline.score_video(video, out, person_boxes=bad)
    monkeypatch.setattr(cli, "module_available", lambda name: False)
    with pytest.raises(ValueError, match="pycolmap") as moving:
        pipeline.score_video(video, out, static_cam=False)
    assert "--moving-cam" in str(moving.value) and "--skip-gvhmr" not in str(moving.value)
    assert not out.exists() and fakes.engines.runs() == []


def test_person_coverage_is_read_from_the_keypoints(tmp_path):
    session = tmp_path / "S"
    (session / "baseline").mkdir(parents=True)
    confs = np.ones((5, 87))
    confs[[1, 3]] = 0
    np.savez(session / "baseline" / "clip_keypoints.npz", confs=confs, person_selection=np.array("boxes"),
             num_detections=np.array([1, 2, 0, 1, 3]))
    assert pipeline.person_coverage(session, "clip") == {"selection": "boxes", "frames_analysed": 5,
                                                         "frames_found": 3, "frames_with_several_people": 2}
    (session / "baseline" / "other_keypoints.npz").write_text("not an npz")
    assert pipeline.person_coverage(session, "other") is None and pipeline.person_coverage(session, "missing") is None


def _load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "humanscore_single.py"
    spec = importlib.util.spec_from_file_location("humanscore_single", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_single_script_end_to_end_with_default_output_dir(tmp_path, fakes, monkeypatch, capfd):
    script = _load_script()
    video = make_video(tmp_path, "Run Fast.mp4")
    monkeypatch.chdir(tmp_path)
    assert script.main([str(video)]) == 0
    out = (tmp_path / "humanscore_output" / "Run_Fast").resolve()
    printed = capfd.readouterr().out
    assert "HumanScore: Run Fast.mp4" in printed and "OVERALL" in printed
    assert f"results: {out / 'humanscore.json'}" in printed
    result = json.loads((out / "humanscore.json").read_text())
    assert result["scores"] == VALUES and (out / "summary.txt").is_file()

    fakes.engines.set("FAKE_ENGINE_FAIL", "mesh")
    assert script.main([str(video), "--out", str(tmp_path / "partial"), "--sample-fps", "0", "--max-side", "0",
                        "--paper-scores"]) == 1
    manifest = json.loads((tmp_path / "partial" / "manifest.json").read_text())
    assert manifest["config"]["requested_fit_options"]["sample_fps"] is None
    assert "paper_scores" in json.loads((tmp_path / "partial" / "humanscore.json").read_text())
    assert script.main([str(tmp_path / "nope.mp4")]) == 2
    assert "error: video not found" in capfd.readouterr().err


def test_cli_person_boxes_flags_are_validated(tmp_path):
    video = make_video(tmp_path)
    boxes = tmp_path / "boxes.json"
    boxes.write_text(json.dumps({"timestamps": [0.0], "boxes_xyxy": [[0, 0, 4, 4]]}))

    def parse(*extra):
        return cli.build_parser().parse_args(["run-all", "--video", str(video), "--session-dir", str(tmp_path / "S"),
                                              *extra])

    args = parse("--person-boxes", str(boxes))
    cli.validate_args(args)
    assert args.person == "boxes" and args.person_boxes == str(boxes.resolve())
    for extra, message in ((("--person", "boxes"), "needs --person-boxes"),
                           (("--person", "track", "--person-boxes", str(boxes)), "cannot be combined"),
                           (("--person-boxes", str(tmp_path / "missing.json")), "not found")):
        with pytest.raises(ValueError, match=message):
            cli.validate_args(parse(*extra))


# scripts/humanscore_single.py with the fake engines in a separate process, which the test signals.
SCRIPT_DRIVER = r'''
import signal, sys
from pathlib import Path
signal.signal(signal.SIGINT, signal.default_int_handler)   # in case the test runner ignores them
signal.signal(signal.SIGTERM, signal.SIG_DFL)
sys.path.insert(0, sys.argv[1])
import pytest
from hmb.test_pipeline import Fakes, _load_script, make_video
tmp, pids = Path(sys.argv[2]), sys.argv[3]
patches = pytest.MonkeyPatch()
Fakes(tmp, patches)
for name in ("DETECT", "MESH", "HADM"):
    patches.setenv(f"FAKE_ENGINE_DELAY_{name}", "300")
patches.setenv("FAKE_ENGINE_PIDS", pids)
patches.setenv("FAKE_ENGINE_CHILD", "1")   # every engine starts a process of its own
sys.exit(_load_script().main([str(make_video(tmp)), "--out", str(tmp / "out")]))
'''


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
def test_single_script_stops_the_engines_and_their_children_on_sigterm_or_sigint(tmp_path, signum):
    from hmb.test_batch import wait_until_gone
    from hmb.test_process_groups import kill_all, recorded_pids, wait_for

    pids = tmp_path / "pids.txt"
    # the driver imports pytest and this module: give it the test runner's import path
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    driver = subprocess.Popen([sys.executable, "-c", SCRIPT_DRIVER, str(Path(__file__).resolve().parents[1]),
                               str(tmp_path), str(pids)], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, start_new_session=True, env=env)
    try:
        recorded = wait_for(pids, [f"{engine}{suffix}" for engine in ("detect", "mesh", "hadm")
                                   for suffix in ("", "-child")], timeout=60)
        driver.send_signal(signum)  # to the script alone, as `kill PID` does
        _, err = driver.communicate(timeout=60)
        assert driver.returncode == 128 + signum
        reason = "terminated by SIGTERM" if signum == signal.SIGTERM else "interrupted"
        assert f"{reason}; run the same command again to resume" in err
        assert wait_until_gone(list(recorded.values())) == []
        log = (tmp_path / "out" / "logs" / "humanscore.log").read_text()
        assert all(f"{engine} finished with code -15" in log for engine in ("detect", "mesh", "hadm"))
    finally:
        driver.kill()
        kill_all(recorded_pids(pids).values())
        driver.wait()
