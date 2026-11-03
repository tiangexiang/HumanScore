"""CPU tests of hmb.batch: run_session layout, option mapping, ordering and failure isolation.

The four engines are replaced by one small fake engine script started as real
subprocesses (so the Stage/concurrency machinery itself is exercised); video
probing is replaced by a fixed probe.
"""
import json
import os
import sys
import time
from pathlib import Path

import pytest

from hmb import batch
from hmb import video as hvideo

FAKE_ENGINE = r'''
import json, os, sys, time
from pathlib import Path

name, jobs_file = sys.argv[1], sys.argv[2]
extra = sys.argv[3:]
spec = json.loads(Path(jobs_file).read_text())
record = Path(os.environ["FAKE_ENGINE_RECORD"])
fail = set(filter(None, os.environ.get("FAKE_ENGINE_FAIL", "").split(",")))
crash = set(filter(None, os.environ.get("FAKE_ENGINE_CRASH", "").split(",")))
started = time.time()
if os.environ.get("FAKE_ENGINE_PIDS"):
    with open(os.environ["FAKE_ENGINE_PIDS"], "a") as fh:
        fh.write(f"{name} {os.getpid()}\n")
if os.environ.get("FAKE_ENGINE_CHILD"):
    # a process the engine starts itself (like a data-loader worker); it must not outlive the engine's group
    import subprocess
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
    with open(os.environ["FAKE_ENGINE_PIDS"], "a") as fh:
        fh.write(f"{name}-child {child.pid}\n")
time.sleep(float(os.environ.get("FAKE_ENGINE_DELAY_" + name.upper(), "0")))
if name in crash:
    print(f"{name}: simulated crash before any status")
    sys.exit(3)
status_name = {"detect": "detect.json", "mesh": "mesh.json", "hadm": "hadm.json", "fit": "fit_engine.json"}[name]
code = 0


def touch(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


for job in spec["jobs"]:
    session, stem = Path(job["session_dir"]), job["stem"]
    status = {"ok": True, "error": None, "engine": name}
    if name in fail:
        status.update(ok=False, error=f"{name} exploded")
    elif name == "detect":
        touch(session / "baseline" / f"{stem}_keypoints.npz", "keypoints")
    elif name == "mesh":
        touch(session / "gvhmr" / stem / "hmr4d_results.pt", "mesh")
    elif name == "hadm":
        touch(session / "cache" / "multi_limbs" / stem / "hadm_results" / "output.json", "{}")
    elif name == "fit":
        if not (session / "baseline" / f"{stem}_keypoints.npz").is_file():
            status.update(ok=False, error="No MeTRAbs keypoints")
        for mode in spec["options"].get("modes", ["rigid", "variable"]):
            if not status["ok"]:
                break
            if mode == "rigid":
                touch(session / "baseline" / f"{stem}_fitted_model.npz", "rigid")
                touch(session / "baseline" / f"{stem}_opensim_skeleton.mot", "mot")
            else:
                touch(session / "bone_fit" / f"{stem}_bone_length.npz", "variable")
                touch(session / "baseline" / f"{stem}_bone_length.npz", "variable")
    if not status["ok"]:
        code = 1
    touch(session / "cache" / "engine_status" / status_name, json.dumps(status))
with record.open("a") as fh:
    fh.write(json.dumps({"engine": name, "start": started, "end": time.time(), "extra": extra,
                         "options": spec["options"], "jobs": spec["jobs"]}) + "\n")
print(f"[fake {name}] done with code {code}")
sys.exit(code)
'''

PROBE = {"fps": 30.0, "num_frames": 12, "width": 64, "height": 48, "duration": 0.4}


class FakeEngines:
    """Patch hmb.batch to launch the fake engine and expose what each engine received."""

    def __init__(self, tmp_path: Path, monkeypatch):
        self.dir = tmp_path / "fake_engines"
        self.dir.mkdir(exist_ok=True)
        self.script = self.dir / "fake_engine.py"
        self.script.write_text(FAKE_ENGINE)
        self.record = self.dir / "record.jsonl"
        self.monkeypatch = monkeypatch
        self.commands = []
        monkeypatch.setenv("FAKE_ENGINE_RECORD", str(self.record))
        for key in ("FAKE_ENGINE_FAIL", "FAKE_ENGINE_CRASH", "FAKE_ENGINE_PIDS", "FAKE_ENGINE_CHILD"):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setattr(hvideo, "probe", lambda video: dict(PROBE))

        def command(name, jobs_file, *, python, force=False, summary=None, extra_args=()):
            cmd = [sys.executable, str(self.script), name, str(jobs_file), *extra_args]
            if force:
                cmd.append("--force")
            self.commands.append(cmd)
            return cmd

        monkeypatch.setattr(batch, "engine_command", command)

    def set(self, key: str, value: str) -> None:
        self.monkeypatch.setenv(key, value)

    def runs(self) -> list:
        if not self.record.is_file():
            return []
        return [json.loads(line) for line in self.record.read_text().splitlines() if line.strip()]

    def engines(self) -> list:
        return [entry["engine"] for entry in self.runs()]

    def last(self, engine: str) -> dict:
        return [entry for entry in self.runs() if entry["engine"] == engine][-1]

    def clear(self) -> None:
        self.record.unlink(missing_ok=True)


@pytest.fixture
def engines(tmp_path, monkeypatch):
    return FakeEngines(tmp_path, monkeypatch)


def make_video(tmp_path: Path, name: str = "My Clip.mp4") -> Path:
    source = tmp_path / "source"
    source.mkdir(exist_ok=True)
    video = source / name
    video.write_bytes(b"fake video bytes")
    return video


def all_outputs(session: Path, stem: str) -> list:
    return [path for paths in batch.session_outputs(session, stem).values() for path in paths]


def test_run_session_writes_the_cli_layout(tmp_path, engines):
    video = make_video(tmp_path)
    session = tmp_path / "session"
    result = batch.run_session(video, session, stem="My_Clip", poll_seconds=0.02)

    assert result["ok"], result["errors"]
    session = session.resolve()
    link = session / "input" / "My_Clip.mp4"
    assert link.is_symlink() and link.resolve() == video.resolve()
    assert result["video"] == str(link)
    plan = hvideo.FramePlan.load(session / "input" / "frames.json")
    assert plan.num_frames == 12 and plan.sample_fps is None and plan.max_side is None
    assert all(path.is_file() for path in all_outputs(session, "My_Clip"))
    assert (session / "baseline" / "My_Clip_keypoints.npz").is_file()
    assert (session / "bone_fit" / "My_Clip_bone_length.npz").is_file()
    assert (session / "gvhmr" / "My_Clip" / "hmr4d_results.pt").is_file()
    assert (session / "cache" / "multi_limbs" / "My_Clip" / "hadm_results" / "output.json").is_file()
    for name in ("detect", "mesh", "hadm", "fit"):
        entry = result["stages"][name]
        assert entry["status"] == "ok" and entry["seconds"] >= 0 and entry["returncode"] == 0
        assert entry["engine_status"]["ok"] is True
        jobs = json.loads((session / "cache" / "engines" / f"{name}.json").read_text())
        assert jobs["jobs"] == [{"session_dir": str(session), "stem": "My_Clip", "video": str(link)}]
        assert (session / "cache" / "engines" / f"{name}.log").is_file()
    assert sorted(engines.engines()) == ["detect", "fit", "hadm", "mesh"]
    # No batch-only files in a CLI session.
    assert not (session / "meta.json").exists()


def test_detect_mesh_hadm_overlap_and_fit_waits_for_detection(tmp_path, engines):
    engines.set("FAKE_ENGINE_DELAY_DETECT", "0.8")
    result = batch.run_session(make_video(tmp_path), tmp_path / "session", poll_seconds=0.02)
    assert result["ok"]
    detect, mesh, hadm, fit = (engines.last(name) for name in ("detect", "mesh", "hadm", "fit"))
    assert mesh["start"] < detect["end"] and hadm["start"] < detect["end"]
    assert fit["start"] >= detect["end"]


def test_existing_outputs_are_reused_and_only_missing_work_runs(tmp_path, engines):
    video, session = make_video(tmp_path), tmp_path / "session"
    batch.run_session(video, session, stem="clip", poll_seconds=0.02)
    engines.clear()
    again = batch.run_session(video, session, stem="clip", poll_seconds=0.02)
    assert again["ok"] and engines.runs() == []
    assert {again["stages"][name]["status"] for name in ("detect", "mesh", "hadm", "fit")} == {"cached"}

    (session / "bone_fit" / "clip_bone_length.npz").unlink()
    result = batch.run_session(video, session, stem="clip", poll_seconds=0.02)
    assert result["ok"] and engines.engines() == ["fit"]
    assert engines.last("fit")["options"]["modes"] == ["variable"]

    # New keypoints make both fits stale.
    engines.clear()
    (session / "baseline" / "clip_keypoints.npz").unlink()
    result = batch.run_session(video, session, stem="clip", stages=("plan", "detect", "fit"), poll_seconds=0.02)
    assert result["ok"] and sorted(engines.engines()) == ["detect", "fit"]
    assert engines.last("fit")["options"]["modes"] == ["rigid", "variable"]
    assert "mesh" not in result["stages"] and "hadm" not in result["stages"]


def test_force_reruns_every_requested_engine(tmp_path, engines):
    video, session = make_video(tmp_path), tmp_path / "session"
    batch.run_session(video, session, stem="clip", poll_seconds=0.02)
    engines.clear()
    result = batch.run_session(video, session, stem="clip", stages=("plan", "mesh", "hadm"), force=True,
                               poll_seconds=0.02)
    assert result["ok"] and sorted(engines.engines()) == ["hadm", "mesh"]
    assert all(entry["extra"][-1] == "--force" for entry in engines.runs())


def test_option_mapping_reaches_each_engine(tmp_path, engines):
    calibration = tmp_path / "camera.json"
    calibration.write_text("{}")
    result = batch.run_session(
        make_video(tmp_path), tmp_path / "session", stem="clip", sample_fps=10.0, max_side=32, person="track",
        fit_options={"max_iters": 5, "camera_calibration": str(calibration)},
        engine_options={"hadm": {"precision": "fp32"}, "fit": {"steps_per_call": 7, "max_iters": 99},
                        "detect": {"camera_calibration": str(calibration)}, "mesh": {"camera": "static"}},
        extra_args={"mesh": ["--gvhmr-root", "/opt/gvhmr"]}, poll_seconds=0.02,
    )
    assert result["ok"]
    fit = engines.last("fit")["options"]
    assert fit["max_iters"] == 5  # fit_options are applied after engine_options["fit"]
    assert fit["steps_per_call"] == 7 and fit["batch"] == 1 and fit["camera_calibration"] == str(calibration)
    assert fit["modes"] == ["rigid", "variable"] and fit["unconstrained_joints"] is True
    detect = engines.last("detect")["options"]
    assert detect == {"person": "track", "batch_size": 8, "camera_calibration": str(calibration)}
    assert engines.last("hadm")["options"] == {"precision": "fp32"}
    mesh = engines.last("mesh")
    assert mesh["options"] == {"camera": "static"} and mesh["extra"] == ["--gvhmr-root", "/opt/gvhmr"]
    assert all("--gvhmr-root" not in entry["extra"] for entry in engines.runs() if entry["engine"] != "mesh")
    plan = hvideo.FramePlan.load(tmp_path / "session" / "input" / "frames.json")
    assert plan.sample_fps == 10.0 and plan.max_side == 32 and plan.indices == [0, 3, 6, 9, 11]
    assert result["plan"]["frames"] == 5 and result["plan"]["source_frames"] == 12


def test_one_failing_engine_does_not_stop_the_others(tmp_path, engines):
    engines.set("FAKE_ENGINE_FAIL", "mesh")
    session = tmp_path / "session"
    result = batch.run_session(make_video(tmp_path), session, stem="clip", poll_seconds=0.02)
    assert not result["ok"]
    assert result["errors"] == {"mesh": "mesh exploded"}
    assert {name: result["stages"][name]["status"] for name in ("detect", "mesh", "hadm", "fit")} == {
        "detect": "ok", "mesh": "error", "hadm": "ok", "fit": "ok"}
    assert (session / "baseline" / "clip_opensim_skeleton.mot").is_file()
    assert not (session / "gvhmr" / "clip" / "hmr4d_results.pt").exists()
    assert "[fake mesh] done with code 1" in result["stages"]["mesh"]["log_tail"]


def test_failed_detection_skips_the_fit_but_not_mesh_or_hadm(tmp_path, engines):
    engines.set("FAKE_ENGINE_FAIL", "detect")
    result = batch.run_session(make_video(tmp_path), tmp_path / "session", stem="clip", poll_seconds=0.02)
    assert not result["ok"]
    assert result["stages"]["fit"]["status"] == "skipped" and "detection failed" in result["errors"]["fit"]
    assert result["stages"]["mesh"]["status"] == "ok" and result["stages"]["hadm"]["status"] == "ok"
    assert "fit" not in engines.engines()


def test_engine_crash_without_status_is_reported_from_its_log(tmp_path, engines):
    session = tmp_path / "session"
    batch.run_session(make_video(tmp_path), session, stem="clip", poll_seconds=0.02)
    (session / "cache" / "multi_limbs" / "clip" / "hadm_results" / "output.json").unlink()
    engines.set("FAKE_ENGINE_CRASH", "hadm")
    result = batch.run_session(make_video(tmp_path), session, stem="clip", poll_seconds=0.02)
    entry = result["stages"]["hadm"]
    # The previous run's (ok) status file must not be mistaken for this run's.
    assert entry["status"] == "error" and entry["engine_status"] is None
    assert entry["error"].startswith("exit code 3; missing ") and "output.json" in entry["error"]
    assert "simulated crash" in entry["log_tail"]


def test_changed_sampling_replans_and_rebuilds_every_product(tmp_path, engines):
    video, session = make_video(tmp_path), tmp_path / "session"
    batch.run_session(video, session, stem="clip", poll_seconds=0.02)
    with pytest.raises(ValueError, match="differs from the requested sampling"):
        batch.plan_session_dir(video, session, "clip", 10.0, None)
    engines.clear()
    result = batch.run_session(video, session, stem="clip", sample_fps=10.0, poll_seconds=0.02)
    assert result["ok"] and result["plan"]["replanned"] and result["plan"]["frames"] == 5
    assert sorted(engines.engines()) == ["detect", "fit", "hadm", "mesh"]
    with pytest.raises(ValueError, match="differs"):
        batch.run_session(video, session, stem="clip", replan=False, poll_seconds=0.02)


def test_sampling_that_selects_the_same_frames_keeps_the_plan_and_products(tmp_path, engines):
    video, session = make_video(tmp_path), tmp_path / "session"
    batch.run_session(video, session, stem="clip", poll_seconds=0.02)
    plan_file = session.resolve() / "input" / "frames.json"
    before = plan_file.read_text()
    engines.clear()
    # 30 fps, 64x48 source: a rate >= 30 Hz selects every frame, a max side >= 64 keeps the frame size.
    for sample_fps, max_side in ((30.0, None), (60.0, 64), (None, 1280)):
        result = batch.run_session(video, session, stem="clip", sample_fps=sample_fps, max_side=max_side,
                                   replan=False, poll_seconds=0.02)
        assert result["ok"] and not result["plan"]["replanned"] and result["plan"]["frames"] == 12
    assert engines.runs() == [] and plan_file.read_text() == before
    # A smaller frame size is a different plan.
    result = batch.run_session(video, session, stem="clip", max_side=32, poll_seconds=0.02)
    assert result["plan"]["replanned"] and sorted(engines.engines()) == ["detect", "fit", "hadm", "mesh"]
    assert hvideo.FramePlan.load(plan_file).output_size == (32, 24)


def test_session_input_must_belong_to_the_same_video(tmp_path, engines):
    video, session = make_video(tmp_path), tmp_path / "session"
    batch.run_session(video, session, stem="clip", stages=("plan",), poll_seconds=0.02)
    other = make_video(tmp_path, "other.mp4")
    with pytest.raises(ValueError, match="use --force"):
        batch.run_session(other, session, stem="clip", stages=("plan",))
    with pytest.raises(ValueError, match="unknown session stage"):
        batch.run_session(video, session, stem="clip", stages=("plan", "score_v2"))


def alive(pid: int) -> bool:
    """The process exists and is not a zombie (an orphan's zombie is reaped by init shortly)."""
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except (OSError, IndexError):
        return False
    return state not in ("Z", "X")


def wait_until_gone(pids, timeout: float = 15.0) -> list:
    """The ``pids`` still alive after ``timeout`` seconds (empty when all are gone)."""
    deadline = time.monotonic() + timeout
    while any(alive(pid) for pid in pids) and time.monotonic() < deadline:
        time.sleep(0.05)
    return [pid for pid in pids if alive(pid)]


def recorded_pids(path: Path) -> dict:
    """``{"mesh": pid, "mesh-child": pid, ...}`` written by the fake engines (FAKE_ENGINE_PIDS)."""
    if not path.is_file():
        return {}
    fields = (line.split() for line in path.read_text().splitlines())
    return {f[0]: int(f[1]) for f in fields if len(f) == 2 and f[1].isdigit()}


def test_interrupt_terminates_running_engines(tmp_path, engines, monkeypatch):
    pids = tmp_path / "pids.txt"
    engines.set("FAKE_ENGINE_PIDS", str(pids))
    engines.set("FAKE_ENGINE_CHILD", "1")
    engines.set("FAKE_ENGINE_DELAY_MESH", "60")
    real_sleep = time.sleep
    interrupted = []

    def interrupt(seconds):
        # Ctrl-C once the mesh engine runs; later sleeps (Popen.wait in terminate) are real.
        if not interrupted and "mesh-child" in recorded_pids(pids):
            interrupted.append(True)
            raise KeyboardInterrupt
        real_sleep(min(seconds, 0.05))

    monkeypatch.setattr(batch.time, "sleep", interrupt)
    started = time.time()
    with pytest.raises(KeyboardInterrupt):
        batch.run_session(make_video(tmp_path), tmp_path / "session", stem="clip", stages=("plan", "mesh"))
    assert time.time() - started < 30
    recorded = recorded_pids(pids)
    with pytest.raises(ProcessLookupError):
        os.kill(recorded["mesh"], 0)
    # the engine's own child is in the engine's process group and is stopped with it
    assert wait_until_gone([recorded["mesh-child"]]) == []


def test_engine_command_matches_the_batch_convention():
    assert batch.engine_command("fit", Path("jobs/fit.json"), python="py", force=True,
                                summary=Path("jobs/fit_summary.json")) == [
        "py", "-m", "hmb.fit_engine", "--jobs", "jobs/fit.json", "--force"]
    assert batch.engine_command("mesh", Path("j.json"), python="py", summary=Path("s.json"),
                                extra_args=["--gvhmr-root", "g"]) == [
        "py", "-m", "hmb.mesh_engine", "--jobs", "j.json", "--summary", "s.json", "--gvhmr-root", "g"]


def test_batch_run_keeps_its_outputs_layout(tmp_path, engines):
    video = make_video(tmp_path, "walk.mp4")
    root = (tmp_path / "root").resolve()
    args = batch.build_parser().parse_args(
        ["--video", str(video), "--root", str(root), "--stages", "plan", "detect", "mesh", "hadm", "fit",
         "--fit-iters", "3", "--person", "first", "--paper-scores"])
    assert batch.run(args) == 0
    session = root / "sessions" / "walk"
    assert json.loads((session / "meta.json").read_text())["frames"] == 10  # 24 Hz default plan of 30 fps
    assert all(path.is_file() for path in all_outputs(session, "walk"))
    for name in ("detect", "mesh", "hadm", "fit"):
        spec = json.loads((root / "jobs" / f"{name}.json").read_text())
        assert spec["jobs"] == [{"session_dir": str(session), "stem": "walk", "video": str(session / "input/walk.mp4")}]
        assert (root / "logs" / f"{name}.log").is_file()
    assert json.loads((root / "jobs" / "fit.json").read_text())["options"]["max_iters"] == 3
    assert json.loads((root / "jobs" / "fit.json").read_text())["options"]["batch"] == 8
    assert json.loads((root / "jobs" / "fit.json").read_text())["options"]["export_grid_hz"] == 24.0
    assert json.loads((root / "jobs" / "fit.json").read_text())["options"]["modes"] == ["rigid", "variable"]
    timings = json.loads((root / "timings.json").read_text())
    assert set(timings["stages"]) == {"plan", "detect", "mesh", "hadm", "fit"}
    assert timings["engine_returncodes"] == {"detect": 0, "mesh": 0, "hadm": 0, "fit": 0}
    assert json.loads((root / "plan_failures.json").read_text()) == []


def test_batch_run_reports_engine_failure(tmp_path, engines):
    engines.set("FAKE_ENGINE_FAIL", "hadm")
    args = batch.build_parser().parse_args(
        ["--video", str(make_video(tmp_path, "walk.mp4")), "--root", str(tmp_path / "root"),
         "--stages", "plan", "detect", "hadm", "fit"])
    assert batch.run(args) == 1
    timings = json.loads((tmp_path / "root" / "timings.json").read_text())
    assert timings["engine_returncodes"] == {"detect": 0, "hadm": 1, "fit": 0}


def test_batch_defaults_fit_only_the_rigid_model_and_score_every_session():
    args = batch.build_parser().parse_args(["--video", "clip.mp4", "--root", "r"])
    assert args.stages == ["plan", "detect", "mesh", "hadm", "fit", "score"] and not args.paper_scores
    assert batch.engine_options(args)["fit"]["modes"] == ["rigid"]
    args = batch.build_parser().parse_args(["--video", "clip.mp4", "--root", "r", "--paper-scores"])
    assert batch.engine_options(args)["fit"]["modes"] == ["rigid", "variable"]


def test_score_tables_copy_the_manifest_keys(tmp_path):
    metrics = batch.METRICS
    rows = [{"session_dir": "/s/a", "scores": {m: 80.0 for m in metrics}, "status": {m: "ok" for m in metrics},
             "tiers": {"anatomy": 80.0, "kinematic": 80.0, "kinetic": 80.0}, "overall": 80.0},
            {"session_dir": "/s/b", "error": "RuntimeError: no session"}]
    metas = {"/s/a": {"id": "a", "video": "/v/a.mp4", "label": 1, "group": "x", "frames": 10, "session_dir": "/s/a"},
             "/s/b": {"id": "b", "label": 0}}
    path = tmp_path / "scores.csv"
    batch.write_table(path, rows, metas)
    import csv

    table = list(csv.DictReader(path.open()))
    assert [r["id"] for r in table] == ["a", "b"] and table[0]["label"] == "1" and table[0]["group"] == "x"
    assert "video" not in table[0] and table[0]["overall"] == "80.0" and table[0]["bone_length_status"] == "ok"
    assert table[1]["error"] == "RuntimeError: no session" and table[1]["overall"] == ""
