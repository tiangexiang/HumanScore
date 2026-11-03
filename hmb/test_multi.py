"""hmb.multi.score_people and scripts/humanscore_multi.py on CPU.

A synthetic clip with three flat-coloured "people" is tracked by a fake YOLO
backend (the real ``track_people`` caching, cleaning, stitching and preview run
around it) and scored by a fake ``score_video``, except in the integration test,
which runs the real ``hmb.pipeline.score_video`` on the fake engines of
``hmb.test_pipeline``.
"""
import importlib.util
import json
import signal
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from hmb import multi, pipeline  # noqa: E402
from hmb.report import is_complete  # noqa: E402
from hmb.scoring.score_session import METRICS, TIERS  # noqa: E402
from hmb.tracking import SAM3SetupError, yolo_backend  # noqa: E402
from hmb.tracking.render import FILL_GRAY, VideoWriter, iter_video  # noqa: E402
from hmb.tracking.tracks import TrackRecorder  # noqa: E402

H, W, N, FPS = 96, 128, 24, 12.0           # 2 s clip
BG = (40, 160, 40)
COLORS = {1: (200, 60, 60), 2: (60, 60, 200), 3: (60, 200, 200)}


def box(oid, f):
    if oid == 1:
        return (8 + f, 4, 48 + f, 92)        # walks to the right (largest: ranked first)
    if oid == 2:
        return (80, 6, 120, 88)              # stands still
    return (50, 10, 70, 90)                  # person 3: only frames 0-5 (0.5 s)


def visible(oid, f):
    return oid != 3 or f < 6


def make_clip(tmp_path, name="Group Clip.mp4"):
    """The clip and its exact tracks (what the fake tracker returns)."""
    path = tmp_path / "videos" / name
    rec = TrackRecorder(W, H)
    with VideoWriter(path, FPS, W, H, crf=0) as writer:
        for f in range(N):
            frame = np.empty((H, W, 3), np.uint8)
            frame[:] = BG
            for oid in (1, 2, 3):
                if not visible(oid, f):
                    continue
                x0, y0, x1, y1 = box(oid, f)
                frame[y0:y1, x0:x1] = COLORS[oid]
                mask = np.zeros((H, W), bool)
                mask[y0:y1, x0:x1] = True
                rec.add(f, oid, mask, 0.9)
            writer.write(frame)
    return path, rec.build(video=str(path), fps=FPS, num_frames=N, backend="yolo")


class FakeTracker:
    def __init__(self, monkeypatch, tracks):
        self.tracks, self.calls = tracks, 0
        monkeypatch.setattr(yolo_backend, "track_yolo", self)

    def __call__(self, video, *, weights=None, device=None, quiet=False):
        import sys

        self.calls += 1
        print("[yolo] done: fake tracker", file=sys.stderr, flush=True)
        print("ultralytics chatter", file=sys.stderr, flush=True)
        return self.tracks.subset(np.arange(len(self.tracks)))


class FakeScorer:
    """Stands in for hmb.pipeline.score_video; person ``fail`` raises."""

    def __init__(self, monkeypatch, fail=()):
        self.calls, self.fail = [], set(fail)
        monkeypatch.setattr(multi, "score_video", self)

    def __call__(self, video, out_dir, *, person_boxes=None, force=False, quiet=False, static_cam=True,
                 sample_fps=24.0, max_side=1280, paper_scores=False):
        pid = int(Path(video).stem.split("_")[-1])
        self.calls.append({"pid": pid, "video": Path(video), "out": Path(out_dir), "boxes": Path(person_boxes),
                           "force": force, "quiet": quiet, "static_cam": static_cam, "sample_fps": sample_fps,
                           "max_side": max_side, "paper_scores": paper_scores})
        if pid in self.fail:
            raise RuntimeError("no scores were produced\nsecond line")
        scores = {m: 50.0 + pid + i for i, m in enumerate(METRICS)}
        result = {"video": str(video), "scores": scores,
                  "tiers": {t: (scores[a] + scores[b]) / 2 for t, (a, b) in TIERS.items()},
                  "overall": sum(scores.values()) / 6, "status": {m: "ok" for m in METRICS}, "reasons": {},
                  "person": {"selection": "boxes", "frames_analysed": N, "frames_found": N - 1},
                  "files": {"session": str(out_dir)}, "timings": {"total_seconds": 1.0}}
        if paper_scores:
            result["paper_scores"] = {"scores": {m: 40.0 for m in METRICS}, "tiers": {t: 40.0 for t in TIERS},
                                      "overall": 40.0, "status": {m: "ok" for m in METRICS}, "reasons": {}}
        (Path(out_dir) / pipeline.RESULT_NAME).write_text(json.dumps(result))
        return result


def run(video, out, **kwargs):
    kwargs.setdefault("tracker", "yolo")
    return multi.score_people(video, out, **kwargs)


def test_people_are_scored_one_by_one_short_tracks_skipped_and_failures_reported(tmp_path, monkeypatch):
    video, tracks = make_clip(tmp_path)
    tracker, scorer = FakeTracker(monkeypatch, tracks), FakeScorer(monkeypatch, fail={2})
    out = tmp_path / "out"
    result = run(video, out, sample_fps=30.0, quiet=False)

    # person 1 (largest area, moves) ranks before person 2; person 3 is visible 0.5 s < 1 s
    assert list(result["people"]) == ["1", "2"] and [c["pid"] for c in scorer.calls] == [1, 2]
    assert [s["id"] for s in result["skipped"]] == [3] and "too short" in result["skipped"][0]["reason"]
    assert result["skipped"][0]["frames_visible"] == 6
    assert (result["num_tracked"], result["num_selected"], result["num_scored"]) == (3, 2, 1)
    assert result["complete"] is False and result["message"] is None

    p1, p2 = result["people"]["1"], result["people"]["2"]
    assert p1["rank"] == 1 and p1["frames_visible"] == N and p1["duration_seconds"] == pytest.approx(2.0)
    assert p1["first_frame"] == 0 and p1["last_frame"] == N - 1 and p1["time_span_seconds"] == [0.0, 2.0]
    assert p1["scores"]["extra_limbs"] == 51.0 and p1["overall"] is not None and is_complete(p1)
    assert p1["status"] == {m: "ok" for m in METRICS} and p1["error"] is None and "paper_scores" not in p1
    assert p1["person"] == {"selection": "boxes", "frames_analysed": N, "frames_found": N - 1}
    assert p2["error"] == "RuntimeError: no scores were produced" and not is_complete(p2)
    assert p2["scores"] == {m: None for m in METRICS} and set(p2["status"].values()) == {"error"}
    assert p2["reasons"]["bone_length"] == p2["error"]
    assert result["settings"] == {"static_cam": True, "sample_fps": 30.0, "max_side": 1280, "paper_scores": False,
                                  "tracker": "yolo"}

    # each person: own directory = session, full-frame masked video and its box track
    call = scorer.calls[0]
    pdir = out / "people" / "person_1"
    assert call["out"] == pdir.resolve() and call["video"] == pdir.resolve() / "person_1.mp4"
    assert call["boxes"] == pdir.resolve() / "person_boxes.json" and call["quiet"] is False
    assert (call["sample_fps"], call["max_side"], call["paper_scores"], call["static_cam"]) == (30.0, 1280, False,
                                                                                                True)
    assert p1["paths"]["humanscore"] == str(pdir.resolve() / "humanscore.json")
    frames = [frame for _, frame in iter_video(call["video"])]
    assert len(frames) == N and frames[0].shape == (H, W, 3)
    x0, y0, x1, y1 = box(2, 3)
    assert np.abs(frames[3][(y0 + y1) // 2, (x0 + x1) // 2].astype(int) - FILL_GRAY).max() <= 6   # other: gray
    x0, y0, x1, y1 = box(1, 3)
    assert np.abs(frames[3][(y0 + y1) // 2, (x0 + x1) // 2].astype(int) - COLORS[1]).max() <= 6   # target kept
    x0, y0, x1, y1 = box(3, 2)
    assert np.abs(frames[2][(y0 + y1) // 2, (x0 + x1) // 2].astype(int) - FILL_GRAY).max() <= 6   # skipped too
    boxes = json.loads(call["boxes"].read_text())
    assert len(boxes["timestamps"]) == len(boxes["boxes_xyxy"]) == N
    assert boxes["boxes_xyxy"][3] == list(map(float, box(1, 3))) and boxes["timestamps"][3] == pytest.approx(0.25)

    # outputs
    on_disk = json.loads((out / "humanscore_multi.json").read_text())
    assert on_disk == json.loads(json.dumps(result))
    for name in ("tracks.json", "masks.npz", "preview.mp4", "preview_sheet.jpg", "selection.json"):
        assert (out / "tracks" / name).is_file(), name
    selection = json.loads((out / "tracks" / "selection.json").read_text())
    assert [s["id"] for s in selection["selected"]] == [1, 2] and [s["id"] for s in selection["skipped"]] == [3]
    assert selection["skipped"][0]["reason"].startswith("too short")
    summary = (out / "summary.txt").read_text()
    assert "P1" in summary and "failed: RuntimeError" in summary and "P3: too short" in summary
    assert tracker.calls == 1


def test_rerun_reuses_tracks_and_person_videos_and_rebuilds_them_when_tracks_change(tmp_path, monkeypatch):
    video, tracks = make_clip(tmp_path)
    tracker, scorer = FakeTracker(monkeypatch, tracks), FakeScorer(monkeypatch)
    out = tmp_path / "out"
    run(video, out, quiet=False)
    pvideo = out / "people" / "person_1" / "person_1.mp4"
    stamp = pvideo.stat().st_mtime_ns
    assert [c["force"] for c in scorer.calls] == [False, False]

    scorer.calls.clear()
    result = run(video, out, quiet=False)
    assert tracker.calls == 1 and pvideo.stat().st_mtime_ns == stamp           # tracks and videos reused
    assert [c["force"] for c in scorer.calls] == [False, False] and result["complete"]

    # new post-processing re-saves the tracks (no re-tracking): the person videos are rebuilt, and a
    # session prepared from the older video is rebuilt too (it would refuse the changed input otherwise)
    (out / "people" / "person_1" / "manifest.json").write_text("{}")
    scorer.calls.clear()
    run(video, out, quiet=False, tracker_options={"stitch_gap_seconds": 0.5})
    assert tracker.calls == 1 and pvideo.stat().st_mtime_ns != stamp
    assert [(c["pid"], c["force"]) for c in scorer.calls] == [(1, True), (2, False)]

    scorer.calls.clear()
    run(video, out, quiet=False, force=True)
    assert tracker.calls == 2 and [c["force"] for c in scorer.calls] == [True, True]

    # stricter rules: person 2 is not scored any more, its stale directory is left alone and not reported
    scorer.calls.clear()
    result = run(video, out, quiet=False, max_people=1)
    assert list(result["people"]) == ["1"] and [c["pid"] for c in scorer.calls] == [1]
    assert {s["id"]: s["reason"] for s in result["skipped"]}[2].startswith("less prominent")


def test_no_person_found(tmp_path, monkeypatch):
    video, tracks = make_clip(tmp_path)
    FakeTracker(monkeypatch, tracks.subset([]))
    scorer = FakeScorer(monkeypatch)
    result = run(video, tmp_path / "out", quiet=False)
    assert result["people"] == {} and result["skipped"] == [] and scorer.calls == []
    assert result["num_tracked"] == 0 and result["complete"] is False
    assert result["message"] == "no person found in the video"
    assert "no person found" in (tmp_path / "out" / "summary.txt").read_text()


def test_quiet_run_shows_tracker_progress_and_logs_the_rest(tmp_path, monkeypatch, capfd):
    video, tracks = make_clip(tmp_path)
    FakeTracker(monkeypatch, tracks)
    FakeScorer(monkeypatch)
    out = tmp_path / "out"
    run(video, out, quiet=True)
    console = capfd.readouterr()
    assert "yolo: done: fake tracker" in console.out and "ultralytics chatter" not in console.out + console.err
    log = (out / "logs" / "tracking.log").read_text()
    assert "ultralytics chatter" in log and "[tracking] yolo: 3 people" in log
    assert "person 1 (1 of 2)" in console.out and "person   3: frames 0-5" in console.out


def test_invalid_inputs_fail_before_tracking(tmp_path, monkeypatch):
    video, tracks = make_clip(tmp_path)
    tracker = FakeTracker(monkeypatch, tracks)
    out = tmp_path / "out"
    with pytest.raises(FileNotFoundError):
        run(tmp_path / "missing.mp4", out)
    with pytest.raises(ValueError, match="tracking backend"):
        run(video, out, tracker="openpose")
    with pytest.raises(ValueError, match="min_visible"):
        run(video, out, min_visible=1.5)
    with pytest.raises(ValueError, match="max_people"):
        run(video, out, max_people=-1)
    from hmb import cli

    monkeypatch.setattr(cli, "module_available", lambda name: False)
    with pytest.raises(ValueError, match="pycolmap") as moving:
        run(video, out, static_cam=False)
    assert "--moving-cam" in str(moving.value) and "--static-cam" not in str(moving.value)
    assert tracker.calls == 0 and not out.exists()


def test_min_visible_default_is_the_tracking_default_everywhere():
    import inspect

    from hmb.tracking.tracks import MIN_VISIBLE, select_people

    script = _load_script()
    assert inspect.signature(multi.score_people).parameters["min_visible"].default == MIN_VISIBLE
    assert inspect.signature(select_people).parameters["min_visible"].default == MIN_VISIBLE
    assert script.build_parser().parse_args(["clip.mp4"]).min_visible == MIN_VISIBLE == 0.3
    text = " ".join(script.build_parser().format_help().split())
    assert "25% of all analysed frames" in text and "unsupported" in text and "Default: 0.3" in text


def test_metric_reasons_reach_the_multi_result_and_summary(tmp_path, monkeypatch):
    video, tracks = make_clip(tmp_path)
    FakeTracker(monkeypatch, tracks)
    scorer = FakeScorer(monkeypatch)
    plain = scorer.__call__

    def bone_length_unsupported(video, out_dir, **kwargs):
        result = plain(video, out_dir, **kwargs)
        if int(Path(video).stem.split("_")[-1]) == 2:
            result["scores"]["bone_length"], result["overall"] = None, None
            result["status"]["bone_length"] = "unsupported"
            result["reasons"] = {"bone_length": "keypoint coverage 0.18 < 0.25 of the analysed frames"}
        return result

    monkeypatch.setattr(multi, "score_video", bone_length_unsupported)
    out = tmp_path / "out"
    result = run(video, out, quiet=False)
    assert result["people"]["1"]["reasons"] == {}
    assert result["people"]["2"]["reasons"] == {"bone_length": "keypoint coverage 0.18 < 0.25 of the analysed frames"}
    summary = (out / "summary.txt").read_text()
    assert "missing scores:" in summary
    assert "P2 bone_length unsupported: keypoint coverage 0.18 < 0.25 of the analysed frames" in summary
    assert "P1 " not in summary.split("missing scores:")[1]


def test_paper_scores_reach_every_person_and_the_table(tmp_path, monkeypatch):
    video, tracks = make_clip(tmp_path)
    FakeTracker(monkeypatch, tracks)
    scorer = FakeScorer(monkeypatch, fail={2})
    out = tmp_path / "out"
    result = run(video, out, quiet=False, paper_scores=True)
    assert [c["paper_scores"] for c in scorer.calls] == [True, True]
    assert result["people"]["1"]["paper_scores"]["overall"] == 40.0 and result["settings"]["paper_scores"]
    assert result["people"]["2"]["paper_scores"]["status"] == {m: "error" for m in METRICS}
    table = (out / "summary.txt").read_text()
    header = next(line for line in table.splitlines() if line.strip().startswith("person"))
    assert header.split()[-3:] == ["overall", "paper", "status"]
    row = next(line for line in table.splitlines() if line.strip().startswith("P1"))
    assert row.split()[-2:] == ["40.0", "ok"]


def _load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "humanscore_multi.py"
    spec = importlib.util.spec_from_file_location("humanscore_multi", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_multi_script_exit_codes_and_default_output_dir(tmp_path, monkeypatch, capfd):
    script = _load_script()
    video, tracks = make_clip(tmp_path)
    tracker = FakeTracker(monkeypatch, tracks)
    scorer = FakeScorer(monkeypatch)
    monkeypatch.chdir(tmp_path)
    assert script.main([str(video), "--tracker", "yolo", "--max-people", "0"]) == 0
    out = (tmp_path / "humanscore_output" / "Group_Clip_multi").resolve()
    printed = capfd.readouterr().out
    assert "every person: Group Clip.mp4" in printed and f"results: {out / 'humanscore_multi.json'}" in printed
    assert [c["quiet"] for c in scorer.calls] == [True, True] and scorer.calls[0]["out"].parent == out / "people"

    scorer.fail = {1}
    assert script.main([str(video), "--tracker", "yolo", "--out", str(tmp_path / "o2")]) == 1
    tracker.tracks = tracks.subset([])
    assert script.main([str(video), "--tracker", "yolo", "--out", str(tmp_path / "o3")]) == 1   # nobody to score
    capfd.readouterr()
    assert script.main([str(tmp_path / "nope.mp4"), "--tracker", "yolo"]) == 2
    assert "error: video not found" in capfd.readouterr().err
    assert script.main([str(video), "--min-visible", "2"]) == 2

    def no_sam3(*args, **kwargs):
        raise SAM3SetupError("The sam3.1 tracker needs SAM 3.1, which was not found")

    monkeypatch.setattr(multi, "check_options", lambda *a, **k: "sam3.1")
    import hmb.tracking.sam3_backend as sam3_backend

    monkeypatch.setattr(sam3_backend, "track_sam3", no_sam3)
    assert script.main([str(video), "--out", str(tmp_path / "o4")]) == 2
    assert "error: The sam3.1 tracker needs SAM 3.1" in capfd.readouterr().err


def test_integration_with_the_real_pipeline_follows_each_persons_boxes(tmp_path, monkeypatch):
    from hmb.test_pipeline import Fakes

    fakes = Fakes(tmp_path, monkeypatch)
    video, tracks = make_clip(tmp_path)
    FakeTracker(monkeypatch, tracks)
    out = tmp_path / "out"
    result = multi.score_people(video, out, tracker="yolo", quiet=False)
    detect = [run["options"] for run in fakes.engines.runs() if run["engine"] == "detect"]
    assert [Path(o["person_boxes"]).parent.name for o in detect] == ["person_1", "person_2"]
    assert all(o["person"] == "boxes" for o in detect)
    mesh = [run["options"] for run in fakes.engines.runs() if run["engine"] == "mesh"]
    assert [o["person_boxes"] for o in mesh] == [o["person_boxes"] for o in detect]   # GVHMR follows them too
    for pid in ("1", "2"):
        person = result["people"][pid]
        saved = json.loads(Path(person["paths"]["humanscore"]).read_text())
        manifest = json.loads((Path(person["paths"]["dir"]) / "manifest.json").read_text())
        assert manifest["config"]["requested_fit_options"]["person"] == "boxes" and person["scores"] == saved["scores"]
        assert person["error"] is None and Path(person["paths"]["summary"]).is_file()
    fakes.engines.clear()
    again = multi.score_people(video, out, tracker="yolo", quiet=False)
    assert fakes.engines.runs() == [] and again["people"]["1"]["scores"] == result["people"]["1"]["scores"]


@pytest.mark.parametrize("signum, status, message", [(signal.SIGTERM, 143, "terminated by SIGTERM"),
                                                     (signal.SIGINT, 130, "interrupted")])
def test_multi_script_reports_an_interrupt_with_the_signal_exit_status(tmp_path, monkeypatch, capfd, signum,
                                                                       status, message):
    # score_people stops the tracker worker and the engines (hmb.process_groups) before this propagates
    from hmb import process_groups

    script = _load_script()
    video, _ = make_clip(tmp_path)

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt if signum == signal.SIGINT else process_groups.Terminated(signum)

    monkeypatch.setattr(script, "score_people", interrupted)
    assert script.main([str(video), "--tracker", "yolo", "--out", str(tmp_path / "o")]) == status
    assert f"{message}; run the same command again to resume" in capfd.readouterr().err
