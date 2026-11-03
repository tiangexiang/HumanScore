"""hmb.detect_engine: batching, frame plans, outputs and person selection (no GPU; fake detector)."""
import json
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from hmb import video as hvideo  # noqa: E402
from hmb.detect_engine import (  # noqa: E402
    DetectEngine, FrameDetections, default_intrinsics, detections_arrays, job_camera, keypoint_arrays,
    load_detections, make_job, reselect_session, select_first, select_track,
)

STEP = 10  # frame k is a uniform image of intensity STEP * k (survives MJPG)


def write_video(path: Path, frames: int = 11, size=(64, 48), fps: float = 24.0) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    assert writer.isOpened()
    for k in range(frames):
        writer.write(np.full((size[1], size[0], 3), STEP * k, np.uint8))
    writer.release()
    return path


class FakeDetector:
    """Stands in for MeTRAbs: one person per frame encoding the frame index, a second
    (smaller) person on frames listed in ``extra``, nobody on frames in ``empty``."""

    def __init__(self, empty=(), extra=(), extra_first=()):
        self.calls = []
        self.empty, self.extra, self.extra_first = set(empty), set(extra), set(extra_first)

    def detect(self, frames, intrinsics, distortion, batch_size, skeleton, kwargs):
        padded = list(frames) + [np.zeros_like(frames[0])] * (batch_size - len(frames))
        self.calls.append({"real": len(frames), "batch": len(padded), "shape": frames.shape[1:],
                           "intrinsics": np.asarray(intrinsics).copy(), "skeleton": skeleton, "kwargs": kwargs})
        boxes, p3, p2 = [], [], []
        for frame in frames:
            k = int(round(float(frame.mean()) / STEP))
            if k in self.empty:
                boxes.append(np.zeros((0, 5), np.float32))
                p3.append(np.zeros((0, 87, 3), np.float32))
                p2.append(np.zeros((0, 87, 2), np.float32))
                continue
            main = np.array([10 + k, 5, 30, 40, 0.9], np.float32)      # large, drifting right
            other = np.array([0, 0, 8, 10, 0.95], np.float32)          # small bystander
            dets = [main]
            if k in self.extra:
                dets = [other, main] if k in self.extra_first else [main, other]
            boxes.append(np.stack(dets))
            p3.append(np.stack([np.full((87, 3), 1000.0 * k + d[0], np.float32) for d in dets]))
            p2.append(np.stack([np.full((87, 2), float(k) + d[0], np.float32) for d in dets]))
        return FrameDetections(boxes, p3, p2), 0


def run(tmp_path, video, detector=None, options=None, force=False, **entry):
    engine = DetectEngine({"warmup": False, **(options or {})}, detector=detector or FakeDetector(), log=lambda m: None)
    engine.load()
    job = {"session_dir": str(tmp_path / "S"), "stem": "clip", "video": str(video), **entry}
    return engine, engine.process(job, force=force)


def test_all_frames_padded_batches_and_baseline_keys(tmp_path):
    video = write_video(tmp_path / "clip.avi", frames=11)
    detector = FakeDetector()
    engine, status = run(tmp_path, video, detector, {"batch_size": 4, "pad_final_batch": True})
    assert status["ok"] and status["frames"] == 11 and status["plan"] == "all_frames"
    # every call has the fixed batch size; the last real batch has 3 frames
    assert [c["batch"] for c in detector.calls] == [4, 4, 4]
    assert [c["real"] for c in detector.calls] == [4, 4, 3]
    assert detector.calls[0]["skeleton"] == "bml_movi_87" and detector.calls[0]["kwargs"] == {}
    np.testing.assert_allclose(detector.calls[0]["intrinsics"], default_intrinsics(64, 48))
    with np.load(tmp_path / "S/baseline/clip_keypoints.npz", allow_pickle=False) as data:
        for key in ("keypoints3d", "keypoints2d", "boxes", "confs", "camera_intrinsics_px", "camera_distortion",
                    "camera_calibration_source", "timestamps", "frame_indices", "fps", "sample_fps",
                    "box_confidence"):
            assert key in data.files
        assert data["keypoints3d"].shape == (11, 87, 3) and data["keypoints3d"].dtype == np.float32
        assert data["boxes"].shape == (11, 5)
        np.testing.assert_array_equal(data["confs"], np.ones((11, 87)))
        # frame k really carries the detection of decoded frame k (alignment through batching)
        np.testing.assert_allclose(data["keypoints3d"][:, 0, 0], 1000.0 * np.arange(11) + 10 + np.arange(11))
        np.testing.assert_array_equal(data["frame_indices"], np.arange(11))
        np.testing.assert_allclose(data["timestamps"], np.arange(11) / 24.0)
        assert float(data["fps"]) == pytest.approx(24.0) and np.isnan(data["sample_fps"])
        np.testing.assert_allclose(data["camera_intrinsics_px"], default_intrinsics(64, 48)[None])
        assert data["camera_intrinsics_px"].shape == (1, 4) and data["camera_distortion"].shape == (1, 5)
        assert str(data["camera_calibration_source"]) == "assumed_fov55"
        np.testing.assert_allclose(data["box_confidence"], 0.9)
    status_file = json.loads((tmp_path / "S/cache/engine_status/detect.json").read_text())
    assert status_file["ok"] is True and status_file["seconds"] >= 0 and status_file["error"] is None


def test_unpadded_final_batch_option(tmp_path):
    video = write_video(tmp_path / "clip.avi", frames=6)
    detector = FakeDetector()
    run(tmp_path, video, detector, {"batch_size": 4})  # default: no padding (exact CLI parity)
    assert [c["batch"] for c in detector.calls] == [4, 2]


def test_missing_detections_match_baseline_semantics(tmp_path):
    video = write_video(tmp_path / "clip.avi", frames=5)
    _, status = run(tmp_path, video, FakeDetector(empty={1, 3}))
    assert status["frames_detected"] == 3
    with np.load(tmp_path / "S/baseline/clip_keypoints.npz") as data:
        assert data["keypoints3d"].dtype == np.float64  # as np.savez of the baseline's mixed list
        np.testing.assert_array_equal(data["confs"][[1, 3]], 0)
        np.testing.assert_array_equal(data["confs"][[0, 2, 4]], 1)
        np.testing.assert_array_equal(data["keypoints3d"][1], 0)
        np.testing.assert_array_equal(data["selected_detection"], [0, -1, 0, -1, 0])


def test_keypoint_arrays_equal_save_metrabs_data(tmp_path):
    from baseline.main import save_metrabs_data

    class T:  # mimics a tf.Tensor row
        def __init__(self, value):
            self.value = value

        def numpy(self):
            return self.value

    rng = np.random.default_rng(0)
    dets = FrameDetections(
        [rng.normal(size=(2, 5)).astype(np.float32), np.zeros((0, 5), np.float32)],
        [rng.normal(size=(2, 87, 3)).astype(np.float32), np.zeros((0, 87, 3), np.float32)],
        [rng.normal(size=(2, 87, 2)).astype(np.float32), np.zeros((0, 87, 2), np.float32)])
    accumulated = {"boxes": [[T(b) for b in dets.boxes[0]], []],
                   "poses3d": [[T(b) for b in dets.poses3d[0]], []],
                   "poses2d": [[T(b) for b in dets.poses2d[0]], []]}
    save_metrabs_data(accumulated, "/x/clip.mp4", str(tmp_path))
    ours = keypoint_arrays(dets, select_first(dets.boxes))
    with np.load(tmp_path / "clip_keypoints.npz") as data:
        for key in ("keypoints3d", "keypoints2d", "boxes", "confs"):
            assert data[key].dtype == ours[key].dtype
            np.testing.assert_array_equal(data[key], ours[key])


def test_frame_plan_is_used_exactly(tmp_path):
    video = write_video(tmp_path / "clip.avi", frames=20, fps=30.0)
    plan = hvideo.plan_frames(video, sample_fps=12.0)
    plan.save(tmp_path / "S/input/frames.json")
    _, status = run(tmp_path, video, FakeDetector(), {"batch_size": 3})
    assert status["ok"] and status["plan"] == "frames.json"
    with np.load(tmp_path / "S/baseline/clip_keypoints.npz") as data:
        np.testing.assert_array_equal(data["frame_indices"], plan.indices)
        np.testing.assert_allclose(data["timestamps"], plan.timestamps)
        np.testing.assert_allclose(data["keypoints3d"][:, 0, 0], 1000.0 * np.array(plan.indices) + 10 + np.array(plan.indices))
        assert float(data["sample_fps"]) == 12.0 and float(data["fps"]) == pytest.approx(30.0)


def test_sample_fps_option_creates_plan_and_conflict_fails_only_that_job(tmp_path):
    video = write_video(tmp_path / "clip.avi", frames=20, fps=30.0)
    _, status = run(tmp_path, video, FakeDetector(), {"sample_fps": 15.0})
    plan = hvideo.FramePlan.load(tmp_path / "S/input/frames.json")
    assert status["ok"] and plan.indices == hvideo.sample_indices(20, 30.0, 15.0).tolist()
    assert status["frames"] == len(plan.indices)

    engine = DetectEngine({"warmup": False}, detector=FakeDetector(), log=lambda m: None).load()
    statuses = engine.run([
        {"session_dir": str(tmp_path / "S"), "stem": "clip", "video": str(video), "sample_fps": 10.0},
        {"session_dir": str(tmp_path / "missing"), "stem": "gone", "video": str(tmp_path / "gone.avi")},
        {"session_dir": str(tmp_path / "T"), "stem": "clip", "video": str(video)},
    ], force=True)
    assert [s["ok"] for s in statuses] == [False, False, True]
    assert "differs from the requested" in statuses[0]["error"]
    assert "gone.avi" in statuses[1]["error"]
    assert json.loads((tmp_path / "missing/cache/engine_status/detect.json").read_text())["ok"] is False


def test_skip_existing_unless_forced(tmp_path):
    video = write_video(tmp_path / "clip.avi", frames=4)
    detector = FakeDetector()
    engine, first = run(tmp_path, video, detector)
    assert first["ok"] and not first["skipped"]
    calls = len(detector.calls)
    again = engine.process({"session_dir": str(tmp_path / "S"), "stem": "clip", "video": str(video)})
    assert again["skipped"] and len(detector.calls) == calls
    forced = engine.process({"session_dir": str(tmp_path / "S"), "stem": "clip", "video": str(video)}, force=True)
    assert not forced["skipped"] and len(detector.calls) > calls


def test_max_side_resize_rescales_camera(tmp_path):
    video = write_video(tmp_path / "clip.avi", frames=3, size=(128, 64))
    plan = hvideo.plan_frames(video, max_side=64)
    plan.save(tmp_path / "S/input/frames.json")
    detector = FakeDetector()
    _, status = run(tmp_path, video, detector)
    assert status["image_size"] == [64, 32] or tuple(status["image_size"]) == (64, 32)
    np.testing.assert_allclose(detector.calls[0]["intrinsics"], default_intrinsics(64, 32))
    with np.load(tmp_path / "S/baseline/clip_keypoints.npz") as data:
        np.testing.assert_array_equal(data["image_size"], [64, 32])
        np.testing.assert_array_equal(data["source_size"], [128, 64])
    calibration = tmp_path / "cam.json"
    calibration.write_text(json.dumps({"fx": 100.0, "fy": 110.0, "cx": 60.0, "cy": 30.0}))
    job = make_job({"session_dir": str(tmp_path / "S"), "stem": "clip", "camera_calibration": str(calibration)})
    intrinsics, distortion, source = job_camera(job, (128, 64), (64, 32))
    np.testing.assert_allclose(intrinsics, [50.0, 55.0, 30.0, 15.0])
    np.testing.assert_array_equal(distortion, np.zeros(5))
    assert source == str(calibration)


def test_explicit_paths_for_other_output_directories(tmp_path):
    video = write_video(tmp_path / "clip.avi", frames=5, fps=30.0)
    out = tmp_path / "anywhere"
    # a plan in the default location must be ignored when plan_path is disabled
    hvideo.plan_frames(video, sample_fps=10.0).save(out / "input/frames.json")
    engine = DetectEngine({"warmup": False, "save_detections": False}, detector=FakeDetector(), log=lambda m: None).load()
    status = engine.process({"session_dir": str(out), "stem": "clip", "video": str(video),
                             "keypoints_path": str(out / "clip_keypoints.npz"), "plan_path": None,
                             "status_path": str(out / "clip_detect_status.json")})
    assert status["ok"] and status["plan"] == "all_frames" and status["frames"] == 5
    assert (out / "clip_keypoints.npz").is_file() and (out / "clip_detect_status.json").is_file()
    assert not (out / "baseline").exists() and not (out / "cache").exists()
    assert not (out / "input/clip.mp4").exists()


def two_people(n=8, swap=(2, 5), gap=(4,)):
    boxes = []
    for t in range(n):
        main = [100 + 3 * t, 50, 80, 200, 0.8]
        bystander = [10, 10, 20, 40, 0.99]
        if t in gap:
            boxes.append(np.zeros((0, 5)))
        elif t in swap:
            boxes.append(np.array([bystander, main], float))
        else:
            boxes.append(np.array([main, bystander], float))
    return boxes


def test_track_follows_the_seeded_person_through_order_swaps_and_gaps():
    boxes = two_people()
    times = np.arange(len(boxes)) / 24.0
    first = select_first(boxes)
    track = select_track(boxes, times, init_seconds=0.5)
    assert first.tolist() == [0, 0, 0, 0, -1, 0, 0, 0]
    assert track.tolist() == [0, 0, 1, 0, -1, 1, 0, 0]


def test_track_seed_is_largest_box_in_init_window_and_tracks_backwards():
    # frame 0 only sees the bystander; the main person (larger) appears at frame 1
    boxes = [np.array([[10, 10, 20, 40, 0.99]], float)] + two_people(6, swap=(), gap=())[1:]
    track = select_track(boxes, np.arange(6) / 24.0, init_seconds=0.5)
    assert track[1:].tolist() == [0] * 5
    # backward from the seed (main person, frame 1): frame 0 only has the distant small bystander,
    # which fails the association gate, so frame 0 is a gap rather than an identity switch
    assert track[0] == -1
    # with a zero-length window the seed is frame 0's only box (the bystander), which is then followed
    assert select_track(boxes, np.arange(6) / 24.0, init_seconds=0.0).tolist() == [0, 1, 1, 1, 1, 1]


def test_track_seed_ignores_a_transient_larger_box_entering_at_the_edge():
    # main subject in every frame; a bystander enters from the left edge and is briefly larger
    main = [567, 68, 255, 659, 0.99]
    boxes = []
    for t in range(14):
        frame = [main]
        if t >= 4:
            frame.append([0, 40, 150 + 15 * t, 680, 0.55])  # grows to 345 x 680 > main
        boxes.append(np.array(frame, float))
    track = select_track(boxes, np.arange(14) / 24.0, init_seconds=0.5)
    assert track.tolist() == [0] * 14


def test_track_nearest_centre_fallback_when_nothing_overlaps():
    # nothing overlaps: the nearest box within the motion gate (1 x sqrt(area)) is taken ...
    boxes = [np.array([[0, 0, 10, 10, 1.0]]), np.array([[100, 100, 10, 10, 1.0], [10, 0, 10, 10, 1.0]])]
    assert select_track(boxes, [0.0, 1.0], init_seconds=0.0).tolist() == [0, 1]
    # ... but a box beyond the gate is not the same person: gap
    far = [np.array([[0, 0, 10, 10, 1.0]]), np.array([[100, 100, 10, 10, 1.0], [30, 0, 10, 10, 1.0]])]
    assert select_track(far, [0.0, 1.0], init_seconds=0.0).tolist() == [0, -1]
    assert select_track([np.zeros((0, 5))] * 3, [0, 1, 2]).tolist() == [-1, -1, -1]


def test_track_mode_and_cached_detections_reselection(tmp_path):
    video = write_video(tmp_path / "clip.avi", frames=6)
    _, status = run(tmp_path, video, FakeDetector(extra={1, 2, 3}, extra_first={2}), {"person": "first"})
    assert status["frames_multi_detection"] == 3 and status["first_vs_track_disagreements"] == 1
    session = tmp_path / "S"
    dets, extra = load_detections(session / "cache/detect/clip_detections.npz")
    assert [len(b) for b in dets.boxes] == [1, 2, 2, 2, 1, 1]
    np.testing.assert_array_equal(extra["frame_indices"], np.arange(6))
    with np.load(session / "baseline/clip_keypoints.npz") as data:
        assert data["keypoints3d"][2, 0, 0] == pytest.approx(2000.0)  # 'first' picked the bystander at frame 2
    reselect_session(session, "clip", person="track")
    with np.load(session / "baseline/clip_keypoints.npz") as data:
        assert data["keypoints3d"][2, 0, 0] == pytest.approx(2000.0 + 12)
        assert str(data["person_selection"]) == "track"
        assert "timestamps" in data.files and data["selected_detection"].tolist() == [0, 0, 1, 0, 0, 0]
    roundtrip = detections_arrays(dets)
    assert roundtrip["boxes"].shape == (9, 5) and roundtrip["poses3d"].shape == (9, 87, 3)


def test_invalid_options_rejected():
    with pytest.raises(ValueError):
        DetectEngine({"person": "largest"})
    with pytest.raises(ValueError):
        DetectEngine({"graph_optimization": "aggressive"})
    with pytest.raises(ValueError):
        make_job({"session_dir": "/tmp/x", "stem": "a", "batch_size": 0})


def test_graph_optimization_mode_validation():
    from hmb.metrabs_model import configure_graph_optimization
    with pytest.raises(ValueError):
        configure_graph_optimization("aggressive")
    assert configure_graph_optimization("full") == "full"


@pytest.mark.skipif(__import__("os").environ.get("HMB_DETECT_ENGINE_GPU_TEST") != "1",
                    reason="loads MeTRAbs on the GPU (~1-2 min); set HMB_DETECT_ENGINE_GPU_TEST=1")
def test_real_metrabs_engine_on_a_short_clip(tmp_path):
    source = Path(__import__("os").environ.get("HMB_TEST_VIDEO", "missing-test-clip.mp4"))
    if not source.is_file():
        pytest.skip("test clip unavailable (set HMB_TEST_VIDEO)")
    plan = hvideo.plan_frames(source, sample_fps=12.0)
    plan.indices, plan.timestamps = plan.indices[:11], plan.timestamps[:11]
    plan.save(tmp_path / "S/input/frames.json")
    engine = DetectEngine({"batch_size": 8, "graph_optimization": "lean"}).load()
    status = engine.process({"session_dir": str(tmp_path / "S"), "stem": "clip", "video": str(source)})
    assert status["ok"], status.get("error")
    assert status["frames"] == 11 and status["padding_detections"] == 0 and status["batches"] == 2
    with np.load(tmp_path / "S/baseline/clip_keypoints.npz") as data:
        assert data["keypoints3d"].shape == (11, 87, 3) and np.all(data["confs"] == 1)
        np.testing.assert_array_equal(data["frame_indices"], plan.indices)
        # camera-frame keypoints in mm, person a few metres in front of the camera
        assert 1000 < np.median(data["keypoints3d"][..., 2]) < 20000


def test_track_does_not_switch_to_bystander_when_subject_drops_out():
    """Regression (review finding): a one-frame subject dropout must become a gap,
    not an identity switch to a distant bystander that stays matched afterwards."""
    from hmb.detect_engine import select_track

    subject = lambda x: np.array([x, 200.0, 250.0, 480.0, 0.95])
    bystander = np.array([20.0, 170.0, 70.0, 95.0, 0.4])
    frames = [np.stack([subject(400 + 3 * t), bystander]) for t in range(10)]
    frames += [bystander[None]]                           # subject missing, bystander visible
    frames += [np.stack([subject(440 + 3 * t), bystander]) for t in range(10)]
    sel = select_track(frames, np.arange(len(frames)) / 24.0)
    assert sel[10] == -1                                  # gap, not the bystander
    assert all(sel[t] == 0 for t in range(len(frames)) if t != 10)


def test_track_follows_fast_motion_without_overlap():
    from hmb.detect_engine import select_track

    # a small fast-moving subject whose boxes stop overlapping between frames
    frames = [np.array([[100.0 + 90 * t, 300.0, 80.0, 120.0, 0.9]]) for t in range(8)]
    sel = select_track(frames, np.arange(8) / 24.0)
    assert sel.tolist() == [0] * 8


# ------------------------------------------------------------------ person='boxes' (external box prior)
def test_select_boxes_follows_the_prior_and_leaves_gaps():
    from hmb.detect_engine import select_boxes, select_person

    a, b = [0.0, 0.0, 10.0, 20.0, 0.9], [30.0, 0.0, 10.0, 20.0, 0.8]
    frames = [np.array([a, b]), np.array([a, b]), np.array([a, b]), np.zeros((0, 5)), np.array([a, b]),
              np.array([a])]
    nan = [np.nan] * 4
    prior = np.array([[31.0, 1.0, 10.0, 20.0],   # on b -> 1
                      [1.0, 0.0, 10.0, 20.0],    # on a -> 0
                      nan,                       # no prior box: gap, never a fallback to someone else
                      [0.0, 0.0, 10.0, 20.0],    # prior but no detection: gap
                      [50.0, 50.0, 10.0, 10.0],  # nothing overlaps: gap
                      [7.0, 0.0, 10.0, 20.0]])   # IoU 3/17 < 0.3: gap
    assert select_boxes(frames, prior).tolist() == [1, 0, -1, -1, -1, -1]
    assert select_boxes(frames, prior, min_iou=0.0).tolist() == [1, 0, -1, -1, -1, 0]
    assert select_person(frames, np.arange(6) / 24.0, "boxes", prior=prior).tolist() == [1, 0, -1, -1, -1, -1]
    with pytest.raises(ValueError):
        select_person(frames, np.arange(6) / 24.0, "boxes")
    with pytest.raises(ValueError):
        select_boxes(frames, prior[:3])


def test_prior_for_frames_matches_by_time_or_index_and_rescales_to_decoded_frames():
    from hmb.detect_engine import load_person_boxes, prior_for_frames

    # track per source frame at 30 fps on a 200 x 100 source; analysed frames 0, 2, 5 decoded at 100 x 50
    track = {"timestamps": [k / 30.0 for k in range(6)],
             "boxes_xyxy": [[10 * k, 10, 10 * k + 40, 90] if k != 2 else None for k in range(6)]}
    prior = load_person_boxes(track)
    out = prior_for_frames(prior, [0, 2, 5], [0.0, 2 / 30.0, 5 / 30.0], (100, 50), (200, 100))
    np.testing.assert_allclose(out[0], [0, 5, 20, 40])
    assert np.isnan(out[1]).all()                       # null box in the track
    np.testing.assert_allclose(out[2], [25, 5, 20, 40])
    # timestamps further than half a track frame from every entry have no prior
    late = prior_for_frames(prior, [9], [9 / 30.0], (200, 100), (200, 100))
    assert np.isnan(late).all()
    # nearest timestamp within tolerance (e.g. rounded container timestamps)
    near = prior_for_frames(prior, [3], [3 / 30.0 + 0.01], (200, 100), (200, 100))
    np.testing.assert_allclose(near[0], [30, 10, 40, 80])
    # explicit frame_indices are matched exactly; image_size names the coordinate frame of the boxes
    indexed = load_person_boxes({"frame_indices": [4, 5], "boxes_xyxy": [[0, 0, 50, 25], [10, 0, 60, 25]],
                                 "image_size": [100, 50]})
    out = prior_for_frames(indexed, [4, 5, 6], [0.0, 0.0, 0.0], (200, 100), (400, 200))
    np.testing.assert_allclose(out[:2], [[0, 0, 100, 50], [20, 0, 100, 50]])
    assert np.isnan(out[2]).all()


def test_load_person_boxes_validation(tmp_path):
    from hmb.detect_engine import load_person_boxes

    path = tmp_path / "boxes.json"
    path.write_text(json.dumps({"timestamps": [0.0, 0.1], "boxes_xyxy": [[0, 0, 5, 5], [3, 3, 3, 9]]}))
    prior = load_person_boxes(path)
    np.testing.assert_allclose(prior["boxes"][0], [0, 0, 5, 5])
    assert np.isnan(prior["boxes"][1]).all()            # degenerate box = no prior
    for bad in ({"timestamps": [0.0]}, {"boxes_xyxy": [[0, 0, 1, 1]]},
                {"timestamps": [0.0, 1.0], "boxes_xyxy": [[0, 0, 1, 1]]},
                {"timestamps": [0.0], "boxes_xyxy": [[0, 0, 1]]},
                {"timestamps": [0.0], "boxes_xyxy": [[0, 0, float("nan"), 1]]},
                {"timestamps": [0.0], "boxes_xyxy": [[0, 0, 1, 1]], "image_size": [10]}):
        with pytest.raises(ValueError):
            load_person_boxes(bad)
    with pytest.raises(ValueError):
        DetectEngine({"person": "boxes"})               # boxes mode needs a track
    with pytest.raises(ValueError):
        DetectEngine({"person": "boxes", "person_boxes": str(path), "prior_min_iou": 1.5})


def test_boxes_mode_engine_follows_the_prior_and_reselection(tmp_path):
    video = write_video(tmp_path / "clip.avi", frames=6)  # 64 x 48, 24 fps
    # the prior follows the bystander ([0, 0, 8, 10]) on frames 1-3, the main person afterwards,
    # and has no box on frame 0
    main = lambda k: [10 + k, 5, 40 + k, 45]
    boxes = [None, [0, 0, 8, 10], [0, 0, 8, 10], [0, 0, 8, 10], main(4), main(5)]
    prior_path = tmp_path / "person_boxes.json"
    prior_path.write_text(json.dumps({"timestamps": [k / 24.0 for k in range(6)], "boxes_xyxy": boxes}))
    detector = FakeDetector(extra={1, 2, 3}, extra_first={2})
    _, status = run(tmp_path, video, detector, person="boxes", person_boxes=str(prior_path))  # per-job options
    assert status["ok"], status.get("error")
    assert status["person"] == "boxes" and status["frames_detected"] == 5
    assert status["prior_frames"] == 5 and status["prior_frames_unmatched"] == 0
    session = tmp_path / "S"
    with np.load(session / "baseline/clip_keypoints.npz") as data:
        assert data["selected_detection"].tolist() == [-1, 1, 0, 1, 0, 0]
        assert str(data["person_selection"]) == "boxes"
        np.testing.assert_array_equal(data["confs"][0], 0)
        np.testing.assert_allclose(data["keypoints3d"][1:4, 0, 0], [1000.0, 2000.0, 3000.0])  # bystander x=0
        np.testing.assert_allclose(data["keypoints3d"][4:, 0, 0], [4014.0, 5015.0])
        assert data["person_prior_boxes"].shape == (6, 4) and np.isnan(data["person_prior_boxes"][0]).all()
        config = json.loads(str(data["detector_config"]))
        assert config["person"] == "boxes" and config["person_boxes"] == str(prior_path)
        assert config["person_boxes_sha256"] and config["prior_min_iou"] == 0.3

    # re-selection from the cached detections: back to 'track' (no prior key), then another prior
    reselect_session(session, "clip", person="track")
    with np.load(session / "baseline/clip_keypoints.npz") as data:
        assert data["selected_detection"].tolist() == [0, 0, 1, 0, 0, 0]
        assert "person_prior_boxes" not in data.files
    prior_path.write_text(json.dumps({"timestamps": [k / 24.0 for k in range(6)],
                                      "boxes_xyxy": [main(k) for k in range(6)]}))
    reselect_session(session, "clip", person="boxes", person_boxes=str(prior_path))
    with np.load(session / "baseline/clip_keypoints.npz") as data:
        assert data["selected_detection"].tolist() == [0, 0, 1, 0, 0, 0]
        assert str(data["person_selection"]) == "boxes" and "person_prior_boxes" in data.files


def test_existing_person_modes_write_no_prior(tmp_path):
    video = write_video(tmp_path / "clip.avi", frames=4)
    for mode in ("first", "track"):
        _, status = run(tmp_path, video, FakeDetector(extra={1}), {"person": mode}, force=True)
        assert status["ok"] and "prior_frames" not in status
        with np.load(tmp_path / "S/baseline/clip_keypoints.npz") as data:
            assert "person_prior_boxes" not in data.files and str(data["person_selection"]) == mode
            assert "person_boxes" not in json.loads(str(data["detector_config"]))
