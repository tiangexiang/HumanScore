"""CPU tests for hmb.mesh_engine and the reusable GVHMR stages in hmb.gvhmr_prepare.

GPU/model integration (bitwise parity with hmb.gvhmr_prepare) is opt-in:
HMB_RUN_GPU_TESTS=1 runs ``test_gpu_parity_with_gvhmr_prepare`` (~2 min).
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
cv2 = pytest.importorskip("cv2")

from hmb import gvhmr_prepare as gp  # noqa: E402
from hmb import mesh_engine as me  # noqa: E402
from hmb import video as hvideo  # noqa: E402

GVHMR_ROOT = gp.REPO_ROOT / "GVHMR"


@pytest.fixture(scope="module")
def hmr4d():
    if not (GVHMR_ROOT / "hmr4d").is_dir():
        pytest.skip("GVHMR source tree missing")
    pytest.importorskip("pytorch_lightning")
    pytest.importorskip("ultralytics")
    if str(GVHMR_ROOT) not in sys.path:
        sys.path.insert(0, str(GVHMR_ROOT))
    gp._defer_optional_imports(GVHMR_ROOT)
    return GVHMR_ROOT


# ----------------------------------------------------------------- options/jobs
def test_options_defaults_and_validation():
    opts = me.resolve_options(None)
    assert opts == me.DEFAULT_OPTIONS
    assert opts["camera"] == "static"
    assert me.resolve_options({"gmc_workers": 0, "unknown": 1})["gmc_workers"] == 0
    for bad in ({"camera": "dpvo"}, {"precision": "int8"}, {"decoder": "ffmpeg"}, {"pose_batch": 0},
                {"gmc_workers": -1}, {"f_mm": -3.0}, {"f_mm": float("nan")}):
        with pytest.raises(ValueError):
            me.resolve_options(bad)


def test_make_job_defaults_and_per_job_overrides(tmp_path):
    job = me.make_job({"session_dir": str(tmp_path / "S"), "video": "/data/clip a.mp4", "precision": "fp32"},
                      {"precision": "bf16", "pose_batch": 32})
    assert job.stem == "clip a"
    assert job.options["precision"] == "fp32" and job.options["pose_batch"] == 32
    assert job.result_path == tmp_path / "S/gvhmr/clip a/hmr4d_results.pt"
    assert job.status_path == tmp_path / "S/cache/engine_status/mesh.json"
    job = me.make_job({"session_dir": str(tmp_path / "S"), "stem": "x"})
    assert job.video == tmp_path / "S/input/x.mp4"
    with pytest.raises(ValueError):
        me.make_job({"stem": "x"})
    with pytest.raises(ValueError):
        me.make_job({"session_dir": str(tmp_path)})


def test_load_jobs_accepts_dict_or_list(tmp_path):
    path = tmp_path / "jobs.json"
    path.write_text(json.dumps({"options": {"precision": "fp32"}, "jobs": [{"session_dir": "a", "stem": "b"}]}))
    assert me.load_jobs(path) == ({"precision": "fp32"}, [{"session_dir": "a", "stem": "b"}])
    path.write_text(json.dumps([{"session_dir": "a", "stem": "b"}]))
    assert me.load_jobs(path) == ({}, [{"session_dir": "a", "stem": "b"}])


# ---------------------------------------------------------------------- decode
def _write_video(path: Path, frames: int = 12, fps: float = 30.0, size=(96, 64)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(frames):
        image = np.full((size[1], size[0], 3), 10 * i % 255, np.uint8)
        cv2.rectangle(image, (5 + i, 5), (40 + i, 50), (0, 200, 255), -1)
        writer.write(image)
    writer.release()
    return path


def test_decode_follows_frame_plan_and_defaults_to_all_frames(tmp_path):
    video = _write_video(tmp_path / "clip.mp4", frames=12, fps=30.0)
    session = tmp_path / "S"
    job = me.make_job({"session_dir": str(session), "video": str(video)})
    decoded = me.decode_job(job)
    assert decoded.indices == list(range(12)) and decoded.plan_source == "all_frames"
    assert decoded.timestamps[3] == pytest.approx(3 / 30.0)
    assert len(decoded.frames_bgr) == len(decoded.small_rgb) == 12
    assert decoded.small_rgb[0].shape == (32, 48, 3)  # img_ds=0.5
    plan = hvideo.plan_frames(video, sample_fps=12.0, max_side=48)
    plan.save(session / "input" / hvideo.PLAN_NAME)
    decoded = me.decode_job(job)
    assert decoded.plan_source == "frames.json"
    assert decoded.indices == plan.indices and decoded.timestamps == plan.timestamps
    assert (decoded.width, decoded.height) == plan.output_size == (48, 32)
    assert decoded.frames_bgr[0].shape == (32, 48, 3)


def test_person_boxes_option_maps_the_subject_track_onto_the_decoded_frames(tmp_path):
    video = _write_video(tmp_path / "clip.mp4", frames=12, fps=30.0)       # 96 x 64 source
    session = tmp_path / "S"
    boxes = tmp_path / "person_boxes.json"
    rows = [[4.0 + i, 6.0, 40.0 + i, 50.0] for i in range(12)]
    rows[6] = None
    boxes.write_text(json.dumps({"timestamps": [i / 30.0 for i in range(12)], "boxes_xyxy": rows}))
    with pytest.raises(ValueError, match="person_boxes"):
        me.resolve_options({"person_boxes": 5})
    plain = me.make_job({"session_dir": str(session), "video": str(video)})
    assert me.job_prior_xyxy(plain, me.decode_job(plain)) is None
    job = me.make_job({"session_dir": str(session), "video": str(video), "person_boxes": str(boxes)})
    plan = hvideo.plan_frames(video, sample_fps=15.0, max_side=48)          # every other frame, half size
    plan.save(session / "input" / hvideo.PLAN_NAME)
    decoded = me.decode_job(job)
    prior = me.job_prior_xyxy(job, decoded)
    assert prior.shape == (len(plan.indices), 4)
    for row, index in zip(prior, plan.indices):
        if rows[index] is None:
            assert np.isnan(row).all()
        else:
            assert np.allclose(row, np.asarray(rows[index]) * 0.5)


def test_decode_rejects_missing_video_and_single_frame(tmp_path):
    job = me.make_job({"session_dir": str(tmp_path / "S"), "video": str(tmp_path / "missing.mp4")})
    with pytest.raises(FileNotFoundError):
        me.decode_job(job)
    video = _write_video(tmp_path / "one.mp4", frames=1)
    with pytest.raises(ValueError, match="two frames"):
        me.decode_job(me.make_job({"session_dir": str(tmp_path / "S"), "video": str(video)}))


# ---------------------------------------------------------------- crops (upstream)
def test_crops_match_upstream_get_batch(hmr4d):
    from hmr4d.utils.preproc.vitfeat_extractor import get_batch
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 256, (3, 300, 500, 3), dtype=np.uint8)
    # third box is large enough to trigger the anti-alias blur branch
    boxes = torch.tensor([[250.0, 150.0, 220.0], [240.0, 160.0, 300.0], [250.0, 150.0, 1300.0]])
    expected, expected_boxes = get_batch(frames, boxes, img_ds=1.0, path_type="np")
    crops, crop_boxes = me.make_crops(list(frames), boxes, 1.0, device=None)
    assert torch.equal(crops, expected) and torch.equal(crop_boxes, expected_boxes)
    u8, _ = me.crop_frames(list(frames), boxes, 1.0)
    assert u8.dtype == torch.uint8 and u8.shape == (3, 256, 256, 3)
    assert torch.allclose(me.normalize_crops(u8, device=None), expected)


# ---------------------------------------------------------------------- tracking
def test_one_track_selects_largest_track_and_fills_gaps(hmr4d):
    from hmr4d.utils.net_utils import moving_average_smooth
    big = lambda t: np.array([10.0 + t, 10.0, 60.0 + t, 90.0], np.float32)  # noqa: E731
    small = np.array([0.0, 0.0, 5.0, 5.0], np.float32)
    history = []
    for t in range(8):
        frame = [{"id": 7, "bbx_xyxy": small}]
        if t not in (0, 3, 4):  # missing at the start and in the middle
            frame.append({"id": 3, "bbx_xyxy": big(t)})
        history.append(frame)
    boxes, mask = gp.one_track_from_history(history, 8, 100, 100)
    assert mask.tolist() == [False, True, True, False, False, True, True, True]
    raw = torch.tensor(np.stack([big(t) for t in range(8)]))
    raw[0] = raw[1]  # edge gap copies the nearest observed box
    expected = moving_average_smooth(moving_average_smooth(raw, window_size=5, dim=0), window_size=5, dim=0)
    assert torch.allclose(boxes, expected)  # middle gap (3, 4) is linear, so equals the true boxes
    with pytest.raises(ValueError, match="no person"):
        gp.one_track_from_history([[], []], 2, 100, 100)


def test_select_track_by_prior_follows_the_subject_not_the_largest_track():
    big = {"id": 3, "bbx_xyxy": np.array([10.0, 10.0, 60.0, 90.0])}
    history = [[big, {"id": 7, "bbx_xyxy": np.array([70.0 + t, 20.0, 90.0 + t, 60.0])}] for t in range(6)]
    history[4] = [big]                                                    # subject missed in one frame
    prior = np.array([[70.0 + t, 22.0, 89.0 + t, 61.0] for t in range(6)])
    prior[2] = np.nan                                                     # unknown subject box
    assert gp.select_track_by_prior(history, prior) == 7
    assert gp.select_track_by_prior(history, np.full((6, 4), np.nan)) is None
    assert gp.select_track_by_prior(history, [[200.0, 200.0, 220.0, 220.0]] * 6) is None
    tie = [[{"id": 9, "bbx_xyxy": np.array([0.0, 0.0, 10.0, 10.0])},
            {"id": 2, "bbx_xyxy": np.array([0.0, 0.0, 10.0, 10.0])}]]
    assert gp.select_track_by_prior(tie, [[0.0, 0.0, 10.0, 10.0]]) == 2


def _textured_frames(n=6, shift=(3, 2), size=(160, 120), seed=0):
    rng = np.random.default_rng(seed)
    base = cv2.GaussianBlur(rng.integers(0, 256, (size[1] * 2, size[0] * 2, 3), dtype=np.uint8), (5, 5), 1.5)
    frames = []
    for i in range(n):
        x, y = 20 + i * shift[0], 20 + i * shift[1]
        frames.append(np.ascontiguousarray(base[y:y + size[1], x:x + size[0]]))
    return frames


def test_parallel_gmc_warps_equal_sequential_ultralytics_gmc(hmr4d):
    from ultralytics.trackers.utils.gmc import GMC
    frames = _textured_frames(7)
    frames[4] = np.zeros_like(frames[4])  # featureless frame: upstream re-initialization branch
    sequence = [0, 1, 2, 4, 5, 6]  # frame 3 had no detection, so GMC never saw it
    gmc = GMC(method="sparseOptFlow")
    expected = [gmc.apply(frames[i]) for i in sequence]
    warps = gp.sparse_optflow_warps(frames, sequence, workers=3)
    assert len(warps) == len(expected)
    for got, want in zip(warps, expected):
        assert np.array_equal(got, want)
    assert gp.sparse_optflow_warps(frames, [], workers=2) == []


def _boxes(rows, shape):
    from ultralytics.engine.results import Boxes
    return Boxes(np.asarray(rows, np.float32).reshape(-1, 6), shape).numpy()


def test_track_detections_precomputed_gmc_matches_inline(hmr4d):
    frames = _textured_frames(9, shift=(2, 1))
    shape = frames[0].shape[:2]
    detections = []
    for t in range(9):
        rows = [] if t == 4 else [[30 + 2 * t, 20 + t, 80 + 2 * t, 110, 0.9, 0]]
        detections.append(_boxes(rows, shape))
    inline, mask_inline = gp.GVHMRModels.track_detections(detections, frames, None)
    sequence = [i for i, d in enumerate(detections) if len(d)]
    warps = gp.sparse_optflow_warps(frames, sequence, workers=2)
    pre, mask_pre = gp.GVHMRModels.track_detections(detections, frames, warps)
    assert torch.equal(inline, pre) and torch.equal(mask_inline, mask_pre)
    assert inline.dtype == torch.float32 and inline.shape == (9, 4)
    assert not mask_pre[4] and mask_pre[1]
    with pytest.raises(RuntimeError, match="warp sequence"):
        gp.GVHMRModels.track_detections(detections, frames, warps + [np.eye(2, 3)])


def test_track_detections_follows_the_prior_boxes(hmr4d):
    frames = _textured_frames(8, shift=(1, 1))
    shape = frames[0].shape[:2]
    big = [5, 5, 75, 115, 0.9, 0]                                         # largest: upstream's choice
    small = lambda t: [100 + t, 30, 140 + t, 100, 0.9, 0]                 # noqa: E731 - the subject
    detections = [_boxes([big, small(t)], shape) for t in range(8)]
    largest, _ = gp.GVHMRModels.track_detections(detections, frames, None)
    assert float(largest[4, 0]) < 20
    prior = np.array([small(t)[:4] for t in range(8)], dtype=float)
    followed, mask = gp.GVHMRModels.track_detections(detections, frames, None, prior_xyxy=prior)
    assert abs(float(followed[4, 0]) - 104) < 2 and bool(mask.all())
    with pytest.raises(ValueError, match="overlaps the given person boxes"):
        gp.GVHMRModels.track_detections(detections, frames, None, prior_xyxy=np.full((8, 4), np.nan))


def test_matmul_precision_is_scoped():
    previous = torch.backends.cuda.matmul.allow_tf32
    with gp.matmul_precision("tf32"):
        assert torch.backends.cuda.matmul.allow_tf32
        with gp.matmul_precision("fp32"):
            assert not torch.backends.cuda.matmul.allow_tf32
    assert torch.backends.cuda.matmul.allow_tf32 == previous
    with pytest.raises(ValueError):
        with gp.matmul_precision("int4"):
            pass


# ------------------------------------------------------------- engine orchestration
class _FakeEngine(me.MeshEngine):
    """MeshEngine with the two GPU phases replaced by deterministic fakes."""

    def __init__(self, fail_stems=(), **kwargs):
        super().__init__(models=object(), verbose=False, **kwargs)
        self.fail_stems = set(fail_stems)
        self.prepared, self.inferred = [], []

    def prepare(self, job):
        if job.stem in self.fail_stems:
            raise RuntimeError(f"boom {job.stem}")
        self.prepared.append(job.stem)
        T = 4
        return me.PreparedJob(job, list(range(T)), [i / 24 for i in range(T)], 64, 48, 24.0, "all_frames",
                              torch.zeros(T, 4), torch.ones(T, 3), torch.tensor([True, True, False, True]),
                              torch.zeros(T, 256, 256, 3, dtype=torch.uint8), torch.ones(T, 3),
                              torch.eye(3).repeat(T, 1, 1), {"decode": 0.01})

    def infer(self, prepared):
        self.inferred.append(prepared.job.stem)
        T = len(prepared.indices)
        params = {"body_pose": torch.zeros(T, 63), "betas": torch.zeros(T, 10),
                  "global_orient": torch.zeros(T, 3), "transl": torch.zeros(T, 3)}
        prediction = {"smpl_params_global": dict(params), "smpl_params_incam": dict(params),
                      "K_fullimg": torch.eye(3).repeat(T, 1, 1), "net_outputs": {}}
        return prediction, {"bbx": {"bbx_xyxy": prepared.bbx_xyxy, "bbx_xys": prepared.bbx_xys,
                                    "tracked": prepared.tracked},
                            "vitpose": torch.zeros(T, 17, 3), "vit_features": torch.zeros(T, 1024)}, {"vitpose": 0.02}


def test_run_isolates_failures_writes_status_and_skips_existing(tmp_path):
    entries = [{"session_dir": str(tmp_path / s), "stem": s, "video": str(tmp_path / f"{s}.mp4")}
               for s in ("a", "bad", "c")]
    entries.insert(2, {"stem": "no_session"})
    engine = _FakeEngine(fail_stems={"bad"})
    statuses = engine.run(entries)
    assert [s["ok"] for s in statuses] == [True, False, False, True]
    assert "boom bad" in statuses[1]["error"] and "session_dir" in statuses[2]["error"]
    result = tmp_path / "a/gvhmr/a/hmr4d_results.pt"
    data = torch.load(result, weights_only=False)
    assert set(data) >= {"smpl_params_global", "smpl_params_incam", "K_fullimg", "net_outputs", "hmb"}
    assert data["hmb"]["frame_indices"] == [0, 1, 2, 3] and data["hmb"]["tracked"].tolist() == [True, True, False, True]
    assert (tmp_path / "a/gvhmr/a/preprocess/vitpose.pt").is_file()
    status = json.loads((tmp_path / "a/cache/engine_status/mesh.json").read_text())
    assert status["ok"] and status["tracked_fraction"] == pytest.approx(0.75) and status["seconds"] >= 0
    assert status["prefetched"] and "vitpose" in status["stages"] and "decode" in status["stages"]
    bad = json.loads((tmp_path / "bad/cache/engine_status/mesh.json").read_text())
    assert bad["ok"] is False and "boom" in bad["error"]
    # second pass: existing outputs are skipped (not prepared), --force recomputes
    engine2 = _FakeEngine()
    statuses = engine2.run(entries[:1])
    assert statuses[0]["skipped"] and engine2.prepared == []
    statuses = engine2.run(entries[:1], force=True)
    assert not statuses[0]["skipped"] and engine2.prepared == ["a"]


def test_process_without_prefetch(tmp_path):
    engine = _FakeEngine(options={"save_preprocess": False})
    status = engine.process({"session_dir": str(tmp_path / "S"), "stem": "v"})
    assert status["ok"] and not status["prefetched"]
    assert (tmp_path / "S/gvhmr/v/hmr4d_results.pt").is_file()
    assert not (tmp_path / "S/gvhmr/v/preprocess").exists()
    provenance = json.loads((tmp_path / "S/gvhmr/v/mesh_engine.json").read_text())
    assert provenance["frames"] == 4 and provenance["options"]["save_preprocess"] is False


def test_single_video_cli_job_writes_plan_once(tmp_path):
    video = _write_video(tmp_path / "clip.mp4", frames=10, fps=30.0)
    args = type("A", (), {"session_dir": str(tmp_path / "S"), "video": str(video), "stem": None,
                          "sample_fps": 15.0, "max_side": None})()
    entry = me._single_job(args)
    assert entry["stem"] == "clip" and (tmp_path / "S/input/clip.mp4").is_symlink()
    plan = hvideo.FramePlan.load(tmp_path / "S/input/frames.json")
    assert plan.sample_fps == 15.0 and plan.indices == [0, 2, 4, 6, 8]
    me._single_job(args)  # same plan again is fine
    args.sample_fps = 10.0
    with pytest.raises(SystemExit):
        me._single_job(args)


# ------------------------------------------------------------------ opt-in GPU
# Any single-person clip (e.g. a few seconds of walking); set HMB_TEST_VIDEO to run.
TEST_VIDEO = Path(os.environ.get("HMB_TEST_VIDEO", "missing-test-clip.mp4"))


@pytest.mark.skipif(os.environ.get("HMB_RUN_GPU_TESTS") != "1", reason="set HMB_RUN_GPU_TESTS=1")
def test_gpu_parity_with_gvhmr_prepare(tmp_path):
    if not torch.cuda.is_available() or not TEST_VIDEO.is_file():
        pytest.skip("needs CUDA and a test clip (HMB_TEST_VIDEO)")
    import argparse
    models = gp.GVHMRModels()
    args = argparse.Namespace(video=str(TEST_VIDEO), output_root=str(tmp_path / "ref"), static_cam=True, f_mm=None)
    ref = torch.load(gp.reconstruct(args, models=models), weights_only=False)
    engine = me.MeshEngine({"decoder": "upstream", "precision": "fp32", "det_batch": 1, "pose_batch": 16,
                            "feature_batch": 16}, models=models, verbose=False).load()
    status = engine.process({"session_dir": str(tmp_path / "S"), "stem": TEST_VIDEO.stem, "video": str(TEST_VIDEO)})
    assert status["ok"], status
    got = torch.load(status["output"], weights_only=False)
    for space in ("smpl_params_global", "smpl_params_incam"):
        for key, value in ref[space].items():
            assert torch.equal(value, got[space][key]), (space, key)
