"""HADM engine: job layout, frame plans, pipelining and preprocessing parity.

CPU tests use a stub detector.  The real-model parity test needs a GPU and the
HADM checkpoint and only runs with ``HMB_HADM_GPU_TESTS=1``.
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from hmb import hadm_engine
from hmb.hadm_engine import (HadmEngine, HadmOptions, JobPaths, job_paths, job_source,
                             max_extra_limb_confidence, outputs_current)
from hmb.video import FramePlan


def _write_video(path: Path, count: int = 12, fps: float = 24.0, size=(64, 48)) -> Path:
    cv2 = pytest.importorskip("cv2")
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        pytest.skip("No mp4v encoder available")
    for i in range(count):
        writer.write(np.full((size[1], size[0], 3), 10 * i, dtype=np.uint8))
    writer.release()
    return path


class _Identity:
    def get_transform(self, image):
        class T:
            @staticmethod
            def apply_image(img):
                return img
        return T()


def _stub_engine(monkeypatch=None, fail_on=None, **options) -> HadmEngine:
    """Engine whose 'detector' reports the frame's mean brightness as a score."""
    torch = pytest.importorskip("torch")
    options.setdefault("jpeg_roundtrip", False)
    engine = HadmEngine(HadmOptions(device="cpu", **options))
    from types import SimpleNamespace
    engine.loaded = object()
    engine.torch = torch
    engine.aug = _Identity()
    engine.model = SimpleNamespace(backbone=SimpleNamespace(net=SimpleNamespace(blocks=[])))
    engine.load = lambda: engine
    calls = []

    def detect_batch(batch):
        calls.append(len(batch))
        return [{"bbox": [[0.0, 0.0, 1.0, 1.0]], "score": [float(img.mean()) / 1000.0],
                 "class": ["human with extra arm"]} for img, _, _ in batch]
    engine.detect_batch = detect_batch
    engine.calls = calls
    return engine


def test_defaults_are_the_parity_preserving_fast_settings():
    options = HadmOptions().validated()
    assert (options.precision, options.attention, options.jpeg_roundtrip, options.input_format) == \
        ("fp16", "sdpa", True, "BGR")
    assert options.batch_size > 1 and options.autocast_scope == "backbone"


def test_options_validate_and_reject_unknown_keys():
    options = HadmOptions.from_dict({"batch_size": "4", "precision": "bf16"}, attention=None)
    assert options.batch_size == 4 and options.precision == "bf16" and options.attention == "sdpa"
    assert options.hadm_root  # default checkout filled in
    with pytest.raises(ValueError, match="Unknown"):
        HadmOptions.from_dict({"batchsize": 4})
    for bad in ({"precision": "int8"}, {"attention": "flash"}, {"batch_size": 0},
                {"score_thresh": float("nan")}, {"autocast_scope": "heads"}, {"input_format": "YUV"}):
        with pytest.raises(ValueError):
            HadmOptions.from_dict(bad)
    assert set(hadm_engine.REFERENCE_OPTIONS) <= set(options.as_dict())


def test_session_layout_paths(tmp_path):
    paths = job_paths({"session_dir": str(tmp_path), "stem": "clip"})
    assert paths.output_json == tmp_path / "cache/multi_limbs/clip/hadm_results/output.json"
    assert paths.frame_index.name == "frame_indices.json"
    assert paths.status == tmp_path / "cache/engine_status/hadm.json"
    other = job_paths({"video": "/x/y/clip2.mp4", "output_dir": str(tmp_path / "o")})
    assert other.stem == "clip2" and other.status.parent == tmp_path / "o"
    with pytest.raises(ValueError):
        job_paths({"stem": "s"})


def test_extra_limb_confidence_uses_only_extra_classes():
    record = {"score": [0.9, 0.02, 0.01], "class": ["human missing arm", "human with extra leg",
                                                     "human with extra hand"]}
    assert max_extra_limb_confidence(record) == 0.02
    assert max_extra_limb_confidence({"score": [], "class": []}) == 0.0


def test_pipeline_keeps_order_for_partial_batches():
    engine = _stub_engine(batch_size=4, cpu_workers=3, prefetch_batches=1)
    frames = [np.full((8, 8, 3), i, dtype=np.uint8) for i in range(10)]
    records, stats = engine.detect(iter(frames), num_frames=10)
    assert [r["score"][0] for r in records] == pytest.approx([i / 1000 for i in range(10)])
    assert engine.calls == [4, 4, 2] and stats["frames"] == 10
    with pytest.raises(RuntimeError, match="Decoded 10 of 11"):
        engine.detect(iter(frames), num_frames=11)


def test_prepare_keeps_bgr_by_default_and_can_flip_to_rgb():
    frame = np.zeros((4, 6, 3), np.uint8)
    frame[..., 0] = 1  # blue channel in BGR
    engine = _stub_engine()
    image, height, width = engine.prepare(frame)
    assert (height, width) == (4, 6) and image[0, 0].tolist() == [1, 0, 0]
    engine.configure(input_format="RGB")
    assert engine.prepare(frame)[0][0, 0].tolist() == [0, 0, 1]
    assert engine.options.detection_settings()["input_format"] == "RGB"


def test_pipeline_propagates_decode_errors():
    engine = _stub_engine(batch_size=2)

    def broken():
        yield np.zeros((4, 4, 3), np.uint8)
        raise OSError("corrupt stream")
    with pytest.raises(OSError, match="corrupt stream"):
        engine.detect(broken())


def test_process_writes_schema_status_and_skips(tmp_path):
    video = _write_video(tmp_path / "clip.mp4", count=12)
    session = tmp_path / "S"
    engine = _stub_engine(batch_size=5)
    job = {"session_dir": str(session), "stem": "clip", "video": str(video)}
    status = engine.process(job)
    assert status["ok"] and not status["skipped"] and status["frames"] == 12, status.get("error")
    out = json.loads((session / "cache/multi_limbs/clip/hadm_results/output.json").read_text())
    assert list(out) == [f"frame_{i:06d}.jpg" for i in range(12)]
    assert set(out["frame_000000.jpg"]) == {"bbox", "score", "class"}
    index = json.loads((session / "cache/multi_limbs/clip/hadm_results/frame_indices.json").read_text())
    assert index["source_indices"] == list(range(12)) and not index["planned"]
    assert index["timestamps"][1] == pytest.approx(1 / 24)
    saved = json.loads((session / "cache/engine_status/hadm.json").read_text())
    assert saved["ok"] and saved["seconds"] >= 0 and saved["error"] is None
    assert engine.process(job)["skipped"]
    assert not engine.process(job, force=True)["skipped"]


def test_plan_is_used_exactly_and_plan_change_invalidates(tmp_path):
    video = _write_video(tmp_path / "clip.mp4", count=12)
    session = tmp_path / "S"
    plan = FramePlan(video=str(video), source_fps=24.0, num_source_frames=12, width=64, height=48,
                     sample_fps=12.0, indices=[0, 2, 4, 6, 8, 10],
                     timestamps=[i / 24 for i in (0, 2, 4, 6, 8, 10)])
    (session / "input").mkdir(parents=True)
    plan.save(session / "input")
    engine = _stub_engine(batch_size=4)
    job = {"session_dir": str(session), "stem": "clip", "video": str(video)}
    status = engine.process(job)
    assert status["ok"] and status["frames"] == 6, status.get("error")
    out = json.loads((session / "cache/multi_limbs/clip/hadm_results/output.json").read_text())
    scores = [out[f"frame_{i:06d}.jpg"]["score"][0] * 1000 for i in range(6)]
    assert np.allclose(scores, [20 * i for i in range(6)], atol=4)  # frames 0,2,4,... (lossy codec)
    index = json.loads((session / "cache/multi_limbs/clip/hadm_results/frame_indices.json").read_text())
    assert index["source_indices"] == plan.indices and index["timestamps"] == plan.timestamps
    assert engine.process(job)["skipped"]
    plan.indices, plan.timestamps = [0, 3, 6, 9], [i / 24 for i in (0, 3, 6, 9)]
    plan.save(session / "input")
    again = engine.process(job)
    assert again["ok"] and not again["skipped"] and again["frames"] == 4


def test_output_without_index_is_reused_only_if_complete(tmp_path):
    paths = JobPaths(tmp_path, "c", tmp_path / "o", tmp_path / "s.json")
    paths.output_dir.mkdir()
    paths.output_json.write_text(json.dumps({"frame_000000.jpg": {"bbox": [], "score": [], "class": []}}))
    source = hadm_engine.FrameSource(None, lambda: iter(()), 2, {"planned": True, "source_indices": [0, 1]})
    assert not outputs_current(paths, source)
    source.num_frames = 1
    assert outputs_current(paths, source)


def test_one_failing_job_does_not_stop_others(tmp_path, monkeypatch):
    video = _write_video(tmp_path / "ok.mp4", count=3)
    jobs = {"options": {"batch_size": 2, "device": "cpu", "jpeg_roundtrip": False},
            "jobs": [{"session_dir": str(tmp_path / "A"), "stem": "missing", "video": str(tmp_path / "no.mp4")},
                     {"session_dir": str(tmp_path / "B"), "stem": "ok", "video": str(video)}]}
    spec = tmp_path / "jobs.json"
    spec.write_text(json.dumps(jobs))
    torch = pytest.importorskip("torch")

    def fake_load(self):
        if self.loaded is None:
            self.loaded, self.torch, self.aug = object(), torch, _Identity()
            self.detect_batch = lambda batch: [{"bbox": [], "score": [], "class": []} for _ in batch]
        return self
    monkeypatch.setattr(HadmEngine, "load", fake_load)
    summary = tmp_path / "summary.json"
    code = hadm_engine.main(["--jobs", str(spec), "--summary", str(summary)])
    results = json.loads(summary.read_text())["jobs"]
    assert code == 1 and not results[0]["ok"] and "FileNotFoundError" in results[0]["error"]
    assert results[1]["ok"] and results[1]["frames"] == 3
    assert json.loads((tmp_path / "A/cache/engine_status/hadm.json").read_text())["ok"] is False


def test_jpeg_roundtrip_matches_imwrite_and_read_image(tmp_path):
    cv2 = pytest.importorskip("cv2")
    pytest.importorskip("detectron2")
    from detectron2.data.detection_utils import read_image
    rng = np.random.default_rng(0)
    frame = cv2.GaussianBlur(rng.integers(0, 255, (48, 64, 3), dtype=np.uint8), (5, 5), 0)
    path = tmp_path / "frame_000000.jpg"
    cv2.imwrite(str(path), frame)
    expected = read_image(str(path), format="BGR")
    assert np.array_equal(hadm_engine.jpeg_roundtrip(frame), expected)
    assert np.array_equal(hadm_engine.read_image_bgr(path), expected)


def test_sdpa_attention_matches_upstream_eager_branch():
    torch = pytest.importorskip("torch")
    pytest.importorskip("detectron2")
    from detectron2.modeling.backbone.vit import Attention
    from detectron2.modeling.backbone.utils import VisionRotaryEmbeddingFast
    torch.manual_seed(0)
    rope = VisionRotaryEmbeddingFast(dim=8, pt_seq_len=16, ft_seq_len=4)
    attn = Attention(64, num_heads=4, qkv_bias=True, rope=rope, xattn=False).eval()
    for p in attn.parameters():
        torch.nn.init.normal_(p, std=0.2)
    x = torch.randn(3, 4, 4, 64)
    with torch.no_grad():
        eager = attn(x)
        fused = hadm_engine.sdpa_attention_forward(attn, x)
    torch.testing.assert_close(fused, eager, rtol=1e-5, atol=1e-5)


class _Blocks:
    def __init__(self, blocks):
        self.blocks = blocks


def test_set_attention_patches_instances_and_restores():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    class A(torch.nn.Module):
        xattn = False

        def forward(self, x):
            return "eager"
    blocks = [SimpleNamespace(attn=A()) for _ in range(3)]
    model = SimpleNamespace(backbone=SimpleNamespace(net=_Blocks(blocks)))
    assert hadm_engine.set_attention(model, "sdpa") == 3
    assert all(b.attn.forward.__func__ is hadm_engine.sdpa_attention_forward for b in blocks)
    assert A().forward(None) == "eager"  # class untouched
    hadm_engine.set_attention(model, "eager")
    assert all(b.attn.forward(None) == "eager" for b in blocks)


def test_hadm_prepare_cli_defaults_are_the_reference_settings(monkeypatch):
    from hmb import hadm_prepare
    for name in ("HMB_HADM_BATCH_SIZE", "HMB_HADM_PRECISION", "HMB_HADM_ATTENTION"):
        monkeypatch.delenv(name, raising=False)
    args = hadm_prepare.build_parser().parse_args(["--hadm-root", "r", "--frames-dir", "f", "--output-dir", "o"])
    assert hadm_prepare.reference_settings(args)
    monkeypatch.setenv("HMB_HADM_PRECISION", "bf16")
    monkeypatch.setenv("HMB_HADM_BATCH_SIZE", "8")
    args = hadm_prepare.build_parser().parse_args(["--hadm-root", "r", "--frames-dir", "f", "--output-dir", "o"])
    assert not hadm_prepare.reference_settings(args) and args.batch_size == 8 and args.precision == "bf16"


@pytest.mark.skipif(os.environ.get("HMB_HADM_GPU_TESTS") != "1", reason="set HMB_HADM_GPU_TESTS=1 (GPU + checkpoint)")
def test_real_model_parity_with_default_inferencer():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from hmb.video import iter_frames
    video = os.environ.get("HMB_TEST_VIDEO", "")  # any clip with > 100 frames
    if not os.path.isfile(video):
        pytest.skip("set HMB_TEST_VIDEO to a test clip")
    frames = [f for _, f in iter_frames(video, [10, 60, 100], rgb=False)]
    engine = HadmEngine(HadmOptions(batch_size=1, precision="fp32", attention="eager",
                                    jpeg_roundtrip=False, resize="pil")).load()
    inferencer = engine.loaded.inferencer()
    with torch.inference_mode():
        reference = [inferencer(f)["instances"].to("cpu") for f in frames]
    for resize in ("pil", "gpu"):  # both bit-identical to upstream DefaultInferencer
        engine.configure(resize=resize)
        records, _ = engine.detect(frames)
        for ref, rec in zip(reference, records):
            assert rec["score"] == ref.scores.tolist() and rec["bbox"] == ref.pred_boxes.tensor.tolist()
    for precision, tol in (("fp32", 1e-4), ("fp16", 3e-3)):
        engine.configure(batch_size=3, attention="sdpa", precision=precision)
        batched, _ = engine.detect(frames)
        for ref, rec in zip(records, batched):
            assert abs(max_extra_limb_confidence(rec) - max_extra_limb_confidence(ref)) < tol


def test_frames_dir_jobs_are_never_jpeg_reencoded(tmp_path, monkeypatch):
    cv2 = pytest.importorskip("cv2")
    pytest.importorskip("detectron2")
    frames = tmp_path / "frames"
    frames.mkdir()
    for i in range(3):
        cv2.imwrite(str(frames / f"img_{i:03d}.jpg"), np.full((8, 8, 3), 40 * i, np.uint8))
    calls = []
    monkeypatch.setattr(hadm_engine, "jpeg_roundtrip", lambda f, q=95: calls.append(1) or f)
    engine = _stub_engine(jpeg_roundtrip=True)
    status = engine.process({"frames_dir": str(frames), "output_dir": str(tmp_path / "out")})
    assert status["ok"] and status["frames"] == 3 and calls == [], status.get("error")
    out = json.loads((tmp_path / "out/output.json").read_text())
    assert list(out) == ["img_000.jpg", "img_001.jpg", "img_002.jpg"]  # original names kept
    engine.detect([np.zeros((8, 8, 3), np.uint8)])
    assert calls == [1]  # in-memory video frames do get the round trip


@pytest.mark.parametrize("shape,out", [((48, 85, 3), (27, 48)), ((30, 20, 3), (45, 30)),
                                       ((64, 64, 3), (64, 64)), ((37, 101, 3), (11, 30)),
                                       ((108, 192, 3), (58, 102))])
def test_gpu_resizer_is_bit_exact_with_pil_bilinear(shape, out):
    torch = pytest.importorskip("torch")
    from PIL import Image
    rng = np.random.default_rng(sum(shape))
    image = rng.integers(0, 256, shape, dtype=np.uint8)
    expected = np.asarray(Image.fromarray(image).resize((out[1], out[0]), Image.BILINEAR))
    got = hadm_engine.PilBilinearResizer("cpu")(torch.from_numpy(image), *out).numpy()
    assert got.dtype == np.uint8 and np.array_equal(got, expected)


def test_engine_gpu_resize_inputs_equal_default_inferencer_preprocessing():
    torch = pytest.importorskip("torch")
    pytest.importorskip("detectron2")
    from detectron2.data import transforms as T
    rng = np.random.default_rng(1)
    frames = [rng.integers(0, 256, (90, 160, 3), dtype=np.uint8) for _ in range(3)]
    engine = _stub_engine(resize="gpu")
    engine.aug = T.ResizeShortestEdge(short_edge_length=64, max_size=64)
    engine.resizer = hadm_engine.PilBilinearResizer("cpu")
    gpu_inputs = engine._inputs([engine.prepare(f) for f in frames])
    engine.configure(resize="pil")
    pil_inputs = engine._inputs([engine.prepare(f) for f in frames])
    for frame, a, b in zip(frames, gpu_inputs, pil_inputs):
        # DefaultInferencer: aug.apply_image -> float32 CHW, original height/width
        reference = torch.as_tensor(engine.aug.get_transform(frame).apply_image(frame)
                                    .astype("float32").transpose(2, 0, 1))
        assert torch.equal(a["image"], reference) and torch.equal(b["image"], reference)
        assert (a["height"], a["width"]) == (90, 160) == (b["height"], b["width"])
