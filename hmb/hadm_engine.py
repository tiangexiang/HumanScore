"""HADM extra-limb detection engine: the input of the extra-limbs score.

Pipeline position (see ``hmb.batch``): runs next to ``hmb.detect_engine`` and
``hmb.mesh_engine``, directly on the video.  One long-lived process loads the
HADM-G detector (EVA-02 ViTDet, EMA weights; see ``hmb.hadm_prepare``) once and
then detects anatomy errors ("human with extra hand / arm / leg / feet", among
other classes) in every planned frame of any number of videos.

Frames are decoded in memory with :mod:`hmb.video` and batched through the
model.  By default each frame first takes a JPEG round trip, so the detector sees
exactly what HADM's own pipeline (frames extracted to JPEG files, read back with
``read_image(format="BGR")``) gives it.  Preprocessing replicates
``DefaultInferencer``: ``ResizeShortestEdge(1024, max_size=1024)`` on the uint8
image (PIL bilinear), float32 CHW, original height/width for box rescaling,
model-side normalisation and square padding to 1024.  Detections keep HADM's
``output.json`` schema ``{frame_%06d.jpg: {bbox, score, class}}``.

Speed knobs (all optional, see :class:`HadmOptions`):

* ``batch_size``: images per forward pass (frames of a clip share one size).
* ``attention="sdpa"``: ``F.scaled_dot_product_attention`` instead of the
  backbone's explicit ``softmax(q k^T) v`` (which materialises a
  16 x 4096 x 4096 matrix per image in every global block).  Same maths.
* ``precision``: ``fp32`` (torch defaults, as HADM), ``tf32`` (TF32
  matmuls), ``bf16`` / ``fp16`` (autocast).  Autocast covers the ViT backbone
  and its feature pyramid only; the RPN and cascade heads run in fp32, as in
  HADM's AMP training loop (``autocast_scope="model"`` autocasts everything).
* ``jpeg_roundtrip``: re-encode every frame like ``cv2.imwrite`` and decode it
  with PIL like ``read_image`` (bit-identical to the original JPEG frame extraction).
* ``input_format``: ``"BGR"`` (default) reproduces HADM's inference, which feeds
  BGR to a detector trained on RGB; ``"RGB"`` is the training-consistent order
  (changes the detections; HumanScore uses ``"BGR"``).

Command line (the same conventions as the other ``hmb.*_engine`` modules)::

    python -m hmb.hadm_engine --jobs jobs.json [--force]

``jobs.json = {"options": {...HadmOptions...}, "jobs": [{"session_dir", "stem",
"video"?, "frames_dir"?, "output_dir"?}]}``.  For a session ``S`` the planned
frames come from ``S/input/frames.json`` (:class:`hmb.video.FramePlan`; all
frames at container rate when absent) and the outputs are
``S/cache/multi_limbs/<stem>/hadm_results/{output.json, frame_indices.json}``
with keys numbered by plan position (0..T-1).  Per-job status and timing go to
``S/cache/engine_status/hadm.json``.  Jobs whose outputs already match the plan
are skipped unless ``--force``; a failing job never stops the others.

Python API::

    engine = HadmEngine(HadmOptions(batch_size=8, precision="bf16")).load()
    status = engine.process({"session_dir": S, "stem": stem})
    records, stats = engine.detect(frames_bgr)   # in-memory frames
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import queue
import sys
import threading
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from time import perf_counter
from typing import Callable, Iterable, Iterator, Optional, Sequence

import numpy as np

from hmb.hadm_prepare import (DEFAULT_CHECKPOINT, DEFAULT_CONFIG, DEFAULT_SCORE_THRESH,
                              default_hadm_root, detections_from_instances, list_frames,
                              load_hadm_model)

ENGINE_NAME = "hadm"
PRECISIONS = ("fp32", "tf32", "bf16", "fp16")
ATTENTIONS = ("sdpa", "eager")
AUTOCAST_SCOPES = ("backbone", "model")
# HADM's DefaultInferencer feeds BGR although the model was trained on RGB
# (coco_loader_lsj: mapper.image_format="RGB"); "BGR" matches the paper.
INPUT_FORMATS = ("BGR", "RGB")
# "pil": detectron2 ResizeTransform on the CPU; "gpu": bit-exact replica of
# Pillow's fixed-point bilinear resampler (see PilBilinearResizer) on the GPU.
RESIZERS = ("gpu", "pil")
# Classes scored by metrics/multi-limbs/eval_extra_limbs.py (EXTRA_CLASSES).
EXTRA_CLASSES = frozenset({"human with extra hand", "human with extra arm",
                           "human with extra leg", "human with extra feet"})
OUTPUT_NAME = "output.json"
INDEX_NAME = "frame_indices.json"


@dataclass
class HadmOptions:
    """Engine options (each is also a command-line flag, e.g. ``--precision fp32``)."""

    hadm_root: str = ""                    # HADM source tree ("" = hmb.hadm_prepare.default_hadm_root())
    config_file: str = DEFAULT_CONFIG      # relative to hadm_root or absolute
    checkpoint: str = DEFAULT_CHECKPOINT   # idem
    score_thresh: float = DEFAULT_SCORE_THRESH
    # The defaults below are the recommended settings: they keep the detections within the
    # run-to-run noise of HADM's own inference (REFERENCE_OPTIONS).  The JPEG round trip is
    # needed for that: the detector is sensitive to JPEG compression of its input.
    batch_size: int = 8                    # frames per forward pass
    precision: str = "fp16"                # see PRECISIONS and the module docstring
    attention: str = "sdpa"                # sdpa (fused kernels) | eager (HADM's explicit softmax)
    autocast_scope: str = "backbone"       # backbone (as HADM's AMP training) | model (everything)
    jpeg_roundtrip: bool = True            # re-encode frames as JPEG first (see the module docstring)
    jpeg_quality: int = 95                 # cv2.imwrite's default quality
    input_format: str = "BGR"              # BGR (as HADM's inference) | RGB
    resize: str = "gpu"                    # gpu (bit-exact Pillow replica) | pil (CPU)
    cudnn_benchmark: bool = False          # let cuDNN pick kernels by timing
    cpu_workers: int = 2                   # threads for JPEG round trip and resizing
    prefetch_batches: int = 2              # prepared batches queued ahead of the GPU
    fast_init: bool = True                 # build the model directly on the GPU (see load_hadm_model)
    device: str = "cuda"

    @classmethod
    def from_dict(cls, values: Optional[dict] = None, **overrides) -> "HadmOptions":
        """Options from a dict plus keyword overrides (None = keep); unknown keys raise."""
        merged = dict(values or {})
        merged.update({k: v for k, v in overrides.items() if v is not None})
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(merged) - known)
        if unknown:
            raise ValueError(f"Unknown HADM engine option(s): {', '.join(unknown)}")
        return cls(**merged).validated()

    def validated(self) -> "HadmOptions":
        """Fill the default HADM root, check every value and cast the integers; returns self."""
        if not self.hadm_root:
            self.hadm_root = str(default_hadm_root())
        if self.precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {PRECISIONS}")
        if self.attention not in ATTENTIONS:
            raise ValueError(f"attention must be one of {ATTENTIONS}")
        if self.autocast_scope not in AUTOCAST_SCOPES:
            raise ValueError(f"autocast_scope must be one of {AUTOCAST_SCOPES}")
        if int(self.batch_size) < 1 or int(self.cpu_workers) < 1 or int(self.prefetch_batches) < 1:
            raise ValueError("batch_size, cpu_workers and prefetch_batches must be >= 1")
        if self.resize not in RESIZERS:
            raise ValueError(f"resize must be one of {RESIZERS}")
        if self.input_format not in INPUT_FORMATS:
            raise ValueError(f"input_format must be one of {INPUT_FORMATS}")
        if not 1 <= int(self.jpeg_quality) <= 100:
            raise ValueError("jpeg_quality must be in [1, 100]")
        if not math.isfinite(self.score_thresh) or not 0 <= self.score_thresh <= 1:
            raise ValueError("score_thresh must be finite and in [0, 1]")
        self.batch_size, self.cpu_workers = int(self.batch_size), int(self.cpu_workers)
        self.prefetch_batches, self.jpeg_quality = int(self.prefetch_batches), int(self.jpeg_quality)
        return self

    def as_dict(self) -> dict:
        return asdict(self)

    def detection_settings(self) -> dict:
        """Options that can change detections (stored next to the outputs)."""
        keys = ("config_file", "checkpoint", "score_thresh", "batch_size", "precision",
                "attention", "autocast_scope", "jpeg_roundtrip", "jpeg_quality", "input_format",
                "resize", "cudnn_benchmark")
        return {k: getattr(self, k) for k in keys}


# Settings that reproduce HADM's own inference (hmb/hadm_prepare.py: per frame, fp32, eager attention).
REFERENCE_OPTIONS = {"batch_size": 1, "precision": "fp32", "attention": "eager", "jpeg_roundtrip": True,
                     "resize": "pil"}


# ----------------------------------------------------------------------------- frames
def read_image_bgr(path) -> np.ndarray:
    """``detectron2.data.detection_utils.read_image(path, format="BGR")``."""
    from detectron2.data.detection_utils import read_image
    return read_image(str(path), format="BGR")


def jpeg_roundtrip(frame_bgr: np.ndarray, quality: int = 95) -> np.ndarray:
    """Encode like ``cv2.imwrite(frame.jpg)`` and decode like ``read_image(BGR)``."""
    import cv2
    from PIL import Image
    from detectron2.data.detection_utils import _apply_exif_orientation, convert_PIL_to_numpy

    params = [] if quality == 95 else [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    ok, buffer = cv2.imencode(".jpg", frame_bgr, params)
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    with Image.open(io.BytesIO(buffer.tobytes())) as image:
        image = _apply_exif_orientation(image)
        return convert_PIL_to_numpy(image, "BGR")


@dataclass
class FrameSource:
    """Frames of one job plus their provenance (written to frame_indices.json)."""

    names: Optional[list]          # None: frame_%06d.jpg by position
    frames: Callable[[], Iterator[np.ndarray]]
    num_frames: Optional[int]
    provenance: dict
    encoded: bool = False          # frames were read from JPEG files: never re-encode


def video_source(video, plan=None) -> FrameSource:
    """Frames of ``video``: the planned frames of ``plan``, or all frames when it is None."""
    from hmb.video import iter_frames

    video = Path(video)
    if plan is None:
        import cv2
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video {video}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        cap.release()
        provenance = {"video": str(video), "planned": False, "source_fps": fps, "sample_fps": None,
                      "max_side": None}

        def frames():
            for _, frame in iter_frames(video, None, rgb=False):
                yield frame
        return FrameSource(None, frames, None, provenance)
    provenance = {"video": str(video), "planned": True, "source_fps": plan.source_fps,
                  "sample_fps": plan.sample_fps, "max_side": plan.max_side,
                  "num_source_frames": plan.num_source_frames,
                  "source_indices": list(plan.indices), "timestamps": list(plan.timestamps)}

    def planned():
        for _, frame in iter_frames(video, plan.indices, rgb=False, max_side=plan.max_side):
            yield frame
    return FrameSource(None, planned, plan.num_frames, provenance)


def frames_dir_source(frames_dir) -> FrameSource:
    """Frames read from the image files of ``frames_dir`` (already JPEG: never re-encoded)."""
    paths = list_frames(frames_dir)
    return FrameSource([p.name for p in paths], lambda: (read_image_bgr(p) for p in paths),
                       len(paths), {"frames_dir": str(Path(frames_dir).resolve()), "planned": False},
                       encoded=True)


# ----------------------------------------------------------------------------- resize
PIL_PRECISION_BITS = 32 - 8 - 2  # Pillow libImaging/Resample.c


def pil_bilinear_taps(in_size: int, out_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Pillow ``precompute_coeffs`` + ``normalize_coeffs_8bpc`` for BILINEAR.

    Returns ``(index, weight)`` of shape ``(out_size, ksize)``: source indices and
    22-bit fixed-point integer weights (unused taps have weight 0).
    """
    scale = filterscale = in_size / out_size
    if filterscale < 1.0:
        filterscale = 1.0
    support = 1.0 * filterscale
    ksize = int(math.ceil(support)) * 2 + 1
    index = np.zeros((out_size, ksize), np.int64)
    weight = np.zeros((out_size, ksize), np.int64)
    ss = 1.0 / filterscale
    for xx in range(out_size):
        center = (xx + 0.5) * scale
        xmin = max(int(center - support + 0.5), 0)
        xmax = min(int(center + support + 0.5), in_size) - xmin
        k = []
        for x in range(xmax):
            t = abs((x + xmin - center + 0.5) * ss)
            k.append(1.0 - t if t < 1.0 else 0.0)
        ww = sum(k)
        if ww != 0.0:
            k = [w / ww for w in k]
        for x, w in enumerate(k):
            index[xx, x] = xmin + x
            weight[xx, x] = int(-0.5 + w * (1 << PIL_PRECISION_BITS)) if w < 0 else \
                int(0.5 + w * (1 << PIL_PRECISION_BITS))
        index[xx, xmax:] = xmin
    return index, weight


class PilBilinearResizer:
    """Bit-exact torch replica of ``PIL.Image.resize(size, BILINEAR)`` for uint8 HWC.

    Pillow resamples horizontally then vertically, accumulating
    ``2**21 + sum(pixel * w_int)`` in integers and clipping ``>> 22`` to uint8
    after each pass; the same integer arithmetic is done here with gathers.
    """

    def __init__(self, device="cuda"):
        self.device = device
        self._taps: dict = {}

    def taps(self, in_size: int, out_size: int):
        """Source indices and weights for one axis (cached per size pair, on the device)."""
        key = (in_size, out_size)
        if key not in self._taps:
            import torch
            index, weight = pil_bilinear_taps(in_size, out_size)
            self._taps[key] = (torch.as_tensor(index, device=self.device),
                               torch.as_tensor(weight, dtype=torch.int32, device=self.device))
        return self._taps[key]

    @staticmethod
    def _clip8(acc):
        return (acc >> PIL_PRECISION_BITS).clamp_(0, 255)

    def __call__(self, image, out_h: int, out_w: int):
        """``image``: (H, W, C) uint8 tensor; returns (out_h, out_w, C) uint8."""
        import torch
        h, w = image.shape[:2]
        x = image.to(torch.int32)
        rounding = 1 << (PIL_PRECISION_BITS - 1)
        if out_w != w:
            index, weight = self.taps(w, out_w)
            acc = (x[:, index, :] * weight[None, :, :, None]).sum(2, dtype=torch.int32) + rounding
            x = self._clip8(acc)
        if out_h != h:
            index, weight = self.taps(h, out_h)
            acc = (x[index, :, :] * weight[:, :, None, None]).sum(1, dtype=torch.int32) + rounding
            x = self._clip8(acc)
        return x.to(torch.uint8)


# ----------------------------------------------------------------------------- attention
def sdpa_attention_forward(self, x):
    """EVA-02 ``Attention.forward`` with fused scaled-dot-product attention.

    Identical projections, rotary embedding and scale as the upstream eager
    branch; only ``softmax(q*scale @ k^T) @ v`` is computed by
    ``F.scaled_dot_product_attention`` (flash / memory-efficient kernels).
    """
    import torch.nn.functional as F

    B, H, W, C = x.shape
    x = x.view(B, -1, C)
    N = H * W
    q = F.linear(input=x, weight=self.q_proj.weight, bias=self.q_bias)
    k = F.linear(input=x, weight=self.k_proj.weight, bias=None)
    v = F.linear(input=x, weight=self.v_proj.weight, bias=self.v_bias)
    q = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    k = k.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    v = v.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    q = self.rope(q).type_as(v)
    k = self.rope(k).type_as(v)
    x = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
    x = x.transpose(1, 2).reshape(B, N, -1)
    x = self.proj(x)
    return x.view(B, H, W, C)


def set_attention(model, mode: str) -> int:
    """Patch (instance level) every backbone attention block; returns the count."""
    import types

    blocks = model.backbone.net.blocks
    for block in blocks:
        attn = block.attn
        if mode == "sdpa":
            if getattr(attn, "xattn", False):
                continue  # xformers kernel already fused
            attn.forward = types.MethodType(sdpa_attention_forward, attn)
        elif "forward" in vars(attn):
            del attn.forward
    return len(blocks)


# ----------------------------------------------------------------------------- outputs
def max_extra_limb_confidence(record: dict) -> float:
    """Highest extra-limb confidence of one frame (``M_t`` of the extra-limbs score), 0 if none."""
    best = 0.0
    for score, cls in zip(record.get("score", []), record.get("class", [])):
        if cls in EXTRA_CLASSES:
            best = max(best, float(score))
    return best


def _write_json(path: Path, payload) -> Path:
    """Write ``payload`` atomically (a temporary file, then rename); returns ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(path)
    return path


@dataclass
class JobPaths:
    """Where one job's detections and status are written."""

    session_dir: Optional[Path]
    stem: str
    output_dir: Path
    status: Path

    @property
    def output_json(self) -> Path:
        return self.output_dir / OUTPUT_NAME

    @property
    def frame_index(self) -> Path:
        return self.output_dir / INDEX_NAME


def job_paths(job: dict) -> JobPaths:
    """Output locations of a job: ``<S>/cache/multi_limbs/<stem>/hadm_results`` unless ``output_dir`` is given."""
    session = Path(job["session_dir"]).expanduser().resolve() if job.get("session_dir") else None
    stem = job.get("stem")
    if not stem:
        source = job.get("video") or job.get("frames_dir")
        if not source:
            raise ValueError("HADM job needs 'stem' or 'video'")
        stem = Path(source).stem if job.get("video") else Path(source).name
    if job.get("output_dir"):
        output_dir = Path(job["output_dir"]).expanduser().resolve()
    elif session is not None:
        output_dir = session / "cache" / "multi_limbs" / stem / "hadm_results"
    else:
        raise ValueError("HADM job needs 'session_dir' or 'output_dir'")
    status = (session / "cache" / "engine_status" / f"{ENGINE_NAME}.json" if session is not None
              else output_dir / f"engine_status_{ENGINE_NAME}.json")
    return JobPaths(session, stem, output_dir, status)


def job_source(job: dict, paths: JobPaths) -> FrameSource:
    """The job's frames: ``frames_dir``, else its video with the session's (or ``frames_json``) plan."""
    from hmb.video import FramePlan, PLAN_NAME

    if job.get("frames_dir"):
        return frames_dir_source(job["frames_dir"])
    plan_path = Path(job["frames_json"]) if job.get("frames_json") else (
        paths.session_dir / "input" / PLAN_NAME if paths.session_dir else None)
    plan = FramePlan.load(plan_path) if plan_path is not None and plan_path.is_file() else None
    video = job.get("video")
    if not video and paths.session_dir is not None:
        candidate = paths.session_dir / "input" / f"{paths.stem}.mp4"
        video = candidate if candidate.exists() else None
    if not video and plan is not None:
        video = plan.video
    if not video:
        raise FileNotFoundError(f"No video for HADM job {paths.stem}")
    if not Path(video).exists():
        raise FileNotFoundError(video)
    return video_source(video, plan)


def outputs_current(paths: JobPaths, source: FrameSource) -> bool:
    """True when output.json exists and covers exactly the planned frames."""
    if not paths.output_json.is_file():
        return False
    index = None
    if paths.frame_index.is_file():
        try:
            index = json.loads(paths.frame_index.read_text())
        except (OSError, ValueError):
            return False
    planned = source.provenance.get("planned")
    if index is None:
        if source.num_frames is None:
            return True  # all-frames output without a frame index, nothing to compare against
        try:
            return len(json.loads(paths.output_json.read_text())) == source.num_frames
        except (OSError, ValueError):
            return False
    if planned:
        return index.get("source_indices") == source.provenance.get("source_indices")
    if source.names is not None:
        return index.get("frame_names") == source.names
    return not index.get("planned", False)


# ----------------------------------------------------------------------------- engine
class _Failure:
    """An exception of the producer thread, passed to the consumer to be re-raised there."""

    def __init__(self, exc: BaseException):
        self.exc = exc


_DONE = object()  # end-of-frames marker on the batch queue


class HadmEngine:
    """Resident HADM detector; ``load()`` once, then ``detect``/``process``/``run``."""

    # Options fixed by load(); changing them needs a new engine.
    MODEL_OPTIONS = ("hadm_root", "config_file", "checkpoint", "score_thresh", "device", "fast_init")

    def __init__(self, options: Optional[HadmOptions] = None, **overrides):
        self.options = HadmOptions.from_dict(asdict(options) if options else None, **overrides)
        self.loaded = None
        self.load_timings: dict = {}

    # -- model -------------------------------------------------------------------------
    def load(self) -> "HadmEngine":
        """Load the detector (once) and apply the attention and cuDNN options."""
        if self.loaded is not None:
            return self
        o = self.options
        started = perf_counter()
        self.loaded = load_hadm_model(o.hadm_root, o.config_file, o.checkpoint, o.score_thresh,
                                      device=o.device, fast_init=o.fast_init)
        import torch
        from detectron2.data import transforms as T

        self.torch = torch
        self.model = self.loaded.model
        self.class_names = self.loaded.class_names
        # DefaultInferencer.__init__ (detectron2/engine/defaults.py)
        size = self.loaded.cfg.model.backbone.net.img_size
        self.aug = T.ResizeShortestEdge(short_edge_length=size, max_size=size)
        self.resizer = PilBilinearResizer(o.device)
        set_attention(self.model, o.attention)
        if o.cudnn_benchmark:
            torch.backends.cudnn.benchmark = True
        self.load_timings = dict(self.loaded.timings, engine_load_seconds=perf_counter() - started)
        return self

    def configure(self, **overrides) -> "HadmEngine":
        """Change runtime options (batch, precision, attention, JPEG ...) without reloading."""
        changed = {k for k, v in overrides.items() if v is not None and v != getattr(self.options, k, v)}
        if self.loaded is not None and changed & set(self.MODEL_OPTIONS):
            raise ValueError(f"Options {sorted(changed & set(self.MODEL_OPTIONS))} need a new engine")
        self.options = HadmOptions.from_dict(self.options.as_dict(), **overrides)
        if self.loaded is not None:
            set_attention(self.model, self.options.attention)
            if self.options.cudnn_benchmark:
                self.torch.backends.cudnn.benchmark = True
        return self

    def _autocast(self, enabled: bool):
        """Autocast context for bf16/fp16 when ``enabled``, else a no-op."""
        precision = self.options.precision
        if enabled and precision in ("bf16", "fp16"):
            dtype = self.torch.bfloat16 if precision == "bf16" else self.torch.float16
            return self.torch.autocast("cuda", dtype=dtype)
        return contextlib.nullcontext()

    @contextlib.contextmanager
    def _matmul_precision(self):
        """Allow TF32 matmuls while the block runs (precision ``tf32`` only); restores the previous state."""
        backends = self.torch.backends
        if self.options.precision != "tf32":
            yield
            return
        previous = backends.cuda.matmul.allow_tf32, backends.cudnn.allow_tf32
        backends.cuda.matmul.allow_tf32 = backends.cudnn.allow_tf32 = True
        try:
            yield
        finally:
            backends.cuda.matmul.allow_tf32, backends.cudnn.allow_tf32 = previous

    def prepare(self, frame_bgr: np.ndarray, jpeg: Optional[bool] = None) -> tuple[np.ndarray, int, int]:
        """CPU part of DefaultInferencer: optional JPEG round trip, then resize."""
        if self.options.jpeg_roundtrip if jpeg is None else jpeg:
            frame_bgr = jpeg_roundtrip(frame_bgr, self.options.jpeg_quality)
        if self.options.input_format == "RGB":
            frame_bgr = frame_bgr[:, :, ::-1]
        height, width = frame_bgr.shape[:2]
        if self.options.resize == "gpu":  # resized in _inputs
            return np.ascontiguousarray(frame_bgr), height, width
        image = self.aug.get_transform(frame_bgr).apply_image(frame_bgr)
        return np.ascontiguousarray(image), height, width

    def _resized_shape(self, height: int, width: int) -> tuple[int, int]:
        """``(height, width)`` after ``ResizeShortestEdge``."""
        size = self.aug.short_edge_length[0]
        return self.aug.get_output_shape(height, width, size, self.aug.max_size)

    def _inputs(self, batch: Sequence[tuple[np.ndarray, int, int]]) -> list[dict]:
        """Model inputs ``{"image": float CHW tensor, "height", "width"}`` (resized on the GPU if chosen)."""
        torch, device = self.torch, self.options.device
        if self.options.resize == "gpu":
            tensors = []
            for image, h, w in batch:
                new_h, new_w = self._resized_shape(h, w)
                gpu = torch.from_numpy(image).to(device, non_blocking=False)
                tensors.append(self.resizer(gpu, new_h, new_w).permute(2, 0, 1).float().contiguous())
            return [{"image": t, "height": h, "width": w} for t, (_, h, w) in zip(tensors, batch)]
        shapes = {image.shape for image, _, _ in batch}
        if len(shapes) == 1:
            stacked = torch.from_numpy(np.stack([image for image, _, _ in batch])).to(device)
            tensors = list(stacked.permute(0, 3, 1, 2).float().contiguous())
        else:
            tensors = [torch.from_numpy(image).to(device).permute(2, 0, 1).float().contiguous()
                       for image, _, _ in batch]
        return [{"image": t, "height": h, "width": w} for t, (_, h, w) in zip(tensors, batch)]

    def forward(self, inputs: list[dict]) -> list[dict]:
        """GeneralizedRCNN.inference with optional backbone autocast."""
        from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN

        model = self.model
        amp = self.options.precision in ("bf16", "fp16")
        with self.torch.inference_mode(), self._matmul_precision():
            images = model.preprocess_image(inputs)
            with self._autocast(True):
                features = model.backbone(images.tensor)
            if amp:
                features = {k: v.float() for k, v in features.items()}
            with self._autocast(self.options.autocast_scope == "model"):
                proposals, _ = model.proposal_generator(images, features, None)
                results, _ = model.roi_heads(images, features, proposals, None)
            return GeneralizedRCNN._postprocess(results, inputs, images.image_sizes)

    def detect_batch(self, batch: Sequence[tuple[np.ndarray, int, int]]) -> list[dict]:
        """``output.json`` records of a batch of :meth:`prepare` results."""
        outputs = self.forward(self._inputs(batch))
        return [detections_from_instances(o["instances"], self.class_names) for o in outputs]

    # -- pipelined detection -----------------------------------------------------------
    def _produce(self, frames: Iterable[np.ndarray], out: queue.Queue, stop: threading.Event,
                 jpeg: Optional[bool] = None):
        """Producer thread: prepare frames in a thread pool and queue them in batches, in order."""
        o = self.options

        def put(item):
            while not stop.is_set():
                try:
                    out.put(item, timeout=0.1)
                    return
                except queue.Full:
                    continue

        try:
            with ThreadPoolExecutor(max_workers=o.cpu_workers, thread_name_prefix="hadm-prep") as pool:
                pending: deque = deque()
                batch: list = []
                limit = o.batch_size * o.prefetch_batches + o.cpu_workers  # frames in flight

                def drain():
                    nonlocal batch
                    batch.append(pending.popleft().result())
                    if len(batch) == o.batch_size:
                        put(batch)
                        batch = []

                for frame in frames:
                    if stop.is_set():
                        return
                    pending.append(pool.submit(self.prepare, frame, jpeg))
                    while len(pending) >= limit:
                        drain()
                while pending:
                    drain()
                if batch:
                    put(batch)
            put(_DONE)
        except BaseException as exc:  # forwarded to the consumer
            put(_Failure(exc))

    def detect(self, frames: Iterable[np.ndarray], num_frames: Optional[int] = None,
               jpeg_roundtrip: Optional[bool] = None):
        """Detections for an iterable of BGR uint8 frames, in order.

        ``jpeg_roundtrip`` overrides the option (``False`` for frames that were
        already read from JPEG files).

        Decoding and CPU preprocessing run in a background thread pool while
        the GPU processes the previous batch.  Returns ``(records, stats)``.
        """
        self.load()
        torch = self.torch
        started = perf_counter()
        stop = threading.Event()
        batches: queue.Queue = queue.Queue(maxsize=self.options.prefetch_batches)
        producer = threading.Thread(target=self._produce, args=(frames, batches, stop, jpeg_roundtrip),
                                    name="hadm-producer", daemon=True)
        records: list = []
        wait = gpu = 0.0
        if str(self.options.device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        producer.start()
        try:
            while True:
                tick = perf_counter()
                item = batches.get()
                wait += perf_counter() - tick
                if item is _DONE:
                    break
                if isinstance(item, _Failure):
                    raise item.exc
                tick = perf_counter()
                records.extend(self.detect_batch(item))
                gpu += perf_counter() - tick
        finally:
            stop.set()
            producer.join(timeout=30)
        seconds = perf_counter() - started
        if num_frames is not None and len(records) != num_frames:
            raise RuntimeError(f"Decoded {len(records)} of {num_frames} planned frames")
        stats = {"frames": len(records), "detect_seconds": seconds, "input_wait_seconds": wait,
                 "model_seconds": gpu, "frames_per_second": len(records) / seconds if seconds else None}
        if str(self.options.device).startswith("cuda"):
            stats["peak_memory_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
        return records, stats

    # -- jobs --------------------------------------------------------------------------
    def process(self, job: dict, force: bool = False) -> dict:
        """Detect one job's frames and write its outputs; never raises (errors are in the status).

        A job whose ``output.json`` already covers exactly its planned frames is skipped unless ``force``.
        """
        started = perf_counter()
        status: dict = {"engine": ENGINE_NAME, "ok": False, "seconds": None, "error": None, "skipped": False}
        paths = None
        try:
            paths = job_paths(job)
            status.update(stem=paths.stem, session_dir=str(paths.session_dir) if paths.session_dir else None,
                          output=str(paths.output_json))
            source = job_source(job, paths)
            if not force and outputs_current(paths, source):
                status.update(ok=True, skipped=True)
            else:
                self.load()
                records, stats = self.detect(source.frames(), source.num_frames,
                                             jpeg_roundtrip=False if source.encoded else None)
                if not records:
                    raise ValueError("No frames decoded")
                names = source.names or [f"frame_{i:06d}.jpg" for i in range(len(records))]
                index = dict(source.provenance, frame_names=names, num_frames=len(records),
                             engine=ENGINE_NAME, settings=self.options.detection_settings())
                if not index.get("planned") and source.names is None:
                    fps = index.get("source_fps") or 0.0
                    index["source_indices"] = list(range(len(records)))
                    index["timestamps"] = [i / fps for i in range(len(records))] if fps > 0 else None
                # Write the index first: output.json is the completion marker.
                _write_json(paths.frame_index, index)
                _write_json(paths.output_json, dict(zip(names, records)))
                # Summary for the status file (0.005 is the paper's per-frame threshold).
                confidences = [max_extra_limb_confidence(r) for r in records]
                status.update(ok=True, **stats, settings=self.options.detection_settings(),
                              model_load=self.load_timings,
                              frames_with_extra_limb_confidence_above_0_005=int(
                                  sum(c > 0.005 for c in confidences)),
                              max_extra_limb_confidence=max(confidences))
        except Exception as exc:
            status["error"] = f"{type(exc).__name__}: {exc}"
            status["traceback"] = traceback.format_exc()
        status["seconds"] = perf_counter() - started
        if paths is not None:
            try:
                _write_json(paths.status, status)
            except OSError as exc:  # pragma: no cover - status is best effort
                status["status_write_error"] = str(exc)
        return status

    def run(self, jobs: Sequence[dict], force: bool = False, log=print) -> list[dict]:
        """:meth:`process` every job in order, logging one line per job; returns their statuses."""
        results = []
        for number, job in enumerate(jobs, 1):
            status = self.process(job, force=force)
            results.append(status)
            if log is not None:
                state = "skip" if status.get("skipped") else ("ok" if status["ok"] else "FAIL")
                extra = (f" {status['frames']} fr {status['frames_per_second']:.1f} fr/s"
                         if status.get("frames") else "")
                log(f"[hadm_engine] {number}/{len(jobs)} {state} {status.get('stem')}"
                    f" {status['seconds']:.1f}s{extra}" + (f" {status['error']}" if status["error"] else ""),
                    flush=True)
        return results


# ----------------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HADM extra-limb detection engine")
    parser.add_argument("--jobs", help="jobs.json ({'options': {...}, 'jobs': [...]})")
    parser.add_argument("--session-dir", help="single job: session directory")
    parser.add_argument("--stem", help="single job: video stem")
    parser.add_argument("--video", help="single job: video path")
    parser.add_argument("--frames-dir", help="single job: directory of extracted frames")
    parser.add_argument("--output-dir", help="single job: output directory override")
    parser.add_argument("--force", action="store_true", help="redo jobs whose detections exist")
    parser.add_argument("--summary", help="write all job statuses to this JSON file")
    # One flag per HadmOptions field, e.g. --precision fp32 or --jpeg-roundtrip false.
    for f in fields(HadmOptions):
        flag = "--" + f.name.replace("_", "-")
        if f.type in ("bool", bool):
            parser.add_argument(flag, dest=f.name, default=None,
                                type=lambda s: s.lower() in ("1", "true", "yes", "on"))
        else:
            cast = {"int": int, "float": float}.get(str(f.type), str)
            parser.add_argument(flag, dest=f.name, default=None, type=cast)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command line (see the module docstring); returns 1 if any job failed."""
    args = build_parser().parse_args(argv)
    spec: dict = {"options": {}, "jobs": []}
    if args.jobs:
        spec = json.loads(Path(args.jobs).read_text())
    if args.session_dir or args.video or args.frames_dir:
        spec.setdefault("jobs", []).append({k: getattr(args, k) for k in
                                            ("session_dir", "stem", "video", "frames_dir", "output_dir")
                                            if getattr(args, k)})
    if not spec.get("jobs"):
        raise SystemExit("No HADM jobs given (use --jobs or --session-dir/--video/--frames-dir)")
    options = dict(spec.get("options") or {})
    overrides = {f.name: getattr(args, f.name) for f in fields(HadmOptions)}
    engine = HadmEngine(HadmOptions.from_dict(options, **overrides))
    results = engine.run(spec["jobs"], force=args.force)
    if args.summary:
        _write_json(Path(args.summary), {"options": engine.options.as_dict(),
                                         "model_load": engine.load_timings, "jobs": results})
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
