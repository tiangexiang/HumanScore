"""GVHMR body-mesh reconstruction engine: the input of the self-collision score.

Pipeline position (see ``hmb.batch``): runs next to ``hmb.detect_engine`` and
``hmb.hadm_engine``, directly on the video.  One long-lived process loads the
four GVHMR networks once (YOLOv8x person detector, ViTPose-H 2-D pose, HMR2
image features, GVHMR) and reconstructs the SMPL-X body of one person in the
planned frames of any number of videos::

    python -m hmb.mesh_engine --jobs jobs.json [--force]
    python -m hmb.mesh_engine --video clip.mp4 --session-dir S [--stem clip] [--sample-fps 24]

``jobs.json`` = ``{"options": {...}, "jobs": [{"session_dir", "stem", "video", ...}]}``;
per-job keys override ``options`` (see ``DEFAULT_OPTIONS``).

Per job the engine decodes exactly the planned frames ``<S>/input/frames.json``
(``hmb.video.FramePlan`` indices and ``max_side``; all frames at container fps if
absent), so the mesh aligns frame-by-frame with every other per-frame artifact
of the session, and writes

* ``<S>/gvhmr/<stem>/hmr4d_results.pt`` -- GVHMR's format (``smpl_params_global``,
  ``smpl_params_incam``, ``K_fullimg``, ``net_outputs``) plus an ``hmb`` dict
  with the frame indices/timestamps, decoded size and the per-frame
  ``tracked`` mask (False where the person box was interpolated);
* ``<S>/gvhmr/<stem>/preprocess/{bbx,vitpose,vit_features}.pt`` (< 1 MB);
* ``<S>/gvhmr/<stem>/mesh_engine.json`` (provenance) and
  ``<S>/cache/engine_status/mesh.json`` (``ok``, ``seconds``, ``error``, stage timings).

Python API (the GVHMR source tree and checkpoints are located by ``hmb.gvhmr_prepare``)::

    engine = MeshEngine({"precision": "tf32"}).load()
    status = engine.process({"session_dir": S, "stem": stem, "video": mp4})
    statuses = engine.run(jobs, force=False)   # pipelined, see below

Each job has two phases: ``prepare`` (decode, batched YOLO detection, GMC +
BoT-SORT tracking, crops; CPU-heavy) and ``infer`` (ViTPose-H, HMR2, GVHMR;
GPU).  ``run`` prepares job k+1 in a worker thread while job k is inferred.

As in the original GVHMR demo, the reconstructed person is the BoT-SORT track
with the largest summed box area, unless the option ``person_boxes`` (the
subject's box track, the JSON format of ``hmb.detect_engine.load_person_boxes``,
e.g. from ``hmb.tracking``) is set: then it is the track that overlaps those
boxes most (``hmb.gvhmr_prepare.select_track_by_prior``), the person the 3-D
keypoints follow as well.

Differences from the original GVHMR demo (``hmb.gvhmr_prepare.reconstruct``) are
limited to (a) frame decoding (OpenCV decode + area downscale of the planned
frames instead of PyAV/ffmpeg scaling of the whole file; ``decoder="upstream"``
restores the latter, for parity tests), (b) batched YOLO detection (BoT-SORT is
still updated frame by frame; its camera-motion (GMC) warps are computed in
parallel with identical results), (c) tunable ViT batch sizes and (d) the ViT
matmul precision.  With ``decoder="upstream", det_batch=1,
pose_batch=feature_batch=16, precision="fp32"`` the output is bitwise identical
to ``hmb.gvhmr_prepare.reconstruct``.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from hmb import video as hvideo
from hmb.gvhmr_prepare import PRECISIONS, REPO_ROOT, GVHMRModels

ENGINE = "mesh"
DEFAULT_OPTIONS: Dict[str, Any] = {
    "camera": "static",       # "static": fixed camera (identity rotation) | "simplevo": estimate the camera
                              # rotation (moving camera; needs the optional pycolmap)
    "f_mm": None,             # full-frame-equivalent focal length in mm; None = GVHMR's default estimate_K
    "precision": "tf32",      # fp32 (as GVHMR) | tf32 | bf16 | fp16 (ViT backbones under autocast)
    "det_batch": 32,          # YOLO detection frames per forward
    "gmc_workers": 3,         # threads for BoT-SORT's sparse-optical-flow GMC (0 = inline, sequential)
    "pose_batch": 64,         # ViTPose crops per forward (x2 with flip test)
    "feature_batch": 64,      # HMR2 crops per forward
    "decoder": "plan",        # "plan" (hmb.video, OpenCV) | "upstream" (PyAV + ffmpeg scale, parity tests only)
    "img_ds": 0.5,            # downscale of the frames the person crops are cut from (as GVHMR)
    "save_preprocess": True,  # also write gvhmr/<stem>/preprocess/*.pt
    "person_boxes": None,     # JSON box track of the subject: reconstruct the track overlapping it (None: largest)
    "person_boxes_sha256": None,  # content digest of person_boxes (recorded so an edited track is recomputed)
}
IMG_DST_SIZE = 256  # side of the square person crop fed to ViTPose and HMR2


# ----------------------------------------------------------------------------- jobs
@dataclass
class Job:
    """One video to reconstruct and where its outputs go (see the module docstring)."""

    session_dir: Path
    stem: str
    video: Path
    options: Dict[str, Any] = field(default_factory=dict)

    @property
    def output_dir(self) -> Path:
        return self.session_dir / "gvhmr" / self.stem

    @property
    def result_path(self) -> Path:
        return self.output_dir / "hmr4d_results.pt"

    @property
    def status_path(self) -> Path:
        return self.session_dir / "cache" / "engine_status" / f"{ENGINE}.json"

    @property
    def plan_path(self) -> Path:
        return self.session_dir / "input" / hvideo.PLAN_NAME


def resolve_options(options: Optional[dict]) -> Dict[str, Any]:
    """``DEFAULT_OPTIONS`` updated with the known keys of ``options``, validated."""
    merged = dict(DEFAULT_OPTIONS)
    merged.update({k: v for k, v in (options or {}).items() if k in DEFAULT_OPTIONS})
    if merged["camera"] not in ("static", "simplevo"):
        raise ValueError(f"camera must be 'static' or 'simplevo', got {merged['camera']!r}")
    if merged["precision"] not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}")
    if merged["decoder"] not in ("plan", "upstream"):
        raise ValueError("decoder must be 'plan' or 'upstream'")
    for key in ("det_batch", "pose_batch", "feature_batch", "gmc_workers"):
        merged[key] = int(merged[key])
        if merged[key] < (0 if key == "gmc_workers" else 1):
            raise ValueError(f"{key} must be positive")
    if merged["f_mm"] is not None and (not math.isfinite(float(merged["f_mm"])) or float(merged["f_mm"]) <= 0):
        raise ValueError("f_mm must be finite and positive")
    if merged["person_boxes"] is not None and not isinstance(merged["person_boxes"], (str, os.PathLike)):
        raise ValueError("person_boxes must be the path of a box track JSON")
    return merged


def job_prior_xyxy(job: Job, decoded: "DecodedVideo") -> Optional[np.ndarray]:
    """The job's ``person_boxes`` as ``(F, 4)`` xyxy boxes of the decoded frames (NaN = unknown), or None."""
    if not job.options["person_boxes"]:
        return None
    from hmb.detect_engine import load_person_boxes, prior_for_frames

    plan = job_plan(job)
    source_size = (int(plan.width), int(plan.height)) if plan is not None else (decoded.width, decoded.height)
    xywh = prior_for_frames(load_person_boxes(Path(job.options["person_boxes"])), decoded.indices,
                            decoded.timestamps, (decoded.width, decoded.height), source_size)
    return np.concatenate([xywh[:, :2], xywh[:, :2] + xywh[:, 2:4]], axis=1)


def make_job(entry: dict, options: Optional[dict] = None) -> Job:
    """A :class:`Job` from one ``jobs.json`` entry; its option keys override ``options``."""
    if "session_dir" not in entry:
        raise ValueError("job needs 'session_dir'")
    session = Path(entry["session_dir"]).expanduser().absolute()
    video = entry.get("video")
    stem = entry.get("stem") or (Path(video).stem if video else None)
    if not stem:
        raise ValueError("job needs 'stem' or 'video'")
    video = Path(video).expanduser().absolute() if video else session / "input" / f"{stem}.mp4"
    merged = dict(options or {})
    merged.update({k: v for k, v in entry.items() if k in DEFAULT_OPTIONS})
    return Job(session, stem, video, resolve_options(merged))


def load_jobs(path: Path) -> tuple[Dict[str, Any], List[dict]]:
    """``(options, jobs)`` of a jobs file (a bare list of jobs is accepted too)."""
    spec = json.loads(Path(path).read_text())
    if isinstance(spec, list):
        spec = {"jobs": spec}
    return dict(spec.get("options") or {}), list(spec.get("jobs") or [])


def write_json(path: Path, payload: dict) -> None:
    """Write ``payload`` atomically (a temporary file, then rename), numpy values included."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(payload, indent=2, default=_json_default))
    tmp.replace(path)


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value))


# --------------------------------------------------------------------------- frames
@dataclass
class DecodedVideo:
    """Planned frames: full-size BGR for YOLO and the downscaled RGB crop source."""
    frames_bgr: List[np.ndarray]
    small_rgb: List[np.ndarray]
    indices: List[int]
    timestamps: List[float]
    width: int
    height: int
    source_fps: float
    plan_source: str
    seconds: float


def job_plan(job: Job) -> Optional[hvideo.FramePlan]:
    """The session's frame plan, or None (then all frames are used)."""
    if job.plan_path.is_file():
        return hvideo.FramePlan.load(job.plan_path)
    return None


def _downscale(frame_rgb: np.ndarray, img_ds: float) -> np.ndarray:
    """Scale a frame by ``img_ds`` (area interpolation)."""
    import cv2
    if img_ds == 1.0:
        return frame_rgb
    height, width = frame_rgb.shape[:2]
    size = (int(width * img_ds), int(height * img_ds))
    return cv2.resize(frame_rgb, size, interpolation=cv2.INTER_AREA)


def decode_job(job: Job) -> DecodedVideo:
    """Decode the planned frames of a job (CPU only).

    Returns the full-size BGR frames (for YOLO) and, scaled by ``img_ds``, the RGB
    frames the person crops are cut from.
    """
    import cv2
    started = perf_counter()
    if not job.video.is_file():
        raise FileNotFoundError(job.video)
    plan = job_plan(job)
    options = job.options
    if plan is not None:
        indices, timestamps, max_side = list(plan.indices), list(plan.timestamps), plan.max_side
        fps, plan_source = float(plan.source_fps), "frames.json"
    else:
        indices, timestamps, max_side, plan_source = None, None, None, "all_frames"
        cap = cv2.VideoCapture(str(job.video))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        cap.release()
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"{job.video} reports an invalid frame rate {fps}")
    frames_bgr, small, decoded_idx = [], [], []
    for index, frame in hvideo.iter_frames(job.video, indices, rgb=False, max_side=max_side):
        frames_bgr.append(frame)
        decoded_idx.append(index)
        if options["decoder"] == "plan":
            small.append(_downscale(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), options["img_ds"]))
    if indices is not None and decoded_idx != [int(i) for i in indices]:
        raise RuntimeError(f"Decoded {len(decoded_idx)} of {len(indices)} planned frames from {job.video}")
    if indices is None:
        indices = decoded_idx
        timestamps = [i / fps for i in decoded_idx]
    if len(frames_bgr) < 2:
        raise ValueError("GVHMR needs at least two frames")
    if options["decoder"] == "upstream":
        if max_side and max(frames_bgr[0].shape[:2]) < max(plan.width, plan.height):
            raise ValueError("decoder='upstream' reproduces full-resolution decoding only (plan.max_side is set)")
        from hmr4d.utils.video_io_utils import read_video_np
        everything = read_video_np(str(job.video), scale=options["img_ds"])
        if everything.shape[0] <= max(indices):
            raise RuntimeError("PyAV decoded fewer frames than OpenCV; use decoder='plan'")
        small = [everything[i] for i in indices]
    height, width = frames_bgr[0].shape[:2]
    return DecodedVideo(frames_bgr, small, [int(i) for i in indices], [float(t) for t in timestamps],
                        int(width), int(height), fps, plan_source, perf_counter() - started)


def crop_frames(small_rgb: List[np.ndarray], bbx_xys, img_ds: float):
    """GVHMR's ``vitfeat_extractor.get_batch`` cropping on pre-decoded, downscaled frames.

    Anti-alias blur when the person box is > 2.2x the 256 px crop, then an
    affine crop (INTER_LINEAR).  Returns ``(crops (F, 256, 256, 3) uint8, crop_bbx_xys (F, 3))``.
    """
    import cv2
    import torch
    from hmr4d.network.hmr2.utils.preproc import crop_and_resize

    gt_center = bbx_xys[:, :2]
    gt_bbx_size = bbx_xys[:, 2]
    ds_factors = ((gt_bbx_size * img_ds * 1.0) / IMG_DST_SIZE / 2.0).numpy()
    crops, boxes = [], []
    for i, (image, d) in enumerate(zip(small_rgb, ds_factors)):
        if d > 1.1:
            image = cv2.GaussianBlur(image, (5, 5), (d - 1) / 2)
        crop, box = crop_and_resize(image, gt_center[i] * img_ds, gt_bbx_size[i] * img_ds, IMG_DST_SIZE,
                                    enlarge_ratio=1.0)
        crops.append(crop)
        boxes.append(box)
    return torch.from_numpy(np.stack(crops)), torch.from_numpy(np.stack(boxes)) / img_ds


def normalize_crops(crops_u8, device: Optional[str] = "cuda"):
    """ImageNet normalization of uint8 crops -> (F, 3, 256, 256) float32 (as GVHMR's get_batch)."""
    from hmr4d.network.hmr2.utils.preproc import IMAGE_MEAN, IMAGE_STD
    if device:
        crops_u8 = crops_u8.to(device, non_blocking=True)
    mean, std = IMAGE_MEAN.to(crops_u8.device), IMAGE_STD.to(crops_u8.device)
    return ((crops_u8 / 255.0 - mean) / std).permute(0, 3, 1, 2)


def make_crops(small_rgb: List[np.ndarray], bbx_xys, img_ds: float, device: Optional[str] = "cuda"):
    """``crop_frames`` + ``normalize_crops`` (normalization on ``device``; None = CPU as GVHMR does)."""
    crops, boxes = crop_frames(small_rgb, bbx_xys, img_ds)
    return normalize_crops(crops, device), boxes


def _camera_rotation_simplevo(frames_bgr, f_mm, step=8, scale=0.5):
    """Camera rotations (F, 3, 3) from GVHMR's SimpleVO (SIFT + pycolmap two-view) on the planned frames."""
    try:
        from hmr4d.utils.preproc.relpose.simple_vo import SimpleVO
    except ImportError as exc:  # pycolmap is optional and not installed by default
        raise RuntimeError("camera='simplevo' requires GVHMR's SimpleVO dependencies (pycolmap)") from exc
    import cv2
    import torch

    class _PlannedSimpleVO(SimpleVO):
        def __init__(self, frames):
            super().__init__(video_path=None, scale=scale, step=step, method="sift", f_mm=f_mm)
            self._frames = frames

        def compute(self):  # SimpleVO.compute, reading in-memory frames instead of the video file
            from hmr4d.utils.preproc.relpose.matcher_wrapper import Matcher
            from hmr4d.utils.preproc.relpose.solver_two_view import (CameraParams, TwoPairSolver,
                                                                     interpolate_missing_frames)
            from hmr4d.utils.preproc.relpose.utils import focal_length_from_mm
            frames = self._frames
            sample = np.arange(0, len(frames), self.step)
            if sample[-1] != len(frames) - 1:
                sample = np.concatenate([sample, [len(frames) - 1]])
            chosen = np.stack([frames[i] for i in sample])
            _, H, W, _ = chosen.shape
            params = CameraParams(W, H, focal_length=focal_length_from_mm(W, H, self.f_mm))
            poses = self.process_video_T_w2c_list_np(chosen, Matcher(self.method), TwoPairSolver(params, solver="pycolmap"))
            return interpolate_missing_frames(poses, sample)

    vo_frames = [cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                 for f in frames_bgr]
    T_w2c = torch.as_tensor(np.asarray(_PlannedSimpleVO(vo_frames).compute())).float()
    return T_w2c[:, :3, :3]


@dataclass
class PreparedJob:
    """Everything up to the ViT crops (CPU + YOLO); frames are not retained."""
    job: Job
    indices: List[int]
    timestamps: List[float]
    width: int
    height: int
    source_fps: float
    plan_source: str
    bbx_xyxy: Any
    bbx_xys: Any
    tracked: Any
    crops: Any          # uint8 (F, 256, 256, 3), or normalized float (F, 3, 256, 256) for decoder='upstream'
    crop_boxes: Any
    camera_rotation: Any
    stages: Dict[str, float]


# --------------------------------------------------------------------------- engine
class MeshEngine:
    """Resident GVHMR models: ``load()`` once, then :meth:`process` or :meth:`run` any number of jobs.

    ``models`` lets callers (tests) share one :class:`hmb.gvhmr_prepare.GVHMRModels`.
    """

    def __init__(self, options: Optional[dict] = None, *, gvhmr_root=None, checkpoints_dir=None,
                 smplx_model_dir=None, models: Optional[GVHMRModels] = None, verbose: bool = True):
        self.options = resolve_options(options)
        self.models = models or GVHMRModels(gvhmr_root, checkpoints_dir, smplx_model_dir)
        self.load_seconds: Dict[str, float] = {}
        self.verbose = verbose
        self._prepare_stream = None

    def log(self, message: str) -> None:
        if self.verbose:
            print(f"[mesh_engine] {message}", flush=True)

    def load(self) -> "MeshEngine":
        """Load the four networks and warm them up."""
        started = perf_counter()
        self.models.setup()
        self.models.load()
        self._warmup()
        self.load_seconds = dict(self.models.load_seconds)
        self.load_seconds["total"] = perf_counter() - started
        self.log("loaded models in " + ", ".join(f"{k} {v:.1f}s" for k, v in self.load_seconds.items()))
        return self

    def _warmup(self) -> None:
        """One tiny forward per network so the first job is not charged for CUDA/cuBLAS init."""
        import torch
        started = perf_counter()
        frames = [np.full((360, 640, 3), 114, dtype=np.uint8) for _ in range(2)]
        self.models.detect_frames(frames, batch_size=2)
        crops = torch.zeros((2, 3, 256, 256), device="cuda")
        boxes = torch.tensor([[128.0, 128.0, 200.0]] * 2)
        precision = self.options["precision"]
        self.models.extract_pose(crops, boxes, batch_size=2, precision=precision)
        self.models.extract_features(crops, batch_size=2, precision=precision)
        torch.cuda.synchronize()
        self.load_seconds["warmup"] = perf_counter() - started

    # ............................................................ two phases
    def prepare(self, job: Job) -> PreparedJob:
        """Decode -> YOLO detection -> GMC -> BoT-SORT -> crops (CPU-heavy; runs in the prefetch thread)."""
        import torch
        from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy
        from hmb.gvhmr_prepare import sparse_optflow_warps

        options = job.options
        decoded = decode_job(job)
        stages = {"decode": decoded.seconds}
        t0 = perf_counter()
        # YOLO runs on its own non-blocking stream: when prepare() runs in the prefetch thread, the
        # main thread's stream syncs (GVHMR IK loop, .cpu()) must not wait for queued YOLO kernels.
        stream = None
        if torch.cuda.is_available():
            if self._prepare_stream is None:
                self._prepare_stream = torch.cuda.Stream()
            stream = self._prepare_stream
        with torch.no_grad(), (torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()):
            detections = self.models.detect_frames(decoded.frames_bgr, batch_size=options["det_batch"])
            if stream is not None:
                stream.synchronize()
        stages["detect"] = perf_counter() - t0
        t0 = perf_counter()
        warps = None
        if options["gmc_workers"] > 0:
            sequence = [i for i, det in enumerate(detections) if len(det)]
            warps = sparse_optflow_warps(decoded.frames_bgr, sequence, workers=options["gmc_workers"])
        prior = job_prior_xyxy(job, decoded)
        if prior is None:
            bbx_xyxy, tracked = self.models.track_detections(detections, decoded.frames_bgr, warps)
        else:  # follow the given subject instead of the largest track
            bbx_xyxy, tracked = self.models.track_detections(detections, decoded.frames_bgr, warps,
                                                             prior_xyxy=prior)
        bbx_xys = get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2).float()
        stages["track"] = perf_counter() - t0
        t0 = perf_counter()
        crops, crop_boxes = crop_frames(decoded.small_rgb, bbx_xys, options["img_ds"])
        if options["decoder"] == "upstream":  # parity: normalize on the CPU exactly like get_batch
            crops = normalize_crops(crops, device=None)
        stages["crops"] = perf_counter() - t0
        length = len(decoded.indices)
        if options["camera"] == "static":
            camera_rotation = torch.eye(3).repeat(length, 1, 1)
        else:
            t0 = perf_counter()
            camera_rotation = _camera_rotation_simplevo(decoded.frames_bgr, options["f_mm"])
            stages["camera"] = perf_counter() - t0
        return PreparedJob(job, decoded.indices, decoded.timestamps, decoded.width, decoded.height,
                           decoded.source_fps, decoded.plan_source, bbx_xyxy, bbx_xys, tracked, crops,
                           crop_boxes, camera_rotation, stages)

    def infer(self, prepared: PreparedJob) -> tuple[dict, dict, Dict[str, float]]:
        """ViTPose -> HMR2 features -> GVHMR (GPU)."""
        import torch
        from hmr4d.utils.geo.hmr_cam import create_camera_sensor, estimate_K

        options = prepared.job.options
        precision = options["precision"]
        stages: Dict[str, float] = {}

        def tick(name, t0):
            torch.cuda.current_stream().synchronize()  # not device-wide: the prefetch thread uses another stream
            stages[name] = perf_counter() - t0

        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            t0 = perf_counter()
            crops = prepared.crops
            crops = normalize_crops(crops, "cuda") if crops.dtype == torch.uint8 else crops
            kp2d = self.models.extract_pose(crops, prepared.crop_boxes, batch_size=options["pose_batch"],
                                            precision=precision)
            tick("vitpose", t0)
            t0 = perf_counter()
            features = self.models.extract_features(crops, batch_size=options["feature_batch"], precision=precision)
            tick("hmr2", t0)
            del crops
            t0 = perf_counter()
            length = len(prepared.indices)
            width, height = prepared.width, prepared.height
            intrinsic = (estimate_K(width, height) if options["f_mm"] is None
                         else create_camera_sensor(width, height, float(options["f_mm"]))[2])
            # The GVHMR network and its post-processing always run in fp32 (they are cheap).
            prediction = self.models.predict(prepared.bbx_xys, kp2d, features, intrinsic.repeat(length, 1, 1),
                                             prepared.camera_rotation, static_cam=options["camera"] == "static")
            tick("gvhmr", t0)
        self.last_gpu_peak_gib = torch.cuda.max_memory_allocated() / 2 ** 30
        preprocess = {"bbx": {"bbx_xyxy": prepared.bbx_xyxy, "bbx_xys": prepared.bbx_xys, "tracked": prepared.tracked},
                      "vitpose": kp2d, "vit_features": features}
        return prediction, preprocess, stages

    # .................................................................. one job
    def process(self, job, force: bool = False, prepared=None) -> dict:
        """Reconstruct one job; never raises (errors are reported in the status).

        ``prepared`` may carry a ``PreparedJob`` (or the exception raised while
        preparing it) produced ahead of time by ``run``.
        """
        started = perf_counter()
        status: Dict[str, Any] = {"engine": ENGINE, "ok": False, "skipped": False, "error": None}
        try:
            if not isinstance(job, Job):
                job = make_job(job, self.options)
            status.update({"session_dir": str(job.session_dir), "stem": job.stem, "video": str(job.video),
                           "output": str(job.result_path), "options": job.options})
            if job.result_path.is_file() and not force:
                status.update({"ok": True, "skipped": True})
                return self._finish(job, status, started)
            prefetched = prepared is not None
            if isinstance(prepared, BaseException):
                raise prepared
            if prepared is None:
                prepared = self.prepare(job)
            prediction, preprocess, infer_stages = self.infer(prepared)
            t0 = perf_counter()
            self._write_outputs(prepared, prediction, preprocess)
            stages = {**prepared.stages, **infer_stages, "write": perf_counter() - t0}
            prepare_seconds = sum(prepared.stages.values())
            status.update({"ok": True, "frames": len(prepared.indices), "width": prepared.width,
                           "height": prepared.height, "source_fps": prepared.source_fps,
                           "plan": prepared.plan_source, "tracked_fraction": float(prepared.tracked.float().mean()),
                           "stages": stages, "prepare_seconds": prepare_seconds,
                           "infer_seconds": sum(infer_stages.values()), "prefetched": prefetched,
                           "gpu_peak_gib": getattr(self, "last_gpu_peak_gib", None)})
        except Exception as exc:  # one failing job must not stop the others
            status.update({"ok": False, "error": f"{type(exc).__name__}: {exc}",
                           "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__, limit=8))})
        status = self._finish(job, status, started)
        if status["ok"] and not status["skipped"] and status.get("prefetched"):
            # wall time charged to this job = its own (overlapped) prepare + infer + write
            status["work_seconds"] = status["prepare_seconds"] + status["seconds"]
        return status

    def _finish(self, job, status: dict, started: float) -> dict:
        """Complete, save and log a job status."""
        status["seconds"] = perf_counter() - started
        status["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        if isinstance(job, Job):
            try:
                write_json(job.status_path, status)
            except OSError as exc:  # pragma: no cover
                status["status_write_error"] = str(exc)
        label = status.get("stem", "?")
        if status["ok"]:
            self.log(f"{label}: {'skipped (exists)' if status['skipped'] else 'ok'} in {status['seconds']:.2f}s"
                     + ("" if status["skipped"] else f" ({status.get('frames')} frames)"))
        else:
            self.log(f"{label}: FAILED {status['error']}")
        return status

    def _write_outputs(self, prepared: PreparedJob, prediction: dict, preprocess: dict) -> None:
        """Write ``hmr4d_results.pt`` (last: it marks completion), ``mesh_engine.json`` and preprocess files."""
        import torch
        job = prepared.job
        out = job.output_dir
        out.mkdir(parents=True, exist_ok=True)
        tracked = prepared.tracked
        prediction["hmb"] = {"engine": "hmb.mesh_engine",
                             "frame_indices": prepared.indices, "timestamps": prepared.timestamps,
                             "width": prepared.width, "height": prepared.height, "source_fps": prepared.source_fps,
                             "tracked": tracked.clone(), "options": dict(job.options)}
        if job.options["save_preprocess"]:
            pre = out / "preprocess"
            pre.mkdir(exist_ok=True)
            torch.save(preprocess["bbx"], pre / "bbx.pt")
            torch.save(preprocess["vitpose"], pre / "vitpose.pt")
            torch.save(preprocess["vit_features"], pre / "vit_features.pt")
        write_json(out / "mesh_engine.json", {"engine": "hmb.mesh_engine",
                                              "video": str(job.video), "plan": prepared.plan_source,
                                              "frames": len(prepared.indices), "width": prepared.width,
                                              "height": prepared.height, "options": job.options,
                                              "tracked_fraction": float(tracked.float().mean())})
        tmp = job.result_path.with_suffix(".pt.part")
        torch.save(prediction, tmp)
        tmp.replace(job.result_path)

    # .................................................................. batch
    def run(self, entries: Iterable, force: bool = False, prefetch: bool = True) -> List[dict]:
        """Process jobs in order; job k+1 is prepared (decode/YOLO/GMC/crops) in a
        worker thread while job k runs ViTPose/HMR2/GVHMR."""
        jobs: List[Any] = []
        for entry in entries:
            try:
                jobs.append(entry if isinstance(entry, Job) else make_job(entry, self.options))
            except Exception as exc:
                jobs.append(exc)

        def needs_work(job):
            return isinstance(job, Job) and (force or not job.result_path.is_file())

        def safe_prepare(job):
            try:
                return self.prepare(job)
            except Exception as exc:
                return exc

        statuses: List[dict] = []
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="mesh-prepare") as pool:
            pending = {}

            def submit(i):
                if prefetch and i < len(jobs) and needs_work(jobs[i]) and i not in pending:
                    pending[i] = pool.submit(safe_prepare, jobs[i])

            submit(0)
            for i, job in enumerate(jobs):
                if isinstance(job, Exception):
                    statuses.append({"engine": ENGINE, "ok": False, "skipped": False,
                                     "error": f"{type(job).__name__}: {job}"})
                    self.log(f"job {i}: invalid ({job})")
                    submit(i + 1)
                    continue
                wait = perf_counter()
                prepared = pending.pop(i).result() if i in pending else None
                wait = perf_counter() - wait
                submit(i + 1)
                status = self.process(job, force=force, prepared=prepared)
                status["prefetch_wait"] = wait
                statuses.append(status)
        return statuses


# ------------------------------------------------------------------------------ CLI
def _single_job(args) -> dict:
    """Job entry of ``--video``/``--session-dir``: links the input video and writes the frame plan."""
    session = Path(args.session_dir).expanduser().absolute()
    video = Path(args.video).expanduser().absolute()
    stem = args.stem or video.stem
    (session / "input").mkdir(parents=True, exist_ok=True)
    link = session / "input" / f"{stem}.mp4"
    if not link.exists() and not link.is_symlink():
        link.symlink_to(video.resolve())
    plan_path = session / "input" / hvideo.PLAN_NAME
    if args.sample_fps is not None or args.max_side is not None:
        plan = hvideo.plan_frames(video, sample_fps=args.sample_fps, max_side=args.max_side)
        if plan_path.is_file():
            existing = hvideo.FramePlan.load(plan_path)
            if existing.indices != plan.indices or existing.max_side != plan.max_side:
                raise SystemExit(f"{plan_path} exists with a different plan; remove it or omit --sample-fps")
        else:
            plan.save(plan_path)
    return {"session_dir": str(session), "stem": stem, "video": str(video)}


def main(argv: Optional[List[str]] = None) -> int:
    """Command line (see the module docstring); returns 1 if any job failed."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jobs", help="jobs.json")
    parser.add_argument("--video", help="single video (with --session-dir)")
    parser.add_argument("--session-dir", help="session directory of the single video")
    parser.add_argument("--stem", help="output name of the single video (default: its file name)")
    parser.add_argument("--sample-fps", type=float, default=None, help="single-video mode: write input/frames.json")
    parser.add_argument("--max-side", type=int, default=None, help="single-video mode: plan max_side")
    parser.add_argument("--force", action="store_true", help="redo jobs whose result exists")
    parser.add_argument("--no-prefetch", action="store_true", help="do not prepare the next job during inference")
    parser.add_argument("--summary", help="write all job statuses to this JSON file")
    # One flag per engine option, e.g. --precision fp32 or --save-preprocess false (see DEFAULT_OPTIONS).
    for key, value in DEFAULT_OPTIONS.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            parser.add_argument(flag, type=lambda s: s.lower() in ("1", "true", "yes"), default=None)
        elif isinstance(value, int):
            parser.add_argument(flag, type=int, default=None)
        elif key == "f_mm" or isinstance(value, float):
            parser.add_argument(flag, type=float, default=None)
        else:
            parser.add_argument(flag, default=None)
    parser.add_argument("--gvhmr-root", default=str(REPO_ROOT / "GVHMR"))
    parser.add_argument("--checkpoints-dir", default=str(REPO_ROOT / "outputs/assets/gvhmr/checkpoints"))
    parser.add_argument("--smplx-model-dir", default=str(REPO_ROOT / "outputs/assets/smplx"))
    args = parser.parse_args(argv)

    if bool(args.jobs) == bool(args.video):
        parser.error("give either --jobs or --video/--session-dir")
    if args.jobs:
        options, entries = load_jobs(Path(args.jobs))
    else:
        if not args.session_dir:
            parser.error("--video needs --session-dir")
        options, entries = {}, [_single_job(args)]
    for key in DEFAULT_OPTIONS:
        if getattr(args, key) is not None:
            options[key] = getattr(args, key)
    started = perf_counter()
    engine = MeshEngine(options, gvhmr_root=args.gvhmr_root, checkpoints_dir=args.checkpoints_dir,
                        smplx_model_dir=args.smplx_model_dir).load()
    statuses = engine.run(entries, force=args.force, prefetch=not args.no_prefetch)
    summary = {"engine": ENGINE, "load_seconds": engine.load_seconds, "wall_seconds": perf_counter() - started,
               "jobs": len(statuses), "ok": sum(bool(s.get("ok")) for s in statuses),
               "failed": sum(not s.get("ok") for s in statuses), "statuses": statuses}
    if args.summary:
        write_json(Path(args.summary), summary)
    print(f"[mesh_engine] {summary['ok']}/{summary['jobs']} ok, wall {summary['wall_seconds']:.1f}s", flush=True)
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
