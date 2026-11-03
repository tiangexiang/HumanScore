"""MeTRAbs 3-D keypoint detection engine: the input of the skeleton fits.

Pipeline position (see ``hmb.batch``): ``plan -> detect -> fit -> score``.  One
long-lived process loads the MeTRAbs detector once (only its detection graph, see
``hmb.metrabs_model``), warms it up once, and then detects the 87 ``bml_movi_87``
body keypoints of one person in the planned frames of any number of videos::

    python -m hmb.detect_engine --jobs jobs.json [--force] [--summary out.json]
    python -m hmb.detect_engine --video clip.mp4 --session-dir S [--stem clip] [--sample-fps 24] [--max-side 1280]

``jobs.json`` = ``{"options": {...}, "jobs": [{"session_dir", "stem", "video", ...}]}``;
per-job keys override ``options`` (see ``DEFAULT_OPTIONS``; ``ENGINE_OPTIONS``
are process-wide and only read from the top-level options).

Input per job: the video and its frame plan ``<S>/input/frames.json``
(:class:`hmb.video.FramePlan`; without a plan, all frames at the container frame
rate).  With the ``sample_fps``/``max_side`` options and no plan file, the plan is
created and saved, so every other engine sees the same frames.  The detector is
called exactly as ``baseline/main.py`` calls it: ``skeleton='bml_movi_87'``, pinhole
intrinsics from the decoded frame size and MeTRAbs' 55 degree field of view (or a
calibration file rescaled to the decoded size), zero distortion and all other
MeTRAbs defaults, on batches of ``batch_size`` frames.

Outputs per job:

* ``<S>/baseline/<stem>_keypoints.npz`` -- the keys of
  ``baseline/main.py:save_metrabs_data`` with identical semantics
  (``keypoints3d`` (T,87,3) mm camera frame, ``keypoints2d`` (T,87,2) px,
  ``boxes`` (T,5) ``[x, y, w, h, score]``, ``confs`` (T,87) = 1 when a person
  was selected and 0 otherwise, ``camera_intrinsics_px`` (1,4)
  ``[fx, fy, cx, cy]`` of the decoded frames, ``camera_distortion`` (1,5),
  ``camera_calibration_source``) plus ``timestamps`` (T,) s, ``frame_indices``
  (T,), ``fps`` (source fps), ``sample_fps`` (NaN = all frames),
  ``image_size``/``source_size`` ``[w, h]``, ``box_confidence`` (T,),
  ``num_detections`` (T,), ``selected_detection`` (T,; -1 = none),
  ``person_selection`` and ``detector_config`` (JSON);
* ``<S>/cache/detect/<stem>_detections.npz`` -- every detection of every
  frame (``counts`` (T,), ``boxes`` (N,5), ``poses3d`` (N,87,3),
  ``poses2d`` (N,87,2)), so the person can be re-selected without the GPU
  (:func:`reselect_session`);
* ``<S>/cache/engine_status/detect.json`` -- ``ok``, ``seconds``, ``error``,
  timings and detection statistics.

Reproducibility: with the defaults (``graph_optimization='lean'``,
``cudnn_autotune=False``, ``batch_size=8``, no padding) the keypoints are
bit-identical to ``baseline/main.py --skip_fitting`` run with
``TF_CUDNN_USE_AUTOTUNE=0``, and identical from run to run.  TensorFlow's default
cuDNN autotuning chooses kernels by timing them, so the keypoints then differ
slightly between runs.  Another ``batch_size`` (or padding) also selects other
cuDNN kernels and changes the fp16 numerics, usually by a few millimetres, but
borderline detections can appear or disappear and ambiguous (mirrored) poses can
flip; keep the default of 8 to compare results.

Person selection (option ``person``):

* ``first``: MeTRAbs' first box in every frame, as ``baseline/main.py`` does.
* ``track`` (what ``hmb.cli`` and ``hmb.batch`` use): seed with the person covering
  the largest box area over the first ``track_init_seconds`` after the first
  detection, then follow that person by maximum IoU with the previously selected
  box (nearest box centre when nothing overlaps; see :func:`select_track`).
  Frames without the person are gaps that do not reset the track.
* ``boxes``: follow an external per-frame box track of the subject (option
  ``person_boxes``, e.g. from ``hmb.tracking``; format in
  :func:`load_person_boxes`).  In each frame the detection with the highest IoU
  with the given box is chosen; a frame without a box, or whose best IoU is below
  ``prior_min_iou``, is a gap (see :func:`select_boxes`).

Python API::

    engine = DetectEngine({"batch_size": 8}).load()      # loads + warms up once
    status = engine.process({"session_dir": S, "stem": stem, "video": mp4})
    statuses = engine.run(jobs, force=False)            # decodes ahead in a thread
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import queue
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from hmb import video as hvideo
from hmb.layout import sanitize_stem
from hmb.metrabs_model import GRAPH_OPTIMIZATION_MODES as GRAPH_MODES

ENGINE = "detect"
NUM_JOINTS = 87  # keypoints of MeTRAbs' bml_movi_87 skeleton
REPO_ROOT = Path(__file__).resolve().parents[1]
PRIOR_MIN_IOU = 0.3  # person='boxes': minimum IoU of the chosen detection with the given box

# Process-wide options (the model is loaded once per engine).
ENGINE_OPTIONS: Dict[str, Any] = {
    "model_path": None,              # MeTRAbs SavedModel dir / hub handle; default $HMB_METRABS_MODEL
    "graph_optimization": "lean",    # full (TF default) | lean (1 Grappler pass, no memory optimizer) | off
    "warmup": True,                  # instantiate the graph at load() with one black batch
    "cudnn_autotune": False,         # False: TF_CUDNN_USE_AUTOTUNE=0 (kernels depend only on input shapes:
                                     # reproducible keypoints, no autotune stalls, same speed). True: TF
                                     # default (kernels chosen by timing; small run-to-run differences).
                                     # None, or the variable already set: leave TensorFlow alone.
}
# Per-job options (job keys override the top-level options).
DEFAULT_OPTIONS: Dict[str, Any] = {
    "batch_size": 8,                 # frames per detect_poses_batched call (as baseline/main.py)
    "person": "first",               # first (as baseline/main.py) | track | boxes (follow ``person_boxes``)
    "person_boxes": None,            # person='boxes': JSON path (or dict) of the subject's per-frame boxes
    "prior_min_iou": PRIOR_MIN_IOU,  # person='boxes': minimum IoU of the chosen detection with the given box
    "skeleton": "bml_movi_87",
    "fov_degrees": 55.0,             # assumed field of view across the larger image side (as baseline/main.py)
    "camera_calibration": None,      # None | JSON path (fx,fy,cx,cy[,distortion])
    "sample_fps": None,              # only used to create <S>/input/frames.json when it is absent
    "max_side": None,                # idem
    "save_detections": True,         # also write <S>/cache/detect/<stem>_detections.npz
    "track_init_seconds": 0.5,       # person='track': seed window after the first detection
    "detector_kwargs": {},           # extra MeTRAbs arguments, e.g. {"suppress_implausible_poses": False}
    "prefetch_batches": 6,           # decoded batches queued ahead of the GPU
    "pad_final_batch": False,        # True: pad the last partial batch with black frames (one input shape,
                                     # so no extra cuDNN autotune for each new remainder size), but this
                                     # slightly changes that batch's keypoints
}
PERSON_MODES = ("first", "track", "boxes")


# ----------------------------------------------------------------------------- jobs
@dataclass
class Job:
    """One video to process and where its outputs go (see the module docstring)."""

    session_dir: Path
    stem: str
    video: Path
    options: Dict[str, Any] = field(default_factory=dict)
    # Optional explicit locations (e.g. ``baseline/main.py`` output directories);
    # ``plan_path`` "" means "no frame plan: all frames".
    overrides: Dict[str, str] = field(default_factory=dict)

    def _path(self, key: str, default: Path) -> Path:
        value = self.overrides.get(key)
        return Path(value) if value else default

    @property
    def keypoints_path(self) -> Path:
        return self._path("keypoints_path", self.session_dir / "baseline" / f"{self.stem}_keypoints.npz")

    @property
    def detections_path(self) -> Path:
        return self._path("detections_path", self.session_dir / "cache" / ENGINE / f"{self.stem}_detections.npz")

    @property
    def status_path(self) -> Path:
        return self._path("status_path", self.session_dir / "cache" / "engine_status" / f"{ENGINE}.json")

    @property
    def plan_path(self) -> Optional[Path]:
        if "plan_path" in self.overrides and not self.overrides["plan_path"]:
            return None
        return self._path("plan_path", self.session_dir / "input" / hvideo.PLAN_NAME)

    @property
    def has_plan(self) -> bool:
        return self.plan_path is not None and self.plan_path.is_file()


PATH_KEYS = ("keypoints_path", "detections_path", "status_path", "plan_path")


def resolve_options(options: Optional[dict]) -> Dict[str, Any]:
    """``DEFAULT_OPTIONS`` updated with the known keys of ``options``, validated and type-cast."""
    merged = dict(DEFAULT_OPTIONS)
    merged.update({k: v for k, v in (options or {}).items() if k in DEFAULT_OPTIONS})
    merged["batch_size"] = int(merged["batch_size"])
    if merged["batch_size"] < 1:
        raise ValueError("batch_size must be positive")
    if merged["person"] not in PERSON_MODES:
        raise ValueError(f"person must be one of {PERSON_MODES}, got {merged['person']!r}")
    if merged["person"] == "boxes" and not merged["person_boxes"]:
        raise ValueError("person='boxes' needs the person_boxes option (a per-frame box track JSON)")
    merged["prior_min_iou"] = float(merged["prior_min_iou"])
    if not 0.0 <= merged["prior_min_iou"] <= 1.0:
        raise ValueError("prior_min_iou must be in [0, 1]")
    merged["fov_degrees"] = float(merged["fov_degrees"])
    if not 0.0 < merged["fov_degrees"] < 180.0:
        raise ValueError("fov_degrees must be in (0, 180)")
    if merged["sample_fps"] is not None:
        merged["sample_fps"] = float(merged["sample_fps"])
    if merged["max_side"] is not None:
        merged["max_side"] = int(merged["max_side"])
    merged["track_init_seconds"] = float(merged["track_init_seconds"])
    merged["prefetch_batches"] = max(1, int(merged["prefetch_batches"]))
    merged["pad_final_batch"] = bool(merged["pad_final_batch"])
    if not isinstance(merged["detector_kwargs"], dict):
        raise ValueError("detector_kwargs must be a dict")
    return merged


def make_job(entry: dict, options: Optional[dict] = None) -> Job:
    """A :class:`Job` from one ``jobs.json`` entry; its option keys override ``options``."""
    if isinstance(entry, Job):
        return entry
    if "session_dir" not in entry:
        raise ValueError("job needs 'session_dir'")
    session = Path(entry["session_dir"]).expanduser().absolute()
    video = entry.get("video")
    stem = entry.get("stem") or (sanitize_stem(Path(video).stem) if video else None)
    if not stem:
        raise ValueError("job needs 'stem' or 'video'")
    video = Path(video).expanduser().absolute() if video else session / "input" / f"{stem}.mp4"
    merged = dict(options or {})
    merged.update({k: v for k, v in entry.items() if k in DEFAULT_OPTIONS})
    overrides = {k: ("" if entry[k] is None else str(entry[k])) for k in PATH_KEYS if k in entry}
    return Job(session, stem, video, resolve_options(merged), overrides)


def load_jobs(path: Path) -> Tuple[Dict[str, Any], List[dict]]:
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
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value))


def _savez_atomic(path: Path, **arrays) -> None:
    """``np.savez`` to a temporary file, then rename, so readers never see a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "wb") as handle:
        np.savez(handle, **arrays)
    tmp.replace(path)


# --------------------------------------------------------------------- camera model
def default_intrinsics(width: int, height: int, fov_degrees: float = 55.0) -> np.ndarray:
    """``[fx, fy, cx, cy]`` exactly as ``baseline/main.py:load_camera_calibration``."""
    focal = max(width, height) / (2.0 * np.tan(np.deg2rad(fov_degrees) / 2.0))
    return np.array([focal, focal, width / 2.0, height / 2.0], dtype=float)


def job_camera(job: Job, source_size: Tuple[int, int], image_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, str]:
    """Pixel intrinsics ``[fx, fy, cx, cy]`` and distortion (5,) for the decoded frames."""
    calibration = job.options["camera_calibration"]
    width, height = image_size
    if not calibration:
        fov = job.options["fov_degrees"]
        return default_intrinsics(width, height, fov), np.zeros(5), f"assumed_fov{fov:g}"
    values = json.loads(Path(calibration).read_text())
    intrinsics = np.array([values[k] for k in ("fx", "fy", "cx", "cy")], dtype=float)
    distortion = np.asarray(values.get("distortion", [0.0] * 5), dtype=float).reshape(-1)
    if distortion.shape != (5,) or not np.all(np.isfinite(distortion)) or not np.all(np.isfinite(intrinsics)) \
            or np.any(intrinsics[:2] <= 0):
        raise ValueError(f"invalid camera calibration {calibration}")
    sx, sy = width / float(source_size[0]), height / float(source_size[1])
    if (sx, sy) != (1.0, 1.0):
        # Calibration refers to the source resolution; rescale to the decoded frames.
        intrinsics = intrinsics * np.array([sx, sy, sx, sy])
    return intrinsics, distortion, str(calibration)


# ------------------------------------------------------------------ person selection
def _box_iou(ref: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """IoU of one ``[x, y, w, h]`` box against (N, >=4) boxes."""
    x1 = np.maximum(ref[0], boxes[:, 0])
    y1 = np.maximum(ref[1], boxes[:, 1])
    x2 = np.minimum(ref[0] + ref[2], boxes[:, 0] + boxes[:, 2])
    y2 = np.minimum(ref[1] + ref[3], boxes[:, 1] + boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    union = ref[2] * ref[3] + boxes[:, 2] * boxes[:, 3] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def select_first(frame_boxes: Sequence[np.ndarray]) -> np.ndarray:
    """``person='first'``: detection 0 of every frame (-1 where nothing was detected)."""
    return np.array([0 if len(b) else -1 for b in frame_boxes], dtype=np.int32)


def _link_area(frame_boxes, ref: np.ndarray, order) -> float:
    """Sum of box areas along a strict max-IoU chain (stops when nothing overlaps)."""
    total = 0.0
    for t in order:
        boxes = np.asarray(frame_boxes[t], dtype=float)
        if not len(boxes):
            continue
        iou = _box_iou(ref, boxes)
        if iou.max() <= 0:
            break
        ref = boxes[int(np.argmax(iou))]
        total += float(ref[2] * ref[3])
    return total


# Gate of person='track' (see associate()).
TRACK_MIN_IOU = 0.2          # overlap that always counts as the same person
TRACK_CENTRE_GATE = 1.0      # max centre motion per frame, in units of sqrt(reference box area)
TRACK_GATE_GROWTH = 0.5      # gate widening per consecutive missed frame
TRACK_AREA_RATIO = 4.0       # accepted box area within [1/r, r] x reference area (non-overlapping case)


def associate(ref: np.ndarray, boxes: np.ndarray, missed: int = 0) -> Optional[int]:
    """Index of the box continuing the track of ``ref``, or None (gated association).

    A box continues the track if it overlaps the reference (IoU >= TRACK_MIN_IOU),
    or if its centre moved less than ``TRACK_CENTRE_GATE * sqrt(area)`` (widened by
    ``TRACK_GATE_GROWTH`` per missed frame) and its area is within a factor
    ``TRACK_AREA_RATIO`` of the reference.  Among candidates the highest IoU wins,
    then the nearest centre.
    """
    iou = _box_iou(ref, boxes)
    centre = ref[:2] + ref[2:4] / 2
    dist = np.linalg.norm(boxes[:, :2] + boxes[:, 2:4] / 2 - centre, axis=1)
    ref_area = max(float(ref[2] * ref[3]), 1e-6)
    area = np.maximum(boxes[:, 2] * boxes[:, 3], 1e-6)
    ratio = np.maximum(area / ref_area, ref_area / area)
    gate = TRACK_CENTRE_GATE * (1.0 + TRACK_GATE_GROWTH * missed) * np.sqrt(ref_area)
    ok = (iou >= TRACK_MIN_IOU) | ((dist <= gate) & (ratio <= TRACK_AREA_RATIO))
    if not ok.any():
        return None
    cand = np.flatnonzero(ok)
    if iou[cand].max() > 0:
        return int(cand[np.argmax(iou[cand])])
    return int(cand[np.argmin(dist[cand])])


def select_track(frame_boxes: Sequence[np.ndarray], timestamps: Sequence[float],
                 init_seconds: float = 0.5) -> np.ndarray:
    """Follow one person: seed = largest person near the start, then max-IoU association.

    Seed: among all boxes in the frames within ``init_seconds`` of the first
    frame that has a detection, the one whose IoU-linked chain through that
    window covers the largest total box area (box area integrated over the
    window, so a large but transient box, e.g. a bystander entering at the
    frame edge, does not beat the persistent main subject; with a single
    frame in the window this is simply the largest box).  Tracking then runs
    forward and backward from the seed; in each frame the box with the
    highest IoU with the most recently selected box is chosen, or the nearest
    box centre if none overlaps, subject to the gate in :func:`associate`.
    Frames without detections, or whose detections all fail the gate (the
    tracked person is missing while someone else is visible), are left
    unselected (-1) and do not reset the reference box.
    """
    count = len(frame_boxes)
    selected = np.full(count, -1, dtype=np.int32)
    detected = [t for t in range(count) if len(frame_boxes[t])]
    if not detected:
        return selected
    times = np.asarray(timestamps, dtype=float)
    window_end = times[detected[0]] + init_seconds
    window = [t for t in detected if times[t] <= window_end + 1e-9]
    best = None
    for position, t in enumerate(window):
        boxes = np.asarray(frame_boxes[t], dtype=float)
        for j, box in enumerate(boxes):
            score = (float(box[2] * box[3]) + _link_area(frame_boxes, box, window[position + 1:])
                     + _link_area(frame_boxes, box, window[:position][::-1]))
            if best is None or score > best[0]:
                best = (score, t, j)
    _, seed, seed_det = best
    selected[seed] = seed_det

    def follow(order):
        ref = np.asarray(frame_boxes[seed][seed_det], dtype=float)
        missed = 0  # consecutive frames since the last accepted association
        for t in order:
            boxes = np.asarray(frame_boxes[t], dtype=float)
            if not len(boxes):
                missed += 1
                continue
            j = associate(ref, boxes, missed)
            if j is None:
                # The tracked person is not among this frame's detections: leave a gap
                # instead of switching to whoever else is visible (e.g. a bystander).
                missed += 1
                continue
            selected[t] = j
            ref = boxes[j]
            missed = 0

    follow(range(seed + 1, count))
    follow(range(seed - 1, -1, -1))
    return selected


def load_person_boxes(source) -> Dict[str, Any]:
    """Read and validate a subject box track (path to JSON, or the parsed dict).

    Format: ``{"timestamps": [t, ...], "boxes_xyxy": [[x0, y0, x1, y1] or null, ...]}``,
    one entry per source frame (seconds and pixels), optionally ``frame_indices``
    (source frame index of each entry; matched exactly instead of by time) and
    ``image_size`` ``[w, h]`` of the box coordinates (default: the source frame size).
    Returns ``{"timestamps" (N,), "boxes" (N, 4) xyxy with NaN rows for null,
    "frame_indices" (N,) or None, "image_size" (w, h) or None}``.
    """
    data = json.loads(Path(source).read_text()) if not isinstance(source, dict) else source
    if not isinstance(data, dict) or "boxes_xyxy" not in data:
        raise ValueError("person_boxes must be a JSON object with 'boxes_xyxy' (and 'timestamps' or 'frame_indices')")
    rows = data["boxes_xyxy"]
    boxes = np.full((len(rows), 4), np.nan)
    for k, box in enumerate(rows):
        if box is None:
            continue
        values = np.asarray(box, dtype=float).reshape(-1)
        if values.shape != (4,) or not np.all(np.isfinite(values)):
            raise ValueError(f"person_boxes: box {k} must be [x0, y0, x1, y1] or null, got {box!r}")
        if values[2] <= values[0] or values[3] <= values[1]:
            continue  # degenerate (empty) box: no prior for this frame
        boxes[k] = values
    out: Dict[str, Any] = {"boxes": boxes, "frame_indices": None, "timestamps": None, "image_size": None}
    for key, cast in (("timestamps", float), ("frame_indices", np.int64)):
        if data.get(key) is not None:
            values = np.asarray(data[key], dtype=cast).reshape(-1)
            if len(values) != len(rows):
                raise ValueError(f"person_boxes: {len(values)} {key} for {len(rows)} boxes")
            if key == "timestamps" and not np.all(np.isfinite(values)):
                raise ValueError("person_boxes: timestamps must be finite")
            out[key] = values
    if out["timestamps"] is None and out["frame_indices"] is None:
        raise ValueError("person_boxes needs 'timestamps' or 'frame_indices'")
    if data.get("image_size") is not None:
        size = [int(v) for v in data["image_size"]]
        if len(size) != 2 or min(size) <= 0:
            raise ValueError("person_boxes: image_size must be [width, height]")
        out["image_size"] = (size[0], size[1])
    return out


def prior_for_frames(prior: Dict[str, Any], frame_indices: Sequence[int], timestamps: Sequence[float],
                     image_size: Tuple[int, int], source_size: Tuple[int, int]) -> np.ndarray:
    """Prior box ``[x, y, w, h]`` (decoded-frame pixels) of every analysed frame; NaN rows = no prior.

    Entries are matched by source frame index when the track has ``frame_indices``,
    otherwise by the nearest timestamp within half the track's median frame interval.
    """
    count = len(frame_indices)
    out = np.full((count, 4), np.nan)
    boxes = prior["boxes"]
    if not len(boxes) or not count:
        return out
    if prior.get("frame_indices") is not None:
        lookup = {int(i): k for k, i in enumerate(prior["frame_indices"])}
        rows = [lookup.get(int(i), -1) for i in frame_indices]
    else:
        times = np.asarray(prior["timestamps"], dtype=float)
        order = np.argsort(times, kind="stable")
        times_sorted = times[order]
        steps = np.diff(times_sorted)
        steps = steps[steps > 0]
        tolerance = 0.5 * float(np.median(steps)) if len(steps) else 0.5 / 24.0
        rows = []
        for t in np.asarray(timestamps, dtype=float):
            pos = int(np.searchsorted(times_sorted, t))
            best = min((p for p in (pos - 1, pos) if 0 <= p < len(times_sorted)),
                       key=lambda p: abs(times_sorted[p] - t))
            rows.append(int(order[best]) if abs(times_sorted[best] - t) <= tolerance + 1e-9 else -1)
    ref_w, ref_h = prior.get("image_size") or source_size
    sx, sy = image_size[0] / float(ref_w), image_size[1] / float(ref_h)
    for t, k in enumerate(rows):
        if k < 0 or not np.all(np.isfinite(boxes[k])):
            continue
        x0, y0, x1, y1 = boxes[k]
        out[t] = [x0 * sx, y0 * sy, (x1 - x0) * sx, (y1 - y0) * sy]
    return out


def select_boxes(frame_boxes: Sequence[np.ndarray], prior_xywh: np.ndarray,
                 min_iou: float = PRIOR_MIN_IOU) -> np.ndarray:
    """Per frame the detection with the highest IoU with the prior box (``person='boxes'``).

    A frame without a prior box (NaN row), without detections, or whose best
    detection overlaps the prior box less than ``min_iou`` is a gap (-1): an
    untracked frame never falls back to another person.
    """
    count = len(frame_boxes)
    if len(prior_xywh) != count:
        raise ValueError(f"{len(prior_xywh)} prior boxes for {count} frames")
    selected = np.full(count, -1, dtype=np.int32)
    for t in range(count):
        ref = np.asarray(prior_xywh[t], dtype=float)
        boxes = np.asarray(frame_boxes[t], dtype=float)
        if not len(boxes) or not np.all(np.isfinite(ref)) or ref[2] <= 0 or ref[3] <= 0:
            continue
        iou = _box_iou(ref, boxes)
        j = int(np.argmax(iou))
        if iou[j] > 0 and iou[j] >= min_iou:
            selected[t] = j
    return selected


def select_person(frame_boxes, timestamps, mode: str, init_seconds: float = 0.5,
                  prior: Optional[np.ndarray] = None, prior_min_iou: float = PRIOR_MIN_IOU) -> np.ndarray:
    """Selected detection index per frame (-1 = none) for ``mode`` first | track | boxes."""
    if mode == "first":
        return select_first(frame_boxes)
    if mode == "track":
        return select_track(frame_boxes, timestamps, init_seconds)
    if mode == "boxes":
        if prior is None:
            raise ValueError("person='boxes' needs per-frame prior boxes")
        return select_boxes(frame_boxes, prior, prior_min_iou)
    raise ValueError(mode)


def _file_sha256(path) -> Optional[str]:
    """SHA-256 of a file's content, or None (e.g. for an inline box track dict)."""
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except (OSError, TypeError):
        return None


def job_prior(options: Dict[str, Any], frame_indices, timestamps, image_size, source_size) -> Optional[np.ndarray]:
    """Prior boxes of a ``person='boxes'`` job aligned with its analysed frames (None otherwise)."""
    if options["person"] != "boxes":
        return None
    return prior_for_frames(load_person_boxes(options["person_boxes"]), frame_indices, timestamps,
                            image_size, source_size)


# ------------------------------------------------------------------------ outputs
@dataclass
class FrameDetections:
    """All MeTRAbs detections of a clip, one array per frame (``n_t`` people in frame t)."""

    boxes: List[np.ndarray]      # per frame (n_t, 5) [x, y, w, h, score]
    poses3d: List[np.ndarray]    # per frame (n_t, 87, 3) mm
    poses2d: List[np.ndarray]    # per frame (n_t, 87, 2) px


def keypoint_arrays(dets: FrameDetections, selected: np.ndarray) -> Dict[str, Any]:
    """Per-frame selection with ``save_metrabs_data`` semantics and dtypes.

    Undetected frames are float64 zeros and detected frames keep MeTRAbs'
    float32 values, then the frame list is stacked as ``np.savez`` does in the
    baseline, so a fully detected clip is float32 and a clip with gaps float64.
    """
    boxes, pose3d, pose2d, confs = [], [], [], []
    for t, j in enumerate(selected):
        if j < 0:
            boxes.append(np.zeros((5)))
            pose3d.append(np.zeros((NUM_JOINTS, 3)))
            pose2d.append(np.zeros((NUM_JOINTS, 2)))
            confs.append(np.zeros((NUM_JOINTS)))
            continue
        boxes.append(dets.boxes[t][j])
        pose3d.append(dets.poses3d[t][j])
        pose2d.append(dets.poses2d[t][j])
        confs.append(np.ones((NUM_JOINTS)))
    if not len(selected):
        raise ValueError("No decodable video frames")
    return {"keypoints3d": np.asarray(pose3d), "keypoints2d": np.asarray(pose2d),
            "boxes": np.asarray(boxes), "confs": np.asarray(confs)}


def detections_arrays(dets: FrameDetections) -> Dict[str, np.ndarray]:
    """Flat arrays of all detections plus per-frame ``counts`` (the ``_detections.npz`` content)."""
    counts = np.array([len(b) for b in dets.boxes], dtype=np.int32)

    def flat(items, shape):
        kept = [np.asarray(x, dtype=np.float32).reshape((-1,) + shape) for x in items]
        return np.concatenate(kept) if kept else np.zeros((0,) + shape, np.float32)

    return {"counts": counts, "boxes": flat(dets.boxes, (5,)),
            "poses3d": flat(dets.poses3d, (NUM_JOINTS, 3)), "poses2d": flat(dets.poses2d, (NUM_JOINTS, 2))}


def load_detections(path: Path) -> Tuple[FrameDetections, Dict[str, np.ndarray]]:
    """Inverse of the cached ``_detections.npz`` (for re-selection without the GPU)."""
    with np.load(path, allow_pickle=False) as data:
        counts = data["counts"]
        splits = np.cumsum(counts)[:-1]
        dets = FrameDetections(list(np.split(data["boxes"], splits)), list(np.split(data["poses3d"], splits)),
                               list(np.split(data["poses2d"], splits)))
        extra = {k: data[k] for k in data.files if k not in ("counts", "boxes", "poses3d", "poses2d")}
    return dets, extra


# ------------------------------------------------------------------------ decoding
@dataclass
class JobFrames:
    """What the decoder thread knows about a job before its first batch."""

    indices: Optional[List[int]]        # planned source frames (None = all frames)
    timestamps: Optional[List[float]]   # their times in seconds (None = index / fps)
    fps: float                          # source frame rate
    sample_fps: Optional[float]
    max_side: Optional[int]
    source_size: Tuple[int, int]        # (width, height) of the source video
    plan_source: str                    # "frames.json" or "all_frames"
    plan_video: Optional[str] = None


def _video_header(video: Path) -> Tuple[float, int, int]:
    """``(fps, width, height)`` from the container header (no decoding)."""
    import cv2
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video {video}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid frame rate {fps!r} for {video}")
    return fps, width, height


def resolve_frames(job: Job) -> JobFrames:
    """Use <S>/input/frames.json; create it from sample_fps/max_side options when absent."""
    if not job.video.is_file():
        raise FileNotFoundError(job.video)
    wants_plan = job.options["sample_fps"] is not None or job.options["max_side"] is not None
    plan = hvideo.FramePlan.load(job.plan_path) if job.has_plan else None
    if plan is None and wants_plan:
        if job.plan_path is None:
            raise ValueError("sample_fps/max_side need a frame plan location (plan_path is disabled)")
        plan = hvideo.plan_frames(job.video, sample_fps=job.options["sample_fps"], max_side=job.options["max_side"])
        if job.has_plan:  # created concurrently by another engine
            existing = hvideo.FramePlan.load(job.plan_path)
            if existing.indices != plan.indices or existing.max_side != plan.max_side:
                raise ValueError(f"{job.plan_path} appeared with a different plan")
            plan = existing
        else:
            plan.save(job.plan_path)
    elif plan is not None and wants_plan:
        requested = hvideo.sample_indices(plan.num_source_frames, plan.source_fps, job.options["sample_fps"])
        if [int(i) for i in requested] != [int(i) for i in plan.indices] or \
                (job.options["max_side"] or None) != (plan.max_side or None):
            raise ValueError(f"{job.plan_path} differs from the requested sample_fps/max_side; "
                             "remove it or drop those options")
    if plan is not None:
        return JobFrames([int(i) for i in plan.indices], [float(t) for t in plan.timestamps],
                         float(plan.source_fps), plan.sample_fps, plan.max_side, (int(plan.width), int(plan.height)),
                         "frames.json", plan.video)
    fps, width, height = _video_header(job.video)
    return JobFrames(None, None, fps, None, None, (width, height), "all_frames")


class _Producer(threading.Thread):
    """Decode planned frames of a list of jobs into fixed-size batches, ahead of the GPU.

    Puts these messages on ``out``, in order, for every job k:
    ``("begin", k, JobFrames)``, ``("batch", k, frames)`` per batch, then
    ``("end", k, info)`` or ``("error", k, message, traceback)``; finally ``("done", None)``.
    """

    def __init__(self, jobs: Sequence[Job], out: "queue.Queue", stop: threading.Event):
        super().__init__(name="detect-engine-decoder", daemon=True)
        self.jobs, self.out, self.stop = jobs, out, stop

    def put(self, item):
        """Queue ``item``; returns False (drop it) once the engine asked the thread to stop."""
        while not self.stop.is_set():
            try:
                self.out.put(item, timeout=0.5)
                return True
            except queue.Full:
                continue
        return False

    def run(self):
        for k, job in enumerate(self.jobs):
            if self.stop.is_set():
                return
            started = perf_counter()
            try:
                frames = resolve_frames(job)
                if not self.put(("begin", k, frames)):
                    return
                batch, decoded = [], []
                size = job.options["batch_size"]
                for index, frame in hvideo.iter_frames(job.video, frames.indices, rgb=True,
                                                       max_side=frames.max_side):
                    decoded.append(index)
                    batch.append(frame)
                    if len(batch) == size:
                        if not self.put(("batch", k, np.stack(batch))):
                            return
                        batch = []
                if batch and not self.put(("batch", k, np.stack(batch))):
                    return
                if frames.indices is not None and decoded != frames.indices:
                    raise RuntimeError(f"Decoded {len(decoded)} of {len(frames.indices)} planned frames from {job.video}")
                if not decoded:
                    raise ValueError(f"No decodable video frames: {job.video}")
                self.put(("end", k, {"indices": decoded, "decode_seconds": perf_counter() - started}))
            except Exception as exc:  # noqa: BLE001 - reported per job
                self.put(("error", k, f"{type(exc).__name__}: {exc}", traceback.format_exc()))
        self.put(("done", None))


# --------------------------------------------------------------------- model layer
def baseline_load_metrabs():
    """``baseline/monocular_demos/utils.py:load_metrabs`` (the loader ``baseline/main.py`` uses).

    Imported by file path: ``metrics/bone_length`` ships a second
    ``monocular_demos`` package, so a bare import could resolve to either.
    """
    import importlib.util
    name = "hmb_baseline_monocular_utils"
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "baseline/monocular_demos/utils.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[name] = module
    return module.load_metrabs


class MetrabsDetector:
    """Thin adapter: fixed-size (padded) batches in, per-frame numpy detections out."""

    def __init__(self, model_path: Optional[str] = None, graph_optimization: str = "lean",
                 cudnn_autotune: Optional[bool] = False):
        self.model_path, self.graph_optimization = model_path, graph_optimization
        self.cudnn_autotune = cudnn_autotune
        self.model = None
        self.timing: Dict[str, Any] = {}
        self._tf = None

    def load(self):
        """Import TensorFlow, configure it (GPU memory growth, cuDNN, Grappler) and load MeTRAbs."""
        started = perf_counter()
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        if self.cudnn_autotune is not None and "TF_CUDNN_USE_AUTOTUNE" not in os.environ:
            # Read once by TensorFlow's cuDNN wrapper at the first convolution of the process.
            os.environ["TF_CUDNN_USE_AUTOTUNE"] = "1" if self.cudnn_autotune else "0"
        import tensorflow as tf
        from hmb.metrabs_model import configure_graph_optimization
        for gpu in tf.config.list_physical_devices("GPU"):
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass  # already initialized (memory growth then is whatever was set before)
        self.graph_optimization = configure_graph_optimization(self.graph_optimization)
        self._tf = tf
        load_metrabs = baseline_load_metrabs()
        self.model = load_metrabs(self.model_path, detection_only=True)
        self.timing = {"import_and_load_seconds": perf_counter() - started,
                       "load_details": getattr(load_metrabs, "last_timing", None),
                       "graph_optimization": self.graph_optimization,
                       "cudnn_autotune": os.environ.get("TF_CUDNN_USE_AUTOTUNE", "default")}
        return self

    def warmup(self, batch_size: int, image_size=(640, 360)):
        """Run one black batch so graph instantiation is not charged to the first video."""
        started = perf_counter()
        width, height = image_size
        frames = np.zeros((batch_size, height, width, 3), np.uint8)
        self.detect(frames, default_intrinsics(width, height), np.zeros(5), batch_size, "bml_movi_87", {})
        self.timing["warmup_seconds"] = perf_counter() - started
        return self.timing["warmup_seconds"]

    def memory_mb(self, reset: bool = False) -> Optional[Dict[str, float]]:
        """TensorFlow allocator use on GPU:0 (current/peak MB since the last reset)."""
        try:
            info = self._tf.config.experimental.get_memory_info("GPU:0")
            if reset:
                self._tf.config.experimental.reset_memory_stats("GPU:0")
            return {"current_mb": info["current"] / 2**20, "peak_mb": info["peak"] / 2**20}
        except Exception:  # noqa: BLE001 - CPU-only runtime
            return None

    def detect(self, frames: np.ndarray, intrinsics: np.ndarray, distortion: np.ndarray, batch_size: int,
               skeleton: str, kwargs: dict) -> Tuple[FrameDetections, int]:
        """Run one batch (padded to ``batch_size``); returns detections of the real frames
        and the number of detections MeTRAbs produced on padding frames (expected 0)."""
        count = len(frames)
        if count < batch_size:
            pad = np.zeros((batch_size - count,) + frames.shape[1:], frames.dtype)
            frames = np.concatenate([frames, pad])
        fx, fy, cx, cy = intrinsics
        intrinsic_matrix = np.array([[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]], dtype=np.float32)
        pred = self.model.detect_poses_batched(
            frames, skeleton=skeleton, intrinsic_matrix=intrinsic_matrix,
            distortion_coeffs=np.asarray(distortion, dtype=np.float32).reshape(1, -1), **kwargs)
        lengths = pred["boxes"].row_lengths().numpy()
        splits = np.cumsum(lengths)[:-1]
        out = []
        for key in ("boxes", "poses3d", "poses2d"):
            values = pred[key].values.numpy() if hasattr(pred[key], "values") else pred[key].flat_values.numpy()
            out.append(np.split(values, splits)[:count])
        return FrameDetections(*out), int(lengths[count:].sum())


# --------------------------------------------------------------------------- engine
class DetectEngine:
    """Resident detector: ``load()`` once, then :meth:`process` or :meth:`run` any number of jobs.

    ``detector`` replaces MeTRAbs (tests pass a fake with the :class:`MetrabsDetector` interface).
    """

    def __init__(self, options: Optional[dict] = None, detector=None, log: Optional[Callable[[str], None]] = None):
        options = dict(options or {})
        self.engine_options = {k: options.get(k, v) for k, v in ENGINE_OPTIONS.items()}
        if self.engine_options["graph_optimization"] not in GRAPH_MODES:
            raise ValueError(f"graph_optimization must be one of {GRAPH_MODES}")
        self.options = resolve_options(options)
        self.detector = detector
        self.log = log or (lambda message: print(f"[detect_engine] {message}", flush=True))
        self.load_info: Dict[str, Any] = {}

    # ............................................................... lifecycle
    def load(self) -> "DetectEngine":
        """Load (and by default warm up) the detector."""
        if self.detector is None:
            self.detector = MetrabsDetector(self.engine_options["model_path"],
                                            self.engine_options["graph_optimization"],
                                            self.engine_options["cudnn_autotune"]).load()
        if self.engine_options["warmup"] and hasattr(self.detector, "warmup"):
            self.detector.warmup(self.options["batch_size"])
        self.load_info = dict(getattr(self.detector, "timing", {}) or {})
        self.log(f"loaded: {json.dumps(self.load_info, default=_json_default)}")
        return self

    # ................................................................ helpers
    def needs_work(self, job: Job, force: bool) -> Tuple[bool, str]:
        """``(needed, reason)``: redo a job unless its keypoints exist and match the frame plan."""
        if force or not job.keypoints_path.is_file():
            return True, "forced" if force else "missing"
        if not job.has_plan and (job.options["sample_fps"] is not None or job.options["max_side"] is not None):
            return True, "frame plan requested but absent"
        if job.has_plan:
            try:
                plan = hvideo.FramePlan.load(job.plan_path)
                with np.load(job.keypoints_path, allow_pickle=False) as data:
                    if "frame_indices" in data.files:
                        if [int(i) for i in data["frame_indices"]] != [int(i) for i in plan.indices]:
                            return True, "frame plan changed"
                    elif len(data["keypoints3d"]) != plan.num_frames:
                        return True, "frame plan changed"
            except Exception as exc:  # noqa: BLE001 - unreadable output: redo
                return True, f"unreadable output ({type(exc).__name__})"
        return False, "exists"

    def _finish(self, job: Job, status: dict, started: float) -> dict:
        """Complete, save and log a job status."""
        status["seconds"] = perf_counter() - started
        status["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        try:
            write_json(job.status_path, status)
        except Exception as exc:  # noqa: BLE001
            status["status_write_error"] = repr(exc)
        self.log(f"{job.stem}: ok={status['ok']} {status['seconds']:.1f}s"
                 + (f" error={status['error']}" if status.get("error") else ""))
        return status

    def _write_outputs(self, job: Job, frames: JobFrames, dets: FrameDetections, indices: List[int],
                       image_size: Tuple[int, int], camera, stats: dict) -> dict:
        """Select the person, write the keypoint (and detection) files and fill ``stats``."""
        intrinsics, distortion, camera_source = camera
        timestamps = (np.asarray(frames.timestamps, dtype=float) if frames.timestamps is not None
                      else np.asarray(indices, dtype=float) / frames.fps)
        prior = job_prior(job.options, indices, timestamps, image_size, frames.source_size)
        selected = select_person(dets.boxes, timestamps, job.options["person"], job.options["track_init_seconds"],
                                 prior, job.options["prior_min_iou"])
        # Diagnostic only (status "first_vs_track_disagreements"): the first/track rule not chosen.
        other = select_person(dets.boxes, timestamps, "track" if job.options["person"] == "first" else "first",
                              job.options["track_init_seconds"])
        payload = keypoint_arrays(dets, selected)
        counts = np.array([len(b) for b in dets.boxes], dtype=np.int32)
        box_conf = np.array([float(dets.boxes[t][j][4]) if j >= 0 else 0.0 for t, j in enumerate(selected)],
                            dtype=np.float32)
        config = {"engine": "hmb.detect_engine", "skeleton": job.options["skeleton"],
                  "batch_size": job.options["batch_size"], "detector_kwargs": job.options["detector_kwargs"],
                  "pad_final_batch": job.options["pad_final_batch"],
                  "graph_optimization": self.engine_options["graph_optimization"], "plan": frames.plan_source,
                  "person": job.options["person"], "track_init_seconds": job.options["track_init_seconds"]}
        if prior is not None:
            config.update(person_boxes=str(job.options["person_boxes"]) if not isinstance(
                job.options["person_boxes"], dict) else "inline",
                person_boxes_sha256=_file_sha256(job.options["person_boxes"]),
                prior_min_iou=job.options["prior_min_iou"])
        payload.update({
            "camera_intrinsics_px": np.asarray(intrinsics, dtype=float).reshape(1, 4),
            "camera_distortion": np.asarray(distortion, dtype=float).reshape(1, 5),
            "camera_calibration_source": np.array(camera_source),
            "timestamps": timestamps,
            "frame_indices": np.asarray(indices, dtype=np.int64),
            "fps": np.float64(frames.fps),
            "sample_fps": np.float64(np.nan if frames.sample_fps is None else frames.sample_fps),
            "image_size": np.asarray(image_size, dtype=np.int64),
            "source_size": np.asarray(frames.source_size, dtype=np.int64),
            "box_confidence": box_conf,
            "num_detections": counts,
            "selected_detection": selected,
            "person_selection": np.array(job.options["person"]),
            "detector_config": np.array(json.dumps(config, sort_keys=True)),
        })
        if prior is not None:
            payload["person_prior_boxes"] = prior  # (T, 4) [x, y, w, h] decoded px, NaN = no prior box
        _savez_atomic(job.keypoints_path, **payload)
        if job.options["save_detections"]:
            _savez_atomic(job.detections_path, **detections_arrays(dets), timestamps=timestamps,
                          frame_indices=np.asarray(indices, dtype=np.int64),
                          image_size=np.asarray(image_size, dtype=np.int64),
                          camera_intrinsics_px=np.asarray(intrinsics, dtype=float).reshape(1, 4))
        detected = selected >= 0
        both = (selected >= 0) | (other >= 0)
        stats.update({
            "frames": int(len(indices)),
            "frames_detected": int(detected.sum()),
            "frames_multi_detection": int((counts > 1).sum()),
            "max_detections": int(counts.max()) if len(counts) else 0,
            "first_vs_track_disagreements": int(((selected != other) & both).sum()),
            **({"prior_frames": int(np.isfinite(prior[:, 0]).sum()),
                "prior_frames_unmatched": int((np.isfinite(prior[:, 0]) & (selected < 0)).sum())}
               if prior is not None else {}),
            "mean_box_confidence": float(box_conf[detected].mean()) if detected.any() else None,
            "output": str(job.keypoints_path),
        })
        return stats

    # ................................................................... batch
    def run(self, entries: Iterable, force: bool = False) -> List[dict]:
        """Process jobs in order; a decoder thread keeps the next batches (and jobs) ready.

        Returns one status per entry (also written to each job's ``status_path``);
        a failing job never stops the others.
        """
        if self.detector is None:
            self.load()
        jobs: List[Any] = []
        for entry in entries:
            try:
                jobs.append(make_job(entry, self.options))
            except Exception as exc:  # noqa: BLE001
                jobs.append(exc)
        statuses: List[Optional[dict]] = [None] * len(jobs)
        work: List[Job] = []
        work_slot: List[int] = []
        for i, job in enumerate(jobs):
            if isinstance(job, Exception):
                statuses[i] = {"engine": ENGINE, "ok": False, "skipped": False, "seconds": 0.0,
                               "error": f"{type(job).__name__}: {job}"}
                self.log(f"job {i}: invalid ({job})")
                continue
            needed, reason = self.needs_work(job, force)
            if not needed:
                statuses[i] = {"engine": ENGINE, "ok": True, "skipped": True, "seconds": 0.0, "reason": reason,
                               "stem": job.stem, "session_dir": str(job.session_dir),
                               "output": str(job.keypoints_path)}
                if not job.status_path.is_file():
                    write_json(job.status_path, statuses[i])
                continue
            work.append(job)
            work_slot.append(i)
        if not work:
            return statuses  # type: ignore[return-value]

        # The decoder thread (_Producer) feeds batches through a bounded queue; this thread
        # runs MeTRAbs and keeps the state of the job being received in ``current``.
        prefetch = max(j.options["prefetch_batches"] for j in work)
        items: "queue.Queue" = queue.Queue(maxsize=prefetch)
        stop = threading.Event()
        producer = _Producer(work, items, stop)
        producer.start()
        current: Dict[str, Any] = {}

        def begin(k, frames):
            """A new job starts: link its input video and reset the per-job state."""
            job = work[k]
            job.keypoints_path.parent.mkdir(parents=True, exist_ok=True)
            link = job.session_dir / "input" / f"{job.stem}.mp4"
            if not job.overrides and not link.exists() and not link.is_symlink():
                try:
                    link.parent.mkdir(parents=True, exist_ok=True)
                    link.symlink_to(job.video.resolve())
                except OSError:
                    pass
            if hasattr(self.detector, "memory_mb"):
                self.detector.memory_mb(reset=True)
            current.clear()
            current.update({"k": k, "frames": frames, "started": perf_counter(), "failed": None,
                            "dets": FrameDetections([], [], []), "image_size": None, "camera": None,
                            "inference_seconds": 0.0, "wait_seconds": 0.0, "batches": 0,
                            "padding_detections": 0, "batch_shapes": set(), "batch_seconds": []})

        def fail(k, message, trace=None):
            """Record job ``k`` as failed."""
            job = work[k]
            started = current.get("started", perf_counter()) if current.get("k") == k else perf_counter()
            status = {"engine": ENGINE, "ok": False, "skipped": False,
                      "stem": job.stem, "session_dir": str(job.session_dir), "video": str(job.video),
                      "options": job.options, "error": message}
            if trace:
                status["traceback"] = trace[-4000:]
            statuses[work_slot[k]] = self._finish(job, status, started)

        try:
            while True:
                waited = perf_counter()
                item = items.get()
                if current:
                    current["wait_seconds"] += perf_counter() - waited
                kind = item[0]
                if kind == "done":
                    break
                k = item[1]
                if kind == "begin":
                    begin(k, item[2])
                    continue
                if kind == "error":
                    if statuses[work_slot[k]] is None:
                        fail(k, item[2], item[3])
                    current.clear()
                    continue
                if current.get("k") != k or current.get("failed"):
                    continue  # batches of a job that already failed
                job = work[k]
                if kind == "batch":
                    frames = item[2]
                    try:
                        if current["image_size"] is None:
                            current["image_size"] = (int(frames.shape[2]), int(frames.shape[1]))
                            current["camera"] = job_camera(job, current["frames"].source_size, current["image_size"])
                        started = perf_counter()
                        pad_to = job.options["batch_size"] if job.options["pad_final_batch"] else len(frames)
                        dets, padded = self.detector.detect(frames, current["camera"][0], current["camera"][1],
                                                            pad_to, job.options["skeleton"],
                                                            job.options["detector_kwargs"])
                        current["inference_seconds"] += perf_counter() - started
                        current["batch_seconds"].append(round(perf_counter() - started, 4))
                        current["batches"] += 1
                        current["padding_detections"] += padded
                        current["batch_shapes"].add(tuple(int(s) for s in frames.shape[1:3]))
                        for name in ("boxes", "poses3d", "poses2d"):
                            getattr(current["dets"], name).extend(getattr(dets, name))
                    except Exception as exc:  # noqa: BLE001
                        current["failed"] = True
                        fail(k, f"{type(exc).__name__}: {exc}", traceback.format_exc())
                    continue
                if kind == "end":
                    info = item[2]
                    try:
                        stats = {"engine": ENGINE, "ok": True, "skipped": False,
                                 "error": None, "stem": job.stem, "session_dir": str(job.session_dir),
                                 "video": str(job.video), "plan": current["frames"].plan_source,
                                 "source_fps": current["frames"].fps, "sample_fps": current["frames"].sample_fps,
                                 "image_size": current["image_size"], "source_size": current["frames"].source_size,
                                 "batch_size": job.options["batch_size"], "person": job.options["person"],
                                 "batches": current["batches"], "decode_seconds": info["decode_seconds"],
                                 "inference_seconds": current["inference_seconds"],
                                 "decode_wait_seconds": current["wait_seconds"],
                                 "padding_detections": current["padding_detections"],
                                 "batch_seconds": current["batch_seconds"],
                                 "graph_optimization": self.engine_options["graph_optimization"]}
                        if len(current["batch_shapes"]) > 1:
                            raise RuntimeError(f"frame size changed within the video: {current['batch_shapes']}")
                        if len(current["dets"].boxes) != len(info["indices"]):
                            raise RuntimeError("detections and decoded frames are misaligned")
                        written = perf_counter()
                        self._write_outputs(job, current["frames"], current["dets"], info["indices"],
                                            current["image_size"], current["camera"], stats)
                        stats["write_seconds"] = perf_counter() - written
                        stats["per_frame_ms"] = 1000.0 * stats["inference_seconds"] / max(1, stats["frames"])
                        if hasattr(self.detector, "memory_mb"):
                            stats["gpu_memory"] = self.detector.memory_mb()
                        statuses[work_slot[k]] = self._finish(job, stats, current["started"])
                    except Exception as exc:  # noqa: BLE001
                        fail(k, f"{type(exc).__name__}: {exc}", traceback.format_exc())
                    current.clear()
        finally:
            stop.set()
            producer.join(timeout=5)
        for k, slot in enumerate(work_slot):
            if statuses[slot] is None:
                fail(k, "RuntimeError: job was not completed")
        return statuses  # type: ignore[return-value]

    def process(self, entry, force: bool = False) -> dict:
        """Process one job (a ``jobs.json`` entry or a :class:`Job`); returns its status."""
        return self.run([entry], force=force)[0]


def reselect_session(session_dir: Path, stem: str, person: str = "track", init_seconds: float = 0.5,
                     output: Optional[Path] = None, person_boxes=None,
                     prior_min_iou: float = PRIOR_MIN_IOU) -> Path:
    """Rewrite ``_keypoints.npz`` from the cached detections with another person selection (CPU only).

    ``person='boxes'`` needs ``person_boxes`` (see :func:`load_person_boxes`)."""
    session_dir = Path(session_dir)
    job = Job(session_dir, stem, session_dir / "input" / f"{stem}.mp4",
              resolve_options({"person": person, "person_boxes": person_boxes, "prior_min_iou": prior_min_iou}))
    dets, extra = load_detections(job.detections_path)
    with np.load(job.keypoints_path, allow_pickle=False) as data:
        previous = {k: data[k] for k in data.files}
    image_size = tuple(int(v) for v in extra["image_size"])
    source_size = tuple(int(v) for v in previous.get("source_size", image_size))
    prior = job_prior(job.options, extra["frame_indices"], extra["timestamps"], image_size, source_size)
    selected = select_person(dets.boxes, extra["timestamps"], person, init_seconds, prior, prior_min_iou)
    payload = dict(previous)
    payload.pop("person_prior_boxes", None)
    if prior is not None:
        payload["person_prior_boxes"] = prior
    payload.update(keypoint_arrays(dets, selected))
    payload["box_confidence"] = np.array([float(dets.boxes[t][j][4]) if j >= 0 else 0.0
                                          for t, j in enumerate(selected)], dtype=np.float32)
    payload["selected_detection"] = selected
    payload["person_selection"] = np.array(person)
    target = Path(output) if output else job.keypoints_path
    _savez_atomic(target, **payload)
    return target


# ------------------------------------------------------------------------------ CLI
def _single_job(args) -> dict:
    """The job entry of ``--video``/``--session-dir``/``--stem``."""
    session = Path(args.session_dir).expanduser().absolute()
    video = Path(args.video).expanduser().absolute()
    return {"session_dir": str(session), "stem": args.stem or sanitize_stem(video.stem), "video": str(video)}


def main(argv: Optional[List[str]] = None) -> int:
    """Command line (see the module docstring); returns 1 if any job failed."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jobs", help="jobs.json")
    parser.add_argument("--video", help="single video (with --session-dir)")
    parser.add_argument("--session-dir", help="session directory of the single video")
    parser.add_argument("--stem", help="output name of the single video (default: its file name)")
    parser.add_argument("--force", action="store_true", help="redo jobs whose keypoints exist")
    parser.add_argument("--summary", help="write engine timing and all job statuses to this JSON file")
    parser.add_argument("--model-path", default=None, help="MeTRAbs SavedModel (default: $HMB_METRABS_MODEL)")
    parser.add_argument("--graph-optimization", choices=GRAPH_MODES, default=None,
                        help="TensorFlow graph optimisation of the first call (default: lean)")
    parser.add_argument("--no-warmup", action="store_true", help="skip the warm-up batch at load time")
    parser.add_argument("--cudnn-autotune", choices=("on", "off"), default=None,
                        help="off = TF_CUDNN_USE_AUTOTUNE=0 (reproducible across processes; the default)")
    parser.add_argument("--batch-size", type=int, default=None, help="frames per detector call (default: 8)")
    parser.add_argument("--person", choices=PERSON_MODES, default=None, help="person selection (default: first)")
    parser.add_argument("--person-boxes", default=None, help="person=boxes: subject box track JSON")
    parser.add_argument("--sample-fps", type=float, default=None, help="create input/frames.json at this rate if absent")
    parser.add_argument("--max-side", type=int, default=None, help="create input/frames.json with this max side if absent")
    parser.add_argument("--camera-calibration", default=None, help="JSON with fx, fy, cx, cy [, distortion]")
    parser.add_argument("--fov-degrees", type=float, default=None,
                        help="assumed field of view without a calibration (default: 55)")
    parser.add_argument("--no-save-detections", action="store_true",
                        help="do not keep all detections in cache/detect/")
    args = parser.parse_args(argv)

    if bool(args.jobs) == bool(args.video):
        parser.error("give either --jobs or --video/--session-dir")
    if args.jobs:
        options, entries = load_jobs(Path(args.jobs))
    else:
        if not args.session_dir:
            parser.error("--video needs --session-dir")
        options, entries = {}, [_single_job(args)]
    overrides = {"model_path": args.model_path, "graph_optimization": args.graph_optimization,
                 "batch_size": args.batch_size, "person": args.person, "sample_fps": args.sample_fps,
                 "max_side": args.max_side, "camera_calibration": args.camera_calibration,
                 "fov_degrees": args.fov_degrees, "person_boxes": args.person_boxes}
    options.update({k: v for k, v in overrides.items() if v is not None})
    if args.no_warmup:
        options["warmup"] = False
    if args.cudnn_autotune:
        options["cudnn_autotune"] = args.cudnn_autotune == "on"
    if args.no_save_detections:
        options["save_detections"] = False
    started = perf_counter()
    engine = DetectEngine(options)
    engine.load()
    loaded = perf_counter() - started
    statuses = engine.run(entries, force=args.force)
    total = perf_counter() - started
    summary = {"engine": ENGINE, "load_seconds": loaded, "total_seconds": total,
               "load_info": engine.load_info, "options": {**engine.engine_options, **engine.options},
               "jobs": statuses}
    if args.summary:
        write_json(Path(args.summary), summary)
    failed = sum(1 for s in statuses if not s.get("ok"))
    print(f"[detect_engine] {len(statuses)} jobs, {failed} failed, load {loaded:.1f}s, total {total:.1f}s", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
