"""Shared video probing, temporal sampling and frame decoding.

Every reconstruction engine (MeTRAbs keypoints, GVHMR mesh, HADM extra-limb
detection) must see the *same* frames with the *same* timestamps, so that all
per-frame results of a session line up.  A sampling plan (:class:`FramePlan`)
selects the source frames closest to a uniform target rate and keeps their exact
source timestamps, so derivative-based metrics use true time even when the
selected frames are not uniformly spaced (e.g. 60 fps -> 24 Hz).  The plan can
also downscale frames to a maximum side length.  ``sample_fps=None`` keeps all
frames, as the original paper implementation does.

The plan is written once per session as ``<session>/input/frames.json``
(``hmb.batch.plan_session_dir``); every engine reads it and decodes only the
planned frames with :func:`iter_frames`.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator, Optional, Sequence

import numpy as np

PLAN_NAME = "frames.json"


@dataclass
class FramePlan:
    """The frames of a video that a session analyses (stored as ``frames.json``)."""

    video: str                      # resolved path of the source video
    source_fps: float               # container frame rate
    num_source_frames: int          # decodable frames in the source
    width: int                      # source frame size (pixels)
    height: int
    sample_fps: Optional[float]     # requested analysis rate; None = all frames
    indices: list = field(default_factory=list)     # selected source frame indices (increasing)
    timestamps: list = field(default_factory=list)  # their source times in seconds (index / source_fps)
    max_side: Optional[int] = None  # decoded frames are downscaled to this longer side (None = source size)

    @property
    def num_frames(self) -> int:
        return len(self.indices)

    @property
    def output_size(self) -> tuple[int, int]:
        """(width, height) of decoded frames after the optional downscale."""
        return output_size(self.width, self.height, self.max_side)

    def as_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Path) -> Path:
        """Write the plan atomically to ``path`` (a directory gets ``frames.json``); returns the file."""
        path = Path(path)
        if path.is_dir():
            path = path / PLAN_NAME
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        tmp.write_text(json.dumps(self.as_dict(), indent=1))
        tmp.replace(path)
        return path

    @classmethod
    def load(cls, path: Path) -> "FramePlan":
        """Read a plan from ``path`` (a file, or a directory containing ``frames.json``)."""
        path = Path(path)
        if path.is_dir():
            path = path / PLAN_NAME
        return cls(**json.loads(path.read_text()))


def output_size(width: int, height: int, max_side: Optional[int]) -> tuple[int, int]:
    """``(width, height)`` after scaling the longer side down to ``max_side`` (never up)."""
    if not max_side or max(width, height) <= max_side:
        return int(width), int(height)
    scale = max_side / float(max(width, height))
    # keep even dimensions for codecs and stable intrinsics
    return int(round(width * scale / 2) * 2), int(round(height * scale / 2) * 2)


def probe(video: Path) -> dict:
    """Return fps, decoded frame count and frame size.

    The frame count is obtained by grabbing every frame, because container
    metadata is unreliable for generated (sometimes variable-rate) videos.
    """
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count = 0
    while cap.grab():
        count += 1
    cap.release()
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"Video {video} reports an invalid frame rate {fps}")
    if count == 0:
        raise ValueError(f"Video {video} has no decodable frames")
    return {"fps": fps, "num_frames": count, "width": width, "height": height,
            "duration": count / fps}


def sample_indices(num_frames: int, fps: float, sample_fps: Optional[float]) -> np.ndarray:
    """Indices of source frames nearest to a uniform ``sample_fps`` grid.

    ``sample_fps`` None, <= 0 or at/above the source rate keeps every frame.
    """
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if sample_fps is None or sample_fps <= 0 or sample_fps >= fps * (1 - 1e-6):
        return np.arange(num_frames)
    last = (num_frames - 1) / fps
    grid = np.arange(0.0, last + 0.5 / sample_fps, 1.0 / sample_fps)
    idx = np.clip(np.rint(grid * fps).astype(int), 0, num_frames - 1)
    return np.unique(idx)


def plan_frames(video: Path, sample_fps: Optional[float] = None, max_side: Optional[int] = None,
                info: Optional[dict] = None) -> FramePlan:
    """Build the :class:`FramePlan` of ``video``; ``info`` (a :func:`probe` result) skips decoding."""
    info = info or probe(video)
    idx = sample_indices(info["num_frames"], info["fps"], sample_fps)
    return FramePlan(video=str(Path(video).resolve()), source_fps=float(info["fps"]),
                     num_source_frames=int(info["num_frames"]), width=int(info["width"]),
                     height=int(info["height"]), sample_fps=sample_fps,
                     indices=[int(i) for i in idx],
                     timestamps=[float(i) / float(info["fps"]) for i in idx], max_side=max_side)


def iter_frames(video: Path, indices: Optional[Sequence[int]] = None, *, rgb: bool = True,
                max_side: Optional[int] = None) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(index, frame)`` for the selected source indices in order.

    Unselected frames are skipped with ``grab()`` (no colour conversion).
    """
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video {video}")
    wanted = None if indices is None else iter(sorted(int(i) for i in indices))
    target = None if wanted is None else next(wanted, None)
    frame_index = -1
    try:
        while True:
            if wanted is not None and target is None:
                break
            if not cap.grab():
                break
            frame_index += 1
            if wanted is not None and frame_index != target:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                break
            if max_side and max(frame.shape[:2]) > max_side:
                w, h = output_size(frame.shape[1], frame.shape[0], max_side)
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            if rgb:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield frame_index, frame
            if wanted is not None:
                target = next(wanted, None)
    finally:
        cap.release()


def read_plan_frames(plan: FramePlan, *, rgb: bool = True) -> np.ndarray:
    """Decode all planned frames into one ``(T, H, W, 3)`` uint8 array."""
    frames = [f for _, f in iter_frames(Path(plan.video), plan.indices, rgb=rgb, max_side=plan.max_side)]
    if len(frames) != plan.num_frames:
        raise RuntimeError(f"Decoded {len(frames)} of {plan.num_frames} planned frames from {plan.video}")
    return np.stack(frames)
