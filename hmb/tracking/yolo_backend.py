"""YOLO tracking backend: ultralytics instance segmentation + BoT-SORT, person class only.

Always available in the HumanScore environment (``ultralytics`` is a GVHMR
dependency).  Weights, in order: the ``weights`` argument,
``$HMB_YOLO_SEG_WEIGHTS``, ``<repo>/outputs/assets/yolo/yolov8x-seg.pt``; when
none exists, the default ``yolov8x-seg.pt`` is downloaded once from the
ultralytics release assets into ``<repo>/outputs/assets/yolo/``.

Frames are decoded with OpenCV (the frame indices of every other HumanScore
stage) and fed one by one to ``model.track(persist=True)``; masks are produced
at the source resolution (``retina_masks``) and each record's box is the tight
box of its mask.
"""
from __future__ import annotations

import contextlib
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

from .render import iter_video, video_info
from .tracks import TrackRecorder, Tracks

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_WEIGHTS = "HMB_YOLO_SEG_WEIGHTS"
DEFAULT_WEIGHTS = "yolov8x-seg.pt"
ASSET_DIR = REPO_ROOT / "outputs" / "assets" / "yolo"
PERSON_CLASS = 0  # COCO class id of "person"


def resolve_weights(weights: Optional[str] = None, download: bool = True) -> Path:
    """Local path of the segmentation weights, downloading the default once if needed."""
    explicit = weights or os.environ.get(ENV_WEIGHTS)
    if explicit:
        path = Path(explicit).expanduser()
        if path.exists():
            return path.resolve()
        if path.parent != Path("."):
            raise FileNotFoundError(f"YOLO segmentation weights not found: {explicit}")
        name = path.name                  # a bare asset name such as "yolov8l-seg.pt"
    else:
        name = DEFAULT_WEIGHTS
    target = ASSET_DIR / name
    if target.exists():
        return target
    if not download:
        raise FileNotFoundError(f"YOLO segmentation weights {target} are missing")
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    from ultralytics.utils.downloads import attempt_download_asset

    print(f"[yolo] downloading {name} into {ASSET_DIR} (one time)", file=sys.stderr, flush=True)
    found = Path(attempt_download_asset(str(target)))
    if not found.exists():
        raise FileNotFoundError(f"Could not download {name} into {ASSET_DIR}; download it manually and set "
                                f"{ENV_WEIGHTS}=/path/to/{name}")
    return found.resolve()


@contextlib.contextmanager
def _trusted_weights(path: Path):
    """Let ``torch.load`` fully unpickle this one (official ultralytics) weight file.

    The installed ultralytics stores whole modules in its weight files, which the
    ``weights_only=True`` default of recent torch versions refuses to load.
    """
    import torch

    original = torch.load
    trusted = Path(path).resolve()

    def load(f, *args, **kwargs):
        if isinstance(f, (str, os.PathLike)) and Path(f).resolve() == trusted:
            kwargs.setdefault("weights_only", False)
        return original(f, *args, **kwargs)

    torch.load = load
    try:
        yield
    finally:
        torch.load = original


def load_model(weights: Optional[str] = None):
    """The ultralytics segmentation model -> ``(model, weights_path)``; rejects non-segmentation weights."""
    # no console chatter and no automatic pip installs from ultralytics
    os.environ.setdefault("YOLO_VERBOSE", "False")
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    from ultralytics import YOLO

    path = resolve_weights(weights)
    with _trusted_weights(path):
        model = YOLO(str(path))
    if getattr(model, "task", None) != "segment":
        raise ValueError(f"{path} is a '{getattr(model, 'task', '?')}' model; the tracking backend needs a "
                         f"segmentation model (*-seg.pt)")
    return model, path


def result_records(result, height: int, width: int):
    """``(ids, scores, masks)`` of one tracked ultralytics result (empty when nothing is tracked)."""
    boxes = getattr(result, "boxes", None)
    masks = getattr(result, "masks", None)
    if boxes is None or masks is None or getattr(boxes, "id", None) is None or len(boxes) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.float32), np.zeros((0, height, width), bool)
    ids = np.asarray(_numpy(boxes.id)).astype(np.int64).reshape(-1)
    scores = np.asarray(_numpy(boxes.conf), dtype=np.float32).reshape(-1)
    data = np.asarray(_numpy(masks.data))
    if data.shape[1:] != (height, width):
        import cv2

        data = np.stack([cv2.resize(m.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
                         for m in data]) if len(data) else np.zeros((0, height, width), np.float32)
    return ids, scores, data > 0.5


def _numpy(x):
    """A torch tensor (on any device) or array-like as a numpy array."""
    return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)


def track_yolo(video, *, weights: Optional[str] = None, tracker: str = "botsort.yaml", conf: float = 0.1,
               iou: float = 0.7, imgsz: int = 640, device=None, quiet: bool = False) -> Tracks:
    """Track people through ``video`` with YOLO segmentation + BoT-SORT."""
    video = Path(video).resolve()
    info = video_info(video)
    height, width = info["height"], info["width"]
    model, path = load_model(weights)
    recorder = TrackRecorder(width, height)
    t0 = last = time.time()
    count = 0
    for index, frame in iter_video(video):
        if frame.shape[:2] != (height, width):
            raise ValueError(f"frame {index} of {video} has size {frame.shape[:2]}, expected {(height, width)}")
        results = model.track(frame, persist=True, tracker=tracker, classes=[PERSON_CLASS], conf=conf, iou=iou,
                              imgsz=imgsz, retina_masks=True, verbose=False, device=device)
        ids, scores, masks = result_records(results[0], height, width)
        for oid, score, mask in zip(ids, scores, masks):
            recorder.add(index, int(oid), mask, float(score))
        count = index + 1
        if not quiet and time.time() - last > 10:
            print(f"[yolo] tracked {count}/{info['num_frames']} frames", file=sys.stderr, flush=True)
            last = time.time()
    elapsed = time.time() - t0
    if not quiet:
        print(f"[yolo] done: {count} frames, {len(set(recorder.obj_id))} id(s), {elapsed:.0f}s "
              f"({count / max(elapsed, 1e-6):.1f} fps)", file=sys.stderr, flush=True)
    meta = {"yolo": {"weights": str(path), "tracker": tracker, "conf": conf, "iou": iou, "imgsz": imgsz,
                     "seconds": round(elapsed, 2)}}
    return recorder.build(video=str(video), fps=info["fps"], num_frames=count, backend="yolo", meta=meta)
