"""``track_people``: multi-person tracking of one video into a directory.

Runs one backend (:mod:`hmb.tracking.sam3_backend` or :mod:`hmb.tracking.yolo_backend`),
keeps its unprocessed output in ``DIR/raw/``, then cleans the masks and re-joins
fragments of the same person (:mod:`hmb.tracking.tracks`) into ``DIR/tracks.json`` and
``DIR/masks.npz``.  Both steps are reused while the video and the settings are unchanged.
Used by :mod:`hmb.multi` and ``python -m hmb.tracking``.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

from .render import write_preview
from .tracks import MASKS_NPZ, TRACKS_JSON, Tracks, clean_masks, stitch_fragments

BACKENDS = ("sam3.1", "yolo")
#: Accepted spellings of the backend names.
ALIASES = {"sam3": "sam3.1", "sam3.1": "sam3.1", "sam31": "sam3.1", "yolo": "yolo"}
PREVIEW_NAME = "preview.mp4"
SHEET_NAME = "preview_sheet.jpg"
RAW_DIR = "raw"


def normalize_backend(backend: str) -> str:
    """The canonical backend name (``"sam3.1"`` or ``"yolo"``); ``ValueError`` for anything else."""
    key = str(backend).strip().lower()
    if key not in ALIASES:
        raise ValueError(f"unknown tracking backend {backend!r}; choose one of {BACKENDS}")
    return ALIASES[key]


def _video_signature(video: Path) -> dict:
    """Identity of the video file (path, size, modification time) for the cache check."""
    st = video.stat()
    return {"path": str(video), "size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}


def _cached(out_dir: Path, request: dict) -> Optional[Tracks]:
    """The tracks saved in ``out_dir`` if they were made for exactly ``request``, else ``None``."""
    if not (out_dir / TRACKS_JSON).exists() or not (out_dir / MASKS_NPZ).exists():
        return None
    try:
        info = json.loads((out_dir / TRACKS_JSON).read_text())
        if info.get("meta", {}).get("request") != request:
            return None
        return Tracks.load(out_dir)
    except Exception:  # unreadable or partial files: track again
        return None


def track_people(video, out_dir, *, backend: str = "sam3.1", prompt: str = "person", force: bool = False,
                 quiet: bool = False, clean: bool = True, stitch: bool = True, stitch_gap_seconds: float = 1.0,
                 preview: bool = True, max_objects: int = 16, sam3_chunk_frames: int = 600,
                 sam3_python: Optional[str] = None, sam3_checkpoint: Optional[str] = None,
                 yolo_weights: Optional[str] = None, device=None) -> Tracks:
    """Track every person in ``video`` and write ``out_dir/{tracks.json, masks.npz, preview.mp4}``.

    The unprocessed tracker output is kept in ``out_dir/raw/`` (same format).

    ``backend`` is ``"sam3.1"`` (SAM 3.1 Object Multiplex, text prompt ``prompt``,
    run in its own environment, see :mod:`hmb.tracking.sam3_backend`) or ``"yolo"``
    (YOLO segmentation + BoT-SORT, see :mod:`hmb.tracking.yolo_backend`).  With
    ``clean`` isolated mask specks are removed (:func:`clean_masks`); with
    ``stitch`` a track that restarts under a new id within ``stitch_gap_seconds``
    at the same place is merged back (:func:`stitch_fragments`).  A previous
    result in ``out_dir`` made with the same video and settings is reused unless
    ``force``.  Returns the :class:`Tracks`.

    SAM 3.1 options: ``max_objects`` (objects one session can hold),
    ``sam3_chunk_frames`` (longer videos are tracked in overlapping sessions of
    this many frames, 0 = one session), ``sam3_python`` / ``sam3_checkpoint``
    (else ``$HMB_SAM3_PYTHON`` / ``$HMB_SAM3_CHECKPOINT``).  YOLO options:
    ``yolo_weights`` (else ``$HMB_YOLO_SEG_WEIGHTS`` or yolov8x-seg.pt), ``device``.
    """
    backend = normalize_backend(backend)
    video = Path(video).resolve()
    if not video.is_file():
        raise FileNotFoundError(f"video not found: {video}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if backend == "sam3.1":
        backend_options = {"prompt": prompt, "max_objects": int(max_objects),
                           "chunk_frames": int(sam3_chunk_frames)}
    else:
        backend_options = {"weights": str(yolo_weights or os.environ.get("HMB_YOLO_SEG_WEIGHTS") or "default")}
    post_options = {"clean": bool(clean), "stitch": bool(stitch), "stitch_gap_seconds": float(stitch_gap_seconds)}
    signature = _video_signature(video)
    raw_request = {"video": signature, "backend": backend, "options": backend_options}
    request = dict(raw_request, postprocess=post_options)

    tracks = None if force else _cached(out_dir, request)
    if tracks is not None:
        if not quiet:
            count = len(tracks.ids)
            print(f"[tracking] reusing {out_dir / TRACKS_JSON} ({count} {'person' if count == 1 else 'people'})",
                  file=sys.stderr, flush=True)
    else:
        # the raw tracker output is kept in out_dir/raw so post-processing can change without re-tracking
        raw_dir = out_dir / RAW_DIR
        raw = None if force else _cached(raw_dir, raw_request)
        if raw is None:
            t0 = time.time()
            if backend == "sam3.1":
                from .sam3_backend import track_sam3

                raw = track_sam3(video, prompt=prompt, python=sam3_python, checkpoint=sam3_checkpoint,
                                 max_objects=max_objects, chunk_frames=sam3_chunk_frames, work_dir=out_dir,
                                 quiet=quiet)
            else:
                from .yolo_backend import track_yolo

                raw = track_yolo(video, weights=yolo_weights, device=device, quiet=quiet)
            raw.meta.update(request=raw_request, seconds=round(time.time() - t0, 2))
            raw.save(raw_dir)
        elif not quiet:
            print(f"[tracking] reusing raw {backend} output in {raw_dir}", file=sys.stderr, flush=True)
        tracks = raw
        if clean:
            tracks = clean_masks(tracks)
        if stitch:
            tracks = stitch_fragments(tracks, max_gap_seconds=stitch_gap_seconds)
        tracks.meta.update(request=request, raw_ids=raw.ids)
        tracks.save(out_dir)
        if not quiet:
            merged = tracks.meta.get("stitched", [])
            count = len(tracks.ids)
            print(f"[tracking] {backend}: {count} {'person' if count == 1 else 'people'} ({len(raw.ids)} raw ids"
                  + (f", {len(merged)} fragment(s) re-joined" if merged else "") + f") -> {out_dir / TRACKS_JSON}",
                  file=sys.stderr, flush=True)
    if preview and (force or not (out_dir / PREVIEW_NAME).exists()
                    or (out_dir / PREVIEW_NAME).stat().st_mtime < (out_dir / TRACKS_JSON).stat().st_mtime):
        write_preview(video, tracks, out_dir / PREVIEW_NAME, sheet_path=out_dir / SHEET_NAME)
    return tracks
