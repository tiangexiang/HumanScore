"""Per-person tracks of one video: storage, mask encoding, fragment stitching, selection.

A :class:`Tracks` holds one record per (frame, person) where the person is
visible: its id, its tight box ``[x0, y0, x1, y1]`` in source pixels (``x1``/``y1``
exclusive), a detection score and its binary mask.  Masks are stored as
uncompressed COCO run-length encodings (column-major, first run = background),
so ``pycocotools.mask.frPyObjects({"size": [H, W], "counts": counts}, H, W)``
decodes them too.

On disk a track set is two files in one directory:

* ``tracks.json`` -- video metadata and, per person id, the frames it is visible
  in and its boxes (human readable; no masks);
* ``masks.npz``   -- the records (``frame``, ``obj_id``, ``box``, ``score``) and the
  concatenated RLE counts with per-record offsets.

Frame ``i`` of the source video has timestamp ``i / fps`` (the convention of
:mod:`hmb.video`).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

TRACKS_JSON = "tracks.json"
MASKS_NPZ = "masks.npz"
#: Layout identifier written to ``tracks.json`` (``"format_version"``).
FORMAT_VERSION = 1


# ----------------------------------------------------------------------------- RLE
def rle_encode(mask: np.ndarray) -> np.ndarray:
    """Uncompressed COCO RLE of a 2-D boolean mask (column-major, starts with a 0-run)."""
    flat = np.asarray(mask, dtype=bool).ravel(order="F")
    if flat.size == 0:
        return np.zeros(0, dtype=np.uint32)
    change = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    bounds = np.concatenate(([0], change, [flat.size]))
    counts = np.diff(bounds)
    if flat[0]:
        counts = np.concatenate(([0], counts))
    return counts.astype(np.uint32)


def rle_decode(counts: np.ndarray, height: int, width: int) -> np.ndarray:
    """Inverse of :func:`rle_encode`; returns a ``(height, width)`` boolean mask."""
    counts = np.asarray(counts, dtype=np.int64)
    if int(counts.sum()) != height * width:
        raise ValueError(f"RLE covers {int(counts.sum())} pixels, expected {height * width}")
    values = np.zeros(len(counts), dtype=bool)
    values[1::2] = True
    return np.repeat(values, counts).reshape(width, height).T


def mask_box(mask: np.ndarray) -> Optional[np.ndarray]:
    """Tight ``[x0, y0, x1, y1]`` box (exclusive max) of a boolean mask, None when empty."""
    mask = np.asarray(mask, dtype=bool)
    cols = np.flatnonzero(mask.any(axis=0))
    if cols.size == 0:
        return None
    rows = np.flatnonzero(mask.any(axis=1))
    return np.array([cols[0], rows[0], cols[-1] + 1, rows[-1] + 1], dtype=np.float32)


def remove_specks(mask: np.ndarray, min_fraction: float = 0.005) -> np.ndarray:
    """Drop connected components smaller than ``min_fraction`` x the largest one.

    Segmenters leave isolated specks (often single pixels, from mask upsampling)
    far from the person; they would stretch the person's box.  Real detached
    parts (a hand seen past an occluder) are far larger than 0.5% of the body.
    Returns ``mask`` itself when nothing is removed.
    """
    import cv2

    box = mask_box(mask)
    if box is None:
        return mask
    x0, y0, x1, y1 = (int(v) for v in box)
    crop = np.ascontiguousarray(mask[y0:y1, x0:x1], dtype=np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(crop, connectivity=8)
    if n <= 2:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.flatnonzero(areas >= float(min_fraction) * areas.max()) + 1
    if len(keep) == n - 1:
        return mask
    out = np.zeros(mask.shape, dtype=bool)
    out[y0:y1, x0:x1] = np.isin(labels, keep)
    return out


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over union of two ``[x0, y0, x1, y1]`` boxes (0 when both are empty)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


# ----------------------------------------------------------------------------- builder
class TrackRecorder:
    """Accumulates per-frame (id, mask[, score]) outputs of a tracker into a :class:`Tracks`."""

    def __init__(self, width: int, height: int):
        self.width = int(width)
        self.height = int(height)
        self.frame: List[int] = []
        self.obj_id: List[int] = []
        self.box: List[np.ndarray] = []
        self.score: List[float] = []
        self.counts: List[np.ndarray] = []

    def add(self, frame: int, obj_id: int, mask: np.ndarray, score: float = 1.0,
            box: Optional[Sequence[float]] = None) -> bool:
        """Record one visible person; an empty mask is ignored (returns False)."""
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (self.height, self.width):
            raise ValueError(f"mask shape {mask.shape} != video size {(self.height, self.width)}")
        tight = mask_box(mask)
        if tight is None:
            return False
        self.frame.append(int(frame))
        self.obj_id.append(int(obj_id))
        self.box.append(tight if box is None else np.asarray(box, dtype=np.float32))
        self.score.append(float(score))
        self.counts.append(rle_encode(mask))
        return True

    def build(self, **meta) -> "Tracks":
        """The recorded people as :class:`Tracks` (``meta``: ``video``, ``fps``, ``num_frames``, ``backend``, ...)."""
        n = len(self.frame)
        lengths = np.array([len(c) for c in self.counts], dtype=np.int64)
        offsets = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)
        counts = np.concatenate(self.counts).astype(np.uint32) if n else np.zeros(0, np.uint32)
        tracks = Tracks(width=self.width, height=self.height,
                        frame=np.asarray(self.frame, dtype=np.int32).reshape(n),
                        obj_id=np.asarray(self.obj_id, dtype=np.int32).reshape(n),
                        box=np.asarray(self.box, dtype=np.float32).reshape(n, 4),
                        score=np.asarray(self.score, dtype=np.float32).reshape(n),
                        rle_offsets=offsets, rle_counts=counts, **meta)
        return tracks.sorted()


# ----------------------------------------------------------------------------- tracks
@dataclass
class Tracks:
    """All tracked people of one video (see the module docstring for the layout)."""

    video: str
    fps: float
    num_frames: int
    width: int
    height: int
    frame: np.ndarray
    obj_id: np.ndarray
    box: np.ndarray
    score: np.ndarray
    rle_offsets: np.ndarray
    rle_counts: np.ndarray
    backend: str = ""
    meta: dict = field(default_factory=dict)

    # -- basic access
    def __len__(self) -> int:
        """Number of (frame, person) records."""
        return int(len(self.frame))

    @property
    def ids(self) -> List[int]:
        """Sorted ids of the tracked people."""
        return sorted(int(i) for i in np.unique(self.obj_id))

    @property
    def timestamps(self) -> np.ndarray:
        """Timestamp in seconds of every source frame (``i / fps``)."""
        return np.arange(self.num_frames, dtype=float) / float(self.fps)

    def records(self, obj_id: Optional[int] = None, frame: Optional[int] = None) -> np.ndarray:
        """Indices of the records of one person and/or one frame (all records without filters)."""
        keep = np.ones(len(self), dtype=bool)
        if obj_id is not None:
            keep &= self.obj_id == int(obj_id)
        if frame is not None:
            keep &= self.frame == int(frame)
        return np.flatnonzero(keep)

    def frames(self, obj_id: int) -> np.ndarray:
        """Frames in which ``obj_id`` is visible."""
        return self.frame[self.records(obj_id)].astype(int)

    def boxes(self, obj_id: int) -> np.ndarray:
        """``(num_frames, 4)`` xyxy boxes of one person, NaN where it is not visible."""
        out = np.full((self.num_frames, 4), np.nan, dtype=np.float64)
        rec = self.records(obj_id)
        frames = self.frame[rec]
        ok = (frames >= 0) & (frames < self.num_frames)
        out[frames[ok]] = self.box[rec[ok]]
        return out

    def decode(self, record: int) -> np.ndarray:
        """Boolean mask of one record."""
        lo, hi = int(self.rle_offsets[record]), int(self.rle_offsets[record + 1])
        return rle_decode(self.rle_counts[lo:hi], self.height, self.width)

    def mask(self, obj_id: int, frame: int) -> Optional[np.ndarray]:
        """Boolean mask of one person in one frame (the union of its records), None when not visible."""
        rec = self.records(obj_id, frame)
        if rec.size == 0:
            return None
        out = self.decode(int(rec[0]))
        for r in rec[1:]:
            out |= self.decode(int(r))
        return out

    def masks_at(self, frame: int) -> Dict[int, np.ndarray]:
        """``{obj_id: mask}`` of every person visible in ``frame``."""
        out: Dict[int, np.ndarray] = {}
        for r in self.records(frame=frame):
            oid = int(self.obj_id[r])
            m = self.decode(int(r))
            out[oid] = out[oid] | m if oid in out else m
        return out

    def by_frame(self) -> List[np.ndarray]:
        """Record indices grouped per frame (list of length ``num_frames``)."""
        order = np.argsort(self.frame, kind="stable")
        groups: List[np.ndarray] = [np.zeros(0, dtype=np.int64)] * self.num_frames
        frames = self.frame[order]
        starts = np.searchsorted(frames, np.arange(self.num_frames), side="left")
        ends = np.searchsorted(frames, np.arange(self.num_frames), side="right")
        for f in range(self.num_frames):
            if ends[f] > starts[f]:
                groups[f] = order[starts[f]:ends[f]]
        return groups

    # -- transforms
    def sorted(self) -> "Tracks":
        """A copy with the records ordered by frame, then id."""
        order = np.lexsort((self.obj_id, self.frame))
        return self.subset(order)

    def subset(self, records: Sequence[int]) -> "Tracks":
        """A copy with only the given records (in that order)."""
        records = np.asarray(records, dtype=np.int64)
        lengths = np.diff(self.rle_offsets)[records] if len(records) else np.zeros(0, np.int64)
        offsets = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)
        if len(records):
            counts = np.concatenate([self.rle_counts[self.rle_offsets[r]:self.rle_offsets[r + 1]]
                                     for r in records]).astype(np.uint32)
        else:
            counts = np.zeros(0, np.uint32)
        return Tracks(video=self.video, fps=self.fps, num_frames=self.num_frames, width=self.width,
                      height=self.height, frame=self.frame[records], obj_id=self.obj_id[records],
                      box=self.box[records], score=self.score[records], rle_offsets=offsets,
                      rle_counts=counts, backend=self.backend, meta=dict(self.meta))

    def relabel(self, mapping: Dict[int, int]) -> "Tracks":
        """A copy with ids renamed by ``mapping`` (ids not in it are kept); used to merge fragments."""
        new = self.subset(np.arange(len(self)))
        new.obj_id = np.array([mapping.get(int(i), int(i)) for i in self.obj_id], dtype=np.int32)
        return new.sorted()

    # -- per-person summary
    def summary(self, obj_id: int) -> dict:
        """Visibility and size of one person: frames, seconds, fraction of the video, box height, prominence.

        ``prominence`` is the box area fraction summed over the frames and divided by
        the video length, so it rewards people who are both large and visible for long.
        """
        rec = self.records(obj_id)
        frames = np.unique(self.frame[rec])
        boxes = self.box[rec]
        heights = (boxes[:, 3] - boxes[:, 1]) / float(self.height)
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) / float(self.width * self.height)
        return {
            "id": int(obj_id),
            "first_frame": int(frames[0]) if frames.size else None,
            "last_frame": int(frames[-1]) if frames.size else None,
            "num_visible_frames": int(frames.size),
            "visible_seconds": float(frames.size / self.fps),
            "visible_fraction": float(frames.size / max(1, self.num_frames)),
            "median_height_fraction": float(np.median(heights)) if heights.size else 0.0,
            "mean_area_fraction": float(np.mean(areas)) if areas.size else 0.0,
            "prominence": float(np.sum(areas)) / max(1, self.num_frames),
            "mean_score": float(np.mean(self.score[rec])) if rec.size else 0.0,
        }

    def person_boxes(self, obj_id: int) -> dict:
        """The ``person_boxes`` JSON payload of :func:`hmb.pipeline.score_video` for one person."""
        boxes = self.boxes(obj_id)
        return {
            "video": self.video,
            "person_id": int(obj_id),
            "fps": float(self.fps),
            "num_frames": int(self.num_frames),
            "width": int(self.width),
            "height": int(self.height),
            "timestamps": [float(t) for t in self.timestamps],
            "boxes_xyxy": [None if not np.all(np.isfinite(b)) else [round(float(v), 2) for v in b]
                           for b in boxes],
        }

    def write_person_boxes(self, obj_id: int, path) -> Path:
        """Write :meth:`person_boxes` to ``path`` (JSON); returns the path."""
        path = Path(path)
        _atomic_write_text(path, json.dumps(self.person_boxes(obj_id)))
        return path

    # -- IO
    def save(self, out_dir) -> Path:
        """Write ``out_dir/masks.npz`` and ``out_dir/tracks.json`` (see the module docstring)."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp = out_dir / (MASKS_NPZ + ".part.npz")
        np.savez_compressed(tmp, frame=self.frame, obj_id=self.obj_id, box=self.box, score=self.score,
                            rle_offsets=self.rle_offsets, rle_counts=self.rle_counts,
                            size=np.array([self.height, self.width], dtype=np.int64))
        os.replace(tmp, out_dir / MASKS_NPZ)
        people = {}
        for oid in self.ids:
            rec = self.records(oid)
            people[str(oid)] = dict(self.summary(oid),
                                    frames=[int(f) for f in self.frame[rec]],
                                    boxes_xyxy=[[round(float(v), 2) for v in b] for b in self.box[rec]],
                                    scores=[round(float(s), 4) for s in self.score[rec]])
        payload = {
            "format": "humanscore-tracks",
            "format_version": FORMAT_VERSION,
            "video": self.video,
            "fps": float(self.fps),
            "num_frames": int(self.num_frames),
            "width": int(self.width),
            "height": int(self.height),
            "backend": self.backend,
            "meta": self.meta,
            "mask_encoding": "uncompressed COCO RLE (column-major) in masks.npz",
            "people": people,
        }
        _atomic_write_text(out_dir / TRACKS_JSON, json.dumps(payload, indent=1, allow_nan=False))
        return out_dir / TRACKS_JSON

    @classmethod
    def load(cls, path) -> "Tracks":
        """Read the tracks saved by :meth:`save` (``path``: the directory or its ``tracks.json``)."""
        path = Path(path)
        out_dir = path if path.is_dir() else path.parent
        info = json.loads((out_dir / TRACKS_JSON).read_text())
        with np.load(out_dir / MASKS_NPZ) as z:
            arrays = {k: z[k] for k in ("frame", "obj_id", "box", "score", "rle_offsets", "rle_counts")}
        return cls(video=info["video"], fps=float(info["fps"]), num_frames=int(info["num_frames"]),
                   width=int(info["width"]), height=int(info["height"]), backend=info.get("backend", ""),
                   meta=info.get("meta", {}), **arrays)


def load_tracks(path) -> Tracks:
    """:meth:`Tracks.load`."""
    return Tracks.load(path)


def _atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` through a temporary file and a rename, so readers never see a partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(text)
    os.replace(tmp, path)


# ----------------------------------------------------------------------------- cleaning
def clean_masks(tracks: Tracks, min_fraction: float = 0.005) -> Tracks:
    """:func:`remove_specks` on every mask; boxes are recomputed from the cleaned masks."""
    recorder = TrackRecorder(tracks.width, tracks.height)
    changed = 0
    for r in range(len(tracks)):
        mask = tracks.decode(r)
        cleaned = remove_specks(mask, min_fraction)
        if cleaned is mask:
            recorder.add(int(tracks.frame[r]), int(tracks.obj_id[r]), mask, float(tracks.score[r]),
                         box=tracks.box[r])
        else:
            changed += 1
            recorder.add(int(tracks.frame[r]), int(tracks.obj_id[r]), cleaned, float(tracks.score[r]))
    meta = dict(tracks.meta)
    meta["cleaned_masks"] = changed
    return recorder.build(video=tracks.video, fps=tracks.fps, num_frames=tracks.num_frames,
                          backend=tracks.backend, meta=meta)


# ----------------------------------------------------------------------------- stitching
def stitch_fragments(tracks: Tracks, max_gap_seconds: float = 1.0, min_iou: float = 0.3,
                     max_center_shift: float = 0.5) -> Tracks:
    """Join a track that ends with one that starts shortly after at the same place.

    Trackers sometimes give a person a new id after a short occlusion or a missed
    detection.  Track ``B`` is appended to track ``A`` when ``B`` starts after ``A``
    ended (no frame in common), within ``max_gap_seconds``, and ``B``'s first box
    overlaps ``A``'s last box (IoU >= ``min_iou``) or its centre moved by less
    than ``max_center_shift`` x the box height.  Candidates are linked greedily,
    shortest gap / best overlap first, each track at most once on either side.
    The merged person keeps the id of the earlier fragment.
    """
    ids = tracks.ids
    if len(ids) < 2:
        return tracks
    max_gap = max(0, int(round(max_gap_seconds * tracks.fps)))
    span = {}
    for oid in ids:
        rec = tracks.records(oid)
        frames = tracks.frame[rec]
        first, last = int(frames.min()), int(frames.max())
        span[oid] = (first, last, tracks.box[rec[np.argmin(frames)]], tracks.box[rec[np.argmax(frames)]])
    candidates = []
    for a in ids:
        a_first, a_last, _, a_box = span[a]
        for b in ids:
            if a == b:
                continue
            b_first, _, b_box, _ = span[b]
            gap = b_first - a_last - 1
            if gap < 0 or gap > max_gap:
                continue
            iou = box_iou(a_box, b_box)
            height = max(a_box[3] - a_box[1], b_box[3] - b_box[1], 1.0)
            ca = np.array([(a_box[0] + a_box[2]) / 2, (a_box[1] + a_box[3]) / 2])
            cb = np.array([(b_box[0] + b_box[2]) / 2, (b_box[1] + b_box[3]) / 2])
            shift = float(np.linalg.norm(ca - cb) / height)
            if iou >= min_iou or shift <= max_center_shift:
                candidates.append((gap, -iou, shift, a, b))
    candidates.sort()
    successor, predecessor = {}, {}
    for gap, neg_iou, shift, a, b in candidates:
        if a in successor or b in predecessor:
            continue
        successor[a] = b
        predecessor[b] = a
    if not successor:
        return tracks
    # follow each chain of fragments from its first track and give every fragment that track's id
    mapping, merges = {}, []
    for oid in ids:
        if oid in predecessor:
            continue
        node = oid
        while node in successor:
            nxt = successor[node]
            mapping[nxt] = oid
            merges.append([int(oid), int(nxt)])
            node = nxt
    out = tracks.relabel(mapping)
    out.meta = dict(tracks.meta)
    out.meta["stitched"] = out.meta.get("stitched", []) + merges
    return out


# ----------------------------------------------------------------------------- selection
#: Default ``min_visible`` (fraction of the video's frames a person must be visible in to be scored).
#: The bone-length score needs usable keypoints in at least 25 % of ALL analysed frames
#: (``hmb.scoring.bone_length.MIN_COVERAGE``) and reports anyone below that as ``unsupported``.  The 3-D
#: keypoint detector does not find a tracked person in every frame the tracker sees it in (occlusion,
#: truncation at the frame border), so the threshold keeps a margin above the 25 % requirement.
MIN_VISIBLE = 0.3


def screen_people(tracks: Tracks, min_duration: float = 1.0, min_visible: float = MIN_VISIBLE,
                  max_people: int = 10, min_height: float = 0.1,
                  min_relative_height: float = 0.25) -> Tuple[List[dict], List[dict]]:
    """:func:`select_people` and the people it leaves out: ``(selected, skipped)``.

    ``skipped`` holds :meth:`Tracks.summary` plus ``reason`` (why the person is
    not scored) for every other tracked person, by id.
    """
    kept, skipped = [], []
    for oid in tracks.ids:
        s = tracks.summary(oid)
        if s["visible_seconds"] + 1e-9 < float(min_duration):
            reason = f"too short: visible {s['visible_seconds']:.2f} s < min duration {float(min_duration):g} s"
        elif s["visible_fraction"] + 1e-9 < float(min_visible):
            reason = (f"too short: visible in {100 * s['visible_fraction']:.0f}% of the frames "
                      f"< min visible {100 * float(min_visible):.0f}%")
        elif s["median_height_fraction"] + 1e-9 < float(min_height):
            reason = (f"too small: median height {100 * s['median_height_fraction']:.0f}% of the frame "
                      f"< {100 * float(min_height):.0f}%")
        else:
            kept.append(s)
            continue
        skipped.append(dict(s, reason=reason))
    if kept and min_relative_height:
        tallest = max(s["median_height_fraction"] for s in kept)
        limit = float(min_relative_height) * tallest
        for s in kept:
            if s["median_height_fraction"] + 1e-9 < limit:
                skipped.append(dict(s, reason=(f"too small: median height {100 * s['median_height_fraction']:.0f}% "
                                               f"of the frame < {100 * float(min_relative_height):.0f}% of the "
                                               f"tallest person's ({100 * tallest:.0f}%)")))
        kept = [s for s in kept if s["median_height_fraction"] + 1e-9 >= limit]
    kept.sort(key=lambda s: (-s["prominence"], s["id"]))
    if max_people is not None and max_people > 0 and len(kept) > int(max_people):
        n = int(max_people)
        for s in kept[n:]:
            skipped.append(dict(s, reason=f"less prominent than the {n} {'person' if n == 1 else 'people'} "
                                          f"kept (max people {n})"))
        kept = kept[: int(max_people)]
    for rank, s in enumerate(kept, 1):
        s["rank"] = rank
    skipped.sort(key=lambda s: s["id"])
    return kept, skipped


def select_people(tracks: Tracks, min_duration: float = 1.0, min_visible: float = MIN_VISIBLE,
                  max_people: int = 10, min_height: float = 0.1,
                  min_relative_height: float = 0.25) -> List[dict]:
    """People worth scoring, most prominent first.

    A person is kept when it is visible for at least ``min_duration`` seconds, in at
    least ``min_visible`` (fraction, default :data:`MIN_VISIBLE` = 0.3) of the video's
    frames, and its median box height is at least ``min_height`` of the frame height
    and at least ``min_relative_height`` of the tallest such person's (distant
    bystanders cannot be reconstructed and are not the subject).  Kept people are
    ranked by prominence (mean over the video of the box area fraction, i.e. large
    and long-visible first) and at most ``max_people`` are returned.  Each entry is
    :meth:`Tracks.summary` plus ``rank`` (1 = most prominent).  :func:`screen_people`
    also returns the people left out, with the reason.
    """
    return screen_people(tracks, min_duration=min_duration, min_visible=min_visible, max_people=max_people,
                         min_height=min_height, min_relative_height=min_relative_height)[0]


def describe_selection(tracks: Tracks, selected: Iterable[dict], skipped: Optional[Iterable[dict]] = None
                       ) -> List[str]:
    """One line per tracked person: frames, visibility, height and whether it is scored (and why not)."""
    lines = []
    chosen = {int(s["id"]) for s in selected}
    reasons = {int(s["id"]): s.get("reason") for s in (skipped or [])}
    for oid in tracks.ids:
        s = tracks.summary(oid)
        tag = "selected" if oid in chosen else "skipped"
        if oid not in chosen and reasons.get(oid):
            tag += f" ({reasons[oid]})"
        lines.append(f"person {oid:>3}: frames {s['first_frame']}-{s['last_frame']}, "
                     f"visible {s['visible_seconds']:.1f}s ({100 * s['visible_fraction']:.0f}%), "
                     f"height {100 * s['median_height_fraction']:.0f}% -> {tag}")
    return lines

