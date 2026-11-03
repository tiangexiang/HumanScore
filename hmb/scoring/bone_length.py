"""Metric (II) bone-length consistency, read directly from the MeTRAbs 3-D keypoints.

Paper (App. F.1): for every bone b, ``e_b(t) = |l_b(t) - median_t l_b| / median_t l_b``;
``E = mean_b mean_t e_b``; ``score = 100 (1 - clip(E / tau, 0, 1))``.  HumanScore takes the
segment lengths ``l_b(t)`` from the MeTRAbs ``bml_movi_87`` 3-D output, which contains the
anatomical joint centres (hip, knee, ankle, shoulder, elbow, wrist, pelvis centre ``mhip``,
``Neck``), so no skeleton fit is needed:

1. **Segments** (``SEGMENTS``): thigh, shank, upper arm, forearm (left/right), trunk
   (pelvis centre - neck), hip width and shoulder width.  Feet, hands and head are short
   and poorly observed; they are not scored.
2. **Observation mask.**  A segment is observed in a frame only when a person was detected
   (``confs > 0``) and both endpoints project inside the image (inset by
   ``EDGE_MARGIN_FRAC``).  A truncated body is a framing problem, not an anatomy defect.
   Clips whose usable frames are fewer than ``MIN_COVERAGE`` get no score ("unsupported").
3. **Global-scale normalisation.**  Monocular metric scale is ambiguous, so per frame the
   log body scale ``g(t)`` (median over observed segments of ``log l_b(t) - log L_b``,
   ``L_b`` the per-segment median over time) is removed.  The residual
   ``r_b(t) = log l_b(t) - log L_b - g(t)`` is invariant to a uniform per-frame scaling of
   the skeleton, and the median keeps one stretched limb from being absorbed into ``g``.
4. **Short-glitch suppression.**  ``r_b`` is median-filtered over a centred window of
   ``FILTER_SECONDS`` (time based, so the same physical window at any frame rate): a
   single-frame estimator glitch does not survive, a deformation of two or more frames does.
5. **Score (paper form).**  ``d_b(t) = exp(|r_b(t)|) - 1`` is the relative length deviation;
   ``E`` = mean over segments of the mean over each segment's observed frames, and
   ``score = 100 (1 - clip(E / TAU_MEAN, 0, 1))``.  ``TAU_MEAN`` = 0.45 is wider than the
   paper's tau = 0.15 because per-frame monocular keypoints are never noise-free (the paper
   scores a fitted skeleton); with 0.45 the estimation noise of an ordinary real video
   costs only a few points.

The result also reports the per-segment statistics and the worst frame/segment.
Pure numpy; no GPU.  Input: ``<S>/baseline/<stem>_keypoints.npz`` (detection engine).

Run it on its own::

    python -m hmb.scoring.bone_length --session-dir S
    python -m hmb.scoring.bone_length --keypoints clip_keypoints.npz --image-size 1280 720
"""
from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from hmb.video import output_size

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Centred temporal median window (seconds); 0.125 s = 3 frames at 24 Hz, i.e. the
#: shortest window that removes an isolated frame.
FILTER_SECONDS = 0.125
#: Inset of the image border (fraction of width/height) for the visibility test of
#: segment endpoints (2 % ~ 25 px at 1280 px).
EDGE_MARGIN_FRAC = 0.02
#: Minimum observed segments in a frame to estimate its global scale g(t).
MIN_SEGMENTS_PER_FRAME = 5
#: Paper: at least five valid frames per segment.
MIN_FRAMES_PER_SEGMENT = 5
#: Minimum fraction of analysed frames with a usable observation, else "unsupported".
MIN_COVERAGE = 0.25
#: Mean relative deviation E at which the score reaches 0 (score = 100 (1 - E / TAU_MEAN)),
#: wide enough that ordinary monocular estimation noise costs only a few points.
TAU_MEAN = 0.45
#: Alternating median iterations for (L_b, g(t)); 2-3 are enough to converge.
SCALE_ITERATIONS = 4

#: ``bml_movi_87`` joint order of MeTRAbs (``baseline/monocular_demos/utils.py``).
JOINT_NAMES: Tuple[str, ...] = (
    "backneck", "upperback", "clavicle", "sternum", "umbilicus", "lfronthead", "lbackhead", "lback",
    "lshom", "lupperarm", "lelbm", "lforearm", "lwrithumbside", "lwripinkieside", "lfin", "lasis",
    "lpsis", "lfrontthigh", "lthigh", "lknem", "lankm", "LHeel", "lfifthmetatarsal", "LBigToe",
    "lcheek", "lbreast", "lelbinner", "lwaist", "lthumb", "lfrontinnerthigh", "linnerknee", "lshin",
    "lfirstmetatarsal", "lfourthtoe", "lscapula", "lbum", "rfronthead", "rbackhead", "rback", "rshom",
    "rupperarm", "relbm", "rforearm", "rwrithumbside", "rwripinkieside", "rfin", "rasis", "rpsis",
    "rfrontthigh", "rthigh", "rknem", "rankm", "RHeel", "rfifthmetatarsal", "RBigToe", "rcheek",
    "rbreast", "relbinner", "rwaist", "rthumb", "rfrontinnerthigh", "rinnerknee", "rshin",
    "rfirstmetatarsal", "rfourthtoe", "rscapula", "rbum", "Head", "mhip", "CHip", "Neck", "LAnkle",
    "LElbow", "LHip", "LHand", "LKnee", "LShoulder", "LWrist", "LFoot", "RAnkle", "RElbow", "RHip",
    "RHand", "RKnee", "RShoulder", "RWrist", "RFoot",
)

#: Segment endpoints = MeTRAbs joint centres.
ENDPOINTS: Dict[str, Tuple[str, ...]] = {
    "hip_l": ("LHip",), "knee_l": ("LKnee",), "ankle_l": ("LAnkle",),
    "shoulder_l": ("LShoulder",), "elbow_l": ("LElbow",), "wrist_l": ("LWrist",),
    "hip_r": ("RHip",), "knee_r": ("RKnee",), "ankle_r": ("RAnkle",),
    "shoulder_r": ("RShoulder",), "elbow_r": ("RElbow",), "wrist_r": ("RWrist",),
    "pelvis": ("mhip",), "neck": ("Neck",),
}

#: Scored segments: (proximal, distal endpoint).
SEGMENT_ENDPOINTS: Dict[str, Tuple[str, str]] = {
    "thigh_l": ("hip_l", "knee_l"), "shank_l": ("knee_l", "ankle_l"),
    "upper_arm_l": ("shoulder_l", "elbow_l"), "forearm_l": ("elbow_l", "wrist_l"),
    "thigh_r": ("hip_r", "knee_r"), "shank_r": ("knee_r", "ankle_r"),
    "upper_arm_r": ("shoulder_r", "elbow_r"), "forearm_r": ("elbow_r", "wrist_r"),
    "trunk": ("pelvis", "neck"), "hip_width": ("hip_l", "hip_r"),
    "shoulder_width": ("shoulder_l", "shoulder_r"),
}
SEGMENTS: Tuple[str, ...] = tuple(SEGMENT_ENDPOINTS)


@dataclass(frozen=True)
class BoneLengthConfig:
    """Constants of the metric (the defaults are the HumanScore values)."""

    segments: Tuple[str, ...] = SEGMENTS
    filter_seconds: float = FILTER_SECONDS
    edge_margin_frac: Optional[float] = EDGE_MARGIN_FRAC  # None: no visibility mask
    min_segments_per_frame: int = MIN_SEGMENTS_PER_FRAME
    min_frames_per_segment: int = MIN_FRAMES_PER_SEGMENT
    min_coverage: float = MIN_COVERAGE
    tau_mean: float = TAU_MEAN

    def as_dict(self) -> dict:
        """JSON-friendly copy (stored as ``config`` in the result)."""
        d = asdict(self)
        d["segments"] = list(self.segments)
        return d


DEFAULT_CONFIG = BoneLengthConfig()


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def endpoint_positions(points: np.ndarray, endpoints: Mapping[str, Tuple[str, ...]] = ENDPOINTS,
                       names: Sequence[str] = JOINT_NAMES) -> Dict[str, np.ndarray]:
    """Mean position of each endpoint's markers. ``points`` (T, J, D) with D = 2 or 3."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 3 or points.shape[1] != len(names):
        raise ValueError(f"points must be (T, {len(names)}, D), got {points.shape}")
    idx = {n: i for i, n in enumerate(names)}
    return {e: points[:, [idx[m] for m in markers]].mean(axis=1) for e, markers in endpoints.items()}


def segment_lengths(points3d: np.ndarray, cfg: BoneLengthConfig = DEFAULT_CONFIG,
                    names: Sequence[str] = JOINT_NAMES) -> np.ndarray:
    """(T, S) Euclidean segment lengths (input units) for ``cfg.segments``."""
    ends = endpoint_positions(points3d, ENDPOINTS, names)
    cols = [np.linalg.norm(ends[SEGMENT_ENDPOINTS[s][0]] - ends[SEGMENT_ENDPOINTS[s][1]], axis=-1)
            for s in cfg.segments]
    return np.stack(cols, axis=1)


def segment_visibility(points2d: Optional[np.ndarray], image_size: Optional[Sequence[float]],
                       cfg: BoneLengthConfig = DEFAULT_CONFIG,
                       names: Sequence[str] = JOINT_NAMES) -> Optional[np.ndarray]:
    """(T, S) True when both endpoints project inside the image (inset by the margin).

    Returns None when 2-D keypoints or the image size are unknown (no mask).
    """
    if points2d is None or image_size is None or cfg.edge_margin_frac is None:
        return None
    w, h = float(image_size[0]), float(image_size[1])
    mx, my = cfg.edge_margin_frac * w, cfg.edge_margin_frac * h
    ends = endpoint_positions(points2d, ENDPOINTS, names)
    inside = {e: np.isfinite(p).all(-1) & (p[:, 0] >= mx) & (p[:, 0] <= w - mx) & (p[:, 1] >= my) & (p[:, 1] <= h - my)
              for e, p in ends.items()}
    return np.stack([inside[SEGMENT_ENDPOINTS[s][0]] & inside[SEGMENT_ENDPOINTS[s][1]] for s in cfg.segments], axis=1)


# ---------------------------------------------------------------------------
# Normalisation and filtering
# ---------------------------------------------------------------------------
def _masked_median(x: np.ndarray, mask: np.ndarray, axis: int) -> np.ndarray:
    """Median of ``x`` over ``axis`` using only entries where ``mask`` is True (NaN if none)."""
    y = np.where(mask, x, np.nan)
    with np.errstate(all="ignore"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.nanmedian(y, axis=axis)


def log_residuals(lengths: np.ndarray, mask: np.ndarray, min_segments: int = MIN_SEGMENTS_PER_FRAME,
                  iterations: int = SCALE_ITERATIONS) -> Dict[str, np.ndarray]:
    """Per-frame, per-segment log length residual after removing the clip's per-segment
    reference length (median) and the per-frame global scale.

    Returns ``residual`` (T, S) (NaN where unobserved), ``log_scale`` (T,), ``reference``
    (S,) (median length) and the updated ``mask`` (frames with fewer than
    ``min_segments`` observed segments have no global scale and are dropped).
    """
    lengths = np.asarray(lengths, dtype=float)
    mask = np.asarray(mask, dtype=bool) & np.isfinite(lengths) & (lengths > 0)
    logl = np.log(np.where(mask, lengths, 1.0))
    T, S = lengths.shape
    g = np.zeros(T)
    frame_ok = mask.sum(axis=1) >= min_segments
    mask = mask & frame_ok[:, None]
    # alternate: reference lengths given the frame scales, then frame scales given the references
    for _ in range(max(1, iterations)):
        ref = _masked_median(logl - g[:, None], mask, axis=0)
        dev = logl - np.nan_to_num(ref)[None, :]
        g = np.nan_to_num(_masked_median(dev, mask, axis=1))
    ref = _masked_median(logl - g[:, None], mask, axis=0)
    residual = np.where(mask, logl - np.nan_to_num(ref)[None, :] - g[:, None], np.nan)
    return {"residual": residual, "log_scale": g, "reference": np.exp(ref), "mask": mask}


def temporal_median(values: np.ndarray, mask: np.ndarray, timestamps: np.ndarray,
                    window_seconds: float) -> np.ndarray:
    """Centred running median of each column over observed samples within
    +-window/2 seconds (NaN where the sample itself is unobserved)."""
    values = np.asarray(values, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if window_seconds is None or window_seconds <= 0:
        return np.where(mask, values, np.nan)
    t = np.asarray(timestamps, dtype=float)
    T = t.size
    if T == 0:
        return values.copy()
    half = 0.5 * window_seconds + 1e-9
    lo = np.searchsorted(t, t - half, side="left")
    hi = np.searchsorted(t, t + half, side="right")
    # gather every sample's window into a padded (T, K, S) array and take one masked median
    K = int((hi - lo).max())
    idx = lo[:, None] + np.arange(K)[None, :]                     # (T, K) neighbour indices
    inside = idx < hi[:, None]
    idx = np.minimum(idx, T - 1)
    win = values[idx]                                              # (T, K, S)
    wm = mask[idx] & inside[:, :, None]
    med = _masked_median(win, wm, axis=1)
    return np.where(mask, med, np.nan)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def frame_timestamps(n: int, timestamps=None, fps: Optional[float] = None) -> np.ndarray:
    """``timestamps`` when they are ``n`` strictly increasing finite values, else ``arange(n) / fps``."""
    if timestamps is not None:
        t = np.asarray(timestamps, dtype=float)
        if t.shape == (n,) and np.isfinite(t).all() and (n < 2 or np.all(np.diff(t) > 0)):
            return t
    if fps is None or not np.isfinite(fps) or fps <= 0:
        raise ValueError("need strictly increasing timestamps or a positive fps")
    return np.arange(n) / float(fps)


def analyze(points3d, confs=None, points2d=None, image_size=None, timestamps=None, fps=None,
            cfg: BoneLengthConfig = DEFAULT_CONFIG, names: Sequence[str] = JOINT_NAMES) -> dict:
    """All intermediate per-frame quantities (for scoring and diagnostics).

    Adds ``deviation`` (T, S) relative length deviations (NaN where not scored),
    ``frame_deviation`` (T,) worst segment per frame and ``valid`` (T,) frames with
    at least one scored segment.
    """
    points3d = np.asarray(points3d, dtype=float)
    T = points3d.shape[0]
    t = frame_timestamps(T, timestamps, fps)
    lengths = segment_lengths(points3d, cfg, names)
    observed = np.isfinite(lengths) & (lengths > 0)
    if confs is not None:
        c = np.asarray(confs, dtype=float).reshape(T, -1)
        observed &= ((c > 0) & np.isfinite(c)).any(axis=1)[:, None]
    visible = segment_visibility(points2d, image_size, cfg, names)
    if visible is not None:
        observed &= visible
    # segments with too few observed frames are not scored at all
    seg_ok = observed.sum(axis=0) >= cfg.min_frames_per_segment
    observed &= seg_ok[None, :]
    res = log_residuals(lengths, observed,
                        min_segments=min(cfg.min_segments_per_frame, max(1, int(seg_ok.sum()))))
    mask = res["mask"]
    filtered = temporal_median(res["residual"], mask, t, cfg.filter_seconds)
    dev = np.where(mask, np.expm1(np.abs(filtered)), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        frame_dev = np.nanmax(np.where(mask, dev, np.nan), axis=1)
    valid = mask.any(axis=1) & np.isfinite(frame_dev)
    return {"timestamps": t, "lengths": lengths, "observed": observed, "visible": visible, "mask": mask,
            "residual": res["residual"], "filtered": filtered, "log_scale": res["log_scale"],
            "reference": res["reference"], "segments": list(cfg.segments), "num_frames": T,
            "num_segments": lengths.shape[1], "deviation": dev, "frame_deviation": frame_dev, "valid": valid}


def mean_deviation(a: dict, cfg: BoneLengthConfig = DEFAULT_CONFIG) -> Tuple[float, Dict[str, dict]]:
    """Paper statistic E = mean over segments of the per-segment mean relative
    deviation (over that segment's observed frames), plus per-segment details."""
    dev, mask = a["deviation"], a["mask"]
    per_seg, seg_means = {}, []
    for j, name in enumerate(a["segments"]):
        m = mask[:, j] & np.isfinite(dev[:, j])
        if m.sum() < cfg.min_frames_per_segment:
            continue
        dj = dev[m, j]
        seg_means.append(float(dj.mean()))
        per_seg[name] = {"reference_length": float(a["reference"][j]), "frames": int(m.sum()),
                         "mean_deviation": float(dj.mean()), "p95_deviation": float(np.percentile(dj, 95)),
                         "max_deviation": float(dj.max())}
    if not seg_means:
        return float("nan"), per_seg
    return float(np.mean(seg_means)), per_seg


def score_from_analysis(a: dict, cfg: BoneLengthConfig = DEFAULT_CONFIG) -> dict:
    """0-100 score ``100 (1 - clip(E / tau_mean, 0, 1))`` and diagnostics from :func:`analyze`.

    The score is NaN ("unsupported") when fewer than ``min_coverage`` of the frames are
    usable or no segment is observed in ``min_frames_per_segment`` frames.
    """
    valid = a["valid"]
    T = a["num_frames"]
    coverage = float(valid.mean()) if T else 0.0
    frame_dev = np.nan_to_num(a["frame_deviation"], nan=0.0)
    E, per_seg = mean_deviation(a, cfg)
    supported = coverage >= cfg.min_coverage and bool(per_seg)
    nan = float("nan")
    score = 100.0 * (1.0 - float(np.clip(E / cfg.tau_mean, 0.0, 1.0))) if supported else nan
    vd = frame_dev[valid]
    dev, mask = a["deviation"], a["mask"]
    worst = None
    if valid.any():
        i = int(np.nanargmax(np.where(valid, frame_dev, -np.inf)))
        j = int(np.nanargmax(np.where(mask[i], np.nan_to_num(dev[i], nan=-np.inf), -np.inf)))
        worst = {"frame": i, "time": float(a["timestamps"][i]), "segment": a["segments"][j],
                 "deviation": float(dev[i, j])}
    return {
        "metric": "bone_length",
        "bone_length_score": float(score),
        "score": float(score),
        "supported": bool(supported),
        "mean_relative_deviation": E,
        "coverage": coverage,
        "frame_deviation_p50": float(np.median(vd)) if vd.size else nan,
        "frame_deviation_p95": float(np.percentile(vd, 95)) if vd.size else nan,
        "frame_deviation_max": float(vd.max()) if vd.size else nan,
        "global_scale_range": float(np.ptp(a["log_scale"][valid])) if valid.any() else nan,
        "worst": worst,
        "per_segment": per_seg,
        "config": cfg.as_dict(),
    }


def score_arrays(points3d, confs=None, points2d=None, image_size=None, timestamps=None, fps=None,
                 cfg: BoneLengthConfig = DEFAULT_CONFIG, names: Sequence[str] = JOINT_NAMES) -> dict:
    """Score raw MeTRAbs arrays. ``points3d`` (T, 87, 3) in any length unit."""
    return score_from_analysis(analyze(points3d, confs, points2d, image_size, timestamps, fps, cfg, names), cfg)


def load_keypoints(path) -> dict:
    """Read a ``<stem>_keypoints.npz`` written by the detection engine."""
    with np.load(path, allow_pickle=False) as d:
        out = {k: d[k] for k in d.files if k in ("keypoints3d", "keypoints2d", "confs", "timestamps", "fps",
                                                 "image_size", "frame_indices", "camera_intrinsics_px")}
    return out


def _file_image_size(kp: dict) -> Optional[list]:
    """``[width, height]`` stored in a keypoints file, else None."""
    if "image_size" not in kp:
        return None
    return [float(x) for x in np.asarray(kp["image_size"]).ravel()[:2]]


def _file_fps(kp: dict) -> Optional[float]:
    """Positive finite frame rate stored in a keypoints file, else None."""
    if "fps" not in kp:
        return None
    f = float(np.asarray(kp["fps"]).ravel()[0])
    return f if np.isfinite(f) and f > 0 else None


def score_keypoints_file(path, image_size=None, fps=None, timestamps=None,
                         cfg: BoneLengthConfig = DEFAULT_CONFIG) -> dict:
    """Score one ``<stem>_keypoints.npz`` (image size, timestamps and fps from the file unless given)."""
    kp = load_keypoints(path)
    size = image_size if image_size is not None else _file_image_size(kp)
    ts = timestamps if timestamps is not None else kp.get("timestamps")
    if fps is None:
        fps = _file_fps(kp)
    out = score_arrays(kp["keypoints3d"], kp.get("confs"), kp.get("keypoints2d"), size, ts, fps, cfg)
    out["keypoints"] = str(path)
    out["image_size"] = list(size) if size is not None else None
    return out


def _plan(session: Path) -> Optional[dict]:
    """The session's frame plan ``<S>/input/frames.json`` (None when absent)."""
    p = session / "input" / "frames.json"
    return json.loads(p.read_text()) if p.is_file() else None


def score_session(session_dir, stem: Optional[str] = None, cfg: BoneLengthConfig = DEFAULT_CONFIG) -> dict:
    """Score ``<S>/baseline/<stem>_keypoints.npz`` of a session.

    Timestamps: keypoints ``timestamps`` -> ``<S>/input/frames.json`` -> video fps.
    Image size: keypoints ``image_size`` -> the plan's decoded size -> the input video.
    Without an image size the visibility mask is skipped (``visibility_mask=False``).
    """
    session = Path(session_dir)
    if stem is None:
        kps = sorted((session / "baseline").glob("*_keypoints.npz"))
        if not kps:
            raise FileNotFoundError(f"no keypoints in {session / 'baseline'}")
        path = kps[0]
        stem = path.name[: -len("_keypoints.npz")]
    else:
        path = session / "baseline" / f"{stem}_keypoints.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    kp = load_keypoints(path)
    plan = _plan(session)
    size = _file_image_size(kp)
    if size is None and plan is not None:
        # the keypoints were detected on the decoded frames, i.e. after the plan's resize
        size = list(output_size(int(plan["width"]), int(plan["height"]), plan.get("max_side")))
    fps = None
    ts = kp.get("timestamps")
    if ts is None and plan is not None and len(plan.get("timestamps", [])) == len(kp["keypoints3d"]):
        ts = plan["timestamps"]
    if ts is None or size is None:
        videos = sorted((session / "input").glob(f"{stem}.*")) if (session / "input").is_dir() else []
        if videos:
            info = _probe(videos[0])
            fps = info.get("fps")
            if size is None and info.get("width"):
                size = [info["width"], info["height"]]
    if fps is None:
        fps = _file_fps(kp)
    out = score_arrays(kp["keypoints3d"], kp.get("confs"), kp.get("keypoints2d"), size, ts, fps, cfg)
    out.update({"keypoints": str(path), "image_size": size, "visibility_mask": size is not None})
    return out


def _probe(video: Path) -> dict:
    """``{fps, width, height}`` of a video read with OpenCV (empty when OpenCV is missing)."""
    try:
        import cv2

        cap = cv2.VideoCapture(str(video))
        info = {"fps": float(cap.get(cv2.CAP_PROP_FPS) or 0) or None,
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
        cap.release()
        return info
    except Exception:  # pragma: no cover - cv2 missing
        return {}


def main(argv=None) -> int:
    """Command line: score a session or one keypoints file; print the score, coverage and worst frame."""
    import argparse

    ap = argparse.ArgumentParser(description="HumanScore bone-length consistency from MeTRAbs keypoints")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--session-dir", help="session directory with baseline/<stem>_keypoints.npz")
    g.add_argument("--keypoints", help="a <stem>_keypoints.npz file")
    ap.add_argument("--stem", help="video stem inside the session (default: the first keypoints file)")
    ap.add_argument("--image-size", type=float, nargs=2, help="width height (with --keypoints)")
    ap.add_argument("--fps", type=float, help="frame rate (with --keypoints, when it has no timestamps)")
    ap.add_argument("--output", help="write the full result JSON here")
    args = ap.parse_args(argv)
    if args.session_dir:
        out = score_session(args.session_dir, args.stem)
    else:
        out = score_keypoints_file(args.keypoints, image_size=args.image_size, fps=args.fps)
    text = json.dumps(out, indent=1, default=lambda v: v.item() if hasattr(v, "item") else str(v))
    if args.output:
        Path(args.output).write_text(text)
    print(json.dumps({k: out[k] for k in ("bone_length_score", "coverage", "worst")}, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
