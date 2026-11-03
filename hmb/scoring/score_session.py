"""Score the six HumanScore metrics of a prepared session.

This is the scoring stage of the pipeline.  After the reconstruction engines have filled
the session directory, ``hmb.cli`` (and through it ``scripts/humanscore_single.py`` and
``scripts/humanscore_multi.py``) runs this module in its own process and ``hmb.batch``
calls :func:`score_session` directly; :mod:`hmb.report` condenses the report for the
console and ``humanscore.json``.

A session is the directory written by ``hmb.cli`` / ``hmb.batch`` (see ``hmb.layout``)::

    <S>/input/<stem>.mp4, <S>/input/frames.json
    <S>/baseline/<stem>_keypoints.npz, _fitted_model.npz, _opensim_skeleton.mot, _opensim_skeleton_grid.mot
    <S>/gvhmr/<stem>/hmr4d_results.pt
    <S>/cache/multi_limbs/<stem>/hadm_results/output.json
    <S>/bone_fit/<stem>_bone_length.npz              (only for paper scores)

Metric -> scorer:

* extra limbs: :mod:`hmb.scoring.extra_limbs` (graded HADM evidence);
* bone length: :mod:`hmb.scoring.bone_length` (MeTRAbs keypoints, no fit);
* joint range: :mod:`hmb.scoring.joint_range` (canonical angles of the rigid fit);
* self-collision: :mod:`hmb.scoring.self_collision` (penetration depth of the GVHMR mesh);
* kinematic extremes and motion smoothness: the paper implementation
  (``metrics/kinematic_extreme``, ``metrics/motion_smoothness``) via
  :mod:`hmb.scoring.paper_scores`.

Each metric is scored independently.  A metric whose inputs are missing or whose
scorer fails gets ``status`` ``missing_input``, ``unsupported`` or ``error``, a null
score and a one-line ``reason`` (:func:`status_reason`); the other metrics are still
scored.  Tier scores are the mean of their two metrics and ``overall`` the mean of the
six (paper Table 1); both are null unless every contributing metric has a score.

With ``paper_scores=True`` (``--paper-scores``) the report also holds the six scores
of the original paper implementation (``paper_scores``, see :mod:`hmb.scoring.paper_scores`).

The report is written to ``<S>/scores.json``.

Usage::

    python -m hmb.scoring.score_session --session-dir S [--paper-scores]
    python -m hmb.scoring.score_session --sessions-root ROOT [--output scores.csv] [--workers 4]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable, Dict, Optional

#: The six metrics, in the paper's order.
METRICS = ("extra_limbs", "bone_length", "joint_range", "self_collision",
           "kinematic_extremes", "motion_smoothness")
#: The three tiers of paper Table 1 and their two metrics each.
TIERS = {"anatomy": ("extra_limbs", "bone_length"),
         "kinematic": ("joint_range", "self_collision"),
         "kinetic": ("kinematic_extremes", "motion_smoothness")}
OUTPUT_NAME = "scores.json"
#: Metrics scored with the paper implementation (``metrics/<dir>``) in the default report.
PAPER_CODE = {"kinematic_extremes": "metrics/kinematic_extreme", "motion_smoothness": "metrics/motion_smoothness"}
#: Longest one-line reason kept in a report.
REASON_MAX_CHARS = 500


def _finite(x) -> bool:
    """A real, finite number (bools are not scores)."""
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _one_line(text) -> str:
    """First non-empty line of ``text``, shortened to ``REASON_MAX_CHARS``."""
    lines = [line.strip() for line in str(text or "").strip().splitlines() if line.strip()]
    line = lines[0] if lines else ""
    return line if len(line) <= REASON_MAX_CHARS else line[: REASON_MAX_CHARS - 3] + "..."


def _below(value: float, limit: float) -> str:
    """``"0.18 < 0.25"`` with enough decimals that the two numbers differ."""
    for digits in (2, 3, 4):
        a, b = f"{value:.{digits}f}", f"{limit:.{digits}f}"
        if a != b:
            return f"{a} < {b}"
    return f"{value:.4g} < {limit:.4g}"


def _number(details: dict, *keys):
    """First finite value of ``keys`` in ``details`` or its ``aggregate`` sub-dict (None if none)."""
    for source in (details, details.get("aggregate") if isinstance(details.get("aggregate"), dict) else {}):
        for key in keys:
            value = source.get(key)
            if _finite(value):
                return value
    return None


def _unsupported_reason(metric: str, details: Optional[dict]) -> str:
    """Why a scorer returned no finite score, in words, from its details dict."""
    details = details if isinstance(details, dict) else {}
    for key in ("reason", "unsupported_reason"):
        if isinstance(details.get(key), str) and details[key].strip():
            return _one_line(details[key])
    config = details.get("config") if isinstance(details.get("config"), dict) else {}
    coverage, min_coverage = _number(details, "coverage"), config.get("min_coverage")
    if metric == "bone_length":
        if _finite(coverage) and _finite(min_coverage) and coverage < min_coverage:
            return f"keypoint coverage {_below(coverage, min_coverage)} of the analysed frames"
        if "per_segment" in details and not details["per_segment"]:
            frames = config.get("min_frames_per_segment")
            return ("no body segment observed in enough frames"
                    + (f" (at least {frames})" if _finite(frames) else ""))
    frames, valid = _number(details, "num_frames"), _number(details, "num_valid")
    if metric == "self_collision" and frames == 0:
        return "the body mesh has no frames"
    if valid == 0:
        what = {"joint_range": "no frame with a detected person",
                "self_collision": "no usable body-mesh frame"}.get(metric, "no usable frame")
        return what + (f" (0 of {int(frames)} analysed frames)" if _finite(frames) else "")
    if metric in PAPER_CODE:
        return "the paper implementation returned no finite score"
    return "the scorer returned no finite score"


def _missing_reason(exc: BaseException) -> str:
    """``missing input <path>`` from a ``FileNotFoundError`` (or its first line)."""
    if isinstance(exc, OSError) and exc.filename:
        return _one_line(f"missing input {exc.filename}")
    arg = exc.args[0] if len(exc.args) == 1 else None
    if isinstance(arg, Path) or (isinstance(arg, str) and arg.startswith(("/", "~")) and "\n" not in arg):
        return _one_line(f"missing input {arg}")
    return _one_line(exc) or "missing input"


def status_reason(metric: str, entry: Optional[dict]) -> Optional[str]:
    """One-line reason of a metric entry whose ``status`` is not ``ok`` (``None`` when ok).

    ``unsupported``: from the scorer details (e.g. ``keypoint coverage 0.18 < 0.25 of the
    analysed frames`` for bone length); ``missing_input`` / ``error``: the first line of the
    error.  An entry that already carries ``reason`` keeps it.
    """
    entry = entry or {}
    status = entry.get("status")
    if status in (None, "ok"):
        return None
    if isinstance(entry.get("reason"), str) and entry["reason"].strip():
        return _one_line(entry["reason"])
    if status == "unsupported":
        return _unsupported_reason(metric, entry.get("details"))
    text = _one_line(entry.get("error"))
    if status == "missing_input":
        return text or "missing input"
    return text or status


def session_stem(session: Path) -> str:
    """Video stem of a session: the input video, else the keypoints file, else the frame plan."""
    plan = session / "input" / "frames.json"
    videos = sorted(p for p in (session / "input").glob("*") if p.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv")) \
        if (session / "input").is_dir() else []
    if videos:
        return videos[0].stem
    kps = sorted((session / "baseline").glob("*_keypoints.npz")) if (session / "baseline").is_dir() else []
    if kps:
        return kps[0].name[: -len("_keypoints.npz")]
    if plan.is_file():
        return Path(json.loads(plan.read_text())["video"]).stem
    raise FileNotFoundError(f"cannot determine the video stem of session {session}")


# ----------------------------------------------------------------------------- scorers
# One function per metric: (session, stem, opts) -> {"score", "details"}.  A missing input
# raises FileNotFoundError; a NaN score means the metric cannot be scored on this clip.
def _extra_limbs(session: Path, stem: str, opts: dict) -> dict:
    """Extra limbs from the HADM detections (:mod:`hmb.scoring.extra_limbs`)."""
    from hmb.scoring import extra_limbs

    details = extra_limbs.score_session(session, stem)
    return {"score": details["score"], "details": details}


def _bone_length(session: Path, stem: str, opts: dict) -> dict:
    """Bone length from the MeTRAbs keypoints (:mod:`hmb.scoring.bone_length`)."""
    from hmb.scoring import bone_length

    details = bone_length.score_session(session, stem)
    return {"score": details["bone_length_score"], "details": details}


def _joint_range(session: Path, stem: str, opts: dict) -> dict:
    """Joint range from the rigid skeleton fit (:mod:`hmb.scoring.joint_range`)."""
    from hmb.scoring import joint_range

    details = joint_range.score_session(session, stem)
    return {"score": details["score"], "details": details}


def _self_collision(session: Path, stem: str, opts: dict) -> dict:
    """Self-collision from the GVHMR body mesh (:mod:`hmb.scoring.self_collision`)."""
    from hmb.scoring import self_collision

    details = self_collision.score_session(session, backend=opts.get("collision_backend", "auto"), write=True)
    return {"score": details["self_collision_score"], "details": details}


# Paper-implementation results of the session being scored.  Both kinetic metrics (and, with
# paper scores, all six) come from one paper run, so the motion files are read only once; the
# cache holds one session and is cleared at the end of every score_session call.
_PAPER_CACHE: Dict[str, dict] = {}


def _paper_entries(session: Path, stem: str, opts: dict, metrics) -> Dict[str, dict]:
    """Paper-implementation entries of ``metrics`` for ``session``, computed once per scoring call."""
    from hmb.scoring import paper_scores

    key = str(session)
    if key not in _PAPER_CACHE:
        _PAPER_CACHE.clear()
        _PAPER_CACHE[key] = {}
    cache = _PAPER_CACHE[key]
    wanted = [m for m in paper_scores.METRICS if m in metrics and m not in cache]
    if wanted:
        rep = paper_scores.score_session(session, stem, metrics=tuple(wanted), write=False,
                                         use_cuda=opts.get("collision_backend", "auto") != "cpu")
        cache.update(rep["metrics"])
    return cache


def _paper_metric(metric: str, session: Path, stem: str, opts: dict) -> dict:
    """``metric`` scored with the paper implementation (kinematic extremes, motion smoothness)."""
    # compute everything this call will need from the paper code in the first paper run
    needed = set(PAPER_CODE) & set(opts.get("metrics", PAPER_CODE))
    if opts.get("paper_scores"):
        needed |= set(METRICS)
    entry = _paper_entries(session, stem, opts, needed | {metric})[metric]
    # re-raise so that the entry gets the same status and reason as any other metric
    if entry["status"] == "missing_input":
        raise FileNotFoundError(entry.get("error"))
    if entry["status"] == "error":
        raise RuntimeError(entry.get("error"))
    return {"score": entry["score"], "details": {"source": f"{PAPER_CODE[metric]} (paper implementation)",
                                                 "outputs": f"paper_metrics/{metric}/"}}


def _kinematic_extremes(session: Path, stem: str, opts: dict) -> dict:
    """Kinematic extremes from the paper implementation."""
    return _paper_metric("kinematic_extremes", session, stem, opts)


def _motion_smoothness(session: Path, stem: str, opts: dict) -> dict:
    """Motion smoothness from the paper implementation."""
    return _paper_metric("motion_smoothness", session, stem, opts)


SCORERS: Dict[str, Callable[[Path, str, dict], dict]] = {
    "extra_limbs": _extra_limbs,
    "bone_length": _bone_length,
    "joint_range": _joint_range,
    "self_collision": _self_collision,
    "kinematic_extremes": _kinematic_extremes,
    "motion_smoothness": _motion_smoothness,
}


def tiers_and_overall(scores: Dict[str, Optional[float]]) -> tuple:
    """``(tiers, overall)``: tier = mean of its two metrics, overall = mean of the six (null unless all finite)."""
    tiers = {}
    for tier, pair in TIERS.items():
        vals = [scores.get(m) for m in pair]
        tiers[tier] = sum(vals) / 2.0 if all(_finite(v) for v in vals) else None
    all6 = [scores.get(m) for m in METRICS]
    overall = sum(all6) / 6.0 if all(_finite(v) for v in all6) else None
    return tiers, overall


def _paper_section(entries: Dict[str, dict]) -> dict:
    """``paper_scores`` section of the report from the paper-implementation entries."""
    metrics = {}
    for m in METRICS:
        entry = dict(entries.get(m) or {"score": None, "status": "error", "error": "not computed"})
        if entry["status"] == "unsupported":
            entry["reason"] = "the paper implementation returned no finite score"
        elif entry["status"] != "ok":
            entry["reason"] = _one_line(entry.get("error")) or entry["status"]
        metrics[m] = entry
    scores = {m: metrics[m]["score"] if _finite(metrics[m]["score"]) else None for m in METRICS}
    tiers, overall = tiers_and_overall(scores)
    return {"scores": scores, "tiers": tiers, "overall": overall,
            "status": {m: metrics[m]["status"] for m in METRICS},
            "reasons": {m: metrics[m]["reason"] for m in METRICS if metrics[m].get("reason")},
            "metrics": metrics}


def score_session(session_dir, stem: Optional[str] = None, *, metrics=METRICS, paper_scores: bool = False,
                  collision_backend: str = "auto", write: bool = True) -> dict:
    """Score ``metrics`` (default all six) of one session and return the report.

    The report holds ``session``, ``stem``, ``scores``, ``tiers``, ``overall``, ``status``,
    ``reasons`` and per-metric ``metrics`` entries (``score``, ``status``, ``details``,
    ``reason``/``error`` when not ok, ``seconds``); with ``paper_scores`` also the
    ``paper_scores`` section (``scores``, ``tiers``, ``overall``, ``status``, ``reasons``,
    ``metrics``).  ``write`` saves it as ``<S>/scores.json``.
    ``collision_backend`` selects the self-collision triangle test (``auto``: torch on
    the GPU when available, else the CPU; ``cpu`` forces the CPU).
    """
    session = Path(session_dir)
    stem = stem or session_stem(session)
    metrics = tuple(m for m in METRICS if m in metrics)
    opts = {"collision_backend": collision_backend, "paper_scores": bool(paper_scores), "metrics": metrics}
    results: Dict[str, dict] = {}
    try:
        for metric in metrics:
            started = time.perf_counter()
            try:
                out = SCORERS[metric](session, stem, opts)
                score = out["score"]
                status = "ok" if _finite(score) else "unsupported"
                entry = {"score": float(score) if _finite(score) else None, "status": status,
                         "details": out.get("details")}
            except FileNotFoundError as exc:
                entry = {"score": None, "status": "missing_input", "error": str(exc), "reason": _missing_reason(exc)}
            except Exception as exc:  # one broken metric must not hide the others
                entry = {"score": None, "status": "error", "error": f"{type(exc).__name__}: {exc}",
                         "traceback": traceback.format_exc(limit=6)}
            if entry["status"] != "ok":
                entry["reason"] = status_reason(metric, entry)
            entry["seconds"] = time.perf_counter() - started
            results[metric] = entry
        paper = _paper_section(_paper_entries(session, stem, opts, METRICS)) if paper_scores else None
    finally:
        _PAPER_CACHE.clear()
    scores = {m: results[m]["score"] for m in results}
    tiers, overall = tiers_and_overall(scores)
    report = {"session": str(session), "stem": stem, "scores": scores, "tiers": tiers, "overall": overall,
              "status": {m: results[m]["status"] for m in results},
              "reasons": {m: results[m]["reason"] for m in results if results[m].get("reason")},
              "metrics": results}
    if paper is not None:
        report["paper_scores"] = paper
    if write:  # write-then-rename, so a reader never sees a half-written report
        tmp = session / (OUTPUT_NAME + ".part")
        tmp.write_text(json.dumps(report, indent=1, default=_json_default))
        tmp.replace(session / OUTPUT_NAME)
    return report


def _json_default(value):
    """JSON fallback for numpy scalars and arrays (anything else as a string)."""
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except ImportError:
        pass
    return str(value)


def _score_one(args) -> dict:
    """``--sessions-root`` worker: one CSV row (scores, tiers, statuses) or an ``error`` row."""
    session, kwargs = args
    try:
        rep = score_session(session, **kwargs)
        row = {"session": str(session), "stem": rep["stem"], **{m: rep["scores"][m] for m in METRICS},
               **rep["tiers"], "overall": rep["overall"],
               **{f"{m}_status": rep["metrics"][m]["status"] for m in METRICS}}
        if "paper_scores" in rep:
            row.update({f"paper_{m}": rep["paper_scores"]["scores"][m] for m in METRICS})
            row["paper_overall"] = rep["paper_scores"]["overall"]
        return row
    except Exception as exc:
        return {"session": str(session), "error": f"{type(exc).__name__}: {exc}"}


def find_sessions(root: Path) -> list:
    """Session directories directly under ``root`` (those with a frame plan, else with keypoints)."""
    return sorted(p.parent.parent for p in Path(root).glob("*/input/frames.json")) or \
        sorted(p.parent.parent for p in Path(root).glob("*/baseline/*_keypoints.npz"))


def main(argv=None) -> int:
    """Command line (see the module docstring); with ``--session-dir`` the exit status is 1 without an overall score."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--session-dir", help="score one session (writes <S>/scores.json)")
    g.add_argument("--sessions-root", help="score every session under this folder into one CSV")
    ap.add_argument("--stem", help="video stem inside the session (default: found from the session)")
    ap.add_argument("--paper-scores", action="store_true",
                    help="also compute the six scores of the original paper implementation (metrics/)")
    ap.add_argument("--collision-backend", default="auto",
                    help="self-collision triangle test: auto (default), torch, cpu, bvh or numpy")
    ap.add_argument("--output", help="CSV path in --sessions-root mode (default: ROOT/scores.csv)")
    ap.add_argument("--workers", type=int, default=1, help="parallel sessions in --sessions-root mode")
    ap.add_argument("--metrics", nargs="+", default=list(METRICS), choices=METRICS, help="score only these metrics")
    args = ap.parse_args(argv)
    kwargs = {"collision_backend": args.collision_backend, "metrics": tuple(args.metrics),
              "paper_scores": args.paper_scores}
    if args.session_dir:
        rep = score_session(args.session_dir, args.stem, **kwargs)
        summary = {key: rep[key] for key in ("scores", "tiers", "overall", "status", "reasons")}
        if "paper_scores" in rep:
            summary["paper_scores"] = {key: rep["paper_scores"][key] for key in ("scores", "overall", "status")}
        print(json.dumps(summary, indent=1))
        return 0 if rep["overall"] is not None else 1
    sessions = find_sessions(Path(args.sessions_root))
    jobs = [(s, kwargs) for s in sessions]
    if args.workers > 1:
        with ProcessPoolExecutor(args.workers) as pool:
            rows = list(pool.map(_score_one, jobs))
    else:
        rows = [_score_one(j) for j in jobs]
    out = Path(args.output or Path(args.sessions_root) / "scores.csv")
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "session", k != "stem", k))
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"scored {len(rows)} sessions -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
