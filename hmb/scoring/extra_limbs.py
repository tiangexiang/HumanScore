"""Metric (I) extra limbs: graded HADM evidence of extra hands, arms, legs or feet.

Evidence (as in the paper, App. F.1): the HADM-G human-artifact detector runs on
every analysed frame (``hmb.hadm_engine``), and ``M_t`` is the highest confidence
over its four classes "human with extra {hand, arm, leg, feet}" in frame ``t``.

Per-frame membership and aggregation::

    m_t = clip((M_t - TAU_FLOOR) / (TAU_SEVERE - TAU_FLOOR), 0, 1)     (0 at or below TAU_FLOOR)
    r   = sum_t m_t / T                      (graded frequency)
    s   = sum_t m_t^2 / sum_t m_t            (mean severity of the flagged mass)
    p   = max over runs (m_t > 0) of sum m_t / T   (graded persistence)
    D   = 0.5 r + 0.3 s + 0.2 p,   score = 100 (1 - clip(D, 0, 1))

This is the paper's r/s/p aggregation with each hard flag ``b_t`` replaced by its
graded membership ``m_t``: a weak 0.005 detection counts much less than a
confident 0.9 one.  When every flagged frame is at full severity the formula is
identical to the paper's.

Constants: ``TAU_FLOOR`` = 0.001 is HADM's output threshold (the smallest
confidence it can report); ``TAU_SEVERE`` = 0.03 is the paper's severe threshold.

Input: ``<S>/cache/multi_limbs/<stem>/hadm_results/output.json`` (written by the HADM
engine).  Output: a result dict (``score``, ``r``, ``s``, ``p``, ...).

Run it on its own::

    python -m hmb.scoring.extra_limbs --session-dir S
    python -m hmb.scoring.extra_limbs --output-json path/to/output.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from hmb.scoring.aggregate import PAPER_WEIGHTS, run_bounds

#: The four HADM-G classes that make up the evidence ``M_t``.
EXTRA_LIMB_CLASSES = (
    "human with extra hand",
    "human with extra arm",
    "human with extra leg",
    "human with extra feet",
)
#: HADM runs with ``test_score_thresh = 0.001``: the smallest confidence it can report.
HADM_SCORE_FLOOR = 0.001
#: Membership is 0 at or below this confidence (the detector floor).
TAU_FLOOR = HADM_SCORE_FLOOR
#: Membership reaches 1 at this confidence (the paper's severe threshold).
TAU_SEVERE = 0.03


def graded_rsp(M: Sequence[float], tau: float = TAU_FLOOR, tau_severe: float = TAU_SEVERE,
               weights: Sequence[float] = PAPER_WEIGHTS) -> dict:
    """Paper r/s/p with each hard flag replaced by its graded membership ``m_t``.

    ``m_t = clip((M_t - tau) / (tau_severe - tau), 0, 1)`` (0 at or below ``tau``);
    ``r = sum m / T``, ``s = sum m^2 / sum m``, ``p = max over runs (m > 0) of sum m / T``,
    ``D = 0.5 r + 0.3 s + 0.2 p``, score ``100 (1 - D)``.
    """
    M = np.asarray(M, dtype=float)
    if M.ndim != 1 or M.size == 0:
        raise ValueError("evidence must be a non-empty 1-D sequence")
    m = np.where(M > tau, np.clip((M - tau) / (tau_severe - tau), 0.0, 1.0), 0.0)
    T = M.size
    mass = float(m.sum())
    r = mass / T
    s = float((m * m).sum() / mass) if mass > 0 else 0.0
    p = 0.0
    on = m > 0
    if on.any():
        # persistence of a run = its membership mass, so a run of weak detections counts little
        p = max(float(m[a:z].sum()) for a, z in zip(*run_bounds(on))) / T
    w = np.asarray(weights, float) / float(np.sum(weights))
    D = float(np.clip(w[0] * r + w[1] * s + w[2] * p, 0.0, 1.0))
    return {"score": 100.0 * (1.0 - D), "D": D, "r": r, "s": s, "p": p, "num_frames": int(T),
            "num_flagged": int(on.sum()), "max_evidence": float(M.max())}


def frame_evidence(path) -> np.ndarray:
    """``M_t`` per frame of a HADM ``output.json``, frames in file-name order (as the paper code)."""
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict) or not data:
        raise ValueError("HADM output must contain at least one frame; cannot score empty inference.")
    classes = set(EXTRA_LIMB_CLASSES)
    M = np.zeros(len(data))
    for i, key in enumerate(sorted(data)):
        rec = data[key] or {}
        scores, labels = rec.get("score", []) or [], rec.get("class", []) or []
        if len(scores) != len(labels):
            raise ValueError(f"Mismatched HADM score/class arrays for frame {key}.")
        for conf, cls in zip(scores, labels):
            if cls not in classes:
                continue
            conf = float(conf)
            if not (0.0 <= conf <= 1.0) or not math.isfinite(conf):
                raise ValueError(f"Invalid HADM confidence {conf} for frame {key}.")
            M[i] = max(M[i], conf)
    return M


def find_hadm_output(session_dir, stem: Optional[str] = None) -> Path:
    """``<S>/cache/multi_limbs/<stem>/hadm_results/output.json``.

    Also accepts the layout of the original HADM runner, ``hadm_results/<timestamp>/output.json``
    (the newest one is used).  Without ``stem`` the first video folder with an output is used.
    """
    root = Path(session_dir) / "cache" / "multi_limbs"
    if stem:
        stems = [stem]
    else:
        stems = sorted(p.name for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
    for name in stems:
        results = root / name / "hadm_results"
        direct = results / "output.json"
        if direct.is_file():
            return direct
        nested = sorted(results.glob("*/output.json"), key=lambda p: p.stat().st_mtime)
        if nested:
            return nested[-1]
    raise FileNotFoundError(f"no HADM output.json under {root}" + (f" for {stem}" if stem else ""))


def score_session(session_dir, stem: Optional[str] = None) -> dict:
    """Extra-limbs score of a session from its HADM detections (details dict with ``score``)."""
    out = find_hadm_output(session_dir, stem)
    res = graded_rsp(frame_evidence(out))
    res.update({"metric": "extra_limbs", "hadm_output": str(out), "tau": TAU_FLOOR, "tau_severe": TAU_SEVERE,
                "extra_limbs_score": res["score"]})
    return res


def main(argv=None) -> int:
    """Command line: score a session or a single HADM ``output.json`` and print the result JSON."""
    parser = argparse.ArgumentParser(description="HumanScore extra-limbs score from HADM detections")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--session-dir", help="session directory with cache/multi_limbs/<stem>/hadm_results/")
    source.add_argument("--output-json", help="a HADM output.json")
    parser.add_argument("--stem", help="video stem inside the session (default: the only one)")
    parser.add_argument("--out", help="also write the result JSON here")
    args = parser.parse_args(argv)
    if args.session_dir:
        result = score_session(args.session_dir, args.stem)
    else:
        result = graded_rsp(frame_evidence(args.output_json))
        result.update({"metric": "extra_limbs", "hadm_output": args.output_json,
                       "extra_limbs_score": result["score"]})
    text = json.dumps(result, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
