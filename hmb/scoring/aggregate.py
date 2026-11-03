"""Per-frame -> per-video aggregation shared by the HumanScore metrics.

Every metric first measures each analysed frame ``t``: a severity ``m_t`` in [0, 1]
and an abnormal flag ``b_t``.  The paper (App. F.1) turns those into one 0-100 score::

    r = mean_t b_t                      (frequency: how often)
    s = sum_t m_t / max(1, sum_t b_t)   (severity: how bad, on the flagged frames)
    p = L_max / T                       (persistence: longest abnormal run / T)
    D = 0.5 r + 0.3 s + 0.2 p,   score = 100 (1 - clip(D, 0, 1)).

Two details make the score depend on evidence rather than on clip length:

* ``s`` averages the severity of *flagged* frames only, so sub-threshold noise
  cannot accumulate into a penalty on long clips.
* Frames without an observation (no detected person, failed reconstruction)
  are removed before r, s and p are computed; ``coverage`` reports the fraction
  of usable frames so callers can abstain on clips with too little evidence.

Used by :mod:`hmb.scoring.joint_range` and :mod:`hmb.scoring.self_collision`;
:mod:`hmb.scoring.extra_limbs` uses a graded variant of the same formula.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

#: Paper weights (alpha, beta, gamma) of frequency, severity and persistence.
PAPER_WEIGHTS = (0.5, 0.3, 0.2)


@dataclass(frozen=True)
class Aggregate:
    """Result of :func:`aggregate`.

    ``score`` is the 0-100 score (NaN when no frame is valid), ``D`` the penalty in
    [0, 1] and ``r``, ``s``, ``p`` its three terms.  ``num_frames`` counts all input
    frames, ``num_valid`` the observed ones, ``num_flagged`` the flagged valid ones;
    ``longest_run`` is the longest flagged run (in valid frames) and ``coverage`` =
    ``num_valid / num_frames``.
    """

    score: float
    D: float
    r: float
    s: float
    p: float
    num_frames: int
    num_valid: int
    num_flagged: int
    longest_run: int
    coverage: float

    def as_dict(self) -> dict:
        return asdict(self)


def run_bounds(flags: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """``(starts, stops)`` of the runs of True values in a 1-D boolean array (``flags[start:stop]``)."""
    padded = np.concatenate([[False], np.asarray(flags, dtype=bool), [False]])
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return edges[::2], edges[1::2]


def longest_run(flags: np.ndarray) -> int:
    """Length of the longest run of True values in a 1-D boolean array."""
    flags = np.asarray(flags, dtype=bool)
    if flags.size == 0 or not flags.any():
        return 0
    starts, stops = run_bounds(flags)
    return int((stops - starts).max())


def aggregate(
    severity: Sequence[float],
    flags: Optional[Sequence[bool]] = None,
    valid: Optional[Sequence[bool]] = None,
    *,
    threshold: float = 0.0,
    weights: Sequence[float] = PAPER_WEIGHTS,
) -> Aggregate:
    """Aggregate per-frame severities into the paper's 0-100 score.

    Args:
        severity: per-frame ``m_t`` (NaN counts as 0; values are clipped to [0, 1]).
        flags: per-frame ``b_t``. Defaults to ``severity > threshold``.
        valid: per-frame observation mask. Invalid frames are removed before
            computing r, s and p (runs are counted on the valid sequence).
        threshold: used only when ``flags`` is None.
        weights: (alpha, beta, gamma), normalised to sum to one.
    """
    m = np.clip(np.nan_to_num(np.asarray(severity, dtype=float), nan=0.0), 0.0, 1.0)
    total = int(m.size)
    if flags is None:
        b = m > threshold
    else:
        b = np.asarray(flags, dtype=bool)
        if b.shape != m.shape:
            raise ValueError("flags and severity must have the same shape")
    if valid is None:
        v = np.ones_like(b, dtype=bool)
    else:
        v = np.asarray(valid, dtype=bool)
        if v.shape != m.shape:
            raise ValueError("valid and severity must have the same shape")
    m, b = m[v], b[v]
    n = int(m.size)
    w = np.asarray(weights, dtype=float)
    if w.shape != (3,) or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("weights must be three non-negative numbers")
    w = w / w.sum()
    if n == 0:  # nothing observed: no score (callers report the metric as unsupported)
        return Aggregate(score=float("nan"), D=float("nan"), r=float("nan"), s=float("nan"),
                         p=float("nan"), num_frames=total, num_valid=0, num_flagged=0,
                         longest_run=0, coverage=0.0)
    flagged = int(b.sum())
    r = flagged / n
    s = float(m[b].sum() / max(1, flagged))
    run = longest_run(b)
    p = run / n
    D = float(np.clip(w[0] * r + w[1] * s + w[2] * p, 0.0, 1.0))
    return Aggregate(score=100.0 * (1.0 - D), D=D, r=float(r), s=s, p=float(p), num_frames=total,
                     num_valid=n, num_flagged=flagged, longest_run=run, coverage=n / max(1, total))


def ramp(x, low: float, high: float):
    """Linear ramp: 0 at ``low`` and below, 1 at ``high`` and above."""
    x = np.asarray(x, dtype=float)
    if not high > low:
        raise ValueError("ramp requires high > low")
    return np.clip((x - low) / (high - low), 0.0, 1.0)
