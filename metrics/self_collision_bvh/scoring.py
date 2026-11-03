"""Video-level aggregation of metric (IV) self-collision (paper implementation).

Turns the per-frame collided-face ratios ``M_t`` computed by ``bvh_metric.py`` into the
video score of the HumanScore paper (supplementary section F.1).  Pure Python, no GPU.
"""

from math import isfinite
from typing import Dict, Iterable


def aggregate_collision_score(
    face_ratios: Iterable[float],
    tau_mild: float = 0.01,
    tau_severe: float = 0.03,
    alpha: float = 0.5,
    beta: float = 0.3,
    gamma: float = 0.2,
    delta: float = 0.0,
) -> Dict[str, float]:
    """Self-collision score from the per-frame collided-face ratios ``M_t`` in [0, 1].

    ::

        m_t = clip((M_t - tau_mild) / (tau_severe - tau_mild), 0, 1)   (m_t > 0: a "positive" frame)
        r_severe = fraction of frames with M_t >= tau_severe
        s = sum_t m_t / max(1, number of positive frames)
        p = (longest run of positive frames) / T
        D = (alpha r_severe + beta s + gamma p + delta r_mild) / (alpha + beta + gamma + delta)
        score = 100 (1 - clip(D, 0, 1))

    Severity and persistence use every positive frame, mild ones included, as the paper
    defines them.  ``delta`` weights the frequency of positive frames (``r_mild``); the
    paper does not use that term, so it defaults to 0.

    Returns the score with its components (``D``, ``r_mild``, ``r_severe``,
    ``severity_mean``, ``continuity``, ``avg_M_t``; ``penetration_frame_rate`` equals
    ``r_severe``).  Raises ``ValueError`` for an empty sequence or invalid settings.
    """
    ratios = [float(value) for value in face_ratios]
    if not ratios:
        raise ValueError("Cannot score an empty mesh sequence.")
    if any(not isfinite(value) or not 0 <= value <= 1 for value in ratios):
        raise ValueError("Collision face ratios must be finite and in [0, 1].")
    if not 0 <= tau_mild < tau_severe <= 1:
        raise ValueError("Require 0 <= tau_mild < tau_severe <= 1.")
    weights = (alpha, beta, gamma, delta)
    if any(not isfinite(weight) or weight < 0 for weight in weights) or sum(weights) <= 0:
        raise ValueError("Collision weights must be finite, nonnegative and sum to more than zero.")

    severity = [min(1.0, max(0.0, (ratio - tau_mild) / (tau_severe - tau_mild)))
                for ratio in ratios]
    positive = [value > 0 for value in severity]
    count = len(ratios)
    r_mild = sum(positive) / count
    r_severe = sum(ratio >= tau_severe for ratio in ratios) / count
    severity_mean = sum(severity) / max(1, sum(positive))

    longest = current = 0
    for value in positive:
        current = current + 1 if value else 0
        longest = max(longest, current)
    continuity = longest / count
    penalty = (alpha * r_severe + beta * severity_mean + gamma * continuity + delta * r_mild) / sum(weights)
    penalty = min(1.0, max(0.0, penalty))
    return {
        "score": 100.0 * (1.0 - penalty),
        "D": penalty,
        "penetration_frame_rate": r_severe,
        "r_mild": r_mild,
        "r_severe": r_severe,
        "severity_mean": severity_mean,
        "continuity": continuity,
        "avg_M_t": sum(ratios) / count,
    }
