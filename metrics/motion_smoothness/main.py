# -*- coding: utf-8 -*-
"""Metric (VI) motion smoothness: the original HumanScore paper implementation.

Flags frames with joint angular accelerations that no human produces (limits in
``definition.py``).  HumanScore uses this code for its motion-smoothness score, both by
default and in the paper scores: ``hmb/scoring/paper_scores.py`` calls
:func:`compute_and_save_for_mot` on ``<S>/baseline/<stem>_opensim_skeleton_grid.mot``, the
fitted skeleton sampled on a uniform 24 Hz grid (finite differences need a uniform step).

Definition (HumanScore paper, supplementary section F.1)::

    per frame t, angular acceleration a_j(t) (deg/s^2) of every coordinate j:
        u_j(t) = clip((|a_j(t)| - amax_j) / (EXCESS_FRACTION_ACC * amax_j), 0, 1)
        m_t = mean_j u_j(t),   b_t = m_t > FRAME_VIOLATION_THRESHOLD
    r = mean_t b_t,   s = sum_t m_t / max(1, sum_t b_t),   p = (longest run of b_t) / T
    score = 100 * (1 - clip(0.5 r + 0.3 s + 0.2 p, 0, 1))

Only the acceleration term is implemented; the paper's optional jerk-energy term needs
calibrated per-coordinate limits and is left out (``debug.jerk_energy_included = false``).

Run on its own (from the repository root)::

    python -m metrics.motion_smoothness.main --mot_path S/baseline/clip_opensim_skeleton_grid.mot --output_dir out

Without ``--mot_path`` every ``.mot`` in ``--mot_dir`` is scored.  Outputs in ``output_dir``:
``<video>_msmooth.json`` and ``.txt`` per video and ``msmooth_scores_summary.csv``.
"""

import os
import json
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd

from .definition import (
    joints_angular_acceleration_limits,
    get_joint_weight,
)
from .utils import (
    preload_motion_data,
    PreloadedMotionData,
)

# Default paths (relative to the repo root)
DEFAULT_MOT_DIR = os.path.join("results", "baseline")
DEFAULT_OUTPUT_DIR = os.path.join("results", "metrics", "motion_smoothness")

# Scoring parameters
FRAME_VIOLATION_THRESHOLD = 0.05    # a frame with m_t > 0.05 is flagged as non-smooth
EXCESS_FRACTION_ACC = 0.5           # exceeding the limit by 50 % of it gives full severity


@dataclass
class MotionSmoothnessResult:
    """Result of one video; written as ``<video>_msmooth.json``."""

    video_name: str
    msmooth_score: float
    frame_violation_rate: float
    mean_violation_severity: float
    max_consecutive_violation_ratio: float
    per_joint_violation_rate: Dict[str, float]
    per_joint_mean_excess_deg_s2: Dict[str, float]
    debug: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _longest_consecutive_true(mask: np.ndarray) -> int:
    """Length of the longest run of True values."""
    max_run = cur = 0
    for v in mask:
        if v:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 0
    return max_run


# ========== Frame-level severity ==========

def _frame_severity(
    acc_deg_s2: np.ndarray,
    joint_names: List[str],
) -> (float, Dict[str, float]):
    """Frame severity m_t in [0, 1] and the per-joint excess over the limit (deg/s^2).

    ``acc_deg_s2``: (J,) angular acceleration per joint for the current frame (deg/s^2).
    Every joint with a limit counts in the mean, also when it is not violated.
    """
    weighted_sum = 0.0
    weight_sum = 0.0
    per_joint_excess: Dict[str, float] = {}

    for j, name in enumerate(joint_names):
        limits = joints_angular_acceleration_limits.get(name)
        if limits is None:
            continue

        amax = max(abs(limits[0]), abs(limits[1]))
        if amax <= 0.0:
            continue

        val = float(abs(acc_deg_s2[j]))
        excess = max(0.0, val - amax)

        # Values >= amax * EXCESS_FRACTION_ACC are clamped to severity 1
        norm = excess / (amax * EXCESS_FRACTION_ACC)
        norm = float(max(0.0, min(1.0, norm)))

        w = get_joint_weight(name)
        if w <= 0.0:
            continue

        weighted_sum += w * norm
        weight_sum += w
        per_joint_excess[name] = excess

    if weight_sum == 0.0:
        return 0.0, per_joint_excess

    return weighted_sum / weight_sum, per_joint_excess


# ========== Video-level scoring ==========

def compute_msmooth_score(
    video_name: str,
    data: PreloadedMotionData,
    alpha: float = 0.5,
    beta: float = 0.3,
    gamma: float = 0.2,
    frame_violation_threshold: float = FRAME_VIOLATION_THRESHOLD,
) -> MotionSmoothnessResult:
    """Motion-smoothness score of one video from its accelerations (see the module docstring).

    Raises ``ValueError`` when there are no finite accelerations of a coordinate with a limit.
    """
    joint_acc = data.joint_acc
    joint_names = data.joint_names

    if joint_acc is None or joint_acc.size == 0:
        raise ValueError("No acceleration data; motion smoothness is unavailable")
    if not np.isfinite(joint_acc).all():
        raise ValueError("Joint accelerations must be finite")
    if not any(name in joints_angular_acceleration_limits for name in joint_names):
        raise ValueError("No recognized joint angular accelerations")

    T, J = joint_acc.shape
    m_t = np.zeros(T, dtype=float)
    bad_mask = np.zeros(T, dtype=bool)

    joint_violation_counts: Dict[str, int] = {}
    joint_excess_values: Dict[str, List[float]] = {}

    for t in range(T):
        # rad/s^2 -> deg/s^2
        acc_deg_s2 = np.degrees(joint_acc[t])
        m, per_excess = _frame_severity(acc_deg_s2, joint_names)
        m = float(max(0.0, min(1.0, m)))
        m_t[t] = m
        bad_mask[t] = m > frame_violation_threshold

        for name, exc in per_excess.items():
            if exc > 0:
                joint_violation_counts[name] = joint_violation_counts.get(name, 0) + 1
                joint_excess_values.setdefault(name, []).append(exc)

    if T > 0:
        frame_violation_rate = float(bad_mask.mean())
        longest_run = _longest_consecutive_true(bad_mask)
        max_consecutive_violation_ratio = float(longest_run / T)
    else:
        frame_violation_rate = 0.0
        max_consecutive_violation_ratio = 0.0

    mean_violation_severity = float(m_t.sum() / max(1, int(bad_mask.sum())))

    # Combined penalty
    D = (
        alpha * frame_violation_rate
        + beta * mean_violation_severity
        + gamma * max_consecutive_violation_ratio
    )
    D = float(max(0.0, min(1.0, D)))
    msmooth_score = float((1.0 - D) * 100.0)

    # Per-joint summary
    per_joint_violation_rate: Dict[str, float] = {}
    per_joint_mean_excess: Dict[str, float] = {}
    for name in joints_angular_acceleration_limits.keys():
        cnt = joint_violation_counts.get(name, 0)
        per_joint_violation_rate[name] = float(cnt / T) if T > 0 else 0.0
        if cnt > 0:
            per_joint_mean_excess[name] = float(
                np.mean(joint_excess_values.get(name, [0.0]))
            )
        else:
            per_joint_mean_excess[name] = 0.0

    debug = {
        "D": D,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "frame_violation_threshold": frame_violation_threshold,
        "scoring_components": ["joint_angular_acceleration"],
        "jerk_energy_included": False,
        "derivative_method": "central_with_one_sided_endpoints",
        "aggregation_weights": "unit_per_dof",
    }

    return MotionSmoothnessResult(
        video_name=video_name,
        msmooth_score=msmooth_score,
        frame_violation_rate=frame_violation_rate,
        mean_violation_severity=mean_violation_severity,
        max_consecutive_violation_ratio=max_consecutive_violation_ratio,
        per_joint_violation_rate=per_joint_violation_rate,
        per_joint_mean_excess_deg_s2=per_joint_mean_excess,
        debug=debug,
    )


# ========== Single-file evaluation & storage ==========

def compute_and_save_for_mot(
    mot_path: str,
    output_dir: str,
    alpha: float,
    beta: float,
    gamma: float,
    frame_violation_threshold: float,
    save_detail_txt: bool = True,
    preloaded: Optional[PreloadedMotionData] = None,
) -> Optional[MotionSmoothnessResult]:
    """Score one ``.mot`` file and write ``<video>_msmooth.json`` (and ``.txt``) into ``output_dir``.

    ``preloaded`` skips loading and differentiating the motion again.  Returns ``None`` when
    the motion cannot be loaded.
    """
    video_name = os.path.splitext(os.path.basename(mot_path))[0]

    try:
        if preloaded is None:
            preloaded = preload_motion_data(mot_path)
    except Exception as e:
        print(f"[WARN] Failed to preload {mot_path}: {e}")
        return None

    result = compute_msmooth_score(
        video_name=video_name,
        data=preloaded,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        frame_violation_threshold=frame_violation_threshold,
    )

    os.makedirs(output_dir, exist_ok=True)

    # JSON
    json_path = os.path.join(output_dir, f"{video_name}_msmooth.json")
    with open(json_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    # TXT (human-readable)
    if save_detail_txt:
        txt_path = os.path.join(output_dir, f"{video_name}_msmooth.txt")
        with open(txt_path, "w") as f:
            f.write(f"Video: {video_name}\n")
            f.write(f"Motion smoothness score: {result.msmooth_score:.3f}\n")
            f.write(f"Frame violation rate: {result.frame_violation_rate:.6f}\n")
            f.write(f"Mean violation severity: {result.mean_violation_severity:.6f}\n")
            f.write(
                f"Max consecutive violation ratio: {result.max_consecutive_violation_ratio:.6f}\n"
            )

    return result


def summary_row(result: MotionSmoothnessResult) -> Dict[str, Any]:
    """One row of ``msmooth_scores_summary.csv``."""
    return {
        "video_name": result.video_name,
        "msmooth_score": result.msmooth_score,
        "frame_violation_rate": result.frame_violation_rate,
        "mean_violation_severity": result.mean_violation_severity,
        "max_consecutive_violation_ratio": result.max_consecutive_violation_ratio,
    }


# ========== CLI ==========

def main():
    """Command-line entry point; see the module docstring."""
    parser = argparse.ArgumentParser(
        description="Metric (VI) motion smoothness, paper implementation: score OpenSim .mot files."
    )
    parser.add_argument(
        "--mot_path",
        type=str,
        default=None,
        help="Single MOT file path; if omitted, iterate over --mot_dir",
    )
    parser.add_argument(
        "--mot_dir",
        type=str,
        default=DEFAULT_MOT_DIR,
        help="Directory containing MOT files (default: results/baseline)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory (default: results/metrics/motion_smoothness)",
    )
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--gamma", type=float, default=0.2)
    parser.add_argument(
        "--frame_violation_threshold",
        type=float,
        default=FRAME_VIOLATION_THRESHOLD,
    )
    parser.add_argument(
        "--no_detail",
        action="store_true",
        help="Do not save txt files; only output json + summary csv",
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Collect MOT files
    if args.mot_path:
        if not os.path.exists(args.mot_path):
            raise FileNotFoundError(args.mot_path)
        mot_files = [args.mot_path]
    else:
        if not os.path.isdir(args.mot_dir):
            raise NotADirectoryError(
                f"--mot_dir '{args.mot_dir}' not found; check path or use --mot_path."
            )
        mot_files = [
            os.path.join(args.mot_dir, f)
            for f in sorted(os.listdir(args.mot_dir))
            if f.lower().endswith(".mot")
        ]

    if not mot_files:
        print("No MOT files found. Nothing to do.")
        return

    print(f"Found {len(mot_files)} MOT files.")

    summary_rows: List[Dict[str, Any]] = []

    for mot_path in mot_files:
        print(f"\nProcessing {mot_path} ...")
        result = compute_and_save_for_mot(
            mot_path=mot_path,
            output_dir=args.output_dir,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
            frame_violation_threshold=args.frame_violation_threshold,
            save_detail_txt=not args.no_detail,
        )
        if result is not None:
            summary_rows.append(summary_row(result))

    if summary_rows:
        summary_path = os.path.join(args.output_dir, "msmooth_scores_summary.csv")
        df = pd.DataFrame(summary_rows)
        df.to_csv(summary_path, index=False)
        print(f"\nSaved motion smoothness summary to: {summary_path}")
    else:
        print("No valid motion smoothness scores computed.")
    if len(summary_rows) != len(mot_files):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
