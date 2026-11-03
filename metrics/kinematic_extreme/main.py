# -*- coding: utf-8 -*-
"""Metric (V) kinematic extremes: the original HumanScore paper implementation.

Flags frames in which a joint turns, or a body segment moves, faster than a human can
(limits in ``definition.py``).  HumanScore uses this code for its kinematic-extremes score,
both by default and in the paper scores: ``hmb/scoring/paper_scores.py`` calls
:func:`compute_and_save_for_mot` on ``<S>/baseline/<stem>_opensim_skeleton_grid.mot``, the
fitted skeleton sampled on a uniform 24 Hz grid (finite differences need a uniform step).

Definition (HumanScore paper, supplementary section F.1)::

    per frame t: angular velocity w_j(t) of every coordinate j, linear speed v_b(t) of every
    segment b (centre of mass, from forward kinematics of the template skeleton)
        u_j(t) = clip((|w_j(t)| - wmax_j) / (EXCESS_FRACTION_ANG * wmax_j), 0, 1)
        u_b(t) = clip((v_b(t) - vmax_b) / (EXCESS_FRACTION_LIN * vmax_b), 0, 1)
        m_t = 0.5 * (mean_j u_j(t) + mean_b u_b(t))   (mean_j u_j(t) without a body model)
        b_t = m_t > FRAME_VIOLATION_THRESHOLD
    r = mean_t b_t,   s = sum_t m_t / max(1, sum_t b_t),   p = (longest run of b_t) / T
    score = 100 * (1 - clip(0.5 r + 0.3 s + 0.2 p, 0, 1))

``s`` sums ``m_t`` over all frames, including those below the threshold, as the paper writes it.

Run on its own (from the repository root)::

    python -m metrics.kinematic_extreme.main --mot_path S/baseline/clip_opensim_skeleton_grid.mot --output_dir out

Without ``--mot_path`` every ``.mot`` in ``--mot_dir`` is scored.  ``--model_path ""`` scores the
joint velocities only.  Outputs in ``output_dir``: ``<video>_kextreme.json`` and ``.txt`` per
video and ``kextreme_scores_summary.csv``.
"""

import os
import json
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd

from .definition import (
    joints_angular_velocity_limits,
    body_linear_velocity_limits,
    get_joint_weight,
    get_body_weight,
)
from .utils import (
    preload_motion_data,
    PreloadedMotionData,
)

# Default paths
DEFAULT_MOT_DIR = os.path.join("results", "baseline")
DEFAULT_OUTPUT_DIR = os.path.join("results", "metrics", "kinematic_extreme")

# Scoring parameters
FRAME_VIOLATION_THRESHOLD = 0.05   # a frame with m_t > 0.05 is flagged as extreme
EXCESS_FRACTION_ANG = 0.5          # exceeding the limit by 50 % of it gives full severity
EXCESS_FRACTION_LIN = 0.5
DEFAULT_MODEL_PATH = "baseline/monocular_demos/biomechanics_mjx/data/humanoid/humanoid_torque.xml"


@dataclass
class KinematicExtremeResult:
    """Result of one video; written as ``<video>_kextreme.json``."""

    video_name: str
    kextreme_score: float
    frame_violation_rate: float
    mean_violation_severity: float
    max_consecutive_violation_ratio: float
    per_joint_violation_rate: Dict[str, float]
    per_joint_mean_excess_deg_per_s: Dict[str, float]
    per_body_violation_rate: Dict[str, float]
    per_body_mean_excess_m_per_s: Dict[str, float]
    debug: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _longest_consecutive_true(mask: np.ndarray) -> int:
    """Length of the longest run of True values."""
    max_run = cur = 0
    for v in mask:
        if v:
            cur += 1
            if cur > max_run:
                max_run = cur
        else:
            cur = 0
    return max_run


# ---------------- Per-frame: angular velocity severity ----------------

def _frame_joint_severity(
    vel_deg_s: np.ndarray,
    joint_names: List[str],
) -> (float, Dict[str, float]):
    """Joint part of the frame severity.

    Args:
        vel_deg_s: (J,) absolute angular velocity per joint (deg/s)
    Returns:
        m_joint: weighted mean of the clipped normalised excesses, in [0, 1]. Every
            joint with a limit counts in the denominator, also when it is not violated.
        per_joint_excess: amount exceeding vmax for each joint (deg/s)
    """
    weighted_sum = 0.0
    weight_sum = 0.0
    per_joint_excess: Dict[str, float] = {}

    for j, name in enumerate(joint_names):
        limits = joints_angular_velocity_limits.get(name)
        if limits is None:
            continue

        vmax = max(abs(limits[0]), abs(limits[1]))
        if vmax <= 0:
            continue

        v = float(abs(vel_deg_s[j]))
        excess = max(0.0, v - vmax)

        norm = excess / (vmax * EXCESS_FRACTION_ANG)
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


# ---------------- Per-frame: linear velocity severity ----------------

def _frame_body_severity(
    speeds_m_s: Optional[np.ndarray],
    body_names: List[str],
) -> (float, Dict[str, float]):
    """Body-segment part of the frame severity (0 without body speeds).

    Args:
        speeds_m_s: (B,) linear speed magnitude per body (m/s)
    Returns:
        m_body: weighted mean of the clipped normalised excesses, in [0, 1]
        per_body_excess: amount exceeding vmax (m/s)
    """
    if speeds_m_s is None:
        return 0.0, {}

    weighted_sum = 0.0
    weight_sum = 0.0
    per_body_excess: Dict[str, float] = {}

    for b, name in enumerate(body_names):
        limits = body_linear_velocity_limits.get(name)
        if limits is None:
            continue

        vmax = max(abs(limits[0]), abs(limits[1]))
        if vmax <= 0:
            continue

        v = float(abs(speeds_m_s[b]))
        excess = max(0.0, v - vmax)

        norm = excess / (vmax * EXCESS_FRACTION_LIN)
        norm = float(max(0.0, min(1.0, norm)))

        w = get_body_weight(name)
        if w <= 0.0:
            continue

        weighted_sum += w * norm
        weight_sum += w
        per_body_excess[name] = excess

    if weight_sum == 0.0:
        return 0.0, per_body_excess

    return weighted_sum / weight_sum, per_body_excess


# ---------------- Video-level scoring ----------------

def compute_kextreme_score(
    video_name: str,
    data: PreloadedMotionData,
    alpha: float = 0.5,
    beta: float = 0.3,
    gamma: float = 0.2,
    frame_violation_threshold: float = FRAME_VIOLATION_THRESHOLD,
) -> KinematicExtremeResult:
    """Kinematic-extremes score of one video from its velocities (see the module docstring).

    ``data`` comes from :func:`metrics.kinematic_extreme.utils.preload_motion_data`.  Raises
    ``ValueError`` when there are no finite joint velocities with a known limit.
    """

    joint_vel = data.joint_angular_vel        # (T, J)
    joint_names = data.joint_names
    body_vel = data.body_linear_vel           # (T, B) or None
    body_names = data.body_names

    if joint_vel is None or joint_vel.size == 0:
        raise ValueError("No joint velocity data; kinematic extremes is unavailable")
    if not np.isfinite(joint_vel).all():
        raise ValueError("Joint velocities must be finite")
    if not any(name in joints_angular_velocity_limits for name in joint_names):
        raise ValueError("No recognized joint angular velocities")

    T = joint_vel.shape[0]
    if body_vel is not None and body_vel.size > 0:
        if not np.isfinite(body_vel).all():
            raise ValueError("Body velocities must be finite")
        T = min(T, body_vel.shape[0])
        joint_vel = joint_vel[:T]
        body_vel = body_vel[:T]
    else:
        body_vel = None

    m_t = np.zeros(T, dtype=float)
    bad_mask = np.zeros(T, dtype=bool)

    joint_violation_counts: Dict[str, int] = {}
    joint_excess_values: Dict[str, List[float]] = {}
    body_violation_counts: Dict[str, int] = {}
    body_excess_values: Dict[str, List[float]] = {}

    for t in range(T):
        # Angular velocity → deg/s
        vel_deg_s = np.degrees(np.abs(joint_vel[t]))
        m_joint, per_j_excess = _frame_joint_severity(vel_deg_s, joint_names)

        # Linear velocity → m/s
        if body_vel is not None:
            m_body, per_b_excess = _frame_body_severity(body_vel[t], body_names)
        else:
            m_body, per_b_excess = 0.0, {}

        # Fusion:
        # - If both exist: average
        # - If only one exists: use that term
        if body_vel is not None:
            m = 0.5 * (m_joint + m_body)
        else:
            m = m_joint

        m = float(max(0.0, min(1.0, m)))
        m_t[t] = m
        bad_mask[t] = m > frame_violation_threshold

        # per-joint violation statistics
        for name, exc in per_j_excess.items():
            if exc > 0:
                joint_violation_counts[name] = joint_violation_counts.get(name, 0) + 1
                joint_excess_values.setdefault(name, []).append(exc)

        # per-segment violation statistics
        for name, exc in per_b_excess.items():
            if exc > 0:
                body_violation_counts[name] = body_violation_counts.get(name, 0) + 1
                body_excess_values.setdefault(name, []).append(exc)

    if T > 0:
        frame_violation_rate = float(bad_mask.mean())
        longest_run = _longest_consecutive_true(bad_mask)
        max_consecutive_violation_ratio = float(longest_run / T)
    else:
        frame_violation_rate = 0.0
        max_consecutive_violation_ratio = 0.0

    # The paper (supplementary F.1) sums m_t over all frames, also those below the threshold.
    mean_violation_severity = float(m_t.sum() / max(1, int(bad_mask.sum())))

    # Combined penalty D & final score
    D = (
        alpha * frame_violation_rate
        + beta * mean_violation_severity
        + gamma * max_consecutive_violation_ratio
    )
    D = float(max(0.0, min(1.0, D)))
    kextreme_score = float((1.0 - D) * 100.0)

    # per-joint
    per_joint_violation_rate: Dict[str, float] = {}
    per_joint_mean_excess: Dict[str, float] = {}
    for name in joints_angular_velocity_limits.keys():
        cnt = joint_violation_counts.get(name, 0)
        per_joint_violation_rate[name] = float(cnt / T) if T > 0 else 0.0
        if cnt > 0:
            per_joint_mean_excess[name] = float(
                np.mean(joint_excess_values.get(name, [0.0]))
            )
        else:
            per_joint_mean_excess[name] = 0.0

    # per-body
    per_body_violation_rate: Dict[str, float] = {}
    per_body_mean_excess: Dict[str, float] = {}
    for name in body_linear_velocity_limits.keys():
        cnt = body_violation_counts.get(name, 0)
        per_body_violation_rate[name] = float(cnt / T) if T > 0 else 0.0
        if cnt > 0:
            per_body_mean_excess[name] = float(
                np.mean(body_excess_values.get(name, [0.0]))
            )
        else:
            per_body_mean_excess[name] = 0.0

    debug = {
        "D": D,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "frame_violation_threshold": frame_violation_threshold,
        "scoring_components": ["joint_angular_velocity", "body_linear_velocity"] if body_vel is not None else ["joint_angular_velocity"],
        "body_model_scale": "template" if body_vel is not None else None,
        "derivative_method": "central_with_one_sided_endpoints",
        "aggregation_weights": "unit_per_dof_and_segment",
    }

    return KinematicExtremeResult(
        video_name=video_name,
        kextreme_score=kextreme_score,
        frame_violation_rate=frame_violation_rate,
        mean_violation_severity=mean_violation_severity,
        max_consecutive_violation_ratio=max_consecutive_violation_ratio,
        per_joint_violation_rate=per_joint_violation_rate,
        per_joint_mean_excess_deg_per_s=per_joint_mean_excess,
        per_body_violation_rate=per_body_violation_rate,
        per_body_mean_excess_m_per_s=per_body_mean_excess,
        debug=debug,
    )


# ---------------- Compute & save ----------------

def compute_and_save_for_mot(
    mot_path: str,
    output_dir: str,
    model_path: Optional[str],
    alpha: float,
    beta: float,
    gamma: float,
    frame_violation_threshold: float,
    save_detail_txt: bool = True,
    preloaded: Optional[PreloadedMotionData] = None,
) -> Optional[KinematicExtremeResult]:
    """Score one ``.mot`` file and write ``<video>_kextreme.json`` (and ``.txt``) into ``output_dir``.

    ``preloaded`` skips loading and differentiating the motion again.  Returns ``None`` when
    the motion cannot be loaded.
    """
    video_name = os.path.splitext(os.path.basename(mot_path))[0]

    try:
        if preloaded is None:
            preloaded = preload_motion_data(mot_path=mot_path, model_path=model_path)
    except Exception as e:
        print(f"[WARN] Failed to preload {mot_path}: {e}")
        return None

    result = compute_kextreme_score(
        video_name=video_name,
        data=preloaded,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        frame_violation_threshold=frame_violation_threshold,
    )

    os.makedirs(output_dir, exist_ok=True)

    # JSON
    json_path = os.path.join(output_dir, f"{video_name}_kextreme.json")
    with open(json_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    # TXT summary
    if save_detail_txt:
        txt_path = os.path.join(output_dir, f"{video_name}_kextreme.txt")
        with open(txt_path, "w") as f:
            f.write(f"Video: {video_name}\n")
            f.write(f"Kinematic extremes score: {result.kextreme_score:.3f}\n")
            f.write(f"Frame violation rate: {result.frame_violation_rate:.6f}\n")
            f.write(f"Mean violation severity: {result.mean_violation_severity:.6f}\n")
            f.write(
                f"Max consecutive violation ratio: {result.max_consecutive_violation_ratio:.6f}\n"
            )

    return result


def summary_row(result: KinematicExtremeResult) -> Dict[str, Any]:
    """One row of ``kextreme_scores_summary.csv``."""
    return {
        "video_name": result.video_name,
        "kextreme_score": result.kextreme_score,
        "frame_violation_rate": result.frame_violation_rate,
        "mean_violation_severity": result.mean_violation_severity,
        "max_consecutive_violation_ratio": result.max_consecutive_violation_ratio,
    }


# ---------------- CLI ----------------

def main():
    """Command-line entry point; see the module docstring."""
    parser = argparse.ArgumentParser(
        description="Metric (V) kinematic extremes, paper implementation: score OpenSim .mot files."
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
        help="Output directory (default: results/metrics/kinematic_extreme)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="MuJoCo skeleton XML for the body-segment speeds (default: the HumanScore skeleton, relative to "
             "the repository root); pass an empty string to score joint velocities only.",
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
        help="Skip per-video txt; only write json + summary csv",
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
                f"--mot_dir '{args.mot_dir}' not found; please check or use --mot_path."
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
            model_path=args.model_path,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
            frame_violation_threshold=args.frame_violation_threshold,
            save_detail_txt=not args.no_detail,
        )
        if result is not None:
            summary_rows.append(summary_row(result))

    if summary_rows:
        summary_path = os.path.join(args.output_dir, "kextreme_scores_summary.csv")
        df = pd.DataFrame(summary_rows)
        df.to_csv(summary_path, index=False)
        print(f"\nSaved summary to: {summary_path}")
    else:
        print("No valid kinematic extremes scores computed.")
    if len(summary_rows) != len(mot_files):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
