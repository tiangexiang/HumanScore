"""Metric (III) joint range of motion: the original HumanScore paper implementation.

Flags frames in which a joint of the fitted skeleton is bent beyond its anatomical range
(limits in ``definition.py``).  With ``--paper-scores``, ``hmb/scoring/paper_scores.py``
calls :func:`compute_rom_for_mot_file` on ``<S>/baseline/<stem>_opensim_skeleton.mot`` (the
skeleton fitted without joint limits, so violations stay visible) and writes the result to
``<S>/paper_metrics/joint_range/``.  The default joint-range score is
``hmb/scoring/joint_range.py``.

Definition (HumanScore paper, supplementary section F.1)::

    per frame t and scored coordinate j with limits [lo_j, hi_j] and tolerance tol = 15 deg:
        e_j(t) = max(0, lo_j - tol - q_j(t)) + max(0, q_j(t) - hi_j - tol)
        u_j(t) = clip(e_j(t) / (ROM_EXCESS_FRACTION * (hi_j - lo_j)), 0, 1)
        m_t = mean_j u_j(t),   b_t = m_t > FRAME_VIOLATION_THRESHOLD
    r = mean_t b_t,   s = sum_t m_t / max(1, sum_t b_t),   p = (longest run of b_t) / T
    score = 100 * (1 - clip(0.5 r + 0.3 s + 0.2 p, 0, 1))

The pelvis (global root) coordinates are not scored.

Run on its own (from the repository root)::

    python -m metrics.range_of_motion.joints_violaton --mot_path S/baseline/clip_opensim_skeleton.mot --output_dir out

Without ``--mot_path`` every ``.mot`` in ``--mot_dir`` is scored.  Outputs in ``output_dir``:
``<video>_rom.json`` and ``.txt`` per video and ``rom_scores_summary.csv``.  (The file name
``joints_violaton`` is kept as is because the command line uses it.)
"""
import os
import json
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import pandas as pd

from .definition import (
    joint_names as ALL_JOINT_NAMES,
    joints_limits,
    pose_param_name2qid,
    OPENSIM_LIM_BOUNDS,
    OPENSIM_LIM_QID2IDX,
)

# =========================
# Paths & hyperparameters
# =========================

DEFAULT_MOT_DIR = os.path.join("results", "baseline")
DEFAULT_OUTPUT_DIR = os.path.join("results", "metrics", "range_of_motion")

# A frame with severity m_t above this is flagged.
FRAME_VIOLATION_THRESHOLD = 0.05

# Exceeding the (tolerance-expanded) limit by this fraction of the joint's range gives severity 1.
ROM_EXCESS_FRACTION = 0.5

# Unit weights for anatomical rotational DoFs. Global root motion is excluded.
JOINT_WEIGHTS: Dict[str, float] = {
    name: 0.0 if name.startswith("pelvis_") else 1.0 for name in joints_limits
}


def get_joint_weight(name: str) -> float:
    """Aggregation weight of a coordinate: 0 for the pelvis and for translations, else 1."""
    if name in JOINT_WEIGHTS:
        return JOINT_WEIGHTS[name]
    # translations (*_tx, *_ty, *_tz) are never scored
    if (
        name.endswith("_tx")
        or name.endswith("_ty")
        or name.endswith("_tz")
        or name in ["pelvis_tx", "pelvis_ty", "pelvis_tz"]
    ):
        return 0.0
    return 1.0


# =========================
# MOT loading
# =========================

def load_mot_file(mot_path: str) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Load joint trajectories from an OpenSim-style MOT file.

    Returns:
        poses: (T, J) numpy.float64, radians
        joint_columns: list[str], length J (column names excluding time)
        time_data: (T,) np.ndarray
    """
    if not os.path.exists(mot_path):
        raise FileNotFoundError(f"MOT file not found: {mot_path}")

    with open(mot_path, "r") as f:
        lines = [ln.strip() for ln in f.readlines()]

    # Locate header starting with "time"
    header_idx = None
    for i, line in enumerate(lines):
        if not line:
            continue
        if line.split()[0] == "time":
            header_idx = i
            break

    if header_idx is None:
        raise ValueError(
            f"Could not find header line starting with 'time' in MOT file: {mot_path}"
        )

    header_tokens = lines[header_idx].split()
    if not header_tokens or header_tokens[0] != "time":
        raise ValueError("Header line must start with 'time'")

    col_names = header_tokens
    data_rows: List[List[float]] = []
    for line in lines[header_idx + 1 :]:
        if not line:
            continue
        parts = line.split()
        if len(parts) != len(col_names):
            # Skip rows with mismatched column counts
            continue
        try:
            vals = [float(x) for x in parts]
        except ValueError:
            continue
        data_rows.append(vals)

    if not data_rows:
        raise ValueError(f"No valid data rows in MOT file: {mot_path}")

    data = np.asarray(data_rows, dtype=float)
    df = pd.DataFrame(data, columns=col_names)

    time_data = df["time"].to_numpy()
    joint_columns = list(col_names[1:])
    joint_data = df[joint_columns].to_numpy(dtype=float)

    if not np.isfinite(data).all():
        raise ValueError(f"MOT contains non-finite values: {mot_path}")
    if len(time_data) > 1 and np.any(np.diff(time_data) <= 0):
        raise ValueError(f"MOT timestamps must be strictly increasing: {mot_path}")
    if any(line.lower().replace(" ", "") == "indegrees=yes" for line in lines[:header_idx]):
        rotational = [not name.endswith(("_tx", "_ty", "_tz")) for name in joint_columns]
        joint_data[:, rotational] = np.deg2rad(joint_data[:, rotational])
    return joint_data, joint_columns, time_data


# =========================
# Per-joint per-frame violations
# =========================

def eval_rot_delta(
    poses: np.ndarray,
    joint_column_names: List[str],
    tol_deg: float = 15.0,
) -> Dict[str, np.ndarray]:
    """Amount (radians, >= 0) by which each limited joint exceeds its range in every frame.

    The range of every joint is first widened by ``tol_deg`` on both sides.

    Returns:
        dict[joint_name] -> (T,) array of the exceedance (0 inside the widened range)
    """
    if hasattr(poses, "detach"):
        poses = poses.detach().cpu().numpy()
    poses = np.asarray(poses, dtype=float)
    if poses.ndim != 2 or poses.shape[1] != len(joint_column_names) or not np.isfinite(poses).all():
        raise ValueError("poses must be a finite (frames, joint columns) array")
    if not np.isfinite(tol_deg) or tol_deg < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    tol_rad = np.deg2rad(tol_deg)
    name_to_idx = {name: idx for idx, name in enumerate(joint_column_names)}
    violations_per_joint: Dict[str, np.ndarray] = {}

    for joint_name in ALL_JOINT_NAMES:
        if joint_name not in name_to_idx:
            continue
        if joint_name not in joints_limits:
            continue

        col_idx = name_to_idx[joint_name]
        qid = pose_param_name2qid[joint_name]
        if qid not in OPENSIM_LIM_QID2IDX:
            continue

        lim_idx = OPENSIM_LIM_QID2IDX[qid]
        lower_bound, upper_bound = OPENSIM_LIM_BOUNDS[lim_idx]

        joint_angles = poses[:, col_idx]
        # Angles are compared as exported (no wrapping to [-pi, pi]).
        normalized_angles = joint_angles

        exceed_lb = np.where(
            normalized_angles < lower_bound - tol_rad,
            normalized_angles - lower_bound + tol_rad,
            np.zeros_like(joint_angles),
        )
        exceed_ub = np.where(
            normalized_angles > upper_bound + tol_rad,
            normalized_angles - upper_bound - tol_rad,
            np.zeros_like(joint_angles),
        )

        violation_amounts = np.abs(exceed_lb) + np.abs(exceed_ub)
        violations_per_joint[joint_name] = violation_amounts

    return violations_per_joint


# =========================
# Video-level ROM Score
# =========================

@dataclass
class RomScoreResult:
    """Result of one video; written as ``<video>_rom.json``."""

    video_name: str
    rom_score: float  # 0-100, higher is better
    frame_violation_rate: float
    mean_violation_severity: float
    max_consecutive_violation_ratio: float
    per_joint_violation_rate: Dict[str, float]
    per_joint_mean_excess_deg: Dict[str, float]
    debug: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _normalize_frame_violation(
    frame_excess_rad: np.ndarray,
    joint_names: List[str],
) -> float:
    """Frame severity m_t in [0, 1]: weighted mean of the exceedances, each relative to
    ``ROM_EXCESS_FRACTION`` times the joint's range and clipped to 1.  Every weighted joint
    counts in the mean, also when it is not violated.
    """
    weighted_sum = 0.0
    weight_sum = 0.0

    for j, name in enumerate(joint_names):
        excess = float(frame_excess_rad[j])
        if name not in joints_limits:
            continue

        w = get_joint_weight(name)
        if w <= 0.0:
            continue

        lb, ub = joints_limits[name]
        rom_width = float(ub - lb)
        if rom_width <= 0.0:
            continue

        # exceedance relative to the joint's range, clipped to [0, 1]
        norm = excess / (rom_width * ROM_EXCESS_FRACTION)
        norm = max(0.0, min(1.0, norm))

        weighted_sum += w * norm
        weight_sum += w

    if weight_sum == 0.0:
        return 0.0
    return weighted_sum / weight_sum


def _longest_consecutive_true(mask: np.ndarray) -> int:
    """Length of the longest run of True values."""
    max_run, cur = 0, 0
    for v in mask:
        if v:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 0
    return max_run


def compute_rom_score_from_violations(
    video_name: str,
    violations_per_joint: Dict[str, np.ndarray],
    alpha: float = 0.5,
    beta: float = 0.3,
    gamma: float = 0.2,
    frame_violation_threshold: float = FRAME_VIOLATION_THRESHOLD,
) -> RomScoreResult:
    """Video-level ROM score from the per-joint, per-frame exceedances of :func:`eval_rot_delta`.

    Raises ``ValueError`` when no weighted joint with a limit is present.
    """
    if not violations_per_joint:
        raise ValueError("No joints with anatomical limits found; ROM is unavailable")

    # Enforce a consistent joint order and keep only weighted joints
    ordered_joints: List[str] = []
    T = None
    for name in ALL_JOINT_NAMES:
        if name not in violations_per_joint:
            continue
        if name not in joints_limits:
            continue
        if get_joint_weight(name) <= 0.0:
            continue
        v = violations_per_joint[name]
        if T is None:
            T = int(v.shape[0])
        else:
            T = min(T, int(v.shape[0]))
        ordered_joints.append(name)

    if T is None or T <= 0 or not ordered_joints:
        raise ValueError("No valid frames/joints for ROM scoring")

    J = len(ordered_joints)
    violation_mat = np.zeros((T, J), dtype=float)
    for j, name in enumerate(ordered_joints):
        v = violations_per_joint[name]
        if hasattr(v, "detach"):
            v = v.detach().cpu().numpy()
        violation_mat[:, j] = np.asarray(v)[:T]
    if not np.isfinite(violation_mat).all() or np.any(violation_mat < 0):
        raise ValueError("ROM violations must be finite and nonnegative")

    # m_t & bad-frame mask
    m_t = np.zeros(T, dtype=float)
    for t in range(T):
        m_t[t] = _normalize_frame_violation(violation_mat[t], ordered_joints)
    b_t = m_t > frame_violation_threshold

    # r: fraction of problematic frames
    frame_violation_rate = float(b_t.mean()) if T > 0 else 0.0

    # s: average severity on bad frames
    mean_violation_severity = float(m_t.sum() / max(1, int(b_t.sum())))

    # p: longest consecutive bad-frame ratio
    longest_run = _longest_consecutive_true(b_t)
    max_consecutive_violation_ratio = float(longest_run / T) if T > 0 else 0.0

    # D ∈ [0,1]; score = 100 * (1 - D)
    D = (
        alpha * frame_violation_rate
        + beta * mean_violation_severity
        + gamma * max_consecutive_violation_ratio
    )
    D = max(0.0, min(1.0, D))
    rom_score = float((1.0 - D) * 100.0)

    # Per-joint statistics (reported, not part of the score)
    per_joint_violation_rate: Dict[str, float] = {}
    per_joint_mean_excess_deg: Dict[str, float] = {}
    for j, name in enumerate(ordered_joints):
        v = violation_mat[:, j]
        mask = v > 0.0
        if mask.any():
            per_joint_violation_rate[name] = float(mask.mean())
            per_joint_mean_excess_deg[name] = float(np.rad2deg(v[mask].mean()))
        else:
            per_joint_violation_rate[name] = 0.0
            per_joint_mean_excess_deg[name] = 0.0

    debug = {
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "frame_violation_threshold": frame_violation_threshold,
        "D": D,
        "ordered_joints": ordered_joints,
        "aggregation_weights": "unit_per_anatomical_dof",
    }

    return RomScoreResult(
        video_name=video_name,
        rom_score=rom_score,
        frame_violation_rate=frame_violation_rate,
        mean_violation_severity=mean_violation_severity,
        max_consecutive_violation_ratio=max_consecutive_violation_ratio,
        per_joint_violation_rate=per_joint_violation_rate,
        per_joint_mean_excess_deg=per_joint_mean_excess_deg,
        debug=debug,
    )


# =========================
# Wrapper: single MOT → files
# =========================

def compute_rom_for_mot_file(
    mot_path: str,
    output_dir: str,
    alpha: float = 0.5,
    beta: float = 0.3,
    gamma: float = 0.2,
    frame_violation_threshold: float = FRAME_VIOLATION_THRESHOLD,
    tolerance_deg: float = 15.0,
    save_detail: bool = True,
    preloaded: Optional[Tuple[np.ndarray, List[str], np.ndarray]] = None,
) -> RomScoreResult:
    """Score one ``.mot`` file and write ``<video>_rom.json`` (and ``.txt``) into ``output_dir``.

    ``preloaded`` is ``(poses, joint_names, times)`` as returned by :func:`load_mot_file`;
    it avoids reading the file again.
    """
    video_name = os.path.splitext(os.path.basename(mot_path))[0]

    poses, joint_column_names, time_data = preloaded if preloaded is not None else load_mot_file(mot_path)
    violations_per_joint = eval_rot_delta(poses, joint_column_names, tol_deg=tolerance_deg)

    result = compute_rom_score_from_violations(
        video_name=video_name,
        violations_per_joint=violations_per_joint,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        frame_violation_threshold=frame_violation_threshold,
    )

    os.makedirs(output_dir, exist_ok=True)

    # Detailed JSON
    json_path = os.path.join(output_dir, f"{video_name}_rom.json")
    with open(json_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    # Optional txt summary for manual inspection
    if save_detail:
        txt_path = os.path.join(output_dir, f"{video_name}_rom.txt")
        with open(txt_path, "w") as f:
            f.write(f"Video: {video_name}\n")
            f.write(f"ROM Score: {result.rom_score:.3f}\n")
            f.write(f"Frame violation rate: {result.frame_violation_rate:.6f}\n")
            f.write(f"Mean violation severity: {result.mean_violation_severity:.6f}\n")
            f.write(
                f"Max consecutive violation ratio: {result.max_consecutive_violation_ratio:.6f}\n\n"
            )

            f.write("Per-joint violation rate:\n")
            for jn, rate in sorted(
                result.per_joint_violation_rate.items(), key=lambda x: -x[1]
            ):
                f.write(f"  {jn}: {rate:.6f}\n")

            f.write("\nPer-joint mean excess (deg):\n")
            for jn, deg in sorted(
                result.per_joint_mean_excess_deg.items(), key=lambda x: -x[1]
            ):
                f.write(f"  {jn}: {deg:.6f}\n")

    return result


def summary_row(result: RomScoreResult) -> Dict[str, Any]:
    """One row of ``rom_scores_summary.csv``."""
    return {
        "video_name": result.video_name,
        "rom_score": result.rom_score,
        "frame_violation_rate": result.frame_violation_rate,
        "mean_violation_severity": result.mean_violation_severity,
        "max_consecutive_violation_ratio": result.max_consecutive_violation_ratio,
    }


# =========================
# CLI
# =========================

def main():
    """Command-line entry point; see the module docstring."""
    parser = argparse.ArgumentParser(
        description="Metric (III) joint range of motion, paper implementation: score OpenSim .mot files."
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
        help="Output directory (default: results/metrics/range_of_motion)",
    )
    parser.add_argument(
        "--no_detail",
        action="store_true",
        help="Skip per-video txt summaries; export JSON & CSV only",
    )
    parser.add_argument(
        '--tolerance_deg',
        type=float,
        default=15.0,
        help='Widen every joint range by this many degrees on both sides (default 15)'
    )
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--gamma", type=float, default=0.2)
    parser.add_argument(
        "--frame_violation_threshold",
        type=float,
        default=FRAME_VIOLATION_THRESHOLD,
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # collect the motions
    if args.mot_path is not None:
        if not os.path.exists(args.mot_path):
            raise FileNotFoundError(args.mot_path)
        mot_files = [args.mot_path]
    else:
        if not os.path.isdir(args.mot_dir):
            raise NotADirectoryError(
                f"--mot_dir '{args.mot_dir}' not found; "
                "set it correctly or use --mot_path."
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
    summary_rows = []

    for mot_path in mot_files:
        print(f"\nProcessing {mot_path} ...")
        try:
            result = compute_rom_for_mot_file(
                mot_path=mot_path,
                output_dir=args.output_dir,
                alpha=args.alpha,
                beta=args.beta,
                gamma=args.gamma,
                frame_violation_threshold=args.frame_violation_threshold,
                tolerance_deg=args.tolerance_deg,
                save_detail=not args.no_detail,
            )
            summary_rows.append(summary_row(result))
        except Exception as e:
            print(f"Error processing {mot_path}: {e}")

    # summary CSV
    if summary_rows:
        summary_path = os.path.join(args.output_dir, "rom_scores_summary.csv")
        df = pd.DataFrame(summary_rows)
        df.to_csv(summary_path, index=False)
        print(f"\nSaved ROM summary to: {summary_path}")
    else:
        print("No valid ROM scores computed.")
    if len(summary_rows) != len(mot_files):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
