# -*- coding: utf-8 -*-
"""Metric (II) bone-length consistency: the original HumanScore paper implementation.

Measures how much the length of nine major bones changes over a video, using the
skeleton fitted with a *time-varying* segment scale (``metrics/bone_length/main.py`` or
``hmb/fit_engine.py``), so that a real person's bones keep their length and a
generated body that stretches or shrinks shows up as a varying length.

Where it sits in HumanScore
    With ``--paper-scores``, ``hmb/scoring/paper_scores.py`` calls
    :func:`process_single_fitted_file` on ``<S>/bone_fit/<stem>_bone_length.npz`` and
    writes the result to ``<S>/paper_metrics/bone_length/``.  The default bone-length
    score (``hmb/scoring/bone_length.py``) does not need this fit.

Definition (HumanScore paper, supplementary section F.1)::

    for each bone b = (proximal body, distal body), over the valid frames t:
        l_b(t) = || p_proximal(t) - p_distal(t) ||      (MuJoCo body origins = joint centres)
        L_b    = median_t l_b(t)
        E_b    = mean_t |l_b(t) - L_b| / (L_b + eps)
    E = mean_b E_b,   score = 100 * (1 - clip(E / tau, 0, 1)),   tau = 0.15

A frame is valid when the bone's endpoints are finite and, if the file has a
``confidence`` array, at least one keypoint was detected in it.  A bone needs at least
``MIN_FRAMES_PER_BONE`` valid frames.

Input: ``<video>_bone_length.npz`` with ``timestamps`` (T,) and ``joints`` (T, bodies, 3)
(optional: ``body_names``, ``confidence``, ``scale_time_varying``).  A file whose
``scale_time_varying`` is false (a rigid fit) is rejected.

Run on its own::

    python -m metrics.bone_length.bone_length --input_path S/bone_fit/clip_bone_length.npz --output_dir out
    python -m metrics.bone_length.bone_length --input_dir results/baseline --output_dir out

Outputs in ``output_dir``: ``<video>_bone_length.json`` and ``.txt`` per video and
``bone_length_scores_summary.csv``.
"""

import os
import json
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd

# ---------------- Paths & hyper-parameters ----------------

DEFAULT_INPUT_DIR = os.path.join("results", "baseline")
DEFAULT_OUTPUT_DIR = os.path.join("results", "metrics", "bone_length")
HUMANOID_XML = Path(__file__).resolve().parents[2] / "baseline/monocular_demos/biomechanics_mjx/data/humanoid/humanoid_torque.xml"

#: Relative length error that maps to a score of 0 (HumanScore supplementary section F.1, metric II).
DEFAULT_TAU = 0.15
#: A bone needs at least this many valid frames to be scored.
MIN_FRAMES_PER_BONE = 5
EPS = 1e-8


# ---------------- Result structure ----------------

@dataclass
class BoneLengthMetricResult:
    """Result of one video; written as ``<video>_bone_length.json``."""

    video_name: str
    bone_length_score: float
    mean_relative_error: float
    median_relative_error: float
    max_relative_error: float
    num_bones_used: int
    per_bone_mean_relative_error: Dict[str, float]
    per_bone_ref_length: Dict[str, float]
    per_bone_cv_percent: Dict[str, float]
    debug: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------- Load fitted skeleton ----------------

def load_fitted_skeleton(npz_path: str) -> Dict[str, Any]:
    """
    Load a fitted skeleton from {video}_bone_length.npz.

    Required keys:
        - 'timestamps': (T,)
        - 'joints': (T, n_bodies, 3)

    Other fields (e.g. scale / qpos / qvel / etc.) are passed through unchanged.
    """
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(npz_path)

    data = np.load(npz_path, allow_pickle=True)
    keys = set(data.files)

    if "timestamps" not in keys or "joints" not in keys:
        raise KeyError(
            f"{npz_path} must contain 'timestamps' and 'joints'. "
            f"Found keys: {sorted(keys)}"
        )

    timestamps = np.asarray(data["timestamps"])
    joints = np.asarray(data["joints"])

    if joints.ndim != 3 or joints.shape[2] != 3:
        raise ValueError(
            f"'joints' must have shape (T, n_bodies, 3), got {joints.shape} in {npz_path}"
        )

    fitted = {"timestamps": timestamps, "joints": joints}
    # Also copy optional fields (do not affect Bone Length computation)
    for k in keys:
        if k not in ("timestamps", "joints"):
            fitted[k] = data[k]

    return fitted


# ---------------- Bone Length Consistency ----------------

class BoneLengthConsistency:
    """Bone lengths of one fitted skeleton and their consistency score.

    ``fitted["joints"]`` holds the MuJoCo body origins in body order, including the
    world body.  Body names come from ``fitted["body_names"]`` or, without them, from the
    bundled ``humanoid_torque.xml``.
    """

    #: bone name -> (proximal body, distal body); body origins are the joint centres.
    MAJOR_BONES = {
        # Lower body
        "femur_r": ("femur_r", "tibia_r"),
        "tibia_r": ("tibia_r", "talus_r"),
        "femur_l": ("femur_l", "tibia_l"),
        "tibia_l": ("tibia_l", "talus_l"),
        # Upper body
        "humerus_r": ("humerus_r", "ulna_r"),
        "forearm_r": ("radius_r", "hand_r"),
        "humerus_l": ("humerus_l", "ulna_l"),
        "forearm_l": ("radius_l", "hand_l"),
        # Torso
        "torso": ("torso", "head"),
    }

    def __init__(self, fitted: Dict[str, Any]):
        self.timestamps = np.asarray(fitted["timestamps"])
        self.joints = np.asarray(fitted["joints"])  # (T, n_bodies, 3)

        if self.joints.ndim != 3 or self.joints.shape[2] != 3:
            raise ValueError(
                f"'joints' must be (T, n_bodies, 3), got {self.joints.shape}"
            )

        self.T, self.n_bodies, _ = self.joints.shape

        if self.timestamps.shape != (self.T,):
            raise ValueError("'timestamps' must contain one value per frame")
        if not np.isfinite(self.timestamps).all() or np.any(np.diff(self.timestamps) <= 0):
            raise ValueError("'timestamps' must be finite and strictly increasing")
        if "scale_time_varying" in fitted:
            marker = np.asarray(fitted["scale_time_varying"])
            if marker.size != 1 or not bool(marker.item()):
                raise ValueError("Bone length requires a variable-scale fit; scale_time_varying is false or invalid")
        self.valid_frame_mask = np.ones(self.T, dtype=bool)
        if "confidence" in fitted:
            confidence = np.asarray(fitted["confidence"], dtype=float)
            if confidence.ndim == 0 or confidence.shape[0] != self.T:
                raise ValueError("'confidence' must contain one row per frame")
            confidence = confidence.reshape(self.T, -1)
            self.valid_frame_mask &= (np.isfinite(confidence) & (confidence > 0)).any(axis=1)

        # MuJoCo assigns body IDs in depth-first XML order. Reading names does
        # not require loading JAX, the fitting model, or its mesh assets.
        if "body_names" in fitted:
            self.body_names = np.asarray(fitted["body_names"]).astype(str).tolist()
        else:
            worldbody = ET.parse(HUMANOID_XML).getroot().find("worldbody")
            if worldbody is None:
                raise ValueError(f"No worldbody in {HUMANOID_XML}")
            self.body_names = ["world"] + [body.attrib["name"] for body in worldbody.iter("body")]
        if len(self.body_names) != self.n_bodies:
            raise ValueError(
                f"Body ordering has {len(self.body_names)} names but joints has "
                f"{self.n_bodies} bodies; provide matching 'body_names' metadata"
            )
        self.name2idx = {name: idx for idx, name in enumerate(self.body_names)}

    def compute_bone_lengths(self) -> Dict[str, np.ndarray]:
        """
        Collect length values for each bone over all valid frames:
            bone_lengths[bone_name] = (K,) sequence of lengths
        """
        bone_lengths: Dict[str, List[float]] = {}

        for bone_name, (p_name, c_name) in self.MAJOR_BONES.items():
            if p_name not in self.name2idx or c_name not in self.name2idx:
                print(f"[BoneLength] Skip bone '{bone_name}': "
                      f"body '{p_name}' or '{c_name}' not found.")
                continue

            pi = self.name2idx[p_name]
            ci = self.name2idx[c_name]

            if pi >= self.n_bodies or ci >= self.n_bodies:
                print(f"[BoneLength] Skip bone '{bone_name}': index out of range.")
                continue

            parents = self.joints[:, pi]
            children = self.joints[:, ci]
            valid = self.valid_frame_mask & np.isfinite(parents).all(axis=1) & np.isfinite(children).all(axis=1)
            vals = np.linalg.norm(children[valid] - parents[valid], axis=1)

            if len(vals) < MIN_FRAMES_PER_BONE:
                print(
                    f"[BoneLength] Bone '{bone_name}' has only {len(vals)} valid frames, skip."
                )
                continue

            bone_lengths[bone_name] = np.asarray(vals, dtype=np.float32)

        return bone_lengths

    @staticmethod
    def compute_stats_and_score(
        bone_lengths: Dict[str, np.ndarray],
        tau: float = DEFAULT_TAU,
        eps: float = EPS,
    ):
        """
        Compute statistics from bone_lengths:
          - per_bone_ref_length[name] = median_t l_b(t)
          - per_bone_mean_relative_error[name] = mean_t |l_b(t)-L_b| / (L_b+eps)
          - per_bone_cv_percent[name] = (std/mean)*100
          - global mean_E / med_E / max_E
          - score = (1 - clip(mean_E / tau, 0, 1)) * 100
        """
        if not np.isfinite(tau) or tau <= 0:
            raise ValueError("tau must be finite and positive")
        if not bone_lengths:
            raise ValueError("No valid bones to evaluate.")

        per_bone_ref_length: Dict[str, float] = {}
        per_bone_mean_rel_error: Dict[str, float] = {}
        per_bone_cv_percent: Dict[str, float] = {}
        all_E: List[float] = []

        for name, arr in bone_lengths.items():
            if arr.size < MIN_FRAMES_PER_BONE:
                continue

            L_b = float(np.median(arr))
            if L_b <= eps:
                continue

            rel = np.abs(arr - L_b) / (L_b + eps)
            E_b = float(rel.mean())

            mean_l = float(arr.mean())
            std_l = float(arr.std())
            cv = (std_l / mean_l * 100.0) if mean_l > 0 else 0.0

            per_bone_ref_length[name] = L_b
            per_bone_mean_rel_error[name] = E_b
            per_bone_cv_percent[name] = cv
            all_E.append(E_b)

        if not all_E:
            raise ValueError("All bones filtered out; cannot compute metric.")

        all_E = np.asarray(all_E, dtype=np.float32)
        mean_E = float(all_E.mean())
        med_E = float(np.median(all_E))
        max_E = float(all_E.max())

        D = float(np.clip(mean_E / tau, 0.0, 1.0))
        score = float((1.0 - D) * 100.0)

        debug = {
            "tau": float(tau),
            "mean_E": mean_E,
            "med_E": med_E,
            "max_E": max_E,
            "num_bones_used": int(len(per_bone_ref_length)),
        }

        return (
            score,
            mean_E,
            med_E,
            max_E,
            per_bone_ref_length,
            per_bone_mean_rel_error,
            per_bone_cv_percent,
            debug,
        )


# ---------------- Single-file processing ----------------

def process_single_fitted_file(
    npz_path: str,
    output_dir: str,
    tau: float = DEFAULT_TAU,
    save_detail_txt: bool = True,
) -> Optional[BoneLengthMetricResult]:
    """Score one ``<video>_bone_length.npz`` and write its JSON (and TXT) into ``output_dir``.

    Returns the result, or ``None`` when the file cannot be loaded or has no valid bone
    (the reason is printed).
    """
    fname = os.path.basename(npz_path)
    if fname.endswith("_bone_length.npz"):
        video_name = fname[: -len("_bone_length.npz")]
    else:
        video_name = os.path.splitext(fname)[0]

    print("\n" + "=" * 80)
    print(f"[BoneLength] Processing {video_name}")
    print("=" * 80)

    try:
        fitted = load_fitted_skeleton(npz_path)
    except Exception as e:
        print(f"[BoneLength] Failed to load fitted skeleton from {npz_path}: {e}")
        return None

    try:
        analyzer = BoneLengthConsistency(fitted)
        bone_lengths = analyzer.compute_bone_lengths()
        if not bone_lengths:
            print(f"[BoneLength] No valid bones for {video_name}, skip.")
            return None

        (
            score,
            mean_E,
            med_E,
            max_E,
            per_bone_ref_length,
            per_bone_mean_rel_error,
            per_bone_cv_percent,
            debug,
        ) = BoneLengthConsistency.compute_stats_and_score(bone_lengths, tau=tau)
        debug.update(
            scale_time_varying=bool(np.asarray(fitted["scale_time_varying"]).item()) if "scale_time_varying" in fitted else "unknown",
            confidence_filter_applied="confidence" in fitted,
            num_valid_detection_frames=int(analyzer.valid_frame_mask.sum()),
            bone_endpoint_convention="anatomical_joint_centers_v1",
        )
    except Exception as e:
        print(f"[BoneLength] Failed to compute metric for {video_name}: {e}")
        return None

    result = BoneLengthMetricResult(
        video_name=video_name,
        bone_length_score=score,
        mean_relative_error=mean_E,
        median_relative_error=med_E,
        max_relative_error=max_E,
        num_bones_used=debug["num_bones_used"],
        per_bone_mean_relative_error=per_bone_mean_rel_error,
        per_bone_ref_length=per_bone_ref_length,
        per_bone_cv_percent=per_bone_cv_percent,
        debug=debug,
    )

    os.makedirs(output_dir, exist_ok=True)

    # JSON
    json_path = os.path.join(output_dir, f"{video_name}_bone_length.json")
    with open(json_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    # short human-readable summary
    if save_detail_txt:
        txt_path = os.path.join(output_dir, f"{video_name}_bone_length.txt")
        with open(txt_path, "w") as f:
            f.write(f"Video: {video_name}\n")
            f.write(f"Bone length consistency score: {score:.3f}\n")
            f.write(f"Mean relative error: {mean_E:.6f}\n")
            f.write(f"Median relative error: {med_E:.6f}\n")
            f.write(f"Max relative error: {max_E:.6f}\n")
            f.write(f"Num bones used: {debug['num_bones_used']}\n")

    print(
        f"[BoneLength] {video_name}: "
        f"score={score:.2f}, meanE={mean_E:.4f}, bones={debug['num_bones_used']}"
    )

    return result


# ---------------- CLI ----------------

def summary_row(result: BoneLengthMetricResult) -> Dict[str, Any]:
    """One row of ``bone_length_scores_summary.csv``."""
    return {
        "video_name": result.video_name,
        "bone_length_score": result.bone_length_score,
        "mean_relative_error": result.mean_relative_error,
        "median_relative_error": result.median_relative_error,
        "max_relative_error": result.max_relative_error,
        "num_bones_used": result.num_bones_used,
    }


def main():
    """Command-line entry point; see the module docstring."""
    parser = argparse.ArgumentParser(
        description="Metric (II) bone-length consistency, paper implementation (reads *_bone_length.npz fits)."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default=None,
        help="Single *_bone_length.npz file to evaluate. If omitted, iterate over --input_dir.",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing *_bone_length.npz files (default: results/baseline)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save metric results (default: results/metrics/bone_length)",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=DEFAULT_TAU,
        help=f"Average relative error threshold tau used to map E to score (default: {DEFAULT_TAU})",
    )
    parser.add_argument(
        "--no_detail",
        action="store_true",
        help="Do not save per-video .txt summaries; only output JSON + summary CSV.",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    fitted_files: List[str] = []
    if args.input_path:
        if not os.path.isfile(args.input_path):
            raise FileNotFoundError(args.input_path)
        fitted_files = [args.input_path]
    else:
        if not os.path.isdir(args.input_dir):
            raise NotADirectoryError(f"input_dir '{args.input_dir}' not found")

        # Find all *_bone_length.npz files
        for root, _, files in os.walk(args.input_dir):
            for f in files:
                if f.endswith("_bone_length.npz"):
                    fitted_files.append(os.path.join(root, f))

        fitted_files = sorted(set(fitted_files))
    if not fitted_files:
        if args.input_path:
            print(f"No valid *_bone_length.npz found at {args.input_path}")
        else:
            print(f"No *_bone_length.npz found under {args.input_dir}")
        return

    print(f"Found {len(fitted_files)} fitted skeleton files.")

    summary_rows: List[Dict[str, Any]] = []

    for npz_path in fitted_files:
        res = process_single_fitted_file(
            npz_path=npz_path,
            output_dir=args.output_dir,
            tau=args.tau,
            save_detail_txt=not args.no_detail,
        )
        if res is not None:
            summary_rows.append(summary_row(res))

    if summary_rows:
        df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(
            args.output_dir, "bone_length_scores_summary.csv"
        )
        df.to_csv(summary_path, index=False)
        print(f"\n[BoneLength] Saved summary to: {summary_path}")
    else:
        print("\n[BoneLength] No valid bone length metrics computed.")
    if len(summary_rows) != len(fitted_files):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
