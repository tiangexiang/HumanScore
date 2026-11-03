# -*- coding: utf-8 -*-
"""Motion loading and angular accelerations for metric (VI) motion smoothness (paper implementation).

* :func:`load_mot` reads an OpenSim ``.mot`` motion (radians; ``inDegrees=yes`` is converted);
* :func:`preload_motion_data` returns the joint angular accelerations that
  ``main.compute_msmooth_score`` scores;
* :func:`compute_joint_vel_acc` returns velocities and accelerations together.

Derivatives use :func:`metrics.kinematic_extreme.utils.central_derivative`: central
differences on the actual timestamps, one-sided at the ends, never across a timestamp gap,
with angles unwrapped first.
"""

import os
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from metrics.kinematic_extreme.utils import central_derivative


# ========= MOT loading =========

@dataclass
class MotData:
    """A loaded ``.mot`` motion: one row per frame, one column per skeleton coordinate."""

    time: np.ndarray           # (T,)
    joint_names: List[str]     # length J
    joint_angles: np.ndarray   # (T, J), radians (or consistent with thresholds)


def load_mot(mot_path: str) -> MotData:
    """
    Load an OpenSim-style MOT file by locating the 'time' header and reading numeric rows.
    """
    if not os.path.exists(mot_path):
        raise FileNotFoundError(f"MOT file not found: {mot_path}")

    with open(mot_path, "r") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    header_idx = None
    for i, line in enumerate(lines):
        if line.split()[0] == "time":
            header_idx = i
            break

    if header_idx is None:
        raise ValueError(f"Could not find 'time' header in {mot_path}")

    header_tokens = lines[header_idx].split()
    if header_tokens[0] != "time":
        raise ValueError("Header line must start with 'time'")

    col_names = header_tokens
    data_rows = []
    for line in lines[header_idx + 1 :]:
        parts = line.split()
        if len(parts) != len(col_names):
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

    time = df["time"].to_numpy()
    joint_names = list(col_names[1:])
    joint_angles = df[joint_names].to_numpy(dtype=float)

    if not np.isfinite(data).all():
        raise ValueError(f"MOT contains non-finite values: {mot_path}")
    if len(time) > 1 and np.any(np.diff(time) <= 0):
        raise ValueError(f"MOT timestamps must be strictly increasing: {mot_path}")
    if any(line.lower().replace(" ", "") == "indegrees=yes" for line in lines[:header_idx]):
        rotational = [not name.endswith(("_tx", "_ty", "_tz")) for name in joint_names]
        joint_angles[:, rotational] = np.deg2rad(joint_angles[:, rotational])

    return MotData(time=time, joint_names=joint_names, joint_angles=joint_angles)


# ========= velocities & accelerations =========

@dataclass
class JointKinematics:
    """Joint angular velocities and accelerations with their frame times."""

    vel_time: np.ndarray          # (Tv,)
    vel: np.ndarray               # (Tv, J), rad/s
    acc_time: np.ndarray          # (Ta,)
    acc: np.ndarray               # (Ta, J), rad/s^2
    joint_names: List[str]


def compute_joint_vel_acc(mot: MotData) -> JointKinematics:
    """
    Compute angular velocity and acceleration (central differences) from joint angles.

    Derivatives remain at original frame times. Timestamp gaps split the
    trajectory into independent segments; no derivative crosses a gap.
    """
    vel_time, vel = central_derivative(mot.joint_angles, mot.time, unwrap_angles=True)
    acc_time, acc = central_derivative(mot.joint_angles, mot.time, order=2, unwrap_angles=True)

    return JointKinematics(
        vel_time=vel_time,
        vel=vel,
        acc_time=acc_time,
        acc=acc,
        joint_names=mot.joint_names,
    )


# ========= preload wrapper =========

@dataclass
class PreloadedMotionData:
    """Everything ``main.compute_msmooth_score`` needs for one motion."""

    joint_names: List[str]
    acc_time: np.ndarray        # (T, )
    joint_acc: np.ndarray       # (T, J), rad/s^2


def preload_motion_data(mot_path: str, *, mot=None) -> PreloadedMotionData:
    """Load a motion (or use ``mot``) and compute its joint angular accelerations."""
    if mot is None:
        mot = load_mot(mot_path)
    # The score uses accelerations only, so the velocities are not kept; order=2 applies
    # the same differencing, segment boundaries and angle unwrapping twice.
    acc_time, acc = central_derivative(mot.joint_angles, mot.time, order=2, unwrap_angles=True)

    return PreloadedMotionData(
        joint_names=mot.joint_names,
        acc_time=acc_time,
        joint_acc=acc,
    )
