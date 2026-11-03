# -*- coding: utf-8 -*-
"""Angular-acceleration limits of metric (VI) motion smoothness (paper implementation).

``joints_angular_acceleration_limits`` gives, per skeleton coordinate (OpenSim names), the
largest plausible angular acceleration in deg/s^2.  Limits are symmetric: only
``max(|min|, |max|)`` is compared with ``|acceleration|``.  They are deliberately loose
(fast kicks and throws stay below them), so that only jerky, unnatural spikes count.
``main.py`` turns them into a per-frame severity.
"""

from typing import Dict, Tuple


# Keep these limits fixed when comparing videos.
joints_angular_acceleration_limits: Dict[str, Tuple[float, float]] = {
    # Pelvis / trunk
    "pelvis_tilt": (-8000.0, 8000.0),
    "pelvis_list": (-8000.0, 8000.0),
    "pelvis_rotation": (-12000.0, 12000.0),
    "lumbar_extension": (-12000.0, 12000.0),
    "lumbar_bending": (-12000.0, 12000.0),
    "lumbar_rotation": (-12000.0, 12000.0),

    # Hips
    "hip_flexion_r": (-20000.0, 20000.0),
    "hip_adduction_r": (-15000.0, 15000.0),
    "hip_rotation_r": (-20000.0, 20000.0),
    "hip_flexion_l": (-20000.0, 20000.0),
    "hip_adduction_l": (-15000.0, 15000.0),
    "hip_rotation_l": (-20000.0, 20000.0),

    # Knees
    "knee_angle_r": (-25000.0, 25000.0),
    "knee_angle_l": (-25000.0, 25000.0),

    # Ankles & feet
    "ankle_angle_r": (-22000.0, 22000.0),
    "subtalar_angle_r": (-18000.0, 18000.0),
    "mtp_angle_r": (-18000.0, 18000.0),

    "ankle_angle_l": (-22000.0, 22000.0),
    "subtalar_angle_l": (-18000.0, 18000.0),
    "mtp_angle_l": (-18000.0, 18000.0),

    # Shoulders & arms
    "arm_flex_r": (-30000.0, 30000.0),
    "arm_add_r": (-30000.0, 30000.0),
    "arm_rot_r": (-60000.0, 60000.0),
    "elbow_flex_r": (-60000.0, 60000.0),
    "pro_sup_r": (-50000.0, 50000.0),
    "wrist_flex_r": (-50000.0, 50000.0),
    "wrist_dev_r": (-50000.0, 50000.0),

    "arm_flex_l": (-30000.0, 30000.0),
    "arm_add_l": (-30000.0, 30000.0),
    "arm_rot_l": (-60000.0, 60000.0),
    "elbow_flex_l": (-60000.0, 60000.0),
    "pro_sup_l": (-50000.0, 50000.0),
    "wrist_flex_l": (-50000.0, 50000.0),
    "wrist_dev_l": (-50000.0, 50000.0),
}


def get_joint_weight(name: str) -> float:
    """Unit weight for every coordinate with a limit (the paper's setting), else 0."""
    return 1.0 if name in joints_angular_acceleration_limits else 0.0
