# -*- coding: utf-8 -*-
"""Velocity limits of metric (V) kinematic extremes (paper implementation).

* ``joints_angular_velocity_limits``: per skeleton coordinate (OpenSim names), deg/s;
* ``body_linear_velocity_limits``: per body segment (MuJoCo body names), m/s;
* ``get_joint_weight`` / ``get_body_weight``: aggregation weights (1 for every listed
  coordinate or segment, 0 otherwise).

Every limit is symmetric: only ``max(|min|, |max|)`` is used.  The limits sit at the upper
end of human capability (fast throws, kicks, sprints), so that only physically implausible
motion counts as a violation.  ``main.py`` turns them into a per-frame severity.
"""

from typing import Dict, Tuple


# ------------------------------------------------
# 1. Joint angular velocity limits (deg/s)
# ------------------------------------------------
# Lenient limits near the extremes of human motion. Keep them fixed when comparing videos.

joints_angular_velocity_limits: Dict[str, Tuple[float, float]] = {
    # Pelvis / trunk (large range of motion)
    "pelvis_tilt": (-360.0, 360.0),
    "pelvis_list": (-360.0, 360.0),
    "pelvis_rotation": (-720.0, 720.0),
    "lumbar_extension": (-720.0, 720.0),
    "lumbar_bending": (-720.0, 720.0),
    "lumbar_rotation": (-720.0, 720.0),

    # Hips
    "hip_flexion_r": (-800.0, 800.0),
    "hip_adduction_r": (-600.0, 600.0),
    "hip_rotation_r": (-800.0, 800.0),
    "hip_flexion_l": (-800.0, 800.0),
    "hip_adduction_l": (-600.0, 600.0),
    "hip_rotation_l": (-800.0, 800.0),

    # Knees
    "knee_angle_r": (-900.0, 900.0),
    "knee_angle_l": (-900.0, 900.0),

    # Ankles
    "ankle_angle_r": (-800.0, 800.0),
    "subtalar_angle_r": (-600.0, 600.0),
    "mtp_angle_r": (-600.0, 600.0),

    "ankle_angle_l": (-800.0, 800.0),
    "subtalar_angle_l": (-600.0, 600.0),
    "mtp_angle_l": (-600.0, 600.0),

    # Shoulders & arms
    "arm_flex_r": (-1200.0, 1200.0),
    "arm_add_r": (-1200.0, 1200.0),
    "arm_rot_r": (-4000.0, 4000.0),
    "elbow_flex_r": (-2500.0, 2500.0),
    "pro_sup_r": (-2000.0, 2000.0),
    "wrist_flex_r": (-2000.0, 2000.0),
    "wrist_dev_r": (-2000.0, 2000.0),

    "arm_flex_l": (-1200.0, 1200.0),
    "arm_add_l": (-1200.0, 1200.0),
    "arm_rot_l": (-4000.0, 4000.0),
    "elbow_flex_l": (-2500.0, 2500.0),
    "pro_sup_l": (-2000.0, 2000.0),
    "wrist_flex_l": (-2000.0, 2000.0),
    "wrist_dev_l": (-2000.0, 2000.0),
}

# ------------------------------------------------
# 2. Body linear velocity limits (m/s)
# ------------------------------------------------
# Keys follow ForwardKinematics body_names:
# [pelvis, femur_r, tibia_r, talus_r, calcn_r, toes_r,
#  femur_l, tibia_l, talus_l, calcn_l, toes_l,
#  torso, head,
#  humerus_r, ulna_r, radius_r, hand_r,
#  humerus_l, ulna_l, radius_l, hand_l]

body_linear_velocity_limits: Dict[str, Tuple[float, float]] = {
    # Pelvis / torso / head
    "pelvis": (-12.0, 12.0),   # near elite sprint COM velocity
    "torso": (-15.0, 15.0),
    "head": (-18.0, 18.0),

    # Lower limbs - right
    "femur_r": (-25.0, 25.0),
    "tibia_r": (-25.0, 25.0),
    "talus_r": (-25.0, 25.0),
    "calcn_r": (-30.0, 30.0),
    "toes_r": (-30.0, 30.0),

    # Lower limbs - left
    "femur_l": (-25.0, 25.0),
    "tibia_l": (-25.0, 25.0),
    "talus_l": (-25.0, 25.0),
    "calcn_l": (-30.0, 30.0),
    "toes_l": (-30.0, 30.0),

    # Upper limbs - right
    "humerus_r": (-35.0, 35.0),
    "ulna_r": (-40.0, 40.0),
    "radius_r": (-40.0, 40.0),
    "hand_r": (-45.0, 45.0),

    # Upper limbs - left
    "humerus_l": (-35.0, 35.0),
    "ulna_l": (-40.0, 40.0),
    "radius_l": (-40.0, 40.0),
    "hand_l": (-45.0, 45.0),
}

# ------------------------------------------------
# 3. Weights for aggregation
# ------------------------------------------------

def get_joint_weight(name: str) -> float:
    """Unit weight for every coordinate with a limit (the paper's setting), else 0."""
    return 1.0 if name in joints_angular_velocity_limits else 0.0


def get_body_weight(name: str) -> float:
    """Unit weight for every body segment with a limit (the paper's setting), else 0."""
    return 1.0 if name in body_linear_velocity_limits else 0.0
