"""Joint limits of metric (III) joint range of motion (paper implementation).

* ``joint_names``: the skeleton coordinates in the column order of the exported ``.mot``;
* ``joints_limits``: ``(lower, upper)`` per coordinate, radians for rotations;
* ``OPENSIM_LIM_*``: the same limits as arrays indexed by coordinate id, used by
  ``joints_violaton.eval_rot_delta``.

The limits are a loose physical-plausibility prior, not clinical norms.  The pelvis
coordinates (global root position and orientation) are listed but get weight 0 in the
score.  Subtalar, metatarsophalangeal, forearm pronation/supination and wrist coordinates
have no limit and are not scored.
"""
import math
import numpy as np

# Joint names in the same order as MOT columns
joint_names = [
    "pelvis_tilt",
    "pelvis_list",
    "pelvis_rotation",
    "pelvis_tx",
    "pelvis_ty",
    "pelvis_tz",
    "hip_flexion_r",
    "hip_adduction_r",
    "hip_rotation_r",
    "knee_angle_r",
    "ankle_angle_r",
    "subtalar_angle_r",
    "mtp_angle_r",
    "hip_flexion_l",
    "hip_adduction_l",
    "hip_rotation_l",
    "knee_angle_l",
    "ankle_angle_l",
    "subtalar_angle_l",
    "mtp_angle_l",
    "lumbar_extension",
    "lumbar_bending",
    "lumbar_rotation",
    "arm_flex_r",
    "arm_add_r",
    "arm_rot_r",
    "elbow_flex_r",
    "pro_sup_r",
    "wrist_flex_r",
    "wrist_dev_r",
    "arm_flex_l",
    "arm_add_l",
    "arm_rot_l",
    "elbow_flex_l",
    "pro_sup_l",
    "wrist_flex_l",
    "wrist_dev_l",
]

# Joint ROM limits (rotations in radians; translations kept for completeness but not scored).
# Comments give the limits in degrees.
joints_limits = {
    # Pelvis orientation
    "pelvis_tilt": (-math.pi, math.pi),
    "pelvis_list": (-math.pi, math.pi),
    "pelvis_rotation": (-math.pi, math.pi),

    # Pelvis translations (metres; not scored)
    "pelvis_tx": (-40.0, 40.0),
    "pelvis_ty": (-5.0, 5.0),
    "pelvis_tz": (-40.0, 40.0),

    # Right leg
    "hip_flexion_r": (-math.pi / 6.0, 2.0 * math.pi / 3.0),        # ≈ -30, 120
    "hip_adduction_r": (-5.0 * math.pi / 18.0, math.pi / 6.0),     # ≈ -50, 30
    "hip_rotation_r": (-2.0 * math.pi / 9.0, 2.0 * math.pi / 9.0), # ≈ -40, 40
    "knee_angle_r": (0.0, 2.0 * math.pi / 3.0),                    # ≈ 0, 120
    "ankle_angle_r": (-2.0 * math.pi / 9.0, math.pi / 6.0),        # ≈ -40, 30

    # Left leg
    "hip_flexion_l": (-math.pi / 6.0, 2.0 * math.pi / 3.0),
    "hip_adduction_l": (-5.0 * math.pi / 18.0, math.pi / 6.0),
    "hip_rotation_l": (-2.0 * math.pi / 9.0, 2.0 * math.pi / 9.0),
    "knee_angle_l": (0.0, 2.0 * math.pi / 3.0),
    "ankle_angle_l": (-2.0 * math.pi / 9.0, math.pi / 6.0),

    # Lumbar / trunk
    "lumbar_extension": (-math.pi / 2.0, math.pi / 2.0),
    "lumbar_bending": (-math.pi / 2.0, math.pi / 2.0),
    "lumbar_rotation": (-math.pi / 2.0, math.pi / 2.0),

    # Right arm
    "arm_flex_r": (-math.pi / 2.0, math.pi / 2.0),
    "arm_add_r": (-2.0 * math.pi / 3.0, math.pi / 2.0),
    "arm_rot_r": (-math.pi / 2.0, math.pi / 2.0),
    "elbow_flex_r": (0.0, 5.0 * math.pi / 6.0),

    # Left arm
    "arm_flex_l": (-math.pi / 2.0, math.pi / 2.0),
    "arm_add_l": (-2.0 * math.pi / 3.0, math.pi / 2.0),
    "arm_rot_l": (-math.pi / 2.0, math.pi / 2.0),
    "elbow_flex_l": (0.0, 5.0 * math.pi / 6.0),
}

# Limits as arrays indexed by coordinate id (position in joint_names).
pose_param_name2qid = {name: qid for qid, name in enumerate(joint_names)}
qid2pose_param_name = {qid: name for qid, name in enumerate(joint_names)}

OPENSIM_LIM_QIDS = []
OPENSIM_LIM_BOUNDS = []

for name, (lower, upper) in joints_limits.items():
    if lower > upper:
        lower, upper = upper, lower
    qid = pose_param_name2qid.get(name)
    if qid is None:
        continue
    OPENSIM_LIM_QIDS.append(qid)
    OPENSIM_LIM_BOUNDS.append([float(lower), float(upper)])

OPENSIM_LIM_BOUNDS = np.asarray(OPENSIM_LIM_BOUNDS, dtype=np.float64)
OPENSIM_LIM_QID2IDX = {qid: idx for idx, qid in enumerate(OPENSIM_LIM_QIDS)}
