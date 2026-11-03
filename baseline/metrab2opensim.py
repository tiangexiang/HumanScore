"""Export a fitted skeleton trajectory as an OpenSim ``.mot`` motion file.

Reads ``qpos`` and ``timestamps`` of a fitted ``.npz`` (``<stem>_fitted_model.npz`` or
``<stem>_bone_length.npz``) and writes one column per joint of ``humanoid_torque.xml`` in
OpenSim conventions: the pelvis translation and orientation are rotated from MuJoCo's
z-up axes to OpenSim's y-up axes, and the knee angles change sign.  Angles are in radians
(``inDegrees=no``); the header line ``signed_pelvis_rotation=yes`` marks that the pelvis
rotation keeps its sign.  ``metrics/kinematic_extreme/utils.py`` inverts this mapping.

HumanScore's fit engine (``hmb/fit_engine.py``) and ``baseline/main.py`` call
:func:`convert_fitted_model_to_mot`.  On its own::

    python baseline/metrab2opensim.py clip_fitted_model.npz clip_opensim_skeleton.mot
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R


def get_joint_names():
    """Read qpos names without initializing JAX, MJX, or the fitting model."""
    import mujoco

    xml_path = Path(__file__).resolve().parent / "monocular_demos/biomechanics_mjx/data/humanoid/humanoid_torque.xml"
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    if model.nq != model.njnt:
        raise ValueError("OpenSim conversion requires the scalar-joint Rajagopal skeleton")
    return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]


def convertDataframeToMot(df, out_path, dt, time_column):
    """Write ``df`` (one column per coordinate) with ``time_column`` as an OpenSim ``.mot`` file.

    The timestamps are written as given (they need not be uniform); ``dt`` is not used.
    """
    numFrames = df.shape[0]
    timestamps = np.asarray(time_column)
    if timestamps.shape != (numFrames,) or not np.all(np.isfinite(timestamps)):
        raise ValueError("Expected one finite timestamp per motion frame")
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("Motion timestamps must be strictly increasing")
    out_file = open(out_path, 'w')
    out_file.write('Coordinates\n')
    out_file.write('version=1\n')
    out_file.write(f'nRows={numFrames}\n')
    out_file.write(f'nColumns={len(df.columns)+1}\n')
    out_file.write('inDegrees=no\n')
    out_file.write('coordinate_convention=opensim_rajagopal_v1\n')
    out_file.write('signed_pelvis_rotation=yes\n\n')
    out_file.write('If the header above contains a line with \'inDegrees\', this indicates whether rotational values are in degrees (yes) or radians (no).\n\n')
    out_file.write('endheader\n')

    out_file.write('time')
    for i in range(len(df.columns)):
        out_file.write('\t' + df.columns[i])
    out_file.write('\n')

    for i in range(numFrames):
        out_file.write(format(float(timestamps[i]), '.12g'))
        for j in range(len(df.columns)):
            out_file.write('\t' + str(df.iloc[i, j]))
        out_file.write('\n')
    out_file.close()
    print(f'Kinematics exported to {out_path}')

def convert_fitted_model_to_mot(fitted_model_path, output_path):
    """Convert a fitted skeleton (``.npz`` with ``qpos`` and ``timestamps``) to an OpenSim ``.mot`` file.

    Args:
        fitted_model_path: Path to the _fitted_model.npz file
        output_path: Path where the .mot file should be saved

    Returns True on success, False (with the reason printed) otherwise.
    """
    if not os.path.exists(fitted_model_path):
        print(f"Error: Fitted model file not found: {fitted_model_path}")
        return False
    
    try:
        with open(fitted_model_path, "rb") as f:
            data = np.load(f, allow_pickle=True)
            print(f"Loaded biomechanics data: {fitted_model_path}")
            print(f"- Timesteps: {len(data['qpos'])}")
            print(f"- Joint positions shape: {data['qpos'].shape}")
            print(f"- Joint velocities shape: {data['qvel'].shape}")

            timestamps = data['timestamps']
            if len(timestamps) > 1:
                dt = float(timestamps[1] - timestamps[0])
                print(f"dt: {dt}")
            else:
                dt = 1.0/30.0

            qpos_df = pd.DataFrame(data['qpos'], columns=get_joint_names())
            # OpenSim measures knee flexion with the opposite sign
            qpos_df['knee_angle_r'] = - qpos_df['knee_angle_r']
            qpos_df['knee_angle_l'] = - qpos_df['knee_angle_l']
            
            # MuJoCo (z up) -> OpenSim (y up) for the pelvis position and orientation
            axis_transform_mat = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])
            txtytz = qpos_df[['pelvis_tx', 'pelvis_ty', 'pelvis_tz']].values
            txtytz_global = axis_transform_mat @ txtytz.T
            qpos_df[['pelvis_tx', 'pelvis_ty', 'pelvis_tz']] = txtytz_global.T

            euler_angles = qpos_df[['pelvis_tilt', 'pelvis_list', 'pelvis_rotation']].values
            
            rot_mats = R.from_euler('ZXY', euler_angles).as_matrix()
            transformed_rot_mats = np.einsum('ij,njk->nik', axis_transform_mat, rot_mats)
            transformed_euler = R.from_matrix(transformed_rot_mats).as_euler('ZXY')
            qpos_df[['pelvis_tilt', 'pelvis_list', 'pelvis_rotation']] = transformed_euler

            convertDataframeToMot(qpos_df, output_path, dt, data['timestamps'])
            print(f"Successfully exported OpenSim skeleton to: {output_path}")
            return True
            
    except Exception as e:
        print(f"Error converting to OpenSim format: {e}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export a fitted skeleton (.npz) as an OpenSim .mot file.")
    parser.add_argument("fitted_model", help="<stem>_fitted_model.npz or <stem>_bone_length.npz")
    parser.add_argument("output", help="output .mot path")
    args = parser.parse_args()
    raise SystemExit(0 if convert_fitted_model_to_mot(args.fitted_model, args.output) else 1)
