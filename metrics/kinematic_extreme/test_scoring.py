"""Regression cases with analytic scores, independent of learned models."""

from pathlib import Path

import numpy as np
import pytest

from metrics.bone_length.bone_length import BoneLengthConsistency, DEFAULT_TAU
from metrics.kinematic_extreme.main import (
    _frame_body_severity,
    _frame_joint_severity,
    compute_kextreme_score,
)
from metrics.kinematic_extreme.utils import (
    ForwardKinematics,
    MotData,
    PreloadedMotionData,
    _load_body_kin_cache,
    _save_body_kin_cache,
    BodyKinematicsData,
    load_mot,
    central_derivative,
    compute_joint_angular_velocities,
)
from metrics.motion_smoothness.main import _frame_severity, compute_msmooth_score
from metrics.motion_smoothness.utils import PreloadedMotionData as AccelerationData
from metrics.motion_smoothness.utils import compute_joint_vel_acc
from metrics.range_of_motion.joints_violaton import (
    _normalize_frame_violation,
    compute_rom_score_from_violations,
    eval_rot_delta,
    load_mot_file,
)


def test_nonviolating_joints_remain_in_severity_denominator():
    names = ["arm_flex_r", "arm_flex_l"]
    assert _frame_joint_severity(np.array([1800., 0.]), names)[0] == pytest.approx(.5)
    assert _frame_severity(np.array([45000., 0.]), names)[0] == pytest.approx(.5)
    assert _frame_body_severity(np.array([18., 0.]), ["pelvis", "torso"])[0] == pytest.approx(.5)
    assert _normalize_frame_violation(np.array([np.pi / 2, 0.]), names) == pytest.approx(.5)


def test_available_zero_body_term_is_averaged():
    n = 4
    data = PreloadedMotionData(
        time=np.arange(n), joint_names=["arm_flex_r", "arm_flex_l"],
        joint_angles=np.zeros((n, 2)), joint_vel_time=np.arange(n),
        joint_angular_vel=np.deg2rad(np.tile([1800., 0.], (n, 1))),
        body_names=["pelvis", "torso"], body_linear_vel_time=np.arange(n),
        body_linear_vel=np.zeros((n, 2)),
    )
    result = compute_kextreme_score("synthetic", data)
    assert result.mean_violation_severity == pytest.approx(.25)
    assert result.kextreme_score == pytest.approx(22.5)
    data.body_linear_vel = None
    result = compute_kextreme_score("synthetic", data)
    assert result.mean_violation_severity == pytest.approx(.5)
    assert result.kextreme_score == pytest.approx(15.)


def test_missing_or_nonfinite_data_cannot_score_perfectly():
    with pytest.raises(ValueError):
        compute_rom_score_from_violations("missing", {})
    with pytest.raises(ValueError):
        compute_rom_score_from_violations("invalid", {"arm_flex_r": np.array([np.nan])})
    with pytest.raises(ValueError):
        compute_msmooth_score("missing", AccelerationData([], np.array([]), np.empty((0, 0))))
    with pytest.raises(ValueError):
        compute_msmooth_score("invalid", AccelerationData(["arm_flex_r"], np.array([0.]), np.array([[np.nan]])))


def test_bone_default_matches_paper_and_five_frames_suffice():
    score, mean_error, *_ = BoneLengthConsistency.compute_stats_and_score(
        {"segment": np.array([.9, 1., 1., 1., 1.1])}
    )
    assert DEFAULT_TAU == .15
    assert mean_error == pytest.approx(.04)
    assert score == pytest.approx(100 * (1 - .04 / .15))
    with pytest.raises(ValueError):
        BoneLengthConsistency.compute_stats_and_score({"segment": np.ones(5)}, tau=float("nan"))


def test_bone_rejects_rigid_fit_invalid_timestamps_and_missing_detections():
    fitted = {"timestamps": np.arange(5, dtype=float), "joints": np.zeros((5, 22, 3)),
              "confidence": np.zeros((5, 87)), "scale_time_varying": True}
    lengths = BoneLengthConsistency(fitted).compute_bone_lengths()
    assert lengths == {}
    with pytest.raises(ValueError, match="No valid bones"):
        BoneLengthConsistency.compute_stats_and_score(lengths)
    fitted["scale_time_varying"] = False
    with pytest.raises(ValueError, match="variable-scale"):
        BoneLengthConsistency(fitted)
    fitted["scale_time_varying"] = True
    fitted["timestamps"][-1] = np.nan
    with pytest.raises(ValueError, match="finite and strictly increasing"):
        BoneLengthConsistency(fitted)
    fitted["timestamps"][-1] = 3.
    with pytest.raises(ValueError, match="finite and strictly increasing"):
        BoneLengthConsistency(fitted)


def test_degree_mot_converts_rotations_but_preserves_translations(tmp_path):
    path = tmp_path / "degrees.mot"
    path.write_text("inDegrees=yes\nendheader\ntime pelvis_tx hip_flexion_r\n0 10 90\n1 20 180\n2 30 90\n")
    mot = load_mot(str(path))
    np.testing.assert_allclose(mot.joint_angles[:, 0], [10, 20, 30])
    np.testing.assert_allclose(mot.joint_angles[:, 1], np.deg2rad([90, 180, 90]))
    rom_angles, names, _ = load_mot_file(str(path))
    np.testing.assert_allclose(rom_angles, mot.joint_angles)
    assert names == mot.joint_names
    assert eval_rot_delta(np.zeros((2, 1)), ["hip_flexion_r"])["hip_flexion_r"].sum() == 0


def test_body_cache_invalidates_subsecond_source_edits(tmp_path):
    import os
    mot_path = tmp_path / "source.mot"
    model_path = tmp_path / "model.xml"
    cache_path = tmp_path / "cache.npz"
    mot_path.write_text("abcd")
    model_path.write_text("model")
    data = BodyKinematicsData(["pelvis"], np.zeros((2, 1, 3)), np.zeros((1, 1)), np.array([.5]))
    _save_body_kin_cache(str(cache_path), str(mot_path), str(model_path), data)
    assert _load_body_kin_cache(str(cache_path), str(mot_path), str(model_path)) is not None
    stat = mot_path.stat()
    os.utime(mot_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    assert _load_body_kin_cache(str(cache_path), str(mot_path), str(model_path)) is None


@pytest.mark.parametrize("changed_dependency", ["implementation", "runtime", "included_xml", "mesh"])
def test_body_cache_invalidates_implementation_runtime_and_model_assets(tmp_path, monkeypatch, changed_dependency):
    import os
    from metrics.kinematic_extreme import utils

    implementation = tmp_path / "implementation.py"
    implementation.write_text("version = 1")
    monkeypatch.setattr(utils, "__file__", str(implementation))
    versions = {"mujoco": "version-1", "numpy": "version-1", "scipy": "version-1"}
    monkeypatch.setattr(utils, "_fk_runtime_versions", lambda: versions.copy())
    model = tmp_path / "model.xml"
    model.write_text('<mujoco><include file="included.xml"/></mujoco>')
    included = tmp_path / "included.xml"
    included.write_text('<mujocoinclude><asset><mesh file="mesh.stl"/></asset></mujocoinclude>')
    mesh = tmp_path / "mesh.stl"
    mesh.write_bytes(b"mesh-version-1")
    mot = tmp_path / "input.mot"
    mot.write_text("motion")
    cache = tmp_path / "cached.npz"
    data = BodyKinematicsData(["pelvis"], np.zeros((2, 1, 3)), np.zeros((1, 1)), np.array([.5]))
    utils._save_body_kin_cache(str(cache), str(mot), str(model), data)
    assert utils._load_body_kin_cache(str(cache), str(mot), str(model)) is not None

    if changed_dependency == "runtime":
        versions["mujoco"] = "version-2"
    else:
        changed = {"implementation": implementation, "included_xml": included, "mesh": mesh}[changed_dependency]
        stat = changed.stat()
        original = changed.read_bytes()
        # Same size and restored modification time still require new FK.
        changed.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
        os.utime(changed, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert utils._load_body_kin_cache(str(cache), str(mot), str(model)) is None


@pytest.mark.parametrize("changed_input", ["mot", "fitted"])
def test_body_cache_hashes_motion_and_companion_fit_when_timestamps_are_preserved(tmp_path, changed_input):
    import os
    from metrics.kinematic_extreme import utils

    model = tmp_path / "model.xml"
    model.write_text("<mujoco/>")
    mot = tmp_path / "clip_opensim_skeleton.mot"
    mot.write_text("time hip_flexion_r\n0 0\n1 0\n")
    fitted = tmp_path / "clip_fitted_model.npz"
    np.savez(fitted, qpos=np.zeros((2, 1)))
    cache = tmp_path / "cached.npz"
    data = BodyKinematicsData(["pelvis"], np.zeros((2, 1, 3)), np.zeros((1, 1)), np.array([.5]))
    utils._save_body_kin_cache(str(cache), str(mot), str(model), data)
    assert utils._load_body_kin_cache(str(cache), str(mot), str(model)) is not None

    changed = mot if changed_input == "mot" else fitted
    stat = changed.stat()
    if changed_input == "mot":
        mot.write_text("time hip_flexion_r\n0 0\n1 1\n")
    else:
        np.savez(fitted, qpos=np.ones((2, 1)))
    assert changed.stat().st_size == stat.st_size
    os.utime(changed, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert utils._load_body_kin_cache(str(cache), str(mot), str(model)) is None


def test_native_fk_inverts_export_maps_columns_and_uses_com():
    mujoco = pytest.importorskip("mujoco")
    from scipy.spatial.transform import Rotation

    model = Path(__file__).resolve().parents[2] / "baseline/monocular_demos/biomechanics_mjx/data/humanoid/humanoid_torque.xml"
    fk = ForwardKinematics(str(model))
    name_to_address = dict(zip(fk.joint_names, fk.qpos_addresses))
    qpos = np.tile(fk.model.qpos0, (3, 1))
    for name, value in {"pelvis_tx": .4, "pelvis_ty": .6, "pelvis_tz": .8, "pelvis_tilt": .2, "pelvis_list": -.3, "pelvis_rotation": -.4, "knee_angle_r": -.7}.items():
        qpos[:, name_to_address[name]] = value
    exported = qpos.copy()
    transform = np.array([[-1., 0., 0.], [0., 0., 1.], [0., 1., 0.]])
    translation = [name_to_address[name] for name in ("pelvis_tx", "pelvis_ty", "pelvis_tz")]
    rotation = [name_to_address[name] for name in ("pelvis_tilt", "pelvis_list", "pelvis_rotation")]
    exported[:, translation] = qpos[:, translation] @ transform.T
    exported[:, rotation] = Rotation.from_matrix(transform @ Rotation.from_euler("ZXY", qpos[:, rotation]).as_matrix()).as_euler("ZXY")
    for name in ("knee_angle_r", "knee_angle_l"):
        exported[:, name_to_address[name]] *= -1
    names = list(reversed(fk.joint_names))
    shuffled = exported[:, [name_to_address[name] for name in names]]
    recovered = fk.qpos_from_mot(MotData(np.arange(3), names, shuffled, metadata={"signed_pelvis_rotation": "yes"}))
    np.testing.assert_allclose(recovered, qpos, atol=1e-12)
    positions = fk.compute_body_positions(recovered)
    data = mujoco.MjData(fk.model)
    data.qpos[:] = qpos[0]
    mujoco.mj_forward(fk.model, data)
    np.testing.assert_allclose(positions[0], data.xipos[1:], atol=1e-12)
    assert np.linalg.norm(positions[0] - data.xpos[1:]) > 0


def test_bone_endpoints_cover_thigh_shank_arm_forearm_and_trunk():
    mujoco = pytest.importorskip("mujoco")
    model = Path(__file__).resolve().parents[2] / "baseline/monocular_demos/biomechanics_mjx/data/humanoid/humanoid_torque.xml"
    fk = ForwardKinematics(str(model))
    data = mujoco.MjData(fk.model)
    mujoco.mj_kinematics(fk.model, data)
    positions = np.tile(data.xpos, (5, 1, 1))
    analyzer = BoneLengthConsistency({"timestamps": np.arange(5), "joints": positions})
    lengths = analyzer.compute_bone_lengths()
    assert lengths["femur_r"][0] == pytest.approx(.36554883)
    assert lengths["tibia_r"][0] == pytest.approx(.480173)
    assert lengths["humerus_r"][0] == pytest.approx(np.linalg.norm([.0154412, -.336305, -.0112719]))
    assert lengths["forearm_r"][0] == pytest.approx(np.linalg.norm([-.00974948, -.261376, .0150836]))
    assert lengths["torso"][0] == pytest.approx(.541458)
    positions[-1, analyzer.name2idx["talus_r"], 0] += .1
    changed = BoneLengthConsistency({"timestamps": np.arange(5), "joints": positions}).compute_bone_lengths()
    assert changed["tibia_r"][-1] != pytest.approx(lengths["tibia_r"][-1])
    np.testing.assert_allclose(changed["femur_r"], lengths["femur_r"])


def test_central_derivatives_match_quadratic_on_actual_timestamps():
    times = np.array([0., .05, .1, .16, .22])
    angles = (.2 * times ** 2 + .5 * times)[:, None]
    motion = MotData(times, ["arm_flex_r"], angles)
    velocity = compute_joint_angular_velocities(motion)
    kinematics = compute_joint_vel_acc(motion)
    np.testing.assert_allclose(velocity.time_mid, times)
    np.testing.assert_allclose(velocity.angular_vel[:, 0], .4 * times + .5, atol=1e-12)
    np.testing.assert_allclose(kinematics.acc[:, 0], .4, atol=1e-12)


def test_derivative_stencils_do_not_cross_timestamp_gaps():
    times = np.array([0., .1, .2, 2., 2.1, 2.2])
    values = np.array([0., 0., 0., 2., 2., 2.])[:, None]
    for order in (1, 2):
        out_time, derivative = central_derivative(values, times, order=order)
        np.testing.assert_allclose(out_time, times)
        np.testing.assert_allclose(derivative, 0., atol=1e-12)


def test_paper_severity_sum_keeps_subthreshold_values_and_scores_bounded():
    n = 4
    data = PreloadedMotionData(
        time=np.arange(n), joint_names=["arm_flex_r"], joint_angles=np.zeros((n, 1)),
        joint_vel_time=np.arange(n), joint_angular_vel=np.deg2rad(np.full((n, 1), 1224.)),
        body_names=[], body_linear_vel_time=None, body_linear_vel=None,
    )
    result = compute_kextreme_score("subthreshold", data)
    assert result.frame_violation_rate == 0
    assert result.mean_violation_severity == pytest.approx(.16)
    assert result.kextreme_score == pytest.approx(95.2)
    acceleration = AccelerationData(["arm_flex_r"], np.arange(n), np.deg2rad(np.full((n, 1), 30600.)))
    assert compute_msmooth_score("subthreshold", acceleration).msmooth_score == pytest.approx(95.2)
    rom = compute_rom_score_from_violations("subthreshold", {"arm_flex_r": np.full(n, .04 * np.pi / 2)})
    assert rom.rom_score == pytest.approx(95.2)
    data.joint_angular_vel = np.repeat(data.joint_angular_vel, 100, axis=0)
    data.joint_vel_time = np.arange(len(data.joint_angular_vel))
    assert compute_kextreme_score("many_subthreshold", data).kextreme_score == 0
