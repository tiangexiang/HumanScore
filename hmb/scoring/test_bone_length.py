"""Tests of the bone-length consistency metric (hmb.scoring.bone_length, MeTRAbs keypoints, no fit)."""
from __future__ import annotations

import ast
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from hmb.scoring import bone_length as bl

REPO = Path(__file__).resolve().parents[2]
W, H, F = 1280, 720, 1000.0
#: Joint centres of every synthetic body point (the scored endpoints plus head, toes and hands).
CENTRE_ENDPOINTS = {**bl.ENDPOINTS, "head": ("Head",), "toe_l": ("LFoot",), "toe_r": ("RFoot",),
                    "hand_l": ("LHand",), "hand_r": ("RHand",)}


def _rot(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * k @ k


def skeleton(T: int = 120, fps: float = 24.0, seed: int = 0, depth: float = 4000.0, stretch=None,
             scale=None, unit: float = 1.0):
    """Synthetic walking-like stick figure in camera coordinates (mm, y down).

    ``stretch``: dict segment -> (T,) multiplicative length factor applied to
    the distal endpoint of that segment (and everything below it).
    ``scale``: (T,) per-frame uniform scale of the whole skeleton about the pelvis.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(T) / fps
    stretch = stretch or {}
    one = np.ones(T)
    idx = {n: i for i, n in enumerate(bl.JOINT_NAMES)}
    pts = np.zeros((T, 87, 3))
    lengths = {"thigh": 420.0, "shank": 400.0, "upper_arm": 290.0, "forearm": 260.0, "trunk": 500.0,
               "hip_half": 90.0, "shoulder_half": 180.0, "head": 180.0, "foot": 140.0, "hand": 80.0}
    phase = rng.uniform(0, 2 * np.pi)
    for i in range(T):
        w = 2 * np.pi * 1.0 * t[i] + phase
        pelvis = np.array([200.0 * math.sin(0.3 * w), 0.0, depth])
        yaw = _rot(np.array([0.0, 1.0, 0.0]), 0.6 * math.sin(0.2 * w))
        up = yaw @ np.array([0.0, -1.0, 0.0])
        side = yaw @ np.array([1.0, 0.0, 0.0])
        neck = pelvis + lengths["trunk"] * stretch.get("trunk", one)[i] * up
        P = {"pelvis": pelvis, "neck": neck, "head": neck + lengths["head"] * up}
        for s, sgn in (("l", 1.0), ("r", -1.0)):
            hip = pelvis + sgn * lengths["hip_half"] * side
            sh = neck + sgn * lengths["shoulder_half"] * side
            swing = sgn * 0.5 * math.sin(w)
            thigh_dir = _rot(side, swing) @ (-up)
            knee = hip + lengths["thigh"] * stretch.get(f"thigh_{s}", one)[i] * thigh_dir
            shank_dir = _rot(side, swing - 0.4 * (1 + math.sin(w))) @ (-up)
            ankle = knee + lengths["shank"] * stretch.get(f"shank_{s}", one)[i] * shank_dir
            toe = ankle + lengths["foot"] * (yaw @ np.array([0.0, 0.0, -1.0]))
            arm_dir = _rot(side, -swing) @ (-up)
            elbow = sh + lengths["upper_arm"] * stretch.get(f"upper_arm_{s}", one)[i] * arm_dir
            fore_dir = _rot(side, -swing + 0.3 + 0.3 * math.sin(w)) @ (-up)
            wrist = elbow + lengths["forearm"] * stretch.get(f"forearm_{s}", one)[i] * fore_dir
            hand = wrist + lengths["hand"] * fore_dir
            P.update({f"hip_{s}": hip, f"knee_{s}": knee, f"ankle_{s}": ankle, f"toe_{s}": toe,
                      f"shoulder_{s}": sh, f"elbow_{s}": elbow, f"wrist_{s}": wrist, f"hand_{s}": hand})
        c = P["pelvis"]
        sc = 1.0 if scale is None else float(scale[i])
        pts[i] = c  # unused markers sit at the pelvis
        for name, markers in CENTRE_ENDPOINTS.items():
            pts[i, idx[markers[0]]] = c + sc * (P[name] - c)
        # medial/lateral marker pairs straddle the joint centre
        for joint, (a, b) in (("knee", ("knem", "innerknee")), ("elbow", ("elbm", "elbinner")),
                              ("wrist", ("writhumbside", "wripinkieside"))):
            for s in ("l", "r"):
                centre = c + sc * (P[f"{joint}_{s}"] - c)
                off = sc * 45.0 * side
                pts[i, idx[f"{s}{a}"]] = centre + off
                pts[i, idx[f"{s}{b}"]] = centre - off
        pts[i, idx["CHip"]] = pts[i, idx["mhip"]]
    pts *= unit
    p2 = np.stack([F * pts[..., 0] / pts[..., 2] + W / 2, F * pts[..., 1] / pts[..., 2] + H / 2], axis=-1)
    return {"keypoints3d": pts, "keypoints2d": p2, "confs": np.ones((T, 87)), "timestamps": t,
            "image_size": (W, H)}


def score(kp, cfg=bl.DEFAULT_CONFIG, **kw):
    return bl.score_arrays(kp["keypoints3d"], kp["confs"], kp["keypoints2d"], kw.pop("image_size", kp["image_size"]),
                           kp["timestamps"], cfg=cfg, **kw)


def run_factor(T, start, stop, value):
    f = np.ones(T)
    f[start:stop] = value
    return f


def test_joint_names_match_metrabs_declaration():
    module = ast.parse((REPO / "baseline/monocular_demos/utils.py").read_text())
    for node in module.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "joint_names" for t in node.targets):
            assert tuple(ast.literal_eval(node.value)) == bl.JOINT_NAMES
            return
    pytest.fail("joint_names not found")


def test_segment_lengths_of_synthetic_skeleton_are_constant():
    kp = skeleton()
    L = bl.segment_lengths(kp["keypoints3d"])
    assert L.shape == (120, len(bl.SEGMENTS))
    np.testing.assert_allclose(L.std(axis=0), 0.0, atol=1e-6)
    j = bl.SEGMENTS.index("thigh_l")
    np.testing.assert_allclose(L[:, j], 420.0, rtol=1e-9)


def test_clean_motion_scores_100():
    out = score(skeleton())
    assert out["bone_length_score"] == pytest.approx(100.0)
    assert out["supported"] and out["coverage"] == pytest.approx(1.0)
    assert out["mean_relative_deviation"] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("segment", ["forearm_l", "thigh_r", "trunk"])
def test_injected_stretch_lowers_score_monotonically(segment):
    T = 120
    scores = []
    for a in (0.0, 0.05, 0.15, 0.25, 0.4, 0.6):
        kp = skeleton(T, stretch={segment: run_factor(T, 40, 70, 1 + a)})
        scores.append(score(kp)["bone_length_score"])
    assert scores[0] == pytest.approx(100.0)
    assert all(b < a for a, b in zip(scores, scores[1:]))  # strictly decreasing
    assert scores[-1] < scores[0] - 2.0


def test_mean_form_matches_the_paper_formula():
    """E = mean over segments of the time-mean relative deviation; score = 100 (1 - E / tau)."""
    T = 120
    kp = skeleton(T, stretch={"forearm_l": run_factor(T, 40, 70, 1.3)})
    out = score(kp)
    n_seg = len(bl.SEGMENTS)
    expected_E = (30 / 120) * 0.3 / n_seg
    assert out["mean_relative_deviation"] == pytest.approx(expected_E, rel=1e-6)
    assert out["bone_length_score"] == pytest.approx(100 * (1 - expected_E / bl.TAU_MEAN), rel=1e-6)
    assert out["per_segment"]["forearm_l"]["mean_deviation"] == pytest.approx((30 / 120) * 0.3, rel=1e-6)


def test_defect_longer_than_half_the_clip_still_lowers_the_score():
    # the reference is the per-segment median, so a defect covering most of the
    # clip becomes the reference and the *normal* frames deviate instead
    T = 120
    out = score(skeleton(T, stretch={"upper_arm_l": run_factor(T, 0, 80, 1.3)}))
    assert out["bone_length_score"] < 100.0
    assert out["per_segment"]["upper_arm_l"]["mean_deviation"] > 0.05


def test_shrinking_is_penalised_like_stretching():
    T = 120
    up = score(skeleton(T, stretch={"shank_l": run_factor(T, 30, 60, 1.3)}))["bone_length_score"]
    down = score(skeleton(T, stretch={"shank_l": run_factor(T, 30, 60, 1 / 1.3)}))["bone_length_score"]
    assert up < 100 and down < 100
    assert down == pytest.approx(up, abs=1.0)


def test_longer_defect_scores_lower():
    T = 120
    s = [score(skeleton(T, stretch={"upper_arm_r": run_factor(T, 20, 20 + n, 1.35)}))["bone_length_score"]
         for n in (3, 10, 30, 60)]
    assert all(b < a for a, b in zip(s, s[1:]))


def test_invariant_to_per_frame_global_scale_rigid_motion_and_units():
    T = 120
    rng = np.random.default_rng(3)
    defect = {"forearm_l": run_factor(T, 40, 70, 1.3)}
    ref = score(skeleton(T, stretch=defect))
    scale = np.exp(rng.uniform(np.log(0.8), np.log(1.25), T))
    scaled = score(skeleton(T, stretch=defect, scale=scale))
    assert scaled["bone_length_score"] == pytest.approx(ref["bone_length_score"], abs=1e-6)
    clean_scaled = score(skeleton(T, scale=scale))
    assert clean_scaled["bone_length_score"] == pytest.approx(100.0)
    metres = skeleton(T, stretch=defect, unit=1e-3)
    assert score(metres)["bone_length_score"] == pytest.approx(ref["bone_length_score"], abs=1e-6)
    # rigid motion of the whole body (rotation about the camera axis + shift)
    kp = skeleton(T, stretch=defect)
    R = _rot(np.array([0.2, 0.1, 1.0]), 0.7)
    moved = dict(kp)
    moved["keypoints3d"] = kp["keypoints3d"] @ R.T + np.array([100.0, -50.0, 300.0])
    out = bl.score_arrays(moved["keypoints3d"], kp["confs"], None, None, kp["timestamps"])
    ref_novis = bl.score_arrays(kp["keypoints3d"], kp["confs"], None, None, kp["timestamps"])
    assert out["bone_length_score"] == pytest.approx(ref_novis["bone_length_score"], abs=1e-6)


def test_single_frame_glitch_is_filtered_but_persistent_defect_is_not():
    T = 120
    glitch = score(skeleton(T, stretch={"thigh_l": run_factor(T, 50, 51, 1.5)}))
    assert glitch["bone_length_score"] == pytest.approx(100.0)
    unfiltered = score(skeleton(T, stretch={"thigh_l": run_factor(T, 50, 51, 1.5)}),
                       replace(bl.DEFAULT_CONFIG, filter_seconds=0.0))
    assert unfiltered["bone_length_score"] < 100.0
    persistent = score(skeleton(T, stretch={"thigh_l": run_factor(T, 50, 53, 1.5)}))
    assert persistent["bone_length_score"] < 100.0


def test_frame_rate_invariance_of_time_based_filter():
    out = {}
    for fps in (24.0, 60.0):
        T = int(round(5.0 * fps))
        f = np.ones(T)
        t = np.arange(T) / fps
        f[(t >= 2.0) & (t < 2.5)] = 1.35
        out[fps] = score(skeleton(T, fps=fps, stretch={"forearm_r": f}))["bone_length_score"]
    assert out[24.0] < 100 and out[60.0] < 100
    assert abs(out[24.0] - out[60.0]) < 0.5


def test_out_of_image_segments_are_not_scored():
    T = 120
    kp = skeleton(T, stretch={"shank_l": run_factor(T, 40, 80, 1.6)})
    # move the left ankle below the image bottom during the stretched frames
    idx = bl.JOINT_NAMES.index("LAnkle")
    kp["keypoints2d"][40:80, idx, 1] = H + 50
    masked = score(kp)
    assert masked["bone_length_score"] == pytest.approx(100.0)
    unmasked = score(kp, image_size=None)
    assert unmasked["bone_length_score"] < 100.0


def test_undetected_frames_are_excluded_and_empty_clips_unsupported():
    T = 120
    kp = skeleton(T, stretch={"forearm_l": run_factor(T, 40, 70, 1.5)})
    kp["confs"][40:70] = 0.0
    assert score(kp)["bone_length_score"] == pytest.approx(100.0)
    kp["confs"][:] = 0.0
    out = score(kp)
    assert not out["supported"] and math.isnan(out["bone_length_score"])


def test_global_scale_is_robust_to_one_stretched_segment():
    T = 60
    lengths = np.tile(np.array([400.0, 380, 290, 260, 400, 380, 290, 260, 500, 180, 360]), (T, 1))
    lengths[20:40, 3] *= 1.4
    res = bl.log_residuals(lengths, np.ones_like(lengths, bool))
    np.testing.assert_allclose(res["log_scale"], 0.0, atol=1e-12)
    np.testing.assert_allclose(res["residual"][20:40, 3], np.log(1.4), atol=1e-12)
    np.testing.assert_allclose(np.delete(res["residual"], 3, axis=1), 0.0, atol=1e-12)


def test_temporal_median_uses_time_window():
    t = np.array([0.0, 0.04, 0.08, 0.12, 0.16])
    x = np.array([[0.0], [0.0], [1.0], [0.0], [0.0]])
    m = np.ones_like(x, bool)
    np.testing.assert_allclose(bl.temporal_median(x, m, t, 0.1)[:, 0], 0.0)
    np.testing.assert_allclose(bl.temporal_median(x, m, t, 0.0)[:, 0], x[:, 0])


def test_score_session_reads_engine_layout(tmp_path):
    T = 96
    kp = skeleton(T, stretch={"upper_arm_l": run_factor(T, 30, 60, 1.4)})
    stem = "clip"
    (tmp_path / "baseline").mkdir()
    np.savez(tmp_path / "baseline" / f"{stem}_keypoints.npz", keypoints3d=kp["keypoints3d"],
             keypoints2d=kp["keypoints2d"].astype(np.float32), confs=kp["confs"], timestamps=kp["timestamps"],
             image_size=np.array(kp["image_size"]), fps=np.array(24.0))
    out = bl.score_session(tmp_path, stem)
    direct = score(kp)
    assert out["bone_length_score"] == pytest.approx(direct["bone_length_score"])
    assert out["visibility_mask"] is True
    # without a stem the session's keypoints file is found
    assert bl.score_session(tmp_path)["bone_length_score"] == pytest.approx(direct["bone_length_score"])
    json.dumps(out, default=str)


def test_plan_timestamps_and_size_fallback(tmp_path):
    T = 48
    kp = skeleton(T)
    (tmp_path / "baseline").mkdir()
    (tmp_path / "input").mkdir()
    np.savez(tmp_path / "baseline" / "c_keypoints.npz", keypoints3d=kp["keypoints3d"],
             keypoints2d=kp["keypoints2d"], confs=kp["confs"])
    plan = {"video": "c.mp4", "source_fps": 24.0, "num_source_frames": T, "width": 2560, "height": 1440,
            "sample_fps": None, "indices": list(range(T)), "timestamps": list(kp["timestamps"]), "max_side": 1280}
    (tmp_path / "input" / "frames.json").write_text(json.dumps(plan))
    out = bl.score_session(tmp_path, "c")
    assert out["image_size"] == [1280, 720]
    assert out["bone_length_score"] == pytest.approx(100.0)
