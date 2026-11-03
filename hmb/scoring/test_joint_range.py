"""Tests of the joint-range metric (hmb.scoring.joint_range, metric III)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from hmb.scoring import joint_range as jr

mujoco = pytest.importorskip("mujoco")

NAMES = None


def model():
    return jr._Model()


def coordinate_names():
    global NAMES
    if NAMES is None:
        m = model()
        NAMES = [n for n in m.joint_names]
    return NAMES


def neutral(T=20):
    """A clean pose sequence: every scored DoF well inside its paper limits."""
    names = coordinate_names()
    q = np.zeros((T, len(names)))
    idx = {n: i for i, n in enumerate(names)}
    t = np.linspace(0, 2 * np.pi, T)
    for d in jr.SCORED_DOFS:
        lo, hi = jr.PAPER_LIMITS_DEG[jr.dof_key(d)]
        mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
        q[:, idx[d]] = np.radians(mid + 0.5 * half * np.sin(t + len(d)))
    return names, q, idx


def write_mot(path, names, q, time=None, in_degrees=False):
    time = np.arange(len(q)) / 24.0 if time is None else time
    vals = np.degrees(q) if in_degrees else q
    if in_degrees:
        for j, n in enumerate(names):
            if n in jr.TRANSLATIONS:
                vals[:, j] = q[:, j]
    lines = ["Coordinates", "version=1", f"nRows={len(q)}", f"nColumns={len(names) + 1}",
             f"inDegrees={'yes' if in_degrees else 'no'}", "", "endheader", "time\t" + "\t".join(names)]
    for t, row in zip(time, vals):
        lines.append(f"{t:.6f}\t" + "\t".join(repr(float(v)) for v in row))
    path.write_text("\n".join(lines) + "\n")


def fk_dev(m, names, a, b):
    xa = m.forward(m.qpos_from_coordinates(a, names))
    xb = m.forward(m.qpos_from_coordinates(b, names))
    return max(float(np.abs(u - v).max()) for u, v in zip(xa, xb))


# ---------------------------------------------------------------------------
# angle utilities
# ---------------------------------------------------------------------------

def test_wrap_to_pi_range_and_equivalence():
    x = np.linspace(-20, 20, 4001)
    w = jr.wrap_to_pi(x)
    assert np.all(w > -np.pi) and np.all(w <= np.pi)
    assert np.allclose(np.mod(x - w, 2 * np.pi) * (np.abs(np.mod(x - w, 2 * np.pi) - 2 * np.pi) > 1e-9), 0, atol=1e-9)
    assert jr.wrap_to_pi(-np.pi) == pytest.approx(np.pi)


def test_wrap_near_picks_representation_closest_to_centre():
    c = math.radians(45.0)  # centre of the arm_flex limits (-90, 180)
    y = jr.wrap_near(math.radians(-165.0), c)
    assert math.degrees(y) == pytest.approx(195.0)
    assert np.all(np.abs(jr.wrap_near(np.linspace(-9, 9, 101), c) - c) <= np.pi + 1e-12)


def test_limits_change_only_shoulder_elevation():
    changed = {k for k in jr.PAPER_LIMITS_DEG if jr.LIMITS_DEG[k] != jr.PAPER_LIMITS_DEG[k]}
    assert changed == {"arm_flex", "arm_add"}
    assert jr.LIMITS_DEG["arm_flex"] == (-90.0, 180.0)
    assert jr.LIMITS_DEG["arm_add"] == (-180.0, 90.0)
    # The rule: every AAOS norm lies inside limit +- tolerance.
    for k, (lo, hi) in jr.AAOS_LIMITS_DEG.items():
        vlo, vhi = jr.LIMITS_DEG[k]
        assert lo >= vlo - jr.TOLERANCE_DEG and hi <= vhi + jr.TOLERANCE_DEG


def test_movi87_order_matches_fitting_code():
    import ast
    from pathlib import Path

    src = (Path(jr.__file__).resolve().parents[2] / "baseline" / "monocular_demos" / "utils.py").read_text()
    tree = ast.parse(src)
    names = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "joint_names" for t in node.targets):
            names = tuple(ast.literal_eval(node.value))
    assert names == jr.MOVI87
    assert set(jr.MOVI87) <= set(model().site_names)


# ---------------------------------------------------------------------------
# exact equivalences (forward kinematics)
# ---------------------------------------------------------------------------

def test_euler_alternative_is_exact_for_every_ball_joint():
    m = model()
    names = coordinate_names()
    rng = np.random.default_rng(0)
    for _ in range(25):
        row = rng.uniform(-np.pi, np.pi, len(names))
        for group, dofs in jr.BALL_JOINTS.items():
            alt = row.copy()
            ia, ib, ic = (names.index(d) for d in dofs)
            alt[ia], alt[ib], alt[ic] = jr.euler_alternative(row[ia], row[ib], row[ic])
            assert not np.allclose(alt, row)
            assert fk_dev(m, names, row, alt) < 1e-9, group


def test_canonicalize_preserves_forward_kinematics_exactly():
    m = model()
    names = coordinate_names()
    rng = np.random.default_rng(1)
    q = rng.uniform(-3 * np.pi, 3 * np.pi, (30, len(names)))
    c = jr.canonicalize(q, names)
    assert c.wrapped and c.euler_switched
    for a, b in zip(q, c.angles):
        assert fk_dev(m, names, a, b) < 1e-9


def test_canonicalize_never_increases_violation():
    names = coordinate_names()
    rng = np.random.default_rng(2)
    q = rng.uniform(-3 * np.pi, 3 * np.pi, (200, len(names)))
    idx = [names.index(d) for d in jr.SCORED_DOFS]
    lo, hi = jr.limits_for(jr.SCORED_DOFS, jr.LIMITS_DEG)
    c = jr.canonicalize(q, names).angles
    w = jr.canonicalize(q, names, euler=False).angles
    sev = lambda a: jr.dof_severity(jr.dof_excess(a, lo, hi), lo, hi)
    for raw, wrapped, can in zip(q, w, c):
        # centred wrapping minimises every DoF's excess over all 2 pi shifts ...
        for k in (-2, -1, 0, 1, 2):
            assert np.all(jr.dof_excess(wrapped[idx], lo, hi) <= jr.dof_excess(raw[idx] + 2 * np.pi * k, lo, hi) + 1e-12)
        # ... and Euler switching never raises the worst per-DoF severity
        assert sev(can[idx]).max() <= sev(wrapped[idx]).max() + 1e-12 <= sev(raw[idx]).max() + 2e-12


def test_limb_flip_is_not_an_exact_equivalence():
    """The arm long-axis flip moves the elbow/forearm by centimetres (template FK)."""
    m = model()
    names = coordinate_names()
    row = np.zeros(len(names))
    i = {n: k for k, n in enumerate(names)}
    row[i["arm_rot_r"]], row[i["elbow_flex_r"]], row[i["pro_sup_r"]] = 0.3, 1.0, 1.4
    flip = row.copy()
    flip[i["arm_rot_r"]] += np.pi
    flip[i["elbow_flex_r"]] *= -1
    xa = m.forward(m.qpos_from_coordinates(row, names))[0]
    xb = m.forward(m.qpos_from_coordinates(flip, names))[0]
    ulna = m.body_names.index("ulna_r")
    assert np.linalg.norm(xa[ulna] - xb[ulna]) > 0.02


# ---------------------------------------------------------------------------
# scoring: controls, monotonicity, invariances
# ---------------------------------------------------------------------------

def test_clean_signal_scores_100():
    names, q, _ = neutral()
    for limits in (jr.PAPER_LIMITS_DEG, None):
        assert jr.score_angles(q, names, limits=limits).score == pytest.approx(100.0)


def paper_reference_score(angles, names, tolerance_deg: float = 15.0, threshold: float = 0.05) -> float:
    """Re-implementation of the paper scorer (``metrics/range_of_motion``) for comparisons.

    Paper limits, no wrapping, mean over the 21 DoFs, literal paper
    ``s = sum_t m_t / max(1, sum_t b_t)`` (sums sub-threshold frames too).
    """
    from hmb.scoring.aggregate import longest_run

    q = np.asarray(angles, dtype=float)
    index = {n: i for i, n in enumerate(names)}
    used = [d for d in jr.SCORED_DOFS if d in index]
    lo, hi = jr.limits_for(used, jr.PAPER_LIMITS_DEG)
    excess = jr.dof_excess(q[:, [index[d] for d in used]], lo, hi, tolerance_deg)
    m = jr.dof_severity(excess, lo, hi).mean(axis=1)
    b = m > threshold
    T = len(m)
    D = 0.5 * b.mean() + 0.3 * m.sum() / max(1, int(b.sum())) + 0.2 * longest_run(b) / T
    return float(100.0 * (1.0 - min(1.0, max(0.0, D))))


def test_parameterisation_artefacts_do_not_lower_the_score():
    names, q, idx = neutral()
    q2 = q.copy()
    q2[:, idx["elbow_flex_l"]] -= 2 * np.pi            # e.g. -341 deg instead of 19 deg
    q2[:, idx["mtp_angle_r"]] += 4 * np.pi
    a, b, c = (idx[d] for d in jr.BALL_JOINTS["shoulder_r"])
    q2[:, a], q2[:, b], q2[:, c] = jr.euler_alternative(q[:, a], q[:, b], q[:, c])
    assert paper_reference_score(q2, names) < 50.0       # the paper code counts these as violations
    assert jr.score_angles(q2, names).score == pytest.approx(100.0)
    assert jr.score_angles(q2, names, canonical=False).score < 100.0


def test_overhead_arm_is_not_a_violation():
    names, q, idx = neutral()
    q[:, idx["arm_flex_r"]] = math.radians(170.0)   # arm raised overhead (AAOS flexion 180)
    assert jr.score_angles(q, names, limits=jr.PAPER_LIMITS_DEG).score < 50.0
    assert jr.score_angles(q, names).score == pytest.approx(100.0)


def test_single_joint_violation_flags_the_frame():
    names, q, idx = neutral()
    q[5:15, idx["knee_angle_l"]] = math.radians(-60.0)   # hyperextended knee, fully saturated
    mx = jr.score_angles(q, names)
    assert mx.num_flagged == 10 and mx.score < 60.0


@pytest.mark.parametrize("dof", ["knee_angle_r", "elbow_flex_l", "hip_rotation_l", "arm_add_r", "lumbar_bending"])
def test_score_decreases_monotonically_with_defect_magnitude(dof):
    names, q, idx = neutral(40)
    lo, hi = jr.LIMITS_DEG[jr.dof_key(dof)]
    # a hinge angle and its 2 pi-equivalents are the same pose, so the largest
    # measurable excess is 180 - half range - tolerance (beyond it the angle is
    # closer to the opposite limit)
    max_excess = 180.0 - 0.5 * (hi - lo) - jr.TOLERANCE_DEG
    scores = []
    for excess in [e for e in (0, 5, 10, 20, 40, 80, 120) if e < max_excess]:
        qq = q.copy()
        qq[10:20, idx[dof]] = math.radians(lo - jr.TOLERANCE_DEG - excess)
        scores.append(jr.score_angles(qq, names).score)
    assert scores[0] == pytest.approx(100.0)
    assert all(b <= a + 1e-9 for a, b in zip(scores, scores[1:]))
    assert scores[-1] < scores[1]


def test_score_decreases_monotonically_with_defect_duration():
    names, q, idx = neutral(60)
    scores = []
    for n in (0, 1, 3, 10, 30, 60):
        qq = q.copy()
        qq[:n, idx["elbow_flex_r"]] = math.radians(-60.0)
        scores.append(jr.score_angles(qq, names).score)
    assert scores[0] == pytest.approx(100.0)
    assert all(b <= a + 1e-9 for a, b in zip(scores, scores[1:]))
    assert scores[-1] < 20.0


def test_frame_rate_invariance_by_frame_repetition():
    names, q, idx = neutral(30)
    q[4:9, idx["ankle_angle_r"]] = math.radians(-90.0)
    base = jr.score_angles(q, names).score
    up = jr.score_angles(np.repeat(q, 3, axis=0), names).score
    assert up == pytest.approx(base)


def test_left_right_symmetry():
    names, q, idx = neutral(30)
    a, b = q.copy(), q.copy()
    a[3:8, idx["hip_adduction_r"]] = math.radians(-90.0)
    b[3:8, idx["hip_adduction_l"]] = math.radians(-90.0)
    assert jr.score_angles(a, names).score == pytest.approx(jr.score_angles(b, names).score)


def test_invalid_frames_are_excluded():
    names, q, idx = neutral(30)
    q[0:10, idx["knee_angle_r"]] = math.radians(-80.0)
    valid = np.ones(30, bool)
    valid[0:10] = False
    r = jr.score_angles(q, names, valid=valid)
    assert r.score == pytest.approx(100.0) and r.num_valid == 20 and r.coverage == pytest.approx(2 / 3)


def test_paper_reference_matches_the_paper_module(tmp_path):
    pytest.importorskip("pandas")
    from metrics.range_of_motion.joints_violaton import compute_rom_score_from_violations, eval_rot_delta

    names, q, idx = neutral(40)
    q[5:12, idx["arm_flex_l"]] = math.radians(150.0)
    q[20:30, idx["knee_angle_r"]] = math.radians(-50.0)
    q[20:30, idx["elbow_flex_r"]] = math.radians(-60.0)
    paper = compute_rom_score_from_violations("x", eval_rot_delta(q, names, tol_deg=15.0)).rom_score
    assert paper_reference_score(q, names) == pytest.approx(paper, abs=1e-9)


def test_load_mot_roundtrip_radians_and_degrees(tmp_path):
    names, q, _ = neutral(5)
    names = list(names)
    for deg in (False, True):
        p = tmp_path / f"x_{deg}.mot"
        write_mot(p, names, q, in_degrees=deg)
        n2, q2, t2 = jr.load_mot(p)
        assert n2 == names and np.allclose(q2, q, atol=1e-9) and len(t2) == 5


def test_rejects_bad_input():
    names, q, _ = neutral(5)
    with pytest.raises(ValueError):
        jr.score_angles(q[:, :-1], names)
    bad = q.copy()
    bad[0, 7] = np.nan
    with pytest.raises(ValueError):
        jr.score_angles(bad, names)
    with pytest.raises(ValueError):
        jr.reduce_frames(np.zeros(3))


# ---------------------------------------------------------------------------
# observation-verified limb flips and session entry point
# ---------------------------------------------------------------------------

def _synthetic_session(tmp_path, true_rows, fitted_rows, stem="clip"):
    """Session whose keypoints are exact FK of ``true_rows`` and whose fit is ``fitted_rows``."""
    m = model()
    names = coordinate_names()
    order = [m.site_names.index(s) for s in jr.MOVI87]
    kp, qpos = [], []
    for tr, fr in zip(true_rows, fitted_rows):
        kp.append(m.forward(m.qpos_from_coordinates(tr, names))[1][order] * 1000.0)
        qpos.append(m.qpos_from_coordinates(fr, names))
    base = tmp_path / "baseline"
    base.mkdir(parents=True, exist_ok=True)
    T = len(true_rows)
    np.savez(base / f"{stem}_keypoints.npz", keypoints3d=np.asarray(kp), confs=np.ones((T, 87)))
    np.savez(base / f"{stem}_fitted_model.npz", qpos=np.asarray(qpos), timestamps=np.arange(T) / 24.0,
             body_scale_factors=np.ones((m.model.nbody, 1)), site_offsets=np.zeros((m.model.nsite, 3)))
    write_mot(base / f"{stem}_opensim_skeleton.mot", names, np.asarray(fitted_rows))
    return tmp_path, names


def _flipped_fit(true_row, names, chain="arm_r"):
    """Best long-axis-flipped configuration for the true pose (what a stuck fit returns)."""
    from scipy.optimize import least_squares

    m = model()
    order = [m.site_names.index(s) for s in jr.MOVI87]
    c = jr.LIMB_CHAINS[chain]
    dofs = list(c["dofs"])
    k_idx = [k for k, s in enumerate(jr.MOVI87) if m.site_body[order[k]] in c["bodies"]]
    target = m.forward(m.qpos_from_coordinates(true_row, names))[1][order][k_idx]
    i = {n: k for k, n in enumerate(names)}

    def res(x):
        row = true_row.copy()
        for d, v in zip(dofs, x):
            row[i[d]] = v
        return (m.forward(m.qpos_from_coordinates(row, names))[1][order][k_idx] - target).ravel()

    x0 = np.array([true_row[i[d]] for d in dofs])
    x0[2] += np.pi
    x0[3] = -x0[3]
    x0[4] += np.pi
    sol = least_squares(res, x0, method="lm")
    row = true_row.copy()
    for d, v in zip(dofs, sol.x):
        row[i[d]] = v
    return row


def test_flip_resolution_recovers_the_anatomical_arm(tmp_path):
    names, q, idx = neutral(6)
    true = q.copy()
    true[:, idx["arm_rot_r"]] = math.radians(20.0)
    true[:, idx["elbow_flex_r"]] = math.radians(70.0)
    true[:, idx["pro_sup_r"]] = math.radians(80.0)
    fitted = np.stack([_flipped_fit(r, names) for r in true])
    assert jr.score_angles(fitted, names).score < 50.0          # the stuck fit looks like a violation
    s, names = _synthetic_session(tmp_path, true, fitted)
    out = jr.score_session(s)
    assert out["flip_resolution"]["accepted"]["arm_r"] == 6
    assert out["score"] == pytest.approx(100.0)


def test_flip_resolution_keeps_a_violation_supported_by_keypoints(tmp_path):
    names, q, idx = neutral(6)
    true = q.copy()
    true[:, idx["arm_rot_r"]] = 0.0
    true[:, idx["elbow_flex_r"]] = math.radians(-70.0)        # genuinely hyperextended elbow
    s, names = _synthetic_session(tmp_path, true, true)
    out = jr.score_session(s)
    assert out["flip_resolution"]["accepted"]["arm_r"] == 0
    assert out["score"] < 50.0


def test_score_session_masks_undetected_frames(tmp_path):
    names, q, idx = neutral(10)
    bad = q.copy()
    bad[:4, idx["knee_angle_l"]] = math.radians(-80.0)
    s, names = _synthetic_session(tmp_path, q, bad)
    kp = np.load(s / "baseline" / "clip_keypoints.npz")
    confs = np.ones((10, 87))
    confs[:4] = 0
    np.savez(s / "baseline" / "clip_keypoints.npz", keypoints3d=kp["keypoints3d"], confs=confs)
    out = jr.score_session(s)
    assert out["detection_mask"] and out["num_valid"] == 6 and out["score"] == pytest.approx(100.0)
    assert jr.score_angles(bad, names).score < 100.0     # without the detection mask


def test_observation_mask_removes_violations_of_unobserved_dofs():
    names, q, idx = neutral(12)
    bad = q.copy()
    bad[:3, idx["elbow_flex_l"]] = math.radians(-80.0)
    every = np.ones((12, len(jr.SCORED_DOFS)), bool)
    assert jr.score_angles(bad, names, observed=every).score == pytest.approx(jr.score_angles(bad, names).score)
    hidden = every.copy()
    hidden[:3, jr.SCORED_DOFS.index("elbow_flex_l")] = False
    assert jr.score_angles(bad, names, observed=hidden).score == pytest.approx(100.0)
    other = every.copy()
    other[:3, jr.SCORED_DOFS.index("elbow_flex_r")] = False      # masking another DoF changes nothing
    assert jr.score_angles(bad, names, observed=other).score == pytest.approx(jr.score_angles(bad, names).score)
    with pytest.raises(ValueError):
        jr.score_angles(bad, names, observed=every[:, :5])


def test_observed_dofs_from_image_bounds(tmp_path):
    T = 3
    P2 = np.full((T, 87, 2), 50.0)
    ix = {n: i for i, n in enumerate(jr.MOVI87)}
    P2[1, ix["LWrist"]] = (-5.0, 50.0)                 # left wrist out of view on frame 1
    P2[2, ix["RAnkle"]] = (50.0, 130.0)                # right ankle below the image on frame 2
    path = tmp_path / "kp.npz"
    np.savez(path, keypoints2d=P2, image_size=np.array([100, 120]))
    obs = jr.observed_dofs(path, T)
    col = {d: j for j, d in enumerate(jr.SCORED_DOFS)}
    assert obs.shape == (T, len(jr.SCORED_DOFS)) and obs[0].all()
    assert not obs[1, col["elbow_flex_l"]] and not obs[1, col["arm_flex_l"]] and obs[1, col["elbow_flex_r"]]
    assert not obs[2, col["knee_angle_r"]] and not obs[2, col["ankle_angle_r"]] and obs[2, col["knee_angle_l"]]
    np.savez(tmp_path / "old.npz", keypoints3d=np.zeros((T, 87, 3)))
    assert jr.observed_dofs(tmp_path / "old.npz", T) is None
