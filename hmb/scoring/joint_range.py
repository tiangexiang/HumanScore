"""Metric (III) joint range of motion of the fitted biomechanical skeleton.

The paper (App. F.1) scores every frame of the fitted OpenSim (Rajagopal) skeleton
against per-DoF limits with a 15 degree tolerance::

    excess_t^d = max(0, theta - (hi + tol), (lo - tol) - theta)
    m_t^d      = min(excess_t^d / (0.5 (hi - lo)), 1)
    b_t        = 1[m_t > 0.05]

and aggregates r/s/p into a 0-100 score.  HumanScore keeps these formulas, the
tolerance, the half-range normalisation, the 0.05 flag threshold, the 21 scored
DoFs and the 0.5/0.3/0.2 aggregation, and measures the angles so that only real
limit violations count:

1. **Angle representation.**  The skeleton fit is unconstrained, so the MOT can
   contain angles such as elbow -341 deg (= +19 deg) or a hip/shoulder Euler
   triplet expressed with the "other" Euler solution.  Both describe exactly the
   same pose.  Every scored coordinate is mapped to its 2 pi-equivalent closest to
   the centre of its limit interval and, for each 3-DoF ball joint, the Euler
   solution with the smaller violation is kept (:func:`canonicalize`).  A pose is
   "inside the limits" iff *some* parameterisation of it is.  Consequence: the
   largest measurable excess of a DoF is 180 deg - half range - tolerance.
2. **Limb flips the keypoints cannot distinguish.**  The fit sometimes rotates a
   humerus or femur by ~180 deg about its long axis and bends the hinge backwards.
   When the session has keypoints and the fitted model, :func:`resolve_limbs`
   re-fits each violating limb chain to its MeTRAbs keypoints from several starts
   (flip, distal reset, clipped into the limits, a bounded in-limit fit, the
   neighbouring frame) and accepts the least violating alternative whose keypoint
   RMS is at most ``FLIP_RMS_MARGIN_MM`` worse than the fit.  It runs once with the
   template marker placement and once with the fitted marker offsets.
3. **Unobserved joints.**  A DoF whose defining keypoints (e.g. shoulder, elbow and
   wrist for the arm DoFs) project outside the image is not scored on that frame;
   frames without a person detection are removed from r/s/p.
4. **Per-frame reduction.**  ``m_t`` is the worst DoF, so a single joint outside
   its limits flags the frame.
5. **Limits** (``LIMITS_DEG``): the paper's Rajagopal ranges, widened only where
   the AAOS normative active range of motion lies beyond limit + tolerance.  Net
   change: shoulder elevation, arm_flex upper 90 -> 180 deg (AAOS flexion 180) and
   arm_add lower -120 -> -180 deg (AAOS abduction 180).  No limit is fitted to data.

Inputs: ``<S>/baseline/<stem>_opensim_skeleton.mot`` (rigid skeleton fit),
``_fitted_model.npz`` and ``_keypoints.npz``.  Output: a result dict (``score``, ``r``,
``s``, ``p``, per-DoF violation rates, what was resolved).

Run it on its own::

    python -m hmb.scoring.joint_range --session-dir S
    python -m hmb.scoring.joint_range --mot clip_opensim_skeleton.mot   (no masks, no limb resolution)
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from hmb.scoring.aggregate import PAPER_WEIGHTS, aggregate
from hmb.scoring.bone_length import JOINT_NAMES

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Paper tolerance (Catelli et al. 2019), unchanged.
TOLERANCE_DEG = 15.0
#: Paper severity normalisation: excess / (SATURATION_FRACTION * (hi - lo)).
SATURATION_FRACTION = 0.5
#: Paper per-(joint, DoF) flag threshold on the normalised severity.
FRAME_THRESHOLD = 0.05
#: Paper aggregation weights (alpha, beta, gamma).
WEIGHTS = PAPER_WEIGHTS
#: Flip resolution: an anatomical alternative of a limb is accepted when its
#: keypoint RMS on that limb's markers is at most this much worse than the
#: (per-frame refined) fitted configuration: half a typical per-limb fit residual and
#: well below the 3-D error of monocular keypoints.
FLIP_RMS_MARGIN_MM = 10.0

#: The 21 DoFs scored by the paper (pelvis, subtalar, mtp, pro_sup and
#: wrist are not scored; neck is not part of the Rajagopal limit table).
SCORED_DOFS: Tuple[str, ...] = (
    "hip_flexion_r", "hip_adduction_r", "hip_rotation_r", "knee_angle_r", "ankle_angle_r",
    "hip_flexion_l", "hip_adduction_l", "hip_rotation_l", "knee_angle_l", "ankle_angle_l",
    "lumbar_extension", "lumbar_bending", "lumbar_rotation",
    "arm_flex_r", "arm_add_r", "arm_rot_r", "elbow_flex_r",
    "arm_flex_l", "arm_add_l", "arm_rot_l", "elbow_flex_l",
)

#: Rajagopal et al. 2016 coordinate ranges as used by the paper
#: (``metrics/range_of_motion/definition.py``), OpenSim sign conventions.
PAPER_LIMITS_DEG: Dict[str, Tuple[float, float]] = {
    "hip_flexion": (-30.0, 120.0),
    "hip_adduction": (-50.0, 30.0),
    "hip_rotation": (-40.0, 40.0),
    "knee_angle": (0.0, 120.0),
    "ankle_angle": (-40.0, 30.0),
    "lumbar_extension": (-90.0, 90.0),
    "lumbar_bending": (-90.0, 90.0),
    "lumbar_rotation": (-90.0, 90.0),
    "arm_flex": (-90.0, 90.0),
    "arm_add": (-120.0, 90.0),
    "arm_rot": (-90.0, 90.0),
    "elbow_flex": (0.0, 150.0),
}

#: AAOS normative active ROM mapped to the OpenSim coordinate signs
#: (hip_adduction < 0 = abduction; ankle_angle > 0 = dorsiflexion;
#: arm_add < 0 = abduction; knee/elbow > 0 = flexion).  Only DoFs with a
#: clinical counterpart are listed.  Values: hip flexion 120 / extension 30,
#: abduction 45 / adduction 30, internal/external rotation 45/45; knee flexion
#: 135; ankle plantarflexion 50 / dorsiflexion 20; shoulder flexion 180 /
#: extension 60, abduction 180, internal 70 / external 90 rotation; elbow 150.
AAOS_LIMITS_DEG: Dict[str, Tuple[float, float]] = {
    "hip_flexion": (-30.0, 120.0),
    "hip_adduction": (-45.0, 30.0),
    "hip_rotation": (-45.0, 45.0),
    "knee_angle": (0.0, 135.0),
    "ankle_angle": (-50.0, 20.0),
    "arm_flex": (-60.0, 180.0),
    "arm_add": (-180.0, 0.0),
    "arm_rot": (-70.0, 90.0),
    "elbow_flex": (0.0, 150.0),
}


def _widen_where_norms_exceed(paper: Mapping[str, Tuple[float, float]],
                              norms: Mapping[str, Tuple[float, float]],
                              tolerance_deg: float) -> Dict[str, Tuple[float, float]]:
    """Paper limits, widened to the normative bound only where that bound lies
    beyond paper limit +- tolerance (i.e. the tolerance cannot absorb it)."""
    out = dict(paper)
    for key, (n_lo, n_hi) in norms.items():
        lo, hi = out[key]
        if n_lo < lo - tolerance_deg:
            lo = n_lo
        if n_hi > hi + tolerance_deg:
            hi = n_hi
        out[key] = (lo, hi)
    return out


#: HumanScore limits.  Net changes vs the paper: arm_flex (-90, 90) -> (-90, 180) and
#: arm_add (-120, 90) -> (-180, 90).  Everything else is the paper's Rajagopal range.
LIMITS_DEG: Dict[str, Tuple[float, float]] = _widen_where_norms_exceed(
    PAPER_LIMITS_DEG, AAOS_LIMITS_DEG, TOLERANCE_DEG)

#: 3-DoF ball joints of the biomechanics model.  In MuJoCo several hinges in
#: one body compose intrinsically, R = Rz(a) Rx(+-b) Ry(+-c); for any such
#: triplet (a, b, c) and (a + pi, pi - b, c + pi) give the same rotation
#: (the left side uses -x/-y axes, for which the same identity holds mod 2 pi).
BALL_JOINTS: Dict[str, Tuple[str, str, str]] = {
    "hip_r": ("hip_flexion_r", "hip_adduction_r", "hip_rotation_r"),
    "hip_l": ("hip_flexion_l", "hip_adduction_l", "hip_rotation_l"),
    "lumbar": ("lumbar_extension", "lumbar_bending", "lumbar_rotation"),
    "shoulder_r": ("arm_flex_r", "arm_add_r", "arm_rot_r"),
    "shoulder_l": ("arm_flex_l", "arm_add_l", "arm_rot_l"),
}

#: Limb chains considered by :func:`resolve_limbs`: (proximal long-axis
#: rotation DoF, hinge DoF, all chain DoFs re-optimised, bodies whose markers
#: are compared).
LIMB_CHAINS: Dict[str, dict] = {
    f"arm_{s}": {
        "axial": f"arm_rot_{s}", "hinge": f"elbow_flex_{s}",
        "dofs": tuple(f"{d}_{s}" for d in ("arm_flex", "arm_add", "arm_rot", "elbow_flex",
                                             "pro_sup", "wrist_flex", "wrist_dev")),
        "bodies": tuple(f"{b}_{s}" for b in ("humerus", "ulna", "radius", "hand")),
    } for s in ("r", "l")
}
LIMB_CHAINS.update({
    f"leg_{s}": {
        "axial": f"hip_rotation_{s}", "hinge": f"knee_angle_{s}",
        "dofs": tuple(f"{d}_{s}" for d in ("hip_flexion", "hip_adduction", "hip_rotation",
                                             "knee_angle", "ankle_angle", "subtalar_angle",
                                             "mtp_angle")),
        "bodies": tuple(f"{b}_{s}" for b in ("femur", "tibia", "talus", "calcn", "toes")),
    } for s in ("r", "l")
})

#: MOT columns that are translations (never wrapped or scored).
TRANSLATIONS = ("pelvis_tx", "pelvis_ty", "pelvis_tz")
#: OpenSim export negates the MuJoCo knee angles (baseline/metrab2opensim.py).
SIGN_FLIPPED_IN_MOT = ("knee_angle_r", "knee_angle_l")

HUMANOID_XML = (Path(__file__).resolve().parents[2] / "baseline" / "monocular_demos"
                / "biomechanics_mjx" / "data" / "humanoid" / "humanoid_torque.xml")

#: MeTRAbs ``bml_movi_87`` keypoint order (:data:`hmb.scoring.bone_length.JOINT_NAMES`);
#: keypoint k corresponds to the model site of the same name.
MOVI87: Tuple[str, ...] = JOINT_NAMES


# ---------------------------------------------------------------------------
# Angle utilities and limits
# ---------------------------------------------------------------------------

def wrap_to_pi(x):
    """Wrap angles (radians) to (-pi, pi]."""
    x = np.asarray(x, dtype=float)
    y = np.mod(x + np.pi, 2.0 * np.pi) - np.pi
    return np.where(y == -np.pi, np.pi, y)


def dof_key(name: str) -> str:
    """Limit-table key of a coordinate name (drops the _r/_l side suffix)."""
    return name[:-2] if name.endswith(("_r", "_l")) else name


def limits_for(dofs: Sequence[str], limits_deg: Mapping[str, Tuple[float, float]]):
    """(lo, hi) arrays in radians for ``dofs``."""
    lo = np.radians([limits_deg[dof_key(d)][0] for d in dofs])
    hi = np.radians([limits_deg[dof_key(d)][1] for d in dofs])
    if np.any(hi <= lo):
        raise ValueError("each limit needs hi > lo")
    return lo, hi


def resolve_limits(limits=None) -> Dict[str, Tuple[float, float]]:
    """The limit table to use: :data:`LIMITS_DEG` by default, else a copy of the given mapping."""
    return LIMITS_DEG if limits is None else dict(limits)


def dof_excess(angles, lo, hi, tolerance_deg: float = TOLERANCE_DEG):
    """Paper excess beyond ``[lo - tol, hi + tol]`` in radians (>= 0)."""
    a = np.asarray(angles, dtype=float)
    tol = math.radians(tolerance_deg)
    return np.maximum(0.0, np.maximum(a - (hi + tol), (lo - tol) - a))


def dof_severity(excess, lo, hi, saturation: float = SATURATION_FRACTION):
    """Paper per-DoF severity ``min(excess / (saturation (hi - lo)), 1)``."""
    return np.clip(np.asarray(excess, dtype=float) / (saturation * (hi - lo)), 0.0, 1.0)


def reduce_frames(severity):
    """Per-frame m_t of (T, D) per-DoF severities: the worst DoF."""
    sev = np.asarray(severity, dtype=float)
    if sev.ndim != 2:
        raise ValueError("severity must be (frames, dofs)")
    if sev.shape[1] == 0:
        return np.zeros(sev.shape[0])
    return sev.max(axis=1)


# ---------------------------------------------------------------------------
# Exact canonicalisation
# ---------------------------------------------------------------------------

def euler_alternative(a, b, c):
    """The second intrinsic Z-X-Y Euler solution of the same rotation."""
    return wrap_to_pi(np.asarray(a) + np.pi), wrap_to_pi(np.pi - np.asarray(b)), wrap_to_pi(np.asarray(c) + np.pi)


def wrap_near(x, centre):
    """2 pi-equivalent of ``x`` closest to ``centre`` (in (centre - pi, centre + pi])."""
    return np.asarray(centre, float) + wrap_to_pi(np.asarray(x, float) - centre)


def _wrap_column(values, name, table):
    """Wrap one MOT column: scored DoFs near the centre of their limits, the others to (-pi, pi]."""
    if name in SCORED_DOFS and dof_key(name) in table:
        lo, hi = table[dof_key(name)]
        return wrap_near(values, math.radians(0.5 * (lo + hi)))
    return wrap_to_pi(values)


@dataclass
class Canonicalization:
    """Result of :func:`canonicalize`: the angles and, per column / ball joint, how many frames changed."""

    angles: np.ndarray                      # (T, N) canonical angles, same columns as input
    wrapped: Dict[str, int] = field(default_factory=dict)       # frames whose value changed by wrapping
    euler_switched: Dict[str, int] = field(default_factory=dict)  # frames using the alternative solution


def canonicalize(angles, names: Sequence[str], limits=None,
                 tolerance_deg: float = TOLERANCE_DEG, wrap: bool = True,
                 euler: bool = True) -> Canonicalization:
    """Map every frame to the exactly equivalent pose representation with least violation.

    ``angles`` is (T, N) in radians with MOT column ``names`` (OpenSim
    convention).  Translations are left untouched.  The pose (forward
    kinematics of every body and site) is unchanged by construction.
    """
    q = np.array(angles, dtype=float, copy=True)
    if q.ndim != 2 or q.shape[1] != len(names):
        raise ValueError("angles must be (frames, len(names))")
    index = {n: i for i, n in enumerate(names)}
    table = resolve_limits(limits)
    out = Canonicalization(angles=q)
    if wrap:
        for n, i in index.items():
            if n in TRANSLATIONS:
                continue
            w = _wrap_column(q[:, i], n, table)
            changed = int(np.sum(np.abs(w - q[:, i]) > 1e-9))
            if changed:
                out.wrapped[n] = changed
            q[:, i] = w
    if euler:
        for group, dofs in BALL_JOINTS.items():
            if not all(d in index for d in dofs):
                continue
            cols = [index[d] for d in dofs]
            orig = q[:, cols]
            alt = np.stack(euler_alternative(orig[:, 0], orig[:, 1], orig[:, 2]), axis=1)
            alt = np.stack([_wrap_column(alt[:, k], d, table) for k, d in enumerate(dofs)], axis=1)
            scored = [k for k, d in enumerate(dofs) if d in SCORED_DOFS and dof_key(d) in table]
            if not scored:
                continue
            lo, hi = limits_for([dofs[k] for k in scored], table)
            sev_o = dof_severity(dof_excess(orig[:, scored], lo, hi, tolerance_deg), lo, hi)
            sev_a = dof_severity(dof_excess(alt[:, scored], lo, hi, tolerance_deg), lo, hi)
            # least violation: worst DoF first, then the sum (strict improvement only)
            key_o = sev_o.max(axis=1) + 1e-6 * sev_o.sum(axis=1)
            key_a = sev_a.max(axis=1) + 1e-6 * sev_a.sum(axis=1)
            use_alt = key_a < key_o - 1e-12
            if use_alt.any():
                q[np.ix_(use_alt, cols)] = alt[use_alt]
                out.euler_switched[group] = int(use_alt.sum())
    out.angles = q
    return out


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

@dataclass
class JointRangeResult:
    """Result of :func:`score_angles` (aggregate terms, per-frame arrays and per-DoF statistics)."""

    score: float
    D: float
    r: float
    s: float
    p: float
    num_frames: int
    num_valid: int
    num_flagged: int
    coverage: float
    frame_severity: np.ndarray
    frame_flags: np.ndarray
    per_dof_violation_rate: Dict[str, float]
    per_dof_max_excess_deg: Dict[str, float]
    settings: dict
    canonicalization: dict

    def as_dict(self, include_frames: bool = False) -> dict:
        """JSON-friendly dict; ``include_frames`` adds the per-frame severities and flags."""
        out = {k: getattr(self, k) for k in ("score", "D", "r", "s", "p", "num_frames", "num_valid",
                                             "num_flagged", "coverage", "per_dof_violation_rate",
                                             "per_dof_max_excess_deg", "settings", "canonicalization")}
        if include_frames:
            out["frame_severity"] = self.frame_severity.tolist()
            out["frame_flags"] = self.frame_flags.astype(int).tolist()
        return out


def score_angles(angles, names: Sequence[str], *, valid=None, limits=None,
                 tolerance_deg: float = TOLERANCE_DEG, threshold: float = FRAME_THRESHOLD,
                 canonical: bool = True, dofs: Sequence[str] = SCORED_DOFS, observed=None) -> JointRangeResult:
    """Joint-range score of a (T, N) OpenSim coordinate array (radians).

    ``valid`` (T,) marks observed frames (e.g. a person was detected); invalid frames
    are excluded from r/s/p.  ``observed`` (T, len(SCORED_DOFS)) bool optionally marks,
    per frame, the DoFs whose defining keypoints are in view (:func:`observed_dofs`); an
    unobserved DoF contributes no excess on that frame.  ``limits`` replaces
    :data:`LIMITS_DEG` (e.g. :data:`PAPER_LIMITS_DEG`).  ``canonical=False`` scores the
    raw angles (no wrapping, no Euler switching).
    """
    q = np.asarray(angles, dtype=float)
    if q.ndim != 2 or q.shape[1] != len(names):
        raise ValueError("angles must be (frames, len(names))")
    if q.shape[0] == 0:
        raise ValueError("no frames")
    if not np.isfinite(q).all():
        raise ValueError("angles must be finite")
    table = resolve_limits(limits)
    canon = {"wrapped": {}, "euler_switched": {}}
    if canonical:
        c = canonicalize(q, names, table, tolerance_deg)
        q = c.angles
        canon = {"wrapped": c.wrapped, "euler_switched": c.euler_switched}
    index = {n: i for i, n in enumerate(names)}
    used = [d for d in dofs if d in index and dof_key(d) in table]
    if not used:
        raise ValueError("no scored DoFs present")
    lo, hi = limits_for(used, table)
    sub = q[:, [index[d] for d in used]]
    excess = dof_excess(sub, lo, hi, tolerance_deg)
    if observed is not None:
        obs = np.asarray(observed, bool)
        if obs.shape != (q.shape[0], len(SCORED_DOFS)):
            raise ValueError("observed must be (frames, len(SCORED_DOFS))")
        excess = np.where(obs[:, [SCORED_DOFS.index(d) for d in used]], excess, 0.0)
    sev = dof_severity(excess, lo, hi)
    m = reduce_frames(sev)
    flags = m > threshold
    agg = aggregate(m, flags, valid, weights=WEIGHTS)
    v = np.ones(len(m), bool) if valid is None else np.asarray(valid, bool)
    per_rate = {d: float(np.mean(excess[v, j] > 0)) if v.any() else float("nan") for j, d in enumerate(used)}
    per_max = {d: float(np.degrees(excess[v, j].max())) if v.any() else float("nan") for j, d in enumerate(used)}
    settings = {"limits": "default" if limits is None else "custom", "tolerance_deg": tolerance_deg,
                "threshold": threshold, "canonical": canonical, "num_dofs": len(used),
                "observation_mask": observed is not None}
    return JointRangeResult(score=agg.score, D=agg.D, r=agg.r, s=agg.s, p=agg.p, num_frames=agg.num_frames,
                            num_valid=agg.num_valid, num_flagged=agg.num_flagged, coverage=agg.coverage,
                            frame_severity=m, frame_flags=flags, per_dof_violation_rate=per_rate,
                            per_dof_max_excess_deg=per_max, settings=settings, canonicalization=canon)


# ---------------------------------------------------------------------------
# Loading artifacts
# ---------------------------------------------------------------------------

def load_mot(path) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Read an OpenSim .mot/.sto coordinate table -> (names, angles radians (T, N), time)."""
    lines = Path(path).read_text().splitlines()
    header = None
    in_degrees = False
    for i, line in enumerate(lines):
        tokens = line.split()
        if not tokens:
            continue
        if line.strip().lower().replace(" ", "") == "indegrees=yes":
            in_degrees = True
        if tokens[0] == "time":
            header = i
            break
    if header is None:
        raise ValueError(f"no 'time' header in {path}")
    cols = lines[header].split()
    rows = [ln.split() for ln in lines[header + 1:] if ln.strip()]
    data = np.asarray([[float(x) for x in r] for r in rows if len(r) == len(cols)], dtype=float)
    if data.size == 0:
        raise ValueError(f"no data rows in {path}")
    names = cols[1:]
    q = data[:, 1:]
    if in_degrees:
        rot = [n not in TRANSLATIONS for n in names]
        q[:, rot] = np.radians(q[:, rot])
    return names, q, data[:, 0]


def detection_valid(keypoints_path, num_frames: int) -> Optional[np.ndarray]:
    """Per-frame validity from MeTRAbs confidences (any keypoint > 0)."""
    try:
        with np.load(keypoints_path, allow_pickle=False) as d:
            confs = np.asarray(d["confs"], float)
    except (OSError, KeyError, ValueError):
        return None
    if confs.ndim != 2 or confs.shape[0] != num_frames:
        return None
    return confs.max(axis=1) > 0


# ---------------------------------------------------------------------------
# Forward kinematics (template model) and observation-verified limb flips
# ---------------------------------------------------------------------------

class _Model:
    """Native MuJoCo model of the fitting skeleton, optionally with fitted scale/site offsets."""

    def __init__(self, xml=HUMANOID_XML, body_scale=None, site_offsets=None):
        import mujoco

        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(xml))
        m = self.model
        if body_scale is not None:
            sc = np.asarray(body_scale, float).reshape(-1)
            if sc.shape != (m.nbody,):
                raise ValueError("body_scale must have one entry per body")
            m.body_pos[:] = m.body_pos * sc[m.body_parentid, None]
            m.site_pos[:] = m.site_pos * sc[m.site_bodyid, None]
        if site_offsets is not None:
            m.site_pos[:] = m.site_pos + np.asarray(site_offsets, float).reshape(m.nsite, 3)
            m.site_sameframe[:] = 0
        self.data = mujoco.MjData(m)
        self.joint_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]
        self.qadr = {n: int(m.jnt_qposadr[i]) for i, n in enumerate(self.joint_names)}
        self.site_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, i) for i in range(m.nsite)]
        self.body_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)]
        self.site_body = [self.body_names[b] for b in m.site_bodyid]

    def qpos_from_coordinates(self, row, names, root=None):
        """MuJoCo ``qpos`` of one MOT row (root pose from ``root``, pelvis coordinates ignored)."""
        q = np.zeros(self.model.nq) if root is None else np.array(root, float, copy=True)
        for n, v in zip(names, row):
            if n in self.qadr and not n.startswith("pelvis"):
                q[self.qadr[n]] = -v if n in SIGN_FLIPPED_IN_MOT else v
        return q

    def forward(self, qpos):
        """Forward kinematics -> (body positions, site positions, body rotation matrices)."""
        self.data.qpos[:] = qpos
        self.mj.mj_kinematics(self.model, self.data)
        return self.data.xpos.copy(), self.data.site_xpos.copy(), self.data.xmat.copy()


@dataclass
class FlipReport:
    """Result of :func:`resolve_limbs`: resolved angles and per-chain accepted/candidate frame counts."""

    angles: np.ndarray
    accepted: Dict[str, int]
    candidates: Dict[str, int]


class _Chain:
    """One limb chain of :data:`LIMB_CHAINS` on a model (index maps, limits, MOT <-> qpos)."""

    def __init__(self, model, names, chain_name, table, tol):
        ch = LIMB_CHAINS[chain_name]
        self.name = chain_name
        self.dofs = ch["dofs"]
        self.index = {n: i for i, n in enumerate(names)}
        self.cols = [self.index[d] for d in self.dofs]
        self.adr = [model.qadr[d] for d in self.dofs]
        self.ax = self.dofs.index(ch["axial"])
        self.hg = self.dofs.index(ch["hinge"])
        self.sign = np.array([-1.0 if d in SIGN_FLIPPED_IN_MOT else 1.0 for d in self.dofs])
        self.scored = [d for d in self.dofs if d in SCORED_DOFS]
        self.scored_cols = [self.index[d] for d in self.scored]
        self.lo, self.hi = limits_for(self.scored, table)
        site_index = {n: i for i, n in enumerate(model.site_names)}
        self.order = np.array([site_index[n] for n in MOVI87])
        self.kp_idx = np.array([k for k, _ in enumerate(MOVI87) if model.site_body[self.order[k]] in ch["bodies"]])
        self.table, self.tol = table, tol

    def sev(self, row_canon):
        """(worst, summed) severity of the chain's scored DoFs in a canonical MOT row."""
        s = dof_severity(dof_excess(row_canon[self.scored_cols], self.lo, self.hi, self.tol), self.lo, self.hi)
        return float(s.max()), float(s.sum())

    def mot_to_x(self, row):
        """Chain coordinates of a MOT row in MuJoCo sign convention."""
        return np.asarray(row, float)[self.cols] * self.sign

    def x_to_row(self, row, x, names):
        """``row`` with the chain coordinates replaced by ``x`` (MuJoCo signs), canonicalised."""
        out = np.array(row, float, copy=True)
        out[self.cols] = wrap_to_pi(np.asarray(x, float)) * self.sign
        return canonicalize(out[None], names, self.table, self.tol).angles[0]


def _rms(f) -> float:
    """Root-mean-square 3-D distance of flattened (x, y, z) residuals."""
    return float(np.sqrt(np.mean(np.sum(np.reshape(f, (-1, 3)) ** 2, 1))))


def _start_points(ch: _Chain, x_fit: np.ndarray, clipped_row: np.ndarray):
    """``(label, x0)`` start configurations of a violating limb chain, besides the fit itself.

    ``x_fit`` are the fitted chain coordinates (MuJoCo signs), ``clipped_row`` the MOT row with
    the scored DoFs clipped into their limits.  Index ``ch.hg + 1`` is the first DoF after the
    hinge (forearm pronation / ankle), the "distal" DoF.
    """
    # half a turn about the long axis of the humerus / femur with the hinge bent the other way
    # puts the elbow / knee and the wrist / ankle in the same place
    flip = x_fit.copy()
    flip[ch.ax] += np.pi
    flip[ch.hg] = -flip[ch.hg]
    flip_distal_pi = flip.copy()
    flip_distal_pi[ch.hg + 1] += np.pi
    flip_distal_reset = flip.copy()
    flip_distal_reset[ch.hg + 1:] = 0.0
    distal_reset = x_fit.copy()
    distal_reset[ch.hg + 1:] = 0.0
    return (("flip", flip), ("flip_distal_pi", flip_distal_pi), ("flip_distal_reset", flip_distal_reset),
            ("distal_reset", distal_reset), ("clipped", ch.mot_to_x(clipped_row)))


def _in_limit_bounds(ch: _Chain, tolerance_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    """Box bounds (MuJoCo signs) that keep the chain's scored DoFs within limits +- tolerance."""
    tol = math.radians(tolerance_deg)
    lb = np.full(len(ch.dofs), -np.inf)
    ub = np.full(len(ch.dofs), np.inf)
    for j, d in enumerate(ch.scored):
        k = ch.dofs.index(d)
        lo_d, hi_d = ch.lo[j] - tol, ch.hi[j] + tol
        lb[k], ub[k] = (lo_d, hi_d) if ch.sign[k] > 0 else (-hi_d, -lo_d)
    return lb, ub


def resolve_limbs_arrays(angles, names: Sequence[str], qpos_fit, body_scale, site_offsets, keypoints_m, valid, *,
                         margin_mm: float = FLIP_RMS_MARGIN_MM, limits=None, tolerance_deg: float = TOLERANCE_DEG,
                         chains: Sequence[str] = tuple(LIMB_CHAINS)) -> Tuple[np.ndarray, Dict[str, dict]]:
    """Least-violating limb configuration among those that explain the limb's keypoints.

    ``angles`` (T, N) canonical MOT coordinates, ``qpos_fit`` (T, nq) fitted qpos (root),
    ``keypoints_m`` (T, 87, 3) MeTRAbs keypoints in metres, ``valid`` (T,) frames to examine.
    For every frame and limb chain whose scored DoFs violate the limits, the chain is re-fitted
    (Levenberg-Marquardt, analytic MuJoCo Jacobian) to the chain's markers from several starts
    (per-frame refined fit; long-axis flip with the hinge bent the other way, with the distal
    DoF turned by pi or reset; distal reset without flip; the fit clipped into the limits;
    the best fit with the scored DoFs constrained to limits +- tolerance; the neighbouring
    frame's resolved configuration, in a forward then a backward pass).
    Among the candidates whose marker RMS is within ``margin_mm`` of the (refined) fit, the least
    violating one replaces the fit if it violates less.  The test does not depend on the Euler
    representation.  Returns the resolved angles and per-chain statistics.
    """
    import mujoco
    from scipy.optimize import least_squares

    names = list(names)
    table = resolve_limits(limits)
    q0 = np.array(angles, float, copy=True)
    q = q0.copy()
    T = q0.shape[0]
    K = np.asarray(keypoints_m, float)
    vmask = np.ones(T, bool) if valid is None else np.asarray(valid, bool)
    model = _Model(HUMANOID_XML, body_scale, site_offsets)
    mm, dd = model.model, model.data
    jacp = np.zeros((3, mm.nv))
    stats: Dict[str, dict] = {}
    for cname in chains:
        if not all(d in names for d in LIMB_CHAINS[cname]["dofs"]):
            continue
        ch = _Chain(model, names, cname, table, tolerance_deg)
        sites = ch.order[ch.kp_idx]
        dofadr = [int(mm.jnt_dofadr[model.joint_names.index(d)]) for d in ch.dofs]
        frames: Dict[int, dict] = {}

        def problem(t, ch=ch, sites=sites, dofadr=dofadr, frames=frames):
            """Candidate configurations of chain ``ch`` in frame ``t`` (computed once per frame)."""
            if t in frames:
                return frames[t]
            base = model.qpos_from_coordinates(q0[t], names, root=qpos_fit[t])
            _, s0, _ = model.forward(base)
            target = K[t, ch.kp_idx] - (K[t].mean(0) - s0[ch.order].mean(0))  # fitter centres both sets

            def fun(x):  # marker residuals of the chain for chain coordinates x
                qq = base.copy()
                qq[ch.adr] = x
                dd.qpos[:] = qq
                mujoco.mj_kinematics(mm, dd)
                return (dd.site_xpos[sites] - target).ravel()

            def jac(x):  # analytic Jacobian of fun from MuJoCo's site Jacobians
                qq = base.copy()
                qq[ch.adr] = x
                dd.qpos[:] = qq
                mujoco.mj_kinematics(mm, dd)
                mujoco.mj_comPos(mm, dd)
                J = np.empty((3 * len(sites), len(dofadr)))
                for i, sid in enumerate(sites):
                    mujoco.mj_jacSite(mm, dd, jacp, None, int(sid))
                    J[3 * i:3 * i + 3] = jacp[:, dofadr]
                return J

            def run(label, x0):  # unconstrained fit from x0 -> candidate
                sol = least_squares(fun, x0, jac=jac, method="lm", max_nfev=200)
                xs = wrap_to_pi(sol.x)
                row = ch.x_to_row(q0[t], xs, names)
                return {"label": label, "x": xs, "rms": _rms(sol.fun), "key": ch.sev(row), "row": row}

            x_fit = base[ch.adr].copy()
            rms_fit = _rms(fun(x_fit))
            # the order matters: on equal (violation, RMS) the earlier candidate wins
            cands = [run("refined", x_fit)]
            clipped = q0[t].copy()
            clipped[ch.scored_cols] = np.clip(q0[t][ch.scored_cols], ch.lo, ch.hi)
            for label, x0 in _start_points(ch, x_fit, clipped):
                cands.append(run(label, x0))
            # best fit that stays within the limits (+- tolerance)
            lb, ub = _in_limit_bounds(ch, tolerance_deg)
            x0 = np.clip(ch.mot_to_x(clipped), lb + 1e-6, ub - 1e-6)
            sol = least_squares(fun, x0, jac=jac, method="trf", bounds=(lb, ub), max_nfev=200)
            row = ch.x_to_row(q0[t], sol.x, names)
            cands.append({"label": "bounded", "x": wrap_to_pi(sol.x), "rms": _rms(sol.fun),
                          "key": ch.sev(row), "row": row})
            frames[t] = {"rms_ref": min(rms_fit, cands[0]["rms"]), "cands": cands, "run": run}
            return frames[t]

        st = {"violating": 0, "accepted": 0, "accepted_by": {}}
        seen = set()
        # a forward and a backward pass, so a resolved frame can seed its neighbours on both sides
        passes = [range(T), range(T - 1, -1, -1)]
        for p_i, order in enumerate(passes):
            for t in order:
                if not vmask[t]:
                    continue
                sev_now = ch.sev(q[t])
                if sev_now[0] <= 0 or ch.sev(q0[t])[0] <= 0:
                    continue
                pr = problem(t)
                if t not in seen:
                    seen.add(t)
                    st["violating"] += 1
                cands = list(pr["cands"])
                # also start from the resolved neighbour visited just before in this pass, unless
                # that start is within 10 deg of an existing candidate
                nb = t - 1 if p_i == 0 else t + 1
                if 0 <= nb < T and vmask[nb]:
                    x_nb = ch.mot_to_x(q[nb])
                    dup = min(float(np.abs(wrap_to_pi(x_nb - c["x"])).max()) for c in cands)
                    if dup > math.radians(10.0):
                        cands.append(pr["run"]("temporal", x_nb))
                # least violating candidate that explains the keypoints about as well as the fit
                best = None
                for c in cands:
                    if c["rms"] > pr["rms_ref"] + margin_mm / 1000.0:
                        continue
                    key = (round(c["key"][0], 9), round(c["key"][1], 9), c["rms"])
                    if best is None or key < best[0]:
                        best = (key, c)
                if best is not None and best[0][0] < sev_now[0] - 1e-9:
                    c = best[1]
                    row = q[t].copy()
                    row[ch.cols] = c["row"][ch.cols]
                    q[t] = canonicalize(row[None], names, table, tolerance_deg).angles[0]
                    st["accepted"] += 1
                    st["accepted_by"][c["label"]] = st["accepted_by"].get(c["label"], 0) + 1
        stats[cname] = st
    return q, stats


def resolve_limbs(angles, names: Sequence[str], fitted_model_path, keypoints_path, *,
                  limits=None, tolerance_deg: float = TOLERANCE_DEG, margin_mm: float = FLIP_RMS_MARGIN_MM,
                  chains: Iterable[str] = tuple(LIMB_CHAINS), valid=None) -> FlipReport:
    """Limb resolution of a session: :func:`resolve_limbs_arrays` verified with the template
    marker placement (fitted per-body scale kept), then with the fitted marker offsets."""
    with np.load(fitted_model_path, allow_pickle=False) as fit:
        qpos_fit = np.asarray(fit["qpos"], float)
        body_scale = None
        if "body_scale_factors" in fit.files:
            bsf = np.asarray(fit["body_scale_factors"], float)
            if bsf.ndim == 1 or (bsf.ndim == 2 and bsf.shape[1] == 1):
                body_scale = bsf.reshape(-1)          # rigid fit: one scale per body
        offsets = np.asarray(fit["site_offsets"], float) if "site_offsets" in fit.files else None
    with np.load(keypoints_path, allow_pickle=False) as kp:
        K = np.asarray(kp["keypoints3d"], float) / 1000.0
    q = np.asarray(angles, float)
    if qpos_fit.shape[0] != q.shape[0] or K.shape[0] != q.shape[0]:
        raise ValueError("fitted model, keypoints and angles must have the same number of frames")
    chains = tuple(chains)
    kw = dict(margin_mm=margin_mm, limits=limits, tolerance_deg=tolerance_deg, chains=chains)
    q1, st1 = resolve_limbs_arrays(q, names, qpos_fit, body_scale, None, K, valid, **kw)
    q2, st2 = resolve_limbs_arrays(q1, names, qpos_fit, body_scale, offsets, K, valid, **kw)
    acc = {c: st1.get(c, {}).get("accepted", 0) + st2.get(c, {}).get("accepted", 0) for c in chains}
    cand = {c: st1.get(c, {}).get("violating", 0) for c in chains}
    return FlipReport(angles=q2, accepted=acc, candidates=cand)


#: MeTRAbs name prefix of each model side.
_SIDE = {"r": "R", "l": "L"}


def _observing_keypoints(dof: str) -> Tuple[str, ...]:
    """Keypoints (MeTRAbs names) that must project inside the image for ``dof`` to be observed."""
    if dof.startswith("lumbar"):
        return ("LHip", "RHip", "LShoulder", "RShoulder", "Neck")
    side = _SIDE[dof[-1]]
    if dof.startswith(("hip_", "knee_")):
        return (f"{side}Hip", f"{side}Knee", f"{side}Ankle")
    if dof.startswith("ankle_"):
        return (f"{side}Knee", f"{side}Ankle", f"{side}Heel", f"{side}BigToe")
    if dof.startswith(("arm_", "elbow_")):
        return (f"{side}Shoulder", f"{side}Elbow", f"{side}Wrist")
    raise KeyError(dof)


def observed_dofs(keypoints_path, num_frames: Optional[int] = None) -> Optional[np.ndarray]:
    """(T, len(SCORED_DOFS)) bool: the DoF's defining keypoints project inside the image.

    ``None`` when the keypoints file has no 2-D keypoints or image size (no mask is applied).
    """
    try:
        with np.load(keypoints_path, allow_pickle=False) as kp:
            P2 = np.asarray(kp["keypoints2d"], float)
            W, H = [float(x) for x in np.asarray(kp["image_size"]).reshape(-1)[:2]]
    except (OSError, KeyError, ValueError):
        return None
    if P2.ndim != 3 or P2.shape[1] != len(MOVI87) or (num_frames is not None and P2.shape[0] != num_frames):
        return None
    inside = (P2[..., 0] >= 0) & (P2[..., 0] < W) & (P2[..., 1] >= 0) & (P2[..., 1] < H)
    ix = {n: i for i, n in enumerate(MOVI87)}
    return np.stack([inside[:, [ix[k] for k in _observing_keypoints(d)]].all(1) for d in SCORED_DOFS], 1)


# ---------------------------------------------------------------------------
# Session-level entry points
# ---------------------------------------------------------------------------

def _session_paths(session_dir, stem=None) -> dict:
    """Input files of the metric (stem from the first ``*_opensim_skeleton.mot`` when not given)."""
    s = Path(session_dir)
    if stem is None:
        mots = sorted((s / "baseline").glob("*_opensim_skeleton.mot"))
        if not mots:
            raise FileNotFoundError(f"no *_opensim_skeleton.mot under {s / 'baseline'}")
        stem = mots[0].name[: -len("_opensim_skeleton.mot")]
    return {
        "stem": stem,
        "mot": s / "baseline" / f"{stem}_opensim_skeleton.mot",
        "fitted_model": s / "baseline" / f"{stem}_fitted_model.npz",
        "keypoints": s / "baseline" / f"{stem}_keypoints.npz",
    }


def score_session(session_dir, stem: Optional[str] = None) -> dict:
    """Score a session (``<S>/baseline/<stem>_opensim_skeleton.mot`` and its keypoints / fitted model).

    Uses the detection mask, the observation mask and the limb resolution when the
    keypoints and the fitted model are present; without them it scores the canonical
    angles (the returned dict records what was used).
    """
    paths = _session_paths(session_dir, stem)
    names, q, time = load_mot(paths["mot"])
    has_keypoints = paths["keypoints"].is_file()
    valid = detection_valid(paths["keypoints"], len(q)) if has_keypoints else None
    flip_info = None
    if has_keypoints and paths["fitted_model"].is_file():
        c = canonicalize(q, names, None, TOLERANCE_DEG)
        try:
            rep = resolve_limbs(c.angles, names, paths["fitted_model"], paths["keypoints"],
                                limits=None, tolerance_deg=TOLERANCE_DEG, valid=valid)
        except (ImportError, KeyError, ValueError, OSError) as exc:  # keep scoring without it
            flip_info = {"error": f"{type(exc).__name__}: {exc}"}
        else:
            q = rep.angles
            flip_info = {"accepted": rep.accepted, "candidates": rep.candidates}
    observed = observed_dofs(paths["keypoints"], len(q)) if has_keypoints else None
    result = score_angles(q, names, valid=valid, observed=observed)
    out = result.as_dict()
    out.update({"metric": "joint_range", "stem": paths["stem"],
                "mot_path": str(paths["mot"]), "detection_mask": valid is not None,
                "flip_resolution": flip_info,
                "duration_s": float(time[-1] - time[0]) if len(time) > 1 else 0.0})
    return out


def main(argv=None) -> int:
    """Command line: score a session or a bare .mot file and print (or write) the result JSON."""
    import argparse

    parser = argparse.ArgumentParser(description="HumanScore joint range of motion")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--session-dir", help="session directory with baseline/<stem>_opensim_skeleton.mot")
    source.add_argument("--mot", help="an OpenSim .mot file (scored without keypoint masks or limb resolution)")
    parser.add_argument("--output", help="write the result JSON here instead of printing it")
    args = parser.parse_args(argv)
    if args.session_dir:
        out = score_session(args.session_dir)
    else:
        names, q, _ = load_mot(args.mot)
        out = score_angles(q, names).as_dict()
    text = json.dumps(out, indent=2, default=float)
    if args.output:
        Path(args.output).write_text(text)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
