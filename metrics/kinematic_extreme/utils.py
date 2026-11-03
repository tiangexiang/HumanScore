# -*- coding: utf-8 -*-
"""Motion loading and velocities for metric (V) kinematic extremes (paper implementation).

* :func:`load_mot` reads an OpenSim ``.mot`` motion (radians; ``inDegrees=yes`` is converted);
* :func:`central_derivative` differentiates on the actual timestamps (central differences,
  one-sided at the ends, never across a timestamp gap); motion smoothness uses it too;
* :class:`ForwardKinematics` maps the OpenSim coordinates back to the MuJoCo skeleton
  (inverting ``baseline/metrab2opensim.py``) and returns segment centres of mass;
* :func:`preload_motion_data` bundles joint angular velocities and segment speeds for
  ``main.compute_kextreme_score``.

Segment speeds are cached in ``.body_kin_cache/`` next to this file, or in
``$KEXTREME_BODY_KIN_CACHE_DIR`` (HumanScore uses ``<S>/cache/kinematic_extreme``).  A cache
entry is keyed by the motion and model paths and is reused only while the motion file,
its sibling ``*_fitted_model.npz``, the skeleton assets, this file and the numerical
library versions are unchanged.
"""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd


# =======================
# MOT loading
# =======================

@dataclass
class MotData:
    """A loaded ``.mot`` motion: one row per frame, one column per skeleton coordinate."""

    time: np.ndarray           # (T,)
    joint_names: List[str]     # length J
    joint_angles: np.ndarray   # (T, J), preferably radians
    source_path: Optional[str] = None
    metadata: dict = field(default_factory=dict)


_BODY_KIN_CACHE_ENV_VAR = "KEXTREME_BODY_KIN_CACHE_DIR"
_BODY_KIN_CACHE_VERSION = 4  # cache file format; bump when the stored arrays change
_DEFAULT_BODY_KIN_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".body_kin_cache")


def _get_body_kin_cache_dir() -> str:
    base = os.environ.get(_BODY_KIN_CACHE_ENV_VAR)
    if base:
        return base
    return _DEFAULT_BODY_KIN_CACHE_DIR


def _body_kin_cache_path(mot_path: str, model_path: str) -> str:
    key_src = f"{os.path.abspath(mot_path)}::{os.path.abspath(model_path)}"
    cache_key = hashlib.md5(key_src.encode("utf-8")).hexdigest()
    return os.path.join(_get_body_kin_cache_dir(), f"{cache_key}.npz")


def _safe_stat(path: Optional[str]):
    if not path:
        return None
    try:
        return os.stat(path)
    except OSError:
        return None


def _fitted_model_path(mot_path: str) -> str:
    """``<stem>_fitted_model.npz`` next to ``<stem>_opensim_skeleton.mot``; '' for other names."""
    suffix = "_opensim_skeleton.mot"
    return mot_path[:-len(suffix)] + "_fitted_model.npz" if mot_path.endswith(suffix) else ""


def _serialize_optional_array(arr: Optional[np.ndarray]):
    if arr is None:
        return np.empty((0,), dtype=np.float64), 0
    return np.asarray(arr), 1


def _fk_runtime_versions() -> dict:
    versions = {"numpy": np.__version__}
    for package in ("mujoco", "scipy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _fk_dependency_identity(path: Path) -> dict:
    """Hash small assets; retain filesystem change identity for large meshes."""
    path = path.resolve()
    try:
        stat = path.stat()
    except OSError:
        return {"path": str(path), "missing": True}
    identity = {"path": str(path), "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns}
    if path.is_file() and stat.st_size <= 8 * 1024 * 1024:
        identity = {"path": str(path), "size": stat.st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return identity


def _fk_cache_signature(model_path: str, mot_path: Optional[str] = None) -> str:
    """Hash of everything cached segment speeds depend on.

    Covers this file, the numerical library versions, every model asset in the skeleton
    folder (a changed mesh or XML also invalidates the cache when the entry-point XML is
    unchanged) and, with ``mot_path``, the motion and its sibling fitted model.
    """
    model = Path(model_path).resolve()
    assets = {model}
    assets.update(path for path in model.parent.rglob("*")
                  if path.is_file() and path.suffix.lower() in {".xml", ".stl", ".obj", ".msh"})
    payload = {
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runtime": _fk_runtime_versions(),
        "model_assets": [_fk_dependency_identity(path) for path in sorted(assets)],
    }
    if mot_path:
        inputs = [Path(mot_path)]
        fitted = _fitted_model_path(mot_path)
        if fitted:
            inputs.append(Path(fitted))
        payload["motion_inputs"] = [_fk_dependency_identity(path) for path in inputs]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _load_body_kin_cache(cache_path: str, mot_path: str, model_path: str) -> Optional["BodyKinematicsData"]:
    """Cached segment kinematics, or ``None`` when absent or no longer valid."""
    mot_stat = _safe_stat(mot_path)
    model_stat = _safe_stat(model_path)
    if mot_stat is None or model_stat is None:
        return None
    if not os.path.exists(cache_path):
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as data:
            if int(data["cache_version"][0]) != _BODY_KIN_CACHE_VERSION:
                return None
            if str(data["fk_signature"].item()) != _fk_cache_signature(model_path, mot_path):
                return None
            if int(data["mot_size"][0]) != mot_stat.st_size:
                return None
            if int(data["mot_mtime_ns"][0]) != mot_stat.st_mtime_ns:
                return None
            if int(data["model_size"][0]) != model_stat.st_size:
                return None
            if int(data["model_mtime_ns"][0]) != model_stat.st_mtime_ns:
                return None
            fitted_stat = _safe_stat(_fitted_model_path(mot_path))
            if int(data["fitted_mtime_ns"][0]) != (fitted_stat.st_mtime_ns if fitted_stat else -1):
                return None

            body_names = data["body_names"].astype(str).tolist()
            positions = data["positions"] if bool(data["has_positions"][0]) else None
            linear_vel = data["linear_vel"] if bool(data["has_linear_vel"][0]) else None
            time_mid = data["time_mid"] if bool(data["has_time_mid"][0]) else None

            return BodyKinematicsData(
                body_names=body_names,
                positions=positions,
                linear_vel=linear_vel,
                time_mid=time_mid,
            )
    except Exception as e:
        print(f"[WARN] Failed to load body kinematics cache {cache_path}: {e}")
        return None


def _save_body_kin_cache(
    cache_path: str,
    mot_path: str,
    model_path: str,
    data: "BodyKinematicsData",
):
    """Store segment kinematics atomically (temporary file, then rename)."""
    mot_stat = _safe_stat(mot_path)
    model_stat = _safe_stat(model_path)
    if mot_stat is None or model_stat is None:
        return

    positions_arr, has_positions = _serialize_optional_array(data.positions)
    linear_vel_arr, has_linear_vel = _serialize_optional_array(data.linear_vel)
    time_mid_arr, has_time_mid = _serialize_optional_array(data.time_mid)

    payload = {
        "cache_version": np.array([_BODY_KIN_CACHE_VERSION], dtype=np.int16),
        "mot_mtime_ns": np.array([mot_stat.st_mtime_ns], dtype=np.int64),
        "mot_size": np.array([mot_stat.st_size], dtype=np.int64),
        "model_mtime_ns": np.array([model_stat.st_mtime_ns], dtype=np.int64),
        "model_size": np.array([model_stat.st_size], dtype=np.int64),
        "body_names": np.asarray(data.body_names, dtype=np.str_),
        "positions": positions_arr,
        "linear_vel": linear_vel_arr,
        "time_mid": time_mid_arr,
        "has_positions": np.array([has_positions], dtype=np.int8),
        "has_linear_vel": np.array([has_linear_vel], dtype=np.int8),
        "has_time_mid": np.array([has_time_mid], dtype=np.int8),
    }
    try:
        payload["fk_signature"] = np.array(_fk_cache_signature(model_path, mot_path))
    except OSError as exc:
        print(f"[WARN] Cannot identify FK dependencies for caching: {exc}")
        return
    fitted_stat = _safe_stat(_fitted_model_path(mot_path))
    payload["fitted_mtime_ns"] = np.array([fitted_stat.st_mtime_ns if fitted_stat else -1], dtype=np.int64)

    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    tmp_fd, tmp_path = tempfile.mkstemp(dir=cache_dir or ".", prefix="bk_", suffix=".npz")
    try:
        with os.fdopen(tmp_fd, "wb") as tmp_file:
            np.savez_compressed(tmp_file, **payload)
        os.replace(tmp_path, cache_path)
    except Exception as e:
        print(f"[WARN] Failed to save body kinematics cache {cache_path}: {e}")
        try:
            os.remove(tmp_path)
        except OSError:
            pass

def load_mot(mot_path: str) -> MotData:
    """
    Load an OpenSim-style MOT file by locating the 'time' header and parsing subsequent rows.
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
        raise ValueError(f"Could not find 'time' header line in {mot_path}")

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
    metadata = {}
    for line in lines[:header_idx]:
        if "=" in line:
            key, value = line.split("=", 1)
            metadata[key.strip().lower()] = value.strip()
    if metadata.get("indegrees", "no").lower() == "yes":
        rotational = [not name.endswith(("_tx", "_ty", "_tz")) for name in joint_names]
        joint_angles[:, rotational] = np.deg2rad(joint_angles[:, rotational])

    return MotData(
        time=time,
        joint_names=joint_names,
        joint_angles=joint_angles,
        source_path=os.path.abspath(mot_path),
        metadata=metadata,
    )


# =======================
# Joint angular velocity
# =======================

@dataclass
class JointVelocityData:
    """Joint angular velocities at the original frame times."""

    time_mid: np.ndarray           # (T,) frame times of the velocities
    joint_names: List[str]
    angular_vel: np.ndarray        # (T, J), rad/s


def central_derivative(
    values: np.ndarray, time: np.ndarray, order: int = 1, unwrap_angles: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """Differentiate independently within contiguous timestamp segments.

    Interior samples use central differences on the actual timestamps. Segment
    endpoints use one-sided differences (second order when available). No
    stencil crosses a gap larger than 1.5 times the median frame interval.
    """
    time = np.asarray(time, dtype=float)
    values = np.asarray(values, dtype=float)
    if time.ndim != 1 or values.ndim < 1 or len(values) != len(time):
        raise ValueError("Derivative values must align with one-dimensional timestamps")
    if order not in (1, 2):
        raise ValueError("Derivative order must be 1 or 2")
    if len(time) < order + 1 or not np.isfinite(time).all() or not np.isfinite(values).all():
        raise ValueError(f"Need at least {order + 1} finite samples for derivative order {order}")
    dt = np.diff(time)
    if np.any(dt <= 0):
        raise ValueError("Derivative timestamps must be strictly increasing")
    boundaries = np.r_[0, np.flatnonzero(dt > 1.5 * np.median(dt)) + 1, len(time)]
    kept_times, derivatives = [], []
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        if stop - start < order + 1:
            continue
        segment = values[start:stop]
        if unwrap_angles:
            segment = np.unwrap(segment, axis=0)
        for _ in range(order):
            segment = np.gradient(segment, time[start:stop], axis=0, edge_order=2 if stop - start >= 3 else 1)
        kept_times.append(time[start:stop])
        derivatives.append(segment)
    if not derivatives:
        raise ValueError("No contiguous timestamp segment is long enough for derivatives")
    return np.concatenate(kept_times), np.concatenate(derivatives, axis=0)


def compute_joint_angular_velocities(mot: MotData) -> JointVelocityData:
    """Joint angular velocities (rad/s) from the motion's angles (radians), angles unwrapped."""
    time_mid, angular_vel = central_derivative(mot.joint_angles, mot.time, unwrap_angles=True)

    return JointVelocityData(
        time_mid=time_mid,
        joint_names=mot.joint_names,
        angular_vel=angular_vel,
    )


# =======================
# Forward Kinematics
# =======================

@dataclass
class BodyKinematicsData:
    """Segment centres of mass and their speeds."""

    body_names: List[str]             # length B
    positions: Optional[np.ndarray]   # (T, B, 3) segment centers of mass, or None
    linear_vel: Optional[np.ndarray]  # (T, B) or None
    time_mid: Optional[np.ndarray]    # (T,) original frame times, or None


class ForwardKinematics:
    """Native MuJoCo FK at segment centers of mass, without JAX compilation."""

    def __init__(self, model_path: str):
        import mujoco
        self._mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.body_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
                           for i in range(1, self.model.nbody)]
        if np.any(self.model.jnt_type < int(mujoco.mjtJoint.mjJNT_SLIDE)):
            raise ValueError("MOT FK requires scalar slide/hinge joints in the model")
        self.joint_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                            for i in range(self.model.njnt)]
        self.qpos_addresses = self.model.jnt_qposadr.copy()

    def compute_body_positions(self, qpos_seq: np.ndarray) -> np.ndarray:
        """
        Args:
            qpos_seq: (T, nq) array
        Returns:
            positions: (T, B, 3) aligned with self.body_names
        """
        qpos_seq = np.asarray(qpos_seq, dtype=float)
        if qpos_seq.ndim != 2 or qpos_seq.shape[1] != self.model.nq or not np.isfinite(qpos_seq).all():
            raise ValueError(f"FK requires finite qpos with shape (T, {self.model.nq})")
        data = self._mujoco.MjData(self.model)
        positions = np.empty((len(qpos_seq), len(self.body_names), 3), dtype=float)
        for i, qpos in enumerate(qpos_seq):
            data.qpos[:] = qpos
            self._mujoco.mj_kinematics(self.model, data)
            positions[i] = data.xipos[1:]
        return positions

    def qpos_from_mot(self, mot: MotData) -> np.ndarray:
        """Invert the repository's OpenSim export and map columns by name."""
        from scipy.spatial.transform import Rotation

        name_to_col = {name: i for i, name in enumerate(mot.joint_names)}
        missing = set(self.joint_names) - set(name_to_col)
        if missing:
            raise ValueError(f"MOT is missing model joint columns: {sorted(missing)}")
        qpos = np.tile(self.model.qpos0, (len(mot.time), 1))
        for name, address in zip(self.joint_names, self.qpos_addresses):
            qpos[:, address] = mot.joint_angles[:, name_to_col[name]]
        address_by_name = dict(zip(self.joint_names, self.qpos_addresses))
        for name in ("knee_angle_r", "knee_angle_l"):
            qpos[:, address_by_name[name]] *= -1
        axis_transform = np.array([[-1., 0., 0.], [0., 0., 1.], [0., 1., 0.]])
        translations = [address_by_name[name] for name in ("pelvis_tx", "pelvis_ty", "pelvis_tz")]
        rotations = [address_by_name[name] for name in ("pelvis_tilt", "pelvis_list", "pelvis_rotation")]
        qpos[:, translations] = qpos[:, translations] @ axis_transform
        rot_mats = Rotation.from_euler("ZXY", qpos[:, rotations]).as_matrix()
        qpos[:, rotations] = Rotation.from_matrix(axis_transform @ rot_mats).as_euler("ZXY")

        # A .mot without "signed_pelvis_rotation=yes" stores |pelvis_rotation|, which loses the
        # sign. Take the exact pose from the matching fitted model instead of guessing it.
        if mot.metadata.get("signed_pelvis_rotation", "no").lower() != "yes":
            fitted_path = _fitted_model_path(mot.source_path or "")
            if not fitted_path or not os.path.isfile(fitted_path):
                raise ValueError("MOT has unsigned pelvis rotation; a matching fitted_model.npz is required for body FK")
            with np.load(fitted_path, allow_pickle=False) as fitted:
                raw_qpos = np.asarray(fitted["qpos"], dtype=float)
                raw_time = np.asarray(fitted["timestamps"], dtype=float)
            if raw_qpos.shape != qpos.shape or raw_time.shape != mot.time.shape or not np.allclose(raw_time, mot.time, rtol=0, atol=1e-5):
                raise ValueError("Sibling fitted model does not match MOT frames/timestamps")
            nonroot = [address for name, address in address_by_name.items() if not name.startswith("pelvis_")]
            if not np.allclose(raw_qpos[:, nonroot], qpos[:, nonroot], rtol=0, atol=1e-5):
                raise ValueError("Sibling fitted model does not match MOT joint angles")
            if not np.allclose(raw_qpos[:, translations], qpos[:, translations], rtol=0, atol=1e-5):
                raise ValueError("Sibling fitted model does not match MOT root translation")
            exported_rotations = Rotation.from_matrix(
                axis_transform @ Rotation.from_euler("ZXY", raw_qpos[:, rotations]).as_matrix()
            ).as_euler("ZXY")
            exported_rotations[:, 2] = np.abs(exported_rotations[:, 2])
            if not np.allclose(exported_rotations, mot.joint_angles[:, [name_to_col[name] for name in ("pelvis_tilt", "pelvis_list", "pelvis_rotation")]], rtol=0, atol=1e-5):
                raise ValueError("Sibling fitted model does not match MOT root rotation")
            qpos = raw_qpos
        return qpos


def compute_body_kinematics(
    mot: MotData,
    model_path: str,
) -> BodyKinematicsData:
    """Segment centres of mass (forward kinematics) and their speeds, using the cache.

    An explicitly requested model must be usable: errors propagate, so a joint-only
    score cannot silently stand in for the full metric.
    """
    source_path = getattr(mot, "source_path", None)
    cache_path = None
    if source_path and model_path:
        cache_path = _body_kin_cache_path(source_path, model_path)
        cached = _load_body_kin_cache(cache_path, source_path, model_path)
        if cached is not None:
            return cached

    fk = ForwardKinematics(model_path)
    qpos_seq = fk.qpos_from_mot(mot)

    positions = fk.compute_body_positions(qpos_seq)  # (T, B, 3)
    if positions.shape[0] != mot.time.shape[0]:
        raise ValueError("FK output T mismatch with MOT time length.")

    if positions.shape[0] < 2:
        return BodyKinematicsData(body_names=fk.body_names, positions=positions,
                                  linear_vel=None, time_mid=None)

    time_mid, linear_velocity = central_derivative(positions, mot.time)
    speed = np.linalg.norm(linear_velocity, axis=-1)

    result = BodyKinematicsData(
        body_names=fk.body_names,
        positions=positions,
        linear_vel=speed,
        time_mid=time_mid,
    )

    if cache_path and source_path:
        _save_body_kin_cache(cache_path, source_path, model_path, result)

    return result



# =======================
# Aggregate into preloaded structure
# =======================

@dataclass
class PreloadedMotionData:
    """Everything ``main.compute_kextreme_score`` needs for one motion."""

    time: np.ndarray
    joint_names: List[str]
    joint_angles: np.ndarray
    joint_vel_time: np.ndarray
    joint_angular_vel: np.ndarray
    body_names: List[str]
    body_linear_vel_time: Optional[np.ndarray]
    body_linear_vel: Optional[np.ndarray]


def preload_motion_data(
    mot_path: str,
    model_path: Optional[str] = None,
    *,
    mot: Optional[MotData] = None,
) -> PreloadedMotionData:
    """Load a motion (or use ``mot``) and compute its joint and segment velocities.

    ``model_path`` is the MuJoCo skeleton used for the segment speeds; with ``None`` or ''
    only joint angular velocities are computed.
    """
    if mot is None:
        mot = load_mot(mot_path)
    jv = compute_joint_angular_velocities(mot)

    body_names: List[str] = []
    body_v_time: Optional[np.ndarray] = None
    body_v: Optional[np.ndarray] = None

    if model_path:
        bk = compute_body_kinematics(mot, model_path)
        body_names = bk.body_names
        body_v_time = bk.time_mid
        body_v = bk.linear_vel

        # Align lengths with joint_angular_vel
        if body_v is not None and body_v_time is not None:
            T = min(jv.angular_vel.shape[0], body_v.shape[0])
            jv.angular_vel = jv.angular_vel[:T]
            jv.time_mid = jv.time_mid[:T]
            body_v = body_v[:T]
            body_v_time = body_v_time[:T]
        else:
            body_names = []
            body_v_time = None
            body_v = None

    return PreloadedMotionData(
        time=mot.time,
        joint_names=mot.joint_names,
        joint_angles=mot.joint_angles,
        joint_vel_time=jv.time_mid,
        joint_angular_vel=jv.angular_vel,
        body_names=body_names,
        body_linear_vel_time=body_v_time,
        body_linear_vel=body_v,
    )
