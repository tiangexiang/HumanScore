"""Body and site positions of a fitted skeleton with native MuJoCo (no MJX compilation).

After a fit, the joint angles and body scales come from the differentiable trajectory;
:func:`export_kinematic_coordinates` only places the scaled skeleton in each pose and
reads the body origins and keypoint sites.  Doing this in native MuJoCo avoids compiling
an MJX program for every export.  Used by ``baseline/main.py``,
``metrics/bone_length/main.py`` and ``hmb/fit_engine.py``.
"""
from __future__ import annotations

import numpy as np


def export_kinematic_coordinates(xml_path, qpos, body_scales, site_offsets, site_reorder=None):
    """World positions of every body and site of the scaled skeleton in each pose.

    Uses the fitter's conventions: a child body's offset is scaled by its parent's scale,
    and a site's local position by its own body's scale before ``site_offsets`` are added.
    Only kinematics are evaluated (no forces, contacts or simulation).

    Args:
        xml_path: skeleton model (``humanoid_torque.xml``).
        qpos: (T, nq) joint positions.
        body_scales: per-body scale factors, (nbody,), (nbody, 1) or (T, nbody[, 1]).
        site_offsets: (nsite, 3) learned site offsets.
        site_reorder: optional site order of the returned sites (the keypoint order).

    Returns:
        ``(bodies, sites)`` of shape (T, nbody, 3) and (T, nsites, 3).
    """
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    poses = np.asarray(qpos, dtype=float)
    if poses.ndim != 2 or poses.shape[1] != model.nq or not np.isfinite(poses).all():
        raise ValueError(f"qpos must contain finite poses with shape (T, {model.nq})")
    frames = len(poses)
    if frames == 0:
        raise ValueError("Cannot export an empty trajectory")
    scales = np.asarray(body_scales, dtype=float)
    if scales.shape == (model.nbody, 1):
        scales = np.broadcast_to(scales[:, 0], (frames, model.nbody))
    elif scales.shape == (model.nbody,):
        scales = np.broadcast_to(scales, (frames, model.nbody))
    elif scales.shape == (frames, model.nbody, 1):
        scales = scales[..., 0]
    if scales.shape != (frames, model.nbody) or not np.isfinite(scales).all():
        raise ValueError(f"Body scales must be finite and align with {frames} frames and {model.nbody} bodies")
    offsets = np.asarray(site_offsets, dtype=float)
    if offsets.shape != (model.nsite, 3) or not np.isfinite(offsets).all():
        raise ValueError(f"Site offsets must be finite and have shape ({model.nsite}, 3)")
    order = np.arange(model.nsite) if site_reorder is None else np.asarray(site_reorder)
    if order.ndim != 1 or not np.issubdtype(order.dtype, np.integer) or np.any((order < 0) | (order >= model.nsite)):
        raise ValueError("Site reordering must contain valid integer site indices")
    base_body_pos, base_site_pos = model.body_pos.copy(), model.site_pos.copy()
    # MuJoCo precomputes shortcuts for sites originally placed at body origins.
    # Learned offsets invalidate those shortcuts; MJX always applies site_pos.
    model.site_sameframe[:] = 0
    data = mujoco.MjData(model)
    bodies = np.empty((frames, model.nbody, 3))
    sites = np.empty((frames, len(order), 3))
    for frame in range(frames):
        # A child's origin is scaled by its parent's scale. A site's local
        # coordinates are scaled by its own body's scale before adding offsets.
        model.body_pos[:] = base_body_pos * scales[frame, model.body_parentid, None]
        model.site_pos[:] = base_site_pos * scales[frame, model.site_bodyid, None] + offsets
        data.qpos[:] = poses[frame]
        mujoco.mj_kinematics(model, data)
        bodies[frame] = data.xpos
        sites[frame] = data.site_xpos[order]
    return bodies, sites
