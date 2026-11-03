"""Export helpers of the skeleton-fit engine (``hmb.fit_engine``).

* :func:`export_times` / :func:`grid_confidence`: a uniform time grid and the
  per-grid-time keypoint confidence, used to export the rigid fit at a fixed
  rate (``<stem>_opensim_skeleton_grid.mot``).  The analysed frames can be
  unevenly spaced in time (see ``hmb.video``); the kinematic-extremes and
  motion-smoothness metrics differentiate the motion and need uniform steps.
* :func:`variable_scale_params`: the per-frame body-scale parameters of a
  variable-scale fit, computed exactly like ``get_scale_params`` in
  ``metrics/bone_length/main.py`` but vectorised and compiled once.
"""
from __future__ import annotations

import numpy as np


def export_times(timestamps, grid_hz: float) -> np.ndarray:
    """Uniform grid ``k / grid_hz`` covering [timestamps[0], timestamps[-1]] (timestamps start at 0)."""
    timestamps = np.asarray(timestamps, dtype=float)
    if grid_hz is None or not grid_hz > 0:
        raise ValueError("grid_hz must be positive")
    span = float(timestamps[-1] - timestamps[0])
    count = int(np.floor(span * grid_hz + 1e-9)) + 1  # 1e-9: keep the end point despite rounding
    return timestamps[0] + np.arange(count, dtype=float) / float(grid_hz)


def grid_confidence(observation_times, confidences, grid_times) -> np.ndarray:
    """Confidence of the nearest observed frame for every grid time (ties -> earlier frame)."""
    observation_times = np.asarray(observation_times, dtype=float)
    confidences = np.asarray(confidences)
    grid_times = np.asarray(grid_times, dtype=float)
    right = np.clip(np.searchsorted(observation_times, grid_times, side="left"), 0, len(observation_times) - 1)
    left = np.clip(right - 1, 0, len(observation_times) - 1)
    choose_left = np.abs(grid_times - observation_times[left]) <= np.abs(observation_times[right] - grid_times)
    return confidences[np.where(choose_left, left, right)]


def _variable_scale_params_impl(model, timestamps):
    """``get_scale_params`` of ``metrics/bone_length/main.py`` (trajectory 0), vectorised over ``timestamps``."""
    import jax
    import jax.numpy as jnp

    # The trajectory output stacks joint angles, then optional actions and extra
    # channels, then the scale parameters; skip everything before the scales.
    scale_start_idx = model.njoints
    if hasattr(model, "nactions"):
        scale_start_idx += model.nactions
    if hasattr(model, "nextra"):
        scale_start_idx += model.nextra
    trajectory = jax.tree_util.tree_map(lambda x: jnp.take(x, 0, axis=0), model.trajectory)

    def get_scale_params(t):
        output = trajectory(t)[..., 0]
        # Same tanh squashing into [lower_limit, upper_limit] as the model's forward pass.
        lower_limit = jnp.broadcast_to(model.lower_limit, output.shape)
        upper_limit = jnp.broadcast_to(model.upper_limit, output.shape)
        range_val = (upper_limit - lower_limit) / 2.0
        range_val = jnp.clip(range_val, 0.01, None)
        middle = (upper_limit + lower_limit) / 2.0
        scaled = jnp.tanh(output) * range_val + middle
        output = jnp.where(model.apply_limits, scaled, output)
        return output[scale_start_idx:]

    return jax.vmap(get_scale_params)(timestamps)


_JITTED = None  # compiled on first use, so importing this module does not import JAX


def variable_scale_params(model, timestamps):
    """Per-frame scale parameters ``(T, n_scales)`` of a variable-scale fit at ``timestamps``."""
    global _JITTED
    if _JITTED is None:
        import equinox as eqx

        _JITTED = eqx.filter_jit(_variable_scale_params_impl)
    return _JITTED(model, timestamps)
