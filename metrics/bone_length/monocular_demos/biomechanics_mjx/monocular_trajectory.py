"""Variable-scale skeleton fit: the network also outputs the body scales at every time point.

Part of the variable-scale copy of the MonocularBiomechanics fitting code
(``metrics/bone_length/monocular_demos``; the rigid-fit original is
``baseline/monocular_demos``).  It differs from the rigid fit only in the scales: the
trajectory network outputs, after the joint angles, one scale parameter per entry of the
scale mixer at every time point (bounded by ``tanh`` to [-2, 2]), and :func:`loss` adds a
smoothness penalty on their frame-to-frame change (``scale_smoothness_regularization``).
The paper's bone-length metric (``metrics/bone_length/bone_length.py``) then measures how
much the bones change length.

* :class:`KineticsWrapper` couples an :class:`ImplicitTrajectory` (time -> joint angles and
  scales, joint angles bounded by ``tanh`` to the joint ranges unless
  ``enforce_joint_limits=False``) with the scaled skeleton (:class:`ForwardKinetics`) and
  keypoint-site offsets.
* :func:`get_model` builds it for a :class:`MonocularDataset`.
* :func:`loss` compares the skeleton's sites with the MeTRAbs keypoints: 3-D distance after
  removing the mean, 2-D reprojection (weight ``lambda_reprojection``), a penalty on the site
  offsets and the scale smoothness; keypoints are weighted by their confidence.
* :func:`fit_model` optimises it with AdamW and an exponentially decaying learning rate
  (:func:`build_optimizer`); the fused helpers below run many iterations per compiled call.
* :func:`export_trajectory` evaluates the fitted joint angles, velocities and scales.

HumanScore's fit engine (``hmb/fit_engine.py``) and ``metrics/bone_length/main.py`` use these
functions.
"""
import functools
import inspect
from collections import OrderedDict
from typing import Callable, Dict, List, Tuple, Union

import equinox as eqx
import jax
import mujoco
import optax
from jax import numpy as jnp
from jaxlie import SE3, SO3
from jaxtyping import Array, Float, Integer, PRNGKeyArray
from ..camera import get_extrinsic, project_distortion
from ..dataset import (
    KeypointDataset,
    MonocularDataset,
    level_floor_from_dataset,
)
from ..implicit_representation import (
    CubicSplineTrajectory,
    ENCODINGS,
    ImplicitTrajectory,
    ExplicitTrajectory,
    calculate_encoding_length,
    fixed_encoding,
)
from mujoco import mjx
from tqdm import trange

from .forward_kinematics import (
    ForwardKinematicsModelDescriptor,
    ForwardKinetics,
    angular_velocity_to_quaternion_derivative,
    normalize_velocity_from_quaternions,
)


# One jitted forward wrapper per ForwardKinetics instance, and one ForwardKinetics
# per (xml, site order) in get_model. Every model of a process then carries the
# *same* static objects. Distinct-but-similar static Modules (they hold JAX
# arrays) make JAX's jit cache compare tree definitions with Module.__eq__,
# which evaluates a nested jitted ``==`` under the cache lock and can deadlock a
# second fit in the same process. Values and compiled programs are unchanged.
_JIT_FORWARD_CACHE: Dict[int, Tuple] = {}
_FORWARD_KINETICS_CACHE: Dict[Tuple, "ForwardKinetics"] = {}


def shared_jit_forward(kinetic_model):
    """The one jitted forward function of ``kinetic_model`` (see the comment above)."""
    entry = _JIT_FORWARD_CACHE.get(id(kinetic_model))
    if entry is None or entry[0] is not kinetic_model:
        entry = (kinetic_model, eqx.filter_jit(kinetic_model))
        _JIT_FORWARD_CACHE[id(kinetic_model)] = entry
    return entry[1]


def cached_forward_kinetics(xml_path=None, joint_names=None):
    """The shared :class:`ForwardKinetics` of an (xml, site order) pair (see the comment above)."""
    try:
        key = (None if xml_path is None else str(xml_path), None if joint_names is None else tuple(joint_names))
        hash(key)
    except TypeError:  # e.g. an array of site indices: build without caching
        return ForwardKinetics(xml_path=xml_path, site_reorder=joint_names)
    if key not in _FORWARD_KINETICS_CACHE:
        _FORWARD_KINETICS_CACHE[key] = ForwardKinetics(xml_path=xml_path, site_reorder=joint_names)
    return _FORWARD_KINETICS_CACHE[key]


class KineticsWrapper(eqx.Module):
    """The fitted model: trajectory network(s) + scaled skeleton + learned site offsets.

    Holds one trajectory per video (stacked, selected with ``trajectory_selection``).
    Calling it with timestamps returns the skeleton state at those times (see ``__call__``).
    """

    trajectory: ImplicitTrajectory | CubicSplineTrajectory
    body_scale: jnp.array
    scale_mixer: jnp.array
    site_offsets: jnp.array
    lower_limit: jnp.array
    upper_limit: jnp.array
    apply_limits: jnp.array
    njoints: int = eqx.field(static=True)  # number of qpos joints
    nvjoints: int = eqx.field(static=True)  # number of vel joints (can be less than qpos if quaternions)
    nactions: int = eqx.field(static=True)
    nextra: int = eqx.field(static=True)
    n_scale_params: int = eqx.field(static=True)  # number of scale parameters
    forward: ForwardKinematicsModelDescriptor
    jit_forward: Callable = eqx.field(static=True)  # ForwardKinetics
    freeroot: bool = eqx.field(static=True)

    def __init__(
        self,
        kinetic_model: ForwardKinetics,
        trajectory=None,
        trajectory_type="implicit",
        scale_mixer=None,
        num_trajectories=1,
        encoding_length=None,
        cubic_sequence_length=None,
        max_time=60,
        min_freq=80,
        key=None,
        action=True,  # set to false for kinematic fits
        nextra=0,  # extra outpts from the implicit
        enforce_joint_limits=True,
        encoding="paper",
        top_hz=None,
        **kwargs,
    ):
        self.njoints = kinetic_model.mjx_model.nq
        self.nvjoints = kinetic_model.mjx_model.nv
        self.nactions = kinetic_model.mjx_model.nu
        nbody = kinetic_model.mjx_model.nbody
        self.nextra = nextra
        self.freeroot = True if kinetic_model.mjx_model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE else False

        # Determine number of scale parameters (before initializing scale_mixer)
        if scale_mixer is not None:
            self.n_scale_params = scale_mixer.shape[1]
        else:
            self.n_scale_params = 1

        max_time += 1.0  # add an additional second of representation space to cover the last frame future predictions

        if action:
            dims = self.njoints + self.nactions
        else:
            # for kinematic fits
            dims = self.njoints
            self.nactions = 0

        if key is None:
            key = jax.random.PRNGKey(0)

        # adds extra output dims
        dims += nextra

        # Add scale parameter dimensions (time-varying scale)
        # The second dimension of scale_mixer is the number of scale parameters (typically ~8)
        if scale_mixer is not None:
            n_scale_params = scale_mixer.shape[1]
        else:
            n_scale_params = 1
        dims += n_scale_params
        
        if trajectory is None:

            trajectory_keys = jax.random.split(key, num_trajectories)
            if trajectory_type.lower() == "implicit":

                if encoding not in ENCODINGS:
                    raise ValueError(f"encoding must be one of {ENCODINGS}")
                if encoding == "fixed":
                    # duration-independent top frequency 2*pi*top_hz; the lowest
                    # frequency covers the same span as the paper encoding
                    # (last timestamp + 1 s)
                    if top_hz is None or encoding_length is not None:
                        raise ValueError("encoding='fixed' needs top_hz and derives encoding_length")
                    encoding_length, encoding_max_time = fixed_encoding(float(max_time), top_hz)
                    time_options = {"max_time": encoding_max_time, "round_max_time": False}
                else:
                    if encoding_length is None:
                        encoding_length = calculate_encoding_length(max_time, min_freq=min_freq)
                    time_options = {"max_time": max_time}

                self.trajectory = [
                    ImplicitTrajectory(joints=dims, spatial_dims=1, encoding_length=encoding_length, **time_options, **kwargs, key=key)
                    for key in trajectory_keys
                ]
            elif trajectory_type.lower() == "cubic":

                if cubic_sequence_length is None:
                    cubic_sequence_length = int(max_time * min_freq)
                self.trajectory = [
                    CubicSplineTrajectory(sequence_length=cubic_sequence_length, joints=dims, spatial_dims=1, max_time=max_time, key=key)
                    for key in trajectory_keys
                ]

        else:
            self.trajectory = trajectory

        if not isinstance(self.trajectory, list):
            self.trajectory = [self.trajectory]

        # make the trajectory something we can index it to while jitting
        def f(*args):
            return jnp.stack(list(args), axis=0)

        self.trajectory = jax.tree_util.tree_map(f, *self.trajectory)
        self.scale_mixer = scale_mixer if scale_mixer is not None else jnp.ones((nbody, 1))

        dof_qpos_jntid = jnp.array(kinetic_model.dof_qpos_jntid)
        if action:
            self.lower_limit = jnp.concatenate(
                [
                    kinetic_model.mjx_model.jnt_range[dof_qpos_jntid, 0],
                    kinetic_model.mjx_model.actuator_ctrlrange[:, 0],
                    -jnp.ones(nextra),
                    -2.0 * jnp.ones(self.n_scale_params),  # scale lower bound (-2 to +2, ~0.135x to 7.4x)
                ]
            )
            self.upper_limit = jnp.concatenate(
                [
                    kinetic_model.mjx_model.jnt_range[dof_qpos_jntid, 1],
                    kinetic_model.mjx_model.actuator_ctrlrange[:, 1],
                    jnp.ones(nextra),
                    2.0 * jnp.ones(self.n_scale_params),  # scale upper bound
                ]
            )
            self.apply_limits = (
                jnp.concatenate(
                    [
                        kinetic_model.mjx_model.jnt_limited[dof_qpos_jntid],
                        kinetic_model.mjx_model.actuator_ctrllimited,
                        jnp.zeros(nextra),
                        jnp.ones(self.n_scale_params),  # apply limits to scale parameters
                    ]
                )
                >= 1
            )
        else:
            # for kinematic fits
            self.lower_limit = jnp.concatenate([kinetic_model.mjx_model.jnt_range[dof_qpos_jntid, 0], 
                                               -jnp.ones(nextra),
                                               -2.0 * jnp.ones(self.n_scale_params)])
            self.upper_limit = jnp.concatenate([kinetic_model.mjx_model.jnt_range[dof_qpos_jntid, 1], 
                                               jnp.ones(nextra),
                                               2.0 * jnp.ones(self.n_scale_params)])
            self.apply_limits = jnp.concatenate([kinetic_model.mjx_model.jnt_limited[dof_qpos_jntid], 
                                                jnp.zeros(nextra),
                                                jnp.ones(self.n_scale_params)]) >= 1
        # The scales are outputs of the trajectory; body_scale (set below) is only a default value.
        if not enforce_joint_limits:
            # Preserve scale/actuator bounds while exposing observed joint violations.
            self.apply_limits = self.apply_limits.at[:self.njoints].set(False)
        self.body_scale = jnp.zeros((self.scale_mixer.shape[1], 1))  # kept as an initial/default value
        self.site_offsets = jnp.zeros((kinetic_model.mjx_model.nsite, 3))

        self.forward = kinetic_model.get_descriptor()
        # NOTE: this throws a warning.
        # UserWarning: A JAX array is being set as static! This can result in unexpected behavior and is usually a mistake to do.
        # but doing the compilation below is quite slow
        self.jit_forward = shared_jit_forward(kinetic_model)

    def __call__(
        self,
        timepoint,
        trajectory_selection=0,
        trajectory_only=False,
        skip_action=False,
        skip_vel=False,
        steps=1,
        dt=1.0 / 30.0,
        fast_inference=False,
        freeze_scale=False,
        margin=None,
        action_repeats=1,
        check_constraints=True,
        return_scale_params=False,
    ):
        scale = 1 + jax.lax.stop_gradient(self.scale_mixer) @ self.body_scale

        scale = jax.lax.cond(freeze_scale, lambda: jax.lax.stop_gradient(scale), lambda: scale)
        site_offsets = jax.lax.cond(freeze_scale, lambda: jax.lax.stop_gradient(self.site_offsets), lambda: self.site_offsets)

        # NOTE: the dt passed here should be the frame timestep divided by the frame interpolation
        # here, we additionally rescale it by the number of action repeats
        model_timestep = None if skip_action else dt / action_repeats
            
        trajectory = jax.tree_util.tree_map(lambda x: jnp.take(x, trajectory_selection, axis=0), self.trajectory)

        if fast_inference:

            def vectorize(f: Callable):
                # vectorize using a scan to avoid the compilation time of vmap
                def scan_fn(carry, x):
                    return (carry + 1, f(x))

                return lambda x: jax.lax.scan(scan_fn, 0, x)[1]

        else:
            vectorize = jax.vmap

        def scale_limits(x):
            # broadcast lower_limit and upper limit to be the same shape and
            # last dimension as the pose

            lower_limit = jax.lax.stop_gradient(jnp.broadcast_to(self.lower_limit, x.shape))
            upper_limit = jax.lax.stop_gradient(jnp.broadcast_to(self.upper_limit, x.shape))

            range = (upper_limit - lower_limit) / 2.0
            # for joints with no range in the model, prevent gradients having divide by zero
            range = jnp.clip(range, 0.01, None)
            middle = (upper_limit + lower_limit) / 2.0

            # apply the bounded limits to all of the joints but ignore the global pose
            # this includes allowing the global pose to wrap around
            scaled = jnp.tanh(x) * range + middle
            x = jnp.where(self.apply_limits, scaled, x)
            return x

        def get_pose_and_action(t):
            pose_action_extra = trajectory(t)[..., 0]
            pose_action_extra = scale_limits(pose_action_extra)
            pose = pose_action_extra[..., : self.njoints]
            action = pose_action_extra[..., self.njoints : (self.njoints + self.nactions)]
            # Extract extra and scale parameters
            extra_start = self.njoints + self.nactions
            scale_start = extra_start + self.nextra
            extra = pose_action_extra[..., extra_start : scale_start]
            body_scale_params = pose_action_extra[..., scale_start :]  # time-varying scale parameters
            return pose, action, extra, body_scale_params

        def process_single_time(t):
            pose, action, extra, body_scale_params = get_pose_and_action(t)

            # Compute time-varying scale
            time_varying_scale = 1 + jax.lax.stop_gradient(self.scale_mixer) @ body_scale_params[..., jnp.newaxis]
            time_varying_scale = jax.lax.cond(freeze_scale, 
                                             lambda: jax.lax.stop_gradient(scale), 
                                             lambda: time_varying_scale)

            if skip_vel:
                vel = jnp.zeros((self.nvjoints))
            else:
                # analyztically compute joint velocities
                vel = jax.jacfwd(lambda x: scale_limits(trajectory(x)[:, 0]))(t)
                raw_vel = vel[..., : self.njoints]

                # for freeroot, this changes from derivative w.r.t. qpos to the qvel with one less element
                vel = normalize_velocity_from_quaternions(pose, raw_vel, self.freeroot)

            if trajectory_only:
                if return_scale_params:
                    return pose, vel, action, extra, body_scale_params
                return pose, vel, action, extra

            if skip_action:
                return (
                    self.jit_forward(
                        joint_angles=pose,
                        joint_velocities=vel,
                        scale=time_varying_scale,  # use time-varying scale
                        site_offsets=site_offsets,
                        margin=margin,
                        check_constraints=check_constraints,
                        timestep=model_timestep,
                    ),
                    (pose, vel, action),
                    extra,
                )

            def scan_function(carry, i):
                pose, action, extra, body_scale_params = get_pose_and_action(t + i * dt)
                # Compute scale at this time step
                step_scale = 1 + jax.lax.stop_gradient(self.scale_mixer) @ body_scale_params[..., jnp.newaxis]
                step_scale = jax.lax.cond(freeze_scale, 
                                         lambda: jax.lax.stop_gradient(scale), 
                                         lambda: step_scale)
                vel = jax.jacfwd(lambda x: scale_limits(trajectory(x)[:, 0]))(t + i * dt)[..., : self.njoints]
                vel = normalize_velocity_from_quaternions(pose, vel, self.freeroot)

                return carry, (pose, vel, action)

            _, (future_pose, future_vel, action) = jax.lax.scan(
                scan_function, (), jnp.arange(steps + 1)
            )
            action = action[:-1]

            # Note: the scale here should use the current time step's scale,
            # not the scale of future scan steps.
            rollout = lambda x, y, z: self.jit_forward(
                joint_angles=x,
                joint_velocities=y,
                action=z,
                scale=time_varying_scale,  # use time-varying scale
                site_offsets=site_offsets,
                margin=margin,
                action_repeats=action_repeats,
                check_constraints=check_constraints,
                timestep=model_timestep,
            )
            return rollout(pose, vel, action), (future_pose, future_vel, action), extra

        return vectorize(process_single_time)(timepoint)



def huber(x, delta=5.0):
    """Huber loss."""
    x = jnp.where(jnp.abs(x) < delta, 0.5 * x**2, delta * (jnp.abs(x) - 0.5 * delta))
    return x

def get_default_wrapper():
    """
    Convenience function to get a model for the 3D MOVI keypoints.

    Returns:
        A tuple of the model and loss function.
    """

    from .forward_kinematics import ForwardKinetics, movi_joint_names

    # default forward kinematic model
    fk = ForwardKinetics(site_reorder=movi_joint_names)

    # default scaler mixer
    scale_mixer = fk.build_default_scale_mixer()

    # use the implicit function wrapper
    fkw = KineticsWrapper(fk, scale_mixer=scale_mixer, key=jax.random.PRNGKey(0))

    return fkw

def get_custom_wrapper(fk: ForwardKinetics, dataset: MonocularDataset, dist=1.5, **kwargs) -> KineticsWrapper:
    """This will buiild an implicit trajectory with custom initializations computed from the dataset."""
    import numpy as np
    from jaxlie import SO3

    min_freq = kwargs.get("min_freq", 80)  # safely handle default

    dims = fk.mjx_model.nq + kwargs.get("nextra", 0) + fk.build_default_scale_mixer().shape[1]
    if "encoding_length" not in kwargs:
        encoding_length = calculate_encoding_length(dataset.max_time, min_freq=min_freq)
    else:
        encoding_length = kwargs["encoding_length"]

    # calculate custom output layer bias
    custom_biases = []
    if dataset.phone_attitude is not None:
        rvec_means = [
            np.median(dataset.get_all_attitude(i)[1], axis=0)
            for i in range(len(dataset))
        ]

        for rvec in rvec_means:
            custom_bias = np.full((dims,), np.nan)

            # Set rotation bias from median rotation vector
            custom_bias[-3:] = rvec

            # Convert rotation vector to rotation matrix and extract direction
            rmat = SO3.exp(rvec).as_matrix()
            z_axis = rmat[:, 2]

            # Project direction onto ground plane (XY plane)
            ground_normal = np.array([0, 0, 1])
            vproj = z_axis - np.dot(z_axis, ground_normal) * ground_normal
            vproj /= np.linalg.norm(vproj)

            # Estimate 2D position in XY plane, assuming person faces away from camera
            x_pos = vproj[0] * dist
            y_pos = vproj[1] * dist

            # Assign position and facing direction depending on freeroot/not
            if fk.joint_names[0].startswith("root"):
                custom_bias[3:5] = 0.7071
                custom_bias[:3] = np.array([x_pos, y_pos, -1.5])
            else:
                custom_bias[:3] = np.array([x_pos, -y_pos, -1.5])

            custom_biases.append(custom_bias)

    else:
        # For staic fits generally the camera is not at origin so just get the freeroot right
        for _ in range(len(dataset)):
            custom_bias = np.full((dims,), np.nan)
            if fk.joint_names[0].startswith("root"):
                custom_bias[3:5] = 0.7071
            custom_biases.append(custom_bias)


    trajectory_keys = jax.random.split(jax.random.PRNGKey(0), len(dataset))
    custom_implicits = [
        ImplicitTrajectory(
            joints=dims, spatial_dims=1, max_time=dataset.max_time, encoding_length=encoding_length, custom_bias=custom_biases[i], key=key
        )
        for i, key in enumerate(trajectory_keys)
    ]

    custom_trajectory = KineticsWrapper(
        fk,
        scale_mixer=fk.build_default_scale_mixer(),
        key=jax.random.PRNGKey(0),
        num_trajectories=len(dataset),
        max_time=dataset.max_time,
        action=False,
        trajectory=custom_implicits,
        **kwargs,
    )

    return custom_trajectory


def get_model(dataset: KeypointDataset | MonocularDataset, xml_path=None, joint_names=None, scale_mappings=None, **kwargs) -> KineticsWrapper:
    """
    Get a kinetics wrapper trajectory for training from a dataset.

    Args:
        dataset (KeypointDataset | MonocularDataset): The dataset (one trajectory per video).
        xml_path (str): The skeleton model.
        joint_names (List[str]): Keypoint names, matched to the model's sites.
        scale_mappings (Dict | None): Custom scale mixer; default ``build_default_scale_mixer``.
        **kwargs: Passed to :class:`KineticsWrapper` (e.g. ``enforce_joint_limits``).

    Returns:
        KineticsWrapper: The model.
    """

    if joint_names is None:
        import warnings

        warnings.warn("Using default joint names. This may not be correct for your dataset.")
        joint_names = (KeypointSet & {"xml_fname": xml_path.split("/")[-1]}).get_site_reorder()

    # default forward kinematic model
    fk = cached_forward_kinetics(xml_path, joint_names)

    if scale_mappings is None:
        scale_mixer = fk.build_default_scale_mixer()
        print("Using default scale mixer.")
    else:
        scale_mixer = fk.build_custom_scale_mixer(scale_mappings)

    if dataset.phone_attitude is not None or fk.joint_names[0].startswith("root"):
        trajectory = get_custom_wrapper(fk, dataset, **kwargs)
    else:
        trajectory = KineticsWrapper(
            fk,
            scale_mixer=scale_mixer,
            key=jax.random.PRNGKey(0),
            num_trajectories=len(dataset),
            max_time=dataset.max_time,
            action=False,
            **kwargs,
        )

    model = trajectory

    return model


def project_dynamic(points: Float[Array, "time joints 3"], calibation: Dict, rvec):
    """Project ``points`` with a camera whose orientation ``rvec`` changes over time (handheld phone)."""

    camera_params = {"mtx": calibation["mtx"], "dist": calibation["dist"]}  # these do not vary with time
    tvec = calibation["tvec"]  # this is the same for all times

    def project_time(_rvec, _tvec, points, _camera_params=camera_params):
        _camera_params["rvec"] = -1 * _rvec.reshape(1, 3)  # reshape for compatibiltiy with multi-camera code
        _camera_params["tvec"] = _tvec.reshape(1, 3)
        return project_distortion(_camera_params, 0, points)

    return jax.vmap(project_time, in_axes=(0, None, 0))(rvec, tvec, points)  # for future changing tvec, remove this none


def get_extrinsic_dynamic(camera_params, i, rvec: Float[Array, "time 3"]):
    """World-to-camera transforms (time, 4, 4) for a time-varying camera orientation ``rvec``."""
    tvec = jnp.take(camera_params["tvec"], i, axis=0) * 1000.0

    def _get_extrinsic(_rvec):
        rot = SO3.exp(-1 * _rvec)
        return SE3.from_rotation_and_translation(rot, tvec).as_matrix()

    return jax.vmap(_get_extrinsic)(rvec)


# if you add args to this function the dont forget to add them to the vmap
def loss(
    model: KineticsWrapper,
    x: Tuple[Integer, Float[Array, "camera_times"], Float[Array, "attitude_times"]],
    y: Tuple[
        Float[Array, "cameras camera_times keypoints 2"],
        Float[Array, "cameras camera_times keypoints 3"],
        Float[Array, "cameras camera_times keypoints"],
        Float[Array, "attitude_times 3"],
    ],
    lambda_reprojection: Array,
    camera_calibration: Dict,
    site_offset_regularization: float = 1e2,
    static: bool = False,
    center_3d: bool = True,
    quat_regularization: float = 100.0,
    ang_pred_regularization: float = 0.0,
    ang_pred_vel_scale: float = 10.0,
    scale_smoothness_regularization: float = 1e3,  # scale smoothness regularization
    dt: float = 1.0 / 30.0,
) -> Tuple[Float, Dict]:
    """This is written for a single trajectory however will work for a batch of trajectories as well."""

    # vmap over the data so this is applied to a single trial
    if isinstance(x[0], jnp.ndarray) and len(x[0].shape) == 1:
        # if there is a batch dimension, automatically vmap this function over it
        _loss = lambda x, y: loss(
            model=model,
            x=x,
            y=y,
            lambda_reprojection=lambda_reprojection,
            site_offset_regularization=site_offset_regularization,
            camera_calibration=camera_calibration,
            static=static,
            center_3d=center_3d,
            quat_regularization=quat_regularization,
            ang_pred_regularization=ang_pred_regularization,
            ang_pred_vel_scale=ang_pred_vel_scale,
            scale_smoothness_regularization=scale_smoothness_regularization,
            dt=dt,
        )
        _loss = jax.vmap(_loss, in_axes=(0, 0), out_axes=(0, 0))
        l, metrics = _loss(x, y)
        return jnp.sum(l), metrics
    elif isinstance(x[0], jnp.ndarray) and len(x[0].shape) > 1:
        raise ValueError("Cannot batch over more than one dimension in the input data.")

    # unpack data. An optional fourth input element marks real (True) vs padding
    # (False) samples. Padding has zero confidence (no keypoint term) and is
    # excluded from the neighbouring-sample scale smoothness term below.
    if len(x) == 4:
        trajectory_id, video_timestamps, attitude_timestamps, sample_valid = x
    else:
        trajectory_id, video_timestamps, attitude_timestamps = x
        sample_valid = None
    keypoints2d, keypoints3d, keypoints_confidences, attitude = y
    keypoints_confidences = keypoints_confidences[0]

    # forward pass at video times
    (state, constraints, next_states), (ang, vel, action), rvec_video = model(
        video_timestamps,
        trajectory_selection=trajectory_id,
        skip_vel=False if ang_pred_regularization > 0 else True,
        skip_action=True,
        steps=0,
        margin=1e-3,
        action_repeats=0,
        dt=dt,
        check_constraints=False,
    )
    if static:
        lambda_attitude = 0.0
        l_attitude = 0.0
        T_nc = get_extrinsic(camera_calibration, 0)
        p_n = jnp.concatenate([state.site_xpos * 1000, jnp.ones((*state.site_xpos.shape[:-1], 1))], axis=-1)
        p_c_hat = (T_nc @ p_n[..., None])[..., :3, 0]
        model_keypoints_mm = p_c_hat

        # reprojection
        if lambda_reprojection > 0.0:
            projected_points = project_distortion(camera_calibration, 0, state.site_xpos * 1000)  # this is performing the extrinsic math twice
    else:
        rvec_video = SO3.exp(rvec_video).normalize().log()
        # attitude
        lambda_attitude = 1e0
        _, _, _, qnc_phone = model(
            attitude_timestamps,
            trajectory_selection=trajectory_id,
            skip_vel=True,
            skip_action=True,
            steps=0,
            margin=1e-3,
            action_repeats=0,
            dt=dt,
            trajectory_only=True,
        )
        q_diff = (SO3.exp(attitude).inverse() @ SO3.exp(qnc_phone).normalize()).parameters()
        quat_angle = lambda q: 2 * jnp.arctan2(jnp.sqrt(q[1] ** 2 + q[2] ** 2 + q[3] ** 2), jnp.abs(q[0]))
        l_attitude = jnp.mean(jax.vmap(quat_angle)(q_diff) * 180 / jnp.pi)

        # keypoint
        T_nc = get_extrinsic_dynamic(camera_calibration, 0, rvec_video)
        p_n = jnp.concatenate([state.site_xpos * 1000, jnp.ones((*state.site_xpos.shape[:-1], 1))], axis=-1)
        p_c_hat = jnp.einsum("tij,tbj->tbi", T_nc, p_n)[..., :3]
        model_keypoints_mm = p_c_hat

        # reprojection
        if lambda_reprojection > 0.0:
            projected_points = project_dynamic(state.site_xpos * 1000, camera_calibration, rvec_video)

    metrics = {}

    # region 3d loss
    lambda_3d = 1e0
    video_keypoints_mm = keypoints3d[0]
    # remove mean from keypoints
    if center_3d:
        model_keypoints_mm = model_keypoints_mm - jnp.mean(model_keypoints_mm, axis=1, keepdims=True)
        video_keypoints_mm = video_keypoints_mm - jnp.mean(video_keypoints_mm, axis=1, keepdims=True)
    dist_3d = huber(jnp.linalg.norm(model_keypoints_mm / 10 - video_keypoints_mm / 10, axis=-1) * keypoints_confidences, delta=10)
    norm = keypoints_confidences.sum() + 1e-7
    dist_3d = jnp.sum(dist_3d) / norm
    # endregion

    # site offset
    l_site_offset = jnp.sum(jnp.square(model.site_offsets))

    # reprojection loss
    if lambda_reprojection > 0.0:
        dist_2d = huber(jnp.linalg.norm(projected_points - keypoints2d[0], axis=-1) * keypoints_confidences, delta=50)
        dist_2d = jnp.nansum(dist_2d) / norm
    else:
        dist_2d = 0.0

    # Log raw metrics
    metrics = {
        "mean_3d_raw": dist_3d,
        "site_offset_raw": l_site_offset,
        "mean_reprojection_raw": dist_2d,
        "attitude_loss_raw": l_attitude,
    }

    # Calculate scaled metrics
    metrics["mean_3d_scaled"] = metrics["mean_3d_raw"] * lambda_3d
    metrics["site_offset_scaled"] = metrics["site_offset_raw"] * site_offset_regularization
    metrics["mean_reprojection_scaled"] = metrics["mean_reprojection_raw"] * lambda_reprojection
    metrics["attitude_loss_scaled"] = metrics["attitude_loss_raw"] * lambda_attitude

    # Scale smoothness regularization (temporal smoothness constraint on scale)
    if scale_smoothness_regularization > 0:
        # Extract scale parameters at all timestamps from the trajectory
        trajectory = jax.tree_util.tree_map(lambda x: jnp.take(x, trajectory_id, axis=0), model.trajectory)
        
        def get_scale_params_at_time(t):
            output = trajectory(t)[..., 0]
            # Apply limits
            lower_limit = jnp.broadcast_to(model.lower_limit, output.shape)
            upper_limit = jnp.broadcast_to(model.upper_limit, output.shape)
            range_val = (upper_limit - lower_limit) / 2.0
            range_val = jnp.clip(range_val, 0.01, None)
            middle = (upper_limit + lower_limit) / 2.0
            scaled_output = jnp.tanh(output) * range_val + middle
            output = jnp.where(model.apply_limits, scaled_output, output)
            # Extract scale parameters (after qpos, action, extra)
            scale_start = model.njoints + model.nactions + model.nextra
            scale_params = output[scale_start:]
            return scale_params
        
        # Compute scale parameters for all frames
        scale_params_all = jax.vmap(get_scale_params_at_time)(video_timestamps)  # shape: (n_frames, n_scale_params)

        # Compute changes between neighboring frames
        scale_changes = jnp.diff(scale_params_all, axis=0)  # shape: (n_frames-1, n_scale_params)

        # Smoothness loss: encourage smooth temporal evolution of scale
        if sample_valid is None:
            scale_smoothness_loss = jnp.mean(jnp.square(scale_changes))
        else:
            # mean over neighbouring pairs of real samples only (identical to the
            # unmasked mean when no sample is padding)
            pair_valid = jnp.logical_and(sample_valid[1:], sample_valid[:-1]).astype(scale_changes.dtype)[:, None]
            scale_smoothness_loss = jnp.sum(jnp.square(scale_changes) * pair_valid) / (
                jnp.sum(pair_valid) * scale_changes.shape[1]
            )
        
        metrics["scale_smoothness_raw"] = scale_smoothness_loss
        metrics["scale_smoothness_scaled"] = scale_smoothness_loss * scale_smoothness_regularization
    else:
        metrics["scale_smoothness_raw"] = 0.0
        metrics["scale_smoothness_scaled"] = 0.0
        scale_smoothness_loss = 0.0

    # Total loss
    l = (metrics["mean_3d_scaled"] + metrics["site_offset_scaled"] + 
         metrics["attitude_loss_scaled"] + metrics["mean_reprojection_scaled"] +
         metrics["scale_smoothness_scaled"])

    # freeroot loss
    if model.freeroot:
        # NOTE: this only applies to the free root right now
        quat_error = 1 - jnp.linalg.norm(ang[..., 3:7], axis=-1)
        metrics["quat_error"] = jnp.mean(jnp.square(quat_error))
        l += metrics["quat_error"] * quat_regularization

    if ang_pred_regularization > 0:
        # use a simple future prediction to get velocity roughly smoothed out
        ang_dt, _, _, _ = model(video_timestamps + dt, trajectory_selection=trajectory_id, trajectory_only=True)

        # we are trying to use our estimates of the future angle to supervise the velocity -- but dont want
        # to use the noisy velocity estimates to influence the pose estimates so stop the gradient here
        ang_dt = jax.lax.stop_gradient(ang_dt)

        if model.freeroot:
            qdot = angular_velocity_to_quaternion_derivative(ang[..., 3:7], vel[..., 3:6])
            vel = jnp.concatenate([vel[..., :3], qdot, vel[..., 6:]], axis=-1)

        pred_ang = ang + vel * dt
        err = pred_ang - ang_dt
        err = err.at[:, :3].set(err[:, :3] * ang_pred_vel_scale)
        metrics["ang_pred"] = jnp.mean(jnp.square(err))
        l = l + metrics["ang_pred"] * ang_pred_regularization

    # make loss the first key in the dictionary by popping and building a new dictionary with the rest
    metrics = {"total": l, **metrics}

    return l, metrics


@eqx.filter_jit
def step(model, opt_state, data, loss_grad, optimizer, **kwargs):
    """One optimiser step: loss and gradients on ``data``, then the parameter update."""
    x, targets = data
    (val, metrics), grads = loss_grad(model, x=x, y=targets, **kwargs)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return val, model, opt_state, metrics


def fetch_data(i, dataset, N, model_device, robust_camera_weights=False):
    """Training batch ``i`` (one sample per video) on the model's device."""
    # fetch the next batch of data and move to the same device as the model
    idx = jnp.arange(N) + i * N
    if robust_camera_weights:
        data = dataset.get_with_camera_weights(idx)
    else:
        data = dataset[idx]
    data = jax.tree_util.tree_map(lambda x: jax.device_put(x, model_device), data)
    return data


def commit_arrays_to_device(tree, device):
    """Avoid a second compile caused by uncommitted-to-committed state changes."""
    return jax.tree_util.tree_map(lambda value: jax.device_put(value, device) if eqx.is_array(value) else value, tree)


@eqx.filter_jit
def export_trajectory(model, timestamps, trajectory_selection=0):
    """Avoid embedding freshly fitted weights as scan constants at each export."""
    return model(timestamps, trajectory_selection=trajectory_selection, steps=0,
                 skip_action=True, fast_inference=True, check_constraints=False,
                 trajectory_only=True, return_scale_params=True)


# ---------------------------------------------------------------------------
# Fused / batched training building blocks.
#
# They reproduce fit_model's iteration exactly (same sample order, loss,
# optimizer and learning-rate schedule as a function of the iteration index)
# but run many iterations per compiled call and can be vmapped over a stack of
# independent models (one per video). NOTE: this section is duplicated
# verbatim in metrics/bone_length/monocular_demos/biomechanics_mjx/
# monocular_trajectory.py; keep both copies identical.
# ---------------------------------------------------------------------------


def build_optimizer(
    max_iters: int,
    lr_init_value: float = 1e-4,
    lr_end_value: float = 1e-7,
    warmup_steps: int = 0,
    transition_begin: int = 0,
    clip_by_global_norm: float = 0.1,
    optimizer_zero_nans: bool = False,
):
    """Return (optimizer, learning_rate_schedule) exactly as fit_model builds them.

    The exponential decay compresses with ``max_iters``: lr goes from
    ``lr_init_value`` to ``lr_end_value`` in ``max_iters // 10`` decay intervals.
    """
    # work out the transition steps to make the desired schedule
    if max_iters < 1 or warmup_steps >= max_iters:
        raise ValueError("max_iters must be positive and greater than warmup_steps")
    transition_steps = 10
    decay_intervals = max(1, (max_iters - warmup_steps) // transition_steps)
    lr_decay_rate = (lr_end_value / lr_init_value) ** (1.0 / decay_intervals)
    learning_rate = optax.warmup_exponential_decay_schedule(
        init_value=0,
        peak_value=lr_init_value,
        end_value=lr_end_value,
        warmup_steps=warmup_steps,
        transition_begin=transition_begin,
        decay_rate=lr_decay_rate,
        transition_steps=transition_steps,
    )

    # create learning rates for different components
    components = [optax.adamw(learning_rate=learning_rate, b1=0.8, weight_decay=1e-5)]
    if optimizer_zero_nans:
        components.append(optax.zero_nans())
    if clip_by_global_norm > 0:
        components.append(optax.clip_by_global_norm(clip_by_global_norm))
    return optax.chain(*components), learning_rate


def padded_cyclic_sample(data: Dict, i, with_valid: bool = False):
    """One training sample of a single-trial video, padded to a fixed length.

    ``data`` holds ``timestamps`` (P,), ``kp2d`` (1, P, K, 2), ``kp3d`` (1, P, K, 3),
    ``conf`` (1, P, K) and ``length`` (scalar number of real frames T <= P).
    Real frames are taken in MonocularDataset's deterministic order for a full
    clip sample, ``(arange(T) + i) % T``; padding frames (zero confidence, a copy
    of the last frame at the last timestamp) follow at their own positions.
    For P == T this is exactly ``dataset[jnp.arange(1) + i]``.
    """
    timestamps = data["timestamps"]
    padded = timestamps.shape[0]
    length = data["length"]
    pos = jnp.arange(padded)
    valid = pos < length
    idx = jnp.where(valid, (pos + i) % length, pos)
    x = (
        jnp.zeros((1,), dtype=int),
        jnp.take(timestamps, idx, axis=0)[None],
        None,
    )
    if with_valid:
        x = x + (valid[None],)
    y = (
        jnp.take(data["kp2d"], idx, axis=1)[None],
        jnp.take(data["kp3d"], idx, axis=1)[None],
        jnp.take(data["conf"], idx, axis=1)[None],
        None,
    )
    return x, y


def make_train_chunk(loss_grad: Callable, optimizer, sample_fn: Callable, steps_per_call: int, static):
    """Build ``chunk(params, opt_state, data, start) -> (params, opt_state, losses, metrics)``.

    One call runs iterations ``start .. start + steps_per_call - 1`` in a
    ``lax.scan``. Every iteration is fit_model's ``step``: sample
    ``sample_fn(data, i)``, value-and-grad of the loss, optimizer update and
    apply. ``params``/``static`` come from ``eqx.partition(model, eqx.is_array)``
    so the chunk can be ``jax.vmap``-ed over a stack of independent models.
    ``loss_grad(model, x, y, data)`` returns ``((loss, metrics), grads)``.
    Returned ``losses`` (steps_per_call,) and ``metrics`` (steps_per_call, ...)
    are per iteration, metrics unreduced as returned by the loss.
    """
    if steps_per_call < 1:
        raise ValueError("steps_per_call must be positive")

    def chunk(params, opt_state, data, start):
        def body(carry, k):
            params, opt_state = carry
            i = start + k
            model = eqx.combine(params, static)
            x, y = sample_fn(data, i)
            (val, metrics), grads = loss_grad(model, x, y, data)
            updates, opt_state = optimizer.update(grads, opt_state, model)
            model = eqx.apply_updates(model, updates)
            return (eqx.filter(model, eqx.is_array), opt_state), (val, metrics)

        (params, opt_state), (vals, metrics) = jax.lax.scan(body, (params, opt_state), jnp.arange(steps_per_call))
        return params, opt_state, vals, metrics

    return chunk


def chunk_schedule(max_iters: int, steps_per_call: int):
    """(start, length) of each fused call: full calls, then one shorter remainder call."""
    if max_iters < 1 or steps_per_call < 1:
        raise ValueError("max_iters and steps_per_call must be positive")
    return [(start, min(steps_per_call, max_iters - start)) for start in range(0, max_iters, steps_per_call)]


def _fit_model_fused(model, opt_state, optimizer, learning_rate, loss_grad, dataset, max_iters, steps_per_call,
                     step_callback, callback_interval):
    """fit_model's loop with ``steps_per_call`` iterations per compiled call."""
    import numpy as np

    N = len(dataset)
    params, static = eqx.partition(model, eqx.is_array)
    params = jax.tree_util.tree_map(jnp.copy, params)  # calls donate params; never the caller's buffers
    sample = lambda data, i: dataset[jnp.arange(N) + i * N]
    chunks = {}
    decile = max(1, max_iters // 10)
    smoothed = None
    metrics = None
    counter = trange(max_iters)
    for start, length in chunk_schedule(max_iters, steps_per_call):
        if length not in chunks:
            chunks[length] = jax.jit(make_train_chunk(lambda m, x, y, data: loss_grad(m, x=x, y=y), optimizer, sample,
                                                      length, static), donate_argnums=(0, 1))
        params, opt_state, vals, step_metrics = chunks[length](params, opt_state, None, jnp.array(start))
        end = start + length
        vals = np.asarray(vals)
        if start <= 1000 < end and np.isnan(vals[1000 - start]):
            raise ValueError("Loss is NaN on iteration 1000.")
        host_metrics = {k: np.asarray(v) for k, v in step_metrics.items()}
        for j, i in enumerate(range(start, end)):
            reduced = {k: v[j].mean(axis=0) for k, v in host_metrics.items()}
            smoothed = reduced if smoothed is None else {k: 0.9 * smoothed[k] + 0.1 * reduced[k] for k in reduced}
            if i > 0 and i % decile == 0:
                print(f"iter: {i} loss: {vals[j]}.")
        metrics = jax.tree_util.tree_map(lambda v: v[length - 1], step_metrics)
        callback_steps = [i for i in range(start, end) if i % callback_interval == 0]
        if step_callback is not None and callback_steps:
            i = callback_steps[-1]
            step_callback(i, eqx.combine(params, static), dataset, {**metrics, "learning_rate": learning_rate(i)})
        counter.update(length)
        counter.set_postfix({k: float(v) for k, v in smoothed.items()})
    counter.close()
    return eqx.combine(params, static), metrics


def fit_model(
    model: KineticsWrapper,
    dataset: Tuple | MonocularDataset,
    lr_end_value: float = 1e-7,
    lr_init_value: float = 1e-4,
    warmup_steps: int = 0,
    transition_begin: int = 0,
    max_iters: int = 25000,
    site_offset_regularization: float = 1e2,
    lambda_reprojection: float = 1e-1,
    clip_by_global_norm: float = 0.1,
    optimizer_zero_nans: bool = False,
    center_3d: bool = True,
    step_callback: Callable | None = None,
    callback_interval: int = 1,
    steps_per_call: int = 1,
    **kwargs,  # additional kwargs go to the loss function
):
    """
    Fits the model to the given keypoints.

    ``steps_per_call > 1`` runs that many optimizer iterations inside one
    compiled ``lax.scan`` call (same data order, objective, optimizer and
    learning-rate schedule per iteration index). Callbacks then fire once per
    call that contains a callback iteration, with the end-of-call model.

    Parameters:
        model (KineticsWrapper): The model (from get_model).
        dataset (MonocularDataset): The dataset.
        lr_end_value (float): The end value of the learning rate.
        lr_init_value (float): The initial value of the learning rate.
        max_iters (int): The maximum number of iterations.
        site_offset_regularization (float): The site offset regularization.
        lambda_reprojection (float): Weight of the 2-D reprojection loss.
        step_callback (Callable): Called as ``step_callback(i, model, dataset, metrics)``
            every ``callback_interval`` iterations.

    Returns:
        (model, metrics): the fitted model and the loss terms of the last iteration.
    """

    if callback_interval < 1:
        raise ValueError("callback_interval must be positive")
    if steps_per_call < 1:
        raise ValueError("steps_per_call must be positive")
    is_static = kwargs.pop("static", dataset.phone_attitude is None)

    # find the camera frame rate for the dataset
    if is_static:
        kwargs['dt'] = 1.0 / 29.0
    else:
        kwargs['dt'] = 1.0 / 30.0

    optimizer, learning_rate = build_optimizer(
        max_iters,
        lr_init_value=lr_init_value,
        lr_end_value=lr_end_value,
        warmup_steps=warmup_steps,
        transition_begin=transition_begin,
        clip_by_global_norm=clip_by_global_norm,
        optimizer_zero_nans=optimizer_zero_nans,
    )

    # Match filter_value_and_grad: boolean masks are fixed model metadata.
    # Otherwise their Adam state disappears after step zero and recompiles JIT.
    model_device = next(iter(model.scale_mixer.devices()))
    model = commit_arrays_to_device(model, model_device)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    opt_state = commit_arrays_to_device(opt_state, model_device)

    loss_grad = eqx.filter_value_and_grad(
        functools.partial(
            loss,
            site_offset_regularization=site_offset_regularization,
            lambda_reprojection=lambda_reprojection,
            camera_calibration=dataset.camera_params,
            static=is_static,
            center_3d=center_3d,
            **kwargs
        ),
        has_aux=True,
    )

    if steps_per_call > 1:
        return _fit_model_fused(model, opt_state, optimizer, learning_rate, loss_grad, dataset, max_iters,
                                steps_per_call, step_callback, callback_interval)

    N = len(dataset)
    local_fetch_data = jax.jit(functools.partial(fetch_data, dataset=dataset, N=N, model_device=model_device))

    smoothed_metrics = None

    @jax.jit
    def update_metrics(new_metrics, old_metrics):
        new_metrics_reduced = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), new_metrics)

        # Function to check if the structures are the same
        def is_same_structure(x, y):
            if isinstance(x, dict) and isinstance(y, dict):
                if x.keys() != y.keys():
                    return False
                return all(is_same_structure(x[k], y[k]) for k in x)
            return type(x) == type(y)

        if old_metrics is None or not is_same_structure(new_metrics_reduced, old_metrics):
            return new_metrics_reduced
        else:
            return jax.tree_util.tree_map(lambda x, y: 0.9 * x + 0.1 * y, old_metrics, new_metrics_reduced)

    counter = trange(max_iters)
    for i in counter:
        data = local_fetch_data(jnp.array(i))
        val, model, opt_state, metrics = step(model, opt_state, data, loss_grad, optimizer)
        if i == 1000:
            if jnp.isnan(val):
                raise ValueError(f"Loss is NaN on iteration {i}.")

        smoothed_metrics = update_metrics(metrics, smoothed_metrics)

        if step_callback is not None and i % callback_interval == 0:
            step_callback(i, model, dataset, {**metrics, "learning_rate": learning_rate(i)})

        if i > 0 and i % max(1, max_iters // 10) == 0:
            display_metrics = smoothed_metrics
            print(f"iter: {i} loss: {val}.")

        if i % 50 == 0:
            display_metrics = {k: v.item() for k, v in smoothed_metrics.items()}

            # make this an OrderedDict and make sure "loss" is the first element, followed by "mean_reprojection" and then the rest
            ordered_display_metrics = OrderedDict(
                sorted(display_metrics.items(), key=lambda x: x[0] not in ["total", "mean_3d_raw", "site_offset_raw"])
            )
            counter.set_postfix(ordered_display_metrics)

            if i % max(1, max_iters // 10) == 0:
                print(display_metrics)

    return model, metrics
