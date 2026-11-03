"""Differentiable forward kinematics of the MuJoCo skeleton (MJX), with body scaling.

Part of the variable-scale copy of the MonocularBiomechanics fitting code
(``metrics/bone_length/monocular_demos``; the rigid-fit original is
``baseline/monocular_demos``).  :class:`ForwardKinetics` wraps the ``humanoid_torque.xml``
skeleton (Rajagopal joint set): given joint angles, per-body scale factors and learned site
offsets it returns body positions and the positions of the keypoint sites that the fit
compares with the MeTRAbs keypoints.  The scale mixer (:func:`create_default_scale_mix`) maps a
few intuitive scales (thighs, legs, arms, pelvis, overall size) to every body.  Only
kinematics are needed for fitting; the dynamics (rollouts with actuators, contacts) are kept
from the upstream code.
"""
from pathlib import Path
import functools
from typing import List, Dict, Tuple
from jaxtyping import Integer, Float, Array

import jax
from jax import numpy as jnp
import equinox as eqx

import os
import mujoco
from mujoco import mjx
from mujoco.mjx import PyTreeNode
from mujoco.mjx._src.types import TrnType


class State(PyTreeNode):
    """
    State variables needed for forward kinematics

    This is a subset of the attributes available in `mjx.Data`, but including
    all of them in the full output dramatically slows things down, even if they
    are not used.

    Attributes:
        qpos: joint position                                          (nq,)
        qvel: joint velocity                                          (nv,)
        xpos:  Cartesian position of body frame                       (nbody, 3)
        site_xpos: Cartesian site position                            (nsite, 3)

    """

    qpos: jax.Array
    qvel: jax.Array
    xpos: jax.Array
    site_xpos: jax.Array
    # This field is always an array; MJX needs an array annotation for a leaf.
    efc_force: jax.Array


class ConstraintViolations(PyTreeNode):
    """
    Track violations of the equality constraints, joint limit constraints, and contact constraints

    Attributes:
        equality_violation: The sum of the absolute value of the equality constraints  (1,)
        limit_violations: The sum of the absolute value of the joint limit constraints (1,)
        contact_violations: The sum of the absolute value of the contact constraints   (1,)
    """

    equality_violation: jax.Array
    limit_violations: jax.Array
    friction_violations: jax.Array
    contact_violations: jax.Array


def create_default_scale_mix(body_names: List[str]) -> Float[Array, "bodies scales"]:
    """
    Create a matrix that maps from intuitive body scalings to transformations

    This is specific to the humanoid model

    Args:
        body_names: The body names.

    Returns:
        A vector of scales to apply to the the body part offsets
    """

    mappings = {
        "pelvis": ["pelvis"],
        "left_thigh": ["femur_l"],
        "left_leg": ["tibia_l", "talus_l", "calcn_l", "toes_l"],
        "right_thigh": ["femur_r"],
        "right_leg": ["tibia_r", "talus_r", "calcn_r", "toes_r"],
        "left_arm": ["humerus_l", "radius_l", "ulna_l", "hand_l"],
        "right_arm": ["humerus_r", "radius_r", "ulna_r", "hand_r"],
        "overall": body_names,
    }

    # add back in world for computing offsets
    body_names = ["world"] + body_names

    scale_mixer = jnp.zeros((len(body_names), len(mappings)))
    for i, (k, v) in enumerate(mappings.items()):
        for b in v:
            scale_mixer = scale_mixer.at[body_names.index(b), i].set(1.0)

    return scale_mixer


def create_custom_scale_mix(body_names: List[str], mappings: dict) -> Float[Array, "bodies scales"]:
    """
    Create a matrix that maps from intuitive body scalings to transformations

    This is specific to the humanoid model

    Args:
        body_names: The body names.

    Returns:
        A vector of scales to apply to the the body part offsets
    """
    # add back in world for computing offsets
    if "overall" not in mappings:
        mappings["overall"] = body_names

    body_names = ["world"] + body_names
    scale_mixer = jnp.zeros((len(body_names), len(mappings)))
    for i, (k, v) in enumerate(mappings.items()):
        for b in v:
            scale_mixer = scale_mixer.at[body_names.index(b), i].set(1.0)

    return scale_mixer


def make_body_pos_scale_mixer(mjx_model: mjx.Model) -> Float[Array, "bodies bodies"]:
    """
    Create a mixer matrix to scale the body origin positions

    If we are scaling, say, the pelvis, we really need to scale the origin
    position of each of the body elements that have it as their parent
    """

    mixer = jnp.zeros((mjx_model.nbody, mjx_model.nbody))

    idx0 = jnp.arange(mjx_model.nbody)
    idx1 = mjx_model.body_parentid

    # set all these elements to 1
    mixer = mixer.at[idx0, idx1].set(1.0)

    return mixer


def make_site_scale_mixer(mjx_model: mjx.Model) -> Float[Array, "sites bodies"]:
    """
    Create a mixer matrix to scale the site positions

    We provide a scaling matrix based on the body parts. Site positions
    are defined relative to this body part, so should also scale propotionally
    """

    mixer = jnp.zeros((mjx_model.nsite, mjx_model.nbody))

    idx0 = jnp.arange(mjx_model.nsite)
    idx1 = mjx_model.site_bodyid

    # set all these elements to 1
    mixer = mixer.at[idx0, idx1].set(1.0)

    return mixer


def make_geom_scale_mixer(mjx_model: mjx.Model) -> Float[Array, "geoms bodies"]:
    """
    Create a mixer matrix to scale the geom positions

    We provide a scaling matrix based on the body parts. Geom positions
    are defined relative to this body part, so should also scale propotionally
    """

    mixer = jnp.zeros((mjx_model.ngeom, mjx_model.nbody))

    idx0 = jnp.arange(mjx_model.ngeom)
    idx1 = mjx_model.geom_bodyid

    # set all these elements to 1
    mixer = mixer.at[idx0, idx1].set(1.0)

    return mixer


def make_vert_scale_mixer(mjx_model):
    """
    Create a mixer matrix to scale the vert positions

    We provide a scaling matrix based on the body parts. Vert positions
    are defined relative to this body part, so should also scale propotionally
    """

    mixer = jnp.zeros((mjx_model.nmeshvert, mjx_model.nbody))

    for i in jnp.arange(mjx_model.nbody):
        for g in jnp.arange(mjx_model.body_geomadr[i], mjx_model.body_geomadr[i] + mjx_model.body_geomnum[i]):
            mesh_id = mjx_model.geom_dataid[g]
            if mesh_id != -1:
                mixer = mixer.at[mjx_model.mesh_vertadr[mesh_id] : mjx_model.mesh_vertadr[mesh_id] + mjx_model.mesh_vertnum[mesh_id], i].set(1.0)

    return mixer


def scale_model(model: mjx.Model, scale: Float[Array, "bodies 1"] | Float) -> mjx.Model:
    """
    Scale the model by a constant or vector

    This works well enough for forward kinematics with mujoco, but
    needs some work to reliably rescale the whole body and inertia
    on the fly. This also attempts to handlemujoco models that are
    using mjx or not.

    Args:
        model: The mujoco model
        scale: The scale factor

    Returns:
        The scaled model
    """

    if isinstance(scale, float):
        scale = jnp.ones((model.nbody, 1)) * scale

    body_mixer = make_body_pos_scale_mixer(model)
    body_scale = body_mixer @ scale

    site_mixer = make_site_scale_mixer(model)
    site_scale = site_mixer @ scale

    geom_mixer = make_geom_scale_mixer(model)
    geom_scale = geom_mixer @ scale

    if type(model) == mujoco._structs.MjModel:
        from copy import deepcopy

        vert_mixer = make_vert_scale_mixer(model)
        vert_scale = vert_mixer @ scale

        # deep copy model
        model = deepcopy(model)
        model.body_pos = model.body_pos * body_scale
        model.site_pos = model.site_pos * site_scale
        model.geom_pos = model.geom_pos * geom_scale
        model.geom_size = model.geom_size * geom_scale
        model.mesh_vert = model.mesh_vert * vert_scale

        if model.ntendon > 0:
            muscle_actuated = model.actuator_trntype == TrnType.TENDON
            data = mujoco.MjData(model)
            data.qpos = model.qpos0
            mujoco.mj_forward(model, data)
            model.actuator_length0 = data.actuator_length
            model.tendon_length0 = data.actuator_length[muscle_actuated]

        return model

    elif type(model) == mujoco.mjx._src.types.Model:
        mjx_model = model

        # NOTE: do not use vertices for MJX. make_vert_scale_mixer is not jittable either.
        # NOTE: it is important to prevent any gradients altering the base model
        mjx_model = mjx_model.replace(
            body_pos=mjx_model.body_pos * body_scale,
            site_pos=mjx_model.site_pos * site_scale,
            geom_pos=mjx_model.geom_pos * geom_scale,
            geom_size=mjx_model.geom_size * geom_scale,
        )

        if model.ntendon > 0:
            muscle_actuated = model.actuator_trntype == TrnType.TENDON
            mjx_data = mjx.make_data(mjx_model).replace(qpos=mjx_model.qpos0)
            mjx_data = mjx.forward(mjx_model, mjx_data)
            mjx_model = mjx_model.replace(actuator_length0=mjx_data.actuator_length, tendon_length0=mjx_data.tendon_length[muscle_actuated])

            # NOTE: we are not scaling the actuator_acc0 and tendon_invweight0 because it is hard and I don't want to.

        return mjx_model

    # brax.base.System, matched by name so that brax is not a dependency
    elif type(model).__name__ == "System" and model.__module__ == "brax.base":

        # NOTE: the embedded mj_model is not scaled; it should not be used afterwards

        sys = model.tree_replace(
            {
                "body_pos": model.body_pos * body_scale,
                "site_pos": model.site_pos * site_scale,
                "geom_pos": model.geom_pos * geom_scale,
                "geom_size": model.geom_size * geom_scale,
            }
        )

        if sys.ntendon > 0:
            muscle_actuated = model.actuator_trntype == TrnType.TENDON
            mjx_data = mjx.make_data(sys).replace(qpos=model.qpos0)
            mjx_data = mjx.forward(sys, mjx_data)
            sys = sys.tree_replace(
                {
                    # brax systems have no actuator_length0
                    "tendon_length0": mjx_data.actuator_length[muscle_actuated],
                }
            )
        return sys

    else:
        raise NotImplementedError(f"Unable to handle data type {type(model)}")


def scale_muscle(model: mjx.Model, scale: Float[Array, "bodies 1"] | Float) -> mjx.Model:
    """
    Scale the muscle strength by a constant or vector

    Args:
        model: The mujoco model
        scale: The scale factor

    Returns:
        The scaled model
    """

    if type(model) == mujoco._structs.MjModel:
        from copy import deepcopy

        # deep copy model
        model = deepcopy(model)
        model.actuator_gainprm[:, 2] = model.actuator_gainprm[:, 2] * scale

        return model

    elif type(model) == mujoco.mjx._src.types.Model:
        mjx_model = model

        actuator_gainprm = model.actuator_gainprm.at[:, 2].set(model.actuator_gainprm[:, 2] * scale)

        mjx_model = mjx_model.replace(actuator_gainprm=actuator_gainprm)

        return mjx_model

    # brax.base.System, matched by name so that brax is not a dependency
    elif type(model).__name__ == "System" and model.__module__ == "brax.base":

        actuator_gainprm = model.actuator_gainprm
        if isinstance(actuator_gainprm, jnp.ndarray):
            actuator_gainprm = actuator_gainprm.at[:, 2].set(actuator_gainprm[:, 2] * scale)
        else:
            actuator_gainprm[:, 2] = actuator_gainprm[:, 2] * scale

        sys = model.tree_replace({"actuator_gainprm": actuator_gainprm})

        return sys

    else:
        raise NotImplementedError(f"Unable to handle data type {type(model)}")


def offset_sites(model: mjx.Model, offsets: Float[Array, "sites 3"]) -> mjx.Model:
    """
    Apply offsets to the sites

    Args:
        model: The mujoco model
        offsets: The scale factor

    Returns:
        The scaled model
    """

    if type(model) == mujoco.MjModel:
        from copy import deepcopy

        # deep copy model
        model = deepcopy(model)
        model.site_pos = model.site_pos + offsets
        return model

    elif type(model) == mjx.Model:
        mjx_model = model
        mjx_model = mjx_model.replace(
            site_pos=mjx_model.site_pos + offsets,
        )
        return mjx_model

    else:
        raise NotImplementedError(f"Unable to handle data type {type(model)}")


def set_margin(model: mjx.Model, margin: Float | None) -> mjx.Model:
    """Set the MJX model geometry margins if not None"""

    if margin is None:
        return model

    if type(model) == mujoco.MjModel:
        import numpy as np
        from copy import deepcopy

        # deep copy model
        model = deepcopy(model)

        model.geom_margin = np.ones_like(model.geom_margin) * margin
        return model

    elif type(model) == mjx.Model:
        geom_margin = jnp.ones_like(model.geom_margin) * margin
        model = model.replace(geom_margin=geom_margin)
        return model

    else:
        raise NotImplementedError(f"Unable to handle data type {type(model)}")


def shift_geom_vertically(model: mjx.Model, geom_idx: Array, vertical_displacement: Float | None) -> mjx.Model:
    """Set the MJX model geometry margins if not None"""

    if vertical_displacement is None:
        return model

    if type(model) == mujoco.MjModel:
        import numpy as np
        from copy import deepcopy

        # deep copy model
        model = deepcopy(model)

        model.geom_pos[geom_idx, 1] += vertical_displacement
        return model

    elif type(model) == mjx.Model:
        geom_pos = model.geom_pos
        geom_pos = geom_pos.at[geom_idx, 1].set(geom_pos[geom_idx, 1] + vertical_displacement)
        model = model.replace(geom_pos=geom_pos)
        return model

    # brax.base.System, matched by name so that brax is not a dependency
    elif type(model).__name__ == "System" and model.__module__ == "brax.base":
        geom_pos = model.geom_pos
        geom_pos = geom_pos.at[geom_idx, 1].set(geom_pos[geom_idx, 1] + vertical_displacement)
        model = model.tree_replace({"geom_pos": geom_pos})
        return model

    else:
        raise NotImplementedError(f"Unable to handle data type {type(model)}")


def set_timestep(model: mjx.Model, timestep: Float) -> mjx.Model:
    """Set the MJX model timestep"""

    if type(model) == mujoco.MjModel:
        model.opt.timestep = timestep
        return model

    elif type(model) == mjx.Model:
        model = model.replace(opt=model.opt.replace(timestep=timestep))
        return model

    else:
        raise NotImplementedError(f"Unable to handle data type {type(model)}")

    return model


# Dictionary to store cached models and their modification times
model_cache: Dict[str, tuple[float, mujoco.MjModel]] = {}


def get_default_model_path(model_name: str = "humanoid_torque.xml") -> str:
    """
    Get the default model path

    Returns:
        The path to the default model
    """

    return (Path(__file__).resolve().parent / "data" / "humanoid" / model_name).as_posix()


def load_default_model(xml_path: str = None) -> mujoco.MjModel:
    """
    Load the default model

    This keeps a local cache of models that have been opened to work around a bug in
    mujoco where it is not properly closing all the file handles from prior opens.

    Args:
        xml_path: The path to the model xml

    Returns:
        The mujoco model
    """

    if xml_path is None:
        xml_path = (Path(__file__).resolve().parent / "data" / "humanoid" / "humanoid_torque.xml").as_posix()

    # Get the last modified time of the XML file
    current_mod_time = os.path.getmtime(xml_path)

    # Check if the model is cached and the file has not been modified
    if xml_path in model_cache and model_cache[xml_path][0] == current_mod_time:
        return model_cache[xml_path][1]

    # Load the model from XML, cache it with the current modification time
    model = mujoco.MjModel.from_xml_path(xml_path)
    model_cache[xml_path] = (current_mod_time, model)

    return model


def reorder_sites(mjx_data: mjx.Data, site_reorder: Integer[Array, "sites"] | None) -> mjx.Data:
    """``mjx_data`` with its sites in ``site_reorder`` order (unchanged for ``None``)."""
    if site_reorder is None:
        return mjx_data
    else:
        site_xpos = mjx_data.site_xpos[site_reorder]
        return mjx_data.replace(site_xpos=site_xpos)


@functools.partial(jax.jit, static_argnames=("return_contact_forces"))
def reduce_state(mjx_data: mjx.Data, return_contact_forces: bool = False) -> State:
    """
    Reduce the mjx data to a state object

    Empirically, this seems to improve performance over returning the full mjx data
    and then only using a subset.

    Args:
        mjx_data: The mujoco data
        return_contact_forces: Return the contact forces or not (default no)

    Returns:
        The state object
    """
    return State(
        qpos=mjx_data.qpos,
        qvel=mjx_data.qvel,
        xpos=mjx_data.xpos,
        site_xpos=mjx_data.site_xpos,
        efc_force=jnp.where(return_contact_forces, mjx_data.efc_force, jnp.zeros(())),
    )


@functools.partial(jax.jit, static_argnames=("return_contact_forces", "check_constraints"))
def forward_kinematics(
    mjx_model, joint_angles, scale, site_reorder=None, site_offsets=None | Array, return_contact_forces=False, check_constraints=False
):
    """State (body and site positions) of the scaled model at ``joint_angles``.

    With ``check_constraints`` also returns the :class:`ConstraintViolations` of that pose.
    """

    scaled_model = scale_model(mjx_model, scale)
    if site_offsets is not None:
        scaled_model = offset_sites(scaled_model, site_offsets)
    mjx_data = mjx.make_data(scaled_model)

    # Match the full forward geometry while omitting unused force/contact work.
    forward = functools.partial(mjx.forward if (check_constraints or return_contact_forces) else mjx.kinematics, scaled_model)
    qpos = mjx_data.qpos.at[:].set(joint_angles)
    mjx_data = mjx_data.replace(qpos=qpos)
    mjx_data = forward(mjx_data)

    mjx_data = reorder_sites(mjx_data, site_reorder)
    state = reduce_state(mjx_data, return_contact_forces)

    if check_constraints:
        ne = mjx_data.ne  # number of equality constraints
        nf = mjx_data.nf  # number of friction constraints
        nl = mjx_data.nl  # number of limit constraints
        nc = mjx_data.ncon  # number of contact constraint counts

        dx = mjx.make_constraint(scaled_model, mjx_data)
        equality_violation = jax.lax.cond(ne > 0, lambda: jnp.mean(dx.efc_aref[:ne] ** 2), lambda: jnp.array(0.0, dx.efc_aref.dtype))
        friction_violations = jax.lax.cond(nf > 0, lambda: jnp.mean(dx.efc_aref[ne : ne + nf] ** 2), lambda: jnp.array(0.0, dx.efc_aref.dtype))
        limit_violations = jax.lax.cond(nl > 0, lambda: jnp.mean(dx.efc_aref[ne + nf : ne + nf + nl] ** 2), lambda: jnp.array(0.0, dx.efc_aref.dtype))
        contact_violations = jax.lax.cond(
            nc > 0, lambda: jnp.mean(dx.efc_aref[ne + nf + nl : ne + nf + nl + nc] ** 2), lambda: jnp.array(0.0, dx.efc_aref.dtype)
        )

        constraints = ConstraintViolations(
            equality_violation=equality_violation,
            friction_violations=friction_violations,
            limit_violations=limit_violations,
            contact_violations=contact_violations,
        )

        return state, constraints

    else:
        return state


def kinetic_step(model, mjx_data, action, action_repeats: int = 8, site_reorder: Integer[Array, "sites"] | None = None):
    """Simulate ``action_repeats`` MuJoCo steps with the actuator controls fixed to ``action``."""

    def step_fn(carry, _):
        mjx_data = carry
        ctrl = mjx_data.ctrl.at[:].set(action)
        mjx_data = mjx_data.replace(ctrl=ctrl)
        mjx_data = mjx.step(model, mjx_data)

        return mjx_data, _

    mjx_data, _ = jax.lax.scan(step_fn, mjx_data, xs=None, length=action_repeats)
    return mjx_data


@functools.partial(jax.jit, static_argnames=("return_contact_forces", "check_constraints", "action_repeats"))
def forward_kinetics(
    mjx_model,
    joint_angles,
    joint_velocities,
    action,
    scale,
    site_reorder=None,
    site_offsets=None,
    return_contact_forces: bool = False,
    check_constraints: bool = False,
    action_repeats: int = 8,
    margin: Float | None = None,
    timestep: Float | None = None,
):
    """State of the scaled model at ``joint_angles``/``joint_velocities``, optionally rolled forward.

    Without ``action`` (and without constraint checks or contact forces) only kinematics are
    evaluated, which is all the keypoint fit needs.  With ``action`` (steps, actuators) the
    model is simulated through the actions (:func:`kinetic_step`).
    """

    scaled_model = scale_model(mjx_model, scale)
    if site_offsets is not None:
        scaled_model = offset_sites(scaled_model, site_offsets)
    if timestep is not None:
        scaled_model = set_timestep(scaled_model, timestep)
    scaled_model = set_margin(scaled_model, margin)
    mjx_data = mjx.make_data(scaled_model)

    qpos = mjx_data.qpos.at[:].set(joint_angles)
    mjx_data = mjx_data.replace(qpos=qpos)

    if joint_velocities is not None:
        qvel = mjx_data.qvel.at[:].set(joint_velocities)
        mjx_data = mjx_data.replace(qvel=qvel)

    geometry_only = action is None and not check_constraints and not return_contact_forces
    initial_state = mjx.kinematics(scaled_model, mjx_data) if geometry_only else mjx.forward(scaled_model, mjx_data)

    if site_reorder is None:
        site_reorder = jnp.arange(mjx_data.site_xpos.shape[0])

    initial_state_reordered = reorder_sites(initial_state, site_reorder)
    state = reduce_state(initial_state_reordered, return_contact_forces)

    if check_constraints:
        ne = mjx_data.ne  # number of equality constraints
        nf = mjx_data.nf  # number of friction constraints
        nl = mjx_data.nl  # number of limit constraints
        nc = mjx_data.ncon  # number of contact constraint counts

        dx = mjx.make_constraint(scaled_model, initial_state)
        equality_violation = jax.lax.cond(ne > 0, lambda: jnp.mean(dx.efc_aref[:ne] ** 2), lambda: jnp.array(0.0, dx.efc_aref.dtype))
        friction_violations = jax.lax.cond(nf > 0, lambda: jnp.mean(dx.efc_aref[ne : ne + nf] ** 2), lambda: jnp.array(0.0, dx.efc_aref.dtype))
        limit_violations = jax.lax.cond(nl > 0, lambda: jnp.mean(dx.efc_aref[ne + nf : ne + nf + nl] ** 2), lambda: jnp.array(0.0, dx.efc_aref.dtype))
        contact_violations = jax.lax.cond(
            nc > 0, lambda: jnp.mean(dx.efc_aref[ne + nf + nl : ne + nf + nl + nc] ** 2), lambda: jnp.array(0.0, dx.efc_aref.dtype)
        )

        constraints = ConstraintViolations(
            equality_violation=equality_violation,
            friction_violations=friction_violations,
            limit_violations=limit_violations,
            contact_violations=contact_violations,
        )

    else:
        constraints = None

    # NOTE: we need to stop gradient on model before propagating through kinetics
    # as right now the gradients seem to become stable and prevent the side/offsets
    # continuing to learn. This is related to how the camera gradients get blocked when
    # included in these reprojection errors.
    scaled_model = jax.lax.stop_gradient(scaled_model)

    if action is not None:

        if len(action.shape) == 1:
            action = jnp.expand_dims(action, 0)

        def scaled_step(carry, action):
            step, mjx_data = carry

            mjx_data = kinetic_step(scaled_model, mjx_data, action, action_repeats=action_repeats, site_reorder=site_reorder)

            mjx_data_reordered = reorder_sites(mjx_data, site_reorder)
            output_state = reduce_state(mjx_data_reordered, return_contact_forces)

            return (step + 1, mjx_data), output_state

        next_states = jax.lax.scan(scaled_step, (0, initial_state), action)[1]

    else:
        next_states = None

    return state, constraints, next_states


def get_qpos_dof(joint_type: int) -> int:
    """Get qpos elements for joint types"""
    if joint_type == mujoco.mjtJoint.mjJNT_FREE:
        return 7  # vel component is 6
    elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
        return 4
    elif joint_type == mujoco.mjtJoint.mjJNT_SLIDE or joint_type == mujoco.mjtJoint.mjJNT_HINGE:
        return 1
    else:
        raise ValueError(f"Unsupported joint type: {joint_type}")


def get_dof_qpos_jntid(mjx_model: mjx.Model) -> List[int]:
    """
    Get lookup from qpos index to joint index

    This makes it easier to handle multiple DOF joints. The model already contains
    a field dof_jntid, which gives the lookup into the joint index for velocity terms
    but for quaternions there is an extra DOF.

    Args:
        mjx_model: The mujoco model

    Returns:
        List of joint indices for each qpos element
    """

    return [addr for addr, jt in enumerate(mjx_model.jnt_type) for _ in range(get_qpos_dof(jt))]


def angular_velocity_to_quaternion_derivative(q: Float[Array, "... 4"], omega: Float[Array, "... 3"]) -> Float[Array, "... 4"]:
    """Convert angular velocity to quaternion derivative"""

    # Ensure q and omega have compatible shapes for broadcasting
    if q.shape[-1] != 4:
        raise ValueError("The last dimension of q must be 4 (representing a quaternion).")
    if omega.shape[-1] != 3:
        raise ValueError("The last dimension of omega must be 3 (representing an angular velocity vector).")

    # Extract quaternion components
    q0, q1, q2, q3 = jnp.split(q, 4, axis=-1)
    omega_x, omega_y, omega_z = jnp.split(omega, 3, axis=-1)

    # Compute the quaternion derivative
    q_dot_0 = -0.5 * (q1 * omega_x + q2 * omega_y + q3 * omega_z)
    q_dot_1 = 0.5 * (q0 * omega_x + q2 * omega_z - q3 * omega_y)
    q_dot_2 = 0.5 * (q0 * omega_y + q3 * omega_x - q1 * omega_z)
    q_dot_3 = 0.5 * (q0 * omega_z + q1 * omega_y - q2 * omega_x)

    # Concatenate the results along the last dimension
    q_dot = jnp.concatenate([q_dot_0, q_dot_1, q_dot_2, q_dot_3], axis=-1)

    return q_dot


def quat_to_angular_velocity(q: Float[Array, "4"], q_dot: Float[Array, "4"]) -> Float[Array, "3"]:
    """Convert quaternion to angular velocity"""
    # Ensure q is a unit quaternion
    q = q / (jnp.linalg.norm(q) + 1e-9)

    # Compute the inverse (conjugate) of q
    q_inv = jnp.array([q[0], -q[1], -q[2], -q[3]])

    # Perform quaternion multiplication
    result = jnp.array(
        [
            q_dot[0] * q_inv[0] - q_dot[1] * q_inv[1] - q_dot[2] * q_inv[2] - q_dot[3] * q_inv[3],
            q_dot[0] * q_inv[1] + q_dot[1] * q_inv[0] + q_dot[2] * q_inv[3] - q_dot[3] * q_inv[2],
            q_dot[0] * q_inv[2] - q_dot[1] * q_inv[3] + q_dot[2] * q_inv[0] + q_dot[3] * q_inv[1],
            q_dot[0] * q_inv[3] + q_dot[1] * q_inv[2] - q_dot[2] * q_inv[1] + q_dot[3] * q_inv[0],
        ]
    )

    # Extract the vector part and multiply by 2
    omega = 2 * result[1:]

    return omega


def normalize_velocity_from_quaternions(qpos: Float[Array, "joints"], qdot: Float[Array, "joints"], freeroot: bool) -> Float[Array, "joints - 1"]:
    """
    Convert derivatives through implicites to MJX velocities

    For bodies with a freeroot, the quaternion part of the pose has four elements
    but the velocity elements has three elements. However, this isn't captured by
    our implicit representations where we simply compute the derivative through the
    network.

    This uses the standard formula to compute the rotation 3-vector and then substitutes
    it in and returns the final velocity.

    Args:
        qpos: The joint positions
        qdot: The joint velocities

    Returns:
        The normalized velocities
    """

    if not freeroot:
        return qdot

    _q = qpos[3:7]
    _qdot = qdot[3:7]
    omega = quat_to_angular_velocity(_q, _qdot)
    qvel = jnp.concatenate([qdot[:3], omega, qdot[7:]])
    return qvel


class ForwardKinematicsModelDescriptor(eqx.Module):
    """Names and paths of a :class:`ForwardKinematics` model without the model arrays."""

    xml_path: str
    body_names: List[str]
    joint_names: List[str]
    joints_and_dof: List[Tuple[str, int]]
    site_names: List[str]
    actuator_names: List[str]
    site_reorder: Integer[Array, "sites"]

    def __init__(self, xml_path, body_names, joint_names, joints_and_dof, site_names, actuator_names, site_reorder):
        self.xml_path = xml_path
        self.body_names = body_names
        self.joint_names = joint_names
        self.joints_and_dof = joints_and_dof
        self.site_names = site_names
        self.actuator_names = actuator_names
        self.site_reorder = site_reorder


class ForwardKinematics(eqx.Module):
    """
    Wrap a Mujoco MJX model with support for scaling and site offsets
    """

    mjx_model: mjx.Model
    xml_path: str | None = eqx.field(static=True)
    model: mujoco.MjModel = eqx.field(static=True)
    body_names: List[str] = eqx.field(static=True)
    joint_names: List[str] = eqx.field(static=True)
    geom_names: List[str] = eqx.field(static=True)
    joints_and_dof: List[Tuple[str, int]] = eqx.field(static=True)
    _dof_qpos_jntid: List[int] = eqx.field(static=True)
    raw_site_names: List[str] = eqx.field(static=True)
    site_names: List[str] = eqx.field(static=True)
    actuator_names: List[str] = eqx.field(static=True)
    site_reorder: Integer[Array, "sites"]
    model_stop_gradient: bool = eqx.field(static=True, default=True)

    def __init__(self, xml_path=None, model=None, site_reorder: List[str] | Integer[Array, "sites"] | None = None, model_stop_gradient: bool = True):

        self.xml_path = xml_path or get_default_model_path()
        model = load_default_model(self.xml_path)
        self.model = model

        self.mjx_model = mjx.put_model(model)

        self.body_names = []
        for i in range(model.nbody):
            self.body_names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i))
        self.body_names = self.body_names[1:]  # drop "world" reference frame

        self._dof_qpos_jntid = get_dof_qpos_jntid(model)

        self.joint_names = []
        self.joints_and_dof = []
        dof_count = 0
        for i in range(model.njnt):
            dof = get_qpos_dof(model.jnt_type[i])

            if dof > 1:
                for j in range(dof):
                    self.joint_names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) + f"_{j}")
            else:
                self.joint_names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i))

            self.joints_and_dof.append((mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i), dof))
            dof_count += dof

        if dof_count != model.nq:
            raise ValueError(f"The total DOF ({dof_count}) does not match the model's nv ({model.nq}).")

        self.raw_site_names = []
        for i in range(model.nsite):
            self.raw_site_names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i))

        self.geom_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(model.ngeom)]

        if site_reorder is not None and isinstance(site_reorder, List) and isinstance(site_reorder[0], str):
            self.site_reorder = jnp.array([self.raw_site_names.index(s) for s in site_reorder])
        elif site_reorder is not None and (
            (isinstance(site_reorder, List) and isinstance(site_reorder[0], int)) or isinstance(site_reorder, jnp.ndarray)
        ):
            self.site_reorder = jnp.array(site_reorder)
        else:
            self.site_reorder = jnp.arange(model.nsite)

        self.site_names = []
        for i in self.site_reorder:
            self.site_names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i))

        self.actuator_names = []
        for i in range(model.nu):
            self.actuator_names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i))

        self.model_stop_gradient = model_stop_gradient

    def build_default_scale_mixer(self):
        """(bodies, scales) matrix of :func:`create_default_scale_mix` for this model."""
        return create_default_scale_mix(self.body_names)

    def build_custom_scale_mixer(self, mappings: Dict[str, List[str]]):
        """(bodies, scales) matrix from ``{scale name: [body names]}``."""
        return create_custom_scale_mix(self.body_names, mappings)

    def __call__(
        self,
        joint_angles: Float[Array, "joints"],
        scale: Float[Array, "scales"],
        site_offsets: Float[Array, "sites 3"] | None = None,
        mjx_model: mjx.Model | None = None,
        check_constraints: bool = True,
        return_contact_forces: bool = False,
    ):
        """
        Returns the joint end positions given joint angles

        Args:
            joint_angles: array of joint angles in radians
            scale: scale factor or array to apply to the model
            site_offsets: offsets to apply to the sites, or none to skip
            mjx_model: optionally override the default model
            check_constraints: check for constraint violations
            return_contact_forces: return the contact forces

        Returns:
            joint_end_positions: array of joint end positions in meters
        """

        mjx_model = mjx_model if mjx_model is not None else self.mjx_model
        mjx_model = jax.lax.cond(self.model_stop_gradient, lambda: jax.lax.stop_gradient(mjx_model), lambda: mjx_model)

        return forward_kinematics(
            mjx_model,
            joint_angles,
            scale,
            self.site_reorder,
            site_offsets,
            return_contact_forces,
            check_constraints,
        )

    @property
    def dof_qpos_jntid(self):
        """Joint id of every qpos entry, as a JAX array."""
        # convenient to have this as a jnp array when accessing, but dont want to
        # store it as this or it upsets the comparison between static fields
        return jnp.array(self._dof_qpos_jntid)

    def get_descriptor(self):
        """:class:`ForwardKinematicsModelDescriptor` of this model."""
        # return a dataclass containing the xml path and all the information we got about the model,
        # but drop the actual model. this is useful for speeding up compute time, as jitting the call
        # and caching that produces more streamlined objects that result in faster code.

        return ForwardKinematicsModelDescriptor(
            xml_path=self.xml_path,
            body_names=self.body_names,
            joint_names=self.joint_names,
            joints_and_dof=self.joints_and_dof,
            site_names=self.site_names,
            actuator_names=self.actuator_names,
            site_reorder=self.site_reorder,
        )


class ForwardKinetics(ForwardKinematics):
    """:class:`ForwardKinematics` that can also roll the model forward under actuator actions."""

    def __call__(
        self,
        joint_angles: Float[Array, "joints"],
        scale: Float[Array, "scales"],
        joint_velocities: Float[Array, "joints"] | None = None,
        action: Float[Array, "steps actuators"] | None = None,
        site_offsets: Float[Array, "sites 3"] | None = None,
        mjx_model: mjx.Model | None = None,
        return_contact_forces: bool = False,
        check_constraints: bool = True,
        action_repeats: int = 8,
        margin: Float | None = None,
        timestep: Float | None = None,
    ):
        """
        Returns the joint end positions given joint angles

        Args:
            joint_angles: array of joint angles in radians
            joint_velocities: array of joint velocities in radians / s
            action: array of actuator forces, can include multiple steps to perform rollouts
            scale: scale factor or array to apply to the model
            site_offsets: offsets to apply to the sites, or none to skip
            return_contact_forces: return the contact forces (default False)
            check_constraints: check for constraint violations (default True)
            action_repeats: number of steps to subsample the action (default 8)
            margin: margin to apply to the geometry, or none to leave unchanged
            timestep: timestep to apply to the model, or none to leave unchanged

        Returns:
            joint_end_positions: array of joint end positions in meters
        """

        mjx_model = mjx_model if mjx_model is not None else self.mjx_model
        mjx_model = jax.lax.cond(self.model_stop_gradient, lambda: jax.lax.stop_gradient(mjx_model), lambda: mjx_model)

        return forward_kinetics(
            mjx_model,
            joint_angles,
            joint_velocities,
            action,
            scale=scale,
            site_reorder=self.site_reorder,
            site_offsets=site_offsets,
            return_contact_forces=return_contact_forces,
            check_constraints=check_constraints,
            action_repeats=action_repeats,
            margin=margin,
            timestep=timestep,
        )


def floor_loss(state: State, constraints: ConstraintViolations, foot_body_offset: float = 0.01):
    """
    Compute loss for tracking floor position accurately.

    Parameters:
        state: State, which contains the body positions
        constraints: ConstraintViolations, including floor violations
    """

    # take the lowest z position for each of the body elements at each time, will likely
    # be toes or calcaneus. skip the first body as that is the floor plane.
    lowest_body = jnp.min(state.xpos[:, 1:, 2], axis=1)

    # the geometries cannot actual hit zero without penetrating the floor. the floor penetration
    # loss prevents this, but an offset further encouages a good local optimum
    lowest_body = lowest_body - foot_body_offset

    lowest_body = jnp.mean(jnp.abs(lowest_body))

    # then get the contact violations, which with the model setup we are using only
    # comes from colliders passing through the floor. this sometimes hits extreme values
    # so we use a sqrt transform and further clip each time point
    floor_violation = jnp.sqrt(constraints.contact_violations)
    floor_violation = jnp.clip(floor_violation, 0.0, 100.0)
    floor_violation = jnp.mean(constraints.contact_violations)

    return lowest_body, floor_violation


# fmt: off

# MeTRAbs bml_movi_87 keypoint names in MeTRAbs order (the skeleton's site names).
movi_joint_names = ['backneck', 'upperback', 'clavicle', 'sternum', 'umbilicus', 'lfronthead', 'lbackhead', 'lback',
 'lshom', 'lupperarm', 'lelbm', 'lforearm', 'lwrithumbside', 'lwripinkieside', 'lfin', 'lasis', 'lpsis', 'lfrontthigh', 'lthigh',
 'lknem', 'lankm', 'LHeel', 'lfifthmetatarsal', 'LBigToe', 'lcheek', 'lbreast', 'lelbinner', 'lwaist', 'lthumb', 'lfrontinnerthigh',
 'linnerknee', 'lshin', 'lfirstmetatarsal', 'lfourthtoe', 'lscapula', 'lbum', 'rfronthead', 'rbackhead', 'rback', 'rshom', 'rupperarm',
 'relbm', 'rforearm', 'rwrithumbside', 'rwripinkieside', 'rfin', 'rasis', 'rpsis', 'rfrontthigh', 'rthigh', 'rknem', 'rankm', 'RHeel',
 'rfifthmetatarsal', 'RBigToe', 'rcheek', 'rbreast', 'relbinner', 'rwaist', 'rthumb', 'rfrontinnerthigh', 'rinnerknee', 'rshin', 'rfirstmetatarsal',
 'rfourthtoe', 'rscapula', 'rbum', 'Head', 'mhip', 'CHip', 'Neck', 'LAnkle', 'LElbow', 'LHip', 'LHand', 'LKnee', 'LShoulder', 'LWrist', 'LFoot',
 'RAnkle', 'RElbow', 'RHip', 'RHand', 'RKnee', 'RShoulder', 'RWrist', 'RFoot']

def get_joint_names(with_hands=False) -> List[str]:
    """
    Get the joint names for the MOVI model used for reconstruction
    """

    joint_names = movi_joint_names

    if with_hands:

        # fmt: off
        hand_names = ['cmc1', 'mcp1', 'ip1', 'tip1', 'mcp2', 'pip2', 'dip2', 'tip2', 'mcp3', 'pip3', 'dip3', 'tip3', 'mcp4', 'pip4', 'dip4', 'tip4', 'mcp5', 'pip5', 'dip5', 'tip5']
        # fmt: on

        left_idx = 95
        right_idx = 116

        halpe_keypoint_idx = []
        halpe_keypoint_names = []
        for i, name in enumerate(hand_names):
            halpe_keypoint_names.append(name + "_l")
        for i, name in enumerate(hand_names):
            halpe_keypoint_names.append(name + "_r")

        return joint_names + halpe_keypoint_names

    return joint_names
# fmt: on
