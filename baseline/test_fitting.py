"""Optimizer state regression; run in the biomechanics environment."""
import importlib.util
import unittest


HAS_FITTING = all(importlib.util.find_spec(name) is not None for name in ("jax", "equinox", "optax", "mujoco"))


@unittest.skipUnless(HAS_FITTING, "requires the biomechanics environment")
class FittingOptimizerTests(unittest.TestCase):
    def test_export_compilation_reuses_dynamic_fitted_weights(self):
        import importlib
        import equinox as eqx
        import jax
        import jax.numpy as jnp
        import numpy as np

        for package in ("baseline", "metrics.bone_length"):
            traces = []

            class Model(eqx.Module):
                value: jax.Array

                def __call__(self, times, **options):
                    traces.append(True)
                    poses = self.value * times[:, None]
                    return poses, jnp.ones_like(poses) * self.value, jnp.empty((len(times), 0)), jnp.empty((len(times), 0))

            module = importlib.import_module(package + ".monocular_demos.biomechanics_mjx.monocular_trajectory")
            with self.subTest(package=package):
                times = jnp.arange(3, dtype=jnp.float32)
                a = module.export_trajectory(Model(jnp.array(2., dtype=jnp.float32)), times)
                b = module.export_trajectory(Model(jnp.array(3., dtype=jnp.float32)), times)
                np.testing.assert_array_equal(a[0], 2 * np.arange(3)[:, None])
                np.testing.assert_array_equal(b[0], 3 * np.arange(3)[:, None])
                self.assertEqual(len(traces), 1)

    def test_device_commitment_prevents_second_trace_without_changing_values(self):
        import importlib
        import jax
        import jax.numpy as jnp
        import numpy as np

        device = jax.devices()[0]
        batch = jax.device_put(jnp.array([2., 3.]), device)
        for package in ("baseline", "metrics.bone_length"):
            with self.subTest(package=package):
                module = importlib.import_module(package + ".monocular_demos.biomechanics_mjx.monocular_trajectory")
                traces = []

                @jax.jit
                def update(parameter, state, batch):
                    traces.append(True)
                    return parameter + batch, state + 1

                parameters = jnp.array([1., -1.])
                state = jnp.array(0, dtype=jnp.int32)
                committed_parameters, committed_state = module.commit_arrays_to_device((parameters, state), device)
                np.testing.assert_array_equal(committed_parameters, parameters)
                np.testing.assert_array_equal(committed_state, state)
                self.assertTrue(committed_parameters.committed)
                for _ in range(3):
                    committed_parameters, committed_state = update(committed_parameters, committed_state, batch)
                self.assertEqual(len(traces), 1)
                np.testing.assert_array_equal(committed_parameters, parameters + 3 * batch)
                self.assertEqual(int(committed_state), 3)

    def test_logging_callback_interval_preserves_training_updates(self):
        import contextlib
        import importlib
        import io
        from unittest.mock import patch
        import equinox as eqx
        import jax
        import jax.numpy as jnp
        import numpy as np

        class Model(eqx.Module):
            value: jax.Array
            scale_mixer: jax.Array

        class Dataset:
            phone_attitude = None
            camera_params = {}

            def __len__(self):
                return 1

            def __getitem__(self, indices):
                return jnp.array(1.), jnp.array(2.)

        class Progress:
            def __init__(self, count):
                self.count = count

            def __iter__(self):
                return iter(range(self.count))

            def set_postfix(self, metrics):
                pass

        def objective(model, x, y, **kwargs):
            value = (model.value * x - y) ** 2
            return value, {"total": jnp.atleast_1d(value)}

        for package in ("baseline", "metrics.bone_length"):
            module = importlib.import_module(package + ".monocular_demos.biomechanics_mjx.monocular_trajectory")
            with self.subTest(package=package), patch.object(module, "loss", objective), \
                    patch.object(module, "trange", Progress), contextlib.redirect_stdout(io.StringIO()):
                counts = []
                model = Model(jnp.array(1.), jnp.zeros((1, 1)))
                outputs = []
                for interval in (1, 2):
                    calls = []
                    fitted, _ = module.fit_model(model, Dataset(), max_iters=4,
                        step_callback=lambda i, *a, **kw: calls.append(i), callback_interval=interval)
                    outputs.append(fitted.value)
                    counts.append(calls)
                self.assertEqual(counts, [[0, 1, 2, 3], [0, 2]])
                np.testing.assert_array_equal(outputs[0], outputs[1])
                with self.assertRaisesRegex(ValueError, "callback_interval"):
                    module.fit_model(model, Dataset(), max_iters=4, callback_interval=0)

    def test_boolean_masks_do_not_change_optimizer_structure_or_updates(self):
        import equinox as eqx
        import jax
        import jax.numpy as jnp
        import numpy as np
        import optax
        from baseline.monocular_demos.biomechanics_mjx.monocular_trajectory import step
        from metrics.bone_length.monocular_demos.biomechanics_mjx.monocular_trajectory import step as bone_step

        class MaskedModel(eqx.Module):
            value: jax.Array
            mask: jax.Array

            def __call__(self, x):
                return jnp.where(self.mask, self.value * x, 0.0)

        model = MaskedModel(jnp.array(1.0, dtype=jnp.float32), jnp.array(True))
        optimizer = optax.adamw(learning_rate=1e-3)
        loss_grad = eqx.filter_value_and_grad(
            lambda model, x, y: ((model(x) - y)**2, {"total": (model(x) - y)**2}),
            has_aux=True,
        )
        data = (jnp.array(1.0), jnp.array(0.0))
        for train_step in (step, bone_step):
            with self.subTest(module=train_step.__module__):
                old_state = optimizer.init(eqx.filter(model, eqx.is_array))
                _, old_updated, _, _ = train_step(model, old_state, data, loss_grad, optimizer)
                state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
                _, updated, next_state, _ = train_step(model, state, data, loss_grad, optimizer)
                self.assertEqual(jax.tree_util.tree_structure(state), jax.tree_util.tree_structure(next_state))
                np.testing.assert_array_equal(updated.value, old_updated.value)
                self.assertTrue(bool(updated.mask))


@unittest.skipUnless(HAS_FITTING, "requires the biomechanics environment")
class GeometryOnlyForwardTests(unittest.TestCase):
    def test_native_export_matches_scaled_mjx_geometry_and_site_order(self):
        import jax.numpy as jnp
        import mujoco
        from mujoco import mjx
        import numpy as np
        from pathlib import Path
        from baseline.native_export import export_kinematic_coordinates
        from baseline.monocular_demos.biomechanics_mjx.forward_kinematics import forward_kinematics

        path = Path(__file__).resolve().parent / "monocular_demos/biomechanics_mjx/data/humanoid/humanoid_torque.xml"
        native = mujoco.MjModel.from_xml_path(str(path))
        model = mjx.put_model(native)
        generator = np.random.default_rng(22)
        poses = np.tile(native.qpos0, (3, 1)) + generator.normal(0, .1, (3, native.nq))
        scales = generator.uniform(.85, 1.15, (3, native.nbody))
        offsets = generator.normal(0, .001, (native.nsite, 3))
        order = np.arange(native.nsite)[::-1].copy()
        for factors in (scales, scales[0, :, None]):
            bodies, sites = export_kinematic_coordinates(path, poses, factors, offsets, order)
            per_frame = factors if factors.ndim == 2 and factors.shape[1] == native.nbody else np.broadcast_to(factors[:, 0], scales.shape)
            for frame in range(len(poses)):
                reference = forward_kinematics(model, jnp.asarray(poses[frame]), jnp.asarray(per_frame[frame, :, None]),
                    site_offsets=jnp.asarray(offsets), site_reorder=jnp.asarray(order))
                np.testing.assert_allclose(bodies[frame], reference.xpos, rtol=2e-6, atol=2e-7)
                np.testing.assert_allclose(sites[frame], reference.site_xpos, rtol=2e-6, atol=2e-7)

    def test_geometry_and_gradients_match_full_dynamics_with_contacts(self):
        import importlib
        import jax
        import jax.numpy as jnp
        import mujoco
        from mujoco import mjx
        import numpy as np

        xml = '''<mujoco><option gravity="0 0 -9.81"/>
          <worldbody><geom type="plane" size="2 2 .1"/>
            <body pos="0 0 .06"><joint name="slide" type="slide" axis="1 0 0"/>
              <geom type="sphere" size=".1"/><site name="root" pos=".03 .02 .01"/>
              <body pos=".2 0 .15"><joint name="hinge" axis="0 1 0"/>
                <geom type="capsule" size=".03 .1"/><site name="end" pos=".15 .02 .12"/>
              </body></body></worldbody></mujoco>'''
        model = mjx.put_model(mujoco.MjModel.from_xml_string(xml))
        qpos = jnp.array([.07, .35])
        qvel = jnp.array([.02, -.04])
        scale = jnp.array([[1.], [1.08], [.93]])
        offsets = jnp.array([[.01, -.02, .03], [-.02, .01, 0.]])
        reorder = jnp.array([1, 0])
        for package in ("baseline", "metrics.bone_length"):
            with self.subTest(package=package):
                module = importlib.import_module(package + ".monocular_demos.biomechanics_mjx.forward_kinematics")

                def evaluate(q, scales, full):
                    return module.forward_kinetics(model, q, qvel, None, scales,
                        site_reorder=reorder, site_offsets=offsets,
                        return_contact_forces=full, check_constraints=False)[0]

                fast, full = evaluate(qpos, scale, False), evaluate(qpos, scale, True)
                for name in ("qpos", "qvel", "xpos", "site_xpos"):
                    np.testing.assert_allclose(getattr(fast, name), getattr(full, name), rtol=1e-6, atol=1e-7)
                # The reference really computes nonzero ground-contact forces.
                self.assertGreater(float(jnp.max(jnp.abs(full.efc_force))), 0.)

                def objective(q, scales, full):
                    state = evaluate(q, scales, full)
                    return jnp.sum(state.site_xpos ** 2) + .3 * jnp.sum(state.xpos ** 2)

                fast_grad = jax.grad(objective, argnums=(0, 1))(qpos, scale, False)
                # The full force solver has a dynamic while-loop, so use
                # forward-mode AD for its independent geometry reference.
                full_grad = jax.jacfwd(objective, argnums=(0, 1))(qpos, scale, True)
                for actual, expected in zip(fast_grad, full_grad):
                    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-7)

                fast_fk = module.forward_kinematics(model, qpos, scale, site_reorder=reorder,
                    site_offsets=offsets)
                full_fk = module.forward_kinematics(model, qpos, scale, site_reorder=reorder,
                    site_offsets=offsets, return_contact_forces=True)
                np.testing.assert_allclose(fast_fk.site_xpos, full_fk.site_xpos, rtol=1e-6, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
