"""Tests for the fused/batched skeleton fitting and hmb.fit_engine (CPU, tiny trajectory MLP)."""
import importlib
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

HAS_FITTING = all(importlib.util.find_spec(name) is not None for name in ("jax", "equinox", "optax", "mujoco"))
REPO = Path(__file__).resolve().parents[1]
TREES = {
    "rigid": ("baseline.monocular_demos", REPO / "baseline/monocular_demos/biomechanics_mjx/data/humanoid/humanoid_torque.xml"),
    "variable": ("metrics.bone_length.monocular_demos",
                 REPO / "metrics/bone_length/monocular_demos/biomechanics_mjx/data/humanoid/humanoid_torque.xml"),
}
TINY = {"features": [16, 16], "concatenate_layers": 1}


def synthetic_keypoints(frames, seed=0, sites=87):
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 250.0, (sites, 3)) + np.array([0.0, 0.0, 3000.0])
    drift = np.cumsum(rng.normal(0, 5.0, (frames, sites, 3)), axis=0)
    kp3d = base[None] + drift
    kp2d = 500.0 + 1000.0 * kp3d[..., :2] / kp3d[..., 2:]
    confs = np.ones((frames, sites))
    return kp2d.astype(np.float32), kp3d, confs


TOY_XML = """<mujoco><compiler autolimits="true"/>
  <worldbody>
    <body name="base">
      <joint name="tx" type="slide" axis="1 0 0"/>
      <joint name="rz" type="hinge" axis="0 0 1" range="-1 1"/>
      <geom type="sphere" size=".05"/><site name="s0" pos="0 0 0"/>
      <body name="link" pos=".2 0 0">
        <joint name="h1" type="hinge" axis="0 1 0" range="-2 2"/>
        <geom type="capsule" size=".02 .1"/><site name="s1" pos=".1 0 0"/><site name="s2" pos=".2 .05 0"/>
      </body>
    </body>
  </worldbody></mujoco>"""
TOY_SITES = ["s0", "s1", "s2"]


def camera():
    return {"mtx": np.array([[1.0, 1.0, 0.5, 0.5]]), "dist": np.zeros((1, 5)), "rvec": np.zeros((1, 3)), "tvec": np.zeros((1, 3))}


def setUpModule():
    if HAS_FITTING:
        import jax

        jax.config.update("jax_enable_x64", True)


@unittest.skipUnless(HAS_FITTING, "requires the biomechanics environment")
class EncodingTests(unittest.TestCase):
    def test_fixed_encoding_has_exact_top_frequency_and_covers_duration(self):
        for package in ("baseline", "metrics.bone_length"):
            module = importlib.import_module(package + ".monocular_demos.implicit_representation")
            with self.subTest(package=package):
                for duration in (3.2, 6.0, 6.8333, 9.0):
                    for top_hz in (4.0, 8.0, 10.0):
                        length, max_time = module.fixed_encoding(duration, top_hz)
                        self.assertAlmostEqual(module.encoding_top_hz(length, max_time), top_hz, places=9)
                        self.assertGreaterEqual(max_time, duration)
                        # one level fewer would not cover the duration
                        self.assertLess(math.pi * 2.0 ** (length - 2) / (2 * math.pi * top_hz), duration)
                with self.assertRaises(ValueError):
                    module.fixed_encoding(5.0, 0.0)

    def test_paper_encoding_bandwidth_depends_on_clip_duration(self):
        from baseline.monocular_demos.implicit_representation import calculate_encoding_length, encoding_top_hz

        tops = {}
        for frames in (121, 141):
            span = np.float32(np.float32((frames - 1) / 24.0) + np.float32(1.0))  # dataset.max_time + 1 s
            length = calculate_encoding_length(span, min_freq=80)
            tops[frames] = encoding_top_hz(length, math.ceil(span))
        self.assertAlmostEqual(tops[121], 64 / 6, places=6)  # 5.00 s clip: L=8, max_time=6
        self.assertAlmostEqual(tops[141], 64 / 7, places=6)  # 5.83 s clip: L=8, max_time=7

    def test_trajectory_time_normalisation_options(self):
        import equinox as eqx
        import jax
        import jax.numpy as jnp
        from baseline.monocular_demos.implicit_representation import ImplicitTrajectory

        key = jax.random.PRNGKey(0)
        default = ImplicitTrajectory(joints=3, spatial_dims=1, encoding_length=4, max_time=5.2, key=key, **TINY)
        self.assertEqual(float(default.max_time), 6.0)
        exact = ImplicitTrajectory(joints=3, spatial_dims=1, encoding_length=4, max_time=5.2, round_max_time=False, key=key, **TINY)
        self.assertAlmostEqual(float(exact.max_time), 5.2, places=6)
        frozen = ImplicitTrajectory(joints=3, spatial_dims=1, encoding_length=4, max_time=5.2, train_max_time=False, key=key, **TINY)
        np.testing.assert_array_equal(default(jnp.array(1.3)), frozen(jnp.array(1.3)))
        objective = lambda m: jnp.sum(m(jnp.array(1.3)) ** 2)
        self.assertNotEqual(float(eqx.filter_grad(objective)(default).max_time), 0.0)
        self.assertEqual(float(eqx.filter_grad(objective)(frozen).max_time), 0.0)

    def test_kinetics_wrapper_fixed_encoding(self):
        import jax
        import jax.numpy as jnp

        for mode, (package, xml) in TREES.items():
            with self.subTest(mode=mode):
                mt = importlib.import_module(package + ".biomechanics_mjx.monocular_trajectory")
                utils = importlib.import_module(package + ".utils")
                fk = mt.ForwardKinetics(xml_path=str(xml), site_reorder=utils.joint_names)
                model = mt.KineticsWrapper(fk, scale_mixer=fk.build_default_scale_mixer(), key=jax.random.PRNGKey(0),
                                           max_time=jnp.array(4.5, jnp.float32), action=False, encoding="fixed", top_hz=8.0, **TINY)
                length, max_time = model.trajectory.encoding_length, float(model.trajectory.max_time[0])
                # covers last timestamp + 1 s = 5.5 s: 2**(L-1) >= 2*8*5.5 -> L = 8, max_time = pi*2**7/(2*pi*8) = 8
                self.assertEqual((length, round(max_time, 5)), (8, 8.0))
                with self.assertRaises(ValueError):
                    mt.KineticsWrapper(fk, scale_mixer=fk.build_default_scale_mixer(), key=jax.random.PRNGKey(0),
                                       max_time=4.5, action=False, encoding="fixed", **TINY)


@unittest.skipUnless(HAS_FITTING, "requires the biomechanics environment")
class FusedTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        xml = Path(cls.tmp.name) / "toy.xml"
        xml.write_text(TOY_XML)
        cls.trees = {}
        for mode, (package, _) in TREES.items():
            mt = importlib.import_module(package + ".biomechanics_mjx.monocular_trajectory")
            dataset_module = importlib.import_module(package + ".dataset")
            frames = 5
            kp2d, kp3d, confs = synthetic_keypoints(frames, seed=1, sites=len(TOY_SITES))
            confs[2] = 0.0  # an undetected frame stays in the scale smoothness sequence
            timestamps = np.arange(frames) / 24.0
            dataset = dataset_module.MonocularDataset(timestamps=[timestamps.copy()], keypoints_2d=[kp2d[None]],
                                                      keypoints_3d=[kp3d[None]], keypoint_confidence=[confs[None]],
                                                      camera_params=camera(), phone_attitude=None)
            model = mt.get_model(dataset, xml_path=str(xml), joint_names=TOY_SITES, scale_mappings={"link": ["link"]},
                                 enforce_joint_limits=False, **TINY)
            settings = dict(site_offset_regularization=1e2, lambda_reprojection=1e-1, static=True, center_3d=True, dt=1.0 / 29.0)
            if mode == "variable":
                settings["scale_smoothness_regularization"] = 1e3
            cls.trees[mode] = dict(mt=mt, dataset=dataset, model=model, settings=settings, frames=frames,
                                   arrays=(timestamps, kp2d, kp3d, confs))
        cls.chunks = {}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def padded_data(self, mode, padded):
        import jax.numpy as jnp

        timestamps, kp2d, kp3d, confs = self.trees[mode]["arrays"]
        extra = padded - len(timestamps)
        last = lambda a: np.concatenate([a, np.repeat(a[-1:], extra, axis=0)])
        return {"timestamps": jnp.asarray(last(timestamps)), "kp2d": jnp.asarray(last(kp2d), dtype=jnp.float64)[None],
                "kp3d": jnp.asarray(last(kp3d))[None], "conf": jnp.asarray(np.concatenate([confs, np.zeros((extra, confs.shape[1]))]))[None],
                "length": jnp.asarray(len(timestamps)), "camera": {k: jnp.asarray(v) for k, v in camera().items()}}

    def test_padded_cyclic_sample_matches_dataset_order(self):
        import jax.numpy as jnp

        for mode, tree in self.trees.items():
            mt, dataset, frames = tree["mt"], tree["dataset"], tree["frames"]
            for padded in (frames, frames + 3):
                data = self.padded_data(mode, padded)
                for i in (0, 1, 4, 7, 123):
                    with self.subTest(mode=mode, padded=padded, i=i):
                        (ix, ts, _), (k2, k3, c, _) = dataset[jnp.arange(1) + i]
                        (jx, pts, _, valid), (p2, p3, pc, _) = mt.padded_cyclic_sample(data, jnp.asarray(i), with_valid=True)
                        np.testing.assert_array_equal(np.asarray(jx), np.asarray(ix))
                        np.testing.assert_array_equal(np.asarray(pts)[:, :frames], np.asarray(ts))
                        np.testing.assert_array_equal(np.asarray(p2)[:, :, :frames], np.asarray(k2))
                        np.testing.assert_array_equal(np.asarray(p3)[:, :, :frames], np.asarray(k3))
                        np.testing.assert_array_equal(np.asarray(pc)[:, :, :frames], np.asarray(c))
                        self.assertTrue(np.all(np.asarray(pc)[:, :, frames:] == 0))
                        np.testing.assert_array_equal(np.asarray(valid)[0], np.arange(padded) < frames)
                        np.testing.assert_array_equal(np.asarray(pts)[0, frames:], np.full(padded - frames, float(data["timestamps"][-1])))

    def loss_and_grad(self, mode, padded, i=3, masked=True):
        import equinox as eqx
        import jax.numpy as jnp

        tree = self.trees[mode]
        mt = tree["mt"]
        x, y = mt.padded_cyclic_sample(self.padded_data(mode, padded), jnp.asarray(i), with_valid=True)
        if not masked:
            x = x[:3]
        settings = tree["settings"]
        fn = eqx.filter_jit(eqx.filter_value_and_grad(lambda m, x, y: mt.loss(m, x, y, camera_calibration=camera(), **settings),
                                                      has_aux=True))
        (value, _), grads = fn(tree["model"], x, y)
        return float(value), grads

    def test_loss_and_gradients_are_invariant_to_padding(self):
        import equinox as eqx
        import jax

        for mode in self.trees:
            with self.subTest(mode=mode):
                frames = self.trees[mode]["frames"]
                value, grads = self.loss_and_grad(mode, frames)
                padded_value, padded_grads = self.loss_and_grad(mode, frames + 4)
                self.assertAlmostEqual(value, padded_value, delta=1e-9 * max(1.0, abs(value)))
                for a, b in zip(jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array)),
                                jax.tree_util.tree_leaves(eqx.filter(padded_grads, eqx.is_array))):
                    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=2e-3, atol=1e-6)

    def test_variable_scale_smoothness_needs_the_padding_mask(self):
        frames = self.trees["variable"]["frames"]
        value, _ = self.loss_and_grad("variable", frames, i=2)
        unmasked, _ = self.loss_and_grad("variable", frames + 4, i=2, masked=False)
        masked, _ = self.loss_and_grad("variable", frames + 4, i=2, masked=True)
        self.assertAlmostEqual(value, masked, delta=1e-9 * max(1.0, abs(value)))
        self.assertNotAlmostEqual(value, unmasked, delta=1e-9 * max(1.0, abs(value)))

    def run_chunk(self, mode, padded, iters, steps_per_call, models=None, datas=None):
        import equinox as eqx
        import jax
        import jax.numpy as jnp

        tree = self.trees[mode]
        mt = tree["mt"]
        optimizer, _ = mt.build_optimizer(iters, lr_init_value=1e-3)
        settings = tree["settings"]
        value_and_grad = eqx.filter_value_and_grad(lambda m, x, y, cam: mt.loss(m, x, y, camera_calibration=cam, **settings), has_aux=True)
        loss_grad = lambda m, x, y, data: value_and_grad(m, x, y, data["camera"])
        sample = lambda data, i: mt.padded_cyclic_sample(data, i, with_valid=(mode == "variable"))
        models = models or [tree["model"]]
        datas = datas or [self.padded_data(mode, padded)] * len(models)
        static = eqx.partition(models[0], eqx.is_array)[1]
        stack = lambda trees: jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *trees)
        params = stack([eqx.filter(m, eqx.is_array) for m in models])
        state = stack([optimizer.init(eqx.filter(m, eqx.is_inexact_array)) for m in models])
        data = stack(datas)
        losses = []
        for start, length in mt.chunk_schedule(iters, steps_per_call):
            key = (mode, iters, length)
            if key not in self.chunks:  # reuse compilations across tests (same model structure)
                self.chunks[key] = jax.jit(jax.vmap(mt.make_train_chunk(loss_grad, optimizer, sample, length, static),
                                                    in_axes=(0, 0, 0, None)))
            params, state, vals, _ = self.chunks[key](params, state, data, jnp.asarray(start))
            losses.append(np.asarray(vals))
        return [eqx.combine(jax.tree_util.tree_map(lambda x: x[b], params), static) for b in range(len(models))], np.concatenate(losses, 1)

    def assert_models_close(self, a, b, rtol=1e-4, atol=1e-6):
        import equinox as eqx
        import jax

        for x, y in zip(jax.tree_util.tree_leaves(eqx.filter(a, eqx.is_inexact_array)),
                        jax.tree_util.tree_leaves(eqx.filter(b, eqx.is_inexact_array))):
            np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=rtol, atol=atol)

    def test_fused_padded_batched_training_matches_fit_model_loop(self):
        for mode, tree in self.trees.items():
            with self.subTest(mode=mode):
                mt, frames = tree["mt"], tree["frames"]
                iters = 4
                kwargs = {"scale_smoothness_regularization": 1e3} if mode == "variable" else {}
                reference, _ = mt.fit_model(tree["model"], tree["dataset"], lr_init_value=1e-3, max_iters=iters, **kwargs)
                fused, _ = mt.fit_model(tree["model"], tree["dataset"], lr_init_value=1e-3, max_iters=iters, steps_per_call=3, **kwargs)
                self.assert_models_close(reference, fused)  # includes a shorter remainder call (3 + 1)
                (unpadded,), _ = self.run_chunk(mode, frames, iters, 4)
                self.assert_models_close(reference, unpadded)
                (padded,), _ = self.run_chunk(mode, frames + 3, iters, 4)
                self.assert_models_close(reference, padded)

    def test_vmapped_fits_are_independent(self):
        import equinox as eqx
        import jax
        import jax.numpy as jnp

        mode = "rigid"
        tree = self.trees[mode]
        other = eqx.tree_at(lambda m: m.site_offsets, tree["model"], tree["model"].site_offsets + 0.01)
        data_a = self.padded_data(mode, 8)
        data_b = dict(data_a, kp3d=data_a["kp3d"] * 1.1)
        (alone_a,), _ = self.run_chunk(mode, 8, 4, 4, models=[tree["model"]], datas=[data_a])
        (alone_b,), _ = self.run_chunk(mode, 8, 4, 4, models=[other], datas=[data_b])
        (batch_a, batch_b), losses = self.run_chunk(mode, 8, 4, 4, models=[tree["model"], other], datas=[data_a, data_b])
        self.assertEqual(losses.shape, (2, 4))
        self.assert_models_close(alone_a, batch_a)
        self.assert_models_close(alone_b, batch_b)

    def test_models_share_static_forward_objects(self):
        # distinct-but-similar static Modules (holding arrays) can deadlock JAX's jit cache
        for mode, tree in self.trees.items():
            with self.subTest(mode=mode):
                mt, dataset = tree["mt"], tree["dataset"]
                xml = str(Path(self.tmp.name) / "toy.xml")
                a = mt.get_model(dataset, xml_path=xml, joint_names=TOY_SITES, scale_mappings={"link": ["link"]}, **TINY)
                b = mt.get_model(dataset, xml_path=xml, joint_names=TOY_SITES, scale_mappings={"link": ["link"]}, **TINY)
                self.assertIs(a.jit_forward, b.jit_forward)
                self.assertIs(a.jit_forward, tree["model"].jit_forward)

    def test_chunk_schedule_and_validation(self):
        mt = self.trees["rigid"]["mt"]
        self.assertEqual(mt.chunk_schedule(10, 4), [(0, 4), (4, 4), (8, 2)])
        self.assertEqual(mt.chunk_schedule(10000, 200)[-1], (9800, 200))
        with self.assertRaises(ValueError):
            mt.chunk_schedule(0, 4)
        with self.assertRaises(ValueError):
            mt.fit_model(self.trees["rigid"]["model"], self.trees["rigid"]["dataset"], max_iters=2, steps_per_call=0)
        optimizer, schedule = mt.build_optimizer(1000, lr_init_value=1e-3)
        self.assertAlmostEqual(float(schedule(0)), 1e-3, places=12)
        self.assertAlmostEqual(float(schedule(1000)), 1e-7, delta=1e-12)


class EngineHelperTests(unittest.TestCase):
    def test_options_buckets_and_grid(self):
        from hmb.fit_engine import FitOptions, bucket_length
        from hmb.fit_export import export_times, grid_confidence

        options = FitOptions.from_dict({"modes": ["rigid"], "buckets": [64, 32]})
        self.assertEqual(options.buckets, (32, 64))
        for bad in ({"encoding": "fixed"}, {"modes": ["nope"]}, {"steps_per_call": 0}, {"unknown": 1}, {"export_grid_hz": 0}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                FitOptions.from_dict(bad)
        self.assertEqual(bucket_length(141, (128, 160)), 160)
        self.assertEqual(bucket_length(160, (128, 160)), 160)
        self.assertEqual(bucket_length(1100, (128, 160)), 1152)
        from hmb.fit_engine import FitEngine

        engine = FitEngine({"batch": 8})
        self.assertEqual([engine.batch_size_for(n) for n in (1, 2, 3, 5, 8, 20)], [1, 2, 4, 8, 8, 8])
        self.assertEqual(FitEngine({"batch": 6}).batch_size_for(5), 6)
        grid = export_times(np.array([0.0, 0.05, 0.1, 0.25]), 20.0)
        np.testing.assert_allclose(grid, np.arange(6) / 20.0)
        conf = grid_confidence(np.array([0.0, 0.05, 0.1, 0.25]), np.array([[1.0], [0.0], [1.0], [0.0]]), grid)
        np.testing.assert_array_equal(conf[:, 0], [1, 0, 1, 1, 0, 0])

    def test_inputs_timestamp_precedence_and_intrinsics(self):
        from hmb.fit_engine import load_video_inputs

        kp2d, kp3d, confs = synthetic_keypoints(4, sites=3)
        with tempfile.TemporaryDirectory() as tmp:
            session = Path(tmp) / "s"
            (session / "baseline").mkdir(parents=True)
            (session / "input").mkdir()
            path = session / "baseline" / "clip_keypoints.npz"
            intrinsics = np.array([[1200.0, 1200.0, 640.0, 360.0]])
            np.savez(path, keypoints2d=kp2d, keypoints3d=kp3d, confs=confs, fps=np.array(20.0),
                     camera_intrinsics_px=intrinsics, camera_distortion=np.zeros((1, 5)), camera_calibration_source=np.array("assumed_fov55"))
            video = load_video_inputs({"session_dir": str(session), "stem": "clip"})
            np.testing.assert_allclose(video.timestamps, np.arange(4) / 20.0)
            self.assertEqual(video.timestamp_source, "keypoints_fps")
            np.testing.assert_allclose(video.camera["mtx"], intrinsics / 1000.0)
            plan = {"video": "v.mp4", "source_fps": 60.0, "num_source_frames": 10, "width": 1280, "height": 720,
                    "sample_fps": 24.0, "indices": [0, 2, 5, 7], "timestamps": [0.0, 2 / 60, 5 / 60, 7 / 60], "max_side": None}
            (session / "input" / "frames.json").write_text(json.dumps(plan))
            np.savez(path, keypoints2d=kp2d, keypoints3d=kp3d, confs=confs, camera_intrinsics_px=intrinsics,
                     camera_distortion=np.zeros((1, 5)))
            video = load_video_inputs({"session_dir": str(session), "stem": "clip"})
            np.testing.assert_allclose(video.timestamps, plan["timestamps"])
            self.assertEqual(video.timestamp_source, "frame_plan")
            np.savez(path, keypoints2d=kp2d, keypoints3d=kp3d, confs=confs, timestamps=np.array([1.0, 1.1, 1.3, 1.4]),
                     camera_intrinsics_px=intrinsics, camera_distortion=np.zeros((1, 5)))
            video = load_video_inputs({"session_dir": str(session), "stem": "clip"})
            np.testing.assert_allclose(video.timestamps, [0.0, 0.1, 0.3, 0.4])
            self.assertEqual(video.timestamp_source, "keypoints_npz")
            np.savez(path, keypoints2d=kp2d, keypoints3d=kp3d, confs=np.zeros_like(confs))
            with self.assertRaisesRegex(ValueError, "no valid person"):
                load_video_inputs({"session_dir": str(session), "stem": "clip"})


@unittest.skipUnless(HAS_FITTING, "requires the biomechanics environment")
class EngineEndToEndTests(unittest.TestCase):
    def test_engine_writes_cli_artifacts_skips_and_isolates_failures(self):
        from hmb.fit_engine import FitEngine

        with tempfile.TemporaryDirectory() as tmp:
            jobs = []
            for name, frames in (("a", 6), ("b", 7)):
                session = Path(tmp) / name
                (session / "baseline").mkdir(parents=True)
                kp2d, kp3d, confs = synthetic_keypoints(frames, seed=frames)
                np.savez(session / "baseline" / f"{name}_keypoints.npz", keypoints2d=kp2d, keypoints3d=kp3d, confs=confs,
                         timestamps=np.arange(frames) / 30.0, camera_intrinsics_px=np.array([[1000.0, 1000.0, 500.0, 500.0]]),
                         camera_distortion=np.zeros((1, 5)), camera_calibration_source=np.array("assumed_fov55"))
                jobs.append({"session_dir": str(session), "stem": name})
            jobs.append({"session_dir": str(Path(tmp) / "missing"), "stem": "missing"})
            engine = FitEngine({"max_iters": 4, "steps_per_call": 2, "batch": 2, "buckets": [8],
                                "trajectory_kwargs": TINY, "export_grid_hz": None,
                                "jax_cache_dir": str(Path(tmp) / "jax_cache")})
            statuses = engine.process_many(jobs)
            self.assertEqual([s["ok"] for s in statuses], [True, True, False])
            self.assertIn("No MeTRAbs keypoints", statuses[2]["error"])
            for job, frames in zip(jobs[:2], (6, 7)):
                session, stem = Path(job["session_dir"]), job["stem"]
                rigid = np.load(session / "baseline" / f"{stem}_fitted_model.npz")
                self.assertEqual(rigid["qpos"].shape, (frames, 40))
                self.assertEqual(rigid["joints"].shape, (frames, 22, 3))
                self.assertEqual(rigid["sites"].shape, (frames, 87, 3))
                self.assertEqual(rigid["scale"].shape, (8, 1))
                self.assertFalse(bool(rigid["scale_time_varying"]))
                np.testing.assert_allclose(rigid["timestamps"], np.arange(frames) / 30.0)
                variable = np.load(session / "bone_fit" / f"{stem}_bone_length.npz")
                self.assertEqual(variable["scale"].shape, (frames, 8))
                self.assertEqual(variable["body_scale_factors"].shape, (frames, 22))
                self.assertTrue(bool(variable["scale_time_varying"]))
                self.assertTrue((session / "baseline" / f"{stem}_bone_length.npz").is_file())
                mot = (session / "baseline" / f"{stem}_opensim_skeleton.mot").read_text()
                self.assertIn("inDegrees=no", mot)
                self.assertEqual(len(mot.strip().split("endheader")[1].strip().splitlines()), frames + 1)
                status = json.loads((session / "cache/engine_status/fit_engine.json").read_text())
                self.assertTrue(status["ok"])
                self.assertEqual(status["modes"]["rigid"]["padded_frames"], 8)
                self.assertEqual(status["modes"]["variable"]["batch_size"], 2)
                for mode in ("rigid", "variable"):
                    self.assertGreater(status["modes"][mode]["keypoint_residual_mm"], 0.0)
                    self.assertTrue(0.0 <= status["modes"][mode]["frac_nonpositive_body_scale"] <= 1.0)
            again = engine.process_many(jobs[:1])
            self.assertTrue(again[0]["modes"]["rigid"]["skipped"])
            # uniform-grid export (same compiled programs: same bucket and batch shape)
            engine.options.export_grid_hz = 20.0
            engine.options.modes = ("rigid",)
            forced = engine.process_many(jobs[:2], force=True)
            self.assertTrue(all(status["ok"] for status in forced))
            base = Path(jobs[1]["session_dir"]) / "baseline"
            observed = np.load(base / "b_fitted_model.npz")
            gridded = np.load(base / "b_fitted_model_grid.npz")
            keypoints = np.load(base / "b_keypoints.npz")
            # the CLI artifact stays on the observation timestamps (aligned with the keypoints) ...
            self.assertEqual(observed["qpos"].shape[0], keypoints["keypoints3d"].shape[0])
            # ... and the grid companion is the same fit on k / grid_hz
            np.testing.assert_allclose(gridded["timestamps"], np.arange(5) / 20.0)
            self.assertEqual(gridded["confidence"].shape, (5, 87))
            self.assertTrue((base / "b_opensim_skeleton_grid.mot").is_file())
            self.assertIn("keypoint_residual_mm", forced[1]["modes"]["rigid"])


if __name__ == "__main__":
    unittest.main()


def test_deterministic_xla_flags(monkeypatch):
    from hmb import fit_engine as fe

    monkeypatch.delenv("HMB_FIT_AUTOTUNE", raising=False)
    assert fe.deterministic_xla_flags("") == "--xla_gpu_autotune_level=0"
    assert fe.deterministic_xla_flags("--xla_foo=1") == "--xla_foo=1 --xla_gpu_autotune_level=0"
    assert fe.deterministic_xla_flags("--xla_gpu_autotune_level=4") == "--xla_gpu_autotune_level=4"   # user's choice kept
    monkeypatch.setenv("HMB_FIT_AUTOTUNE", "1")
    assert fe.deterministic_xla_flags("--xla_foo=1") == "--xla_foo=1"


def test_standalone_fit_commands_disable_xla_autotuning(monkeypatch, tmp_path):
    """baseline/main.py and metrics/bone_length/main.py (the standalone fit command lines) set
    XLA_FLAGS by hmb.fit_engine's rule before configure_runtime imports JAX; detection-only runs
    (TensorFlow) keep the caller's flags."""
    import os
    import sys

    seen = []
    for module_name in ("baseline.main", "metrics.bone_length.main"):
        module = importlib.import_module(module_name)
        monkeypatch.setattr(module, "configure_runtime", lambda use_tensorflow=True, use_jax=True: seen.append(
            (use_jax, os.environ.get("XLA_FLAGS"))))
        monkeypatch.setattr(module, "process_videos_with_metrabs", lambda *args, **kwargs: "detected")
        monkeypatch.setattr(module, "process_videos_with_biomechanics", lambda *args, **kwargs: "fitted")
        common = [module_name, "--video", str(tmp_path / "clip.mp4"), "--output_dir", str(tmp_path / "out"),
                  "--skip_rendering", "--skip_opensim"]

        def run(*extra, flags="--xla_foo=1", autotune=None):
            monkeypatch.setenv("XLA_FLAGS", flags)
            if autotune is None:
                monkeypatch.delenv("HMB_FIT_AUTOTUNE", raising=False)
            else:
                monkeypatch.setenv("HMB_FIT_AUTOTUNE", autotune)
            monkeypatch.setattr(sys, "argv", [*common, *extra])
            module.main()
            return seen[-1]

        assert run("--skip_metrabs") == (True, "--xla_foo=1 --xla_gpu_autotune_level=0")  # the CLI's fit call
        assert run("--skip_metrabs", flags="") == (True, "--xla_gpu_autotune_level=0")
        assert run("--skip_metrabs", flags="--xla_gpu_autotune_level=4") == (True, "--xla_gpu_autotune_level=4")
        assert run("--skip_metrabs", autotune="1") == (True, "--xla_foo=1")  # HMB_FIT_AUTOTUNE=1 keeps autotuning
        assert run("--skip_fitting") == (False, "--xla_foo=1")  # the CLI's detection call: TensorFlow only
