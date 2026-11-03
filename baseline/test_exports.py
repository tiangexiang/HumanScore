"""Regression checks for motion exports; no detector weights or JAX required."""
from pathlib import Path
import importlib.util
import json
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from baseline.metrab2opensim import convertDataframeToMot, convert_fitted_model_to_mot, get_joint_names


def read_mot(path):
    lines = path.read_text().splitlines()
    return pd.read_csv(path, sep="\t", skiprows=lines.index("endheader") + 1)


class MotionExportTests(unittest.TestCase):
    @unittest.skipUnless(all(importlib.util.find_spec(name) for name in ("jax", "jaxlie", "cv2")), "requires biomechanics camera dependencies")
    def test_camera_projectors_match_opencv_distortion(self):
        import cv2
        import jax
        jax.config.update("jax_enable_x64", True)
        from baseline.monocular_demos.camera import project_distortion
        from metrics.bone_length.monocular_demos.camera import project_distortion as project_bone

        points = np.array([[100.0, 200.0, 1500.0], [-250.0, 350.0, 1800.0], [500.0, -200.0, 1000.0]])
        intrinsics = np.array([[1200.0, 0.0, 640.0], [0.0, 1250.0, 360.0], [0.0, 0.0, 1.0]])
        distortion = np.array([0.05, -0.03, 0.002, -0.004, 0.01])
        calibration = {"mtx": np.array([[1.2, 1.25, 0.64, 0.36]]), "dist": distortion[None],
                       "rvec": np.zeros((1, 3)), "tvec": np.zeros((1, 3))}
        expected = cv2.projectPoints(points, np.zeros(3), np.zeros(3), intrinsics, distortion)[0][:, 0, :]
        for project in (project_distortion, project_bone):
            np.testing.assert_allclose(np.asarray(project(calibration, 0, points)), expected, rtol=0, atol=1e-9)

    def test_camera_intrinsics_match_metrabs_fov_and_explicit_calibration(self):
        from baseline.main import load_camera_calibration

        calibration = load_camera_calibration(image_size=(1280, 720))
        focal = 1280 / (2 * np.tan(np.deg2rad(55) / 2))
        np.testing.assert_allclose(calibration["mtx"][0] * 1000, [focal, focal, 640, 360])
        np.testing.assert_allclose(calibration["dist"], 0)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "camera.json"
            values = {"fx": 1200, "fy": 1250, "cx": 500, "cy": 400}
            path.write_text(json.dumps(values))
            explicit = load_camera_calibration(path)
            np.testing.assert_allclose(explicit["mtx"][0] * 1000, [1200, 1250, 500, 400])
            values["fx"] = -1
            path.write_text(json.dumps(values))
            with self.assertRaisesRegex(ValueError, "positive"):
                load_camera_calibration(path)

    def test_preserves_irregular_timestamps(self):
        timestamps = np.array([0.0, 0.04166666667, 0.0835])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "motion.mot"
            convertDataframeToMot(pd.DataFrame({"knee_angle_r": [0.0, 0.1, 0.2]}), path, 0.04, timestamps)
            np.testing.assert_allclose(read_mot(path)["time"], timestamps, rtol=0, atol=1e-12)

    def test_rejects_nonincreasing_timestamps(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                convertDataframeToMot(pd.DataFrame({"x": [0.0, 0.0]}), Path(folder) / "bad.mot", 0.1, [0.1, 0.1])

    def test_root_transform_is_invertible_and_knee_sign_is_correct(self):
        names = get_joint_names()
        qpos = np.random.default_rng(42).uniform(-1.0, 1.0, (16, len(names)))
        timestamps = np.arange(len(qpos)) / 24.0
        axis_transform = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])
        angles = ["pelvis_tilt", "pelvis_list", "pelvis_rotation"]
        translation = ["pelvis_tx", "pelvis_ty", "pelvis_tz"]
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "fit.npz"
            target = Path(folder) / "motion.mot"
            np.savez(source, qpos=qpos, qvel=np.zeros_like(qpos), timestamps=timestamps)
            self.assertTrue(convert_fitted_model_to_mot(source, target))
            exported = read_mot(target)
            self.assertIn("signed_pelvis_rotation=yes", target.read_text())
            self.assertTrue(np.any(exported["pelvis_rotation"] < 0))
            recovered = np.einsum("ij,njk->nik", axis_transform.T, Rotation.from_euler("ZXY", exported[angles]).as_matrix())
            original = Rotation.from_euler("ZXY", qpos[:, [names.index(x) for x in angles]]).as_matrix()
            np.testing.assert_allclose(recovered, original, atol=1e-12)
            np.testing.assert_allclose(exported[translation].to_numpy() @ axis_transform, qpos[:, [names.index(x) for x in translation]], atol=1e-12)
            for knee in ["knee_angle_l", "knee_angle_r"]:
                np.testing.assert_allclose(exported[knee], -qpos[:, names.index(knee)], atol=1e-12)

    def test_missing_detection_and_dotted_stem(self):
        from baseline.main import save_metrabs_data
        from metrics.bone_length.main import save_metrabs_data as save_bone_data

        for save in [save_metrabs_data, save_bone_data]:
            with self.subTest(module=save.__module__), tempfile.TemporaryDirectory() as folder:
                save({"boxes": [[]], "poses3d": [[]], "poses2d": [[]]}, "/data/clip.part1.mp4", folder)
                with np.load(Path(folder) / "clip.part1_keypoints.npz") as data:
                    self.assertEqual(data["keypoints3d"].shape, (1, 87, 3))
                    self.assertTrue(np.all(data["confs"] == 0))
                with self.assertRaisesRegex(ValueError, "No decodable video frames"):
                    save(None, "empty.mp4", folder)


if __name__ == "__main__":
    unittest.main()
