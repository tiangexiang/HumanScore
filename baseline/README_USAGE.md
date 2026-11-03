# `baseline/main.py`: keypoints and skeleton fit

`baseline/` holds the [MonocularBiomechanics](https://github.com/IntelligentSensingAndRehabilitation/MonocularBiomechanics)
code that HumanScore uses to reconstruct a person's motion: MeTRAbs keypoint detection and a MuJoCo
biomechanical skeleton fitted to the keypoints. HumanScore runs the same detection and fit through its
engines (`hmb/detect_engine.py`, `hmb/fit_engine.py`), so you do not need this script to score a video.
It is the standalone command line of the same steps.

For each video, `main.py` runs four steps (each can be skipped):

1. MeTRAbs keypoint detection: `<stem>_keypoints.npz`
2. skeleton fit (one fixed scale per body part): `<stem>_fitted_model.npz`
3. rendering of the fitted skeleton: `<stem>_mjx.mp4` (needs MuJoCo rendering; `MUJOCO_GL=egl` on a
   server)
4. OpenSim export: `<stem>_opensim_skeleton.mot`

## Usage

Run it from this folder, in the HumanScore environment:

```bash
cd baseline
python main.py --video ../clip.mp4 --output_dir ../results/baseline
python main.py --video_dir ../videos --output_dir ../results/baseline --skip_rendering
python main.py --config videos.json        # {"videos": ["../a.mp4", "../b.mp4"]}
```

Steps whose inputs already exist can be skipped, for example to render and export again:

```bash
python main.py --video ../clip.mp4 --skip_metrabs --skip_fitting
```

| option | default | meaning |
|---|---|---|
| `--video` | - | one video |
| `--video_dir` | `../data` | every `.mp4`, `.avi`, `.mov` and `.mkv` in this folder (used without `--video`/`--config`) |
| `--config` | - | JSON file with a `videos` list |
| `--output_dir` | `../results/baseline` | all outputs |
| `--skip_metrabs`, `--skip_fitting`, `--skip_rendering`, `--skip_opensim` | off | skip a step |
| `--metrabs-model` | installed model | MeTRAbs SavedModel folder or TF Hub handle (also `HMB_METRABS_MODEL`) |
| `--max-iters` | 10000 | fitting iterations |
| `--unconstrained-joints` | off | fit without anatomical joint bounds, so joint-range violations stay visible (HumanScore fits this way) |
| `--camera-calibration` | 55-degree field of view | JSON with `fx`, `fy`, `cx`, `cy` in pixels and optional `distortion` (five coefficients) |
| `--export-backend` | `native` | export body and site positions with native MuJoCo or with MJX (`mjx`) |

Existing output files are overwritten. Videos fitted in one call must have the same frame size.

## Output files

* `<stem>_keypoints.npz`: per frame the MeTRAbs 3-D (`keypoints3d`, mm, camera frame) and 2-D
  (`keypoints2d`, pixels) keypoints of the `bml_movi_87` skeleton, the person box and the keypoint
  confidences, plus the frame timestamps and the camera intrinsics.
* `<stem>_fitted_model.npz`: joint angles `qpos` and velocities `qvel` per frame, body positions
  (`joints`) and keypoint sites (`sites`) in metres, the per-body scales and the fit settings.
* `<stem>_mjx.mp4`: the fitted skeleton.
* `<stem>_opensim_skeleton.mot`: the joint angles in OpenSim coordinates (radians), as the joint-range
  metric reads them. (HumanScore's fit engine also writes `<stem>_opensim_skeleton_grid.mot`, the same
  motion on a uniform 24 Hz grid, for the velocity and acceleration metrics.)

## Code

* `main.py`: the command line (detection, fit, rendering, export).
* `metrab2opensim.py`: OpenSim `.mot` export.
* `native_export.py`: body and site positions of the fitted skeleton with native MuJoCo.
* `monocular_demos/`: the fitting code (camera model, datasets, implicit trajectory network, MuJoCo
  skeleton, optimiser) and the skeleton model in `biomechanics_mjx/data/humanoid/`.
* `test_exports.py`, `test_fitting.py`: tests.

`metrics/bone_length/` has a copy of `monocular_demos/` whose segment scales may change over time; it is
used only for the paper's bone-length metric.
