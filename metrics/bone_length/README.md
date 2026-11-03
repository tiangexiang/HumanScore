# (II) Bone-length consistency: paper implementation

The original HumanScore paper implementation of metric (II). A real person's bones keep their length.
This metric fits a biomechanical skeleton whose segment scales may change over time and measures how much
nine major bones change length over the video.

HumanScore runs this code only with `--paper-scores`: it then also runs the variable-scale fit (through
`hmb/fit_engine.py`) and scores `<S>/bone_fit/<stem>_bone_length.npz`. The default bone-length score
(`hmb/scoring/bone_length.py`) does not need this fit.

## Definition

For each bone `b` (proximal body, distal body) and each valid frame `t`:

```
l_b(t) = || p_proximal(t) - p_distal(t) ||        joint centre to joint centre
L_b    = median_t l_b(t)                          reference length
E_b    = mean_t |l_b(t) - L_b| / (L_b + eps)      mean relative deviation
E      = mean_b E_b
score  = 100 * (1 - clip(E / tau, 0, 1))
```

| constant | value | meaning |
|---|---|---|
| `tau` (`DEFAULT_TAU`) | 0.15 | a mean relative deviation of 15 % or more scores 0 (paper, supplementary F.1) |
| `MIN_FRAMES_PER_BONE` | 5 | a bone needs at least this many valid frames |
| `eps` | 1e-8 | avoids division by zero |

A frame is valid when both endpoints are finite and at least one keypoint was detected in it (when the
file has a `confidence` array).

Bones (MuJoCo body names of `humanoid_torque.xml`):

| bone | proximal → distal |
|---|---|
| `femur_r`, `femur_l` | femur → tibia (hip to knee) |
| `tibia_r`, `tibia_l` | tibia → talus (knee to ankle) |
| `humerus_r`, `humerus_l` | humerus → ulna (shoulder to elbow) |
| `forearm_r`, `forearm_l` | radius → hand (forearm pivot to wrist) |
| `torso` | torso → head (lower spine to neck) |

## Run on its own

From the repository root:

```bash
# 1. variable-scale fit: writes results/bone_fit/<video>_keypoints.npz and <video>_bone_length.npz
python -m metrics.bone_length.main --video clip.mp4 --output_dir results/bone_fit --skip_rendering --skip_opensim

# 2. score one fit, or every *_bone_length.npz under a folder
python -m metrics.bone_length.bone_length --input_path results/bone_fit/clip_bone_length.npz --output_dir results/metrics/bone_length
python -m metrics.bone_length.bone_length --input_dir results/bone_fit --output_dir results/metrics/bone_length
```

| option of `bone_length` | default | meaning |
|---|---|---|
| `--input_path` | - | one `*_bone_length.npz` |
| `--input_dir` | `results/baseline` | folder searched for `*_bone_length.npz` (used without `--input_path`) |
| `--output_dir` | `results/metrics/bone_length` | scores |
| `--tau` | 0.15 | see above |
| `--no_detail` | off | skip the per-video `.txt` files |

The input `.npz` needs `timestamps` (T,) and `joints` (T, bodies, 3); optional keys are `body_names`,
`confidence` and `scale_time_varying`. A rigid fit (`scale_time_varying=False`) is rejected, because a
rigid skeleton cannot change bone length.

## Output

* `<output_dir>/<video>_bone_length.json`: `bone_length_score` (0-100, higher means more consistent),
  `mean_relative_error` (`E`), median and maximum of `E_b`, and per bone the deviation
  (`per_bone_mean_relative_error`), the reference length in metres (`per_bone_ref_length`) and the
  coefficient of variation in percent (`per_bone_cv_percent`).
* `<output_dir>/<video>_bone_length.txt`: short summary.
* `<output_dir>/bone_length_scores_summary.csv`: one row per video.

## Files

* `main.py`: MeTRAbs detection and the variable-scale skeleton fit (standalone CLI).
* `bone_length.py`: the score.
* `monocular_demos/`: the variable-scale version of `baseline/monocular_demos` (the fitting code).
