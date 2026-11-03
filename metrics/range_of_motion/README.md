# (III) Joint range of motion: paper implementation

The original HumanScore paper implementation of metric (III). It flags frames in which a joint of the
fitted skeleton is bent beyond its anatomical range.

HumanScore runs this code only with `--paper-scores`, on `<S>/baseline/<stem>_opensim_skeleton.mot`.
The skeleton is fitted without joint limits, so violations stay visible. The default joint-range score
is `hmb/scoring/joint_range.py`.

## Definition

For frame `t` and every scored coordinate `j` with range `[lo_j, hi_j]` (radians) and a tolerance of
15 degrees on both sides:

```
e_j(t) = max(0, lo_j - tol - q_j(t)) + max(0, q_j(t) - hi_j - tol)     exceedance
u_j(t) = clip(e_j(t) / (0.5 * (hi_j - lo_j)), 0, 1)
m_t = mean_j u_j(t)
b_t = m_t > 0.05

r = mean_t b_t,   s = sum_t m_t / max(1, sum_t b_t),   p = (longest run of b_t = 1) / T
score = 100 * (1 - clip(0.5 r + 0.3 s + 0.2 p, 0, 1))
```

Every scored coordinate has weight 1 and stays in the mean also when it is not violated.

| constant | value | where |
|---|---|---|
| tolerance | 15 degrees | `--tolerance_deg` |
| `ROM_EXCESS_FRACTION` | 0.5 (an excess of half the joint's range gives full severity) | `joints_violaton.py` |
| `FRAME_VIOLATION_THRESHOLD` | 0.05 | `joints_violaton.py` |
| weights | 0.5 / 0.3 / 0.2 (frequency / severity / persistence) | `--alpha`, `--beta`, `--gamma` |

Joint ranges (`definition.py`, degrees):

| coordinate | range | coordinate | range |
|---|---|---|---|
| `hip_flexion` | -30 to 120 | `lumbar_extension`, `lumbar_bending`, `lumbar_rotation` | -90 to 90 |
| `hip_adduction` | -50 to 30 | `arm_flex` | -90 to 90 |
| `hip_rotation` | -40 to 40 | `arm_add` | -120 to 90 |
| `knee_angle` | 0 to 120 | `arm_rot` | -90 to 90 |
| `ankle_angle` | -40 to 30 | `elbow_flex` | 0 to 150 |

Left and right sides use the same ranges. The pelvis coordinates (the global root) are not scored;
subtalar, toe, forearm pronation/supination and wrist coordinates have no range and are not scored.

## Run on its own

From the repository root:

```bash
python -m metrics.range_of_motion.joints_violaton \
    --mot_path humanscore_output/clip/baseline/clip_opensim_skeleton.mot \
    --output_dir results/metrics/range_of_motion
```

| option | default | meaning |
|---|---|---|
| `--mot_path` | - | one `.mot` file |
| `--mot_dir` | `results/baseline` | score every `.mot` in this folder (used without `--mot_path`) |
| `--output_dir` | `results/metrics/range_of_motion` | scores |
| `--tolerance_deg` | 15 | widen every range by this many degrees on both sides |
| `--alpha`, `--beta`, `--gamma` | 0.5, 0.3, 0.2 | weights |
| `--frame_violation_threshold` | 0.05 | flag threshold on `m_t` |
| `--no_detail` | off | skip the per-video `.txt` files |

The motion must be an OpenSim `.mot` file whose header line starts with `time`, with columns named as in
`definition.py` (as `baseline/metrab2opensim.py` writes them). Angles are in radians unless the header
says `inDegrees=yes`.

## Output

* `<output_dir>/<video>_rom.json`: `rom_score` (0-100, higher is better), `frame_violation_rate` (`r`),
  `mean_violation_severity` (`s`), `max_consecutive_violation_ratio` (`p`), per coordinate the
  violation rate and the mean excess in degrees, and the settings (`debug`).
* `<output_dir>/<video>_rom.txt`: summary with the per-coordinate statistics.
* `<output_dir>/rom_scores_summary.csv`: one row per video.
