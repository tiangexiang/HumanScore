# (VI) Motion smoothness: paper implementation

The original HumanScore paper implementation of metric (VI). It flags frames with joint angular
accelerations that no human produces: jerky, unnatural spikes.

HumanScore uses this code for its motion-smoothness score, both by default and in the paper scores
(`--paper-scores`). It scores `<S>/baseline/<stem>_opensim_skeleton_grid.mot`, the fitted skeleton
sampled on a uniform 24 Hz grid (finite differences need a uniform time step).

## Definition

Angular accelerations are second central differences on the actual timestamps (one-sided at the ends;
no difference crosses a timestamp gap; angles are unwrapped first). For frame `t` and coordinate `j`:

```
u_j(t) = clip((|a_j(t)| - amax_j) / (0.5 * amax_j), 0, 1)      angular acceleration a, deg/s^2
m_t = mean_j u_j(t)
b_t = m_t > 0.05

r = mean_t b_t,   s = sum_t m_t / max(1, sum_t b_t),   p = (longest run of b_t = 1) / T
score = 100 * (1 - clip(0.5 r + 0.3 s + 0.2 p, 0, 1))
```

Every coordinate with a limit has weight 1 and stays in the mean also when it is not violated. `s`
sums `m_t` over all frames, also those below the threshold, as the paper writes it. Only the
acceleration term is implemented: the paper's optional jerk-energy term needs calibrated
per-coordinate limits and is left out (`debug.jerk_energy_included` is `false`).

| constant | value | where |
|---|---|---|
| `FRAME_VIOLATION_THRESHOLD` | 0.05 | `main.py` |
| `EXCESS_FRACTION_ACC` | 0.5 (50 % above the limit gives full severity) | `main.py` |
| weights | 0.5 / 0.3 / 0.2 (frequency / severity / persistence) | `--alpha`, `--beta`, `--gamma` |
| angular acceleration limits | 8,000-60,000 deg/s^2 per coordinate | `definition.py` |

The limits sit at the upper end of human capability (fast kicks and throws stay below them).

## Run on its own

From the repository root:

```bash
python -m metrics.motion_smoothness.main \
    --mot_path humanscore_output/clip/baseline/clip_opensim_skeleton_grid.mot \
    --output_dir results/metrics/motion_smoothness
```

| option | default | meaning |
|---|---|---|
| `--mot_path` | - | one `.mot` file |
| `--mot_dir` | `results/baseline` | score every `.mot` in this folder (used without `--mot_path`) |
| `--output_dir` | `results/metrics/motion_smoothness` | scores |
| `--alpha`, `--beta`, `--gamma` | 0.5, 0.3, 0.2 | weights |
| `--frame_violation_threshold` | 0.05 | flag threshold on `m_t` |
| `--no_detail` | off | skip the per-video `.txt` files |

The motion must be an OpenSim `.mot` file whose header line starts with `time`, with at least three
frames per contiguous stretch of timestamps. Angles are in radians unless the header says
`inDegrees=yes`.

## Output

* `<output_dir>/<video>_msmooth.json`: `msmooth_score` (0-100, higher is smoother),
  `frame_violation_rate` (`r`), `mean_violation_severity` (`s`), `max_consecutive_violation_ratio` (`p`),
  per coordinate the violation rate and the mean excess in deg/s^2, and the settings (`debug`).
* `<output_dir>/<video>_msmooth.txt`: short summary.
* `<output_dir>/msmooth_scores_summary.csv`: one row per video.

## Files

* `definition.py`: the acceleration limits and weights.
* `utils.py`: motion loading and derivatives.
* `main.py`: the score and the command line.

Kinematic extremes (`metrics/kinematic_extreme`) checks velocities, how fast joints move; this metric
checks accelerations, how abruptly the velocities change.
