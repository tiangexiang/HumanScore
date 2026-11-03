# (V) Kinematic extremes: paper implementation

The original HumanScore paper implementation of metric (V). It flags frames in which a joint turns, or a
body segment moves, faster than a human can.

HumanScore uses this code for its kinematic-extremes score, both by default and in the paper scores
(`--paper-scores`). It scores `<S>/baseline/<stem>_opensim_skeleton_grid.mot`, the fitted skeleton
sampled on a uniform 24 Hz grid (finite differences need a uniform time step).

## Definition

Velocities are central differences on the actual timestamps (one-sided at the ends; no difference
crosses a timestamp gap). Segment speeds are those of the segment centres of mass, from MuJoCo forward
kinematics of the template skeleton (`humanoid_torque.xml`).

For frame `t`, coordinate `j` and segment `b`:

```
u_j(t) = clip((|w_j(t)| - wmax_j) / (0.5 * wmax_j), 0, 1)       angular velocity w, deg/s
u_b(t) = clip((v_b(t) - vmax_b) / (0.5 * vmax_b), 0, 1)         linear speed v, m/s
m_t = 0.5 * (mean_j u_j(t) + mean_b u_b(t))                     (mean_j u_j(t) without a body model)
b_t = m_t > 0.05

r = mean_t b_t,   s = sum_t m_t / max(1, sum_t b_t),   p = (longest run of b_t = 1) / T
score = 100 * (1 - clip(0.5 r + 0.3 s + 0.2 p, 0, 1))
```

Every coordinate and segment with a limit has weight 1 and stays in the mean also when it is not
violated. `s` sums `m_t` over all frames, also those below the threshold, as the paper writes it.

| constant | value | where |
|---|---|---|
| `FRAME_VIOLATION_THRESHOLD` | 0.05 | `main.py` |
| `EXCESS_FRACTION_ANG`, `EXCESS_FRACTION_LIN` | 0.5 (50 % above the limit gives full severity) | `main.py` |
| weights | 0.5 / 0.3 / 0.2 (frequency / severity / persistence) | `--alpha`, `--beta`, `--gamma` |
| angular velocity limits | 360-4000 deg/s per coordinate | `definition.py` |
| linear speed limits | 12 m/s (pelvis) to 45 m/s (hands) per segment | `definition.py` |

The limits sit at the upper end of human capability (fast throws, kicks and sprints), so that only
physically implausible motion counts.

## Run on its own

From the repository root:

```bash
python -m metrics.kinematic_extreme.main \
    --mot_path humanscore_output/clip/baseline/clip_opensim_skeleton_grid.mot \
    --output_dir results/metrics/kinematic_extreme
```

| option | default | meaning |
|---|---|---|
| `--mot_path` | - | one `.mot` file |
| `--mot_dir` | `results/baseline` | score every `.mot` in this folder (used without `--mot_path`) |
| `--output_dir` | `results/metrics/kinematic_extreme` | scores |
| `--model_path` | `baseline/.../humanoid_torque.xml` | skeleton for the segment speeds; `""` scores joint velocities only |
| `--alpha`, `--beta`, `--gamma` | 0.5, 0.3, 0.2 | weights |
| `--frame_violation_threshold` | 0.05 | flag threshold on `m_t` |
| `--no_detail` | off | skip the per-video `.txt` files |

The motion must be an OpenSim `.mot` file whose header line starts with `time`. Angles are in radians
unless the header says `inDegrees=yes`. For the segment speeds the columns must name the coordinates of
`humanoid_torque.xml`, as `baseline/metrab2opensim.py` writes them.

Segment speeds are cached in `metrics/kinematic_extreme/.body_kin_cache/` (or in
`$KEXTREME_BODY_KIN_CACHE_DIR`). A cache entry is reused only while the motion, the skeleton files and
the code are unchanged.

## Output

* `<output_dir>/<video>_kextreme.json`: `kextreme_score` (0-100, higher is better),
  `frame_violation_rate` (`r`), `mean_violation_severity` (`s`), `max_consecutive_violation_ratio` (`p`),
  per coordinate and per segment the violation rate and the mean excess, and the settings (`debug`).
* `<output_dir>/<video>_kextreme.txt`: short summary.
* `<output_dir>/kextreme_scores_summary.csv`: one row per video.

## Files

* `definition.py`: the velocity limits and weights.
* `utils.py`: motion loading, derivatives, forward kinematics and the segment-speed cache.
* `main.py`: the score and the command line.
* `test_scoring.py`: tests with analytic scores for this and the other paper metrics.
