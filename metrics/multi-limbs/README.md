# (I) Extra limbs: paper implementation

The original HumanScore paper implementation of metric (I). It scores how often, how strongly and for
how long the [HADM](https://github.com/wangkaihong/HADM) human-artefact detector (HADM-G) sees a person
with an extra hand, arm, leg or foot.

HumanScore runs this code only with `--paper-scores`, on the detections it has already computed for
the default extra-limbs score. You do not need to run it by hand.

## Definition

For frame `t` of `T`, `M_t` is the highest confidence of the four classes "human with extra
{hand, arm, leg, feet}" (0 when below `tau_prob`). Then

```
b_t = 1 if M_t > tau_mild else 0
m_t = clip((M_t - tau_mild) / (tau_severe - tau_mild), 0, 1)

r = mean_t b_t                           frequency
s = sum_t m_t / max(1, sum_t b_t)        severity
p = (longest run of b_t = 1) / T         persistence
score = 100 * (1 - clip(0.5 r + 0.3 s + 0.2 p, 0, 1))
```

| constant | value | meaning |
|---|---|---|
| `tau_prob` | 0.005 | detections below this confidence are ignored |
| `tau_mild` | 0.005 | a frame is flagged above this confidence |
| `tau_severe` | 0.03 | confidence of full severity |
| weights | 0.5 / 0.3 / 0.2 | frequency / severity / persistence |

## Run on its own

HADM is set up by the HumanScore installer (`metrics/multi-limbs/HADM`, weights in
`HADM/pretrained_models/`). The folder name contains a hyphen, so run the script by its path from
the repository root:

```bash
# video -> frames -> HADM -> score
python metrics/multi-limbs/eval_extra_limbs.py --video_path clip.mp4 --output_dir out/extra_limbs

# score an existing HADM output.json without running the detector
python metrics/multi-limbs/eval_extra_limbs.py --video_path clip.mp4 \
    --detections-json path/to/output.json --output_dir out/extra_limbs
```

Main options:

| option | default | meaning |
|---|---|---|
| `--video_path` / `--video_dir` | - / `data` | one video, or every `.mp4` in a folder |
| `--detections-json` | - | score this HADM `output.json` (needs `--video_path` for the name) |
| `--hadm_root` | `metrics/multi-limbs/HADM` | HADM checkout |
| `--work_dir` | `results/metrics/extra_limbs/mid_outputs` | extracted frames and HADM outputs, reused on later runs |
| `--output_dir` | `results/metrics/extra_limbs` | scores |
| `--skip_inference` | off | only score detections that already exist |
| `--overwrite_frames` | off | extract the frames again |
| `--tau_prob`, `--tau_mild`, `--tau_severe`, `--alpha`, `--beta`, `--gamma` | see above | score constants |

## Output

* `<output_dir>/<video>_extra_limbs.json`: `extra_limbs_score` (0-100, higher means fewer extra
  limbs), `r`, `severity_mean` (`s`), `continuity` (`p`), `D`, `avg_M_t`, `num_frames` and the path of
  the scored detections.
* `<output_dir>/extra_limbs_scores_summary.csv`: one row per video.
