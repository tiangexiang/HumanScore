# Paper implementation of the six metrics

This folder holds the original implementation of the six metrics of the
[HumanScore paper](https://cs.stanford.edu/~xtiange/projects/humanscore/). The HumanScore scorer
(`hmb/scoring/`) uses it in two ways:

* **Always**, for (V) kinematic extremes and (VI) motion smoothness: these two HumanScore scores are
  computed by this code.
* **With `--paper-scores`**, for all six metrics: the result is reported as `paper_scores` next to the
  HumanScore scores, computed on the same reconstruction (`hmb/scoring/paper_scores.py`, outputs in
  `<S>/paper_metrics/<metric>/`).

You do not need to run anything here by hand: `scripts/humanscore_single.py` does it. Each folder also has
a command line for scoring files on their own; its README gives the definition, the constants and the
command.

| metric | folder | scores | input in a HumanScore session `<S>` |
|---|---|---|---|
| (I) extra limbs | [`multi-limbs/`](multi-limbs/README.md) | HADM-G extra-limb detections | `cache/multi_limbs/<stem>/hadm_results/output.json` |
| (II) bone-length consistency | [`bone_length/`](bone_length/README.md) | skeleton fitted with time-varying segment scales | `bone_fit/<stem>_bone_length.npz` |
| (III) joint range of motion | [`range_of_motion/`](range_of_motion/README.md) | joint angles of the fitted skeleton | `baseline/<stem>_opensim_skeleton.mot` |
| (IV) self-collision | [`self_collision_bvh/`](self_collision_bvh/README.md) | self-intersection of the SMPL-X body mesh | `gvhmr/<stem>/hmr4d_results.pt` |
| (V) kinematic extremes | [`kinematic_extreme/`](kinematic_extreme/README.md) | joint and segment velocities | `baseline/<stem>_opensim_skeleton_grid.mot` |
| (VI) motion smoothness | [`motion_smoothness/`](motion_smoothness/README.md) | joint angular accelerations | `baseline/<stem>_opensim_skeleton_grid.mot` |

Every score lies in [0, 100] (higher is more plausible). Apart from bone length, each metric turns a
per-frame severity `m_t` and a flag `b_t` into a penalty of this form (paper, supplementary section F.1;
self-collision counts only its severe frames in `r`):

```
r = mean_t b_t                       frequency of flagged frames
s = sum_t m_t / max(1, sum_t b_t)    severity
p = (longest run of b_t = 1) / T     persistence
score = 100 * (1 - clip(0.5 r + 0.3 s + 0.2 p, 0, 1))
```

Run the command lines from the repository root (`python -m metrics.<folder>.<module>`). The extra-limbs
folder name contains a hyphen, so that script is run by its path:
`python metrics/multi-limbs/eval_extra_limbs.py`.
