# HumanScore metrics

This page defines the six scores of HumanScore, as computed by `hmb/scoring/`. They are the six
metrics of the [HumanScore paper](https://arxiv.org/abs/2604.20157), in three tiers:

| tier | metric | evidence | code |
|---|---|---|---|
| anatomy | (I) extra limbs | HADM-G human-artefact detector, per frame | `hmb/scoring/extra_limbs.py` |
| anatomy | (II) bone length | MeTRAbs 3-D joint centres | `hmb/scoring/bone_length.py` |
| kinematic | (III) joint range | joint angles of the fitted skeleton | `hmb/scoring/joint_range.py` |
| kinematic | (IV) self-collision | GVHMR SMPL-X body mesh | `hmb/scoring/self_collision.py` |
| kinetic | (V) kinematic extremes | fitted skeleton on a uniform 24 Hz grid | `metrics/kinematic_extreme/` |
| kinetic | (VI) motion smoothness | fitted skeleton on a uniform 24 Hz grid | `metrics/motion_smoothness/` |

Every score lies in [0, 100], and higher is more plausible. A tier score is the mean of its two
metrics, and the overall score is the mean of all six (the paper's Table 1). A tier or overall
score is null when one of its metrics has no score.

## Reconstruction

One video and one person give five kinds of evidence. All of it is computed on the same frames.

* **Frame plan.** The video is analysed on the source frames nearest a uniform 24 Hz grid, with
  their true timestamps. Frames are downscaled to at most 1280 px on the long side. A 24 fps clip
  and a 60 fps clip are therefore looked at the same number of times per second.
* **MeTRAbs** (`bml_movi_87` skeleton: 87 keypoints, including the anatomical joint centres) gives
  3-D and 2-D keypoints per frame. The person followed is the largest person seen near the start
  of the clip. When that person is missing from a frame, the frame becomes a gap; the tracker
  never switches to someone else. `scripts/humanscore_multi.py` instead passes each person's track.
* **Skeleton fit.** A MuJoCo biomechanical skeleton (the Rajagopal joint set, with OpenSim
  coordinate conventions) is fitted to the keypoints as one continuous trajectory (10,000
  optimizer steps). The fit has no joint-angle bounds, so joint-range violations stay visible. The
  trajectory is exported at the analysed frames and on a uniform 24 Hz grid.
* **GVHMR** reconstructs a world-frame SMPL-X body mesh per frame. It assumes a static camera
  unless `--moving-cam` is given.
* **HADM-G** (human-artefact detector) runs on every analysed frame.

## From frames to a score

Each metric measures every analysed frame `t` of `T`: a severity `m_t` in [0, 1] and an abnormal
flag `b_t`. The paper (supplementary section F.1) combines them into one score:

```
r = mean_t b_t                       frequency: how often
s = sum_t m_t / max(1, sum_t b_t)    severity: how bad
p = L_max / T                        persistence: longest run of flagged frames
D = 0.5 r + 0.3 s + 0.2 p,           score = 100 (1 - clip(D, 0, 1))
```

Kinematic extremes and motion smoothness use this formula as written. Extra limbs uses a graded
form of it (I), and bone length uses the paper's mean-deviation form instead (II). Joint range and
self-collision (`hmb/scoring/aggregate.py`) define two terms more precisely:

* `s` is the mean severity of the **flagged** frames: `sum_t m_t b_t / max(1, sum_t b_t)`.
  Harmless sub-threshold noise therefore never adds up to a penalty on a long clip.
* Frames without an observation (no person detected, a joint outside the image) are removed
  before `r`, `s` and `p` are computed. The fraction of usable frames is reported as `coverage`.

## The metrics

### (I) Extra limbs

* **Evidence.** `M_t` is the highest HADM-G confidence over the four classes "human with extra
  {hand, arm, leg, feet}" in frame `t`.
* **Per frame.** A graded membership `m_t = clip((M_t - 0.001) / (0.03 - 0.001), 0, 1)`, which is
  0 at or below the detector's output floor of 0.001.
* **Aggregation.** The paper's r/s/p with each hard flag replaced by its membership:
  `r = sum m_t / T`, `s = sum m_t^2 / sum m_t`, `p = max over runs (m_t > 0) of sum m_t / T`,
  `D = 0.5 r + 0.3 s + 0.2 p`. A weak detection therefore counts much less than a confident one.
  When every flagged frame has full severity, this equals the paper's formula.
* **Constants.** 0.001 is the smallest confidence HADM-G reports; 0.03 is the paper's severe
  threshold. Detections in a single frame count like any other.

### (II) Bone length

The lengths are read directly from the MeTRAbs joint centres, without a skeleton fit.

* **Evidence.** 11 segments: thigh, shank, upper arm and forearm (left and right), the trunk
  (pelvis centre to neck), hip width and shoulder width. Hands, feet and head are short and poorly
  observed, so they are not scored.
* **Observation mask.** A segment counts in a frame only when the person was detected and both
  endpoints lie inside the image, at least 2 % of its size from the border. A body cut off by the
  frame is a framing problem, not an anatomy error. With fewer than 25 % usable frames, the score
  is `unsupported`.
* **Body scale removed.** The overall size of a monocular reconstruction drifts with depth and
  pose. Per frame, the log body scale `g(t)` is the median over the observed segments of
  `log l_b(t) - log L_b`, where `L_b` is the segment's median length over the clip. The residual
  `r_b(t) = log l_b(t) - log L_b - g(t)` does not change when the whole skeleton is scaled, and the
  median keeps one stretched limb from being absorbed into `g`.
* **Glitch filter.** `r_b` is median-filtered over a centred window of 0.125 s (3 frames at
  24 Hz). A one-frame estimation glitch is removed; a deformation of two or more frames stays.
* **Score.** `d_b(t) = exp(|r_b(t)|) - 1` is the relative length change.
  `E = mean_b mean_t d_b(t)` over the observed frames of each segment, and
  `score = 100 (1 - clip(E / 0.45, 0, 1))`.
* **Constant.** `tau = 0.45`. The paper's 0.15 assumes a noise-free fitted skeleton; per-frame
  monocular keypoints always carry some noise, and with 0.45 that noise costs only a few points.

### (III) Joint range

The paper's formula: 21 degrees of freedom (DoFs) of the hips, knees, ankles, lumbar spine,
shoulders and elbows, a 15 degree tolerance, the half-range severity
`m = min(excess / (0.5 (hi - lo)), 1)`, a flag threshold of 0.05 and r/s/p aggregation. The angles
are measured so that only real limit violations count:

* **Equivalent angles.** The fit is unbounded, so it can return, for example, an elbow angle of
  -341 degrees, which is the same pose as +19 degrees. Every angle is mapped to its 360-degree
  equivalent closest to the centre of its range. For each 3-DoF ball joint, the Euler solution
  with the smaller violation is kept.
* **Flipped limbs.** The fit sometimes rotates an upper arm or thigh by about 180 degrees around
  its long axis and bends the elbow or knee backwards. For each violating frame and limb, other
  anatomical configurations are fitted to that limb's keypoints (the flipped limb, the forearm or
  foot reset to neutral, the fit clipped into the limits, a bounded fit, the neighbouring frame).
  The least violating one replaces the fit if it matches the keypoints at most 10 mm (RMS) worse.
* **Unobserved joints.** A DoF whose keypoints lie outside the image is not scored in that frame,
  and frames without a detected person are removed.
* **Worst joint per frame.** `m_t` is the largest severity over the 21 DoFs, so one joint out of
  range flags the frame.
* **Limits.** The paper's Rajagopal ranges, widened only where the clinical normal range (AAOS)
  lies beyond limit + tolerance. Only shoulder elevation changes: arm flexion up to 180 degrees
  (paper 90) and abduction to 180 degrees (paper 120). The table is `LIMITS_DEG` in
  `hmb/scoring/joint_range.py`.

### (IV) Self-collision

The metric measures how deep one body part sinks into another, on GVHMR's SMPL-X mesh.

* **Unobserved geometry removed.** The eyeballs (which intersect their sockets in every pose) and
  the finger vertices are removed. GVHMR estimates the 21 body joints only, so its fingers keep a
  fixed half-curled pose that carries no information from the video.
* **Non-adjacent parts only.** Faces are grouped into 14 body parts by the SMPL-X skinning
  weights. Only pairs of parts that are neither the same part nor parent and child count. Skin
  folding at a bent joint is thus left to joint range (III), and the rest pose has no counted pair.
* **Depth.** A vertex of part A is inside part B when its generalised winding number with respect
  to B exceeds 1/2, and its depth is its exact distance to B's surface. Only vertices connected to
  an actual triangle intersection count. The frame depth is the largest depth over all colliding
  part pairs.
* **Per frame and aggregation.** `m_t = clip((depth - contact) / (severe - contact), 0, 1)`,
  `b_t = depth > contact`, and `D = 0.5 r_severe + 0.3 s + 0.2 p` with
  `r_severe = mean_t (depth >= severe)`, as in the paper.
* **Constants.** `contact = 2 cm`: hands on hips, clapping or crossed legs make the reconstructed
  surfaces overlap a little, and this tolerance covers that plus the error of a monocular body
  mesh. `severe = 5 cm`: the contact tolerance plus about the radius of a forearm.

### (V) Kinematic extremes

The paper's code (`metrics/kinematic_extreme/`) on the skeleton trajectory sampled on the uniform
24 Hz grid. Finite differences need a uniform time step, and the common grid makes the result
independent of the video's frame rate.

* **Evidence.** The angular velocity of every joint coordinate (central differences) and the
  linear speed of every body segment's centre of mass (forward kinematics of the skeleton).
* **Per frame.** `u = clip((|x| - x_max) / (0.5 x_max), 0, 1)` for each coordinate and segment.
  `m_t` is the mean of the joint term and the segment term, each a mean over its coordinates or
  segments, and `b_t = m_t > 0.05`.
* **Aggregation.** The paper's r/s/p as written above.
* **Limits** (`metrics/kinematic_extreme/definition.py`): from 360 to 4000 degrees/s per joint
  coordinate and from 12 to 45 m/s per segment, at the upper end of human capability (fast throws,
  kicks and sprints).

### (VI) Motion smoothness

The paper's code (`metrics/motion_smoothness/`) on the same 24 Hz grid.

* **Evidence.** The angular acceleration of every joint coordinate (central differences).
* **Per frame.** `u = clip((|a| - a_max) / (0.5 a_max), 0, 1)` per coordinate, `m_t` is the mean
  over the coordinates and `b_t = m_t > 0.05`.
* **Aggregation.** The paper's r/s/p as written above. The paper's optional jerk-energy term is
  not used.
* **Limits** (`metrics/motion_smoothness/definition.py`): from 8,000 to 60,000 degrees/s² per
  joint coordinate. Fast kicks and throws stay below them, so only jerky, unnatural spikes count.

## Constants

| metric | constant | value | where it comes from |
|---|---|---|---|
| all | aggregation weights (r, s, p) | 0.5 / 0.3 / 0.2 | paper |
| extra limbs | detector floor / full severity | 0.001 / 0.03 | HADM-G output threshold / paper |
| bone length | `tau`, glitch window, border inset, minimum coverage | 0.45, 0.125 s, 2 %, 25 % | set so that ordinary keypoint noise costs a few points / one frame at 24 Hz / framing / minimum evidence |
| joint range | tolerance, flag threshold, limits | 15 degrees, 0.05, Rajagopal + AAOS shoulder | paper / clinical norms |
| joint range | limb-flip margin | 10 mm RMS | below the 3-D error of monocular keypoints |
| self-collision | contact / severe depth | 2 cm / 5 cm | contact and mesh error / contact + forearm radius |
| kinematic extremes, smoothness | limits, flag threshold, excess fraction | `definition.py`, 0.05, 0.5 | paper |

All scores also depend on the frame plan (24 Hz, at most 1280 px). If you change it with
`--sample-fps` or `--max-side`, the constants no longer match the evidence exactly.

## Paper scores

`--paper-scores` (or `paper_scores=True` in Python) also computes the six metrics with the paper's
original code in `metrics/`, on the same reconstruction. The result appears as `paper_scores` in
`humanscore.json` (scores, tiers, overall, status, reasons), and each metric's own output is
written to `<session>/paper_metrics/<metric>/`. [metrics/README.md](../metrics/README.md) describes
each paper metric. Compared with the scores above:

* **Extra limbs**: a frame is flagged when the confidence exceeds 0.005 (full severity at 0.03),
  instead of the graded membership.
* **Bone length**: a second skeleton fit with segment scales that may change over time (written to
  `<session>/bone_fit/`, which makes the run longer); 9 bones of that fit; `tau = 0.15`; no
  observation mask and no body-scale removal.
* **Joint range**: the fitted angles as they are, without equivalent angles, limb flips or
  masks; `m_t` is the mean over the 21 DoFs; the paper's limits.
* **Self-collision**: the fraction of all SMPL-X faces in intersecting triangle pairs (eyeballs
  and fingers included), flagged above 1 % and fully severe at 3 %.
* **Kinematic extremes and motion smoothness**: the same code on the same input, so the same
  scores.

## Limitations

* The scores describe the **reconstructed** body. MeTRAbs keeps limb lengths fairly stable, so a
  limb stretched in the image changes its 3-D length less. GVHMR meshes rarely show one limb
  passing through the body. On hard motions (flips, cartwheels, fast spins), many joint and
  kinetic flags come from a failed reconstruction rather than from the video.
* The skeleton fit assumes a static camera. With `--moving-cam`, only the body mesh accounts for
  camera motion, so a moving camera can still raise segment speeds.
* One person is scored at a time. In crowded scenes, occlusion degrades every estimator. The
  multi-person script masks the other tracked people but cannot recover hidden limbs, and a
  tracker can split one person into several identities, which are then scored separately.
