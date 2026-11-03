# (IV) Self-collision: paper implementation

The original HumanScore paper implementation of metric (IV). It measures how much of the reconstructed
body mesh intersects itself: a body part passing through another is impossible for a real person.

HumanScore runs this code only with `--paper-scores`, on GVHMR's world-frame SMPL-X mesh
(`<S>/gvhmr/<stem>/hmr4d_results.pt`). The default self-collision score is
`hmb/scoring/self_collision.py`.

## Definition

For each frame `t`, the [torch-mesh-isect](https://github.com/vchoutas/torch-mesh-isect) BVH finds the
pairs of intersecting triangles. Pairs whose triangles share a vertex are neighbours on the surface and
are dropped. `M_t` is the fraction of the mesh's faces in a remaining pair (0 when the frame has fewer
than `min_num_pairs` pairs or a ratio below `min_face_ratio`). Then

```
m_t = clip((M_t - tau_mild) / (tau_severe - tau_mild), 0, 1)       a frame with m_t > 0 is "colliding"
r_severe = fraction of frames with M_t >= tau_severe
s = sum_t m_t / max(1, number of colliding frames)
p = (longest run of colliding frames) / T
score = 100 * (1 - clip(0.5 r_severe + 0.3 s + 0.2 p, 0, 1))
```

| constant | value | meaning |
|---|---|---|
| `tau_mild` | 0.01 | a frame collides when more than 1 % of the faces intersect |
| `tau_severe` | 0.03 | 3 % or more gives full severity and counts as severe |
| `min_num_pairs`, `min_face_ratio` | 1, 0 | noise filter (command-line defaults, used by HumanScore) |
| weights | 0.5 / 0.3 / 0.2 | `--alpha`, `--beta`, `--gamma`; `--delta` (default 0) adds the frequency of colliding frames |

## Run on its own

Needs CUDA, the compiled `mesh_intersection` extension (installed by the HumanScore installer) and the
licensed `SMPLX_NEUTRAL.npz` model. The model is looked up in `$HMB_SMPLX_MODEL` (the file),
`$HMB_SMPLX_MODEL_DIR` (its folder), `outputs/assets/smplx/` and the GVHMR body-model folder. From the
repository root:

```bash
python -m metrics.self_collision_bvh.main \
    --input_path humanscore_output/clip/gvhmr/clip/hmr4d_results.pt \
    --output_dir results/metrics/self_collision_bvh
```

| option | default | meaning |
|---|---|---|
| `--input_path` | - | one `.pt` (GVHMR) or `.pkl` (PromptHMR-style) file |
| `--input_dir` | `results/baseline` | score every `.pt`/`.pkl` under this folder (used without `--input_path`) |
| `--output_dir` | `results/metrics/self_collision_bvh` | scores |
| `--use_incam` | off | use GVHMR's camera-frame instead of world-frame body parameters |
| `--person_id` | first | person of a `.pkl` file |
| `--to_smpl` | off | map the mesh to SMPL topology first |
| `--min_face_ratio`, `--min_num_pairs` | 0, 1 | noise filter |
| `--max_collisions` | 150 | BVH pair slots per triangle |
| `--alpha`, `--beta`, `--gamma`, `--delta` | 0.5, 0.3, 0.2, 0 | weights |

Every frame is checked. Frames are processed in chunks so that the padded BVH output stays below
256 MiB (five frames for SMPL-X); chunking does not change the score.

## Output

* `<output_dir>/<sequence>_self_collision_bvh.json`: `self_collision_bvh_score` (0-100, higher means
  fewer collisions), `r_severe` (also as `penetration_frame_rate`), `r_mild`, `severity_mean` (`s`),
  `continuity` (`p`), `avg_M_t`, `face_collision_ratio`, `avg_pairs_per_frame` and the settings.
* `<output_dir>/self_collision_bvh_scores_summary.csv`: one row per sequence.
* `<output_dir>/all_self_collision_bvh_results.json`: every result, including failures.

## Files

* `smpl_utils.py`: SMPL-X meshes from GVHMR or PromptHMR results.
* `bvh_metric.py`: per-frame self-intersection with the BVH.
* `scoring.py`: the video score.
* `main.py`: `evaluate_sequence` and the command line.
