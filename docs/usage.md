Using HumanScore
================

This page explains the output in detail and the other ways to run HumanScore. For installation and
the basic command, see the [README](../README.md); for how each score is computed, see
[metrics.md](metrics.md).

## Reading the results

`humanscore.json` looks like this:

```json
{
 "video": "/path/to/video.mp4",
 "scores": {"extra_limbs": 100.0, "bone_length": 98.52, "joint_range": 100.0,
            "self_collision": 100.0, "kinematic_extremes": 100.0, "motion_smoothness": 100.0},
 "tiers": {"anatomy": 99.26, "kinematic": 100.0, "kinetic": 100.0},
 "overall": 99.75,
 "status": {"extra_limbs": "ok", "bone_length": "ok", "joint_range": "ok",
            "self_collision": "ok", "kinematic_extremes": "ok", "motion_smoothness": "ok"},
 "reasons": {},
 "person": {"selection": "track", "frames_analysed": 141, "frames_found": 141,
            "frames_with_several_people": 0},
 "files": {"session": "...", "report": "...", "summary": "...", "log": "...", "engine_logs": "..."},
 "timings": {"total_seconds": 164, "reconstruction_seconds": 157, "scoring_seconds": 6}
}
```

* **`scores`**: 100 means that nothing implausible was found. A score drops when a problem appears
  in many frames, when it is severe, and when it lasts long.
* **`tiers`** and **`overall`**: the means described above.
* **`status`**: `ok` when the score was computed. Otherwise it says why the score is missing:
  `unsupported` (too little evidence, for example the person is visible in too few frames),
  `missing_input` (a reconstruction step produced nothing) or `error` (see `scores.json` and the
  log). `reasons` then gives one line, for example
  `"bone_length": "keypoint coverage 0.18 < 0.25 of the analysed frames"`. A tier or the overall
  score is `null` when one of its scores is missing.
* **`person`**: in how many of the analysed frames the person was found. If several people are
  visible, the script follows the largest person near the start of the video.

The script exits with status 0 when every score was computed, 1 when some are missing and 2 on an
error. Ctrl-C stops it cleanly, and running it again continues where it stopped.

The scores describe the reconstructed 3-D body, not the pixels. On hard real videos (fast spins,
heavy occlusion, motion blur) the reconstruction itself can fail and lower a score, so compare
videos of similar motions.

## Options of `humanscore_single.py`

| option | meaning |
|---|---|
| `--out DIR` | save the results in `DIR` (default: `humanscore_output/<video name>`) |
| `--moving-cam` | the camera moves (the default assumes a static camera) |
| `--paper-scores` | also compute the six scores with the paper's original code; they appear under `paper_scores` ([details](metrics.md#paper-scores)) |
| `--force` | recompute everything instead of reusing earlier results |
| `--sample-fps FPS`, `--max-side PX` | analysed frame rate (default 24) and image size (default 1280 px); the score constants are set for these defaults |

`python scripts/humanscore_single.py --help` shows them all.

## Several people in a video

```bash
python scripts/humanscore_multi.py video.mp4                  # SAM 3.1 (needs --with-sam3)
python scripts/humanscore_multi.py video.mp4 --tracker yolo   # YOLO, no extra setup
```

The script first tracks every person through the video. It keeps people who are visible for at
least 1 s and in at least 30 % of the frames, and who are not too small; at most 10, the most
prominent first. It then scores each person on its own, with the other people painted gray.

The results are saved in `humanscore_output/video_multi/`:

* `summary.txt`: one row per person, with the six scores, the overall score and the status.
* `humanscore_multi.json`: for each person, the same fields as `humanscore.json` plus the frames
  in which the person is visible. People who were not scored are listed under `skipped` with the
  reason.
* `people/person_<id>/`: that person's video and its own `humanscore.json` and `summary.txt`.
* `tracks/preview.mp4`: the tracks drawn on the video. Check here that each person kept one id.

SAM 3.1 needs about 30 GB of GPU memory, and long videos are tracked in pieces of 600 frames. The
YOLO tracker needs no extra setup, but when people occlude each other it can split one person into
two ids, which are then scored separately. `--help` lists the selection options (`--max-people`,
`--min-duration`, `--min-visible`).

## More ways to run it

Run these from the repository root with the environment activated.

* **From Python**: `score_video` returns the content of `humanscore.json`, and `score_people` that
  of `humanscore_multi.json`. Both take the script options as keyword arguments, for example
  `paper_scores=True` or `static_cam=False`.

  ```python
  from hmb.pipeline import score_video

  result = score_video("video.mp4", "humanscore_output/video")
  print(result["overall"], result["scores"])
  ```

* **Many videos**: `python -m hmb.batch --manifest videos.json --root results/run1`, where
  `videos.json` is a list like `[{"id": "clip1", "video": "clip1.mp4"}, ...]`. Each model is loaded
  only once for the whole list. The scores of all videos are written to `results/run1/scores.csv`.
* **Single steps**: `python -m hmb.cli prepare` (reconstruction), `score` (scoring),
  `run-all` (both) and `doctor` (checks the installation). Add `--help` to any of them.
* **Tests**: `python -m pytest -q`. Tests that need the GPU or a video run only when you ask for
  them, with `HMB_RUN_GPU_TESTS=1` and `HMB_TEST_VIDEO=/path/to/clip.mp4`.

## Limitations

* The scores describe the reconstructed body. A failed reconstruction of a real video (fast spins,
  flips, heavy occlusion, motion blur) can lower the joint-range and kinetic scores.
* The skeleton fit assumes a static camera. `--moving-cam` corrects only the body mesh, so a
  moving camera can still lower the kinetic scores.
* The score constants are set for the default frame rate (24 fps) and image size (1280 px).
* People are scored one at a time. When people occlude each other, hidden limbs cannot be
  recovered, and a tracker can split one person into several ids.
