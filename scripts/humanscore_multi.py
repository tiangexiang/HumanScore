#!/usr/bin/env python
"""HumanScore for every person in a video, end to end.

    python scripts/humanscore_multi.py VIDEO [--out DIR] [--tracker sam3.1|yolo] [--min-duration 1.0]
                                             [--min-visible 0.3] [--max-people 10] [--moving-cam]
                                             [--paper-scores] [--force] [--sample-fps 24] [--max-side 1280]

Tracks every person through the video (SAM 3.1 with the text prompt "person",
or YOLO segmentation + BoT-SORT), keeps the people who are visible long enough
(by default at least 1 s and in at least 30 % of the frames: the bone-length
score needs the person's keypoints in 25 % of all analysed frames) and large
enough, and scores them one by one: for each person a full-frame copy
of the video with the other tracked people painted flat gray is reconstructed
and scored by the single-person pipeline (scripts/humanscore_single.py), with
the 3-D keypoint detector and the body-mesh tracker following that person's track.

Outputs in DIR (default ./humanscore_output/<video stem>_multi):
  tracks/                 tracks.json, masks.npz, preview.mp4 (scored people in
                          colour, skipped in gray), preview_sheet.jpg, selection.json
  people/person_<id>/     person_<id>.mp4 (other people masked), person_boxes.json,
                          humanscore.json, summary.txt and the reconstruction
  humanscore_multi.json   every person: frames visible, duration, scores, tiers,
                          overall, status, reasons, paths; skipped people with the reason
  summary.txt             the table printed at the end
  logs/tracking.log       the tracker's output
Running the command again reuses the tracks and every finished step.

The sam3.1 tracker needs the sam3 package (bash install.sh --with-sam3, or a
separate environment in $HMB_SAM3_PYTHON) and access to the gated
facebook/sam3.1 checkpoint; see the README.
--tracker yolo needs no extra setup.  Exit status: 0 = every selected person was
scored completely, 1 = no person to score or some scores missing, 2 = error,
130/143 = interrupted by Ctrl-C/SIGTERM (tracker and engines are stopped; run again to resume).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# make ``hmb`` importable when the script is run from any directory
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hmb import process_groups  # noqa: E402
from hmb.multi import (MULTI_RESULT_NAME, TRACKERS, TRACKING_LOG_NAME, default_out_dir,  # noqa: E402
                       format_multi_table, score_people)
from hmb.pipeline import SUMMARY_NAME  # noqa: E402
from hmb.tracking import SAM3CheckpointError, SAM3SetupError  # noqa: E402
from hmb.tracking.tracks import MIN_VISIBLE  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """The command-line options (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip(),
                                 epilog="Each person is scored like scripts/humanscore_single.py scores a "
                                        "single-person video; see that script for the scores.")
    ap.add_argument("video", help="input video (mp4, mov, avi, mkv)")
    ap.add_argument("--out", metavar="DIR", default=None,
                    help="output directory (default: ./humanscore_output/<video stem>_multi)")
    ap.add_argument("--tracker", choices=TRACKERS, default="sam3.1",
                    help="multi-person tracker: sam3.1 (SAM 3.1, text prompt 'person'; default) or yolo "
                         "(YOLO segmentation + BoT-SORT, no extra setup)")
    ap.add_argument("--min-duration", metavar="S", type=float, default=1.0,
                    help="skip people visible for less than S seconds. Default: 1.0")
    ap.add_argument("--min-visible", metavar="F", type=float, default=MIN_VISIBLE,
                    help="skip people visible in less than this fraction of the frames. The bone-length "
                         "score needs the person's keypoints in at least 25%% of all analysed frames (else it "
                         "is 'unsupported'), and the keypoint detector misses a tracked person in some frames, "
                         "so lower values can admit people whose bone length is unsupported. "
                         f"Default: {MIN_VISIBLE:g}")
    ap.add_argument("--max-people", metavar="N", type=int, default=10,
                    help="score at most the N most prominent people (largest and longest visible; "
                         "0 = no limit). Default: 10")
    ap.add_argument("--moving-cam", action="store_true",
                    help="the camera moves: estimate its rotation for the body mesh (needs pycolmap). Default: "
                         "static camera, as in the HumanScore evaluation")
    ap.add_argument("--sample-fps", metavar="FPS", type=float, default=24.0,
                    help="analysed frame rate (source frames nearest to this rate; 0 = all frames). Default: 24")
    ap.add_argument("--max-side", metavar="PX", type=int, default=1280,
                    help="downscale frames to this longer side (0 = source resolution). Default: 1280")
    ap.add_argument("--paper-scores", action="store_true",
                    help="also compute the six scores of the original paper implementation for every person")
    ap.add_argument("--force", action="store_true",
                    help="track and recompute everything instead of reusing earlier results")
    return ap


def main(argv=None) -> int:
    """Track and score every person and print the summary table; returns the exit status."""
    args = build_parser().parse_args(argv)
    if args.sample_fps < 0 or args.max_side < 0:
        print("error: --sample-fps and --max-side must be >= 0", file=sys.stderr)
        return 2
    if args.min_duration < 0 or not 0 <= args.min_visible <= 1 or args.max_people < 0:
        print("error: --min-duration and --max-people must be >= 0 and --min-visible in [0, 1]", file=sys.stderr)
        return 2
    out = Path(args.out) if args.out else default_out_dir(args.video)
    try:
        result = score_people(args.video, out, tracker=args.tracker, min_duration=args.min_duration,
                              min_visible=args.min_visible, max_people=args.max_people or None,
                              static_cam=not args.moving_cam, sample_fps=args.sample_fps or None,
                              max_side=args.max_side or None, paper_scores=args.paper_scores,
                              force=args.force, quiet=True)
    except KeyboardInterrupt as exc:  # Ctrl-C, or SIGTERM/SIGHUP (process_groups.Terminated)
        print(f"{process_groups.describe(exc)}; run the same command again to resume", file=sys.stderr)
        return process_groups.exit_status(exc)
    except (SAM3SetupError, SAM3CheckpointError) as exc:  # the message explains the SAM 3.1 setup
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        log = out.expanduser().resolve() / TRACKING_LOG_NAME
        if log.is_file():
            print(f"tracking log: {log}", file=sys.stderr)
        return 2
    print()
    print(format_multi_table(result))
    root = out.expanduser().resolve()
    print(f"\nresults: {root / MULTI_RESULT_NAME}\nsummary: {root / SUMMARY_NAME}")
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
