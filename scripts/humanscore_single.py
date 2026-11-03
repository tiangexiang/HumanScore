#!/usr/bin/env python
"""HumanScore for a video with one person, end to end.

    python scripts/humanscore_single.py VIDEO [--out DIR] [--moving-cam] [--paper-scores] [--force]
                                              [--sample-fps 24] [--max-side 1280]

Reconstructs the person (MeTRAbs 3-D keypoints, GVHMR body mesh, HADM extra-limb
detections, biomechanical skeleton fit) and computes the six HumanScore
plausibility scores (extra limbs, bone length, joint range, self-collision,
kinematic extremes, motion smoothness; 0-100, higher is more plausible), their
three tiers and the overall score.  --paper-scores adds the six scores of the
original paper implementation.

Outputs in DIR (default ./humanscore_output/<video stem>):
  humanscore.json   scores, tiers, overall, per-metric status and reasons, timings
  summary.txt       the table printed at the end
  scores.json       the detailed scorer report
  logs/             the full console log (engine logs: cache/engines/)
plus the reconstruction artifacts; running the command again reuses them.

If several people are visible, the largest person near the start is followed;
use scripts/humanscore_multi.py to score every person.  Exit status: 0 = all
requested scores computed, 1 = some scores missing (see summary), 2 = error,
130/143 = interrupted by Ctrl-C/SIGTERM (the engines are stopped; run again to resume).
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
from hmb.pipeline import LOG_NAME, RESULT_NAME, SUMMARY_NAME, default_out_dir, format_table, score_video  # noqa: E402
from hmb.report import is_complete  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """The command-line options (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip(),
                                 epilog="Scores are computed for one person; see scripts/humanscore_multi.py for "
                                        "videos with several people.")
    ap.add_argument("video", help="input video (mp4, mov, avi, mkv)")
    ap.add_argument("--out", metavar="DIR", default=None, help="output directory (default: ./humanscore_output/<video stem>)")
    ap.add_argument("--moving-cam", action="store_true",
                    help="the camera moves: estimate its rotation for the body mesh (needs pycolmap). Default: "
                         "static camera, as in the HumanScore evaluation")
    ap.add_argument("--sample-fps", metavar="FPS", type=float, default=24.0,
                    help="analysed frame rate (source frames nearest to this rate; 0 = all frames). Default: 24")
    ap.add_argument("--max-side", metavar="PX", type=int, default=1280,
                    help="downscale frames to this longer side (0 = source resolution). Default: 1280")
    ap.add_argument("--paper-scores", action="store_true",
                    help="also compute the six scores of the original paper implementation (adds a second skeleton fit)")
    ap.add_argument("--force", action="store_true", help="recompute everything instead of reusing earlier results")
    return ap


def main(argv=None) -> int:
    """Score one video and print the summary table; returns the exit status."""
    args = build_parser().parse_args(argv)
    if args.sample_fps < 0 or args.max_side < 0:
        print("error: --sample-fps and --max-side must be >= 0", file=sys.stderr)
        return 2
    out = Path(args.out) if args.out else default_out_dir(args.video)
    try:
        result = score_video(args.video, out, static_cam=not args.moving_cam, sample_fps=args.sample_fps or None,
                             max_side=args.max_side or None, paper_scores=args.paper_scores,
                             force=args.force, quiet=True)
    except KeyboardInterrupt as exc:  # Ctrl-C, or SIGTERM/SIGHUP (process_groups.Terminated)
        print(f"{process_groups.describe(exc)}; run the same command again to resume", file=sys.stderr)
        return process_groups.exit_status(exc)
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        log = out.expanduser().resolve() / LOG_NAME
        if log.is_file():
            print(f"log: {log}", file=sys.stderr)
        return 2
    print()
    print(format_table(result))
    session = Path(result["files"]["session"])
    print(f"\nresults: {session / RESULT_NAME}\nsummary: {session / SUMMARY_NAME}")
    return 0 if is_complete(result) else 1


if __name__ == "__main__":
    raise SystemExit(main())
