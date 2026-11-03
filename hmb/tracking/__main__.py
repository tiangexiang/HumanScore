"""``python -m hmb.tracking VIDEO --out DIR [--backend sam3.1|yolo]``: track and select people.

Writes ``DIR/tracks.json`` + ``DIR/masks.npz`` (all tracks), ``DIR/selection.json``
(people kept by the selection rules, most prominent first), ``DIR/preview.mp4``
and ``DIR/preview_sheet.jpg`` (masks, boxes and ids; skipped people in gray) and,
with ``--person-videos``, ``DIR/person_<id>.mp4`` (others masked gray) and
``DIR/person_<id>_boxes.json`` for every selected person.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .api import BACKENDS, PREVIEW_NAME, SHEET_NAME, track_people
from .render import write_person_video, write_preview
from .sam3_backend import SAM3CheckpointError, SAM3SetupError, check_sam3
from .tracks import MIN_VISIBLE, describe_selection, screen_people

SELECTION_NAME = "selection.json"


def build_parser() -> argparse.ArgumentParser:
    """The command-line options (see the module docstring)."""
    p = argparse.ArgumentParser(prog="python -m hmb.tracking",
                                description="Track every person in a video (SAM 3.1 or YOLO) and select the "
                                            "people to score; optionally write one masked video per person.")
    p.add_argument("video", nargs="?", help="input video")
    p.add_argument("--out", help="output directory")
    p.add_argument("--backend", "--tracker", dest="backend", default="sam3.1", choices=BACKENDS,
                   help="tracker (default: sam3.1)")
    p.add_argument("--min-duration", type=float, default=1.0, help="min visible seconds of a person (1.0)")
    p.add_argument("--min-visible", type=float, default=MIN_VISIBLE,
                   help=f"min fraction of frames a person is visible ({MIN_VISIBLE:g}; below about 0.25 the "
                        "bone-length score is unsupported)")
    p.add_argument("--max-people", type=int, default=10, help="keep at most this many people (10)")
    p.add_argument("--min-height", type=float, default=0.1,
                   help="min median box height as a fraction of the frame height (0.1)")
    p.add_argument("--min-relative-height", type=float, default=0.25,
                   help="min median box height relative to the tallest kept person (0.25)")
    p.add_argument("--person-videos", action="store_true", help="also write one masked video per selected person")
    p.add_argument("--no-stitch", action="store_true", help="do not re-join fragments of the same person")
    p.add_argument("--no-preview", action="store_true", help="do not write preview.mp4 / preview_sheet.jpg")
    p.add_argument("--yolo-weights", default=None, help="YOLO *-seg.pt weights (default: yolov8x-seg.pt)")
    p.add_argument("--sam3-python", default=None, help="python of the SAM 3.1 environment ($HMB_SAM3_PYTHON)")
    p.add_argument("--sam3-checkpoint", default=None, help="local sam3.1_multiplex.pt ($HMB_SAM3_CHECKPOINT)")
    p.add_argument("--sam3-max-objects", type=int, default=16,
                   help="max objects one SAM 3.1 session tracks (16)")
    p.add_argument("--sam3-chunk-frames", type=int, default=600,
                   help="track longer videos in SAM 3.1 sessions of this many frames (600; 0 = one session)")
    p.add_argument("--check-sam3", action="store_true", help="only check the SAM 3.1 setup and exit")
    p.add_argument("--force", action="store_true", help="re-run tracking even if DIR has a result")
    p.add_argument("--quiet", action="store_true", help="do not print the tracker's progress")
    return p


def main(argv=None) -> int:
    """Track, select and write the outputs; returns the exit status (2 when SAM 3.1 is not set up)."""
    args = build_parser().parse_args(argv)
    try:
        if args.check_sam3:
            print(json.dumps(check_sam3(args.sam3_python, args.sam3_checkpoint), indent=1))
            return 0
        if not args.video or not args.out:
            build_parser().error("VIDEO and --out are required")
        out = Path(args.out)
        tracks = track_people(args.video, out, backend=args.backend, force=args.force, quiet=args.quiet,
                              stitch=not args.no_stitch, preview=False, max_objects=args.sam3_max_objects,
                              sam3_chunk_frames=args.sam3_chunk_frames, sam3_python=args.sam3_python,
                              sam3_checkpoint=args.sam3_checkpoint, yolo_weights=args.yolo_weights)
    except (SAM3SetupError, SAM3CheckpointError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    selected, skipped = screen_people(tracks, min_duration=args.min_duration, min_visible=args.min_visible,
                                      max_people=args.max_people, min_height=args.min_height,
                                      min_relative_height=args.min_relative_height)
    (out / SELECTION_NAME).write_text(json.dumps({
        "video": tracks.video, "backend": tracks.backend,
        "rules": {"min_duration": args.min_duration, "min_visible": args.min_visible,
                  "max_people": args.max_people, "min_height": args.min_height,
                  "min_relative_height": args.min_relative_height},
        "selected": selected, "skipped": skipped}, indent=1))
    print("\n".join(describe_selection(tracks, selected, skipped)))
    if not args.no_preview:
        write_preview(args.video, tracks, out / PREVIEW_NAME, selected=[s["id"] for s in selected],
                      sheet_path=out / SHEET_NAME)
        print(f"preview: {out / PREVIEW_NAME}")
    if args.person_videos:
        for s in selected:
            pid = s["id"]
            path = write_person_video(args.video, tracks, pid, out / f"person_{pid}.mp4")
            tracks.write_person_boxes(pid, out / f"person_{pid}_boxes.json")
            print(f"person {pid}: {path}")
    print(f"{len(selected)} of {len(tracks.ids)} tracked people selected -> {out / SELECTION_NAME}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
