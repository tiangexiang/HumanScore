"""Multi-person tracking for HumanScore (preprocessing of multi-person videos, used by :mod:`hmb.multi`).

::

    from hmb.tracking import track_people, select_people, write_person_video

    tracks = track_people("clip.mp4", "out/tracks", backend="sam3.1")   # or backend="yolo"
    for person in select_people(tracks, min_duration=1.0, min_visible=0.3, max_people=10):
        pid = person["id"]
        write_person_video("clip.mp4", tracks, pid, f"out/people/person_{pid}/person_{pid}.mp4")
        tracks.write_person_boxes(pid, f"out/people/person_{pid}/person_boxes.json")

Backends: ``sam3.1`` (SAM 3.1 Object Multiplex with the text prompt "person",
run as a subprocess, in this environment or a separate one, see :mod:`hmb.tracking.sam3_backend`)
and ``yolo`` (ultralytics YOLO segmentation + BoT-SORT, no extra setup).
Command line: ``python -m hmb.tracking VIDEO --out DIR [--backend sam3.1|yolo]``.

Modules: :mod:`~hmb.tracking.api` (``track_people``), :mod:`~hmb.tracking.tracks` (storage,
stitching, selection), :mod:`~hmb.tracking.render` (person videos and previews) and the two
backends.
"""
from .api import BACKENDS, track_people
from .render import write_person_video, write_preview
from .sam3_backend import SAM3CheckpointError, SAM3SetupError, check_sam3
from .tracks import (MIN_VISIBLE, Tracks, describe_selection, load_tracks, rle_decode, rle_encode, screen_people,
                     select_people, stitch_fragments)

__all__ = [
    "BACKENDS",
    "MIN_VISIBLE",
    "SAM3CheckpointError",
    "SAM3SetupError",
    "Tracks",
    "check_sam3",
    "describe_selection",
    "load_tracks",
    "rle_decode",
    "rle_encode",
    "screen_people",
    "select_people",
    "stitch_fragments",
    "track_people",
    "write_person_video",
    "write_preview",
]
