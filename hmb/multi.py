"""HumanScore for every person in one video (the API behind ``scripts/humanscore_multi.py``).

::

    from hmb.multi import score_people

    result = score_people("group.mp4", "humanscore_output/group_multi", tracker="sam3.1")
    for pid, person in result["people"].items():
        print(pid, person["overall"], person["status"])

The people are processed one by one:

1. **Track** every person through the video with :func:`hmb.tracking.track_people`
   (SAM 3.1 with the text prompt "person", or YOLO segmentation + BoT-SORT) into
   ``DIR/tracks/``: ``tracks.json``, ``masks.npz``, ``preview.mp4`` and
   ``preview_sheet.jpg`` (scored people in colour, skipped people in gray) and
   ``selection.json``.
2. **Select** the people to score (:func:`hmb.tracking.screen_people`): visible for
   at least ``min_duration`` seconds and in at least ``min_visible`` of the frames
   (default 0.3, :data:`hmb.tracking.tracks.MIN_VISIBLE`: the bone-length score needs
   keypoints in 25 % of all analysed frames), tall enough, at most ``max_people``
   (most prominent first).  The others are listed under ``skipped`` with the reason.
3. **Score** each selected person: ``DIR/people/person_<id>/person_<id>.mp4`` is the
   full-frame video with every other tracked person painted flat gray (so the
   camera geometry is unchanged), ``person_boxes.json`` the person's box per source
   frame.  :func:`hmb.pipeline.score_video` scores that video into the same
   directory (its session directory), with MeTRAbs and GVHMR following those boxes.  A person
   whose scoring fails is recorded with the error and the next one is processed.
4. Write ``DIR/humanscore_multi.json`` (:func:`score_people`'s result) and
   ``DIR/summary.txt`` (:func:`format_multi_table`).

Running the same call again reuses the tracks, the person videos and every
verified reconstruction step.
"""
from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

from hmb import process_groups
from hmb.pipeline import (RESULT_NAME, SUMMARY_NAME, _console, _write, build_cli_args, score_video,
                          validate_options)
from hmb.pipeline import LOG_NAME as PERSON_LOG_NAME
from hmb.report import is_complete, reason_lines
from hmb.scoring.score_session import METRICS, TIERS
from hmb.tracking.tracks import MIN_VISIBLE

# Files and folders of the output directory DIR (relative paths).
MULTI_RESULT_NAME = "humanscore_multi.json"
TRACKS_DIR = "tracks"
PEOPLE_DIR = "people"
SELECTION_NAME = "selection.json"      # in TRACKS_DIR
PERSON_BOXES_NAME = "person_boxes.json"  # in each person's directory
TRACKING_LOG_NAME = "logs/tracking.log"
TRACKERS = ("sam3.1", "yolo")
#: Metric column headers of the summary table.
SHORT_LABELS = {"extra_limbs": "ext.limbs", "bone_length": "bone.len", "joint_range": "joint.rng",
                "self_collision": "self-coll", "kinematic_extremes": "kin.extr", "motion_smoothness": "smooth"}
# tracker lines shown on the console of a quiet run (everything else goes to the tracking log)
TRACKING_PATTERNS = ((re.compile(r"^\[(sam3\.1|yolo|tracking)\] (.*)"), lambda m: f"  {m[1]}: {m[2]}"),)


def default_out_dir(video, root="humanscore_output") -> Path:
    """``./humanscore_output/<video stem>_multi`` (next to the single-person default, never inside it)."""
    from hmb.layout import sanitize_stem

    return Path(root) / f"{sanitize_stem(Path(video).stem)}_multi"


def person_dir(out_dir, person_id) -> Path:
    """``DIR/people/person_<id>``: the person's masked video and its session directory."""
    return Path(out_dir) / PEOPLE_DIR / f"person_{int(person_id)}"


def _number(name: str, value, *, upper: Optional[float] = None) -> float:
    """``value`` as a float; ``ValueError`` unless it is a finite number in ``[0, upper]``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not value >= 0 or math.isinf(value):
        raise ValueError(f"{name} must be a non-negative number, got {value!r}")
    if upper is not None and value > upper:
        raise ValueError(f"{name} must be at most {upper:g}, got {value!r}")
    return float(value)


def check_options(video: Path, out_dir: Path, *, tracker: str, min_duration, min_visible, max_people,
                  static_cam: bool, sample_fps, max_side, paper_scores: bool = False) -> str:
    """Reject invalid inputs before tracking starts; returns the canonical tracker name."""
    from hmb.tracking.api import normalize_backend

    if not video.is_file():
        raise FileNotFoundError(f"video not found: {video}")
    tracker = normalize_backend(tracker)
    _number("min_duration", min_duration)
    _number("min_visible", min_visible, upper=1.0)
    if max_people is not None and (isinstance(max_people, bool) or not isinstance(max_people, int) or max_people < 0):
        raise ValueError(f"max_people must be a non-negative integer (0 or None: no limit), got {max_people!r}")
    for name, value in (("sample_fps", sample_fps), ("max_side", max_side)):
        if value is not None:
            _number(name, value)
    # the per-person scoring options (e.g. a moving camera without pycolmap), checked on a stand-in session
    args = build_cli_args(video, person_dir(out_dir, 0), static_cam=static_cam, sample_fps=sample_fps,
                          max_side=max_side, paper_scores=paper_scores)
    validate_options(args)
    return tracker


# ----------------------------------------------------------------------------- per person
def _tracking_info(person: dict, fps: float) -> dict:
    """The tracking part of a person's result entry, from :meth:`hmb.tracking.Tracks.summary`."""
    first, last = person.get("first_frame"), person.get("last_frame")
    return {
        "first_frame": first,
        "last_frame": last,
        "frames_visible": person.get("num_visible_frames"),
        "duration_seconds": round(float(person.get("visible_seconds") or 0.0), 3),
        "visible_fraction": round(float(person.get("visible_fraction") or 0.0), 4),
        "time_span_seconds": ([round(first / fps, 3), round((last + 1) / fps, 3)]
                              if first is not None and last is not None and fps > 0 else None),
        "median_height_fraction": round(float(person.get("median_height_fraction") or 0.0), 4),
        "mean_track_score": round(float(person.get("mean_score") or 0.0), 4),
    }


def _empty_scores(state: str, paper_scores: bool, reason: str = "") -> dict:
    """Score keys of a person that was not scored (``state`` for every metric)."""
    scores = {"scores": {m: None for m in METRICS}, "tiers": {t: None for t in TIERS}, "overall": None,
              "status": {m: state for m in METRICS}, "reasons": {m: reason for m in METRICS} if reason else {}}
    if paper_scores:
        scores["paper_scores"] = {key: value for key, value in scores.items()}
    return scores


def _person_entry(person: dict, fps: float, pdir: Path, paper_scores: bool) -> dict:
    """A person's result entry before scoring: tracking info, empty scores and the file paths."""
    pid = int(person["id"])
    entry = {"id": pid, "rank": person.get("rank")}
    entry.update(_tracking_info(person, fps))
    entry.update(_empty_scores("error", paper_scores, "not scored"))
    entry.update({
        "error": None,
        "person": None,
        "seconds": None,
        "paths": {"dir": str(pdir), "video": str(pdir / f"person_{pid}.mp4"),
                  "person_boxes": str(pdir / PERSON_BOXES_NAME), "humanscore": str(pdir / RESULT_NAME),
                  "summary": str(pdir / SUMMARY_NAME), "log": str(pdir / PERSON_LOG_NAME)},
    })
    return entry


def _score_person(video: Path, tracks, person: dict, out: Path, *, score_options: dict, force: bool,
                  quiet: bool, say: Callable[[str], None]) -> dict:
    """Write one person's masked video and box track, score it; never raises (except on interrupt)."""
    from hmb.tracking import write_person_video
    from hmb.tracking.tracks import TRACKS_JSON

    pid = int(person["id"])
    pdir = person_dir(out, pid)
    entry = _person_entry(person, tracks.fps, pdir, score_options["paper_scores"])
    pvideo, boxes = Path(entry["paths"]["video"]), Path(entry["paths"]["person_boxes"])
    started = time.perf_counter()
    try:
        pdir.mkdir(parents=True, exist_ok=True)
        tracks_json = out / TRACKS_DIR / TRACKS_JSON
        rewritten = False
        # the person video depends only on the tracks: rebuild it when they are newer (or on force)
        if force or not pvideo.is_file() or pvideo.stat().st_mtime_ns < tracks_json.stat().st_mtime_ns:
            say(f"person {pid}: writing {pvideo.name} (other people masked)")
            write_person_video(video, tracks, pid, pvideo)
            rewritten = True
        tracks.write_person_boxes(pid, boxes)
        # a session prepared from a previous person video (older tracks) must be rebuilt
        rescore = force or (rewritten and (pdir / "manifest.json").is_file())
        result = score_video(pvideo, pdir, person_boxes=boxes, force=rescore, quiet=quiet, **score_options)
    except KeyboardInterrupt:
        raise
    except Exception as exc:  # noqa: BLE001 - one failing person must not stop the others
        message = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
        entry["error"] = f"{type(exc).__name__}: {message}"
        entry.update(_empty_scores("error", score_options["paper_scores"], entry["error"]))
        say(f"person {pid}: FAILED ({entry['error']}); continuing with the next person")
    else:
        for key in ("scores", "tiers", "overall", "status", "reasons", "paper_scores", "person"):
            if key in result:
                entry[key] = result[key]
    entry["seconds"] = round(time.perf_counter() - started, 1)
    return entry


# ----------------------------------------------------------------------------- selection
def _save_selection(video: Path, tracks, tracks_dir: Path, rules: dict, selected: List[dict],
                    skipped: List[dict], force: bool) -> tuple:
    """Write ``tracks/selection.json`` and redraw the preview if needed -> ``(selection_path, preview_path)``.

    The preview draws the selected people in colour, so it is redrawn when the tracks
    or the selection changed (or with ``force``).
    """
    from hmb.tracking import write_preview
    from hmb.tracking.api import PREVIEW_NAME, SHEET_NAME
    from hmb.tracking.tracks import TRACKS_JSON

    # same format as ``python -m hmb.tracking``'s selection.json
    selection = {"video": str(video), "backend": tracks.backend, "rules": rules, "selected": selected,
                 "skipped": skipped}
    selection_path = tracks_dir / SELECTION_NAME
    previous_selection = None
    if selection_path.is_file():
        try:
            previous_selection = json.loads(selection_path.read_text())
        except ValueError:
            previous_selection = None
    preview = tracks_dir / PREVIEW_NAME
    # compare through a JSON round trip, as the saved selection was (tuples become lists)
    if (force or previous_selection != json.loads(json.dumps(selection)) or not preview.is_file()
            or preview.stat().st_mtime_ns < (tracks_dir / TRACKS_JSON).stat().st_mtime_ns):
        write_preview(video, tracks, preview, selected=[s["id"] for s in selected], sheet_path=tracks_dir / SHEET_NAME)
    _write(selection_path, json.dumps(selection, indent=1) + "\n")
    return selection_path, preview


def _skipped_entries(skipped: List[dict], fps: float) -> List[dict]:
    """The ``skipped`` list of the result: id, tracking info and the reason."""
    entries: List[dict] = []
    for s in skipped:
        item = {"id": int(s["id"])}
        item.update(_tracking_info(s, fps))
        item["reason"] = s["reason"]
        entries.append(item)
    return entries


# ----------------------------------------------------------------------------- API
@process_groups.raise_on_termination()
def score_people(video, out_dir, *, tracker: str = "sam3.1", min_duration: float = 1.0,
                 min_visible: float = MIN_VISIBLE, max_people: Optional[int] = 10, min_height: float = 0.1,
                 min_relative_height: float = 0.25, static_cam: bool = True, sample_fps: Optional[float] = 24.0,
                 max_side: Optional[int] = 1280, paper_scores: bool = False, force: bool = False,
                 quiet: bool = True, tracker_options: Optional[dict] = None,
                 progress: Optional[Callable[[str], None]] = None) -> dict:
    """Track every person in ``video`` and score the selected people one by one into ``out_dir``.

    Options:

    * ``tracker``: ``"sam3.1"`` (default; needs SAM 3.1, see :mod:`hmb.tracking.sam3_backend`)
      or ``"yolo"``.  ``tracker_options`` go to :func:`hmb.tracking.track_people`
      (e.g. ``sam3_python``, ``yolo_weights``).
    * ``min_duration`` (seconds), ``min_visible`` (fraction of the frames), ``max_people``
      (0/None: no limit), ``min_height`` and ``min_relative_height``: the selection rules
      of :func:`hmb.tracking.screen_people`.
    * ``static_cam``, ``sample_fps``, ``max_side`` and ``paper_scores`` are passed to
      :func:`hmb.pipeline.score_video` for every person.
    * ``force``: track again and rebuild everything.
    * ``quiet``: keep the console to progress lines (tracker output goes to
      ``out_dir/logs/tracking.log``, each person's to its session log).
    * ``progress(message)`` receives the progress messages (default: printed with a
      ``[humanscore]`` prefix).

    Returns (and writes to ``out_dir/humanscore_multi.json``)::

        {"video", "tracker", "video_info", "num_tracked", "num_selected", "num_scored",
         "complete", "message",
         "people": {"<id>": {"id", "rank", "frames_visible", "duration_seconds", "visible_fraction",
                             "first_frame", "last_frame", "scores", "tiers", "overall", "status",
                             "reasons", "paper_scores" (only when requested), "person", "error",
                             "seconds", "paths", ...}},          (most prominent first)
         "skipped": [{"id", ..., "reason"}],
         "selection_rules", "settings", "timings", "files"}

    Raises ``FileNotFoundError``/``ValueError`` for invalid inputs (before any work) and
    the tracker's errors (:class:`hmb.tracking.SAM3SetupError`,
    :class:`hmb.tracking.SAM3CheckpointError`, ``RuntimeError``); a person whose scoring
    fails is reported in its entry instead.  Ctrl-C and SIGTERM/SIGHUP
    (:class:`hmb.process_groups.Terminated`) stop the tracker and the engines and
    propagate (see :func:`hmb.pipeline.score_video`).
    """
    from hmb.tracking import describe_selection, screen_people, track_people
    from hmb.tracking.api import SHEET_NAME
    from hmb.tracking.tracks import MASKS_NPZ, TRACKS_JSON

    started = time.perf_counter()
    video = Path(video).expanduser().resolve()
    out = Path(out_dir).expanduser().resolve()
    tracker = check_options(video, out, tracker=tracker, min_duration=min_duration, min_visible=min_visible,
                            max_people=max_people, static_cam=static_cam, sample_fps=sample_fps,
                            max_side=max_side, paper_scores=paper_scores)
    say = progress or (lambda message: print(f"[humanscore] {message}", flush=True))
    out.mkdir(parents=True, exist_ok=True)
    for name in (MULTI_RESULT_NAME, SUMMARY_NAME):
        (out / name).unlink(missing_ok=True)  # never leave a previous run's result behind a failed run
    tracks_dir = out / TRACKS_DIR

    # 1. tracking
    say(f"tracking people in {video.name} with {tracker}"
        + (f" (log: {out / TRACKING_LOG_NAME})" if quiet else ""))
    t0 = time.perf_counter()
    with _console(out / TRACKING_LOG_NAME, quiet, patterns=TRACKING_PATTERNS):
        tracks = track_people(video, tracks_dir, backend=tracker, force=force, quiet=False, preview=False,
                              **(tracker_options or {}))
    tracking_seconds = time.perf_counter() - t0

    # 2. selection
    rules = {"min_duration": float(min_duration), "min_visible": float(min_visible),
             "max_people": int(max_people) if max_people else None, "min_height": float(min_height),
             "min_relative_height": float(min_relative_height)}
    selected, skipped = screen_people(tracks, min_duration=min_duration, min_visible=min_visible,
                                      max_people=max_people, min_height=min_height,
                                      min_relative_height=min_relative_height)
    selection_path, preview = _save_selection(video, tracks, tracks_dir, rules, selected, skipped, force)

    say(f"{len(tracks.ids)} {'person' if len(tracks.ids) == 1 else 'people'} tracked, {len(selected)} to score"
        + (f", {len(skipped)} skipped" if skipped else "") + f"; preview: {preview}")
    for line in describe_selection(tracks, selected, skipped):
        say("  " + line)
    message = None
    if not tracks.ids:
        message = "no person found in the video"
    elif not selected:
        message = "no tracked person passes the selection rules (see skipped)"
    if message:
        say(message + "; nothing to score")

    # 3. people, one by one
    score_options = {"static_cam": bool(static_cam), "sample_fps": sample_fps, "max_side": max_side,
                     "paper_scores": bool(paper_scores)}
    people: Dict[str, dict] = {}
    for k, person in enumerate(selected, 1):
        say(f"person {person['id']} ({k} of {len(selected)}): visible {person['visible_seconds']:.1f} s "
            f"(frames {person['first_frame']}-{person['last_frame']})")
        entry = _score_person(video, tracks, person, out, score_options=score_options, force=force, quiet=quiet,
                              say=say)
        people[str(entry["id"])] = entry

    # 4. result
    num_scored = sum(1 for p in people.values() if p["error"] is None
                     and any(v is not None for v in p["scores"].values()))
    result = {
        "video": str(video),
        "tracker": tracker,
        "video_info": {"fps": float(tracks.fps), "num_frames": int(tracks.num_frames), "width": int(tracks.width),
                       "height": int(tracks.height),
                       "duration_seconds": round(tracks.num_frames / tracks.fps, 3) if tracks.fps else None},
        "num_tracked": len(tracks.ids),
        "num_selected": len(selected),
        "num_scored": num_scored,
        "complete": bool(people) and all(p["error"] is None and is_complete(p) for p in people.values()),
        "message": message,
        "people": people,
        "skipped": _skipped_entries(skipped, tracks.fps),
        "selection_rules": rules,
        "settings": dict(score_options, tracker=tracker),
        "timings": {"tracking_seconds": round(tracking_seconds, 1),
                    "people_seconds": {pid: p["seconds"] for pid, p in people.items()},
                    "total_seconds": round(time.perf_counter() - started, 1)},
        "files": {"tracks": str(tracks_dir / TRACKS_JSON), "masks": str(tracks_dir / MASKS_NPZ),
                  "preview": str(preview), "preview_sheet": str(tracks_dir / SHEET_NAME),
                  "selection": str(selection_path), "people": str(out / PEOPLE_DIR),
                  "summary": str(out / SUMMARY_NAME),
                  "tracking_log": str(out / TRACKING_LOG_NAME) if quiet else None},
    }
    _write(out / MULTI_RESULT_NAME, json.dumps(result, indent=1, allow_nan=False) + "\n")
    _write(out / SUMMARY_NAME, format_multi_table(result) + "\n")
    return result


# ----------------------------------------------------------------------------- table
def _cell(value, width: int) -> str:
    """A score right-aligned in ``width`` characters with one decimal, ``-`` when there is none."""
    return f"{value:{width}.1f}" if isinstance(value, (int, float)) and not isinstance(value, bool) \
        else " " * (width - 1) + "-"


def _person_state(person: dict) -> str:
    """The status column: ``ok``, ``failed: <error>`` or ``incomplete: <metric> <status>, ...``."""
    if person.get("error"):
        return "failed: " + person["error"]
    if is_complete(person):
        return "ok"
    states = dict(person.get("status") or {})
    states.update({f"paper {m}": s for m, s in ((person.get("paper_scores") or {}).get("status") or {}).items()})
    missing = [f"{m} {s}" for m, s in states.items() if s != "ok"]
    return "incomplete: " + ", ".join(missing) if missing else "incomplete"


def format_multi_table(result: dict) -> str:
    """Short human-readable table of a :func:`score_people` result (one row per person)."""
    paper = bool(result.get("settings", {}).get("paper_scores"))
    people = list(result.get("people", {}).values())
    tracked, skipped = result.get("num_tracked", 0), result.get("skipped") or []
    lines = [f"HumanScore, every person: {Path(result['video']).name}",
             f"  tracker {result.get('tracker')}: {tracked} {'person' if tracked == 1 else 'people'} tracked, "
             f"{len(people)} scored one by one" + (f", {len(skipped)} skipped" if skipped else "")]
    if result.get("message"):
        lines.append(f"  {result['message']}")
    if people:
        header = f"  {'person':7s} {'visible':14s}" + "".join(f" {SHORT_LABELS[m]:>9s}" for m in METRICS)
        header += f" {'overall':>8s}" + (f" {'paper':>6s}" if paper else "") + "  status"
        lines += [""] + (["  paper = overall of the original paper implementation"] if paper else []) + [
            header, "  " + "-" * (len(header) - 2)]
        for p in people:
            visible = f"{p.get('duration_seconds') or 0:.1f} s ({100 * (p.get('visible_fraction') or 0):.0f}%)"
            row = f"  {'P' + str(p['id']):7s} {visible:14s}" + "".join(
                f" {_cell((p.get('scores') or {}).get(m), 9)}" for m in METRICS)
            row += f" {_cell(p.get('overall'), 8)}"
            if paper:
                row += f" {_cell((p.get('paper_scores') or {}).get('overall'), 6)}"
            lines.append(row + f"  {_person_state(p)}")
        notes = [f"    P{p['id']} {line}" for p in people if not p.get("error") for line in reason_lines(p)]
        if notes:
            lines += ["", "  missing scores:"] + notes
    if skipped:
        lines += ["", "  skipped:"]
        lines += [f"    P{s['id']}: {s['reason']}" for s in skipped]
    timing = result.get("timings") or {}
    lines += ["", "  scores 0-100, higher = more plausible"
              + (f"; tracking {timing['tracking_seconds']:.0f} s" if timing.get("tracking_seconds") is not None else "")
              + (f", total {timing['total_seconds']:.0f} s" if timing.get("total_seconds") is not None else "")]
    files = result.get("files") or {}
    if people:
        lines.append(f"  per person: {files.get('people', 'people')}/person_<id>/summary.txt")
    if files.get("preview"):
        lines.append(f"  tracks preview: {files['preview']}")
    return "\n".join(lines)
