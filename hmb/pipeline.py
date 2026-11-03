"""HumanScore for one person in one video, end to end (the API behind ``scripts/humanscore_single.py``).

::

    from hmb.pipeline import score_video

    result = score_video("clip.mp4", "humanscore_output/clip")
    print(result["overall"], result["scores"]["bone_length"])

:func:`score_video` runs what ``python -m hmb.cli run-all`` runs (it calls
:func:`hmb.cli.execute_video`):

1. the reconstruction engines (``hmb.batch.run_session``): MeTRAbs 3-D keypoints, the GVHMR
   body mesh and the HADM extra-limb detector run in parallel, then the skeleton fit;
2. the six HumanScore scores (``hmb.scoring.score_session``).

``out_dir`` is the session directory (layout in :mod:`hmb.layout`), so a second call
reuses every verified artifact.  The call writes

* ``out_dir/humanscore.json``::

      {"video": "/abs/clip.mp4",
       "scores": {six metrics}, "tiers": {"anatomy", "kinematic", "kinetic"}, "overall",
       "status": {metric: "ok" | "unsupported" | "missing_input" | "error"},
       "reasons": {metric: one line, for every metric that is not ok},
       "paper_scores": {"scores", "tiers", "overall", "status", "reasons"},   (only with paper_scores=True)
       "person": {"selection", "frames_analysed", "frames_found", "frames_with_several_people"},
       "files": {"session", "report", "summary", "log", "engine_logs"},
       "timings": {"total_seconds", "reconstruction_seconds", "scoring_seconds"}}

  e.g. ``"reasons": {"bone_length": "keypoint coverage 0.18 < 0.25 of the analysed frames"}``;
* ``out_dir/summary.txt`` -- the same as a short table (:func:`format_table`);
* ``out_dir/logs/humanscore.log`` -- the console output of the run when ``quiet=True``
  (engine logs are in ``out_dir/cache/engines/*.log``);
* ``out_dir/scores.json`` -- the detailed scorer report (per-metric diagnostics and full errors).

Scores are 0-100, higher is more plausible.  Tier scores are the mean of their two
metrics and ``overall`` the mean of the six; they are null unless every contributing
metric was scored.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Iterator, Optional

from hmb import process_groups
from hmb.report import format_scores, is_complete, reason_lines
from hmb.scoring.score_session import METRICS, TIERS

# Files of the session directory that the result refers to (paths relative to ``out_dir``).
RESULT_NAME = "humanscore.json"
SUMMARY_NAME = "summary.txt"
REPORT_NAME = "scores.json"
LOG_NAME = "logs/humanscore.log"
#: Engine names in the console progress lines.
ENGINE_LABELS = {"detect": "3-D keypoints (MeTRAbs)", "mesh": "body mesh (GVHMR)",
                 "hadm": "extra-limb detector (HADM)", "fit": "skeleton fit"}
MOVING_CAMERA_HELP = ("a moving camera (--moving-cam, static_cam=False) is estimated for the body mesh with GVHMR's "
                      "visual odometry, which needs pycolmap; it is not installed. Install pycolmap in this "
                      "environment, or score with a static camera (omit --moving-cam; the HumanScore evaluation "
                      "protocol).")


def default_out_dir(video, root="humanscore_output") -> Path:
    """``./humanscore_output/<video stem>`` (the stem as ``hmb.cli`` sanitises it)."""
    from hmb.layout import sanitize_stem

    return Path(root) / sanitize_stem(Path(video).stem)


# ----------------------------------------------------------------------------- console
class _ConsoleFilter:
    """``sys.stdout``/``sys.stderr`` replacement of a quiet run: everything goes to the log, a few
    progress lines (engine start/finish, scoring stages, failures) also go to the console.

    ``PATTERNS`` maps the log lines of :mod:`hmb.cli` and :mod:`hmb.batch` to the short
    console messages (``(regex, render)`` pairs).
    """

    PATTERNS = (
        (re.compile(r"^\[session\] started (\w+)"), lambda m: f"  {ENGINE_LABELS.get(m[1], m[1])}: running"),
        (re.compile(r"^\[session\] (\w+) finished with code (-?\d+) in ([\d.]+)s"),
         lambda m: f"  {ENGINE_LABELS.get(m[1], m[1])}: "
                   + (f"done in {float(m[3]):.0f} s" if m[2] == "0" else f"FAILED (exit code {m[2]}), see the log")),
        (re.compile(r"^\[HMB\] Scoring: (.*)"), lambda m: f"  scoring: {m[1]}"),
        (re.compile(r"^\[HMB\] (\S+ engine failed: .*)"), lambda m: f"  warning: {m[1]}"),
        (re.compile(r"^\[HMB\] (Preparation failed.*)"), lambda m: f"  warning: {m[1]}"),
        (re.compile(r"^\[HMB\] (scoring failed.*|scores incomplete.*)"), lambda m: f"  warning: {m[1]}"),
    )

    def __init__(self, log, echo: Callable[[str], None], patterns=None):
        self.log, self.echo, self.buffer, self.seen = log, echo, "", set()
        self.patterns = self.PATTERNS if patterns is None else tuple(patterns)

    def write(self, text: str) -> int:
        """Append ``text`` to the log and echo every completed line that matches a pattern."""
        self.log.write(text)
        self.buffer += text
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            self._match(line)
        return len(text)

    def _match(self, line: str) -> None:
        for pattern, render in self.patterns:
            found = pattern.match(line.strip())
            if found:
                message = render(found)
                # a progress line is shown once even if the log repeats it; warnings are always shown
                if message not in self.seen or "warning" in message:
                    self.seen.add(message)
                    self.echo(message)
                return

    def flush(self) -> None:
        self.log.flush()

    def isatty(self) -> bool:
        return False

    def writable(self) -> bool:
        return True

    def fileno(self) -> int:  # libraries that write to the descriptor reach the log as well
        return self.log.fileno()

    @property
    def encoding(self) -> str:
        return self.log.encoding


@contextlib.contextmanager
def _console(log_path: Path, quiet: bool, prefix: str = "[humanscore]",
             patterns=None) -> Iterator[Callable[[str], None]]:
    """Yield ``say(message)``.  With ``quiet`` the process stdout/stderr (file descriptors 1 and 2,
    so also every child process) are redirected to ``log_path`` and only progress lines are shown
    (lines matching ``patterns``, ``(regex, render)`` pairs; default: the engine/scoring lines)."""
    if not quiet:
        def say(message: str) -> None:
            print(f"{prefix} {message.strip()}", flush=True)

        yield say
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            stream.flush()
    # keep the real stdout/stderr descriptors: ``say`` writes to the terminal through them while
    # descriptors 1 and 2 point to the log, and they are restored afterwards
    saved = [os.dup(1), os.dup(2)]
    console = os.fdopen(os.dup(saved[0]), "w", buffering=1)
    log = open(log_path, "a", buffering=1)
    log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
    log.flush()

    def say(message: str) -> None:
        console.write(f"{prefix} {message.strip()}\n")
        console.flush()

    old = sys.stdout, sys.stderr
    try:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        sys.stdout = _ConsoleFilter(log, say, patterns)  # type: ignore[assignment]
        sys.stderr = _ConsoleFilter(log, say, patterns)  # type: ignore[assignment]
        yield say
    finally:
        for stream in (sys.stdout, sys.stderr):
            with contextlib.suppress(Exception):
                stream.flush()
        sys.stdout, sys.stderr = old
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        for fd in saved:
            os.close(fd)
        log.close()
        console.close()


# ----------------------------------------------------------------------------- result
def _finite(value) -> Optional[float]:
    """``value`` as a float when it is a finite number, else ``None`` (JSON has no NaN)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _read_json(path: Path) -> Optional[dict]:
    """Parsed JSON file, ``None`` when it is missing or unreadable."""
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def person_coverage(session: Path, stem: str) -> Optional[dict]:
    """How often the scored person was found by MeTRAbs (``baseline/<stem>_keypoints.npz``)."""
    path = session / "baseline" / f"{stem}_keypoints.npz"
    try:
        import numpy as np

        with np.load(path, allow_pickle=False) as data:
            confs = np.asarray(data["confs"])
            frames = int(len(confs))
            # frames where the person was not found have zero keypoint confidence
            found = int((confs.reshape(frames, -1)[:, 0] > 0).sum()) if frames else 0
            mode = str(data["person_selection"]) if "person_selection" in data.files else None
            multi = int((np.asarray(data["num_detections"]) > 1).sum()) if "num_detections" in data.files else None
    except Exception:  # noqa: BLE001 - missing or unreadable keypoints: no coverage information
        return None
    return {"selection": mode, "frames_analysed": frames, "frames_found": found,
            "frames_with_several_people": multi}


def frame_sampling(session: Path) -> Optional[dict]:
    """Analysed frames of the session's frame plan (``input/frames.json``), for the summary table."""
    plan = _read_json(Path(session) / "input" / "frames.json")
    if not plan:
        return None
    return {"analysed_frames": len(plan.get("indices") or []), "source_frames": plan.get("num_source_frames"),
            "source_fps": plan.get("source_fps"), "sample_fps": plan.get("sample_fps"),
            "max_side": plan.get("max_side")}


def _timings(payload: dict, wall: float) -> dict:
    """The ``timings`` entry of ``humanscore.json`` from a :class:`hmb.performance.RunTimings` payload."""
    by_name: Dict[str, dict] = {}
    for entry in payload.get("stages") or []:
        by_name[entry.get("stage")] = entry  # the last record of a stage wins
    return {"total_seconds": wall,
            "reconstruction_seconds": _finite((by_name.get("reconstruction_engines") or {}).get("seconds")),
            "scoring_seconds": _finite((by_name.get("scoring") or {}).get("seconds"))}


def build_result(session_dir, video, summary: dict, *, wall_seconds: float,
                 timings_payload: Optional[dict] = None) -> dict:
    """``humanscore.json`` of a scored session from its score summary (:func:`hmb.report.summarize_report`).

    ``paper_scores`` is included when the summary has it.
    """
    session = Path(session_dir)
    manifest = _read_json(session / "manifest.json") or {}
    stem = str(manifest.get("video_name") or Path(video).stem)
    result = {
        "video": str(Path(video).resolve()),
        "scores": {m: _finite((summary.get("scores") or {}).get(m)) for m in METRICS},
        "tiers": {t: _finite((summary.get("tiers") or {}).get(t)) for t in TIERS},
        "overall": _finite(summary.get("overall")),
        "status": {m: (summary.get("status") or {}).get(m, "error") for m in METRICS},
        "reasons": dict(summary.get("reasons") or {}),
    }
    if manifest.get("preparation_error"):
        # a failed reconstruction explains every missing score that has no reason of its own
        for metric, state in result["status"].items():
            if state != "ok" and not result["reasons"].get(metric):
                result["reasons"][metric] = f"reconstruction failed: {manifest['preparation_error']}"
    if summary.get("paper_scores") is not None:
        result["paper_scores"] = summary["paper_scores"]
    result.update({
        "person": person_coverage(session, stem),
        "files": {"session": str(session), "report": str(session / REPORT_NAME),
                  "summary": str(session / SUMMARY_NAME), "log": str(session / LOG_NAME),
                  "engine_logs": str(session / "cache" / "engines")},
        "timings": _timings(timings_payload or _read_json(session / "timings.json") or {}, wall_seconds),
    })
    return result


def format_table(result: dict, sampling: Optional[dict] = None) -> str:
    """Short human-readable table of a :func:`score_video` result.

    ``sampling`` (:func:`frame_sampling`) adds the analysed-frames line; it is read from
    the session directory when not given.
    """
    if sampling is None and (result.get("files") or {}).get("session"):
        sampling = frame_sampling(Path(result["files"]["session"]))
    head = [f"HumanScore: {Path(result['video']).name}"]
    if sampling:
        rate = (f"sampled at {sampling['sample_fps']:g} fps" if sampling.get("sample_fps")
                else f"all frames, {sampling.get('source_fps') or 0:.4g} fps")
        head.append(f"  frames analysed: {sampling['analysed_frames']} of {sampling.get('source_frames')} ({rate}"
                    + (f", max side {sampling['max_side']} px" if sampling.get("max_side") else "") + ")")
    person = result.get("person")
    if person:
        # with person boxes (multi-person script) the other people are masked and the selection follows
        # the given track, so the detector's count of further people says nothing about the subject
        following_boxes = person.get("selection") == "boxes"
        head.append(f"  person found in {person['frames_found']}/{person['frames_analysed']} frames"
                    + (" (selection: follows the given person boxes)" if following_boxes
                       else f" (selection: {person.get('selection')})")
                    + (f"; several people visible in {person['frames_with_several_people']} frames"
                       if person.get("frames_with_several_people") and not following_boxes else ""))
    lines = [format_scores(result, "\n".join(head))]
    timing = result.get("timings") or {}
    lines += ["", "  scores 0-100, higher = more plausible"
              + (f"; wall time {timing['total_seconds']:.0f} s" if timing.get("total_seconds") is not None else "")]
    if not is_complete(result):
        failed = {m: s for m, s in result["status"].items() if s != "ok"}
        failed.update({f"paper {m}": s for m, s in ((result.get("paper_scores") or {}).get("status") or {}).items()
                       if s != "ok"})
        details = Path((result.get("files") or {}).get("session", ".")) / RESULT_NAME
        lines.append("  incomplete: " + ", ".join(f"{m} {s}" for m, s in failed.items()) + f" (details: {details})")
        lines += [f"    {line}" for line in reason_lines(result)]
    return "\n".join(lines)


def _write(path: Path, text: str) -> None:
    """Write ``text`` atomically (temporary file, then rename): a reader never sees half a file."""
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(text)
    tmp.replace(path)


# ----------------------------------------------------------------------------- API
def build_cli_args(video, out_dir, *, static_cam=True, sample_fps=24.0, max_side=1280, paper_scores=False,
                   person="track", person_boxes=None, force=False) -> argparse.Namespace:
    """The ``hmb.cli run-all`` arguments of a :func:`score_video` call."""
    from hmb import cli

    argv = ["run-all", "--video", str(video), "--session-dir", str(out_dir),
            "--sample-fps", repr(float(sample_fps or 0)), "--max-side", str(int(max_side or 0))]
    if person_boxes is not None:
        argv += ["--person", "boxes", "--person-boxes", str(person_boxes)]
    else:
        argv += ["--person", person]
    if not static_cam:
        argv.append("--moving-cam")
    if paper_scores:
        argv.append("--paper-scores")
    if force:
        argv.append("--force")
    return cli.build_parser().parse_args(argv)



def validate_options(args: argparse.Namespace) -> None:
    """:func:`hmb.cli.validate_args` worded for the end-user scripts and the API."""
    from hmb import cli

    try:
        cli.validate_args(args)
    except ValueError as exc:
        if not args.static_cam and "pycolmap" in str(exc):
            raise ValueError(MOVING_CAMERA_HELP) from None
        raise


@process_groups.raise_on_termination()
def score_video(video, out_dir, *, static_cam: bool = True, sample_fps: Optional[float] = 24.0,
                max_side: Optional[int] = 1280, paper_scores: bool = False, person: str = "track",
                person_boxes=None, force: bool = False, quiet: bool = False) -> dict:
    """Reconstruct and score ONE person in ONE video into the session directory ``out_dir``.

    Options:

    * ``static_cam``: the camera does not move (GVHMR uses an identity camera rotation;
      the HumanScore evaluation protocol).  False estimates the camera motion (needs pycolmap).
    * ``sample_fps`` / ``max_side``: analysed frame rate and decoded resolution
      (None or 0: all frames / source resolution).
    * ``paper_scores``: also compute the six scores of the original paper implementation
      (``metrics/``; adds the variable-scale skeleton fit).
    * ``person``: which detection MeTRAbs follows, ``"track"`` (the largest person near
      the start, followed with gated association) or ``"first"``.
    * ``person_boxes``: path to a JSON ``{"timestamps": [...], "boxes_xyxy": [[x0, y0, x1, y1]
      or null, ...]}`` with one box per source frame; the selection then follows these
      boxes (``person="boxes"``; used by :mod:`hmb.multi`).
    * ``force``: rebuild every artifact (otherwise verified artifacts of an earlier call are reused).
    * ``quiet``: send the console output (including the engines') to
      ``out_dir/logs/humanscore.log`` and print only progress lines.

    Returns (and writes to ``out_dir/humanscore.json``) the result dict described in
    the module docstring; :func:`hmb.report.is_complete` tells whether every requested
    score was computed.  Raises ``FileNotFoundError``/``ValueError`` for invalid inputs
    before any work starts.

    Every engine and the scorer run in a process group of their own.  On Ctrl-C
    (``KeyboardInterrupt``) or SIGTERM/SIGHUP (raised as :class:`hmb.process_groups.Terminated`,
    a ``KeyboardInterrupt``) those groups are stopped before the exception propagates;
    calling again resumes.
    """
    from hmb import cli
    from hmb.performance import RunTimings

    started = time.perf_counter()
    video = Path(video).expanduser().resolve()
    if not video.is_file():
        raise FileNotFoundError(f"video not found: {video}")
    if person_boxes is None and person not in ("track", "first"):
        raise ValueError(f"person must be 'track' or 'first' (or pass person_boxes), got {person!r}")
    if person_boxes is not None:
        person_boxes = Path(person_boxes).expanduser().resolve()
        person = "boxes"
    for name, value in (("sample_fps", sample_fps), ("max_side", max_side)):
        if value is not None and (not isinstance(value, (int, float)) or not value >= 0 or value == float("inf")):
            raise ValueError(f"{name} must be a positive number, or 0/None to disable it")
    out = Path(out_dir).expanduser().resolve()
    args = build_cli_args(video, out, static_cam=static_cam, sample_fps=sample_fps, max_side=max_side,
                          paper_scores=paper_scores, person=person, person_boxes=person_boxes, force=force)
    validate_options(args)  # option errors (e.g. moving camera without pycolmap) before any work
    out.mkdir(parents=True, exist_ok=True)
    for name in (RESULT_NAME, SUMMARY_NAME):
        (out / name).unlink(missing_ok=True)  # never leave a previous run's result behind a failed run
    timings = RunTimings(out, "humanscore")
    with _console(out / LOG_NAME, quiet) as say:
        if quiet:
            say(f"{video.name} -> {out}  (log: {out / LOG_NAME})")
        try:
            summary, succeeded = cli.execute_video(args, timings)
        except BaseException:
            timings.finish(False)
            raise
        timings.finish(succeeded)
    result = build_result(out, video, summary, wall_seconds=time.perf_counter() - started,
                          timings_payload=timings.payload)
    _write(out / RESULT_NAME, json.dumps(result, indent=1, allow_nan=False) + "\n")
    _write(out / SUMMARY_NAME, format_table(result, frame_sampling(out)) + "\n")
    if quiet:
        print(f"[humanscore] done in {result['timings']['total_seconds']:.0f} s"
              + ("" if is_complete(result) else " (incomplete, see summary)"), flush=True)
    return result
