"""Batch HumanScore pipeline: many videos through persistent, concurrent engines.

Every model is loaded once per run instead of once per video, and the GPU is
shared by independent stages::

    plan  ->  detect (MeTRAbs, TF)  ->  fit (JAX/MJX, rigid [+ variable scale])  ->  score
          ->  mesh   (GVHMR, torch)  ----------------------------------------------^
          ->  hadm   (HADM, torch)   ----------------------------------------------^

``detect``, ``mesh`` and ``hadm`` start together as separate processes (they only
need the video); ``fit`` starts when detection has finished.  Every engine reads
the same ``<S>/input/frames.json`` sampling plan, so all per-frame artifacts
align.  Engines skip sessions whose outputs exist, so an interrupted run resumes
by re-running the same command.

Usage::

    python -m hmb.batch --manifest videos.json --root outputs/batch/NAME
    python -m hmb.batch --video clip.mp4 --root outputs/batch/one

``videos.json`` is a list (or ``{"videos": [...]}``) of ``{"id": ..., "video": ...}``
entries; any extra keys (e.g. a label) are copied to ``<S>/meta.json`` and to the
score tables.

Outputs: ``ROOT/sessions/<id>/`` (the ``hmb.cli`` session layout), ``ROOT/jobs/*.json``,
``ROOT/logs/*.log``, ``ROOT/timings.json`` and ``ROOT/scores.{csv,json}``; with
``--paper-scores`` also ``ROOT/paper_scores.{csv,json}`` (the variable-scale fit then runs too).

Single video: :func:`run_session` processes ONE video into an explicit session
directory with the same concurrent engine processes.  ``hmb.cli`` (and through it
``scripts/humanscore_single.py`` and ``scripts/humanscore_multi.py``) is built on
it; its engine job files and logs are under ``<S>/cache/engines/``.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from hmb import process_groups

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGES = ("plan", "detect", "mesh", "hadm", "fit", "score")  # stages of a batch run
SESSION_STAGES = ("plan", "detect", "mesh", "hadm", "fit")   # stages of run_session (no scoring)
ENGINES = ("detect", "mesh", "hadm", "fit")
ENGINE_MODULES = {"detect": "hmb.detect_engine", "mesh": "hmb.mesh_engine", "hadm": "hmb.hadm_engine",
                  "fit": "hmb.fit_engine"}
# <S>/cache/engine_status/<file> written by each engine for every job.
ENGINE_STATUS_FILES = {"detect": "detect.json", "mesh": "mesh.json", "hadm": "hadm.json", "fit": "fit_engine.json"}
FIT_MODES = ("rigid", "variable")
METRICS = ("extra_limbs", "bone_length", "joint_range", "self_collision",
           "kinematic_extremes", "motion_smoothness")


def safe_id(value: str) -> str:
    """``value`` with every run of characters other than letters, digits, ``._-`` replaced by ``_``."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_") or "video"


def load_manifest(path: Path) -> List[dict]:
    """Manifest entries with a unique, file-system-safe ``id`` each (default: the video's file name)."""
    data = json.loads(Path(path).read_text())
    entries = data["videos"] if isinstance(data, dict) else data
    out, seen = [], set()
    for entry in entries:
        entry = dict(entry)
        if "video" not in entry:
            raise ValueError(f"manifest entry without 'video': {entry}")
        entry["id"] = safe_id(entry.get("id") or Path(entry["video"]).stem)
        if entry["id"] in seen:
            raise ValueError(f"duplicate manifest id {entry['id']}")
        seen.add(entry["id"])
        out.append(entry)
    return out


def _write_json(path: Path, payload) -> None:
    """Write ``payload`` atomically (a temporary file, then rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(payload, indent=1, default=str))
    tmp.replace(path)


# ----------------------------------------------------------------------------- session files
def session_outputs(session_dir: Path, stem: str) -> Dict[str, List[Path]]:
    """Completion markers of every reconstruction product of one session (CLI layout)."""
    s = Path(session_dir)
    return {
        "detect": [s / "baseline" / f"{stem}_keypoints.npz"],
        "rigid": [s / "baseline" / f"{stem}_fitted_model.npz", s / "baseline" / f"{stem}_opensim_skeleton.mot"],
        "variable": [s / "bone_fit" / f"{stem}_bone_length.npz", s / "baseline" / f"{stem}_bone_length.npz"],
        "mesh": [s / "gvhmr" / stem / "hmr4d_results.pt"],
        "hadm": [s / "cache" / "multi_limbs" / stem / "hadm_results" / "output.json"],
    }


def _remove(path: Path) -> None:
    """Delete a file, symlink or directory tree if it exists."""
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def clear_fit_outputs(session_dir: Path, stem: str, modes: Iterable[str] = FIT_MODES) -> None:
    """Delete the outputs of the given fit modes (so the next run refits them)."""
    s, outputs = Path(session_dir), session_outputs(session_dir, stem)
    for mode in modes:
        for path in outputs[mode]:
            path.unlink(missing_ok=True)
        if mode == "variable":
            (s / "bone_fit" / f"{stem}_keypoints.npz").unlink(missing_ok=True)


def clear_frame_outputs(session_dir: Path, stem: str) -> None:
    """Remove every artifact that depends on the frame plan (all reconstruction products)."""
    s = Path(session_dir)
    (s / "baseline" / f"{stem}_keypoints.npz").unlink(missing_ok=True)
    (s / "cache" / "detect" / f"{stem}_detections.npz").unlink(missing_ok=True)
    clear_fit_outputs(s, stem)
    _remove(s / "gvhmr" / stem)
    _remove(s / "cache" / "multi_limbs" / stem)


def effective_frames(plan) -> dict:
    """What the engines decode from a plan: video, source frames and timestamps, output size.

    The requested ``sample_fps``/``max_side`` are left out: e.g. a rate at or
    above the source rate, or a max side above the frame size, selects the same
    frames as no option at all.
    """
    frames = plan.as_dict()
    frames.pop("sample_fps", None)
    frames.pop("max_side", None)
    frames["video"] = str(Path(frames["video"]).resolve())
    frames["output_size"] = list(plan.output_size)
    return frames


def plan_session_dir(video, session_dir, stem: str, sample_fps: Optional[float] = None,
                     max_side: Optional[int] = None, *, force: bool = False, replan: bool = False):
    """Link ``<S>/input/<stem><ext>`` to ``video`` and write ``<S>/input/frames.json``.

    An existing plan of the same video and sampling is reused without decoding;
    so is one whose frames (:func:`effective_frames`) the requested sampling
    selects again, which keeps its file and every product.  Otherwise a plan
    that differs raises ``ValueError`` unless ``force`` (always re-plan) or
    ``replan``: then the plan is replaced and, if its frames changed, every
    frame-dependent artifact of the session is removed.  ``video`` may also be
    the session's own input link.  Returns ``(link, plan, replanned)``.
    """
    from hmb.video import FramePlan, plan_frames

    session = Path(session_dir)
    video = Path(video).expanduser().absolute()
    target = video.resolve()
    if not target.is_file():
        raise FileNotFoundError(video)
    link = session / "input" / f"{stem}{video.suffix.lower() or '.mp4'}"
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        if link.resolve() != target:
            if not force:
                raise ValueError(f"{link} points to {link.resolve()}, not {target}; use --force")
            link.unlink()
    if not link.exists():
        link.symlink_to(target)
    plan_path = session / "input" / "frames.json"
    previous = None
    if plan_path.is_file():
        try:
            previous = FramePlan.load(plan_path)
        except (OSError, ValueError, TypeError):
            previous = None
    if previous is not None and not force:
        if Path(previous.video).resolve() == target:
            if previous.sample_fps == sample_fps and previous.max_side == max_side:
                return link, previous, False
            # Same video: its probe is in the plan, so no decoding is needed to compare the frames.
            info = {"fps": previous.source_fps, "num_frames": previous.num_source_frames,
                    "width": previous.width, "height": previous.height}
            if effective_frames(plan_frames(target, sample_fps, max_side, info=info)) == effective_frames(previous):
                return link, previous, False
        if not replan:
            raise ValueError(f"existing plan {plan_path} differs from the requested sampling; use --force")
    plan = plan_frames(target, sample_fps, max_side)
    replanned = previous is not None and effective_frames(previous) != effective_frames(plan)
    if replanned and replan:
        clear_frame_outputs(session, stem)
    if previous is None or previous.as_dict() != plan.as_dict():
        plan.save(plan_path)
    return link, plan, replanned


def plan_session(entry: dict, root: Path, sample_fps: Optional[float], max_side: Optional[int],
                 force: bool = False) -> dict:
    """Create ``<S>/input/<id>.mp4`` (symlink), ``frames.json`` and ``meta.json``."""
    session = root / "sessions" / entry["id"]
    stem = entry["id"]
    link, plan, _ = plan_session_dir(entry["video"], session, stem, sample_fps, max_side, force=force)
    meta = dict(entry)
    meta.update({"session_dir": str(session), "stem": stem, "frames": plan.num_frames,
                 "source_fps": plan.source_fps, "sample_fps": sample_fps, "max_side": max_side})
    _write_json(session / "meta.json", meta)
    return {"session_dir": str(session), "stem": stem, "video": str(link)}


# ----------------------------------------------------------------------------- engine processes
class Stage:
    """One engine subprocess: its command, log file and start/finish times."""

    def __init__(self, name: str, cmd: List[str], log: Path, env: dict, tag: str = "batch"):
        self.name, self.cmd, self.log, self.env, self.tag = name, cmd, log, env, tag
        self.proc: Optional[subprocess.Popen] = None
        self.started = self.finished = None

    def start(self) -> "Stage":
        """Start the engine with stdout and stderr going to the log file."""
        self.log.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.log.open("w")
        self.started = time.time()
        try:
            # own process group: terminate() also stops whatever the engine started
            self.proc = process_groups.popen(self.cmd, cwd=str(REPO_ROOT), env=self.env, stdout=self.handle,
                                             stderr=subprocess.STDOUT)
        except BaseException:
            self.handle.close()
            raise
        print(f"[{self.tag}] started {self.name}: {' '.join(self.cmd[-4:])} (log {self.log})", flush=True)
        return self

    def poll(self) -> Optional[int]:
        """Exit code, or None while running; the first call after the exit records the finish."""
        code = self.proc.poll()
        if code is not None and self.finished is None:
            self.finished = time.time()
            self.handle.close()
            print(f"[{self.tag}] {self.name} finished with code {code} in {self.finished - self.started:.1f}s",
                  flush=True)
        return code

    def wait(self, poll_seconds: float = 2.0) -> int:
        """Block until the engine exits; returns its exit code."""
        while self.poll() is None:
            time.sleep(poll_seconds)
        return self.proc.returncode

    def terminate(self, grace: float = process_groups.GRACE_SECONDS) -> None:
        """Stop the engine and its process group (SIGTERM, SIGKILL after ``grace`` seconds)."""
        terminate_stages([self], grace)

    def timing(self) -> dict:
        """Start/finish times, seconds, exit code and log path."""
        return {"started": self.started, "finished": self.finished,
                "seconds": None if self.finished is None else self.finished - self.started,
                "returncode": None if self.proc is None else self.proc.returncode, "log": str(self.log)}


def terminate_stages(stages: Iterable[Stage], grace: float = process_groups.GRACE_SECONDS) -> None:
    """Stop the running engines of ``stages`` together with their process groups
    (:func:`hmb.process_groups.terminate`) and record their finish."""
    stages = [st for st in stages if st.proc is not None]
    try:
        process_groups.terminate([st.proc for st in stages], grace)
    finally:
        for st in stages:
            st.poll()


def default_engine_options(*, person: str = "track", fit_iters: int = 10000, fit_batch: int = 8,
                           fit_modes: Sequence[str] = FIT_MODES, fit_steps_per_call: int = 200,
                           fit_grid_hz: Optional[float] = 24.0,
                           overrides: Optional[Dict[str, dict]] = None) -> Dict[str, dict]:
    """Options of the four engines (``detect``, ``mesh``, ``hadm``, ``fit``), then ``overrides`` per engine.

    The fit also exports the rigid fit on a uniform ``fit_grid_hz`` grid
    (``<S>/baseline/<stem>_opensim_skeleton_grid.mot``), which the kinematic-extremes and
    motion-smoothness scores read.
    """
    opts = {
        "detect": {"person": person, "batch_size": 8},
        "mesh": {},
        "hadm": {},
        "fit": {"max_iters": fit_iters, "batch": fit_batch, "modes": list(fit_modes),
                "steps_per_call": fit_steps_per_call, "export_grid_hz": fit_grid_hz, "unconstrained_joints": True},
    }
    for name, extra in (overrides or {}).items():
        opts.setdefault(name, {}).update(extra)
    return opts


def engine_options(args) -> Dict[str, dict]:
    """Engine options of a batch run (the variable-scale fit only with ``--paper-scores``)."""
    modes = list(FIT_MODES) if args.paper_scores else ["rigid"]
    return default_engine_options(
        person=args.person, fit_iters=args.fit_iters, fit_batch=args.fit_batch, fit_modes=modes,
        fit_steps_per_call=args.fit_steps_per_call, fit_grid_hz=args.fit_grid_hz or None,
        overrides=json.loads(args.engine_options) if args.engine_options else None)


def child_env() -> dict:
    """Environment of the engine processes: the repository on ``PYTHONPATH``, GPU memory on demand."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT)] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p])
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    env.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    return env


def engine_command(name: str, jobs_file: Path, *, python: str, force: bool = False,
                   summary: Optional[Path] = None, extra_args: Sequence[str] = ()) -> List[str]:
    """Command line of engine ``name`` (``python -m hmb.<name>_engine --jobs ...``)."""
    cmd = [python, "-m", ENGINE_MODULES[name], "--jobs", str(jobs_file)]
    if summary is not None and name != "fit":  # the fit engine prints its summary instead
        cmd += ["--summary", str(summary)]
    cmd += list(extra_args)
    if force:
        cmd.append("--force")
    return cmd


def run_engine_stages(jobs: List[dict], opts: Dict[str, dict], stages: Iterable[str], *, jobs_dir: Path,
                      logs_dir: Path, env: dict, python: Optional[str] = None, force: bool = False,
                      serial: bool = False, extra_args: Optional[Dict[str, Sequence[str]]] = None,
                      poll_seconds: float = 2.0, fit_needs_detect: bool = False,
                      job_files: Optional[Iterable[str]] = None, tag: str = "batch"):
    """Run the engine processes: ``detect || mesh || hadm``, then ``fit`` once detection ended.

    Writes ``jobs_dir/<engine>.json`` (for ``job_files``, default every engine),
    logs to ``logs_dir/<engine>.log`` and engine summaries to
    ``jobs_dir/<engine>_summary.json``.  ``fit_needs_detect`` skips the fit when
    detection failed (single-session use; a batch fit still serves the other
    sessions).  On any exception the running engines are terminated with their
    process groups; while the engines run, SIGTERM/SIGHUP raise
    :class:`hmb.process_groups.Terminated` like Ctrl-C raises ``KeyboardInterrupt``
    (both propagate after the cleanup).
    Returns ``(returncodes, timings, skipped)`` keyed by engine name.
    """
    stages = tuple(stages)
    python = python or sys.executable
    extra_args = extra_args or {}
    for name in (ENGINES if job_files is None else tuple(job_files)):
        _write_json(jobs_dir / f"{name}.json", {"options": opts.get(name, {}), "jobs": jobs})

    def make(name: str) -> Stage:
        cmd = engine_command(name, jobs_dir / f"{name}.json", python=python, force=force,
                             summary=jobs_dir / f"{name}_summary.json", extra_args=extra_args.get(name, ()))
        return Stage(name, cmd, logs_dir / f"{name}.log", env, tag=tag)

    running: Dict[str, Stage] = {}
    skipped: Dict[str, str] = {}

    def wait_for(target: Stage) -> int:
        """Wait for ``target``, polling every running stage so each finish time is recorded on time."""
        while target.poll() is None:
            for other in running.values():
                other.poll()
            time.sleep(poll_seconds)
        return target.proc.returncode

    def launch(name: str) -> None:
        running[name] = make(name)  # registered before it starts, so an interrupt always reaches it
        running[name].start()

    with process_groups.raise_on_termination():
        try:
            for name in ("detect", "mesh", "hadm"):
                if name in stages and jobs:
                    launch(name)
                    if serial:
                        wait_for(running[name])
            if "fit" in stages and jobs:
                if "detect" in running:
                    wait_for(running["detect"])
                    if fit_needs_detect and running["detect"].proc.returncode != 0:
                        skipped["fit"] = "detection failed"
                if "fit" not in skipped:
                    launch("fit")
            codes = {name: wait_for(st) for name, st in running.items()}
        except BaseException:
            terminate_stages(running.values())
            raise
    return codes, {name: st.timing() for name, st in running.items()}, skipped


def engine_status(session_dir: Path, name: str) -> Optional[dict]:
    """The status file engine ``name`` wrote for a session, or None."""
    path = Path(session_dir) / "cache" / "engine_status" / ENGINE_STATUS_FILES[name]
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _log_tail(path: Path, lines: int = 15) -> str:
    """The last ``lines`` lines of a log file ("" if unreadable)."""
    try:
        return "\n".join(Path(path).read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


def run_session(video, session_dir, *, stem: Optional[str] = None, sample_fps: Optional[float] = None,
                max_side: Optional[int] = None, person: str = "first", fit_options: Optional[dict] = None,
                engine_options: Optional[Dict[str, dict]] = None, stages: Iterable[str] = SESSION_STAGES,
                force: bool = False, replan: bool = True, serial: bool = False, env: Optional[dict] = None,
                python: Optional[str] = None, extra_args: Optional[Dict[str, Sequence[str]]] = None,
                poll_seconds: float = 0.5) -> dict:
    """Reconstruct ONE video into ``session_dir`` (``hmb.cli`` layout) with the engines.

    Layout: ``<S>/input/<stem><ext>`` (symlink) and ``frames.json``,
    ``<S>/baseline/<stem>_{keypoints,fitted_model,bone_length}.npz`` and
    ``_opensim_skeleton.mot``, ``<S>/bone_fit/<stem>_bone_length.npz``,
    ``<S>/gvhmr/<stem>/hmr4d_results.pt``,
    ``<S>/cache/multi_limbs/<stem>/hadm_results/output.json``; engine status in
    ``<S>/cache/engine_status/``, engine jobs/logs/summaries in ``<S>/cache/engines/``.

    ``stages`` is a subset of :data:`SESSION_STAGES` (the plan is always made or
    reused: every engine reads it).  Engines whose outputs exist are not
    started (``force`` reruns them); running detection removes the old fits.
    ``fit_options`` (e.g. ``max_iters``, ``modes``, ``camera_calibration``) are
    applied after ``engine_options["fit"]``; ``engine_options`` holds per-engine
    overrides (``{"detect": {...}, "mesh": {...}, "hadm": {...}, "fit": {...}}``)
    on top of :func:`default_engine_options` (batch 1 fits, ``person``).
    ``extra_args`` are extra command-line arguments per engine (e.g. the mesh
    engine's ``--gvhmr-root``).  One failing engine never stops the others
    (except that a fit needs detection).

    Returns ``{"ok", "session_dir", "stem", "video", "plan", "stages", "errors",
    "seconds"}``; ``stages[name]["status"]`` is ``ok``, ``cached`` (outputs
    existed), ``error`` or ``skipped``.
    """
    started = time.time()
    stages = tuple(stages)
    unknown = sorted(set(stages) - set(SESSION_STAGES))
    if unknown:
        raise ValueError(f"unknown session stage(s) {unknown}; choose from {SESSION_STAGES}")
    from hmb.layout import sanitize_stem

    session = Path(session_dir).expanduser().resolve()
    stem = stem or sanitize_stem(Path(video).stem)
    result: dict = {"ok": True, "session_dir": str(session), "stem": stem, "stages": {}, "errors": {}}

    t0 = time.time()
    link, plan, replanned = plan_session_dir(video, session, stem, sample_fps, max_side, force=force, replan=replan)
    result["video"] = str(link)
    result["plan"] = {"path": str(session / "input" / "frames.json"), "frames": plan.num_frames,
                      "source_frames": plan.num_source_frames, "source_fps": plan.source_fps,
                      "sample_fps": plan.sample_fps, "max_side": plan.max_side, "replanned": replanned}
    result["stages"]["plan"] = {"status": "ok", "seconds": time.time() - t0}

    opts = default_engine_options(person=person, fit_batch=1)
    for name, extra in (engine_options or {}).items():
        opts.setdefault(name, {}).update(extra or {})
    opts["fit"].update(fit_options or {})
    modes = [m for m in FIT_MODES if m in (opts["fit"].get("modes") or FIT_MODES)]
    outputs = session_outputs(session, stem)

    def missing(key: str) -> bool:
        return force or not all(path.is_file() for path in outputs[key])

    # Run only the engines whose outputs are missing (all with ``force``); a fit without
    # keypoints needs detection first.
    run: List[str] = []
    pending_modes = [m for m in modes if missing(m)] if "fit" in stages else []
    if "detect" in stages and missing("detect") or pending_modes and not outputs["detect"][0].is_file():
        run.append("detect")
        clear_fit_outputs(session, stem)  # new keypoints invalidate both fits
        pending_modes = modes if "fit" in stages else []
    for name in ("mesh", "hadm"):
        if name in stages and missing(name):
            run.append(name)
    if pending_modes:
        run.append("fit")
        opts["fit"]["modes"] = list(pending_modes)
    for name in run:  # never read a previous run's status for this run
        (session / "cache" / "engine_status" / ENGINE_STATUS_FILES[name]).unlink(missing_ok=True)

    engines_dir = session / "cache" / "engines"
    codes, timings, skipped = {}, {}, {}
    if run:
        jobs = [{"session_dir": str(session), "stem": stem, "video": str(link)}]
        codes, timings, skipped = run_engine_stages(
            jobs, opts, run, jobs_dir=engines_dir, logs_dir=engines_dir, env=env or child_env(), python=python,
            force=force, serial=serial, extra_args=extra_args, poll_seconds=poll_seconds, fit_needs_detect=True,
            job_files=run, tag="session")

    # One result entry per requested engine: cached, skipped, ok or error (with the log tail).
    for name in ("detect", "mesh", "hadm", "fit"):
        if name not in stages:
            continue
        keys = pending_modes if name == "fit" else [name]
        if name not in run:
            result["stages"][name] = {"status": "cached", "seconds": 0.0}
            continue
        if name in skipped:
            entry = {"status": "skipped", "seconds": 0.0, "error": f"not run: {skipped[name]}"}
        else:
            status = engine_status(session, name)
            produced = all(path.is_file() for key in keys for path in outputs[key])
            ok = codes.get(name) == 0 and produced and (status is None or status.get("ok", True))
            entry = {**timings.get(name, {}), "status": "ok" if ok else "error", "engine_status": status,
                     "options": opts.get(name, {})}
            if not ok:
                error = (status or {}).get("error")
                if not error and not produced:
                    error = f"exit code {codes.get(name)}; missing " + ", ".join(
                        str(p) for key in keys for p in outputs[key] if not p.is_file())
                tail = _log_tail(Path(timings.get(name, {}).get("log", engines_dir / f"{name}.log")))
                entry["error"] = error or f"exit code {codes.get(name)}"
                entry["log_tail"] = tail
        if entry["status"] != "ok":
            result["ok"] = False
            result["errors"][name] = entry["error"]
        result["stages"][name] = entry
    result["seconds"] = time.time() - started
    return result


# ----------------------------------------------------------------------------- scoring
def _score_session(args) -> dict:
    """Score one session (``hmb.scoring.score_session``) -> one row of the score tables."""
    session, paper_scores = args
    from hmb.scoring.score_session import score_session

    try:
        rep = score_session(session, paper_scores=paper_scores)
        row = {"session_dir": session, "scores": rep["scores"], "tiers": rep["tiers"], "overall": rep["overall"],
               "status": rep["status"], "seconds": {m: rep["metrics"][m]["seconds"] for m in rep["metrics"]}}
        if paper_scores:
            row["paper_scores"] = {k: rep["paper_scores"][k] for k in ("scores", "tiers", "overall", "status")}
        return row
    except Exception as exc:
        return {"session_dir": session, "error": f"{type(exc).__name__}: {exc}"}


def write_table(path: Path, rows: List[dict], metas: Dict[str, dict]) -> None:
    """CSV with one row per session: its manifest keys, the six scores and statuses, tiers and overall.

    ``rows`` are :func:`_score_session` rows, or their ``paper_scores`` sections (with ``session_dir``).
    """
    flat = []
    for row in rows:
        meta = metas.get(row["session_dir"], {})
        rec = {k: v for k, v in meta.items() if k not in ("video", "session_dir", "stem") and not isinstance(v, (dict, list))}
        rec["session_dir"] = row["session_dir"]
        for m in METRICS:
            rec[m] = (row.get("scores") or {}).get(m)
            rec[f"{m}_status"] = (row.get("status") or {}).get(m)
        for t, v in (row.get("tiers") or {}).items():
            rec[t] = v
        rec["overall"] = row.get("overall")
        rec["error"] = row.get("error")
        flat.append(rec)
    keys = list(dict.fromkeys(k for rec in flat for k in rec)) if flat else ["session_dir"]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(flat)


# ----------------------------------------------------------------------------- batch driver
def run(args) -> int:
    """The batch run of ``python -m hmb.batch`` (see the module docstring); returns the exit status."""
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.video:
        entries = [{"id": safe_id(Path(args.video).stem), "video": args.video}]
    else:
        entries = load_manifest(Path(args.manifest))
    if args.limit:
        entries = entries[: args.limit]
    stages = tuple(args.stages)
    env = child_env()
    timings: dict = {"stages": {}, "num_videos": len(entries), "started": time.time()}
    sample_fps = None if args.sample_fps <= 0 else args.sample_fps
    max_side = None if args.max_side <= 0 else args.max_side

    # 1. Frame plans, in parallel threads (probing decodes each video once).
    t0 = time.time()
    jobs, failures = [], []
    if "plan" in stages or not all((root / "sessions" / e["id"] / "input" / "frames.json").is_file() for e in entries):
        with ThreadPoolExecutor(max(1, args.plan_workers)) as pool:
            futures = {pool.submit(plan_session, e, root, sample_fps, max_side, args.force): e for e in entries}
            for fut, e in futures.items():
                try:
                    jobs.append(fut.result())
                except Exception as exc:
                    failures.append({"id": e["id"], "error": f"{type(exc).__name__}: {exc}"})
    else:
        jobs = [{"session_dir": str(root / "sessions" / e["id"]), "stem": e["id"],
                 "video": str(next((root / "sessions" / e["id"] / "input").glob(f"{e['id']}.*")))} for e in entries]
    timings["stages"]["plan"] = {"seconds": time.time() - t0, "failures": failures}
    _write_json(root / "plan_failures.json", failures)
    print(f"[batch] planned {len(jobs)} sessions ({len(failures)} failed) in {time.time() - t0:.1f}s", flush=True)

    # 2. Reconstruction engines.
    codes, engine_timings, _ = run_engine_stages(
        jobs, engine_options(args), stages, jobs_dir=root / "jobs", logs_dir=root / "logs", env=env,
        python=sys.executable, force=args.force, serial=args.serial)
    timings["stages"].update(engine_timings)

    # 3. Scores, in parallel processes.

    metas = {}
    for job in jobs:
        try:
            metas[job["session_dir"]] = json.loads((Path(job["session_dir"]) / "meta.json").read_text())
        except OSError:
            metas[job["session_dir"]] = {}
    sessions = [j["session_dir"] for j in jobs]
    if "score" in stages and sessions:
        t = time.time()
        with ProcessPoolExecutor(max(1, args.score_workers)) as pool:
            rows = list(pool.map(_score_session, [(s, args.paper_scores) for s in sessions]))
        write_table(root / "scores.csv", rows, metas)
        _write_json(root / "scores.json", [{k: v for k, v in r.items() if k != "paper_scores"} for r in rows])
        if args.paper_scores:
            paper = [{"session_dir": r["session_dir"], **(r.get("paper_scores") or {"error": r.get("error")})}
                     for r in rows]
            write_table(root / "paper_scores.csv", paper, metas)
            _write_json(root / "paper_scores.json", paper)
        timings["stages"]["score"] = {"seconds": time.time() - t}
    timings["finished"] = time.time()
    timings["wall_seconds"] = timings["finished"] - timings["started"]
    timings["engine_returncodes"] = codes
    _write_json(root / "timings.json", timings)
    print(f"[batch] done: {len(sessions)} sessions in {timings['wall_seconds']:.1f}s; engine codes {codes}", flush=True)
    return 0 if not failures and all(c == 0 for c in codes.values()) else 1


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest", help='videos.json: a list of {"id": ..., "video": ...}')
    src.add_argument("--video", help="a single video")
    ap.add_argument("--root", required=True, help="output folder of the run")
    ap.add_argument("--stages", nargs="+", default=list(STAGES), choices=STAGES,
                    help="stages to run (default: all)")
    ap.add_argument("--paper-scores", action="store_true",
                    help="also run the variable-scale fit and write paper_scores.csv (original paper implementation)")
    ap.add_argument("--sample-fps", type=float, default=24.0, help="analysed frame rate; 0 = all frames")
    ap.add_argument("--max-side", type=int, default=1280, help="decoded longer side; 0 = source resolution")
    ap.add_argument("--person", default="track", choices=("first", "track"), help="MeTRAbs person selection")
    ap.add_argument("--fit-iters", type=int, default=10000, help="iterations of each skeleton fit")
    ap.add_argument("--fit-batch", type=int, default=8, help="videos fitted together per compiled program")
    ap.add_argument("--fit-grid-hz", type=float, default=24.0,
                    help="rate of the uniform-grid rigid-fit export read by kinematic extremes and "
                         "motion smoothness; 0 = off")
    ap.add_argument("--fit-steps-per-call", type=int, default=200, help="fit iterations per compiled call")
    ap.add_argument("--engine-options", help='JSON overrides, e.g. {"hadm": {"precision": "fp32"}}')
    ap.add_argument("--score-workers", type=int, default=4, help="parallel scoring processes")
    ap.add_argument("--plan-workers", type=int, default=6, help="parallel frame-plan threads")
    ap.add_argument("--serial", action="store_true", help="run detect/mesh/hadm one after another")
    ap.add_argument("--limit", type=int, default=0, help="process only the first N manifest entries")
    ap.add_argument("--force", action="store_true", help="rerun every stage instead of reusing outputs")
    return ap


def main(argv=None) -> int:
    """Command line; on Ctrl-C or SIGTERM the engines are stopped and the run can be resumed."""
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except KeyboardInterrupt as exc:  # Ctrl-C, or SIGTERM/SIGHUP while engines run (process_groups.Terminated)
        print(f"[batch] {process_groups.describe(exc)}; run the same command again to resume", file=sys.stderr,
              flush=True)
        return process_groups.exit_status(exc)


if __name__ == "__main__":
    raise SystemExit(main())
