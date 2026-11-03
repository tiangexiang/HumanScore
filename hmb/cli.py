"""HumanScore command line for one video (``python -m hmb.cli``), for advanced use.

Most users run ``scripts/humanscore_single.py``, which calls ``run-all`` and adds a short
summary.  Commands::

    python -m hmb.cli run-all --video clip.mp4 [--session-dir DIR] [--paper-scores]
    python -m hmb.cli prepare --video clip.mp4 [--session-dir DIR]
    python -m hmb.cli score   --session-dir DIR [--paper-scores] [--rescore]
    python -m hmb.cli doctor

* ``prepare``  reconstructs one video into a session directory: the frame plan, then the
  MeTRAbs 3-D keypoints, the GVHMR body mesh and the HADM extra-limb detections
  (concurrent engine processes, ``hmb.batch.run_session``), then the skeleton fit.
* ``score``    scores a prepared session (``hmb.scoring.score_session``; writes ``<S>/scores.json``).
* ``run-all``  is ``prepare`` followed by ``score``.
* ``doctor``   checks dependencies and model files without loading any model (:mod:`hmb.doctor`).

The defaults are the HumanScore protocol: 24 analysed frames per second, frames
downscaled to a longer side of 1280 px, the largest person near the start followed
through the video, a static camera.  ``--paper-scores`` also builds the variable-scale
skeleton fit and adds the six scores of the original paper implementation (``metrics/``).

Every product is reused while its fingerprint (:mod:`hmb.result_cache`) is unchanged, so
running a command again resumes or returns the verified results.  The session layout
is described in :mod:`hmb.layout`.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Optional

from . import process_groups
from .layout import SessionLayout, link_or_copy_file, read_manifest, sanitize_stem, write_manifest
from .model_paths import default_metrabs_model
from .performance import RunTimings
from .report import format_scores, is_complete, json_safe, summarize_report
from .result_cache import (
    hadm_inference_fingerprint,
    load_cached_scores,
    reconstruction_fingerprints,
    save_cached_scores,
    score_fingerprint,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
#: Interpreter of the engine and scorer subprocesses (the one running this module).
PYTHON = sys.executable or "python3"
#: Parent folder of the default session directories (``./humanscore_output/<video stem>``).
DEFAULT_OUTPUT_ROOT = "humanscore_output"

#: Reconstruction products the scores read ("rigid"/"variable" are the two skeleton fits).
#: The variable-scale fit is read only by the paper's bone-length metric.
RECONSTRUCTION = {"scores": frozenset({"detect", "rigid", "mesh", "hadm"}),
                  "paper_scores": frozenset({"variable"})}


# ----------------------------------------------------------------------------- session files
def default_smplx_dir() -> Path:
    """Folder of ``SMPLX_NEUTRAL.npz``: ``$HMB_SMPLX_MODEL``'s folder, ``outputs/assets/smplx`` or GVHMR's."""
    override = os.environ.get("HMB_SMPLX_MODEL")
    if override and Path(override).expanduser().is_file():
        return Path(override).expanduser().resolve().parent
    local = REPO_ROOT / "outputs/assets/smplx"
    return (
        local if (local / "SMPLX_NEUTRAL.npz").is_file() else REPO_ROOT / "GVHMR/inputs/checkpoints/body_models/smplx"
    )


def file_signature(path: Path) -> dict | None:
    """``{path, size, mtime_ns}`` of a large file (e.g. a checkpoint), ``None`` when missing."""
    if not path.is_file():
        return None
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def canonical_video_path(layout: SessionLayout, src_video: Path) -> Path:
    """``<S>/input/<sanitised stem><ext>``: the session's link to the input video."""
    stem = sanitize_stem(src_video.stem)
    suffix = src_video.suffix.lower() or ".mp4"
    return layout.input_dir / f"{stem}{suffix}"


def baseline_artifacts(layout: SessionLayout, canonical_stem: str) -> Dict[str, Path]:
    """Paths of the keypoints, rigid fit, OpenSim motion and variable-scale fit of a session."""
    return {
        "keypoints_npz": layout.baseline_dir / f"{canonical_stem}_keypoints.npz",
        "fitted_model_npz": layout.baseline_dir / f"{canonical_stem}_fitted_model.npz",
        "opensim_mot": layout.baseline_dir / f"{canonical_stem}_opensim_skeleton.mot",
        "bone_length_npz": layout.baseline_dir / f"{canonical_stem}_bone_length.npz",
    }


def invalidate_scores(layout: SessionLayout) -> None:
    """Remove the session's score report and per-metric outputs (they are recomputed on demand)."""
    for path in (layout.session_dir / "scores.json", layout.cache_dir / "score_cache.json"):
        path.unlink(missing_ok=True)
    for directory in (layout.metrics_dir / "self_collision", layout.session_dir / "paper_metrics"):
        _remove_path(directory)


def write_session_manifest(layout: SessionLayout, source_video: Path, prepared_video: Path, canonical_stem: str,
                           include_gvhmr: bool, provenance: dict | None = None, config: dict | None = None,
                           status: str = "preparing") -> None:
    """Write ``<S>/manifest.json``: the source video identity, the artifact paths and the session config."""
    artifacts = baseline_artifacts(layout, canonical_stem)
    payload = {
        "session_dir": str(layout.session_dir.resolve()),
        "source_video": str(source_video.resolve()),
        "prepared_video": str(prepared_video),
        "video_name": canonical_stem,
        "source_stat": {"size": source_video.stat().st_size, "mtime_ns": source_video.stat().st_mtime_ns},
        "provenance": provenance or {},
        "config": config or {},
        "preparation_status": status,
        "artifacts": {
            "keypoints_npz": str(artifacts["keypoints_npz"].resolve()),
            "fitted_model_npz": str(artifacts["fitted_model_npz"].resolve()),
            "opensim_mot": str(artifacts["opensim_mot"].resolve()),
            "bone_length_npz": str(artifacts["bone_length_npz"].resolve()),
            "gvhmr_result_pt": (str((layout.gvhmr_root / canonical_stem / "hmr4d_results.pt").resolve())
                                if include_gvhmr else ""),
        },
    }
    write_manifest(layout.manifest_path, payload)


# ----------------------------------------------------------------------------- options
def _remove_path(path: Path) -> None:
    """Delete a file, symlink or directory tree; nothing happens when it does not exist."""
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _optional_positive(value, flag: str, cast):
    """``None``/0 disable an option; otherwise it must be a finite positive number."""
    if value is None:
        return None
    value = cast(value)
    if value == 0:
        return None
    if not value > 0 or value == float("inf"):
        raise ValueError(f"{flag} must be positive (0: disabled)")
    return value


def required_reconstruction(paper_scores: bool = False) -> set:
    """Reconstruction products (detect, rigid, variable, mesh, hadm) the requested scores read."""
    return set(RECONSTRUCTION["scores"]) | (set(RECONSTRUCTION["paper_scores"]) if paper_scores else set())


def reconstruction_needs(args: argparse.Namespace) -> set:
    """What ``prepare``/``run-all`` build: the products of the requested scores (no mesh with ``--skip-gvhmr``)."""
    needs = required_reconstruction(getattr(args, "paper_scores", False))
    if getattr(args, "skip_gvhmr", False):
        needs.discard("mesh")
    return needs


def module_available(name: str) -> bool:
    """Whether ``name`` can be imported here (looked up, not imported)."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def validate_args(args: argparse.Namespace) -> None:
    """Reject option combinations the engines cannot honour, before any work starts."""
    validate_person_boxes(args)
    if getattr(args, "person", None) is None:
        args.person = "track"
    if getattr(args, "max_iters", 10000) < 1:
        raise ValueError("--max-iters must be positive")
    args.sample_fps = _optional_positive(getattr(args, "sample_fps", None), "--sample-fps", float)
    args.max_side = _optional_positive(getattr(args, "max_side", None), "--max-side", int)
    overrides = getattr(args, "engine_options", None) or {}
    if isinstance(overrides, str):
        try:
            overrides = json.loads(overrides)
        except ValueError as exc:
            raise ValueError(f"--engine-options is not valid JSON: {exc}") from None
    if not isinstance(overrides, dict) or any(
        name not in ("detect", "mesh", "hadm", "fit") or not isinstance(value, dict) for name, value in overrides.items()
    ):
        raise ValueError('--engine-options must map detect/mesh/hadm/fit to option objects, e.g. {"hadm": {"precision": "fp32"}}')
    args.engine_options = overrides
    if "mesh" in reconstruction_needs(args) and not args.static_cam and not module_available("pycolmap"):
        raise ValueError(
            "--moving-cam estimates the camera motion for the body mesh with visual odometry (GVHMR SimpleVO), "
            "which needs pycolmap; it is not installed. Install pycolmap, or score with a static camera (omit "
            "--moving-cam; the HumanScore evaluation protocol)."
        )


def validate_person_boxes(args: argparse.Namespace) -> None:
    """``--person-boxes`` (a subject box track) implies ``--person boxes``; check the file once."""
    boxes = getattr(args, "person_boxes", None)
    person = getattr(args, "person", None)
    if not boxes:
        if person == "boxes":
            raise ValueError("--person boxes needs --person-boxes (the subject's per-frame box track JSON)")
        return
    if person not in (None, "boxes"):
        raise ValueError(f"--person-boxes selects the person by its boxes; it cannot be combined with --person {person}")
    path = Path(boxes).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"--person-boxes file not found: {path}")
    from .detect_engine import load_person_boxes

    try:
        load_person_boxes(path)
    except (ValueError, TypeError, KeyError) as exc:
        raise ValueError(f"--person-boxes {path}: {exc}") from None
    args.person, args.person_boxes = "boxes", str(path)


def fused_steps_per_call(max_iters: int, preferred: int = 200, minimum: int = 25) -> int:
    """Fit iterations per compiled ``lax.scan`` call: a divisor of ``max_iters`` when possible.

    Every distinct call length is a separate XLA compilation, so e.g. 300 iterations use
    2 x 150 instead of 200 + 100.  The default 10000 iterations keep the engine default of 200.
    """
    if max_iters <= preferred:
        return max_iters
    for steps in range(preferred, minimum - 1, -1):
        if max_iters % steps == 0:
            return steps
    return preferred


def engine_options(args: argparse.Namespace, config: dict, needs: set) -> Dict[str, dict]:
    """Per-engine options of a session (recorded in the manifest and fingerprinted)."""
    from .batch import FIT_MODES, default_engine_options

    calibration = getattr(args, "camera_calibration", None)
    modes = [mode for mode in FIT_MODES if mode in needs] or list(FIT_MODES)
    opts = default_engine_options(person=getattr(args, "person", None) or "track", fit_iters=args.max_iters, fit_batch=1,
                                  fit_modes=modes)
    opts["detect"]["camera_calibration"] = calibration
    if opts["detect"]["person"] == "boxes":
        # the content digest makes an edited box track invalidate the detection (and both fits) and the mesh;
        # GVHMR reconstructs the track overlapping the same boxes (not the largest one)
        digest = hashlib.sha256(Path(args.person_boxes).read_bytes()).hexdigest()
        opts["detect"].update(person_boxes=args.person_boxes, person_boxes_sha256=digest)
        opts["mesh"].update(person_boxes=args.person_boxes, person_boxes_sha256=digest)
    if getattr(args, "metrabs_model", None):
        opts["detect"]["model_path"] = args.metrabs_model
    opts["mesh"].update(camera="static" if args.static_cam else "simplevo", f_mm=getattr(args, "gvhmr_f_mm", None))
    opts["hadm"]["hadm_root"] = config["hadm_root"]
    opts["fit"].update(
        camera_calibration=calibration,
        steps_per_call=fused_steps_per_call(args.max_iters),
        jax_cache_dir=os.environ.get("HMB_JAX_CACHE_DIR", str(REPO_ROOT / "outputs/runtime_cache/jax")),
    )
    for name, extra in (getattr(args, "engine_options", None) or {}).items():
        opts.setdefault(name, {}).update(extra)
    return opts


def _record_engine_timings(timings: RunTimings, result: dict) -> None:
    """Per-engine wall times of one concurrent run (engines overlap; do not sum them)."""
    for name in ("detect", "mesh", "hadm", "fit"):
        entry = result["stages"].get(name)
        if entry is None:
            continue
        record = {
            "stage": f"engine:{name}",
            "kind": "stage",
            "parallel_group": "reconstruction_engines",
            "cache_hit": entry["status"] == "cached",
            "status": "ok" if entry["status"] in ("ok", "cached") else "error",
            "seconds": entry.get("seconds") or 0.0,
        }
        if entry.get("error"):
            record["error"] = entry["error"]
        status = entry.get("engine_status")
        if isinstance(status, dict):
            record["details"] = {key: value for key, value in status.items() if key not in ("traceback", "options")}
        timings.payload["stages"].append(json_safe(record))
    timings.save()


# ----------------------------------------------------------------------------- prepare
def _resolve_input_paths(args: argparse.Namespace) -> None:
    """Make ``--camera-calibration`` and a local ``--metrabs-model`` absolute (the manifest records them)."""
    calibration = getattr(args, "camera_calibration", None)
    if calibration:
        args.camera_calibration = str(Path(calibration).expanduser().resolve())
    model = getattr(args, "metrabs_model", None)
    if model and "://" not in model:
        args.metrabs_model = str(Path(model).expanduser().resolve())


def _requested_fit_options(args: argparse.Namespace) -> dict:
    """The reconstruction options of a session (manifest ``config.requested_fit_options``, fingerprinted)."""
    fit_options = {
        "max_iters": args.max_iters,
        "metrabs_model": getattr(args, "metrabs_model", None),
        "camera_calibration": getattr(args, "camera_calibration", None),
        "gvhmr_f_mm": getattr(args, "gvhmr_f_mm", None),
        "static_cam": bool(args.static_cam),
        "sample_fps": args.sample_fps,
        "max_side": args.max_side,
        "person": getattr(args, "person", None) or "track",
    }
    if fit_options["person"] == "boxes":
        fit_options["person_boxes"] = args.person_boxes
    if fit_options["camera_calibration"]:
        # an edited calibration file must invalidate the fits even though its path is unchanged
        fit_options["camera_calibration_sha256"] = hashlib.sha256(Path(args.camera_calibration).read_bytes()).hexdigest()
    return fit_options


def _previous_manifest(layout: SessionLayout, video_path: Path, force: bool) -> dict:
    """The session's existing manifest (``{}`` for a new session).

    Refuses a session that belongs to another video and, without ``force``, a video
    that changed since the session was prepared.
    """
    if not layout.manifest_path.exists():
        return {}
    previous = read_manifest(layout.manifest_path)
    if Path(previous["source_video"]).resolve() != video_path:
        raise ValueError(f"Session {layout.session_dir} belongs to a different video; choose another --session-dir.")
    current_stat = {"size": video_path.stat().st_size, "mtime_ns": video_path.stat().st_mtime_ns}
    if previous.get("source_stat", current_stat) != current_stat and not force:
        raise ValueError("Input video has changed since preparation; use --force to rebuild its artifacts.")
    return previous


def _clear_generated(layout: SessionLayout) -> None:
    """``--force``: remove every generated artifact of the session (the manifest is rewritten afterwards)."""
    for directory in (layout.baseline_dir, layout.bone_fit_dir, layout.gvhmr_root, layout.metrics_dir,
                      layout.cache_dir):
        _remove_path(directory)
        directory.mkdir(parents=True)
    _remove_path(layout.session_dir / "paper_metrics")
    for output in ("scores.json", "input/frames.json"):
        (layout.session_dir / output).unlink(missing_ok=True)


def _session_config(args: argparse.Namespace, layout: SessionLayout, fit_options: dict, needs: set) -> dict:
    """The manifest ``config``: asset folders and file signatures, requested options and engine options."""
    from . import batch

    config = {
        "hadm_root": str(Path(getattr(args, "hadm_root", REPO_ROOT / "metrics/multi-limbs/HADM")).resolve()),
        "gvhmr_root": str(Path(getattr(args, "gvhmr_root", REPO_ROOT / "GVHMR")).resolve()),
        "smplx_model_dir": str(Path(getattr(args, "smplx_model_dir", default_smplx_dir())).resolve()),
        "camera_calibration": getattr(args, "camera_calibration", None),
        "frame_plan": str(layout.input_dir / "frames.json"),
        "gvhmr_checkpoints_dir": str(REPO_ROOT / "outputs/assets/gvhmr/checkpoints"),
        "requested_fit_options": fit_options,
    }
    config["requested_execution_options"] = {
        "engine_modules": dict(batch.ENGINE_MODULES),
        "serial_engines": bool(getattr(args, "serial_engines", False)),
        "jax_cache_dir": os.environ.get("HMB_JAX_CACHE_DIR", str(REPO_ROOT / "outputs/runtime_cache/jax")),
    }
    config["smplx_asset_signature"] = file_signature(Path(config["smplx_model_dir"]) / "SMPLX_NEUTRAL.npz")
    config["hadm_checkpoint_signature"] = file_signature(
        Path(config["hadm_root"]) / "pretrained_models/HADM-G_0249999.pth"
    )
    config["engine_options"] = engine_options(args, config, needs)
    return config


def _drop_outputs_of_changed_models(layout: SessionLayout, stem: str, old_config: dict, config: dict) -> None:
    """Remove the HADM detections or the body mesh when their checkpoint or model folder changed."""
    if any(old_config.get(key) != config.get(key) for key in ("hadm_root", "hadm_checkpoint_signature")):
        _remove_path(layout.cache_dir / "multi_limbs")
    if any(
        old_config.get(key) != config.get(key) for key in ("smplx_model_dir", "smplx_asset_signature", "gvhmr_root")
    ):
        _remove_path(layout.gvhmr_root / stem)


def _drop_stale_reconstruction(layout: SessionLayout, stem: str, previous_stages: dict,
                               stage_fingerprints: dict) -> None:
    """Remove every reconstruction product whose fingerprint changed (new keypoints also invalidate both fits)."""
    from . import batch

    if previous_stages.get("detection") != stage_fingerprints["detection"]:
        baseline_artifacts(layout, stem)["keypoints_npz"].unlink(missing_ok=True)
        (layout.cache_dir / "detect" / f"{stem}_detections.npz").unlink(missing_ok=True)
        batch.clear_fit_outputs(layout.session_dir, stem)
    if previous_stages.get("rigid_fit") != stage_fingerprints["rigid_fit"]:
        batch.clear_fit_outputs(layout.session_dir, stem, ("rigid",))
    if previous_stages.get("variable_fit") != stage_fingerprints["variable_fit"]:
        batch.clear_fit_outputs(layout.session_dir, stem, ("variable",))
    if previous_stages.get("mesh") != stage_fingerprints["mesh"]:
        _remove_path(layout.gvhmr_root / stem)


def _engine_stages(needs: set) -> list:
    """The ``hmb.batch.run_session`` stages that build ``needs`` (the frame plan always runs)."""
    stages = ["plan"]
    if needs & {"detect", "rigid", "variable"}:
        stages.append("detect")
    stages += [name for name in ("mesh", "hadm") if name in needs]
    if needs & {"rigid", "variable"}:
        stages.append("fit")
    return stages


def _engine_failures(result: dict, config: dict, video_path: Path, detector_cache: Path) -> Dict[str, str]:
    """``{engine: error}`` of a ``run_session`` result; prints each failed engine's error and log tail.

    New HADM detections are kept only if HADM's inputs did not change while it ran; their
    fingerprint is then saved next to them (``inference_identity.json``).
    """
    failures = dict(result["errors"])
    hadm_entry = result["stages"].get("hadm", {})
    if hadm_entry.get("status") in ("ok", "cached"):
        if hadm_inference_fingerprint(config, video_path, REPO_ROOT) != config["hadm_inference_fingerprint"]:
            _remove_path(detector_cache)
            failures["hadm"] = "HADM inference inputs changed while computing detections; run again."
        else:
            write_manifest(detector_cache / "inference_identity.json",
                           {"fingerprint": config["hadm_inference_fingerprint"]})
    for name, entry in result["stages"].items():
        if entry.get("status") == "error":
            print(f"[HMB] {name} engine failed: {entry.get('error')}", file=sys.stderr)
            if entry.get("log_tail"):
                print(entry["log_tail"], file=sys.stderr)
    return failures


def prepare_session(args: argparse.Namespace, timings: RunTimings | None = None, needs=None) -> Path:
    """Reconstruct ``args.video`` into its session directory with the engines (``hmb.batch.run_session``).

    Steps: check the session against the video, write the frame plan, remove the
    artifacts whose reconstruction fingerprint changed (engine code, frame plan, engine
    options, model files), then run the engines for the missing products only.  A
    failing engine does not stop the others; the manifest is then left in
    ``preparing`` state and a ``RuntimeError`` names the failed engines.  Returns the
    session directory.
    """
    from . import batch

    validate_args(args)
    needs = set(reconstruction_needs(args) if needs is None else needs)
    video_path = Path(args.video).expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    _resolve_input_paths(args)

    canonical_stem = sanitize_stem(video_path.stem)
    layout = SessionLayout.create(session_dir_from_args(args))
    fit_options = _requested_fit_options(args)
    previous = _previous_manifest(layout, video_path, args.force)
    old_config = previous.get("config", {})

    prepared_video = canonical_video_path(layout, video_path)
    if args.force:
        _clear_generated(layout)  # before the new source generation is recorded
    if args.force or not prepared_video.exists():
        link_or_copy_file(video_path, prepared_video)

    config = _session_config(args, layout, fit_options, needs)
    if timings:
        timings.payload.update(
            source_video=str(video_path),
            requested_fit_options=fit_options,
            requested_execution_options=config["requested_execution_options"],
        )
    if previous:
        _drop_outputs_of_changed_models(layout, canonical_stem, old_config, config)

    # One frame plan for every engine; a changed plan removes all frame-dependent artifacts.
    plan_existed = (layout.input_dir / "frames.json").is_file()
    with timings.stage("frame_plan") if timings else nullcontext() as stage:
        _, plan, replanned = batch.plan_session_dir(
            prepared_video, layout.session_dir, canonical_stem, args.sample_fps, args.max_side, force=args.force,
            replan=True,
        )
        plan_summary = {"frames": plan.num_frames, "source_frames": plan.num_source_frames,
                        "source_fps": plan.source_fps, "sample_fps": plan.sample_fps, "max_side": plan.max_side,
                        "replanned": replanned}
        if stage is not None:
            stage["details"] = plan_summary
            stage["cache_hit"] = plan_existed and not replanned and not args.force
    if replanned:
        invalidate_scores(layout)

    stage_fingerprints = reconstruction_fingerprints(config, video_path, layout, REPO_ROOT)
    if previous and not args.force:
        _drop_stale_reconstruction(layout, canonical_stem, old_config.get("stage_fingerprints", {}),
                                   stage_fingerprints)
    config["stage_fingerprints"] = stage_fingerprints
    # HADM detections are reused only while their own fingerprint is unchanged
    detector_cache = layout.cache_dir / "multi_limbs"
    config["hadm_inference_fingerprint"] = hadm_inference_fingerprint(config, video_path, REPO_ROOT)
    try:
        saved_inference = read_manifest(detector_cache / "inference_identity.json")
    except (OSError, ValueError):
        saved_inference = {}
    if saved_inference.get("fingerprint") != config["hadm_inference_fingerprint"]:
        _remove_path(detector_cache)
    provenance = {"mesh_camera": config["engine_options"]["mesh"].get("camera"), "frame_sampling": plan_summary}
    # Leave a manifest even if reconstruction fails, so the available scores can be computed and the run resumed.
    write_session_manifest(
        layout, video_path, prepared_video, canonical_stem, not args.skip_gvhmr, provenance=provenance, config=config
    )

    stages = _engine_stages(needs)
    extra_args = {"mesh": ["--gvhmr-root", config["gvhmr_root"], "--smplx-model-dir", config["smplx_model_dir"],
                           "--checkpoints-dir", config["gvhmr_checkpoints_dir"]]}
    with timings.stage("reconstruction_engines", group=True) if timings else nullcontext() as group:
        result = batch.run_session(
            prepared_video, layout.session_dir, stem=canonical_stem, sample_fps=args.sample_fps,
            max_side=args.max_side, person=fit_options["person"], engine_options=config["engine_options"],
            stages=stages, force=False, replan=False, serial=bool(getattr(args, "serial_engines", False)),
            python=PYTHON, extra_args=extra_args,
        )
        if timings:
            _record_engine_timings(timings, result)
            group["details"] = {"stages": stages, "ok": result["ok"]}
            if not result["ok"]:
                group.update(status="error", error="; ".join(f"{k}: {v}" for k, v in result["errors"].items()))
    failures = _engine_failures(result, config, video_path, detector_cache)
    config["stage_fingerprints"] = reconstruction_fingerprints(config, video_path, layout, REPO_ROOT)
    config["engine_run"] = json_safe({
        name: {key: entry.get(key) for key in ("status", "seconds", "error")}
        for name, entry in result["stages"].items()
    })
    write_session_manifest(
        layout=layout,
        source_video=video_path,
        prepared_video=prepared_video,
        canonical_stem=canonical_stem,
        include_gvhmr=not args.skip_gvhmr,
        provenance=provenance,
        config=config,
        status="preparing" if failures else "ready",
    )
    if failures:
        raise RuntimeError("Reconstruction engine(s) failed: " + "; ".join(f"{k}: {v}" for k, v in failures.items()))
    print(f"[HMB] Prepared session: {layout.session_dir}")
    return layout.session_dir


# ----------------------------------------------------------------------------- score
def session_smplx_dir(session_dir: Path) -> Path | None:
    """The session's ``config.smplx_model_dir`` (``manifest.json``) when it holds ``SMPLX_NEUTRAL.npz``."""
    try:
        value = read_manifest(Path(session_dir) / "manifest.json").get("config", {}).get("smplx_model_dir")
    except (OSError, ValueError, AttributeError):
        return None
    if not value or not (Path(value) / "SMPLX_NEUTRAL.npz").is_file():
        return None
    return Path(value)


def score_session_subprocess(session_dir: Path, stem: str, paper_scores: bool = False) -> None:
    """Run ``python -m hmb.scoring.score_session`` for one session in its own process group;
    it writes ``<S>/scores.json``."""
    cmd = [PYTHON, "-m", "hmb.scoring.score_session", "--session-dir", str(session_dir), "--stem", stem]
    if paper_scores:
        cmd.append("--paper-scores")
    print(f"[HMB] Running: {' '.join(cmd)}", flush=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    model_dir = session_smplx_dir(Path(session_dir))
    if model_dir is not None:
        # self-collision uses the SMPL-X model the session's mesh was built with
        env.update({"HMB_SMPLX_MODEL": "", "HMB_SMPLX_MODEL_DIR": str(model_dir)})
    process = process_groups.run(cmd, cwd=str(REPO_ROOT), env=env)
    # Exit status 1 only means that "overall" is null (some metric has no score); the report records why.
    if not (Path(session_dir) / "scores.json").is_file():
        raise RuntimeError(f"hmb.scoring.score_session exited with {process.returncode} without writing scores.json")


def run_scores(manifest: dict, layout: SessionLayout, timings: RunTimings, *, paper_scores: bool = False,
               rescore: bool = False) -> tuple[dict, bool]:
    """Score a prepared session -> ``(summary, success)`` (:func:`hmb.report.summarize_report`).

    The report ``<S>/scores.json`` is reused while its inputs, the scoring code and the
    runtime are unchanged (``hmb.result_cache.load_cached_scores``); ``rescore`` recomputes
    it.  ``success``: all six scores (and the paper scores when requested) were computed.
    """
    report_path = layout.session_dir / "scores.json"
    report, error, cache_hit = None, None, False
    print("[HMB] Scoring: six HumanScore metrics" + (" and the paper scores" if paper_scores else ""), flush=True)
    with timings.stage("scoring") as stage:
        try:
            report = None if rescore else load_cached_scores(manifest, layout, REPO_ROOT, paper_scores)
            cache_hit = report is not None
            if report is None:
                fingerprint = score_fingerprint(manifest, layout, REPO_ROOT, paper_scores)
                report_path.unlink(missing_ok=True)
                score_session_subprocess(layout.session_dir, str(manifest["video_name"]), paper_scores)
                report = json.loads(report_path.read_text())
                validate_source(manifest)
                if score_fingerprint(manifest, layout, REPO_ROOT, paper_scores) != fingerprint:
                    raise ValueError("Input artifacts changed while computing the scores; run again.")
                save_cached_scores(fingerprint, layout, paper_scores)
            else:
                print("[HMB] Reusing verified scores")
        except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError) as exc:
            report, error = None, str(exc)
            stage.update(status="error", error=error)
            print(f"[HMB] scoring failed: {exc}", file=sys.stderr)
        stage["cache_hit"] = cache_hit
    summary = summarize_report(report, error=error)
    if not paper_scores:
        summary.pop("paper_scores", None)
    ok = error is None and is_complete(summary)
    failed = {name: state for name, state in summary["status"].items() if state != "ok"}
    if failed and error is None:
        print(f"[HMB] scores incomplete: {failed}", file=sys.stderr)
    return json_safe(summary), ok


def score(args: argparse.Namespace, timings: RunTimings | None = None) -> tuple[dict, bool]:
    """``score``: score the session ``args.session_dir`` and print the table -> ``(summary, success)``."""
    layout = SessionLayout.create(Path(args.session_dir).expanduser().resolve())
    manifest = read_manifest(layout.manifest_path)
    validate_source(manifest)
    owns_timings = timings is None
    timings = timings or RunTimings(layout.session_dir, "score")
    summary, ok = run_scores(manifest, layout, timings, paper_scores=getattr(args, "paper_scores", False),
                             rescore=getattr(args, "rescore", False))
    print()
    print(format_scores(summary, f"[HMB] Scores for {manifest['video_name']} (0-100, higher is more plausible)"))
    print(f"[HMB] Report: {layout.session_dir / 'scores.json'}")
    if owns_timings:
        timings.finish(ok)
    return summary, ok


def validate_source(manifest: dict) -> None:
    """Raise ``ValueError`` when the video or a model file changed since the session was prepared."""
    source = Path(manifest["source_video"])
    stat = source.stat()
    expected = manifest.get("source_stat")
    if expected and expected != {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}:
        raise ValueError("Input video has changed since preparation; run prepare --force before scoring.")
    config = manifest.get("config", {})
    hadm_signature = config.get("hadm_checkpoint_signature")
    if hadm_signature and file_signature(Path(hadm_signature["path"])) != hadm_signature:
        raise ValueError("HADM checkpoint changed; run prepare again to invalidate detector caches.")
    model_signature = config.get("smplx_asset_signature")
    if model_signature and file_signature(Path(model_signature["path"])) != model_signature:
        raise ValueError("SMPL-X asset changed; run prepare again to invalidate old scores.")
    options = config.get("requested_fit_options", {})
    camera_path, digest = options.get("camera_calibration"), options.get("camera_calibration_sha256")
    if camera_path and digest and hashlib.sha256(Path(camera_path).read_bytes()).hexdigest() != digest:
        raise ValueError("Camera calibration changed; run prepare --force to refit the trajectories.")


# ----------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    """The ``prepare`` / ``score`` / ``run-all`` / ``doctor`` argument parser."""
    # allow_abbrev=False: an option is only ever matched by its full name, so a shortened or mistyped
    # option is an error instead of silently selecting another one
    parser = argparse.ArgumentParser(prog="python -m hmb.cli", allow_abbrev=False,
                                     description="HumanScore for one video: prepare a session, score it, or both.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Reconstruct one video into a session directory.",
                                    allow_abbrev=False)
    score_cmd = subparsers.add_parser("score", help="Score a prepared session (writes <session>/scores.json).",
                                      allow_abbrev=False)
    run_all = subparsers.add_parser("run-all", help="Prepare one video and score it.", allow_abbrev=False)

    def add_asset_options(command):
        command.add_argument("--hadm-root", default=os.environ.get("HMB_HADM_ROOT", str(REPO_ROOT / "metrics/multi-limbs/HADM")),
                             help="HADM source folder (with pretrained_models/).")
        command.add_argument("--gvhmr-root", default=os.environ.get("HMB_GVHMR_ROOT", str(REPO_ROOT / "GVHMR")),
                             help="GVHMR source folder.")
        command.add_argument("--smplx-model-dir", default=os.environ.get("HMB_SMPLX_MODEL_DIR", str(default_smplx_dir())),
                             help="Folder holding SMPLX_NEUTRAL.npz.")

    def add_paper_option(command):
        command.add_argument("--paper-scores", action="store_true",
                             help="Also compute the six scores of the original paper implementation (metrics/); "
                                  "this adds the variable-scale skeleton fit.")

    for command in (prepare, run_all):
        command.add_argument("--video", required=True, help="Input video path.")
        command.add_argument("--session-dir", default="",
                             help=f"Session directory. Default: ./{DEFAULT_OUTPUT_ROOT}/<video stem>.")
        command.add_argument("--force", action="store_true", help="Rebuild every artifact instead of reusing them.")
        command.add_argument("--moving-cam", dest="static_cam", action="store_false",
                             help="The camera moves: estimate its rotation for the body mesh (needs pycolmap). "
                                  "Default: static camera (the HumanScore protocol).")
        command.add_argument("--skip-gvhmr", action="store_true",
                             help="Debugging: skip the body mesh (self-collision is then missing).")
        add_paper_option(command)
        add_asset_options(command)
        command.add_argument("--sample-fps", type=float, default=24.0,
                             help="Analyse the source frames nearest to this rate (true timestamps kept). "
                                  "Default 24 (the HumanScore protocol); 0 = every frame.")
        command.add_argument("--max-side", type=int, default=1280,
                             help="Downscale decoded frames to this longer side. Default 1280; 0 = source resolution.")
        command.add_argument("--person", choices=("first", "track", "boxes"), default=None,
                             help="MeTRAbs person selection: track (default; the largest early person, followed with "
                                  "gated IoU/centre association, a missing subject becomes a gap), first (first "
                                  "detection per frame) or boxes (follow --person-boxes).")
        command.add_argument("--person-boxes", default=None,
                             help='JSON {"timestamps": [...], "boxes_xyxy": [[x0, y0, x1, y1] or null, ...]} with the '
                                  "subject's box per source frame (e.g. from hmb.tracking); MeTRAbs then selects the "
                                  "detection with the highest IoU with that box. Implies --person boxes.")
        command.add_argument("--metrabs-model", default=default_metrabs_model(),
                             help="Local MeTRAbs SavedModel or TF Hub handle (default: $HMB_METRABS_MODEL, else the "
                                  "model install.sh unpacks in outputs/assets/metrabs/, else the TF Hub handle).")
        command.add_argument("--camera-calibration", default=None,
                             help="Camera calibration JSON. Default: MeTRAbs-compatible 55 degree field of view.")
        command.add_argument("--gvhmr-f-mm", type=float, default=None,
                             help="Full-frame-equivalent focal length for the body mesh (default: GVHMR's estimate).")
        command.add_argument("--max-iters", type=int, default=10000,
                             help="Iterations of each skeleton fit; smaller values are for tests only.")
        command.add_argument("--engine-options", default=None,
                             help='JSON per-engine overrides, e.g. {"hadm": {"precision": "fp32"}}.')
        command.add_argument("--serial-engines", action="store_true",
                             help="Run detection, mesh and HADM one after another (lower peak GPU memory).")

    score_cmd.add_argument("--session-dir", required=True, help="Prepared session directory.")
    add_paper_option(score_cmd)
    for command in (score_cmd, run_all):
        command.add_argument("--rescore", action="store_true",
                             help="Recompute the scores (the reconstruction is still reused).")

    doctor = subparsers.add_parser(
        "doctor", allow_abbrev=False,
        help="Check dependencies and model files without loading ML models or downloading anything.",
        description="Required checks cover single-person scoring; AUTO items are downloaded on first use. "
                    "Optional checks (SAM 3.1, YOLO segmentation weights, ffmpeg) cover the multi-person script "
                    "and do not affect the exit status (0 = ready, 1 = a required item is missing).")
    add_asset_options(doctor)
    doctor.add_argument("--json", action="store_true", help="Print diagnostics as JSON.")
    doctor.add_argument("--skip-optional", action="store_true",
                        help="Skip the optional multi-person checks (the SAM 3.1 check starts its Python environment).")
    return parser


# ----------------------------------------------------------------------------- entry points
def session_dir_from_args(args: argparse.Namespace) -> Path:
    """``--session-dir``, else ``./humanscore_output/<sanitised video stem>``."""
    if args.session_dir:
        return Path(args.session_dir).expanduser().resolve()
    return (Path(DEFAULT_OUTPUT_ROOT) / sanitize_stem(Path(args.video).stem)).resolve()


def execute_video(args: argparse.Namespace, timings: RunTimings) -> tuple[dict, bool]:
    """``run-all``: prepare and score one video -> ``(summary, success)``.

    When the reconstruction fails after the session was created, the available
    artifacts are still scored (the failed products' metrics are reported as missing).
    """
    validate_args(args)
    preparation_failed = False
    try:
        with timings.stage("prepare", group=True):
            session_dir = prepare_session(args, timings=timings)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        session_dir = session_dir_from_args(args)
        manifest_path = session_dir / "manifest.json"
        if not manifest_path.is_file():
            raise
        manifest = read_manifest(manifest_path)
        if Path(manifest["source_video"]).resolve() != Path(args.video).expanduser().resolve():
            raise
        if "changed since preparation" in str(exc):
            raise  # never score a changed video with the artifacts of the old one
        preparation_failed = True
        manifest["preparation_error"] = str(exc)
        write_manifest(manifest_path, manifest)
        print(f"[HMB] Preparation failed; scoring the available artifacts: {exc}", file=sys.stderr)
    summary, ok = score(argparse.Namespace(session_dir=str(session_dir), paper_scores=args.paper_scores,
                                           rescore=args.rescore), timings=timings)
    return summary, ok and not preparation_failed


def main(argv: Optional[list] = None) -> None:
    """Command-line entry point; exits with 0 on success, 1 when scores are missing."""
    parser = build_parser()
    args = parser.parse_args(argv)
    # SIGTERM/SIGHUP raise process_groups.Terminated (a KeyboardInterrupt) like Ctrl-C does, so the
    # engines and the scorer (each in its own process group) are stopped before exiting.
    with process_groups.raise_on_termination():
        try:
            run_command(parser, args)
        except KeyboardInterrupt as exc:
            print(f"[HMB] {process_groups.describe(exc)}; child processes stopped. Run the same command again "
                  "to resume.", file=sys.stderr, flush=True)
            raise SystemExit(process_groups.exit_status(exc)) from None


def run_command(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Run the parsed command (``doctor``, ``score``, ``prepare`` or ``run-all``)."""
    if args.command == "doctor":
        from .doctor import diagnose, format_report

        report = diagnose(REPO_ROOT, Path(args.hadm_root), Path(args.gvhmr_root), Path(args.smplx_model_dir),
                          multi_person=not args.skip_optional)
        print(json.dumps(report, indent=2) if args.json else format_report(report))
        raise SystemExit(0 if report["ready"] else 1)

    if args.command == "score":
        raise SystemExit(0 if score(args)[1] else 1)

    try:
        validate_args(args)
    except ValueError as exc:
        parser.error(str(exc))

    if args.command == "prepare":
        timings = RunTimings(session_dir_from_args(args), "prepare")
        try:
            with timings.stage("prepare", group=True):
                prepare_session(args, timings=timings)
        except BaseException:
            timings.finish(False)
            raise
        timings.finish(True)
        return

    if args.command == "run-all":
        timings = RunTimings(session_dir_from_args(args), "run-all")
        try:
            _, succeeded = execute_video(args, timings)
        except BaseException:
            timings.finish(False)
            raise
        timings.finish(succeeded)
        print(f"[HMB] Timings: {timings.path} ({timings.payload['wall_seconds']:.3f}s)")
        raise SystemExit(0 if succeeded else 1)

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
