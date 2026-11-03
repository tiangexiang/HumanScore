"""Fingerprints that decide when a session's artifacts and scores can be reused.

Running HumanScore again on the same video reuses everything that is still valid.
Every reusable product of a session has a fingerprint: a hash over everything its
numbers depend on.  A product is reused only while its fingerprint is unchanged.
``hmb.cli`` stores the reconstruction fingerprints in ``<S>/manifest.json``
(``config.stage_fingerprints``) and deletes a product whose fingerprint changed;
the HADM detections keep theirs in ``<S>/cache/multi_limbs/inference_identity.json``
and the scores in ``<S>/cache/score_cache.json``.

=================  ==========================================================================
product            fingerprint covers
=================  ==========================================================================
detection          video, MeTRAbs model files, camera calibration, detection code (``baseline/main.py``
                   helpers, ``baseline/monocular_demos/utils.py``, ``hmb/detect_engine.py``), frame
                   plan, detect-engine options, TF/OpenCV versions
rigid / variable   keypoints, detection fingerprint, iterations, camera, skeleton model files, fit code
fit                (``baseline`` or ``metrics/bone_length`` ``monocular_demos``, ``hmb/fit_engine.py``),
                   frame plan, fit-engine options, JAX/MuJoCo versions
mesh               video, SMPL-X model, GVHMR checkpoints and code, mesh-engine options, torch versions
HADM detections    video, HADM checkpoint and code, HADM-engine options, frame plan, torch/detectron2
scores             every session artifact the scorer reads, the scorer code (``hmb/scoring``,
                   ``metrics/kinematic_extreme``, ``metrics/motion_smoothness``; with paper scores also
                   the other ``metrics/`` code and the variable-scale fit), the runtime versions
=================  ==========================================================================

Large model files use filesystem identity (size, nanosecond modification and change time),
avoiding a multi-gigabyte read just to reuse cached numbers; small input/code/result files are
content hashed.  No ML library is imported.
"""

from __future__ import annotations

import hashlib
import ast
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys
from functools import lru_cache

from .hadm_prepare import DEFAULT_CHECKPOINT as HADM_CHECKPOINT, DEFAULT_SCORE_THRESH as HADM_SCORE_THRESH
from .layout import SessionLayout, sanitize_stem, write_manifest

# Format of the score cache record; increase it to invalidate every saved score cache.
CACHE_VERSION = 1
_SMALL_FILE_LIMIT = 8 * 1024 * 1024  # files up to this size are content hashed
# Engine options that change scheduling or side outputs, never the reconstructed numbers.
EXECUTION_ONLY_OPTIONS = frozenset({"jax_cache_dir", "modes", "prefetch_batches", "save_detections",
                                    "save_preprocess", "gmc_workers", "cpu_workers", "warmup"})
#: ``<S>/cache/<SCORE_CACHE_NAME>`` records the fingerprint of the verified ``<S>/scores.json``.
SCORE_CACHE_NAME = "score_cache.json"


@lru_cache(maxsize=2048)
def _digest_by_identity(path: Path, size: int, mtime_ns: int, ctime_ns: int) -> str:
    """SHA-256 of a file; the stat values only key the in-process cache (a changed file is rehashed)."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_file(path: Path) -> str:
    """SHA-256 of a file's content (hashed at most once per process while the file is unchanged)."""
    path = path.resolve()
    stat = path.stat()
    return _digest_by_identity(path, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def file_identity(path: Path) -> dict:
    """Identity of one input file: its content hash when small, its size and times when large."""
    path = path.expanduser().resolve()
    try:
        stat = path.stat()
    except OSError:
        return {"path": str(path), "missing": True}
    identity = {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns}
    if path.is_file() and stat.st_size <= _SMALL_FILE_LIMIT:
        # Metadata-only changes and byte-identical reimports do not invalidate
        # a content-hashed input.
        identity = {"path": str(path), "size": stat.st_size, "sha256": digest_file(path)}
    return identity


#: Distributions that install the same modules under another name; the first one installed is
#: recorded under the requested name (the lightweight environment installs opencv-python-headless).
_ALTERNATIVE_DISTRIBUTIONS = {"opencv-python": ("opencv-python-headless",)}


def _runtime_versions(packages: tuple[str, ...]) -> dict:
    """Installed versions of ``packages`` (None if missing) plus the Python version and interpreter."""
    result = {"python": sys.version, "executable": str(Path(sys.executable).resolve())}
    for package in packages:
        result[package] = None
        for name in (package, *_ALTERNATIVE_DISTRIBUTIONS.get(package, ())):
            try:
                result[package] = importlib.metadata.version(name)
                break
            except importlib.metadata.PackageNotFoundError:
                continue
    return result


def _hash_payload(payload: dict) -> str:
    """The fingerprint: SHA-256 of the payload as canonical JSON."""
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _function_identity(path: Path, names: set[str]) -> dict:
    """Hash of a module's top-level code, keeping only the functions/classes in ``names``.

    Edits to the other functions of a large script (e.g. ``baseline/main.py``) do not
    invalidate a product that never calls them.  Comments and formatting do not count.
    """
    if not path.is_file():
        return file_identity(path)
    module = ast.parse(path.read_text())
    selected = [
        ast.dump(node, include_attributes=False)
        for node in module.body
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        or getattr(node, "name", "") in names
    ]
    return {"path": str(path), "functions_sha256": hashlib.sha256("\n".join(selected).encode()).hexdigest()}


def _code_identity(paths) -> list[dict]:
    """Content hashes of source files (``test_*`` files are ignored)."""
    return [
        {"path": str(path), "sha256": digest_file(path) if path.is_file() else None}
        for path in sorted(set(paths))
        if not path.name.startswith("test_")
    ]


def _package_code(name: str, subdirectory: str = "") -> list[Path]:
    """Source files of an installed package (its ``subdirectory`` only, if given); [] if not installed."""
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return []
    if spec is None or not spec.origin:
        return []
    path = Path(spec.origin)
    if path.suffix != ".py":
        return [path]
    return sorted((path.parent / subdirectory).rglob("*.py")) if subdirectory else [path]


def engine_settings(config: dict, engine: str) -> dict:
    """Numerically relevant options of one engine recorded in the session's config."""
    options = ((config or {}).get("engine_options") or {}).get(engine) or {}
    return {key: options[key] for key in sorted(options) if key not in EXECUTION_ONLY_OPTIONS}


def _frame_plan_path(config: dict, layout: SessionLayout | None = None) -> Path:
    """The session's ``frames.json`` (from the config, else the layout)."""
    return Path((config or {}).get("frame_plan") or (layout.input_dir / "frames.json" if layout else "missing-frames.json"))


def frame_plan_identity(config: dict, layout: SessionLayout | None = None) -> dict:
    """:func:`file_identity` of the session's frame plan."""
    return file_identity(_frame_plan_path(config, layout))


def _plan_sampling(config: dict, layout: SessionLayout | None) -> list:
    """``[sample_fps, max_side]`` stored in the frame plan (the request only when there is no plan).

    A request that selects the plan's frames again keeps that plan file
    (``hmb.batch.plan_session_dir``), so it must not change the fingerprints.
    """
    try:
        plan = json.loads(_frame_plan_path(config, layout).read_text())
        return [plan["sample_fps"], plan["max_side"]]
    except (OSError, ValueError, KeyError, TypeError):
        options = (config or {}).get("requested_fit_options", {})
        return [options.get("sample_fps"), options.get("max_side")]


def _engine_payload(config: dict, layout: SessionLayout | None, engine: str, code: list[Path]) -> dict:
    """The part of a fingerprint that describes the engine run: frame plan, settings and code."""
    return {
        "engine": engine,
        "plan": frame_plan_identity(config, layout),
        "sampling": _plan_sampling(config, layout),
        "settings": engine_settings(config, engine),
        "code": _code_identity(code),
    }


def reconstruction_fingerprints(config: dict, video: Path, layout: SessionLayout, repo: Path) -> dict[str, str]:
    """Fingerprints ``{"detection", "rigid_fit", "variable_fit", "mesh"}`` of a session (see the table above).

    The fits include the detection fingerprint, so new keypoints invalidate both fits.
    """
    options = config.get("requested_fit_options", {})
    baseline = repo / "baseline"
    model = Path(options.get("metrabs_model") or "missing-savedmodel")
    model_files = (
        [model / "saved_model.pb", *sorted((model / "variables").glob("*")), *sorted((model / "assets").rglob("*"))]
        if model.is_dir()
        else []
    )
    detection = {
        "video": file_identity(video),
        "handle": options.get("metrabs_model"),
        "model": [file_identity(path) for path in model_files if path.is_file()],
        "camera": [options.get("camera_calibration"), options.get("camera_calibration_sha256")],
        "functions": _function_identity(
            baseline / "main.py",
            {"configure_runtime", "load_camera_calibration", "save_metrabs_data", "process_videos_with_metrabs"},
        ),
        "code": _code_identity([baseline / "monocular_demos/utils.py", repo / "hmb/metrabs_model.py"]),
        "runtime": _runtime_versions(("tensorflow", "tensorflow-hub", "opencv-python")),
    }
    detection["engine"] = _engine_payload(
        config, layout, "detect", [repo / "hmb/detect_engine.py", repo / "hmb/video.py"])
    detection["engine"]["person"] = options.get("person")
    fingerprints = {"detection": _hash_payload(detection)}
    for name, folder in (("rigid_fit", baseline), ("variable_fit", repo / "metrics/bone_length")):
        modules = folder / "monocular_demos"
        geometry = modules / "biomechanics_mjx/data/humanoid"
        geometry_files = [
            file_identity(path)
            for path in sorted(geometry.rglob("*"))
            if path.is_file() and path.suffix in {".xml", ".stl"}
        ]
        payload = {
            "keypoints": file_identity(layout.baseline_dir / f"{sanitize_stem(video.stem)}_keypoints.npz"),
            "detection": fingerprints["detection"],
            "iterations": options.get("max_iters"),
            "camera": detection["camera"],
            "geometry": geometry_files,
            "functions": _function_identity(
                folder / "main.py",
                {"configure_runtime", "load_metrabs_data", "get_framerate", "process_videos_with_biomechanics"},
            ),
            "code": _code_identity(
                [
                    modules / "camera.py",
                    modules / "dataset.py",
                    modules / "utils.py",
                    modules / "implicit_representation.py",
                    modules / "biomechanics_mjx/monocular_trajectory.py",
                    modules / "biomechanics_mjx/forward_kinematics.py",
                    baseline / "native_export.py",
                    baseline / "metrab2opensim.py",
                ]
            ),
            "runtime": _runtime_versions(("jax", "jaxlib", "equinox", "optax", "mujoco", "mujoco-mjx")),
        }
        # One fit engine serves both fits; it reuses the baseline encoding helpers for both.
        payload["engine"] = _engine_payload(
            config, layout, "fit",
            [repo / "hmb/fit_engine.py", repo / "hmb/fit_export.py", repo / "hmb/video.py",
             baseline / "monocular_demos/implicit_representation.py"])
        fingerprints[name] = _hash_payload(payload)
    gvhmr = Path(config.get("gvhmr_root", repo / "GVHMR"))
    mesh = {
        "video": detection["video"],
        "static_cam": options.get("static_cam"),
        "f_mm": options.get("gvhmr_f_mm"),
        "model": file_identity(
            Path(config.get("smplx_model_dir", repo / "outputs/assets/smplx")) / "SMPLX_NEUTRAL.npz"
        ),
        "checkpoints": [
            file_identity(path)
            for path in sorted((gvhmr / "inputs/checkpoints").rglob("*"))
            if path.is_file() and path.suffix in {".pt", ".pth", ".ckpt"}
        ],
        "code": _code_identity([repo / "hmb/gvhmr_prepare.py", *gvhmr.glob("hmr4d/**/*.py")]),
        "runtime": _runtime_versions(("torch", "torchvision", "pytorch3d", "smplx", "timm", "ultralytics")),
    }
    mesh["engine"] = _engine_payload(config, layout, "mesh", [repo / "hmb/mesh_engine.py", repo / "hmb/video.py"])
    checkpoints = Path(config.get("gvhmr_checkpoints_dir") or repo / "outputs/assets/gvhmr/checkpoints")
    mesh["engine"]["checkpoints"] = [
        file_identity(path) for path in sorted(checkpoints.rglob("*"))
        if path.is_file() and path.suffix in {".pt", ".pth", ".ckpt"}
    ]
    fingerprints["mesh"] = _hash_payload(mesh)
    return fingerprints


def hadm_inference_fingerprint(config: dict, video: Path, repo: Path) -> str:
    """Fingerprint of the session's HADM detections."""
    root = Path(config.get("hadm_root", repo / "metrics/multi-limbs/HADM"))
    payload = {
        "video": file_identity(video),
        "checkpoint": file_identity(root / HADM_CHECKPOINT),
        "code": _code_identity(
            [
                repo / "hmb/hadm_prepare.py",
                *root.glob("projects/ViTDet/**/*.py"),
                *_package_code("detectron2", "modeling"),
            ]
        ),
        "runtime": _runtime_versions(("torch", "torchvision", "detectron2", "timm")),
        "threshold": HADM_SCORE_THRESH,
    }
    payload["engine"] = _engine_payload(config, None, "hadm", [repo / "hmb/hadm_engine.py", repo / "hmb/video.py"])
    return _hash_payload(payload)


# ----------------------------------------------------------------------------- scores
def score_input_files(manifest: dict, layout: SessionLayout, repo: Path, paper_scores: bool = False) -> list[Path]:
    """Session artifacts and assets read by ``hmb.scoring.score_session`` (missing files are recorded too)."""
    stem = sanitize_stem(str(manifest["video_name"]))
    session, config = layout.session_dir, manifest.get("config", {})
    hadm = layout.cache_dir / "multi_limbs" / stem / "hadm_results"
    humanoid = repo / "baseline/monocular_demos/biomechanics_mjx/data/humanoid"
    files = [
        Path(manifest["source_video"]),
        layout.input_dir / "frames.json",
        layout.baseline_dir / f"{stem}_keypoints.npz",
        layout.baseline_dir / f"{stem}_fitted_model.npz",
        layout.baseline_dir / f"{stem}_opensim_skeleton.mot",
        layout.baseline_dir / f"{stem}_opensim_skeleton_grid.mot",
        session / "gvhmr" / stem / "hmr4d_results.pt",
        hadm / "output.json",
        hadm / "frame_indices.json",
        Path(config.get("smplx_model_dir", repo / "outputs/assets/smplx")) / "SMPLX_NEUTRAL.npz",
        repo / "outputs/assets/smplx/SMPLX_NEUTRAL.npz",
        *sorted(p for p in humanoid.rglob("*") if p.is_file() and p.suffix in {".xml", ".stl"}),
    ]
    if paper_scores:
        files += [layout.bone_fit_dir / f"{stem}_bone_length.npz", layout.baseline_dir / f"{stem}_bone_length.npz"]
    return files


def score_code_files(repo: Path, paper_scores: bool = False) -> list[Path]:
    """Code whose change invalidates the scores (``test_*`` files are dropped by the identity)."""
    metrics = repo / "metrics"
    code = [
        *sorted((repo / "hmb/scoring").glob("*.py")),
        repo / "hmb/video.py",
        repo / "hmb/result_cache.py",
        *sorted((metrics / "kinematic_extreme").glob("*.py")),
        *sorted((metrics / "motion_smoothness").glob("*.py")),
    ]
    if paper_scores:
        code += [
            metrics / "bone_length/bone_length.py",
            *sorted((metrics / "range_of_motion").glob("*.py")),
            *sorted((metrics / "self_collision_bvh").glob("*.py")),
            metrics / "multi-limbs/eval_extra_limbs.py",
            *_package_code("mesh_intersection", "."),
        ]
    return code


def score_fingerprint(manifest: dict, layout: SessionLayout, repo: Path, paper_scores: bool = False) -> str:
    """Fingerprint of the session's scores (with or without the paper scores)."""
    packages = ("numpy", "scipy", "pandas", "torch", "smplx", "mujoco")
    payload = {
        "version": CACHE_VERSION,
        "scores": "humanscore",
        "paper_scores": bool(paper_scores),
        "files": [file_identity(path) for path in score_input_files(manifest, layout, repo, paper_scores)],
        "code": _code_identity(score_code_files(repo, paper_scores)),
        "runtime": _runtime_versions(packages + (("mesh-intersection",) if paper_scores else ())),
    }
    return _hash_payload(payload)


def score_cache_path(layout: SessionLayout) -> Path:
    """``<S>/cache/score_cache.json``."""
    return layout.cache_dir / SCORE_CACHE_NAME


def saved_score_request(layout: SessionLayout) -> dict | None:
    """``{"paper_scores": bool}`` of the cached ``scores.json``, else ``None``."""
    try:
        saved = json.loads(score_cache_path(layout).read_text())
        return {"paper_scores": bool(saved["paper_scores"])}
    except (OSError, ValueError, KeyError, TypeError):
        return None


def load_cached_scores(manifest: dict, layout: SessionLayout, repo: Path, paper_scores: bool = False) -> dict | None:
    """The verified ``<S>/scores.json`` report, else ``None``.

    A report with paper scores also serves a request without them, not the other way round.
    """
    report = layout.session_dir / "scores.json"
    try:
        saved = json.loads(score_cache_path(layout).read_text())
        if saved["version"] != CACHE_VERSION or (paper_scores and not saved["paper_scores"]):
            return None
        if saved["fingerprint"] != score_fingerprint(manifest, layout, repo, bool(saved["paper_scores"])):
            return None
        if saved["result_sha256"] != digest_file(report):
            return None
        return json.loads(report.read_text())
    except (OSError, ValueError, KeyError, TypeError):
        return None


def save_cached_scores(fingerprint: str, layout: SessionLayout, paper_scores: bool = False) -> None:
    """Record ``fingerprint`` as the one of the current ``<S>/scores.json``."""
    write_manifest(
        score_cache_path(layout),
        {"version": CACHE_VERSION, "fingerprint": fingerprint, "paper_scores": bool(paper_scores),
         "result_sha256": digest_file(layout.session_dir / "scores.json")},
    )
