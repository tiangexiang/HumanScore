"""Installation check (``python -m hmb.cli doctor``): dependencies and model files.

It works before the numerical stack is fully installed: modules are only looked up
(``importlib.util.find_spec``), no model is loaded and nothing is downloaded.

Every check has a ``status``:

* ``ok``: present;
* ``first-use``: not present yet, but downloaded automatically on first use (it does not make
  the installation "not ready");
* ``missing``: absent; a missing *required* item makes the installation "not ready".

Required checks cover single-person scoring (``scripts/humanscore_single.py``).  Optional checks
cover the multi-person tracker of ``scripts/humanscore_multi.py`` (SAM 3.1 or YOLO) and never
change ``ready``.  The locations checked are the ones the engines use:

* GVHMR, ViTPose, HMR2 and YOLO checkpoints: ``GVHMR/inputs/checkpoints`` or
  ``outputs/assets/gvhmr/checkpoints`` (:func:`hmb.gvhmr_prepare.prepare_assets` downloads them there);
* MeTRAbs: ``$HMB_METRABS_MODEL``, ``outputs/assets/metrabs/`` or the TensorFlow Hub cache;
* YOLO segmentation weights (``--tracker yolo``): ``$HMB_YOLO_SEG_WEIGHTS`` or ``outputs/assets/yolo``.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

from .model_paths import METRABS_HANDLE, METRABS_LOCAL, REPO_ROOT as MODEL_ROOT

#: Python modules the engines and the scorer import (looked up, not imported).
PYTHON_MODULES = ("numpy", "scipy", "pandas", "cv2", "torch", "jax", "jaxlie", "equinox", "optax",
                  "tensorflow", "tensorflow_hub", "mujoco", "smplx", "pytorch3d", "detectron2",
                  "mesh_intersection", "hydra", "ultralytics")
#: Commands single-person scoring needs (ffmpeg is only used by the multi-person script: optional check).
REQUIRED_COMMANDS = ("nvidia-smi",)
#: Directory :func:`hmb.gvhmr_prepare.prepare_assets` downloads the GVHMR checkpoints into.
GVHMR_ASSET_DIR = "outputs/assets/gvhmr/checkpoints"
#: Display names of the checkpoints in :data:`hmb.gvhmr_prepare.CHECKPOINTS`.
GVHMR_CHECKPOINT_NAMES = {
    "gvhmr/gvhmr_siga24_release.ckpt": "GVHMR weights",
    "hmr2/epoch=10-step=25000.ckpt": "HMR2 weights",
    "vitpose/vitpose-h-multi-coco.pth": "ViTPose weights",
    "yolo/yolov8x.pt": "YOLO person detector (GVHMR)",
}
#: Default MeTRAbs handle of ``baseline/monocular_demos/utils.py:load_metrabs``.
METRABS_DEFAULT_HANDLE = METRABS_HANDLE
#: Segmentation weights of ``--tracker yolo``, as in :mod:`hmb.tracking.yolo_backend`.
YOLO_SEG_ENV = "HMB_YOLO_SEG_WEIGHTS"
YOLO_SEG_DEFAULT = "yolov8x-seg.pt"
YOLO_SEG_DIR = "outputs/assets/yolo"
SMPLX_URL = "https://smpl-x.is.tue.mpg.de"
#: Console label of each status.
LABELS = {"ok": "OK", "first-use": "AUTO", "missing": "MISSING"}


def _check(name: str, status: str, detail: str, required: bool = True) -> Dict[str, Any]:
    """One check result: ``{name, status, ok, required, detail}``."""
    if status not in LABELS:
        raise ValueError(f"unknown check status {status!r}")
    return {"name": name, "status": status, "ok": status != "missing", "required": bool(required),
            "detail": detail}


def _file_check(name: str, path: Path, hint: str, required: bool = True) -> Dict[str, Any]:
    """``ok`` when ``path`` is a non-empty file, else ``missing`` with ``hint`` (how to get it)."""
    if path.is_file() and path.stat().st_size > 0:
        return _check(name, "ok", str(path), required)
    return _check(name, "missing", f"{path} not found; {hint}", required)


def module_checks(modules: Iterable[str] = PYTHON_MODULES) -> List[Dict[str, Any]]:
    """Whether each Python module can be found by this interpreter (without importing it)."""
    checks = []
    for module in modules:
        try:
            found = importlib.util.find_spec(module)
        except (ImportError, ValueError) as exc:
            checks.append(_check(f"python:{module}", "missing", str(exc)))
            continue
        checks.append(_check(f"python:{module}", "ok" if found else "missing",
                             str(found.origin) if found else "not installed in this interpreter"))
    return checks


def gvhmr_checkpoint_checks(gvhmr_root: Path, asset_dir: Path) -> List[Dict[str, Any]]:
    """The four GVHMR checkpoints: ``ok`` when either location holds the complete file (exact size),
    else ``first-use`` (``prepare_assets`` downloads them on the first mesh reconstruction)."""
    from .gvhmr_prepare import CHECKPOINTS

    checks = []
    for relative, (size, _) in CHECKPOINTS.items():
        name = GVHMR_CHECKPOINT_NAMES.get(relative, f"GVHMR checkpoint {relative}")
        candidates = (gvhmr_root / "inputs/checkpoints" / relative, asset_dir / relative)
        found = next((p for p in candidates if p.is_file() and p.stat().st_size == size), None)
        if found is not None:
            checks.append(_check(name, "ok", str(found)))
            continue
        partial = next((p for p in candidates if p.is_file()), None)
        note = (f"incomplete file {partial} ({partial.stat().st_size} of {size} bytes); " if partial else "")
        checks.append(_check(name, "first-use", f"{note}downloaded on first use ({size / 1e9:.1f} GB) "
                                                f"into {asset_dir / relative}"))
    return checks


def metrabs_check(environ: Mapping[str, str], local: Optional[Path] = None) -> Dict[str, Any]:
    """MeTRAbs: ``$HMB_METRABS_MODEL`` (local SavedModel or hub handle), else ``local`` (the SavedModel
    that ``install.sh`` unpacks, :data:`hmb.model_paths.METRABS_LOCAL`), else the default handle,
    which TensorFlow Hub downloads on first use into ``$TFHUB_CACHE_DIR`` (default: the system
    temporary directory) under the SHA-1 of the handle."""
    value = environ.get("HMB_METRABS_MODEL", "")
    if not value and local is not None and (Path(local) / "saved_model.pb").is_file():
        return _check("MeTRAbs model", "ok", f"{local} (installed by install.sh)")
    if value and "://" not in value:
        path = Path(value).expanduser()
        if (path / "saved_model.pb").is_file():
            return _check("MeTRAbs model", "ok", f"{path} (HMB_METRABS_MODEL)")
        return _check("MeTRAbs model", "missing",
                      f"HMB_METRABS_MODEL={value} has no saved_model.pb; point it to an unpacked SavedModel "
                      f"or unset it to download {METRABS_DEFAULT_HANDLE} on first use")
    handle = value or METRABS_DEFAULT_HANDLE
    cache = Path(environ.get("TFHUB_CACHE_DIR") or os.path.join(tempfile.gettempdir(), "tfhub_modules"))
    module = cache.expanduser() / hashlib.sha1(handle.encode("utf8")).hexdigest()
    if (module / "saved_model.pb").is_file():
        return _check("MeTRAbs model", "ok", f"{handle}, cached at {module}")
    keep = "" if environ.get("TFHUB_CACHE_DIR") else "; set TFHUB_CACHE_DIR to keep it outside the temporary directory"
    return _check("MeTRAbs model", "first-use",
                  f"downloaded on first use from {handle} into the TensorFlow Hub cache {cache}{keep}; `bash install.sh` "
                  "unpacks it into outputs/assets/metrabs/ instead, or set HMB_METRABS_MODEL to a local SavedModel")


def yolo_seg_check(repo_root: Path, environ: Mapping[str, str]) -> Dict[str, Any]:
    """Segmentation weights of ``--tracker yolo`` (optional); see :mod:`hmb.tracking.yolo_backend`."""
    name = "YOLO segmentation weights (--tracker yolo)"
    explicit = environ.get(YOLO_SEG_ENV)
    weights = YOLO_SEG_DEFAULT
    if explicit:
        path = Path(explicit).expanduser()
        if path.exists():
            return _check(name, "ok", f"{path} ({YOLO_SEG_ENV})", required=False)
        if path.parent != Path("."):
            return _check(name, "missing", f"{YOLO_SEG_ENV}={explicit} does not exist", required=False)
        weights = path.name
    target = repo_root / YOLO_SEG_DIR / weights
    if target.exists():
        return _check(name, "ok", str(target), required=False)
    return _check(name, "first-use", f"{weights} is downloaded on first use of --tracker yolo into "
                                     f"{target.parent}", required=False)


@contextlib.contextmanager
def _offline_hub():
    """No Hugging Face download while checking: the SAM 3.1 checkpoint is several GB."""
    previous = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = previous


def default_sam3_check() -> dict:
    """:func:`hmb.tracking.check_sam3` (imports torch and sam3 in the SAM 3.1 environment; loads no
    model) with downloads disabled."""
    from .tracking.sam3_backend import check_sam3

    with _offline_hub():
        return check_sam3()


def sam3_check(check: Callable[[], dict] = default_sam3_check) -> Dict[str, Any]:
    """SAM 3.1 tracker (optional): its Python environment, CUDA and a local checkpoint."""
    name = "SAM 3.1 tracker (--tracker sam3.1, default of humanscore_multi.py)"
    fallback = "without it, run humanscore_multi.py with --tracker yolo"
    try:
        info = check()
    except Exception as exc:  # noqa: BLE001 - every failure is reported, none is fatal
        kind = type(exc).__name__
        first = next((line.strip() for line in str(exc).splitlines() if line.strip()), kind)
        first = first.split(" Details:")[0].rstrip(".")
        if kind == "SAM3CheckpointError":
            detail = (f"environment found, checkpoint not available locally ({first}). After access to "
                      "facebook/sam3.1 was granted, run `hf auth login` and `bash install.sh --with-sam3` (it downloads "
                      "the checkpoint into outputs/assets/sam3/), or set HMB_SAM3_CHECKPOINT to a local "
                      "sam3.1_multiplex.pt (doctor never downloads it); " + fallback)
        elif kind == "SAM3SetupError":
            detail = (f"not set up ({first}); install it with `bash install.sh --with-sam3` "
                      f"(install/INSTALL.md, SAM 3.1); {fallback}")
        else:
            detail = f"check failed ({kind}: {first}); {fallback}"
        return _check(name, "missing", detail, required=False)
    detail = (f"python {info.get('python')}, torch {info.get('torch')}, checkpoint {info.get('checkpoint')}")
    if not info.get("cuda"):
        return _check(name, "missing", f"{detail}; CUDA is not available in the SAM 3.1 environment",
                      required=False)
    return _check(name, "ok", detail, required=False)


def ffmpeg_check(environ: Mapping[str, str], which: Callable[[str], Optional[str]] = shutil.which) -> Dict[str, Any]:
    """Encoder of the multi-person videos (:func:`hmb.tracking.render.find_ffmpeg`; optional)."""
    name = "ffmpeg (multi-person videos)"
    exe = environ.get("HMB_FFMPEG") or which("ffmpeg")
    if exe:
        return _check(name, "ok", exe, required=False)
    if importlib.util.find_spec("imageio_ffmpeg") is not None:
        return _check(name, "ok", "imageio-ffmpeg", required=False)
    return _check(name, "missing", "not on PATH (install.sh adds imageio-ffmpeg when ffmpeg is missing); person "
                                   "videos fall back to the OpenCV encoder", required=False)


def diagnose(repo_root: Path, hadm_root: Path, gvhmr_root: Path, smplx_dir: Path, *,
             environ: Optional[Mapping[str, str]] = None, modules: Iterable[str] = PYTHON_MODULES,
             commands: Iterable[str] = REQUIRED_COMMANDS, multi_person: bool = True,
             sam3: Callable[[], dict] = default_sam3_check,
             which: Callable[[str], Optional[str]] = shutil.which) -> Dict[str, Any]:
    """All checks; ``ready`` is true when no required check is ``missing``."""
    from .hadm_prepare import DEFAULT_CHECKPOINT, DEFAULT_CONFIG

    environ = os.environ if environ is None else environ
    repo_root, hadm_root, gvhmr_root = Path(repo_root), Path(hadm_root), Path(gvhmr_root)
    restore = "run bash install/check_sources.sh --init"
    checks = module_checks(modules)
    checks += [
        _file_check("HADM inference source", hadm_root / DEFAULT_CONFIG, restore),
        _file_check("HADM-G weights", hadm_root / DEFAULT_CHECKPOINT,
                    "run `bash install.sh` (downloads HADM-G_0249999.pth from the link in "
                    "https://github.com/wangkaihong/HADM into outputs/assets/hadm/ and links it here)"),
        _file_check("GVHMR source", gvhmr_root / "hmr4d/configs/demo.yaml", restore),
    ]
    checks += gvhmr_checkpoint_checks(gvhmr_root, repo_root / GVHMR_ASSET_DIR)
    checks.append(_file_check(
        "SMPL-X neutral model", Path(smplx_dir) / "SMPLX_NEUTRAL.npz",
        f"licensed: register at {SMPLX_URL}, download SMPL-X v1.1 and run `bash install.sh --smplx-dir "
        f"/path/to/models_smplx_v1_1.zip`, or place SMPLX_NEUTRAL.npz (a symlink is fine) in "
        f"{repo_root / 'outputs/assets/smplx'}, or set HMB_SMPLX_MODEL to the file"))
    checks.append(_file_check(
        "biomechanics skeleton",
        repo_root / "baseline/monocular_demos/biomechanics_mjx/data/humanoid/humanoid_torque.xml",
        "part of the repository; check the clone"))
    checks.append(metrabs_check(environ, repo_root / METRABS_LOCAL.relative_to(MODEL_ROOT)))
    for command in commands:
        path = which(command)
        checks.append(_check(command, "ok" if path else "missing", path or "not on PATH"))
    if multi_person:
        checks += [sam3_check(sam3), yolo_seg_check(repo_root, environ), ffmpeg_check(environ, which)]
    required = [c for c in checks if c["required"]]
    return {"python": sys.executable,
            "ready": all(c["ok"] for c in required),
            "first_use": [c["name"] for c in checks if c["status"] == "first-use"],
            "optional_missing": [c["name"] for c in checks if not c["required"] and not c["ok"]],
            "checks": checks}


def format_report(report: Mapping[str, Any]) -> str:
    """Console form of :func:`diagnose`'s report."""
    def line(check):
        label = LABELS[check["status"]] if check["required"] or check["ok"] else "OPTIONAL"
        return f"{label:8s} {check['name']}: {check['detail']}"

    checks = report["checks"]
    lines = [f"[HMB] Python: {report['python']}", "Required (single-person scoring):"]
    lines += [line(c) for c in checks if c["required"]]
    optional = [c for c in checks if not c["required"]]
    if optional:
        lines += ["Optional (multi-person tracking, scripts/humanscore_multi.py):"]
        lines += [line(c) for c in optional]
    missing = [c["name"] for c in checks if c["required"] and not c["ok"]]
    if report["ready"]:
        summary = "Ready."
        if report.get("first_use"):
            summary += f" {len(report['first_use'])} item(s) marked AUTO are downloaded on first use."
    else:
        summary = f"Not ready: {len(missing)} required item(s) missing ({', '.join(missing)})."
    if report.get("optional_missing"):
        summary += f" Optional, not available: {', '.join(report['optional_missing'])}."
    lines.append(summary)
    return "\n".join(lines)
