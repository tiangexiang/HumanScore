"""Session directory layout and manifest I/O.

A *session* is the working directory of one video, written ``<S>`` in the docstrings
(default ``./humanscore_output/<video stem>``).  Everything a run produces stays in it,
so running the same command again reuses the finished steps::

    <S>/manifest.json        source video identity, artifact paths, options and fingerprints
    <S>/input/               link to (or copy of) the video; frames.json = the analysed frames
    <S>/baseline/            MeTRAbs 3-D keypoints (<stem>_keypoints.npz) and the rigid
                             skeleton fit (<stem>_fitted_model.npz, <stem>_opensim_skeleton.mot)
    <S>/bone_fit/            variable-scale skeleton fit (empty unless --paper-scores)
    <S>/gvhmr/<stem>/        GVHMR body mesh (hmr4d_results.pt)
    <S>/metrics/             per-metric outputs of the scorer (e.g. self_collision/)
    <S>/paper_metrics/       per-metric outputs of the paper implementation (--paper-scores)
    <S>/cache/               engine jobs and logs (engines/), detections, HADM outputs, score cache
    <S>/scores.json          the scorer's detailed report
    <S>/timings.json         wall time of every stage of the last run

``scripts/humanscore_single.py`` (:mod:`hmb.pipeline`) adds ``humanscore.json``,
``summary.txt`` and ``logs/``.  :class:`SessionLayout` names the directories that the
reconstruction creates; the manifest helpers write JSON atomically.
"""
from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


def sanitize_stem(stem: str) -> str:
    """File-system safe version of a video stem (``"my clip (1)"`` -> ``"my_clip_1"``).

    Every artifact of a session is named after this stem, so it must not contain
    spaces, path separators or shell-special characters.
    """
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", stem.strip())
    cleaned = cleaned.strip("._")
    return cleaned or "video"


@dataclass
class SessionLayout:
    """Paths of one session directory (see the module docstring)."""

    session_dir: Path
    input_dir: Path
    baseline_dir: Path
    bone_fit_dir: Path
    gvhmr_root: Path
    metrics_dir: Path
    cache_dir: Path
    manifest_path: Path

    @classmethod
    def create(cls, session_dir: Path) -> "SessionLayout":
        """Layout of ``session_dir``; creates the directories that do not exist yet."""
        layout = cls(
            session_dir=session_dir,
            input_dir=session_dir / "input",
            baseline_dir=session_dir / "baseline",
            bone_fit_dir=session_dir / "bone_fit",
            gvhmr_root=session_dir / "gvhmr",
            metrics_dir=session_dir / "metrics",
            cache_dir=session_dir / "cache",
            manifest_path=session_dir / "manifest.json",
        )
        for path in [
            layout.session_dir,
            layout.input_dir,
            layout.baseline_dir,
            layout.bone_fit_dir,
            layout.gvhmr_root,
            layout.metrics_dir,
            layout.cache_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)
        return layout


def link_or_copy_file(src: Path, dst: Path) -> None:
    """Make ``dst`` a symlink to ``src`` (a copy where symlinks are not supported)."""
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def write_manifest(manifest_path: Path, payload: Dict[str, Any]) -> None:
    """Write ``payload`` as JSON; a temporary file and a rename keep a crash from truncating it."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, allow_nan=False)
    temporary.replace(manifest_path)


def read_manifest(manifest_path: Path) -> Dict[str, Any]:
    """Read a JSON file written by :func:`write_manifest`."""
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)
