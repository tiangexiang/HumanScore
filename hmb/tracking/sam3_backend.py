"""SAM 3.1 tracking backend: runs :mod:`hmb.tracking.sam3_worker` as a subprocess.

``bash install.sh --with-sam3`` installs the ``sam3`` package into the HumanScore
environment itself, so no second Python environment is needed; a separate SAM 3.1
environment also works.  The worker always runs in its own process, so the tracker's
GPU memory is released before the scoring engines start.  Its interpreter is, in order:

1. the ``python`` argument;
2. ``$HMB_SAM3_PYTHON`` (a separate SAM 3.1 environment);
3. the current interpreter, when ``sam3`` is importable in it.

Otherwise :class:`SAM3SetupError` explains how to install it.  The checkpoint
(``facebook/sam3.1`` / ``sam3.1_multiplex.pt``) is gated on Hugging Face; when it
cannot be obtained, :class:`SAM3CheckpointError` explains how to get access.  The
checkpoint is ``$HMB_SAM3_CHECKPOINT`` if set, else the copy that ``install.sh --with-sam3``
puts in ``outputs/assets/sam3/``, else the Hugging Face cache.
"""
from __future__ import annotations

import collections
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

from .. import process_groups
from .tracks import Tracks

WORKER = Path(__file__).with_name("sam3_worker.py")
# environment variables of a separate SAM 3.1 interpreter and of a local checkpoint
ENV_PYTHON = "HMB_SAM3_PYTHON"
ENV_CHECKPOINT = "HMB_SAM3_CHECKPOINT"
HF_REPO = "facebook/sam3.1"
HF_FILE = "sam3.1_multiplex.pt"

SETUP_HELP = f"""\
SAM 3.1 (package sam3, https://github.com/facebookresearch/sam3) is installed into the HumanScore
environment itself; no second Python environment is needed. One-time setup, from the repository root:

    bash install.sh --with-sam3                # adds sam3, checks access, downloads the checkpoint

A separate SAM 3.1 environment also works: set {ENV_PYTHON} to its python.
The SAM 3.1 weights are gated: request access at https://huggingface.co/{HF_REPO} first.
Alternatively use the YOLO tracker (--tracker yolo), which needs no extra setup."""

CHECKPOINT_HELP = f"""\
The SAM 3.1 checkpoint ({HF_REPO}, file {HF_FILE}) is gated on Hugging Face. To use the sam3.1 tracker:
  1. request access at https://huggingface.co/{HF_REPO} (log in, accept the license on the model page);
  2. once granted, log in from the environment that runs SAM 3.1: `hf auth login` (its `hf` CLI, e.g. {{hf}}),
     with a token that can read gated repositories (if HF_HOME is set, the token is stored under $HF_HOME),
     and run `bash install.sh --with-sam3` again: it downloads {HF_FILE} into outputs/assets/sam3/;
  3. or download {HF_FILE} yourself and set {ENV_CHECKPOINT}=/path/to/{HF_FILE}.
Or use the YOLO tracker instead (--tracker yolo / backend="yolo")."""


class SAM3SetupError(RuntimeError):
    """SAM 3.1 environment missing or broken."""


class SAM3CheckpointError(RuntimeError):
    """The gated SAM 3.1 checkpoint could not be obtained."""


def resolve_python(python: Optional[str] = None) -> str:
    """Interpreter of the SAM 3.1 environment (argument, ``$HMB_SAM3_PYTHON``, or the current one)."""
    for source, candidate in (("python argument", python), (f"${ENV_PYTHON}", os.environ.get(ENV_PYTHON))):
        if candidate:
            path = Path(candidate).expanduser()
            if not path.exists():
                raise SAM3SetupError(f"{source} points to {candidate}, which does not exist.\n\n{SETUP_HELP}")
            if not os.access(path, os.X_OK):
                raise SAM3SetupError(f"{source} points to {candidate}, which is not executable.")
            return str(path)
    if importlib.util.find_spec("sam3") is not None:
        return sys.executable
    raise SAM3SetupError(f"The sam3.1 tracker needs SAM 3.1, which was not found: {ENV_PYTHON} is not set "
                         f"and the package 'sam3' is not importable in {sys.executable}.\n\n{SETUP_HELP}")


def _hf_cli(python: str) -> str:
    """The ``hf`` command next to ``python`` (named in the login instructions), else plain ``hf``."""
    candidate = Path(python).with_name("hf")
    return str(candidate) if candidate.exists() else "hf"


def worker_env(python: str) -> dict:
    """Environment of the worker: a foreign interpreter must not see this environment's PYTHONPATH."""
    env = dict(os.environ)
    if os.path.realpath(python) != os.path.realpath(sys.executable):
        for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONUSERBASE"):
            env.pop(key, None)
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def _run_worker(cmd, env, quiet: bool):
    """Run the worker, relaying its stderr unless ``quiet``; returns (code, error dict or None, tail)."""
    tail = collections.deque(maxlen=60)
    error = None
    # The worker gets a process group of its own, stopped on Ctrl-C and (while it runs) on SIGTERM/SIGHUP,
    # which raise hmb.process_groups.Terminated: an orphaned worker would keep ~25 GB of GPU memory.
    with process_groups.raise_on_termination():
        proc = process_groups.popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True,
                                    errors="replace", bufsize=1)
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                line = line.rstrip("\n")
                if line.startswith("HMB_SAM3_ERROR "):
                    try:
                        error = json.loads(line[len("HMB_SAM3_ERROR "):])
                    except json.JSONDecodeError:
                        error = {"kind": "runtime", "message": line}
                    continue
                tail.append(line)
                if not quiet:
                    print(line, file=sys.stderr, flush=True)
            code = proc.wait()
        except BaseException:
            process_groups.terminate([proc])
            raise
        finally:
            proc.stdout.close()
    return code, error, list(tail)


def _raise_for(code: int, error: Optional[dict], tail, python: str) -> None:
    """Raise the error matching a failed worker run (exit codes of :mod:`hmb.tracking.sam3_worker`)."""
    message = (error or {}).get("message") or "\n".join(tail[-20:]) or f"exit code {code}"
    kind = (error or {}).get("kind")
    if code == 3 or kind == "checkpoint":
        raise SAM3CheckpointError(f"SAM 3.1 checkpoint not available: {message}\n\n"
                                  + CHECKPOINT_HELP.replace("{hf}", _hf_cli(python)))
    if code == 4 or kind == "import":
        raise SAM3SetupError(f"SAM 3.1 environment is broken: {message}\n\n{SETUP_HELP}")
    details = "\n".join(tail[-20:])
    raise RuntimeError(f"SAM 3.1 tracking failed ({kind or 'exit code ' + str(code)}): {message}"
                       + (f"\n--- worker output (last lines) ---\n{details}" if details else ""))


def check_sam3(python: Optional[str] = None, checkpoint: Optional[str] = None) -> dict:
    """Verify the SAM 3.1 environment and checkpoint access without running the model."""
    python = resolve_python(python)
    cmd = [python, "-P", str(WORKER), "-", "--out", "-", "--check"]
    if checkpoint:
        cmd += ["--checkpoint", str(checkpoint)]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=worker_env(python))
    error = None
    for line in proc.stderr.splitlines():
        if line.startswith("HMB_SAM3_ERROR "):
            error = json.loads(line[len("HMB_SAM3_ERROR "):])
    if proc.returncode != 0:
        _raise_for(proc.returncode, error, proc.stderr.splitlines(), python)
    return json.loads(proc.stdout.strip().splitlines()[-1])


def track_sam3(video, *, prompt: str = "person", python: Optional[str] = None,
               checkpoint: Optional[str] = None, max_objects: int = 16, chunk_frames: int = 600,
               chunk_overlap: int = 16, compile_model: bool = False, offload_video: str = "auto",
               work_dir=None, quiet: bool = False) -> Tracks:
    """Track every ``prompt`` instance through ``video`` with SAM 3.1 (see the module docstring).

    SAM 3.1 keeps per-frame state for the whole session (GPU memory grows by
    roughly 25 MB per 720p frame on top of ~25 GB), so videos longer than
    ``chunk_frames`` are tracked in sessions of ``chunk_frames`` frames that
    share ``chunk_overlap`` frames; ids continue across chunks by mask overlap
    (a person absent at a chunk boundary gets a new id).  ``chunk_frames=0``
    tracks the whole video in one session.
    """
    video = Path(video).resolve()
    python = resolve_python(python)
    checkpoint = checkpoint or os.environ.get(ENV_CHECKPOINT)
    with tempfile.TemporaryDirectory(prefix="hmb_sam3_", dir=work_dir) as tmp:
        out = Path(tmp) / "sam3_tracks.npz"
        cmd = [python, "-P", str(WORKER), str(video), "--out", str(out), "--prompt", prompt,
               "--max-objects", str(int(max_objects)), "--offload-video", offload_video,
               "--chunk-frames", str(int(chunk_frames)), "--chunk-overlap", str(int(chunk_overlap))]
        if checkpoint:
            cmd += ["--checkpoint", str(checkpoint)]
        if compile_model:
            cmd.append("--compile")
        code, error, tail = _run_worker(cmd, worker_env(python), quiet)
        if code != 0 or not out.exists():
            _raise_for(code, error, tail, python)
        with np.load(out) as z:
            data = {k: z[k] for k in z.files}
        meta = json.loads(Path(str(out) + ".json").read_text()) if Path(str(out) + ".json").exists() else {}
    height, width = (int(v) for v in data["size"])
    meta.update(python=python)
    tracks = Tracks(video=str(video), fps=float(data["fps"]), num_frames=int(data["num_frames"]),
                    width=width, height=height, frame=data["frame"].astype(np.int32),
                    obj_id=data["obj_id"].astype(np.int32), box=data["box"].astype(np.float32).reshape(-1, 4),
                    score=data["score"].astype(np.float32), rle_offsets=data["rle_offsets"].astype(np.int64),
                    rle_counts=data["rle_counts"].astype(np.uint32), backend="sam3.1", meta={"sam3": meta})
    return tracks.sorted()
