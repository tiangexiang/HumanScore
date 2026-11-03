"""Biomechanical skeleton-fitting engine: rigid and variable-scale fits to the keypoints.

Pipeline position (see ``hmb.batch``): ``plan -> detect -> fit -> score``.  The
fit turns the MeTRAbs 3-D keypoints of ``hmb.detect_engine`` into the motion of
a MuJoCo humanoid skeleton (joint angles over time), the input of the joint-range,
kinematic-extremes and motion-smoothness scores.  Two fits exist:

* ``rigid``: one scale per body segment for the whole clip (``baseline/main.py``);
  always run.
* ``variable``: time-varying segment scales (``metrics/bone_length/main.py``);
  only needed for the paper's bone-length score (``--paper-scores``).

The engine optimises the same objective as those two scripts, with the same
initialisation, data order, optimizer and learning-rate schedule per iteration,
but

* loads JAX/MuJoCo and builds/compiles the training program once per process,
* fuses ``steps_per_call`` optimizer iterations into one compiled ``lax.scan``,
* pads clips to a frame-count bucket (zero-confidence copies of the last frame
  at the last timestamp; masked out of the scale-smoothness term) so videos of
  different lengths share one compiled program, and
* ``vmap``-s independent fits of ``batch`` videos through that program.

Command line::

    python -m hmb.fit_engine --jobs jobs.json [--force]

``jobs.json``::

    {"options": {"max_iters": 10000, "steps_per_call": 200, "batch": 4,
                 "modes": ["rigid", "variable"], "export_grid_hz": null, "unconstrained_joints": true},
     "jobs": [{"session_dir": "...", "stem": "...", "keypoints": "...?", "video": "...?"}]}

Per job it writes the files of the original scripts (plus a few ``fit_*``
provenance keys inside the npz files):

* rigid: ``<S>/baseline/<stem>_fitted_model.npz`` and ``<S>/baseline/<stem>_opensim_skeleton.mot``
  (at the analysed frames), and with ``export_grid_hz`` the same fit sampled on a
  uniform time grid, ``<stem>_fitted_model_grid.npz`` and ``<stem>_opensim_skeleton_grid.mot``
* variable: ``<S>/bone_fit/<stem>_bone_length.npz`` and a copy ``<S>/baseline/<stem>_bone_length.npz``

and a status file ``<S>/cache/engine_status/fit_engine.json``. Jobs whose
outputs exist are skipped unless ``--force``; one failing job never stops the
others. Python API: ``FitEngine(options).load()`` then ``process(job)`` or
``process_many(jobs)``.

Environment: ``HMB_JAX_CACHE_DIR`` (compiled-program cache), ``HMB_FIT_AUTOTUNE=1``
(see :func:`deterministic_xla_flags`), ``HMB_FIT_ENGINE_WATCHDOG=<seconds>``
(periodic stack dumps to stderr, to diagnose a stalled fit).
"""
from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import math
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
ENGINE = "fit_engine"
MODES = ("rigid", "variable")
RIGID_XML = REPO_ROOT / "baseline/monocular_demos/biomechanics_mjx/data/humanoid/humanoid_torque.xml"
VARIABLE_XML = REPO_ROOT / "metrics/bone_length/monocular_demos/biomechanics_mjx/data/humanoid/humanoid_torque.xml"
# Clip lengths (frames) are padded up to one of these, so clips share compiled programs.
DEFAULT_BUCKETS = (32, 48, 64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512, 640, 768, 1024)

# Fit settings of the original scripts: baseline/main.py and metrics/bone_length/main.py call
# fit_model(model, dataset, lr_init_value=1e-3, max_iters=N[, scale_smoothness_regularization=1e3])
# and fit_model's defaults supply the rest (static camera: dt = 1/29 s).
LOSS_SETTINGS = dict(site_offset_regularization=1e2, lambda_reprojection=1e-1, static=True, center_3d=True, dt=1.0 / 29.0)
VARIABLE_SCALE_SMOOTHNESS = 1e3
LR_INIT, LR_END, CLIP_NORM = 1e-3, 1e-7, 0.1


@dataclasses.dataclass
class FitOptions:
    """Options of ``jobs.json`` (``"options"``)."""

    max_iters: int = 10000                 # optimizer iterations per fit
    steps_per_call: int = 200              # iterations fused into one compiled call
    batch: int = 4                         # videos fitted together (vmap) per compiled program
    modes: tuple = MODES                   # fits to run: rigid and/or variable
    export_grid_hz: Optional[float] = None  # also export the rigid fit on a uniform grid at this rate
    unconstrained_joints: bool = True      # False: enforce the model's joint limits during the fit
    buckets: tuple = DEFAULT_BUCKETS       # padded clip lengths (frames)
    pad_batches: bool = True               # fill partial batches with copies (fewer programs to compile)
    camera_calibration: Optional[str] = None  # JSON intrinsics; default: those stored with the keypoints
    jax_cache_dir: Optional[str] = None    # persistent compilation cache (default $HMB_JAX_CACHE_DIR)
    trajectory_kwargs: Optional[dict] = None  # extra ImplicitTrajectory arguments (tests use a tiny network)

    @classmethod
    def from_dict(cls, values: Optional[dict]) -> "FitOptions":
        """Options from a dict; unknown keys raise."""
        values = dict(values or {})
        unknown = set(values) - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"Unknown fit_engine options: {sorted(unknown)}")
        options = cls(**values)
        options.modes = tuple(options.modes)
        options.buckets = tuple(sorted(int(b) for b in options.buckets))
        options.validate()
        return options

    def validate(self) -> None:
        if self.max_iters < 1 or self.steps_per_call < 1 or self.batch < 1:
            raise ValueError("max_iters, steps_per_call and batch must be positive")
        if not self.modes or any(mode not in MODES for mode in self.modes):
            raise ValueError(f"modes must be a non-empty subset of {MODES}")
        if self.export_grid_hz is not None and not self.export_grid_hz > 0:
            raise ValueError("export_grid_hz must be positive")

    def as_dict(self) -> dict:
        values = dataclasses.asdict(self)
        values["modes"] = list(self.modes)
        values["buckets"] = list(self.buckets)
        return values


def deterministic_xla_flags(flags: Optional[str] = None) -> str:
    """XLA_FLAGS with GPU kernel autotuning disabled, unless the caller chose a level.

    With autotuning on, XLA picks GPU kernels by timing them at compile time, so two fresh
    compiles of the same fit can use different kernels; the non-convex skeleton fit can then end
    in a different local optimum on hard clips and give noticeably different scores.  With
    ``--xla_gpu_autotune_level=0`` the fits of two fresh compiles are bit-identical, with the same
    loss and speed.  ``HMB_FIT_AUTOTUNE=1`` keeps XLA's default autotuning.
    """
    flags = (os.environ.get("XLA_FLAGS", "") if flags is None else flags).strip()
    if os.environ.get("HMB_FIT_AUTOTUNE", "0") not in ("", "0", "false", "False") or "xla_gpu_autotune_level" in flags:
        return flags
    return (flags + " --xla_gpu_autotune_level=0").strip()


def configure_jax(cache_dir: Optional[str] = None) -> None:
    """The runtime configuration of ``baseline/main.py:configure_runtime`` (x64, persistent cache),
    plus reproducible GPU kernel selection (:func:`deterministic_xla_flags`; XLA reads the flags
    when the GPU backend initialises, i.e. before the first compile)."""
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ["XLA_FLAGS"] = deterministic_xla_flags()
    import jax

    cache_dir = cache_dir or os.environ.get("HMB_JAX_CACHE_DIR",
                                            os.environ.get("JAX_COMPILATION_CACHE_DIR", str(REPO_ROOT / ".jax_cache")))
    jax.config.update("jax_compilation_cache_dir", cache_dir)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_enable_x64", True)


def bucket_length(frames: int, buckets: Iterable[int]) -> int:
    """Smallest bucket >= frames; beyond the largest bucket round up to a multiple of 128."""
    for bucket in sorted(buckets):
        if bucket >= frames:
            return int(bucket)
    return int(128 * math.ceil(frames / 128))


def session_paths(session_dir: Path, stem: str) -> dict:
    """Input, output and status paths of one session (see the module docstring)."""
    session_dir = Path(session_dir)
    return {
        "keypoints": session_dir / "baseline" / f"{stem}_keypoints.npz",
        "fitted_model": session_dir / "baseline" / f"{stem}_fitted_model.npz",
        "mot": session_dir / "baseline" / f"{stem}_opensim_skeleton.mot",
        # uniform-grid companions (only with export_grid_hz): same rigid fit, evaluated on k / grid_hz
        "fitted_model_grid": session_dir / "baseline" / f"{stem}_fitted_model_grid.npz",
        "mot_grid": session_dir / "baseline" / f"{stem}_opensim_skeleton_grid.mot",
        "bone_fit": session_dir / "bone_fit" / f"{stem}_bone_length.npz",
        "bone_copy": session_dir / "baseline" / f"{stem}_bone_length.npz",
        "bone_keypoints": session_dir / "bone_fit" / f"{stem}_keypoints.npz",
        "video": session_dir / "input" / f"{stem}.mp4",
        "plan": session_dir / "input" / "frames.json",
        "status": session_dir / "cache" / "engine_status" / f"{ENGINE}.json",
    }


def mode_outputs(paths: dict, mode: str, grid: bool = False) -> list:
    """Files that one fit mode writes (``grid``: including the uniform-grid export)."""
    if mode == "rigid":
        return [paths["fitted_model"], paths["mot"]] + ([paths["fitted_model_grid"], paths["mot_grid"]] if grid else [])
    return [paths["bone_fit"], paths["bone_copy"]]


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(payload, indent=1, default=str))
    tmp.replace(path)


def _atomic_npz(path: Path, **arrays) -> None:
    """``np.savez`` to a temporary file, then rename, so readers never see a partial file."""
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "wb") as handle:
        np.savez(handle, **arrays)
    tmp.replace(path)


@dataclasses.dataclass
class PreparedVideo:
    """Host-side inputs of one job, already in the fitter's dtypes and conventions."""

    job: dict
    stem: str
    session_dir: Path
    paths: dict
    timestamps: Any  # (T,) aligned so timestamps[0] == 0 (MonocularDataset convention)
    keypoints2d: Any  # (T, K, 2)
    keypoints3d: Any  # (T, K, 3)
    confs: Any  # (T, K)
    camera: dict  # mtx (1,4)/1000, dist (1,5), rvec, tvec
    calibration_source: str
    keypoint_calibration_source: str
    timestamp_source: str

    @property
    def frames(self) -> int:
        return int(len(self.timestamps))


def keypoint_residual_mm(sites_m, keypoints3d_mm, confs) -> Optional[float]:
    """Mean 3D distance (mm) between fitted sites and MeTRAbs keypoints over detected samples,
    both centred per frame as in the fitting loss (fit-quality diagnostic, not a metric)."""
    import numpy as np

    sites = np.asarray(sites_m) * 1000.0
    keypoints = np.asarray(keypoints3d_mm, dtype=float)
    if sites.shape != keypoints.shape:
        return None
    sites = sites - sites.mean(axis=1, keepdims=True)
    keypoints = keypoints - keypoints.mean(axis=1, keepdims=True)
    mask = np.asarray(confs) > 0
    if not mask.any():
        return None
    return float((np.linalg.norm(sites - keypoints, axis=-1) * mask).sum() / mask.sum())


def load_video_inputs(job: dict, camera_calibration: Optional[str] = None) -> PreparedVideo:
    """Resolve keypoints, timestamps and intrinsics for one job (no JAX).

    Timestamps: keypoints npz ``timestamps`` (s) if present, else the session's
    frame plan (``input/frames.json``) timestamps, else ``arange(T) / fps`` with
    fps from the npz ``fps`` key or the video container. Intrinsics: the npz
    ``camera_intrinsics_px``/``camera_distortion`` written by detection, else the
    default of ``baseline/main.py`` (MeTRAbs 55 deg FOV) for the plan's output size
    or the video size.
    """
    import numpy as np

    session_dir = Path(job["session_dir"]).expanduser().resolve()
    stem = job.get("stem") or (Path(job["video"]).stem if job.get("video") else None)
    if not stem:
        raise ValueError("job needs a stem (or a video to derive it from)")
    paths = session_paths(session_dir, stem)
    keypoint_path = Path(job.get("keypoints") or paths["keypoints"])
    if not keypoint_path.is_file():
        raise FileNotFoundError(f"No MeTRAbs keypoints at {keypoint_path}")
    video = Path(job["video"]) if job.get("video") else paths["video"]
    plan = None
    if paths["plan"].is_file():
        from hmb.video import FramePlan

        plan = FramePlan.load(paths["plan"])

    with np.load(keypoint_path, allow_pickle=False) as data:
        keypoints2d = np.asarray(data["keypoints2d"])
        keypoints3d = np.asarray(data["keypoints3d"])
        confs = np.asarray(data["confs"])
        stored_ts = np.asarray(data["timestamps"], dtype=float) if "timestamps" in data else None
        stored_fps = float(np.asarray(data["fps"])) if "fps" in data else None
        intrinsics = np.asarray(data["camera_intrinsics_px"], dtype=float).reshape(-1) if "camera_intrinsics_px" in data else None
        distortion = np.asarray(data["camera_distortion"], dtype=float).reshape(-1) if "camera_distortion" in data else None
        keypoint_source = str(data["camera_calibration_source"].item()) if "camera_calibration_source" in data else "unknown_cached"
    frames = len(keypoints2d)
    if frames == 0 or not np.any(confs > 0):
        raise ValueError(f"Video {stem}: no valid person detections found")
    if keypoints3d.shape[:2] != (frames, keypoints2d.shape[1]) or confs.shape != keypoints2d.shape[:2]:
        raise ValueError("keypoints2d, keypoints3d and confs disagree in shape")

    if stored_ts is not None:
        timestamps, timestamp_source = stored_ts, "keypoints_npz"
    elif plan is not None and len(plan.timestamps) == frames:
        timestamps, timestamp_source = np.asarray(plan.timestamps, dtype=float), "frame_plan"
    else:
        if stored_fps is not None:
            fps, timestamp_source = stored_fps, "keypoints_fps"
        else:
            from baseline.main import get_framerate

            fps, timestamp_source = get_framerate(video), "video_fps"
        timestamps = np.arange(0, frames) / fps
    if timestamps.shape != (frames,) or not np.all(np.isfinite(timestamps)) or np.any(np.diff(timestamps) <= 0):
        raise ValueError("timestamps must be finite, strictly increasing and one per keypoint frame")
    timestamps = timestamps - timestamps[0]

    if camera_calibration is None and intrinsics is not None:
        # The intrinsics the keypoints were detected with (the fitter works in metres: px / 1000).
        camera = {
            "mtx": intrinsics[:4].reshape(1, 4) / 1000.0,
            "dist": (distortion if distortion is not None else np.zeros(5)).reshape(1, 5),
            "rvec": np.zeros((1, 3)),
            "tvec": np.zeros((1, 3)),
        }
        calibration_source = keypoint_source
    else:
        from baseline.main import load_camera_calibration

        image_size = plan.output_size if plan is not None else None
        camera = load_camera_calibration(camera_calibration, video_path=video, image_size=image_size)
        calibration_source = str(camera_calibration or "assumed_fov55")
        if intrinsics is not None and not np.allclose(intrinsics[:4], camera["mtx"][0] * 1000.0):
            raise ValueError("Keypoint camera intrinsics differ from fitting calibration; regenerate keypoints or use matching calibration")
    return PreparedVideo(job=job, stem=stem, session_dir=session_dir, paths=paths, timestamps=timestamps,
                         keypoints2d=keypoints2d, keypoints3d=keypoints3d, confs=confs, camera=camera,
                         calibration_source=calibration_source, keypoint_calibration_source=keypoint_source,
                         timestamp_source=timestamp_source)


class FitEngine:
    """Load once, fit many videos. ``process_many`` batches compatible videos.

    Videos are grouped by (encoding length, padded frame count); each group is
    fitted together through one compiled program per mode.
    """

    def __init__(self, options: Optional[dict | FitOptions] = None):
        self.options = options if isinstance(options, FitOptions) else FitOptions.from_dict(options)
        self.loaded = False
        self._trees: dict = {}
        self._templates: dict = {}
        self._chunks: dict = {}
        self._optimizer = None
        self._learning_rate = None

    # ------------------------------------------------------------------ setup
    def load(self) -> "FitEngine":
        """Configure JAX, build the skeleton models of the requested modes and the optimizer."""
        if self.loaded:
            return self
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        configure_jax(self.options.jax_cache_dir)
        import jax

        from baseline.monocular_demos.biomechanics_mjx import monocular_trajectory as rigid_mt
        from baseline.monocular_demos.utils import joint_names as rigid_joint_names
        from metrics.bone_length.monocular_demos.biomechanics_mjx import monocular_trajectory as variable_mt
        from metrics.bone_length.monocular_demos.utils import joint_names as variable_joint_names

        self.device = jax.devices()[0]
        for mode, module, names, xml in (("rigid", rigid_mt, rigid_joint_names, RIGID_XML),
                                         ("variable", variable_mt, variable_joint_names, VARIABLE_XML)):
            if mode in self.options.modes:
                # shared with get_model(): one ForwardKinetics / jitted forward per process
                fk = module.cached_forward_kinetics(str(xml), names)
                self._trees[mode] = {"module": module, "fk": fk, "scale_mixer": fk.build_default_scale_mixer()}
        self._optimizer, self._learning_rate = rigid_mt.build_optimizer(
            self.options.max_iters, lr_init_value=LR_INIT, lr_end_value=LR_END, clip_by_global_norm=CLIP_NORM)
        self.loaded = True
        return self

    # --------------------------------------------------------------- building
    def initial_model(self, mode: str, max_time: float):
        """The model ``get_model()`` of the original scripts builds for a one-clip dataset with this ``max_time``.

        get_model constructs ``KineticsWrapper(ForwardKinetics(xml, site_reorder=joint_names),
        scale_mixer=default, key=PRNGKey(0), num_trajectories=1, max_time=dataset.max_time,
        action=False, enforce_joint_limits=...)``; the ForwardKinetics is reused here.
        Returns the model expressed in the cached template tree structure.
        """
        import jax
        import jax.numpy as jnp

        tree = self._trees[mode]
        module = tree["module"]
        model = module.KineticsWrapper(
            tree["fk"], scale_mixer=tree["scale_mixer"], key=jax.random.PRNGKey(0), num_trajectories=1,
            max_time=jnp.array(max_time, dtype=jnp.float32), action=False,
            enforce_joint_limits=not self.options.unconstrained_joints,
            **(self.options.trajectory_kwargs or {}),
        )
        key = (mode, int(model.trajectory.encoding_length))
        if key not in self._templates:
            self._templates[key] = model
        template = self._templates[key]
        leaves = jax.tree_util.tree_leaves(model)
        treedef = jax.tree_util.tree_structure(template)
        if len(leaves) != treedef.num_leaves:
            raise RuntimeError("model structure differs from the cached template")
        return jax.tree_util.tree_unflatten(treedef, leaves)

    def encoding_length_for(self, video: PreparedVideo) -> int:
        """Encoding length KineticsWrapper derives for this clip (without building a model)."""
        import numpy as np
        from baseline.monocular_demos.implicit_representation import calculate_encoding_length

        span = np.float32(np.float32(video.timestamps[-1]) + np.float32(1.0))  # dataset.max_time + 1 s (float32)
        return int(calculate_encoding_length(span, min_freq=80))

    def padded_data(self, video: PreparedVideo, padded: int) -> dict:
        """Device-ready arrays in MonocularDataset dtypes, padded to ``padded`` frames."""
        import numpy as np
        import jax.numpy as jnp

        frames = video.frames
        if padded < frames:
            raise ValueError("padded length shorter than the clip")
        extra = padded - frames
        take_last = lambda a: np.concatenate([a, np.repeat(a[-1:], extra, axis=0)], axis=0)
        confs = np.concatenate([video.confs, np.zeros((extra,) + video.confs.shape[1:])], axis=0)
        f64 = jnp.float64
        return {
            "timestamps": jnp.asarray(take_last(video.timestamps), dtype=f64),
            "kp2d": jnp.asarray(take_last(video.keypoints2d), dtype=f64)[None],
            "kp3d": jnp.asarray(take_last(video.keypoints3d), dtype=f64)[None],
            "conf": jnp.asarray(confs, dtype=f64)[None],
            "length": jnp.asarray(frames, dtype=int),
            "camera": {k: jnp.asarray(np.asarray(v, dtype=float), dtype=f64) for k, v in video.camera.items()},
        }

    def _loss_grad(self, mode: str):
        """``(loss, aux), grads`` function of the original fitting loss of ``mode``."""
        import equinox as eqx

        module = self._trees[mode]["module"]
        settings = dict(LOSS_SETTINGS)
        if mode == "variable":
            settings["scale_smoothness_regularization"] = VARIABLE_SCALE_SMOOTHNESS
        value_and_grad = eqx.filter_value_and_grad(functools.partial(module.loss, **settings), has_aux=True)
        return lambda model, x, y, data: value_and_grad(model, x=x, y=y, camera_calibration=data["camera"])

    def chunk_function(self, mode: str, encoding_length: int, length: int):
        """jit(vmap(chunk)) for one (mode, encoding length, steps per call); shapes key the jit cache."""
        import jax

        key = (mode, encoding_length, length)
        if key not in self._chunks:
            self._chunks[key] = jax.jit(self.batched_chunk(mode, encoding_length, length), donate_argnums=(0, 1))
        return self._chunks[key]

    def batched_chunk(self, mode: str, encoding_length: int, length: int):
        """Un-jitted ``vmap`` of the training chunk over a stack of independent models."""
        import equinox as eqx
        import jax

        module = self._trees[mode]["module"]
        _, static = eqx.partition(self._templates[(mode, encoding_length)], eqx.is_array)
        sample = functools.partial(module.padded_cyclic_sample, with_valid=(mode == "variable"))
        chunk = module.make_train_chunk(self._loss_grad(mode), self._optimizer, sample, length, static)
        return jax.vmap(chunk, in_axes=(0, 0, 0, None))

    # --------------------------------------------------------------- fitting
    def fit_batch(self, mode: str, videos: list, padded: int, batch_size: Optional[int] = None) -> dict:
        """Fit ``videos`` (same encoding length) jointly; returns trained models and timings.

        With ``pad_batches`` the stack is filled up to ``batch_size`` (default:
        ``options.batch``) with copies of the first video so the compiled program is reused.
        """
        import equinox as eqx
        import jax
        import jax.numpy as jnp
        import numpy as np
        from baseline.monocular_demos.implicit_representation import encoding_top_hz

        module = self._trees[mode]["module"]
        timings = {}
        started = time.perf_counter()
        models = [self.initial_model(mode, float(np.float32(v.timestamps[-1]))) for v in videos]
        lengths = {int(m.trajectory.encoding_length) for m in models}
        if len(lengths) != 1:
            raise ValueError("a batch must share one encoding length")
        encoding_length = lengths.pop()
        real = len(videos)
        batch_size = self.options.batch if batch_size is None else batch_size
        if self.options.pad_batches and real < batch_size:
            models = models + [models[0]] * (batch_size - real)
            videos = list(videos) + [videos[0]] * (batch_size - real)
        stack = lambda trees: jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *trees)
        params = stack([eqx.filter(m, eqx.is_array) for m in models])
        opt_state = stack([self._optimizer.init(eqx.filter(m, eqx.is_inexact_array)) for m in models])
        data = stack([self.padded_data(v, padded) for v in videos])
        params, opt_state, data = jax.device_put((params, opt_state, data), self.device)
        max_time_init = np.asarray(params.trajectory.max_time).reshape(len(videos), -1)[:, 0]
        timings["build_seconds"] = time.perf_counter() - started

        losses, last_metrics, compile_seconds = [], None, 0.0
        train_started = time.perf_counter()
        for start, length in module.chunk_schedule(self.options.max_iters, self.options.steps_per_call):
            fn = self.chunk_function(mode, encoding_length, length)
            call_started = time.perf_counter()
            params, opt_state, vals, metrics = fn(params, opt_state, data, jnp.asarray(start))
            if start == 0:
                jax.block_until_ready(vals)
                compile_seconds = time.perf_counter() - call_started
            vals = np.asarray(vals)  # (B, length); one host sync per call
            losses.append(vals)
            last_metrics = metrics
            # The original fit_model stops with an error when the loss is NaN at iteration 1000.
            if start <= 1000 < start + length and np.any(np.isnan(vals[:real, 1000 - start])):
                bad = [videos[b].stem for b in range(real) if np.isnan(vals[b, 1000 - start])]
                raise FloatingPointError(f"Loss is NaN on iteration 1000 for {bad}")
        jax.block_until_ready(params)
        timings["train_seconds"] = time.perf_counter() - train_started
        timings["first_call_seconds"] = compile_seconds
        losses = np.concatenate(losses, axis=1)
        _, static = eqx.partition(self._templates[(mode, encoding_length)], eqx.is_array)
        out = []
        max_time_final = np.asarray(params.trajectory.max_time).reshape(len(videos), -1)[:, 0]
        for b in range(real):
            trained = eqx.combine(jax.tree_util.tree_map(lambda x: x[b], params), static)
            out.append({
                "model": trained,
                "final_loss": float(losses[b, -1]),
                "loss_curve_every_100": losses[b, ::100].tolist(),
                "final_metrics": {k: float(np.asarray(v)[b, -1].mean()) for k, v in last_metrics.items()},
                "max_time_init": float(max_time_init[b]),
                "max_time_final": float(max_time_final[b]),
                "encoding_length": encoding_length,
                "top_hz_init": encoding_top_hz(encoding_length, float(max_time_init[b])),
                "top_hz_final": encoding_top_hz(encoding_length, float(max_time_final[b])),
            })
        timings["batch_size"] = len(videos)
        timings["real_videos"] = real
        timings["padded_frames"] = padded
        return {"results": out, "timings": timings}

    # ---------------------------------------------------------------- export
    def export(self, mode: str, video: PreparedVideo, model, padded: int, fit_info: dict) -> dict:
        """Write the output files of one trained model; returns export details."""
        import jax
        import jax.numpy as jnp
        import numpy as np
        from baseline.native_export import export_kinematic_coordinates
        from hmb.fit_export import export_times, grid_confidence, variable_scale_params

        module = self._trees[mode]["module"]
        started = time.perf_counter()
        # The main files are always written at the observation timestamps (so they align
        # frame-by-frame with the keypoints); with export_grid_hz the rigid fit is additionally
        # evaluated on a uniform grid k / grid_hz (``*_grid`` files) for derivative-based metrics.
        times, conf = video.timestamps, video.confs

        def evaluate(eval_at):
            """Model outputs at ``eval_at`` (padded to a bucket length, so programs are reused)."""
            n = len(eval_at)
            length = bucket_length(n, self.options.buckets)
            padded_times = jnp.asarray(np.concatenate([eval_at, np.repeat(eval_at[-1:], length - n)]), dtype=jnp.float64)
            out = module.export_trajectory(model, padded_times, trajectory_selection=0)
            return jax.tree_util.tree_map(lambda x: np.asarray(x)[:n], out), padded_times

        frames = len(times)
        outputs, eval_times = evaluate(times)
        engine_keys = dict(
            fit_encoding_length=np.array(fit_info["encoding_length"]),
            fit_max_time_init=np.array(fit_info["max_time_init"]), fit_max_time_final=np.array(fit_info["max_time_final"]),
            fit_export_grid_hz=np.array(np.nan if not self.options.export_grid_hz else float(self.options.export_grid_hz)),
            fit_timestamp_source=np.array(video.timestamp_source), fit_padded_frames=np.array(padded),
            fit_steps_per_call=np.array(self.options.steps_per_call), fit_final_loss=np.array(fit_info["final_loss"]),
        )
        common = dict(
            confidence=np.array(conf),
            joint_limits_enforced=np.array(not self.options.unconstrained_joints),
            fitting_iterations=np.array(self.options.max_iters),
            site_offsets=np.asarray(model.site_offsets),
            export_backend=np.array("native"),
            camera_intrinsics_px=np.array(video.camera["mtx"]) * 1000.0,
            camera_distortion=np.array(video.camera["dist"]),
            camera_calibration_source=np.array(video.calibration_source),
            keypoint_camera_calibration_source=np.array(video.keypoint_calibration_source),
        )
        paths = video.paths
        if mode == "rigid":
            qpos, qvel, _, rnc = outputs
            body_scales = 1.0 + np.asarray(model.scale_mixer) @ np.asarray(model.body_scale)
            joints, sites = export_kinematic_coordinates(model.forward.xml_path, qpos, body_scales, model.site_offsets,
                                                         model.forward.site_reorder)
            _atomic_npz(paths["fitted_model"], timestamps=np.array(times), qpos=np.array(qpos), qvel=np.array(qvel),
                        rnc=np.array(rnc), sites=sites, joints=joints, scale=np.array(model.body_scale),
                        body_scale_factors=body_scales, scale_parameterization=np.array("additive"),
                        scale_time_varying=np.array(False), **common, **engine_keys)
            _write_mot(paths["fitted_model"], paths["mot"])
            if self.options.export_grid_hz:
                grid_times = export_times(video.timestamps, self.options.export_grid_hz)
                grid_conf = grid_confidence(video.timestamps, video.confs, grid_times)
                (gqpos, gqvel, _, grnc), _ = evaluate(grid_times)
                gjoints, gsites = export_kinematic_coordinates(model.forward.xml_path, gqpos, body_scales,
                                                               model.site_offsets, model.forward.site_reorder)
                _atomic_npz(paths["fitted_model_grid"], timestamps=np.array(grid_times), qpos=np.array(gqpos),
                            qvel=np.array(gqvel), rnc=np.array(grnc), sites=gsites, joints=gjoints,
                            scale=np.array(model.body_scale), body_scale_factors=body_scales,
                            scale_parameterization=np.array("additive"), scale_time_varying=np.array(False),
                            **{**common, "confidence": np.array(grid_conf)}, **engine_keys)
                _write_mot(paths["fitted_model_grid"], paths["mot_grid"])
        else:
            qpos, qvel, _, rnc, geometry_scale_params = outputs
            scale_params = np.asarray(variable_scale_params(model, eval_times))[:frames]
            joints, sites = export_kinematic_coordinates(
                model.forward.xml_path, qpos, 1.0 + np.asarray(geometry_scale_params) @ np.asarray(model.scale_mixer).T,
                model.site_offsets, model.forward.site_reorder)
            _atomic_npz(paths["bone_fit"], timestamps=np.array(times), qpos=np.array(qpos), qvel=np.array(qvel),
                        rnc=np.array(rnc), sites=sites, joints=joints, scale=scale_params,
                        scale_mixer=np.array(model.scale_mixer),
                        body_scale_factors=1.0 + scale_params @ np.array(model.scale_mixer).T,
                        scale_parameterization=np.array("additive"), scale_time_varying=np.array(True),
                        **common, **engine_keys)
            paths["bone_copy"].parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(paths["bone_fit"], paths["bone_copy"])
            body_scales = 1.0 + scale_params @ np.array(model.scale_mixer).T  # (T, nbody)
        # bodies excluding the world body: rigid (nbody, 1) -> axis 0, variable (T, nbody) -> axis 1
        bodies = np.asarray(body_scales)[1:, 0] if mode == "rigid" else np.asarray(body_scales)[:, 1:]
        details = {"export_seconds": time.perf_counter() - started, "export_frames": frames,
                   "min_body_scale_factor": float(np.min(bodies)),
                   "frac_nonpositive_body_scale": float(np.mean(bodies <= 0))}
        details["keypoint_residual_mm"] = keypoint_residual_mm(sites, video.keypoints3d, video.confs)
        return details

    # ---------------------------------------------------------------- driver
    def process(self, job: dict, force: bool = False) -> dict:
        """Fit and export one job; returns its status."""
        return self.process_many([job], force=force)[0]

    def process_many(self, jobs: list, force: bool = False) -> list:
        """Fit and export all jobs; returns one status dict per job (also written to disk)."""
        self.load()
        statuses = [{"ok": True, "seconds": 0.0, "error": None, "modes": {},
                     "options": self.options.as_dict()} for _ in jobs]
        prepared: list = [None] * len(jobs)
        for index, job in enumerate(jobs):
            started = time.perf_counter()
            try:
                prepared[index] = load_video_inputs(job, self.options.camera_calibration)
                statuses[index].update(stem=prepared[index].stem, frames=prepared[index].frames,
                                       timestamp_source=prepared[index].timestamp_source)
            except Exception as exc:
                statuses[index].update(ok=False, error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
            statuses[index]["seconds"] += time.perf_counter() - started

        for mode in self.options.modes:
            # Skip jobs whose outputs exist; clear partial outputs of the others.
            pending = []
            for index, video in enumerate(prepared):
                if video is None:
                    continue
                if not force and all(path.is_file() for path in mode_outputs(video.paths, mode, bool(self.options.export_grid_hz))):
                    statuses[index]["modes"][mode] = {"ok": True, "skipped": True}
                    continue
                for path in mode_outputs(video.paths, mode, bool(self.options.export_grid_hz)):
                    path.unlink(missing_ok=True)
                if mode == "variable":
                    video.paths["bone_keypoints"].parent.mkdir(parents=True, exist_ok=True)
                    keypoints = Path(video.job.get("keypoints") or video.paths["keypoints"])
                    if keypoints.resolve() != video.paths["bone_keypoints"].resolve():
                        shutil.copy2(keypoints, video.paths["bone_keypoints"])
                pending.append(index)
            # Videos with the same encoding length and padded length share one compiled program.
            groups: dict = {}
            for index in pending:
                video = prepared[index]
                groups.setdefault((self.encoding_length_for(video), bucket_length(video.frames, self.options.buckets)),
                                  []).append(index)
            for (encoding_length, padded), members in sorted(groups.items()):
                size = self.batch_size_for(len(members))
                for offset in range(0, len(members), size):
                    batch = members[offset: offset + size]
                    self._run_batch(mode, batch, prepared, statuses, padded, size)
        for index, status in enumerate(statuses):
            video = prepared[index]
            status["ok"] = status["ok"] and all(entry.get("ok") for entry in status["modes"].values())
            if not status["ok"] and status["error"] is None:
                status["error"] = "; ".join(f"{mode}: {entry.get('error')}" for mode, entry in status["modes"].items()
                                            if not entry.get("ok"))
            status_path = (video.paths["status"] if video is not None else
                           session_paths(Path(jobs[index]["session_dir"]).expanduser().resolve(),
                                         jobs[index].get("stem") or "unknown")["status"])
            try:
                _atomic_json(status_path, status)
            except Exception as exc:  # never fail other jobs because of a status write
                status["status_write_error"] = str(exc)
        return statuses

    def batch_size_for(self, videos: int) -> int:
        """Compiled batch size for a group: ``batch``, or the next power of two for small groups.

        Partial batches are padded with copies (results discarded) so only
        power-of-two batch programs are ever compiled.
        """
        size = 1
        while size < min(videos, self.options.batch):
            size *= 2
        return min(size, self.options.batch)

    def _run_batch(self, mode: str, batch: list, prepared: list, statuses: list, padded: int, size: int) -> None:
        """Fit and export one batch; on failure retry its videos one by one to isolate the bad one."""
        started = time.perf_counter()
        try:
            fitted = self.fit_batch(mode, [prepared[i] for i in batch], padded, size)
        except Exception as exc:
            if len(batch) > 1:  # isolate the failing video(s)
                for index in batch:
                    self._run_batch(mode, [index], prepared, statuses, padded, 1)
                return
            statuses[batch[0]]["modes"][mode] = {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                                                 "traceback": traceback.format_exc()}
            statuses[batch[0]]["seconds"] += time.perf_counter() - started
            return
        share = (time.perf_counter() - started) / len(batch)
        for index, result in zip(batch, fitted["results"]):
            entry = {key: value for key, value in result.items() if key != "model"}
            entry.update(fitted["timings"])
            entry["fit_seconds_share"] = share
            try:
                entry.update(self.export(mode, prepared[index], result["model"], padded, result))
                entry["ok"] = bool(math.isfinite(result["final_loss"]))
                if not entry["ok"]:
                    entry["error"] = "non-finite final loss"
            except Exception as exc:
                entry.update(ok=False, error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
            statuses[index]["modes"][mode] = entry
            statuses[index]["seconds"] += share + entry.get("export_seconds", 0.0)


def _write_mot(fitted_model: Path, mot: Path) -> None:
    """Convert a fitted-model npz to an OpenSim ``.mot`` file (written atomically)."""
    from baseline.metrab2opensim import convert_fitted_model_to_mot

    mot.unlink(missing_ok=True)
    tmp = mot.with_name(mot.name + ".part")
    if not convert_fitted_model_to_mot(str(fitted_model), str(tmp)):
        raise RuntimeError(f"OpenSim conversion failed for {fitted_model}")
    tmp.replace(mot)


def main(argv: Optional[list] = None) -> int:
    """Command line (see the module docstring); prints a summary, returns 1 if any job failed."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--jobs", required=True, help="jobs.json with {'options': {...}, 'jobs': [...]}")
    parser.add_argument("--force", action="store_true", help="refit even if outputs exist")
    args = parser.parse_args(argv)
    watchdog = float(os.environ.get("HMB_FIT_ENGINE_WATCHDOG", "0") or 0)
    if watchdog > 0:  # periodic all-thread stack dumps to stderr, to diagnose stalls
        import faulthandler

        faulthandler.dump_traceback_later(watchdog, repeat=True, file=sys.stderr)
    spec = json.loads(Path(args.jobs).read_text())
    engine = FitEngine(spec.get("options"))
    statuses = engine.process_many(spec.get("jobs", []), force=args.force)
    summary = [{"stem": s.get("stem"), "ok": s["ok"], "seconds": round(s["seconds"], 2), "error": s["error"]} for s in statuses]
    print(json.dumps(summary, indent=1))
    return 0 if all(s["ok"] for s in statuses) else 1


if __name__ == "__main__":
    raise SystemExit(main())
