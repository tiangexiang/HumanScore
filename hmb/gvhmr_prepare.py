"""GVHMR models and stages for the mesh engine, without rendering or training imports.

GVHMR (``GVHMR/``, a pinned copy of the original source tree, left unchanged)
reconstructs the SMPL-X body of a person from a video.  The person tracker
(YOLOv8x + BoT-SORT), ViTPose, HMR2 features, the GVHMR network and its
post-processing are GVHMR's own code; this module only defers imports that
inference does not need (moving-camera odometry, native PyTorch3D ops),
downloads the checkpoints and handles outputs.

* :func:`prepare_assets` downloads the four checkpoints (from the Hugging Face
  mirror used by the GVHMR authors' demo) and links the SMPL-X body model into
  ``GVHMR/inputs/checkpoints/``.
* :class:`GVHMRModels` loads the networks once, so a long-lived process
  (``hmb.mesh_engine``) reconstructs many videos without paying the import and
  checkpoint cost per video, and exposes each stage (detection, tracking, 2-D
  pose, features, GVHMR) on in-memory frames.
* :func:`reconstruct` is the original single-video demo path (GVHMR's video
  readers, frame-by-frame YOLO tracking, batch size 16, fp32).  The mesh engine
  reproduces it bit for bit with its parity settings; the GPU parity test
  compares the two.
"""
from __future__ import annotations

import argparse
import contextlib
import filecmp
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import types
import urllib.request
from time import perf_counter
from typing import Dict, Iterable, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
# Checkpoint path under GVHMR/inputs/checkpoints -> (size in bytes, Google Drive id of the
# authors' original link).  They are downloaded from the Hugging Face mirror that the
# authors' demo (linked in the GVHMR README) uses,
# https://huggingface.co/spaces/LittleFrog/GVHMR/raw/main/app.py, because the original
# Drive links often fail with download-quota errors.
CHECKPOINTS = {
    "gvhmr/gvhmr_siga24_release.ckpt": (163508011, "1c9iCeKFN4Kr6cMPJ9Ss6Jdc3SZFnO5NP"),
    "hmr2/epoch=10-step=25000.ckpt": (2709494041, "1X5hvVqvqI9tvjUCb2oAlZxtgIKD9kvsc"),
    "vitpose/vitpose-h-multi-coco.pth": (2549075546, "1sR8xZD9wrZczdDVo6zKscNLwvarIRhP5"),
    "yolo/yolov8x.pt": (136867539, None),
}
COMPONENTS = ("tracker", "vitpose", "hmr2", "gvhmr")
# "fp32" keeps PyTorch defaults (strict fp32 matmul, as GVHMR itself runs),
# "tf32" allows TF32 tensor-core matmuls, "bf16"/"fp16" also run the two ViT-H
# backbones (ViTPose, HMR2) under autocast.
PRECISIONS = ("fp32", "tf32", "bf16", "fp16")
UPSTREAM_BATCH = 16  # GVHMR's batch size for ViTPose and HMR2
# GVHMR's tracker settings (hmr4d/utils/preproc/tracker.py): persons only, confidence >= 0.5.
YOLO_ARGS = {"conf": 0.5, "classes": 0, "verbose": False}


def prepare_assets(gvhmr_root: Path, asset_dir: Path, smplx_dir: Path) -> None:
    """Make the checkpoints and the SMPL-X model available under ``gvhmr_root/inputs/checkpoints``.

    Missing checkpoints are downloaded into ``asset_dir`` (checked by size) and
    symlinked into the GVHMR tree.  The SMPL-X model needs a license and is
    never downloaded: ``smplx_dir/SMPLX_NEUTRAL.npz`` must exist (or already be linked).
    """
    for relative, (expected_size, _) in CHECKPOINTS.items():
        target = gvhmr_root / "inputs/checkpoints" / relative
        if target.is_file() and target.stat().st_size == expected_size:
            continue
        asset = asset_dir / relative
        if not asset.is_file() or asset.stat().st_size != expected_size:
            asset.parent.mkdir(parents=True, exist_ok=True)
            url = "https://huggingface.co/camenduru/GVHMR/resolve/main/" + relative.replace("=", "%3D") + "?download=true"
            temporary = asset.with_suffix(asset.suffix + ".part")
            print(f"[GVHMR] Downloading author-endorsed checkpoint: {relative}", flush=True)
            try:
                with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as output:
                    shutil.copyfileobj(response, output, length=4 * 1024 * 1024)
                if temporary.stat().st_size != expected_size:
                    raise ValueError(f"Incorrect checkpoint size: {relative}")
                temporary.replace(asset)
            finally:
                temporary.unlink(missing_ok=True)
        if target.absolute() != asset.absolute():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.unlink(missing_ok=True)
            target.symlink_to(asset.resolve())
    source = smplx_dir / "SMPLX_NEUTRAL.npz"
    target = gvhmr_root / "inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz"
    if source.is_file() and source.resolve() != target.resolve():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.unlink(missing_ok=True)
        target.symlink_to(source.resolve())
    elif not target.is_file():
        raise FileNotFoundError(f"Licensed local body model is required: {source}")


def prepare_video_input(video: Path, output_root: Path, static_cam: bool, f_mm: float | None):
    """Copy ``video`` to ``<output_root>/<stem>/0_input_video.mp4`` for :func:`reconstruct`.

    The fingerprint (resolved source path, size, modification time and the camera
    options) is checked against an existing ``reconstruction.json``, so cached
    preprocessing of another input is never reused.  Returns
    ``(session, prepared_video, fingerprint)``.
    """
    video = video.expanduser().absolute()
    session = output_root / video.stem
    session.mkdir(parents=True, exist_ok=True)
    preprocess = session / "preprocess"
    preprocess.mkdir(exist_ok=True)
    prepared_video = session / "0_input_video.mp4"
    fingerprint = {"source": str(video.resolve()), "size": video.stat().st_size,
                   "mtime_ns": video.stat().st_mtime_ns, "static_cam": bool(static_cam), "f_mm": f_mm}
    provenance = session / "reconstruction.json"
    if provenance.exists() and json.loads(provenance.read_text())["input"] != fingerprint:
        raise ValueError("GVHMR cache input/options changed; use a fresh output directory")
    if not provenance.exists() and any(preprocess.iterdir()):
        raise ValueError("Unverified GVHMR preprocess cache found; use a fresh output directory")
    if prepared_video.is_file() and not filecmp.cmp(video, prepared_video, shallow=False):
        if provenance.exists():
            raise ValueError("Prepared GVHMR video changed after preprocessing; use a fresh output directory")
        shutil.copy2(video, prepared_video)
    elif not prepared_video.exists():
        shutil.copy2(video, prepared_video)
    return session, prepared_video, fingerprint


class _DeferredKNN:
    """Stands in for ``pytorch3d.ops.knn``: imports it only if a KNN function is really used."""

    def __getattr__(self, name):
        try:
            module = importlib.import_module("pytorch3d.ops.knn")
        except ImportError as exc:
            raise RuntimeError("This optional geometry operation requires native PyTorch3D; GVHMR inference uses only its Python rotation transforms") from exc
        return getattr(module, name)


def _defer_optional_imports(gvhmr_root: Path) -> None:
    """Let GVHMR's inference modules import without its optional heavy dependencies."""
    # GVHMR's preproc/__init__.py imports moving-camera VO and optional SLAM.
    # Load its submodules as a namespace so static mode never imports them.
    package_name = "hmr4d.utils.preproc"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(gvhmr_root / "hmr4d/utils/preproc")]
        package.__package__ = package_name
        sys.modules[package_name] = package
    # geo_transform eagerly imports native KNN for a helper unused by inference.
    # Defer that one import while executing the unchanged upstream functions.
    # Official PyTorch3D's PYTORCH3D_NO_EXTENSION=1 distribution supplies all
    # required rotation transforms; no replacement mathematical code is used.
    module_name = "hmr4d.utils.geo_transform"
    if module_name not in sys.modules:
        path = gvhmr_root / "hmr4d/utils/geo_transform.py"
        source = path.read_text()
        eager = "import pytorch3d.ops.knn as knn"
        if eager not in source:
            raise RuntimeError("GVHMR geometry import changed; review the pinned inference adapter")
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        module.__dict__["_DeferredKNN"] = _DeferredKNN
        sys.modules[module_name] = module
        exec(compile(source.replace(eager, "knn = _DeferredKNN()", 1), str(path), "exec"), module.__dict__)


@contextlib.contextmanager
def _trusted_legacy_checkpoints(checkpoint_dir: Path):
    """Allow GVHMR's pickle-based ``torch.load`` only for the known checkpoint files."""
    import torch
    original = torch.load
    trusted = {(checkpoint_dir / name).resolve() for name in CHECKPOINTS}

    def load(path, *args, **kwargs):
        if isinstance(path, (str, Path)) and Path(path).resolve() in trusted:
            kwargs.setdefault("weights_only", False)
            # These pinned checkpoints use PyTorch's ZIP serialization.
            # Mapping avoids eagerly duplicating multi-GB tensor storage on
            # the host before upstream load_state_dict copies the used tensors.
            kwargs.setdefault("mmap", True)
        return original(path, *args, **kwargs)

    torch.load = load
    try:
        yield
    finally:
        torch.load = original


@contextlib.contextmanager
def _working_directory(path: Path):
    """Run inside the GVHMR root: GVHMR resolves several checkpoint paths relative to it."""
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


@contextlib.contextmanager
def matmul_precision(precision: str):
    """Scope TF32 matmuls to one stage and restore the process defaults."""
    import torch
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = precision != "fp32"
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def _autocast(precision: str):
    """Autocast context for bf16/fp16, else a no-op."""
    import torch
    if precision in ("bf16", "fp16"):
        return torch.autocast("cuda", dtype=torch.bfloat16 if precision == "bf16" else torch.float16)
    return contextlib.nullcontext()


def prepare_shared_crops(video_path, boxes, pose_missing, features_missing, get_batch):
    """Decode and normalize identical ViTPose/HMR2 inputs once per video.

    Both upstream extractors otherwise call the same deterministic get_batch
    implementation independently, including video decode, Gaussian filtering,
    crop/resize and normalization. Their tensor APIs preserve those exact inputs.
    """
    if pose_missing or features_missing:
        return get_batch(str(video_path), boxes)
    return None, boxes


def one_track_from_history(track_history, num_frames: int, width: int, height: int):
    """GVHMR's ``Tracker.get_one_track`` post-processing for an in-memory track history.

    Selects the track with the largest summed normalized box area, fills missing
    frames by linear interpolation (edge frames copy the nearest box), and
    applies GVHMR's double 5-frame moving average.  Returns
    ``(bbx_xyxy (F, 4) float64, tracked_mask (F,) bool)``.
    """
    import numpy as np
    import torch
    from collections import defaultdict
    from hmr4d.utils.net_utils import moving_average_smooth
    from hmr4d.utils.seq_utils import (frame_id_to_mask, get_frame_id_list_from_mask,
                                       linear_interpolate_frame_ids, rearrange_by_mask)

    id_to_frame_ids, id_to_boxes = defaultdict(list), defaultdict(list)
    for frame_id, frame in enumerate(track_history):
        for det in frame:
            id_to_frame_ids[det["id"]].append(frame_id)
            id_to_boxes[det["id"]].append(det["bbx_xyxy"])
    if not id_to_frame_ids:
        raise ValueError("YOLO tracking found no person track in the video")
    area = {}
    for key, boxes in id_to_boxes.items():
        boxes = np.array(boxes)
        wh = boxes[:, 2:] - boxes[:, :2]
        area[key] = (wh[:, 0] * wh[:, 1] / width / height).sum()
    track_id = sorted(area.items(), key=lambda item: item[1], reverse=True)[0][0]
    frame_ids = torch.tensor(id_to_frame_ids[track_id])
    boxes = torch.tensor(np.array(id_to_boxes[track_id]))
    mask = frame_id_to_mask(frame_ids, num_frames)
    boxes = rearrange_by_mask(boxes, mask)
    boxes = linear_interpolate_frame_ids(boxes, get_frame_id_list_from_mask(~mask))
    if not (boxes.sum(1) != 0).all():
        raise ValueError("Person track could not be interpolated over all frames")
    boxes = moving_average_smooth(boxes, window_size=5, dim=0)
    boxes = moving_average_smooth(boxes, window_size=5, dim=0)
    return boxes, mask


def select_track_by_prior(track_history, prior_xyxy):
    """Id of the track that overlaps a known subject box track most, or None when none overlaps it.

    ``prior_xyxy`` (F, 4) is the subject's box in every frame of ``track_history``
    (NaN rows: unknown), e.g. from a multi-person tracker.  Each track scores the
    sum over frames of its box IoU with the prior box; the highest sum wins (ties:
    the lower id).  Replaces GVHMR's "largest summed area" rule when the
    subject is known, so a larger bystander (or another person's masked
    silhouette) is never reconstructed instead.
    """
    import numpy as np

    prior = np.asarray(prior_xyxy, dtype=float).reshape(-1, 4)
    score = {}
    for frame_id, frame in enumerate(track_history):
        if frame_id >= len(prior) or not np.all(np.isfinite(prior[frame_id])):
            continue
        p = prior[frame_id]
        p_area = max(0.0, p[2] - p[0]) * max(0.0, p[3] - p[1])
        for det in frame:
            b = np.asarray(det["bbx_xyxy"], dtype=float)
            inter = (max(0.0, min(b[2], p[2]) - max(b[0], p[0]))
                     * max(0.0, min(b[3], p[3]) - max(b[1], p[1])))
            union = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]) + p_area - inter
            if inter > 0 and union > 0:
                score[det["id"]] = score.get(det["id"], 0.0) + inter / union
    if not score:
        return None
    return min(score, key=lambda key: (-score[key], key))


# ultralytics GMC(method="sparseOptFlow", downscale=2) parameters (trackers/utils/gmc.py)
GMC_DOWNSCALE = 2
GMC_FEATURES = dict(maxCorners=1000, qualityLevel=0.01, minDistance=1, blockSize=3, useHarrisDetector=False, k=0.04)


class _PrecomputedGMC:
    """Stands in for ultralytics' GMC, returning warps computed ahead of time."""

    def __init__(self, warps):
        self.warps = list(warps)
        self.used = 0

    def apply(self, raw_frame=None, detections=None):
        """The next precomputed warp (BoT-SORT calls this once per update)."""
        warp = self.warps[self.used]
        self.used += 1
        return warp.copy()

    def reset_params(self):  # BOTSORT.reset()
        self.used = 0


def _gmc_frame(frame_bgr):
    """Downscaled grey image and corner features of one frame, as ultralytics' GMC computes them."""
    import cv2
    height, width = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    if GMC_DOWNSCALE > 1:
        gray = cv2.resize(gray, (width // GMC_DOWNSCALE, height // GMC_DOWNSCALE))
    return gray, cv2.goodFeaturesToTrack(gray, mask=None, **GMC_FEATURES)


def _gmc_pair(prev, current):
    """``GMC.applySparseOptFlow`` for one (previous, current) call pair."""
    import cv2
    import numpy as np
    prev_gray, prev_keypoints = prev
    gray = current[0]
    H = np.eye(2, 3)
    if prev_keypoints is None:  # upstream re-initializes on this call
        return H
    matched, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_keypoints, None)
    good = np.asarray(status).reshape(-1).astype(bool)
    prev_points, curr_points = prev_keypoints[good], matched[good]
    if prev_points.shape[0] > 4:
        H, _ = cv2.estimateAffinePartial2D(prev_points, curr_points, cv2.RANSAC)
        if GMC_DOWNSCALE > 1:
            H[0, 2] *= GMC_DOWNSCALE
            H[1, 2] *= GMC_DOWNSCALE
    return H


def sparse_optflow_warps(frames_bgr: Sequence, sequence: Sequence[int], workers: int = 4,
                         cache: Optional[dict] = None) -> list:
    """GMC warps of BoT-SORT's update calls on ``frames_bgr[sequence]``.

    Each warp depends only on the two images of its call pair, so they are
    computed independently (OpenCV releases the GIL) and are identical to
    ultralytics' sequential computation.  ``cache`` maps frame index ->
    ``(gray, keypoints)`` (e.g. precomputed while decoding).
    """
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor
    sequence = list(sequence)
    if not sequence:
        return []
    cache = {} if cache is None else cache
    missing = [i for i in sequence if i not in cache]
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        for index, value in zip(missing, pool.map(lambda i: _gmc_frame(frames_bgr[i]), missing)):
            cache[index] = value
        pairs = list(zip(sequence[:-1], sequence[1:]))
        warps = list(pool.map(lambda pair: _gmc_pair(cache[pair[0]], cache[pair[1]]), pairs))
    return [np.eye(2, 3)] + warps


class GVHMRModels:
    """The four GVHMR networks, loaded once and reused across videos.

    ``gvhmr_root`` defaults to ``GVHMR/``, ``checkpoints_dir`` (download folder)
    to ``outputs/assets/gvhmr/checkpoints/`` and ``smplx_model_dir`` to
    ``outputs/assets/smplx/``.  Each stage method loads the component it needs.
    """

    def __init__(self, gvhmr_root: Optional[Path] = None, checkpoints_dir: Optional[Path] = None,
                 smplx_model_dir: Optional[Path] = None):
        self.root = Path(gvhmr_root or REPO_ROOT / "GVHMR").expanduser().resolve()
        self.checkpoints_dir = Path(checkpoints_dir or REPO_ROOT / "outputs/assets/gvhmr/checkpoints").expanduser().resolve()
        self.smplx_dir = Path(smplx_model_dir or REPO_ROOT / "outputs/assets/smplx").expanduser().resolve()
        self.tracker = self.vitpose = self.hmr2 = self.gvhmr = None
        self.cfg = None
        self.load_seconds: Dict[str, float] = {}
        self._ready = False

    # ------------------------------------------------------------------ setup
    def setup(self) -> "GVHMRModels":
        """Check the GPU and source tree, import GVHMR, prepare the assets and compose its config."""
        if self._ready:
            return self
        import torch
        started = perf_counter()
        if not (self.root / "hmr4d/configs/demo.yaml").is_file():
            raise FileNotFoundError(f"Restore the pinned GVHMR source: {self.root}")
        if not torch.cuda.is_available():
            raise RuntimeError("GVHMR reconstruction requires a CUDA GPU")
        if str(self.root) not in sys.path:
            sys.path.insert(0, str(self.root))
        os.environ.setdefault("YOLO_AUTOINSTALL", "false")
        _defer_optional_imports(self.root)
        with _working_directory(self.root):
            from hydra import compose, initialize_config_module
            import hmr4d.utils.preproc.tracker  # noqa: F401  (ultralytics)
            import hmr4d.utils.preproc.vitpose  # noqa: F401
            import hmr4d.utils.preproc.vitfeat_extractor  # noqa: F401
            # Register only model groups, avoiding GVHMR's training registry.
            import hmr4d.model.gvhmr.gvhmr_pl_demo  # noqa: F401
            import hmr4d.model.gvhmr.utils.endecoder  # noqa: F401
            import hmr4d.network.gvhmr.relative_transformer  # noqa: F401
            import hmr4d.model.gvhmr.pipeline.gvhmr_pipeline  # noqa: F401
            prepare_assets(self.root, self.checkpoints_dir, self.smplx_dir)
            with initialize_config_module(version_base="1.3", config_module="hmr4d.configs"):
                self.cfg = compose(config_name="demo", overrides=["video_name=engine", "static_cam=True"])
        self._ready = True
        self.load_seconds["setup"] = perf_counter() - started
        return self

    def compose_cfg(self, video_name: str, output_root: Path, static_cam: bool):
        """GVHMR's demo config with per-video output paths (as ``tools/demo/demo.py`` lays them out)."""
        from hydra import compose, initialize_config_module
        self.setup()
        with _working_directory(self.root), initialize_config_module(version_base="1.3", config_module="hmr4d.configs"):
            return compose(config_name="demo", overrides=[f"video_name={json.dumps(video_name)}",
                                                         f"output_root={json.dumps(str(output_root))}",
                                                         f"static_cam={static_cam}"])

    def load(self, components: Iterable[str] = COMPONENTS) -> Dict[str, float]:
        """Load the given components (already loaded ones are kept); returns load seconds."""
        import torch
        self.setup()
        components = tuple(components)
        unknown = set(components) - set(COMPONENTS)
        if unknown:
            raise ValueError(f"Unknown GVHMR components: {sorted(unknown)}")
        with torch.no_grad(), _working_directory(self.root), _trusted_legacy_checkpoints(self.root / "inputs/checkpoints"):
            for name in components:
                if getattr(self, name) is not None:
                    continue
                started = perf_counter()
                if name == "tracker":
                    from hmr4d.utils.preproc.tracker import Tracker
                    self.tracker = Tracker()
                elif name == "vitpose":
                    from hmr4d.utils.preproc.vitpose import VitPoseExtractor
                    self.vitpose = VitPoseExtractor(tqdm_leave=False)
                elif name == "hmr2":
                    from hmr4d.utils.preproc.vitfeat_extractor import Extractor
                    self.hmr2 = Extractor(tqdm_leave=False)
                else:
                    from hydra.utils import instantiate
                    model = instantiate(self.cfg.model, _recursive_=False)
                    model.load_pretrained_model(self.cfg.ckpt_path)
                    self.gvhmr = model.eval().cuda()
                torch.cuda.synchronize()
                self.load_seconds[name] = perf_counter() - started
        return dict(self.load_seconds)

    def release(self, *components: str) -> None:
        """Free the given components (all when none are given) and the CUDA cache."""
        import torch
        for name in components or COMPONENTS:
            setattr(self, name, None)
        torch.cuda.empty_cache()

    # ----------------------------------------------------------------- stages
    def track_video(self, video_path: Path):
        """GVHMR's frame-by-frame YOLO tracking of a video file (float32 boxes)."""
        self.load(("tracker",))
        return self.tracker.get_one_track(str(video_path)).float()

    def detect_frames(self, frames_bgr: Sequence, batch_size: int = UPSTREAM_BATCH) -> list:
        """Batched YOLOv8x person detection (GVHMR's thresholds); one numpy ``Boxes`` per frame."""
        self.load(("tracker",))
        yolo = self.tracker.yolo
        batch_size = max(1, int(batch_size))
        detections = []
        for start in range(0, len(frames_bgr), batch_size):
            chunk = [frames_bgr[i] for i in range(start, min(len(frames_bgr), start + batch_size))]
            detections += [result.boxes.cpu().numpy()
                           for result in yolo.predict(chunk, device="cuda", stream=False, **YOLO_ARGS)]
        return detections

    @staticmethod
    def track_detections(detections: list, frames_bgr: Sequence, warps: Optional[list] = None,
                         prior_xyxy=None):
        """BoT-SORT over per-frame detections, as ultralytics does for a video stream.

        ``warps`` optionally supplies the GMC affine of every ``tracker.update``
        call (frames with >= 1 detection, in order), e.g. from
        ``sparse_optflow_warps``; otherwise GMC runs inline.  ``prior_xyxy``
        (F, 4; NaN rows = unknown) is the subject's known box per frame: the
        track that overlaps it most is used (:func:`select_track_by_prior`)
        instead of the largest one.  Returns ``(bbx_xyxy (F, 4) float32,
        tracked_mask (F,) bool)``.
        """
        import torch
        from ultralytics.trackers.track import TRACKER_MAP
        from ultralytics.utils import IterableSimpleNamespace, ops, yaml_load
        from ultralytics.utils.checks import check_yaml

        cfg = IterableSimpleNamespace(**yaml_load(check_yaml("botsort.yaml")))
        tracker = TRACKER_MAP[cfg.tracker_type](args=cfg, frame_rate=30)  # as GVHMR: a new tracker per video
        if warps is not None:
            if cfg.gmc_method != "sparseOptFlow":
                raise RuntimeError("Precomputed GMC warps assume ultralytics' sparseOptFlow method")
            tracker.gmc = _PrecomputedGMC(warps)
        history = []
        height, width = frames_bgr[0].shape[:2]
        for det, image in zip(detections, frames_bgr):
            tracks = tracker.update(det, image) if len(det) else ()
            if len(tracks) == 0:
                history.append([])
                continue
            # ultralytics: results.update(boxes=tracks[:, :-1]) -> [xyxy, id, conf, cls], clipped to the image
            data = ops.clip_boxes(torch.as_tensor(tracks[:, :-1]), (height, width))
            ids = data[:, 4].int().tolist()
            xyxy = data[:, :4].numpy()
            history.append([{"id": ids[i], "bbx_xyxy": xyxy[i]} for i in range(len(ids))])
        if warps is not None and tracker.gmc.used != len(warps):
            raise RuntimeError("GMC warp sequence does not match the tracker updates")
        if prior_xyxy is not None:
            chosen = select_track_by_prior(history, prior_xyxy)
            if chosen is None:
                raise ValueError("No person track of the mesh detector overlaps the given person boxes")
            history = [[det for det in frame if det["id"] == chosen] for frame in history]
        boxes, mask = one_track_from_history(history, len(frames_bgr), width, height)
        return boxes.float(), mask

    def extract_pose(self, crops, crop_boxes, batch_size: int = UPSTREAM_BATCH, precision: str = "fp32"):
        """``VitPoseExtractor.extract`` on precomputed crops with a tunable batch size."""
        import numpy as np
        import torch
        from hmr4d.utils.geo.flip_utils import flip_heatmap_coco17
        from hmr4d.utils.kpts.kp2d_utils import keypoints_from_heatmaps

        self.load(("vitpose",))
        pose = self.vitpose.pose
        outputs = []
        with torch.no_grad(), matmul_precision(precision):
            for start in range(0, crops.shape[0], batch_size):
                batch = crops[start:start + batch_size, :, :, 32:224].cuda(non_blocking=True)
                with _autocast(precision):
                    heatmap, heatmap_flipped = pose(torch.cat([batch, batch.flip(3)], dim=0)).chunk(2)
                heatmap, heatmap_flipped = heatmap.float(), heatmap_flipped.float()
                heatmap = (heatmap + flip_heatmap_coco17(heatmap_flipped)) * 0.5
                boxes = crop_boxes[start:start + batch_size]
                center = boxes[:, :2].numpy()
                scale = (torch.cat((boxes[:, [2]] * 24 / 32, boxes[:, [2]]), dim=1) / 200).numpy()
                preds, maxvals = keypoints_from_heatmaps(heatmaps=heatmap.cpu().numpy(), center=center,
                                                         scale=scale, use_udp=True)
                outputs.append(torch.from_numpy(np.concatenate((preds, maxvals), axis=-1)))
        return torch.cat(outputs, dim=0).clone()

    def extract_features(self, crops, batch_size: int = UPSTREAM_BATCH, precision: str = "fp32"):
        """``Extractor.extract_video_features`` on precomputed crops (F, 1024)."""
        import torch
        self.load(("hmr2",))
        features = []
        with torch.no_grad(), matmul_precision(precision):
            for start in range(0, crops.shape[0], batch_size):
                batch = crops[start:start + batch_size].cuda(non_blocking=True)
                with _autocast(precision):
                    feature = self.hmr2.extractor({"img": batch})
                features.append(feature.float().cpu())
        return torch.cat(features, dim=0).clone()

    def predict(self, bbx_xys, kp2d, features, K_fullimg, camera_rotation, static_cam: bool,
                precision: str = "fp32") -> dict:
        """GVHMR network + its post-processing; returns the ``hmr4d_results`` dict on the CPU."""
        import torch
        from hmr4d.utils.geo_transform import compute_cam_angvel
        from hmr4d.utils.net_utils import detach_to_cpu

        self.load(("gvhmr",))
        length = int(bbx_xys.shape[0])
        data = {"length": torch.tensor(length), "bbx_xys": bbx_xys, "kp2d": kp2d, "K_fullimg": K_fullimg,
                "cam_angvel": compute_cam_angvel(camera_rotation), "f_imgseq": features}
        with torch.no_grad(), matmul_precision(precision):
            prediction = detach_to_cpu(self.gvhmr.predict(data, static_cam=static_cam))
        for name in ("smpl_params_incam", "smpl_params_global"):
            for key, value in prediction[name].items():
                if value.shape[0] != length or not torch.isfinite(value).all():
                    raise ValueError(f"Invalid reconstructed {name}/{key}")
        return prediction


def reconstruct(args: argparse.Namespace, models: Optional[GVHMRModels] = None) -> Path:
    """Reconstruct one video exactly as GVHMR's demo does; returns the ``hmr4d_results.pt`` path.

    ``args`` needs ``video``, ``output_root``, ``static_cam`` and ``f_mm``
    (focal length in mm or None); ``gvhmr_root``, ``checkpoints_dir`` and
    ``smplx_model_dir`` are read only when ``models`` is not given.  Outputs go
    to ``<output_root>/<video stem>/`` in the demo's layout; each preprocessing
    result that already exists there is reused.
    """
    import torch

    video = Path(args.video).expanduser().absolute()
    output_root = Path(args.output_root).expanduser().resolve()
    if not video.is_file():
        raise FileNotFoundError(video)
    if args.f_mm is not None and (not math.isfinite(args.f_mm) or args.f_mm <= 0):
        raise ValueError("Focal length must be finite and positive")
    started = perf_counter()
    models = models or GVHMRModels(args.gvhmr_root, args.checkpoints_dir, args.smplx_model_dir)
    models.setup()
    from hmr4d.utils.video_io_utils import get_video_lwh
    from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy, estimate_K, create_camera_sensor
    if not args.static_cam:
        try:
            from hmr4d.utils.preproc.relpose.simple_vo import SimpleVO
        except ImportError as exc:
            raise RuntimeError("Moving-camera reconstruction requires GVHMR's SimpleVO dependencies, including "
                               "pycolmap; use static_cam=True only for a known static camera") from exc

    session, prepared_video, fingerprint = prepare_video_input(video, output_root, args.static_cam, args.f_mm)
    provenance = session / "reconstruction.json"
    length, width, height = get_video_lwh(str(prepared_video))
    if length < 2:
        raise ValueError("GVHMR needs at least two decoded frames")
    provenance.write_text(json.dumps({"input": fingerprint, "frames": length, "width": width, "height": height,
        "checkpoint_source": "Author-linked LittleFrog/GVHMR demo checkpoint mirrors", "renders": False}, indent=2))
    cfg = models.compose_cfg(video.stem, output_root, args.static_cam)
    result_path = Path(cfg.paths.hmr4d_results)
    stages: Dict[str, float] = {}

    def timed(name, function, *a, **k):
        t0 = perf_counter()
        value = function(*a, **k)
        torch.cuda.synchronize()
        stages[name] = stages.get(name, 0.0) + perf_counter() - t0
        return value

    with torch.no_grad():
        bbx_path, pose_path, features_path = (Path(cfg.paths.bbx), Path(cfg.paths.vitpose), Path(cfg.paths.vit_features))
        if not bbx_path.is_file():
            boxes = timed("track", models.track_video, prepared_video)
            torch.save({"bbx_xyxy": boxes, "bbx_xys": get_bbx_xys_from_xyxy(boxes, base_enlarge=1.2).float()}, bbx_path)
            models.release("tracker")
        boxes = torch.load(bbx_path)["bbx_xys"]
        pose_missing, features_missing = not pose_path.is_file(), not features_path.is_file()
        from hmr4d.utils.preproc.vitfeat_extractor import get_batch
        crops, crop_boxes = timed("crops", prepare_shared_crops, prepared_video, boxes, pose_missing, features_missing, get_batch)
        if pose_missing:
            torch.save(timed("vitpose", models.extract_pose, crops, crop_boxes), pose_path)
            models.release("vitpose")
        if features_missing:
            torch.save(timed("hmr2", models.extract_features, crops), features_path)
            models.release("hmr2")
        del crops
        torch.cuda.empty_cache()
        if args.static_cam:
            camera_rotation = torch.eye(3).repeat(length, 1, 1)
        else:
            slam_path = Path(cfg.paths.slam)
            if not slam_path.is_file():
                vo = SimpleVO(str(prepared_video), scale=.5, step=8, method="sift", f_mm=args.f_mm)
                torch.save(torch.as_tensor(timed("camera", vo.compute)).float(), slam_path)
            camera_rotation = torch.load(slam_path, weights_only=True)[:, :3, :3].float()
        intrinsic = estimate_K(width, height) if args.f_mm is None else create_camera_sensor(width, height, args.f_mm)[2]
        prediction = timed("gvhmr", models.predict, boxes, torch.load(pose_path), torch.load(features_path),
                           intrinsic.repeat(length, 1, 1), camera_rotation, args.static_cam)
        temporary = result_path.with_suffix(".pt.part")
        torch.save(prediction, temporary)
        temporary.replace(result_path)
    profile = {"frames": length, "load_seconds": dict(models.load_seconds), "stage_seconds": stages,
               "total_seconds": perf_counter() - started}
    (result_path.parent / "runtime_profile.json").write_text(json.dumps(profile, indent=2))
    print(f"[GVHMR] Reconstructed {length} frames: {result_path}", flush=True)
    print(f"[GVHMR] Shared video decode/crops: {stages.get('crops', 0.0):.3f}s", flush=True)
    return result_path

