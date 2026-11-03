"""HADM extra-limb detector: model loading and inference on a folder of frames.

HADM (``metrics/multi-limbs/HADM``, an EVA-02 ViTDet detector) finds human
anatomy errors such as extra hands, arms, legs or feet in an image.  HADM's own
inference launcher loads the regular weights, overwrites every parameter and
buffer with the EMA (exponential moving average) weights, renders every image and
saves its training config.  This module loads the identical EMA weights once and
writes only the detections.

* :func:`load_hadm_model` builds the detector; the multi-video engine
  ``hmb.hadm_engine`` (used by HumanScore) loads it the same way.
* The command line runs HADM on a folder of extracted frames and writes
  ``<output-dir>/output.json`` (``{frame name: {bbox, score, class}}``).  The
  paper implementation (``metrics/multi-limbs/eval_extra_limbs.py``, used for
  ``--paper-scores``) calls it::

      python hmb/hadm_prepare.py --hadm-root metrics/multi-limbs/HADM --frames-dir FRAMES --output-dir OUT

  Its defaults reproduce HADM's own inference (one frame at a time with
  ``DefaultInferencer``, fp32, eager attention).  Faster settings are opt-in
  through ``--batch-size``, ``--precision`` and ``--attention``, or the
  environment variables ``HMB_HADM_BATCH_SIZE``, ``HMB_HADM_PRECISION`` and
  ``HMB_HADM_ATTENTION`` (which ``eval_extra_limbs.py`` passes through).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Optional

# Relative to the HADM root.  HADM-G is HADM's Global Human Artifact Detection Model (extra limbs etc.).
DEFAULT_CONFIG = "projects/ViTDet/configs/eva2_o365_to_coco/demo_global.py"
DEFAULT_CHECKPOINT = "pretrained_models/HADM-G_0249999.pth"
DEFAULT_SCORE_THRESH = 0.001  # keep detections down to this confidence; scoring applies its own thresholds
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def default_hadm_root() -> Path:
    """HADM source tree: ``$HMB_HADM_ROOT``, default ``metrics/multi-limbs/HADM``."""
    env = os.environ.get("HMB_HADM_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[1] / "metrics" / "multi-limbs" / "HADM"


def apply_ema_parameters(model, checkpoint) -> None:
    """Copy the EMA weights into ``model``, as HADM's ``EMAState`` does (by parameter/buffer name).

    Validate the entire EMA state before copying so a partial training
    checkpoint cannot silently leave random initialization in the detector.
    """
    import torch

    ema = checkpoint.get("ema_state")
    if not isinstance(ema, dict) or not ema:
        raise ValueError("HADM inference requires a complete ema_state checkpoint")
    state = dict(model.named_parameters())
    state.update(dict(model.named_buffers()))
    for name, value in state.items():
        if name not in ema or not torch.is_tensor(ema[name]) or ema[name].shape != value.shape:
            raise ValueError(f"HADM EMA parameter missing or incompatible: {name}")
    with torch.no_grad():
        for name, value in state.items():
            value.copy_(ema[name])


def resolve_assets(hadm_root, config_file: str = DEFAULT_CONFIG,
                   checkpoint: str = DEFAULT_CHECKPOINT) -> tuple[Path, Path, Path]:
    """Absolute (root, config, checkpoint); relative paths are under ``hadm_root``."""
    root = Path(hadm_root).expanduser().resolve()
    paths = []
    for value in (config_file, checkpoint):
        path = Path(value).expanduser()
        paths.append(path if path.is_absolute() else root / path)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    return root, paths[0], paths[1]


@dataclass
class HadmModel:
    """A ready-to-run HADM detector (eval mode, EMA weights, on ``device``)."""

    model: Any
    cfg: Any
    class_names: list
    device: str
    timings: dict = field(default_factory=dict)

    def inferencer(self):
        """HADM's per-image ``DefaultInferencer`` around this model."""
        from detectron2.engine.defaults import DefaultInferencer
        return DefaultInferencer(self.cfg, self.model)

    def bboxes(self) -> dict:
        """Optional ``inference.bbox_file`` of the config (unused by the default config)."""
        path = self.cfg.inference.get("bbox_file", None)
        return json.loads(Path(path).read_text()) if path and Path(path).is_file() else {}


def load_hadm_model(hadm_root=None, config_file: str = DEFAULT_CONFIG,
                    checkpoint: str = DEFAULT_CHECKPOINT,
                    score_thresh: float = DEFAULT_SCORE_THRESH, device: str = "cuda",
                    fast_init: bool = False) -> HadmModel:
    """Build the HADM EVA-02 ViTDet detector with its complete EMA state.

    ``fast_init`` constructs the module directly on ``device`` (default tensor
    device), so the random initialisation of ~0.4 B parameters runs on the GPU
    instead of the CPU and no CPU copy of the model is made.  (The meta device
    cannot be used: the ViT constructor calls ``.item()``.)  Every parameter and
    buffer is validated and overwritten by :func:`apply_ema_parameters`, so the
    weights are identical to the regular path.
    """
    started = perf_counter()
    if not math.isfinite(score_thresh) or not 0 <= score_thresh <= 1:
        raise ValueError("Detection score threshold must be finite and in [0, 1]")
    root, config_path, checkpoint_path = resolve_assets(hadm_root or default_hadm_root(),
                                                        config_file, checkpoint)
    # Keep a compiled Detectron2 overlay ahead of its unbuilt source tree.
    if str(root) not in sys.path:
        sys.path.append(str(root))
    import torch
    from detectron2.config import LazyConfig, instantiate
    from detectron2.data import MetadataCatalog

    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("HADM inference requires a CUDA GPU")
    cfg = LazyConfig.load(str(config_path))
    for predictor in cfg.model.roi_heads.box_predictors:
        predictor.test_score_thresh = score_thresh
    if importlib.util.find_spec("xformers") is None:
        cfg.model.backbone.net.xattn = False
    # Activation checkpointing only saves backward-pass memory. In evaluation
    # it wraps the same forward call with unnecessary autograd machinery.
    cfg.model.backbone.net.use_act_checkpoint = False
    cfg.train.device = device
    imported = perf_counter()
    if fast_init:
        with torch.device(device):
            model = instantiate(cfg.model)
    else:
        model = instantiate(cfg.model)
    constructed = perf_counter()
    # The source contains model+EMA (~5.4 GB). Mapping it reads only EMA pages;
    # regular weights and optimizer/training state are unused during inference.
    state = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=False)
    apply_ema_parameters(model, state)
    del state
    model = model.to(device).eval()
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    loaded = perf_counter()
    class_names = list(MetadataCatalog.get(cfg.dataloader.test.dataset.names).thing_classes)
    timings = {"imports_config_seconds": imported - started,
               "construct_seconds": constructed - imported,
               "ema_load_seconds": loaded - constructed,
               "model_load_seconds": loaded - imported, "fast_init": bool(fast_init)}
    return HadmModel(model=model, cfg=cfg, class_names=class_names, device=device, timings=timings)


def list_frames(frames_dir) -> list[Path]:
    """Image files of ``frames_dir`` in name order (at least one)."""
    frames = Path(frames_dir).expanduser().resolve()
    paths = sorted(path for path in frames.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise ValueError(f"No extracted frames found in {frames}")
    return paths


def detections_from_instances(instances, class_names) -> dict:
    """The ``output.json`` record of one image: ``{"bbox": [...], "score": [...], "class": [...]}``."""
    instances = instances.to("cpu")
    return {"bbox": instances.get("pred_boxes").tensor.tolist(),
            "score": instances.get("scores").tolist(),
            "class": [class_names[index] for index in instances.get("pred_classes").tolist()]}


def write_detections(output_dir, detections: dict, timing: Optional[dict] = None) -> Path:
    """Write ``output.json`` atomically (and ``inference_timing.json``); returns ``output.json``."""
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    target = output / "output.json"
    temporary = output / "output.json.part"
    temporary.write_text(json.dumps(detections), encoding="utf-8")
    temporary.replace(target)
    if timing is not None:
        (output / "inference_timing.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
    return target


def _env_default(name: str, default, cast):
    """``cast($name)`` if the environment variable is set and not empty, else ``default``."""
    value = os.environ.get(name)
    return cast(value) if value not in (None, "") else default


def reference_settings(args: argparse.Namespace) -> bool:
    """True for the original inference settings (batch 1, fp32, eager attention)."""
    return args.batch_size == 1 and args.precision == "fp32" and args.attention == "eager"


def infer(args: argparse.Namespace) -> Path:
    """Run HADM on ``args.frames_dir`` and write ``args.output_dir/output.json``.

    The reference settings use HADM's ``DefaultInferencer`` frame by frame;
    any other settings run the batched ``hmb.hadm_engine`` on the same frames.
    """
    started = perf_counter()
    paths = list_frames(args.frames_dir)
    if not reference_settings(args):
        return _infer_with_engine(args, paths, started)
    loaded_model = load_hadm_model(args.hadm_root, args.config_file, args.checkpoint,
                                   args.score_thresh, device="cuda", fast_init=True)
    import torch
    from detectron2.data.detection_utils import read_image

    loaded = perf_counter()
    inferencer = loaded_model.inferencer()
    bboxes = loaded_model.bboxes()
    detections = {}
    with torch.inference_mode():
        for path in paths:
            bbox = bboxes[path.name]["bbox"] if path.name in bboxes else None
            result = inferencer(read_image(str(path), format="BGR"), bbox=bbox)["instances"]
            detections[path.name] = detections_from_instances(result, loaded_model.class_names)
    torch.cuda.synchronize()
    finished = perf_counter()
    timing = {"backend": "upstream_hadm_ema_headless", "frames": len(paths),
              "imports_config_seconds": loaded_model.timings["imports_config_seconds"],
              "model_load_seconds": loaded_model.timings["model_load_seconds"],
              "inference_seconds": finished - loaded, "total_seconds": finished - started,
              "visualizations": False, "regular_checkpoint_weights_loaded": False}
    target = write_detections(args.output_dir, detections, timing)
    print(f"[HADM] {len(paths)} frames: load {timing['model_load_seconds']:.2f}s, "
          f"inference {timing['inference_seconds']:.2f}s; {target}", flush=True)
    return target


def _infer_with_engine(args: argparse.Namespace, paths: list, started: float) -> Path:
    """:func:`infer` with the batched engine (frames are read from JPEG files: no re-encoding)."""
    from hmb.hadm_engine import HadmEngine, HadmOptions, read_image_bgr

    engine = HadmEngine(HadmOptions(hadm_root=str(args.hadm_root), config_file=args.config_file,
                                    checkpoint=args.checkpoint, score_thresh=args.score_thresh,
                                    batch_size=args.batch_size, precision=args.precision,
                                    attention=args.attention))
    engine.load()
    loaded = perf_counter()
    records, stats = engine.detect((read_image_bgr(path) for path in paths), len(paths),
                                   jpeg_roundtrip=False)
    finished = perf_counter()
    detections = {path.name: record for path, record in zip(paths, records)}
    timing = {"backend": "hmb_hadm_engine", "frames": len(paths), "options": engine.options.as_dict(),
              **engine.load_timings, "inference_seconds": finished - loaded,
              "total_seconds": finished - started, **stats,
              "visualizations": False, "regular_checkpoint_weights_loaded": False}
    target = write_detections(args.output_dir, detections, timing)
    print(f"[HADM engine] {len(paths)} frames: load {engine.load_timings['model_load_seconds']:.2f}s, "
          f"inference {timing['inference_seconds']:.2f}s; {target}", flush=True)
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hadm-root", required=True, help="HADM source tree")
    parser.add_argument("--frames-dir", required=True, help="folder of extracted frames (.jpg/.jpeg/.png)")
    parser.add_argument("--output-dir", required=True, help="where output.json is written")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="relative to --hadm-root or absolute")
    parser.add_argument("--config-file", default=DEFAULT_CONFIG, help="relative to --hadm-root or absolute")
    parser.add_argument("--score-thresh", type=float, default=DEFAULT_SCORE_THRESH,
                        help="lowest detection confidence kept")
    parser.add_argument("--batch-size", type=int,
                        default=_env_default("HMB_HADM_BATCH_SIZE", 1, int), help="frames per forward pass")
    parser.add_argument("--precision", choices=("fp32", "tf32", "bf16", "fp16"),
                        default=_env_default("HMB_HADM_PRECISION", "fp32", str))
    parser.add_argument("--attention", choices=("eager", "sdpa"),
                        default=_env_default("HMB_HADM_ATTENTION", "eager", str),
                        help="sdpa: fused attention kernels (same maths)")
    return parser


def main() -> None:
    """Command line (see the module docstring)."""
    infer(build_parser().parse_args())


if __name__ == "__main__":
    main()
