"""Metric (I) extra limbs: the original HumanScore paper implementation.

Scores how often, how strongly and for how long the HADM-G human-artefact detector
reports a person with an extra hand, arm, leg or foot (HumanScore paper,
supplementary section F.1).

Where it sits in HumanScore
    With ``--paper-scores``, ``hmb/scoring/paper_scores.py`` calls
    :func:`compute_extra_limbs_score` on the session's HADM detections
    (``<S>/cache/multi_limbs/<stem>/hadm_results/output.json``).  The default
    extra-limbs score (``hmb/scoring/extra_limbs.py``) reads the same detections.

Definition (frame t of T, then the video)::

    M_t = highest confidence of the four extra-limb classes (0 when below tau_prob)
    b_t = 1 if M_t > tau_mild else 0
    m_t = clip((M_t - tau_mild) / (tau_severe - tau_mild), 0, 1)
    r = mean_t b_t,   s = sum_t m_t / max(1, sum_t b_t),   p = (longest run of b_t = 1) / T
    score = 100 * (1 - clip(0.5 r + 0.3 s + 0.2 p, 0, 1))

    defaults: tau_prob = tau_mild = 0.005, tau_severe = 0.03

Standalone use (video -> frames -> HADM -> score)::

    python metrics/multi-limbs/eval_extra_limbs.py --video_path clip.mp4 --output_dir out
    # score an existing HADM output.json without running the detector
    python metrics/multi-limbs/eval_extra_limbs.py --video_path clip.mp4 --detections-json output.json --output_dir out

Outputs: ``<output_dir>/<video>_extra_limbs.json`` per video and
``<output_dir>/extra_limbs_scores_summary.csv``.  The folder name ``multi-limbs``
contains a hyphen, so run this file by its path.
"""
import os
import json
import argparse
import subprocess
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Tuple, Optional

import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='.*cudnn.*')
warnings.filterwarnings('ignore', category=UserWarning, message='.*CUDNN.*')

#: HADM-G classes that count as an extra limb.
EXTRA_CLASSES = {
    "human with extra hand",
    "human with extra arm",
    "human with extra leg",
    "human with extra feet",
}

def _is_mjx_variant(video_stem: str) -> bool:
    """True for ``*_mjx`` videos: skeleton renderings written by ``baseline/main.py``, not inputs."""
    return video_stem.lower().endswith("_mjx")


def find_latest_output_json(results_dir: str) -> str:
    """
    Recursively search the HADM inference output directory for output.json.

    Supports both layouts:
      1) results_dir/output.json
      2) results_dir/<timestamp>/output.json (nested subdirectories)

    Returns:
        Path to the most recently modified output.json; empty string if none.
    """
    results_dir = os.path.abspath(results_dir)

    # Case 1: file directly in the root
    direct = os.path.join(results_dir, "output.json")
    if os.path.isfile(direct):
        return direct

    # Case 2: timestamped subdirectories / deeper structure
    candidates = []
    for root, dirs, files in os.walk(results_dir):
        if "output.json" in files:
            path = os.path.join(root, "output.json")
            candidates.append(path)

    if not candidates:
        return ""

    # Pick the most recently modified file
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def _list_image_files(dir_path: str) -> List[str]:
    """Sorted image file names (jpg/jpeg/png) in ``dir_path``; empty if it does not exist."""
    if not os.path.isdir(dir_path):
        return []
    return sorted(
        f for f in os.listdir(dir_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )


def _pick_frames_dir(mid_dir: str) -> str:
    """
    Prefer <mid_dir>/frames if it exists; otherwise fall back to mid_dir itself
    if it already contains extracted images.
    """
    frames_dir = os.path.join(mid_dir, "frames")
    if _list_image_files(frames_dir):
        return frames_dir
    if _list_image_files(mid_dir):
        return mid_dir
    return frames_dir


def extract_frames_from_video(
    video_path: str,
    frames_dir: str,
    overwrite: bool = False,
) -> Tuple[int, int, int]:
    """
    Extract frames from a single video into JPG images named frame_XXXXXX.jpg.

    Frames already present in ``frames_dir`` are reused unless ``overwrite``.

    Returns:
        T, H, W: number of frames and the frame height and width (for the log).
    """
    import cv2

    os.makedirs(frames_dir, exist_ok=True)

    # If frames already exist and overwrite=False, reuse them
    if not overwrite:
        existing = sorted(
            f for f in os.listdir(frames_dir) if f.lower().endswith((".jpg", ".png"))
        )
        if existing:
            sample_img = os.path.join(frames_dir, existing[0])
            img = cv2.imread(sample_img)
            if img is None:
                raise RuntimeError(f"Cannot read existing frame: {sample_img}")
            h, w = img.shape[:2]
            return len(existing), h, w

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    idx = 0
    h = w = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if h == 0:
            h, w = frame.shape[:2]
        out_path = os.path.join(frames_dir, f"frame_{idx:06d}.jpg")
        cv2.imwrite(out_path, frame)
        idx += 1

    cap.release()

    if idx == 0:
        raise RuntimeError(f"No frames extracted from {video_path}")

    return idx, h, w


def _extract_frames_worker(
    args: Tuple[str, str, str, bool],
) -> Tuple[str, int, int, int]:
    """
    Worker for parallel frame extraction (must be top-level for ProcessPoolExecutor).
    args: (name, video_path, frames_dir, overwrite)
    Returns: (name, T, H, W)
    """
    name, video_path, frames_dir, overwrite = args
    T, H, W = extract_frames_from_video(video_path, frames_dir, overwrite=overwrite)
    return (name, T, H, W)


def run_hadm_inference(
    hadm_root: str,
    frames_dir: str,
    results_dir: str,
    config_file: str = "projects/ViTDet/configs/eva2_o365_to_coco/demo_global.py",
    checkpoint_path: str = "pretrained_models/HADM-G_0249999.pth",
    num_gpus: int = 1,
    score_thresh: float = 0.001,
    gpu_id: Optional[int] = None,
):
    """Run HADM-G on every image in ``frames_dir`` and write ``output.json`` under ``results_dir``.

    By default the detector runs through ``hmb/hadm_prepare.py``, which loads the same
    EMA weights as HADM's own launcher but writes only the detections.  Set
    ``HMB_HADM_UPSTREAM=1`` (or use more than one GPU) to call HADM's
    ``tools/lazyconfig_train_net.py`` directly instead.

    ``frames_dir`` and ``results_dir`` are made absolute because the child runs with
    ``hadm_root`` as its working directory; ``config_file`` and ``checkpoint_path`` stay
    relative to ``hadm_root``.  ``gpu_id`` restricts the child to one GPU.
    """
    for required in ("tools/lazyconfig_train_net.py", config_file, checkpoint_path):
        path = os.path.join(hadm_root, required)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Required HADM asset not found: {path}")

    # Absolute paths to ensure accessibility when cwd switches to hadm_root
    frames_dir_abs = os.path.abspath(frames_dir)
    results_dir_abs = os.path.abspath(results_dir)
    os.makedirs(results_dir_abs, exist_ok=True)

    train_out_dir_abs = os.path.join(results_dir_abs, "train_out")

    # Use sys.executable to ensure we use the same Python interpreter
    import sys
    python_exe = sys.executable
    
    cmd = [
        python_exe,
        "tools/lazyconfig_train_net.py",
        "--num-gpus",
        str(num_gpus),
        "--inference",
        "--config-file",
        config_file,  # relative to hadm_root
        f"train.output_dir={train_out_dir_abs}",
        f"train.init_checkpoint={checkpoint_path}",  # relative to hadm_root
        "dataloader.train.total_batch_size=1",
        "train.model_ema.enabled=True",
        "train.model_ema.use_ema_weights_for_eval_only=True",
        f"inference.input_dir={frames_dir_abs}",
        f"inference.output_dir={results_dir_abs}",
        f"model.roi_heads.box_predictors.0.test_score_thresh={score_thresh}",
        f"model.roi_heads.box_predictors.1.test_score_thresh={score_thresh}",
        f"model.roi_heads.box_predictors.2.test_score_thresh={score_thresh}",
    ]
    import importlib.util
    if importlib.util.find_spec("xformers") is None:
        # The pinned EVA backbone provides standard softmax attention with the
        # same weights when its optional xformers kernel is unavailable.
        cmd.append("model.backbone.net.xattn=False")

    # Only the detection JSON is needed, not HADM's rendered frames or training setup.
    use_adapter = os.environ.get("HMB_HADM_UPSTREAM", "0") != "1" and num_gpus == 1
    if use_adapter:
        from pathlib import Path
        adapter = Path(__file__).resolve().parents[2] / "hmb/hadm_prepare.py"
        cmd = [python_exe, str(adapter), "--hadm-root", os.path.abspath(hadm_root),
               "--frames-dir", frames_dir_abs, "--output-dir", results_dir_abs,
               "--config-file", config_file, "--checkpoint", checkpoint_path,
               "--score-thresh", str(score_thresh)]

    print("[ExtraLimbs] Running HADM:", " ".join(cmd))
    
    # Set PYTHONPATH to include hadm_root (for detectron2 import)
    env = os.environ.copy()
    # The official HADM checkpoint holds EMA and training state, not only tensors, which
    # recent PyTorch refuses to load by default (weights_only). Allow it in this child only.
    env.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    env.setdefault("WANDB_MODE", "disabled")
    hadm_root_abs = os.path.abspath(hadm_root)
    # hmb/hadm_prepare.py appends hadm_root to sys.path itself, behind the installed (compiled,
    # patched) detectron2. On PYTHONPATH, hadm_root would put the unbuilt detectron2 source tree
    # in front of it, which fails to import with Pillow >= 10 (Image.LINEAR).
    if not use_adapter:
        if "PYTHONPATH" in env:
            env["PYTHONPATH"] = f"{env['PYTHONPATH']}:{hadm_root_abs}"
        else:
            env["PYTHONPATH"] = hadm_root_abs
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    ret = subprocess.run(cmd, cwd=hadm_root, env=env)
    if ret.returncode != 0:
        raise RuntimeError(f"HADM inference failed for {frames_dir_abs}")


def run_hadm_inference_batched(
    hadm_root: str,
    tasks: List[Tuple[str, str, str]],
    gpu_id: Optional[int] = None,
    config_file: str = "projects/ViTDet/configs/eva2_o365_to_coco/demo_global.py",
    checkpoint_path: str = "pretrained_models/HADM-G_0249999.pth",
    score_thresh: float = 0.001,
) -> None:
    """Run HADM once for several videos (one model load).

    ``tasks`` is a list of ``(name, frames_dir, results_dir)``.  The tasks are written to a
    temporary manifest that HADM reads through ``inference.manifest_path``; a HADM checkout
    without manifest support falls back to one :func:`run_hadm_inference` call per video.
    """
    import sys
    import tempfile

    if not tasks:
        return
    hadm_root_abs = os.path.abspath(hadm_root)
    inference_script = os.path.join(hadm_root_abs, "tools/lazyconfig_train_net.py")
    if not os.path.isfile(inference_script):
        raise FileNotFoundError(f"HADM inference source not found: {inference_script}")
    with open(inference_script, "r", encoding="utf-8") as source:
        supports_manifest = "manifest_path" in source.read()
    if not supports_manifest:
        print("[ExtraLimbs] Upstream HADM does not support batched manifests; infer each video separately.")
        for _name, frames_dir, results_dir in tasks:
            run_hadm_inference(
                hadm_root=hadm_root_abs, frames_dir=frames_dir,
                results_dir=results_dir, gpu_id=gpu_id,
                config_file=config_file, checkpoint_path=checkpoint_path,
                score_thresh=score_thresh,
            )
        return
    manifest = [
        {"input_dir": os.path.abspath(fd), "output_dir": os.path.abspath(rd)}
        for _name, fd, rd in tasks
    ]
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, prefix="hadm_manifest_"
    ) as f:
        json.dump(manifest, f, indent=2)
        manifest_path = f.name

    try:
        python_exe = sys.executable
        # First entry's results_dir for train.output_dir (HADM may still write something)
        _, _fd0, rd0 = tasks[0]
        train_out_dir_abs = os.path.join(os.path.abspath(rd0), "train_out")
        cmd = [
            python_exe,
            "tools/lazyconfig_train_net.py",
            "--num-gpus",
            "1",
            "--inference",
            "--config-file",
            config_file,
            f"train.output_dir={train_out_dir_abs}",
            f"train.init_checkpoint={checkpoint_path}",
            "dataloader.train.total_batch_size=1",
            "train.model_ema.enabled=True",
            "train.model_ema.use_ema_weights_for_eval_only=True",
            f"inference.manifest_path={os.path.abspath(manifest_path)}",
            f"model.roi_heads.box_predictors.0.test_score_thresh={score_thresh}",
            f"model.roi_heads.box_predictors.1.test_score_thresh={score_thresh}",
            f"model.roi_heads.box_predictors.2.test_score_thresh={score_thresh}",
        ]
        print(
            f"[ExtraLimbs] Running HADM batched (1 process, {len(tasks)} videos) on GPU {gpu_id}..."
        )
        env = os.environ.copy()
        env.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
        env.setdefault("WANDB_MODE", "disabled")
        if "PYTHONPATH" in env:
            env["PYTHONPATH"] = f"{hadm_root_abs}:{env['PYTHONPATH']}"
        else:
            env["PYTHONPATH"] = hadm_root_abs
        if gpu_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        ret = subprocess.run(
            cmd, cwd=hadm_root_abs, env=env,
            capture_output=True, text=True,
        )
        if ret.returncode != 0:
            err_msg = (
                f"HADM batched inference failed for {len(tasks)} videos (manifest: {manifest_path})"
            )
            if ret.stdout:
                err_msg += f"\n[stdout]\n{ret.stdout[-8000:]}"  # last 8k chars
            if ret.stderr:
                err_msg += f"\n[stderr]\n{ret.stderr[-8000:]}"
            raise RuntimeError(err_msg)
    finally:
        try:
            os.unlink(manifest_path)
        except OSError:
            pass


def _hadm_inference_worker(
    args: Tuple[str, str, str, int, str],
) -> str:
    """
    Worker for parallel HADM inference (top-level for ProcessPoolExecutor).
    args: (hadm_root, frames_dir, results_dir, gpu_id, name)
    Returns: name (on success).
    """
    hadm_root, frames_dir, results_dir, gpu_id, name = args
    run_hadm_inference(
        hadm_root=hadm_root,
        frames_dir=frames_dir,
        results_dir=results_dir,
        num_gpus=1,
        gpu_id=gpu_id,
    )
    return name


def compute_extra_limbs_score(
    output_json_path: str,
    # Confidence threshold: only trust detections above this prob
    tau_prob: float = 0.005,
    # Piecewise thresholds for M_t = maximum extra-limb confidence
    tau_mild: float = 0.005,
    tau_severe: float = 0.03,
    # Aggregation weights (auto-normalized): frequency + severity + continuity
    alpha: float = 0.5,   # r: frequency of frames with visible extra limbs
    beta: float = 0.3,    # s: mean severity when extra limbs appear
    gamma: float = 0.2,   # p: continuity of extra-limb segments
) -> Dict[str, float]:
    """Extra-limbs score of one video from its HADM ``output.json``.

    The file maps each frame image to its detections::

        {"frame_000000.jpg": {"bbox": [...], "score": [...], "class": [...]}, ...}

    Per frame t:

    * ``M_t`` = highest score among :data:`EXTRA_CLASSES` (0 when below ``tau_prob``);
    * ``M_t <= tau_mild``: ``b_t = 0, m_t = 0``;
      ``tau_mild < M_t < tau_severe``: ``b_t = 1, m_t = (M_t - tau_mild) / (tau_severe - tau_mild)``;
      ``M_t >= tau_severe``: ``b_t = 1, m_t = 1``.

    Per video: ``r = mean_t b_t``, ``s = sum_t m_t / max(1, sum_t b_t)``,
    ``p = longest run of b_t = 1 / T``, ``D = w1 r + w2 s + w3 p`` with the weights
    normalised to sum to 1, and ``score = 100 (1 - clip(D, 0, 1))`` (higher is better).
    Bounding-box size does not enter the score.

    Returns ``{"score", "D", "r", "severity_mean", "continuity", "avg_M_t", "num_frames"}``.
    Raises ``ValueError`` for an empty or malformed file.
    """
    if not os.path.isfile(output_json_path):
        raise FileNotFoundError(output_json_path)

    with open(output_json_path, "r") as f:
        data = json.load(f)

    if not isinstance(data, dict) or not data:
        raise ValueError("HADM output must contain at least one frame; cannot score empty inference.")
    if not 0 <= tau_mild < tau_severe <= 1:
        raise ValueError("Require 0 <= tau_mild < tau_severe <= 1.")

    # Sort frames to keep continuity computation stable
    frame_keys = sorted(data.keys())

    M_ts: List[float] = []
    b_ts: List[int] = []
    m_ts: List[float] = []

    for k in frame_keys:
        det = data[k]
        scores = det.get("score", [])
        classes = det.get("class", [])

        if not scores or not classes:
            M_ts.append(0.0)
            b_ts.append(0)
            m_ts.append(0.0)
            continue

        if len(scores) != len(classes):
            raise ValueError(f"Mismatched HADM score/class arrays for frame {k}.")

        max_p = 0.0

        for p, cls in zip(scores, classes):
            if cls not in EXTRA_CLASSES:
                continue
            p = float(p)
            if not 0 <= p <= 1:
                raise ValueError(f"Invalid HADM confidence {p} for frame {k}.")
            max_p = max(max_p, p)

        if max_p < float(tau_prob):
            M_t = 0.0
        else:
            M_t = float(max_p)

        # Piecewise mapping to b_t, m_t
        if M_t <= tau_mild:
            b = 0
            m = 0.0
        elif M_t >= tau_severe:
            b = 1
            m = 1.0
        else:
            b = 1
            m = float((M_t - tau_mild) / max(tau_severe - tau_mild, 1e-8))

        M_ts.append(M_t)
        b_ts.append(b)
        m_ts.append(m)

    T = len(M_ts)
    if T == 0:
        return dict(
            score=100.0,
            D=0.0,
            r=0.0,
            severity_mean=0.0,
            continuity=0.0,
            avg_M_t=0.0,
            num_frames=0,
        )

    # Frequency
    r = float(sum(b_ts)) / float(T)

    # Severity (average m_t over frames with b_t=1)
    num_pos = sum(b_ts)
    if num_pos > 0:
        severity_mean = float(sum(m_ts) / float(num_pos))
    else:
        severity_mean = 0.0

    # Continuity: longest stretch of consecutive extra-limb frames
    longest = 0
    cur = 0
    for b in b_ts:
        if b == 1:
            cur += 1
            if cur > longest:
                longest = cur
        else:
            cur = 0
    continuity = float(longest) / float(T)

    # Normalize weights
    w_sum = alpha + beta + gamma
    if w_sum <= 0:
        w1, w2, w3 = 0.5, 0.3, 0.2
    else:
        w1, w2, w3 = alpha / w_sum, beta / w_sum, gamma / w_sum

    D_raw = w1 * r + w2 * severity_mean + w3 * continuity
    D = float(max(0.0, min(1.0, D_raw)))
    score = 100.0 * (1.0 - D)

    avg_M_t = float(sum(M_ts) / T)

    return dict(
        score=float(score),
        D=D,
        r=r,
        severity_mean=severity_mean,
        continuity=continuity,
        avg_M_t=avg_M_t,
        num_frames=T,
    )


def _normalize_video_name(name: str) -> str:
    """Video name of a cached work-dir folder: '-' becomes '_' before any '__suffix'.

    Example: ``clapping-hands__gentle`` -> ``clapping_hands__gentle``.
    """
    if "__" not in name:
        return name.replace("-", "_")
    action, suffix = name.rsplit("__", 1)
    action = action.replace("-", "_")
    return f"{action}__{suffix}"


def _norm_key(name: str) -> str:
    """Lower-case lookup key that treats '-' and '_' alike (the '__suffix' is kept)."""
    if "__" not in name:
        return name.replace("-", "_").lower()
    action, suffix = name.rsplit("__", 1)
    return f"{action.replace('-', '_')}__{suffix}".lower()


def _build_work_dir_index(work_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    """Index the work-dir folders by :func:`_norm_key`, recording cached frames and detections.

    Lets a video reuse an earlier folder even when its name differs by '-' vs '_' or case.
    """
    index: Dict[str, List[Dict[str, Any]]] = {}
    if not work_dir or not os.path.isdir(work_dir):
        return index

    for d in sorted(os.listdir(work_dir)):
        mid_dir = os.path.join(work_dir, d)
        if not os.path.isdir(mid_dir):
            continue
        results_dir = os.path.join(mid_dir, "hadm_results")
        output_json = find_latest_output_json(results_dir) if os.path.isdir(results_dir) else ""
        has_output = bool(output_json)
        frames_dir = _pick_frames_dir(mid_dir)
        has_frames = bool(_list_image_files(frames_dir))
        rec = {
            "raw_dir": d,
            "mid_dir": mid_dir,
            "has_output": has_output,
            "output_json": output_json,
            "output_mtime": os.path.getmtime(output_json) if has_output else 0.0,
            "frames_dir": frames_dir,
            "has_frames": has_frames,
        }
        index.setdefault(_norm_key(d), []).append(rec)
    return index


def _select_best_mid_dir(
    work_index: Dict[str, List[Dict[str, Any]]],
    work_dir: str,
    video_name: str,
) -> Dict[str, Any]:
    """Work folder for ``video_name``.

    Prefers a folder with cached detections (newest ``output.json``), then one with
    extracted frames, else ``<work_dir>/<video_name>``.
    """
    key = _norm_key(video_name)
    cands = work_index.get(key, [])
    if cands:
        # Prefer cached outputs; tie-break by newest output.json mtime.
        with_output = [c for c in cands if c.get("has_output")]
        if with_output:
            with_output.sort(key=lambda c: c.get("output_mtime", 0.0), reverse=True)
            return with_output[0]
        # Otherwise, prefer having frames
        with_frames = [c for c in cands if c.get("has_frames")]
        if with_frames:
            return with_frames[0]
        return cands[0]

    # No match in index → use a canonical mid_dir path under work_dir
    mid_dir = os.path.join(work_dir, video_name)
    return {
        "raw_dir": video_name,
        "mid_dir": mid_dir,
        "has_output": False,
        "output_json": "",
        "output_mtime": 0.0,
        "frames_dir": os.path.join(mid_dir, "frames"),
        "has_frames": False,
    }


def main():
    """Command-line entry point; see the module docstring."""
    parser = argparse.ArgumentParser(
        description="Metric (I) extra limbs, paper implementation: video -> frames -> HADM -> score."
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default="",
        help="Single input video path. If set, ignore directory scanning under --video_dir.",
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default="data",
        help="Input video directory containing *.mp4 files",
    )
    parser.add_argument(
        "--detections-json", type=str, default="",
        help="Score one existing HADM output.json without extracting frames or running inference; requires --video_path.",
    )
    parser.add_argument(
        "--hadm_root",
        type=str,
        default="metrics/multi-limbs/HADM",
        help="Path to the HADM repository root",
    )
    parser.add_argument(
        "--work_dir",
        type=str,
        default="results/metrics/extra_limbs/mid_outputs",
        help="Intermediate directory (frames + HADM outputs)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/metrics/extra_limbs",
        help="Output directory for per-video JSON + summary CSV",
    )
    parser.add_argument(
        "--overwrite_frames",
        action="store_true",
        help="Re-extract frames even if they already exist",
    )
    parser.add_argument(
        "--extract_workers",
        type=int,
        default=1,
        help="Number of CPU workers for parallel frame extraction (default 1 = sequential). "
        "Use e.g. 24 to utilize 24 CPUs for extract step.",
    )
    parser.add_argument(
        "--skip_inference",
        action="store_true",
        help="Skip rerunning HADM; assume output.json already exists",
    )
    parser.add_argument(
        "--hadm_gpu_ids",
        type=str,
        default="",
        help="Comma-separated GPU IDs for HADM inference (e.g. '0,1'). "
        "If set, multiple HADM processes can run in parallel, each bound to one GPU.",
    )
    parser.add_argument(
        "--hadm_processes_per_gpu",
        type=int,
        default=1,
        help="Number of concurrent HADM inference processes per GPU (default 1). "
        "Used only when --no_hadm_batched (multi-process mode).",
    )
    parser.add_argument(
        "--no_hadm_batched",
        action="store_true",
        help="Disable batched HADM (default: batched = one process per GPU, one model load, N videos in batch). "
        "When set, use multiple processes per GPU instead (may serialize on GPU and slow each video).",
    )
    parser.add_argument(
        "--hadm_batch_size",
        type=int,
        default=5,
        help="Max number of videos per HADM batched process (default 5). "
        "E.g. 10 videos with batch_size=5 run 2 batched processes (5+5). Avoid OOM or timeout.",
    )
    parser.add_argument(
        "--score_from_work_dir",
        action="store_true",
        help=(
            "Rescore from precomputed mid_outputs under --work_dir (no mp4 needed). "
            "Each subdir under --work_dir is treated as one video; we search its hadm_results for output.json."
        ),
    )
    # Thresholds of the score (see compute_extra_limbs_score)
    parser.add_argument("--tau_prob", type=float, default=0.005,
                        help="Ignore extra-limb detections below this confidence (default 0.005)")
    parser.add_argument("--tau_mild", type=float, default=0.005,
                        help="A frame is flagged above this confidence (default 0.005)")
    parser.add_argument("--tau_severe", type=float, default=0.03,
                        help="Confidence of full severity (default 0.03)")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Aggregation weight: frequency of frames with extra limbs (default 0.5)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.3,
        help="Aggregation weight: mean severity when extra limbs appear (default 0.3)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.2,
        help="Aggregation weight: continuity of extra-limb segments (default 0.2)",
    )

    args = parser.parse_args()

    if args.detections_json:
        if not args.video_path:
            parser.error("--detections-json requires --video_path to identify the input clip")
        name = os.path.splitext(os.path.basename(args.video_path))[0]
        stats = compute_extra_limbs_score(
            args.detections_json, tau_prob=args.tau_prob,
            tau_mild=args.tau_mild, tau_severe=args.tau_severe,
            alpha=args.alpha, beta=args.beta, gamma=args.gamma,
        )
        result = {
            "video": name, "extra_limbs_score": stats.pop("score"),
            "scoring": "humanscore-paper", "inference": "cached",
            "source_output_json": os.path.abspath(args.detections_json), **stats,
        }
        os.makedirs(args.output_dir, exist_ok=True)
        out_json = os.path.join(args.output_dir, f"{name}_extra_limbs.json")
        with open(out_json, "w", encoding="utf-8") as output:
            json.dump(result, output, indent=2)
        print(f"[ExtraLimbs] {name}: score={result['extra_limbs_score']:.2f} (cached detections: {args.detections_json})")
        return

    import cv2
    import pandas as pd

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[ExtraLimbs] Output dir (scores + summary CSV): {os.path.abspath(args.output_dir)}")

    work_index = _build_work_dir_index(args.work_dir)

    if args.score_from_work_dir:
        # Rescore all cached mid outputs under work_dir.
        # This is useful when mp4 files are missing or when you want to reuse cached HADM outputs.
        mid_dirs = [
            d for d in sorted(os.listdir(args.work_dir))
            if os.path.isdir(os.path.join(args.work_dir, d))
        ]
        video_files = []  # unused in this mode
    else:
        if args.video_path:
            if not os.path.isfile(args.video_path):
                raise FileNotFoundError(args.video_path)
            video_files = [os.path.basename(args.video_path)]
            args.video_dir = os.path.dirname(os.path.abspath(args.video_path)) or "."
        else:
            video_files = sorted(
                f
                for f in os.listdir(args.video_dir)
                if f.lower().endswith(".mp4") and not f.lower().endswith("_mjx.mp4")
            )
        mid_dirs = []

    all_results = []

    # ---------- Phase 1: parallel frame extraction (when extract_workers > 1) ----------
    extracted_names: set = set()
    if not args.score_from_work_dir and video_files and args.extract_workers > 1:
        extract_tasks: List[Tuple[str, str, str, bool]] = []
        for vf in video_files:
            video_path = os.path.join(args.video_dir, vf)
            name = os.path.splitext(vf)[0]
            if _is_mjx_variant(name):
                continue
            if not args.overwrite_frames:
                chosen = _select_best_mid_dir(work_index, args.work_dir, name)
                mid_dir = chosen["mid_dir"]
            else:
                mid_dir = os.path.join(args.work_dir, name)
            frames_dir = _pick_frames_dir(mid_dir)
            results_dir = os.path.join(mid_dir, "hadm_results")
            if not args.overwrite_frames and find_latest_output_json(results_dir):
                continue
            extract_tasks.append((name, video_path, frames_dir, args.overwrite_frames))

        if extract_tasks:
            n_workers = min(args.extract_workers, len(extract_tasks))
            print(
                f"[ExtraLimbs] Parallel frame extraction with {n_workers} workers "
                f"for {len(extract_tasks)} videos..."
            )
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                future_to_name = {
                    executor.submit(_extract_frames_worker, t): t[0]
                    for t in extract_tasks
                }
                for future in as_completed(future_to_name):
                    name = future_to_name[future]
                    try:
                        _name, T, H, W = future.result()
                        extracted_names.add(_name)
                        print(f"[ExtraLimbs] Extracted {_name}: {T} frames, {W}x{H}")
                    except Exception as e:
                        print(f"[ExtraLimbs] Extract failed for {name}: {e}")

    # ---------- Phase 2: parallel HADM inference (when hadm_gpu_ids + hadm_processes_per_gpu) ----------
    hadm_done_names: set = set()
    if (
        not args.score_from_work_dir
        and video_files
        and not args.skip_inference
    ):
        hadm_tasks: List[Tuple[str, str, str]] = []  # (name, frames_dir, results_dir)
        for vf in video_files:
            name = os.path.splitext(vf)[0]
            if _is_mjx_variant(name):
                continue
            if not args.overwrite_frames:
                chosen = _select_best_mid_dir(work_index, args.work_dir, name)
                mid_dir = chosen["mid_dir"]
            else:
                mid_dir = os.path.join(args.work_dir, name)
            frames_dir = _pick_frames_dir(mid_dir)
            results_dir = os.path.join(mid_dir, "hadm_results")
            if not args.overwrite_frames and find_latest_output_json(results_dir):
                continue
            # Only add if frames already exist (Phase 1 extracted, or cached from before)
            if not _list_image_files(frames_dir):
                continue
            hadm_tasks.append((name, frames_dir, results_dir))

        gpu_ids: List[int] = []
        if args.hadm_gpu_ids.strip():
            for s in args.hadm_gpu_ids.strip().split(","):
                s = s.strip()
                if s.isdigit():
                    gpu_ids.append(int(s))
        use_batched = not args.no_hadm_batched

        if gpu_ids and hadm_tasks:
            if use_batched:
                # Single process per GPU per chunk: one model load per run, up to hadm_batch_size videos per run.
                batch_size = max(1, args.hadm_batch_size)
                tasks_per_gpu: Dict[int, List[Tuple[str, str, str]]] = {}
                for i, (name, fd, rd) in enumerate(hadm_tasks):
                    gid = gpu_ids[i % len(gpu_ids)]
                    tasks_per_gpu.setdefault(gid, []).append((name, fd, rd))
                total_batches = sum(
                    (len(tasks) + batch_size - 1) // batch_size
                    for tasks in tasks_per_gpu.values()
                )
                print(
                    f"[ExtraLimbs] HADM batched mode: up to {batch_size} videos per process, "
                    f"{len(hadm_tasks)} videos -> {total_batches} batch(es) across {len(tasks_per_gpu)} GPU(s)"
                )

                def run_gpu_chunks(gid: int, gpu_tasks: List[Tuple[str, str, str]]) -> None:
                    for start in range(0, len(gpu_tasks), batch_size):
                        chunk = gpu_tasks[start : start + batch_size]
                        run_hadm_inference_batched(
                            hadm_root=args.hadm_root,
                            tasks=chunk,
                            gpu_id=gid,
                        )
                        print(f"[ExtraLimbs] HADM batched done on GPU {gid}: {len(chunk)} videos")

                if len(tasks_per_gpu) == 1:
                    gid = next(iter(tasks_per_gpu))
                    run_gpu_chunks(gid, tasks_per_gpu[gid])
                else:
                    with ThreadPoolExecutor(max_workers=len(tasks_per_gpu)) as executor:
                        futures = {
                            executor.submit(
                                run_gpu_chunks,
                                gid,
                                tasks_per_gpu[gid],
                            ): gid
                            for gid in sorted(tasks_per_gpu.keys())
                        }
                        for future in as_completed(futures):
                            gid = futures[future]
                            try:
                                future.result()
                            except Exception as e:
                                print(f"[ExtraLimbs] HADM batched failed on GPU {gid}: {e}")
                for name, _, _ in hadm_tasks:
                    hadm_done_names.add(name)
            else:
                # Multi-process mode: hadm_processes_per_gpu processes per GPU (may serialize on GPU).
                per_gpu = max(1, args.hadm_processes_per_gpu)
                slots: List[int] = [gid for gid in gpu_ids for _ in range(per_gpu)]
                n_workers = min(len(slots), len(hadm_tasks))
                print(
                    f"[ExtraLimbs] Parallel HADM inference: {n_workers} workers "
                    f"({len(gpu_ids)} GPU(s) x {per_gpu} process(es) per GPU) for {len(hadm_tasks)} videos"
                )
                task_args = [
                    (
                        args.hadm_root,
                        fd,
                        rd,
                        slots[i % len(slots)],
                        name,
                    )
                    for i, (name, fd, rd) in enumerate(hadm_tasks)
                ]
                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    future_to_name = {
                        executor.submit(_hadm_inference_worker, t): t[4]
                        for t in task_args
                    }
                    for future in as_completed(future_to_name):
                        name = future_to_name[future]
                        try:
                            _name = future.result()
                            hadm_done_names.add(_name)
                            print(f"[ExtraLimbs] HADM done: {_name}")
                        except Exception as e:
                            print(f"[ExtraLimbs] HADM failed for {name}: {e}")
                for name, _, _ in hadm_tasks:
                    hadm_done_names.add(name)

    if args.score_from_work_dir:
        # 1) Collect candidates (some dirs may be frames-only or incomplete)
        candidates: List[Dict[str, Any]] = []
        for d in mid_dirs:
            mid_dir = os.path.join(args.work_dir, d)
            results_dir = os.path.join(mid_dir, "hadm_results")
            output_json_path = find_latest_output_json(results_dir)
            if not output_json_path:
                continue
            try:
                mtime = os.path.getmtime(output_json_path)
            except OSError:
                mtime = 0.0
            candidates.append(
                {
                    "raw_dir": d,
                    "mid_dir": mid_dir,
                    "output_json": output_json_path,
                    "output_mtime": float(mtime),
                    "video": _normalize_video_name(d),
                }
            )

        # 2) De-duplicate by normalized video name (protect against '-' vs '_' variants).
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for c in candidates:
            grouped.setdefault(c["video"], []).append(c)

        selected: List[Dict[str, Any]] = []
        for video, items in grouped.items():
            if _is_mjx_variant(video):
                continue
            if len(items) == 1:
                selected.append(items[0])
                continue
            # Pick the newest output.json among duplicates
            items_sorted = sorted(items, key=lambda x: x.get("output_mtime", 0.0), reverse=True)
            chosen = items_sorted[0]
            print(
                f"[ExtraLimbs] Warning: duplicated cached dirs map to same video '{video}'. "
                f"Choosing '{chosen['raw_dir']}' and skipping {[i['raw_dir'] for i in items_sorted[1:]]}"
            )
            selected.append(chosen)

        # 3) Rescore
        for item in sorted(selected, key=lambda x: x["video"]):
            name = item["video"]
            if _is_mjx_variant(name):
                continue
            output_json_path = item["output_json"]

            stats = compute_extra_limbs_score(
                output_json_path=output_json_path,
                tau_prob=args.tau_prob,
                tau_mild=args.tau_mild,
                tau_severe=args.tau_severe,
                alpha=args.alpha,
                beta=args.beta,
                gamma=args.gamma,
            )

            result = {
                "video": name,
                "extra_limbs_score": stats["score"],
                "scoring": "humanscore-paper",
                "D": stats["D"],
                "r": stats["r"],
                "severity_mean": stats["severity_mean"],
                "continuity": stats["continuity"],
                "avg_M_t": stats["avg_M_t"],
                "num_frames": stats.get("num_frames", 0),
                # where the scored detections came from
                "source_mid_dir": item["raw_dir"],
                "source_output_json": output_json_path,
                "inference": "cached",
            }
            all_results.append(result)

            out_json = os.path.join(args.output_dir, f"{name}_extra_limbs.json")
            with open(out_json, "w") as f:
                json.dump(result, f, indent=2)

        # Summary CSV (same format as the video pipeline)
        if all_results:
            df = pd.DataFrame(all_results)
            summary_csv = os.path.join(args.output_dir, "extra_limbs_scores_summary.csv")
            df.to_csv(summary_csv, index=False)
            print(f"[ExtraLimbs] Summary saved to {summary_csv}")
        else:
            raise RuntimeError("No valid cached HADM results to rescore.")
        return

    for vf in video_files:
        video_path = os.path.join(args.video_dir, vf)
        name = os.path.splitext(vf)[0]

        print(f"[ExtraLimbs] Processing video: {name}")

        # Prefer reusing existing cached mid_outputs (even if name differs by '-'/'_' or case)
        if not args.overwrite_frames:
            chosen = _select_best_mid_dir(work_index, args.work_dir, name)
            mid_dir = chosen["mid_dir"]
        else:
            mid_dir = os.path.join(args.work_dir, name)

        frames_dir = _pick_frames_dir(mid_dir)
        results_dir = os.path.join(mid_dir, "hadm_results")

        reuse_cached = False
        output_json_path = ""
        if not args.overwrite_frames:
            cached = find_latest_output_json(results_dir)
            if cached:
                reuse_cached = True
                output_json_path = cached
                print(
                    f"[ExtraLimbs] {name}: detected cached HADM output, "
                    "skip extraction/inference."
                )

        if not reuse_cached:
            # 1) Extract or reuse frames (already done in parallel phase if extract_workers > 1)
            if name in extracted_names:
                # Frames were extracted in phase 1; get T, H, W from existing frames
                existing = _list_image_files(frames_dir)
                if existing:
                    sample_img = os.path.join(frames_dir, existing[0])
                    img = cv2.imread(sample_img)
                    if img is not None:
                        H, W = img.shape[0], img.shape[1]
                        T = len(existing)
                    else:
                        T, H, W = 0, 0, 0
                else:
                    T, H, W = 0, 0, 0
                print(f"[ExtraLimbs] {name}: frames ready ({T} frames, {W}x{H}) [from parallel extract]")
            else:
                T, H, W = extract_frames_from_video(
                    video_path, frames_dir, overwrite=args.overwrite_frames
                )
                print(f"[ExtraLimbs] {name}: frames ready ({T} frames, {W}x{H})")

            # 2) HADM inference (only if not skipping and not already done in Phase 2)
            if not args.skip_inference and name not in hadm_done_names:
                run_hadm_inference(
                    hadm_root=args.hadm_root,
                    frames_dir=frames_dir,
                    results_dir=results_dir,
                )

            output_json_path = find_latest_output_json(results_dir)

        if not os.path.isfile(output_json_path):
            print(
                f"[ExtraLimbs] Warning: HADM output not found, skip scoring for {name}. "
                f"Looked in results_dir={results_dir} (output_json_path={output_json_path!r})"
            )
            continue

        # 3) Compute extra-limbs score
        stats = compute_extra_limbs_score(
            output_json_path=output_json_path,
            tau_prob=args.tau_prob,
            tau_mild=args.tau_mild,
            tau_severe=args.tau_severe,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
        )

        result = {
            "video": name,
            "extra_limbs_score": stats["score"],
            "scoring": "humanscore-paper",
            "D": stats["D"],
            "r": stats["r"],
            "severity_mean": stats["severity_mean"],
            "continuity": stats["continuity"],
            "avg_M_t": stats["avg_M_t"],
            "num_frames": stats.get("num_frames", 0),
            "inference": "cached" if reuse_cached and name not in hadm_done_names else "fresh",
            "source_output_json": os.path.abspath(output_json_path),
        }
        all_results.append(result)

        # per-video JSON
        out_json = os.path.join(args.output_dir, f"{name}_extra_limbs.json")
        with open(out_json, "w") as f:
            json.dump(result, f, indent=2)

        print(
            f"[ExtraLimbs] {name}: score={result['extra_limbs_score']:.2f}, "
            f"r={result['r']:.4f}, severity={result['severity_mean']:.4f}, "
            f"continuity={result['continuity']:.4f}"
        )

    # Summary CSV
    if all_results:
        df = pd.DataFrame(all_results)
        summary_csv = os.path.join(args.output_dir, "extra_limbs_scores_summary.csv")
        df.to_csv(summary_csv, index=False)
        print(f"[ExtraLimbs] Summary saved to {summary_csv}")
    else:
        raise RuntimeError("No valid extra-limb scores computed; inspect HADM inference failures above.")

    if len(all_results) != len(video_files):
        raise RuntimeError(f"Scored {len(all_results)} of {len(video_files)} requested videos.")


if __name__ == "__main__":
    main()
