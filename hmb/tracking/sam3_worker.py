"""SAM 3.1 multi-person tracking worker (runs in the Python environment that has ``sam3``).

Executed as a script by :mod:`hmb.tracking.sam3_backend` with an interpreter that
has torch with CUDA and the ``sam3`` package (https://github.com/facebookresearch/sam3):
the HumanScore environment itself after ``bash install.sh --with-sam3``, or a separate
environment (``$HMB_SAM3_PYTHON``).  Because that interpreter may be a different one, the
worker must not import anything from the HumanScore project: only the standard library and
numpy at module level; torch / sam3 / huggingface_hub are imported where they are used.

It opens a SAM 3.1 (Object Multiplex) video session on the input video, prompts
frame 0 with the text ``person`` (SAM 3.1 keeps detecting people that appear
later), propagates through the whole video and writes ``--out`` (an ``.npz``)
with one record per visible (frame, object): ``frame``, ``obj_id``, ``box``
(tight xyxy of the mask, exclusive max), ``score`` and the mask as an
uncompressed column-major COCO RLE (``rle_offsets`` / ``rle_counts``), plus
``size`` = (H, W) and ``num_frames``.

Exit codes: 0 ok, 2 bad input, 3 checkpoint not accessible (gated repo / not
logged in / offline), 4 sam3 not importable, 1 anything else.  On failure the
last stderr line is ``HMB_SAM3_ERROR {json}`` with ``kind`` and ``message``.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import time

import numpy as np

HF_REPO = "facebook/sam3.1"
HF_FILE = "sam3.1_multiplex.pt"
# exit codes (see the module docstring); the backend maps them to its error classes
EXIT_INPUT, EXIT_CHECKPOINT, EXIT_IMPORT = 2, 3, 4


class WorkerError(Exception):
    """An expected failure: reported as ``HMB_SAM3_ERROR {kind, message}`` with exit code ``code``."""

    def __init__(self, kind: str, message: str, code: int = 1):
        super().__init__(message)
        self.kind, self.message, self.code = kind, message, code


def log(msg: str) -> None:
    """Progress line on stderr (relayed to the console or the tracking log by the backend)."""
    print(f"[sam3.1] {msg}", file=sys.stderr, flush=True)


def rle_encode(mask: np.ndarray) -> np.ndarray:
    """Uncompressed COCO RLE (column-major, first run = background); see hmb.tracking.tracks."""
    flat = np.asarray(mask, dtype=bool).ravel(order="F")
    if flat.size == 0:
        return np.zeros(0, dtype=np.uint32)
    change = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    counts = np.diff(np.concatenate(([0], change, [flat.size])))
    if flat[0]:
        counts = np.concatenate(([0], counts))
    return counts.astype(np.uint32)


def tight_box(mask: np.ndarray):
    """``[x0, y0, x1, y1]`` of a boolean mask (exclusive max), ``None`` when it is empty."""
    cols = np.flatnonzero(mask.any(axis=0))
    if cols.size == 0:
        return None
    rows = np.flatnonzero(mask.any(axis=1))
    return [float(cols[0]), float(rows[0]), float(cols[-1] + 1), float(rows[-1] + 1)]


# ----------------------------------------------------------------------------- checkpoint
# Where install.sh --with-sam3 puts the checkpoint (hmb/model_paths.py; this worker may run in another
# interpreter, so the path is derived from this file rather than imported).
LOCAL_CHECKPOINT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                "outputs", "assets", "sam3", HF_FILE)


def resolve_checkpoint(explicit):
    """Local checkpoint path: --checkpoint, $HMB_SAM3_CHECKPOINT, the installer's copy in
    outputs/assets/sam3/, else the Hugging Face hub cache/download."""
    path = explicit or os.environ.get("HMB_SAM3_CHECKPOINT")
    if path:
        if not os.path.isfile(path):
            raise WorkerError("checkpoint", f"SAM 3.1 checkpoint not found: {path}", EXIT_CHECKPOINT)
        return path
    local = os.environ.get("HMB_SAM3_LOCAL_CHECKPOINT", LOCAL_CHECKPOINT)   # tests point it elsewhere
    if os.path.isfile(local):
        return local
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise WorkerError("import", f"huggingface_hub is not installed in this environment ({exc})",
                          EXIT_IMPORT)
    try:
        return hf_hub_download(repo_id=HF_REPO, filename=HF_FILE)
    except Exception as exc:  # classify by name: the class layout differs between hub versions
        names = {cls.__name__ for cls in type(exc).__mro__}
        status = getattr(getattr(exc, "response", None), "status_code", None)
        text = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
        if names & {"GatedRepoError", "RepositoryNotFoundError"} or status in (401, 403):
            reason = "access to the gated repository was refused (no access granted yet, or not logged in)"
        elif names & {"LocalEntryNotFoundError", "OfflineModeIsEnabled"}:
            reason = "it is not in the local Hugging Face cache and could not be downloaded (offline?)"
        else:
            reason = f"download failed ({type(exc).__name__})"
        raise WorkerError("checkpoint", f"Cannot get {HF_FILE} from https://huggingface.co/{HF_REPO}: "
                                        f"{reason}. Details: {text}", EXIT_CHECKPOINT)


# ----------------------------------------------------------------------------- model
def use_flash_attention_3(torch) -> bool:
    """FlashAttention 3 is Hopper-only and optional; fall back to PyTorch SDPA elsewhere."""
    if os.environ.get("HMB_SAM3_FA3", "").lower() in ("0", "false", "no"):
        return False
    try:
        import flash_attn_interface  # noqa: F401
    except Exception:
        return False
    return torch.cuda.get_device_capability()[0] >= 9


def build_predictor(checkpoint: str, max_objects: int, compile_model: bool):
    """The SAM 3.1 Object Multiplex video predictor -> ``(predictor, uses_flash_attention_3)``."""
    import torch
    from sam3.model_builder import build_sam3_multiplex_video_predictor

    fa3 = use_flash_attention_3(torch)
    captured = io.StringIO()
    # the builder prints the (expected) partial key matches of its two-stage load; keep them out of the console
    with contextlib.redirect_stdout(captured):
        predictor = build_sam3_multiplex_video_predictor(
            checkpoint_path=checkpoint, max_num_objects=int(max_objects), use_fa3=fa3,
            compile=bool(compile_model), warm_up=bool(compile_model))
    return predictor, fa3


def start_session(predictor, video: str, offload_video_to_cpu: bool) -> str:
    """``start_session`` request, tolerating predictors whose model rejects the newer kwargs."""
    request = dict(type="start_session", resource_path=video, offload_video_to_cpu=offload_video_to_cpu)
    try:
        return predictor.handle_request(request)["session_id"]
    except TypeError as exc:
        if "unexpected keyword argument" not in str(exc):
            raise
    # some sam3 revisions pass offload_state_to_cpu, which the multiplex model's init_state lacks:
    # start the session the way Sam3BasePredictor.start_session does, with the supported kwargs only.
    import inspect
    import uuid

    import torch

    params = inspect.signature(predictor.model.init_state).parameters
    kwargs = dict(resource_path=video, offload_video_to_cpu=offload_video_to_cpu,
                  async_loading_frames=getattr(predictor, "async_loading_frames", False))
    kwargs = {k: v for k, v in kwargs.items() if k in params}
    with torch.inference_mode():
        state = predictor.model.init_state(**kwargs)
    session_id = str(uuid.uuid4())
    now = time.time()
    predictor._all_inference_states[session_id] = {"state": state, "session_id": session_id,
                                                   "start_time": now, "last_use_time": now}
    return session_id


def count_frames(video: str) -> tuple:
    """``(num_frames, fps, width, height)``; frames are counted by reading the video (container counts can be wrong)."""
    import cv2

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise WorkerError("input", f"cannot open video {video}", EXIT_INPUT)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    if n == 0:
        raise WorkerError("input", f"video {video} has no decodable frames", EXIT_INPUT)
    return n, fps, width, height


# ----------------------------------------------------------------------------- chunking
def chunk_spans(num_frames: int, chunk: int, overlap: int) -> list:
    """``[(start, end), ...]`` covering ``num_frames``; consecutive spans share ``overlap`` frames."""
    if chunk <= 0 or num_frames <= chunk:
        return [(0, num_frames)]
    overlap = max(1, min(int(overlap), chunk // 2))
    spans, start = [], 0
    while True:
        end = min(num_frames, start + chunk)
        spans.append((start, end))
        if end >= num_frames:
            return spans
        start = end - overlap


def match_ids(prev: dict, head: dict, min_iou: float = 0.5) -> dict:
    """Map this chunk's ids to the previous chunk's by mask IoU over their shared frames.

    ``prev`` / ``head`` = ``{frame: {id: mask}}``; greedy, best IoU first, one-to-one.
    """
    inter, union = {}, {}
    for f in set(prev) & set(head):
        for g, mg in prev[f].items():
            for l, ml in head[f].items():
                i = int(np.logical_and(mg, ml).sum())
                if i:
                    inter[(g, l)] = inter.get((g, l), 0) + i
                    union[(g, l)] = union.get((g, l), 0) + int(np.logical_or(mg, ml).sum())
    mapping, used = {}, set()
    for iou, (g, l) in sorted(((inter[k] / union[k], k) for k in inter), reverse=True):
        if iou < min_iou:
            break
        if l in mapping or g in used:
            continue
        mapping[l] = g
        used.add(g)
    return mapping


class FrameReader:
    """Sequential RGB PIL frames of a video for chunked sessions (keeps only what the next chunk needs)."""

    def __init__(self, path: str):
        import cv2

        self.cv2 = cv2
        self.cap = cv2.VideoCapture(path)
        self.next = 0
        self.cache = {}

    def get(self, start: int, end: int) -> list:
        """Frames ``start .. end - 1`` as RGB PIL images (frames before ``start`` are dropped)."""
        from PIL import Image

        for i in [i for i in self.cache if i < start]:
            del self.cache[i]
        while self.next < end:
            ok, frame = self.cap.read()
            if not ok:
                break
            self.cache[self.next] = Image.fromarray(self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB))
            self.next += 1
        return [self.cache[i] for i in range(start, end) if i in self.cache]


# ----------------------------------------------------------------------------- main
def silence_progress_bars() -> None:
    """sam3 draws tqdm bars (with explicit ``disable=False``); this worker reports progress itself."""
    if os.environ.get("HMB_SAM3_TQDM", "") == "1":
        return
    try:
        import tqdm
    except ImportError:
        return
    original = tqdm.tqdm.__init__

    def quiet_init(self, *a, **k):
        k["disable"] = True
        original(self, *a, **k)

    tqdm.tqdm.__init__ = quiet_init


def run(args) -> dict:
    """Track the video and write ``--out`` (see the module docstring); returns the metadata."""
    video = os.path.abspath(args.video)
    if not os.path.isfile(video):
        raise WorkerError("input", f"video not found: {video}", EXIT_INPUT)
    os.environ.setdefault("LOG_LEVEL", "ERROR")     # sam3's own loggers (repeat warnings every frame)
    silence_progress_bars()
    import warnings

    # library deprecation chatter (timm.models.layers, ...) and torch's kernel-cache notice
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", message=".*kernel cache directory could not be created.*")
    try:
        import torch
        import sam3  # noqa: F401
    except Exception as exc:
        raise WorkerError("import", f"cannot import torch/sam3 in {sys.executable}: {exc}", EXIT_IMPORT)
    if not torch.cuda.is_available():
        raise WorkerError("cuda", "SAM 3.1 needs a CUDA GPU, but torch.cuda.is_available() is False")

    num_frames, fps, width, height = count_frames(video)
    log(f"{os.path.basename(video)}: {num_frames} frames {width}x{height} @ {fps:.3f} fps")
    checkpoint = resolve_checkpoint(args.checkpoint)
    t0 = time.time()
    predictor, fa3 = build_predictor(checkpoint, args.max_objects, args.compile)
    log(f"model ready in {time.time() - t0:.0f}s (FlashAttention3={'on' if fa3 else 'off'})")

    spans = chunk_spans(num_frames, args.chunk_frames, args.chunk_overlap)
    reader = FrameReader(video) if len(spans) > 1 else None
    if reader is not None:
        log(f"long video: {len(spans)} chunks of <= {args.chunk_frames} frames, {args.chunk_overlap}-frame overlaps "
            f"(ids carried across chunks by mask overlap)")
    t0 = time.time()
    # records of the whole video (frame, video-level id, box, score, RLE)
    frames, ids, boxes, scores, counts = [], [], [], [], []
    seen = set()
    # prev_tail: low-resolution masks of the previous chunk's last ``overlap`` frames, to match ids
    next_id, prev_tail, prev_end = 0, {}, 0
    last_report = time.time()
    for k, (start, end) in enumerate(spans):
        # one SAM 3.1 session per chunk: prompt its first frame with the text, propagate forward
        length = end - start
        offload = args.offload_video == "yes" or (args.offload_video == "auto" and length > args.offload_after)
        resource = video if reader is None else reader.get(start, end)
        session = start_session(predictor, resource, offload)
        resp = predictor.handle_request(dict(type="add_prompt", session_id=session, frame_index=0, text=args.prompt))
        first = resp.get("outputs") or {}
        if k == 0:
            log(f"prompt '{args.prompt}' on frame 0: {len(first.get('out_obj_ids', []))} object(s)")
        tail_from = spans[k + 1][0] if k + 1 < len(spans) else end
        head, tail, local = {}, {}, []
        session_ids, cap_logged = set(), False
        request = dict(type="propagate_in_video", session_id=session, propagation_direction="forward",
                       start_frame_index=0, evict_cached_frame_outputs=True)
        for item in predictor.handle_stream_request(request):
            f = start + int(item["frame_index"])
            out = item["outputs"] or {}
            seen.add(f)
            obj_ids = np.asarray(out.get("out_obj_ids", []), dtype=np.int64).reshape(-1)
            masks = np.asarray(out.get("out_binary_masks", np.zeros((0, height, width), bool)))
            probs = np.asarray(out.get("out_probs", np.ones(len(obj_ids))), dtype=np.float32).reshape(-1)
            session_ids.update(int(i) for i in obj_ids)
            if len(session_ids) >= args.max_objects and not cap_logged:
                log(f"warning: {len(session_ids)} objects reached the SAM 3.1 limit (--max-objects "
                    f"{args.max_objects}) at frame {f}; people appearing later in this session may be missed")
                cap_logged = True
            for j, oid in enumerate(obj_ids):
                m = masks[j].astype(bool)
                if m.shape != (height, width):
                    raise WorkerError("output", f"mask shape {m.shape} != video size {(height, width)}")
                box = tight_box(m)
                if box is None:
                    continue
                if f < prev_end:          # covered by the previous chunk: only used to match ids
                    head.setdefault(f, {})[int(oid)] = m[::4, ::4]
                    continue
                if f >= tail_from:        # also covered by the next chunk
                    tail.setdefault(f, {})[int(oid)] = m[::4, ::4]
                local.append((f, int(oid), box, float(probs[j]) if j < len(probs) else 1.0, rle_encode(m)))
            if time.time() - last_report > 10:
                log(f"tracked {len(seen)}/{num_frames} frames")
                last_report = time.time()
        with contextlib.suppress(Exception):
            predictor.handle_request(dict(type="close_session", session_id=session))
        if reader is not None:
            torch.cuda.empty_cache()
        # per-chunk ids -> video ids
        matched = match_ids(prev_tail, head) if k else {}
        local_ids = sorted({r[1] for r in local} | {l for d in head.values() for l in d}
                           | {l for d in tail.values() for l in d})
        if k == 0:
            mapping = {l: l for l in local_ids}
            next_id = max(local_ids, default=-1) + 1
        else:
            mapping = dict(matched)
            for l in local_ids:
                if l not in mapping:
                    mapping[l] = next_id
                    next_id += 1
        for f, l, box, score, rle in local:
            frames.append(f)
            ids.append(mapping[l])
            boxes.append(box)
            scores.append(score)
            counts.append(rle)
        prev_tail = {f: {mapping[l]: m for l, m in d.items()} for f, d in tail.items()}
        prev_end = end
        if len(spans) > 1:
            log(f"chunk {k + 1}/{len(spans)} (frames {start}-{end - 1}): {len(local_ids)} object(s)"
                + (f", {len(matched)} continued from the previous chunk" if k else ""))
    elapsed = time.time() - t0
    log(f"done: {len(seen)} frames, {len(set(ids))} id(s), {elapsed:.0f}s "
        f"({len(seen) / max(elapsed, 1e-6):.1f} fps), peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")

    # write the records (same layout as hmb.tracking.tracks masks.npz) and the metadata
    lengths = np.array([len(c) for c in counts], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)
    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".part.npz"
    np.savez_compressed(
        tmp, frame=np.asarray(frames, np.int32), obj_id=np.asarray(ids, np.int32),
        box=np.asarray(boxes, np.float32).reshape(-1, 4), score=np.asarray(scores, np.float32),
        rle_offsets=offsets, rle_counts=(np.concatenate(counts) if counts else np.zeros(0)).astype(np.uint32),
        size=np.array([height, width], np.int64), num_frames=np.array(num_frames, np.int64),
        fps=np.array(fps, np.float64), frames_seen=np.array(len(seen), np.int64))
    os.replace(tmp, out_path)
    meta = {"model": f"{HF_REPO}/{HF_FILE}", "checkpoint": checkpoint, "prompt": args.prompt,
            "max_objects": int(args.max_objects), "flash_attention_3": fa3,
            "chunks": [list(span) for span in spans], "seconds": round(elapsed, 2),
            "frames_seen": len(seen), "torch": torch.__version__}
    with open(out_path + ".json", "w") as fh:
        json.dump(meta, fh)
    return meta


def parse_args(argv=None):
    """Worker options (set by :func:`hmb.tracking.sam3_backend.track_sam3`)."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("video")
    p.add_argument("--out", required=True, help="output .npz")
    p.add_argument("--prompt", default="person")
    p.add_argument("--max-objects", type=int, default=16)
    p.add_argument("--checkpoint", default=None, help="local sam3.1_multiplex.pt (default: Hugging Face)")
    p.add_argument("--offload-video", choices=("auto", "yes", "no"), default="auto",
                   help="keep decoded frames on the CPU (auto: when longer than --offload-after frames)")
    p.add_argument("--offload-after", type=int, default=1500)
    p.add_argument("--chunk-frames", type=int, default=600,
                   help="track long videos in chunks of this many frames (GPU memory grows with the "
                        "session length, ~25 MB per frame at 720p); 0 = one session")
    p.add_argument("--chunk-overlap", type=int, default=16, help="frames shared by consecutive chunks")
    p.add_argument("--compile", action="store_true", help="torch.compile the model (slow start, faster tracking)")
    p.add_argument("--check", action="store_true", help="only check imports, CUDA and checkpoint access")
    return p.parse_args(argv)


def check(args) -> dict:
    """``--check``: imports, CUDA and checkpoint access, without loading the model."""
    try:
        import torch
        import sam3  # noqa: F401
    except Exception as exc:
        raise WorkerError("import", f"cannot import torch/sam3 in {sys.executable}: {exc}", EXIT_IMPORT)
    checkpoint = resolve_checkpoint(args.checkpoint)
    return {"python": sys.executable, "torch": torch.__version__, "cuda": torch.cuda.is_available(),
            "checkpoint": checkpoint}


def main(argv=None) -> int:
    """Run the worker; returns the exit code."""
    args = parse_args(argv)
    try:
        if args.check:
            print(json.dumps(check(args)), flush=True)
        else:
            run(args)
        return 0
    except WorkerError as exc:
        print("HMB_SAM3_ERROR " + json.dumps({"kind": exc.kind, "message": exc.message}), file=sys.stderr,
              flush=True)
        return exc.code
    except Exception as exc:  # surface unexpected failures in the same machine-readable way
        import traceback

        traceback.print_exc()
        print("HMB_SAM3_ERROR " + json.dumps({"kind": "runtime", "message": f"{type(exc).__name__}: {exc}"}),
              file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
