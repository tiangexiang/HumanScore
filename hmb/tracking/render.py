"""Videos made from tracks: one masked video per person, and a colour preview of all ids.

Person videos keep the full frame, the source frame rate, frame count and
resolution (so camera intrinsics, timestamps and box coordinates stay those of
the source); only the pixels of the *other* tracked people, dilated by a few
pixels, are painted flat gray (114, the letterbox gray of YOLO), and the
target's own mask is always kept.  Videos are H.264 MP4s written by an
``ffmpeg`` subprocess (``libx264``), falling back to OpenCV's writer when no
ffmpeg binary is available.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .tracks import Tracks

#: Gray level that replaces the other people (YOLO's letterbox gray).
FILL_GRAY = 114


# ----------------------------------------------------------------------------- IO helpers
def iter_video(video) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield ``(index, BGR frame)`` for every decodable frame (OpenCV order, like the trackers)."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video {video}")
    index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield index, frame
            index += 1
    finally:
        cap.release()


def video_info(video) -> dict:
    """``{fps, width, height, num_frames}``; frames are counted by reading the video (container counts can be wrong)."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video {video}")
    try:
        info = {"fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
        count = 0
        while cap.grab():
            count += 1
        info["num_frames"] = count
    finally:
        cap.release()
    if not np.isfinite(info["fps"]) or info["fps"] <= 0:
        raise ValueError(f"Video {video} reports an invalid frame rate {info['fps']}")
    if info["num_frames"] == 0:
        raise ValueError(f"Video {video} has no decodable frames")
    return info


def find_ffmpeg() -> Optional[str]:
    """``$HMB_FFMPEG``, ``ffmpeg`` on the PATH or the imageio-ffmpeg binary; ``None`` when there is none."""
    exe = os.environ.get("HMB_FFMPEG") or shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg  # type: ignore

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _fps_arg(fps: float) -> str:
    """Frame rate for ffmpeg's ``-r``: a fraction (e.g. ``30000/1001``) when one matches exactly, else a decimal."""
    frac = Fraction(float(fps)).limit_denominator(1001)
    if abs(float(frac) - fps) > 1e-6 * fps:
        return f"{fps:.9g}"
    return f"{frac.numerator}/{frac.denominator}"


class VideoWriter:
    """Write BGR frames to an H.264 MP4 (ffmpeg pipe; OpenCV fallback)."""

    def __init__(self, path, fps: float, width: int, height: int, crf: int = 16):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.tmp = self.path.with_name(self.path.stem + ".part" + self.path.suffix)
        self.width, self.height = int(width), int(height)
        self.frames = 0
        self.proc = None
        self.cv_writer = None
        exe = find_ffmpeg()
        if exe:
            # yuv420p needs even sizes; keep the exact resolution with 4:4:4 otherwise
            pix_fmt = "yuv420p" if self.width % 2 == 0 and self.height % 2 == 0 else "yuv444p"
            cmd = [exe, "-hide_banner", "-loglevel", "error", "-y",
                   "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{self.width}x{self.height}",
                   "-r", _fps_arg(fps), "-i", "-", "-an",
                   "-c:v", "libx264", "-preset", "medium", "-crf", str(int(crf)), "-pix_fmt", pix_fmt,
                   "-movflags", "+faststart", str(self.tmp)]
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        else:
            import cv2

            for fourcc in ("avc1", "mp4v"):
                writer = cv2.VideoWriter(str(self.tmp), cv2.VideoWriter_fourcc(*fourcc), float(fps),
                                         (self.width, self.height))
                if writer.isOpened():
                    self.cv_writer = writer
                    break
            if self.cv_writer is None:
                raise RuntimeError("No video encoder available: install ffmpeg (or imageio-ffmpeg)")

    def write(self, frame: np.ndarray) -> None:
        """Append one BGR frame of the writer's size."""
        if frame.shape[:2] != (self.height, self.width):
            raise ValueError(f"frame {frame.shape[:2]} != writer size {(self.height, self.width)}")
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        if self.proc is not None:
            try:
                self.proc.stdin.write(frame.tobytes())
            except BrokenPipeError as exc:
                err = self.proc.stderr.read().decode(errors="replace")
                raise RuntimeError(f"ffmpeg failed writing {self.path}: {err.strip()}") from exc
        else:
            self.cv_writer.write(frame)
        self.frames += 1

    def close(self) -> Path:
        """Finish the file and move it into place (the video is written to a ``.part`` file first)."""
        if self.proc is not None:
            self.proc.stdin.close()
            err = self.proc.stderr.read().decode(errors="replace")
            code = self.proc.wait()
            if code != 0:
                raise RuntimeError(f"ffmpeg exited with code {code} writing {self.path}: {err.strip()}")
        elif self.cv_writer is not None:
            self.cv_writer.release()
        os.replace(self.tmp, self.path)
        return self.path

    def abort(self) -> None:
        """Stop the encoder and delete the partial file."""
        if self.proc is not None:
            try:
                self.proc.stdin.close()
            except Exception:
                pass
            self.proc.kill()
            self.proc.wait()
        elif self.cv_writer is not None:
            self.cv_writer.release()
        if self.tmp.exists():
            self.tmp.unlink()

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()


# ----------------------------------------------------------------------------- person video
def default_dilation(width: int, height: int) -> int:
    """Dilation radius of the other people's masks: 1% of the longer side, at least 3 px."""
    return max(3, int(round(0.01 * max(width, height))))


def _disk(radius: int) -> np.ndarray:
    """Disk-shaped structuring element for :func:`cv2.dilate`."""
    import cv2

    radius = max(0, int(radius))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def person_frame(frame: np.ndarray, target: Optional[np.ndarray], others: Sequence[np.ndarray],
                 dilate_px: int, fill: int = FILL_GRAY) -> np.ndarray:
    """One frame of a person video: dilated ``others`` masks filled gray, ``target`` pixels kept.

    The dilation also covers the soft mask borders (hair, motion blur) of the other
    people, so no fragment of them is left for the keypoint detector to pick up.
    """
    import cv2

    if not others:
        return frame
    union = np.zeros(frame.shape[:2], dtype=np.uint8)
    for m in others:
        union |= m.astype(np.uint8)
    if dilate_px > 0:
        union = cv2.dilate(union, _disk(dilate_px))
    paint = union.astype(bool)
    if target is not None:
        paint &= ~target
    out = frame.copy()
    out[paint] = fill
    return out


def write_person_video(video, tracks: Tracks, person_id: int, out_path, *,
                       dilate_px: Optional[int] = None, fill: int = FILL_GRAY,
                       crf: int = 16) -> Path:
    """Write the source video with every tracked person except ``person_id`` masked out.

    Full frame, same fps / resolution / frame count as ``video``; the other
    tracked people's masks, dilated by ``dilate_px`` (default 1% of the longer
    side), are filled with flat gray ``fill``; the target's own mask pixels are
    always kept.  Returns ``out_path``.
    """
    person_id = int(person_id)
    if person_id not in tracks.ids:
        raise KeyError(f"person {person_id} is not in the tracks (ids: {tracks.ids})")
    dilate = default_dilation(tracks.width, tracks.height) if dilate_px is None else int(dilate_px)
    groups = tracks.by_frame()
    writer = VideoWriter(out_path, tracks.fps, tracks.width, tracks.height, crf=crf)
    with writer:
        for index, frame in iter_video(video):
            if frame.shape[:2] != (tracks.height, tracks.width):
                raise ValueError(f"{video} frame size {frame.shape[:2]} differs from the tracks "
                                 f"({tracks.height}, {tracks.width})")
            target, others = None, []
            if index < tracks.num_frames:
                for r in groups[index]:
                    m = tracks.decode(int(r))
                    if int(tracks.obj_id[r]) == person_id:
                        target = m if target is None else (target | m)
                    else:
                        others.append(m)
            writer.write(person_frame(frame, target, others, dilate, fill))
    return Path(out_path)


# ----------------------------------------------------------------------------- preview
def id_color(obj_id: int) -> Tuple[int, int, int]:
    """Stable, well separated BGR colour per id (golden-ratio hue walk)."""
    import colorsys

    hue = (0.13 + 0.61803398875 * int(obj_id)) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def draw_tracks(frame: np.ndarray, items: Iterable[Tuple[int, np.ndarray, np.ndarray]],
                selected: Optional[set] = None, alpha: float = 0.45) -> np.ndarray:
    """Overlay masks (alpha), boxes and id labels; ``items`` = (id, mask, box)."""
    import cv2

    out = frame.copy()
    overlay = frame.copy()
    items = list(items)
    for oid, mask, _ in items:
        if selected is not None and oid not in selected:
            continue
        overlay[mask] = id_color(oid)
    out = cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)
    scale = max(0.5, min(out.shape[:2]) / 720.0)
    thick = max(1, int(round(2 * scale)))
    for oid, mask, box in items:
        color = id_color(oid) if selected is None or oid in selected else (160, 160, 160)
        x0, y0, x1, y1 = [int(round(v)) for v in box]
        cv2.rectangle(out, (x0, y0), (x1 - 1, y1 - 1), color, thick)
        label = f"P{oid}" if selected is None or oid in selected else f"P{oid} (skip)"
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7 * scale, thick)
        ty = max(th + 4, y0 - 4)
        cv2.rectangle(out, (x0, ty - th - 4), (x0 + tw + 6, ty + base - 2), color, -1)
        cv2.putText(out, label, (x0 + 3, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * scale, (0, 0, 0), thick,
                    cv2.LINE_AA)
    return out


def write_preview(video, tracks: Tracks, out_path, *, selected: Optional[Sequence[int]] = None,
                  max_side: int = 960, sheet_path=None, sheet_frames: int = 6) -> Path:
    """Colour preview of all tracks (mask + box + ``P<id>``), optionally a contact sheet JPEG.

    ``selected`` ids are drawn in colour, the others in gray marked "(skip)".
    The preview is downscaled to ``max_side`` pixels.
    """
    import cv2

    chosen = None if selected is None else {int(i) for i in selected}
    scale = min(1.0, float(max_side) / max(tracks.width, tracks.height)) if max_side else 1.0
    w = int(round(tracks.width * scale / 2) * 2)
    h = int(round(tracks.height * scale / 2) * 2)
    groups = tracks.by_frame()
    sheet_at = set()
    if sheet_path is not None and tracks.num_frames:
        sheet_at = {int(round(x)) for x in np.linspace(0, tracks.num_frames - 1, sheet_frames)}
    sheet: List[np.ndarray] = []
    with VideoWriter(out_path, tracks.fps, w, h, crf=23) as writer:
        for index, frame in iter_video(video):
            items = []
            if index < tracks.num_frames:
                for r in groups[index]:
                    items.append((int(tracks.obj_id[r]), tracks.decode(int(r)), tracks.box[r]))
            drawn = draw_tracks(frame, items, chosen)
            cv2.putText(drawn, f"frame {index}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2,
                        cv2.LINE_AA)
            small = cv2.resize(drawn, (w, h), interpolation=cv2.INTER_AREA) if (w, h) != drawn.shape[1::-1] \
                else drawn
            writer.write(small)
            if index in sheet_at:
                sheet.append(small)
    if sheet_path is not None and sheet:
        cols = 3 if len(sheet) > 4 else 2
        rows = int(np.ceil(len(sheet) / cols))
        blank = np.zeros_like(sheet[0])
        tiles = sheet + [blank] * (rows * cols - len(sheet))
        grid = np.vstack([np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(rows)])
        Path(sheet_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(sheet_path), grid, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return Path(out_path)
