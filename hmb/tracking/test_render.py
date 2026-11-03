"""Person videos and previews (hmb.tracking.render) and the ``python -m hmb.tracking`` command on a synthetic video."""
import json

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from hmb.tracking import __main__ as tracking_cli  # noqa: E402
from hmb.tracking.render import (FILL_GRAY, VideoWriter, default_dilation, person_frame, video_info,  # noqa: E402
                                 write_person_video, write_preview)
from hmb.tracking.tracks import TrackRecorder  # noqa: E402

H, W, N, FPS = 96, 128, 12, 12.0
BG = (40, 160, 40)        # BGR green background
A_COLOR = (200, 60, 60)   # person A (target)
B_COLOR = (60, 60, 200)   # person B (other)


def a_box(f):
    return (10 + f, 20, 40 + f, 90)


def b_box(f):
    return (75, 5, 115, 85)


def synthetic(tmp_path):
    """A video with two flat-coloured 'people' and their exact tracks."""
    path = tmp_path / "two_people.mp4"
    rec = TrackRecorder(W, H)
    with VideoWriter(path, FPS, W, H, crf=0) as writer:
        for f in range(N):
            frame = np.empty((H, W, 3), np.uint8)
            frame[:] = BG
            for oid, box, color in ((1, a_box(f), A_COLOR), (2, b_box(f), B_COLOR)):
                x0, y0, x1, y1 = box
                frame[y0:y1, x0:x1] = color
                m = np.zeros((H, W), bool)
                m[y0:y1, x0:x1] = True
                if oid == 2 and f >= N - 2:
                    continue   # B leaves the tracks for the last two frames
                rec.add(f, oid, m)
            writer.write(frame)
    tracks = rec.build(video=str(path), fps=FPS, num_frames=N, backend="test")
    return path, tracks


def test_writer_keeps_size_fps_and_count(tmp_path):
    path, _ = synthetic(tmp_path)
    info = video_info(path)
    assert (info["width"], info["height"], info["num_frames"]) == (W, H, N)
    assert info["fps"] == pytest.approx(FPS, rel=1e-3)


def test_writer_odd_size(tmp_path):
    with VideoWriter(tmp_path / "odd.mp4", 25.0, 33, 17) as writer:
        for _ in range(3):
            writer.write(np.full((17, 33, 3), 90, np.uint8))
    info = video_info(tmp_path / "odd.mp4")
    assert (info["width"], info["height"], info["num_frames"]) == (33, 17, 3)


def test_person_frame_fills_dilated_others_and_keeps_target():
    frame = np.full((20, 20, 3), 7, np.uint8)
    target = np.zeros((20, 20), bool)
    target[5:15, 2:9] = True
    other = np.zeros((20, 20), bool)
    other[5:15, 9:16] = True                       # touches the target
    out = person_frame(frame, target, [other], dilate_px=2)
    assert (out[target] == 7).all()                 # target always kept
    assert (out[other] == FILL_GRAY).all()
    assert (out[10, 17] == FILL_GRAY).all()         # dilation ring
    assert (out[10, 19] == 7).all()                 # beyond the dilation
    assert person_frame(frame, None, [], 3) is frame


def test_write_person_video(tmp_path):
    path, tracks = synthetic(tmp_path)
    out = write_person_video(path, tracks, 1, tmp_path / "person_1.mp4")
    info = video_info(out)
    assert (info["width"], info["height"], info["num_frames"]) == (W, H, N)
    assert info["fps"] == pytest.approx(FPS, rel=1e-3)
    dil = default_dilation(W, H)
    cap = cv2.VideoCapture(str(out))
    for f in range(N):
        ok, frame = cap.read()
        assert ok
        ax0, ay0, ax1, ay1 = a_box(f)
        bx0, by0, bx1, by1 = b_box(f)
        target = frame[ay0 + 2:ay1 - 2, ax0 + 2:ax1 - 2].reshape(-1, 3).mean(0)
        assert np.allclose(target, A_COLOR, atol=12)
        inner_b = frame[by0 + 3:by1 - 3, bx0 + 3:bx1 - 3].reshape(-1, 3).mean(0)
        if f < N - 2:
            assert np.allclose(inner_b, FILL_GRAY, atol=6)                      # other person painted
            assert np.allclose(frame[by0 + 10, bx1 + dil - 1], FILL_GRAY, atol=12)  # dilated ring
        else:
            assert np.allclose(inner_b, B_COLOR, atol=12)                       # untracked: untouched
        assert np.allclose(frame[2, 2], BG, atol=12)                            # background untouched
    cap.release()
    with pytest.raises(KeyError):
        write_person_video(path, tracks, 99, tmp_path / "nobody.mp4")


def test_write_preview_and_sheet(tmp_path):
    path, tracks = synthetic(tmp_path)
    out = write_preview(path, tracks, tmp_path / "preview.mp4", selected=[1], sheet_path=tmp_path / "sheet.jpg")
    info = video_info(out)
    assert info["num_frames"] == N
    assert (tmp_path / "sheet.jpg").stat().st_size > 0


def test_cli_with_stubbed_tracker(tmp_path, monkeypatch):
    path, tracks = synthetic(tmp_path)
    out_dir = tmp_path / "tracks"

    def fake_track_people(video, out, **kwargs):
        assert kwargs["backend"] == "yolo"
        tracks.save(out)
        return tracks

    monkeypatch.setattr(tracking_cli, "track_people", fake_track_people)
    code = tracking_cli.main([str(path), "--out", str(out_dir), "--backend", "yolo", "--min-duration", "0.5",
                              "--min-visible", "0.1", "--person-videos"])
    assert code == 0
    selection = json.loads((out_dir / "selection.json").read_text())
    assert [s["id"] for s in selection["selected"]] == [2, 1]   # B is bigger
    for pid in (1, 2):
        assert video_info(out_dir / f"person_{pid}.mp4")["num_frames"] == N
        boxes = json.loads((out_dir / f"person_{pid}_boxes.json").read_text())
        assert len(boxes["boxes_xyxy"]) == N and len(boxes["timestamps"]) == N
    assert (out_dir / "preview.mp4").exists() and (out_dir / "preview_sheet.jpg").exists()
