"""Track storage, RLE masks, speck removal, fragment stitching and person selection (hmb.tracking.tracks)."""
import json

import numpy as np
import pytest

from hmb.tracking.tracks import (TrackRecorder, Tracks, box_iou, clean_masks, mask_box, remove_specks,
                                 describe_selection, rle_decode, rle_encode, screen_people, select_people,
                                 stitch_fragments)

H, W = 48, 64


def rect(x0, y0, x1, y1, h=H, w=W):
    m = np.zeros((h, w), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def make_tracks(spec, num_frames=30, fps=10.0):
    """spec: {obj_id: {frame: (x0, y0, x1, y1)}}"""
    rec = TrackRecorder(W, H)
    for oid, frames in spec.items():
        for f, box in frames.items():
            rec.add(f, oid, rect(*box), score=0.9)
    return rec.build(video="synthetic.mp4", fps=fps, num_frames=num_frames, backend="test")


@pytest.mark.parametrize("mask", [
    np.zeros((H, W), bool),
    np.ones((H, W), bool),
    rect(0, 0, 1, 1),
    rect(W - 1, H - 1, W, H),
    rect(5, 7, 30, 40),
])
def test_rle_roundtrip_edge_cases(mask):
    counts = rle_encode(mask)
    assert counts.dtype == np.uint32 and int(counts.sum()) == H * W
    assert np.array_equal(rle_decode(counts, H, W), mask)


def test_rle_roundtrip_random_and_coco_layout():
    rng = np.random.default_rng(0)
    for _ in range(20):
        mask = rng.random((H, W)) > 0.7
        assert np.array_equal(rle_decode(rle_encode(mask), H, W), mask)
    # column-major, first run = background (COCO uncompressed RLE)
    m = np.zeros((2, 3), bool)
    m[1, 0] = True          # Fortran order: [0, 1, 0, 0, 0, 0]
    assert rle_encode(m).tolist() == [1, 1, 4]
    m[0, 0] = True          # starts with foreground: leading zero-length background run
    assert rle_encode(m).tolist() == [0, 2, 4]


def test_rle_matches_pycocotools_when_available():
    mask_utils = pytest.importorskip("pycocotools.mask")
    rng = np.random.default_rng(1)
    mask = rng.random((H, W)) > 0.5
    rle = mask_utils.frPyObjects({"size": [H, W], "counts": rle_encode(mask).tolist()}, H, W)
    assert np.array_equal(mask_utils.decode(rle).astype(bool), mask)


def test_rle_decode_rejects_wrong_size():
    with pytest.raises(ValueError):
        rle_decode(np.array([5, 5], np.uint32), H, W)


def test_mask_box_and_iou():
    assert mask_box(np.zeros((H, W), bool)) is None
    assert mask_box(rect(3, 4, 10, 20)).tolist() == [3, 4, 10, 20]
    assert box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)
    assert box_iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1 / 3)
    assert box_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_recorder_access_and_roundtrip(tmp_path):
    spec = {1: {f: (2 + f, 5, 12 + f, 30) for f in range(0, 10)},
            7: {f: (40, 10, 50, 40) for f in range(5, 20)}}
    tracks = make_tracks(spec)
    assert tracks.ids == [1, 7] and len(tracks) == 25
    assert np.all(np.diff(tracks.frame) >= 0)            # sorted by frame
    boxes = tracks.boxes(1)
    assert boxes.shape == (30, 4)
    assert boxes[3].tolist() == [5, 5, 15, 30] and np.isnan(boxes[10]).all()
    assert np.array_equal(tracks.mask(7, 6), rect(40, 10, 50, 40))
    assert tracks.mask(7, 2) is None
    assert set(tracks.masks_at(6)) == {1, 7}
    groups = tracks.by_frame()
    assert len(groups) == 30 and len(groups[6]) == 2 and len(groups[25]) == 0

    tracks.save(tmp_path)
    again = Tracks.load(tmp_path)
    assert again.ids == tracks.ids and again.fps == 10.0 and again.num_frames == 30
    for r in range(len(tracks)):
        assert np.array_equal(again.decode(r), tracks.decode(r))
    info = json.loads((tmp_path / "tracks.json").read_text())
    assert info["people"]["7"]["first_frame"] == 5 and info["people"]["7"]["num_visible_frames"] == 15


def test_recorder_ignores_empty_and_checks_shape():
    rec = TrackRecorder(W, H)
    assert rec.add(0, 1, np.zeros((H, W), bool)) is False
    with pytest.raises(ValueError):
        rec.add(0, 1, np.ones((H + 1, W), bool))
    tracks = rec.build(video="v", fps=24.0, num_frames=3)
    assert len(tracks) == 0 and tracks.ids == []


def test_person_boxes_payload(tmp_path):
    tracks = make_tracks({3: {1: (1, 2, 11, 22), 2: (2, 2, 12, 22)}}, num_frames=4, fps=24.0)
    payload = tracks.person_boxes(3)
    assert payload["timestamps"] == pytest.approx([0, 1 / 24, 2 / 24, 3 / 24])
    assert payload["boxes_xyxy"] == [None, [1, 2, 11, 22], [2, 2, 12, 22], None]
    path = tracks.write_person_boxes(3, tmp_path / "p.json")
    assert json.loads(path.read_text())["boxes_xyxy"][1] == [1, 2, 11, 22]


def test_remove_specks_keeps_body_parts_and_drops_specks():
    body = rect(10, 5, 30, 45)                       # 800 px
    hand = rect(40, 20, 45, 25)                      # 25 px = 3% of the body: kept
    speck = rect(60, 2, 61, 3)                       # 1 px: dropped
    mask = body | hand | speck
    cleaned = remove_specks(mask)
    assert np.array_equal(cleaned, body | hand)
    assert remove_specks(body) is body               # unchanged mask is returned as is


def test_clean_masks_recomputes_boxes():
    rec = TrackRecorder(W, H)
    rec.add(0, 1, rect(10, 5, 30, 45) | rect(60, 2, 61, 3))
    rec.add(1, 1, rect(10, 5, 30, 45))
    tracks = rec.build(video="v", fps=10.0, num_frames=2)
    assert tracks.box[0].tolist() == [10, 2, 61, 45]
    cleaned = clean_masks(tracks)
    assert cleaned.box[0].tolist() == [10, 5, 30, 45]
    assert cleaned.meta["cleaned_masks"] == 1
    assert np.array_equal(cleaned.decode(1), tracks.decode(1))


def test_stitch_joins_restarted_track_only():
    spec = {
        1: {f: (10, 10, 20, 40) for f in range(0, 10)},
        2: {f: (11, 10, 21, 40) for f in range(13, 30)},     # same place after a 3-frame gap: joined
        3: {f: (45, 10, 55, 40) for f in range(13, 30)},     # starts at the same time elsewhere: not joined
        4: {f: (45, 10, 55, 40) for f in range(0, 12)},      # overlaps 3? no: ends at 11, 3 starts at 13 -> joined
    }
    tracks = stitch_fragments(make_tracks(spec), max_gap_seconds=0.5)
    assert tracks.ids == [1, 4]
    assert tracks.frames(1).tolist() == list(range(10)) + list(range(13, 30))
    assert sorted(map(tuple, tracks.meta["stitched"])) == [(1, 2), (4, 3)]


def test_stitch_respects_gap_and_overlap():
    spec = {1: {f: (10, 10, 20, 40) for f in range(0, 10)},
            2: {f: (10, 10, 20, 40) for f in range(25, 30)},     # gap 15 frames > 0.5 s at 10 fps
            3: {f: (10, 10, 20, 40) for f in range(5, 12)}}      # overlaps track 1 in time
    tracks = stitch_fragments(make_tracks(spec), max_gap_seconds=0.5)
    assert tracks.ids == [1, 2, 3]
    assert "stitched" not in tracks.meta


def test_select_people_rules_and_ranking():
    spec = {
        1: {f: (0, 0, 20, 40) for f in range(0, 30)},          # big, whole video
        2: {f: (30, 0, 40, 20) for f in range(0, 30)},         # smaller, whole video
        3: {f: (30, 20, 50, 46) for f in range(0, 8)},         # 0.8 s: too short
        4: {f: (50, 0, 52, 3) for f in range(0, 30)},          # tiny: below min_height
        5: {f: (0, 0, 30, 46) for f in range(20, 30)},         # 1 s, 33% of frames
    }
    tracks = make_tracks(spec)
    chosen = select_people(tracks, min_duration=1.0, min_visible=0.2, max_people=10)
    assert [s["id"] for s in chosen] == [1, 5, 2]
    assert [s["rank"] for s in chosen] == [1, 2, 3]
    assert [s["id"] for s in select_people(tracks, max_people=1)] == [1]
    assert [s["id"] for s in select_people(tracks, min_visible=0.5)] == [1, 2]
    everyone = dict(min_duration=0.0, min_visible=0.0, min_height=0.0, min_relative_height=0.0)
    assert {s["id"] for s in select_people(tracks, **everyone)} == {1, 2, 3, 4, 5}
    # relative height: person 2 (20 px) is under half the tallest kept person (46 px)
    assert [s["id"] for s in select_people(tracks, min_relative_height=0.5)] == [1, 5]


def test_default_min_visible_leaves_room_for_the_bone_length_coverage():
    import inspect

    from hmb.tracking import MIN_VISIBLE, __main__ as tracking_cli
    from hmb.scoring.bone_length import MIN_COVERAGE

    # the bone-length score needs keypoints in 25 % of ALL analysed frames; admission keeps a margin above it
    assert MIN_VISIBLE == 0.3 and MIN_VISIBLE > MIN_COVERAGE
    for function in (select_people, screen_people):
        assert inspect.signature(function).parameters["min_visible"].default == MIN_VISIBLE
    assert tracking_cli.build_parser().parse_args(["clip.mp4"]).min_visible == MIN_VISIBLE
    spec = {1: {f: (0, 0, 20, 40) for f in range(100)},        # whole video
            2: {f: (30, 0, 50, 40) for f in range(27)},         # 2.7 s, 27 % of the frames
            3: {f: (30, 0, 50, 40) for f in range(70, 100)}}    # 3.0 s, 30 %
    selected, skipped = screen_people(make_tracks(spec, num_frames=100))
    assert sorted(s["id"] for s in selected) == [1, 3]
    assert [s["id"] for s in skipped] == [2] and "< min visible 30%" in skipped[0]["reason"]


def test_screen_people_gives_the_reason_for_every_skipped_person():
    spec = {
        1: {f: (0, 0, 20, 40) for f in range(0, 30)},          # big, whole video
        2: {f: (30, 0, 40, 20) for f in range(0, 30)},         # smaller, whole video
        3: {f: (30, 20, 50, 46) for f in range(0, 8)},         # 0.8 s: too short
        4: {f: (50, 0, 52, 3) for f in range(0, 30)},          # tiny: below min_height
        5: {f: (0, 0, 30, 46) for f in range(20, 30)},         # 1 s, 33% of frames
    }
    tracks = make_tracks(spec)
    selected, skipped = screen_people(tracks, min_duration=1.0, min_visible=0.2, max_people=10)
    assert selected == select_people(tracks, min_duration=1.0, min_visible=0.2, max_people=10)
    reasons = {s["id"]: s["reason"] for s in skipped}
    assert sorted(reasons) == [3, 4]
    assert reasons[3].startswith("too short: visible 0.80 s < min duration 1 s")
    assert reasons[4].startswith("too small: median height 6% of the frame < 10%")
    _, skipped = screen_people(tracks, min_visible=0.5)
    assert {s["id"]: s["reason"] for s in skipped}[5].startswith("too short: visible in 33% of the frames")
    _, skipped = screen_people(tracks, min_relative_height=0.5)
    assert "of the tallest person's" in {s["id"]: s["reason"] for s in skipped}[2]
    _, skipped = screen_people(tracks, max_people=1)
    assert sorted(s["id"] for s in skipped) == [2, 3, 4, 5]
    assert {s["id"]: s["reason"] for s in skipped}[5] == "less prominent than the 1 person kept (max people 1)"
    lines = describe_selection(tracks, selected, screen_people(tracks)[1])
    assert lines[2].endswith("-> skipped (too short: visible 0.80 s < min duration 1 s)")
    assert lines[0].endswith("-> selected")
