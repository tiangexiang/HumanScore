"""Tests of hmb/gvhmr_prepare.py: checkpoint download and loading, SMPL-X linking, video staging
and the shared decode of the pose and feature stages (no GPU, no GVHMR checkpoints needed)."""

import io
import json
from pathlib import Path

import pytest

from hmb import gvhmr_prepare as adapter


def test_download_into_checkpoint_directory_does_not_create_self_symlink(tmp_path, monkeypatch):
    root = tmp_path / "gvhmr"
    assets = root / "inputs/checkpoints"
    model = tmp_path / "models"
    model.mkdir()
    (model / "SMPLX_NEUTRAL.npz").write_bytes(b"body")
    monkeypatch.setattr(adapter, "CHECKPOINTS", {"network/weights.pt": (3, None)})
    monkeypatch.setattr(adapter.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"abc"))
    adapter.prepare_assets(root, assets, model)
    checkpoint = assets / "network/weights.pt"
    assert checkpoint.read_bytes() == b"abc"
    assert not checkpoint.is_symlink()


def test_requested_body_model_replaces_previous_target(tmp_path, monkeypatch):
    monkeypatch.setattr(adapter, "CHECKPOINTS", {})
    root = tmp_path / "gvhmr"
    target = root / "inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"old")
    models = tmp_path / "new_models"
    models.mkdir()
    source = models / "SMPLX_NEUTRAL.npz"
    source.write_bytes(b"new")
    adapter.prepare_assets(root, tmp_path / "assets", models)
    assert target.resolve() == source
    assert target.read_bytes() == b"new"


def test_video_alias_stem_is_preserved_and_original_source_fingerprinted(tmp_path):
    source = tmp_path / "original name.mp4"
    source.write_bytes(b"video")
    alias = tmp_path / "original_name.mp4"
    alias.symlink_to(source)
    session, prepared, fingerprint = adapter.prepare_video_input(alias, tmp_path / "outputs", True, None)
    assert session.name == "original_name"
    assert prepared.read_bytes() == b"video"
    assert fingerprint["source"] == str(source)


def test_unknown_copied_video_is_replaced_but_unknown_preprocessing_rejected(tmp_path):
    source = tmp_path / "video.mp4"
    source.write_bytes(b"correct")
    output = tmp_path / "outputs"
    session = output / "video"
    session.mkdir(parents=True)
    prepared = session / "0_input_video.mp4"
    prepared.write_bytes(b"stale")
    _, prepared, fingerprint = adapter.prepare_video_input(source, output, True, None)
    assert prepared.read_bytes() == b"correct"
    (session / "preprocess/bbx.pt").write_bytes(b"unverified")
    with pytest.raises(ValueError, match="Unverified"):
        adapter.prepare_video_input(source, output, True, None)
    (session / "reconstruction.json").write_text(json.dumps({"input": fingerprint}))
    adapter.prepare_video_input(source, output, True, None)
    with pytest.raises(ValueError, match="options changed"):
        adapter.prepare_video_input(source, output, False, None)
    prepared.write_bytes(b"changed")
    with pytest.raises(ValueError, match="video changed"):
        adapter.prepare_video_input(source, output, True, None)


def test_pose_and_features_share_one_decode_and_crop():
    calls = []
    expected_crops, expected_boxes = object(), object()

    def get_batch(path, boxes):
        calls.append((path, boxes))
        return expected_crops, expected_boxes

    boxes = object()
    crops, crop_boxes = adapter.prepare_shared_crops(Path("clip.mp4"), boxes, True, True, get_batch)
    assert crops is expected_crops and crop_boxes is expected_boxes
    assert calls == [("clip.mp4", boxes)]
    assert adapter.prepare_shared_crops(Path("clip.mp4"), boxes, False, False, get_batch) == (None, boxes)
    assert len(calls) == 1


def test_checkpoint_mapping_is_scoped_to_verified_assets(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    calls = []
    original = lambda path, *args, **kwargs: calls.append((path, kwargs))
    fake_torch = SimpleNamespace(load=original)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    name = next(iter(adapter.CHECKPOINTS))
    with adapter._trusted_legacy_checkpoints(tmp_path):
        fake_torch.load(tmp_path / name, map_location="cpu")
        fake_torch.load(tmp_path / "unrelated.pt")
    assert calls[0][1] == {"map_location": "cpu", "weights_only": False, "mmap": True}
    assert calls[1][1] == {}
    assert fake_torch.load is original
