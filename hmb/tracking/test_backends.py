"""Tracking backends without a GPU: the SAM 3.1 worker against fake torch/sam3/hub modules, and the YOLO backend."""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from hmb.tracking import api, sam3_backend, sam3_worker, yolo_backend
from hmb.tracking.render import VideoWriter
from hmb.tracking.sam3_backend import SAM3CheckpointError, SAM3SetupError, resolve_python, track_sam3, worker_env
from hmb.tracking.tracks import rle_decode, rle_encode

H, W, N = 40, 56, 8

FAKE_TORCH = '''
import contextlib
__version__ = "0.0+fake"
class cuda:
    @staticmethod
    def is_available():
        return True
    @staticmethod
    def get_device_capability():
        return (8, 0)
    @staticmethod
    def max_memory_allocated():
        return 0
    @staticmethod
    def empty_cache():
        pass
def inference_mode():
    return contextlib.nullcontext()
'''

FAKE_HUB = '''
import os
class GatedRepoError(Exception):
    pass
def hf_hub_download(repo_id, filename):
    assert (repo_id, filename) == ("facebook/sam3.1", "sam3.1_multiplex.pt")
    if os.environ.get("FAKE_HF_MODE") == "gated":
        raise GatedRepoError("401 Client Error: access to model facebook/sam3.1 is restricted")
    return "/fake/sam3.1_multiplex.pt"
'''

# Mimics a sam3 revision whose start_session passes a kwarg the multiplex model rejects; per-session object ids
# (session k numbers its objects from 100 * k); a chunk (list of PIL frames) is located in the video by the
# gray level of its first frame (frame f is filled with 20 + 20 f).
FAKE_BUILDER = '''
import numpy as np
H, W, N = %d, %d, %d

class Model:
    def init_state(self, resource_path, offload_video_to_cpu=False, async_loading_frames=False):
        if isinstance(resource_path, list):
            first = np.asarray(resource_path[0], dtype=float).mean()
            return {"offset": int(round((first - 20) / 20)), "frames": len(resource_path)}
        return {"offset": 0, "frames": N}

class Predictor:
    def __init__(self, **kw):
        assert kw["checkpoint_path"] == "/fake/sam3.1_multiplex.pt" and kw["use_fa3"] is False
        self.model = Model()
        self._all_inference_states = {}
    def handle_request(self, request):
        t = request["type"]
        if t == "start_session":
            raise TypeError("init_state() got an unexpected keyword argument 'offload_state_to_cpu'")
        if t == "add_prompt":
            assert request["session_id"] in self._all_inference_states and request["text"] == "person"
            return {"frame_index": 0, "outputs": self.outputs(request["session_id"], 0)}
        return {"is_success": True}
    def outputs(self, sid, local):
        state = self._all_inference_states[sid]["state"]
        base = 100 * list(self._all_inference_states).index(sid)
        f = state["offset"] + local
        ids, masks = [], []
        a = np.zeros((H, W), bool); a[5:30, 2 + f:12 + f] = True
        a[0, W - 1] = True                         # a speck: removed by clean_masks
        ids.append(base + 0); masks.append(a)
        if f >= 3:
            b = np.zeros((H, W), bool); b[10:38, 35:50] = True
            ids.append(base + 4); masks.append(b)
        return {"out_obj_ids": np.array(ids), "out_probs": np.full(len(ids), 0.9, np.float32),
                "out_binary_masks": np.stack(masks), "out_boxes_xywh": np.zeros((len(ids), 4))}
    def handle_stream_request(self, request):
        assert request["type"] == "propagate_in_video"
        sid = request["session_id"]
        for local in range(self._all_inference_states[sid]["state"]["frames"]):
            yield {"frame_index": local, "outputs": self.outputs(sid, local)}

def build_sam3_multiplex_video_predictor(**kw):
    print("Missing keys: [noise that must not reach the console]")
    return Predictor(**kw)
''' % (H, W, N)


@pytest.fixture
def fake_sam3(tmp_path, monkeypatch):
    root = tmp_path / "fake_site"
    (root / "torch").mkdir(parents=True)
    (root / "torch" / "__init__.py").write_text(FAKE_TORCH)
    (root / "huggingface_hub").mkdir()
    (root / "huggingface_hub" / "__init__.py").write_text(FAKE_HUB)
    (root / "sam3").mkdir()
    (root / "sam3" / "__init__.py").write_text("")
    (root / "sam3" / "model_builder.py").write_text(FAKE_BUILDER)
    monkeypatch.setenv("PYTHONPATH", str(root) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    monkeypatch.setenv("HMB_SAM3_PYTHON", sys.executable)
    monkeypatch.delenv("HMB_SAM3_CHECKPOINT", raising=False)
    # the checkpoint that install.sh may have placed in outputs/assets/sam3/ must not be used here
    monkeypatch.setenv("HMB_SAM3_LOCAL_CHECKPOINT", str(tmp_path / "no-local-checkpoint.pt"))
    video = tmp_path / "clip.mp4"
    with VideoWriter(video, 24.0, W, H, crf=0) as writer:
        for f in range(N):
            writer.write(np.full((H, W, 3), 20 + 20 * f, np.uint8))
    return video


def test_worker_rle_matches_project_rle():
    rng = np.random.default_rng(3)
    for _ in range(10):
        mask = rng.random((H, W)) > 0.6
        assert np.array_equal(sam3_worker.rle_encode(mask), rle_encode(mask))
        assert np.array_equal(rle_decode(sam3_worker.rle_encode(mask), H, W), mask)
    assert sam3_worker.tight_box(np.zeros((3, 3), bool)) is None


def test_resolve_python_order(monkeypatch, tmp_path):
    monkeypatch.delenv("HMB_SAM3_PYTHON", raising=False)
    assert resolve_python(sys.executable) == sys.executable
    monkeypatch.setenv("HMB_SAM3_PYTHON", str(tmp_path / "missing" / "python"))
    with pytest.raises(SAM3SetupError, match="does not exist"):
        resolve_python()
    monkeypatch.setenv("HMB_SAM3_PYTHON", sys.executable)
    assert resolve_python() == sys.executable
    monkeypatch.delenv("HMB_SAM3_PYTHON")
    monkeypatch.setattr(sam3_backend.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(SAM3SetupError) as err:
        resolve_python()
    text = str(err.value)
    assert "HMB_SAM3_PYTHON" in text and "facebookresearch/sam3" in text and "--tracker yolo" in text


def test_worker_env_isolates_foreign_interpreter(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/project/overlay")
    assert "PYTHONPATH" not in worker_env("/opt/other/bin/python")
    assert worker_env(sys.executable)["PYTHONPATH"] == "/project/overlay"


def test_sam3_backend_end_to_end_with_fake_sam3(fake_sam3, tmp_path):
    tracks = track_sam3(fake_sam3, quiet=True)
    assert tracks.backend == "sam3.1" and tracks.num_frames == N and (tracks.height, tracks.width) == (H, W)
    assert tracks.ids == [0, 4]
    assert tracks.frames(4).tolist() == list(range(3, N))
    assert tracks.box[tracks.records(0, 2)[0]].tolist() == [4, 0, W, 30]    # speck stretches the raw box
    assert tracks.meta["sam3"]["prompt"] == "person"
    # through track_people: specks cleaned, raw output kept, second call reuses the result
    out = tmp_path / "tracks"
    people = api.track_people(fake_sam3, out, backend="sam3.1", quiet=True, preview=False)
    assert people.box[people.records(0, 2)[0]].tolist() == [4, 5, 14, 30]
    assert (out / "raw" / "tracks.json").exists() and (out / "tracks.json").exists()
    again = api.track_people(fake_sam3, out, backend="sam3.1", quiet=True, preview=False)
    assert again.ids == people.ids and len(again) == len(people)


def test_sam3_chunked_tracking_keeps_ids(fake_sam3):
    whole = track_sam3(fake_sam3, quiet=True, chunk_frames=0)
    chunked = track_sam3(fake_sam3, quiet=True, chunk_frames=5, chunk_overlap=2)   # spans 0-4, 3-7
    assert chunked.meta["sam3"]["chunks"] == [[0, 5], [3, 8]]
    assert chunked.ids == whole.ids == [0, 4]
    for oid in (0, 4):
        assert chunked.frames(oid).tolist() == whole.frames(oid).tolist()
        assert np.array_equal(chunked.boxes(oid), whole.boxes(oid), equal_nan=True)


def test_chunk_spans_and_id_matching():
    assert sam3_worker.chunk_spans(100, 0, 16) == [(0, 100)]
    assert sam3_worker.chunk_spans(100, 100, 16) == [(0, 100)]
    spans = sam3_worker.chunk_spans(100, 40, 10)
    assert spans == [(0, 40), (30, 70), (60, 100)]
    a = np.zeros((10, 10), bool); a[:5] = True
    b = np.zeros((10, 10), bool); b[6:] = True
    prev = {3: {7: a, 9: b}, 4: {7: a, 9: b}}
    head = {3: {0: b, 1: a, 2: ~(a | b)}, 4: {0: b, 1: a}}
    assert sam3_worker.match_ids(prev, head) == {0: 9, 1: 7}
    assert sam3_worker.match_ids({}, head) == {}


def test_sam3_gated_checkpoint_message(fake_sam3, monkeypatch):
    monkeypatch.setenv("FAKE_HF_MODE", "gated")
    with pytest.raises(SAM3CheckpointError) as err:
        track_sam3(fake_sam3, quiet=True)
    text = str(err.value)
    assert "huggingface.co/facebook/sam3.1" in text and "hf auth login" in text
    assert "HMB_SAM3_CHECKPOINT" in text and "yolo" in text
    with pytest.raises(SAM3CheckpointError):
        sam3_backend.check_sam3()


def test_sam3_missing_local_checkpoint(fake_sam3, monkeypatch, tmp_path):
    monkeypatch.setenv("HMB_SAM3_CHECKPOINT", str(tmp_path / "nope.pt"))
    with pytest.raises(SAM3CheckpointError, match="not found"):
        track_sam3(fake_sam3, quiet=True)


def test_sam3_not_installed(monkeypatch, tmp_path):
    root = tmp_path / "site"
    (root / "torch").mkdir(parents=True)
    (root / "torch" / "__init__.py").write_text(FAKE_TORCH)
    (root / "sam3.py").write_text("raise ImportError('No module named sam3 (simulated)')\n")
    monkeypatch.setenv("PYTHONPATH", str(root))
    monkeypatch.setenv("HMB_SAM3_PYTHON", sys.executable)
    video = tmp_path / "v.mp4"
    with VideoWriter(video, 24.0, W, H) as writer:
        writer.write(np.zeros((H, W, 3), np.uint8))
    with pytest.raises(SAM3SetupError, match="environment is broken"):
        track_sam3(video, quiet=True)


def test_normalize_backend():
    assert api.normalize_backend("SAM3.1") == "sam3.1"
    assert api.normalize_backend("sam3") == "sam3.1"
    assert api.normalize_backend("yolo") == "yolo"
    with pytest.raises(ValueError):
        api.normalize_backend("bytetrack")


# ----------------------------------------------------------------------------- yolo
class FakeTensor:
    def __init__(self, a):
        self.a = np.asarray(a)

    def cpu(self):
        return self

    def numpy(self):
        return self.a


def test_yolo_result_records():
    empty = SimpleNamespace(boxes=SimpleNamespace(id=None, conf=FakeTensor([0.9])),
                            masks=SimpleNamespace(data=FakeTensor(np.ones((1, H, W)))))
    ids, scores, masks = yolo_backend.result_records(empty, H, W)
    assert len(ids) == 0 and masks.shape == (0, H, W)

    class Boxes:
        id = FakeTensor([3.0, 5.0])
        conf = FakeTensor([0.8, 0.6])

        def __len__(self):
            return 2

    low = np.zeros((2, H // 2, W // 2), np.float32)
    low[0, 2:10, 2:10] = 1.0
    low[1, 5:15, 15:25] = 0.9
    result = SimpleNamespace(boxes=Boxes(), masks=SimpleNamespace(data=FakeTensor(low)))
    ids, scores, masks = yolo_backend.result_records(result, H, W)
    assert ids.tolist() == [3, 5] and scores.tolist() == pytest.approx([0.8, 0.6])
    assert masks.shape == (2, H, W) and masks.dtype == bool
    assert masks[0, 10, 10] and not masks[0, 30, 30]


def test_yolo_weights_resolution(monkeypatch, tmp_path):
    monkeypatch.setattr(yolo_backend, "ASSET_DIR", tmp_path / "assets")
    monkeypatch.delenv(yolo_backend.ENV_WEIGHTS, raising=False)
    with pytest.raises(FileNotFoundError):
        yolo_backend.resolve_weights(download=False)
    local = tmp_path / "custom-seg.pt"
    local.write_bytes(b"x")
    monkeypatch.setenv(yolo_backend.ENV_WEIGHTS, str(local))
    assert yolo_backend.resolve_weights() == local.resolve()
    with pytest.raises(FileNotFoundError):
        yolo_backend.resolve_weights(str(tmp_path / "missing" / "w-seg.pt"))
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "yolov8x-seg.pt").write_bytes(b"x")
    monkeypatch.delenv(yolo_backend.ENV_WEIGHTS)
    assert yolo_backend.resolve_weights() == tmp_path / "assets" / "yolov8x-seg.pt"


def test_worker_imports_only_numpy_and_stdlib():
    source = open(sam3_worker.__file__).read()
    top_level = [line for line in source.splitlines() if line.startswith(("import ", "from "))]
    allowed = {"argparse", "contextlib", "io", "json", "os", "sys", "time", "numpy", "__future__"}
    for line in top_level:
        module = line.split()[1].split(".")[0]
        assert module in allowed, line
    assert "import hmb" not in source and "from hmb" not in source
