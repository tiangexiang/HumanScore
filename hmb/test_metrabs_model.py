"""The inference-only model must keep weights and the exact forward graph."""
import importlib.util
import json

import pytest

from hmb.metrabs_model import load_detection_model, prepare_detection_cache, select_detection_graph


pytestmark = pytest.mark.skipif(importlib.util.find_spec("tensorflow") is None,
                                reason="requires TensorFlow SavedModel runtime")


def make_model(tmp_path):
    import tensorflow as tf
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    class Model(tf.Module):
        def __init__(self):
            self.weight = tf.Variable(2.0)

        @tf.function(input_signature=[tf.TensorSpec([], tf.float32)])
        def detect_poses_batched(self, value):
            return self.weight * value + 1.0

        @tf.function(input_signature=[tf.TensorSpec([], tf.float32)])
        def unused_training_export(self, value):
            return self.weight * value**3

    source = tmp_path / "original"
    with tf.device("/CPU:0"):
        model = Model()
        tf.saved_model.save(model, str(source))
    return source


def test_closure_keeps_forward_function_bytes_and_removes_unused_exports(tmp_path):
    from tensorflow.core.protobuf import saved_model_pb2
    source = make_model(tmp_path)
    proto = saved_model_pb2.SavedModel()
    proto.ParseFromString((source / "saved_model.pb").read_bytes())
    original = {f.signature.name: f.SerializeToString() for f in proto.meta_graphs[0].graph_def.library.function}
    filters, statistics = select_detection_graph(proto)
    assert "root.detect_poses_batched" in filters
    assert statistics["retained_function_count"] < statistics["original_function_count"]
    for function in proto.meta_graphs[0].graph_def.library.function:
        assert function.SerializeToString() == original[function.signature.name]


def test_detection_cache_reuses_original_checkpoint_and_matches_forward(tmp_path):
    import tensorflow as tf
    source = make_model(tmp_path)
    original_pb = (source / "saved_model.pb").read_bytes()
    cache = prepare_detection_cache(source, tmp_path / "cache")
    assert (cache / "variables").is_symlink()
    assert (source / "saved_model.pb").read_bytes() == original_pb
    with tf.device("/CPU:0"):
        loaded = load_detection_model(source, tmp_path / "cache")
        assert loaded.detect_poses_batched(tf.constant(3.0)).numpy() == 7.0
    metadata = json.loads((cache / "hmb_detection_cache.json").read_text())
    assert metadata["validated"] is True
    assert prepare_detection_cache(source, tmp_path / "cache") == cache
