"""Fast loading of the MeTRAbs 3-D pose detector: only its ``detect_poses_batched`` graph.

The published MeTRAbs SavedModel contains thousands of training and export
functions that HumanScore never calls, and loading all of them is slow.  On first
use :func:`prepare_detection_cache` writes a reduced copy of the model graph that
keeps exactly the functions and resources ``detect_poses_batched`` depends on;
its weights are symlinks to the original model, so no weights, thresholds,
augmentations or operators change.  The reduced copy is inference-only.

Cache location: ``$HMB_METRABS_CACHE``, default ``outputs/runtime_cache/metrabs/``,
one folder per source model and TensorFlow version.

Used by ``baseline/monocular_demos/utils.py:load_metrabs(detection_only=True)``,
which the detection engine (``hmb.detect_engine``) calls::

    model = load_detection_model(Path(saved_model_dir))
    predictions = model.detect_poses_batched(frames, skeleton="bml_movi_87", ...)
"""
from __future__ import annotations

from collections import deque
import hashlib
import json
import os
from pathlib import Path
import tempfile
from time import perf_counter
from types import SimpleNamespace


# Part of the cache key: increase it when the graph selection or the cache layout changes,
# so caches written by older code are rebuilt instead of reused.
FORMAT_VERSION = 1
ENTRYPOINT = "detect_poses_batched"
REPO_ROOT = Path(__file__).resolve().parents[1]
GRAPH_OPTIMIZATION_MODES = ("full", "lean", "off")


def configure_graph_optimization(mode: str | None = None) -> str:
    """Choose how much Grappler optimisation the first MeTRAbs call pays for.

    The first ``detect_poses_batched`` call of a process instantiates a large
    function graph and runs TensorFlow's Grappler meta optimizer on it on the
    CPU, which is most of the first-call cost.  Grappler runs two
    meta-optimizer passes by default; the second pass and the memory optimizer
    barely change this inference graph.

    ``full``: TensorFlow defaults.
    ``lean``: one meta-optimizer pass and no memory optimizer (the detection engine's default).
    ``off``:  no Grappler at all (fastest first call, slower warm inference).

    The setting is process-wide and must be applied before the first call of
    any ``tf.function``. ``mode=None`` reads ``HMB_METRABS_GRAPH_OPTIMIZATION``
    (default ``full``). Returns the mode actually applied.
    """
    mode = (mode or os.environ.get("HMB_METRABS_GRAPH_OPTIMIZATION") or "full").lower()
    if mode not in GRAPH_OPTIMIZATION_MODES:
        raise ValueError(f"graph optimization must be one of {GRAPH_OPTIMIZATION_MODES}, got {mode!r}")
    if mode == "full":
        return mode
    import tensorflow as tf
    if mode == "off":
        tf.config.optimizer.set_experimental_options({"disable_meta_optimizer": True})
        return mode
    try:
        from tensorflow.core.protobuf import config_pb2, rewriter_config_pb2
        from tensorflow.python.eager import context

        ctx = context.context()
        config = config_pb2.ConfigProto()
        if getattr(ctx, "_config", None) is not None:
            config.CopyFrom(ctx._config)
        rewrite = config.graph_options.rewrite_options
        rewrite.meta_optimizer_iterations = rewriter_config_pb2.RewriterConfig.ONE
        rewrite.memory_optimization = rewriter_config_pb2.RewriterConfig.NO_MEM_OPT
        ctx._config = config
        # Function calls serialize the context config into each call; drop the
        # cached copy so the new rewrite options are used from the next call on.
        ctx._thread_local_data.function_call_options = None
        return mode
    except (AttributeError, ImportError) as error:  # private TF API moved: keep defaults
        print(f"[MeTRAbs] lean graph optimization unavailable ({error}); using TF defaults", flush=True)
        return "full"


class DetectionModel(SimpleNamespace):
    """The loaded detector: ``detect_poses_batched`` plus load timings and cache statistics."""

    def __repr__(self):
        return f"MeTRAbsDetectionModel(functions={self.cache_statistics['retained_function_count']}, cache={self.detection_cache!r})"


def source_signature(source: Path, tensorflow_version: str) -> dict:
    """What the reduced graph depends on: the graph file's hash, the weight files' identity and TF."""
    protobuf = source / "saved_model.pb"
    variables = source / "variables"
    return {"source": str(source.resolve()), "tensorflow": tensorflow_version,
            "format_version": FORMAT_VERSION,
            "protobuf_sha256": hashlib.sha256(protobuf.read_bytes()).hexdigest(),
            "variables": [{"name": path.name, "size": path.stat().st_size,
                           "mtime_ns": path.stat().st_mtime_ns}
                          for path in sorted(variables.iterdir()) if path.is_file()]}


def select_detection_graph(saved_model) -> tuple[list[str], dict]:
    """Prune ``saved_model`` in place to what ``detect_poses_batched`` needs.

    Walks the object graph and the function library from the entry point and
    keeps every object, concrete function, captured tensor and save/restore
    function reachable from it.  Returns the object paths to pass to
    ``tf.__internal__.saved_model.load_partial`` and function/object counts.
    """
    if len(saved_model.meta_graphs) != 1:
        raise ValueError("Detection-only loading requires a single TF2 metagraph")
    graph = saved_model.meta_graphs[0]
    objects = graph.object_graph_def
    entrypoint = next((ref.node_id for ref in objects.nodes[0].children
                       if ref.local_name == ENTRYPOINT), None)
    if entrypoint is None or objects.nodes[entrypoint].WhichOneof("kind") != "function":
        raise ValueError(f"MeTRAbs SavedModel has no {ENTRYPOINT} function")
    functions = {function.signature.name: function for function in graph.graph_def.library.function}
    original_functions = len(functions)
    needed_nodes, needed_functions = set(), set()
    node_queue, function_queue = [entrypoint], []

    def append_function_references(attributes):
        for value in attributes:
            if value.HasField("func") and value.func.name in functions:
                function_queue.append(value.func.name)
            function_queue.extend(value.name for value in value.list.func if value.name in functions)

    while node_queue or function_queue:
        while node_queue:
            node_id = node_queue.pop()
            if node_id in needed_nodes:
                continue
            needed_nodes.add(node_id)
            node = objects.nodes[node_id]
            node_queue.extend(ref.node_id for ref in node.children)
            node_queue.extend(ref.node_id for ref in node.dependencies)
            for saveable in node.saveable_objects.values():
                node_queue.extend((saveable.save_function, saveable.restore_function))
            kind = node.WhichOneof("kind")
            if kind == "function":
                function_queue.extend(node.function.concrete_functions)
            elif kind == "bare_concrete_function":
                function_queue.append(node.bare_concrete_function.concrete_function_name)
            elif kind == "captured_tensor":
                function_queue.append(node.captured_tensor.concrete_function)
        while function_queue:
            name = function_queue.pop()
            if name in needed_functions:
                continue
            needed_functions.add(name)
            if name in objects.concrete_functions:
                node_queue.extend(objects.concrete_functions[name].bound_inputs)
            function = functions[name]
            append_function_references(function.attr.values())
            for node in function.node_def:
                if node.op in functions:
                    function_queue.append(node.op)
                append_function_references(node.attr.values())

    kept = [function for function in graph.graph_def.library.function
            if function.signature.name in needed_functions]
    del graph.graph_def.library.function[:]
    graph.graph_def.library.function.extend(kept)
    # Backward graph serialization is irrelevant for this explicitly
    # inference-only cache. Keep any gradient definitions already in closure.
    gradients = [value for value in graph.graph_def.library.registered_gradients
                 if value.gradient_func in needed_functions]
    del graph.graph_def.library.registered_gradients[:]
    graph.graph_def.library.registered_gradients.extend(gradients)
    for name in list(objects.concrete_functions):
        if name not in needed_functions:
            del objects.concrete_functions[name]

    # Attribute path ("root.a.b") of every object, as load_partial expects them.
    paths, queue = {0: "root"}, deque([0])
    while queue:
        node_id = queue.popleft()
        for ref in objects.nodes[node_id].children:
            if ref.node_id not in paths:
                paths[ref.node_id] = paths[node_id] + "." + ref.local_name
                queue.append(ref.node_id)
    # A StaticHashTable's save/restore functions have no attribute path in the
    # original model. Alias only those hidden objects for load_partial's filter.
    for node_id in sorted(needed_nodes - paths.keys()):
        alias = objects.nodes[0].children.add()
        alias.local_name, alias.node_id = f"_hmb_dependency_{node_id}", node_id
        paths[node_id] = "root." + alias.local_name
    filters = [paths[node_id] for node_id in sorted(needed_nodes)]
    statistics = {"original_function_count": original_functions,
                  "retained_function_count": len(needed_functions),
                  "retained_object_count": len(needed_nodes)}
    return filters, statistics


def prepare_detection_cache(source: Path, cache_root: Path | None = None) -> Path:
    """Folder of the reduced detection graph of the SavedModel ``source``, built if missing."""
    import tensorflow as tf
    from tensorflow.core.protobuf import saved_model_pb2

    source = source.expanduser().resolve()
    signature = source_signature(source, tf.__version__)
    root = cache_root or Path(os.environ.get("HMB_METRABS_CACHE", REPO_ROOT / "outputs/runtime_cache/metrabs"))
    root = root.expanduser().resolve()
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:24]
    target = root / key
    marker = target / "hmb_detection_cache.json"
    if marker.is_file() and (target / "saved_model.pb").is_file():
        cached = json.loads(marker.read_text())
        if cached["source_signature"] == signature and cached.get("validated"):
            return target
    root.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    saved_model = saved_model_pb2.SavedModel()
    saved_model.ParseFromString((source / "saved_model.pb").read_bytes())
    filters, statistics = select_detection_graph(saved_model)
    with tempfile.TemporaryDirectory(prefix=key + ".", dir=root) as temporary:
        staging = Path(temporary)
        (staging / "saved_model.pb").write_bytes(saved_model.SerializeToString())
        for directory in ("variables", "assets"):
            if (source / directory).exists():
                (staging / directory).symlink_to(source / directory, target_is_directory=True)
        payload = {"source_signature": signature, "filters": filters, "statistics": statistics,
                   "build_seconds": perf_counter() - started, "inference_only": True, "validated": False}
        (staging / "hmb_detection_cache.json").write_text(json.dumps(payload, indent=2))
        # Atomic publication: interrupted model preparation never becomes a hit.
        if target.exists():
            import shutil
            shutil.rmtree(target)
        staging.rename(target)
    return target


def load_detection_model(source: Path, cache_root: Path | None = None) -> DetectionModel:
    """Load ``detect_poses_batched`` of the MeTRAbs SavedModel ``source`` via the reduced graph."""
    import tensorflow as tf

    prepare_started = perf_counter()
    cache = prepare_detection_cache(source, cache_root)
    prepare_seconds = perf_counter() - prepare_started
    metadata = json.loads((cache / "hmb_detection_cache.json").read_text())
    started = perf_counter()
    parts = tf.__internal__.saved_model.load_partial(
        str(cache), metadata["filters"], options=tf.saved_model.LoadOptions(allow_partial_checkpoint=True))
    # allow_partial_checkpoint skips the checkpoint objects that were pruned away, but every
    # variable that was kept must be restored: never run the detector on uninitialised weights.
    for value in parts.values():
        if isinstance(value, tf.Variable) and not bool(tf.raw_ops.VarIsInitializedOp(resource=value.handle).numpy()):
            raise ValueError(f"MeTRAbs variable was not restored: {value.name}")
    model_load_seconds = perf_counter() - started
    if not metadata.get("validated"):
        metadata["validated"] = True
        marker = cache / "hmb_detection_cache.json"
        temporary = marker.with_suffix(".json.part")
        temporary.write_text(json.dumps(metadata, indent=2))
        temporary.replace(marker)
    # Retain all captured variables and table resources for the lifetime of the
    # function, just as the full SavedModel retains them under its root object.
    return DetectionModel(detect_poses_batched=parts["root." + ENTRYPOINT], _parts=parts,
                           load_seconds=model_load_seconds,
                           detection_cache=str(cache), cache_statistics=metadata["statistics"],
                           _hmb_timing={"mode": "detection_only", "cache_prepare_seconds": prepare_seconds,
                                        "model_load_seconds": model_load_seconds,
                                        **metadata["statistics"]})
