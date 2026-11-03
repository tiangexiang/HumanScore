"""Video reading, MeTRAbs loading and the keypoint names of the fit.

Part of the MonocularBiomechanics fitting code (``baseline/monocular_demos``; the same file is
in ``metrics/bone_length/monocular_demos``).

* :func:`load_metrabs` loads the MeTRAbs pose estimator (the model installed by
  ``install.sh`` when present, else ``$HMB_METRABS_MODEL`` or the TF Hub handle);
* :func:`video_reader` yields RGB frames in batches;
* ``joint_names`` lists the 87 ``bml_movi_87`` keypoints in MeTRAbs order; the fit matches
  them to the skeleton's sites of the same name.
"""
import cv2
import numpy as np
from tqdm import tqdm

def jax_memory_limit():
    """Stop JAX from preallocating most of the GPU memory (call before JAX starts)."""
    import os
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

def tensorflow_memory_limit():
    """Let TensorFlow grow its GPU memory on demand, so JAX can share the GPU."""
    # limit tensorflow memory. there are also other approaches
    # https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

def video_reader(filename: str, batch_size: int = 8, width: int | None = None):
    """
    Read a video file and yield RGB frames in batches, optionally downsampled to ``width``.

    Args:
        filename: (str) The path to the video file.
        batch_size: (int) The number of frames to yield at once.
        width: (int | None) The width to downsample to. If None, the original width is used.

    Returns:
        A tuple of (generator, n_frames) where generator yields batches and n_frames is total frame count
    """

    cap = cv2.VideoCapture(filename)
    
    # Get total frame count
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    def frame_generator():
        cap = cv2.VideoCapture(filename)  # Create new capture object for generator
        frames = []
        while True:
            ret, frame = cap.read()

            if ret is False:
                if len(frames) > 0:
                    frames = np.array(frames)
                    yield frames
                cap.release()
                return
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if width is not None:
                    # downsample to keep the aspect ratio and output the specified width
                    scale = width / frame.shape[1]
                    height = int(frame.shape[0] * scale)
                    frame = cv2.resize(frame, (width, height))

                frames.append(frame)

                if len(frames) >= batch_size:
                    frames = np.array(frames)
                    yield frames
                    frames = []
    
    cap.release()  # Release the initial capture object
    return frame_generator(), n_frames

def _import_tensorflow_hub():
    """``import tensorflow_hub``, which needs Keras 2 (``tf-keras``).

    TensorFlow routes ``tf.keras`` to Keras 3 when that is installed. The lightweight environment
    (install/install_slim.sh) does not install Keras 3, which MeTRAbs inference never uses; there,
    ``tf.keras`` is routed to ``tf-keras`` before tensorflow_hub first touches it.
    """
    import importlib.util
    import os

    if importlib.util.find_spec("keras") is None:
        os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
    import tensorflow_hub as hub
    return hub


def _local_metrabs_model():
    """The SavedModel that install.sh unpacks (hmb/model_paths.py), if present; same model as the hub handle."""
    from pathlib import Path

    for parent in Path(__file__).resolve().parents:
        if (parent / "hmb/model_paths.py").is_file():
            local = parent / "outputs/assets/metrabs/metrabs_eff2xl_y4_384px_800k_28ds"
            return str(local) if (local / "saved_model.pb").is_file() else None
    return None


def load_metrabs(model_path=None, detection_only=False):
    """Load (and cache) the MeTRAbs model.

    ``model_path`` is a SavedModel folder or TF Hub handle; without it ``$HMB_METRABS_MODEL``,
    the model installed by ``install.sh`` or the public ``metrabs_xl`` handle is used.  With
    ``detection_only`` only the batched detection entry point is loaded
    (``hmb.metrabs_model.load_detection_model``, faster); ``HMB_METRABS_FULL_LOAD=1`` loads
    the full model instead.  The load timing is kept in ``load_metrabs.last_timing``.
    """
    import os
    import sys
    from pathlib import Path
    from time import perf_counter

    handle = model_path or os.environ.get("HMB_METRABS_MODEL") or _local_metrabs_model() or 'https://bit.ly/metrabs_xl'
    fast = detection_only and os.environ.get("HMB_METRABS_FULL_LOAD", "0") != "1"
    key = (str(handle), fast)
    if load_metrabs.model is not None and load_metrabs.handle == key:
        return load_metrabs.model
    started = perf_counter()
    if fast:
        # Scripts may run from inside baseline/; put the repository root on sys.path so
        # that the shared hmb loader can be imported.
        repository = next(parent for parent in Path(__file__).resolve().parents
                          if (parent / "hmb/metrabs_model.py").is_file())
        if str(repository) not in sys.path:
            sys.path.insert(0, str(repository))
        from hmb.metrabs_model import load_detection_model
        source = Path(handle).expanduser()
        if not (source / "saved_model.pb").is_file():
            hub = _import_tensorflow_hub()
            source = Path(hub.resolve(handle))
        try:
            load_metrabs.model = load_detection_model(source)
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            # A SavedModel the fast loader cannot read falls back to the full load.
            # CUDA and out-of-memory errors are not caught on purpose.
            print(f"[MeTRAbs] Detection-only load unavailable ({error}); loading full model.")
            hub = _import_tensorflow_hub()
            load_metrabs.model = hub.load(handle)
    else:
        hub = _import_tensorflow_hub()
        load_metrabs.model = hub.load(handle)
    load_metrabs.handle = key
    load_metrabs.last_timing = getattr(load_metrabs.model, "_hmb_timing", {
        "mode": "full_model", "model_load_seconds": perf_counter() - started})
    print(f"[MeTRAbs] {load_metrabs.last_timing}", flush=True)
    return load_metrabs.model

load_metrabs.model = None
load_metrabs.handle = None
load_metrabs.last_timing = None

joint_names = [
    "backneck",
    "upperback",
    "clavicle",
    "sternum",
    "umbilicus",
    "lfronthead",
    "lbackhead",
    "lback",
    "lshom",
    "lupperarm",
    "lelbm",
    "lforearm",
    "lwrithumbside",
    "lwripinkieside",
    "lfin",
    "lasis",
    "lpsis",
    "lfrontthigh",
    "lthigh",
    "lknem",
    "lankm",
    "LHeel",
    "lfifthmetatarsal",
    "LBigToe",
    "lcheek",
    "lbreast",
    "lelbinner",
    "lwaist",
    "lthumb",
    "lfrontinnerthigh",
    "linnerknee",
    "lshin",
    "lfirstmetatarsal",
    "lfourthtoe",
    "lscapula",
    "lbum",
    "rfronthead",
    "rbackhead",
    "rback",
    "rshom",
    "rupperarm",
    "relbm",
    "rforearm",
    "rwrithumbside",
    "rwripinkieside",
    "rfin",
    "rasis",
    "rpsis",
    "rfrontthigh",
    "rthigh",
    "rknem",
    "rankm",
    "RHeel",
    "rfifthmetatarsal",
    "RBigToe",
    "rcheek",
    "rbreast",
    "relbinner",
    "rwaist",
    "rthumb",
    "rfrontinnerthigh",
    "rinnerknee",
    "rshin",
    "rfirstmetatarsal",
    "rfourthtoe",
    "rscapula",
    "rbum",
    "Head",
    "mhip",
    "CHip",
    "Neck",
    "LAnkle",
    "LElbow",
    "LHip",
    "LHand",
    "LKnee",
    "LShoulder",
    "LWrist",
    "LFoot",
    "RAnkle",
    "RElbow",
    "RHip",
    "RHand",
    "RKnee",
    "RShoulder",
    "RWrist",
    "RFoot",
]
