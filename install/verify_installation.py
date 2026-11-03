#!/usr/bin/env python3
"""Check the Python packages of a HumanScore environment: imports, versions and GPU access.

This is the first part of the installers' smoke test (``install/smoke_test.sh``). It imports the
libraries of every pipeline stage and runs a tiny GPU operation where a stage needs the GPU:

* PyTorch (body mesh, extra limbs, self-collision) and the core scientific libraries;
* GVHMR's 3-D libraries (PyTorch3D, SMPL-X, trimesh) and the torch-mesh-isect BVH kernel;
* JAX and MuJoCo/MJX (skeleton fit) and TensorFlow (MeTRAbs 3-D keypoints);
* Detectron2 (the HADM-G extra-limbs detector).

It needs no model files (``python -m hmb.cli doctor`` checks those). Run it with the
environment's interpreter::

    python install/verify_installation.py

Exit status: 0 when every check passes, 1 otherwise.
"""

import os
import sys
from typing import Dict, List

# The checks need little GPU memory: JAX would otherwise reserve 75 % of it, which fails (and
# makes the installers' smoke tests fail) when other jobs use the GPU.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


#: Set by install.sh when it builds on a machine without a usable GPU (--cuda-arch): the GPU checks
#: are reported but do not fail; on the GPU machine, run this script (or install.sh) again.
NO_GPU = os.environ.get("HMB_VERIFY_NO_GPU") == "1"


def print_header(text: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_result(name: str, success: bool, details: str = "") -> None:
    """Print one check result: a green tick or a red cross, the name and details."""
    status = "✓" if success else "✗"
    color = "\033[92m" if success else "\033[91m"
    reset = "\033[0m"

    result = f"{color}{status}{reset} {name}"
    if details:
        result += f": {details}"
    print(f"  {result}")


def check_python() -> bool:
    """Report the Python version (never fails; HumanScore is pinned to Python 3.11)."""
    print_header("Python Environment")

    version = sys.version_info
    print_result(
        "Python version",
        True,
        f"{version.major}.{version.minor}.{version.micro}",
    )
    if (version.major, version.minor) != (3, 11):
        print("    Warning: the pinned dependencies are for Python 3.11; other versions are untested.")

    return True


def check_pytorch() -> bool:
    """PyTorch imports and sees a CUDA GPU."""
    print_header("PyTorch Environment")

    results: List[bool] = []

    try:
        import torch

        print_result("PyTorch", True, torch.__version__)
        results.append(True)

        cuda_available = torch.cuda.is_available()
        print_result("CUDA available", cuda_available)

        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            print_result("GPU", True, device_name)

            device_count = torch.cuda.device_count()
            print_result("Number of GPUs", True, str(device_count))
        else:
            print("    Warning: PyTorch sees no CUDA GPU; HumanScore needs an NVIDIA GPU.")

        results.append(cuda_available or NO_GPU)
    except ImportError as e:
        print_result("PyTorch", False, str(e))
        results.append(False)

    return all(results)


def check_core_libraries() -> bool:
    """NumPy, SciPy, Matplotlib, OpenCV and Pillow import."""
    print_header("Core Libraries")

    libraries = [
        ("NumPy", "numpy"),
        ("SciPy", "scipy"),
        ("Matplotlib", "matplotlib"),
        ("OpenCV", "cv2"),
        ("Pillow", "PIL"),
    ]

    results: List[bool] = []
    for name, module in libraries:
        try:
            lib = __import__(module)
            version = getattr(lib, "__version__", "unknown")
            print_result(name, True, version)
            results.append(True)
        except ImportError as e:
            print_result(name, False, str(e))
            results.append(False)

    return all(results)


def check_3d_libraries() -> bool:
    """PyTorch3D, SMPL-X and trimesh import, and the torch-mesh-isect BVH kernel runs."""
    print_header("3D Vision Libraries (GVHMR)")

    results: List[bool] = []

    # PyTorch3D
    try:
        import pytorch3d

        print_result("PyTorch3D", True, pytorch3d.__version__)
        results.append(True)
    except ImportError:
        print_result("PyTorch3D", False, "Not installed")
        results.append(False)

    # SMPL-X
    try:
        import smplx  # type: ignore[import]

        print_result("SMPL-X", True)
        results.append(True)
    except ImportError:
        print_result("SMPL-X", False, "Not installed")
        results.append(False)

    # Trimesh
    try:
        import trimesh

        print_result("Trimesh", True, trimesh.__version__)
        results.append(True)
    except ImportError:
        print_result("Trimesh", False, "Not installed")
        results.append(False)

    # torch-mesh-isect
    try:
        from mesh_intersection.bvh_search_tree import BVH
        import torch

        if NO_GPU and not torch.cuda.is_available():
            import bvh_cuda  # noqa: F401  (the CUDA kernel runs only on a GPU)

            print_result("torch-mesh-isect (BVH)", True, "imported (no GPU: not run)")
        else:
            bvh = BVH(max_collisions=16)
            device = "cuda" if torch.cuda.is_available() else "cpu"

            triangles = torch.randn(1, 5, 3, 3).to(device)
            _ = bvh(triangles)

            print_result("torch-mesh-isect (BVH)", True, f"Working (device: {device})")
        results.append(True)
    except Exception as e:
        print_result("torch-mesh-isect (BVH)", False, str(e))
        results.append(False)

    return all(results)


def check_jax_mujoco() -> bool:
    """JAX runs on the GPU, and MuJoCo and MJX import (the skeleton fit)."""
    print_header("JAX & MuJoCo (MonocularBiomechanics)")

    results: List[bool] = []

    # JAX
    try:
        import jax
        import jax.numpy as jnp

        print_result("JAX", True, jax.__version__)

        x = jnp.array([1, 2, 3])
        _ = x.sum()  # simple operation to trigger backend

        devices = jax.devices()
        on_gpu = bool(devices) and devices[0].platform == "gpu"
        if devices:
            device_type = devices[0].platform
            print_result("JAX devices", on_gpu, f"{device_type} ({len(devices)} device(s))")
            if not on_gpu:
                print("    Warning: JAX runs on the CPU; the skeleton fit needs the GPU (cuDNN >= 9.8).")
        else:
            print_result("JAX devices", False, "No devices found")

        results.append(on_gpu or NO_GPU)
    except ImportError as e:
        print_result("JAX", False, str(e))
        results.append(False)

    # MuJoCo
    try:
        import mujoco

        print_result("MuJoCo", True, mujoco.__version__)
        results.append(True)
    except ImportError as e:
        print_result("MuJoCo", False, str(e))
        results.append(False)

    # MuJoCo MJX
    try:
        import mujoco.mjx  # type: ignore[import]

        print_result("MuJoCo MJX", True)
        results.append(True)
    except ImportError as e:
        print_result("MuJoCo MJX", False, str(e))
        results.append(False)

    return all(results)


def check_tensorflow() -> bool:
    """TensorFlow and TensorFlow Hub import, and TensorFlow sees a GPU (MeTRAbs 3-D keypoints)."""
    print_header("TensorFlow (MeTRAbs)")

    try:
        import importlib.util

        import tensorflow as tf
        # As baseline/monocular_demos/utils.py: without Keras 3 (install_slim.sh), tensorflow_hub uses tf-keras.
        if importlib.util.find_spec("keras") is None:
            os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
        import tensorflow_hub  # noqa: F401

        print_result("TensorFlow", True, tf.__version__)
        gpus = tf.config.list_physical_devices("GPU")
        print_result("TensorFlow GPUs", bool(gpus), str(len(gpus)))
        return bool(gpus) or NO_GPU
    except ImportError as e:
        print_result("TensorFlow", False, str(e))
        return False


def check_detectron2() -> bool:
    """Detectron2 (the HADM fork) and its config system import."""
    print_header("Detectron2 (HADM)")

    try:
        import detectron2
        from detectron2 import model_zoo  # noqa: F401
        from detectron2.config import get_cfg  # noqa: F401

        print_result("Detectron2", True, detectron2.__version__)
        print_result("Detectron2 config import", True)
        return True
    except ImportError as e:
        print_result("Detectron2", False, str(e))
        return False


def main() -> int:
    """Run every check, print a summary and return the exit status."""
    print("\n" + "=" * 60)
    print("  HumanScore - Installation Check")
    print("=" * 60)

    checks = [
        ("Python environment", check_python),
        ("PyTorch", check_pytorch),
        ("Core libraries", check_core_libraries),
        ("3D vision (GVHMR)", check_3d_libraries),
        ("JAX & MuJoCo", check_jax_mujoco),
        ("TensorFlow", check_tensorflow),
        ("Detectron2 (HADM)", check_detectron2),
    ]

    results: Dict[str, bool] = {}
    for name, check in checks:
        try:
            results[name] = check()
        except Exception as e:
            print(f"\nError: check '{name}' failed with exception:")
            print(f"  {str(e)}")
            results[name] = False

    # Summary
    print_header("Installation Summary")

    all_passed = True
    for name, result in results.items():
        status = "PASSED" if result else "FAILED"
        color = "\033[92m" if result else "\033[91m"
        reset = "\033[0m"
        print(f"  {color}{status}{reset} - {name}")

        if not result:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("\n  All checks passed: the Python environment is ready.\n")
        print("  Next steps:")
        print("    - python -m hmb.cli doctor                      (model files)")
        print("    - python scripts/humanscore_single.py video.mp4 (README.md, Score a video)")
        print("")
        return 0
    print("\n  Some checks failed; see the errors above.\n")
    print("  Troubleshooting:")
    print("    - install/TROUBLESHOOTING.md lists common causes and fixes")
    print("    - recreate the environment: bash install.sh --reinstall (model files are kept)")
    print("")
    return 1


if __name__ == "__main__":
    sys.exit(main())
