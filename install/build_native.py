#!/usr/bin/env python3
"""Build the two CUDA extensions of HumanScore: Detectron2 (HADM fork) and torch-mesh-isect.

HADM-G (the extra-limbs detector) runs on Detectron2, and the self-collision score uses the BVH
kernels of torch-mesh-isect. Neither publishes wheels for PyTorch 2.8, so both installers compile
them from the pinned sources in ``HADM/`` and ``torch-mesh-isect/``:

``install/install_slim.sh`` (run by ``./install.sh``) calls it with uv as the installer, the system
gcc/g++ and a CUDA compiler fetched into a temporary directory that it deletes afterwards::

    python install/build_native.py --python .venv-humanscore/bin/python --uv "$(command -v uv)" \\
        --cuda-root "$TMP/cuda-12.8.1" --fetch-cuda --build-dir "$TMP/build"

The sources are copied to ``--build-dir`` and patched there, so the git submodules stay
untouched: torch-mesh-isect gets the PyTorch 2 API (``TORCH_CHECK``, ``data_ptr``) and NVIDIA's
``helper_math.h``; Detectron2 imports MMCV only for soft NMS (inference uses hard NMS) and wandb
only for training logs. The copies are then installed into the interpreter's environment with
``--no-deps`` (the inference dependencies are in ``requirements/``).

The interpreter must already contain PyTorch and torchvision. ``--fetch-cuda`` downloads NVIDIA's
checksum-verified CUDA compiler, runtime headers and CCCL (about 81 MB) for ``--cuda-version``,
which must match PyTorch's CUDA version (12.8.1 for the cu128 wheels).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request


REPO_ROOT = Path(__file__).resolve().parents[1]
CUDA_REDIST = "https://developer.download.nvidia.com/compute/cuda/redist/"
HELPER_MATH_URL = "https://raw.githubusercontent.com/NVIDIA/cuda-samples/v12.8/Common/helper_math.h"


def fetch_cuda(destination: Path, version: str = "12.8.1") -> None:
    """Download and unpack NVIDIA's CUDA compiler, runtime and CCCL ``version`` into ``destination``.

    The archives are listed, with their SHA-256, in NVIDIA's ``redistrib_<version>.json``; each is
    verified before it is unpacked. Archives already next to ``destination`` are reused.
    """
    with urllib.request.urlopen(CUDA_REDIST + f"redistrib_{version}.json", timeout=60) as response:
        manifest = json.load(response)
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("cuda_nvcc", "cuda_cudart", "cuda_cccl"):
        metadata = manifest[name]["linux-x86_64"]
        archive = destination.parent / Path(metadata["relative_path"]).name
        if not archive.is_file():
            print(f"Downloading NVIDIA {name}", flush=True)
            urllib.request.urlretrieve(CUDA_REDIST + metadata["relative_path"], archive)
        if hashlib.sha256(archive.read_bytes()).hexdigest() != metadata["sha256"]:
            raise RuntimeError(f"CUDA archive checksum mismatch: {archive}")
        with tarfile.open(archive) as bundle:
            top_directory = bundle.getmembers()[0].name.split("/")[0]
            bundle.extractall(destination.parent, filter="data")
        # The three archives unpack to separate folders; merge them into one CUDA root.
        shutil.copytree(destination.parent / top_directory, destination, dirs_exist_ok=True)


def _replace(file: Path, *pairs: tuple[str, str]) -> None:
    """Apply text replacements ``(old, new)`` to ``file`` in place."""
    text = file.read_text()
    for old, new in pairs:
        text = text.replace(old, new)
    file.write_text(text)


def prepare_source(component: str, build_dir: Path) -> Path:
    """Copy the pinned source of ``component`` (``mesh`` or ``hadm``) to ``build_dir`` and patch it.

    Returns the patched copy, ready for ``pip install``. No numerical code is changed.
    """
    source = REPO_ROOT / ("torch-mesh-isect" if component == "mesh" else "HADM")
    if not (source / "setup.py").is_file():
        raise FileNotFoundError(f"Missing pinned source: {source}. Run bash install/check_sources.sh --init")
    destination = build_dir / component
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns(
        ".git", "build", "*.egg-info", "pretrained_models", "inputs", "outputs", "__pycache__"),
                    ignore_dangling_symlinks=True, dirs_exist_ok=True)
    if component == "mesh":
        # torch-mesh-isect was written for PyTorch 1.x: AT_CHECK, Tensor.type() and Tensor.data<T>()
        # no longer exist in PyTorch 2.
        _replace(destination / "src/bvh.cpp", ("AT_CHECK", "TORCH_CHECK"), ("x.type().is_cuda()", "x.is_cuda()"))
        _replace(destination / "src/bvh_cuda_op.cu", ("triangles.type()", "triangles.scalar_type()"),
                 ("triangles.data<scalar_t>()", "triangles.data_ptr<scalar_t>()"))
        # include/ by absolute path, so the compiler finds it from pip's build directory.
        _replace(destination / "setup.py", ("    'include',", "    osp.join(here, 'include'),"))
        # helper_math.h is part of NVIDIA's CUDA samples, which the CUDA toolkit no longer ships.
        with urllib.request.urlopen(HELPER_MATH_URL, timeout=60) as response:
            (destination / "include/helper_math.h").write_bytes(response.read())
    else:
        # Inference uses hard NMS: import MMCV (a separate CUDA build) only if soft NMS is requested.
        for relative in ("detectron2/modeling/roi_heads/fast_rcnn.py",
                         "detectron2/modeling/test_time_augmentation.py"):
            _replace(destination / relative, ("import mmcv\n", ""), (
                "from mmcv.ops import soft_nms",
                "def soft_nms(*args, **kwargs):\n"
                "    from mmcv.ops import soft_nms as implementation\n"
                "    return implementation(*args, **kwargs)",
            ))
        # wandb is used only by the training event writer.
        _replace(destination / "detectron2/utils/events.py", ("\nimport wandb\n", "\n"), (
            "    def __init__(self, output_dir, window_size: int = 20):\n        wandb.init",
            "    def __init__(self, output_dir, window_size: int = 20):\n"
            "        global wandb\n        import wandb\n        wandb.init",
        ))
        # Pillow 10 removed the Image.LINEAR alias.
        _replace(destination / "detectron2/data/transforms/transform.py", ("Image.LINEAR", "Image.BILINEAR"))
    return destination


def main() -> None:
    """Parse the options, prepare the build environment and install the selected components."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--python", required=True, help="Interpreter with installed PyTorch/torchvision.")
    parser.add_argument("--cuda-root", type=Path, required=True, help="Matching CUDA toolkit root.")
    parser.add_argument("--cuda-version", default="12.8.1", help="CUDA release fetched by --fetch-cuda.")
    parser.add_argument("--component", choices=("mesh", "hadm", "all"), default="all",
                        help="mesh = torch-mesh-isect, hadm = Detectron2 (default: both).")
    parser.add_argument("--build-dir", type=Path,
                        default=Path(tempfile.gettempdir()) / f"hmb-native-build-{os.getuid()}",
                        help="Where the patched source copies are built.")
    parser.add_argument("--cuda-arch", default="", help="For example 8.0 on A100 (default: the visible GPUs).")
    parser.add_argument("--jobs", type=int, default=2, help="Parallel compile jobs (MAX_JOBS).")
    parser.add_argument("--fetch-cuda", action="store_true",
                        help="Download the CUDA compiler into --cuda-root unless it is already there.")
    parser.add_argument("--uv", default=None,
                        help="Install with this uv executable (uv pip install --python ...) instead of the "
                             "interpreter's pip; for environments without pip, such as uv virtual environments.")
    args = parser.parse_args()
    args.cuda_root = args.cuda_root.resolve()
    args.build_dir = args.build_dir.resolve()
    if args.fetch_cuda and not (args.cuda_root / "bin/nvcc").is_file():
        fetch_cuda(args.cuda_root, args.cuda_version)
    if not (args.cuda_root / "bin/nvcc").is_file():
        parser.error("--cuda-root must contain bin/nvcc and match the selected PyTorch CUDA version")
    args.build_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(CUDA_HOME=str(args.cuda_root), MAX_JOBS=str(args.jobs))
    # The interpreter's bin/ (ninja, when installed there) also for an environment that is not activated.
    environment["PATH"] = os.pathsep.join((str(args.cuda_root / "bin"), str(Path(args.python).parent),
                                           environment.get("PATH", "")))
    if args.cuda_arch:
        environment["TORCH_CUDA_ARCH_LIST"] = args.cuda_arch
    # NVIDIA's PyTorch wheels already include headers for CUDA math libraries.
    nvidia_includes = subprocess.check_output([
        args.python, "-c", "import pathlib,sysconfig; "
        "p=pathlib.Path(sysconfig.get_paths()['purelib'])/'nvidia'; "
        "print(':'.join(str(x) for x in p.glob('*/include')))",
    ], text=True).strip()
    environment["CPATH"] = nvidia_includes + os.pathsep + environment.get("CPATH", "")
    if args.uv:
        # --reinstall: rebuild even if the same version is installed (as pip's --upgrade for a local
        # directory); --no-cache: the wheels built from the temporary copies are not kept in uv's cache.
        pip_command = [args.uv, "pip", "install", "--python", args.python, "--no-deps", "--no-build-isolation",
                       "--reinstall", "--no-cache"]
    else:
        pip_command = [args.python, "-m", "pip", "install", "--upgrade", "--no-deps", "--no-build-isolation"]
    components = ("mesh", "hadm") if args.component == "all" else (args.component,)
    for component in components:
        source = prepare_source(component, args.build_dir)
        subprocess.run(pip_command + [str(source)], env=environment, check=True)
    print(f"Built {', '.join(components)}. Installed into the environment.")


if __name__ == "__main__":
    main()
