Installation
============

One command installs HumanScore:

```bash
bash install.sh --smplx-dir ~/Downloads/models_smplx_v1_1.zip   # add --with-sam3 for several people
source .venv-humanscore/bin/activate
```

The [README](../README.md#install) covers the normal case. This page explains what the installer
does, its options, where the model files go, and how to install in special situations.
[TROUBLESHOOTING.md](TROUBLESHOOTING.md) lists common problems.

Requirements
------------

* **System.** Linux x86-64 with glibc 2.28 or newer (Ubuntu 20.04, RHEL 8 or newer).
* **GPU.** An NVIDIA GPU with compute capability 7.0 or newer and 16 GB of memory or more. The
  multi-person tracker SAM 3.1 needs about 30 GB.
* **Driver.** A driver for CUDA 12.8 (570 or newer). The installer accepts drivers from 525.60.13
  on, with a warning below 570.
* **Tools.** `git`, `curl`, `tar` and gcc/g++ 9 to 14 (on Ubuntu: `sudo apt-get install git curl
  build-essential`). ffmpeg is optional: without it, the installer adds a private copy for the
  multi-person script.
* **Network.** HTTPS access to pypi.org, files.pythonhosted.org, download.pytorch.org, github.com,
  raw.githubusercontent.com and developer.download.nvidia.com (environment), and to
  huggingface.co, www.dropbox.com and omnomnom.vision.rwth-aachen.de (model files).
* **Disk.** About 25 GB: the environment takes about 9 GB, the model files about 13 GB (plus
  3.5 GB for SAM 3.1), and uv's package cache (`~/.cache/uv`) about 1 GB more when it is on the
  same file system as the environment, but about 10 GB on another one. The build also needs about
  1.5 GB of temporary space in `$TMPDIR`. The installer checks the free space before it starts.

What `install.sh` does
----------------------

| step | what it does |
|---|---|
| preflight | checks the system, driver, GPU, tools, compiler, ffmpeg, network and free disk; stops with a fix for each problem |
| sources | fetches the pinned GVHMR, HADM and torch-mesh-isect sources (git submodules) |
| environment | builds the Python environment `./.venv-humanscore` with `install/install_slim.sh`: a private Python, the hash-pinned packages of `requirements/lock-slim.txt`, and the two CUDA extensions compiled with your gcc and a CUDA compiler that it downloads and deletes afterwards |
| ffmpeg | only when no ffmpeg is installed: a private copy for the person videos of the multi-person script |
| smoke | quick checks: imports, GPU access, the command-line entry points |
| assets | downloads the model files from their official sources (below); they download while the environment is built |
| smplx | places the SMPL-X model that you downloaded yourself |
| sam3 | with `--with-sam3`: Hugging Face login and access check, then the SAM 3.1 checkpoint |
| doctor | `python -m hmb.cli doctor`: checks every package and model file |
| selftest | with `--self-test`: scores a short public clip and checks the result |

**Re-running.** Each finished step is recorded in `outputs/install/stamps/`. A second run skips
every step whose inputs have not changed, and interrupted downloads continue where they stopped.
`--force-step NAME` redoes one step, and `--reinstall` redoes everything.

**Logs and failures.** Each run writes a full log to `outputs/install/logs/`. When a step fails,
the installer prints the step, the failing command and the log file. Fix the cause and run the same
command again.

**What it touches.** The installer needs no sudo and never edits your shell start-up files, so
activate the environment in every new shell. It writes only inside the checkout (the environment,
`outputs/`, and links in `GVHMR/inputs/checkpoints/` and
`metrics/multi-limbs/HADM/pretrained_models/`), to uv's cache and Python folder, and, if no uv is
installed, to a private copy of uv in `~/.local/share/humanscore/`.

**Exit status.** 0: ready. 3: almost ready; the summary says what is missing (usually the SMPL-X
model). 1: a step failed. 2: wrong options.

Options
-------

| option | meaning |
|---|---|
| `--smplx-dir PATH` | the SMPL-X model: the zip `models_smplx_v1_1.zip`, a folder that contains `SMPLX_NEUTRAL.npz`, or that file (`--smpl-dir` is the same option) |
| `--with-sam3` | also install SAM 3.1, the default tracker of `scripts/humanscore_multi.py` (gated, see below) |
| `--self-test` | at the end, score a short public clip and check the output (a few minutes) |
| `--prefix PATH` | where to create the environment (default `./.venv-humanscore`) |
| `--cuda-arch LIST` | GPU architectures for the CUDA extensions, for example `"8.0;8.6;8.9;9.0"` (default: the GPUs of this machine; see below) |
| `--skip-assets` | do not download the model files |
| `--jobs N` | parallel model downloads (default 4) |
| `--dry-run` | run the checks and print the plan; change nothing |
| `-y`, `--yes` | ask no questions (servers, CI); also the default when there is no terminal |
| `--force-step NAME` | redo one finished step: `sources`, `environment`, `ffmpeg`, `smoke`, `assets`, `smplx`, `sam3`, `doctor` or `selftest` (`sam3` and `selftest` imply `--with-sam3` and `--self-test`) |
| `--reinstall` | recreate the environment and redo every step (model files are checked, not downloaded again) |
| `--uninstall` | remove the environment and the installer's own files (add `--remove-assets` to delete the model files too) |
| `--log FILE` | write the log to `FILE` |

`bash install.sh --help` prints the same list.

The SMPL-X body model
---------------------

The SMPL-X model is licensed and may not be redistributed, so you download it yourself:

1. Register at [smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de) and accept the license.
2. Download "SMPL-X v1.1 (NPZ+PKL)" (`models_smplx_v1_1.zip`).
3. Run `bash install.sh --smplx-dir /path/to/models_smplx_v1_1.zip`.

The installer takes `SMPLX_NEUTRAL.npz` out of the zip and puts it in `outputs/assets/smplx/`.
Only the neutral model is used. You can also place the file yourself, or point to it with
`HMB_SMPLX_MODEL=/path/to/SMPLX_NEUTRAL.npz`.

SAM 3.1 (optional, for several people)
--------------------------------------

`scripts/humanscore_multi.py` tracks people with [SAM 3.1](https://github.com/facebookresearch/sam3)
by default. Its checkpoint is gated:

1. Request access at [huggingface.co/facebook/sam3.1](https://huggingface.co/facebook/sam3.1).
   Access is approved by hand, which can take a while.
2. Run `bash install.sh --with-sam3` (with your usual options). The installer installs SAM 3.1
   into the same environment and checks your Hugging Face login. In a terminal it offers to run
   `hf auth login`; on a server, set `HF_TOKEN=hf_...` (a token with read access from
   [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)).
3. When the access is granted, it downloads the checkpoint (3.5 GB) to `outputs/assets/sam3/`.

If you are not logged in yet, or the access is still pending, the rest of the installation still
finishes, and the summary says what to do. Run `bash install.sh --with-sam3` again later. Without
SAM 3.1, use `scripts/humanscore_multi.py --tracker yolo`. `python -m hmb.tracking --check-sam3`
checks the SAM 3.1 setup on its own.

Model files
-----------

The installer downloads these files, checks each against its SHA-256 checksum, and links it to
where the code reads it. [assets.json](assets.json) lists the exact sources, checksums and
licenses.

| file | size | location | source |
|---|---|---|---|
| GVHMR, HMR2 and ViTPose checkpoints | 5.4 GB | `outputs/assets/gvhmr/checkpoints/` (linked into `GVHMR/inputs/checkpoints/`) | the GVHMR authors |
| YOLOv8x person detector | 0.14 GB | as above | Ultralytics |
| HADM-G extra-limb detector | 5.7 GB | `outputs/assets/hadm/` (linked into `metrics/multi-limbs/HADM/pretrained_models/`) | the HADM authors |
| MeTRAbs 3-D keypoint model | 1.5 GB | `outputs/assets/metrabs/` | the MeTRAbs author |
| YOLOv8x segmentation (`--tracker yolo`) | 0.14 GB | `outputs/assets/yolo/` | Ultralytics |
| SAM 3.1 (`--with-sam3`) | 3.5 GB | `outputs/assets/sam3/` | `facebook/sam3.1` on Hugging Face |
| SMPL-X neutral model (by hand) | 0.14 GB | `outputs/assets/smplx/` | [smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de) |

Downloads are retried, resume after an interruption, and fall back to a mirror when one exists.
The downloader `install/fetch_assets.py` needs only Python's standard library, so you can also run
it by hand:

```bash
python3 install/fetch_assets.py plan                  # what is present and what is missing
python3 install/fetch_assets.py fetch                 # download and check the missing files
python3 install/fetch_assets.py check                 # check sizes only (no network)
python3 install/fetch_assets.py smplx --source /path/to/models_smplx_v1_1.zip
```

**No internet on the machine?** Copy `outputs/assets/` from another installation, then run
`python3 install/fetch_assets.py fetch`. It checks the copies and creates the links without
downloading anything.

**Other locations.** These environment variables point to a file elsewhere: `HMB_SMPLX_MODEL`,
`HMB_METRABS_MODEL` (an unpacked SavedModel), `HMB_SAM3_CHECKPOINT` and `HMB_YOLO_SEG_WEIGHTS`.
To keep all model files on another disk, make `outputs/assets` a symbolic link before installing.

Self-test
---------

`bash install.sh --self-test` (also on a finished installation) downloads a short public video (a
single-leg squat, Wikimedia Commons, CC BY-SA 4.0) and scores it with
`scripts/humanscore_single.py --paper-scores`. The test passes when the script succeeds and
`humanscore.json` holds all six scores, the tiers, the overall score and the paper scores. The
result is in `outputs/install/self-test/basic_single_leg_squat/`. The self-test needs the SMPL-X
model.

Building for another GPU
------------------------

The two CUDA extensions are compiled for the GPUs of the machine that runs the installer. To build
on a machine without a GPU, or for several GPU types, give the architectures:

```bash
bash install.sh --smplx-dir ~/Downloads/models_smplx_v1_1.zip --cuda-arch "8.0;8.6;8.9;9.0"
```

Without a GPU, the summary says `BUILT WITHOUT A GPU` (exit status 3). Run `bash install.sh` once
on the GPU machine to check the environment there.

The compiler
------------

The CUDA extensions need gcc/g++ 9 to 14. The installer uses `gcc` and `g++` from `PATH`, or the
compiler given by `CC` and `CXX`:

```bash
CC=gcc-13 CXX=g++-13 bash install.sh --smplx-dir ~/Downloads/models_smplx_v1_1.zip
```

On a cluster, a compiler module usually works (`module load gcc/13`). Without sudo or modules,
install gcc/g++ in your home directory (for example the conda-forge packages `gcc_linux-64` and
`gxx_linux-64`) and pass it with `CC=... CXX=...`.

The uv package cache
--------------------

The environment is built with [uv](https://github.com/astral-sh/uv), which keeps the downloaded
packages in its cache (`~/.cache/uv`, or `UV_CACHE_DIR`). The environment does not need the cache
after the installation: `uv cache clean` frees it (a later reinstall then downloads again). If uv
was not installed before, the installer uses a private copy; its summary prints the path to use
instead of `uv`.

Building only the environment
-----------------------------

`install/install_slim.sh` is the environment step of `install.sh`. To build only the environment,
without model files:

```bash
bash install/check_sources.sh --init       # the pinned GVHMR, HADM and torch-mesh-isect sources
bash install/install_slim.sh               # add --with-sam3 for SAM 3.1
source .venv-humanscore/bin/activate
python3 install/fetch_assets.py fetch      # the model files
python -m hmb.cli doctor                   # exit status 0 = ready
```

`bash install/install_slim.sh --help` lists its options. The environment has no `pip`; add packages
with `uv pip install`.

Checking an installation
------------------------

```bash
python -m hmb.cli doctor                                       # packages, sources and model files
bash install/check_sources.sh --assets                         # the same files, without Python
bash install/smoke_test.sh --python .venv-humanscore/bin/python   # imports, GPUs, entry points
python -m pytest -q                                            # the test suite
```

`doctor` downloads nothing. Items marked `AUTO` are downloaded on first use, and the multi-person
checks (SAM 3.1, YOLO segmentation weights, ffmpeg) are optional and never make it fail.

Uninstall
---------

```bash
bash install.sh --uninstall                    # environment and installer files; keeps model files
bash install.sh --uninstall --remove-assets    # also the downloaded model files
```

It lists what it removes and asks first (`--yes` skips the question). It keeps the SMPL-X model,
your results and uv's cache, and prints the commands that remove the cache too.

For maintainers: the lock file
------------------------------

`requirements/lock-slim.txt` pins every package with its hashes. It is generated from
`requirements/slim.in`, `requirements/slim-excludes.txt` and `requirements/slim-overrides.txt`.
After editing one of them, regenerate it:

```bash
uv pip compile requirements/slim.in --excludes requirements/slim-excludes.txt \
    --overrides requirements/slim-overrides.txt --python-version 3.11 \
    --python-platform x86_64-manylinux_2_28 --torch-backend cu128 --generate-hashes \
    -o requirements/lock-slim.txt
```
