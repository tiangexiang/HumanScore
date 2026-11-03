Troubleshooting
===============

Start with:

```bash
source .venv-humanscore/bin/activate
python -m hmb.cli doctor
```

It checks every package, source folder and model file, and says what is missing. Items marked
`AUTO` are downloaded on first use and are not errors. The multi-person checks are optional.

When you report a problem, include the output of `python -m hmb.cli doctor --json` and
`nvidia-smi`, the full error message and, for a scoring run, `<output folder>/logs/humanscore.log`.

Installation
------------

`bash install.sh --dry-run` repeats the checks without changing anything. After a failure, the
installer names the step, the failing command and the log (`outputs/install/logs/`). Fix the cause
and run the same command again: finished steps are skipped and downloads resume.

* **`cannot reach: HOST`.** The machine needs HTTPS access to the hosts listed in
  [INSTALL.md](INSTALL.md#requirements). Behind a proxy, set `export https_proxy=http://HOST:PORT`
  (and `http_proxy`) before running the installer. Without internet access, build the environment
  elsewhere and copy `outputs/assets/` (see [INSTALL.md, Model files](INSTALL.md#model-files)).
* **`not enough free disk`.** The message lists each file system with the space it needs. Free
  space, or move the parts: `--prefix` for the environment, `UV_CACHE_DIR` for uv's cache,
  `TMPDIR` for the build. To keep the model files elsewhere, make `outputs/assets` a symbolic link
  before installing.
* **`C/C++ compiler not found` or `gcc N found`.** Install gcc/g++ 9 to 14
  (`sudo apt-get install build-essential` on Ubuntu, `sudo dnf install gcc gcc-c++` on
  RHEL/Fedora, `module load gcc/13` on a cluster) and run again. To choose a compiler:
  `CC=gcc-13 CXX=g++-13 bash install.sh ...`. Without sudo or modules, install gcc/g++ in your home
  directory (for example the conda-forge packages `gcc_linux-64` and `gxx_linux-64`) and pass it with
  `CC=... CXX=...`.
* **`NVIDIA driver ... is too old` or `nvidia-smi not found`.** Install an NVIDIA driver 570 or
  newer. To build on a machine without a GPU, see
  [INSTALL.md, Building for another GPU](INSTALL.md#building-for-another-gpu).
* **`CUDA_VISIBLE_DEVICES='' hides every GPU`.** An empty `CUDA_VISIBLE_DEVICES` (or `-1`) hides
  the GPUs from PyTorch, JAX and TensorFlow. Unset it, or set it to the GPU to use
  (`export CUDA_VISIBLE_DEVICES=0`).
* **A model download fails or is slow.** Each file is retried several times, and a mirror is tried
  when the official host fails. Partial files (`*.part`) are kept, so run the installer again to
  resume. `SHA-256 ... does not match` means a host sent a different file: it is deleted and the
  next source is tried. HADM-G (5.7 GB) comes from Dropbox, which is often the slowest host.
  `python3 install/fetch_assets.py plan` shows what is still missing.
* **SAM 3.1: `access to facebook/sam3.1 not granted yet` or `not logged in`.** Request access at
  <https://huggingface.co/facebook/sam3.1> (approved by hand), create a token with read access at
  <https://huggingface.co/settings/tokens>, run `.venv-humanscore/bin/hf auth login`, then
  `bash install.sh --with-sam3`. On a server, `HF_TOKEN=hf_... bash install.sh --with-sam3 --yes`
  works too. The rest of the installation does not depend on it.
* **Error-looking lines in the smoke step.** TensorFlow prints `Unable to register cuFFT factory`
  (and the same for cuDNN and cuBLAS) and `computation placer already registered` when it loads
  next to PyTorch and JAX. These lines are harmless. The step passed when it prints
  `[SMOKE] All smoke tests passed`.
* **`ALMOST READY` (exit status 3).** Usually only the SMPL-X model is missing: see Model files
  below. The summary says what to do.
* **`another install.sh is running`.** Wait for it to finish. If no installer is running, delete
  `outputs/install/install.lock`.
* **Start over.** `bash install.sh --uninstall` (add `--remove-assets` to delete the model files
  too), then install again.

Model files
-----------

* **`SMPL-X neutral model` missing, or `Unknown model type body`.** Download the SMPL-X model as
  described in the [README](../README.md#install) and run
  `bash install.sh --smplx-dir /path/to/models_smplx_v1_1.zip`. You can also place the file
  yourself (a symbolic link is fine):

  ```bash
  mkdir -p outputs/assets/smplx
  cp SMPLX_NEUTRAL.npz outputs/assets/smplx/   # or: export HMB_SMPLX_MODEL=/path/to/SMPLX_NEUTRAL.npz
  ```

  The installer warns when the file differs from the reference SMPL-X file; scores can then
  differ slightly.
* **`HADM-G weights` missing.** Run `bash install.sh`, or
  `python3 install/fetch_assets.py fetch --only hadm-g`.
* **A GVHMR, ViTPose, HMR2 or YOLO download fails on first use.** This happens only without
  `install.sh`. Run `python3 install/fetch_assets.py fetch`. On a machine without internet access,
  copy the files into `outputs/assets/gvhmr/checkpoints/` (`doctor` prints the expected paths).
* **MeTRAbs is downloaded again after a reboot.** This also happens only without `install.sh`:
  TensorFlow Hub then caches the model in the temporary directory. Run
  `python3 install/fetch_assets.py fetch --only metrabs`, which keeps it in
  `outputs/assets/metrabs/`.

Running
-------

* **CUDA out of memory.** Stop other jobs on the GPU. To run the three detectors one after another
  instead of at the same time (less GPU memory, a little slower):
  `python -m hmb.cli run-all --video video.mp4 --serial-engines`. The result is in
  `humanscore_output/video/`, as with the script.
* **The wrong person was scored.** With several people in the video, `scripts/humanscore_single.py`
  follows the largest person near the start. Use `scripts/humanscore_multi.py` to score each
  person.
* **`bone_length` is `unsupported`.** The person was found in too few frames (less than 25 % of
  the analysed frames with both ends of the bones inside the image). The other scores are still
  valid.
* **`--moving-cam` fails with a `pycolmap` error.** The moving-camera mode needs `pycolmap`, which
  the installer includes. Check `python -c "import pycolmap"`, or score with a static camera (the
  default).
* **JAX runs on the CPU (`cuDNN not found`, `Falling back to cpu`).** Check that the driver
  supports CUDA 12.8 and that only one cuDNN is installed:

  ```bash
  uv pip list | grep -i cudnn                                          # one nvidia-cudnn-cu12 line
  python -c "import jax; print([d.platform for d in jax.devices()])"    # expected: ['gpu']
  ```

* **Kinematic-extremes, smoothness or joint-range scores differ slightly between machines.** These
  scores come from a skeleton fit that the GPU compiles. Two runs on the same GPU type give the
  same fit. Different GPU types use different kernels, so small differences between GPU models
  remain, and on a difficult clip the fit can end in a different local optimum. Setting
  `HMB_FIT_AUTOTUNE=1` lets the compiler time its kernels, which makes runs less repeatable.

Several people (SAM 3.1)
------------------------

Run `python -m hmb.tracking --check-sam3` first.

* **`SAM3SetupError` (SAM 3.1 not found).** Install it with `bash install.sh --with-sam3`, or use
  `--tracker yolo`, which needs no extra setup.
* **`SAM3CheckpointError` (checkpoint not available).** The checkpoint is gated: see the SAM 3.1
  entry under Installation. Alternatively, download `sam3.1_multiplex.pt` yourself and set
  `HMB_SAM3_CHECKPOINT=/path/to/sam3.1_multiplex.pt`.
* **CUDA out of memory in SAM 3.1.** SAM 3.1 needs about 30 GB, and more for long tracking
  sessions. Videos longer than 600 frames are tracked in pieces of 600 frames. For shorter pieces,
  call the multi-person pipeline from Python, in the repository root:

  ```python
  from hmb.multi import score_people
  score_people("video.mp4", "humanscore_output/video_multi", tracker_options={"sam3_chunk_frames": 300})
  ```

  Or use `--tracker yolo`.
* **One person appears twice in the results.** The tracker split the person into two ids, for
  example after an occlusion. This happens more often with `--tracker yolo`. Check
  `tracks/preview.mp4` to see which ids belong to the same person.

CUDA extensions
---------------

* **`no kernel image is available for execution on the device`.** The extensions were built for a
  different GPU. Rebuild them on this machine with `bash install/install_slim.sh --rebuild-native`
  (it builds for the GPUs it finds), or for several GPU types with
  `HMB_CUDA_ARCH="8.0;8.6;8.9;9.0" bash install/install_slim.sh --rebuild-native`.
* **`unsupported GNU version` or `command 'gcc' failed` during the build.** The compiler must be
  gcc/g++ 9 to 14. Select one with `CC`/`CXX` (see Installation above).
* **`helper_math.h` not found, `AT_CHECK` not declared, or `No module named 'mmcv'`.** The
  extension was built from the unmodified sources. Build it with the installer
  (`bash install/install_slim.sh --rebuild-native`), which patches a copy of the sources.
* **A TensorFlow tool such as `tensorboard` fails with `ModuleNotFoundError`.** The
  environment leaves out packages that HumanScore does not use. Install what you need with
  `uv pip install tensorboard`. For the same reason, `uv pip check` lists some packages as missing;
  this is expected.
