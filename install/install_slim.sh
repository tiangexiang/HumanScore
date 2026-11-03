#!/usr/bin/env bash
# Builds the HumanScore Python environment (the "environment" step of ./install.sh).
#
#   bash install/install_slim.sh [--venv PATH] [--with-sam3] [options]     (--help lists them)
#
# Steps:
#   1. uv: an existing uv >= 0.13.6, or the pinned release downloaded (checksum-verified) into a
#      private directory; nothing on PATH is replaced
#   2. a uv virtual environment (default ./.venv-humanscore) with a uv-managed CPython 3.11
#   3. requirements/lock-slim.txt: only the distributions the pipeline uses, hash-pinned, installed
#      in one pass without dependency resolution; --with-sam3 adds requirements/lock-slim-sam3.txt
#   4. PyTorch3D from source (Python operators only)
#   5. Detectron2/HADM and torch-mesh-isect (install/build_native.py), compiled with the system
#      gcc/g++ and a CUDA compiler fetched into a temporary directory that is deleted afterwards
#   6. removal of files no workload reads (headers, test suites, unused CUDA libraries; --no-strip)
#   7. smoke tests (install/smoke_test.sh)
#
# Re-running it on an existing environment only installs what is missing or outdated. Most users
# run ./install.sh, which also downloads the model files and checks the result.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

VENV_DIR="${HMB_VENV:-${REPO_ROOT}/.venv-humanscore}"
PYTHON_VERSION="${HMB_PYTHON_VERSION:-3.11.18}"
TORCH_BACKEND="${HMB_TORCH_BACKEND:-cu128}"
# uv (step 1): HMB_UV_VERSION selects another release to download.
UV_PINNED_VERSION="0.13.6"
UV_PINNED_SHA256="f29c3e94ffabd7e6f3ba521dba169a9b7ec12100bf3257087f0f24f6123f3645"   # x86-64 archive
UV_VERSION="${HMB_UV_VERSION:-${UV_PINNED_VERSION}}"
UV_MIN_VERSION="${UV_PINNED_VERSION}"   # older uv releases cannot download CPython 3.11.18
UV_DIR="${HMB_UV_DIR:-${XDG_DATA_HOME:-${HOME}/.local/share}/humanscore/uv-${UV_VERSION}}"
# NVIDIA CUDA compiler for the native extensions; must match the CUDA version of PyTorch (cu128).
CUDA_REDIST_VERSION="${HMB_CUDA_REDIST_VERSION:-12.8.1}"
CUDA_ARCH="${HMB_CUDA_ARCH:-}"   # e.g. 8.0 for A100; empty = the GPUs visible while building
PYTORCH3D_REVISION="${HMB_PYTORCH3D_REVISION:-2d224e66523039dbde1d24a86ecef4f8b1dcb038}"
NINJA_VERSION="1.13.2"           # build-only; removed again after the native build
NATIVE_CFLAGS="-g0"              # appended to CFLAGS of the native builds (no debug information)
BUILD_JOBS="${HMB_BUILD_JOBS:-$(( $(nproc 2>/dev/null || echo 4) < 8 ? $(nproc 2>/dev/null || echo 4) : 8 ))}"
BUILD_TMPDIR="${HMB_BUILD_TMPDIR:-${TMPDIR:-/tmp}}"

LOCK="${REPO_ROOT}/requirements/lock-slim.txt"
SAM3_LOCK="${REPO_ROOT}/requirements/lock-slim-sam3.txt"

WITH_SAM3=0
FORCE_RECREATE=0
REBUILD_NATIVE=0
STRIP=1
SKIP_SMOKE_TESTS=0
PRINT_UV=0
UV="${HMB_UV:-}"
LOG_FILE=""

usage() {
    cat <<EOF
Usage: bash install/install_slim.sh [options]

Creates the HumanScore Python environment: a uv virtual environment (CPython ${PYTHON_VERSION}) with
the distributions of requirements/lock-slim.txt (PyTorch ${TORCH_BACKEND}, TensorFlow, JAX, MuJoCo and
only the others the pipeline uses), PyTorch3D (Python operators), and Detectron2/HADM and
torch-mesh-isect compiled with the system gcc/g++ and CUDA ${CUDA_REDIST_VERSION}.

Options:
  --venv PATH          Environment directory (default: ${VENV_DIR})
  --with-sam3          Also install SAM 3.1 (sam3 + triton, requirements/lock-slim-sam3.txt), the
                       default multi-person tracker of scripts/humanscore_multi.py
  --force-recreate     Delete and recreate the environment if it exists
  --rebuild-native     Rebuild Detectron2/HADM and torch-mesh-isect even if they are installed
  --no-strip           Keep the files no workload reads (headers, test suites, unused CUDA
                       libraries; listed in STRIP_FILES of this script)
  --skip-smoke-tests   Skip the post-install smoke tests
  --print-uv           Only make sure a suitable uv exists (downloading it if needed) and print its path
  --uv PATH            uv executable to use (default: uv on PATH if >= ${UV_MIN_VERSION}, else uv ${UV_VERSION})
  --log FILE           Also write all output to FILE
  -h, --help           Show this help

Environment variables: HMB_VENV, HMB_PYTHON_VERSION, HMB_TORCH_BACKEND, HMB_UV, HMB_UV_VERSION,
HMB_UV_DIR, HMB_CUDA_REDIST_VERSION, HMB_CUDA_ARCH (e.g. "8.0;8.6;8.9;9.0" when building for other
GPUs), HMB_BUILD_JOBS, HMB_BUILD_TMPDIR, HMB_PYTORCH3D_REVISION; uv's own (UV_CACHE_DIR, ...).
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --venv) VENV_DIR="$2"; shift 2 ;;
        --with-sam3) WITH_SAM3=1; shift ;;
        --force-recreate) FORCE_RECREATE=1; shift ;;
        --rebuild-native) REBUILD_NATIVE=1; shift ;;
        --no-strip) STRIP=0; shift ;;
        --skip-smoke-tests) SKIP_SMOKE_TESTS=1; shift ;;
        --print-uv) PRINT_UV=1; shift ;;
        --uv) UV="$2"; shift 2 ;;
        --log) LOG_FILE="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[ERROR] Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ -n "${LOG_FILE}" ]]; then
    exec > >(tee -a "${LOG_FILE}") 2>&1
fi

case "${VENV_DIR}" in
    /*) ;;
    *) VENV_DIR="$(pwd)/${VENV_DIR}" ;;
esac
PY="${VENV_DIR}/bin/python"

START_SECONDS="${SECONDS}"
STEP_NAMES=()     # titles and durations of the finished steps, for the summary
STEP_TIMES=()
TMP_ROOT=""       # temporary build directory, deleted on exit

log() {
    printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

fail() {
    echo "[ERROR] $*" >&2
    exit 1
}

CURRENT_STEP=""

on_exit() {   # EXIT trap: delete the temporary build directory; name the step that failed
    local code=$?
    if [[ -n "${TMP_ROOT}" && -d "${TMP_ROOT}" ]]; then
        rm -rf "${TMP_ROOT}"
    fi
    if [[ "${code}" -ne 0 && -n "${CURRENT_STEP}" ]]; then
        echo "[ERROR] step failed: ${CURRENT_STEP} (exit status ${code}). Fix the cause and run the" \
             "script again: finished steps are skipped." >&2
    fi
}
trap on_exit EXIT
trap 'exit 130' INT TERM

# step TITLE FUNCTION: runs FUNCTION, logs its wall time and keeps it for the summary.
step() {
    local title="$1" started="${SECONDS}"
    shift
    printf '\n[%s] ==> %s\n' "$(date '+%H:%M:%S')" "${title}"
    CURRENT_STEP="${title}"
    "$@"
    CURRENT_STEP=""
    local elapsed=$(( SECONDS - started ))
    STEP_NAMES+=("${title}")
    STEP_TIMES+=("${elapsed}")
    log "done: ${title} (${elapsed} s)"
}

version_ge() {   # version_ge A B: A >= B
    [[ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -n 1)" == "$2" ]]
}

# The build and the installed environment must not see another environment's packages.
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE PYTHONSTARTUP

check_prerequisites() {   # tools, a supported gcc/g++, Linux x86-64, the lock and the pinned sources
    local cmd
    for cmd in curl tar sha256sum "${CC:-gcc}" "${CXX:-g++}"; do
        command -v "${cmd}" >/dev/null 2>&1 || fail "Required command not found: ${cmd} (C/C++ compiler: set CC/CXX)"
    done
    local gcc_major
    gcc_major="$("${CC:-gcc}" -dumpversion 2>/dev/null | cut -d. -f1 || true)"
    # CUDA 12.8 accepts host compilers up to gcc 14.
    [[ "${gcc_major}" =~ ^[0-9]+$ ]] && (( gcc_major >= 9 && gcc_major <= 14 )) \
        || fail "${CC:-gcc} ${gcc_major} found; CUDA ${CUDA_REDIST_VERSION} needs gcc 9-14 (set CC/CXX)"
    log "host compiler: $("${CXX:-g++}" --version | head -n 1)"
    command -v ffmpeg >/dev/null 2>&1 || echo "[INFO] ffmpeg not found: only the per-person videos of humanscore_multi.py use it" \
        "(./install.sh then adds imageio-ffmpeg; otherwise they fall back to OpenCV's encoder)" >&2
    command -v nvidia-smi >/dev/null 2>&1 || echo "[WARN] nvidia-smi not found: the pipeline needs an NVIDIA GPU" >&2
    [[ "$(uname -m)" == x86_64 ]] || fail "only Linux x86-64 is supported (the lock pins x86-64 wheels)"
    [[ -f "${LOCK}" ]] || fail "lock file missing: ${LOCK}"
    local source
    for source in HADM/setup.py torch-mesh-isect/setup.py GVHMR/tools/demo/demo.py; do
        [[ -f "${REPO_ROOT}/${source}" ]] || fail "Pinned source missing: ${source}. Run: bash install/check_sources.sh --init"
    done
}

ensure_uv() {   # sets UV: --uv, a recent enough uv (PATH or private copy), else downloads the pinned one
    if [[ -z "${UV}" ]]; then
        local candidate
        for candidate in "$(command -v uv 2>/dev/null || true)" "${UV_DIR}/uv"; do
            [[ -n "${candidate}" && -x "${candidate}" ]] || continue
            if version_ge "$("${candidate}" --version | awk '{print $2}')" "${UV_MIN_VERSION}"; then
                UV="${candidate}"
                break
            fi
        done
    fi
    if [[ -z "${UV}" ]]; then
        local url="https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-x86_64-unknown-linux-gnu.tar.gz"
        local download expected
        download="$(mktemp -d)"
        log "downloading uv ${UV_VERSION} (official release) into ${UV_DIR}"
        curl -fsSL --retry 3 -o "${download}/uv.tar.gz" "${url}"
        # The pinned release is checked against the checksum above; another HMB_UV_VERSION against
        # the checksum file published with it.
        if [[ "${UV_VERSION}" == "${UV_PINNED_VERSION}" ]]; then
            expected="${UV_PINNED_SHA256}"
        else
            expected="$(curl -fsSL --retry 3 "${url}.sha256" | awk '{print $1}')"
        fi
        echo "${expected}  ${download}/uv.tar.gz" | sha256sum -c --quiet - || fail "uv checksum mismatch"
        tar -xzf "${download}/uv.tar.gz" -C "${download}"
        mkdir -p "${UV_DIR}"
        install -m 755 "${download}/uv-x86_64-unknown-linux-gnu/uv" "${UV_DIR}/uv"
        rm -rf "${download}"
        UV="${UV_DIR}/uv"
    fi
    [[ -x "${UV}" ]] || fail "uv not executable: ${UV}"
    log "uv: ${UV} ($("${UV}" --version))"
}

create_venv() {   # a new environment, or checks that the existing one has the expected Python
    if [[ -e "${VENV_DIR}" && "${FORCE_RECREATE}" -eq 1 ]]; then
        [[ -f "${VENV_DIR}/pyvenv.cfg" ]] || fail "${VENV_DIR} exists but is not a virtual environment; not deleting it"
        log "removing existing environment ${VENV_DIR} (--force-recreate)"
        rm -rf "${VENV_DIR}"
    fi
    # uv-managed CPython (python-build-standalone), shared by uv environments of this user; --no-bin:
    # no python3.11 executable is added to ~/.local/bin.
    "${UV}" python install --no-bin "${PYTHON_VERSION}"
    if [[ -e "${VENV_DIR}" ]]; then
        [[ -x "${PY}" && -f "${VENV_DIR}/pyvenv.cfg" ]] || fail "${VENV_DIR} exists but is not a virtual environment (use another --venv)"
        local found
        found="$("${PY}" -c 'import platform; print(platform.python_version())')"
        [[ "${found}" == "${PYTHON_VERSION}" ]] \
            || fail "${VENV_DIR} has Python ${found}, expected ${PYTHON_VERSION}; use --force-recreate"
        log "using existing environment ${VENV_DIR} (Python ${found})"
        return
    fi
    "${UV}" venv --python "${PYTHON_VERSION}" --managed-python --prompt humanscore "${VENV_DIR}"
    # Marks the environment as created here (bash install.sh --uninstall deletes only such environments).
    printf 'created by install/install_slim.sh on %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" > "${VENV_DIR}/.humanscore-env"
}

install_lock() {
    # A complete lock with hashes: --no-deps installs exactly these distributions (no resolver pass),
    # so the exclusions of requirements/slim-excludes.txt are not pulled back in.
    "${UV}" pip install --python "${PY}" --no-deps --require-hashes --torch-backend "${TORCH_BACKEND}" \
        -r "${LOCK}"
}

install_sam3() {
    "${UV}" pip install --python "${PY}" --no-deps --require-hashes -r "${SAM3_LOCK}"
}

install_pytorch3d() {
    # No PyTorch3D wheel exists for PyTorch 2.8; GVHMR uses only its Python rotation transforms.
    PYTORCH3D_NO_EXTENSION=1 "${UV}" pip install --python "${PY}" --no-deps --no-build-isolation \
        "pytorch3d @ https://github.com/facebookresearch/pytorch3d/archive/${PYTORCH3D_REVISION}.tar.gz"
}

# The native extensions are rebuilt only when something they are compiled from changes: PyTorch,
# CUDA, GPU architectures, compiler flags, the HADM / torch-mesh-isect revisions or build_native.py.
native_stamp() {
    local torch_version hadm mesh
    torch_version="$("${PY}" -c 'import importlib.metadata as m; print(m.version("torch"))')"
    hadm="$(git -C "${REPO_ROOT}/HADM" rev-parse HEAD 2>/dev/null || echo unknown)"
    mesh="$(git -C "${REPO_ROOT}/torch-mesh-isect" rev-parse HEAD 2>/dev/null || echo unknown)"
    printf 'torch=%s cuda=%s arch=%s cflags=%s hadm=%s mesh=%s build_native=%s\n' "${torch_version}" \
        "${CUDA_REDIST_VERSION}" "${CUDA_ARCH:-visible}" "${NATIVE_CFLAGS}" "${hadm}" "${mesh}" \
        "$(sha256sum "${SCRIPT_DIR}/build_native.py" | cut -c1-16)"
}

build_native() {
    local stamp_file="${VENV_DIR}/.humanscore-native-build"
    local stamp
    stamp="$(native_stamp)"
    if [[ "${REBUILD_NATIVE}" -eq 0 && -f "${stamp_file}" && "$(cat "${stamp_file}")" == "${stamp}" ]] \
        && "${PY}" -c 'import torch, detectron2._C, bvh_cuda' >/dev/null 2>&1; then
        log "Detectron2/HADM and torch-mesh-isect already built (${stamp_file}); --rebuild-native rebuilds them"
        return
    fi
    mkdir -p "${BUILD_TMPDIR}"
    TMP_ROOT="$(mktemp -d "${BUILD_TMPDIR%/}/humanscore-build.XXXXXX")"
    log "temporary build directory (deleted afterwards): ${TMP_ROOT}"
    # ninja only for the build (PyTorch's extension builder); removed below.
    "${UV}" pip install --python "${PY}" "ninja==${NINJA_VERSION}"
    local arch_args=()
    [[ -n "${CUDA_ARCH}" ]] && arch_args=(--cuda-arch "${CUDA_ARCH}")
    # -g0: uv's CPython (python-build-standalone) compiles extensions with -g; the debug
    # information would add about 28 MB to the two extensions.
    CC="${CC:-gcc}" CXX="${CXX:-g++}" CFLAGS="${CFLAGS:+${CFLAGS} }${NATIVE_CFLAGS}" \
        "${PY}" "${SCRIPT_DIR}/build_native.py" \
        --python "${PY}" --uv "${UV}" \
        --cuda-root "${TMP_ROOT}/cuda-${CUDA_REDIST_VERSION}" --cuda-version "${CUDA_REDIST_VERSION}" --fetch-cuda \
        --build-dir "${TMP_ROOT}/build" --jobs "${BUILD_JOBS}" ${arch_args[@]+"${arch_args[@]}"}
    "${UV}" pip uninstall --python "${PY}" ninja
    "${PY}" -c 'import torch, detectron2._C, bvh_cuda' || fail "native extensions do not import"
    echo "${stamp}" > "${stamp_file}"
    rm -rf "${TMP_ROOT}"
    TMP_ROOT=""
}

# Files inside installed distributions that no HumanScore workload reads: build headers, test
# suites, and CUDA libraries that nothing loads. No library in the environment names the NVIDIA
# libraries below, except libcudnn.so.9, which loads libcudnn_adv (cuDNN's RNN, CTC-loss and legacy
# multi-head-attention API) only on first use of those operations; none of the models calls them.
# `uv pip install --reinstall PKG` restores a distribution's files. PyTorch's headers stay: the
# native extensions are built against them.
STRIP_FILES=(
    tensorflow/include                               # TensorFlow C++ headers (custom-op builds)
    nvidia/cudnn/lib/libcudnn_adv.so.9               # cuDNN RNN / CTC / legacy multi-head attention
    nvidia/cusolver/lib/libcusolverMg.so.11          # multi-GPU dense solver
    nvidia/cuda_nvrtc/lib/libnvrtc.alt.so.12         # alternative NVRTC build
    nvidia/cuda_nvrtc/lib/libnvrtc-builtins.alt.so.12.8
    nvidia/cuda_nvcc/nvvm/lib64/libnvvm.so           # NVVM compiler library (XLA and TensorFlow use ptxas)
    nvidia/cuda_cupti/lib/libnvperf_host.so          # CUPTI profiler (PM sampling, range profiling)
    nvidia/cuda_cupti/lib/libnvperf_target.so
    nvidia/cuda_cupti/lib/libpcsamplingutil.so
    nvidia/cuda_cupti/lib/libcheckpoint.so
)
STRIP_TESTS=(numpy scipy pandas sympy matplotlib networkx)   # their tests/ directories

strip_unused_files() {   # delete STRIP_FILES and STRIP_TESTS; the list is kept in .humanscore-stripped
    local site
    site="$("${PY}" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
    local record="${VENV_DIR}/.humanscore-stripped"
    local paths=() relative package
    for relative in "${STRIP_FILES[@]}"; do
        [[ -e "${site}/${relative}" ]] && paths+=("${site}/${relative}")
    done
    for package in "${STRIP_TESTS[@]}"; do
        [[ -d "${site}/${package}" ]] || continue
        while IFS= read -r -d '' dir; do
            paths+=("${dir}")
        done < <(find "${site}/${package}" -type d -name tests -prune -print0)
    done
    if [[ ${#paths[@]} -eq 0 ]]; then
        log "nothing to strip (already stripped; list: ${record})"
        return
    fi
    local bytes
    bytes="$(du -scb "${paths[@]}" | tail -n 1 | cut -f1)"
    printf '%s\n' "${paths[@]#"${site}/"}" >> "${record}"
    rm -rf "${paths[@]}"
    log "removed ${#paths[@]} files and directories, $(( bytes / 1000000 )) MB (list: ${record})"
}

run_smoke_tests() {
    bash "${SCRIPT_DIR}/smoke_test.sh" --python "${PY}"
}

print_summary() {
    local elapsed=$(( SECONDS - START_SECONDS )) i
    echo
    echo "============================================================"
    echo "HumanScore lightweight environment ready in $(( elapsed / 60 )) min $(( elapsed % 60 )) s"
    echo "============================================================"
    for i in "${!STEP_NAMES[@]}"; do
        printf '  %5d s  %s\n' "${STEP_TIMES[$i]}" "${STEP_NAMES[$i]}"
    done
    echo
    # Decimal GB (du -sb), the unit of the documented sizes; du -sh would print GiB (9.3 GB as 8.7G).
    echo "Environment: ${VENV_DIR} ($(du -sb "${VENV_DIR}" 2>/dev/null | awk '{ printf "%.1f GB", $1 / 1e9 }'))"
    echo "Python:      $("${PY}" -c 'import sys; print(sys.version.split()[0], "at", sys.base_prefix)')"
    echo "uv:          ${UV}"
    if [[ "$(command -v uv 2>/dev/null || true)" != "${UV}" ]]; then
        echo "             (not the uv on PATH: where install/*.md says 'uv', run ${UV})"
    fi
    echo "uv cache:    $("${UV}" cache dir) (outside the environment; '${UV} cache clean' frees it)"
    if [[ "${WITH_SAM3}" -eq 0 ]]; then
        echo "SAM 3.1:     not installed (--with-sam3); humanscore_multi.py --tracker yolo needs no extra setup"
    fi
    # ./install.sh continues with the model files and doctor itself.
    [[ -n "${HMB_INSTALL_PARENT:-}" ]] && return 0
    cat <<EOF

Next steps:
  source ${VENV_DIR}/bin/activate
  python -m hmb.cli doctor     # exit status 0 = ready; model files: python3 install/fetch_assets.py fetch
EOF
}

main() {   # the steps of the header, in order
    if [[ "${PRINT_UV}" -eq 1 ]]; then
        ensure_uv >&2
        echo "${UV}"
        return
    fi
    log "HumanScore lightweight installation into ${VENV_DIR}"
    step "prerequisites" check_prerequisites
    step "uv" ensure_uv
    step "Python ${PYTHON_VERSION} and virtual environment" create_venv
    step "locked distributions (requirements/lock-slim.txt)" install_lock
    if [[ "${WITH_SAM3}" -eq 1 ]]; then
        step "SAM 3.1 (requirements/lock-slim-sam3.txt)" install_sam3
    fi
    step "PyTorch3D ${PYTORCH3D_REVISION:0:7} (Python operators)" install_pytorch3d
    step "Detectron2/HADM and torch-mesh-isect (CUDA ${CUDA_REDIST_VERSION})" build_native
    if [[ "${STRIP}" -eq 1 ]]; then
        step "strip headers and test suites" strip_unused_files
    fi
    if [[ "${SKIP_SMOKE_TESTS}" -eq 0 ]]; then
        step "smoke tests" run_smoke_tests
    fi
    print_summary
}

main "$@"
