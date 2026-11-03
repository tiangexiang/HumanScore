#!/usr/bin/env bash
# HumanScore installer: one command from a fresh clone to a working installation.
#
#   bash install.sh [--with-sam3] [--self-test] [--smplx-dir PATH] [options]     (--help lists them)
#
# Steps (each prints its duration; a re-run skips the finished ones):
#   preflight    machine checks with actionable errors: Linux x86-64, NVIDIA driver and GPU, disk,
#                tools (git, curl, tar, gcc/g++ 9-14), ffmpeg, network; nothing is changed
#   sources      pinned GVHMR, HADM and torch-mesh-isect sources (install/check_sources.sh --init)
#   environment  Python environment (install/install_slim.sh: uv, CPython 3.11.18, hash-pinned
#                requirements/lock-slim.txt, PyTorch3D, Detectron2/HADM and torch-mesh-isect built
#                with the system compiler and a temporary CUDA 12.8.1 compiler)
#   ffmpeg       only without ffmpeg on PATH: imageio-ffmpeg (requirements/lock-ffmpeg.txt)
#   smoke        imports, GPU access of PyTorch/TensorFlow/JAX, CLI entry points
#   assets       model files of install/assets.json from their official sources: resumable,
#                SHA-256 verified, placed where the code reads them (downloaded in the background
#                while the environment is built when a system python3 is available)
#   smplx        the licensed SMPL-X model: found, or taken from --smplx-dir; never downloaded
#   sam3         with --with-sam3: Hugging Face login and gated access, SAM 3.1 checkpoint
#                (never fails the installation)
#   doctor       python -m hmb.cli doctor
#   selftest     with --self-test: scores a short public clip with scripts/humanscore_single.py
#
# No sudo; shell start-up files are never edited. The full log is written to
# outputs/install/logs/. Exit status: 0 ready; 3 almost ready (the SMPL-X model, or model files
# skipped with --skip-assets, are still missing; the summary says what to do); 1 a step failed
# (the message names it and the log); 2 usage error. Details: README.md (Install),
# install/INSTALL.md, install/TROUBLESHOOTING.md.

# `sh install.sh` would stop at the next line with "Illegal option -o pipefail".
if [ -z "${BASH_VERSION:-}" ]; then echo "[ERROR] install.sh needs bash: run  bash install.sh $*" >&2; exit 2; fi
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="${REPO_ROOT}/install"
STATE_DIR="${REPO_ROOT}/outputs/install"
LOG_DIR="${STATE_DIR}/logs"
FETCH="${INSTALL_DIR}/fetch_assets.py"

PYTHON_VERSION="${HMB_PYTHON_VERSION:-3.11.18}"
UV_VERSION="${HMB_UV_VERSION:-0.13.6}"
# NVIDIA driver: CUDA 12.x runs on >= 525.60.13 through minor-version compatibility; the native
# driver branch of CUDA 12.8, which the PyTorch wheels are built with, is R570 (>= 570.26).
MIN_DRIVER="525.60.13"
CUDA128_DRIVER="570.26"
MIN_COMPUTE="7.0"                  # PyTorch 2.8 cu128 wheels: sm_70 to sm_120
MIN_GLIBC="2.28"                   # the lock pins manylinux_2_28 wheels
# Disk space of a finished installation (decimal GB), used by the preflight disk check.
ENV_GB=8.7; ENV_SAM3_GB=9.3; UV_PYTHON_GB=0.1; BUILD_GB=1.5
UV_CACHE_SEPARATE_GB=10.1          # uv cache on another file system than the environment: a copy
UV_CACHE_SAME_GB=1.2               # same file system: the environment hard-links the cache's files
MARGIN_GB=2

# Option defaults (see usage below).
WITH_SAM3=0; SELF_TEST=0; SKIP_ASSETS=0; DRY_RUN=0; UNINSTALL=0; REMOVE_ASSETS=0; REINSTALL=0; YES=0
PREFIX="${REPO_ROOT}/.venv-humanscore"
CUDA_ARCH_OPT=""
SMPLX_SOURCE=""
JOBS=4
LOG_FILE=""
declare -A FORCE_STEPS=()
STEP_NAMES="sources environment ffmpeg smoke assets smplx sam3 doctor selftest"

usage() {
    cat <<EOF
Usage: bash install.sh [options]

Installs HumanScore into a Python environment (default ./.venv-humanscore), downloads the model
files, and checks the result. Re-running it skips finished steps and resumes interrupted downloads.

Options:
  --with-sam3          also install SAM 3.1, the default tracker of scripts/humanscore_multi.py
                       (gated on Hugging Face: request access at huggingface.co/facebook/sam3.1)
  --self-test          finish by scoring a short public clip (a few minutes)
  --smplx-dir PATH     the SMPL-X model: SMPLX_NEUTRAL.npz, a folder containing it, or the SMPL-X
                       v1.1 zip from smpl-x.is.tue.mpg.de (models_smplx_v1_1.zip)
  --smpl-dir PATH      the same (a body-model folder with smplx/ in it); SMPL itself is not used
  --prefix PATH        environment location (default: ${PREFIX#"${REPO_ROOT}/"})
  --cuda-arch LIST     GPU architectures for the CUDA extensions, e.g. "8.0" or "8.0;8.6;8.9;9.0"
                       (default: the GPUs of this machine; needed when building without a GPU)
  --skip-assets        do not download model files (also skips --self-test)
  --jobs N             parallel model downloads (default ${JOBS})
  --dry-run            run the checks and print the plan; change nothing
  -y, --yes            non-interactive: no questions (servers, CI); also when stdin is not a terminal
  --force-step NAME    redo a finished step (repeatable): ${STEP_NAMES// /, }
                       (selftest and sam3 imply --self-test and --with-sam3)
  --reinstall          recreate the environment and redo every step (downloaded files are
                       re-verified, not downloaded again)
  --uninstall          remove the environment and the installer's own files (state, private uv,
                       partial downloads); keeps model files, the SMPL-X model and your results
  --remove-assets      with --uninstall: also delete the downloaded model files
  --log FILE           log file (default: outputs/install/logs/install-DATE.log)
  -h, --help           this help

Environment variables: CC/CXX (host compiler), TMPDIR or HMB_BUILD_TMPDIR (build space),
UV_CACHE_DIR (uv cache), HF_TOKEN / HF_HOME (Hugging Face), https_proxy, and the HMB_* variables of
install/install_slim.sh. Exit status: 0 ready, 3 almost ready, 1 failed, 2 usage error.
EOF
}

die_usage() { echo "[ERROR] $*" >&2; echo "Run: bash install.sh --help" >&2; exit 2; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --with-sam3) WITH_SAM3=1; shift ;;
        --self-test) SELF_TEST=1; shift ;;
        --smplx-dir|--smpl-dir) [[ $# -ge 2 ]] || die_usage "$1 needs a path"; SMPLX_SOURCE="$2"; shift 2 ;;
        --prefix) [[ $# -ge 2 ]] || die_usage "--prefix needs a path"; PREFIX="$2"; shift 2 ;;
        --cuda-arch) [[ $# -ge 2 ]] || die_usage "--cuda-arch needs a value"; CUDA_ARCH_OPT="$2"; shift 2 ;;
        --skip-assets) SKIP_ASSETS=1; shift ;;
        --jobs) [[ $# -ge 2 && "$2" =~ ^[1-9][0-9]*$ ]] || die_usage "--jobs needs a positive number"; JOBS="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -y|--yes) YES=1; shift ;;
        --force-step)
            [[ $# -ge 2 && " ${STEP_NAMES} " == *" $2 "* ]] || die_usage "--force-step takes one of: ${STEP_NAMES}"
            FORCE_STEPS[$2]=1
            # Redoing an opt-in step also requests it.
            [[ "$2" == selftest ]] && SELF_TEST=1
            [[ "$2" == sam3 ]] && WITH_SAM3=1
            shift 2 ;;
        --reinstall) REINSTALL=1; shift ;;
        --uninstall) UNINSTALL=1; shift ;;
        --remove-assets) REMOVE_ASSETS=1; shift ;;
        --log) [[ $# -ge 2 ]] || die_usage "--log needs a file"; LOG_FILE="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) die_usage "unknown option: $1" ;;
    esac
done
[[ -z "${CUDA_ARCH_OPT}" || "${CUDA_ARCH_OPT}" =~ ^[0-9]+\.[0-9]+(\+PTX)?(\;[0-9]+\.[0-9]+(\+PTX)?)*$ ]] \
    || die_usage "--cuda-arch: expected values like 8.0 or \"8.0;8.6\", got '${CUDA_ARCH_OPT}'"
(( REMOVE_ASSETS == 0 || UNINSTALL == 1 )) || die_usage "--remove-assets is an option of --uninstall"
case "${PREFIX}" in /*) ;; *) PREFIX="$(pwd)/${PREFIX}" ;; esac
PREFIX="${PREFIX%/}"
PY="${PREFIX}/bin/python"
# Stamps of finished steps, per environment location (several --prefix can share a checkout).
STAMP_DIR="${STATE_DIR}/stamps/$(basename "${PREFIX}")-$(printf '%s' "${PREFIX}" | sha256sum | cut -c1-8)"
case "${SMPLX_SOURCE}" in ""|/*) ;; *) SMPLX_SOURCE="$(pwd)/${SMPLX_SOURCE}" ;; esac

# Questions only with a terminal on both ends and without --yes.
INTERACTIVE=0
[[ "${YES}" -eq 0 && -t 0 && -t 1 ]] && INTERACTIVE=1
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE PYTHONSTARTUP

# ----------------------------------------------------------------------------- output helpers
# info/warn/fail_check print one preflight line; warnings and failures are also collected for the
# summary (a failed check stops the installation before anything is changed).
log()  { printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*"; }
info() { printf '  %-6s %s\n' "$1" "$2"; }
warn() { printf '  %-6s %s\n' "WARN" "$*"; WARNINGS+=("$*"); }
fail_check() { printf '  %-6s %s\n' "FAIL" "$*"; FAILURES+=("$*"); }
die() { echo "[ERROR] $*" >&2; exit 1; }
version_ge() { [[ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -n 1)" == "$2" ]]; }   # $1 >= $2
gb() { awk -v b="$1" 'BEGIN { printf "%.1f GB", b / 1e9 }'; }
# Decimal GB of du -sb, the unit of the documented sizes (du -sh prints GiB: 9.3 GB shows as 8.8G).
size_gb() { local bytes; bytes="$(du -sb "$@" 2>/dev/null | tail -n 1 | cut -f1)"; [[ -n "${bytes}" ]] && gb "${bytes}" || echo "?"; }
to_bytes() { awk -v g="$1" 'BEGIN { printf "%.0f", g * 1e9 }'; }
WARNINGS=(); FAILURES=(); STEP_RESULT=""

# ----------------------------------------------------------------------------- failure handling
# On failure the EXIT trap names the step, the failed command and the log, and explains how to
# resume. FINISHED=1 marks paths that report their own outcome (summary, dry run, usage errors).
CURRENT_STEP="startup"; FAILED_COMMAND=""; FAILED_LINE=""; PREFETCH_PID=""; TEE_PID=""; FINISHED=0
CHILD_PID=""; CHILD_COMMAND=""
on_err() {   # ERR trap: remember the failed command (of run_child, not its `return`) and its line
    FAILED_COMMAND="${BASH_COMMAND}"
    # shellcheck disable=SC2016
    [[ "${FAILED_COMMAND}" == 'return "${rc}"' && -n "${CHILD_COMMAND}" ]] && FAILED_COMMAND="${CHILD_COMMAND}"
    FAILED_LINE="${BASH_LINENO[0]:-?}"
}

# bash runs a trap only after the foreground command has ended, so a SIGTERM to install.sh alone
# (kill PID, timeout, a CI job being cancelled) would wait for the whole step. The long commands
# therefore run in the background while install.sh waits for them (wait returns on a signal), and
# on exit their process trees are stopped. Asynchronous commands ignore SIGINT, so Ctrl-C reaches
# them through this as well.
run_child() {   # run_child CMD...: CMD with stdin from /dev/null; returns its exit status
    CHILD_COMMAND="$*"
    "$@" < /dev/null &
    CHILD_PID=$!
    local rc=0
    wait "${CHILD_PID}" || rc=$?
    CHILD_PID=""
    return "${rc}"
}
descendants() {   # descendants PID: every process below PID
    ps -eo pid=,ppid= 2>/dev/null | awk -v root="$1" '
        { parent[$1] = $2 }
        END { for (p in parent) { q = parent[p]; while (q != root && (q in parent) && q > 1) q = parent[q]; if (q == root) print p } }'
}
stop_tree() {   # stop_tree PID: SIGTERM to PID and its descendants; SIGKILL after 15 s
    local root="$1" pids i
    kill -0 "${root}" 2>/dev/null || { wait "${root}" 2>/dev/null || true; return 0; }
    pids="$(descendants "${root}" | tr '\n' ' ')"
    # shellcheck disable=SC2086
    kill -TERM "${root}" ${pids} 2>/dev/null || true
    for i in $(seq 1 75); do kill -0 "${root}" 2>/dev/null || break; sleep 0.2; done
    if kill -0 "${root}" 2>/dev/null; then
        pids="$(descendants "${root}" | tr '\n' ' ')"
        # shellcheck disable=SC2086
        kill -KILL "${root}" ${pids} 2>/dev/null || true
    fi
    wait "${root}" 2>/dev/null || true
}
on_exit() {   # EXIT trap: stop child processes, explain a failure, flush the log
    local code=$?
    trap - ERR
    trap '' INT TERM HUP   # a second Ctrl-C must not cut the clean-up short
    [[ -n "${CHILD_PID}" ]] && stop_tree "${CHILD_PID}"
    [[ -n "${PREFETCH_PID}" ]] && stop_tree "${PREFETCH_PID}"
    if [[ "${FINISHED}" -eq 0 && "${code}" -ne 0 && "${code}" -ne 2 && "${code}" -ne 3 ]]; then
        echo
        if [[ "${code}" -eq 130 ]]; then
            echo "[INTERRUPTED] during step '${CURRENT_STEP}'." >&2
        else
            echo "[ERROR] step '${CURRENT_STEP}' failed (exit status ${code})." >&2
            [[ -n "${FAILED_COMMAND}" ]] && echo "        command (install.sh line ${FAILED_LINE}): ${FAILED_COMMAND}" >&2
        fi
        [[ -n "${LOG_FILE}" ]] && echo "        full log: ${LOG_FILE}" >&2
        echo "        Run the same command again to resume: finished steps are skipped and downloads" >&2
        if [[ " ${STEP_NAMES} " == *" ${CURRENT_STEP} "* ]]; then
            echo "        continue where they stopped (--force-step ${CURRENT_STEP} redoes this step)." >&2
        else
            echo "        continue where they stopped." >&2
        fi
        echo "        Common causes and fixes: install/TROUBLESHOOTING.md" >&2
    fi
    if [[ -n "${TEE_PID}" ]]; then
        exec 1>&- 2>&-
        wait "${TEE_PID}" 2>/dev/null || true
    fi
}
trap on_err ERR
trap on_exit EXIT
trap 'exit 130' INT TERM HUP

# ----------------------------------------------------------------------------- shared lookups
find_uv() {   # path of a uv >= UV_VERSION: $HMB_UV, uv on PATH, or install_slim.sh's private copy
    local candidate
    for candidate in "${HMB_UV:-}" "$(command -v uv 2>/dev/null || true)" \
                     "${HMB_UV_DIR:-${XDG_DATA_HOME:-${HOME}/.local/share}/humanscore/uv-${UV_VERSION}}/uv"; do
        [[ -n "${candidate}" && -x "${candidate}" ]] || continue
        if version_ge "$("${candidate}" --version 2>/dev/null | awk '{print $2}')" "${UV_VERSION}"; then
            echo "${candidate}"; return 0
        fi
    done
    return 1
}

uv_cache_dir() {   # where uv keeps its download cache (for the disk check and the summary)
    if [[ -n "${UV_CACHE_DIR:-}" ]]; then echo "${UV_CACHE_DIR}"; return; fi
    local uv; uv="$(find_uv || true)"
    if [[ -n "${uv}" ]]; then "${uv}" cache dir 2>/dev/null && return; fi
    echo "${XDG_CACHE_HOME:-${HOME}/.cache}/uv"
}

# File-system helpers for paths that may not exist yet (measured at their nearest existing parent).
existing_parent() { local p="$1"; while [[ ! -e "${p}" ]]; do p="$(dirname "${p}")"; done; echo "${p}"; }
fs_device() { stat -c %d "$(existing_parent "$1")"; }
fs_free() { df -P -B1 "$(existing_parent "$1")" | awk 'NR == 2 { print $4 }'; }

# A Python for the asset tools before the environment exists (standard library only, >= 3.8).
SYS_PY=""
for candidate in python3 python; do
    if command -v "${candidate}" >/dev/null 2>&1 \
        && "${candidate}" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)' 2>/dev/null; then
        SYS_PY="$(command -v "${candidate}")"; break
    fi
done
tool_python() {   # the environment's Python when it exists, else the system one
    if [[ -x "${PY}" ]]; then echo "${PY}"; elif [[ -n "${SYS_PY}" ]]; then echo "${SYS_PY}"; else return 1; fi
}

# ----------------------------------------------------------------------------- stamps
# A finished step writes STAMP_DIR/NAME: line 1 a key over its inputs (options, lock files,
# installer scripts), line 2 the date. A re-run skips the step while the key is unchanged and the
# files it produced are still there (step_valid).
key_of() {   # key_of VALUE... -- FILE...: hash of values and file contents
    {
        while [[ $# -gt 0 && "$1" != "--" ]]; do printf '%s\n' "$1"; shift; done
        [[ $# -gt 0 ]] && shift
        local file
        for file in "$@"; do [[ -f "${file}" ]] && sha256sum "${file}" | cut -c1-64 || echo "missing ${file}"; done
    } | sha256sum | cut -c1-32
}

ENV_KEY=""; ASSETS_KEY=""
compute_keys() {   # keys of the environment (and smoke/ffmpeg) steps and of the model-file steps
    local cc_version
    cc_version="$("${CC:-gcc}" -dumpfullversion 2>/dev/null || "${CC:-gcc}" -dumpversion 2>/dev/null || echo none)"
    ENV_KEY="$(key_of "prefix=${PREFIX}" "sam3=${WITH_SAM3}" "arch=${CUDA_ARCH}" "cc=${cc_version}" \
        "python=${PYTHON_VERSION}" -- "${REPO_ROOT}/requirements/lock-slim.txt" \
        "${REPO_ROOT}/requirements/lock-slim-sam3.txt" "${INSTALL_DIR}/install_slim.sh" "${INSTALL_DIR}/build_native.py")"
    ASSETS_KEY="$(key_of "root=${REPO_ROOT}" -- "${INSTALL_DIR}/assets.json" "${FETCH}")"
}

selftest_key() {   # the self-test depends on the environment, the model files and self_test.py
    key_of "${ENV_KEY}" "${ASSETS_KEY}" -- "${INSTALL_DIR}/self_test.py"; }

stamp_ok() {   # stamp_ok NAME KEY
    local file="${STAMP_DIR}/$1"
    [[ -z "${FORCE_STEPS[$1]:-}" && -f "${file}" && "$(head -n 1 "${file}")" == "$2" ]]
}

step_valid() {   # quick re-check of a stamped step: the files it produced are still there
    case "$1" in
        environment) [[ -x "${PY}" && -f "${PREFIX}/pyvenv.cfg" && -f "${PREFIX}/.humanscore-native-build" ]] ;;
        smoke) [[ -x "${PY}" ]] ;;
        ffmpeg) "${PY}" -c 'import imageio_ffmpeg' >/dev/null 2>&1 ;;
        # --verified: a model file changed since it was verified (same size, other content) makes the
        # step run again, which re-hashes it and downloads it again if it differs.
        assets) "${PY}" -I "${FETCH}" check --groups core --verified >/dev/null 2>&1 ;;
        sam3) "${PY}" -I "${FETCH}" check --only sam3.1 --verified >/dev/null 2>&1 ;;
        selftest) grep -q '"passed": true' "${STATE_DIR}/self-test/basic_single_leg_squat/self_test.json" 2>/dev/null ;;
        *) return 0 ;;
    esac
}

STEP_REPORT=()
run_step() {   # run_step NAME TITLE FUNCTION [KEY]: KEY empty = always run
    local name="$1" title="$2" func="$3" key="${4:-}" started="${SECONDS}"
    CURRENT_STEP="${name}"
    if [[ -n "${key}" ]] && stamp_ok "${name}" "${key}" && step_valid "${name}"; then
        log "skip ${name}: done on $(sed -n 2p "${STAMP_DIR}/${name}") (--force-step ${name} redoes it)"
        STEP_REPORT+=("${name}|skipped (done before)|0")
        return 0
    fi
    printf '\n[%s] ==> %s: %s\n' "$(date '+%H:%M:%S')" "${name}" "${title}"
    rm -f "${STAMP_DIR}/${name}"
    "${func}"
    local elapsed=$(( SECONDS - started ))
    [[ -n "${key}" ]] && printf '%s\n%s\n' "${key}" "$(date '+%Y-%m-%d %H:%M:%S')" > "${STAMP_DIR}/${name}"
    STEP_REPORT+=("${name}|${STEP_RESULT:-done}|${elapsed}")
    STEP_RESULT=""
    log "done: ${name} (${elapsed} s)"
}

# ----------------------------------------------------------------------------- preflight
# Read-only checks of the machine. Each prints OK / NOTE / WARN / FAIL with what to do; any FAIL
# stops the installation before anything is changed (also the whole of --dry-run).
GPU_NAMES=""; CUDA_ARCH=""; NEED_FFMPEG=0
NO_GPU=0   # building with --cuda-arch where no GPU is usable: the smoke tests skip their GPU checks
preflight() {
    echo "Checks:"
    # Platform.
    if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
        fail_check "Linux x86-64 required (found $(uname -s) $(uname -m)): the lock pins Linux x86-64 wheels"
    else
        local glibc
        glibc="$(getconf GNU_LIBC_VERSION 2>/dev/null | awk '{print $2}')"
        if [[ -z "${glibc}" ]]; then
            warn "could not determine the glibc version (needs >= ${MIN_GLIBC} for manylinux_2_28 wheels)"
        elif version_ge "${glibc}" "${MIN_GLIBC}"; then
            info OK "Linux x86-64, glibc ${glibc}"
        else
            fail_check "glibc ${glibc} is older than ${MIN_GLIBC}: the PyTorch/TensorFlow wheels need a newer distribution (e.g. Ubuntu 20.04+, RHEL 8+)"
        fi
    fi
    local bash_major="${BASH_VERSINFO[0]}"
    (( bash_major >= 4 )) || fail_check "bash 4 or newer required (found ${BASH_VERSION})"

    # NVIDIA driver and GPUs.
    local smi_out
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        if [[ -n "${CUDA_ARCH_OPT}" ]]; then
            NO_GPU=1
            warn "nvidia-smi not found: building for --cuda-arch ${CUDA_ARCH_OPT} without GPU checks; HumanScore needs an NVIDIA GPU to run (re-run install.sh on a GPU machine to verify)"
        else
            fail_check "nvidia-smi not found: install the NVIDIA driver (>= ${CUDA128_DRIVER}, CUDA 12.8 branch R570, or >= ${MIN_DRIVER}), or pass --cuda-arch to build on a machine without GPU"
        fi
    elif ! smi_out="$(nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv,noheader,nounits 2>&1)"; then
        fail_check "nvidia-smi failed: ${smi_out%%$'\n'*} (driver not loaded, or driver/library mismatch: reboot after a driver update)"
    else
        local name driver cap mem caps=() low_mem=0 gpu_driver=""
        while IFS=, read -r name driver cap mem; do
            name="$(echo "${name}" | xargs)"; driver="$(echo "${driver}" | xargs)"; gpu_driver="${driver}"
            cap="$(echo "${cap}" | xargs)"; mem="$(echo "${mem}" | xargs)"
            GPU_NAMES+="${GPU_NAMES:+, }${name} (${mem} MiB, compute ${cap})"
            [[ "${cap}" =~ ^[0-9]+\.[0-9]+$ ]] && caps+=("${cap}")
            if [[ "${cap}" =~ ^[0-9]+\.[0-9]+$ ]] && ! version_ge "${cap}" "${MIN_COMPUTE}"; then
                fail_check "GPU ${name} has compute capability ${cap}; PyTorch 2.8 (cu128) needs ${MIN_COMPUTE} or newer"
            fi
            [[ "${mem}" =~ ^[0-9]+$ ]] && (( mem < 32000 )) && low_mem=1
        done <<< "${smi_out}"
        driver="${gpu_driver}"   # read clears its variables at the end of input
        if [[ ! "${driver}" =~ ^[0-9]+(\.[0-9]+)*$ ]]; then
            fail_check "could not read the NVIDIA driver version from nvidia-smi ('${driver}')"
        elif ! version_ge "${driver}" "${MIN_DRIVER}"; then
            fail_check "NVIDIA driver ${driver} is too old for CUDA 12: install >= ${CUDA128_DRIVER} (R570, CUDA 12.8; minimum ${MIN_DRIVER})"
        elif ! version_ge "${driver}" "${CUDA128_DRIVER}"; then
            warn "NVIDIA driver ${driver} < ${CUDA128_DRIVER} (R570, CUDA 12.8): runs through CUDA minor-version compatibility, which is untested; update the driver if CUDA errors appear"
        else
            info OK "NVIDIA driver ${driver} (CUDA 12.8 needs >= ${CUDA128_DRIVER}); GPUs: ${GPU_NAMES}"
        fi
        if [[ -z "${CUDA_ARCH_OPT}" ]]; then
            CUDA_ARCH="$(printf '%s\n' ${caps[@]+"${caps[@]}"} | sort -u -V | paste -sd ';')"
            [[ -n "${CUDA_ARCH}" ]] || warn "could not read the compute capability; the extensions are built for the visible GPUs"
        fi
        if (( low_mem && WITH_SAM3 )); then
            warn "a GPU has less than 32 GB: SAM 3.1 needs about 23-27 GB for clips of up to 270 frames (humanscore_multi.py --tracker yolo needs less)"
        fi
        # nvidia-smi lists every GPU, but CUDA (PyTorch, JAX, TensorFlow) sees none with these values.
        local visible="${CUDA_VISIBLE_DEVICES-unset}"
        if [[ "${visible}" != unset && ( -z "${visible//[[:space:]]/}" || "${visible}" =~ ^[[:space:]]*-1 || "${visible}" == NoDevFiles ) ]]; then
            if [[ -n "${CUDA_ARCH_OPT}" ]]; then
                NO_GPU=1
                warn "CUDA_VISIBLE_DEVICES='${visible}' hides the GPUs: building for --cuda-arch ${CUDA_ARCH_OPT} without GPU checks"
            else
                fail_check "CUDA_VISIBLE_DEVICES='${visible}' hides every GPU from CUDA (nvidia-smi still lists them), so the smoke tests and HumanScore cannot use one: unset it or name a GPU (export CUDA_VISIBLE_DEVICES=0); to build without a GPU anyway, add --cuda-arch"
            fi
        fi
    fi
    [[ -n "${CUDA_ARCH_OPT}" ]] && CUDA_ARCH="${CUDA_ARCH_OPT}"
    [[ -n "${CUDA_ARCH}" ]] && info OK "CUDA extensions will be built for compute capability ${CUDA_ARCH}"
    compute_keys   # the environment's stamp depends on CUDA_ARCH

    # Tools.
    local tool missing=()
    for tool in git curl tar gzip sha256sum awk sort df stat; do
        command -v "${tool}" >/dev/null 2>&1 || missing+=("${tool}")
    done
    if [[ ${#missing[@]} -gt 0 ]]; then
        fail_check "missing tools: ${missing[*]} (Ubuntu/Debian: sudo apt-get install -y git curl tar coreutils; RHEL/Fedora: sudo dnf install -y git curl tar)"
    else
        info OK "git, curl, tar, coreutils"
    fi
    local cc="${CC:-gcc}" cxx="${CXX:-g++}" major
    if ! command -v "${cc}" >/dev/null 2>&1 || ! command -v "${cxx}" >/dev/null 2>&1; then
        fail_check "C/C++ compiler not found (${cc}/${cxx}). The two CUDA extensions are compiled with gcc/g++ 9-14:
           Ubuntu/Debian: sudo apt-get install -y build-essential;  RHEL/Fedora: sudo dnf install -y gcc gcc-c++
           HPC clusters: module load gcc/13 (or another 9-14), then re-run;  another compiler: CC=gcc-13 CXX=g++-13 bash install.sh
           Without sudo or modules: install gcc/g++ 9-14 in your home directory (for example the conda-forge packages
           gcc_linux-64 and gxx_linux-64) and pass it with CC=/path/to/gcc CXX=/path/to/g++ bash install.sh.
           (install.sh does not download a compiler itself: extensions built by a foreign gcc can need a newer
           libstdc++ than the system has.)"
    else
        major="$("${cc}" -dumpversion 2>/dev/null | cut -d. -f1)"
        if [[ "${major}" =~ ^[0-9]+$ ]] && (( major >= 9 && major <= 14 )); then
            info OK "compiler: $("${cxx}" --version | head -n 1)"
        else
            fail_check "${cc} version ${major} found; CUDA 12.8 needs gcc/g++ 9 to 14: install one and select it with CC=gcc-13 CXX=g++-13 bash install.sh"
        fi
    fi
    if command -v ffmpeg >/dev/null 2>&1 || [[ -n "${HMB_FFMPEG:-}" ]]; then
        info OK "ffmpeg: ${HMB_FFMPEG:-$(command -v ffmpeg)}"
    else
        NEED_FFMPEG=1
        info NOTE "ffmpeg not found: imageio-ffmpeg (a static ffmpeg, 30 MB) will be installed into the environment for the multi-person videos; single-person scoring does not need ffmpeg"
    fi
    if [[ -n "${SYS_PY}" ]]; then
        info OK "system Python for the model downloads: ${SYS_PY} (they run while the environment is built)"
    else
        info NOTE "no system python3 >= 3.8: model files are downloaded after the environment is built"
    fi

    # Pinned sources.
    if bash "${INSTALL_DIR}/check_sources.sh" >/dev/null 2>&1; then
        info OK "pinned sources: GVHMR, HADM, torch-mesh-isect"
    elif [[ -e "${REPO_ROOT}/.git" ]]; then
        if [[ -f "${REPO_ROOT}/.gitmodules" ]]; then
            info NOTE "pinned sources missing or at another revision: fetched in step 'sources' (git submodule update --init)"
        else
            fail_check "the repository has no .gitmodules: clone it with git (git clone --recursive URL) instead of copying files"
        fi
    else
        fail_check "not a git checkout: clone the repository with git (git clone --recursive URL); a downloaded archive lacks the pinned sources"
    fi

    check_network
    check_disk
}

check_network() {   # every host the remaining steps download from must answer (mirrors: a warning)
    local hosts=() host kind
    local env_pending=1
    stamp_ok environment "${ENV_KEY}" && step_valid environment && env_pending=0
    if (( env_pending )); then
        for host in pypi.org files.pythonhosted.org download.pytorch.org github.com raw.githubusercontent.com \
                    developer.download.nvidia.com; do hosts+=("${host}|required|environment"); done
    fi
    if (( ! SKIP_ASSETS )); then
        local groups="core"; (( SELF_TEST )) && groups+=",selftest"
        local tool; tool="$(tool_python || true)"
        if [[ -n "${tool}" ]]; then
            while IFS=$'\t' read -r host kind; do
                [[ -n "${host}" ]] || continue
                hosts+=("${host}|$([[ "${kind}" == primary ]] && echo required || echo optional)|model files")
            done < <("${tool}" -I "${FETCH}" hosts --groups "${groups}" 2>/dev/null)
        else
            for host in huggingface.co github.com www.dropbox.com omnomnom.vision.rwth-aachen.de; do
                hosts+=("${host}|required|model files"); done
        fi
        (( WITH_SAM3 )) && hosts+=("huggingface.co|optional|SAM 3.1")
    fi
    local seen=" " entry purpose level unreachable=() optional_down=()
    for entry in ${hosts[@]+"${hosts[@]}"}; do
        IFS='|' read -r host level purpose <<< "${entry}"
        [[ "${seen}" == *" ${host} "* ]] && continue
        seen+="${host} "
        if ! curl -sS -o /dev/null --max-time 20 --retry 2 -r 0-0 "https://${host}/" 2>/dev/null; then
            if [[ "${level}" == required ]]; then unreachable+=("${host} (${purpose})"); else optional_down+=("${host}"); fi
        fi
    done
    if [[ ${#unreachable[@]} -gt 0 ]]; then
        fail_check "cannot reach: ${unreachable[*]}. Check the connection; behind a proxy, export https_proxy=http://HOST:PORT (curl, uv and Python use it)"
    elif [[ "${seen}" != " " ]]; then
        info OK "network: $(echo ${seen} | tr ' ' ',' | sed 's/,/, /g')"
    fi
    [[ ${#optional_down[@]} -eq 0 ]] || warn "mirror hosts not reachable (only used if a primary source fails): ${optional_down[*]}"
}

check_disk() {   # free space per file system for the environment, uv cache, build and model files
    declare -A need=() where=()
    add_need() {   # add_need PATH BYTES
        local dev; dev="$(fs_device "$1")"
        need[${dev}]=$(( ${need[${dev}]:-0} + $2 ))
        where[${dev}]="${where[${dev}]:-$(existing_parent "$1")}"
    }
    local env_pending=1
    stamp_ok environment "${ENV_KEY}" && step_valid environment && env_pending=0
    if (( env_pending )) && [[ ! -f "${PREFIX}/.humanscore-native-build" ]]; then
        add_need "${PREFIX}" "$(to_bytes "$( (( WITH_SAM3 )) && echo "${ENV_SAM3_GB}" || echo "${ENV_GB}")")"
        add_need "${UV_PYTHON_INSTALL_DIR:-${XDG_DATA_HOME:-${HOME}/.local/share}/uv/python}" "$(to_bytes "${UV_PYTHON_GB}")"
        local cache; cache="$(uv_cache_dir)"
        if [[ "$(fs_device "${cache}")" == "$(fs_device "${PREFIX}")" ]]; then
            add_need "${cache}" "$(to_bytes "${UV_CACHE_SAME_GB}")"
        else
            add_need "${cache}" "$(to_bytes "${UV_CACHE_SEPARATE_GB}")"
        fi
        add_need "${HMB_BUILD_TMPDIR:-${TMPDIR:-/tmp}}" "$(to_bytes "${BUILD_GB}")"
    fi
    if (( ! SKIP_ASSETS )); then
        local groups="core"; (( WITH_SAM3 )) && groups+=",sam3"; (( SELF_TEST )) && groups+=",selftest"
        local tool path bytes free; tool="$(tool_python || true)"
        if [[ -n "${tool}" ]]; then
            while IFS=$'\t' read -r path bytes free; do
                [[ -n "${path}" ]] && add_need "${path}" "${bytes}"
            done < <("${tool}" -I "${FETCH}" plan --groups "${groups}" --disk-tsv 2>/dev/null)
        else   # no Python yet to read the manifest: approximate size of the model files (GB)
            add_need "${REPO_ROOT}/outputs/assets" "$(to_bytes "$( (( WITH_SAM3 )) && echo 17.5 || echo 14.0)")"
        fi
    fi
    local dev ok=1 lines=""
    for dev in "${!need[@]}"; do
        local free; free="$(fs_free "${where[${dev}]}")"
        local want=$(( ${need[${dev}]} + $(to_bytes "${MARGIN_GB}") ))
        lines+="${where[${dev}]}: needs $(gb "${need[${dev}]}") (+${MARGIN_GB} GB margin), $(gb "${free}") free; "
        (( free >= want )) || ok=0
    done
    if [[ -z "${lines}" ]]; then
        info OK "disk: nothing left to install"
    elif (( ok )); then
        info OK "disk: ${lines%; }"
    else
        fail_check "not enough free disk: ${lines%; }. Free space, or choose other locations: --prefix (environment), UV_CACHE_DIR (uv cache), TMPDIR (build); model files go to outputs/assets/ of this checkout"
    fi
}

print_plan() {   # what each step will do (shown by --dry-run and before the interactive prompt)
    echo
    echo "Plan (environment: ${PREFIX}):"
    local name state tool smplx_found=0 access="" access_rc=0
    tool="$(tool_python || true)"
    if [[ -n "${SMPLX_SOURCE}" ]]; then
        [[ -e "${SMPLX_SOURCE}" ]] && smplx_found=1
    elif [[ -n "${tool}" ]] && "${tool}" -I "${FETCH}" plan --groups manual --json 2>/dev/null | grep -q '"status": "ok"'; then
        smplx_found=1
    fi
    for name in ${STEP_NAMES}; do
        case "${name}" in
            ffmpeg) (( NEED_FFMPEG )) || { printf '  %-12s %s\n' "${name}" "not needed (ffmpeg found)"; continue; } ;;
            sam3) (( WITH_SAM3 )) || { printf '  %-12s %s\n' "${name}" "not requested (--with-sam3)"; continue; } ;;
            selftest) (( SELF_TEST && ! SKIP_ASSETS )) || { printf '  %-12s %s\n' "${name}" "not requested (--self-test)"; continue; } ;;
            assets) (( SKIP_ASSETS )) && { printf '  %-12s %s\n' "${name}" "skipped (--skip-assets)"; continue; } ;;
        esac
        state="will run"
        case "${name}" in
            sources|doctor) state="will run (a quick check)" ;;
            smplx)
                if (( smplx_found )); then state="will run (a quick check)"
                elif [[ -n "${SMPLX_SOURCE}" ]]; then state="FAILS: --smplx-dir ${SMPLX_SOURCE} does not exist"
                else state="MISSING: licensed, not downloadable; without --smplx-dir the installation ends ALMOST READY (exit 3)"; fi ;;
            selftest)
                if (( ! smplx_found )); then state="skipped: needs the SMPL-X model (--smplx-dir)"
                else stamp_ok selftest "$(selftest_key)" && step_valid selftest && state="done (skipped)"; fi ;;
            environment|smoke) stamp_ok "${name}" "${ENV_KEY}" && step_valid "${name}" && state="done (skipped)" ;;
            assets|sam3) stamp_ok "${name}" "${ASSETS_KEY}" && [[ -x "${PY}" ]] && step_valid "${name}" && state="done (skipped)" ;;&
            sam3)
                if [[ "${state}" == "will run" && -n "${tool}" ]]; then
                    access_rc=0; access="$("${tool}" -I "${FETCH}" hf-access --name sam3.1 2>&1)" || access_rc=$?
                    (( access_rc == 0 )) || state="will run, but SAM 3.1 stays pending (the rest is installed): ${access#\[assets\] }"
                fi ;;
            ffmpeg) stamp_ok "${name}" "${ENV_KEY}" && [[ -x "${PY}" ]] && step_valid "${name}" && state="done (skipped)" ;;
        esac
        printf '  %-12s %s\n' "${name}" "${state}"
    done
    local groups="core"; (( WITH_SAM3 )) && groups+=",sam3"; (( SELF_TEST )) && groups+=",selftest"
    if (( ! SKIP_ASSETS )) && [[ -n "${tool}" ]]; then
        echo
        echo "Model files (install/assets.json):"
        "${tool}" -I "${FETCH}" plan --groups "${groups},manual" || true
    fi
    echo
    echo "Expected time: about 10 to 20 minutes on a fast connection, mostly for the downloads. The environment"
    echo "is built while the model files (13 GB, +3.5 GB SAM 3.1) download; HADM-G (5.7 GB from Dropbox) usually"
    echo "takes longest. The self-test adds a few minutes."
}

# ----------------------------------------------------------------------------- steps
# One function per step of the header; run_step adds the stamp, the timing and the summary line.
step_sources() {
    if bash "${INSTALL_DIR}/check_sources.sh"; then
        return
    fi
    log "restoring the pinned sources (git submodule update --init)"
    bash "${INSTALL_DIR}/check_sources.sh" --init
}

step_environment() {
    local args=(--venv "${PREFIX}" --skip-smoke-tests)
    (( WITH_SAM3 )) && args+=(--with-sam3)
    (( REINSTALL )) && args+=(--force-recreate)
    run_child env HMB_CUDA_ARCH="${CUDA_ARCH}" HMB_INSTALL_PARENT=1 bash "${INSTALL_DIR}/install_slim.sh" "${args[@]}"
    printf 'created by install.sh on %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" > "${PREFIX}/.humanscore-install"
}

# uv for extra installs: on PATH, the private copy, or downloaded again by install_slim.sh.
need_uv() { find_uv || bash "${INSTALL_DIR}/install_slim.sh" --print-uv; }

step_ffmpeg() {
    local uv; uv="$(need_uv)" || die "could not find or download uv"
    "${uv}" pip install --python "${PY}" --no-deps --require-hashes -r "${REPO_ROOT}/requirements/lock-ffmpeg.txt"
    "${PY}" -c 'import imageio_ffmpeg; print("[ffmpeg] imageio-ffmpeg", imageio_ffmpeg.get_ffmpeg_exe())'
}

step_smoke() {
    run_child env HMB_VERIFY_NO_GPU="${NO_GPU}" bash "${INSTALL_DIR}/smoke_test.sh" --python "${PY}"
    # Headless MuJoCo rendering (PyOpenGL + EGL) for the optional skeleton videos of baseline/main.py.
    if MUJOCO_GL=egl "${PY}" - <<'EOF' >/dev/null 2>&1
import mujoco
model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><light pos="0 0 3"/><geom size=".1"/></worldbody></mujoco>')
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
renderer = mujoco.Renderer(model, 64, 64)
renderer.update_scene(data)
assert renderer.render().shape == (64, 64, 3)
EOF
    then
        log "headless MuJoCo rendering (MUJOCO_GL=egl, PyOpenGL): OK"
    else
        log "WARN: headless MuJoCo rendering (MUJOCO_GL=egl) failed; only the optional skeleton videos of baseline/main.py need it (scoring does not render)"
    fi
}

start_prefetch() {   # start the model downloads in the background (needs a system python3)
    (( SKIP_ASSETS )) && return 0
    [[ -n "${SYS_PY}" ]] || return 0
    stamp_ok assets "${ASSETS_KEY}" && [[ -x "${PY}" ]] && step_valid assets && return 0
    PREFETCH_LOG="${LOG_DIR}/assets-${RUN_ID}.log"
    local groups="core"
    # The SAM 3.1 checkpoint joins the background download when access is already granted (the
    # login questions, if any, come in step sam3; its failure never affects the core files).
    if (( WITH_SAM3 )) && "${SYS_PY}" -I "${FETCH}" hf-access --name sam3.1 >/dev/null 2>&1; then
        groups+=",sam3"
    fi
    local verify=()
    (( REINSTALL )) && verify=(--verify)   # --reinstall re-hashes the files that are there
    "${SYS_PY}" -I "${FETCH}" fetch --groups "${groups}" --jobs "${JOBS}" ${verify[@]+"${verify[@]}"} > "${PREFETCH_LOG}" 2>&1 &
    PREFETCH_PID=$!
    log "model files (${groups}): downloading in the background while the environment is built (progress: ${PREFETCH_LOG})"
}

step_assets() {   # wait for the background download, or download (the rest) here
    if [[ -n "${PREFETCH_PID}" ]]; then
        log "waiting for the background download (${PREFETCH_LOG})"
        local rc=0
        wait "${PREFETCH_PID}" || rc=$?
        PREFETCH_PID=""
        grep -E 'done in|FAILED|ERROR|present and verified|checking SHA' "${PREFETCH_LOG}" | sed 's/^/  /' || true
        if (( rc == 0 )); then
            "${PY}" -I "${FETCH}" check --groups core
            return
        fi
        log "the background download did not finish (exit status ${rc}); resuming it here"
    fi
    local verify=()
    (( REINSTALL )) && verify=(--verify)
    run_child "${PY}" -I "${FETCH}" fetch --groups core --jobs "${JOBS}" ${verify[@]+"${verify[@]}"}
}

SMPLX_OK=0; SMPLX_MESSAGE=""
step_smplx() {   # never fails: a missing model ends the installation "almost ready" (exit status 3)
    local rc=0 output
    output="$("${PY}" -I "${FETCH}" smplx ${SMPLX_SOURCE:+--source "${SMPLX_SOURCE}"} 2>&1)" || rc=$?
    echo "${output}"
    case "${rc}" in
        0) SMPLX_OK=1; STEP_RESULT="found" ;;
        3) SMPLX_MESSAGE="${output}"; STEP_RESULT="MISSING (manual download)" ;;
        *) SMPLX_MESSAGE="--smplx-dir ${SMPLX_SOURCE}: ${output}"; STEP_RESULT="INVALID source" ;;
    esac
}

SAM3_STATUS="not requested (--with-sam3)"
sam3_access() { "${PY}" -I "${FETCH}" hf-access --name sam3.1; }
step_sam3() {
    # Never fails the installation: problems are reported in the summary. A checkpoint that is
    # already there is verified (SHA-256) without the network.
    local verify=()
    (( REINSTALL )) && verify=(--verify)
    if { (( ! REINSTALL )) && "${PY}" -I "${FETCH}" check --only sam3.1 --verified >/dev/null 2>&1; } \
        || { "${PY}" -I "${FETCH}" check --only sam3.1 >/dev/null 2>&1 \
             && run_child "${PY}" -I "${FETCH}" fetch --only sam3.1 ${verify[@]+"${verify[@]}"}; }; then
        SAM3_STATUS="ready (outputs/assets/sam3/sam3.1_multiplex.pt)"; return 0
    fi
    local rc=0
    sam3_access || rc=$?
    if (( rc == 4 || rc == 5 )) && (( INTERACTIVE )); then
        local answer=""
        echo
        echo "SAM 3.1 is gated: you need a Hugging Face account with access to https://huggingface.co/facebook/sam3.1"
        echo "and a token (https://huggingface.co/settings/tokens, type 'Read')."
        read -r -p "Log in to Hugging Face now with 'hf auth login'? [Y/n] " answer < /dev/tty || answer="n"
        if [[ -z "${answer}" || "${answer}" =~ ^[Yy] ]]; then
            "${PREFIX}/bin/hf" auth login < /dev/tty > /dev/tty 2>&1 || log "hf auth login did not complete"
            rc=0; sam3_access || rc=$?
        fi
    fi
    case "${rc}" in
        0)
            local fetch_rc=0
            run_child "${PY}" -I "${FETCH}" fetch --only sam3.1 || fetch_rc=$?
            if (( fetch_rc == 0 )); then
                SAM3_STATUS="ready (outputs/assets/sam3/sam3.1_multiplex.pt)"; return 0
            fi
            SAM3_STATUS="download failed (re-run bash install.sh --with-sam3 to resume)" ;;
        4) SAM3_STATUS="not logged in to Hugging Face: request access at https://huggingface.co/facebook/sam3.1, run '${PREFIX}/bin/hf auth login', then bash install.sh --with-sam3" ;;
        5) SAM3_STATUS="Hugging Face token rejected: run '${PREFIX}/bin/hf auth login' again, then bash install.sh --with-sam3" ;;
        6) SAM3_STATUS="access to facebook/sam3.1 not granted yet (request it at https://huggingface.co/facebook/sam3.1; approval can take a while), then bash install.sh --with-sam3" ;;
        *) SAM3_STATUS="could not reach huggingface.co; re-run bash install.sh --with-sam3 later" ;;
    esac
    STEP_RESULT="pending"
    log "SAM 3.1: ${SAM3_STATUS}. The core installation is not affected; humanscore_multi.py --tracker yolo works without it."
    return 1
}

DOCTOR_RC=0; DOCTOR_MISSING=""
step_doctor() {   # the final verdict: python -m hmb.cli doctor checks every file and package
    DOCTOR_RC=0
    (cd "${REPO_ROOT}" && "${PY}" -m hmb.cli doctor) || DOCTOR_RC=$?
    # Required items that are missing (the SMPL-X model alone means "almost ready").
    DOCTOR_MISSING="$(cd "${REPO_ROOT}" && "${PY}" -m hmb.cli doctor --json --skip-optional 2>/dev/null | "${PY}" -c '
import json, sys
report = json.load(sys.stdin)
print("; ".join(c["name"] for c in report["checks"] if c["required"] and not c["ok"]))' 2>/dev/null || true)"
    if (( DOCTOR_RC == 0 )); then STEP_RESULT="ready"; else STEP_RESULT="missing: ${DOCTOR_MISSING:-see above}"; fi
}

SELFTEST_STATUS="not requested (--self-test)"
step_selftest() {
    run_child "${PY}" -I "${FETCH}" fetch --groups selftest
    local args=(--python "${PY}" --out "${STATE_DIR}/self-test")
    [[ -n "${FORCE_STEPS[selftest]:-}" || "${REINSTALL}" -eq 1 ]] && args+=(--force)
    run_child "${PY}" -I "${INSTALL_DIR}/self_test.py" "${args[@]}"   # runs in the repository root
    SELFTEST_STATUS="passed (overall $("${PY}" -c 'import json, sys; print("%.2f" % json.load(open(sys.argv[1]))["overall"])' \
        "${STATE_DIR}/self-test/basic_single_leg_squat/self_test.json" 2>/dev/null || echo ?))"
}

# ----------------------------------------------------------------------------- uninstall
do_uninstall() {   # --uninstall: list what is removed and what is kept, confirm, remove
    local data="${XDG_DATA_HOME:-${HOME}/.local/share}/humanscore" items=() keep=()
    if (( ! DRY_RUN )) && [[ -d "${STATE_DIR}" ]]; then
        # Not while an installation runs in this checkout (it would lose its partial downloads).
        exec 9> "${STATE_DIR}/install.lock"
        if command -v flock >/dev/null 2>&1 && ! flock -n 9; then
            die "install.sh is running in this checkout (${STATE_DIR}/install.lock); stop it first"
        fi
    fi
    if [[ -e "${PREFIX}" ]]; then
        [[ -f "${PREFIX}/pyvenv.cfg" ]] || die "${PREFIX} is not a Python environment; not deleting it"
        case "${PREFIX}" in /|"${HOME}"|"${REPO_ROOT}") die "refusing to delete ${PREFIX}" ;; esac
        # Only an environment of the HumanScore installers (markers of install.sh and install_slim.sh).
        if [[ ! -e "${PREFIX}/.humanscore-install" && ! -e "${PREFIX}/.humanscore-env" \
              && ! -e "${PREFIX}/.humanscore-native-build" ]]; then
            die "${PREFIX} was not created by install.sh (no .humanscore-install marker); not deleting it"
        fi
        items+=("environment ${PREFIX} ($(du -sh "${PREFIX}" 2>/dev/null | cut -f1))")
    fi
    [[ -d "${STAMP_DIR}" ]] && items+=("installer state ${STAMP_DIR}")
    [[ -d "${data}" ]] && items+=("private uv ${data}")
    # HMB_UV_DIR may be a folder with other programs: only the uv that install_slim.sh put there.
    [[ -n "${HMB_UV_DIR:-}" && -f "${HMB_UV_DIR}/uv" ]] && items+=("private uv ${HMB_UV_DIR}/uv")
    local partial
    partial="$(find "${REPO_ROOT}/outputs/assets" \( -name '*.part' -o -name '*.extracting' \) 2>/dev/null | head -n 50 || true)"
    [[ -n "${partial}" ]] && items+=("partial downloads in outputs/assets/")
    (( REMOVE_ASSETS )) && items+=("downloaded model files of install/assets.json (core, sam3, selftest) and their links")
    keep+=("your results and outputs/ (except the items above), outputs/runtime_cache (compiled-program caches; delete it by hand if you like)")
    (( REMOVE_ASSETS )) || keep+=("model files in outputs/assets/ (add --remove-assets to delete them)")
    keep+=("the SMPL-X model (yours; never touched)")
    local cache python_dir
    cache="$(uv_cache_dir)"
    python_dir="${UV_PYTHON_INSTALL_DIR:-${XDG_DATA_HOME:-${HOME}/.local/share}/uv/python}"
    keep+=("the uv cache ${cache} (shared with other uv projects: 'uv cache clean' empties it, or delete the folder if nothing else uses it)")
    keep+=("uv's CPython ${PYTHON_VERSION} in ${python_dir} (shared: 'uv python uninstall ${PYTHON_VERSION}', or delete ${python_dir}/cpython-${PYTHON_VERSION}-*)")
    echo "HumanScore uninstall. Will remove:"
    [[ ${#items[@]} -gt 0 ]] && printf '  - %s\n' "${items[@]}" || echo "  (nothing installed found)"
    echo "Will keep:"
    printf '  - %s\n' "${keep[@]}"
    (( DRY_RUN )) && { echo "(dry run: nothing removed)"; return 0; }
    [[ ${#items[@]} -gt 0 ]] || return 0
    if (( ! YES )); then
        (( INTERACTIVE )) || die "uninstall needs confirmation: run it in a terminal or add --yes"
        local answer=""
        read -r -p "Proceed? [y/N] " answer < /dev/tty || true
        [[ "${answer}" =~ ^[Yy] ]] || { echo "Nothing removed."; return 0; }
    fi
    if (( REMOVE_ASSETS )); then
        local tool; tool="$(tool_python)" || die "removing model files needs python3 (or the environment)"
        "${tool}" -I "${FETCH}" remove --groups core,sam3,selftest
    fi
    [[ -n "${partial}" ]] && find "${REPO_ROOT}/outputs/assets" \( -name '*.part' -o -name '*.extracting' \) -prune -exec rm -rf {} +
    [[ -e "${PREFIX}" ]] && rm -rf "${PREFIX}"
    rm -rf "${STAMP_DIR}" "${data}"
    if [[ -n "${HMB_UV_DIR:-}" && -f "${HMB_UV_DIR}/uv" ]]; then
        rm -f "${HMB_UV_DIR}/uv"
        rmdir "${HMB_UV_DIR}" 2>/dev/null || true
    fi
    echo "Removed. Logs stay in ${LOG_DIR}."
}

# ----------------------------------------------------------------------------- summary
print_summary() {   # print_summary STATUS: step table, locations, warnings, what to do next
    local status="$1" elapsed=$(( SECONDS - START_SECONDS )) entry name result secs
    echo
    echo "======================================================================"
    echo "HumanScore installation: ${status}   ($(( elapsed / 60 )) min $(( elapsed % 60 )) s)"
    echo "======================================================================"
    for entry in ${STEP_REPORT[@]+"${STEP_REPORT[@]}"}; do
        IFS='|' read -r name result secs <<< "${entry}"
        printf '  %-12s %6s s  %s\n' "${name}" "${secs}" "${result}"
    done
    echo
    echo "Environment:  ${PREFIX} ($(size_gb "${PREFIX}"))"
    if (( SKIP_ASSETS )); then
        echo "Model files:  not downloaded (--skip-assets); doctor above lists what is missing"
    else
        echo "Model files:  ${REPO_ROOT}/outputs/assets ($(size_gb -L "${REPO_ROOT}/outputs/assets"), SHA-256 verified)"
    fi
    if (( SMPLX_OK )); then
        echo "SMPL-X:       ${REPO_ROOT}/outputs/assets/smplx/SMPLX_NEUTRAL.npz"
    else
        echo "SMPL-X:       MISSING (see below)"
    fi
    echo "SAM 3.1:      ${SAM3_STATUS}"
    echo "Self-test:    ${SELFTEST_STATUS}"
    local cache; cache="$(uv_cache_dir)"
    local uv; uv="$(find_uv || echo uv)"
    local cache_size="?"
    if [[ -d "${cache}" && -d "${PREFIX}" ]]; then
        local both env_only
        both="$(du -sb --total "${PREFIX}" "${cache}" 2>/dev/null | tail -n 1 | cut -f1)"
        env_only="$(du -sb "${PREFIX}" 2>/dev/null | cut -f1)"
        [[ -n "${both}" && -n "${env_only}" ]] && cache_size="$(gb $(( both - env_only )))"
        [[ "$(fs_device "${cache}")" == "$(fs_device "${PREFIX}")" ]] && cache_size+=" beyond the environment's hard links into it"
    fi
    echo "uv cache:     ${cache} (${cache_size}); the environment works without it:"
    echo "              '${uv} cache clean' frees it (a later reinstall then downloads again)"
    [[ -n "${LOG_FILE}" ]] && echo "Log:          ${LOG_FILE}"
    if [[ ${#WARNINGS[@]} -gt 0 ]]; then
        echo; echo "Warnings:"; printf '  - %s\n' "${WARNINGS[@]}"
    fi
    if [[ -n "${SMPLX_MESSAGE}" ]]; then
        echo; echo "To finish:"; echo "${SMPLX_MESSAGE}" | sed 's/^/  /'
    fi
    cat <<EOF

Activate the environment in each new shell (install.sh does not edit shell start-up files):
  source ${PREFIX}/bin/activate
Then, from ${REPO_ROOT}:
  python scripts/humanscore_single.py video.mp4
  python -m hmb.cli doctor            # re-check the installation at any time
EOF
}

# ----------------------------------------------------------------------------- main
# Order: uninstall (if asked) | lock + log | preflight | plan (dry run, or a prompt in a terminal)
# | the steps of the header | summary and exit status.
START_SECONDS="${SECONDS}"
RUN_ID="$(date '+%Y%m%d-%H%M%S')"

if (( UNINSTALL )); then
    FINISHED=1          # uninstall reports its own errors
    CURRENT_STEP="uninstall"
    do_uninstall
    exit 0
fi

if (( ! DRY_RUN )); then
    # One installer per checkout at a time (the lock is inherited by the background download).
    mkdir -p "${STAMP_DIR}" "${LOG_DIR}"
    exec 9> "${STATE_DIR}/install.lock"
    if command -v flock >/dev/null 2>&1 && ! flock -n 9; then
        FINISHED=1
        die "another install.sh is running in this checkout, or a model download it started is still
        finishing (an installer stopped with kill -9 leaves it running; it ends by itself). Lock:
        ${STATE_DIR}/install.lock"
    fi
    LOG_FILE="${LOG_FILE:-${LOG_DIR}/install-${RUN_ID}.log}"
    mkdir -p "$(dirname "${LOG_FILE}")"
    # tee ignores Ctrl-C, so the messages printed while stopping still reach the terminal and the log.
    exec > >(trap '' INT TERM HUP; exec tee -a "${LOG_FILE}") 2>&1
    TEE_PID=$!
    (( REINSTALL )) && rm -f "${STAMP_DIR}"/*
fi

echo "HumanScore installer - $(date '+%Y-%m-%d %H:%M:%S') - ${REPO_ROOT}"
echo "Options: environment ${PREFIX}; SAM 3.1 $( (( WITH_SAM3 )) && echo yes || echo no); self-test $( (( SELF_TEST )) && echo yes || echo no)$( (( SKIP_ASSETS )) && echo '; --skip-assets')$( (( DRY_RUN )) && echo '; DRY RUN')"
[[ -n "${LOG_FILE}" ]] && echo "Log: ${LOG_FILE}"
echo
CURRENT_STEP="preflight"
CUDA_ARCH="${CUDA_ARCH_OPT}"
compute_keys
preflight

if (( DRY_RUN )); then
    print_plan
    echo
    if [[ ${#FAILURES[@]} -gt 0 ]]; then
        echo "Dry run: ${#FAILURES[@]} check(s) failed; fix them before installing."
        FINISHED=1; exit 1
    fi
    echo "Dry run: all checks passed; nothing was changed. Run the same command without --dry-run to install."
    FINISHED=1; exit 0
fi

if [[ ${#FAILURES[@]} -gt 0 ]]; then
    echo
    echo "[ERROR] ${#FAILURES[@]} check(s) failed:"
    printf '  - %s\n' "${FAILURES[@]}"
    echo "Nothing was installed. Fix the above and run the same command again (bash install.sh --dry-run repeats the checks)."
    FINISHED=1
    exit 1
fi

if (( INTERACTIVE )); then
    print_plan
    answer=""
    read -r -p "Proceed? [Y/n] " answer < /dev/tty || answer="y"
    [[ -z "${answer}" || "${answer}" =~ ^[Yy] ]] || { echo "Nothing was installed."; FINISHED=1; exit 0; }
fi

run_step sources "pinned GVHMR, HADM and torch-mesh-isect sources" step_sources
start_prefetch   # model files download while the environment is built
run_step environment "Python ${PYTHON_VERSION} environment (install/install_slim.sh)" step_environment "${ENV_KEY}"
(( NEED_FFMPEG )) && run_step ffmpeg "imageio-ffmpeg (no ffmpeg on PATH)" step_ffmpeg "$(key_of "${ENV_KEY}" -- "${REPO_ROOT}/requirements/lock-ffmpeg.txt")"
# Without a GPU no stamp: the next run on a GPU machine repeats the smoke tests with their GPU checks.
run_step smoke "smoke tests$( (( NO_GPU )) && echo ' (no GPU: GPU checks skipped)')" step_smoke "$( (( NO_GPU )) || echo "${ENV_KEY}")"
if (( SKIP_ASSETS )); then
    STEP_REPORT+=("assets|skipped (--skip-assets)|0")
else
    run_step assets "model files (install/assets.json, group core)" step_assets "${ASSETS_KEY}"
fi
run_step smplx "SMPL-X body model (licensed, not downloadable)" step_smplx
# SAM 3.1 is optional and gated: its step never fails the installation, so it does not use run_step.
if (( WITH_SAM3 )); then
    if (( SKIP_ASSETS )); then
        SAM3_STATUS="package installed; checkpoint skipped (--skip-assets)"
    else
        CURRENT_STEP="sam3"
        if stamp_ok sam3 "${ASSETS_KEY}" && step_valid sam3; then
            SAM3_STATUS="ready (outputs/assets/sam3/sam3.1_multiplex.pt)"
            STEP_REPORT+=("sam3|skipped (done before)|0")
        else
            sam3_started="${SECONDS}"
            rm -f "${STAMP_DIR}/sam3"
            printf '\n[%s] ==> sam3: SAM 3.1 checkpoint (gated on Hugging Face)\n' "$(date '+%H:%M:%S')"
            if step_sam3; then
                printf '%s\n%s\n' "${ASSETS_KEY}" "$(date '+%Y-%m-%d %H:%M:%S')" > "${STAMP_DIR}/sam3"
                STEP_REPORT+=("sam3|done|$(( SECONDS - sam3_started ))")
            else
                STEP_REPORT+=("sam3|PENDING (see SAM 3.1 below)|$(( SECONDS - sam3_started ))")
            fi
        fi
    fi
fi
run_step doctor "python -m hmb.cli doctor" step_doctor
if (( SELF_TEST )); then
    if (( SKIP_ASSETS )); then
        SELFTEST_STATUS="skipped (--skip-assets)"
    elif (( ! SMPLX_OK )); then
        SELFTEST_STATUS="skipped: needs the SMPL-X model"
        STEP_REPORT+=("selftest|skipped (needs SMPL-X)|0")
    else
        run_step selftest "score a public sample clip (scripts/humanscore_single.py)" step_selftest "$(selftest_key)"
        [[ "${SELFTEST_STATUS}" == "not requested (--self-test)" ]] && SELFTEST_STATUS="passed (earlier run)"
    fi
fi

CURRENT_STEP="summary"
FINISHED=1
# doctor decides; model files skipped with --skip-assets or the SMPL-X model alone make it "almost".
if (( DOCTOR_RC == 0 && SMPLX_OK )); then
    print_summary "READY"
    exit 0
fi
if (( NO_GPU )) && [[ "$(echo "${DOCTOR_MISSING}" | tr ';' '\n' | sed 's/^ *//' | grep -vxE 'nvidia-smi|SMPL-X neutral model|' || true)" == "" ]]; then
    print_summary "BUILT WITHOUT A GPU (compute capability ${CUDA_ARCH}): run bash install.sh again on the GPU machine to verify it"
    exit 3
fi
if [[ "${DOCTOR_MISSING}" == "SMPL-X neutral model" ]] || (( SKIP_ASSETS )); then
    print_summary "ALMOST READY"
    exit 3
fi
if (( DOCTOR_RC == 0 )); then
    print_summary "NOT READY (--smplx-dir does not point to a SMPL-X model; see below)"
else
    # Packages missing from a finished environment (deleted files): uv sees their metadata as
    # installed, so only recreating the environment repairs it.
    hint=""
    [[ "${DOCTOR_MISSING}" == *python:* ]] && hint="; the environment is damaged: repeat the command with --reinstall (model files are kept)"
    print_summary "NOT READY (python -m hmb.cli doctor: missing ${DOCTOR_MISSING:-items listed above})${hint}"
fi
exit 1
