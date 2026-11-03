#!/usr/bin/env bash
# Smoke test of a HumanScore environment: core imports and GPU access (install/verify_installation.py),
# then the --help of every command-line entry point (paper metric code, scorer, CLI, scripts). It
# needs no model files. ./install.sh runs it as its "smoke" step; run it again at any time:
#
#   bash install/smoke_test.sh --python .venv-humanscore/bin/python
#
# Exit status 0 when every check passes; the first failing check stops it.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# The `python -m metrics...` and `hmb...` checks import from the repository root.
cd "${REPO_ROOT}"

PYTHON_EXE="${REPO_ROOT}/.venv-humanscore/bin/python"   # default: the environment of ./install.sh

usage() {
    cat <<EOF
Usage: bash install/smoke_test.sh [--python PATH]

  --python PATH     interpreter of the environment to test (default: .venv-humanscore/bin/python)
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --python)
            PYTHON_EXE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

# run_in_env CMD...: CMD inside the environment (python = its interpreter); run_in_env_bash STRING
# runs a shell command line there.
[[ -x "${PYTHON_EXE}" ]] || { echo "[ERROR] not an executable interpreter: ${PYTHON_EXE}" >&2; exit 1; }
# As in an activated environment: the interpreter's bin/ first on PATH.
ENV_BIN="$(cd "$(dirname "${PYTHON_EXE}")" && pwd)"
PYTHON_EXE="${ENV_BIN}/$(basename "${PYTHON_EXE}")"

run_in_env() {
    local command="$1"
    shift
    [[ "${command}" == python ]] && command="${PYTHON_EXE}"
    PATH="${ENV_BIN}:${PATH}" "${command}" "$@"
}

run_in_env_bash() {
    PATH="${ENV_BIN}:${PATH}" bash -c "$*"
}

echo "[SMOKE] Verifying core dependencies"
run_in_env python "${REPO_ROOT}/install/verify_installation.py"

echo "[SMOKE] Verifying baseline CLI"
run_in_env_bash "cd '${REPO_ROOT}/baseline' && python main.py --help >/dev/null"

echo "[SMOKE] Verifying metric CLIs (the original paper implementation, used by --paper-scores)"
run_in_env python -m metrics.bone_length.main --help >/dev/null
run_in_env python -m metrics.motion_smoothness.main --help >/dev/null
run_in_env python -m metrics.kinematic_extreme.main --help >/dev/null
run_in_env python -m metrics.self_collision_bvh.main --help >/dev/null
run_in_env python -m metrics.range_of_motion.joints_violaton --help >/dev/null

echo "[SMOKE] Verifying extra-limbs entrypoint"
run_in_env python "${REPO_ROOT}/metrics/multi-limbs/eval_extra_limbs.py" --help >/dev/null

echo "[SMOKE] Verifying user-facing CLI"
run_in_env python -m hmb.cli --help >/dev/null
run_in_env python -m hmb.cli prepare --help >/dev/null
run_in_env python -m hmb.cli score --help >/dev/null
run_in_env python -m hmb.scoring.score_session --help >/dev/null
run_in_env python -m hmb.batch --help >/dev/null
run_in_env python "${REPO_ROOT}/scripts/humanscore_single.py" --help >/dev/null
run_in_env python "${REPO_ROOT}/scripts/humanscore_multi.py" --help >/dev/null

echo "[SMOKE] All smoke tests passed"
