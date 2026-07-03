#!/usr/bin/env bash
# Local/WSL wrapper: resolves repo + Coppelia paths, then runs the headless launcher.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -f "${ROOT}/simulation/env_wsl_local.sh" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/simulation/env_wsl_local.sh"
else
  export REAL_CARTPOLE_ROOT="${ROOT}"
  export ROOT="${ROOT}"
  export COPPELIA_ROOT="${COPPELIA_ROOT:-${HOME}/coppelia_runtime/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu22_04}"
  export PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python3}"
  export COPPELIA_PYDEPS="${COPPELIA_PYDEPS:-${ROOT}/.venv/lib/python3.12/site-packages}"
fi

if [[ ! -x "${COPPELIA_ROOT}/coppeliaSim.sh" ]]; then
  echo "CoppeliaSim not found at ${COPPELIA_ROOT}" >&2
  echo "Run: bash simulation/setup_coppelia_wsl.sh" >&2
  exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python not found at ${PYTHON_BIN}" >&2
  echo "Run: bash simulation/setup_coppelia_wsl.sh" >&2
  exit 1
fi

# Rewrite ROOT inside the cluster launcher for this machine.
TMP_LAUNCHER="$(mktemp)"
trap 'rm -f "${TMP_LAUNCHER}"' EXIT
sed "s|^ROOT=\"/common/users/ss5772/real_Cartpole\"|ROOT=\"${ROOT}\"|" \
  "${ROOT}/simulation/launch_coppeliasim_x_axis_headless.sh" > "${TMP_LAUNCHER}"
chmod +x "${TMP_LAUNCHER}"

export COPPELIA_ROOT
export COPPELIA_PYDEPS
export PYTHON_BIN
export REAL_CARTPOLE_ROOT="${ROOT}"
exec bash "${TMP_LAUNCHER}" "$@"
