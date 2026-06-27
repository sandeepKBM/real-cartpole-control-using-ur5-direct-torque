#!/usr/bin/env bash
# CoppeliaSim + Python setup for WSL2 (Ubuntu 22.04), no sudo required.
# System packages (xvfb, ffmpeg, python3.12) still need one sudo apt line — see end of script.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
COPPELIA_URL="https://downloads.coppeliarobotics.com/V4_10_0_rev0/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu22_04.tar.xz"
COPPELIA_DIR_NAME="CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu22_04"
COPPELIA_INSTALL_ROOT="${COPPELIA_INSTALL_ROOT:-${HOME}/coppelia_runtime}"
COPPELIA_ROOT="${COPPELIA_INSTALL_ROOT}/${COPPELIA_DIR_NAME}"
VENV_DIR="${ROOT}/.venv"
PYTHON_BIN="${PYTHON_BIN:-}"

pick_python() {
  if [[ -n "${PYTHON_BIN}" && -x "${PYTHON_BIN}" ]]; then
    echo "${PYTHON_BIN}"
    return 0
  fi
  for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      echo "$(command -v "${candidate}")"
      return 0
    fi
  done
  return 1
}

echo "==> Repo root: ${ROOT}"
echo "==> CoppeliaSim target: ${COPPELIA_ROOT}"

mkdir -p "${COPPELIA_INSTALL_ROOT}"
if [[ ! -x "${COPPELIA_ROOT}/coppeliaSim.sh" ]]; then
  echo "==> Downloading CoppeliaSim Edu V4.10 Ubuntu 22.04 (~400 MB)..."
  tmp_archive="${COPPELIA_INSTALL_ROOT}/${COPPELIA_DIR_NAME}.tar.xz"
  wget -q --show-progress -O "${tmp_archive}" "${COPPELIA_URL}"
  echo "==> Extracting..."
  tar -xJf "${tmp_archive}" -C "${COPPELIA_INSTALL_ROOT}"
  rm -f "${tmp_archive}"
  chmod +x "${COPPELIA_ROOT}/coppeliaSim.sh"
else
  echo "==> CoppeliaSim already present"
fi

if [[ ! -f "${COPPELIA_ROOT}/models/robots/non-mobile/UR5.ttm" ]]; then
  echo "ERROR: UR5.ttm missing under ${COPPELIA_ROOT}" >&2
  exit 1
fi

SELECTED_PYTHON="$(pick_python)" || {
  echo "ERROR: No python3 found. Install python3.12 or python3.10." >&2
  exit 1
}
echo "==> Using Python: ${SELECTED_PYTHON} ($("${SELECTED_PYTHON}" --version))"

echo "==> Creating venv at ${VENV_DIR}..."
"${SELECTED_PYTHON}" -m venv "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip wheel
pip install "coppeliasim-zmqremoteapi-client>=2.0.4" numpy pyyaml

PY_SITE="$(python -c 'import site; print(site.getsitepackages()[0])')"

mkdir -p "${ROOT}/outputs/control_runs"

cat > "${ROOT}/simulation/env_wsl_local.sh" <<EOF
# Source in WSL before Coppelia runs:
#   source simulation/env_wsl_local.sh
export REAL_CARTPOLE_ROOT="${ROOT}"
export ROOT="${ROOT}"
export COPPELIA_ROOT="${COPPELIA_ROOT}"
export COPPELIA_PYDEPS="${PY_SITE}"
export PYTHON_BIN="${VENV_DIR}/bin/python3"
export PATH="${VENV_DIR}/bin:\${PATH}"
export LD_LIBRARY_PATH="\${COPPELIA_ROOT}:\${LD_LIBRARY_PATH:-}"
EOF

echo ""
echo "==> Python + CoppeliaSim runtime ready."
echo "    CoppeliaSim : ${COPPELIA_ROOT}"
echo "    Python venv : ${VENV_DIR}"
echo "    Env file    : ${ROOT}/simulation/env_wsl_local.sh"
echo ""

missing=()
for tool in xvfb-run ffmpeg xdpyinfo; do
  command -v "${tool}" >/dev/null 2>&1 || missing+=("${tool}")
done

if ((${#missing[@]} > 0)); then
  echo "==> Missing system tools: ${missing[*]}"
  echo "    Run this once in WSL (sudo password required):"
  echo ""
  echo "    sudo apt-get update && sudo apt-get install -y \\"
  echo "      xvfb xauth x11-utils ffmpeg python3.12 python3.12-venv"
  echo ""
else
  echo "==> System tools look good (xvfb, ffmpeg present)."
fi

echo "Next probe command:"
echo "  source simulation/env_wsl_local.sh"
echo "  bash simulation/launch_coppeliasim_x_axis_headless_local.sh --probe-only --no-video --task-frame-mode mujoco_attachment_dummy"
