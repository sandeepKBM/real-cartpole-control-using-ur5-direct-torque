export REAL_CARTPOLE_ROOT=/mnt/c/Users/sandr/Downloads/real-cartpole-control-using-ur5-direct-torque
export ROOT=/mnt/c/Users/sandr/Downloads/real-cartpole-control-using-ur5-direct-torque

_COPPELIA_DIR_NAME=CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu22_04

_resolve_coppelia_root() {
  local candidate
  for candidate in \
    "${COPPELIA_ROOT:-}" \
    "${HOME}/coppelia_runtime/${_COPPELIA_DIR_NAME}" \
    "/home/kbm/coppelia_runtime/${_COPPELIA_DIR_NAME}" \
    "${ROOT}/third_party/coppelia_runtime/${_COPPELIA_DIR_NAME}"; do
    if [[ -n "${candidate}" && -x "${candidate}/coppeliaSim.sh" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

_resolve_pydeps() {
  local pyver
  pyver="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo 3.10)"
  local candidate
  for candidate in \
    "${COPPELIA_PYDEPS:-}" \
    "${HOME}/.local/lib/python${pyver}/site-packages" \
    "/home/kbm/.local/lib/python${pyver}/site-packages"; do
    if [[ -n "${candidate}" && -d "${candidate}/coppeliasim_zmqremoteapi_client" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

if ! _resolved_coppelia="$(_resolve_coppelia_root)"; then
  echo "ERROR: CoppeliaSim not found. Install with: bash simulation/setup_coppelia_wsl.sh" >&2
  echo "       (run as your normal WSL user, not root, unless you reinstall under /root)" >&2
  return 1 2>/dev/null || exit 1
fi
export COPPELIA_ROOT="${_resolved_coppelia}"

if ! _resolved_pydeps="$(_resolve_pydeps)"; then
  echo "ERROR: Python ZMQ client not found. Run: bash simulation/bootstrap_pip_wsl.sh" >&2
  return 1 2>/dev/null || exit 1
fi
export COPPELIA_PYDEPS="${_resolved_pydeps}"

export PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
export PATH="/home/kbm/.local/bin:${HOME}/.local/bin:${PATH}"
export LD_LIBRARY_PATH="${COPPELIA_ROOT}:${LD_LIBRARY_PATH:-}"

if [[ "$(id -u)" -eq 0 && "${HOME}" == "/root" ]]; then
  echo "WARNING: running as root; using CoppeliaSim/Python from ${_resolved_coppelia}" >&2
  echo "WARNING: prefer: su - kbm   (or open Ubuntu as your normal user)" >&2
fi
