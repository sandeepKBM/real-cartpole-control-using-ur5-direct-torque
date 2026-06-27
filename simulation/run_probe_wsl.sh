#!/usr/bin/env bash
set -euo pipefail
ROOT="/mnt/c/Users/sandr/Downloads/real-cartpole-control-using-ur5-direct-torque"
# shellcheck disable=SC1091
source "${ROOT}/simulation/env_wsl_local.sh"
COPPELIA="${COPPELIA_ROOT}"
PORT=23000
echo "Using COPPELIA_ROOT=${COPPELIA}"
RUNNER="${ROOT}/simulation/run_coppeliasim_x_axis_headless.py"
STATE_DIR="${ROOT}/outputs/control_runs/coppelia_x_axis_headless_state"
SIM_LOG="${ROOT}/outputs/control_runs/coppelia_x_axis_headless.log"
RUNNER_LOG="${ROOT}/outputs/control_runs/coppelia_x_axis_headless_runner.log"

mkdir -p "${STATE_DIR}" "$(dirname "${SIM_LOG}")"

if command -v ss >/dev/null 2>&1 && ss -ltn "sport = :${PORT}" | grep -q LISTEN; then
  echo "Port ${PORT} busy; stopping stale CoppeliaSim..."
  pkill -f "zmqRemoteApi.rpcPort=${PORT}" || true
  pkill -f coppeliaSim || true
  sleep 2
fi

export COPPELIA_ROOT COPPELIA_PYDEPS REAL_CARTPOLE_ROOT
export REAL_CARTPOLE_ENABLE_VIDEO_SMOKE=0
export LD_LIBRARY_PATH="${COPPELIA}:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${COPPELIA}/programming/zmqRemoteApi/clients/python/src:${COPPELIA_PYDEPS}:${PYTHONPATH:-}"

# Qt offscreen works on WSL without extra apt packages (wayland/xcb need libxkbcommon-x11).
# Set COPPELIA_QT_PLATFORM=xcb or wayland after installing display libs for a GUI window.
export DISPLAY="${DISPLAY:-:0}"
export WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-wayland-0}"
if [[ -z "${XDG_RUNTIME_DIR:-}" ]]; then
  if [[ -d "/run/user/$(id -u)" ]]; then
    export XDG_RUNTIME_DIR="/run/user/$(id -u)"
  fi
fi

launch_coppelia() {
  local platform="${1:-offscreen}"
  if [[ "${platform}" == "xcb" ]]; then
    unset QT_QPA_PLATFORM
  else
    export QT_QPA_PLATFORM="${platform}"
  fi
  ./coppeliaSim.sh \
    -GzmqRemoteApi.rpcPort="${PORT}" -GzmqRemoteApi.cntPort="$((PORT + 1))" \
    >"${SIM_LOG}" 2>&1 &
  COPPELIA_PID=$!
}

cd "${COPPELIA}"
_PLATFORM="${COPPELIA_QT_PLATFORM:-offscreen}"
echo "Starting CoppeliaSim on port ${PORT} (QT_QPA_PLATFORM=${_PLATFORM})..."

if command -v xvfb-run >/dev/null 2>&1 && [[ "${FORCE_XVFB:-0}" == "1" ]]; then
  echo "  mode=xvfb (FORCE_XVFB=1)"
  xvfb-run -a ./coppeliaSim.sh -GzmqRemoteApi.rpcPort="${PORT}" -GzmqRemoteApi.cntPort="$((PORT + 1))" >"${SIM_LOG}" 2>&1 &
  COPPELIA_PID=$!
else
  launch_coppelia "${_PLATFORM}"
fi

cleanup() {
  kill "${COPPELIA_PID}" 2>/dev/null || true
  wait "${COPPELIA_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Waiting for ZMQ port ${PORT}..."
for _ in $(seq 1 60); do
  if ss -ltn "sport = :${PORT}" 2>/dev/null | grep -q LISTEN; then
    sleep 2
    if ss -ltn "sport = :${PORT}" 2>/dev/null | grep -q LISTEN && pgrep -f coppeliaSim >/dev/null 2>&1; then
      echo "RPC port open and CoppeliaSim still running."
      break
    fi
    echo "Port opened but CoppeliaSim died; retrying..." >&2
  fi
  if ! kill -0 "${COPPELIA_PID}" 2>/dev/null; then
    echo "CoppeliaSim exited early. Log tail:" >&2
    tail -40 "${SIM_LOG}" >&2 || true
    if grep -q 'Could not load the Qt platform plugin' "${SIM_LOG}" 2>/dev/null; then
      echo "" >&2
      echo "FIX option A (no GUI, works now): COPPELIA_QT_PLATFORM=offscreen bash simulation/run_probe_wsl.sh" >&2
      echo "FIX option B (GUI window): install missing lib, then use xcb:" >&2
      echo "  sudo apt-get install -y libxkbcommon-x11-0 libxkbcommon0 libxcb-cursor0 \\" >&2
      echo "    libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util0 \\" >&2
      echo "    libxcb-xinerama0 libxcb-xfixes0 libfontconfig1" >&2
      echo "  bash simulation/diagnose_display.sh" >&2
    fi
    exit 1
  fi
  sleep 1
 done

if ! ss -ltn "sport = :${PORT}" 2>/dev/null | grep -q LISTEN; then
  echo "Timed out waiting for port ${PORT}" >&2
  tail -40 "${SIM_LOG}" >&2 || true
  exit 1
fi

echo "Running probe-only runner..."
cd "${ROOT}"
"${PYTHON_BIN}" "${RUNNER}" \
  --probe-only --no-video --task-frame-mode mujoco_attachment_dummy \
  --coppelia-root "${COPPELIA}" \
  --port "${PORT}" 2>&1 | tee "${RUNNER_LOG}"
PROBE_EXIT=${PIPESTATUS[0]}
echo "Probe exit code: ${PROBE_EXIT}"
exit "${PROBE_EXIT}"
