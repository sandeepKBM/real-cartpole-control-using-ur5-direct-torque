#!/usr/bin/env bash
set -euo pipefail
ROOT="/mnt/c/Users/sandr/Downloads/real-cartpole-control-using-ur5-direct-torque"
source "${ROOT}/simulation/env_wsl_local.sh"

COPPELIA="${COPPELIA_ROOT}"
PORT="${PORT:-23050}"

pkill -f coppeliaSim 2>/dev/null || true
sleep 2

export COPPELIA_ROOT COPPELIA_PYDEPS REAL_CARTPOLE_ROOT
export REAL_CARTPOLE_ENABLE_VIDEO_SMOKE=0
export LD_LIBRARY_PATH="${COPPELIA}:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${COPPELIA}/programming/zmqRemoteApi/clients/python/src:${COPPELIA_PYDEPS}:${PYTHONPATH:-}"
export DISPLAY="${DISPLAY:-:0}"
export WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-wayland-0}"
if [[ -z "${XDG_RUNTIME_DIR:-}" && -d "/run/user/$(id -u)" ]]; then
  export XDG_RUNTIME_DIR="/run/user/$(id -u)"
fi

cd "${COPPELIA}"
./coppeliaSim.sh \
  -GzmqRemoteApi.rpcPort="${PORT}" -GzmqRemoteApi.cntPort="$((PORT + 1))" \
  >/dev/null 2>&1 &
COPPELIA_PID=$!
cleanup() { kill "${COPPELIA_PID}" 2>/dev/null || true; wait "${COPPELIA_PID}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

echo "Waiting for port ${PORT}..."
for _ in $(seq 1 90); do
  if ss -ltn "sport = :${PORT}" 2>/dev/null | grep -q LISTEN; then
    sleep 5
    echo "RPC ready."
    break
  fi
  sleep 1
done

cd "${ROOT}"
PORT="${PORT}" "${PYTHON_BIN}" tools/probe_gravity_multi_pose.py
