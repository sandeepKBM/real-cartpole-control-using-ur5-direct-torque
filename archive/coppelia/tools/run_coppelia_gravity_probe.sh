#!/usr/bin/env bash
set -euo pipefail
ROOT="/mnt/c/Users/sandr/Downloads/real-cartpole-control-using-ur5-direct-torque"
source "${ROOT}/simulation/env_wsl_local.sh"

COPPELIA="${COPPELIA_ROOT}"
PORT="${PORT:-23000}"
SIM_LOG="${ROOT}/outputs/control_runs/coppelia_gravity_probe_sim.log"

pkill -f coppeliaSim 2>/dev/null || true
sleep 3

export COPPELIA_ROOT COPPELIA_PYDEPS REAL_CARTPOLE_ROOT
export REAL_CARTPOLE_ENABLE_VIDEO_SMOKE=0
export LD_LIBRARY_PATH="${COPPELIA}:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${COPPELIA}/programming/zmqRemoteApi/clients/python/src:${COPPELIA_PYDEPS}:${PYTHONPATH:-}"
export DISPLAY="${DISPLAY:-:0}"
export WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-wayland-0}"
if [[ -z "${XDG_RUNTIME_DIR:-}" && -d "/run/user/$(id -u)" ]]; then
  export XDG_RUNTIME_DIR="/run/user/$(id -u)"
fi
unset QT_QPA_PLATFORM

mkdir -p "${ROOT}/outputs/control_runs"

cd "${COPPELIA}"
./coppeliaSim.sh \
  -GzmqRemoteApi.rpcPort="${PORT}" -GzmqRemoteApi.cntPort="$((PORT + 1))" \
  >"${SIM_LOG}" 2>&1 &
COPPELIA_PID=$!
cleanup() { kill "${COPPELIA_PID}" 2>/dev/null || true; wait "${COPPELIA_PID}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

echo "CoppeliaSim PID=${COPPELIA_PID}, waiting for port ${PORT}..."
for i in $(seq 1 90); do
  if ss -ltn "sport = :${PORT}" 2>/dev/null | grep -q LISTEN; then
    echo "Port ${PORT} listening after ~${i}s, waiting 8s for API init..."
    sleep 8
    echo "RPC ready."
    break
  fi
  if ! kill -0 "${COPPELIA_PID}" 2>/dev/null; then
    echo "CoppeliaSim exited early. Last log:" >&2
    tail -20 "${SIM_LOG}" >&2 || true
    exit 1
  fi
  sleep 1
done

if ! ss -ltn "sport = :${PORT}" 2>/dev/null | grep -q LISTEN; then
  echo "Timed out waiting for port ${PORT}. Log tail:" >&2
  tail -20 "${SIM_LOG}" >&2 || true
  exit 1
fi

cd "${ROOT}"
PORT="${PORT}" "${PYTHON_BIN}" tools/probe_coppelia_gravity_native.py
echo "Probe exit code: $?"
