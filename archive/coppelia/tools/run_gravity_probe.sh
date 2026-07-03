#!/bin/bash
set -eu
ROOT="/mnt/c/Users/sandr/Downloads/real-cartpole-control-using-ur5-direct-torque"
source "${ROOT}/simulation/env_wsl_local.sh"
export DISPLAY="${DISPLAY:-:0}"
unset QT_QPA_PLATFORM

COPPELIA="${COPPELIA_ROOT}"
PORT="${PORT:-23000}"
SIM_LOG="/tmp/coppelia_gravity_probe.log"

pgrep -f coppeliaSim >/dev/null 2>&1 && kill $(pgrep -f coppeliaSim) 2>/dev/null
sleep 2

cd "${COPPELIA}"
./coppeliaSim.sh \
  -GzmqRemoteApi.rpcPort="${PORT}" \
  -GzmqRemoteApi.cntPort="$((PORT+1))" \
  >"${SIM_LOG}" 2>&1 &
SIM_PID=$!

cleanup() {
  kill "${SIM_PID}" 2>/dev/null || true
  wait "${SIM_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Waiting for port ${PORT}..."
for i in $(seq 1 90); do
  if ss -ltn "sport = :${PORT}" 2>/dev/null | grep -q LISTEN; then
    sleep 2
    if ss -ltn "sport = :${PORT}" 2>/dev/null | grep -q LISTEN; then
      echo "RPC ready."
      break
    fi
  fi
  if ! kill -0 "${SIM_PID}" 2>/dev/null; then
    echo "CoppeliaSim exited early" >&2
    tail -20 "${SIM_LOG}" >&2
    exit 1
  fi
  sleep 1
done

cd "${ROOT}"
PORT="${PORT}" "${PYTHON_BIN}" tools/probe_gravity_sign.py
echo "Probe complete."
