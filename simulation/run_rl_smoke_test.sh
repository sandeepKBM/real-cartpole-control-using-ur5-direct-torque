#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${SCRIPT_DIR}/env_wsl_local.sh"

PORT="${PORT:-23000}"

# Kill stale CoppeliaSim
if ss -ltn 2>/dev/null | grep -q ":${PORT} "; then
  echo "Killing stale CoppeliaSim on port ${PORT}"
  pkill -f "zmqRemoteApi.rpcPort=${PORT}" 2>/dev/null || true
  sleep 2
fi

rm -f "${COPPELIA_ROOT}/addOns/ur5_video_smoke_addon.lua"
export REAL_CARTPOLE_ENABLE_VIDEO_SMOKE=0
unset QT_QPA_PLATFORM 2>/dev/null || true

SIM_LOG="${ROOT}/outputs/control_runs/rl_coppelia_sim.log"
mkdir -p "$(dirname "${SIM_LOG}")"

echo "Starting CoppeliaSim (DISPLAY=${DISPLAY:-unset})..."
cd "${COPPELIA_ROOT}"
./coppeliaSim.sh \
  -GzmqRemoteApi.rpcPort="${PORT}" \
  -GzmqRemoteApi.cntPort="$((PORT + 1))" \
  >"${SIM_LOG}" 2>&1 &
SIM_PID=$!
cd "${ROOT}"
echo "CoppeliaSim PID=${SIM_PID}"

echo "Waiting for ZMQ port ${PORT}..."
for i in $(seq 1 90); do
  if ss -ltn "sport = :${PORT}" 2>/dev/null | grep -q LISTEN; then
    sleep 2
    if ss -ltn "sport = :${PORT}" 2>/dev/null | grep -q LISTEN && kill -0 "${SIM_PID}" 2>/dev/null; then
      echo "RPC ready."
      break
    fi
  fi
  if ! kill -0 "${SIM_PID}" 2>/dev/null; then
    echo "ERROR: CoppeliaSim died before port opened"
    cat "${SIM_LOG}" 2>/dev/null | tail -20
    exit 1
  fi
  sleep 1
done

sleep 8

echo "--- CoppeliaSim log (first 20 lines) ---"
head -20 "${SIM_LOG}" 2>/dev/null || echo "(no log yet)"
echo "---"

echo "Running smoke test..."
PYTHONPATH="${ROOT}:${COPPELIA_PYDEPS}:${PYTHONPATH:-}" \
PYTHONUNBUFFERED=1 \
  python3 -u "${ROOT}/rl/smoke_test_env.py" 2>&1
TEST_EXIT=$?

kill "${SIM_PID}" 2>/dev/null || true
wait "${SIM_PID}" 2>/dev/null || true
echo "Smoke test exit code: ${TEST_EXIT}"
exit ${TEST_EXIT}
