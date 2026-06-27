#!/usr/bin/env bash
# Evaluate a trained PPO policy for UR5 Y-transport on WSL.
#
# Usage:
#   bash simulation/run_rl_eval_wsl.sh
#   MODEL=outputs/rl_models/checkpoints/ppo_y_transport_50000_steps.zip EPISODES=3 bash simulation/run_rl_eval_wsl.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${SCRIPT_DIR}/env_wsl_local.sh"

PORT="${PORT:-23000}"
MODEL="${MODEL:-${ROOT}/outputs/rl_models/ppo_y_transport}"
EPISODES="${EPISODES:-5}"

if ss -ltn 2>/dev/null | grep -q ":${PORT} "; then
  pkill -f "zmqRemoteApi.rpcPort=${PORT}" 2>/dev/null || true
  sleep 2
fi

rm -f "${COPPELIA_ROOT}/addOns/ur5_video_smoke_addon.lua"
export REAL_CARTPOLE_ENABLE_VIDEO_SMOKE=0

SIM_LOG="${ROOT}/outputs/control_runs/rl_eval_coppelia_sim.log"
mkdir -p "$(dirname "${SIM_LOG}")"

cd "${COPPELIA_ROOT}"
./coppeliaSim.sh \
  -GzmqRemoteApi.rpcPort="${PORT}" \
  -GzmqRemoteApi.cntPort="$((PORT + 1))" \
  >"${SIM_LOG}" 2>&1 &
SIM_PID=$!
cd "${ROOT}"

for i in $(seq 1 90); do
  if ss -ltn "sport = :${PORT}" 2>/dev/null | grep -q LISTEN; then
    sleep 2
    if kill -0 "${SIM_PID}" 2>/dev/null; then
      break
    fi
  fi
  if ! kill -0 "${SIM_PID}" 2>/dev/null; then
    echo "ERROR: CoppeliaSim died before port opened"
    tail -20 "${SIM_LOG}"
    exit 1
  fi
  sleep 1
done

sleep 5

PYTHONPATH="${ROOT}:${COPPELIA_PYDEPS}:${PYTHONPATH:-}" \
PYTHONUNBUFFERED=1 \
  python3 -u "${ROOT}/rl/eval_policy.py" \
    --model "${MODEL}" \
    --port "${PORT}" \
    --coppelia-root "${COPPELIA_ROOT}" \
    --episodes "${EPISODES}"

EVAL_EXIT=$?
kill "${SIM_PID}" 2>/dev/null || true
wait "${SIM_PID}" 2>/dev/null || true
exit ${EVAL_EXIT}
