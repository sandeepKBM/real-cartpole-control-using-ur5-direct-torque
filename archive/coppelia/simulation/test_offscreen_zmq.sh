#!/usr/bin/env bash
set -euo pipefail
ROOT="/mnt/c/Users/sandr/Downloads/real-cartpole-control-using-ur5-direct-torque"
source "${ROOT}/simulation/env_wsl_local.sh"
C="${COPPELIA_ROOT}"
PORT=23988
cd "${C}"
export LD_LIBRARY_PATH="${C}:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${C}/programming/zmqRemoteApi/clients/python/src:${COPPELIA_PYDEPS}:${PYTHONPATH:-}"

pkill -f "zmqRemoteApi.rpcPort=${PORT}" 2>/dev/null || true
sleep 1

echo "Launch offscreen..."
QT_QPA_PLATFORM=offscreen ./coppeliaSim.sh \
  -GzmqRemoteApi.rpcPort="${PORT}" -GzmqRemoteApi.cntPort="$((PORT+1))" \
  >/tmp/coppelia_offscreen_zmq.log 2>&1 &
PID=$!
sleep 10

if ! kill -0 "${PID}" 2>/dev/null; then
  echo "FAIL: sim died"
  tail -20 /tmp/coppelia_offscreen_zmq.log
  exit 1
fi
if ! ss -ltn | grep -q "${PORT}"; then
  echo "FAIL: port not listening"
  tail -20 /tmp/coppelia_offscreen_zmq.log
  kill "${PID}" 2>/dev/null || true
  exit 1
fi
echo "OK: sim running, port ${PORT} open"

cd "${ROOT}"
python3 - <<PY
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
c = RemoteAPIClient(host="127.0.0.1", port=${PORT})
sim = c.require("sim")
print("require(sim) OK, state=", sim.getSimulationState())
PY

kill "${PID}" 2>/dev/null || true
echo "ZMQ probe OK"
