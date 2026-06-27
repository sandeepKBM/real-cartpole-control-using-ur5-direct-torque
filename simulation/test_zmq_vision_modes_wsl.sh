#!/usr/bin/env bash
# Compare ZMQ vision buffer under different Coppelia launch modes on WSL.
set -euo pipefail
ROOT="/mnt/c/Users/sandr/Downloads/real-cartpole-control-using-ur5-direct-torque"
source "${ROOT}/simulation/env_wsl_local.sh"
export DISPLAY="${DISPLAY:-:0}"
unset QT_QPA_PLATFORM

probe_vision() {
  local label="$1"
  local port="$2"
  local extra_flag="${3:-}"
  pkill -f coppeliaSim 2>/dev/null || true
  sleep 1
  cd "${COPPELIA_ROOT}"
  if [[ -n "${extra_flag}" ]]; then
    ./coppeliaSim.sh "${extra_flag}" \
      -GzmqRemoteApi.rpcPort="${port}" -GzmqRemoteApi.cntPort="$((port+1))" \
      >/tmp/coppelia_vision_${label}.log 2>&1 &
  else
    ./coppeliaSim.sh \
      -GzmqRemoteApi.rpcPort="${port}" -GzmqRemoteApi.cntPort="$((port+1))" \
      >/tmp/coppelia_vision_${label}.log 2>&1 &
  fi
  local pid=$!
  for _ in $(seq 1 40); do
    ss -ltn "sport = :${port}" 2>/dev/null | grep -q LISTEN && break
    sleep 1
  done
  if ! ss -ltn "sport = :${port}" 2>/dev/null | grep -q LISTEN; then
    echo "${label}: Coppelia failed to open port"
    tail -5 "/tmp/coppelia_vision_${label}.log" || true
    kill "${pid}" 2>/dev/null || true
    return 1
  fi
  python3 - <<PY
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
port = int("${port}")
sim = RemoteAPIClient("127.0.0.1", port).require("sim")
scene = "${COPPELIA_ROOT}/system/dfltscn.ttt"
model = "${COPPELIA_ROOT}/models/robots/non-mobile/UR5.ttm"
sim.loadScene(scene)
sim.loadModel(model)
opts = 1 | 2 | 4
h = int(sim.createVisionSensor(opts, [640, 360, 0, 0],
    [0.02, 6.0, 1.01, 0.1, 0, 0, 0.82, 0.86, 0.92, 0, 0]))
sim.setExplicitHandling(h, 1)
try:
    sim.setObjectInt32Param(h, 1008, -1)
except Exception:
    pass
for bid in (10, 16):
    try:
        sim.setBoolParam(bid, True)
    except Exception:
        pass
# smoke camera pose
pose = [1.35, 0.0, 0.96, 0.653281, 0.270598, 0.270598, 0.653281]
sim.setObjectPose(h + sim.handleflag_wxyzquat, pose, sim.handle_world)
sim.setStepping(True)
sim.startSimulation()
sim.step()
try:
    sim.handleVisionSensor(getattr(sim, "handle_all_except_explicit", -3))
except Exception:
    pass
sim.handleVisionSensor(h)
img, res = sim.getVisionSensorImg(h, 0, 0.0, [0, 0], [0, 0])
if isinstance(img, (bytes, bytearray, memoryview)):
    b = bytes(img)
elif isinstance(img, list):
    b = bytes(img)
else:
    b = b""
need = int(res[0]) * int(res[1]) * 3
bsum = int(sum(b[:min(need, 8000)]))
arr = np.frombuffer(b[:need], dtype=np.uint8).reshape(int(res[1]), int(res[0]), 3)
print("${label}: byte_sum8000=", bsum, "std=", float(np.std(arr)), "res=", res)
sim.stopSimulation()
PY
  kill "${pid}" 2>/dev/null || true
  wait "${pid}" 2>/dev/null || true
}

probe_vision plain 23100 ""
probe_vision emulated_h 23101 "-h"
LIBGL_ALWAYS_SOFTWARE=1 probe_vision sw_gl 23102 ""
