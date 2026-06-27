#!/usr/bin/env bash
# Minimal ZMQ vision smoke on WSL: open-loop torque pulse + MP4.
# Passes when summary first_frame_std_rgb > 1.0 (not a black buffer).
set -euo pipefail
ROOT="/mnt/c/Users/sandr/Downloads/real-cartpole-control-using-ur5-direct-torque"
# shellcheck disable=SC1091
source "${ROOT}/simulation/env_wsl_local.sh"

COPPELIA="${COPPELIA_ROOT}"
PORT="${PORT:-23096}"
RUNNER="${ROOT}/simulation/run_coppeliasim_x_axis_headless.py"
SIM_LOG="${ROOT}/outputs/control_runs/zmq_vision_wsl_sim.log"
RUNNER_LOG="${ROOT}/outputs/control_runs/zmq_vision_wsl_runner.log"
VIDEO_OUT="${ROOT}/outputs/control_runs/zmq_vision_wsl_smoke.mp4"
SUMMARY_OUT="${ROOT}/outputs/control_runs/zmq_vision_wsl_smoke_summary.json"

mkdir -p "${ROOT}/outputs/control_runs"

pkill -f "zmqRemoteApi.rpcPort=${PORT}" 2>/dev/null || true
pkill -f coppeliaSim 2>/dev/null || true
sleep 1

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

echo "=== ZMQ vision smoke (xcb + DISPLAY=${DISPLAY}) ==="
echo "  port=${PORT}"
echo "  video=${VIDEO_OUT}"

cd "${COPPELIA}"
./coppeliaSim.sh \
  -GzmqRemoteApi.rpcPort="${PORT}" -GzmqRemoteApi.cntPort="$((PORT + 1))" \
  >"${SIM_LOG}" 2>&1 &
COPPELIA_PID=$!

cleanup() {
  kill "${COPPELIA_PID}" 2>/dev/null || true
  wait "${COPPELIA_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 60); do
  if ss -ltn "sport = :${PORT}" 2>/dev/null | grep -q LISTEN; then
    sleep 2
    break
  fi
  if ! kill -0 "${COPPELIA_PID}" 2>/dev/null; then
    echo "CoppeliaSim exited early:" >&2
    tail -30 "${SIM_LOG}" >&2 || true
    exit 1
  fi
  sleep 1
done

if ! ss -ltn "sport = :${PORT}" 2>/dev/null | grep -q LISTEN; then
  echo "Timed out waiting for port ${PORT}" >&2
  exit 1
fi

cd "${ROOT}"
export COPPELIASIM_DEBUG_VISION=1
"${PYTHON_BIN}" "${RUNNER}" \
  --coppelia-root "${COPPELIA}" \
  --port "${PORT}" \
  --torque-pulse \
  --torque-pulse-steps 20 \
  --duration 2 \
  --task-frame-mode mujoco_attachment_dummy \
  --video-camera smoke \
  --video-name "$(basename "${VIDEO_OUT}")" \
  --summary-name "$(basename "${SUMMARY_OUT}")" \
  --fps 10 \
  2>&1 | tee "${RUNNER_LOG}"

python3 - <<PY
import json
from pathlib import Path
p = Path("${SUMMARY_OUT}")
if not p.exists():
    print("FAIL: no summary JSON")
    raise SystemExit(1)
s = json.loads(p.read_text())
std = float(s.get("first_frame_std_rgb") or 0.0)
mean = float(s.get("first_frame_mean_rgb") or 0.0)
frames = int(s.get("frames_written") or 0)
print(f"frames_written={frames}")
print(f"first_frame_mean_rgb={mean}")
print(f"first_frame_std_rgb={std}")
if std > 1.0 and frames > 0:
    print("PASS: ZMQ vision buffer has non-black variance")
    raise SystemExit(0)
print("FAIL: vision buffer still flat (black). Use DISPLAY=:0 and unset QT_QPA_PLATFORM (xcb).")
raise SystemExit(1)
PY
