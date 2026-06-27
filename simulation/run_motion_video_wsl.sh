#!/usr/bin/env bash
# CoppeliaSim external ZMQ X-transport + MP4 (WSL). Not MuJoCo.
set -euo pipefail
ROOT="/mnt/c/Users/sandr/Downloads/real-cartpole-control-using-ur5-direct-torque"
# shellcheck disable=SC1091
source "${ROOT}/simulation/env_wsl_local.sh"

COPPELIA="${COPPELIA_ROOT}"
PORT="${PORT:-23000}"
TARGET_DX="${TARGET_DX:-0.005}"
DURATION="${DURATION:-3}"
SETTLE="${SETTLE_DURATION:-1}"
VIDEO_NAME="${VIDEO_NAME:-coppeliasim_ur5_wsl_x_transport.mp4}"
SUMMARY_NAME="${SUMMARY_NAME:-coppeliasim_ur5_wsl_x_transport_summary.json}"
RUNNER="${ROOT}/simulation/run_coppeliasim_x_axis_headless.py"
SIM_LOG="${ROOT}/outputs/control_runs/coppelia_x_axis_headless.log"
RUNNER_LOG="${ROOT}/outputs/control_runs/coppelia_x_axis_headless_motion.log"
VIDEO_OUT="${ROOT}/demonstration_videos/ur5e_coppeliasim/${VIDEO_NAME}"
SUMMARY_OUT="${ROOT}/outputs/control_runs/${SUMMARY_NAME}"

mkdir -p "${ROOT}/outputs/control_runs" "${ROOT}/demonstration_videos/ur5e_coppeliasim"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ERROR: ffmpeg required. Run: sudo apt-get install -y ffmpeg" >&2
  exit 1
fi

pkill -f "zmqRemoteApi.rpcPort=${PORT}" 2>/dev/null || true
pkill -f coppeliaSim 2>/dev/null || true
sleep 2

export COPPELIA_ROOT COPPELIA_PYDEPS REAL_CARTPOLE_ROOT
export REAL_CARTPOLE_ENABLE_VIDEO_SMOKE=0
export LD_LIBRARY_PATH="${COPPELIA}:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${COPPELIA}/programming/zmqRemoteApi/clients/python/src:${COPPELIA_PYDEPS}:${PYTHONPATH:-}"
export DISPLAY="${DISPLAY:-:0}"

launch_coppelia() {
  local platform="${1:-xcb}"
  if [[ "${platform}" == "xcb" ]]; then
    unset QT_QPA_PLATFORM
  else
    export QT_QPA_PLATFORM="${platform}"
  fi
  cd "${COPPELIA}"
  ./coppeliaSim.sh \
    -GzmqRemoteApi.rpcPort="${PORT}" -GzmqRemoteApi.cntPort="$((PORT + 1))" \
    >"${SIM_LOG}" 2>&1 &
  COPPELIA_PID=$!
}

_PLATFORM="${COPPELIA_QT_PLATFORM:-xcb}"
echo "Using COPPELIA_ROOT=${COPPELIA}"
echo "Starting CoppeliaSim (QT_QPA_PLATFORM=${_PLATFORM}) port=${PORT}..."
launch_coppelia "${_PLATFORM}"

cleanup() {
  kill "${COPPELIA_PID}" 2>/dev/null || true
  wait "${COPPELIA_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Waiting for ZMQ port ${PORT}..."
for _ in $(seq 1 90); do
  if ss -ltn "sport = :${PORT}" 2>/dev/null | grep -q LISTEN; then
    sleep 2
    if ss -ltn "sport = :${PORT}" 2>/dev/null | grep -q LISTEN && pgrep -f coppeliaSim >/dev/null 2>&1; then
      echo "RPC ready."
      break
    fi
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

echo "Running CoppeliaSim ZMQ X-transport (target_dx=${TARGET_DX} m, duration=${DURATION}s)..."
cd "${ROOT}"
set +e
"${PYTHON_BIN}" "${RUNNER}" \
  --coppelia-root "${COPPELIA}" \
  --port "${PORT}" \
  --accel-x-transport \
  --accel-torque-policy ik_joint_pd \
  --target-dx "${TARGET_DX}" \
  --duration "${DURATION}" \
  --settle-duration "${SETTLE}" \
  --task-frame-mode mujoco_attachment_dummy \
  --video-camera smoke \
  --video-name "${VIDEO_NAME}" \
  --summary-name "${SUMMARY_NAME}" \
  --fps 20 \
  2>&1 | tee "${RUNNER_LOG}"
RUN_EXIT=${PIPESTATUS[0]}
set -e

echo ""
echo "=== Results ==="
if [[ -f "${VIDEO_OUT}" ]]; then
  ls -lh "${VIDEO_OUT}"
else
  echo "Video missing: ${VIDEO_OUT}"
fi
if [[ -f "${SUMMARY_OUT}" ]]; then
  python3 -c "import json; s=json.load(open('${SUMMARY_OUT}')); print('success=', s.get('success')); print('failure_reasons=', s.get('failure_reasons')); print('x_net_displacement_m=', s.get('x_net_displacement_m')); print('first_frame_std_rgb=', s.get('first_frame_std_rgb'))"
else
  echo "Summary missing: ${SUMMARY_OUT}"
fi
echo "Runner exit code: ${RUN_EXIT}"
exit "${RUN_EXIT}"
