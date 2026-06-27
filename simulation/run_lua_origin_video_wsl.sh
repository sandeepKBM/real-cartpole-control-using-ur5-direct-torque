#!/usr/bin/env bash
# CoppeliaSim Lua transport video (WSL, visible PNG path).
# Reference: EE green triad axis slides parallel to base green (world Y).
# Pure CoppeliaSim — not MuJoCo, not external ZMQ.
set -euo pipefail
ROOT="/mnt/c/Users/sandr/Downloads/real-cartpole-control-using-ur5-direct-torque"
# shellcheck disable=SC1091
source "${ROOT}/simulation/env_wsl_local.sh"

COPPELIA="${COPPELIA_ROOT}"
FRAME_DIR="${ROOT}/outputs/control_runs/coppelia_origin_acquisition_frames"
STATE_DIR="${ROOT}/outputs/control_runs/coppelia_origin_acquisition_state"
VIDEO_PATH="${ROOT}/demonstration_videos/ur5e_coppeliasim/coppeliasim_ur5_wsl_y_full_sweep.mp4"
SUMMARY_PATH="${STATE_DIR}/coppeliasim_ur5_wsl_y_full_sweep_summary.json"
ADDON_SOURCE="${ROOT}/simulation/ur5_origin_acquisition_video_addon.lua"
ADDON_TARGET="${COPPELIA}/addOns/ur5_origin_acquisition_video_addon.lua"
SIM_LOG="${STATE_DIR}/coppelia.log"
DONE_MARKER="${STATE_DIR}/ur5_origin_acquisition_done.txt"

FPS="${FPS:-25}"
SIM_TIMEOUT="${SIM_TIMEOUT:-180}"
TARGET_DX_M="${TARGET_DX_M:-0.06}"
EE_TARGET_Z_M="${EE_TARGET_Z_M:-0.4}"
A_X_MAX_MPS2="${A_X_MAX_MPS2:-3.0}"
V_X_MAX_MPS="${V_X_MAX_MPS:-0.6}"
START_AT_TRANSPORT_PLANE="${START_AT_TRANSPORT_PLANE:-1}"
TASK_FRAME_MODE="${TASK_FRAME_MODE:-mujoco_attachment_dummy}"
LOCK_SHOULDER_PAN="${LOCK_SHOULDER_PAN:-1}"
# Full end-to-end sweep on green (Y): EE green || base green, base pan locked.
MUJOCO_LIKE_X_SWEEP="${MUJOCO_LIKE_X_SWEEP:-1}"
MUJOCO_LIKE_SWEEP_AXIS="${MUJOCO_LIKE_SWEEP_AXIS:-y}"
MUJOCO_LIKE_SWEEP_LEGS="${MUJOCO_LIKE_SWEEP_LEGS:-3}"
if [[ "${MUJOCO_LIKE_X_SWEEP}" == "1" ]]; then
  ORIGIN_MOVE_FRAMES="${ORIGIN_MOVE_FRAMES:-1}"
  GAP_FRAMES="${GAP_FRAMES:-0}"
  ACCEL_FRAMES="${ACCEL_FRAMES:-360}"
  FRAME_COUNT="${FRAME_COUNT:-360}"
else
  if [[ "${START_AT_TRANSPORT_PLANE}" == "1" ]]; then
    ORIGIN_MOVE_FRAMES="${ORIGIN_MOVE_FRAMES:-1}"
  else
    ORIGIN_MOVE_FRAMES="${ORIGIN_MOVE_FRAMES:-80}"
  fi
  GAP_FRAMES="${GAP_FRAMES:-25}"
  ACCEL_FRAMES="${ACCEL_FRAMES:-80}"
  FRAME_COUNT="${FRAME_COUNT:-$((ORIGIN_MOVE_FRAMES + GAP_FRAMES + ACCEL_FRAMES))}"
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ERROR: ffmpeg required" >&2
  exit 1
fi

mkdir -p "${FRAME_DIR}" "${STATE_DIR}" "${COPPELIA}/addOns" "$(dirname "${VIDEO_PATH}")"
rm -f "${FRAME_DIR}"/frame_*.png "${VIDEO_PATH}" "${SIM_LOG}" "${SUMMARY_PATH}" "${DONE_MARKER}"
rm -f "${COPPELIA}/addOns/ur5_video_smoke_addon.lua"
rm -f "${COPPELIA}/addOns/ur5_controller_video_addon.lua"
cp -f "${ADDON_SOURCE}" "${ADDON_TARGET}"

export DISPLAY="${DISPLAY:-:0}"
export LD_LIBRARY_PATH="${COPPELIA}:${LD_LIBRARY_PATH:-}"
unset QT_QPA_PLATFORM

pkill -f coppeliaSim 2>/dev/null || true
sleep 1

echo "CoppeliaSim Lua transport video (green-on-green / world Y)"
echo "  COPPELIA_ROOT=${COPPELIA}"
echo "  DISPLAY=${DISPLAY}"
echo "  sweep_axis=${MUJOCO_LIKE_SWEEP_AXIS}  legs=${MUJOCO_LIKE_SWEEP_LEGS}  frames=${FRAME_COUNT}"
echo "  lock_shoulder_pan=${LOCK_SHOULDER_PAN}  task_frame=${TASK_FRAME_MODE}"
echo "  video=${VIDEO_PATH}"
echo "  frames_dir=${FRAME_DIR}"
echo "  summary=${SUMMARY_PATH}"

cd "${COPPELIA}"
COPPELIA_ROOT="${COPPELIA}" REAL_CARTPOLE_ROOT="${ROOT}" \
  FRAME_COUNT="${FRAME_COUNT}" FPS="${FPS}" \
  ORIGIN_MOVE_FRAMES="${ORIGIN_MOVE_FRAMES}" GAP_FRAMES="${GAP_FRAMES}" ACCEL_FRAMES="${ACCEL_FRAMES}" \
  TARGET_DX_M="${TARGET_DX_M}" V_X_MAX_MPS="${V_X_MAX_MPS}" A_X_MAX_MPS2="${A_X_MAX_MPS2}" \
  EE_TARGET_Z_M="${EE_TARGET_Z_M}" START_AT_TRANSPORT_PLANE="${START_AT_TRANSPORT_PLANE}" \
  TASK_FRAME_MODE="${TASK_FRAME_MODE}" LOCK_SHOULDER_PAN="${LOCK_SHOULDER_PAN}" \
  MUJOCO_LIKE_X_SWEEP="${MUJOCO_LIKE_X_SWEEP}" \
  MUJOCO_LIKE_SWEEP_AXIS="${MUJOCO_LIKE_SWEEP_AXIS}" \
  MUJOCO_LIKE_SWEEP_LEGS="${MUJOCO_LIKE_SWEEP_LEGS}" \
  VIDEO_PATH="${VIDEO_PATH}" SUMMARY_PATH="${SUMMARY_PATH}" \
  SHOW_EE_TRIAD=1 SHOW_BASE_TRIAD=1 \
  ./coppeliaSim.sh -h -vscriptinfos >"${SIM_LOG}" 2>&1 &
SIM_PID=$!

cleanup() {
  kill "${SIM_PID}" 2>/dev/null || true
  wait "${SIM_PID}" 2>/dev/null || true
  rm -f "${ADDON_TARGET}"
}
trap cleanup EXIT INT TERM

deadline=$((SECONDS + SIM_TIMEOUT))
got_frames=0
while kill -0 "${SIM_PID}" 2>/dev/null; do
  n=$(find "${FRAME_DIR}" -maxdepth 1 -type f -name 'frame_*.png' 2>/dev/null | wc -l | tr -d '[:space:]')
  if [[ "${n}" -ge "${FRAME_COUNT}" ]]; then
    got_frames=1
    break
  fi
  if [[ "${SECONDS}" -ge "${deadline}" ]]; then
    break
  fi
  sleep 1
done

if [[ "${got_frames}" -eq 1 ]]; then
  sleep 3
fi
kill "${SIM_PID}" 2>/dev/null || true
wait "${SIM_PID}" 2>/dev/null || true
SIM_PID=""
trap - EXIT INT TERM
rm -f "${ADDON_TARGET}"

n=$(find "${FRAME_DIR}" -maxdepth 1 -type f -name 'frame_*.png' 2>/dev/null | wc -l | tr -d '[:space:]')
if [[ "${n}" -lt 1 ]]; then
  echo "No frames captured. Log tail:" >&2
  tail -40 "${SIM_LOG}" >&2 || true
  exit 1
fi

echo "Encoding ${n} frames -> ${VIDEO_PATH}"
ffmpeg -y -framerate "${FPS}" -i "${FRAME_DIR}/frame_%08d.png" \
  -c:v libx264 -pix_fmt yuv420p "${VIDEO_PATH}"

if [[ -f "${SUMMARY_PATH}" ]]; then
  python3 -c "import json; s=json.load(open('${SUMMARY_PATH}')); print('success=', s.get('success')); print('sweep_axis=', s.get('sweep_axis_label')); print('lock_shoulder_pan=', s.get('lock_shoulder_pan')); print('shoulder_pan_locked_rad=', s.get('shoulder_pan_locked_rad')); print('q_start[0]=', (s.get('q_start_rad') or [None])[0]); print('q_final[0]=', (s.get('q_final_rad') or [None])[0]); print('axis_net_displacement_m=', s.get('axis_net_displacement_m'))"
else
  echo "Warning: no summary JSON at ${SUMMARY_PATH}"
fi

ls -lh "${VIDEO_PATH}"
echo "Done: ${VIDEO_PATH}"
