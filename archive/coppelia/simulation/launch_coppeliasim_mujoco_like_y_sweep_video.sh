#!/usr/bin/env bash
set -euo pipefail

ROOT="/common/users/ss5772/real_Cartpole"
COPPELIA_ROOT="${COPPELIA_ROOT:-${ROOT}/third_party/coppelia_runtime/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu24_04}"
FRAME_DIR="${ROOT}/outputs/control_runs/coppelia_mujoco_like_y_sweep_frames"
STATE_DIR="${ROOT}/outputs/control_runs/coppelia_mujoco_like_y_sweep_state"
SIM_LOG="${STATE_DIR}/coppelia.log"
BOOT_LOG="${STATE_DIR}/bootstrap.log"
VIDEO_PATH="${ROOT}/demonstration_videos/ur5e_coppeliasim/coppeliasim_ur5_mujoco_like_y_sweep.mp4"
SUMMARY_PATH="${STATE_DIR}/coppeliasim_ur5_mujoco_like_y_sweep_summary.json"
ADDON_SOURCE="${ROOT}/simulation/ur5_origin_acquisition_video_addon.lua"
ADDON_TARGET="${COPPELIA_ROOT}/addOns/ur5_mujoco_like_y_sweep_video_addon.lua"
SMOKE_ADDON_TARGET="${COPPELIA_ROOT}/addOns/ur5_video_smoke_addon.lua"
CONTROLLER_ADDON_TARGET="${COPPELIA_ROOT}/addOns/ur5_controller_video_addon.lua"
ACCEL_ADDON_TARGET="${COPPELIA_ROOT}/addOns/ur5_acceleration_transport_video_addon.lua"
FIXED_Z_ACCEL_ADDON_TARGET="${COPPELIA_ROOT}/addOns/ur5_fixed_z_acceleration_transport_addon.lua"
ORIGIN_ADDON_TARGET="${COPPELIA_ROOT}/addOns/ur5_origin_acquisition_video_addon.lua"
START_MARKER="${STATE_DIR}/ur5_mujoco_like_y_sweep_addon_started.txt"
LOAD_MARKER="${STATE_DIR}/ur5_mujoco_like_y_sweep_addon_loaded.txt"
SENSING_MARKER="${STATE_DIR}/ur5_mujoco_like_y_sweep_addon_sensing.txt"
DONE_MARKER="${STATE_DIR}/ur5_mujoco_like_y_sweep_done.txt"
FPS="${FPS:-15}"
SIM_TIMEOUT="${SIM_TIMEOUT:-120}"
TARGET_DX_M="${TARGET_DX_M:-0.060}"
V_X_MAX_MPS="${V_X_MAX_MPS:-0.12}"
A_X_MAX_MPS2="${A_X_MAX_MPS2:-0.25}"
EE_TARGET_Z_M="${EE_TARGET_Z_M:-0.540}"
ACCEL_ROT_WEIGHT="${ACCEL_ROT_WEIGHT:-0.05}"
MODEL_BASE_Z_OFFSET_M="${MODEL_BASE_Z_OFFSET_M:-0.0}"
START_AT_TRANSPORT_PLANE="${START_AT_TRANSPORT_PLANE:-1}"
MUJOCO_LIKE_X_SWEEP="${MUJOCO_LIKE_X_SWEEP:-1}"
MUJOCO_LIKE_SWEEP_AXIS="${MUJOCO_LIKE_SWEEP_AXIS:-y}"
TASK_FRAME_MODE="${TASK_FRAME_MODE:-mujoco_attachment_dummy}"
MUJOCO_LIKE_SWEEP_LEGS="${MUJOCO_LIKE_SWEEP_LEGS:-3}"
FRAME_COUNT="${FRAME_COUNT:-744}"
ORIGIN_MOVE_FRAMES="${ORIGIN_MOVE_FRAMES:-1}"
GAP_FRAMES="${GAP_FRAMES:-0}"
ACCEL_FRAMES="${ACCEL_FRAMES:-744}"

cleanup() {
  if [[ -n "${SIM_PID:-}" ]] && kill -0 "${SIM_PID}" 2>/dev/null; then
    kill "${SIM_PID}" 2>/dev/null || true
    wait "${SIM_PID}" 2>/dev/null || true
  fi
  if [[ -n "${LOCAL_XVFB_PID:-}" ]] && kill -0 "${LOCAL_XVFB_PID}" 2>/dev/null; then
    kill "${LOCAL_XVFB_PID}" 2>/dev/null || true
    wait "${LOCAL_XVFB_PID}" 2>/dev/null || true
  fi
  rm -f "${ADDON_TARGET}"
}
trap cleanup EXIT INT TERM

if [[ ! -x "${COPPELIA_ROOT}/coppeliaSim.sh" ]]; then
  echo "Missing CoppeliaSim at ${COPPELIA_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${ADDON_SOURCE}" ]]; then
  echo "Missing add-on at ${ADDON_SOURCE}" >&2
  exit 1
fi

mkdir -p "${FRAME_DIR}"
mkdir -p "${STATE_DIR}"
mkdir -p "$(dirname "${VIDEO_PATH}")"
mkdir -p "${COPPELIA_ROOT}/addOns"

rm -f "${FRAME_DIR}"/frame_*.png "${VIDEO_PATH}" "${SIM_LOG}" "${BOOT_LOG}" "${SUMMARY_PATH}"
rm -f "${LOAD_MARKER}" "${START_MARKER}" "${SENSING_MARKER}" "${DONE_MARKER}"
rm -f "${SMOKE_ADDON_TARGET}" "${CONTROLLER_ADDON_TARGET}" "${ACCEL_ADDON_TARGET}" "${FIXED_Z_ACCEL_ADDON_TARGET}" "${ORIGIN_ADDON_TARGET}"
cp -f "${ADDON_SOURCE}" "${ADDON_TARGET}"

log_bootstrap() {
  printf '%s\n' "$*" >>"${BOOT_LOG}"
}

log_bootstrap "COPPELIA_ROOT=${COPPELIA_ROOT}"
log_bootstrap "ADDON_SOURCE=${ADDON_SOURCE}"
log_bootstrap "FRAME_COUNT=${FRAME_COUNT}"
log_bootstrap "ORIGIN_MOVE_FRAMES=${ORIGIN_MOVE_FRAMES}"
log_bootstrap "GAP_FRAMES=${GAP_FRAMES}"
log_bootstrap "ACCEL_FRAMES=${ACCEL_FRAMES}"
log_bootstrap "TARGET_DX_M=${TARGET_DX_M}"
log_bootstrap "V_X_MAX_MPS=${V_X_MAX_MPS}"
log_bootstrap "A_X_MAX_MPS2=${A_X_MAX_MPS2}"
log_bootstrap "EE_TARGET_Z_M=${EE_TARGET_Z_M}"
log_bootstrap "ACCEL_ROT_WEIGHT=${ACCEL_ROT_WEIGHT}"
log_bootstrap "MODEL_BASE_Z_OFFSET_M=${MODEL_BASE_Z_OFFSET_M}"
log_bootstrap "START_AT_TRANSPORT_PLANE=${START_AT_TRANSPORT_PLANE}"
log_bootstrap "MUJOCO_LIKE_X_SWEEP=${MUJOCO_LIKE_X_SWEEP}"
log_bootstrap "MUJOCO_LIKE_SWEEP_AXIS=${MUJOCO_LIKE_SWEEP_AXIS}"
log_bootstrap "TASK_FRAME_MODE=${TASK_FRAME_MODE}"
log_bootstrap "MUJOCO_LIKE_SWEEP_LEGS=${MUJOCO_LIKE_SWEEP_LEGS}"

choose_display() {
  local line
  while IFS= read -r line; do
    [[ -z "${line}" ]] && continue
    local disp="${line%%|*}"
    local auth="${line#*|}"
    if [[ -n "${auth}" ]] && DISPLAY="${disp}" XAUTHORITY="${auth}" xdpyinfo >/dev/null 2>&1; then
      echo "${disp}|${auth}"
      return 0
    fi
  done < <(ps -ef | awk '
    /[X]vfb :[0-9]+/ {
      d=""; a="";
      for (i=1;i<=NF;i++) {
        if ($i ~ /^:[0-9]+$/) d=$i;
        if ($i == "-auth" && i < NF) a=$(i+1);
      }
      if (d != "") print d "|" a;
    }
  ')
  return 1
}

wait_for_provided_display() {
  local attempt
  for attempt in 1 2 3 4 5 6 7 8 9 10; do
    if DISPLAY="${DISPLAY}" XAUTHORITY="${XAUTHORITY}" xdpyinfo >/dev/null 2>&1; then
      echo "${DISPLAY}|${XAUTHORITY}"
      return 0
    fi
    sleep 1
  done
  return 1
}

USE_RAW_XVFB="${COPPELIA_USE_RAW_XVFB:-0}"

start_raw_xvfb() {
  local display_num
  local display
  for display_num in 91 92 93 94 95 96 97 98 99 100; do
    display=":${display_num}"
    Xvfb "${display}" -screen 0 1920x1080x24 -nolisten tcp -ac \
      >"${STATE_DIR}/xvfb-${display_num}.log" 2>&1 &
    LOCAL_XVFB_PID=$!
    sleep 2
    if DISPLAY="${display}" xdpyinfo >/dev/null 2>&1; then
      echo "${display}|"
      return 0
    fi
    kill "${LOCAL_XVFB_PID}" 2>/dev/null || true
    wait "${LOCAL_XVFB_PID}" 2>/dev/null || true
  done
  return 1
}

DISPLAY_SPEC="${COPPELIA_DISPLAY_SPEC:-}"
if [[ -z "${DISPLAY_SPEC}" ]] && [[ -n "${DISPLAY:-}" ]] && [[ -n "${XAUTHORITY:-}" ]]; then
  DISPLAY_SPEC="$(wait_for_provided_display || true)"
fi
if [[ -z "${DISPLAY_SPEC}" ]]; then
  DISPLAY_SPEC="$(choose_display || true)"
fi
if [[ -z "${DISPLAY_SPEC}" ]]; then
  if [[ "${USE_RAW_XVFB}" == "1" ]] && command -v Xvfb >/dev/null 2>&1; then
    DISPLAY_SPEC="$(start_raw_xvfb || true)"
  fi
fi
log_bootstrap "DISPLAY_SPEC=${DISPLAY_SPEC:-<none>}"

cd "${COPPELIA_ROOT}"
if [[ -n "${DISPLAY_SPEC}" ]]; then
  DISPLAY_VALUE="${DISPLAY_SPEC%%|*}"
  XAUTH_VALUE="${DISPLAY_SPEC#*|}"
  log_bootstrap "launch_branch=existing_display"
  DISPLAY="${DISPLAY_VALUE}" XAUTHORITY="${XAUTH_VALUE}" COPPELIA_ROOT="${COPPELIA_ROOT}" REAL_CARTPOLE_ROOT="${ROOT}" OUTPUT_DIR="${FRAME_DIR}" STATE_DIR="${STATE_DIR}" VIDEO_PATH="${VIDEO_PATH}" SUMMARY_PATH="${SUMMARY_PATH}" FRAME_PREFIX="frame" FRAME_COUNT="${FRAME_COUNT}" FPS="${FPS}" ORIGIN_MOVE_FRAMES="${ORIGIN_MOVE_FRAMES}" GAP_FRAMES="${GAP_FRAMES}" ACCEL_FRAMES="${ACCEL_FRAMES}" TARGET_DX_M="${TARGET_DX_M}" V_X_MAX_MPS="${V_X_MAX_MPS}" A_X_MAX_MPS2="${A_X_MAX_MPS2}" EE_TARGET_Z_M="${EE_TARGET_Z_M}" ACCEL_ROT_WEIGHT="${ACCEL_ROT_WEIGHT}" MODEL_BASE_Z_OFFSET_M="${MODEL_BASE_Z_OFFSET_M}" START_AT_TRANSPORT_PLANE="${START_AT_TRANSPORT_PLANE}" MUJOCO_LIKE_X_SWEEP="${MUJOCO_LIKE_X_SWEEP}" MUJOCO_LIKE_SWEEP_AXIS="${MUJOCO_LIKE_SWEEP_AXIS}" TASK_FRAME_MODE="${TASK_FRAME_MODE}" MUJOCO_LIKE_SWEEP_LEGS="${MUJOCO_LIKE_SWEEP_LEGS}" SHOW_EE_TRIAD=1 SHOW_BASE_TRIAD=1 \
    "${COPPELIA_ROOT}/coppeliaSim.sh" -h -vscriptinfos \
    >"${SIM_LOG}" 2>&1 &
  SIM_PID=$!
else
  if [[ "${USE_RAW_XVFB}" == "1" ]] && command -v Xvfb >/dev/null 2>&1; then
    log_bootstrap "launch_branch=raw_xvfb"
    COPPELIA_ROOT="${COPPELIA_ROOT}" REAL_CARTPOLE_ROOT="${ROOT}" OUTPUT_DIR="${FRAME_DIR}" STATE_DIR="${STATE_DIR}" VIDEO_PATH="${VIDEO_PATH}" SUMMARY_PATH="${SUMMARY_PATH}" FRAME_PREFIX="frame" FRAME_COUNT="${FRAME_COUNT}" FPS="${FPS}" ORIGIN_MOVE_FRAMES="${ORIGIN_MOVE_FRAMES}" GAP_FRAMES="${GAP_FRAMES}" ACCEL_FRAMES="${ACCEL_FRAMES}" TARGET_DX_M="${TARGET_DX_M}" V_X_MAX_MPS="${V_X_MAX_MPS}" A_X_MAX_MPS2="${A_X_MAX_MPS2}" EE_TARGET_Z_M="${EE_TARGET_Z_M}" ACCEL_ROT_WEIGHT="${ACCEL_ROT_WEIGHT}" MODEL_BASE_Z_OFFSET_M="${MODEL_BASE_Z_OFFSET_M}" START_AT_TRANSPORT_PLANE="${START_AT_TRANSPORT_PLANE}" MUJOCO_LIKE_X_SWEEP="${MUJOCO_LIKE_X_SWEEP}" MUJOCO_LIKE_SWEEP_AXIS="${MUJOCO_LIKE_SWEEP_AXIS}" TASK_FRAME_MODE="${TASK_FRAME_MODE}" MUJOCO_LIKE_SWEEP_LEGS="${MUJOCO_LIKE_SWEEP_LEGS}" SHOW_EE_TRIAD=1 SHOW_BASE_TRIAD=1 \
      "${COPPELIA_ROOT}/coppeliaSim.sh" -h -vscriptinfos \
      >"${SIM_LOG}" 2>&1 &
    SIM_PID=$!
  else
    log_bootstrap "launch_branch=xvfb-run"
    xvfb-run -a /usr/bin/env COPPELIA_ROOT="${COPPELIA_ROOT}" REAL_CARTPOLE_ROOT="${ROOT}" OUTPUT_DIR="${FRAME_DIR}" STATE_DIR="${STATE_DIR}" VIDEO_PATH="${VIDEO_PATH}" SUMMARY_PATH="${SUMMARY_PATH}" FRAME_PREFIX="frame" FRAME_COUNT="${FRAME_COUNT}" FPS="${FPS}" ORIGIN_MOVE_FRAMES="${ORIGIN_MOVE_FRAMES}" GAP_FRAMES="${GAP_FRAMES}" ACCEL_FRAMES="${ACCEL_FRAMES}" TARGET_DX_M="${TARGET_DX_M}" V_X_MAX_MPS="${V_X_MAX_MPS}" A_X_MAX_MPS2="${A_X_MAX_MPS2}" EE_TARGET_Z_M="${EE_TARGET_Z_M}" ACCEL_ROT_WEIGHT="${ACCEL_ROT_WEIGHT}" MODEL_BASE_Z_OFFSET_M="${MODEL_BASE_Z_OFFSET_M}" START_AT_TRANSPORT_PLANE="${START_AT_TRANSPORT_PLANE}" MUJOCO_LIKE_X_SWEEP="${MUJOCO_LIKE_X_SWEEP}" MUJOCO_LIKE_SWEEP_AXIS="${MUJOCO_LIKE_SWEEP_AXIS}" TASK_FRAME_MODE="${TASK_FRAME_MODE}" MUJOCO_LIKE_SWEEP_LEGS="${MUJOCO_LIKE_SWEEP_LEGS}" SHOW_EE_TRIAD=1 SHOW_BASE_TRIAD=1 \
      "${COPPELIA_ROOT}/coppeliaSim.sh" -h -vscriptinfos \
      >"${SIM_LOG}" 2>&1 &
    SIM_PID=$!
  fi
fi

deadline=$((SECONDS + SIM_TIMEOUT))
frame_target_reached=0
while kill -0 "${SIM_PID}" 2>/dev/null; do
  frame_count=$(find "${FRAME_DIR}" -maxdepth 1 -type f -name 'frame_*.png' | wc -l | tr -d '[:space:]')
  if [[ "${frame_count}" -ge "${FRAME_COUNT}" ]]; then
    frame_target_reached=1
    break
  fi
  if [[ "${SECONDS}" -ge "${deadline}" ]]; then
    break
  fi
  sleep 1
done

if [[ "${frame_target_reached}" -eq 1 ]]; then
  grace_deadline=$((SECONDS + 5))
  while kill -0 "${SIM_PID}" 2>/dev/null && [[ ! -f "${DONE_MARKER}" ]] && [[ "${SECONDS}" -lt "${grace_deadline}" ]]; do
    sleep 1
  done
fi

if kill -0 "${SIM_PID}" 2>/dev/null; then
  kill "${SIM_PID}" 2>/dev/null || true
  wait "${SIM_PID}" 2>/dev/null || true
else
  wait "${SIM_PID}" 2>/dev/null || true
fi
SIM_PID=""

if ! compgen -G "${FRAME_DIR}/frame_*.png" > /dev/null; then
  echo "No MuJoCo-like sweep video frames were captured." >&2
  [[ -f "${LOAD_MARKER}" ]] && echo "Load marker present." >&2 || echo "Load marker missing." >&2
  [[ -f "${START_MARKER}" ]] && echo "Start marker present." >&2 || echo "Start marker missing." >&2
  [[ -f "${SENSING_MARKER}" ]] && echo "Sensing marker present." >&2 || echo "Sensing marker missing." >&2
  [[ -f "${DONE_MARKER}" ]] && echo "Done marker present." >&2 || echo "Done marker missing." >&2
  sed -n '1,260p' "${SIM_LOG}" >&2 || true
  exit 1
fi

ffmpeg -y -framerate "${FPS}" -i "${FRAME_DIR}/frame_%08d.png" \
  -c:v libx264 -pix_fmt yuv420p -movflags +faststart \
  "${VIDEO_PATH}" >/dev/null 2>&1

if [[ ! -f "${VIDEO_PATH}" ]]; then
  echo "Failed to encode MuJoCo-like sweep video." >&2
  exit 1
fi

echo "MuJoCo-like sweep video written to ${VIDEO_PATH}"
