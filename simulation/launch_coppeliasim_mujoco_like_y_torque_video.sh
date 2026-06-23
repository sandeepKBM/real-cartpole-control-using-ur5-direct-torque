#!/usr/bin/env bash
set -euo pipefail

while [[ $# -gt 0 ]]; do
  case "$1" in
    --y-accel-preset)
      if [[ $# -lt 2 ]]; then
        echo "--y-accel-preset requires y_pos or y_neg." >&2
        exit 1
      fi
      Y_ACCEL_PRESET="$2"
      shift 2
      ;;
    --accel-direction)
      if [[ $# -lt 2 ]]; then
        echo "--accel-direction requires a direction value." >&2
        exit 1
      fi
      ACCEL_DIRECTION="$2"
      shift 2
      ;;
    --help|-h)
      cat <<'EOF'
Usage: bash simulation/launch_coppeliasim_mujoco_like_y_torque_video.sh [--y-accel-preset y_pos|y_neg] [--accel-direction ...]
EOF
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

ROOT="/common/users/ss5772/real_Cartpole"
COPPELIA_ROOT="${COPPELIA_ROOT:-${ROOT}/third_party/coppelia_runtime/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu24_04}"
ADDON_SOURCE="${ROOT}/simulation/ur5_mujoco_like_y_torque_addon.lua"
ADDON_TARGET="${COPPELIA_ROOT}/addOns/zz_ur5_mujoco_like_y_torque_addon.lua"
SMOKE_ADDON_TARGET="${COPPELIA_ROOT}/addOns/ur5_video_smoke_addon.lua"
CONTROLLER_ADDON_TARGET="${COPPELIA_ROOT}/addOns/ur5_controller_video_addon.lua"
ACCEL_ADDON_TARGET="${COPPELIA_ROOT}/addOns/ur5_acceleration_transport_video_addon.lua"
FIXED_Z_ACCEL_ADDON_TARGET="${COPPELIA_ROOT}/addOns/ur5_fixed_z_acceleration_transport_addon.lua"
ORIGIN_ADDON_TARGET="${COPPELIA_ROOT}/addOns/ur5_origin_acquisition_video_addon.lua"
LEGACY_TORQUE_ADDON_TARGET="${COPPELIA_ROOT}/addOns/ur5_mujoco_like_y_torque_addon.lua"
FPS="${FPS:-25}"
SIM_TIMEOUT="${SIM_TIMEOUT:-180}"
FRAME_COUNT="${FRAME_COUNT:-185}"
SETTLE_DURATION_S="${SETTLE_DURATION_S:-1.0}"
ACCEL_DIRECTION_REQUEST="${ACCEL_DIRECTION:-}"
if [[ -z "${ACCEL_DIRECTION_REQUEST}" && -n "${Y_ACCEL_PRESET:-}" ]]; then
  case "${Y_ACCEL_PRESET,,}" in
    y_pos|pos|positive)
      ACCEL_DIRECTION_REQUEST="1"
      ;;
    y_neg|neg|negative)
      ACCEL_DIRECTION_REQUEST="-1"
      ;;
    *)
      echo "Invalid Y_ACCEL_PRESET=${Y_ACCEL_PRESET}; expected y_pos or y_neg." >&2
      exit 1
      ;;
  esac
fi
if [[ -z "${ACCEL_DIRECTION_REQUEST}" ]]; then
  ACCEL_DIRECTION_REQUEST="1"
  ACCEL_DIRECTION_SOURCE="internal_default"
else
  ACCEL_DIRECTION_SOURCE="env_override"
fi
case "${ACCEL_DIRECTION_REQUEST,,}" in
  1|+1|y+|+y|positive|pos)
    ACCEL_DIRECTION=1
    RUN_LABEL="pos"
    ;;
  -1|y-|-y|negative|neg)
    ACCEL_DIRECTION=-1
    RUN_LABEL="neg"
    ;;
  *)
    echo "Invalid ACCEL_DIRECTION=${ACCEL_DIRECTION_REQUEST}; expected 1/+1/y+/+y/positive or -1/y-/-y/negative." >&2
    exit 1
    ;;
esac

TRAVEL_DISTANCE_ENV="${TRAVEL_DISTANCE_M:-}"
TRAVEL_DISTANCE_FALLBACK="${TARGET_DX_M:-}"
if [[ -n "${TRAVEL_DISTANCE_ENV}" ]]; then
  TRAVEL_DISTANCE_M="${TRAVEL_DISTANCE_ENV}"
  TRAVEL_DISTANCE_SOURCE="env_override"
elif [[ -n "${TRAVEL_DISTANCE_FALLBACK}" ]]; then
  TRAVEL_DISTANCE_M="${TRAVEL_DISTANCE_FALLBACK}"
  TRAVEL_DISTANCE_SOURCE="compatibility_fallback_input"
else
  TRAVEL_DISTANCE_M="0.35"
  TRAVEL_DISTANCE_SOURCE="internal_default"
fi

ACCEL_MAGNITUDE_ENV="${ACCEL_MAGNITUDE_MPS2:-}"
ACCEL_MAGNITUDE_FALLBACK="${A_AXIS_MAX_MPS2:-}"
if [[ -n "${ACCEL_MAGNITUDE_ENV}" ]]; then
  ACCEL_MAGNITUDE_MPS2="${ACCEL_MAGNITUDE_ENV}"
  ACCEL_MAGNITUDE_SOURCE="env_override"
elif [[ -n "${ACCEL_MAGNITUDE_FALLBACK}" ]]; then
  ACCEL_MAGNITUDE_MPS2="${ACCEL_MAGNITUDE_FALLBACK}"
  ACCEL_MAGNITUDE_SOURCE="compatibility_fallback_input"
else
  ACCEL_MAGNITUDE_MPS2="0.25"
  ACCEL_MAGNITUDE_SOURCE="internal_default"
fi

TARGET_DX_M="${TARGET_DX_M:-${TRAVEL_DISTANCE_M}}"
A_AXIS_MAX_MPS2="${A_AXIS_MAX_MPS2:-${ACCEL_MAGNITUDE_MPS2}}"
V_AXIS_MAX_MPS="${V_AXIS_MAX_MPS:-0.12}"
TRANSPORT_AXIS="${TRANSPORT_AXIS:-y}"
RUN_ROOT="${OUTPUT_DIR:-${ROOT}/outputs/control_runs/coppelia_y_accel_direction_${RUN_LABEL}}"
FRAME_DIR="${FRAME_DIR:-${RUN_ROOT}/frames}"
STATE_DIR="${STATE_DIR:-${RUN_ROOT}/state}"
SIM_LOG="${STATE_DIR}/coppelia.log"
BOOT_LOG="${STATE_DIR}/bootstrap.log"
VIDEO_PATH="${VIDEO_PATH:-${ROOT}/demonstration_videos/ur5e_coppeliasim/coppeliasim_ur5_y_accel_direction_${RUN_LABEL}.mp4}"
SUMMARY_PATH="${SUMMARY_PATH:-${STATE_DIR}/coppeliasim_ur5_y_accel_direction_${RUN_LABEL}_summary.json}"
LOAD_MARKER="${STATE_DIR}/ur5_y_accel_direction_addon_loaded.txt"
START_MARKER="${STATE_DIR}/ur5_y_accel_direction_addon_started.txt"
SENSING_MARKER="${STATE_DIR}/ur5_y_accel_direction_addon_sensing.txt"
DONE_MARKER="${STATE_DIR}/ur5_y_accel_direction_done.txt"
CONFIGURED_MARKER="${STATE_DIR}/ur5_y_accel_direction_configured.txt"
TASK_FRAME_MODE="${TASK_FRAME_MODE:-mujoco_attachment_dummy}"
TASK_ORIENTATION_TARGET="${TASK_ORIENTATION_TARGET:-initial}"
TASK_FRAME_LOCAL_Z_OFFSET_M="${TASK_FRAME_LOCAL_Z_OFFSET_M:--0.2}"
TRANSPORT_PLANE_Z_OFFSET_M="${TRANSPORT_PLANE_Z_OFFSET_M:-0.0}"

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
rm -f "${LOAD_MARKER}" "${START_MARKER}" "${SENSING_MARKER}" "${DONE_MARKER}" "${CONFIGURED_MARKER}"
rm -f "${SMOKE_ADDON_TARGET}" "${CONTROLLER_ADDON_TARGET}" "${ACCEL_ADDON_TARGET}" "${FIXED_Z_ACCEL_ADDON_TARGET}" "${ORIGIN_ADDON_TARGET}" "${LEGACY_TORQUE_ADDON_TARGET}"
cp -f "${ADDON_SOURCE}" "${ADDON_TARGET}"

log_bootstrap() {
  printf '%s\n' "$*" >>"${BOOT_LOG}"
}

log_bootstrap "COPPELIA_ROOT=${COPPELIA_ROOT}"
log_bootstrap "ADDON_SOURCE=${ADDON_SOURCE}"
log_bootstrap "RUN_ROOT=${RUN_ROOT}"
log_bootstrap "RUN_LABEL=${RUN_LABEL}"
log_bootstrap "FRAME_COUNT=${FRAME_COUNT}"
log_bootstrap "FPS=${FPS}"
log_bootstrap "SETTLE_DURATION_S=${SETTLE_DURATION_S}"
log_bootstrap "Y_ACCEL_PRESET=${Y_ACCEL_PRESET:-<unset>}"
log_bootstrap "ACCEL_DIRECTION_REQUEST=${ACCEL_DIRECTION_REQUEST}"
log_bootstrap "ACCEL_DIRECTION_SOURCE=${ACCEL_DIRECTION_SOURCE}"
log_bootstrap "ACCEL_DIRECTION=${ACCEL_DIRECTION}"
log_bootstrap "TRAVEL_DISTANCE_M=${TRAVEL_DISTANCE_M}"
log_bootstrap "TRAVEL_DISTANCE_SOURCE=${TRAVEL_DISTANCE_SOURCE}"
log_bootstrap "ACCEL_DIRECTION=${ACCEL_DIRECTION}"
log_bootstrap "ACCEL_MAGNITUDE_MPS2=${ACCEL_MAGNITUDE_MPS2}"
log_bootstrap "ACCEL_MAGNITUDE_SOURCE=${ACCEL_MAGNITUDE_SOURCE}"
log_bootstrap "TARGET_DX_M=${TARGET_DX_M}"
log_bootstrap "A_AXIS_MAX_MPS2=${A_AXIS_MAX_MPS2}"
log_bootstrap "V_AXIS_MAX_MPS=${V_AXIS_MAX_MPS}"
log_bootstrap "TRANSPORT_AXIS=${TRANSPORT_AXIS}"
log_bootstrap "TASK_FRAME_MODE=${TASK_FRAME_MODE}"
log_bootstrap "TASK_ORIENTATION_TARGET=${TASK_ORIENTATION_TARGET}"
log_bootstrap "TASK_FRAME_LOCAL_Z_OFFSET_M=${TASK_FRAME_LOCAL_Z_OFFSET_M}"
log_bootstrap "TRANSPORT_PLANE_Z_OFFSET_M=${TRANSPORT_PLANE_Z_OFFSET_M}"
log_bootstrap "CONFIGURED_MARKER=${CONFIGURED_MARKER}"
log_bootstrap "torque_loop_mode=internal_lua_render"

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
  DISPLAY="${DISPLAY_VALUE}" XAUTHORITY="${XAUTH_VALUE}" COPPELIA_ROOT="${COPPELIA_ROOT}" REAL_CARTPOLE_ROOT="${ROOT}" OUTPUT_DIR="${FRAME_DIR}" STATE_DIR="${STATE_DIR}" VIDEO_PATH="${VIDEO_PATH}" SUMMARY_PATH="${SUMMARY_PATH}" FRAME_PREFIX="frame" FRAME_COUNT="${FRAME_COUNT}" FPS="${FPS}" SETTLE_DURATION_S="${SETTLE_DURATION_S}" Y_ACCEL_PRESET="${Y_ACCEL_PRESET:-}" TRAVEL_DISTANCE_M="${TRAVEL_DISTANCE_M}" TRAVEL_DISTANCE_SOURCE="${TRAVEL_DISTANCE_SOURCE}" TARGET_DX_M="${TARGET_DX_M}" ACCEL_DIRECTION="${ACCEL_DIRECTION}" ACCEL_DIRECTION_SOURCE="${ACCEL_DIRECTION_SOURCE}" ACCEL_MAGNITUDE_MPS2="${ACCEL_MAGNITUDE_MPS2}" ACCEL_MAGNITUDE_SOURCE="${ACCEL_MAGNITUDE_SOURCE}" A_AXIS_MAX_MPS2="${A_AXIS_MAX_MPS2}" V_AXIS_MAX_MPS="${V_AXIS_MAX_MPS}" TRANSPORT_AXIS="${TRANSPORT_AXIS}" TASK_FRAME_MODE="${TASK_FRAME_MODE}" TASK_ORIENTATION_TARGET="${TASK_ORIENTATION_TARGET}" TASK_FRAME_LOCAL_Z_OFFSET_M="${TASK_FRAME_LOCAL_Z_OFFSET_M}" TRANSPORT_PLANE_Z_OFFSET_M="${TRANSPORT_PLANE_Z_OFFSET_M}" SHOW_EE_TRIAD=1 SHOW_BASE_TRIAD=1 USE_EXTERNAL_STEP_PUMP=0 \
    "${COPPELIA_ROOT}/coppeliaSim.sh" -h -vscriptinfos \
    >"${SIM_LOG}" 2>&1 &
  SIM_PID=$!
else
  if [[ "${USE_RAW_XVFB}" == "1" ]] && command -v Xvfb >/dev/null 2>&1; then
    log_bootstrap "launch_branch=raw_xvfb"
    COPPELIA_ROOT="${COPPELIA_ROOT}" REAL_CARTPOLE_ROOT="${ROOT}" OUTPUT_DIR="${FRAME_DIR}" STATE_DIR="${STATE_DIR}" VIDEO_PATH="${VIDEO_PATH}" SUMMARY_PATH="${SUMMARY_PATH}" FRAME_PREFIX="frame" FRAME_COUNT="${FRAME_COUNT}" FPS="${FPS}" SETTLE_DURATION_S="${SETTLE_DURATION_S}" Y_ACCEL_PRESET="${Y_ACCEL_PRESET:-}" TRAVEL_DISTANCE_M="${TRAVEL_DISTANCE_M}" TRAVEL_DISTANCE_SOURCE="${TRAVEL_DISTANCE_SOURCE}" TARGET_DX_M="${TARGET_DX_M}" ACCEL_DIRECTION="${ACCEL_DIRECTION}" ACCEL_DIRECTION_SOURCE="${ACCEL_DIRECTION_SOURCE}" ACCEL_MAGNITUDE_MPS2="${ACCEL_MAGNITUDE_MPS2}" ACCEL_MAGNITUDE_SOURCE="${ACCEL_MAGNITUDE_SOURCE}" A_AXIS_MAX_MPS2="${A_AXIS_MAX_MPS2}" V_AXIS_MAX_MPS="${V_AXIS_MAX_MPS}" TRANSPORT_AXIS="${TRANSPORT_AXIS}" TASK_FRAME_MODE="${TASK_FRAME_MODE}" TASK_ORIENTATION_TARGET="${TASK_ORIENTATION_TARGET}" TASK_FRAME_LOCAL_Z_OFFSET_M="${TASK_FRAME_LOCAL_Z_OFFSET_M}" TRANSPORT_PLANE_Z_OFFSET_M="${TRANSPORT_PLANE_Z_OFFSET_M}" SHOW_EE_TRIAD=1 SHOW_BASE_TRIAD=1 USE_EXTERNAL_STEP_PUMP=0 \
      "${COPPELIA_ROOT}/coppeliaSim.sh" -h -vscriptinfos \
      >"${SIM_LOG}" 2>&1 &
    SIM_PID=$!
  else
    log_bootstrap "launch_branch=xvfb-run"
    xvfb-run -a /usr/bin/env COPPELIA_ROOT="${COPPELIA_ROOT}" REAL_CARTPOLE_ROOT="${ROOT}" OUTPUT_DIR="${FRAME_DIR}" STATE_DIR="${STATE_DIR}" VIDEO_PATH="${VIDEO_PATH}" SUMMARY_PATH="${SUMMARY_PATH}" FRAME_PREFIX="frame" FRAME_COUNT="${FRAME_COUNT}" FPS="${FPS}" SETTLE_DURATION_S="${SETTLE_DURATION_S}" Y_ACCEL_PRESET="${Y_ACCEL_PRESET:-}" TRAVEL_DISTANCE_M="${TRAVEL_DISTANCE_M}" TRAVEL_DISTANCE_SOURCE="${TRAVEL_DISTANCE_SOURCE}" TARGET_DX_M="${TARGET_DX_M}" ACCEL_DIRECTION="${ACCEL_DIRECTION}" ACCEL_DIRECTION_SOURCE="${ACCEL_DIRECTION_SOURCE}" ACCEL_MAGNITUDE_MPS2="${ACCEL_MAGNITUDE_MPS2}" ACCEL_MAGNITUDE_SOURCE="${ACCEL_MAGNITUDE_SOURCE}" A_AXIS_MAX_MPS2="${A_AXIS_MAX_MPS2}" V_AXIS_MAX_MPS="${V_AXIS_MAX_MPS}" TRANSPORT_AXIS="${TRANSPORT_AXIS}" TASK_FRAME_MODE="${TASK_FRAME_MODE}" TASK_ORIENTATION_TARGET="${TASK_ORIENTATION_TARGET}" TASK_FRAME_LOCAL_Z_OFFSET_M="${TASK_FRAME_LOCAL_Z_OFFSET_M}" TRANSPORT_PLANE_Z_OFFSET_M="${TRANSPORT_PLANE_Z_OFFSET_M}" SHOW_EE_TRIAD=1 SHOW_BASE_TRIAD=1 USE_EXTERNAL_STEP_PUMP=0 \
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
  echo "No torque-controller video frames were captured." >&2
  [[ -f "${LOAD_MARKER}" ]] && echo "Load marker present." >&2 || echo "Load marker missing." >&2
  [[ -f "${START_MARKER}" ]] && echo "Start marker present." >&2 || echo "Start marker missing." >&2
  [[ -f "${SENSING_MARKER}" ]] && echo "Sensing marker present." >&2 || echo "Sensing marker missing." >&2
  [[ -f "${DONE_MARKER}" ]] && echo "Done marker present." >&2 || echo "Done marker missing." >&2
  sed -n '1,260p' "${SIM_LOG}" >&2 || true
  exit 1
fi

captured_count=$(find "${FRAME_DIR}" -maxdepth 1 -type f -name 'frame_*.png' | wc -l | tr -d '[:space:]')
if [[ "${captured_count}" -lt "${FRAME_COUNT}" ]]; then
  echo "Only ${captured_count}/${FRAME_COUNT} torque-controller video frames were captured." >&2
  [[ -f "${STEPPER_LOG}" ]] && sed -n '1,260p' "${STEPPER_LOG}" >&2 || true
  [[ -f "${SIM_LOG}" ]] && sed -n '1,260p' "${SIM_LOG}" >&2 || true
  exit 1
fi

ffmpeg -y -framerate "${FPS}" -i "${FRAME_DIR}/frame_%08d.png" \
  -c:v libx264 -pix_fmt yuv420p -movflags +faststart \
  "${VIDEO_PATH}" >/dev/null 2>&1

if [[ ! -f "${VIDEO_PATH}" ]]; then
  echo "Failed to encode torque-controller video." >&2
  exit 1
fi

echo "Torque-controller video written to ${VIDEO_PATH}"
