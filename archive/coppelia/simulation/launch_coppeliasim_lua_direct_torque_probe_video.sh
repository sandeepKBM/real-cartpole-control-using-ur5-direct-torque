#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COPPELIA_EXE_DEFAULT="${ROOT}/third_party/coppelia_runtime/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu24_04/coppeliaSim.sh"
COPPELIA_ROOT_DEFAULT="${ROOT}/third_party/coppelia_runtime/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu24_04"

COPPELIASIM_EXE="${COPPELIASIM_EXE:-${COPPELIA_EXE_DEFAULT}}"
COPPELIA_ROOT="${COPPELIA_ROOT:-$(cd "$(dirname "${COPPELIASIM_EXE}")" && pwd)}"
COPPELIASIM_SCENE="${COPPELIASIM_SCENE:-${COPPELIA_ROOT}/system/dfltscn.ttt}"
ADDON_SOURCE="${ROOT}/simulation/ur5_lua_direct_torque_probe_addon.lua"
ADDON_TARGET="${COPPELIA_ROOT}/addOns/zz_ur5_lua_direct_torque_probe_addon.lua"
BASE_OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/control_runs/lua_direct_torque_probe}"
LUA_TORQUE_MODE="${LUA_TORQUE_MODE:-single_joint_probe}"
TORQUE_NM="${TORQUE_NM:-0.05}"
ACTIVE_TORQUE_DURATION_S="${ACTIVE_TORQUE_DURATION_S:-1.0}"
TOTAL_DURATION_S="${TOTAL_DURATION_S:-4.0}"
SETTLE_DURATION_S="${SETTLE_DURATION_S:-0.5}"
MIN_ABS_DISPLACEMENT_RAD="${MIN_ABS_DISPLACEMENT_RAD:-1e-5}"
HANDLE_RESOLUTION_TIMEOUT_S="${HANDLE_RESOLUTION_TIMEOUT_S:-10.0}"
ACCEL_MAGNITUDE_MPS2="${ACCEL_MAGNITUDE_MPS2:-0.25}"
TRAVEL_DISTANCE_M="${TRAVEL_DISTANCE_M:-0.35}"
Y_ACCEL_PRESET="${Y_ACCEL_PRESET:-}"
FORCE_XVFB="${FORCE_XVFB:-1}"
FPS="${FPS:-20}"
SIM_TIMEOUT="${SIM_TIMEOUT:-120}"
COPPELIASIM_EXTRA_ARGS="${COPPELIASIM_EXTRA_ARGS:-}"

normalize_accel_direction() {
  local raw="${1:-}"
  local s="${raw,,}"
  s="${s//[[:space:]]/}"
  case "${s}" in
    1|+1|y+|+y|positive|pos)
      printf '%s\n' "1"
      ;;
    -1|y-|-y|negative|neg)
      printf '%s\n' "-1"
      ;;
    *)
      if [[ -n "${s}" ]] && [[ "${s}" =~ ^-?[0-9]+([.][0-9]+)?$ ]]; then
        awk -v v="${s}" 'BEGIN { if (v >= 0) print 1; else print -1; }'
      else
        printf '%s\n' "1"
      fi
      ;;
  esac
}

MODE_OUTPUT_SUBDIR="single_joint_probe"
VIDEO_BASENAME="coppeliasim_ur5_lua_direct_torque_single_joint.mp4"
case "${LUA_TORQUE_MODE}" in
  single_joint_probe)
    LUA_DIRECT_TORQUE_MAX_NM="${LUA_DIRECT_TORQUE_MAX_NM:-0.05}"
    CARTESIAN_FORCE_SCALE_N_PER_MPS2="${CARTESIAN_FORCE_SCALE_N_PER_MPS2:-4.0}"
    MODEL_HEIGHT_SCALE="${MODEL_HEIGHT_SCALE:-1.0}"
    ;;
  all_joint_micro_torque_probe)
    if [[ "${ACTIVE_TORQUE_DURATION_S}" == "1.0" ]]; then
      ACTIVE_TORQUE_DURATION_S="2.0"
    fi
    LUA_DIRECT_TORQUE_MAX_NM="${LUA_DIRECT_TORQUE_MAX_NM:-0.05}"
    CARTESIAN_FORCE_SCALE_N_PER_MPS2="${CARTESIAN_FORCE_SCALE_N_PER_MPS2:-4.0}"
    MODEL_HEIGHT_SCALE="${MODEL_HEIGHT_SCALE:-1.0}"
    MODE_OUTPUT_SUBDIR="all_joint_micro_torque_probe"
    VIDEO_BASENAME="coppeliasim_ur5_lua_all_joint_micro_torque_probe.mp4"
    ;;
  y_axis_constant_wrench_probe)
    LUA_DIRECT_TORQUE_MAX_NM="${LUA_DIRECT_TORQUE_MAX_NM:-0.50}"
    CARTESIAN_FORCE_SCALE_N_PER_MPS2="${CARTESIAN_FORCE_SCALE_N_PER_MPS2:-4.0}"
    MODEL_HEIGHT_SCALE="${MODEL_HEIGHT_SCALE:-0.8}"
    MODEL_BASE_Z_OFFSET_M="${MODEL_BASE_Z_OFFSET_M:--0.15}"
    if [[ -z "${ACCEL_DIRECTION}" ]]; then
      case "${Y_ACCEL_PRESET}" in
        y_neg) ACCEL_DIRECTION="-1" ;;
        *) ACCEL_DIRECTION="1" ;;
      esac
    fi
    ACCEL_DIRECTION="$(normalize_accel_direction "${ACCEL_DIRECTION}")"
    export ACCEL_DIRECTION
    if [[ "${ACCEL_DIRECTION}" == "-1" ]]; then
      MODE_OUTPUT_SUBDIR="y_axis_constant_wrench_neg"
      VIDEO_BASENAME="coppeliasim_ur5_lua_y_constant_wrench_neg.mp4"
    else
      MODE_OUTPUT_SUBDIR="y_axis_constant_wrench_pos"
      VIDEO_BASENAME="coppeliasim_ur5_lua_y_constant_wrench_pos.mp4"
    fi
    ;;
  y_axis_accel_direction)
    LUA_DIRECT_TORQUE_MAX_NM="${LUA_DIRECT_TORQUE_MAX_NM:-0.50}"
    CARTESIAN_FORCE_SCALE_N_PER_MPS2="${CARTESIAN_FORCE_SCALE_N_PER_MPS2:-20.0}"
    MODEL_HEIGHT_SCALE="${MODEL_HEIGHT_SCALE:-0.8}"
    MODEL_BASE_Z_OFFSET_M="${MODEL_BASE_Z_OFFSET_M:--0.15}"
    if [[ -z "${ACCEL_DIRECTION}" ]]; then
      case "${Y_ACCEL_PRESET}" in
        y_neg) ACCEL_DIRECTION="-1" ;;
        *) ACCEL_DIRECTION="1" ;;
      esac
    fi
    ACCEL_DIRECTION="$(normalize_accel_direction "${ACCEL_DIRECTION}")"
    export ACCEL_DIRECTION
    if [[ "${ACCEL_DIRECTION}" == "-1" ]]; then
      MODE_OUTPUT_SUBDIR="y_axis_accel_direction_neg"
      VIDEO_BASENAME="coppeliasim_ur5_lua_y_accel_direct_torque_neg.mp4"
    else
      MODE_OUTPUT_SUBDIR="y_axis_accel_direction_pos"
      VIDEO_BASENAME="coppeliasim_ur5_lua_y_accel_direct_torque_pos.mp4"
    fi
    ;;
  *)
    LUA_DIRECT_TORQUE_MAX_NM="${LUA_DIRECT_TORQUE_MAX_NM:-0.05}"
    CARTESIAN_FORCE_SCALE_N_PER_MPS2="${CARTESIAN_FORCE_SCALE_N_PER_MPS2:-4.0}"
    MODEL_HEIGHT_SCALE="${MODEL_HEIGHT_SCALE:-1.0}"
    MODEL_BASE_Z_OFFSET_M="${MODEL_BASE_Z_OFFSET_M:-0.0}"
    ;;
esac

OUTPUT_DIR="${BASE_OUTPUT_DIR}/${MODE_OUTPUT_SUBDIR}"
FRAME_DIR="${FRAME_DIR:-${OUTPUT_DIR}/frames}"
VIDEO_PATH="${VIDEO_PATH:-${OUTPUT_DIR}/${VIDEO_BASENAME}}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_DIR}/lua_direct_torque_probe_summary.json}"
SIM_LOG="${SIM_LOG:-${OUTPUT_DIR}/coppeliasim.log}"
BOOT_LOG="${BOOT_LOG:-${OUTPUT_DIR}/bootstrap.log}"
DONE_MARKER="${DONE_MARKER:-${OUTPUT_DIR}/lua_direct_torque_probe_done.txt}"
LOAD_MARKER="${LOAD_MARKER:-${OUTPUT_DIR}/lua_direct_torque_probe_loaded.txt}"
START_MARKER="${START_MARKER:-${OUTPUT_DIR}/lua_direct_torque_probe_started.txt}"
SENSING_MARKER="${SENSING_MARKER:-${OUTPUT_DIR}/lua_direct_torque_probe_sensing.txt}"

export OUTPUT_DIR FRAME_DIR VIDEO_PATH SUMMARY_PATH SIM_LOG BOOT_LOG DONE_MARKER LOAD_MARKER START_MARKER SENSING_MARKER
export LUA_TORQUE_MODE TORQUE_NM ACTIVE_TORQUE_DURATION_S TOTAL_DURATION_S MIN_ABS_DISPLACEMENT_RAD
export LUA_DIRECT_TORQUE_MAX_NM HANDLE_RESOLUTION_TIMEOUT_S ACCEL_MAGNITUDE_MPS2 TRAVEL_DISTANCE_M
export CARTESIAN_FORCE_SCALE_N_PER_MPS2 FORCE_XVFB FPS SETTLE_DURATION_S MODEL_HEIGHT_SCALE MODEL_BASE_Z_OFFSET_M

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

if [[ ! -e "${COPPELIASIM_EXE}" ]]; then
  echo "Missing CoppeliaSim executable: ${COPPELIASIM_EXE}" >&2
  exit 1
fi
if [[ ! -x "${COPPELIASIM_EXE}" ]]; then
  echo "CoppeliaSim executable is not executable: ${COPPELIASIM_EXE}" >&2
  exit 1
fi
if [[ ! -f "${ADDON_SOURCE}" ]]; then
  echo "Missing Lua direct torque probe add-on: ${ADDON_SOURCE}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}" "${FRAME_DIR}" "$(dirname "${VIDEO_PATH}")" "${COPPELIA_ROOT}/addOns"
rm -f "${FRAME_DIR}"/frame_*.png "${VIDEO_PATH}" "${SIM_LOG}" "${BOOT_LOG}" "${SUMMARY_PATH}" "${DONE_MARKER}" "${LOAD_MARKER}" "${START_MARKER}" "${SENSING_MARKER}"
rm -f "${COPPELIA_ROOT}/addOns/ur5_video_smoke_addon.lua"
rm -f "${COPPELIA_ROOT}/addOns/ur5_controller_video_addon.lua"
rm -f "${COPPELIA_ROOT}/addOns/ur5_mujoco_like_y_torque_addon.lua"
rm -f "${COPPELIA_ROOT}/addOns/ur5_acceleration_transport_video_addon.lua"
rm -f "${COPPELIA_ROOT}/addOns/ur5_fixed_z_acceleration_transport_addon.lua"
rm -f "${COPPELIA_ROOT}/addOns/ur5_origin_acquisition_video_addon.lua"
rm -f "${COPPELIA_ROOT}/addOns/zz_real_cartpole_controller_keepalive.lua"
rm -f "${COPPELIA_ROOT}/addOns/zz_zmq_runtime_no_ur5_keepalive.lua"
rm -f "${COPPELIA_ROOT}/addOns/Simulation stepper.lua"
cp -f "${ADDON_SOURCE}" "${ADDON_TARGET}"

log_bootstrap() {
  printf '%s\n' "$*" >>"${BOOT_LOG}"
}

log_bootstrap "COPPELIA_ROOT=${COPPELIA_ROOT}"
log_bootstrap "COPPELIASIM_EXE=${COPPELIASIM_EXE}"
log_bootstrap "COPPELIASIM_SCENE=${COPPELIASIM_SCENE}"
log_bootstrap "OUTPUT_DIR=${OUTPUT_DIR}"
log_bootstrap "FRAME_DIR=${FRAME_DIR}"
log_bootstrap "VIDEO_PATH=${VIDEO_PATH}"
log_bootstrap "SUMMARY_PATH=${SUMMARY_PATH}"
log_bootstrap "TORQUE_NM=${TORQUE_NM}"
log_bootstrap "ACTIVE_TORQUE_DURATION_S=${ACTIVE_TORQUE_DURATION_S}"
log_bootstrap "TOTAL_DURATION_S=${TOTAL_DURATION_S}"
log_bootstrap "SETTLE_DURATION_S=${SETTLE_DURATION_S}"
log_bootstrap "MIN_ABS_DISPLACEMENT_RAD=${MIN_ABS_DISPLACEMENT_RAD}"
log_bootstrap "LUA_TORQUE_MODE=${LUA_TORQUE_MODE}"
log_bootstrap "HANDLE_RESOLUTION_TIMEOUT_S=${HANDLE_RESOLUTION_TIMEOUT_S}"
log_bootstrap "LUA_DIRECT_TORQUE_MAX_NM=${LUA_DIRECT_TORQUE_MAX_NM}"
log_bootstrap "ACCEL_DIRECTION=${ACCEL_DIRECTION:-}"
log_bootstrap "ACCEL_MAGNITUDE_MPS2=${ACCEL_MAGNITUDE_MPS2}"
log_bootstrap "TRAVEL_DISTANCE_M=${TRAVEL_DISTANCE_M}"
log_bootstrap "CARTESIAN_FORCE_SCALE_N_PER_MPS2=${CARTESIAN_FORCE_SCALE_N_PER_MPS2}"
log_bootstrap "MODEL_HEIGHT_SCALE=${MODEL_HEIGHT_SCALE}"
log_bootstrap "MODEL_BASE_Z_OFFSET_M=${MODEL_BASE_Z_OFFSET_M}"
log_bootstrap "FORCE_XVFB=${FORCE_XVFB}"

launch_prefix=()
if [[ "${FORCE_XVFB}" == "1" || -z "${DISPLAY:-}" ]]; then
  launch_prefix=(xvfb-run -a)
fi

launch_args=(
  "${launch_prefix[@]}"
  "${COPPELIASIM_EXE}"
)
if [[ -n "${COPPELIASIM_SCENE}" ]]; then
  launch_args+=("${COPPELIASIM_SCENE}")
fi
if [[ -n "${COPPELIASIM_EXTRA_ARGS}" ]]; then
  read -r -a extra_args <<< "${COPPELIASIM_EXTRA_ARGS}"
  launch_args+=("${extra_args[@]}")
fi

echo "[lua-direct-torque] exact launch command: "
printf '%q ' "${launch_args[@]}"
echo

SIM_PID=""
LOCAL_XVFB_PID=""

if [[ "${FORCE_XVFB}" == "1" || -z "${DISPLAY:-}" ]]; then
  log_bootstrap "launch_mode=xvfb_run_resident_plain"
else
  log_bootstrap "launch_mode=display_resident_plain"
fi
log_bootstrap "direct_torque_probe_addon=${ADDON_TARGET}"

"${launch_args[@]}" >"${SIM_LOG}" 2>&1 &
SIM_PID=$!
echo "[lua-direct-torque] launched PID: ${SIM_PID}"

deadline=$((SECONDS + SIM_TIMEOUT))
while kill -0 "${SIM_PID}" 2>/dev/null; do
  if [[ -f "${SUMMARY_PATH}" ]] && [[ -f "${DONE_MARKER}" ]]; then
    break
  fi
  if [[ "${SECONDS}" -ge "${deadline}" ]]; then
    break
  fi
  sleep 1
done

if kill -0 "${SIM_PID}" 2>/dev/null; then
  wait "${SIM_PID}" 2>/dev/null || true
else
  wait "${SIM_PID}" 2>/dev/null || true
fi
SIM_PID=""

if ! compgen -G "${FRAME_DIR}/frame_*.png" > /dev/null; then
  echo "No frames were captured by the Lua direct torque probe." >&2
  sed -n '1,240p' "${SIM_LOG}" >&2 || true
fi

video_found=""
if [[ -f "${VIDEO_PATH}" ]]; then
  video_found="${VIDEO_PATH}"
fi

if compgen -G "${FRAME_DIR}/frame_*.png" > /dev/null; then
  ffmpeg -y -framerate "${FPS}" -i "${FRAME_DIR}/frame_%08d.png" -c:v libx264 -pix_fmt yuv420p -movflags +faststart "${VIDEO_PATH}" >/dev/null 2>&1 || true
  if [[ -f "${VIDEO_PATH}" ]]; then
    video_found="${VIDEO_PATH}"
  fi
fi

if [[ -f "${SUMMARY_PATH}" ]]; then
  python - "${SUMMARY_PATH}" "${video_found}" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
video_path = sys.argv[2] or None
data = json.loads(summary_path.read_text(encoding='utf-8'))
motion_ok = bool(data.get('motion_ok'))
if video_path:
    data['video_path'] = video_path
    data['video_produced'] = True
    data['video_note'] = None
    if motion_ok:
        data['error'] = None
else:
    data['video_path'] = None
    data['video_produced'] = False
    if motion_ok:
        data['video_note'] = 'Lua direct torque motion succeeded, but no video artifact was produced.'
summary_path.write_text(json.dumps(data, indent=2), encoding='utf-8')
PY
fi

echo "[lua-direct-torque] summary: ${SUMMARY_PATH}"
echo "[lua-direct-torque] log: ${SIM_LOG}"
if [[ -n "${video_found}" ]]; then
  echo "[lua-direct-torque] video: ${video_found}"
else
  echo "[lua-direct-torque] video: <none>"
fi

if [[ -f "${SUMMARY_PATH}" ]]; then
  if python - "${SUMMARY_PATH}" <<'PY'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
raise SystemExit(0 if data.get('success') else 1)
PY
  then
    exit 0
  fi
fi

exit 1
