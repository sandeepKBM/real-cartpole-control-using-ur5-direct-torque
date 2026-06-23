#!/usr/bin/env bash
set -euo pipefail

ROOT="/common/users/ss5772/real_Cartpole"
COPPELIA_ROOT="${COPPELIA_ROOT:-${ROOT}/third_party/coppelia_runtime/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu24_04}"
FRAME_DIR="${ROOT}/outputs/control_runs/coppelia_controller_video_frames"
STATE_DIR="${ROOT}/outputs/control_runs/coppelia_controller_video_state"
SIM_LOG="${STATE_DIR}/coppelia.log"
BOOT_LOG="${STATE_DIR}/bootstrap.log"
VIDEO_PATH="${ROOT}/demonstration_videos/ur5e_coppeliasim/coppeliasim_ur5_controller_video.mp4"
ADDON_SOURCE="${ROOT}/simulation/ur5_controller_video_addon.lua"
ADDON_TARGET="${COPPELIA_ROOT}/addOns/ur5_controller_video_addon.lua"
SMOKE_ADDON_TARGET="${COPPELIA_ROOT}/addOns/ur5_video_smoke_addon.lua"
START_MARKER="${STATE_DIR}/ur5_controller_video_addon_started.txt"
LOAD_MARKER="${STATE_DIR}/ur5_controller_video_addon_loaded.txt"
SENSING_MARKER="${STATE_DIR}/ur5_controller_video_addon_sensing.txt"
DONE_MARKER="${STATE_DIR}/ur5_controller_video_done.txt"
FPS="${FPS:-20}"
SIM_TIMEOUT="${SIM_TIMEOUT:-60}"
FRAME_COUNT="${FRAME_COUNT:-80}"

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

rm -f "${FRAME_DIR}"/frame_*.png "${VIDEO_PATH}" "${SIM_LOG}" "${BOOT_LOG}"
rm -f "${LOAD_MARKER}" "${START_MARKER}" "${SENSING_MARKER}" "${DONE_MARKER}"
rm -f "${SMOKE_ADDON_TARGET}"
cp -f "${ADDON_SOURCE}" "${ADDON_TARGET}"

log_bootstrap() {
  printf '%s\n' "$*" >>"${BOOT_LOG}"
}

log_bootstrap "COPPELIA_ROOT=${COPPELIA_ROOT}"
log_bootstrap "ADDON_SOURCE=${ADDON_SOURCE}"
log_bootstrap "FRAME_COUNT=${FRAME_COUNT}"

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
  DISPLAY="${DISPLAY_VALUE}" XAUTHORITY="${XAUTH_VALUE}" COPPELIA_ROOT="${COPPELIA_ROOT}" REAL_CARTPOLE_ROOT="${ROOT}" FRAME_COUNT="${FRAME_COUNT}" \
    "${COPPELIA_ROOT}/coppeliaSim.sh" -h -vscriptinfos \
    >"${SIM_LOG}" 2>&1 &
  SIM_PID=$!
else
  if [[ "${USE_RAW_XVFB}" == "1" ]] && command -v Xvfb >/dev/null 2>&1; then
    log_bootstrap "launch_branch=raw_xvfb"
    COPPELIA_ROOT="${COPPELIA_ROOT}" REAL_CARTPOLE_ROOT="${ROOT}" FRAME_COUNT="${FRAME_COUNT}" \
      "${COPPELIA_ROOT}/coppeliaSim.sh" -h -vscriptinfos \
      >"${SIM_LOG}" 2>&1 &
    SIM_PID=$!
  else
    log_bootstrap "launch_branch=xvfb-run"
    xvfb-run -a /usr/bin/env COPPELIA_ROOT="${COPPELIA_ROOT}" REAL_CARTPOLE_ROOT="${ROOT}" FRAME_COUNT="${FRAME_COUNT}" \
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
  echo "No controller video frames were captured." >&2
  [[ -f "${LOAD_MARKER}" ]] && echo "Load marker present." >&2 || echo "Load marker missing." >&2
  [[ -f "${START_MARKER}" ]] && echo "Start marker present." >&2 || echo "Start marker missing." >&2
  [[ -f "${SENSING_MARKER}" ]] && echo "Sensing marker present." >&2 || echo "Sensing marker missing." >&2
  [[ -f "${DONE_MARKER}" ]] && echo "Done marker present." >&2 || echo "Done marker missing." >&2
  sed -n '1,260p' "${SIM_LOG}" >&2 || true
  exit 1
fi

frame_count=$(find "${FRAME_DIR}" -maxdepth 1 -type f -name 'frame_*.png' | wc -l | tr -d '[:space:]')
if [[ "${frame_count}" -lt "${FRAME_COUNT}" ]]; then
  echo "Only captured ${frame_count}/${FRAME_COUNT} controller video frames." >&2
  sed -n '1,260p' "${SIM_LOG}" >&2 || true
  exit 1
fi

ffmpeg -y -framerate "${FPS}" -i "${FRAME_DIR}/frame_%08d.png" -c:v libx264 -pix_fmt yuv420p "${VIDEO_PATH}"
echo "Saved controller video: ${VIDEO_PATH}"
