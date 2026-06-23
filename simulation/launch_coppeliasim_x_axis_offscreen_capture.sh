#!/usr/bin/env bash
set -euo pipefail

ROOT="/common/users/ss5772/real_Cartpole"
COPPELIA_ROOT="${COPPELIA_ROOT:-${ROOT}/third_party/coppelia_runtime/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu24_04}"
COPPELIA_PYDEPS="${COPPELIA_PYDEPS:-${ROOT}/third_party/coppelia_pydeps}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PORT="${PORT:-23000}"
ZMQ_CNT_PORT="${ZMQ_CNT_PORT:-$((PORT + 1))}"
RUN_SUFFIX="${RUN_SUFFIX:-}"
FRAME_COUNT="${FRAME_COUNT:-120}"
CAPTURE_SKIP_FRAMES="${CAPTURE_SKIP_FRAMES:-12}"
FPS="${FPS:-25}"
FFMPEG_BIN="${FFMPEG_BIN:-}"
if [[ -n "${RUN_SUFFIX}" ]]; then
  SIM_LOG="${ROOT}/outputs/control_runs/${RUN_SUFFIX}_coppeliasim_x_axis_offscreen_capture.log"
  RUNNER_LOG="${ROOT}/outputs/control_runs/${RUN_SUFFIX}_coppeliasim_x_axis_offscreen_capture_runner.log"
  STATE_DIR="${ROOT}/outputs/control_runs/${RUN_SUFFIX}_coppeliasim_x_axis_offscreen_capture_state"
  FRAME_DIR="${ROOT}/outputs/control_runs/${RUN_SUFFIX}_coppeliasim_x_axis_offscreen_capture_frames"
  VIDEO_PATH="${ROOT}/demonstration_videos/ur5e_coppeliasim/${RUN_SUFFIX}_coppeliasim_x_axis_offscreen_capture.mp4"
  TRACE_NAME="${RUN_SUFFIX}_coppeliasim_x_axis_offscreen_capture.jsonl"
  SUMMARY_NAME="${RUN_SUFFIX}_coppeliasim_x_axis_offscreen_capture_summary.json"
else
  SIM_LOG="${ROOT}/outputs/control_runs/coppeliasim_x_axis_offscreen_capture.log"
  RUNNER_LOG="${ROOT}/outputs/control_runs/coppeliasim_x_axis_offscreen_capture_runner.log"
  STATE_DIR="${ROOT}/outputs/control_runs/coppeliasim_x_axis_offscreen_capture_state"
  FRAME_DIR="${ROOT}/outputs/control_runs/coppeliasim_x_axis_offscreen_capture_frames"
  VIDEO_PATH="${ROOT}/demonstration_videos/ur5e_coppeliasim/coppeliasim_ur5_x_axis_offscreen_capture.mp4"
  TRACE_NAME="coppeliasim_ur5_x_axis_offscreen_capture.jsonl"
  SUMMARY_NAME="coppeliasim_ur5_x_axis_offscreen_capture_summary.json"
fi
RUNNER_SCRIPT="${ROOT}/simulation/run_coppeliasim_x_axis_headless.py"
RUNNER_EXTRA_ARGS="${RUNNER_EXTRA_ARGS:-}"
MAX_SIM_SECONDS="${MAX_SIM_SECONDS:-600}"
SIM_TIMEOUT="${SIM_TIMEOUT:-120}"
COPPELIASIM_LAUNCH_MODE="${COPPELIASIM_LAUNCH_MODE:-xvfb_resident_plain}"
FORCE_XVFB="${FORCE_XVFB:-1}"
COPPELIA_EXTRA_ARGS="${COPPELIA_EXTRA_ARGS:-}"
XVFB_RUN_BIN="${XVFB_RUN_BIN:-}"
CAPTURE_ADDON_SOURCE="${ROOT}/simulation/ur5_external_controller_capture_addon.lua"
CAPTURE_ADDON_TARGET="${COPPELIA_ROOT}/addOns/zz_real_cartpole_external_controller_capture.lua"
SMOKE_ADDON_TARGET="${COPPELIA_ROOT}/addOns/ur5_video_smoke_addon.lua"
CONTROLLER_ADDON_TARGET="${COPPELIA_ROOT}/addOns/ur5_controller_video_addon.lua"
ACCEL_ADDON_TARGET="${COPPELIA_ROOT}/addOns/ur5_acceleration_transport_video_addon.lua"
FIXED_Z_ACCEL_ADDON_TARGET="${COPPELIA_ROOT}/addOns/ur5_fixed_z_acceleration_transport_addon.lua"
PY_EXIT_FILE="${STATE_DIR}/python_exit_code.txt"
COPPELIA_PID=""

if [[ ! -x "${COPPELIA_ROOT}/coppeliaSim.sh" ]]; then
  echo "Missing CoppeliaSim at ${COPPELIA_ROOT}" >&2
  exit 1
fi
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Missing Python binary: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -f "${CAPTURE_ADDON_SOURCE}" ]]; then
  echo "Missing capture add-on at ${CAPTURE_ADDON_SOURCE}" >&2
  exit 1
fi

mkdir -p "${STATE_DIR}" "${FRAME_DIR}" "$(dirname "${VIDEO_PATH}")"
rm -f "${SIM_LOG}" "${RUNNER_LOG}" "${PY_EXIT_FILE}"
rm -f "${FRAME_DIR}"/frame_*.png "${VIDEO_PATH}"

if [[ -z "${FFMPEG_BIN}" ]]; then
  if command -v ffmpeg >/dev/null 2>&1; then
    FFMPEG_BIN="$(command -v ffmpeg)"
  elif [[ -x /usr/bin/ffmpeg ]]; then
    FFMPEG_BIN="/usr/bin/ffmpeg"
  elif [[ -x /bin/ffmpeg ]]; then
    FFMPEG_BIN="/bin/ffmpeg"
  else
    echo "Missing ffmpeg binary; set FFMPEG_BIN or install ffmpeg." >&2
    exit 1
  fi
fi

resolve_xvfb_run() {
  if command -v xvfb-run >/dev/null 2>&1; then
    printf '%s\n' "$(command -v xvfb-run)"
    return 0
  fi

  local candidate candidate_dir
  while IFS= read -r candidate; do
    candidate_dir="$(dirname "${candidate}")"
    if [[ -x "${candidate_dir}/Xvfb" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done < <(find /common/home/ss5772/.tmp -path '*/xvfb-run' -type f 2>/dev/null | sort)

  return 1
}

if [[ -z "${XVFB_RUN_BIN}" ]]; then
  if XVFB_RUN_BIN="$(resolve_xvfb_run)"; then
    :
  else
    XVFB_RUN_BIN=""
  fi
fi

XVFB_RUN_DIR=""
if [[ -n "${XVFB_RUN_BIN}" ]]; then
  XVFB_RUN_DIR="$(cd "$(dirname "${XVFB_RUN_BIN}")" && pwd)"
fi

rm -f "${SMOKE_ADDON_TARGET}"
rm -f "${CONTROLLER_ADDON_TARGET}"
rm -f "${ACCEL_ADDON_TARGET}"
rm -f "${FIXED_Z_ACCEL_ADDON_TARGET}"
cp -f "${CAPTURE_ADDON_SOURCE}" "${CAPTURE_ADDON_TARGET}"

cleanup() {
  if [[ -n "${PYTHON_PID:-}" ]] && kill -0 "${PYTHON_PID}" 2>/dev/null; then
    kill "${PYTHON_PID}" 2>/dev/null || true
    wait "${PYTHON_PID}" 2>/dev/null || true
  fi
  if [[ -n "${COPPELIA_PID:-}" ]] && kill -0 "${COPPELIA_PID}" 2>/dev/null; then
    kill "${COPPELIA_PID}" 2>/dev/null || true
    wait "${COPPELIA_PID}" 2>/dev/null || true
  fi
  rm -f "${CAPTURE_ADDON_TARGET}"
}
trap cleanup EXIT INT TERM

export COPPELIA_ROOT
export COPPELIA_PYDEPS
export REAL_CARTPOLE_ROOT="${ROOT}"
export REAL_CARTPOLE_ENABLE_VIDEO_SMOKE=0
export OUTPUT_DIR="${FRAME_DIR}"
export STATE_DIR="${STATE_DIR}"
export VIDEO_PATH="${VIDEO_PATH}"
export SUMMARY_PATH="${STATE_DIR}/coppelia_external_controller_capture_summary.txt"
export FRAME_PREFIX="frame"
export FRAME_COUNT="${FRAME_COUNT}"
export CAPTURE_SKIP_FRAMES="${CAPTURE_SKIP_FRAMES}"
export FPS="${FPS}"
export LD_LIBRARY_PATH="${COPPELIA_ROOT}:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${COPPELIA_ROOT}/programming/zmqRemoteApi/clients/python/src:${COPPELIA_PYDEPS}:${PYTHONPATH:-}"
case "${COPPELIASIM_LAUNCH_MODE}" in
  legacy_headless)
    export QT_QPA_PLATFORM="offscreen"
    ;;
  *)
    export QT_QPA_PLATFORM="xcb"
    ;;
esac

RUNTIME_DIR="${STATE_DIR}/xdg-runtime"
mkdir -p "${RUNTIME_DIR}"
chmod 700 "${RUNTIME_DIR}" 2>/dev/null || true
if [[ -n "${XDG_RUNTIME_DIR:-}" && -d "${XDG_RUNTIME_DIR}" && -w "${XDG_RUNTIME_DIR}" ]]; then
  export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR}"
else
  export XDG_RUNTIME_DIR="${RUNTIME_DIR}"
fi
export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=${XDG_RUNTIME_DIR}/bus}"

mkdir -p "$(dirname "${SIM_LOG}")"
cd "${COPPELIA_ROOT}"

launch_coppelia_bg() {
  local -a launch_prefix=()
  local -a launch_args=()
  local -a extra_args=()
  if [[ -n "${COPPELIA_EXTRA_ARGS}" ]]; then
    read -r -a extra_args <<< "${COPPELIA_EXTRA_ARGS}"
  fi

  case "${COPPELIASIM_LAUNCH_MODE}" in
    xvfb_resident_plain)
      if [[ -z "${XVFB_RUN_BIN}" ]]; then
        echo "Missing xvfb-run; set XVFB_RUN_BIN or install xvfb-run." >&2
        return 1
      fi
      launch_prefix=("${XVFB_RUN_BIN}" -a)
      ;;
    resident_plain)
      if [[ "${FORCE_XVFB}" == "1" || -z "${DISPLAY:-}" ]]; then
        if [[ -z "${XVFB_RUN_BIN}" ]]; then
          echo "Missing xvfb-run; set XVFB_RUN_BIN or install xvfb-run." >&2
          return 1
        fi
        launch_prefix=("${XVFB_RUN_BIN}" -a)
      fi
      ;;
    legacy_headless)
      echo "WARNING: legacy_headless is known to fail ZMQ attach in this environment." >&2
      ;;
    *)
      echo "Unsupported COPPELIASIM_LAUNCH_MODE: ${COPPELIASIM_LAUNCH_MODE}" >&2
      return 1
      ;;
  esac

  if [[ "${COPPELIASIM_LAUNCH_MODE}" == "resident_plain" && "${FORCE_XVFB}" != "1" && -z "${DISPLAY:-}" ]]; then
    echo "resident_plain requires an available DISPLAY unless FORCE_XVFB=1." >&2
    return 1
  fi

  launch_args+=("${launch_prefix[@]}")
  launch_args+=("${COPPELIA_ROOT}/coppeliaSim.sh")
  if [[ "${COPPELIASIM_LAUNCH_MODE}" == "legacy_headless" ]]; then
    launch_args+=(-h -vscriptinfos)
  fi
  launch_args+=("-GzmqRemoteApi.rpcPort=${PORT}")
  launch_args+=("-GzmqRemoteApi.cntPort=${ZMQ_CNT_PORT}")
  if [[ -n "${COPPELIA_SCENE:-}" ]]; then
    launch_args+=("${COPPELIA_SCENE}")
  fi
  if [[ ${#extra_args[@]} -gt 0 ]]; then
    launch_args+=("${extra_args[@]}")
  fi
  if [[ "${COPPELIASIM_LAUNCH_MODE}" != "legacy_headless" ]]; then
    for forbidden in -h -vscriptinfos; do
      if printf '%s\n' "${launch_args[@]}" | grep -qx -- "${forbidden}"; then
        echo "Refusing to use -h/-vscriptinfos for the capture lane in this environment." >&2
        return 1
      fi
    done
  fi
  echo -n "[launcher] exact launch command: "
  printf '%q ' env PATH="${XVFB_RUN_DIR:+${XVFB_RUN_DIR}:}${PATH}" QT_QPA_PLATFORM="${QT_QPA_PLATFORM}" COPPELIA_ROOT="${COPPELIA_ROOT}" REAL_CARTPOLE_ROOT="${ROOT}" OUTPUT_DIR="${FRAME_DIR}" STATE_DIR="${STATE_DIR}" VIDEO_PATH="${VIDEO_PATH}" SUMMARY_PATH="${SUMMARY_PATH}" FRAME_PREFIX="${FRAME_PREFIX}" FRAME_COUNT="${FRAME_COUNT}" CAPTURE_SKIP_FRAMES="${CAPTURE_SKIP_FRAMES}" FPS="${FPS}" "${launch_args[@]}"
  echo
  env \
    PATH="${XVFB_RUN_DIR:+${XVFB_RUN_DIR}:}${PATH}" \
    QT_QPA_PLATFORM="${QT_QPA_PLATFORM}" \
    COPPELIA_ROOT="${COPPELIA_ROOT}" \
    REAL_CARTPOLE_ROOT="${ROOT}" \
    OUTPUT_DIR="${FRAME_DIR}" \
    STATE_DIR="${STATE_DIR}" \
    VIDEO_PATH="${VIDEO_PATH}" \
    SUMMARY_PATH="${SUMMARY_PATH}" \
    FRAME_PREFIX="${FRAME_PREFIX}" \
    FRAME_COUNT="${FRAME_COUNT}" \
    CAPTURE_SKIP_FRAMES="${CAPTURE_SKIP_FRAMES}" \
    FPS="${FPS}" \
    "${launch_args[@]}" >"${SIM_LOG}" 2>&1 &
  COPPELIA_PID=$!
}

wait_for_rpc_port() {
  local deadline=$((SECONDS + SIM_TIMEOUT))
  while [[ "${SECONDS}" -lt "${deadline}" ]]; do
    if { : > "/dev/tcp/127.0.0.1/${PORT}"; } 2>/dev/null; then
      return 0
    fi
    sleep 0.1
  done
  return 1
}

RUNNER_CMD=(
  "${PYTHON_BIN}"
  "${RUNNER_SCRIPT}"
  --port "${PORT}"
  --no-video
  --trace-name "${TRACE_NAME}"
  --summary-name "${SUMMARY_NAME}"
)
if [[ -n "${RUNNER_EXTRA_ARGS}" ]]; then
  read -r -a RUNNER_EXTRA_ARGS_ARRAY <<< "${RUNNER_EXTRA_ARGS}"
  RUNNER_CMD+=("${RUNNER_EXTRA_ARGS_ARRAY[@]}")
fi

echo "[launcher] Starting CoppeliaSim in background"
set +e
launch_coppelia_bg
COPPELIA_START_EXIT=$?
set -e
if [[ "${COPPELIA_START_EXIT}" -ne 0 ]]; then
  echo "[launcher] Failed to start CoppeliaSim" >&2
  exit "${COPPELIA_START_EXIT}"
fi
if ! wait_for_rpc_port; then
  echo "[launcher] RPC port ${PORT} did not start listening" >&2
  exit 1
fi

echo "[launcher] Starting Python runner in foreground"
set +e
timeout "${MAX_SIM_SECONDS}" "${RUNNER_CMD[@]}" >"${RUNNER_LOG}" 2>&1
PY_EXIT=$?
set -e
if [[ -f "${PY_EXIT_FILE}" ]]; then
  PY_EXIT="$(cat "${PY_EXIT_FILE}" 2>/dev/null || echo "${PY_EXIT}")"
fi

if [[ -n "${COPPELIA_PID:-}" ]]; then
  if kill -0 "${COPPELIA_PID}" 2>/dev/null; then
    kill "${COPPELIA_PID}" 2>/dev/null || true
    wait "${COPPELIA_PID}" 2>/dev/null || true
  else
    wait "${COPPELIA_PID}" 2>/dev/null || true
  fi
fi

if ! compgen -G "${FRAME_DIR}/frame_*.png" > /dev/null; then
  echo "No external-controller capture frames were captured." >&2
  sed -n '1,260p' "${SIM_LOG}" >&2 || true
  sed -n '1,260p' "${RUNNER_LOG}" >&2 || true
  exit 1
fi

"${FFMPEG_BIN}" -y -framerate "${FPS}" -i "${FRAME_DIR}/frame_%08d.png" -c:v libx264 -pix_fmt yuv420p "${VIDEO_PATH}"

if [[ "${PY_EXIT}" -ne 0 ]]; then
  echo "[launcher] Python runner exited with code ${PY_EXIT}" >&2
fi

echo "Saved external-controller capture video: ${VIDEO_PATH}"
