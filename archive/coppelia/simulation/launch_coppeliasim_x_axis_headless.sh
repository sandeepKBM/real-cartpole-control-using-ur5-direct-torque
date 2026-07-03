#!/usr/bin/env bash
set -euo pipefail

ROOT="/common/users/ss5772/real_Cartpole"
COPPELIA_ROOT="${COPPELIA_ROOT:-${ROOT}/third_party/coppelia_runtime/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu24_04}"
COPPELIA_PYDEPS="${COPPELIA_PYDEPS:-${ROOT}/third_party/coppelia_pydeps}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PORT="${PORT:-23000}"
ZMQ_CNT_PORT="${ZMQ_CNT_PORT:-$((PORT + 1))}"
RUN_SUFFIX="${RUN_SUFFIX:-}"
if [[ -n "${RUN_SUFFIX}" ]]; then
  SIM_LOG="${ROOT}/outputs/control_runs/${RUN_SUFFIX}_coppeliasim_x_axis_headless.log"
  RUNNER_LOG="${ROOT}/outputs/control_runs/${RUN_SUFFIX}_coppeliasim_x_axis_headless_runner.log"
  STATE_DIR="${ROOT}/outputs/control_runs/${RUN_SUFFIX}_coppeliasim_x_axis_headless_state"
else
  SIM_LOG="${ROOT}/outputs/control_runs/coppeliasim_x_axis_headless.log"
  RUNNER_LOG="${ROOT}/outputs/control_runs/coppeliasim_x_axis_headless_runner.log"
  STATE_DIR="${ROOT}/outputs/control_runs/coppeliasim_x_axis_headless_state"
fi
RUNNER_SCRIPT="${ROOT}/simulation/run_coppeliasim_x_axis_headless.py"
MAX_SIM_SECONDS="${MAX_SIM_SECONDS:-600}"
SIM_TIMEOUT="${SIM_TIMEOUT:-75}"
COPPELIASIM_LAUNCH_MODE="${COPPELIASIM_LAUNCH_MODE:-xvfb_resident_plain}"
FORCE_XVFB="${FORCE_XVFB:-${COPPELIA_USE_RAW_XVFB:-1}}"
COPPELIASIM_EXTRA_ARGS="${COPPELIASIM_EXTRA_ARGS:-}"
KEEPALIVE_SOURCE="${ROOT}/simulation/real_cartpole_controller_keepalive.lua"
KEEPALIVE_TARGET="${COPPELIA_ROOT}/addOns/zz_real_cartpole_controller_keepalive.lua"
PY_EXIT_FILE="${STATE_DIR}/python_exit_code.txt"
RPC_RELEASE_FILE="${STATE_DIR}/rpc_connect_release.txt"
RPC_READY_FILE="${STATE_DIR}/rpc_connect_ready.txt"
COPPELIA_PID=""
LOCAL_XVFB_PID=""
RUNNER_ARGS=()
LEGACY_MARKER_HANDOFF=0
ALLOW_EXISTING_SIM=0
for arg in "$@"; do
  case "${arg}" in
    --legacy-marker-handoff)
      LEGACY_MARKER_HANDOFF=1
      ;;
    --allow-existing-sim)
      ALLOW_EXISTING_SIM=1
      ;;
    *)
      RUNNER_ARGS+=("${arg}")
      ;;
  esac
done

if [[ ! -x "${COPPELIA_ROOT}/coppeliaSim.sh" ]]; then
  echo "Missing CoppeliaSim at ${COPPELIA_ROOT}" >&2; exit 1
fi
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Missing Python binary: ${PYTHON_BIN}" >&2; exit 1
fi
if [[ ! -f "${RUNNER_SCRIPT}" ]]; then
  echo "Missing controller runner at ${RUNNER_SCRIPT}" >&2; exit 1
fi
if [[ "${LEGACY_MARKER_HANDOFF}" -eq 1 && ! -f "${KEEPALIVE_SOURCE}" ]]; then
  echo "Missing keepalive add-on at ${KEEPALIVE_SOURCE}" >&2; exit 1
fi
mkdir -p "${STATE_DIR}"
rm -f "${RUNNER_LOG}" "${PY_EXIT_FILE}" "${RPC_RELEASE_FILE}" "${RPC_READY_FILE}"

rm -f "${COPPELIA_ROOT}/addOns/ur5_video_smoke_addon.lua"
rm -f "${COPPELIA_ROOT}/addOns/ur5_controller_video_addon.lua"
rm -f "${COPPELIA_ROOT}/addOns/ur5_acceleration_transport_video_addon.lua"
rm -f "${COPPELIA_ROOT}/addOns/ur5_fixed_z_acceleration_transport_addon.lua"
rm -f "${COPPELIA_ROOT}/addOns/coppeliasim_keepalive.lua"
rm -f "${COPPELIA_ROOT}/addOns/real_cartpole_controller_keepalive.lua"
rm -f "${COPPELIA_ROOT}/addOns/zz_real_cartpole_keepalive.lua"
rm -f "${COPPELIA_ROOT}/addOns/zz_real_cartpole_controller_keepalive.lua"
if [[ "${LEGACY_MARKER_HANDOFF}" -eq 1 ]]; then
  if [[ ! -f "${KEEPALIVE_SOURCE}" ]]; then
    echo "Missing keepalive add-on at ${KEEPALIVE_SOURCE}" >&2
    exit 1
  fi
  rm -f "${KEEPALIVE_TARGET}"
  cp -f "${KEEPALIVE_SOURCE}" "${KEEPALIVE_TARGET}"
else
  rm -f "${KEEPALIVE_TARGET}"
fi

cleanup() {
  if [[ -n "${PYTHON_PID:-}" ]] && kill -0 "${PYTHON_PID}" 2>/dev/null; then
    kill "${PYTHON_PID}" 2>/dev/null || true
    wait "${PYTHON_PID}" 2>/dev/null || true
  fi
  if [[ -n "${COPPELIA_PID:-}" ]] && kill -0 "${COPPELIA_PID}" 2>/dev/null; then
    kill "${COPPELIA_PID}" 2>/dev/null || true
    wait "${COPPELIA_PID}" 2>/dev/null || true
  fi
  if [[ -n "${LOCAL_XVFB_PID:-}" ]] && kill -0 "${LOCAL_XVFB_PID}" 2>/dev/null; then
    kill "${LOCAL_XVFB_PID}" 2>/dev/null || true
    wait "${LOCAL_XVFB_PID}" 2>/dev/null || true
  fi
  rm -f "${KEEPALIVE_TARGET}"
}
trap cleanup EXIT INT TERM

PORT_ALREADY_LISTENING=0
if command -v ss >/dev/null 2>&1 && ss -ltn "sport = :${PORT}" | grep -q LISTEN; then
  PORT_ALREADY_LISTENING=1
  if [[ "${ALLOW_EXISTING_SIM}" != "1" ]]; then
    echo "The ZMQ port is already in use. A stale CoppeliaSim instance may still be running. Stop it or pass --allow-existing-sim." >&2
    echo "Likely stale processes:" >&2
    ps -ef | grep -E "coppeliaSim|zmqRemoteApi.rpcPort=${PORT}" | grep -v grep >&2 || true
    exit 1
  fi
fi

export COPPELIA_ROOT
export COPPELIA_PYDEPS
export REAL_CARTPOLE_ROOT="${ROOT}"
export REAL_CARTPOLE_ENABLE_VIDEO_SMOKE=0
export REAL_CARTPOLE_PY_EXIT_FILE="${PY_EXIT_FILE}"
export REAL_CARTPOLE_RPC_HOST="${REAL_CARTPOLE_RPC_HOST:-127.0.0.1}"
export REAL_CARTPOLE_RPC_PORT="${PORT}"
export REAL_CARTPOLE_RPC_CONNECT_DELAY_S="${REAL_CARTPOLE_RPC_CONNECT_DELAY_S:-0.0}"
export REAL_CARTPOLE_RPC_CONNECT_PORT_WAIT_S="${REAL_CARTPOLE_RPC_CONNECT_PORT_WAIT_S:-15.0}"
if [[ "${LEGACY_MARKER_HANDOFF}" -eq 1 ]]; then
  export REAL_CARTPOLE_RPC_CONNECT_RELEASE_FILE="${RPC_RELEASE_FILE}"
  export REAL_CARTPOLE_RPC_CONNECT_READY_FILE="${RPC_READY_FILE}"
  export REAL_CARTPOLE_RPC_CONNECT_READY_WAIT_S="${REAL_CARTPOLE_RPC_CONNECT_READY_WAIT_S:-5.0}"
  export REAL_CARTPOLE_RPC_CONNECT_GRACE_S="${REAL_CARTPOLE_RPC_CONNECT_GRACE_S:-5.0}"
fi
# Verified fixed-Z transport seed from
# `outputs/control_runs/fixed_z_x_transport_firstpass_z0.540_seed1.json`.
export Q_START_RAD="${Q_START_RAD:-0.0,-0.1133064268431449,-0.664621645801302,4.921777393344012,-6.283185307179586,5.280928640069786}"
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

mkdir -p "$(dirname "${SIM_LOG}")"
cd "${COPPELIA_ROOT}"

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

launch_coppelia_bg() {
  local -a launch_prefix=()
  local -a launch_args=()
  local -a extra_args=()
  if [[ -n "${COPPELIASIM_EXTRA_ARGS}" ]]; then
    read -r -a extra_args <<< "${COPPELIASIM_EXTRA_ARGS}"
  fi

  case "${COPPELIASIM_LAUNCH_MODE}" in
    xvfb_resident_plain)
      if [[ -n "${XVFB_RUN_BIN:-}" ]]; then
        launch_prefix=("${XVFB_RUN_BIN}" -a)
      else
        launch_prefix=(xvfb-run -a)
      fi
      ;;
    resident_plain)
      if [[ "${FORCE_XVFB}" == "1" || -z "${DISPLAY:-}" ]]; then
        if [[ -n "${XVFB_RUN_BIN:-}" ]]; then
          launch_prefix=("${XVFB_RUN_BIN}" -a)
        else
          launch_prefix=(xvfb-run -a)
        fi
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
  if [[ -n "${COPPELIASIM_SCENE:-}" ]]; then
    launch_args+=("${COPPELIASIM_SCENE}")
  fi
  if [[ ${#extra_args[@]} -gt 0 ]]; then
    launch_args+=("${extra_args[@]}")
  fi

  if [[ "${COPPELIASIM_LAUNCH_MODE}" != "legacy_headless" ]]; then
    for forbidden in -h -vscriptinfos; do
      if printf '%s\n' "${launch_args[@]}" | grep -qx -- "${forbidden}"; then
        echo "Refusing to use -h/-vscriptinfos for external ZMQ validation because this launch mode binds the RPC port but does not service require('sim') in this environment." >&2
        return 1
      fi
    done
  fi

  echo -n "[launcher] exact launch command: "
  printf '%q ' "${launch_args[@]}"
  echo
  "${launch_args[@]}" >"${SIM_LOG}" 2>&1 &
  COPPELIA_PID=$!
}

launch_python_fg() {
  (
    cd "${ROOT}"
    timeout "${MAX_SIM_SECONDS}" "${RUNNER_CMD[@]}"
  ) >"${RUNNER_LOG}" 2>&1
}

wait_for_rpc_port() {
  local deadline=$((SECONDS + 60))
  while [[ "${SECONDS}" -lt "${deadline}" ]]; do
    if { : > "/dev/tcp/127.0.0.1/${PORT}"; } 2>/dev/null; then
      return 0
    fi
    if command -v ss >/dev/null 2>&1 && ss -ltn "sport = :${PORT}" | grep -q LISTEN; then
      return 0
    fi
    sleep 0.2
  done
  return 1
}

RUNNER_CMD=(
  "${PYTHON_BIN}"
  "${RUNNER_SCRIPT}"
  --port "${PORT}"
)
if [[ "${LEGACY_MARKER_HANDOFF}" -eq 1 ]]; then
  RUNNER_CMD+=(--legacy-marker-handoff --preloaded-scene)
fi
AUTO_VIDEO_CAMERA=()
has_accel_transport=0
has_video_camera=0
for arg in "${RUNNER_ARGS[@]}"; do
  case "${arg}" in
    --accel-x-transport)
      has_accel_transport=1
      ;;
    --video-camera|--video-camera=*)
      has_video_camera=1
      ;;
  esac
done
if [[ "${has_accel_transport}" -eq 1 && "${has_video_camera}" -eq 0 ]]; then
  # The smoke camera is only known-good for the dedicated smoke test, where the
  # arm is intentionally posed for framing.  Torque transport runs follow the EE
  # so the motion stays visible even while the arm leaves the smoke framing pose.
  AUTO_VIDEO_CAMERA=(--video-camera ee)
fi
if [[ ${#AUTO_VIDEO_CAMERA[@]} -gt 0 ]]; then
  RUNNER_CMD+=("${AUTO_VIDEO_CAMERA[@]}")
fi
if [[ ${#RUNNER_ARGS[@]} -gt 0 ]]; then
  RUNNER_CMD+=("${RUNNER_ARGS[@]}")
fi

if [[ "${PORT_ALREADY_LISTENING}" -eq 0 ]]; then
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
  sleep 2
else
  echo "[launcher] Reusing existing CoppeliaSim on port ${PORT} (allow-existing-sim)"
fi

echo "[launcher] Starting Python runner in foreground"
set +e
launch_python_fg
PY_EXIT=$?
set -e

COPPELIA_EXIT=0
if [[ -n "${COPPELIA_PID:-}" ]]; then
  if kill -0 "${COPPELIA_PID}" 2>/dev/null; then
    kill "${COPPELIA_PID}" 2>/dev/null || true
    wait "${COPPELIA_PID}" 2>/dev/null || true
  else
    set +e
    wait "${COPPELIA_PID}" 2>/dev/null
    COPPELIA_EXIT=$?
    set -e
  fi
fi

if [[ -f "${PY_EXIT_FILE}" ]]; then
  PY_EXIT="$(cat "${PY_EXIT_FILE}" 2>/dev/null || echo 0)"
fi
if [[ "${COPPELIA_EXIT}" -ne 0 ]]; then
  echo "[launcher] CoppeliaSim exited with code ${COPPELIA_EXIT}" >&2
fi
if [[ "${PY_EXIT}" -ne 0 ]]; then
  echo "[launcher] Python runner exited with code ${PY_EXIT}" >&2
fi
RUNNER_EXIT="0"
if [[ "${PY_EXIT}" -ne 0 ]]; then
  RUNNER_EXIT="${PY_EXIT}"
elif [[ "${COPPELIA_EXIT}" -ne 0 ]]; then
  RUNNER_EXIT="${COPPELIA_EXIT}"
fi

echo "[launcher] Final exit code: ${RUNNER_EXIT}"
exit "${RUNNER_EXIT}"
