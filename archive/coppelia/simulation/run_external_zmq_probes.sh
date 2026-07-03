#!/usr/bin/env bash
set -euo pipefail

ROOT="/common/users/ss5772/real_Cartpole"
COPPELIA_ROOT="${COPPELIA_ROOT:-${ROOT}/third_party/coppelia_runtime/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu24_04}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
ZMQ_RPC_PORT="${ZMQ_RPC_PORT:-${PORT:-23000}}"
ZMQ_CNT_PORT="${ZMQ_CNT_PORT:-$((ZMQ_RPC_PORT + 1))}"
COPPELIASIM_LAUNCH_MODE="${COPPELIASIM_LAUNCH_MODE:-xvfb_resident_plain}"
FORCE_XVFB="${FORCE_XVFB:-1}"
COPPELIASIM_SCENE="${COPPELIASIM_SCENE:-}"
COPPELIASIM_EXTRA_ARGS="${COPPELIASIM_EXTRA_ARGS:-}"
ALLOW_EXISTING_SIM="${ALLOW_EXISTING_SIM:-0}"

for arg in "$@"; do
  case "${arg}" in
    --allow-existing-sim)
      ALLOW_EXISTING_SIM=1
      ;;
    *)
      echo "Unknown argument: ${arg}" >&2
      exit 1
      ;;
  esac
done

HANDSHAKE_SUMMARY="${ROOT}/outputs/control_runs/external_zmq_handshake/summary.json"
SINGLE_TORQUE_SUMMARY="${ROOT}/outputs/control_runs/external_zmq_single_joint_torque/summary.json"
SIM_LOG="${ROOT}/outputs/control_runs/external_zmq_probe_coppelia.log"
HANDSHAKE_SCRIPT="${ROOT}/simulation/probe_external_zmq_handshake.py"
SINGLE_TORQUE_SCRIPT="${ROOT}/simulation/probe_external_zmq_single_joint_torque.py"

COPPELIA_PID=""

cleanup() {
  if [[ -n "${COPPELIA_PID:-}" ]] && kill -0 "${COPPELIA_PID}" 2>/dev/null; then
    kill "${COPPELIA_PID}" 2>/dev/null || true
    wait "${COPPELIA_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if [[ ! -x "${COPPELIA_ROOT}/coppeliaSim.sh" ]]; then
  echo "Missing CoppeliaSim at ${COPPELIA_ROOT}" >&2
  exit 1
fi
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Missing Python binary: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -f "${HANDSHAKE_SCRIPT}" ]]; then
  echo "Missing handshake probe at ${HANDSHAKE_SCRIPT}" >&2
  exit 1
fi
if [[ ! -f "${SINGLE_TORQUE_SCRIPT}" ]]; then
  echo "Missing single-joint torque probe at ${SINGLE_TORQUE_SCRIPT}" >&2
  exit 1
fi
mkdir -p "$(dirname "${HANDSHAKE_SUMMARY}")"
mkdir -p "$(dirname "${SINGLE_TORQUE_SUMMARY}")"
rm -f "${HANDSHAKE_SUMMARY}" "${SINGLE_TORQUE_SUMMARY}" "${SIM_LOG}"
rm -f "${COPPELIA_ROOT}/addOns/ur5_video_smoke_addon.lua"
rm -f "${COPPELIA_ROOT}/addOns/ur5_controller_video_addon.lua"
rm -f "${COPPELIA_ROOT}/addOns/ur5_acceleration_transport_video_addon.lua"
rm -f "${COPPELIA_ROOT}/addOns/ur5_fixed_z_acceleration_transport_addon.lua"
rm -f "${COPPELIA_ROOT}/addOns/coppeliasim_keepalive.lua"
rm -f "${COPPELIA_ROOT}/addOns/real_cartpole_controller_keepalive.lua"
rm -f "${COPPELIA_ROOT}/addOns/zz_real_cartpole_keepalive.lua"
rm -f "${COPPELIA_ROOT}/addOns/zz_real_cartpole_controller_keepalive.lua"

export COPPELIA_PYDEPS="${COPPELIA_PYDEPS:-${ROOT}/third_party/coppelia_pydeps}"
export REAL_CARTPOLE_ROOT="${ROOT}"
export REAL_CARTPOLE_ENABLE_VIDEO_SMOKE=0
case "${COPPELIASIM_LAUNCH_MODE}" in
  legacy_headless)
    export QT_QPA_PLATFORM="offscreen"
    ;;
  *)
    export QT_QPA_PLATFORM="xcb"
    ;;
esac
COPPELIA_PYDEPS_SITE_PACKAGES="${COPPELIA_PYDEPS}/lib/python3.12/site-packages"
COPPELIA_PYDEPS_DIST_PACKAGES="${COPPELIA_PYDEPS}/lib/python3.12/dist-packages"
export LD_LIBRARY_PATH="${COPPELIA_ROOT}:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${COPPELIA_ROOT}/programming/zmqRemoteApi/clients/python/src:${COPPELIA_PYDEPS_SITE_PACKAGES}:${COPPELIA_PYDEPS_DIST_PACKAGES}:${COPPELIA_PYDEPS}:${PYTHONPATH:-}"

echo "Using resident xvfb/plain CoppeliaSim launch mode for external ZMQ probes. The old -h/-vscriptinfos path is known to break require('sim') in this environment."

launch_prefix=()
launch_suffix=()
case "${COPPELIASIM_LAUNCH_MODE}" in
  xvfb_resident_plain)
    launch_prefix=(xvfb-run -a)
    ;;
  resident_plain)
    if [[ "${FORCE_XVFB}" == "1" || -z "${DISPLAY:-}" ]]; then
      launch_prefix=(xvfb-run -a)
    fi
    ;;
  legacy_headless)
    echo "WARNING: legacy_headless is known to fail ZMQ attach in this environment." >&2
    launch_suffix=(-h -vscriptinfos)
    ;;
  *)
    echo "Unsupported COPPELIASIM_LAUNCH_MODE: ${COPPELIASIM_LAUNCH_MODE}" >&2
    exit 1
    ;;
esac

port_listening=0
if command -v ss >/dev/null 2>&1 && ss -ltn "sport = :${ZMQ_RPC_PORT}" | grep -q LISTEN; then
  port_listening=1
fi

if [[ "${port_listening}" -eq 1 && "${ALLOW_EXISTING_SIM}" != "1" ]]; then
  echo "The ZMQ port is already in use. A stale CoppeliaSim instance may still be running. Stop it or pass --allow-existing-sim." >&2
  exit 1
fi

if [[ "${port_listening}" -eq 0 ]]; then
  echo "[probe-runner] Starting CoppeliaSim for probes on rpc port ${ZMQ_RPC_PORT} cnt port ${ZMQ_CNT_PORT}"
  launch_cmd=(
    "${launch_prefix[@]}"
    "${COPPELIA_ROOT}/coppeliaSim.sh"
    "${launch_suffix[@]}"
    "-GzmqRemoteApi.rpcPort=${ZMQ_RPC_PORT}"
    "-GzmqRemoteApi.cntPort=${ZMQ_CNT_PORT}"
  )
  if [[ -n "${COPPELIASIM_SCENE}" ]]; then
    launch_cmd+=("${COPPELIASIM_SCENE}")
  fi
  if [[ "${COPPELIASIM_LAUNCH_MODE}" != "legacy_headless" ]]; then
    for forbidden in -h -vscriptinfos; do
      if printf '%s\n' "${launch_cmd[@]}" | grep -qx -- "${forbidden}"; then
        echo "Refusing to use -h/-vscriptinfos for external ZMQ validation because this launch mode binds the RPC port but does not service require('sim') in this environment." >&2
        exit 1
      fi
    done
  fi
  if [[ -n "${COPPELIASIM_EXTRA_ARGS}" ]]; then
    read -r -a extra_args <<< "${COPPELIASIM_EXTRA_ARGS}"
    launch_cmd+=("${extra_args[@]}")
  fi
  echo -n "[probe-runner] exact launch command: "
  printf '%q ' "${launch_cmd[@]}"
  echo
  "${launch_cmd[@]}" >"${SIM_LOG}" 2>&1 &
  COPPELIA_PID=$!

  deadline=$((SECONDS + 60))
  while [[ "${SECONDS}" -lt "${deadline}" ]]; do
    if command -v ss >/dev/null 2>&1 && ss -ltn "sport = :${ZMQ_RPC_PORT}" | grep -q LISTEN; then
      break
    fi
    sleep 0.2
  done
  if ! (command -v ss >/dev/null 2>&1 && ss -ltn "sport = :${ZMQ_RPC_PORT}" | grep -q LISTEN); then
    echo "CoppeliaSim did not open the ZMQ port ${ZMQ_RPC_PORT} within 60 seconds." >&2
    sed -n '1,240p' "${SIM_LOG}" >&2 || true
    exit 1
  fi
fi

echo "Before debugging controller motion, run simulation/run_hpc_zmq_attach_probe.sh if running on HPC or if require('sim') times out."
echo "[probe-runner] Running handshake probe"
"${PYTHON_BIN}" "${HANDSHAKE_SCRIPT}" --host 127.0.0.1 --port "${ZMQ_RPC_PORT}" --summary-json "${HANDSHAKE_SUMMARY}"

echo "[probe-runner] Running single-joint torque probe"
"${PYTHON_BIN}" "${SINGLE_TORQUE_SCRIPT}" --host 127.0.0.1 --port "${ZMQ_RPC_PORT}" --summary-json "${SINGLE_TORQUE_SUMMARY}"

echo "[probe-runner] Handshake summary: ${HANDSHAKE_SUMMARY}"
echo "[probe-runner] Single-joint summary: ${SINGLE_TORQUE_SUMMARY}"
