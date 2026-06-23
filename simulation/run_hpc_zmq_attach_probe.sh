#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

: "${COPPELIASIM_EXE:?COPPELIASIM_EXE is required}"
ZMQ_RPC_PORT="${ZMQ_RPC_PORT:-23000}"
ZMQ_CNT_PORT="${ZMQ_CNT_PORT:-$((ZMQ_RPC_PORT + 1))}"
COPPELIASIM_LAUNCH_MODE="${COPPELIASIM_LAUNCH_MODE:-xvfb_resident_plain}"
FORCE_XVFB="${FORCE_XVFB:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/control_runs/hpc_zmq_attach}"
STARTUP_GRACE_S="${STARTUP_GRACE_S:-5}"
ATTACH_TIMEOUT_S="${ATTACH_TIMEOUT_S:-60}"
COPPELIASIM_SCENE="${COPPELIASIM_SCENE:-}"
COPPELIASIM_EXTRA_ARGS="${COPPELIASIM_EXTRA_ARGS:-}"
COPPELIA_ROOT="$(cd "$(dirname "${COPPELIASIM_EXE}")" && pwd)"
export COPPELIA_ROOT
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

mkdir -p "${OUTPUT_DIR}"

printf 'date: %s\n' "$(date)"
printf 'hostname: %s\n' "$(hostname)"
printf 'pwd: %s\n' "$(pwd)"
printf 'SLURM_JOB_ID: %s\n' "${SLURM_JOB_ID:-}"
printf 'SLURM_JOB_NODELIST: %s\n' "${SLURM_JOB_NODELIST:-}"
printf 'SLURMD_NODENAME: %s\n' "${SLURMD_NODENAME:-}"
printf 'COPPELIASIM_EXE: %s\n' "${COPPELIASIM_EXE}"
printf 'COPPELIASIM_SCENE: %s\n' "${COPPELIASIM_SCENE:-}"
printf 'ZMQ_RPC_PORT: %s\n' "${ZMQ_RPC_PORT}"
printf 'ZMQ_CNT_PORT: %s\n' "${ZMQ_CNT_PORT}"

if [[ ! -e "${COPPELIASIM_EXE}" ]]; then
  echo "COPPELIASIM_EXE does not exist: ${COPPELIASIM_EXE}" >&2
  exit 1
fi
if [[ ! -x "${COPPELIASIM_EXE}" ]]; then
  echo "COPPELIASIM_EXE is not executable: ${COPPELIASIM_EXE}" >&2
  exit 1
fi

SUMMARY_JSON="${OUTPUT_DIR}/hpc_zmq_attach_summary.json"
COPPELIA_LOG="${OUTPUT_DIR}/coppeliasim.log"

rm -f "${COPPELIA_ROOT}/addOns/ur5_video_smoke_addon.lua"
rm -f "${COPPELIA_ROOT}/addOns/ur5_controller_video_addon.lua"
rm -f "${COPPELIA_ROOT}/addOns/ur5_acceleration_transport_video_addon.lua"
rm -f "${COPPELIA_ROOT}/addOns/ur5_fixed_z_acceleration_transport_addon.lua"
rm -f "${COPPELIA_ROOT}/addOns/coppeliasim_keepalive.lua"
rm -f "${COPPELIA_ROOT}/addOns/real_cartpole_controller_keepalive.lua"
rm -f "${COPPELIA_ROOT}/addOns/zz_real_cartpole_keepalive.lua"
rm -f "${COPPELIA_ROOT}/addOns/zz_real_cartpole_controller_keepalive.lua"

extra_args=()
if [[ -n "${COPPELIASIM_EXTRA_ARGS}" ]]; then
  read -r -a extra_args <<< "${COPPELIASIM_EXTRA_ARGS}"
fi

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

launch_args=(
  "${launch_prefix[@]}"
  "${COPPELIASIM_EXE}"
  "${launch_suffix[@]}"
  "-GzmqRemoteApi.rpcPort=${ZMQ_RPC_PORT}"
  "-GzmqRemoteApi.cntPort=${ZMQ_CNT_PORT}"
)
if [[ -n "${COPPELIASIM_SCENE}" ]]; then
  launch_args+=("${COPPELIASIM_SCENE}")
fi
if [[ ${#extra_args[@]} -gt 0 ]]; then
  launch_args+=("${extra_args[@]}")
fi

if [[ "${COPPELIASIM_LAUNCH_MODE}" != "legacy_headless" ]]; then
  for forbidden in -h -vscriptinfos; do
    if printf '%s\n' "${launch_args[@]}" | grep -qx -- "${forbidden}"; then
      echo "Refusing to use -h/-vscriptinfos for external ZMQ validation because this launch mode binds the RPC port but does not service require('sim') in this environment." >&2
      exit 1
    fi
  done
fi

COPPELIA_PID=""
cleanup() {
  if [[ -n "${COPPELIA_PID}" ]] && kill -0 "${COPPELIA_PID}" 2>/dev/null; then
    kill "${COPPELIA_PID}" 2>/dev/null || true
    wait "${COPPELIA_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "[hpc-zmq-attach] launch mode: ${COPPELIASIM_LAUNCH_MODE}"
echo -n "[hpc-zmq-attach] exact launch command: "
printf '%q ' "${launch_args[@]}"
echo
echo "[hpc-zmq-attach] launching CoppeliaSim"
"${launch_args[@]}" >"${COPPELIA_LOG}" 2>&1 &
COPPELIA_PID=$!
echo "[hpc-zmq-attach] launched PID: ${COPPELIA_PID}"

if ! kill -0 "${COPPELIA_PID}" 2>/dev/null; then
  echo "Blocking layer: CoppeliaSim residency. The simulator exited before the ZMQ attach probe could complete." >&2
  exit 1
fi

sleep "${STARTUP_GRACE_S}"

if ! kill -0 "${COPPELIA_PID}" 2>/dev/null; then
  echo "Blocking layer: CoppeliaSim residency. The simulator exited before the ZMQ attach probe could complete." >&2
  exit 1
fi

PROBE_RC=0
if python simulation/probe_hpc_zmq_attach.py \
  --host 127.0.0.1 \
  --rpc-port "${ZMQ_RPC_PORT}" \
  --cnt-port "${ZMQ_CNT_PORT}" \
  --timeout-s "${ATTACH_TIMEOUT_S}" \
  --summary-json "${SUMMARY_JSON}" \
  --coppeliasim-log "${COPPELIA_LOG}"; then
  :
else
  PROBE_RC=$?
fi

printf 'summary_json: %s\n' "${SUMMARY_JSON}"
printf 'coppeliasim_log: %s\n' "${COPPELIA_LOG}"

exit "${PROBE_RC}"
