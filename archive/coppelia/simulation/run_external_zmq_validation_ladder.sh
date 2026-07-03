#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

: "${COPPELIASIM_EXE:?COPPELIASIM_EXE is required}"
ZMQ_RPC_PORT="${ZMQ_RPC_PORT:-23000}"
ZMQ_CNT_PORT="${ZMQ_CNT_PORT:-$((ZMQ_RPC_PORT + 1))}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/control_runs/external_zmq_validation_ladder}"
RUN_FULL_CONTROLLER="${RUN_FULL_CONTROLLER:-0}"
STARTUP_GRACE_S="${STARTUP_GRACE_S:-10}"
ATTACH_TIMEOUT_S="${ATTACH_TIMEOUT_S:-90}"
COPPELIASIM_LAUNCH_MODE="${COPPELIASIM_LAUNCH_MODE:-xvfb_resident_plain}"
FORCE_XVFB="${FORCE_XVFB:-1}"
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
export LD_LIBRARY_PATH="${COPPELIASIM_EXE%/*}:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${COPPELIASIM_EXE%/*}/programming/zmqRemoteApi/clients/python/src:${COPPELIA_PYDEPS_SITE_PACKAGES}:${COPPELIA_PYDEPS_DIST_PACKAGES}:${COPPELIA_PYDEPS}:${PYTHONPATH:-}"

mkdir -p "${OUTPUT_DIR}"

printf 'hostname: %s\n' "$(hostname)"
printf 'date: %s\n' "$(date)"
printf 'pwd: %s\n' "$(pwd)"
printf 'SLURM_JOB_ID: %s\n' "${SLURM_JOB_ID:-}"
printf 'SLURM_JOB_NODELIST: %s\n' "${SLURM_JOB_NODELIST:-}"
printf 'SLURMD_NODENAME: %s\n' "${SLURMD_NODENAME:-}"
printf 'COPPELIASIM_EXE: %s\n' "${COPPELIASIM_EXE}"
printf 'COPPELIASIM_SCENE: %s\n' "${COPPELIASIM_SCENE:-}"
printf 'ZMQ_RPC_PORT: %s\n' "${ZMQ_RPC_PORT}"
printf 'ZMQ_CNT_PORT: %s\n' "${ZMQ_CNT_PORT}"
printf 'RUN_FULL_CONTROLLER: %s\n' "${RUN_FULL_CONTROLLER}"
printf 'COPPELIASIM_LAUNCH_MODE: %s\n' "${COPPELIASIM_LAUNCH_MODE}"
printf 'FORCE_XVFB: %s\n' "${FORCE_XVFB}"

FINAL_SUMMARY_JSON="${OUTPUT_DIR}/validation_ladder_summary.json"
FULL_HELP_TXT="${OUTPUT_DIR}/run_coppeliasim_x_axis_headless_help.txt"
FULL_RUN_LOG="${OUTPUT_DIR}/run_coppeliasim_x_axis_headless.log"
FULL_COPPELIA_LOG="${OUTPUT_DIR}/coppelia_full_controller.log"
FULL_CONTROLLER_SUMMARY="outputs/control_runs/external_zmq_validation_ladder_controller_summary.json"
FULL_CONTROLLER_TRACE="outputs/control_runs/external_zmq_validation_ladder_controller.jsonl"
ATTACH_SUMMARY_PATH="outputs/control_runs/hpc_zmq_attach/hpc_zmq_attach_summary.json"
ZERO_TORQUE_SUMMARY_PATH=""
SINGLE_TORQUE_SUMMARY_PATH=""
FULL_CONTROLLER_SUMMARY_PATH=""

SUCCESS="false"
ATTACH_ONLY_PASSED="false"
ZERO_TORQUE_PROBE_PASSED="false"
SINGLE_JOINT_TORQUE_PROBE_PASSED="false"
FULL_CONTROLLER_REQUESTED="false"
FULL_CONTROLLER_PASSED="null"
BLOCKING_LAYER="null"
ERROR_MESSAGE=""
FINAL_SUMMARY_WRITTEN=0

FULL_COPPELIA_PID=""

cleanup() {
  set +e
  if [[ -n "${FULL_COPPELIA_PID}" ]] && kill -0 "${FULL_COPPELIA_PID}" 2>/dev/null; then
    kill "${FULL_COPPELIA_PID}" 2>/dev/null || true
    wait "${FULL_COPPELIA_PID}" 2>/dev/null || true
  fi
  if [[ "${FINAL_SUMMARY_WRITTEN}" -eq 0 ]]; then
    write_final_summary
    FINAL_SUMMARY_WRITTEN=1
  fi
}
trap cleanup EXIT INT TERM

json_lookup() {
  local path="$1"
  local key="$2"
  python - "$path" "$key" <<'PY'
import json
import sys

path = sys.argv[1]
key = sys.argv[2]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
value = data.get(key)
if isinstance(value, bool):
    print("true" if value else "false")
elif value is None:
    print("null")
else:
    print(value)
PY
}

newest_summary_path() {
  local pattern="$1"
  local found
  found="$(find outputs/control_runs -type f -path "$pattern" -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)"
  printf '%s\n' "${found}"
}

require_summary_value() {
  local path="$1"
  local key="$2"
  local expected="$3"
  local layer="$4"
  local failure_message="$5"
  local actual
  if ! actual="$(json_lookup "${path}" "${key}")"; then
    fail_stage "${layer}" "${failure_message}"
  fi
  if [[ "${actual}" != "${expected}" ]]; then
    fail_stage "${layer}" "${failure_message}"
  fi
}

write_final_summary() {
  SUMMARY_PATH="${FINAL_SUMMARY_JSON}" \
  SUCCESS="${SUCCESS}" \
  ATTACH_ONLY_PASSED="${ATTACH_ONLY_PASSED}" \
  ZERO_TORQUE_PROBE_PASSED="${ZERO_TORQUE_PROBE_PASSED}" \
  SINGLE_JOINT_TORQUE_PROBE_PASSED="${SINGLE_JOINT_TORQUE_PROBE_PASSED}" \
  FULL_CONTROLLER_REQUESTED="${FULL_CONTROLLER_REQUESTED}" \
  FULL_CONTROLLER_PASSED="${FULL_CONTROLLER_PASSED}" \
  BLOCKING_LAYER="${BLOCKING_LAYER}" \
  ERROR_MESSAGE="${ERROR_MESSAGE}" \
  ATTACH_SUMMARY_PATH="${ATTACH_SUMMARY_PATH}" \
  ZERO_TORQUE_SUMMARY_PATH="${ZERO_TORQUE_SUMMARY_PATH}" \
  SINGLE_TORQUE_SUMMARY_PATH="${SINGLE_TORQUE_SUMMARY_PATH}" \
  FULL_CONTROLLER_SUMMARY_PATH="${FULL_CONTROLLER_SUMMARY_PATH}" \
  python - <<'PY'
import json
import os
from pathlib import Path


def parse_bool(value: str | None, default: bool | None = None):
    if value is None:
        return default
    value = value.strip().lower()
    if value == "true":
        return True
    if value == "false":
        return False
    if value == "null":
        return None
    return value


summary = {
    "success": parse_bool(os.environ.get("SUCCESS"), False),
    "attach_only_passed": parse_bool(os.environ.get("ATTACH_ONLY_PASSED"), False),
    "zero_torque_probe_passed": parse_bool(os.environ.get("ZERO_TORQUE_PROBE_PASSED"), False),
    "single_joint_torque_probe_passed": parse_bool(os.environ.get("SINGLE_JOINT_TORQUE_PROBE_PASSED"), False),
    "full_controller_requested": parse_bool(os.environ.get("FULL_CONTROLLER_REQUESTED"), False),
    "full_controller_passed": parse_bool(os.environ.get("FULL_CONTROLLER_PASSED"), None),
    "blocking_layer": parse_bool(os.environ.get("BLOCKING_LAYER"), None),
    "error": os.environ.get("ERROR_MESSAGE") or None,
    "attach_summary_path": os.environ.get("ATTACH_SUMMARY_PATH") or None,
    "zero_torque_summary_path": os.environ.get("ZERO_TORQUE_SUMMARY_PATH") or None,
    "single_joint_torque_summary_path": os.environ.get("SINGLE_JOINT_TORQUE_SUMMARY_PATH") or None,
    "full_controller_summary_path": os.environ.get("FULL_CONTROLLER_SUMMARY_PATH") or None,
}
path = Path(os.environ["SUMMARY_PATH"])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
PY
}

fail_stage() {
  local layer="$1"
  local message="$2"
  SUCCESS="false"
  BLOCKING_LAYER="${layer}"
  ERROR_MESSAGE="${message}"
  write_final_summary
  FINAL_SUMMARY_WRITTEN=1
  echo "[ladder] ERROR: ${message}" >&2
  echo "[ladder] blocking_layer=${layer}" >&2
  exit 1
}

if [[ ! -e "${COPPELIASIM_EXE}" ]]; then
  fail_stage "attach_only" "COPPELIASIM_EXE does not exist: ${COPPELIASIM_EXE}"
fi
if [[ ! -x "${COPPELIASIM_EXE}" ]]; then
  fail_stage "attach_only" "COPPELIASIM_EXE is not executable: ${COPPELIASIM_EXE}"
fi

echo "[ladder] Stage 1: HPC/ZMQ attach-only probe"
if ! COPPELIASIM_EXE="${COPPELIASIM_EXE}" \
  COPPELIASIM_SCENE="${COPPELIASIM_SCENE}" \
  COPPELIASIM_LAUNCH_MODE="${COPPELIASIM_LAUNCH_MODE}" \
  FORCE_XVFB="${FORCE_XVFB}" \
  COPPELIASIM_EXTRA_ARGS="${COPPELIASIM_EXTRA_ARGS}" \
  ZMQ_RPC_PORT="${ZMQ_RPC_PORT}" \
  ZMQ_CNT_PORT="${ZMQ_CNT_PORT}" \
  STARTUP_GRACE_S="${STARTUP_GRACE_S}" \
  ATTACH_TIMEOUT_S="${ATTACH_TIMEOUT_S}" \
  bash simulation/run_hpc_zmq_attach_probe.sh; then
  fail_stage "attach_only" "Blocking layer: CoppeliaSim/ZMQ attach. Do not debug controller gains yet."
fi

if [[ ! -f "${ATTACH_SUMMARY_PATH}" ]]; then
  fail_stage "attach_only" "Blocking layer: CoppeliaSim/ZMQ attach. Do not debug controller gains yet."
fi
if ! ATTACH_SUCCESS="$(json_lookup "${ATTACH_SUMMARY_PATH}" success)"; then
  fail_stage "attach_only" "Blocking layer: CoppeliaSim/ZMQ attach. Do not debug controller gains yet."
fi
if ! ATTACH_REQUIRE_SIM_OK="$(json_lookup "${ATTACH_SUMMARY_PATH}" require_sim_ok)"; then
  fail_stage "attach_only" "Blocking layer: CoppeliaSim/ZMQ attach. Do not debug controller gains yet."
fi
if ! ATTACH_GET_STATE_OK="$(json_lookup "${ATTACH_SUMMARY_PATH}" get_simulation_state_ok)"; then
  fail_stage "attach_only" "Blocking layer: CoppeliaSim/ZMQ attach. Do not debug controller gains yet."
fi
if [[ "${ATTACH_SUCCESS}" != "true" || "${ATTACH_REQUIRE_SIM_OK}" != "true" || "${ATTACH_GET_STATE_OK}" != "true" ]]; then
  fail_stage "attach_only" "Blocking layer: CoppeliaSim/ZMQ attach. Do not debug controller gains yet."
fi
ATTACH_ONLY_PASSED="true"

echo "[ladder] Stage 2: zero-torque and single-joint torque probes"
if ! ZMQ_RPC_PORT="${ZMQ_RPC_PORT}" \
  ZMQ_CNT_PORT="${ZMQ_CNT_PORT}" \
  COPPELIA_ROOT="${COPPELIA_ROOT}" \
  COPPELIASIM_LAUNCH_MODE="${COPPELIASIM_LAUNCH_MODE}" \
  FORCE_XVFB="${FORCE_XVFB}" \
  COPPELIASIM_SCENE="${COPPELIASIM_SCENE}" \
  COPPELIASIM_EXTRA_ARGS="${COPPELIASIM_EXTRA_ARGS}" \
  bash simulation/run_external_zmq_probes.sh; then
  echo "[ladder] external ZMQ probe runner returned nonzero; inspecting summaries" >&2
fi

ZERO_SUMMARY="$(newest_summary_path '*/external_zmq_handshake/summary.json')"
SINGLE_SUMMARY="$(newest_summary_path '*/external_zmq_single_joint_torque/summary.json')"
ZERO_TORQUE_SUMMARY_PATH="${ZERO_SUMMARY}"
SINGLE_TORQUE_SUMMARY_PATH="${SINGLE_SUMMARY}"
if [[ -z "${ZERO_SUMMARY}" ]]; then
  fail_stage "zero_torque_probe" "Blocking layer: Python-owned stepping/startup."
fi
if [[ -z "${SINGLE_SUMMARY}" ]]; then
  fail_stage "single_joint_torque_probe" "Blocking layer: joint dynamic mode / force-torque mode / torque API path."
fi

require_summary_value "${ZERO_SUMMARY}" success true zero_torque_probe "Blocking layer: Python-owned stepping/startup."
require_summary_value "${ZERO_SUMMARY}" controller_family python_zmq_external_zero_torque_probe zero_torque_probe "Blocking layer: Python-owned stepping/startup."
require_summary_value "${ZERO_SUMMARY}" stepping_owner python_zmq zero_torque_probe "Blocking layer: Python-owned stepping/startup."
require_summary_value "${ZERO_SUMMARY}" uses_direct_torque_control true zero_torque_probe "Blocking layer: Python-owned stepping/startup."
require_summary_value "${ZERO_SUMMARY}" simulation_started_by python zero_torque_probe "Blocking layer: Python-owned stepping/startup."
require_summary_value "${ZERO_SUMMARY}" lua_motion_enabled false zero_torque_probe "Blocking layer: Python-owned stepping/startup."
ZERO_TORQUE_PROBE_PASSED="true"

require_summary_value "${SINGLE_SUMMARY}" success true single_joint_torque_probe "Blocking layer: joint dynamic mode / force-torque mode / torque API path."
require_summary_value "${SINGLE_SUMMARY}" controller_family python_zmq_external_single_joint_torque_probe single_joint_torque_probe "Blocking layer: joint dynamic mode / force-torque mode / torque API path."
require_summary_value "${SINGLE_SUMMARY}" stepping_owner python_zmq single_joint_torque_probe "Blocking layer: joint dynamic mode / force-torque mode / torque API path."
require_summary_value "${SINGLE_SUMMARY}" uses_direct_torque_control true single_joint_torque_probe "Blocking layer: joint dynamic mode / force-torque mode / torque API path."
require_summary_value "${SINGLE_SUMMARY}" simulation_started_by python single_joint_torque_probe "Blocking layer: joint dynamic mode / force-torque mode / torque API path."
require_summary_value "${SINGLE_SUMMARY}" lua_motion_enabled false single_joint_torque_probe "Blocking layer: joint dynamic mode / force-torque mode / torque API path."
require_summary_value "${SINGLE_SUMMARY}" joint_0_displacement_nonzero true single_joint_torque_probe "Blocking layer: joint dynamic mode / force-torque mode / torque API path."
SINGLE_JOINT_TORQUE_PROBE_PASSED="true"

if [[ "${RUN_FULL_CONTROLLER}" != "1" ]]; then
  SUCCESS="true"
  FULL_CONTROLLER_REQUESTED="false"
  FULL_CONTROLLER_PASSED="null"
  BLOCKING_LAYER="null"
  ERROR_MESSAGE=""
  write_final_summary
  FINAL_SUMMARY_WRITTEN=1
  echo "External ZMQ attach, stepping, and single-joint torque probes passed. Full Cartesian impedance controller was not run because RUN_FULL_CONTROLLER=0."
  echo "[ladder] final_summary=${FINAL_SUMMARY_JSON}"
  exit 0
fi

FULL_CONTROLLER_REQUESTED="true"
FULL_CONTROLLER_SUMMARY_PATH="${FULL_CONTROLLER_SUMMARY}"
echo "[ladder] Stage 3: full Cartesian impedance controller run requested"

if ! python simulation/run_coppeliasim_x_axis_headless.py --help >"${FULL_HELP_TXT}" 2>&1; then
  fail_stage "full_controller_help" "Could not print help for simulation/run_coppeliasim_x_axis_headless.py."
fi
for flag in --no-video --probe-only --task-frame-mode; do
  if ! grep -q -- "${flag}" "${FULL_HELP_TXT}"; then
    fail_stage "full_controller_help" "Expected flag ${flag} was not present in the live runner help output."
  fi
done

if command -v ss >/dev/null 2>&1 && ss -ltn "sport = :${ZMQ_RPC_PORT}" | grep -q LISTEN; then
  fail_stage "full_controller" "The ZMQ RPC port is already in use before the full controller launch."
fi

rm -f "${FULL_CONTROLLER_SUMMARY}" "${FULL_CONTROLLER_TRACE}" "${FULL_RUN_LOG}" "${FULL_HELP_TXT}" "${FULL_COPPELIA_LOG}"

FULL_CONTROLLER_LAUNCH_PREFIX=()
case "${COPPELIASIM_LAUNCH_MODE}" in
  xvfb_resident_plain)
    FULL_CONTROLLER_LAUNCH_PREFIX=(xvfb-run -a)
    ;;
  resident_plain)
    if [[ "${FORCE_XVFB}" == "1" || -z "${DISPLAY:-}" ]]; then
      FULL_CONTROLLER_LAUNCH_PREFIX=(xvfb-run -a)
    fi
    ;;
  legacy_headless)
    echo "WARNING: legacy_headless is known to fail ZMQ attach in this environment." >&2
    ;;
  *)
    fail_stage "full_controller" "Unsupported COPPELIASIM_LAUNCH_MODE: ${COPPELIASIM_LAUNCH_MODE}"
    ;;
esac

if [[ "${COPPELIASIM_LAUNCH_MODE}" == "resident_plain" && "${FORCE_XVFB}" != "1" && -z "${DISPLAY:-}" ]]; then
  fail_stage "full_controller" "resident_plain requires an available DISPLAY unless FORCE_XVFB=1."
fi

FULL_CONTROLLER_COPPELIA_ARGS=(
  "${FULL_CONTROLLER_LAUNCH_PREFIX[@]}"
  "${COPPELIASIM_EXE}"
)
if [[ "${COPPELIASIM_LAUNCH_MODE}" == "legacy_headless" ]]; then
  FULL_CONTROLLER_COPPELIA_ARGS+=(-h -vscriptinfos)
fi
FULL_CONTROLLER_COPPELIA_ARGS+=("-GzmqRemoteApi.rpcPort=${ZMQ_RPC_PORT}")
FULL_CONTROLLER_COPPELIA_ARGS+=("-GzmqRemoteApi.cntPort=${ZMQ_CNT_PORT}")
if [[ -n "${COPPELIASIM_SCENE}" ]]; then
  FULL_CONTROLLER_COPPELIA_ARGS+=("${COPPELIASIM_SCENE}")
fi
if [[ -n "${COPPELIASIM_EXTRA_ARGS}" ]]; then
  read -r -a full_extra_args <<< "${COPPELIASIM_EXTRA_ARGS}"
  FULL_CONTROLLER_COPPELIA_ARGS+=("${full_extra_args[@]}")
fi
if [[ "${COPPELIASIM_LAUNCH_MODE}" != "legacy_headless" ]]; then
  for forbidden in -h -vscriptinfos; do
    if printf '%s\n' "${FULL_CONTROLLER_COPPELIA_ARGS[@]}" | grep -qx -- "${forbidden}"; then
      fail_stage "full_controller" "Refusing to use -h/-vscriptinfos for external ZMQ validation because this launch mode binds the RPC port but does not service require('sim') in this environment."
    fi
  done
fi

echo -n "[ladder] exact launch command: "
printf '%q ' "${FULL_CONTROLLER_COPPELIA_ARGS[@]}"
echo
echo "[ladder] launching CoppeliaSim for the live controller"
"${FULL_CONTROLLER_COPPELIA_ARGS[@]}" \
  >"${FULL_COPPELIA_LOG}" 2>&1 &
FULL_COPPELIA_PID=$!

sleep "${STARTUP_GRACE_S}"

echo "[ladder] running the live Cartesian impedance controller"
FULL_CONTROLLER_EXIT=0
set +e
python simulation/run_coppeliasim_x_axis_headless.py \
  --coppelia-root "${COPPELIA_ROOT}" \
  --host 127.0.0.1 \
  --port "${ZMQ_RPC_PORT}" \
  --no-video \
  --duration 3 \
  --settle-duration 1 \
  --target-dx 0.005 \
  --task-frame-mode mujoco_attachment_dummy \
  --summary-name external_zmq_validation_ladder_controller_summary.json \
  --trace-name external_zmq_validation_ladder_controller.jsonl \
  >"${FULL_RUN_LOG}" 2>&1
FULL_CONTROLLER_EXIT=$?
set -e

if [[ ! -f "${FULL_CONTROLLER_SUMMARY}" ]]; then
  FULL_CONTROLLER_PASSED="false"
  fail_stage "full_controller" "Live Cartesian impedance controller did not pass."
fi
if ! FULL_CONTROLLER_SUCCESS="$(json_lookup "${FULL_CONTROLLER_SUMMARY}" success)"; then
  FULL_CONTROLLER_PASSED="false"
  fail_stage "full_controller" "Live Cartesian impedance controller did not pass."
fi
if ! FULL_CONTROLLER_FAMILY="$(json_lookup "${FULL_CONTROLLER_SUMMARY}" controller_family)"; then
  FULL_CONTROLLER_PASSED="false"
  fail_stage "full_controller" "Live Cartesian impedance controller did not pass."
fi
if ! FULL_CONTROLLER_DIRECT="$(json_lookup "${FULL_CONTROLLER_SUMMARY}" uses_direct_torque_control)"; then
  FULL_CONTROLLER_PASSED="false"
  fail_stage "full_controller" "Live Cartesian impedance controller did not pass."
fi
if ! FULL_CONTROLLER_STEPPING_OWNER="$(json_lookup "${FULL_CONTROLLER_SUMMARY}" stepping_owner)"; then
  FULL_CONTROLLER_PASSED="false"
  fail_stage "full_controller" "Live Cartesian impedance controller did not pass."
fi
if ! FULL_CONTROLLER_STARTED_BY="$(json_lookup "${FULL_CONTROLLER_SUMMARY}" simulation_started_by)"; then
  FULL_CONTROLLER_PASSED="false"
  fail_stage "full_controller" "Live Cartesian impedance controller did not pass."
fi
if ! FULL_CONTROLLER_LUA_MOTION="$(json_lookup "${FULL_CONTROLLER_SUMMARY}" lua_motion_enabled)"; then
  FULL_CONTROLLER_PASSED="false"
  fail_stage "full_controller" "Live Cartesian impedance controller did not pass."
fi
if [[ "${FULL_CONTROLLER_SUCCESS}" != "true" || "${FULL_CONTROLLER_FAMILY}" != "python_zmq_external_cartesian_impedance" || "${FULL_CONTROLLER_DIRECT}" != "true" || "${FULL_CONTROLLER_STEPPING_OWNER}" != "python_zmq" || "${FULL_CONTROLLER_STARTED_BY}" != "python" || "${FULL_CONTROLLER_LUA_MOTION}" != "false" ]]; then
  FULL_CONTROLLER_PASSED="false"
  fail_stage "full_controller" "Live Cartesian impedance controller did not pass."
fi
if [[ "${FULL_CONTROLLER_EXIT}" -ne 0 ]]; then
  FULL_CONTROLLER_PASSED="false"
  fail_stage "full_controller" "Live Cartesian impedance controller did not pass."
fi
FULL_CONTROLLER_PASSED="true"

SUCCESS="true"
BLOCKING_LAYER="null"
ERROR_MESSAGE=""
write_final_summary
FINAL_SUMMARY_WRITTEN=1
echo "External ZMQ attach, stepping, single-joint torque, and full Cartesian impedance controller passed."
echo "[ladder] final_summary=${FINAL_SUMMARY_JSON}"
exit 0
