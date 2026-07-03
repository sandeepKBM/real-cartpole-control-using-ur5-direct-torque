#!/usr/bin/env bash
set -euo pipefail

ROOT="/common/users/ss5772/real_Cartpole"
COPPELIA_ROOT="${COPPELIA_ROOT:-${ROOT}/third_party/coppelia_runtime/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu24_04}"
STATE_DIR="${ROOT}/outputs/control_runs/coppelia_origin_acquisition_state"
SIM_LOG="${STATE_DIR}/coppelia_x_range_sweep.log"
BOOT_LOG="${STATE_DIR}/x_range_sweep_bootstrap.log"
SUMMARY_PATH="${STATE_DIR}/coppeliasim_ur5_x_range_height_sweep_summary.json"
ADDON_SOURCE="${ROOT}/simulation/ur5_origin_acquisition_video_addon.lua"
ADDON_TARGET="${COPPELIA_ROOT}/addOns/ur5_origin_acquisition_video_addon.lua"
SMOKE_ADDON_TARGET="${COPPELIA_ROOT}/addOns/ur5_video_smoke_addon.lua"
CONTROLLER_ADDON_TARGET="${COPPELIA_ROOT}/addOns/ur5_controller_video_addon.lua"
ACCEL_ADDON_TARGET="${COPPELIA_ROOT}/addOns/ur5_acceleration_transport_video_addon.lua"
FIXED_Z_ACCEL_ADDON_TARGET="${COPPELIA_ROOT}/addOns/ur5_fixed_z_acceleration_transport_addon.lua"
LOAD_MARKER="${STATE_DIR}/ur5_origin_acquisition_addon_loaded.txt"
START_MARKER="${STATE_DIR}/ur5_origin_acquisition_addon_started.txt"
DONE_MARKER="${STATE_DIR}/ur5_origin_acquisition_done.txt"
SIM_TIMEOUT="${SIM_TIMEOUT:-90}"
SWEEP_Z_MIN_M="${SWEEP_Z_MIN_M:-0.35}"
SWEEP_Z_MAX_M="${SWEEP_Z_MAX_M:-0.95}"
SWEEP_Z_STEP_M="${SWEEP_Z_STEP_M:-0.025}"
RANGE_SCAN_MAX_M="${RANGE_SCAN_MAX_M:-0.35}"
RANGE_SCAN_STEP_M="${RANGE_SCAN_STEP_M:-0.0025}"
RANGE_POSITION_TOL_M="${RANGE_POSITION_TOL_M:-0.005}"
ORIENTATION_TOL_DEG="${ORIENTATION_TOL_DEG:-3.0}"
ACCEL_ROT_WEIGHT="${ACCEL_ROT_WEIGHT:-0.05}"
MODEL_BASE_Z_OFFSET_M="${MODEL_BASE_Z_OFFSET_M:-0.0}"

cleanup() {
  if [[ -n "${SIM_PID:-}" ]] && kill -0 "${SIM_PID}" 2>/dev/null; then
    kill "${SIM_PID}" 2>/dev/null || true
    wait "${SIM_PID}" 2>/dev/null || true
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

mkdir -p "${STATE_DIR}"
mkdir -p "${COPPELIA_ROOT}/addOns"

rm -f "${SIM_LOG}" "${BOOT_LOG}" "${SUMMARY_PATH}"
rm -f "${LOAD_MARKER}" "${START_MARKER}" "${DONE_MARKER}"
rm -f "${SMOKE_ADDON_TARGET}" "${CONTROLLER_ADDON_TARGET}" "${ACCEL_ADDON_TARGET}" "${FIXED_Z_ACCEL_ADDON_TARGET}"
cp -f "${ADDON_SOURCE}" "${ADDON_TARGET}"

log_bootstrap() {
  printf '%s\n' "$*" >>"${BOOT_LOG}"
}

log_bootstrap "COPPELIA_ROOT=${COPPELIA_ROOT}"
log_bootstrap "SWEEP_Z_MIN_M=${SWEEP_Z_MIN_M}"
log_bootstrap "SWEEP_Z_MAX_M=${SWEEP_Z_MAX_M}"
log_bootstrap "SWEEP_Z_STEP_M=${SWEEP_Z_STEP_M}"
log_bootstrap "RANGE_SCAN_MAX_M=${RANGE_SCAN_MAX_M}"
log_bootstrap "RANGE_SCAN_STEP_M=${RANGE_SCAN_STEP_M}"
log_bootstrap "RANGE_POSITION_TOL_M=${RANGE_POSITION_TOL_M}"
log_bootstrap "ORIENTATION_TOL_DEG=${ORIENTATION_TOL_DEG}"

cd "${COPPELIA_ROOT}"
xvfb-run -a /usr/bin/env \
  COPPELIA_ROOT="${COPPELIA_ROOT}" \
  REAL_CARTPOLE_ROOT="${ROOT}" \
  RANGE_SWEEP_ONLY=1 \
  SUMMARY_PATH="${SUMMARY_PATH}" \
  SWEEP_Z_MIN_M="${SWEEP_Z_MIN_M}" \
  SWEEP_Z_MAX_M="${SWEEP_Z_MAX_M}" \
  SWEEP_Z_STEP_M="${SWEEP_Z_STEP_M}" \
  RANGE_SCAN_MAX_M="${RANGE_SCAN_MAX_M}" \
  RANGE_SCAN_STEP_M="${RANGE_SCAN_STEP_M}" \
  RANGE_POSITION_TOL_M="${RANGE_POSITION_TOL_M}" \
  ORIGIN_ORIENTATION_TOL_DEG="${ORIENTATION_TOL_DEG}" \
  ACCEL_ROT_WEIGHT="${ACCEL_ROT_WEIGHT}" \
  MODEL_BASE_Z_OFFSET_M="${MODEL_BASE_Z_OFFSET_M}" \
  "${COPPELIA_ROOT}/coppeliaSim.sh" -h -vscriptinfos \
  >"${SIM_LOG}" 2>&1 &
SIM_PID=$!

deadline=$((SECONDS + SIM_TIMEOUT))
while kill -0 "${SIM_PID}" 2>/dev/null; do
  if [[ -f "${DONE_MARKER}" ]]; then
    break
  fi
  if [[ "${SECONDS}" -ge "${deadline}" ]]; then
    break
  fi
  sleep 1
done

if kill -0 "${SIM_PID}" 2>/dev/null; then
  kill "${SIM_PID}" 2>/dev/null || true
  wait "${SIM_PID}" 2>/dev/null || true
else
  wait "${SIM_PID}" 2>/dev/null || true
fi
SIM_PID=""

if [[ ! -f "${SUMMARY_PATH}" ]]; then
  echo "Missing range sweep summary JSON: ${SUMMARY_PATH}" >&2
  sed -n '1,260p' "${SIM_LOG}" >&2 || true
  exit 1
fi

echo "Saved X range height sweep summary: ${SUMMARY_PATH}"
