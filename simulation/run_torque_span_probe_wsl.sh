#!/usr/bin/env bash
# Measure EE Y transport span (no video) for torque policies on WSL.
set -euo pipefail
ROOT="/mnt/c/Users/sandr/Downloads/real-cartpole-control-using-ur5-direct-torque"
# shellcheck disable=SC1091
source "${ROOT}/simulation/env_wsl_local.sh"

COPPELIA="${COPPELIA_ROOT}"
CONFIG="${ROOT}/ros2_ws/src/ur5_x_axis_controller_ros/config/controller_coppelia_mpc.yaml"
RUNNER="${ROOT}/simulation/run_coppeliasim_x_axis_headless.py"
OUT_DIR="${ROOT}/outputs/control_runs/torque_span_probe"
mkdir -p "${OUT_DIR}"

export DISPLAY="${DISPLAY:-:0}"
unset QT_QPA_PLATFORM
export COPPELIA_ROOT COPPELIA_PYDEPS REAL_CARTPOLE_ROOT
export REAL_CARTPOLE_ENABLE_VIDEO_SMOKE=0
export Q_START_RAD="0,0.0443244063,-1.67570517,5.09435844,-6.28345754,5.96178335"
export LD_LIBRARY_PATH="${COPPELIA}:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${COPPELIA}/programming/zmqRemoteApi/clients/python/src:${COPPELIA_PYDEPS}:${PYTHONPATH:-}"

run_case() {
  local name="$1"
  shift
  local port=$((23100 + RANDOM % 500))
  local sim_log="${OUT_DIR}/${name}_sim.log"
  local run_log="${OUT_DIR}/${name}_runner.log"
  local summary="${OUT_DIR}/${name}_summary.json"

  pkill -f "zmqRemoteApi.rpcPort=${port}" 2>/dev/null || true
  pkill -f coppeliaSim 2>/dev/null || true
  sleep 1

  cd "${COPPELIA}"
  ./coppeliaSim.sh \
    -GzmqRemoteApi.rpcPort="${port}" -GzmqRemoteApi.cntPort="$((port + 1))" \
    >"${sim_log}" 2>&1 &
  local pid=$!

  for _ in $(seq 1 60); do
    ss -ltn "sport = :${port}" 2>/dev/null | grep -q LISTEN && break
    sleep 1
  done
  if ! ss -ltn "sport = :${port}" 2>/dev/null | grep -q LISTEN; then
    echo "${name}: Coppelia failed to start" >&2
    kill "${pid}" 2>/dev/null || true
    return 1
  fi
  sleep 2

  cd "${ROOT}"
  set +e
  "${PYTHON_BIN}" "${RUNNER}" \
    --coppelia-root "${COPPELIA}" \
    --port "${port}" \
    --config "${CONFIG}" \
    --no-video \
    --accel-x-transport \
    --transport-axis y \
    --hold-transport-start-pose \
    --task-frame-mode mujoco_attachment_dummy \
    --accel-torque-policy qp_torque \
    --spawn-coppelia-pendulum \
    --summary-name "torque_span_probe/${name}_summary.json" \
    "$@" \
    >"${run_log}" 2>&1
  local rc=$?
  set -e

  kill "${pid}" 2>/dev/null || true
  wait "${pid}" 2>/dev/null || true

  if [[ -f "${ROOT}/outputs/control_runs/torque_span_probe/${name}_summary.json" ]]; then
  python3 - <<PY
import json
p = "${ROOT}/outputs/control_runs/torque_span_probe/${name}_summary.json"
s = json.load(open(p))
print("${name}",
      "success=", s.get("success"),
      "policy=", s.get("accel_torque_policy"),
      "profile=", s.get("accel_profile"),
      "net_y_m=", s.get("transport_axis_net_displacement_m"),
      "span_y_m=", s.get("transport_axis_span_m"),
      "z_drift_m=", s.get("max_abs_fixed_axis_2_drift_m"),
      "stop=", s.get("safety_stop_reason"),
      sep="")
PY
  else
    echo "${name}: no summary (rc=${rc})"
    tail -5 "${run_log}" 2>/dev/null || true
  fi
}

echo "=== Torque Y-span probe (qp_torque, no video) ==="

run_case "mpc_qp_dy02" \
  --accel-profile mpc --target-dx 0.02 \
  --duration 8 --settle-duration 2 \
  --a-x-max 0.06 --v-x-max 0.03

run_case "mpc_qp_dy06" \
  --accel-profile mpc --target-dx 0.06 \
  --duration 12 --settle-duration 2 \
  --a-x-max 0.06 --v-x-max 0.03

run_case "recip_qp_stroke06" \
  --accel-profile reciprocating --reciprocating-stroke-m 0.06 \
  --duration 0 --settle-duration 2 \
  --a-x-max 0.05 --v-x-max 0.025 --reciprocating-hold-s 0.5

run_case "recip_qp_stroke10" \
  --accel-profile reciprocating --reciprocating-stroke-m 0.10 \
  --duration 0 --settle-duration 2 \
  --a-x-max 0.04 --v-x-max 0.02 --reciprocating-hold-s 0.5

run_case "ppt_qp_dy08" \
  --accel-profile point_to_point --target-dx 0.08 \
  --duration 14 --settle-duration 2 \
  --a-x-max 0.05 --v-x-max 0.025

echo "=== Done ==="
