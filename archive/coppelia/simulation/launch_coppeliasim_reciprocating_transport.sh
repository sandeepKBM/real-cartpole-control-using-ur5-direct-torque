#!/usr/bin/env bash
# CoppeliaSim reciprocating EE transport (origin -> +stroke -> -stroke -> origin).
set -euo pipefail

ROOT="/common/users/ss5772/real_Cartpole"
CONFIG="${ROOT}/ros2_ws/src/ur5_x_axis_controller_ros/config/controller_coppelia_reciprocating.yaml"
PORT="${PORT:-23280}"
RUN_SUFFIX="${RUN_SUFFIX:-coppelia_reciprocating}"

exec bash "${ROOT}/simulation/launch_coppeliasim_x_axis_headless.sh" \
  --config "${CONFIG}" \
  --accel-x-transport \
  --accel-profile reciprocating \
  --accel-torque-policy "${ACCEL_TORQUE_POLICY:-cartesian_impedance}" \
  --transport-axis x \
  --reciprocating-stroke-m "${RECIPROCATING_STROKE_M:-0.018}" \
  --reciprocating-hold-s "${RECIPROCATING_HOLD_S:-0.35}" \
  --a-x-max "${A_X_MAX:-0.03}" \
  --v-x-max "${V_X_MAX:-0.018}" \
  --settle-duration "${SETTLE_DURATION:-2.0}" \
  --duration "${DURATION:-0}" \
  --no-video \
  "$@"
