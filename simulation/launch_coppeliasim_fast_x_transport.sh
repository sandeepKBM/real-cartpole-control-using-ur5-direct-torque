#!/usr/bin/env bash
# CoppeliaSim fast X transport: auto-computed speed under joint/safety limits.
set -euo pipefail

ROOT="/common/users/ss5772/real_Cartpole"
CONFIG="${ROOT}/ros2_ws/src/ur5_x_axis_controller_ros/config/controller_coppelia_fast_x.yaml"
PORT="${PORT:-23280}"
RUN_SUFFIX="${RUN_SUFFIX:-coppelia_fast_x}"

exec bash "${ROOT}/simulation/launch_coppeliasim_x_axis_headless.sh" \
  --config "${CONFIG}" \
  --accel-x-transport \
  --accel-profile fast_x \
  --accel-torque-policy "${ACCEL_TORQUE_POLICY:-ik_joint_pd}" \
  --transport-axis x \
  --target-dx "${TARGET_DX_M:-0.03}" \
  --settle-duration "${SETTLE_DURATION:-2.0}" \
  --duration "${DURATION:-0}" \
  --spawn-coppelia-pendulum \
  --no-video \
  "$@"
