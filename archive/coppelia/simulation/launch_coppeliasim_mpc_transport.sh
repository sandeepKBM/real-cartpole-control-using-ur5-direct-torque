#!/usr/bin/env bash
# CoppeliaSim cart-pole MPC transport (MuJoCo pole observer + IK joint PD inner loop).
set -euo pipefail

ROOT="/common/users/ss5772/real_Cartpole"
CONFIG="${ROOT}/ros2_ws/src/ur5_x_axis_controller_ros/config/controller_coppelia_mpc.yaml"
PORT="${PORT:-23300}"
RUN_SUFFIX="${RUN_SUFFIX:-coppelia_mpc}"

exec bash "${ROOT}/simulation/launch_coppeliasim_x_axis_headless.sh" \
  --config "${CONFIG}" \
  --accel-x-transport \
  --accel-profile mpc \
  --accel-torque-policy ik_joint_pd \
  --transport-axis x \
  --target-dx "${TARGET_DX:-0.02}" \
  --a-x-max "${A_X_MAX:-0.08}" \
  --v-x-max "${V_X_MAX:-0.04}" \
  --settle-duration "${SETTLE_DURATION:-2.0}" \
  --duration "${DURATION:-8.0}" \
  --mpc-horizon "${MPC_HORIZON:-20}" \
  --no-video \
  "$@"
