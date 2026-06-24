#!/usr/bin/env bash
# CoppeliaSim fast X transport with EE-following video capture.
set -euo pipefail

ROOT="/common/users/ss5772/real_Cartpole"
CONFIG="${ROOT}/ros2_ws/src/ur5_x_axis_controller_ros/config/controller_coppelia_fast_x.yaml"
PORT="${PORT:-23280}"
RUN_SUFFIX="${RUN_SUFFIX:-coppelia_fast_x_video}"
VIDEO_NAME="${VIDEO_NAME:-coppelia_fast_x_transport.mp4}"

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
  --video-name "${VIDEO_NAME}" \
  --video-camera ee \
  --fps "${FPS:-25}" \
  --width "${WIDTH:-960}" \
  --height "${HEIGHT:-540}" \
  "$@"
