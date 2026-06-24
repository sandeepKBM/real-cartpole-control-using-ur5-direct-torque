#!/usr/bin/env bash
# Fast X transport video via CoppeliaSim offscreen PNG capture (reliable in container).
set -euo pipefail

ROOT="/common/users/ss5772/real_Cartpole"
CONFIG="${ROOT}/ros2_ws/src/ur5_x_axis_controller_ros/config/controller_coppelia_fast_x.yaml"
RUN_SUFFIX="${RUN_SUFFIX:-coppelia_fast_x_video}"
VIDEO_PATH="${VIDEO_PATH:-${ROOT}/demonstration_videos/ur5e_coppeliasim/coppelia_fast_x_transport.mp4}"
FRAME_COUNT="${FRAME_COUNT:-160}"
CAPTURE_SKIP_FRAMES="${CAPTURE_SKIP_FRAMES:-2}"
FPS="${FPS:-25}"

export RUN_SUFFIX
export VIDEO_PATH
export FRAME_COUNT
export CAPTURE_SKIP_FRAMES
export FPS
export RUNNER_EXTRA_ARGS="--config ${CONFIG} --accel-x-transport --accel-profile fast_x --accel-torque-policy ${ACCEL_TORQUE_POLICY:-ik_joint_pd} --transport-axis x --target-dx ${TARGET_DX_M:-0.03} --settle-duration ${SETTLE_DURATION:-2.0} --duration ${DURATION:-0} --spawn-coppelia-pendulum"

bash "${ROOT}/simulation/launch_coppeliasim_x_axis_offscreen_capture.sh" "$@"
capture_exit=$?

trace_jsonl="${ROOT}/outputs/control_runs/${RUN_SUFFIX}_coppeliasim_x_axis_offscreen_capture.jsonl"
replay_video="${VIDEO_PATH}"
if [[ -f "${trace_jsonl}" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-python3}"
  FFMPEG_BIN="${FFMPEG_BIN:-/usr/bin/ffmpeg}"
  export FFMPEG_BIN
  "${PYTHON_BIN}" "${ROOT}/simulation/render_coppelia_trace_mujoco_mp4.py" \
    "${trace_jsonl}" \
    --out "${replay_video}" \
    --fps "${FPS}" || true
fi
exit "${capture_exit}"
