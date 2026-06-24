#!/usr/bin/env bash
# Fast X transport: run Coppelia torque control, then build a 3D UR5 replay MP4.
set -euo pipefail

ROOT="/common/users/ss5772/real_Cartpole"
RUN_SUFFIX="${RUN_SUFFIX:-coppelia_fast_x_video}"
VIDEO_PATH="${VIDEO_PATH:-${ROOT}/demonstration_videos/ur5e_coppeliasim/coppelia_fast_x_transport.mp4}"

export RUN_SUFFIX
export VIDEO_PATH
export MUJOCO_GL=egl
export FFMPEG_BIN="${FFMPEG_BIN:-/usr/bin/ffmpeg}"

bash "${ROOT}/simulation/launch_coppeliasim_fast_x_transport_offscreen_video_container.sh" "$@"

trace_jsonl="${ROOT}/outputs/control_runs/${RUN_SUFFIX}_coppeliasim_x_axis_offscreen_capture.jsonl"
if [[ -f "${trace_jsonl}" ]]; then
  python3 "${ROOT}/simulation/render_coppelia_trace_mujoco_mp4.py" \
    "${trace_jsonl}" \
    --out "${VIDEO_PATH}" \
    --fps "${FPS:-25}"
  echo "UR5 replay video: ${VIDEO_PATH}"
fi
