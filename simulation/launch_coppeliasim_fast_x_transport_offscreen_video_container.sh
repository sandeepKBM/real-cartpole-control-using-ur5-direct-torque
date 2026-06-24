#!/usr/bin/env bash
set -euo pipefail

ROOT="/common/users/ss5772/real_Cartpole"
SIF="${REAL_CARTPOLE_SIF:-/common/users/ss5772/containers/aha_u2404.sif}"
INNER_SCRIPT="${ROOT}/outputs/control_runs/_coppelia_fast_x_offscreen_video_container_inner.sh"

mkdir -p "$(dirname "${INNER_SCRIPT}")"
cat >"${INNER_SCRIPT}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export PATH=/common/users/ss5772/miniforge3/bin:/usr/bin:/bin
export PYTHON_BIN=/common/users/ss5772/miniforge3/bin/python3
export FFMPEG_BIN=/common/home/ss5772/.tmp/hostffmpeg/ffmpeg
export LD_LIBRARY_PATH=/common/home/ss5772/.tmp/container_bind_libs:${ROOT}/third_party/coppelia_runtime/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu24_04
export XDG_RUNTIME_DIR=/common/home/ss5772/.tmp/xdg-runtime-\$\$
mkdir -p "\${XDG_RUNTIME_DIR}"
export PORT=${PORT:-23330}
export RUN_SUFFIX=${RUN_SUFFIX:-coppelia_fast_x_video}
export VIDEO_PATH=${VIDEO_PATH:-${ROOT}/demonstration_videos/ur5e_coppeliasim/coppelia_fast_x_transport.mp4}
export TARGET_DX_M=${TARGET_DX_M:-0.03}
export SETTLE_DURATION=${SETTLE_DURATION:-2.0}
export DURATION=${DURATION:-0}
export FRAME_COUNT=${FRAME_COUNT:-160}
export CAPTURE_SKIP_FRAMES=${CAPTURE_SKIP_FRAMES:-2}
export FPS=${FPS:-25}
cd ${ROOT}
exec bash simulation/launch_coppeliasim_fast_x_transport_offscreen_video.sh "\$@"
EOF
chmod +x "${INNER_SCRIPT}"

if [[ ! -f "${SIF}" ]]; then
  echo "Missing Singularity image: ${SIF}" >&2
  exit 1
fi

exec singularity exec \
  --bind "${ROOT}:${ROOT}" \
  --bind /common/home/ss5772/.tmp:/common/home/ss5772/.tmp \
  --bind /common/users/ss5772/miniforge3:/common/users/ss5772/miniforge3 \
  --bind /usr/bin/xkbcomp:/usr/bin/xkbcomp \
  --bind /usr/share/X11/xkb:/usr/lib/X11/xkb \
  --bind /usr/share/X11/xkb:/usr/share/X11/xkb \
  "${SIF}" \
  bash "${INNER_SCRIPT}" "$@"
