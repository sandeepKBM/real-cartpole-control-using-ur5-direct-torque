#!/usr/bin/env bash
set -euo pipefail

ROOT="/common/users/ss5772/real_Cartpole"
SIF="${REAL_CARTPOLE_SIF:-/common/users/ss5772/containers/aha_u2404.sif}"

if [[ ! -f "${SIF}" ]]; then
  echo "Missing Singularity image: ${SIF}" >&2
  exit 1
fi
if ! command -v singularity >/dev/null 2>&1; then
  echo "Missing singularity binary" >&2
  exit 1
fi
if [[ ! -d /usr/share/X11/xkb ]]; then
  echo "Missing host XKB tree: /usr/share/X11/xkb" >&2
  exit 1
fi
if [[ ! -d /common/home/ss5772/.tmp/container_bind_libs ]]; then
  echo "Missing container library dir: /common/home/ss5772/.tmp/container_bind_libs" >&2
  exit 1
fi

exec singularity exec \
  --bind "${ROOT}:${ROOT}" \
  --bind /common/home/ss5772/.tmp:/common/home/ss5772/.tmp \
  --bind /common/users/ss5772/miniforge3:/common/users/ss5772/miniforge3 \
  --bind /usr/bin/xkbcomp:/usr/bin/xkbcomp \
  --bind /usr/share/X11/xkb:/usr/lib/X11/xkb \
  --bind /usr/share/X11/xkb:/usr/share/X11/xkb \
  --env Q_START_RAD="${Q_START_RAD:-}" \
  "${SIF}" \
  bash -lc 'export PATH=/common/users/ss5772/miniforge3/bin:/usr/bin:/bin${PATH:+:$PATH}; export PYTHON_BIN=/common/users/ss5772/miniforge3/bin/python3; export FFMPEG_BIN=/common/home/ss5772/.tmp/hostffmpeg/ffmpeg; export LD_LIBRARY_PATH=/common/home/ss5772/.tmp/container_bind_libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; export Q_START_RAD="${Q_START_RAD:-}"; cd /common/users/ss5772/real_Cartpole && exec bash simulation/launch_coppeliasim_x_axis_offscreen_capture.sh'
