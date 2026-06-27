#!/usr/bin/env bash
set -euo pipefail
C="${COPPELIA_ROOT:-/home/kbm/coppelia_runtime/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu22_04}"
cd "${C}"
export LD_LIBRARY_PATH="${C}:${LD_LIBRARY_PATH:-}"

check_plugin() {
  local name="$1"
  local plugin
  plugin="$(find "${C}" -name "${name}" 2>/dev/null | head -1 || true)"
  if [[ -z "${plugin}" ]]; then
    echo "=== ${name}: NOT FOUND ==="
    return 0
  fi
  echo "=== ${name} ==="
  echo "${plugin}"
  ldd "${plugin}" 2>&1 | grep 'not found' || echo "(no missing libs)"
  echo ""
}

check_plugin libqwayland.so
check_plugin libqxcb.so

echo "=== Try launch modes (8s each) ==="
export DISPLAY="${DISPLAY:-:0}"
export WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-wayland-0}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"

for mode in wayland xcb offscreen; do
  echo "--- QT_QPA_PLATFORM=${mode} ---"
  rm -f /tmp/coppelia_mode_test.log
  if [[ "${mode}" == "xcb" ]]; then
    unset QT_QPA_PLATFORM
    ./coppeliaSim.sh -GzmqRemoteApi.rpcPort=23990 > /tmp/coppelia_mode_test.log 2>&1 &
  else
    QT_QPA_PLATFORM="${mode}" ./coppeliaSim.sh -GzmqRemoteApi.rpcPort=23990 > /tmp/coppelia_mode_test.log 2>&1 &
  fi
  pid=$!
  sleep 8
  if kill -0 "${pid}" 2>/dev/null; then
    echo "OK: still running (pid=${pid})"
    kill "${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
  else
    echo "FAIL"
    tail -6 /tmp/coppelia_mode_test.log
  fi
  pkill -f 'zmqRemoteApi.rpcPort=23990' 2>/dev/null || true
  sleep 1
done

echo ""
echo "=== Recommended apt install ==="
echo "sudo apt-get update && sudo apt-get install -y \\"
echo "  libxkbcommon0 libxcb-cursor0 libxcb-icccm4 libxcb-image0 \\"
echo "  libxcb-keysyms1 libxcb-render-util0 libxcb-xinerama0 libxcb-xfixes0 \\"
echo "  libwayland-client0 libwayland-cursor0 libwayland-egl1 \\"
echo "  libfontconfig1 libdbus-1-3 libnss3 libasound2 x11-utils xvfb"
