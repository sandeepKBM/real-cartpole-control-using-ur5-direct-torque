#!/usr/bin/env bash
set -euo pipefail
C="${HOME}/coppelia_runtime/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu22_04"
cd "${C}"
export LD_LIBRARY_PATH="${C}:${LD_LIBRARY_PATH:-}"

echo "=== Display env ==="
echo "DISPLAY=${DISPLAY:-unset}"
echo "WAYLAND_DISPLAY=${WAYLAND_DISPLAY:-unset}"

echo "=== Missing xcb plugin deps ==="
PLUGIN="$(find "${C}" -name 'libqxcb.so' 2>/dev/null | head -1 || true)"
if [[ -n "${PLUGIN}" ]]; then
  ldd "${PLUGIN}" 2>&1 | grep -i 'not found' || echo "(none missing for libqxcb.so)"
else
  echo "libqxcb.so not found"
fi

echo "=== Test 1: DISPLAY=:0 xcb ==="
export DISPLAY="${DISPLAY:-:0}"
unset QT_QPA_PLATFORM
./coppeliaSim.sh -GzmqRemoteApi.rpcPort=23094 -GzmqRemoteApi.cntPort=23093 > /tmp/coppelia_xcb.log 2>&1 &
PID=$!
sleep 12
if kill -0 "${PID}" 2>/dev/null; then
  echo "xcb: STILL RUNNING pid=${PID}"
  ss -ltn | grep 23094 || true
  kill "${PID}" 2>/dev/null || true
else
  echo "xcb: EXITED"
  tail -8 /tmp/coppelia_xcb.log
fi

echo "=== Test 2: Wayland ==="
export QT_QPA_PLATFORM=wayland
export WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-wayland-0}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)/}"
./coppeliaSim.sh -GzmqRemoteApi.rpcPort=23092 -GzmqRemoteApi.cntPort=23091 > /tmp/coppelia_wl.log 2>&1 &
PID=$!
sleep 12
if kill -0 "${PID}" 2>/dev/null; then
  echo "wayland: STILL RUNNING pid=${PID}"
  ss -ltn | grep 23092 || true
  kill "${PID}" 2>/dev/null || true
else
  echo "wayland: EXITED"
  tail -8 /tmp/coppelia_wl.log
fi
