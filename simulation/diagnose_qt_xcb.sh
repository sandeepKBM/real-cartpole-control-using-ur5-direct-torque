#!/usr/bin/env bash
set -euo pipefail
C="${COPPELIA_ROOT:-/home/kbm/coppelia_runtime/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu22_04}"
PLUGIN="$(find "${C}" -name 'libqxcb.so' 2>/dev/null | head -1 || true)"
echo "=== CoppeliaSim Qt xcb plugin ==="
echo "COPPELIA_ROOT=${C}"
if [[ -z "${PLUGIN}" ]]; then
  echo "libqxcb.so not found under ${C}"
  exit 1
fi
echo "Plugin: ${PLUGIN}"
echo ""
echo "=== Missing shared libraries ==="
MISSING="$(ldd "${PLUGIN}" 2>&1 | grep 'not found' || true)"
if [[ -z "${MISSING}" ]]; then
  echo "(none reported by ldd on libqxcb.so)"
else
  echo "${MISSING}"
fi
echo ""
echo "=== Install fix (run once) ==="
echo "sudo apt-get update"
echo "sudo apt-get install -y libxkbcommon0 libxcb-cursor0 libxcb-icccm4 libxcb-image0 \\"
echo "  libxcb-keysyms1 libxcb-render-util0 libxcb-xinerama0 libxcb-xfixes0 \\"
echo "  libfontconfig1 libdbus-1-3 libnss3 libasound2 x11-utils"
