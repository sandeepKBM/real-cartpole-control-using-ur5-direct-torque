#!/usr/bin/env bash
set -euo pipefail
COPPELIA="${HOME}/coppelia_runtime/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu22_04"
cd "${COPPELIA}"
export LD_LIBRARY_PATH="${COPPELIA}:${LD_LIBRARY_PATH:-}"
export QT_QPA_PLATFORM=offscreen
timeout 20 ./coppeliaSim.sh -H -GzmqRemoteApi.rpcPort=23099 2>&1 | head -50
