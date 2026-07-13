#!/usr/bin/env bash
# Full factory reset of URSim Docker (no persisted volumes on this setup).
# WARNING: destroys all PolyScope programs/installations inside the container.
set -euo pipefail

NAME="${URSIM_CONTAINER_NAME:-ursim}"
IMAGE="${URSIM_IMAGE:-universalrobots/ursim_e-series}"
ROBOT_TYPE="${UR_ROBOT_TYPE:-UR5}"

echo "Stopping and removing container: ${NAME}"
docker rm -f "${NAME}" 2>/dev/null || true

echo "Starting fresh ${IMAGE} (robot=${ROBOT_TYPE})"
docker run -d --name "${NAME}" \
  -p 5900:5900 -p 6080:6080 -p 29999:29999 \
  -p 30001:30001 -p 30002:30002 -p 30003:30003 -p 30004:30004 \
  -e "UR_ROBOT_TYPE=${ROBOT_TYPE}" \
  "${IMAGE}"

echo "Container started. UI: http://localhost:6080/vnc.html?host=localhost&port=6080"
echo "Wait ~30-90s for PolyScope, then run:"
echo "  python tools/_ursim_wait_and_power_on.py"
