#!/usr/bin/env bash
# Pre-flight verification for real UR5e / URSim before lab day.
#
# Usage:
#   conda activate mujoco_ur5e
#   bash tools/ur5e_preflight.sh --robot-ip <IP>
#
# Optional flags:
#   --skip-tests          Skip offline pytest -m hardware
#   --skip-watch          Skip live --watch liveness loop (runs --once only)
#   --motion-axis y       Axis for dry-run + tiny move (default: y)
#   --motion-direction left
#   --tiny-distance-m 0.02
#
# URSim Docker (typical; not tracked in this repo):
#   docker run -d --name ursim -p 5900:5900 -p 6080:6080 universalrobots/ursim_e-series
#   # Then power on the simulated robot in the URSim UI and note its IP (often 172.16.71.77).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ROBOT_IP=""
SKIP_TESTS=0
SKIP_WATCH=0
AXIS="y"
DIRECTION="left"
TINY_DIST="0.02"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --robot-ip) ROBOT_IP="${2:-}"; shift 2 ;;
    --skip-tests) SKIP_TESTS=1; shift ;;
    --skip-watch) SKIP_WATCH=1; shift ;;
    --motion-axis) AXIS="${2:-}"; shift 2 ;;
    --motion-direction) DIRECTION="${2:-}"; shift 2 ;;
    --tiny-distance-m) TINY_DIST="${2:-}"; shift 2 ;;
    -h|--help)
      sed -n '2,20p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$ROBOT_IP" ]]; then
  echo "ERROR: --robot-ip is required (no default)." >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
echo "=== UR5e preflight ==="
echo "  repo:      $ROOT"
echo "  python:    $($PYTHON_BIN --version 2>&1)"
echo "  robot_ip:  $ROBOT_IP"
echo "  axis:      $AXIS ($DIRECTION)"
echo ""

step() { echo ""; echo ">>> $*"; }

if [[ "$SKIP_TESTS" -eq 0 ]]; then
  step "1/5 Offline hardware tests (mocked RTDE)"
  $PYTHON_BIN -m pytest -m hardware -q
fi

step "2/5 Receive-only connect (--once)"
$PYTHON_BIN tools/ur5e_connect.py --robot-ip "$ROBOT_IP" --once

if [[ "$SKIP_WATCH" -eq 0 ]]; then
  step "3/5 Liveness watch (~5 s, then Ctrl-C if still running)"
  timeout 5s $PYTHON_BIN tools/ur5e_connect.py --robot-ip "$ROBOT_IP" --watch --frequency-hz 125 \
    || [[ $? -eq 124 ]]  # timeout exit
else
  echo ">>> 3/5 Skipped --watch"
fi

step "4/5 Motion dry-run (no network)"
$PYTHON_BIN tools/ur5e_move.py \
  --robot-ip "$ROBOT_IP" \
  --axis "$AXIS" \
  --direction "$DIRECTION" \
  --distance-m "$TINY_DIST" \
  --dry-run

step "5/5 Tiny real move (requires typed MOVE)"
echo "If this is URSim, power on the robot in the UI first."
$PYTHON_BIN tools/ur5e_move.py \
  --robot-ip "$ROBOT_IP" \
  --axis "$AXIS" \
  --direction "$DIRECTION" \
  --distance-m "$TINY_DIST" \
  --i-understand-this-moves-the-robot

echo ""
echo "=== PREFLIGHT PASS ==="
echo "Next: confirm physical direction matches intent, then try --distance-m 0.15"
