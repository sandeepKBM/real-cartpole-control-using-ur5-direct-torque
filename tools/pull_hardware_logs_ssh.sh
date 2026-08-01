#!/usr/bin/env bash
# On-demand rsync-over-SSH pull of real-hardware run artifacts from
# thinkrobot to this machine (westeros). Replaces an earlier untested
# rclone+Box-OAuth pair of scripts (2026-07-31) with the transfer path
# already proven working tonight (thinkrobot and westeros are on the
# same subnet, scp already works between them) -- no cloud account, no
# OAuth, nothing to configure beyond SSH access you already have.
#
# Pulls only small text artifacts (run_record.json, summary.json,
# trace.jsonl / supervisor_trace.jsonl, run_log.jsonl/csv), not video or
# other large binaries.
#
# Usage (run from westeros):
#   HARDWARE_HOST=thinkrobot HARDWARE_USER=ros2humble \
#     ./tools/pull_hardware_logs_ssh.sh
#
# If SSH only works in the other direction (westeros unreachable from
# the thinkrobot side, or vice versa), run the equivalent rsync command from
# thinkrobot instead, with source/dest swapped -- same flags, same include
# patterns, just reversed.
#
# Optional:
#   REMOTE_REPO_DIR   - path to the repo on thinkrobot
#                        (default: ~/real-cartpole-control-using-ur5-direct-torque)
#   REMOTE_SUBDIRS    - space-separated output subdirs to pull
#                        (default: "outputs/hardware_transport outputs/hardware_urscript")
#   LOCAL_DIR         - local destination root (default: outputs/hardware_transport_remote)
#   DRY_RUN=1         - preview what would be transferred without copying
set -euo pipefail

HARDWARE_HOST="${HARDWARE_HOST:?set HARDWARE_HOST to the thinkrobot hostname or IP on the subnet}"
HARDWARE_USER="${HARDWARE_USER:?set HARDWARE_USER to the SSH username on thinkrobot}"
REMOTE_REPO_DIR="${REMOTE_REPO_DIR:-~/real-cartpole-control-using-ur5-direct-torque}"
REMOTE_SUBDIRS="${REMOTE_SUBDIRS:-outputs/hardware_transport outputs/hardware_urscript}"
LOCAL_DIR="${LOCAL_DIR:-outputs/hardware_transport_remote}"

if ! command -v rsync >/dev/null 2>&1; then
  echo "rsync not found on PATH -- install it first." >&2
  exit 2
fi

RSYNC_FLAGS=(-avz --prune-empty-dirs --include='*/' --include='*.json' --include='*.jsonl' --exclude='*')
if [ "${DRY_RUN:-0}" = "1" ]; then
  RSYNC_FLAGS+=(--dry-run -n)
fi

mkdir -p "${LOCAL_DIR}"
for subdir in ${REMOTE_SUBDIRS}; do
  dest="${LOCAL_DIR}/$(basename "${subdir}")"
  mkdir -p "${dest}"
  echo "[pull_hardware_logs] ${HARDWARE_USER}@${HARDWARE_HOST}:${REMOTE_REPO_DIR}/${subdir}/ -> ${dest}/"
  rsync "${RSYNC_FLAGS[@]}" \
    "${HARDWARE_USER}@${HARDWARE_HOST}:${REMOTE_REPO_DIR}/${subdir}/" \
    "${dest}/"
done
