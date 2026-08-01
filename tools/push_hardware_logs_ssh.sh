#!/usr/bin/env bash
# Companion to tools/pull_hardware_logs_ssh.sh: run this ONE on thinkrobot
# (the machine producing real-hardware logs) to push new run artifacts UP to
# westeros over SSH. Same rsync/no-cloud approach as the puller -- pick
# whichever direction has an SSH connection that actually works for you;
# you do not need both scripts running, just whichever direction is reachable.
#
# Pulls only small text artifacts (run_record.json, summary.json,
# trace.jsonl / supervisor_trace.jsonl, run_log.jsonl/csv), not video or
# other large binaries -- same include patterns as the puller.
#
# Usage (run from thinkrobot):
#   WESTEROS_HOST=westeros WESTEROS_USER=ss5772 \
#     ./tools/push_hardware_logs_ssh.sh
#
# Optional:
#   LOCAL_SUBDIRS      - space-separated output subdirs to push
#                         (default: "outputs/hardware_transport outputs/hardware_urscript")
#   REMOTE_REPO_DIR     - path to the repo on westeros
#                         (default: /common/users/ss5772/real_Cartpole)
#   REMOTE_DEST_DIR     - destination root under REMOTE_REPO_DIR on westeros
#                         (default: outputs/hardware_transport_remote)
#   DRY_RUN=1           - preview what would be transferred without copying
set -euo pipefail

WESTEROS_HOST="${WESTEROS_HOST:?set WESTEROS_HOST to the westeros hostname or IP on the subnet}"
WESTEROS_USER="${WESTEROS_USER:?set WESTEROS_USER to your SSH username on westeros}"
LOCAL_SUBDIRS="${LOCAL_SUBDIRS:-outputs/hardware_transport outputs/hardware_urscript}"
REMOTE_REPO_DIR="${REMOTE_REPO_DIR:-/common/users/ss5772/real_Cartpole}"
REMOTE_DEST_DIR="${REMOTE_DEST_DIR:-outputs/hardware_transport_remote}"

if ! command -v rsync >/dev/null 2>&1; then
  echo "rsync not found on PATH -- install it first." >&2
  exit 2
fi

RSYNC_FLAGS=(-avz --prune-empty-dirs --include='*/' --include='*.json' --include='*.jsonl' --exclude='*')
if [ "${DRY_RUN:-0}" = "1" ]; then
  RSYNC_FLAGS+=(--dry-run -n)
fi

for subdir in ${LOCAL_SUBDIRS}; do
  if [ ! -d "${subdir}" ]; then
    echo "[push_hardware_logs] ${subdir} does not exist locally -- skipping"
    continue
  fi
  dest_subdir="${REMOTE_DEST_DIR}/$(basename "${subdir}")"
  echo "[push_hardware_logs] ${subdir}/ -> ${WESTEROS_USER}@${WESTEROS_HOST}:${REMOTE_REPO_DIR}/${dest_subdir}/"
  ssh "${WESTEROS_USER}@${WESTEROS_HOST}" "mkdir -p '${REMOTE_REPO_DIR}/${dest_subdir}'"
  rsync "${RSYNC_FLAGS[@]}" \
    "${subdir}/" \
    "${WESTEROS_USER}@${WESTEROS_HOST}:${REMOTE_REPO_DIR}/${dest_subdir}/"
done
