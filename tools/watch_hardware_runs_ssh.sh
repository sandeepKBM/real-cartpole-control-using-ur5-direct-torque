#!/usr/bin/env bash
# Watch thinkrobot for a NEW real-hardware run directory to appear, then pull
# just that one directory automatically -- companion to
# tools/pull_hardware_logs_ssh.sh (bulk pull) for the common case: "I'm about
# to ask the user to run one real command, and I want the result pulled the
# moment it's written, without asking them to paste JSON or tell me the
# timestamp."
#
# Usage (run from westeros, BEFORE telling the user to run their command, in
# the background -- it blocks until it finds something new or times out):
#   HARDWARE_HOST=<ip> HARDWARE_USER=<user> ./tools/watch_hardware_runs_ssh.sh
#
# Snapshots the remote subdir's current run directories first, then polls for
# a name that didn't exist in the snapshot. Only pulls once that new
# directory has BOTH summary.json and trace.jsonl present (a run still being
# written won't have summary.json yet -- that's written last), so a half-
# written run is never pulled prematurely.
#
# Prints the local path of the newly-pulled run directory on success and
# exits 0. Exits 1 on timeout with nothing new. Exits 2 on a real error
# (bad SSH, bad remote path).
set -euo pipefail

HARDWARE_HOST="${HARDWARE_HOST:?set HARDWARE_HOST to the thinkrobot hostname or IP on the subnet}"
HARDWARE_USER="${HARDWARE_USER:?set HARDWARE_USER to the SSH username on thinkrobot}"
REMOTE_REPO_DIR="${REMOTE_REPO_DIR:-~/real-cartpole-control-using-ur5-direct-torque}"
WATCH_SUBDIR="${WATCH_SUBDIR:-outputs/hardware_transport}"
LOCAL_DIR="${LOCAL_DIR:-outputs/hardware_transport_remote/$(basename "${WATCH_SUBDIR}")}"
POLL_INTERVAL_S="${POLL_INTERVAL_S:-3}"
TIMEOUT_S="${TIMEOUT_S:-300}"

ssh_ls() {
  ssh "${HARDWARE_USER}@${HARDWARE_HOST}" \
    "ls -1 ${REMOTE_REPO_DIR}/${WATCH_SUBDIR}/ 2>/dev/null || true"
}

echo "[watch_hardware_runs] snapshotting existing runs under ${WATCH_SUBDIR}/ on ${HARDWARE_HOST}..."
before="$(ssh_ls)"

echo "[watch_hardware_runs] watching for a new run directory (poll every ${POLL_INTERVAL_S}s, timeout ${TIMEOUT_S}s)..."
elapsed=0
new_dir=""
while [ "${elapsed}" -lt "${TIMEOUT_S}" ]; do
  now="$(ssh_ls)"
  # Directories present now but not in the original snapshot.
  candidate="$(comm -13 <(echo "${before}" | sort) <(echo "${now}" | sort) | head -1)"
  if [ -n "${candidate}" ]; then
    # Confirm it's actually complete (summary.json written last).
    has_summary="$(ssh "${HARDWARE_USER}@${HARDWARE_HOST}" \
      "test -f ${REMOTE_REPO_DIR}/${WATCH_SUBDIR}/${candidate}/summary.json && echo yes || echo no")"
    if [ "${has_summary}" = "yes" ]; then
      new_dir="${candidate}"
      break
    fi
  fi
  sleep "${POLL_INTERVAL_S}"
  elapsed=$((elapsed + POLL_INTERVAL_S))
done

if [ -z "${new_dir}" ]; then
  echo "[watch_hardware_runs] timed out after ${TIMEOUT_S}s with no new complete run." >&2
  exit 1
fi

echo "[watch_hardware_runs] found new run: ${new_dir} -- pulling..."
mkdir -p "${LOCAL_DIR}"
rsync -avz \
  "${HARDWARE_USER}@${HARDWARE_HOST}:${REMOTE_REPO_DIR}/${WATCH_SUBDIR}/${new_dir}/" \
  "${LOCAL_DIR}/${new_dir}/"

echo "[watch_hardware_runs] pulled to: ${LOCAL_DIR}/${new_dir}"
