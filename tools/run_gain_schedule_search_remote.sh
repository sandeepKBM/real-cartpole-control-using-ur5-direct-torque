#!/usr/bin/env bash
# Launch the per-cell gain-schedule DE sweep (velocity_gain_tuning/scheduling/search.py)
# on a shared remote host (ilab1-4) with this repo's remote-compute hygiene applied
# (AGENTS.md sec 8): BLAS thread pinning so N worker processes don't each spawn a
# full-core-count thread pool, and everything run in the FOREGROUND so the job does
# not depend on systemd-logind lingering (which is unreliable on these hosts).
#
# Usage (from westeros, wrapping ONE persistent ssh connection):
#   ssh ilab3.cs.rutgers.edu 'bash /common/users/ss5772/real_Cartpole/tools/run_gain_schedule_search_remote.sh <workers> <out.json> [extra args...]'
# NOTE: deliberately NOT `set -u` -- this env's conda activate.d hooks
# (ros-jazzy-ros-workspace_activate.sh) reference unset vars like CONDA_BUILD
# and abort the whole activation under nounset.
set -eo pipefail

REPO=/common/users/ss5772/real_Cartpole
WORKERS="${1:?usage: run_gain_schedule_search_remote.sh <workers> <output-json> [extra args]}"
OUT="${2:?usage: run_gain_schedule_search_remote.sh <workers> <output-json> [extra args]}"
shift 2

source /common/users/ss5772/miniforge3/etc/profile.d/conda.sh
conda activate mujoco_ur5e

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$REPO"
exec python -m velocity_gain_tuning.scheduling.search \
    --workers "$WORKERS" \
    --output-json "$OUT" \
    "$@"
