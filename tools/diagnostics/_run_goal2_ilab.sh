#!/usr/bin/env bash
# Goal 2 swing-up: ARM_Q0 (wrist_2 ~ 0 deg, the REAL hardware pose) with the
# default local-Z-hinge pendulum, driven by the SINGLE-AXIS REDUCED-TASK
# controller (x_task_yz_corridor_qp: tracks X + orientation, Y/Z as HOCBF
# corridors) rather than plain OSC, which is structurally mismatched at
# cond(J)=1396.
#
# Exists as a file rather than an inline ssh string because the remote shell
# starts in $HOME, not the repo, and repeated inline attempts kept losing the
# `cd`. Keeping it here also makes the exact invocation reviewable.
set -euo pipefail

REPO=/common/users/ss5772/real_Cartpole
PY=/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python
OUT=/common/home/ss5772/.claude/jobs/d0ed049c/tmp/goal2/singleaxis_reduced_armq0.json

cd "$REPO"
mkdir -p "$(dirname "$OUT")"

# One worker per process; parallelism comes from DE's worker pool, not from
# each process spawning its own full-core BLAS thread pool.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

echo "host=$(hostname -s) cwd=$(pwd) load=$(uptime | sed 's/.*average: //')"

exec "$PY" tools/diagnostics/pendulum_swingup_energy_shaping.py \
  --pendulum-xml assets/ur5e_pendulum/pendulum_attachment.xml \
  --controller-kind x_task_yz_corridor_qp \
  --config config/ur5e_mujoco_torque_x_task_yz_corridor_qp_enabled.yaml \
  --a-max-upper 12.0 \
  --maxiter 40 --popsize 16 --seed 2 \
  --duration-s 10.0 --final-duration-s 14.0 \
  --output-json "$OUT"
