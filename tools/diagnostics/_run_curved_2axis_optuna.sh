#!/usr/bin/env bash
# CURVED 2-D pump + 2-AXIS controller + TPE search, at the singular ARM_Q0.
# Best configuration available: both drive channels perpendicular to the hinge
# (in-plane horizontal + vertical), the guard resolving in the same frame the
# controller drives in, and the energy sign measured for this pose.
# SSH keepalives required (foreground child of the ssh session).
set -euo pipefail
cd /common/users/ss5772/real_Cartpole
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
W="${CURVED_WORKERS:-48}"
N="${N_TRIALS:-1200}"
OUT="${OUT_JSON:-/common/home/ss5772/.claude/jobs/3a0b9f03/tmp/curved_2axis_optuna.json}"
mkdir -p "$(dirname "$OUT")"
echo "host=$(hostname -s) cores=$(nproc) load=$(uptime|sed 's/.*average: //') workers=$W trials=$N"
exec /common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python \
  tools/diagnostics/pendulum_swingup_curved.py \
  --pendulum-xml assets/ur5e_pendulum/pendulum_attachment.xml \
  --start-q-rad -2.3688 -2.1801 -1.8838 -0.7962 0.004714693 0.0206 \
  --controller-kind x_task_yz_corridor_qp \
  --config config/ur5e_mujoco_torque_x_task_inplane_goal2.yaml \
  --search-backend optuna --n-trials "$N" \
  --a-max-upper 12.0 --kick-amplitude-upper 0.15 --seed 7 \
  --duration-s 12.0 --final-duration-s 16.0 --workers "$W" \
  --output-json "$OUT"
