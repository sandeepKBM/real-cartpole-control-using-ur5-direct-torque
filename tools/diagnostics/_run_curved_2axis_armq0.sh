#!/usr/bin/env bash
# CURVED 2-D pump THROUGH THE 2-AXIS CONTROLLER at the singular ARM_Q0.
# The missing cell: ilab4 runs the 2-axis controller with a SINGLE-axis pump,
# ilab3 runs a curved pump through plain OSC. This is both at once -- the
# in-plane config tracks rows (0,2), the horizontal drive and the vertical,
# both perpendicular to the hinge. SSH keepalives required (see _run_goal2_inplane).
set -euo pipefail
cd /common/users/ss5772/real_Cartpole
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
W="${CURVED_WORKERS:-40}"
OUT="${OUT_JSON:-/common/home/ss5772/.claude/jobs/3a0b9f03/tmp/curved_2axis_armq0.json}"
mkdir -p "$(dirname "$OUT")"
echo "host=$(hostname -s) cores=$(nproc) load=$(uptime|sed 's/.*average: //') workers=$W"
exec /common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python \
  tools/diagnostics/pendulum_swingup_curved.py \
  --pendulum-xml assets/ur5e_pendulum/pendulum_attachment.xml \
  --start-q-rad -2.3688 -2.1801 -1.8838 -0.7962 0.004714693 0.0206 \
  --controller-kind x_task_yz_corridor_qp \
  --config config/ur5e_mujoco_torque_x_task_inplane_goal2.yaml \
  --a-max-upper 12.0 --maxiter 50 --popsize 20 --seed 4 \
  --duration-s 10.0 --final-duration-s 14.0 --workers "$W" \
  --output-json "$OUT"
