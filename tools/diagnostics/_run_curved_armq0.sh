#!/usr/bin/env bash
# CURVED 2-D pump at the SINGULAR ARM_Q0 -- the pose this lane has never run at.
# Sign is now measured per (pose, asset, direction); see pendulum_swingup_curved.
# SSH keepalives required (foreground child of the ssh session, see the Goal 2
# launcher for why a broken pipe kills it).
set -euo pipefail
cd /common/users/ss5772/real_Cartpole
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
W="${CURVED_WORKERS:-48}"
OUT="${OUT_JSON:-/common/home/ss5772/.claude/jobs/3a0b9f03/tmp/curved_armq0_signfixed.json}"
mkdir -p "$(dirname "$OUT")"
echo "host=$(hostname -s) cores=$(nproc) load=$(uptime|sed 's/.*average: //') workers=$W"
exec /common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python \
  tools/diagnostics/pendulum_swingup_curved.py \
  --pendulum-xml assets/ur5e_pendulum/pendulum_attachment.xml \
  --start-q-rad -2.3688 -2.1801 -1.8838 -0.7962 0.004714693 0.0206 \
  --a-max-upper 12.0 --maxiter 50 --popsize 20 --seed 3 \
  --duration-s 10.0 --final-duration-s 14.0 --workers "$W" \
  --output-json "$OUT"
