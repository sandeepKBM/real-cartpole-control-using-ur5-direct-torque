#!/bin/bash
# Reproduce the single-axis flip-AND-hold at the SINGULAR ARM_Q0 (cond(J)=1396),
# guards ON. velocity-swingup -> torque-LQR-hold, one drive axis (in-plane
# horizontal). Achieved 2026-08-24; see tests/mujoco/test_pendulum_singular_flip_hold.py.
#
# The three ingredients (all in the command below):
#   1. zhold config: world-Z is a tracked task axis (kp_z=400*Lambda_zz), else
#      the horizontal drive sags the EE into the floor guard in 0.38 s.
#   2. --velocity-hold: the catch stays in velocity tracking; position tracking
#      lag mis-phases the balancing cart-accel and PUMPS the pole.
#   3. goal2_singular_flip_hold_lqr.json: the cascade 4-state LQR with phi/phidot
#      gains x1.5 (the cascade tuned on placed-inverted under-weights the pole
#      for the harder swing-up arrival; x1.5 crosses into a sustained hold,
#      x3 overshoots and falls -- a narrow window).
#
# Result (measured): engages at t=2.67 s (phi=17.1 deg, thetadot=-2.62), holds
# max|phi|=0.299 rad for the full 10 s, guard_fired=False, tip clears floor by
# 3.6 cm. K=0 from the identical arrival falls to pi -- the hold is control, not
# hinge friction.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$HERE"
export PYTHONPATH="$HERE"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
PY="${PY:-python}"
OUT="${1:-outputs/pendulum_renders/goal2_singular_flip_hold.json}"
mkdir -p "$(dirname "$OUT")"

# The validated swing-up schedule (from the hold-scoring search):
#   a_slow a_sharp e_center e_width db_slow db_sharp e_target
SCHED="5.7958928681177 18.883718056153565 0.35043620661279723 0.31761609998870677 0.1584433121534668 0.09906939841856013 0.9978187516750392"

exec "$PY" -u tools/diagnostics/pendulum_two_phase_swingup.py \
  --pendulum-xml assets/ur5e_pendulum/pendulum_attachment.xml \
  --start-q-rad -2.3688 -2.1801 -1.8838 -0.7962 0.004714693 0.0206 \
  --config config/ur5e_mujoco_torque_x_task_yz_corridor_qp_goal2_singular_zhold.yaml \
  --controller-kind x_task_yz_corridor_qp --transport-axis-index 0 \
  --velocity-swingup --velocity-hold --duration-s 14.0 \
  --evaluate $SCHED \
  --lqr-json tools/diagnostics/goal2_singular_flip_hold_lqr.json \
  --s-switch 1.2 --phi-switch-max-rad 0.30 --hold-s 10.0 \
  --output-json "$OUT"
