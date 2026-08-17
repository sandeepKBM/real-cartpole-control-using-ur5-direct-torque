#!/usr/bin/env bash
# GOAL 2, IN-PLANE task rotation. See the config header for the measured basis.
#
# DE_WORKERS is bounded EXPLICITLY. The unset default is cpu_count*0.9 (~87 on a
# 96-core ilab node) and both prior Goal 2 attempts DEADLOCKED at DE start with
# that: 174 procs alive across two launches, ~14 min CPU each over 14-37 h
# elapsed, 2 runnable vs 1216 sleeping, no output past the search banner. Memory
# was NOT the cause (81.4 GB cgroup cap vs ~30 GB used), so worker count is the
# remaining suspect. 32 leaves headroom on a shared teaching machine either way.
#
# LAUNCH NOTE: invoke over SSH WITH KEEPALIVES --
#   ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=10 -o TCPKeepAlive=yes ...
# A previous run of this exact script died to "client_loop: send disconnect:
# Broken pipe" ~2 min in: the remote job is a FOREGROUND child of the ssh
# session (deliberately, since AGENTS.md 8 documents nohup+disown being
# unreliable on these hosts when logind Linger is off), so a transient network
# drop kills it. Keepalives cover the drop without giving up that property.
set -euo pipefail
cd /common/users/ss5772/real_Cartpole
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export DE_WORKERS="${DE_WORKERS:-32}"
OUT="${OUT_JSON:-/common/home/ss5772/.claude/jobs/3a0b9f03/tmp/goal2_inplane_signfixed.json}"
mkdir -p "$(dirname "$OUT")"
echo "host=$(hostname -s) cores=$(nproc) load=$(uptime|sed 's/.*average: //') DE_WORKERS=$DE_WORKERS"
exec /common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python \
  tools/diagnostics/pendulum_swingup_energy_shaping.py \
  --pendulum-xml assets/ur5e_pendulum/pendulum_attachment.xml \
  --start-q-rad -2.3688 -2.1801 -1.8838 -0.7962 0.004714693 0.0206 \
  --controller-kind x_task_yz_corridor_qp \
  --config config/ur5e_mujoco_torque_x_task_inplane_goal2.yaml \
  --a-max-upper 12.0 --maxiter 60 --popsize 24 --seed 2 \
  --duration-s 10.0 --final-duration-s 14.0 \
  --output-json "$OUT"
