"""Independently re-measure the Lambda sub-blocks the gain conversion claims.

Claims under test (from config/ur5e_mujoco_torque_x_task_yz_corridor_qp.yaml):
  * Lambda_xx = 5.9298 at ARM_Q0 with lambda_regularization = 0.1
  * Lambda[3:6,3:6] diag = [0.12063, 0.14517, 3.27179]
  * factor = trace(Lambda_rot)/3 = 1.1792
Also reports cond(J), cond(J_reduced) with/without shoulder_pan, and mu(ARM_Q0).
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "/common/users/ss5772/real_Cartpole")

import mujoco

from controller_core.manipulability_cbf import manipulability
from simulation.ur5e_mujoco_torque import load_model, expand_mass_matrix

ARM_Q0 = np.array([-2.3688, -2.1801, -1.8838, -0.7962, 0.004714693, 0.0206])

model, data, site_id, joint_ids, _ = load_model(
    "/common/users/ss5772/real_Cartpole/assets/ur5e_torque/scene.xml"
)
data.qpos[:] = 0.0
data.qvel[:] = 0.0
for i, jid in enumerate(joint_ids):
    data.qpos[int(model.jnt_qposadr[jid])] = float(ARM_Q0[i])
mujoco.mj_forward(model, data)

jacp = np.zeros((3, model.nv))
jacr = np.zeros((3, model.nv))
mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
J = np.vstack([jacp[:, :6], jacr[:, :6]])
M = expand_mass_matrix(model, data)[:6, :6]
Minv = np.linalg.inv(M)

for eps in (0.1,):
    Lam = np.linalg.inv(J @ Minv @ J.T + eps * np.eye(6))
    print(f"eps = {eps}")
    print(f"  Lambda_xx           = {Lam[0,0]:.5f}   (claimed 5.9298)")
    print(f"  Lambda_rot diag     = [{Lam[3,3]:.5f}, {Lam[4,4]:.5f}, {Lam[5,5]:.5f}]"
          f"   (claimed [0.12063, 0.14517, 3.27179])")
    tr3 = float(np.trace(Lam[3:6, 3:6]) / 3.0)
    print(f"  trace(Lambda_rot)/3 = {tr3:.5f}   (claimed 1.1792)")
    ev = np.linalg.eigvalsh(0.5 * (Lam[3:6, 3:6] + Lam[3:6, 3:6].T))
    print(f"  Lambda_rot eigs     = {np.round(ev,5)}  span factor {ev.max()/ev.min():.1f}")

print()
print(f"cond(J)              = {np.linalg.cond(J):.1f}   (config claims ~1396)")
J_red = np.vstack([J[0:1, :], J[3:6, :]])
print(f"cond(J_reduced)      = {np.linalg.cond(J_red):.1f}  (claimed 10.1)")
J_red_nopan = J_red[:, 1:]
print(f"cond(J_red no pan)   = {np.linalg.cond(J_red_nopan):.1f}  (claimed 519), "
      f"rank = {np.linalg.matrix_rank(J_red_nopan)}")
print(f"mu(ARM_Q0)           = {manipulability(J):.6e}  (claimed 4.326e-4)")
print(f"J world-X row        = {np.round(J[0,:],4)}  (claims pan 0.2366, elbow -0.2346)")
print(f"J rz row             = {np.round(J[5,:],4)}  (claims pan rz coeff 1.0)")
