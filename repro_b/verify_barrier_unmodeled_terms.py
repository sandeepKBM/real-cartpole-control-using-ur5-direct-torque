"""How large are the terms the corridor HOCBF's dynamics model OMITS?

The row assumes  qddot = M^-1 (tau_ctrl - bias),  bias = 0 on the MuJoCo path
(the adapter adds gravity + optional Coriolis itself).  Omitted, in order of
the module docstring's own accounting:
  1. Jdot_a qd            -- named in the docstring
  2. Coriolis C(q,qd)qd   -- named ("standing approximation")
  3. joint friction + viscous damping in the PLANT MODEL -- NOT named anywhere

Each enters hddot as +/- J_a M^-1 (omitted torque), directly comparable to
the b-vector terms the row already carries.
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "/common/users/ss5772/real_Cartpole")

import mujoco

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

jacp = np.zeros((3, model.nv)); jacr = np.zeros((3, model.nv))
mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
J = np.vstack([jacp[:, :6], jacr[:, :6]])
M = expand_mass_matrix(model, data)[:6, :6]
Minv = np.linalg.inv(M)

fl = np.array([model.dof_frictionloss[int(model.jnt_dofadr[j])] for j in joint_ids])
dmp = np.array([model.dof_damping[int(model.jnt_dofadr[j])] for j in joint_ids])
print(f"model frictionloss (Nm)   = {fl}")
print(f"model joint damping       = {dmp}")

a1 = a2 = 10.0
w = 0.05
print(f"\nHOCBF gains a1=a2={a1};  a1*a2*h at corridor CENTRE = {a1*a2*w:.3f} (m/s^2)")
print("(this is the whole 'budget' the b-vector carries at the centre of the corridor)\n")

for axis, name in ((1, "Y"), (2, "Z")):
    lie = J[axis, :] @ Minv
    # worst-case friction contribution: signs chosen adversarially
    fric_worst = float(np.sum(np.abs(lie) * fl))
    print(f"axis {name}:  ||J_{name} M^-1||_1 -weighted friction accel = {fric_worst:.4f} m/s^2 "
          f"(worst-case sign)")
    # viscous damping at a representative |qd| = 1.0 rad/s
    visc = float(np.sum(np.abs(lie) * dmp * 1.0))
    print(f"          viscous damping accel at |qd|=1 rad/s        = {visc:.4f} m/s^2")
    # Coriolis at a representative qd
    qd = np.full(6, 0.5)
    data.qvel[:] = 0.0
    for i, jid in enumerate(joint_ids):
        data.qvel[int(model.jnt_dofadr[jid])] = qd[i]
    mujoco.mj_forward(model, data)
    bias_live = np.asarray(data.qfrc_bias)[:6].copy()
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    bias_static = np.asarray(data.qfrc_bias)[:6].copy()
    cor = bias_live - bias_static
    print(f"          Coriolis accel at qd=0.5 rad/s (all joints)   = "
          f"{float(lie @ cor):+.4f} m/s^2")
