"""Frame-free, unambiguous test of the rotational P term's sign.

Define V(q) = ||orientation_error_vec_wxyz(quat0, quat(q))||^2 -- exactly the
scalar the controller reports as `orientation_error_norm` (squared), and the
scalar the safety guard trips on.  Numerically differentiate it w.r.t. q.

A RESTORING torque must produce joint acceleration that DESCENDS V:
        qddot . grad_q V  <  0     with qddot = M^-1 tau.

Four candidate rotational P terms are compared, all with kd_rot = 0, kp_x = 0,
so only the P term is present and task_excluded_joints=[0] applied as shipped:
   A: m = +kp * e             <-- AS CODED (controller.py line 436)
   B: m = -kp * e
   C: m = +kp * R_des @ e
   D: m = -kp * R_des @ e     <-- textbook restoring world moment
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "/common/users/ss5772/real_Cartpole")

import mujoco

from controller_core.kinematics_utils import orientation_error_vec_wxyz, quat_to_rotmat, rotmat_to_quat
from simulation.ur5e_mujoco_torque import load_model, expand_mass_matrix

ARM_Q0 = np.array([-2.3688, -2.1801, -1.8838, -0.7962, 0.004714693, 0.0206])

model, data, site_id, joint_ids, _ = load_model(
    "/common/users/ss5772/real_Cartpole/assets/ur5e_torque/scene.xml"
)


def kin(q):
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    for i, jid in enumerate(joint_ids):
        data.qpos[int(model.jnt_qposadr[jid])] = float(q[i])
    mujoco.mj_forward(model, data)
    R = np.asarray(data.site_xmat[site_id]).reshape(3, 3).copy()
    quat = rotmat_to_quat(R)
    jacp = np.zeros((3, model.nv)); jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    J = np.vstack([jacp[:, :6], jacr[:, :6]])
    M = expand_mass_matrix(model, data)[:6, :6]
    return quat, R, J, M


quat0, R0, _, _ = kin(ARM_Q0)


def V(q):
    quat, _, _, _ = kin(q)
    return float(np.sum(orientation_error_vec_wxyz(quat0, quat) ** 2))


def gradV(q, h=1e-6):
    g = np.zeros(6)
    for i in range(6):
        qp = q.copy(); qp[i] += h
        qm = q.copy(); qm[i] -= h
        g[i] = (V(qp) - V(qm)) / (2 * h)
    return g


rng = np.random.default_rng(11)
tally = {"A (+kp*e, AS CODED)": 0, "B (-kp*e)": 0,
         "C (+kp*R@e)": 0, "D (-kp*R@e, textbook)": 0}
print(f"{'trial':>5} {'|e|':>8} | " + " | ".join(f"{k:>22}" for k in tally))
for trial in range(8):
    q = ARM_Q0 + rng.normal(size=6) * 0.03
    quat, R, J, M = kin(q)
    e = orientation_error_vec_wxyz(quat0, quat)
    g = gradV(q)
    Minv = np.linalg.inv(M)
    J_red = np.vstack([J[0:1, :], J[3:6, :]]).copy()
    J_red[:, 0] = 0.0
    row = []
    for key, m in (
        ("A (+kp*e, AS CODED)", +e),
        ("B (-kp*e)", -e),
        ("C (+kp*R@e)", +R0 @ e),
        ("D (-kp*R@e, textbook)", -R0 @ e),
    ):
        tau = J_red.T @ np.concatenate([[0.0], m])
        qddot = Minv @ tau
        d = float(qddot @ g)
        tally[key] += int(d < 0)
        row.append(f"{d:+.3e} {'DESC' if d < 0 else 'ASC '}")
    print(f"{trial:>5} {np.linalg.norm(e):8.5f} | " + " | ".join(f"{r:>22}" for r in row))

print("\nDescends V (restoring) out of 8 trials:")
for k, v in tally.items():
    print(f"  {k:>24}: {v}/8")
