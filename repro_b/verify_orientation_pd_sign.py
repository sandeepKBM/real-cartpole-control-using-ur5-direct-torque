"""Is the rotational P term in x_task_yz_corridor_qp RESTORING or DESTABILIZING?

controller.py line 436:   m_rot = kp_rot * e_rot - kd_rot * omega
with e_rot = orientation_error_vec_wxyz(quat0, quat)
           = 2 * vec( conj(q_des) (x) q_cur )

For q_cur = q_des (x) exp(delta/2),  e_rot ~= delta, the BODY-frame rotation
vector taking DESIRED -> CURRENT.  A restoring moment must therefore be
-k * (R_des @ delta) expressed in the WORLD frame (which is the frame the
angular rows of J, and hence J.T, live in).

Test 1 (pure quaternion algebra): confirm e_rot ~= delta (desired-frame),
       i.e. it points along (current - desired), NOT (desired - current).
Test 2 (real model): at ARM_Q0 perturbed, with kd_rot = 0, kp_rot = 1, kp_x = 0,
       compute alpha_world = (J M^-1 tau)[3:6] and the second derivative of the
       error vector,  e_ddot ~= R_des^T alpha_world.
       RESTORING  <=>  e . e_ddot < 0.
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "/common/users/ss5772/real_Cartpole")

import mujoco

from controller_core.kinematics_utils import (
    orientation_error_vec_wxyz,
    quat_multiply_wxyz,
    quat_to_rotmat,
    rotmat_to_quat,
)
from simulation.ur5e_mujoco_torque import load_model, expand_mass_matrix

ARM_Q0 = np.array([-2.3688, -2.1801, -1.8838, -0.7962, 0.004714693, 0.0206])


def exp_quat(v):
    v = np.asarray(v, float)
    th = np.linalg.norm(v)
    if th < 1e-14:
        return np.array([1.0, 0.0, 0.0, 0.0])
    ax = v / th
    return np.concatenate([[np.cos(th / 2)], ax * np.sin(th / 2)])


print("=" * 74)
print("TEST 1 -- what does orientation_error_vec_wxyz actually return?")
print("=" * 74)
rng = np.random.default_rng(3)
q_des = rng.normal(size=4); q_des /= np.linalg.norm(q_des)
delta = np.array([0.01, -0.02, 0.03])           # body-frame perturbation
q_cur = quat_multiply_wxyz(q_des, exp_quat(delta))
e = orientation_error_vec_wxyz(q_des, q_cur)
print(f"  applied body-frame delta (des -> cur) = {delta}")
print(f"  e_rot returned                        = {np.round(e, 6)}")
print(f"  e_rot ~= +delta ?  max err = {np.max(np.abs(e - delta)):.2e}")
print(f"  e_rot ~= -delta ?  max err = {np.max(np.abs(e + delta)):.2e}")
R_des = quat_to_rotmat(q_des)
print(f"  world-frame perturbation R_des@delta  = {np.round(R_des @ delta, 6)}")
print("  => e_rot points along (CURRENT - DESIRED), in the DESIRED body frame.")
print("     A restoring world moment is therefore  -k * R_des @ e_rot.")
print("     The code applies  m_rot = +kp_rot * e_rot  (no rotation, + sign).")

print()
print("=" * 74)
print("TEST 2 -- closed-form on the real model at ARM_Q0: restoring or not?")
print("=" * 74)
model, data, site_id, joint_ids, _ = load_model(
    "/common/users/ss5772/real_Cartpole/assets/ur5e_torque/scene.xml"
)


def kinematics(q):
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


quat0, R0, _, _ = kinematics(ARM_Q0)

n_restoring = 0
n_destab = 0
for trial in range(6):
    dq = rng.normal(size=6) * 0.02
    q = ARM_Q0 + dq
    quat, R, J, M = kinematics(q)
    e = orientation_error_vec_wxyz(quat0, quat)
    if np.linalg.norm(e) < 1e-8:
        continue
    kp_rot = 1.0
    # exactly the controller's construction, with kp_x = kd_rot = 0 so ONLY the
    # rotational P term is present.
    m_rot = kp_rot * e
    J_red = np.vstack([J[0:1, :], J[3:6, :]])
    J_red = J_red.copy(); J_red[:, 0] = 0.0        # task_excluded_joints = [0]
    wrench = np.concatenate([[0.0], m_rot])
    tau = J_red.T @ wrench
    alpha_world = (J @ np.linalg.inv(M) @ tau)[3:6]
    # e lives in the DESIRED body frame; its second derivative is R0^T alpha.
    e_ddot = R0.T @ alpha_world
    dot = float(e @ e_ddot)
    dot_noframe = float(e @ alpha_world)
    verdict = "RESTORING" if dot < 0 else "DESTABILISING (positive feedback)"
    n_restoring += int(dot < 0)
    n_destab += int(dot >= 0)
    print(f"  trial {trial}: |e|={np.linalg.norm(e):.5f}  "
          f"e.e_ddot={dot:+.5e}  (no-frame e.alpha={dot_noframe:+.4e})  -> {verdict}")

print(f"\n  restoring: {n_restoring}/6   destabilising: {n_destab}/6")
print("  (with the SIGN FLIPPED, m_rot = -kp_rot*e_rot, every sign above inverts.)")
