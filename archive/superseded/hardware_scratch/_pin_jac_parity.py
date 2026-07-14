"""Quick MuJoCo vs Pinocchio Jacobian parity check (dev helper)."""
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pinocchio as pin
from controller_core.model_dynamics import DEFAULT_UR5E_MJCF, PinocchioUR5eDynamics

REPO = Path(__file__).resolve().parents[1]
model = mujoco.MjModel.from_xml_path(str(REPO / "assets/ur5e_torque/scene.xml"))
data = mujoco.MjData(model)
site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
dyn = PinocchioUR5eDynamics(DEFAULT_UR5E_MJCF)
frame_id = dyn.model.getFrameId("attachment_site")

q = np.array([0.0, -1.5707963267948966, 0.0, -1.5707963267948966, 0.0, 0.0])
data.qpos[:6] = q
mujoco.mj_forward(model, data)
jacp = np.zeros((3, model.nv))
jacr = np.zeros((3, model.nv))
mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
J_mj = np.vstack([jacp[:, :6], jacr[:, :6]])

pin.computeJointJacobians(dyn.model, dyn.data, q)
pin.updateFramePlacements(dyn.model, dyn.data)
for ref_name, ref in [
    ("LOCAL_WORLD_ALIGNED", pin.LOCAL_WORLD_ALIGNED),
    ("WORLD", pin.WORLD),
    ("LOCAL", pin.LOCAL),
]:
    J_pin = pin.getFrameJacobian(dyn.model, dyn.data, frame_id, ref)
    err = np.max(np.abs(J_mj - J_pin))
    print(ref_name, "max err", err)

M_mj = np.zeros((model.nv, model.nv))
mujoco.mj_fullM(model, data, M_mj)
M_mj = M_mj[:6, :6]
M_pin = dyn.mass_matrix(q)
print("M max err", np.max(np.abs(M_mj - M_pin)))

J_pin = pin.getFrameJacobian(dyn.model, dyn.data, frame_id, pin.LOCAL_WORLD_ALIGNED)
print("J_mj\n", J_mj)
print("J_pin\n", J_pin)
print("diff\n", J_mj - J_pin)
# Try swapping angular/linear rows
J_pin_swap = np.vstack([J_pin[3:6], J_pin[0:3]])
q_fail = np.array([1.72111261, -2.89294764, -2.88406335, -6.07549745, 3.93667287, 5.18684343])
data.qpos[:6]=q_fail; mujoco.mj_forward(model,data)
jacp,jacr=np.zeros((3,model.nv)),np.zeros((3,model.nv))
mujoco.mj_jacSite(model,data,jacp,jacr,site_id)
J_mj=np.vstack([jacp[:,:6],jacr[:,:6]])
pin.computeJointJacobians(dyn.model,dyn.data,q_fail)
pin.updateFramePlacements(dyn.model,dyn.data)
for ref_name, ref in [("LWA", pin.LOCAL_WORLD_ALIGNED), ("WORLD", pin.WORLD), ("LOCAL", pin.LOCAL)]:
    Jp=pin.getFrameJacobian(dyn.model,dyn.data,frame_id,ref)
    Jfix=-Jp.copy(); Jfix[5]=Jp[5]
    print(ref_name, "raw", np.max(np.abs(J_mj-Jp)), "fix", np.max(np.abs(J_mj-Jfix)))
