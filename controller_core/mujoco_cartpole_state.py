"""Read cart-pole task state from a MuJoCo UR5e + pendulum model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import mujoco
import numpy as np

from controller_core.controller_interfaces import ControllerState

JOINT_NAME_ORDER: tuple[str, ...] = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)

DEFAULT_ATTACHMENT_SITE = "attachment_site"
DEFAULT_PENDULUM_HINGE = "pendulum_hinge"


@dataclass(frozen=True)
class MuJoCoCartPoleHandles:
  model: mujoco.MjModel
  data: mujoco.MjData
  attachment_site_id: int
  pendulum_hinge_id: int
  pendulum_qpos_adr: int
  pendulum_dof_adr: int
  has_pendulum: bool


def default_cartpole_scene_candidates(repo_root: Path) -> tuple[Path, ...]:
    menagerie = repo_root / "mujoco_menagerie" / "universal_robots_ur5e"
    return (
        menagerie / "scene_ur5e_cartpole.xml",
        menagerie / "scene.xml",
        menagerie / "ur5e.xml",
    )


def build_mujoco_cartpole_observer(
    scene_candidates: Sequence[Path],
) -> MuJoCoCartPoleHandles | None:
    for scene in scene_candidates:
        if not Path(scene).exists():
            continue
        try:
            model = mujoco.MjModel.from_xml_path(str(scene))
            data = mujoco.MjData(model)
            site_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, DEFAULT_ATTACHMENT_SITE))
            hinge_id = int(
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, DEFAULT_PENDULUM_HINGE)
            )
            has_pendulum = hinge_id >= 0
            pendulum_qpos_adr = int(model.jnt_qposadr[hinge_id]) if has_pendulum else -1
            pendulum_dof_adr = int(model.jnt_dofadr[hinge_id]) if has_pendulum else -1
            return MuJoCoCartPoleHandles(
                model=model,
                data=data,
                attachment_site_id=site_id,
                pendulum_hinge_id=hinge_id,
                pendulum_qpos_adr=pendulum_qpos_adr,
                pendulum_dof_adr=pendulum_dof_adr,
                has_pendulum=has_pendulum,
            )
        except Exception:
            continue
    return None


def _set_robot_joint_state(
    handles: MuJoCoCartPoleHandles,
    q: np.ndarray,
    qd: np.ndarray,
) -> bool:
    model, data = handles.model, handles.data
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    qd = np.asarray(qd, dtype=np.float64).reshape(-1)
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    for idx, joint_name in enumerate(JOINT_NAME_ORDER):
        jid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name))
        if jid < 0:
            return False
        qadr = int(model.jnt_qposadr[jid])
        vadr = int(model.jnt_dofadr[jid])
        if qadr < model.nq:
            data.qpos[qadr] = float(q[idx])
        if vadr < model.nv:
            data.qvel[vadr] = float(qd[idx])
    return True


def read_mujoco_cartpole_state(
    handles: MuJoCoCartPoleHandles | None,
    q: np.ndarray,
    qd: np.ndarray,
    *,
    time_s: float = 0.0,
    dt_s: float = 0.05,
    target_x: float = 0.0,
    target_theta: float = 0.0,
    transport_axis_index: int = 0,
) -> ControllerState | None:
    """Mirror Coppelia / hardware joint state into the MuJoCo cart-pole observer."""
    if handles is None:
        return None
    if not _set_robot_joint_state(handles, q, qd):
        return None
    model, data = handles.model, handles.data
    mujoco.mj_forward(model, data)
    site_id = int(handles.attachment_site_id)
    if site_id < 0:
        return None
    ee_pos = np.asarray(data.site_xpos[site_id], dtype=np.float64)
    site_jacp = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jacSite(model, data, site_jacp, None, site_id)
    ee_vel = site_jacp @ np.asarray(data.qvel, dtype=np.float64)

    theta = 0.0
    theta_dot = 0.0
    if handles.has_pendulum:
        theta = float(data.qpos[int(handles.pendulum_qpos_adr)])
        theta_dot = float(data.qvel[int(handles.pendulum_dof_adr)])
        # Hinge axis +X: positive angle leans the pole toward world +Y in the
        # default flange pose. Map to project convention (+theta = lean +X) with
        # the attachment-site world frame x component sign.
        site_xmat = np.asarray(data.site_xmat[site_id], dtype=np.float64).reshape(3, 3)
        lean_sign = float(np.sign(site_xmat[0, 2])) if abs(site_xmat[0, 2]) > 1.0e-6 else 1.0
        theta = lean_sign * theta
        theta_dot = lean_sign * theta_dot

    return ControllerState(
        x=float(ee_pos[int(transport_axis_index)]),
        x_dot=float(ee_vel[int(transport_axis_index)]),
        theta=float(theta),
        theta_dot=float(theta_dot),
        time_s=float(time_s),
        dt_s=float(dt_s),
        target_x=float(target_x),
        target_theta=float(target_theta),
    )
