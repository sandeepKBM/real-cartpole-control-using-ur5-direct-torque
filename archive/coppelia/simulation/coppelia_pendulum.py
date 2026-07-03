"""Spawn and read a CoppeliaSim pendulum hinged from the UR5 task frame."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

PENDULUM_HINGE_ALIAS = "real_cartpole_pendulum_hinge"
PENDULUM_POLE_ALIAS = "real_cartpole_pendulum_pole"
DEFAULT_POLE_LENGTH_M = 0.4


@dataclass
class CoppeliaPendulumHandles:
    hinge_handle: int
    pole_handle: int
    parent_handle: int
    pole_length_m: float

    @property
    def available(self) -> bool:
        return int(self.hinge_handle) >= 0


def _object_exists(sim: Any, alias: str) -> int | None:
    try:
        return int(sim.getObject(f"/{alias}"))
    except Exception:
        pass
    for prefix in ("/UR5/connection/real_cartpole_mujoco_attachment_site", "/UR5/UR5_connection"):
        try:
            return int(sim.getObject(f"{prefix}/{alias}"))
        except Exception:
            continue
    return None


def ensure_coppelia_pendulum(
    sim: Any,
    *,
    parent_handle: int,
    pole_length_m: float = DEFAULT_POLE_LENGTH_M,
    parent_path_hint: str = "",
) -> CoppeliaPendulumHandles | None:
    """
    Create (or reuse) a passive revolute pendulum under the task-frame parent.

    The hinge axis is local +X; positive angle is mapped to project convention
  (+theta leans toward world +X) at read time using the parent orientation.
    """
    pole_length_m = max(float(pole_length_m), 0.05)
    existing = _object_exists(sim, PENDULUM_HINGE_ALIAS)
    if existing is not None:
        pole = _object_exists(sim, PENDULUM_POLE_ALIAS) or -1
        return CoppeliaPendulumHandles(
            hinge_handle=int(existing),
            pole_handle=int(pole),
            parent_handle=int(parent_handle),
            pole_length_m=pole_length_m,
        )

    try:
        joint = int(
            sim.createJoint(
                sim.joint_revolute_subtype,
                sim.jointmode_passive,
                0,
            )
        )
        sim.setObjectAlias(joint, PENDULUM_HINGE_ALIAS)
        sim.setObjectParent(joint, int(parent_handle), True)
        sim.setJointPosition(joint, 0.0)
        try:
            sim.setObjectInt32Param(joint, sim.jointintparam_motor_enabled, 0)
        except Exception:
            pass

        radius = 0.03
        pole = int(
            sim.createPrimitiveShape(
                sim.primitiveshape_cylinder,
                [radius, pole_length_m, 0.0],
                0,
            )
        )
        sim.setObjectAlias(pole, PENDULUM_POLE_ALIAS)
        sim.setObjectParent(pole, joint, True)
        sim.setObjectPosition(pole, joint, [0.0, 0.0, pole_length_m * 0.5])
        try:
            sim.setShapeColor(
                pole,
                0,
                sim.colorcomponent_ambient_diffuse,
                [1.0, 0.2, 0.1],
            )
        except Exception:
            pass
        return CoppeliaPendulumHandles(
            hinge_handle=joint,
            pole_handle=pole,
            parent_handle=int(parent_handle),
            pole_length_m=pole_length_m,
        )
    except Exception as exc:
        print(f"[pendulum] failed to spawn Coppelia pendulum: {exc}", flush=True)
        if parent_path_hint:
            print(f"[pendulum] parent path hint: {parent_path_hint}", flush=True)
        return None


def read_coppelia_pendulum_state(
    sim: Any,
    handles: CoppeliaPendulumHandles | None,
    *,
    parent_handle: int | None = None,
) -> tuple[float, float] | None:
    """Return ``(theta_rad, theta_dot_radps)`` in project convention."""
    if handles is None or not handles.available:
        return None
    try:
        theta_local = float(sim.getJointPosition(int(handles.hinge_handle)))
        theta_dot_local = float(sim.getJointVelocity(int(handles.hinge_handle)))
    except Exception:
        return None

    parent = int(parent_handle if parent_handle is not None else handles.parent_handle)
    try:
        pose = sim.getObjectPose(parent, -1)
        mat = np.array(pose, dtype=np.float64).reshape(7)
        quat_xyzw = mat[3:]
        # Build rotation matrix from quaternion (x,y,z,w)
        x, y, z, w = quat_xyzw
        rot = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )
        lean_sign = float(np.sign(rot[0, 2])) if abs(rot[0, 2]) > 1.0e-6 else 1.0
    except Exception:
        lean_sign = 1.0
    return float(lean_sign * theta_local), float(lean_sign * theta_dot_local)
