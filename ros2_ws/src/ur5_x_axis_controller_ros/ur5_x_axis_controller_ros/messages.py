"""
Small helpers to convert between ROS messages and the dicts/numpy arrays the
simulator-independent controller_core expects. Kept in one place so the
controller node, the bridge node, and tests all agree on field layout.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from geometry_msgs.msg import PoseStamped, TwistStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, MultiArrayDimension


# Canonical UR5 joint order used everywhere in this package.
UR5_JOINT_NAMES: tuple[str, ...] = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)


def joint_state_to_arrays(msg: JointState, joint_names: Sequence[str] = UR5_JOINT_NAMES) -> tuple[np.ndarray, np.ndarray]:
    """Reorder a ``JointState`` to the canonical UR5 order and return ``(q, qd)``."""
    if not msg.name:
        raise ValueError("JointState has no joint names; cannot reorder safely.")
    index = {n: i for i, n in enumerate(msg.name)}
    missing = [n for n in joint_names if n not in index]
    if missing:
        raise KeyError(f"JointState is missing expected UR5 joints: {missing}")
    q = np.array([msg.position[index[n]] for n in joint_names], dtype=np.float64)
    if msg.velocity and len(msg.velocity) == len(msg.name):
        qd = np.array([msg.velocity[index[n]] for n in joint_names], dtype=np.float64)
    else:
        qd = np.zeros(len(joint_names), dtype=np.float64)
    return q, qd


def pose_to_pos_quat(msg: PoseStamped) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(ee_pos, ee_quat)`` with ``ee_quat`` in ``[w, x, y, z]`` order."""
    p = msg.pose.position
    q = msg.pose.orientation
    return (
        np.array([p.x, p.y, p.z], dtype=np.float64),
        np.array([q.w, q.x, q.y, q.z], dtype=np.float64),
    )


def twist_to_lin_ang(msg: TwistStamped) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(ee_lin_vel, ee_ang_vel)`` world-frame arrays."""
    t = msg.twist
    return (
        np.array([t.linear.x, t.linear.y, t.linear.z], dtype=np.float64),
        np.array([t.angular.x, t.angular.y, t.angular.z], dtype=np.float64),
    )


def float_array_from(data: Iterable[float], label: str | None = None) -> Float64MultiArray:
    msg = Float64MultiArray()
    arr = np.asarray(list(data), dtype=np.float64).reshape(-1)
    msg.data = arr.tolist()
    dim = MultiArrayDimension()
    dim.label = label or "joint"
    dim.size = int(arr.shape[0])
    dim.stride = int(arr.shape[0])
    msg.layout.dim = [dim]
    msg.layout.data_offset = 0
    return msg


def jacobian_from_multiarray(msg: Float64MultiArray, num_joints: int = 6) -> np.ndarray:
    """Parse a 6xN or 3xN Jacobian published as a flat ``Float64MultiArray``.

    The bridge node stacks position rows first (3 rows), then rotation rows
    (3 rows), so the flat array is either 18 or 36 elements.
    """
    data = np.asarray(msg.data, dtype=np.float64).reshape(-1)
    if data.size == 3 * num_joints:
        return data.reshape(3, num_joints)
    if data.size == 6 * num_joints:
        return data.reshape(6, num_joints)
    raise ValueError(
        f"Unexpected Jacobian size {data.size}; expected {3*num_joints} or {6*num_joints}."
    )
