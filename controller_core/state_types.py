"""
Typed state containers used on the wire between adapters and the controller.

The goal is one canonical shape: any simulator or real-robot adapter must
produce a ``RobotState`` dict with these exact keys and shapes. Mismatches are
caught eagerly by ``as_robot_state``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypedDict

import numpy as np


class RobotState(TypedDict, total=False):
    """One snapshot passed to the controller each cycle.

    Required keys:
      - ``time`` (float): wall or sim seconds since the start of the run.
      - ``q`` (np.ndarray shape [6]): joint positions in the canonical order
        ``[shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3]``.
      - ``qd`` (np.ndarray shape [6]): joint velocities in the same order.
      - ``ee_pos`` (np.ndarray shape [3]): end-effector position in the world
        frame in meters.
      - ``ee_quat`` (np.ndarray shape [4]): end-effector orientation
        ``[w, x, y, z]`` in the world frame.
      - ``target_x`` (float): commanded world-X target for the end effector.
      - ``target_x_vel`` (float): optional desired world-X target velocity.

    Optional keys:
      - ``ee_lin_vel`` (np.ndarray shape [3]): world-frame linear velocity.
        If absent the controller falls back to ``J_pos @ qd``.
      - ``ee_ang_vel`` (np.ndarray shape [3]): world-frame angular velocity.
      - ``jacobian_pos`` (np.ndarray shape [3, 6]): ``J_pos`` at the end
        effector in world frame. Required only for the J^T adapter.
      - ``jacobian_rot`` (np.ndarray shape [3, 6]): ``J_rot`` at the end
        effector in world frame. Optional, reserved for future 6D tasks.
      - ``gravity_torque`` (np.ndarray shape [6]): pre-computed gravity
        compensation torque from an external dynamics library or the
        simulator. If absent, gravity compensation is skipped.
      - ``jacobian`` (np.ndarray shape [6, 6]): stacked world-frame Jacobian
        ``[J_pos; J_rot]`` so ``[v; omega] = J @ qd`` (first three rows linear).
      - ``hold_current_pose`` (bool): optional settle-phase hint that asks the
        impedance controller to zero its task-space wrench against the current
        pose before transport begins.
    """

    time: float
    q: np.ndarray
    qd: np.ndarray
    ee_pos: np.ndarray
    ee_quat: np.ndarray
    ee_lin_vel: np.ndarray
    ee_ang_vel: np.ndarray
    target_x: float
    target_x_vel: float
    jacobian: np.ndarray
    jacobian_pos: np.ndarray
    jacobian_rot: np.ndarray
    gravity_torque: np.ndarray


ControlMode = Literal["torque", "cartesian_x_force"]


@dataclass
class ControlOutput:
    """Structured output of the simulator-independent controller.

    The controller itself returns ``mode == "cartesian_x_force"``; the
    J^T adapter (see ``kinematics_utils.cartesian_force_to_joint_torque``)
    produces the corresponding ``mode == "torque"`` output.
    """

    mode: ControlMode
    fx: float | None = None
    tau: np.ndarray | None = None
    # Diagnostics for logging and regression tests.
    x_error: float | None = None
    ee_vx: float | None = None
    saturated: bool = False

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"mode": self.mode, "saturated": bool(self.saturated)}
        if self.fx is not None:
            out["Fx"] = float(self.fx)
        if self.tau is not None:
            out["tau"] = np.asarray(self.tau, dtype=np.float64).tolist()
        if self.x_error is not None:
            out["x_error"] = float(self.x_error)
        if self.ee_vx is not None:
            out["ee_vx"] = float(self.ee_vx)
        return out


def _asarray_1d(value: Any, name: str, length: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.shape[0] != length:
        raise ValueError(f"{name} must have length {length}; got shape {np.asarray(value).shape}")
    return arr


def _asarray_2d(value: Any, name: str, rows: int, cols: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (rows, cols):
        raise ValueError(f"{name} must have shape {(rows, cols)}; got {arr.shape}")
    return arr


def as_robot_state(raw: dict[str, Any], num_joints: int = 6) -> RobotState:
    """Validate and normalize a raw dict into a ``RobotState``.

    Any missing required field raises ``KeyError``. Any shape mismatch raises
    ``ValueError``. Optional fields are only validated if present.
    """
    required = ("time", "q", "qd", "ee_pos", "ee_quat", "target_x")
    for key in required:
        if key not in raw:
            raise KeyError(f"RobotState missing required key: {key!r}")

    state: RobotState = {
        "time": float(raw["time"]),
        "q": _asarray_1d(raw["q"], "q", num_joints),
        "qd": _asarray_1d(raw["qd"], "qd", num_joints),
        "ee_pos": _asarray_1d(raw["ee_pos"], "ee_pos", 3),
        "ee_quat": _asarray_1d(raw["ee_quat"], "ee_quat", 4),
        "target_x": float(raw["target_x"]),
    }

    if "ee_lin_vel" in raw and raw["ee_lin_vel"] is not None:
        state["ee_lin_vel"] = _asarray_1d(raw["ee_lin_vel"], "ee_lin_vel", 3)
    if "ee_ang_vel" in raw and raw["ee_ang_vel"] is not None:
        state["ee_ang_vel"] = _asarray_1d(raw["ee_ang_vel"], "ee_ang_vel", 3)
    if "jacobian_pos" in raw and raw["jacobian_pos"] is not None:
        state["jacobian_pos"] = _asarray_2d(raw["jacobian_pos"], "jacobian_pos", 3, num_joints)
    if "jacobian_rot" in raw and raw["jacobian_rot"] is not None:
        state["jacobian_rot"] = _asarray_2d(raw["jacobian_rot"], "jacobian_rot", 3, num_joints)
    if "gravity_torque" in raw and raw["gravity_torque"] is not None:
        state["gravity_torque"] = _asarray_1d(raw["gravity_torque"], "gravity_torque", num_joints)
    if "jacobian" in raw and raw["jacobian"] is not None:
        state["jacobian"] = _asarray_2d(raw["jacobian"], "jacobian", 6, num_joints)
    if "target_x_vel" in raw and raw["target_x_vel"] is not None:
        state["target_x_vel"] = float(raw["target_x_vel"])
    if "hold_current_pose" in raw and raw["hold_current_pose"] is not None:
        state["hold_current_pose"] = bool(raw["hold_current_pose"])

    return state


def as_impedance_robot_state(raw: dict[str, Any], num_joints: int = 6) -> RobotState:
    """Like ``as_robot_state`` but requires full Cartesian impedance inputs."""
    required = (
        "time",
        "q",
        "qd",
        "ee_pos",
        "ee_quat",
        "ee_lin_vel",
        "ee_ang_vel",
        "target_x",
        "jacobian",
    )
    for key in required:
        if key not in raw or raw[key] is None:
            raise KeyError(f"Impedance RobotState missing required key: {key!r}")
    state: RobotState = {
        "time": float(raw["time"]),
        "q": _asarray_1d(raw["q"], "q", num_joints),
        "qd": _asarray_1d(raw["qd"], "qd", num_joints),
        "ee_pos": _asarray_1d(raw["ee_pos"], "ee_pos", 3),
        "ee_quat": _asarray_1d(raw["ee_quat"], "ee_quat", 4),
        "ee_lin_vel": _asarray_1d(raw["ee_lin_vel"], "ee_lin_vel", 3),
        "ee_ang_vel": _asarray_1d(raw["ee_ang_vel"], "ee_ang_vel", 3),
        "target_x": float(raw["target_x"]),
        "jacobian": _asarray_2d(raw["jacobian"], "jacobian", 6, num_joints),
    }
    if "target_x_vel" in raw and raw["target_x_vel"] is not None:
        state["target_x_vel"] = float(raw["target_x_vel"])
    if "gravity_torque" in raw and raw["gravity_torque"] is not None:
        state["gravity_torque"] = _asarray_1d(raw["gravity_torque"], "gravity_torque", num_joints)
    if "hold_current_pose" in raw and raw["hold_current_pose"] is not None:
        state["hold_current_pose"] = bool(raw["hold_current_pose"])
    return state
