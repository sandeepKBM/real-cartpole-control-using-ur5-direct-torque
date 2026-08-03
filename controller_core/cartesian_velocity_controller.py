"""Resolved-rate Cartesian velocity controller -- P-only kinematic control
law for the UR5e's native RTDE ``speedL`` interface. Deliberately has NO
dynamics: no gravity compensation, no mass matrix, no Jacobian, no torque.

Why this exists: UR5e has no native torque interface -- every torque-control
mechanism in this package (x_axis_cartesian_impedance.py, torque_task_qp.py,
hard_constraint_qp.py) exists to fake compliant force behavior on a robot
that is natively position/velocity-controlled, and essentially every
documented Y-drift/orientation bug this repo has fought (see AGENTS.md
section 3) is a consequence of that dynamics modeling, not of the transport
task itself. ``speedL`` resolves a commanded Cartesian velocity to joint
velocities via the Jacobian ON THE ROBOT'S OWN FIRMWARE -- this controller
never touches a Jacobian or mass matrix at all, so none of that bug class
can occur here. The real tradeoff: this gives zero force compliance, so it
is only appropriate for phases where nothing needs to push back on the
end-effector (pure point-to-point transport / range characterization) --
not for eventual swing-up once a physical pole is mounted and pole-arm
interaction forces matter. See hardware/velocity_transport.py's module
docstring for the real-hardware wiring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .kinematics_utils import orientation_error_vec_wxyz
from .state_types import as_robot_state


@dataclass
class CartesianVelocityConfig:
    """Velocity gains are 1/s (v_cmd = kp * position_error), NOT force gains
    like CartesianImpedanceConfig's kp_x/kp_y/kp_z (N/m) -- do not reuse
    impedance-tuned gain values here, they are dimensionally different
    quantities entirely."""

    kp_x: float = 2.0
    kp_y: float = 2.0
    kp_z: float = 2.0
    kp_rot: float = 2.0
    max_lin_speed_mps: float = 0.25
    max_ang_speed_radps: float = 0.5

    @classmethod
    def from_controller_yaml_section(cls, ctrl: dict) -> "CartesianVelocityConfig":
        vc = ctrl.get("velocity_control", {}) or {}
        return cls(
            kp_x=float(vc.get("kp_x", 2.0)),
            kp_y=float(vc.get("kp_y", 2.0)),
            kp_z=float(vc.get("kp_z", 2.0)),
            kp_rot=float(vc.get("kp_rot", 2.0)),
            max_lin_speed_mps=float(vc.get("max_lin_speed_mps", 0.25)),
            max_ang_speed_radps=float(vc.get("max_ang_speed_radps", 0.5)),
        )


class CartesianVelocityController:
    """xd_cmd = feedforward + kp * (target - actual), clamped to configured
    speed ceilings. Holds Y/Z/orientation at their reset-time values unless
    the caller supplies a moving target_ee_pos/target_ee_vel for them."""

    def __init__(self, config: CartesianVelocityConfig) -> None:
        self.cfg = config
        self._initialized = False
        self._p0 = np.zeros(3, dtype=np.float64)
        self._quat0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def reset_from_state(self, state: dict[str, Any]) -> None:
        st = as_robot_state(state)
        self._p0 = np.asarray(st["ee_pos"], dtype=np.float64).reshape(3).copy()
        self._quat0 = np.asarray(st["ee_quat"], dtype=np.float64).reshape(4).copy()
        self._initialized = True

    @property
    def initialized(self) -> bool:
        return self._initialized

    def compute(self, state: dict[str, Any]) -> np.ndarray:
        """Returns a 6D Cartesian velocity command [vx,vy,vz,wx,wy,wz] in the
        axis-angle/rotvec convention RTDE's speedL/getActualTCPPose use --
        NOT a quaternion, deliberately unlike every torque-path controller
        in this package (hence returning a plain ndarray, not a
        CartesianImpedanceOutput)."""
        if not self._initialized:
            raise RuntimeError("Call reset_from_state() before compute().")
        st = as_robot_state(state)
        p = np.asarray(st["ee_pos"], dtype=np.float64).reshape(3)
        quat = np.asarray(st["ee_quat"], dtype=np.float64).reshape(4)
        p_des = np.asarray(st.get("target_ee_pos", self._p0), dtype=np.float64).reshape(3)
        v_ff = np.asarray(st.get("target_ee_vel", np.zeros(3)), dtype=np.float64).reshape(3)

        pos_err = p_des - p
        kp_lin = np.array([self.cfg.kp_x, self.cfg.kp_y, self.cfg.kp_z], dtype=np.float64)
        v_cmd = v_ff + kp_lin * pos_err

        e_rot = orientation_error_vec_wxyz(self._quat0, quat)
        w_cmd = self.cfg.kp_rot * e_rot

        v_norm = float(np.linalg.norm(v_cmd))
        if v_norm > self.cfg.max_lin_speed_mps and v_norm > 1.0e-9:
            v_cmd = v_cmd * (self.cfg.max_lin_speed_mps / v_norm)
        w_norm = float(np.linalg.norm(w_cmd))
        if w_norm > self.cfg.max_ang_speed_radps and w_norm > 1.0e-9:
            w_cmd = w_cmd * (self.cfg.max_ang_speed_radps / w_norm)

        return np.concatenate([v_cmd, w_cmd]).astype(np.float64)
