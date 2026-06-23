"""
Constrained Cartesian impedance / PD torque law for UR5 X transport.

Stabilizes X tracking while holding initial Y, Z, tool orientation, and a
rest joint posture. Pure numpy; no simulator imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .kinematics_utils import orientation_error_vec_wxyz
from .state_types import as_impedance_robot_state


JOINT_NAME_ORDER: tuple[str, ...] = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)


@dataclass
class CartesianImpedanceConfig:
    kp_x: float = 25.0
    kd_x: float = 8.0
    kp_y: float = 80.0
    kd_y: float = 15.0
    kp_z: float = 120.0
    kd_z: float = 20.0
    kp_rot: float = 20.0
    kd_rot: float = 5.0
    kp_posture: float = 2.0
    kd_posture: float = 0.5
    kd_joint: float = 0.8
    tau_max_nm: np.ndarray = field(
        default_factory=lambda: np.array([8.0, 8.0, 8.0, 2.5, 2.5, 2.5], dtype=np.float64)
    )
    jacobian_singular_cond_max: float = 1.0e5
    torque_headroom: float = 0.9
    task_resample_factor: float = 0.5
    task_resample_min_scale: float = 1.0 / 16384.0
    task_resample_max_iters: int = 14

    @classmethod
    def from_controller_yaml_section(cls, ctrl: dict) -> "CartesianImpedanceConfig":
        gains = ctrl.get("gains", {}) or {}
        mode = str(ctrl.get("torque_limits_mode", "initial")).lower()
        lim_key = (
            "torque_limits_initial"
            if mode == "initial"
            else "torque_limits_after_stable"
        )
        lim_dict = ctrl.get(lim_key, {}) or {}
        tau_list = [float(lim_dict[name]) for name in JOINT_NAME_ORDER]
        tm = np.asarray(tau_list, dtype=np.float64)
        return cls(
            kp_x=float(gains.get("kp_x", 25.0)),
            kd_x=float(gains.get("kd_x", 8.0)),
            kp_y=float(gains.get("kp_y", 80.0)),
            kd_y=float(gains.get("kd_y", 15.0)),
            kp_z=float(gains.get("kp_z", 120.0)),
            kd_z=float(gains.get("kd_z", 20.0)),
            kp_rot=float(gains.get("kp_rot", 20.0)),
            kd_rot=float(gains.get("kd_rot", 5.0)),
            kp_posture=float(gains.get("kp_posture", 2.0)),
            kd_posture=float(gains.get("kd_posture", 0.5)),
            kd_joint=float(gains.get("kd_joint", 0.8)),
            tau_max_nm=tm,
            jacobian_singular_cond_max=float(
                ctrl.get("jacobian_singular_cond_max", 1.0e5)
            ),
            torque_headroom=float(ctrl.get("torque_headroom", 0.9)),
            task_resample_factor=float(ctrl.get("task_resample_factor", 0.5)),
            task_resample_min_scale=float(ctrl.get("task_resample_min_scale", 1.0 / 16384.0)),
            task_resample_max_iters=int(ctrl.get("task_resample_max_iters", 14)),
        )


@dataclass
class CartesianImpedanceOutput:
    tau: np.ndarray
    tau_preclip: np.ndarray
    wrench: np.ndarray
    tau_task_nominal: np.ndarray
    tau_task: np.ndarray
    tau_damping: np.ndarray
    tau_posture: np.ndarray
    tau_gravity: np.ndarray
    tau_saturated: np.ndarray
    jacobian_cond: float
    singular_scale: float
    task_backtrack_scale: float
    task_scale: float
    task_backtrack_iters: int
    task_feasible: bool
    x_error: float
    y_error: float
    z_error: float
    orientation_error_vec: np.ndarray
    orientation_error_norm: float


class XAxisCartesianImpedanceController:
    """Full 6D Cartesian impedance + posture + joint damping (+ optional gravity).

    The task-space wrench is mapped through ``J.T`` and then backtracked if the
    resulting joint torques exceed the configured headroom around the per-joint
    torque limits.
    """

    def __init__(self, config: CartesianImpedanceConfig) -> None:
        self.cfg = config
        self._initialized = False
        self._hold_reference_initialized = False
        self._x0 = 0.0
        self._y0 = 0.0
        self._z0 = 0.0
        self._quat0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._q_rest = np.zeros(6, dtype=np.float64)

    def reset_from_state(self, state: dict[str, Any]) -> None:
        st = as_impedance_robot_state(state)
        ee = np.asarray(st["ee_pos"], dtype=np.float64).reshape(3)
        self._x0 = float(ee[0])
        self._y0 = float(ee[1])
        self._z0 = float(ee[2])
        self._quat0 = np.asarray(st["ee_quat"], dtype=np.float64).reshape(4).copy()
        self._q_rest = np.asarray(st["q"], dtype=np.float64).reshape(6).copy()
        self._hold_reference_initialized = False
        self._initialized = True

    @property
    def initialized(self) -> bool:
        return self._initialized

    @staticmethod
    def _torque_within_limits(tau: np.ndarray, limit: np.ndarray) -> bool:
        tau = np.asarray(tau, dtype=np.float64).reshape(6)
        limit = np.asarray(limit, dtype=np.float64).reshape(6)
        return bool(np.all(np.abs(tau) <= limit + 1e-12))

    def _backtrack_task_scale(
        self,
        tau_nominal: np.ndarray,
        tau_limit: np.ndarray,
    ) -> tuple[float, np.ndarray, int, bool]:
        """Geometrically shrink the full torque candidate until it fits the limit box."""
        tau_nominal = np.asarray(tau_nominal, dtype=np.float64).reshape(6)
        tau_limit = np.asarray(tau_limit, dtype=np.float64).reshape(6)

        resample_factor = float(self.cfg.task_resample_factor)
        if not np.isfinite(resample_factor) or resample_factor <= 0.0 or resample_factor >= 1.0:
            resample_factor = 0.5

        min_scale = max(float(self.cfg.task_resample_min_scale), 0.0)
        max_iters = max(int(self.cfg.task_resample_max_iters), 0)

        task_scale = 1.0
        tau_candidate = task_scale * tau_nominal
        feasible = self._torque_within_limits(tau_candidate, tau_limit)
        iters = 0

        while (not feasible) and (iters < max_iters) and (task_scale > min_scale + 1e-12):
            next_scale = max(task_scale * resample_factor, min_scale)
            if next_scale >= task_scale - 1e-12:
                break
            next_candidate = next_scale * tau_nominal
            if np.allclose(next_candidate, tau_candidate, rtol=0.0, atol=1e-12):
                task_scale = next_scale
                tau_candidate = next_candidate
                break
            task_scale = next_scale
            tau_candidate = next_candidate
            feasible = self._torque_within_limits(tau_candidate, tau_limit)
            iters += 1

        return task_scale, tau_candidate, iters, feasible

    def compute(self, state: dict[str, Any]) -> CartesianImpedanceOutput:
        if not self._initialized:
            raise RuntimeError("Call reset_from_state() before compute().")
        st = as_impedance_robot_state(state)
        q = np.asarray(st["q"], dtype=np.float64).reshape(6)
        qd = np.asarray(st["qd"], dtype=np.float64).reshape(6)
        p = np.asarray(st["ee_pos"], dtype=np.float64).reshape(3)
        quat = np.asarray(st["ee_quat"], dtype=np.float64).reshape(4)
        v = np.asarray(st["ee_lin_vel"], dtype=np.float64).reshape(3)
        omega = np.asarray(st["ee_ang_vel"], dtype=np.float64).reshape(3)
        J = np.asarray(st["jacobian"], dtype=np.float64).reshape(6, 6)

        hold_current_pose = bool(st.get("hold_current_pose", False))
        if hold_current_pose:
            # Capture the settle reference once, then keep that reference
            # fixed so posture and gravity compensation can actually hold the
            # arm instead of re-zeroing the restoring torque every step.
            if not self._hold_reference_initialized:
                self._x0 = float(p[0])
                self._y0 = float(p[1])
                self._z0 = float(p[2])
                self._quat0 = quat.copy()
                self._q_rest = q.copy()
                self._hold_reference_initialized = True
            x_des = self._x0
            x_vel_des = 0.0
            y_des = self._y0
            z_des = self._z0
            quat_ref = self._quat0
        else:
            x_des = float(st["target_x"])
            x_vel_des = float(st.get("target_x_vel", 0.0))
            y_des = self._y0
            z_des = self._z0
            quat_ref = self._quat0

        x_err = x_des - float(p[0])
        y_err = y_des - float(p[1])
        z_err = z_des - float(p[2])

        Fx = self.cfg.kp_x * x_err + self.cfg.kd_x * (x_vel_des - float(v[0]))
        Fy = self.cfg.kp_y * y_err - self.cfg.kd_y * float(v[1])
        Fz = self.cfg.kp_z * z_err - self.cfg.kd_z * float(v[2])

        e_rot = orientation_error_vec_wxyz(quat_ref, quat)
        ori_norm = float(np.linalg.norm(e_rot))
        M = self.cfg.kp_rot * e_rot - self.cfg.kd_rot * omega

        wrench = np.array([Fx, Fy, Fz, M[0], M[1], M[2]], dtype=np.float64)

        # Jacobian conditioning: scale wrench down near singularities.
        cond = float(np.linalg.cond(J))
        singular_scale = 1.0
        if cond > self.cfg.jacobian_singular_cond_max > 0.0:
            singular_scale = float(self.cfg.jacobian_singular_cond_max / cond)
        wrench_scaled = wrench * singular_scale
        tau_task_nominal = J.T @ wrench_scaled
        tau_damping = -self.cfg.kd_joint * qd
        tau_posture = self.cfg.kp_posture * (self._q_rest - q) - self.cfg.kd_posture * qd

        g = np.zeros(6, dtype=np.float64)
        if "gravity_torque" in st and st["gravity_torque"] is not None:
            g = np.asarray(st["gravity_torque"], dtype=np.float64).reshape(6)

        tau_bias = tau_damping + tau_posture + g
        tau_limit = np.asarray(self.cfg.tau_max_nm, dtype=np.float64).reshape(6)
        tau_headroom = np.clip(float(self.cfg.torque_headroom), 0.0, 1.0)
        tau_limit_headroom = tau_limit * max(tau_headroom, 0.0)
        if not np.any(tau_limit_headroom > 0.0):
            tau_limit_headroom = tau_limit.copy()

        tau_nominal = tau_task_nominal + tau_bias
        task_backtrack_scale, tau_preclip, task_backtrack_iters, task_feasible = self._backtrack_task_scale(
            tau_nominal=tau_nominal,
            tau_limit=tau_limit_headroom,
        )

        tau_task = task_backtrack_scale * tau_task_nominal
        tau_damping = task_backtrack_scale * tau_damping
        tau_posture = task_backtrack_scale * tau_posture
        g = task_backtrack_scale * g
        tau = tau_preclip

        tau_clipped = np.clip(tau, -tau_limit, +tau_limit)
        saturated = np.abs(tau - tau_clipped) > 1e-10
        task_scale = float(singular_scale * task_backtrack_scale)

        return CartesianImpedanceOutput(
            tau=tau_clipped,
            tau_preclip=tau_preclip,
            wrench=wrench,
            tau_task_nominal=tau_task_nominal,
            tau_task=tau_task,
            tau_damping=tau_damping,
            tau_posture=tau_posture,
            tau_gravity=g,
            tau_saturated=saturated.astype(np.float64),
            jacobian_cond=cond,
            singular_scale=singular_scale,
            task_backtrack_scale=float(task_backtrack_scale),
            task_scale=task_scale,
            task_backtrack_iters=int(task_backtrack_iters),
            task_feasible=bool(task_feasible),
            x_error=float(x_err),
            y_error=float(y_err),
            z_error=float(z_err),
            orientation_error_vec=e_rot,
            orientation_error_norm=ori_norm,
        )
