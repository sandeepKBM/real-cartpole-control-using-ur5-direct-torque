"""Task-space torque allocation via box-constrained QP (torque + velocity limits)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .box_qp import solve_box_qp
from .kinematics_utils import orientation_error_vec_wxyz
from .state_types import as_impedance_robot_state
from .x_axis_cartesian_impedance import CartesianImpedanceConfig, CartesianImpedanceOutput


@dataclass
class TorqueTaskQPConfig(CartesianImpedanceConfig):
    """Extends impedance gains with QP-specific limits."""

    max_joint_velocity_radps: float = 2.5
    posture_regularization: float = 0.35
    velocity_torque_coupling_kp: np.ndarray = field(
        default_factory=lambda: np.array([8.0, 12.0, 12.0, 4.0, 4.0, 4.0], dtype=np.float64)
    )
    velocity_torque_coupling_kd: np.ndarray = field(
        default_factory=lambda: np.array([4.0, 6.0, 6.0, 2.0, 2.0, 2.0], dtype=np.float64)
    )
    enforce_velocity_torque_bounds: bool = True

    @classmethod
    def from_controller_yaml_section(cls, ctrl: dict) -> "TorqueTaskQPConfig":
        base = CartesianImpedanceConfig.from_controller_yaml_section(ctrl)
        safe = ctrl.get("safety", {}) or {}
        return cls(
            kp_x=base.kp_x,
            kd_x=base.kd_x,
            kp_y=base.kp_y,
            kd_y=base.kd_y,
            kp_z=base.kp_z,
            kd_z=base.kd_z,
            kp_rot=base.kp_rot,
            kd_rot=base.kd_rot,
            kp_posture=base.kp_posture,
            kd_posture=base.kd_posture,
            kd_joint=base.kd_joint,
            tau_max_nm=base.tau_max_nm,
            jacobian_singular_cond_max=base.jacobian_singular_cond_max,
            torque_headroom=base.torque_headroom,
            task_resample_factor=base.task_resample_factor,
            task_resample_min_scale=base.task_resample_min_scale,
            task_resample_max_iters=base.task_resample_max_iters,
            max_joint_velocity_radps=float(safe.get("max_joint_velocity_radps", 2.5)),
            posture_regularization=float(ctrl.get("qp_posture_regularization", 0.35)),
            enforce_velocity_torque_bounds=bool(ctrl.get("qp_enforce_velocity_bounds", True)),
        )


def _velocity_implied_torque_bounds(
    q: np.ndarray,
    qd: np.ndarray,
    q_rest: np.ndarray,
    *,
    kp: np.ndarray,
    kd: np.ndarray,
    qd_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Linear PD map: ``qdot_cmd = qd + (tau - kp*e) / kd`` bounded by ``|qdot_cmd| <= qd_max``."""
    q = np.asarray(q, dtype=np.float64).reshape(6)
    qd = np.asarray(qd, dtype=np.float64).reshape(6)
    q_rest = np.asarray(q_rest, dtype=np.float64).reshape(6)
    kp = np.asarray(kp, dtype=np.float64).reshape(6)
    kd = np.asarray(kd, dtype=np.float64).reshape(6)
    qd_lim = abs(float(qd_max))
    bias = kp * (q_rest - q)
    lo = np.full(6, -np.inf, dtype=np.float64)
    hi = np.full(6, np.inf, dtype=np.float64)
    for i in range(6):
        kdi = float(kd[i])
        if kdi < 1.0e-9:
            continue
        lo[i] = float(bias[i] + kdi * (-qd_lim - qd[i]))
        hi[i] = float(bias[i] + kdi * (qd_lim - qd[i]))
        if lo[i] > hi[i]:
            mid = 0.5 * (lo[i] + hi[i])
            lo[i] = mid
            hi[i] = mid
    return lo, hi


class TorqueTaskQPController:
    """Cartesian task torques from a coupled QP with box limits on ``tau``."""

    def __init__(self, config: TorqueTaskQPConfig) -> None:
        self.cfg = config
        self._initialized = False
        self._hold_reference_initialized = False
        self._p0 = np.zeros(3, dtype=np.float64)
        self._quat0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._q_rest = np.zeros(6, dtype=np.float64)
        self._transport_axis_index = 0

    def reset_from_state(self, state: dict[str, Any]) -> None:
        st = as_impedance_robot_state(state)
        p = np.asarray(st["ee_pos"], dtype=np.float64).reshape(3)
        self._p0 = p.copy()
        self._quat0 = np.asarray(st["ee_quat"], dtype=np.float64).reshape(4).copy()
        self._q_rest = np.asarray(st["q"], dtype=np.float64).reshape(6).copy()
        self._transport_axis_index = int(st.get("transport_axis_index", 0))
        self._hold_reference_initialized = False
        self._initialized = True

    @property
    def initialized(self) -> bool:
        return self._initialized

    def _desired_pose(self, st: dict[str, Any], p: np.ndarray, quat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        hold_current_pose = bool(st.get("hold_current_pose", False))
        if hold_current_pose:
            if not self._hold_reference_initialized:
                self._p0 = np.asarray(p, dtype=np.float64).reshape(3).copy()
                self._quat0 = np.asarray(quat, dtype=np.float64).reshape(4).copy()
                self._q_rest = np.asarray(st["q"], dtype=np.float64).reshape(6).copy()
                self._hold_reference_initialized = True
            p_des = self._p0.copy()
            v_des = np.zeros(3, dtype=np.float64)
            quat_ref = self._quat0.copy()
        else:
            axis_idx = int(st.get("transport_axis_index", self._transport_axis_index))
            axis_idx = int(np.clip(axis_idx, 0, 2))
            target_ee_pos = st.get("target_ee_pos")
            target_ee_vel = st.get("target_ee_vel")
            if target_ee_pos is not None:
                p_des = np.asarray(target_ee_pos, dtype=np.float64).reshape(3).copy()
            else:
                p_des = self._p0.copy()
                if axis_idx == 0:
                    p_des[0] = float(st.get("target_x", p_des[0]))
                else:
                    p_des[axis_idx] = float(st.get("target_axis", p_des[axis_idx]))
                for j in range(3):
                    if j != axis_idx:
                        p_des[j] = float(self._p0[j])
            if target_ee_vel is not None:
                v_des = np.asarray(target_ee_vel, dtype=np.float64).reshape(3).copy()
            else:
                v_des = np.zeros(3, dtype=np.float64)
                if axis_idx == 0:
                    v_des[0] = float(st.get("target_x_vel", 0.0))
                else:
                    v_des[axis_idx] = float(st.get("target_axis_vel", 0.0))
            quat_ref = self._quat0.copy()
        return p_des, v_des, quat_ref

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
        jacobian = np.asarray(st["jacobian"], dtype=np.float64).reshape(6, 6)

        p_des, v_des, quat_ref = self._desired_pose(st, p, quat)
        axis_idx = int(np.clip(int(st.get("transport_axis_index", self._transport_axis_index)), 0, 2))
        kp_axis = (self.cfg.kp_x, self.cfg.kp_y, self.cfg.kp_z)
        kd_axis = (self.cfg.kd_x, self.cfg.kd_y, self.cfg.kd_z)

        # Single-axis transport: drive only the selected world axis (e.g. green / Y);
        # hold the two orthogonal axes at the start pose so motion stays parallel.
        forces = np.zeros(3, dtype=np.float64)
        hold_all_cartesian = bool(st.get("hold_all_cartesian_axes", False))
        hold_orthogonal_only = bool(st.get("hold_orthogonal_axes_only", False))
        for j in range(3):
            if hold_orthogonal_only and j == axis_idx:
                continue
            if hold_all_cartesian:
                pos_err_j = float(p_des[j] - p[j])
                vel_err_j = float(v_des[j] - v[j])
                forces[j] = kp_axis[j] * pos_err_j + kd_axis[j] * vel_err_j
            elif j == axis_idx:
                pos_err_j = float(p_des[j] - p[j])
                vel_err_j = float(v_des[j] - v[j])
                forces[j] = kp_axis[j] * pos_err_j + kd_axis[j] * vel_err_j
            else:
                hold_err = float(self._p0[j] - p[j])
                forces[j] = kp_axis[j] * hold_err - kd_axis[j] * float(v[j])

        x_err = float(p_des[0] - p[0])
        y_err = float(p_des[1] - p[1])
        z_err = float(p_des[2] - p[2])
        fx, fy, fz = float(forces[0]), float(forces[1]), float(forces[2])
        e_rot = orientation_error_vec_wxyz(quat_ref, quat)
        ori_norm = float(np.linalg.norm(e_rot))
        m = self.cfg.kp_rot * e_rot - self.cfg.kd_rot * omega
        wrench = np.array([fx, fy, fz, m[0], m[1], m[2]], dtype=np.float64)

        cond = float(np.linalg.cond(jacobian))
        singular_scale = 1.0
        if cond > self.cfg.jacobian_singular_cond_max > 0.0:
            singular_scale = float(self.cfg.jacobian_singular_cond_max / cond)
        wrench_scaled = wrench * singular_scale

        task_weights = np.diag(
            [
                max(self.cfg.kp_x, 1.0e-6),
                max(self.cfg.kp_y, 1.0e-6),
                max(self.cfg.kp_z, 1.0e-6),
                max(self.cfg.kp_rot, 1.0e-6),
                max(self.cfg.kp_rot, 1.0e-6),
                max(self.cfg.kp_rot, 1.0e-6),
            ]
        ).astype(np.float64)
        j_t = jacobian.T
        lam = float(max(self.cfg.posture_regularization, 1.0e-6))
        hessian = 2.0 * (j_t @ task_weights @ jacobian + lam * np.eye(6, dtype=np.float64))

        tau_task_nominal = j_t @ wrench_scaled
        tau_damping = -self.cfg.kd_joint * qd
        tau_posture = self.cfg.kp_posture * (self._q_rest - q) - self.cfg.kd_posture * qd
        gravity = np.zeros(6, dtype=np.float64)
        if st.get("gravity_torque") is not None:
            gravity = np.asarray(st["gravity_torque"], dtype=np.float64).reshape(6)
        tau_des = tau_task_nominal + tau_damping + tau_posture + gravity
        linear = -hessian @ tau_des

        tau_limit = np.asarray(self.cfg.tau_max_nm, dtype=np.float64).reshape(6)
        headroom = float(np.clip(self.cfg.torque_headroom, 0.0, 1.0))
        tau_hi = tau_limit * max(headroom, 1.0e-6)
        tau_lo = -tau_hi

        if self.cfg.enforce_velocity_torque_bounds:
            vel_lo, vel_hi = _velocity_implied_torque_bounds(
                q,
                qd,
                self._q_rest,
                kp=np.asarray(self.cfg.velocity_torque_coupling_kp, dtype=np.float64),
                kd=np.asarray(self.cfg.velocity_torque_coupling_kd, dtype=np.float64),
                qd_max=float(self.cfg.max_joint_velocity_radps),
            )
            tau_lo = np.maximum(tau_lo, vel_lo)
            tau_hi = np.minimum(tau_hi, vel_hi)
            bad = tau_lo > tau_hi
            if np.any(bad):
                mid = 0.5 * (tau_lo + tau_hi)
                tau_lo = np.where(bad, mid, tau_lo)
                tau_hi = np.where(bad, mid, tau_hi)

        tau_qp = solve_box_qp(hessian, linear, tau_lo, tau_hi)
        tau_clipped = np.clip(tau_qp, -tau_limit, +tau_limit)
        saturated = np.abs(tau_qp - tau_clipped) > 1e-10
        task_feasible = bool(np.all(np.abs(tau_qp) <= tau_hi + 1e-9))

        return CartesianImpedanceOutput(
            tau=tau_clipped,
            tau_preclip=tau_qp,
            wrench=wrench,
            tau_task_nominal=tau_task_nominal,
            tau_task=tau_task_nominal,
            tau_damping=tau_damping,
            tau_posture=tau_posture,
            tau_gravity=gravity,
            tau_saturated=saturated.astype(np.float64),
            jacobian_cond=cond,
            singular_scale=singular_scale,
            task_backtrack_scale=1.0,
            task_scale=float(singular_scale),
            task_backtrack_iters=0,
            task_feasible=task_feasible,
            x_error=x_err,
            y_error=y_err,
            z_error=z_err,
            orientation_error_vec=e_rot,
            orientation_error_norm=ori_norm,
        )
