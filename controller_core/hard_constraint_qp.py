"""Task-space torque allocation via QP with a GENUINE hard Cartesian
constraint (not just a soft weighted cost) -- built 2026-08-03 specifically
to close the gap identified in the existing torque_task_qp.py: that
controller only hard-constrains joint torque/velocity via box bounds; every
Cartesian axis (X/Y/Z/orientation) is still a soft, weighted PD-style cost,
exactly like the plain impedance controller. This file adds a REAL
inequality-constrained axis on top of TorqueTaskQPController's machinery.

Physics: to first order (ignoring Jdot@qd, the same omission the rest of
this codebase already has -- see reduced_task_dims' docstring in
x_axis_cartesian_impedance.py, confirmed absent everywhere in this repo),
joint acceleration is qdd = Minv @ (tau - h), so Cartesian acceleration for
row i of J is a_i = J[i,:] @ Minv @ (tau - h) -- LINEAR in tau. That means a
target band on a_i is a genuine linear inequality constraint on the QP
decision variable tau, solvable via constrained_box_qp.solve_constrained_box_qp
(dual ascent, reusing the existing box_qp.solve_box_qp as the inner solve).

Constraint form used here: rather than bounding raw a_y, bound it around the
SAME PD-computed desired acceleration the soft cost already targets
(a_y_des = kp_y*(y0-y) - kd_y*vy), with a tight tolerance band
(hard_y_tolerance_mps2). This makes the QP find the torque that gets Y's
acceleration close to what it's SUPPOSED to be doing, as a hard requirement,
rather than "as close as the weighted sum of competing objectives happens to
land" -- the qualitative difference from every soft-gain approach tried
earlier this session (kp_y/kd_y sweep, y_integral_action,
y_coupling_feedforward), all of which hit the identical wall because none of
them could actually GUARANTEE Y tracked its target if X-tracking demanded
enough torque.

h (the dynamics bias subtracted before mapping tau to acceleration) is
approximated as gravity_torque from state -- the same real, per-cycle
gravity-compensation torque already used everywhere else in this repo.
Coriolis is NOT included (assumed small at the low joint velocities this
whole session has operated at, same approximation the rest of the codebase
already makes) -- flagged explicitly, not silently assumed away.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .constrained_box_qp import solve_constrained_box_qp
from .kinematics_utils import orientation_error_vec_wxyz
from .state_types import as_impedance_robot_state
from .torque_task_qp import TorqueTaskQPConfig, _velocity_implied_torque_bounds
from .x_axis_cartesian_impedance import CartesianImpedanceOutput


@dataclass
class HardYConstraintQPConfig(TorqueTaskQPConfig):
    """TorqueTaskQPConfig + a genuine hard constraint on Y-axis acceleration.

    Default off (hard_y_constraint=False) = byte-identical to
    TorqueTaskQPController (falls through to solve_box_qp with no extra
    constraint rows, verified in tests). hard_y_tolerance_mps2 sizes the
    band around the PD-desired Y acceleration the QP is required to land
    inside, NOT a position-drift tolerance directly -- position tracking of
    that band is only as good as the acceleration target itself, same
    caveat as any acceleration-level task control.
    """

    hard_y_constraint: bool = False
    hard_y_tolerance_mps2: float = 0.05
    dual_sweeps: int = 4
    dual_root_iters: int = 10

    @classmethod
    def from_controller_yaml_section(cls, ctrl: dict) -> "HardYConstraintQPConfig":
        base = TorqueTaskQPConfig.from_controller_yaml_section(ctrl)
        base_kwargs = {f.name: getattr(base, f.name) for f in base.__dataclass_fields__.values()}
        # TorqueTaskQPConfig.from_controller_yaml_section only propagates a
        # fixed subset of CartesianImpedanceConfig fields from `base` (see
        # its own cls(...) call) -- nullspace_posture/lambda_regularization
        # are NOT among them, so `base` always carries the class default
        # (False / 1e-6) regardless of YAML content. Overwrite them by
        # reading straight from ctrl instead of trusting base_kwargs.
        base_kwargs["nullspace_posture"] = bool(ctrl.get("nullspace_posture", False))
        base_kwargs["lambda_regularization"] = float(ctrl.get("lambda_regularization", 1.0e-6))
        return cls(
            **base_kwargs,
            hard_y_constraint=bool(ctrl.get("hard_y_constraint", False)),
            hard_y_tolerance_mps2=float(ctrl.get("hard_y_tolerance_mps2", 0.05)),
            dual_sweeps=int(ctrl.get("dual_sweeps", 4)),
            dual_root_iters=int(ctrl.get("dual_root_iters", 10)),
        )


class HardYConstraintQPController:
    """Same task/cost structure as TorqueTaskQPController, but Y-axis
    acceleration is a genuine hard constraint when hard_y_constraint=True,
    not folded into the soft weighted cost."""

    def __init__(self, config: HardYConstraintQPConfig) -> None:
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
            axis_idx = int(np.clip(int(st.get("transport_axis_index", self._transport_axis_index)), 0, 2))
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
        lam_reg = float(max(self.cfg.posture_regularization, 1.0e-6))
        hessian = 2.0 * (j_t @ task_weights @ jacobian + lam_reg * np.eye(6, dtype=np.float64))

        tau_task_nominal = j_t @ wrench_scaled
        tau_damping = -self.cfg.kd_joint * qd
        tau_posture = self.cfg.kp_posture * (self._q_rest - q) - self.cfg.kd_posture * qd

        # Nullspace-consistent posture projection (mirrors
        # XAxisCartesianImpedanceController's nullspace_posture mechanism,
        # controller_core/x_axis_cartesian_impedance.py lines ~1483-1595,
        # ported here 2026-08-03). Without this, tau_posture above is added
        # directly into tau_des below and this QP's own cost (see module
        # docstring: absent box-bound saturation, the weighted-least-squares
        # solution is exactly tau=tau_des) reproduces it in tau_qp -- i.e.
        # posture torque leaks straight into the task with NO projection at
        # all, the exact bug class already found and fixed for the impedance
        # controller. Flag-gated (nullspace_posture, default False, inherited
        # from CartesianImpedanceConfig) and mass_matrix-gated -- byte-
        # identical to the pre-existing behavior otherwise. Uses the static
        # lambda_regularization eps (no adaptive scheduling ported -- that
        # needs cond_task/lambda_cond_low/high plumbing not yet present in
        # this simpler controller family; a straight static-eps nullspace
        # projector is itself already validated upstream as the pre-adaptive
        # historical default).
        mass_matrix_provided = st.get("mass_matrix") is not None
        if bool(self.cfg.nullspace_posture) and mass_matrix_provided:
            # Project against POSITION rows only (jacobian[0:3,:]), NOT the
            # full 6D jacobian -- found necessary 2026-08-03 after the
            # full-6D version measured a wash (no X-tracking gain, no
            # orientation gain either): this controller's kp_rot=0 means the
            # task wrench itself supplies zero orientation-RESTORING force
            # (only kd_rot*omega damping), so posture-toward-q_rest is the
            # ONLY restoring mechanism available for orientation. Projecting
            # posture out of the full 6D task (as the impedance controller's
            # nullspace_posture does) also projects it out of orientation,
            # removing that sole restoring path -- confirmed by measurement,
            # not just derivation (orientation_error was 0.248 vs 0.245
            # baseline with the full-6D projector, i.e. no help). Position-
            # only mirrors the effective row-selection the impedance
            # controller's best -45/-40deg configs use via reduced_task_dims
            # (xyz+rz), simplified here to xyz only since this controller has
            # no reduced_task_dims/split-task mechanism to select rz alone.
            J_pos = jacobian[0:3, :]
            m_mat_ns = np.asarray(st["mass_matrix"], dtype=np.float64).reshape(6, 6)
            m_inv_ns = np.linalg.inv(m_mat_ns)
            eps_ns = max(float(self.cfg.lambda_regularization), 0.0)
            a_mat_ns = J_pos @ m_inv_ns @ J_pos.T
            eye_task_ns = np.eye(a_mat_ns.shape[0], dtype=np.float64)
            lambda_mat_nullspace = np.linalg.inv(a_mat_ns + eps_ns * eye_task_ns)
            j_bar_ns = m_inv_ns @ J_pos.T @ lambda_mat_nullspace
            nullspace_proj = np.eye(6) - J_pos.T @ j_bar_ns.T
            tau_posture = nullspace_proj @ tau_posture

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
                q, qd, self._q_rest,
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

        a_ineq = None
        b_ineq = None
        hard_y_active = bool(self.cfg.hard_y_constraint) and (st.get("mass_matrix") is not None)
        if hard_y_active:
            m_mat = np.asarray(st["mass_matrix"], dtype=np.float64).reshape(6, 6)
            m_inv = np.linalg.inv(m_mat)
            j_y_minv = jacobian[1, :] @ m_inv  # row vector, a_y = j_y_minv @ (tau - h)
            h_bias = gravity  # approximation: Coriolis omitted, see module docstring
            a_y_bias = float(j_y_minv @ h_bias)
            # a_y = j_y_minv @ tau - a_y_bias  (linear in tau)
            a_y_des = fy  # same PD law the soft cost already targets (kp_y*y_err - kd_y*vy)
            tol = max(float(self.cfg.hard_y_tolerance_mps2), 1.0e-6)
            # a_y <= a_y_des + tol  =>  j_y_minv @ tau <= a_y_des + tol + a_y_bias
            # a_y >= a_y_des - tol  =>  -j_y_minv @ tau <= -(a_y_des - tol) - a_y_bias
            a_ineq = np.vstack([j_y_minv, -j_y_minv])
            b_ineq = np.array(
                [a_y_des + tol + a_y_bias, -(a_y_des - tol) - a_y_bias],
                dtype=np.float64,
            )

        tau_qp, dual_lambda, hard_y_feasible = solve_constrained_box_qp(
            hessian, linear, tau_lo, tau_hi, a_ineq, b_ineq,
            dual_sweeps=self.cfg.dual_sweeps, dual_root_iters=self.cfg.dual_root_iters,
        )
        tau_clipped = np.clip(tau_qp, -tau_limit, +tau_limit)
        saturated = np.abs(tau_qp - tau_clipped) > 1e-10
        task_feasible = bool(np.all(np.abs(tau_qp) <= tau_hi + 1e-9)) and hard_y_feasible

        return CartesianImpedanceOutput(
            tau=tau_clipped,
            tau_preclip=tau_qp,
            wrench=wrench,
            tau_task_nominal=tau_task_nominal,
            tau_task=tau_task_nominal,
            tau_damping=tau_damping,
            tau_posture=tau_posture,
            tau_orient_wrist=np.zeros(6, dtype=np.float64),
            tau_friction_ff=np.zeros(6, dtype=np.float64),
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
