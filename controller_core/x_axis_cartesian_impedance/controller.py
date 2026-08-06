"""
XAxisCartesianImpedanceController -- the controller class for the X-axis
Cartesian impedance / PD torque law.

Split out of the former single-file ``x_axis_cartesian_impedance.py`` module
(pure structural refactor; see the package ``__init__.py`` for the original
module docstring). ``compute()`` is kept as one large method rather than
force-split into internal helper functions -- its many flags interact in a
single linear sequence (not a clean dispatch), so splitting its internals
would risk a real behavior change; only the CONFIG dataclass, OUTPUT
dataclass, constants, and small parsing helpers were separated into their
own files.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..kinematics_utils import orientation_error_vec_wxyz
from ..state_types import as_impedance_robot_state
from .config import CartesianImpedanceConfig
from .constants import WRIST_ORIENTATION_MASK
from .output import CartesianImpedanceOutput


class XAxisCartesianImpedanceController:
    """Full 6D Cartesian impedance + posture + joint damping (+ optional gravity).

    The task-space wrench is mapped through ``J.T`` and then backtracked if the
    resulting joint torques exceed the configured headroom around the per-joint
    torque limits.
    """

    #: Gain fields a gain-scheduling policy (or any other caller) may update
    #: mid-episode via ``set_gains()``. Must stay in sync with
    #: ``transport_metrics.GAIN_FIELDS`` -- a cross-module test asserts this.
    _SCHEDULABLE_GAIN_FIELDS: tuple[str, ...] = (
        "kp_x", "kd_x", "kp_y", "kd_y", "kp_z", "kd_z",
        "kp_rot", "kd_rot", "kp_posture", "kd_posture", "kd_joint",
    )

    def __init__(self, config: CartesianImpedanceConfig) -> None:
        self.cfg = config
        self._initialized = False
        self._hold_reference_initialized = False
        self._x0 = 0.0
        self._y0 = 0.0
        self._z0 = 0.0
        self._quat0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._q_rest = np.zeros(6, dtype=np.float64)
        self._posture_reanchored = False
        self._x_des_at_anchor = 0.0
        self._y_integral = 0.0
        self._x_integral = 0.0
        # LuGre bristle-deflection state, one scalar per joint. Genuinely
        # stateful (integrated over time), same lifecycle class as
        # _q_rest/_quat0/_y_integral: zeroed here and in reset_from_state(),
        # but deliberately NOT touched by set_gains() -- a gain change must
        # never wipe accumulated stiction state mid-episode (matches
        # set_gains()'s own documented contract).
        self._friction_z = np.zeros(6, dtype=np.float64)
        # Karnopp stick-slip hysteresis latch, one bool per joint. Same lifecycle
        # class as _friction_z: persistent across compute() calls (the hysteresis
        # IS the state -- see karnopp_qd_stick_enter_radps's docstring), zeroed
        # (here: reset to "stuck") in reset_from_state(), not touched by
        # set_gains(). Initialized True (stuck): a fresh episode typically starts
        # from a held/rest pose, and the hysteresis self-corrects within a cycle
        # or two regardless of the initial guess.
        self._karnopp_stuck = np.ones(6, dtype=bool)

    def reset_from_state(self, state: dict[str, Any]) -> None:
        st = as_impedance_robot_state(state)
        ee = np.asarray(st["ee_pos"], dtype=np.float64).reshape(3)
        self._x0 = float(ee[0])
        self._y0 = float(ee[1])
        self._z0 = float(ee[2])
        self._quat0 = np.asarray(st["ee_quat"], dtype=np.float64).reshape(4).copy()
        self._q_rest = np.asarray(st["q"], dtype=np.float64).reshape(6).copy()
        self._hold_reference_initialized = False
        self._posture_reanchored = False
        self._x_des_at_anchor = float(np.asarray(st["ee_pos"], dtype=np.float64).reshape(3)[0])
        self._y_integral = 0.0
        self._x_integral = 0.0
        self._friction_z = np.zeros(6, dtype=np.float64)
        self._karnopp_stuck = np.ones(6, dtype=bool)
        self._initialized = True

    @property
    def initialized(self) -> bool:
        return self._initialized

    def set_gains(self, gains: dict[str, float]) -> None:
        """Overwrite a subset of the 11 scheduled gain fields on ``self.cfg`` in place.

        For a gain-scheduling policy that calls this every step. Does not
        touch ``tau_max_nm``, the P3 flags, or any instance state (``_q_rest``,
        posture-reanchor bookkeeping, hold reference) -- a gain change must
        never reset the posture anchor or hold reference mid-episode.
        """
        unknown = set(gains) - set(self._SCHEDULABLE_GAIN_FIELDS)
        if unknown:
            raise ValueError(f"set_gains got unknown gain field(s): {sorted(unknown)}")
        validated: dict[str, float] = {}
        for key, value in gains.items():
            value = float(value)
            if not np.isfinite(value):
                raise ValueError(f"set_gains: {key} must be finite, got {value!r}")
            validated[key] = value
        for key, value in validated.items():
            setattr(self.cfg, key, value)

    @staticmethod
    def _torque_within_limits(tau: np.ndarray, limit: np.ndarray) -> bool:
        tau = np.asarray(tau, dtype=np.float64).reshape(6)
        limit = np.asarray(limit, dtype=np.float64).reshape(6)
        return bool(np.all(np.abs(tau) <= limit + 1e-12))

    def _scheduled_lambda_regularization(self, cond: float) -> float:
        """Interpolate eps in log(cond(J)) space between the far-field value
        (used when the task is well-conditioned) and ``lambda_regularization``
        (used unchanged as the near-singularity ceiling)."""
        eps_far = max(float(self.cfg.lambda_regularization_far), 0.0)
        eps_near = max(float(self.cfg.lambda_regularization), 0.0)
        cond_low = max(float(self.cfg.lambda_cond_low), 1.0)
        cond_high = max(float(self.cfg.lambda_cond_high), cond_low * (1.0 + 1e-9))
        cond = max(float(cond), 1.0)
        log_frac = (np.log(cond) - np.log(cond_low)) / (np.log(cond_high) - np.log(cond_low))
        log_frac = float(np.clip(log_frac, 0.0, 1.0))
        return eps_far + log_frac * (eps_near - eps_far)

    def _scheduled_wrench_lambda_regularization(self, cond: float) -> float:
        """Same log(cond(J)) interpolation as _scheduled_lambda_regularization,
        applied to the wrench-shaping Lambda instead of the nullspace
        projector's -- see wrench_lambda_adaptive_regularization's docstring
        for why these are kept as two separate schedulers, not one shared
        one."""
        eps_far = max(float(self.cfg.wrench_lambda_regularization_far), 0.0)
        eps_near = max(float(self.cfg.lambda_regularization), 0.0)
        cond_low = max(float(self.cfg.lambda_cond_low), 1.0)
        cond_high = max(float(self.cfg.lambda_cond_high), cond_low * (1.0 + 1e-9))
        cond = max(float(cond), 1.0)
        log_frac = (np.log(cond) - np.log(cond_low)) / (np.log(cond_high) - np.log(cond_low))
        log_frac = float(np.clip(log_frac, 0.0, 1.0))
        return eps_far + log_frac * (eps_near - eps_far)

    def _lugre_step(self, qd: np.ndarray, dt: float) -> np.ndarray:
        """One explicit-Euler update of the LuGre bristle-deflection state.

        Implements the plan's equations exactly
        (docs/hardware/LUGRE_FRICTION_MODEL_PLAN.md sec 1/3.2):

            dz/dt = qd - |qd| * z / g(qd)
            tau_friction = sigma0 * z + sigma1 * dz/dt + sigma2 * qd
            g(qd) = Fc + (Fs - Fc) * exp(-(qd/vs)^2)

        Mutates ``self._friction_z`` in place (persistent per-joint state,
        see reset_from_state()/__init__()). Numerical-stability note (plan
        sec 3.2): explicit Euler on this ODE is only conditionally stable;
        at 500 Hz (dt=0.002s) with physically realistic qd this is expected
        to be fine, but the sim-side validation sweep must check for a
        growing, non-decaying z trace at hold, or a |qd| guard trip that
        wasn't there under the static model -- not special-cased here
        speculatively.
        """
        sigma0 = np.asarray(self.cfg.lugre_sigma0_nm_per_rad, dtype=np.float64).reshape(6)
        sigma1 = np.asarray(self.cfg.lugre_sigma1_nm_s_per_rad, dtype=np.float64).reshape(6)
        sigma2 = np.asarray(self.cfg.lugre_sigma2_nm_s_per_rad, dtype=np.float64).reshape(6)
        fc = np.asarray(self.cfg.lugre_fc_nm, dtype=np.float64).reshape(6)
        fs = np.asarray(self.cfg.lugre_fs_nm, dtype=np.float64).reshape(6)
        vs = np.maximum(np.asarray(self.cfg.lugre_vs_radps, dtype=np.float64).reshape(6), 1e-9)
        g = fc + (fs - fc) * np.exp(-((qd / vs) ** 2))
        g_safe = np.maximum(g, 1e-9)
        z_dot = qd - np.abs(qd) * self._friction_z / g_safe
        self._friction_z = self._friction_z + dt * z_dot
        return sigma0 * self._friction_z + sigma1 * z_dot + sigma2 * qd

    def _karnopp_step(self, qd: np.ndarray, driving_torque: np.ndarray) -> np.ndarray:
        """Karnopp (1985) stick-slip switching friction feedforward.

        See ``karnopp_qd_stick_enter_radps``'s docstring on ``CartesianImpedanceConfig``
        for the full derivation and real-hardware evidence motivating this. Two
        regimes, selected per joint via a velocity hysteresis latch
        (``self._karnopp_stuck``, mutated in place -- persistent state, same
        lifecycle as ``self._friction_z``):

          - Stuck (``|qd|`` below ``karnopp_qd_stick_enter_radps``, or already
            latched stuck and not yet above ``karnopp_qd_stick_exit_radps``):
            **contributes zero feedforward torque.** An earlier version of this
            method set ``tau_stuck = clip(driving_torque, -fs, fs)`` -- intended
            to "cancel" driving_torque, but ``driving_torque`` is built from
            terms (``tau_task_nominal``, ``tau_damping``, ``tau_posture``,
            ``tau_orient_wrist``, ``g``) that are ALSO summed into ``tau_bias``
            separately by the caller, so adding a second (near-)copy of them
            here roughly DOUBLED the commanded torque on any stuck joint --
            confirmed by direct measurement (2026-08-02): full-controller
            output with friction_model="karnopp" came out at exactly 2x the
            "static" model's output for an identical stuck-joint state. Beyond
            that correctness bug, the underlying premise doesn't hold up either:
            real static friction on the physical robot already self-adjusts to
            cancel whatever net torque is applied, up to its real breakaway
            limit -- that IS what "stuck" means -- so sending additional
            feedforward torque changes neither the robot's real behavior (real
            friction just absorbs more) nor the qdd_residual diagnostic gap
            that motivated this feature (that gap reflects an incomplete rigid-
            body DYNAMICS MODEL used for prediction, not insufficient commanded
            torque; fixing it needs a friction-aware model for the predictor,
            not more feedforward torque from the controller). Zero is the only
            value defensible without a fresh, separately-validated design for
            genuine breakaway assistance -- see
            docs/status/karnopp_stiction_friction_model_2026-08-02.md for the
            full trace and reasoning. With this branch at zero, "karnopp"
            reduces to "static"'s own qd==0 behavior for a stuck joint --
            no worse, but no better either; the sliding branch below is
            unaffected and still provides real value.
          - Sliding (latched free, ``|qd|`` above ``karnopp_qd_stick_exit_radps``):
            standard kinetic Coulomb ``lugre_fc_nm`` + viscous
            ``friction_ff_viscous``, same physical quantities and sign
            convention as the static model's own sliding-regime behavior.

        No time integration/ODE is needed for this model (unlike LuGre) -- the
        only state is which regime each joint is latched into, which needs no
        ``dt``. ``driving_torque`` is accepted for interface stability (the
        caller still computes and passes it) but is no longer read here.
        """
        qd = np.asarray(qd, dtype=np.float64).reshape(6)
        driving_torque = np.asarray(driving_torque, dtype=np.float64).reshape(6)
        del driving_torque  # unused -- see docstring; kept as a parameter for interface stability
        enter = np.asarray(self.cfg.karnopp_qd_stick_enter_radps, dtype=np.float64).reshape(6)
        exit_threshold = np.asarray(self.cfg.karnopp_qd_stick_exit_radps, dtype=np.float64).reshape(6)
        fc = np.asarray(self.cfg.lugre_fc_nm, dtype=np.float64).reshape(6)
        viscous = np.asarray(self.cfg.friction_ff_viscous, dtype=np.float64).reshape(6)
        abs_qd = np.abs(qd)

        # Hysteresis update: stuck->free only above exit_threshold, free->stuck
        # only below enter -- the dead zone between the two thresholds holds
        # whatever the previous latch state was, preventing chatter right at a
        # single boundary (the standard fix for a naive one-threshold switch).
        became_free = self._karnopp_stuck & (abs_qd > exit_threshold)
        became_stuck = (~self._karnopp_stuck) & (abs_qd < enter)
        stuck = np.where(became_free, False, self._karnopp_stuck)
        stuck = np.where(became_stuck, True, stuck)
        self._karnopp_stuck = stuck

        tau_stuck = np.zeros(6, dtype=np.float64)
        tau_slide = fc * np.sign(qd) + viscous * qd
        return np.where(stuck, tau_stuck, tau_slide)

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

        if self.cfg.y_coupling_feedforward:
            y_des = y_des - self.cfg.y_coupling_gain * (x_des - self._x0)

        x_err = x_des - float(p[0])
        y_err = y_des - float(p[1])
        z_err = z_des - float(p[2])

        if self.cfg.posture_reanchor_on_settle:
            x_tol = float(self.cfg.reanchor_x_tol_m)
            if self._posture_reanchored and abs(float(x_des) - self._x_des_at_anchor) > x_tol:
                # The x target moved on to a new plateau: re-arm.
                self._posture_reanchored = False
            if (
                not self._posture_reanchored
                and abs(x_err) <= x_tol
                and float(np.max(np.abs(qd))) <= float(self.cfg.reanchor_qd_tol_radps)
            ):
                self._q_rest = q.copy()
                self._x_des_at_anchor = float(x_des)
                self._posture_reanchored = True

        use_y_corridor = self.cfg.y_control_mode == "corridor"
        if use_y_corridor and bool(self.cfg.y_integral_action):
            raise ValueError(
                "y_control_mode='corridor' and y_integral_action are mutually exclusive -- "
                "the integral term would keep accumulating inside the deadband even though "
                "the P-term contributes nothing there; that interaction is unanalyzed."
            )

        use_y_integral = bool(self.cfg.y_integral_action)
        if use_y_integral:
            # dt_s is optional on the state contract (AGENTS.md sec 2); fall
            # back to the 500 Hz direct_torque loop period, this controller's
            # primary real-hardware cadence, if the caller doesn't supply it.
            dt = float(st.get("dt_s", 1.0 / 500.0))
            if not np.isfinite(dt) or dt <= 0.0:
                dt = 1.0 / 500.0
            y_integral_limit = max(float(self.cfg.y_integral_limit_m_s), 0.0)
            self._y_integral += y_err * dt
            if y_integral_limit > 0.0:
                self._y_integral = float(np.clip(self._y_integral, -y_integral_limit, y_integral_limit))
            else:
                self._y_integral = 0.0
        Fy_integral = self.cfg.ki_y * self._y_integral if use_y_integral else 0.0

        use_x_integral = bool(self.cfg.x_integral_action)
        if use_x_integral:
            # Accumulate ONLY while the target is holding still (|x_vel_des|
            # below a small threshold), not during an active move. Found
            # necessary by direct sim testing (2026-08-02): accumulating
            # throughout the move phase too winds the integral up against the
            # large, transient tracking error a fast move naturally has
            # (nothing to do with friction), and that windup then overshoots
            # the target once the hold phase begins -- x_error was measured
            # crossing zero and continuing to grow negative for the rest of
            # the hold, a real closed-loop failure a synthetic single-
            # integrator test could not have caught. This feature exists to
            # close a HOLD-phase steady-state gap (the real evidence is a
            # flat, non-decaying plateau during hold), not to help move-phase
            # tracking, which kp_x/kd_x already handle -- gating to hold-only
            # matches that motivating case exactly and removes the windup
            # path entirely, without needing a reset-on-move-start (frozen,
            # not reset, so a brief blip in x_vel_des doesn't wipe progress).
            x_vel_des_abs = abs(float(x_vel_des))
            if x_vel_des_abs < 1e-4:
                dt = float(st.get("dt_s", 1.0 / 500.0))
                if not np.isfinite(dt) or dt <= 0.0:
                    dt = 1.0 / 500.0
                x_integral_limit = max(float(self.cfg.x_integral_limit_m_s), 0.0)
                self._x_integral += x_err * dt
                if x_integral_limit > 0.0:
                    self._x_integral = float(np.clip(self._x_integral, -x_integral_limit, x_integral_limit))
                else:
                    self._x_integral = 0.0
        Fx_integral = self.cfg.ki_x * self._x_integral if use_x_integral else 0.0

        Fx = self.cfg.kp_x * x_err + self.cfg.kd_x * (x_vel_des - float(v[0])) + Fx_integral
        if use_y_corridor:
            y_soft = max(float(self.cfg.y_soft_limit_m), 0.0)
            y_hard = max(float(self.cfg.y_hard_limit_m), y_soft + 1e-9)
            abs_y_err = abs(y_err)
            if abs_y_err <= y_soft:
                corridor_scale = 0.0
            elif abs_y_err >= y_hard:
                corridor_scale = 1.0
            else:
                t = (abs_y_err - y_soft) / (y_hard - y_soft)
                corridor_scale = float(3.0 * t**2 - 2.0 * t**3)  # smoothstep: C1 at both ends
            Fy = corridor_scale * (self.cfg.y_corridor_kp * y_err - self.cfg.y_corridor_kd * float(v[1]))
        else:
            corridor_scale = 1.0
            Fy = self.cfg.kp_y * y_err - self.cfg.kd_y * float(v[1]) + Fy_integral
        Fz = self.cfg.kp_z * z_err - self.cfg.kd_z * float(v[2])

        e_rot = orientation_error_vec_wxyz(quat_ref, quat)
        ori_norm = float(np.linalg.norm(e_rot))
        M = self.cfg.kp_rot * e_rot - self.cfg.kd_rot * omega

        wrench = np.array([Fx, Fy, Fz, M[0], M[1], M[2]], dtype=np.float64)

        # Jacobian conditioning: needed both for singular_scale below and,
        # when lambda_adaptive_regularization is on, to schedule eps.
        cond = float(np.linalg.cond(J))

        # Base/wrist task split (default off = historical behavior; see
        # split_base_wrist_task's docstring above for the full evidence and
        # rationale). When off, J_task/wrench_task/cond_task are exactly
        # J/wrench/cond (a 6x6 identity substitution), so every computation
        # below that uses them is unchanged arithmetic, not just
        # coincidentally-equal output.
        use_split_base_wrist = bool(self.cfg.split_base_wrist_task)
        use_reduced_task_dims = bool(self.cfg.reduced_task_dims)
        if use_split_base_wrist and use_reduced_task_dims:
            raise ValueError(
                "split_base_wrist_task and reduced_task_dims are mutually exclusive -- "
                "their interaction (column selection AND row selection at once) is untested."
            )
        if use_split_base_wrist:
            # Position rows only, base-joint columns only (shoulder_pan,
            # shoulder_lift, elbow -- JOINT_NAME_ORDER[0:3]); wrist columns
            # stay exactly zero so translation-task torque structurally
            # cannot route through them. The rotational wrench (M) is
            # dropped from this pipeline -- the wrist-only rotation
            # sub-Jacobian is exactly singular at the motivating pose (step-1
            # evidence above), so routing M through it would just relocate
            # the same problem, not fix it. Orientation stays with
            # nullspace_posture (recomputed below against this same reduced
            # J_task) and, if enabled, wrist_orientation_task.
            J_task = np.zeros((3, 6), dtype=np.float64)
            J_task[:, 0:3] = J[0:3, 0:3]
            wrench_task = wrench[0:3].copy()
            cond_task = float(np.linalg.cond(J[0:3, 0:3]))
        elif use_reduced_task_dims:
            # True row selection (S @ J), not a zeroed/masked 6x6 -- see
            # reduced_task_dims' docstring for why a mask would break the
            # Lambda inversion. selected_dims order matches wrench's own
            # [Fx,Fy,Fz,Mx,My,Mz] row order exactly.
            dim_flags = [
                self.cfg.task_dim_x,
                self.cfg.task_dim_y,
                self.cfg.task_dim_z,
                self.cfg.task_dim_rx,
                self.cfg.task_dim_ry,
                self.cfg.task_dim_rz,
            ]
            selected = [i for i, flag in enumerate(dim_flags) if flag]
            if not selected:
                raise ValueError("reduced_task_dims is on but no task_dim_* flag is True -- empty task.")
            J_task = J[selected, :]
            wrench_task = wrench[selected]
            cond_task = float(np.linalg.cond(J_task)) if len(selected) > 1 else float(np.linalg.norm(J_task))
        else:
            J_task = J
            wrench_task = wrench
            cond_task = cond

        locked_flags = [
            self.cfg.task_lock_shoulder_pan,
            self.cfg.task_lock_shoulder_lift,
            self.cfg.task_lock_elbow,
            self.cfg.task_lock_wrist_1,
            self.cfg.task_lock_wrist_2,
            self.cfg.task_lock_wrist_3,
        ]
        if any(locked_flags):
            # Zero the locked joints' columns -- tau_task = J_task^T @ ...
            # then has an exact zero row for each locked joint, so no task
            # force can reach it regardless of wrench_task's value. See
            # task_lock_shoulder_pan's docstring for the full rationale and
            # the known risk of locking too many joints against the task's
            # own dimensionality.
            J_task = J_task.copy()
            J_task[:, locked_flags] = 0.0
            cond_task = float(np.linalg.cond(J_task)) if J_task.shape[0] > 1 else float(np.linalg.norm(J_task))

        # Operational-space terms (P3, flag-gated; default off).
        use_shaping = bool(self.cfg.task_space_inertia_shaping)
        use_nullspace = bool(self.cfg.nullspace_posture)
        use_adaptive_eps = bool(self.cfg.lambda_adaptive_regularization)
        use_accel_ff = bool(self.cfg.acceleration_feedforward)
        if use_accel_ff and use_reduced_task_dims:
            # acceleration_feedforward's accel_ff_vec[0]/[1]/[2] indexing
            # below hardcodes x/y/z as the first 3 wrench_task rows, which
            # only holds for the natural (unselected) or split_base_wrist_task
            # row orders. An arbitrary reduced_task_dims selection (e.g.
            # y,z only) breaks that assumption silently. Not yet made
            # selection-aware -- raise rather than corrupt.
            raise ValueError(
                "acceleration_feedforward + reduced_task_dims is not yet supported: "
                "accel_ff_vec's x/y/z indexing assumes the natural row order."
            )
        mass_matrix_provided = "mass_matrix" in st and st["mass_matrix"] is not None
        lambda_mat: np.ndarray | None = None
        lambda_mat_nullspace: np.ndarray | None = None
        m_inv: np.ndarray | None = None
        use_wrench_adaptive_eps = bool(self.cfg.wrench_lambda_adaptive_regularization)
        eps_wrench = (
            self._scheduled_wrench_lambda_regularization(cond_task)
            if use_wrench_adaptive_eps
            else max(float(self.cfg.lambda_regularization), 0.0)
        )
        eps_effective = eps_wrench
        # acceleration_feedforward alone is also a trigger for this block (not
        # just task_space_inertia_shaping/nullspace_posture): it needs the
        # same Lambda = (J_task M^-1 J_task^T + eps I)^-1 this block already
        # builds for those two flags, reused rather than recomputed.
        if use_shaping or use_nullspace or use_accel_ff:
            if mass_matrix_provided:
                m_mat = np.asarray(st["mass_matrix"], dtype=np.float64).reshape(6, 6)
            else:
                m_mat = np.eye(6, dtype=np.float64)
            m_inv = np.linalg.inv(m_mat)
            # lambda_mat (wrench shaping) uses the static, previously-validated
            # eps by default: reducing it unconditionally destabilizes the
            # shaped wrench itself (measured joint-velocity blowup at
            # cond(J)~1e3-1e4, well short of the exact singularity) -- a
            # separate failure mode from the nullspace-projector leak
            # lambda_adaptive_regularization targets. wrench_lambda_adaptive_
            # regularization (separate flag, see its own docstring) schedules
            # THIS eps by live cond_task instead, with a more conservative
            # far-field floor informed directly by that failure mode. Uses
            # J_task (== J when split_base_wrist_task is off), so a_mat is 3x3
            # in split mode and 6x6 otherwise.
            a_mat = J_task @ m_inv @ J_task.T
            eye_task = np.eye(a_mat.shape[0], dtype=np.float64)
            lambda_mat = np.linalg.inv(a_mat + eps_wrench * eye_task)
            if use_adaptive_eps:
                eps_effective = self._scheduled_lambda_regularization(cond_task)
                lambda_mat_nullspace = np.linalg.inv(a_mat + eps_effective * eye_task)
            else:
                lambda_mat_nullspace = lambda_mat

        # Acceleration feedforward: mass-weighted addition to the task wrench,
        # built from Lambda's own diagonal (the wrench-shaping lambda_mat,
        # NOT lambda_mat_nullspace -- same distinction the wrench-shaping vs.
        # nullspace-projector split above already makes). Graceful no-op
        # (accel_ff_active stays False, wrench_task unchanged) unless a real
        # mass_matrix was supplied this cycle -- see acceleration_feedforward's
        # docstring above for why an identity-matrix fallback here would be
        # the wrong kind of "silent."
        #
        # use_shaping branches which physical quantity wrench_task represents
        # at this point in the pipeline (see the wrench-shaping comment below):
        # when shaping is on, wrench_task is a desired task ACCELERATION and
        # the single lambda_for_wrench @ wrench_task step below converts the
        # combined PD+feedforward acceleration into a force exactly once: add
        # raw target_accel here, not a pre-Lambda-scaled force, or shaping
        # would apply Lambda a second time (an effective Lambda^2 scaling --
        # found and fixed 2026-08-02, confirmed numerically and live in
        # config/ur5e_mujoco_torque_osc_tuned_split_base_wrist_accel_ff.yaml,
        # which sets both flags together). When shaping is off, wrench_task
        # is used directly as a FORCE (no further Lambda multiplication), so
        # the feedforward term must be mass-weighted here to be dimensionally
        # a force contribution.
        accel_ff_active = False
        wrench_accel_ff = np.zeros(3, dtype=np.float64)
        if use_accel_ff and mass_matrix_provided and lambda_mat is not None:
            lambda_diag = np.diag(lambda_mat)
            target_x_accel = float(st.get("target_x_accel", 0.0))
            target_y_accel = float(st.get("target_y_accel", 0.0))
            target_z_accel = float(st.get("target_z_accel", 0.0))
            accel_ff_vec = np.zeros(wrench_task.shape[0], dtype=np.float64)
            if use_shaping:
                accel_ff_vec[0] = target_x_accel
                if wrench_task.shape[0] >= 2:
                    accel_ff_vec[1] = target_y_accel
                if wrench_task.shape[0] >= 3:
                    accel_ff_vec[2] = target_z_accel
            else:
                accel_ff_vec[0] = lambda_diag[0] * target_x_accel
                if wrench_task.shape[0] >= 2:
                    accel_ff_vec[1] = lambda_diag[1] * target_y_accel
                if wrench_task.shape[0] >= 3:
                    accel_ff_vec[2] = lambda_diag[2] * target_z_accel
            wrench_task = wrench_task + accel_ff_vec
            wrench_accel_ff = accel_ff_vec[:3].copy()
            accel_ff_active = True

        singular_scale = 1.0
        if cond_task > self.cfg.jacobian_singular_cond_max > 0.0:
            singular_scale = float(self.cfg.jacobian_singular_cond_max / cond_task)
        use_diagonal_shaping = bool(self.cfg.lambda_diagonal_shaping)
        if use_shaping and lambda_mat is not None:
            # Wrench is treated as a desired task acceleration; Lambda maps it
            # to a dynamically consistent task force. The nullspace projector
            # below always uses the full (undiagonalized) lambda_mat -- only
            # the wrench-shaping step is affected by lambda_diagonal_shaping.
            lambda_for_wrench = np.diag(np.diag(lambda_mat)) if use_diagonal_shaping else lambda_mat
            wrench_effective = lambda_for_wrench @ wrench_task
        else:
            wrench_effective = wrench_task
        wrench_scaled = wrench_effective * singular_scale
        tau_task_nominal = J_task.T @ wrench_scaled
        tau_damping = -self.cfg.kd_joint * qd
        if self.cfg.posture_kp_by_joint is not None or self.cfg.posture_kd_by_joint is not None:
            kp_vec = (
                self.cfg.posture_kp_by_joint
                if self.cfg.posture_kp_by_joint is not None
                else np.full(6, self.cfg.kp_posture, dtype=np.float64)
            )
            kd_vec = (
                self.cfg.posture_kd_by_joint
                if self.cfg.posture_kd_by_joint is not None
                else np.full(6, self.cfg.kd_posture, dtype=np.float64)
            )
            tau_posture = kp_vec * (self._q_rest - q) - kd_vec * qd
        else:
            tau_posture = self.cfg.kp_posture * (self._q_rest - q) - self.cfg.kd_posture * qd
        if use_nullspace and lambda_mat_nullspace is not None and m_inv is not None:
            # Dynamically consistent nullspace projector: posture torques can
            # no longer produce task-space acceleration. Uses
            # lambda_mat_nullspace (== lambda_mat unless lambda_adaptive_
            # regularization is on), not the wrench-shaping lambda_mat, and
            # J_task (== J when split_base_wrist_task is off) so the
            # projector sees the same reduced/rank-3 task the wrench-shaping
            # step used.
            j_bar = m_inv @ J_task.T @ lambda_mat_nullspace
            nullspace_proj = np.eye(6) - J_task.T @ j_bar.T
            tau_posture = nullspace_proj @ tau_posture

        use_wrist_orientation_task = bool(self.cfg.wrist_orientation_task)
        tau_orient_wrist = np.zeros(6, dtype=np.float64)
        if use_wrist_orientation_task:
            # Deliberately NOT part of the shared Lambda-weighted wrench
            # pipeline above -- that pipeline is where kp_rot was found to be
            # unstable near the wrist_2=0 singularity. This is a plain
            # joint-space PD term (computed exactly like tau_posture),
            # reusing e_rot/omega already computed for the wrench's own
            # (currently zero-gain) rotational block, masked to act mostly
            # through the wrist chain -- see WRIST_ORIENTATION_MASK.
            J_rot = J[3:6, :]
            m_wrist = self.cfg.kp_rot_wrist * e_rot - self.cfg.kd_rot_wrist * omega
            tau_orient_wrist = (J_rot.T @ m_wrist) * WRIST_ORIENTATION_MASK

        # Gravity is computed here, BEFORE the friction-feedforward block below,
        # so the karnopp branch's driving_torque can include it -- a pure
        # reordering of two mutually-independent computations (gravity doesn't
        # depend on anything the friction block computes, and the static/lugre
        # friction branches never read g), so this changes no existing output
        # (verified byte-identical for every pre-existing friction_model value).
        g = np.zeros(6, dtype=np.float64)
        if "gravity_torque" in st and st["gravity_torque"] is not None:
            g = np.asarray(st["gravity_torque"], dtype=np.float64).reshape(6)

        use_friction_feedforward = bool(self.cfg.friction_feedforward)
        friction_model_used = str(self.cfg.friction_model) if use_friction_feedforward else "static"
        tau_friction_ff = np.zeros(6, dtype=np.float64)
        if use_friction_feedforward:
            if self.cfg.friction_model == "lugre":
                # dt_s is optional on the state contract (AGENTS.md sec 2);
                # fall back to the 500 Hz direct_torque loop period, this
                # controller's primary real-hardware cadence, matching the
                # same fallback already used by y_integral_action above.
                dt = float(st.get("dt_s", 1.0 / 500.0))
                if not np.isfinite(dt) or dt <= 0.0:
                    dt = 1.0 / 500.0
                tau_friction_ff = self._lugre_step(qd, dt)
            elif self.cfg.friction_model == "karnopp":
                # driving_torque is every other joint-space bias term already
                # computed above this cycle (task torque, damping, posture,
                # wrist-orientation, gravity) -- see _karnopp_step's docstring.
                driving_torque = tau_task_nominal + tau_damping + tau_posture + tau_orient_wrist + g
                tau_friction_ff = self._karnopp_step(qd, driving_torque)
            else:
                coulomb = np.asarray(self.cfg.friction_ff_coulomb_nm, dtype=np.float64).reshape(6)
                viscous = np.asarray(self.cfg.friction_ff_viscous, dtype=np.float64).reshape(6)
                deadband = max(float(self.cfg.friction_ff_qd_deadband), 1e-9)
                tau_friction_ff = coulomb * np.tanh(qd / deadband) + viscous * qd

        tau_bias = tau_damping + tau_posture + tau_orient_wrist + tau_friction_ff + g
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
        tau_orient_wrist = task_backtrack_scale * tau_orient_wrist
        tau_friction_ff = task_backtrack_scale * tau_friction_ff
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
            tau_orient_wrist=tau_orient_wrist,
            tau_friction_ff=tau_friction_ff,
            tau_gravity=g,
            tau_saturated=saturated.astype(np.float64),
            jacobian_cond=cond_task,
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
            inertia_shaping_active=use_shaping,
            lambda_diagonal_shaping_active=bool(use_shaping and use_diagonal_shaping),
            lambda_adaptive_regularization_active=bool((use_shaping or use_nullspace) and use_adaptive_eps),
            lambda_regularization_effective=float(eps_effective),
            nullspace_posture_active=use_nullspace,
            mass_matrix_provided=bool(mass_matrix_provided),
            posture_reanchored=bool(self._posture_reanchored),
            wrist_orientation_task_active=use_wrist_orientation_task,
            friction_feedforward_active=use_friction_feedforward,
            friction_model_used=friction_model_used,
            friction_z=self._friction_z.copy(),
            friction_karnopp_stuck=self._karnopp_stuck.astype(np.float64).copy(),
            y_integral_action_active=use_y_integral,
            y_integral_value=float(self._y_integral),
            x_integral_action_active=use_x_integral,
            x_integral_value=float(self._x_integral),
            split_base_wrist_task_active=use_split_base_wrist,
            y_corridor_scale=corridor_scale,
            acceleration_feedforward_active=accel_ff_active,
            wrench_accel_ff=wrench_accel_ff,
        )
