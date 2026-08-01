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

# Fixed per-joint weighting for the optional wrist-orientation task term
# (``wrist_orientation_task``, see CartesianImpedanceConfig below). Shape
# matches ``JOINT_NAME_ORDER``. Values are the ``face_mask``/``orientation
# mask`` ratios from the two pre-torque-lane kinematic controllers that
# originated this task-partition idea (``archive/legacy_mujoco/controller.py``
# -- ``split_forearm_origin_face_controller`` (C1) and
# ``differential_ik_split_controller`` (C2), documented in
# ``docs/archive/SLSQP_CONTROLLER_REFERENCE.md``): both use
# ``face_mask = [0, 0, 0, 1.25, 1.55, 1.25]`` over
# ``[shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3]`` --
# exactly zero on the three proximal joints, heavily weighted on the wrist
# chain with wrist_2 (the joint that sits at 0 at the transport singularity)
# weighted highest. Normalized here to a 1.0 peak; only the *shape* is taken
# from the legacy controllers (an overall scale is absorbed by
# ``kp_rot_wrist``/``kd_rot_wrist``), not the literal legacy PD gains.
WRIST_ORIENTATION_MASK: np.ndarray = np.array(
    [0.0, 0.0, 0.0, 1.25 / 1.55, 1.0, 1.25 / 1.55], dtype=np.float64
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
    # Operational-space upgrades (P3). Both default off = historical behavior.
    # With shaping on, the PD wrench is interpreted as a desired task-space
    # acceleration and premultiplied by Lambda(q) = (J M^-1 J^T + eps I)^-1.
    # With nullspace_posture on, the posture PD torque is projected into the
    # task nullspace via the dynamically consistent pseudoinverse so it cannot
    # disturb the end-effector. Both need ``mass_matrix`` in the state dict;
    # if absent, M falls back to identity (kinematic pseudoinverse).
    task_space_inertia_shaping: bool = False
    nullspace_posture: bool = False
    lambda_regularization: float = 1.0e-6
    # Diagonal-only Lambda for the wrench shaping step (default off = historical
    # P3 behavior). Root cause of the Z-drift/orientation-coupling ceiling found
    # 2026-07-25: away from the wrist_2=0 singularity, Lambda = (J M^-1 J^T +
    # eps I)^-1 develops large off-diagonal terms (e.g. X-Z), so the shaped
    # wrench's Z-row picks up Lambda[2,0]*Fx even when z_err is ~0 -- the large
    # X-restoring force leaks into a spurious Z command. With this on, only
    # diag(Lambda) is used to shape the wrench (each task-space channel only
    # responds to its own raw-wrench component), eliminating that leak; the
    # nullspace-posture projector still uses the full (undiagonalized) Lambda,
    # since that math needs the true dynamically-consistent pseudoinverse, and
    # it is a separate, unaffected term (measured healthy across the same dx
    # sweep that exposed the wrench-shaping coupling).
    lambda_diagonal_shaping: bool = False
    # Adaptive lambda_regularization (default off = historical behavior: a
    # single static eps == lambda_regularization everywhere). Root cause found
    # 2026-07-25: a static eps cannot be correct both at the singularity and
    # away from it. At the exact wrist_2=0 singularity (cond(J)~5e16) a small
    # eps makes Lambda's diagonal blow up (measured Lambda[3,3] 9.8->88->670 as
    # eps drops 0.1->0.01->0.001) -- eps=0.1 tames that. But away from the
    # singularity (cond(J)~700, well within the well-conditioned range the arm
    # spends most of a transport move in), that same eps=0.1 corrupts the
    # nullspace-posture projector: at eps=0 the projector correctly nulls a
    # representative posture torque's task-space effect (0.0 leak, matching
    # theory for a full-rank task), but at eps=0.1 it leaks a real, sustained
    # 0.074 rad/s^2 task acceleration (0.015-0.05 rad/s^2 per rotational axis)
    # -- a direct, mechanistic explanation for the orientation drift that
    # eventually trips the safety guard at large X-displacement. With this
    # flag on, eps is interpolated in log(cond(J)) space between
    # lambda_regularization_far (used when cond(J) <= lambda_cond_low, i.e.
    # most of a transport move) and lambda_regularization (used unchanged as
    # the near-singularity ceiling when cond(J) >= lambda_cond_high), instead
    # of being a single fixed value -- but ONLY for the nullspace-posture
    # projector's Lambda. A first attempt also scheduled the wrench-shaping
    # Lambda and caused a real regression (joint velocity >3.0 rad/s in
    # previously-trivial cases at cond(J) values far from the singularity,
    # e.g. alpha=0/dx=0.15-0.20): reducing eps amplifies the wrench-shaping
    # Lambda's diagonal in ways the tuned gains were never validated against.
    # Wrench shaping therefore always uses the static lambda_regularization,
    # unaffected by this flag.
    lambda_adaptive_regularization: bool = False
    lambda_regularization_far: float = 1.0e-4
    lambda_cond_low: float = 1.0e4
    lambda_cond_high: float = 1.0e8
    # Posture re-anchoring (default off = historical behavior). In move+hold
    # trajectories ``_q_rest`` stays the reset pose, so during the hold the
    # posture anchor fights the task force; at a task singularity that force
    # couple pumps unbounded self-motion drift. With this flag on, ``_q_rest``
    # is re-captured once when the x target is reached and the arm has settled
    # (|x_err| <= reanchor_x_tol_m and max|qd| <= reanchor_qd_tol_radps), and
    # re-armed whenever the x target moves on by more than the tolerance.
    posture_reanchor_on_settle: bool = False
    reanchor_x_tol_m: float = 2.0e-3
    reanchor_qd_tol_radps: float = 0.05
    # Wrist-orientation task (default off = historical behavior). Root cause
    # found 2026-07-27/28 (AGENTS.md sec 3, "The ceiling is directional, not
    # just a magnitude limit"): with kp_rot=0 (required -- turning kp_rot back
    # on is unstable near the wrist_2=0 transport singularity through the
    # shared, eps-regularized Lambda-weighted wrench pipeline), orientation is
    # held only as a side effect of the nullspace-posture projector, and that
    # projector's restoring authority is itself asymmetric with wrist_2 sign
    # at the height_alpha=0.5 pose -- not a tunable-gain problem with the
    # existing architecture (a direct kp_posture/kd_posture/kd_joint sweep at
    # the exact failing case barely moved the outcome).
    #
    # This flag adds a SEPARATE joint-space PD torque term that gives
    # orientation its own dedicated authority via the wrist joints only,
    # structurally isolated from the shared Lambda-weighted wrench pipeline
    # that made kp_rot unstable (translated from two pre-torque-lane
    # kinematic controllers that split position and orientation by which
    # joints are responsible for which task -- see WRIST_ORIENTATION_MASK's
    # docstring and archive/legacy_mujoco/controller.py):
    #
    #   tau_orient_wrist = (J_rot.T @ (kp_rot_wrist * e_rot - kd_rot_wrist * omega)) * WRIST_ORIENTATION_MASK
    #
    # Reuses the same orientation_error_vec / omega already computed for the
    # existing (currently zero-gain) kp_rot/kd_rot term -- does not recompute
    # them, and does not touch kp_rot/kd_rot, which remain the historical
    # zero-gain path when this flag is off. This term is summed into the same
    # joint-space bias (tau_damping + tau_posture + ... + gravity) as
    # tau_posture, so it flows through the existing geometric-backtracking
    # and hard-clip logic identically -- no bypass, no special-cased safety
    # path.
    wrist_orientation_task: bool = False
    kp_rot_wrist: float = 0.0
    kd_rot_wrist: float = 0.0
    # Model-based joint-friction feedforward (default off = historical
    # behavior). Added 2026-07-31 to address a real sim-to-real gap: the real
    # UR5e only achieved ~55-72% of a small commanded X displacement with
    # steady-state hold-phase torque that never decayed toward zero -- a
    # friction/stiction signature the (until tonight, frictionless) sim never
    # showed. A same-night sim smoke test confirmed pure proportional gain
    # cannot close this: even with real joint friction added to the MuJoCo
    # model, closed-loop kp_x alone still recovered ~99% of target
    # displacement in sim, far short of the real shortfall -- a disturbance
    # like friction needs either feedforward cancellation or integral action
    # to fully zero out at steady state, not more gain. This term is the
    # feedforward half:
    #
    #   tau_friction_ff = friction_ff_coulomb_nm * tanh(qd / friction_ff_qd_deadband)
    #                     + friction_ff_viscous * qd
    #
    # tanh(qd/eps) is used instead of a hard sign(qd) deliberately: a sign()
    # discontinuity right at qd~=0 -- exactly the hold-phase regime this is
    # meant to fix -- would chatter/oscillate instead of settle. Defaults
    # mirror assets/ur5e_torque/ur5e_torque.xml's own frictionloss/damping
    # values (size3 joints 5.0 Nm / 0.4, size1 joints 1.0 Nm / 0.15) as a
    # starting point, independently tunable since the real robot's true
    # friction need not match the sim model exactly. Summed into the same
    # joint-space bias (tau_damping + tau_posture + ... + gravity) as
    # tau_orient_wrist, so it flows through the existing geometric-
    # backtracking and hard-clip logic identically -- no bypass.
    friction_feedforward: bool = False
    friction_ff_coulomb_nm: np.ndarray = field(
        default_factory=lambda: np.array([5.0, 5.0, 5.0, 1.0, 1.0, 1.0], dtype=np.float64)
    )
    friction_ff_viscous: np.ndarray = field(
        default_factory=lambda: np.array([0.4, 0.4, 0.4, 0.15, 0.15, 0.15], dtype=np.float64)
    )
    # Default 0.05, NOT a smaller value: a same-night sim smoke test found
    # deadband=0.01 produces a genuine closed-loop limit cycle (typical
    # hold-phase |qd| sits ~0.005-0.02 rad/s, right in the tanh term's steep
    # transition region at that setting, acting like a large local negative-
    # damping gain). 0.05 settles cleanly instead -- see
    # config/ur5e_mujoco_torque_osc_tuned_friction_ff.yaml's header for the
    # measured before/after numbers.
    friction_ff_qd_deadband: float = 0.05
    # Y-axis integral action (default off = historical behavior: pure kp_y/
    # kd_y PD, ki_y implicitly 0). Added 2026-08-01 to address the -45 deg
    # base-rotation Y-drift failure (AGENTS.md sec 3): diagnosis (see
    # docs/status/neg45_y_axis_diagnosis_and_fix_2026-08-01.md) ruled out
    # geometric backtracking (task_backtrack_scale measured == 1.0 throughout
    # the failing move -- torque never gets close to tau_limit, peak ~4.7% of
    # headroom), the global cond(J) singular_scale (measured == 1.0 throughout
    # at jacobian_singular_cond_max=1e18), kd_joint (0x and 2x both fail
    # identically to baseline), and the wrench-shaping Lambda's off-diagonal
    # X->Y leak (lambda_diagonal_shaping has zero effect on the Y-drift
    # magnitude, matching the already-documented Phase-2 beam-search result).
    # A direct large-multiplier kp_y/kd_y sweep (not previously tried -- prior
    # attempts topped out around 1.6x-1.7x, both live on real hardware and in
    # the staged gain search) found the true behavior is a genuine steady-
    # state trade-off, not a simple insufficient-bandwidth problem: raising
    # kp_y/kd_y by 5x-10x does stop the Y-drift guard trip, but X tracking
    # then stalls at a *steady-state* equilibrium roughly 45-55% short of the
    # commanded X target (confirmed non-transient -- unchanged from a 1s move
    # to a 3s move + 2s hold) -- i.e. proportional/derivative gain on Y can
    # only trade X authority for Y authority at this pose, never satisfy both,
    # consistent with a persistent kinematic/dynamic X-Y coupling disturbance
    # that a P/D term structurally cannot null at steady state (the same
    # class of problem friction_feedforward already solves for joint
    # friction, just on a different axis and with an integral rather than a
    # feedforward term, since the coupling's exact model isn't known the way
    # friction's is). This flag adds a standard clamped integral term to the
    # existing Fy PD law:
    #
    #   Fy = kp_y*y_err - kd_y*v_y + ki_y*y_integral
    #   y_integral(t) = clip(y_integral(t-1) + y_err*dt, -y_integral_limit_m_s, +y_integral_limit_m_s)
    #
    # Anti-windup clamp mirrors controller_core/lqr_controller.py's existing
    # `_x_integral` pattern (accumulate, then clip every step -- never clip
    # only occasionally). y_integral resets to 0 in reset_from_state(), same
    # lifecycle as _q_rest/_quat0. Not added to _SCHEDULABLE_GAIN_FIELDS
    # (matches kp_rot_wrist/kd_rot_wrist precedent: config/gain-override only,
    # not part of the live gain-scheduling contract tested against
    # transport_metrics.GAIN_FIELDS).
    y_integral_action: bool = False
    ki_y: float = 0.0
    y_integral_limit_m_s: float = 0.02

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
            task_space_inertia_shaping=bool(ctrl.get("task_space_inertia_shaping", False)),
            nullspace_posture=bool(ctrl.get("nullspace_posture", False)),
            lambda_regularization=float(ctrl.get("lambda_regularization", 1.0e-6)),
            lambda_diagonal_shaping=bool(ctrl.get("lambda_diagonal_shaping", False)),
            lambda_adaptive_regularization=bool(ctrl.get("lambda_adaptive_regularization", False)),
            lambda_regularization_far=float(ctrl.get("lambda_regularization_far", 1.0e-4)),
            lambda_cond_low=float(ctrl.get("lambda_cond_low", 1.0e4)),
            lambda_cond_high=float(ctrl.get("lambda_cond_high", 1.0e8)),
            posture_reanchor_on_settle=bool(ctrl.get("posture_reanchor_on_settle", False)),
            reanchor_x_tol_m=float(ctrl.get("reanchor_x_tol_m", 2.0e-3)),
            reanchor_qd_tol_radps=float(ctrl.get("reanchor_qd_tol_radps", 0.05)),
            wrist_orientation_task=bool(ctrl.get("wrist_orientation_task", False)),
            kp_rot_wrist=float(gains.get("kp_rot_wrist", 0.0)),
            kd_rot_wrist=float(gains.get("kd_rot_wrist", 0.0)),
            friction_feedforward=bool(ctrl.get("friction_feedforward", False)),
            friction_ff_coulomb_nm=(
                np.array(
                    [float(ctrl["friction_ff_coulomb_nm"][name]) for name in JOINT_NAME_ORDER],
                    dtype=np.float64,
                )
                if "friction_ff_coulomb_nm" in ctrl
                else np.array([5.0, 5.0, 5.0, 1.0, 1.0, 1.0], dtype=np.float64)
            ),
            friction_ff_viscous=(
                np.array(
                    [float(ctrl["friction_ff_viscous"][name]) for name in JOINT_NAME_ORDER],
                    dtype=np.float64,
                )
                if "friction_ff_viscous" in ctrl
                else np.array([0.4, 0.4, 0.4, 0.15, 0.15, 0.15], dtype=np.float64)
            ),
            friction_ff_qd_deadband=float(ctrl.get("friction_ff_qd_deadband", 0.05)),
            y_integral_action=bool(ctrl.get("y_integral_action", False)),
            ki_y=float(gains.get("ki_y", 0.0)),
            y_integral_limit_m_s=float(ctrl.get("y_integral_limit_m_s", 0.02)),
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
    tau_orient_wrist: np.ndarray
    tau_friction_ff: np.ndarray
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
    inertia_shaping_active: bool = False
    lambda_diagonal_shaping_active: bool = False
    lambda_adaptive_regularization_active: bool = False
    lambda_regularization_effective: float = 0.0
    nullspace_posture_active: bool = False
    mass_matrix_provided: bool = False
    posture_reanchored: bool = False
    wrist_orientation_task_active: bool = False
    friction_feedforward_active: bool = False
    y_integral_action_active: bool = False
    y_integral_value: float = 0.0


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

        Fx = self.cfg.kp_x * x_err + self.cfg.kd_x * (x_vel_des - float(v[0]))
        Fy = self.cfg.kp_y * y_err - self.cfg.kd_y * float(v[1]) + Fy_integral
        Fz = self.cfg.kp_z * z_err - self.cfg.kd_z * float(v[2])

        e_rot = orientation_error_vec_wxyz(quat_ref, quat)
        ori_norm = float(np.linalg.norm(e_rot))
        M = self.cfg.kp_rot * e_rot - self.cfg.kd_rot * omega

        wrench = np.array([Fx, Fy, Fz, M[0], M[1], M[2]], dtype=np.float64)

        # Jacobian conditioning: needed both for singular_scale below and,
        # when lambda_adaptive_regularization is on, to schedule eps.
        cond = float(np.linalg.cond(J))

        # Operational-space terms (P3, flag-gated; default off).
        use_shaping = bool(self.cfg.task_space_inertia_shaping)
        use_nullspace = bool(self.cfg.nullspace_posture)
        use_adaptive_eps = bool(self.cfg.lambda_adaptive_regularization)
        mass_matrix_provided = "mass_matrix" in st and st["mass_matrix"] is not None
        lambda_mat: np.ndarray | None = None
        lambda_mat_nullspace: np.ndarray | None = None
        m_inv: np.ndarray | None = None
        eps_wrench = max(float(self.cfg.lambda_regularization), 0.0)
        eps_effective = eps_wrench
        if use_shaping or use_nullspace:
            if mass_matrix_provided:
                m_mat = np.asarray(st["mass_matrix"], dtype=np.float64).reshape(6, 6)
            else:
                m_mat = np.eye(6, dtype=np.float64)
            m_inv = np.linalg.inv(m_mat)
            # lambda_mat (wrench shaping) always uses the static, previously-
            # validated eps: reducing it destabilizes the shaped wrench itself
            # (measured joint-velocity blowup at cond(J)~1e3-1e4, well short of
            # the exact singularity) -- a separate failure mode from the
            # nullspace-projector leak the adaptive schedule targets. Only the
            # nullspace projector's Lambda is scheduled.
            a_mat = J @ m_inv @ J.T
            lambda_mat = np.linalg.inv(a_mat + eps_wrench * np.eye(6))
            if use_adaptive_eps:
                eps_effective = self._scheduled_lambda_regularization(cond)
                lambda_mat_nullspace = np.linalg.inv(a_mat + eps_effective * np.eye(6))
            else:
                lambda_mat_nullspace = lambda_mat

        singular_scale = 1.0
        if cond > self.cfg.jacobian_singular_cond_max > 0.0:
            singular_scale = float(self.cfg.jacobian_singular_cond_max / cond)
        use_diagonal_shaping = bool(self.cfg.lambda_diagonal_shaping)
        if use_shaping and lambda_mat is not None:
            # Wrench is treated as a desired task acceleration; Lambda maps it
            # to a dynamically consistent task force. The nullspace projector
            # below always uses the full (undiagonalized) lambda_mat -- only
            # the wrench-shaping step is affected by lambda_diagonal_shaping.
            lambda_for_wrench = np.diag(np.diag(lambda_mat)) if use_diagonal_shaping else lambda_mat
            wrench_effective = lambda_for_wrench @ wrench
        else:
            wrench_effective = wrench
        wrench_scaled = wrench_effective * singular_scale
        tau_task_nominal = J.T @ wrench_scaled
        tau_damping = -self.cfg.kd_joint * qd
        tau_posture = self.cfg.kp_posture * (self._q_rest - q) - self.cfg.kd_posture * qd
        if use_nullspace and lambda_mat_nullspace is not None and m_inv is not None:
            # Dynamically consistent nullspace projector: posture torques can
            # no longer produce task-space acceleration. Uses
            # lambda_mat_nullspace (== lambda_mat unless lambda_adaptive_
            # regularization is on), not the wrench-shaping lambda_mat.
            j_bar = m_inv @ J.T @ lambda_mat_nullspace
            nullspace_proj = np.eye(6) - J.T @ j_bar.T
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

        use_friction_feedforward = bool(self.cfg.friction_feedforward)
        tau_friction_ff = np.zeros(6, dtype=np.float64)
        if use_friction_feedforward:
            coulomb = np.asarray(self.cfg.friction_ff_coulomb_nm, dtype=np.float64).reshape(6)
            viscous = np.asarray(self.cfg.friction_ff_viscous, dtype=np.float64).reshape(6)
            deadband = max(float(self.cfg.friction_ff_qd_deadband), 1e-9)
            tau_friction_ff = coulomb * np.tanh(qd / deadband) + viscous * qd

        g = np.zeros(6, dtype=np.float64)
        if "gravity_torque" in st and st["gravity_torque"] is not None:
            g = np.asarray(st["gravity_torque"], dtype=np.float64).reshape(6)

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
            inertia_shaping_active=use_shaping,
            lambda_diagonal_shaping_active=bool(use_shaping and use_diagonal_shaping),
            lambda_adaptive_regularization_active=bool((use_shaping or use_nullspace) and use_adaptive_eps),
            lambda_regularization_effective=float(eps_effective),
            nullspace_posture_active=use_nullspace,
            mass_matrix_provided=bool(mass_matrix_provided),
            posture_reanchored=bool(self._posture_reanchored),
            wrist_orientation_task_active=use_wrist_orientation_task,
            friction_feedforward_active=use_friction_feedforward,
            y_integral_action_active=use_y_integral,
            y_integral_value=float(self._y_integral),
        )
