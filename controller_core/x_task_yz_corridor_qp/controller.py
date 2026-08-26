"""
``XTaskYZCorridorQPController`` -- the reduced (X + orientation) task torque
QP with Y/Z corridor HOCBF rows and the manipulability CBF row, all in ONE
solve.

Full formulation, design forks and measured results:
``docs/status/x_task_yz_corridor_qp_2026-08-13.md``. The essentials, because
they have to be readable here too:

1. REDUCED TASK. ``J_reduced = vstack([J[0:1,:], J[3:6,:]])`` (4x6): the
   world-X row and the three orientation rows. Y and Z are excluded by
   CONSTRUCTION. The QP Hessian

       H = 2 (J_reduced^T W J_reduced + posture_regularization I)

   is therefore built from a matrix that has never seen the Y/Z rows, and the
   Tikhonov term is what keeps it full rank on a 4-row task. Absent active
   inequality rows and absent box saturation, the minimizer of
   ``0.5 (tau - tau_des)^T H (tau - tau_des)`` is EXACTLY ``tau_des``, so Y
   and Z have exactly two sources of authority: the small ``tau_yz_soft``
   bias inside ``tau_des``, and the corridor rows at the walls. Nothing else.
   That claim is asserted numerically (byte-identical, not approximately) in
   ``tests/unit/test_x_task_yz_corridor_qp.py``.

2. CORRIDOR HOCBF. For ``h = y_max - y(q)``:

       hdot  = -J_y qd                                (exact)
       hddot = -(Jdot_y qd + J_y qddot)  ~=  -J_y M^-1 (tau - bias)

   dropping ``Jdot_y qd``. That omission is the SAME standing approximation
   ``hard_constraint_qp.py`` states and uses, and it is deliberately not the
   finite-difference curvature ``manipulability_cbf.py`` computes: that
   curvature exists there because ``mu(q)`` is genuinely a curved function of
   ``q``, whereas ``y(q)``'s only nonlinearity beyond the linear term IS the
   dropped ``Jdot_y qd``. Named cost: the barrier's second-order model is
   slightly optimistic when the Jacobian's Y row is rotating fast, so the
   corridor can be transiently overshot by a small amount rather than being a
   hard invariant (measured, see the design doc). Named benefit: no extra
   ``jacobian_fn`` evaluations at all, against an already-tight real-time
   budget. Named deferred item: whether including it changes behavior has NOT
   been measured.

   With the HOCBF condition ``hddot + (a1+a2) hdot + a1 a2 h >= 0`` and
   ``A tau <= b``:

       y_max row:  A = +J_y M^-1
                   b = +J_y M^-1 bias - (a1+a2)(J_y qd) + a1 a2 (y_max - y)
       y_min row:  A = -J_y M^-1
                   b = -J_y M^-1 bias + (a1+a2)(J_y qd) + a1 a2 (y - y_min)

   and the same pair for Z with ``J_z``. ``a1``/``a2`` are
   ``yz_corridor_alpha1``/``yz_corridor_alpha2`` used DIRECTLY as the two
   distinct HOCBF gains -- matching ``manipulability_cbf_constraint_row``'s
   own convention exactly, so the two mechanisms in this one QP read their
   alphas the same way.

3. MANIPULABILITY ROW. ``manipulability_cbf_constraint_row(...)`` called
   directly (never ``manipulability_cbf_filter``, which does its own closed
   single-row solve and is not composable with other rows).

4. ONE SOLVE. All rows are stacked and handed to
   ``solve_constrained_box_qp`` once, with the torque-headroom box
   intersected with ``_velocity_implied_torque_bounds``.

5. EXCLUDED JOINTS (``task_excluded_joints``, default ``(0,)`` =
   shoulder_pan). A 4-row task on 6 joints leaves 2 redundant DOF, and until
   2026-08-13 nothing said which joints were allowed to absorb them -- only
   the soft posture spring, which measurably was not enough (shoulder_pan
   swung 4.3-13.2 deg across the 5-case matrix, at a pose that pins it for
   real wall/base clearance). TWO changes, applied together:

     (a) that joint's COLUMN of ``J_reduced`` is zeroed, so no task torque
         reaches it and -- just as importantly -- the Hessian stops coupling
         it to the other joints;
     (b) its box is pinned shut, ``tau_lo[i] = tau_hi[i] = tau_hold[i]``,
         where ``tau_hold`` is the non-task bias (gravity + posture spring +
         joint damping).

   (b) alone is an exact guarantee but a bad controller: with the column
   still in the Hessian, pinning it makes the QP re-optimize the OTHER five
   joints to make up the lost ``j_x`` projection, which over-drives the
   wrists into the CBF row and diverges (measured). (a) alone is a good
   controller but not a guarantee: damping, posture, ``tau_yz_soft`` and any
   QP deviation under an active row still reach the joint. Together, (a)
   makes ``H``'s off-diagonal coupling to the pinned coordinate exactly zero,
   so the pin costs the free coordinates nothing, and (b) makes the commanded
   torque exactly ``tau_hold[i]`` bit-for-bit, whatever the rows do. Full
   evidence in ``task_excluded_joints``' config docstring.

5. EXCLUDED JOINTS (``task_excluded_joints``, default ``(0,)`` =
   shoulder_pan). A 4-row task on 6 joints leaves 2 redundant DOF, and until
   2026-08-13 nothing said which joints were allowed to absorb them -- only
   the soft posture spring, which measurably was not enough (shoulder_pan
   swung 4.3-13.2 deg across the 5-case matrix, at a pose that pins it for
   real wall/base clearance). Each excluded joint's box is pinned shut,
   ``tau_lo[i] = tau_hi[i] = tau_hold[i]``, where ``tau_hold`` is the
   non-task bias (gravity + posture spring + joint damping). Because
   ``solve_box_qp`` clips every iterate into the box, this is an EXACT
   structural guarantee on the commanded torque, not a tendency -- and
   unlike zeroing that joint's Jacobian column it also lets the QP move the
   REMAINING joints to compensate. Full rationale, the measured
   mechanism-vs-mechanism comparison, and the pose-dependence caveat are in
   ``task_excluded_joints``' config docstring.

6. ORIENTATION HOCBF (2026-08-15, opt-in via ``cfg.orientation_cbf``). Bounds
   ``||orientation_error||^2`` with a fifth row in the SAME QP, rather than
   only tracking it via the 3 rotation task rows above. Full motivation in
   ``config.py``'s ``orientation_cbf`` docstring; the algebra, because it has
   a real subtlety worth having in one place:

       h = theta_max^2 - e^T e            (e = orientation_error_vec_wxyz)
       hdot  = -2 e^T edot
       hddot = -2 edot^T edot - 2 e^T eddot                        (exact)
       eddot ~= jac_rot_ref M^-1 (tau - bias)      (dropping the Jdot term)

   MEASURED, not assumed: ``edot`` is ``R_ref^T @ (J_r qd)``, NOT bare
   ``+-J_r qd`` in the world frame -- ``e`` is expressed in the FIXED
   reference frame ``self._quat0``/``self._R0`` was captured in (the same
   ``e = 2*vec(conj(q_ref)*q_cur)`` construction that already lives at the
   top of ``compute()``), so its rate of change is the world angular velocity
   ROTATED into that frame, not the raw quantity. Finite-differenced directly
   against this repo's own ``orientation_error_vec_wxyz``/Jacobian at ARM_Q0:
   ``R_ref^T @ (J_r qd)`` matches to 1e-7 at ``e=0`` and 1-6% (small-angle
   residual) up to ``||e||=0.21`` rad, while bare ``+J_r qd`` is off by
   20-60% even at ``e=0`` and bare ``-J_r qd`` by 100-200% -- i.e. this is a
   real, load-bearing correction (this pose's tool orientation is far from
   world identity), not a sign nuance. ``jac_rot_ref := self._R0.T @
   jac[3:6, :]`` (the FULL Jacobian's orientation rows, never remapped by
   ``task_frame=="tool"`` -- see that field's docstring) is used everywhere
   the naive derivation would use ``J_r``. Substituting the verified-positive
   ``eddot`` into the HOCBF condition ``hddot + (a1+a2) hdot + a1 a2 h >= 0``
   and solving for ``A tau <= b``:

       A = 2 e^T jac_rot_ref M^-1                                   (1, 6)
       b = -2 edot^T edot + 2 e^T jac_rot_ref M^-1 bias
           + (a1 + a2) hdot + a1 a2 h

   (both terms flip sign relative to a naive ``edot = -J_r qd`` derivation,
   since that assumed sign is itself wrong here). One more row stacked into
   the same ``rows_a``/``rows_b`` list, same single solve as every other
   mechanism in this file.

NOT IMPLEMENTED HERE (see the config docstring): Lambda shaping, nullspace
projection, SCI, friction feedforward, integral action, off-X transport axes.
Coriolis is omitted from the dynamics bias throughout -- the standing
approximation of every torque path in this repo.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from ..constrained_box_qp import solve_constrained_box_qp
from ..kinematics_utils import orientation_error_vec_wxyz, quat_to_rotmat
from ..manipulability_cbf import (
    JacobianDerivativeFn,
    JacobianFn,
    manipulability,
    manipulability_cbf_constraint_row,
    manipulability_directional_curvature,
    manipulability_gradient,
)
from ..state_types import as_impedance_robot_state
from ..torque_task_qp import _velocity_implied_torque_bounds
from .config import XTaskYZCorridorQPConfig
from .output import XTaskYZCorridorQPOutput


class XTaskYZCorridorQPController:
    """Reduced-task (X + orientation) torque QP with a Y/Z corridor."""

    def __init__(
        self,
        config: XTaskYZCorridorQPConfig,
        *,
        jacobian_fn: JacobianFn | None = None,
        jacobian_derivative_fn: JacobianDerivativeFn | None = None,
    ) -> None:
        self.cfg = config
        # Keyword-only and optional, mirroring
        # XAxisCartesianImpedanceController: used ONLY by the manipulability
        # CBF, which needs J at PERTURBED q -- something no snapshot of J(q)
        # on the per-cycle state can supply (and state_types normalizes the
        # state to plain arrays, so a callable could not travel there anyway).
        self._jacobian_fn: JacobianFn | None = jacobian_fn
        # Optional analytic dJ/dq provider for the manipulability CBF gradient.
        # When supplied, grad_mu comes from ONE closed-form kinematic call
        # instead of 2*n finite-difference jacobian_fn evals; when None the
        # gradient is finite-differenced exactly as before (default, unchanged).
        # It MUST be the derivative of the SAME Jacobian jacobian_fn returns.
        self._jacobian_derivative_fn: JacobianDerivativeFn | None = jacobian_derivative_fn
        # Pre-compile the optional Numba QP kernels at construction (before any
        # control cycle), so a real-time loop never pays the one-time JIT cost
        # on its first cycle. No-op if numba is absent (the numpy fallback runs),
        # and disk-cached so this is a fast load after the first ever compile.
        from ..constrained_box_qp import numba_warmup as _numba_warmup

        _numba_warmup()
        self._initialized = False
        self._hold_reference_initialized = False
        self._p0 = np.zeros(3, dtype=np.float64)
        # Runtime view of cfg.task_velocity_rows. Mutable so a phase-switched
        # run flips the drive row between velocity tracking (swing-up) and
        # position tracking (catch) on ONE controller instance -- the switch
        # stays a source switch, not a controller swap, which is the property
        # that made the end-to-end Goal-1 result checkable.
        self.task_velocity_rows: tuple[int, ...] = tuple(
            int(r) for r in (config.task_velocity_rows or ()))
        self._quat0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._q_rest = np.zeros(6, dtype=np.float64)
        #: R_tool snapshotted at reset; identity until then. Only read when
        #: cfg.task_frame == "tool".
        self._R0 = np.eye(3, dtype=np.float64)
        #: Warm-start buffers for the QP solve (cfg.qp_warm_start). The previous
        #: cycle's primal (tau) and dual (lambda); None means "no warm start yet"
        #: (the first cycle after a reset is a cold solve). Reset in
        #: reset_from_state so a re-anchored run never reuses a stale solution.
        self._warm_x: np.ndarray | None = None
        self._warm_lam: np.ndarray | None = None

    # ---------------------------------------------------------------- setup
    def reset_from_state(self, state: dict[str, Any]) -> None:
        st = as_impedance_robot_state(state)
        self._p0 = np.asarray(st["ee_pos"], dtype=np.float64).reshape(3).copy()
        self._quat0 = np.asarray(st["ee_quat"], dtype=np.float64).reshape(4).copy()
        self._q_rest = np.asarray(st["q"], dtype=np.float64).reshape(6).copy()
        self._R0 = quat_to_rotmat(self._quat0)
        self._hold_reference_initialized = False
        self._initialized = True
        # Drop any warm-start solution from a previous run: after a re-anchor the
        # references (and therefore the QP solution) can jump, so seeding from the
        # old solution would be a stale, wrong warm start.
        self._warm_x = None
        self._warm_lam = None

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def y_bounds(self) -> tuple[float, float]:
        """(y_min, y_max) of the corridor, absolute world coordinates."""
        w = float(self.cfg.y_corridor_half_width_m)
        return float(self._p0[1]) - w, float(self._p0[1]) + w

    @property
    def z_bounds(self) -> tuple[float, float]:
        w = float(self.cfg.z_corridor_half_width_m)
        return float(self._p0[2]) - w, float(self._p0[2]) + w

    # ------------------------------------------------------------- internals
    def _task_frames(self, quat: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
        """(R_task, R_corr) for this cycle, or (None, None) in the world frame.

        ``None`` is deliberately not ``np.eye(3)``: it lets every caller SKIP
        the rotation entirely rather than multiply by an identity, which is
        what keeps the world-frame path byte-identical to the pre-2026-08-14
        code rather than merely numerically equal (an identity matmul is not
        guaranteed to reproduce the same floating-point result as no matmul).
        """
        # An explicit constant basis wins over both "world" and "tool", and is
        # used for BOTH the task rows and the corridor row: it does not rotate,
        # so the frozen-vs-live question task_frame_update exists to answer
        # simply does not arise, and the corridor stays the stationary
        # half-space an HOCBF with fixed bounds assumes.
        #
        # WHY THIS EXISTS (measured 2026-08-16, ARM_Q0): the drive axis is
        # hardwired to row 0 (parsing.TRANSPORT_AXIS_ROW), so the frame decides
        # what row 0 physically IS. Under "tool" row 0 is tool X, which at this
        # pose is near-vertical [-0.094 -0.085 +0.992] -- and vertical pivot
        # acceleration exerts ZERO hinge torque at the hanging equilibrium, so
        # the swing-up drove an axis with no authority over the pendulum and
        # tripped the Z corridor in 0.134 s. Neither "world" nor "tool" can name
        # the axis this pose needs (the in-plane horizontal), hence an explicit
        # rotation rather than a third named frame.
        rot_cfg = getattr(self.cfg, "task_rotation", None)
        if rot_cfg is not None:
            r_fixed = np.asarray(rot_cfg, dtype=np.float64).reshape(3, 3)
            return r_fixed, r_fixed
        if str(getattr(self.cfg, "task_frame", "world")).lower() != "tool":
            return None, None
        mode = str(getattr(self.cfg, "task_frame_update", "frozen")).lower()
        r_live = quat_to_rotmat(np.asarray(quat, dtype=np.float64).reshape(4))
        r_task = r_live if mode in ("live", "hybrid") else self._R0
        r_corr = self._R0 if mode in ("frozen", "hybrid") else r_live
        return r_task, r_corr

    def _desired_task(
        self,
        st: dict[str, Any],
        p: np.ndarray,
        quat: np.ndarray,
        axes: tuple[int, ...],
        *,
        rot: np.ndarray | None = None,
        p0: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Desired position and velocity for each TRACKED translation axis.

        ``hold_current_pose`` re-anchors the whole reference (position,
        orientation, posture) ONCE, matching TorqueTaskQPController's own
        settle-phase behavior -- the MuJoCo harness sets that flag during a
        settle phase and expects the controller to hold where it actually is,
        not where it started.

        Generalized 2026-08-13 from the X-only ``_desired_x``. The X branch is
        UNCHANGED, expression for expression, so a default ``task_axis_rows =
        (0,)`` config produces bit-identical numbers. For a non-X tracked axis
        the reference is ``target_ee_pos``/``target_ee_vel`` when the caller
        supplies them (the MuJoCo transport harness does, and it fills the
        untransported components with the START pose -- which is exactly
        "hold this axis where it began" and is what tracking Z means here),
        falling back to the captured ``p0`` and zero velocity otherwise.
        There is deliberately no ``target_y``/``target_z`` scalar fallback to
        invent: the ``target_x``/``target_x_vel`` scalars exist for the
        transport axis only.
        """
        if bool(st.get("hold_current_pose", False)):
            if not self._hold_reference_initialized:
                self._p0 = np.asarray(p, dtype=np.float64).reshape(3).copy()
                self._quat0 = np.asarray(quat, dtype=np.float64).reshape(4).copy()
                self._q_rest = np.asarray(st["q"], dtype=np.float64).reshape(6).copy()
                self._hold_reference_initialized = True
            held = self._p0 if rot is None else rot.T @ self._p0
            return (
                np.array([float(held[a]) for a in axes], dtype=np.float64),
                np.zeros(len(axes), dtype=np.float64),
            )
        p0_f = self._p0 if p0 is None else np.asarray(p0, dtype=np.float64).reshape(3)
        target_ee_pos = st.get("target_ee_pos")
        if target_ee_pos is not None:
            tp = np.asarray(target_ee_pos, dtype=np.float64).reshape(3)
            if rot is not None:
                tp = rot.T @ tp
            des = np.array([float(tp[a]) for a in axes], dtype=np.float64)
        else:
            des = np.array(
                [
                    float(st.get("target_x", p0_f[0])) if a == 0 else float(p0_f[a])
                    for a in axes
                ],
                dtype=np.float64,
            )
        target_ee_vel = st.get("target_ee_vel")
        if target_ee_vel is not None:
            tv = np.asarray(target_ee_vel, dtype=np.float64).reshape(3)
            if rot is not None:
                tv = rot.T @ tv
            vel = np.array([float(tv[a]) for a in axes], dtype=np.float64)
        else:
            vel = np.array(
                [float(st.get("target_x_vel", 0.0)) if a == 0 else 0.0 for a in axes],
                dtype=np.float64,
            )
        return des, vel

    @staticmethod
    def _corridor_rows(
        *,
        j_row: np.ndarray,
        m_inv: np.ndarray,
        bias: np.ndarray,
        qd: np.ndarray,
        value: float,
        lower: float,
        upper: float,
        alpha1: float,
        alpha2: float,
    ) -> tuple[np.ndarray, float, np.ndarray, float]:
        """Both HOCBF rows for one Cartesian axis: (A_max, b_max, A_min, b_min).

        ``j_row`` is that axis's row of the FULL Jacobian (so ``j_row @ qd``
        is the axis's Cartesian velocity, exactly). Static and pure so the
        algebra and sign convention can be unit-tested without a controller
        or a simulator, the same call ``manipulability_cbf_constraint_row``
        makes.
        """
        j_row = np.asarray(j_row, dtype=np.float64).reshape(-1)
        lie = j_row @ np.asarray(m_inv, dtype=np.float64)  # row multiplying tau
        lie_bias = float(lie @ np.asarray(bias, dtype=np.float64).reshape(-1))
        v_axis = float(j_row @ np.asarray(qd, dtype=np.float64).reshape(-1))
        a_sum = float(alpha1) + float(alpha2)
        a_prod = float(alpha1) * float(alpha2)
        # h_max = upper - value ; hdot = -v_axis ; hddot ~= -lie @ (tau - bias)
        a_max = lie.reshape(1, -1)
        b_max = lie_bias - a_sum * v_axis + a_prod * (float(upper) - float(value))
        # h_min = value - lower ; hdot = +v_axis ; hddot ~= +lie @ (tau - bias)
        a_min = -lie.reshape(1, -1)
        b_min = -lie_bias + a_sum * v_axis + a_prod * (float(value) - float(lower))
        return a_max, float(b_max), a_min, float(b_min)

    @staticmethod
    def _orientation_cbf_row(
        *,
        e: np.ndarray,
        jac_rot_ref: np.ndarray,
        m_inv: np.ndarray,
        bias: np.ndarray,
        qd: np.ndarray,
        max_error_rad: float,
        alpha1: float,
        alpha2: float,
    ) -> tuple[np.ndarray, float, float, float, np.ndarray]:
        """``(A, b, h, hdot, edot)`` for the orientation HOCBF row.

        ``jac_rot_ref`` MUST already be ``self._R0.T @ jac[3:6, :]`` -- the
        ``R_ref^T`` rotation is not optional here, see the module docstring's
        item 6 and ``config.py``'s ``orientation_cbf`` docstring for the
        finite-difference evidence. Static and pure, mirroring
        ``_corridor_rows``/``manipulability_cbf_constraint_row``, so the
        algebra and the measured sign convention can be unit-tested without a
        controller or a simulator.
        """
        e = np.asarray(e, dtype=np.float64).reshape(3)
        jac_rot_ref = np.asarray(jac_rot_ref, dtype=np.float64).reshape(3, -1)
        m_inv = np.asarray(m_inv, dtype=np.float64)
        bias = np.asarray(bias, dtype=np.float64).reshape(-1)
        qd = np.asarray(qd, dtype=np.float64).reshape(-1)

        edot = jac_rot_ref @ qd  # R_ref^T @ (J_r qd), verified positive sign
        row = e @ jac_rot_ref  # e^T @ jac_rot_ref, shape (n,)
        lie = row @ m_inv  # e^T @ jac_rot_ref @ M^-1, shape (n,)
        lie_bias = float(lie @ bias)

        h_val = float(max_error_rad) ** 2 - float(e @ e)
        hdot = -2.0 * float(e @ edot)
        a_sum = float(alpha1) + float(alpha2)
        a_prod = float(alpha1) * float(alpha2)

        a_row = (2.0 * lie).reshape(1, -1)
        b_scalar = -2.0 * float(edot @ edot) + 2.0 * lie_bias + a_sum * hdot + a_prod * h_val
        return a_row, float(b_scalar), h_val, hdot, edot

    # -------------------------------------------------------------- compute
    def compute(self, state: dict[str, Any]) -> XTaskYZCorridorQPOutput:
        if not self._initialized:
            raise RuntimeError("Call reset_from_state() before compute().")
        st = as_impedance_robot_state(state)

        axis_idx = int(st.get("transport_axis_index", 0) or 0)
        if axis_idx != 0:
            # Loud rather than ignored: this controller's task Jacobian is
            # hardcoded to J's world-X row, so honoring an off-X request would
            # mean transporting along X while the caller's targets, guards and
            # drift tolerances all describe another axis -- exactly the silent
            # axis mismatch transport_axis_index exists to remove.
            raise ValueError(
                "XTaskYZCorridorQPController is world-X only: the reduced task Jacobian "
                f"is J[0:1,:] + J[3:6,:] by construction, but the state requests "
                f"transport_axis_index={axis_idx}. Use the impedance controller family for "
                "off-X transport."
            )

        q = np.asarray(st["q"], dtype=np.float64).reshape(6)
        qd = np.asarray(st["qd"], dtype=np.float64).reshape(6)
        p = np.asarray(st["ee_pos"], dtype=np.float64).reshape(3)
        quat = np.asarray(st["ee_quat"], dtype=np.float64).reshape(4)
        v = np.asarray(st["ee_lin_vel"], dtype=np.float64).reshape(3)
        omega = np.asarray(st["ee_ang_vel"], dtype=np.float64).reshape(3)
        jac = np.asarray(st["jacobian"], dtype=np.float64).reshape(6, 6)

        use_corridor = bool(self.cfg.yz_corridor_enabled)
        use_manip_cbf = bool(self.cfg.manipulability_cbf)
        use_orientation_cbf = bool(self.cfg.orientation_cbf)
        mass_matrix_provided = st.get("mass_matrix") is not None
        if use_manip_cbf and self._jacobian_fn is None:
            raise ValueError(
                "manipulability_cbf is on but this controller was constructed without "
                "jacobian_fn -- grad_mu needs J at perturbed q, and the per-cycle state "
                "carries J(q) at the current q only. Pass "
                "XTaskYZCorridorQPController(cfg, jacobian_fn=...) (the MuJoCo adapter's "
                "build_controller() does this for you)."
            )
        if (use_manip_cbf or use_corridor or use_orientation_cbf) and not mass_matrix_provided:
            raise ValueError(
                "yz_corridor_enabled/manipulability_cbf/orientation_cbf is on but the "
                "per-cycle state has no mass_matrix -- every constraint row here is built "
                "from J M^-1, so without M there is no constraint to enforce. Substituting "
                "identity would silently enforce a barrier for a robot this is not."
            )

        # Which translation axes are tracked, which are bounded. Validated as a
        # pair at config-parse time (disjoint, X always tracked); re-read here
        # rather than cached so a caller that mutates cfg between cycles -- the
        # diagnostic harness does exactly that for the corridor half-widths --
        # cannot silently desync.
        task_axes = tuple(int(a) for a in (self.cfg.task_axis_rows or (0,)))
        corridor_axes = tuple(int(a) for a in (self.cfg.corridor_axis_rows or ()))

        # --- 1.0 task frame ---------------------------------------------- #
        # `task_axis_rows`/`corridor_axis_rows` index rows of the POSITION
        # block. Under task_frame == "tool" that block is pre-rotated by
        # R_tool^T so the rows become tool X/Y/Z; the three orientation rows
        # are left untouched (already body-referenced via the quaternion
        # error). Every position quantity that feeds a row -- p, v, p0 and the
        # targets -- is rotated by the SAME R as the row it drives, or the
        # signs would not match.
        #
        # R_tool is time-varying, so which R each consumer sees is a real
        # choice (see task_frame_update's docstring): "live" tracks the true
        # hinge, "frozen" keeps the barrier a stationary half-space, "hybrid"
        # is live for the task rows and frozen for the corridor row.
        #
        # world frame => R is None => every branch below is SKIPPED, so the
        # default path is untouched expression-for-expression and stays
        # byte-identical.
        r_task, r_corr = self._task_frames(quat)

        def _rot(r, vec):
            return vec if r is None else r.T @ vec

        jac_task = jac if r_task is None else np.vstack([r_task.T @ jac[0:3, :], jac[3:6, :]])
        jac_corr = jac if r_corr is None else np.vstack([r_corr.T @ jac[0:3, :], jac[3:6, :]])
        p_task, v_task = _rot(r_task, p), _rot(r_task, v)
        p_corr, v_corr = _rot(r_corr, p), _rot(r_corr, v)
        p0_task, p0_corr = _rot(r_task, self._p0), _rot(r_corr, self._p0)

        pos_des, vel_des = self._desired_task(st, p, quat, task_axes, rot=r_task, p0=p0_task)

        # --- 1.1 reduced task ------------------------------------------- #
        # Row selection, not a mask: the tracked translation rows stacked on
        # the three orientation rows. The default (0,) reproduces the original
        # `vstack([jac[0:1,:], jac[3:6,:]])` exactly -- same rows, same order,
        # same values -- which is asserted byte-identically in the unit tests.
        j_reduced = np.vstack([jac_task[list(task_axes), :], jac_task[3:6, :]])  # (len+3, 6)

        # Excluded joints, part (a): zero their columns. Two consequences,
        # BOTH of which are wanted (see the module docstring's item 5):
        #   * tau_task_nominal = J_reduced.T @ wrench has an exact zero row
        #     for each excluded joint -- the same construction
        #     split_base_wrist_active_joints uses in the impedance controller;
        #   * H = 2(J_reduced.T W J_reduced + reg I) becomes exactly diagonal
        #     in each excluded coordinate (its only remaining entry there is
        #     the Tikhonov 2*reg). That is what makes part (b)'s pin free:
        #     with H[free, excluded] == 0 the pinned coordinate no longer
        #     couples into the free ones, so the QP holds the free joints at
        #     tau_des instead of over-driving them to make up the lost task
        #     projection.
        excluded_joints = tuple(int(i) for i in (self.cfg.task_excluded_joints or ()))
        # Keep the UN-excluded rows for the Lambda (task-space inertia) model.
        # Lambda is a property of the MECHANISM -- M and the true J -- not of which
        # joints we have chosen to let act. Building it from the column-zeroed
        # Jacobian inverts a rank-deficient matrix and produces numerical garbage:
        # measured at ARM_Q0 with the default task_excluded_joints=(0,),
        #     cond(J M^-1 J^T + eps I) = 236 (no exclusion) -> 184195 (excluded)
        #     Lambda_zz                =   2.73             ->   9915
        # which turned a wrench of [144, 0, 0, 0] into [1689, -1070, -975, 9820].
        # The exclusion still applies in full to the TORQUE mapping below
        # (j_reduced.T @ wrench and the Hessian), which is where it belongs.
        j_reduced_full = j_reduced
        if excluded_joints:
            j_reduced = j_reduced.copy()
            j_reduced[:, list(excluded_joints)] = 0.0

        # Per-axis translation gains, indexed by world row. kp_y/kd_y and
        # kp_z/kd_z do DOUBLE DUTY by design: as the soft centering bias for an
        # axis in corridor_axis_rows, and as genuine task gains for the same
        # axis in task_axis_rows. The role follows the row set, and the two
        # sets are disjoint, so a given axis only ever reads them one way --
        # but the VALUES appropriate to the two roles are completely different
        # (a soft bias is deliberately ~1e-2 of a task gain), which is why the
        # X+Z config re-derives them rather than inheriting.
        kp_axis = (float(self.cfg.kp_x), float(self.cfg.kp_y), float(self.cfg.kp_z))
        kd_axis = (float(self.cfg.kd_x), float(self.cfg.kd_y), float(self.cfg.kd_z))

        # VELOCITY ROWS: drop the position term, leaving kd*(vel_des - v).
        # self.task_velocity_rows is a RUNTIME view of cfg.task_velocity_rows so
        # a phase-switched run can turn it on for the swing-up and off for the
        # catch without rebuilding the controller (cfg is frozen). See the
        # config field's docstring for why this is a row mode rather than a
        # separate velocity controller.
        vel_rows = frozenset(int(r) for r in (self.task_velocity_rows or ()))
        f_task = np.array(
            [
                (0.0 if a in vel_rows
                 else kp_axis[a] * float(pos_des[i] - p_task[a]))
                + kd_axis[a] * float(vel_des[i] - v_task[a])
                for i, a in enumerate(task_axes)
            ],
            dtype=np.float64,
        )
        x_err = float(pos_des[task_axes.index(0)] - p_task[0])
        e_rot = orientation_error_vec_wxyz(self._quat0, quat)
        ori_norm = float(np.linalg.norm(e_rot))
        m_rot = self.cfg.kp_rot * e_rot - self.cfg.kd_rot * omega
        wrench_reduced = np.concatenate([f_task, m_rot.astype(np.float64)])

        # --- OPTIONAL task-space inertia shaping (2026-08-15, default OFF) ----
        # Without this the wrench above is a raw FORCE, and force along an axis is
        # NOT acceleration along that axis whenever Lambda is non-diagonal.
        # Measured on the compiled model with pendulum_attachment_realrod.xml, a
        # pure +X unit force produces an end-effector acceleration pointing:
        #     ARM_Q0 (wrist_2 ~ 0):   [ 0.9172, -0.0727, -0.3918] -> 23.48 deg off X
        #     wrist_2 = -90 deg:      [ 0.9711, -0.2343,  0.0455] -> 13.81 deg off X
        # because Lambda carries large off-diagonals (Lambda_xz = 1.659 against
        # Lambda_xx = 7.419 at ARM_Q0; Lambda_xy = 0.997 against 4.821 at -90).
        # So pushing X MANUFACTURES the very Y/Z motion the corridor rows then
        # fight -- the barriers end up throttling X to suppress a side effect of X.
        # This is NOT a kinematic limit: continuation IK reaches dx = 0.400 m with
        # |dy|,|dz| held inside 0.03 m, and pinv(J) @ xhat gives exactly [1, 0, 0].
        #
        # With shaping on, kp_*/kd_* become ACCELERATION gains (the same semantic
        # change task_space_inertia_shaping already makes in the sibling
        # x_axis_cartesian_impedance controller, which is why the OSC configs that
        # set it are Lambda-compensated and this controller was not).
        #
        # Lambda is built on the REDUCED rows only -- the 4x4 task actually being
        # commanded -- not the full 6x6, so it inverts exactly the map this
        # controller uses. Default False keeps every existing config bit-for-bit.
        lambda_reduced = None
        if bool(getattr(self.cfg, "task_space_inertia_shaping", False)):
            if not mass_matrix_provided:
                raise ValueError(
                    "task_space_inertia_shaping is on but the per-cycle state has no "
                    "mass_matrix; Lambda cannot be formed. build_mujoco_state() supplies "
                    "it -- pass it, or turn the flag off."
                )
            # Local inversion: the shared m_inv further down in compute() is
            # computed AFTER this point (it is only needed by the constraint rows),
            # so referencing it here would NameError the moment the flag is on.
            m_inv_task = np.linalg.inv(
                np.asarray(st["mass_matrix"], dtype=np.float64).reshape(6, 6)
            )
            eps = float(max(getattr(self.cfg, "lambda_regularization", 0.1), 1.0e-9))
            # j_reduced_full, NOT the column-zeroed j_reduced -- see the
            # exclusion block above for the measured blow-up this avoids.
            j_m_inv = j_reduced_full @ m_inv_task
            lambda_reduced = np.linalg.inv(
                j_m_inv @ j_reduced_full.T
                + eps * np.eye(j_reduced_full.shape[0], dtype=np.float64)
            )
            if bool(getattr(self.cfg, "lambda_diagonal_shaping", False)):
                # Diagonalize for the wrench step only: keeps the per-axis inertia
                # scaling while removing the cross-axis leak that is the whole
                # point of this block.
                lambda_reduced = np.diag(np.diag(lambda_reduced))
            wrench_reduced = lambda_reduced @ wrench_reduced

        cond = float(np.linalg.cond(jac))
        singular_scale = 1.0
        if cond > self.cfg.jacobian_singular_cond_max > 0.0:
            singular_scale = float(self.cfg.jacobian_singular_cond_max / cond)
        wrench_scaled = wrench_reduced * singular_scale

        # W = diag(gain) encodes each row's PRIORITY when a CBF binds. For a
        # velocity row kp is not applied at all, so kp would read as ~0 priority
        # and the QP would sacrifice the drive axis to any competing row. Use
        # that row's kd, which is the gain it is actually being driven by.
        _w_diag = [max(kd_axis[a] if a in vel_rows else kp_axis[a], 1.0e-6)
                   for a in task_axes] + [
            max(self.cfg.kp_rot, 1.0e-6)
        ] * 3
        if lambda_reduced is not None:
            # HESSIAN/LINEAR CONSISTENCY (2026-08-15). H = 2(J^T W J + reg I) sets
            # how the QP trades off task rows against each other when a corridor or
            # CBF row binds; W = diag(kp_*) encodes their relative PRIORITY. With
            # Lambda shaping on, kp_* changes meaning from a force-domain stiffness
            # to an ACCELERATION gain, so the raw numbers no longer express the same
            # relative priority -- a 4-row task mixing one translation gain with
            # three rotation gains gets its trade-off silently re-weighted by
            # whatever Lambda's diagonal happens to be.
            #
            # Rescaling by diag(Lambda) restores the EFFECTIVE force-domain
            # stiffness each row actually applies through the shaped wrench, so the
            # Hessian and the linear term describe the same task in the same units.
            # Inert while no inequality row is active (the unconstrained minimizer is
            # tau_des regardless of H), which is exactly why this was invisible
            # before -- and exactly why it matters for a corridor/CBF experiment.
            _lam_d = np.abs(np.diag(lambda_reduced))
            _w_diag = [
                w * float(max(_lam_d[i], 1.0e-9)) for i, w in enumerate(_w_diag)
            ]
        task_weights = np.diag(_w_diag).astype(np.float64)
        lam_reg = float(max(self.cfg.posture_regularization, 1.0e-6))
        hessian = 2.0 * (
            j_reduced.T @ task_weights @ j_reduced + lam_reg * np.eye(6, dtype=np.float64)
        )
        tau_task_nominal = j_reduced.T @ wrench_scaled

        # --- soft centering of the CORRIDOR axes (linear term only, never
        #     the Hessian) ------------------------------------------------ #
        # Applied to corridor_axis_rows only. An axis promoted to a task row
        # gets its authority from the task; adding the bias on top would
        # double-count it and, worse, do so with a gain deliberately sized to
        # be negligible. y_error/z_error are still REPORTED for both, since a
        # trace consumer wants the distance from the start pose regardless of
        # which mechanism is holding it.
        y0, z0 = float(p0_corr[1]), float(p0_corr[2])
        y_err = y0 - float(p_corr[1])
        z_err = z0 - float(p_corr[2])
        f_soft = (
            0.0,
            float(self.cfg.kp_y) * y_err - float(self.cfg.kd_y) * float(v_corr[1]),
            float(self.cfg.kp_z) * z_err - float(self.cfg.kd_z) * float(v_corr[2]),
        )
        if corridor_axes:
            soft_terms = [jac_corr[a, :] * f_soft[a] for a in corridor_axes]
            tau_yz_soft = soft_terms[0]
            for term in soft_terms[1:]:
                tau_yz_soft = tau_yz_soft + term
        else:
            tau_yz_soft = np.zeros(6, dtype=np.float64)

        tau_damping = -self.cfg.kd_joint * qd
        tau_posture = self.cfg.kp_posture * (self._q_rest - q) - self.cfg.kd_posture * qd
        if self.cfg.posture_joint_weights is not None:
            # Per-joint scaling of the posture spring AND damper together, so a
            # weighted joint stays critically-damped relative to its own
            # stiffness rather than becoming springy. Weight 1.0 reproduces the
            # unweighted term exactly; the branch is skipped entirely when the
            # field is None, so existing configs are bit-for-bit unchanged.
            tau_posture = tau_posture * np.asarray(
                self.cfg.posture_joint_weights, dtype=np.float64
            ).reshape(6)
        gravity = np.zeros(6, dtype=np.float64)
        if st.get("gravity_torque") is not None:
            gravity = np.asarray(st["gravity_torque"], dtype=np.float64).reshape(6)

        # `tau_hold` is the NON-task part of tau_des: gravity compensation plus
        # the joint-space posture spring and damper. It is what an excluded
        # joint is pinned to below -- see task_excluded_joints' docstring for
        # why the pin is against this and not against 0.0.
        # (Written as its own sum rather than by subtracting from tau_des, so
        # tau_des's summation order -- and therefore its exact floating-point
        # value -- is unchanged from before this mechanism existed.)
        # FRICTION FEEDFORWARD. Friction opposes motion (tau_f = -c*sign(qd)), so
        # cancelling it means ADDING +c*sign(qd); tanh is the smooth surrogate for
        # sign, with the deadband setting how sharply it switches. Summed into the
        # same joint-space bias as gravity/posture, i.e. INSIDE tau_des, so it is
        # bounded by the QP's own torque box and traded off against the corridor
        # rows exactly like every other bias term -- there is no bypass path.
        tau_friction_ff = np.zeros(6, dtype=np.float64)
        if bool(self.cfg.friction_feedforward):
            coulomb = np.asarray(self.cfg.friction_ff_coulomb_nm, dtype=np.float64).reshape(6)
            viscous = np.asarray(self.cfg.friction_ff_viscous, dtype=np.float64).reshape(6)
            deadband = max(float(self.cfg.friction_ff_qd_deadband), 1e-9)
            tau_friction_ff = coulomb * np.tanh(qd / deadband) + viscous * qd

        tau_hold = tau_damping + tau_posture + gravity + tau_friction_ff
        tau_des = (tau_task_nominal + tau_damping + tau_posture + tau_yz_soft
                   + gravity + tau_friction_ff)
        linear = -hessian @ tau_des

        # --- torque box -------------------------------------------------- #
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

        # --- excluded joints, part (b): pin the box shut ------------------ #
        # LAST, deliberately: applied after the headroom box and after the
        # velocity-implied bounds, so a pin can never be widened back out by
        # either of them. The pin value is itself clipped into the box that
        # was in force, so the pin can only ever be as permissive as the
        # limits already allowed (a hold torque larger than the headroom box
        # is clamped, not granted).
        #
        # With lo == hi, solve_box_qp's every iterate is np.clip(..., lo, hi)
        # at that index, so tau_preclip[i] == tau_hold_clipped[i] EXACTLY --
        # a structural guarantee, independent of the Hessian, of which
        # inequality rows are active, and of the solver's iteration budget.
        # Part (a) above is what makes it cheap; this part is what makes it a
        # guarantee. Measured necessity: with (a) alone and the corridor rows
        # active, shoulder_pan still moved 0.03 deg at dx=-0.06 m and 2.12 deg
        # at dx=+0.12 m -- the corridor rows route torque straight into the
        # joint the zeroed column was supposed to protect.
        if excluded_joints:
            idx = np.asarray(excluded_joints, dtype=int)
            pin = np.clip(tau_hold[idx], tau_lo[idx], tau_hi[idx])
            tau_lo = tau_lo.copy()
            tau_hi = tau_hi.copy()
            tau_lo[idx] = pin
            tau_hi[idx] = pin

        # --- 1.2/1.3 constraint rows ------------------------------------- #
        rows_a: list[np.ndarray] = []
        rows_b: list[float] = []
        # Bounds are half-widths around the START position expressed in the
        # SAME frame as the corridor rows (p0_corr), so a tool-frame corridor
        # brackets tool Z about where tool Z began. In the world frame p0_corr
        # is self._p0 and these are exactly self.y_bounds/self.z_bounds.
        y_min = float(p0_corr[1]) - float(self.cfg.y_corridor_half_width_m)
        y_max = float(p0_corr[1]) + float(self.cfg.y_corridor_half_width_m)
        z_min = float(p0_corr[2]) - float(self.cfg.z_corridor_half_width_m)
        z_max = float(p0_corr[2]) + float(self.cfg.z_corridor_half_width_m)
        n_corridor_rows = 0
        m_inv: np.ndarray | None = None
        use_joint_corridor = bool(self.cfg.joint_corridor_enabled) and bool(self.cfg.joint_corridor_joints)
        if use_corridor or use_manip_cbf or use_orientation_cbf or use_joint_corridor:
            m_inv = np.linalg.inv(
                np.asarray(st["mass_matrix"], dtype=np.float64).reshape(6, 6)
            )

        # Slot each corridor axis' (max, min) row pair into the fixed
        # (y_max, y_min, z_max, z_min) reporting order, so
        # `yz_corridor_active_rows` keeps its shape and meaning no matter how
        # many axes are actually bounded. An axis that is TRACKED instead
        # simply contributes no rows and reports False for both of its slots.
        corridor_slot: dict[int, int] = {}
        if use_corridor and corridor_axes:
            assert m_inv is not None
            axis_bounds = {1: (y_min, y_max), 2: (z_min, z_max)}
            for axis in corridor_axes:
                lower, upper = axis_bounds[axis]
                a_max, b_max, a_min, b_min = self._corridor_rows(
                    j_row=jac_corr[axis, :], m_inv=m_inv, bias=gravity, qd=qd,
                    value=float(p_corr[axis]), lower=lower, upper=upper,
                    alpha1=float(self.cfg.yz_corridor_alpha1),
                    alpha2=float(self.cfg.yz_corridor_alpha2),
                )
                corridor_slot[axis] = len(rows_a)
                rows_a += [a_max, a_min]
                rows_b += [b_max, b_min]
            n_corridor_rows = 2 * len(corridor_axes)

        # JOINT CORRIDOR: a hard bound on |q_j - q_j(0)| for each listed joint.
        # j_row = e_j because qdot_j = e_j . qdot exactly, so `_corridor_rows`
        # applies verbatim -- same HOCBF algebra, same sign convention, already
        # unit-tested for the Cartesian case.
        if use_joint_corridor:
            assert m_inv is not None
            hw = float(self.cfg.joint_corridor_half_width_rad)
            for j in self.cfg.joint_corridor_joints:
                e_j = np.zeros(6, dtype=np.float64)
                e_j[j] = 1.0
                q0_j = float(self._q_rest[j])
                a_max, b_max, a_min, b_min = self._corridor_rows(
                    j_row=e_j, m_inv=m_inv, bias=gravity, qd=qd,
                    value=float(q[j]), lower=q0_j - hw, upper=q0_j + hw,
                    alpha1=float(self.cfg.joint_corridor_alpha1),
                    alpha2=float(self.cfg.joint_corridor_alpha2),
                )
                rows_a += [a_max, a_min]
                rows_b += [b_max, b_min]

        mu_val: float | None = None
        cbf_h: float | None = None
        manip_row_index: int | None = None
        if use_manip_cbf:
            assert self._jacobian_fn is not None and m_inv is not None
            mu_val = float(manipulability(jac))
            cbf_h = mu_val - float(self.cfg.manipulability_cbf_epsilon)
            grad_mu = manipulability_gradient(
                self._jacobian_fn, q, step=float(self.cfg.manipulability_cbf_fd_step),
                jacobian_derivative_fn=self._jacobian_derivative_fn,
            )
            grad_norm = float(np.linalg.norm(grad_mu))
            if np.isfinite(grad_norm) and grad_norm > 0.0:
                # Mirrors manipulability_cbf_filter's own no-usable-direction
                # branch, reimplemented locally because that filter is a
                # closed single-row solve and cannot be composed with the
                # corridor rows. A degenerate 0 @ tau <= b row would be either
                # vacuous or unsatisfiable-by-construction; skipping is right.
                curvature = manipulability_directional_curvature(
                    self._jacobian_fn, q, qd,
                    step=float(self.cfg.manipulability_cbf_curvature_step),
                )
                a_row, b_scalar = manipulability_cbf_constraint_row(
                    grad_mu=grad_mu,
                    m_inv=m_inv,
                    bias=gravity,
                    qd=qd,
                    mu=mu_val,
                    curvature=curvature,
                    epsilon=float(self.cfg.manipulability_cbf_epsilon),
                    alpha1=float(self.cfg.manipulability_cbf_alpha1),
                    alpha2=float(self.cfg.manipulability_cbf_alpha2),
                )
                manip_row_index = len(rows_a)
                rows_a.append(a_row)
                rows_b.append(float(b_scalar))

        orient_cbf_h: float | None = None
        orient_cbf_row_index: int | None = None
        if use_orientation_cbf:
            assert m_inv is not None
            # Orientation rows are never remapped by task_frame=="tool" (see
            # that field's docstring) -- always the FULL Jacobian's rows
            # 3:6, rotated by the FIXED self._R0 (see the module docstring's
            # item 6 for why this rotation is required, not optional).
            jac_rot_ref = self._R0.T @ jac[3:6, :]
            a_row, b_scalar, orient_cbf_h, _hdot, _edot = self._orientation_cbf_row(
                e=e_rot, jac_rot_ref=jac_rot_ref, m_inv=m_inv, bias=gravity, qd=qd,
                max_error_rad=float(self.cfg.orientation_cbf_max_error_rad),
                alpha1=float(self.cfg.orientation_cbf_alpha1),
                alpha2=float(self.cfg.orientation_cbf_alpha2),
            )
            orient_cbf_row_index = len(rows_a)
            rows_a.append(a_row)
            rows_b.append(float(b_scalar))

        a_ineq = np.vstack(rows_a) if rows_a else None
        b_ineq = np.array(rows_b, dtype=np.float64) if rows_b else None

        # "Binding" is measured at tau_des (the unconstrained minimizer), not
        # by comparing tau before/after: with the box or another row active
        # the QP can move tau for reasons unrelated to this row, and an
        # always-True "the row exists" flag would make a trace unreadable.
        binding = np.zeros(len(rows_b), dtype=bool)
        if a_ineq is not None and b_ineq is not None:
            binding = (a_ineq @ tau_des - b_ineq) > 0.0

        # --- 1.4 one solve ----------------------------------------------- #
        # Warm start (cfg.qp_warm_start): seed the solve from the previous cycle's
        # primal/dual and run a smaller inner budget. Only when a warm buffer
        # actually exists (the first cycle after a reset stays a cold 80-iter
        # solve); the solver itself re-checks the shapes and falls back to cold if
        # the row count changed since the buffer was captured. This changes only
        # the solve's iteration count, not its fixed point -- see the solver
        # docstring and config.qp_warm_start.
        warm_ready = (
            bool(self.cfg.qp_warm_start)
            and self._warm_x is not None
            and self._warm_lam is not None
        )
        solve_kwargs: dict[str, Any] = dict(
            dual_sweeps=int(self.cfg.dual_sweeps),
            dual_root_iters=int(self.cfg.dual_root_iters),
        )
        if warm_ready:
            solve_kwargs["x_warm"] = self._warm_x
            solve_kwargs["lam_warm"] = self._warm_lam
            solve_kwargs["max_iters"] = int(self.cfg.qp_warm_max_iters)
            # Convergence gate: if the warm solve does not converge below this
            # residual (a rare fast-transient cycle), the solver redoes it as a
            # plain cold 80-iter solve, so accuracy is >= cold by construction.
            # None disables the gate. See cfg.qp_warm_fallback_tol.
            if getattr(self.cfg, "qp_warm_fallback_tol", None) is not None:
                solve_kwargs["fallback_tol"] = float(self.cfg.qp_warm_fallback_tol)
        t_start = time.perf_counter()
        tau_qp, _dual, feasible = solve_constrained_box_qp(
            hessian,
            linear,
            tau_lo,
            tau_hi,
            a_ineq,
            b_ineq,
            **solve_kwargs,
        )
        qp_solve_time_s = float(time.perf_counter() - t_start)
        # Store this cycle's solution as the next cycle's warm start. Captured
        # even when this cycle was cold (bootstraps the first warm cycle).
        if bool(self.cfg.qp_warm_start):
            self._warm_x = np.asarray(tau_qp, dtype=np.float64).copy()
            self._warm_lam = np.asarray(_dual, dtype=np.float64).copy()

        tau_clipped = np.clip(tau_qp, -tau_limit, +tau_limit)
        saturated = np.abs(tau_qp - tau_clipped) > 1e-10

        active = [False, False, False, False]
        for axis, start in corridor_slot.items():
            base = 0 if axis == 1 else 2
            active[base] = bool(binding[start])
            active[base + 1] = bool(binding[start + 1])
        corridor_active = (active[0], active[1], active[2], active[3])
        manip_active = bool(binding[manip_row_index]) if manip_row_index is not None else False
        orient_active = bool(binding[orient_cbf_row_index]) if orient_cbf_row_index is not None else False
        # solve_constrained_box_qp reports ONE feasibility flag for the whole
        # row set (it stops distinguishing rows once any is unreachable), so
        # every mechanism reports the same flag -- honest, and only
        # meaningful for a mechanism that actually contributed rows.
        return XTaskYZCorridorQPOutput(
            tau=tau_clipped,
            tau_preclip=tau_qp,
            wrench_reduced=wrench_reduced,
            tau_task_nominal=tau_task_nominal,
            tau_damping=tau_damping,
            tau_posture=tau_posture,
            tau_yz_soft=tau_yz_soft,
            tau_gravity=gravity,
            tau_saturated=saturated.astype(np.float64),
            tau_hold=tau_hold,
            jacobian_cond=cond,
            singular_scale=singular_scale,
            x_error=x_err,
            y_error=y_err,
            z_error=z_err,
            orientation_error_vec=e_rot,
            orientation_error_norm=ori_norm,
            y_min=y_min,
            y_max=y_max,
            z_min=z_min,
            z_max=z_max,
            yz_corridor_active_rows=corridor_active,
            yz_corridor_feasible=bool(feasible) if use_corridor else True,
            manipulability_cbf_active=manip_active,
            manipulability=mu_val,
            manipulability_cbf_h=cbf_h,
            manipulability_cbf_feasible=bool(feasible) if use_manip_cbf else True,
            qp_num_ineq_rows=int(len(rows_b)),
            qp_solve_time_s=qp_solve_time_s,
            task_excluded_joints=excluded_joints,
            task_axis_rows=task_axes,
            corridor_axis_rows=corridor_axes,
            orientation_cbf_active=orient_active,
            orientation_cbf_h=orient_cbf_h,
            orientation_cbf_feasible=bool(feasible) if use_orientation_cbf else True,
        )
