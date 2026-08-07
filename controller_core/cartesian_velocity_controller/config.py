"""Configuration for the resolved-rate Cartesian velocity controller.

Velocity gains are 1/s (v_cmd = kp * position_error), NOT force gains like
CartesianImpedanceConfig's kp_x/kp_y/kp_z (N/m) -- do not reuse
impedance-tuned gain values here, they are dimensionally different
quantities entirely.

For the full design history behind reduced_task_dims, split_base_wrist_task,
and ik_seeded_resolution (why each exists, what was tried and rejected, and
the measured evidence behind the current defaults), see this package's
``__init__.py`` docstring and ``modes.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..x_axis_cartesian_impedance import JOINT_NAME_ORDER


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
    reduced_task_dims: bool = True
    task_dim_rx: bool = False
    task_dim_ry: bool = False
    task_dim_rz: bool = True
    kp_posture: float = 1.0
    pinv_damping: float = 0.005
    posture_reanchor_on_settle: bool = True
    reanchor_pos_tol_m: float = 0.002
    reanchor_settle_cycles: int = 10
    split_base_wrist_task: bool = False
    ik_seeded_resolution: bool = False
    ik_iterations: int = 6
    ik_joint_gain: float = 4.0
    # QP-constrained IK (ik_seeded_resolution only): replaces the plain
    # damped-least-squares Newton step with a genuine box-constrained QP
    # (reuses box_qp.solve_box_qp, already validated by torque_task_qp.py's
    # identical pattern) so each IK iteration's joint-space step can never
    # produce a q_k that violates joint position limits -- unlike the plain
    # Newton step, which has no way to represent "stop here, this joint is
    # at its limit" and must rely entirely on an external safety monitor
    # catching a violation after the fact. None (the default for both
    # bounds) = unconstrained (permissive +-2pi-equivalent bounds), byte-
    # compatible with the pre-QP behavior; the caller (hardware/
    # velocity_transport.py, the kinematic sim) is responsible for
    # supplying real UR5e joint limits, since controller_core stays
    # simulator/hardware-independent.
    joint_pos_lower: np.ndarray | None = None
    joint_pos_upper: np.ndarray | None = None
    joint_vel_limit_radps: float | None = None
    qp_task_weight: float = 1.0e4
    # Hard bound (rad) on how far the REDUNDANT (null-space) part of
    # compute_ik_seeded's per-iteration solve may wander from q_rest,
    # enforced EXACTLY via null-space-basis coordinate clipping (added and
    # corrected twice the SAME day, 2026-08-06 -- see modes.py for the
    # full mechanism, both prior wrong versions, and why). None (default)
    # reproduces the exact prior behavior byte-for-byte.
    #
    # Replaces ik_posture_gain/ik_posture_activation_joint_dev_rad (both
    # REMOVED -- see this file's git history), a SOFT task_w-relative
    # quadratic pull that real gain searches almost never converged to
    # using even when correctly scaled and gated. This hard constraint
    # needs no such learning: the null-space coordinate is PROVABLY
    # confined to [-max_dev, max_dev], and task achievement (J_task @ dq)
    # is PROVABLY unaffected by however aggressively it clips.
    #
    # IMPORTANT SCOPE LIMIT, discovered validating this fix (2026-08-06):
    # this mechanism only helps failures that are genuinely REDUNDANT
    # (null-space) phenomena -- confirmed for neg40/neg45_wrist2offset's
    # wrist_2 runaway (a real fix, task accuracy preserved exactly). It
    # CANNOT help hanging_alpha_0_5's -X orientation failure: a direct
    # check found the pure minimum-norm, ZERO-null-space-component task
    # solution for +X motion at that pose already induces real rx/ry
    # rotation -- the coupling lives in the TASK (row) space itself, not
    # the null space, so no null-space-projected mechanism (this one, or
    # the removed soft pull) can fix it without a real X-tracking accuracy
    # trade-off. Matches this repo's already-documented torque-control-
    # lane finding for a different pose/axis (structural, not a search
    # gap). Do not expect this field to help that failure.
    ik_max_joint_deviation_rad: float | None = None

    # --- orientation_priority (added 2026-08-06, ik_seeded_resolution only,
    # default OFF -- reproduces the exact prior behavior bit-for-bit) ---
    #
    # Why a NEW mechanism family was needed: every redundancy-resolution
    # mechanism this controller has (the removed soft posture pull, and
    # ik_max_joint_deviation_rad above) acts on the NULL space of J_task,
    # and a direct linear-algebra check proved hanging_alpha_0_5's -X
    # orientation failure lives in the ROW space -- the pure minimum-norm,
    # zero-null-space-component task step already induces real rx/ry
    # rotation there. Independently confirmed by search: the per-cell
    # differential_evolution oracle (velocity_gain_tuning/scheduling/,
    # searching the FULL 6D gain space per cell) found NO guard-clean
    # solution anywhere for that cell. Structural to the COST FUNCTION, not
    # a tuning gap: with task_dim_rx/task_dim_ry both False (the default),
    # rx/ry are checked by the safety guard but appear nowhere in what the
    # QP actually minimizes.
    #
    # What this does: runs compute_ik_seeded's Newton solve a SECOND time
    # with the disabled rotation axes PROMOTED to co-primary task rows
    # (weight = qp_task_weight * orientation_priority_weight; 1.0, i.e.
    # equal weight, is the validated default), then blends between the two
    # solutions by how much position accuracy that promotion actually cost:
    #
    #   blend = smooth_falloff(|p_des - FK(q_promoted)|,
    #                          residual_tol_m, residual_falloff_m, power)
    #   q_target = q_position_only + blend * (q_promoted - q_position_only)
    #
    # Where the full 6-DOF pose is reachable the promoted solve hits the
    # position target exactly AND drives orientation error to ~0, so it wins
    # outright (blend == 1.0). Where it is not, the residual grows, the
    # blend decays to EXACTLY 0.0, and behaviour is bit-for-bit today's.
    #
    # Why not simply flip task_dim_rx/task_dim_ry on -- the zero-code
    # alternative, which IS what the promoted solve computes? Because
    # unconditionally is the problem, not the promotion itself: measured at
    # hanging_alpha_0_5, always-on rx/ry drives orientation error to exactly
    # 0.0000 and recovers a real failing case at 100% X-tracking, but at
    # displacements past the reach boundary the square, essentially-undamped
    # 6-row solve goes ill-conditioned, the arm RETREATS in X, and the
    # orientation trips become worse orthogonal-drift trips. The residual
    # gate is precisely the "is this promotion free here?" test that
    # distinguishes the two regimes.
    #
    # The blend gate reads only the promoted solve's own residual, never
    # q_current, so ik_seeded_resolution's path-independence property (see
    # modes.py) is preserved exactly.
    #
    # THE DEFAULT BAND IS DELIBERATELY VERY TIGHT (0.1 mm to 0.5 mm), i.e.
    # very nearly a hard accept/reject rather than a gradual blend -- and
    # that is a MEASURED choice, not a guess. Sweeping the band over the full
    # 128-cell grid (tools/evaluate_orientation_priority.py, see
    # docs/status/task_priority_orientation_hanging_2026-08-06.md) found a
    # clean monotone trend: the wider the band, the worse the result --
    # 111/128 at (1e-4, 5e-4), 107 at (2e-4, 2e-3), 99 at (2e-3, 1e-2), 95 at
    # (5e-4, 5e-2), 85 at (2e-3, 1e-1, linear) -- against a 104/128 baseline.
    # Cause, traced: a PARTIAL blend emits a q_target that neither solve
    # endorses, and worse, the blend weight sweeps through the band DURING a
    # move as the commanded target advances, so q_target migrates between two
    # different IK branches mid-move. The joint-space P law chases that
    # migration at ik_joint_gain and trips the joint-velocity guard -- every
    # single one of the 12 cells the wide band broke was a
    # joint_velocity_guard trip at 3.00-3.22 rad/s against a 3.0 limit, all
    # of them cells the tight band leaves untouched. Widen this band only
    # with grid evidence in hand.
    #
    # Tightening further makes no difference (a pure-step gate scores the
    # same 111/128), so the small residual band is kept purely as robustness
    # against floating-point noise in the residual, not for its blending.
    orientation_priority: bool = False
    orientation_priority_weight: float = 1.0
    orientation_priority_residual_tol_m: float = 0.0001
    orientation_priority_residual_falloff_m: float = 0.0005
    orientation_priority_falloff_power: float = 2.0

    # --- singularity_velocity_scaling (added 2026-08-07, ik_seeded_
    # resolution only, default OFF -- reproduces the exact prior behavior
    # bit-for-bit) ---
    #
    # Why: two documented real qd spikes this session (18.14 and
    # 161.57 rad/s, docs/status/nullspace_v2_search_results_2026-08-06.md)
    # both trace to ``wrist_2`` crossing exactly through 0 -- a real UR
    # kinematic singularity. Every existing mechanism in this controller
    # only changes how the QP's least-squares solve is REGULARIZED
    # (``pinv_damping``, the analogous damping in
    # ``velocity_gain_tuning``'s downstream qd-reconstruction) -- none of
    # them reduce the COMMANDED task velocity itself as the arm approaches
    # a singular configuration; they only change how aggressively a
    # fixed-magnitude target is chased. This is the standard industrial
    # fix instead: scale the commanded Cartesian velocity DOWN as
    # singularity proximity grows, so joint velocity stays bounded because
    # the input is throttled, not just because its inversion is damped.
    #
    # Signal: sigma_min of the FULL 6x6 Jacobian, NOT ``J_task``'s own
    # task-selected submatrix, despite ``J_task``'s SVD (via
    # ``kinematics_utils.null_space_basis``, already used elsewhere in
    # this module) being the obvious first choice for consistency with
    # the rest of this file. Measured directly (not assumed) before
    # picking this: replaying the exact 161.57 rad/s case
    # (``neg45_wrist2offset``, ``dx=-0.029m``, 1.0s move,
    # ``ik_max_joint_deviation_rad=0.01``) with per-step SVDs of both
    # matrices found ``sigma_min(J_task)`` (task_dim_rz on, rx/ry off --
    # 4x6) essentially FLAT at ~0.085-0.094 for the entire episode
    # (cond~15, matching this repo's own independent finding in
    # ``velocity_gain_tuning/envs/velocity_transport_env.py``'s
    # ``qd_estimate_damping`` docstring: "the controller's own reduced-task
    # Jacobian stays well-conditioned... throughout"), while
    # ``sigma_min(full 6x6 J)`` at the SAME q_current drops from ~0.07 to
    # 0.0056 right at the guard trip. A signal keyed to ``J_task`` alone
    # would never have engaged for either documented spike -- the wrist_2
    # singularity kills the arm's ability to independently actuate
    # ``rx``/``ry``, which are excluded from ``J_task`` by
    # ``task_dim_rx``/``task_dim_ry`` defaulting off, but it is still very
    # much present in ``J_full`` and in whatever the real robot's
    # ``speedL`` firmware (or this repo's own qd-reconstruction/
    # integration step) has to invert to realize the commanded ``xd_cmd``.
    # This is what governs the real hazard, not ``J_task``.
    #
    # Applied at TWO points, both scaling the task residual/output BEFORE
    # it is used, never touching regularization: (1) inside
    # ``_ik_newton_solve``'s per-iteration ``task_err_full_k`` (scaled by
    # ``sigma_min`` of that iteration's own full Jacobian at ``q_k``) --
    # covers the (currently untested but plausible) case where the
    # NEWTON SOLVE itself, seeded from ``q_rest``, approaches a
    # singularity; (2) on the FINAL ``qd_joint`` (the joint-space P-law
    # output) before it is converted to ``xd_cmd`` via ``jac_current`` --
    # this is the point that actually fixes both documented cases, since
    # ``q_rest`` for both never leaves a safe pose and the real danger is
    # entirely in how ``q_current`` drifts toward the singularity over
    # many control cycles.
    #
    # Validated (docs referenced in the calling driver's own status doc):
    # re-running the exact 161.57 rad/s case with this flag on and
    # ``singularity_sigma_min_stop``/``singularity_sigma_min_full_speed``
    # at their defaults below eliminates the guard trip; the arm slows
    # rather than blows up as it nears the singularity -- see this
    # package's own validation doc/tests for the exact numbers (peak
    # |qd|, achieved X) rather than restating them here, where they would
    # drift out of sync with the code.
    #
    # Threshold defaults calibrated from real data, not guessed: a survey
    # of ``sigma_min(full J)`` across all 4 pose scenarios x 12 dx
    # fractions x a full episode (~17k samples) found p10~0.042,
    # p25~0.057, median~0.071 -- ``singularity_sigma_min_full_speed=0.03``
    # sits comfortably below that population's typical/safe operating
    # range (so ordinary safe moves are rarely throttled at all) while
    # still being well above the ~0.0056 floor measured at the actual
    # guard trip; ``singularity_sigma_min_stop=0.003`` sits just under
    # that observed failure floor.
    singularity_velocity_scaling: bool = False
    singularity_sigma_min_stop: float = 0.003
    singularity_sigma_min_full_speed: float = 0.03
    singularity_scale_power: float = 2.0

    # --- singularity_windup_clamp_rad (added 2026-08-07, singularity_
    # velocity_scaling only, default OFF -- reproduces singularity_velocity_
    # scaling's own behavior bit-for-bit when unset) ---
    #
    # Why: singularity_velocity_scaling (above) fixed the two originally
    # documented spikes, but a full 128-cell grid re-evaluation
    # (velocity_gain_tuning.evaluate.evaluate_gains, gains from the
    # nullspace_v2 search result) found it net NEUTRAL, not a net win --
    # 105/128 both mechanism-on and mechanism-off. 2 cells flip fail->pass
    # (a real win) but 2 DIFFERENT cells (neg40_wrist2offset and
    # neg45_wrist2offset, both dx=-0.0464m) flip pass->fail, and worst-case
    # |qd| across the grid actually ROSE slightly (4.90->5.08 rad/s).
    #
    # Root cause, diagnosed not guessed: this mode's own design recomputes
    # q_target FRESH from (q_rest, p_des) every cycle -- entirely
    # independent of q_current (see modes.py's "Actual fix: ik_seeded_
    # resolution" docstring for why: this is what makes the mode path-
    # independent). qd_joint = ik_joint_gain * (q_target - q_current) is
    # then throttled by scale_current when q_current nears a singularity.
    # But throttling q_current deliberately holds the REAL robot state
    # back while q_target keeps advancing at the caller's nominal
    # (un-throttled) rate -- so the gap (q_target - q_current) grows every
    # cycle the throttle stays active. The moment sigma_min(jac_current)
    # recovers and scale_current snaps back toward 1.0, it multiplies
    # against a now-large accumulated gap at close to full ik_joint_gain
    # strength -- a genuine windup/release spike, structurally identical
    # to classic PID integral windup even though there is no explicit
    # integrator here (the "accumulator" is the geometric gap between an
    # externally-driven target and an artificially-held-back real state).
    #
    # Fix: bound ||q_target - q_current|| (per-joint, inf-norm, matching
    # ik_max_joint_deviation_rad's own per-joint clip style) to this value
    # BEFORE it is multiplied by ik_joint_gain, but ONLY on cycles that are
    # ACTIVELY throttled this instant (scale_current < 1.0), not merely
    # whenever singularity_velocity_scaling is on. A first version gated
    # only on the flag (unconditional whenever the mechanism was on) was
    # measured to also bind on cycles that were never throttled at all --
    # e.g. large hanging_alpha_0_5 moves, whose naturally large per-cycle
    # gap has nothing to do with a singularity -- introducing 4 new
    # orientation-guard regressions on the full 128-cell grid. Scoping to
    # scale_current < 1.0 removed all 4 while leaving the windup fix fully
    # intact, since the windup this mechanism targets can only accumulate
    # while scale_current < 1.0 in the first place (see modes.py, where
    # jac_current/scale_current are now computed BEFORE qd_joint so this
    # gate is available). Not a generic always-on clamp like
    # joint_vel_limit_radps -- that field still exists separately and is
    # unaffected by this one.
    #
    # Default None (off), matching every other opt-in mechanism in this
    # file -- with singularity_velocity_scaling on but this field unset,
    # behavior is bit-for-bit the same as before this field existed.
    # 0.03 rad is the validated value (see velocity_gain_tuning's env
    # config and this package's tests), chosen by direct sweep {0.01 ..
    # 0.05} over the two known regression cells plus the cells nearest
    # them: looser than ~0.04 rad leaves part of the release spike
    # unclamped (regression cells still trip, or a nearby cell,
    # unrotated_wrist2offset dx=-0.098m, spikes to 6.52 rad/s -- worse
    # than any variant without the clamp); tighter than ~0.02 rad starts
    # reproducing ik_max_joint_deviation_rad's own known failure shape
    # (over-constraining real task-necessary motion). 0.03 rad sits in the
    # flat, safe middle of that sweep.
    #
    # Validated (velocity_gain_tuning.evaluate.evaluate_gains, same
    # nullspace_v2 gains, full 128-cell grid, singularity_velocity_scaling
    # AND this field both on, value 0.03): 111/128 vs a 105/128 tie
    # (mechanism fully off, or on without this field) -- 6 cells flip
    # fail->pass (both target regression cells among them), ZERO cells
    # flip pass->fail anywhere in the grid, and the documented real win
    # (neg45_wrist2offset, dx=-0.029m, tight ik_max_joint_deviation_rad)
    # is reproduced exactly unchanged (this field never engages there --
    # that case was never a windup case to begin with).
    #
    # One honest caveat, not fixed by this field: singularity_velocity_
    # scaling alone (with or without this clamp) raises the grid's
    # worst-case |qd| from 4.90 to 5.08 rad/s -- traced to a DIFFERENT,
    # already-failing cell (neg45_wrist2offset, dx=+0.058m, fails in every
    # variant including fully off) whose peak |qd| is measured completely
    # unaffected by this field at every value tried (0.005-0.05 rad, all
    # give exactly 5.0804) -- the guard trips before scale_current ever
    # drops under 1.0 there, so the clamp cannot engage. This is a
    # pre-existing property of singularity_velocity_scaling itself, not
    # something this field introduces or can fix; the anti-windup fix's
    # own scope is the pass/fail grid and the two documented regression
    # cells, not this separate, already-failing cell's peak magnitude.
    singularity_windup_clamp_rad: float | None = None

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
            reduced_task_dims=bool(vc.get("reduced_task_dims", True)),
            task_dim_rx=bool(vc.get("task_dim_rx", False)),
            task_dim_ry=bool(vc.get("task_dim_ry", False)),
            task_dim_rz=bool(vc.get("task_dim_rz", True)),
            kp_posture=float(vc.get("kp_posture", 1.0)),
            pinv_damping=float(vc.get("pinv_damping", 0.005)),
            posture_reanchor_on_settle=bool(vc.get("posture_reanchor_on_settle", True)),
            reanchor_pos_tol_m=float(vc.get("reanchor_pos_tol_m", 0.002)),
            reanchor_settle_cycles=int(vc.get("reanchor_settle_cycles", 10)),
            split_base_wrist_task=bool(vc.get("split_base_wrist_task", False)),
            ik_seeded_resolution=bool(vc.get("ik_seeded_resolution", False)),
            ik_iterations=int(vc.get("ik_iterations", 6)),
            ik_joint_gain=float(vc.get("ik_joint_gain", 4.0)),
            joint_pos_lower=(
                np.array([float(vc["joint_pos_lower"][name]) for name in JOINT_NAME_ORDER], dtype=np.float64)
                if "joint_pos_lower" in vc
                else None
            ),
            joint_pos_upper=(
                np.array([float(vc["joint_pos_upper"][name]) for name in JOINT_NAME_ORDER], dtype=np.float64)
                if "joint_pos_upper" in vc
                else None
            ),
            joint_vel_limit_radps=(
                float(vc["joint_vel_limit_radps"]) if "joint_vel_limit_radps" in vc else None
            ),
            qp_task_weight=float(vc.get("qp_task_weight", 1.0e4)),
            ik_max_joint_deviation_rad=(
                float(vc["ik_max_joint_deviation_rad"]) if "ik_max_joint_deviation_rad" in vc else None
            ),
            orientation_priority=bool(vc.get("orientation_priority", False)),
            orientation_priority_weight=float(vc.get("orientation_priority_weight", 1.0)),
            orientation_priority_residual_tol_m=float(vc.get("orientation_priority_residual_tol_m", 0.002)),
            orientation_priority_residual_falloff_m=float(
                vc.get("orientation_priority_residual_falloff_m", 0.010)
            ),
            orientation_priority_falloff_power=float(vc.get("orientation_priority_falloff_power", 2.0)),
            singularity_velocity_scaling=bool(vc.get("singularity_velocity_scaling", False)),
            singularity_sigma_min_stop=float(vc.get("singularity_sigma_min_stop", 0.003)),
            singularity_sigma_min_full_speed=float(vc.get("singularity_sigma_min_full_speed", 0.03)),
            singularity_scale_power=float(vc.get("singularity_scale_power", 2.0)),
            singularity_windup_clamp_rad=(
                float(vc["singularity_windup_clamp_rad"]) if "singularity_windup_clamp_rad" in vc else None
            ),
        )
