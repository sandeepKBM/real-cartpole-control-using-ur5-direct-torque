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
        )
