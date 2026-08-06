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
        )
