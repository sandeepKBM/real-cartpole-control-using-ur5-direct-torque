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
    # Null-space posture pull toward q_rest inside compute_ik_seeded's
    # per-iteration Newton-QP solve (added 2026-08-06) -- see modes.py's
    # compute_ik_seeded for the full rationale. 0.0 (default) reproduces
    # the exact prior behavior byte-for-byte; a nonzero value pulls
    # whichever rotation axes are NOT in the task (task_dim_rx/ry when
    # False, by default) back toward their q_rest value each Newton
    # iteration, the same mechanism reduced_task_dims already uses via
    # kp_posture -- kept as a SEPARATE field rather than reusing kp_posture
    # since the two operate at different scales (kp_posture is a per-cycle
    # rate gain; this is a per-Newton-iteration position-step fraction).
    ik_posture_gain: float = 0.0

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
            ik_posture_gain=float(vc.get("ik_posture_gain", 0.0)),
        )
