"""
CartesianImpedanceOutput -- the structured return type of
XAxisCartesianImpedanceController.compute().

Split out of the former single-file ``x_axis_cartesian_impedance.py`` module
(pure structural refactor; see the package ``__init__.py`` for the original
module docstring).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


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
    friction_model_used: str = "static"
    friction_z: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))
    friction_karnopp_stuck: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))
    y_integral_action_active: bool = False
    y_integral_value: float = 0.0
    x_integral_action_active: bool = False
    x_integral_value: float = 0.0
    split_base_wrist_task_active: bool = False
    y_corridor_scale: float = 1.0
    acceleration_feedforward_active: bool = False
    wrench_accel_ff: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
