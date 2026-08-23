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
    #: True when the nullspace projector's eps came from the inertia-scaled
    #: schedule (nullspace_inertia_adaptive_regularization) rather than from
    #: the static lambda_regularization or the log(cond) schedule. Additive
    #: (2026-08-12); the eps it picked is reported in
    #: lambda_regularization_effective, same field both schedulers already use.
    nullspace_inertia_adaptive_regularization_active: bool = False
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
    # Which joint columns the split task actually drove this cycle (indices
    # into JOINT_NAME_ORDER), or None when split_base_wrist_task is off.
    # Additive (2026-08-12) alongside split_base_wrist_active_joints: with the
    # set configurable, `split_base_wrist_task_active: True` alone no longer
    # says which joints were active, and a trace that can't tell (0,1,2) from
    # (1,2,3) can't be re-read later. Defaults to None, so nothing that
    # already consumes this dataclass is affected.
    split_base_wrist_active_joints: tuple[int, ...] | None = None
    # Which translation task rows the split task actually regulated this cycle
    # (0=X, 1=Y, 2=Z), or None when split_base_wrist_task is off. Additive
    # (2026-08-12) alongside split_base_wrist_task_dims, for the same reason
    # split_base_wrist_active_joints above is recorded: with both the rows and
    # the columns configurable, `split_base_wrist_task_active: True` says
    # neither, and a trace that cannot tell a 3x3 task from a 1x3 one cannot be
    # re-read later. Always the full (0, 1, 2) unless the config selects a
    # subset, so nothing that already consumes this dataclass is affected.
    split_base_wrist_task_dims: tuple[int, ...] | None = None
    y_corridor_scale: float = 1.0
    acceleration_feedforward_active: bool = False
    wrench_accel_ff: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    # Which world axis the task/transport axis actually was this cycle (0=X
    # default, 1=Y, 2=Z) -- resolved from the state's transport_axis_index if
    # present, else CartesianImpedanceConfig.transport_axis_index. Additive
    # (2026-08-12) and needed to read the three error fields above
    # unambiguously: ``x_error`` is the TASK-axis error (so it stays the right
    # signal for the axis-error-growth guard fed from it), ``y_error`` is the
    # error of the axis holding the kp_y/kd_y role, ``z_error`` the kp_z/kd_z
    # one -- see CartesianImpedanceConfig.transport_axis_index for the full
    # axis->gain mapping. For the default axis 0 all three are exactly the
    # world X/Y/Z errors they always were.
    transport_axis_index: int = 0
    # Singularity-consistent inversion (SCI) diagnostics -- all None unless
    # CartesianImpedanceConfig.svd_singularity_filtering is on, so nothing that
    # already consumes this dataclass is affected (additive, 2026-08-12).
    # See that flag's docstring for the math these three describe.
    svd_singularity_filtering_active: bool = False
    #: Task-space singular values sigma_i actually filtered this cycle, in the
    #: SAME order as the other two arrays below. Mass-weighted (sqrt of
    #: eigh(J_task M^-1 J_task^T)) when task_space_inertia_shaping is on, raw
    #: svd(J_task) values when it is off -- the two branches genuinely filter
    #: different operators, see the flag's docstring.
    svd_task_singular_values: np.ndarray | None = None
    #: Per-direction damping factor lambda_i (0 for every direction at or above
    #: svd_sigma_threshold -- those are inverted exactly).
    svd_damping_lambda: np.ndarray | None = None
    #: Per-direction surviving fraction of the ideal undamped response,
    #: a_i = sigma_i^2 / (sigma_i^2 + lambda_i^2), in [0, 1]. 1.0 means that
    #: direction is untouched by the filter; ~0 means it is a lost direction
    #: the controller has backed off from.
    svd_direction_attenuation: np.ndarray | None = None
    # Manipulability-CBF diagnostics -- all inert unless
    # CartesianImpedanceConfig.manipulability_cbf is on, so nothing that
    # already consumes this dataclass is affected (additive, 2026-08-13).
    # See that flag's docstring, and controller_core/manipulability_cbf.py for
    # the derivation these describe.
    #: True when the flag is on AND the CBF row was actually binding this
    #: cycle (i.e. the QP ran and changed the torque). Deliberately NOT "the
    #: flag is on": an always-True field would make it impossible to read from
    #: a trace when the filter did anything, which is the whole question.
    manipulability_cbf_active: bool = False
    #: mu(q) = prod_i sigma_i(J), from the FULL 6x6 state Jacobian (never
    #: J_task -- see the flag's docstring). None when the flag is off.
    manipulability: float | None = None
    #: h = mu - manipulability_cbf_epsilon. Negative means the configuration
    #: is already inside the barrier's exclusion region and the high-order CBF
    #: is in recovery mode rather than merely holding.
    manipulability_cbf_h: float | None = None
    #: hdot = grad_mu . qd. Negative means moving TOWARD the singular set.
    manipulability_cbf_h_dot: float | None = None
    #: b - A @ tau_nominal, i.e. how much slack the constraint had at the
    #: unfiltered torque. Negative is exactly the condition that makes
    #: manipulability_cbf_active True.
    manipulability_cbf_slack: float | None = None
    #: ||tau_cbf - tau_nominal||, how far the filter moved the command.
    manipulability_cbf_delta_tau_norm: float = 0.0
    #: False when the CBF row cannot be satisfied inside the torque-headroom
    #: box -- reported, never silently clamped.
    manipulability_cbf_feasible: bool = True
