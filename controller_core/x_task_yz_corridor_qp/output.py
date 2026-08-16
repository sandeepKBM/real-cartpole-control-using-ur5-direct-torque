"""
``XTaskYZCorridorQPOutput`` -- the structured return of
``XTaskYZCorridorQPController.compute()``.

Deliberately NOT ``CartesianImpedanceOutput``: that dataclass is built around
a 6-row wrench and a set of mechanisms (backtracking, Lambda shaping, SCI,
friction models) this controller does not have, and reusing it would mean
emitting a pile of permanently-zero fields that read as "the mechanism ran
and found nothing" rather than "the mechanism does not exist here".

No ``as_dict()``: ``MujocoUR5eTorqueAdapter._controller_step`` already falls
back to ``dict(vars(output))``, which is exactly the right serialization for
a plain dataclass of arrays/scalars.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class XTaskYZCorridorQPOutput:
    #: Final commanded torque, hard-clipped to +-tau_max_nm.
    tau: np.ndarray
    #: The QP solution before that final clip (they differ only if the
    #: velocity-implied bounds pushed the box outside the torque limits).
    tau_preclip: np.ndarray
    #: The 4-row reduced task wrench [Fx, Mx, My, Mz], BEFORE the singular
    #: back-off scale. Four entries, not six -- Y and Z are not task rows.
    wrench_reduced: np.ndarray
    #: ``J_reduced.T @ (wrench_reduced * singular_scale)``.
    tau_task_nominal: np.ndarray
    tau_damping: np.ndarray
    tau_posture: np.ndarray
    #: The soft Y/Z-centering bias. This is the ONLY Y/Z authority in the
    #: objective; everything else Y/Z is a constraint row.
    tau_yz_soft: np.ndarray
    #: Gravity torque as THIS controller applied it (zero in the MuJoCo lane,
    #: whose adapter compensates gravity itself and does not put
    #: ``gravity_torque`` on the state).
    tau_gravity: np.ndarray
    #: Per-joint 1.0/0.0 flags for the final clip.
    tau_saturated: np.ndarray
    #: The non-task bias torque ``tau_damping + tau_posture + tau_gravity``.
    #: Reported because it is exactly what an excluded joint's commanded
    #: torque is pinned to -- so a trace can be checked against the claim
    #: ("tau[i] == tau_hold[i] for every excluded i") directly, without
    #: re-deriving the posture/damping terms from q/qd.
    tau_hold: np.ndarray
    #: cond(J) of the FULL 6x6 Jacobian -- reporting + the singular back-off
    #: scale only; the task itself uses J_reduced.
    jacobian_cond: float
    singular_scale: float
    x_error: float
    #: Signed distance from the corridor CENTER (y0/z0), not a tracking error
    #: -- this controller has no Y/Z tracking task. Kept named *_error for
    #: continuity with every other controller's trace fields.
    y_error: float
    z_error: float
    orientation_error_vec: np.ndarray
    orientation_error_norm: float
    #: Corridor walls, absolute world coordinates, captured at reset.
    y_min: float
    y_max: float
    z_min: float
    z_max: float
    #: Which of the four corridor rows were BINDING this cycle, in the order
    #: (y_max, y_min, z_max, z_min). "Binding" = the row was violated at the
    #: unconstrained minimizer ``tau_des`` and the QP therefore had to move
    #: the torque -- not merely "the row was built".
    yz_corridor_active_rows: tuple[bool, bool, bool, bool] = (False, False, False, False)
    #: False when the corridor rows cannot be met inside the torque box.
    #: Reported, never silently clamped.
    yz_corridor_feasible: bool = True
    #: Same "binding, not merely built" semantics, for the manipulability row.
    manipulability_cbf_active: bool = False
    #: mu(q) from the full 6x6 Jacobian; None when the flag is off.
    manipulability: float | None = None
    #: h = mu - epsilon; None when the flag is off.
    manipulability_cbf_h: float | None = None
    manipulability_cbf_feasible: bool = True
    #: How many inequality rows the QP actually carried this cycle (0, 1, 4,
    #: or 5) -- the direct driver of the solve cost below.
    qp_num_ineq_rows: int = 0
    #: Wall-clock seconds spent inside ``solve_constrained_box_qp`` this
    #: cycle. Measured, not estimated: this controller's real-time viability
    #: is an open question and this is the field that answers it. Compare
    #: against 2.0e-3 s (the 500 Hz direct_torque budget).
    qp_solve_time_s: float = 0.0
    #: Which world translation rows were TRACKED as task rows and which were
    #: BOUNDED by corridor rows this cycle. Reported because every other field
    #: here (`x_error`, `y_error`, `z_error`, `wrench_reduced`'s length,
    #: `yz_corridor_active_rows`) changes meaning with them, and a trace with
    #: no record of the row sets cannot be read after the fact.
    task_axis_rows: tuple[int, ...] = (0,)
    corridor_axis_rows: tuple[int, ...] = (1, 2)
    #: The joint indices whose torque was pinned to ``tau_hold`` this cycle
    #: (``config.task_excluded_joints``). Empty tuple = the mechanism is off.
    task_excluded_joints: tuple[int, ...] = ()
    #: Same "binding, not merely built" semantics as ``yz_corridor_active_rows``
    #: / ``manipulability_cbf_active``, for the orientation HOCBF row.
    orientation_cbf_active: bool = False
    #: ``h = orientation_cbf_max_error_rad^2 - ||orientation_error||^2`` this
    #: cycle. Negative means the error is already outside the barrier's
    #: nominal region (the HOCBF still drives it back, per the module
    #: docstring); ``None`` when ``config.orientation_cbf`` is off.
    orientation_cbf_h: float | None = None
    #: False when the orientation row cannot be met inside the torque box.
    #: ``True`` (not merely defaulted) when ``config.orientation_cbf`` is off.
    orientation_cbf_feasible: bool = True
