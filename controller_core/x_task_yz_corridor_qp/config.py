"""
``XTaskYZCorridorQPConfig`` -- gains + corridor/CBF knobs for the reduced
(X + orientation) task QP controller.

Extends ``TorqueTaskQPConfig`` (NOT ``CartesianImpedanceConfig`` directly) so
the QP-specific machinery is inherited verbatim rather than re-derived:
``max_joint_velocity_radps``, ``posture_regularization``,
``velocity_torque_coupling_kp``/``kd``, ``enforce_velocity_torque_bounds``,
plus the impedance gains ``kp_x``/``kd_x``, ``kp_rot``/``kd_rot``,
``kp_posture``/``kd_posture``, ``kd_joint``, ``tau_max_nm``,
``jacobian_singular_cond_max`` and ``torque_headroom``.

OUT OF SCOPE IN V1 (inherited, deliberately NOT wired into ``compute()``, so
that setting them in a YAML has provably no effect here):
``task_space_inertia_shaping``, ``nullspace_posture``,
``lambda_diagonal_shaping``, ``lambda_adaptive_regularization``,
``svd_singularity_filtering``, ``wrist_orientation_task``,
``friction_feedforward``, ``reduced_task_dims``, ``split_base_wrist_task``,
``y_control_mode``, ``ki_x``/``ki_y``, ``acceleration_feedforward``,
``posture_reanchor_on_settle``, ``transport_axis_index`` != 0. Those are
features of the ``x_axis_cartesian_impedance`` family; this controller is a
different, deliberately small torque law and reproducing them here would be
copying, not reuse. ``compute()`` raises loudly on the one of these that
would silently change the physical meaning of the task
(``transport_axis_index`` != 0) rather than ignoring it.
"""

from __future__ import annotations

import numpy as np

from controller_core.x_axis_cartesian_impedance.config import JOINT_NAME_ORDER

import math

from dataclasses import dataclass

from ..torque_task_qp import TorqueTaskQPConfig
from .parsing import (
    _parse_corridor_half_width,
    _parse_manipulability_cbf_alpha,
    _parse_axis_row_sets,
    _parse_task_frame,
    _parse_manipulability_cbf_epsilon,
    _parse_task_excluded_joints,
)


@dataclass
class XTaskYZCorridorQPConfig(TorqueTaskQPConfig):
    """Reduced-task QP gains + Y/Z corridor HOCBF + manipulability CBF.

    With ``yz_corridor_enabled=False`` and ``manipulability_cbf=False`` (both
    defaults) the QP has zero inequality rows and
    ``solve_constrained_box_qp`` reduces to exactly ``solve_box_qp`` -- i.e.
    the flags-off configuration is a plain box-constrained reduced-task QP,
    and is asserted byte-identical to that in
    ``tests/unit/test_x_task_yz_corridor_qp.py``.
    """

    # --- Soft Y/Z centering ------------------------------------------------
    # These four fields REDEFINE the role of the inherited kp_y/kd_y/kp_z/kd_z.
    # In every other controller in this repo they are the gains of a genuine
    # Cartesian HOLD task that sits in the task Jacobian and competes for
    # torque. Here they are a low-priority joint-space PD BIAS
    # (``tau_yz_soft``) that enters only the QP's linear term, never the
    # Hessian -- it nudges Y/Z back toward their start values without ever
    # reshaping the 4D task.
    #
    # The defaults are overridden (5.0/2.0 vs the inherited 80/15 and 120/20)
    # precisely BECAUSE the role changed: the inherited values were tuned as
    # hard-task gains against a full 6D task, and reusing them unchanged as a
    # bias would recreate the stiff Y/Z hold this controller exists to remove.
    # Sized to be an order of magnitude below kp_x (400 in the tuned configs)
    # so "Y/Z move freely inside the corridor" is true by construction rather
    # than by luck.
    kp_y: float = 5.0
    kd_y: float = 2.0
    kp_z: float = 5.0
    kd_z: float = 2.0

    # --- Joints held OUT of the task -------------------------------------
    #: Joint indices (into ``JOINT_NAME_ORDER``) whose commanded torque is
    #: PINNED to their non-task bias torque -- gravity + posture spring +
    #: joint damping -- so no task, corridor or CBF torque can reach them.
    #: Default ``(0,)`` = shoulder_pan. ``()`` disables the mechanism and
    #: reproduces the pre-2026-08-13 behavior exactly.
    #:
    #: WHY THIS EXISTS (measured, 2026-08-13). The reduced task is 4 rows
    #: (world X + 3 orientation) on a 6-joint arm, so 2 DOF are genuinely
    #: redundant and NOTHING in the QP said which joints were allowed to
    #: absorb them -- only the soft posture spring
    #: (``kp_posture``*(q_rest - q)) discouraged drift. At ARM_Q0 the world-X
    #: row of J gives shoulder_pan the single LARGEST coefficient of any joint
    #: (0.2366, vs elbow's -0.2346), so the X task actively PREFERRED to swing
    #: the base. Measured shoulder_pan excursions over the 5-case matrix:
    #: 4.32 / 5.20 / 8.93 / 12.16 / 13.15 deg. ARM_Q0 pins shoulder_pan at
    #: -135.7 deg for a physical reason (wall/base clearance in the real lab),
    #: so "a spring that mostly holds it" is not an acceptable guarantee.
    #:
    #: MECHANISM: BOTH the Jacobian-column zeroing that
    #: ``split_base_wrist_active_joints`` uses
    #: (``x_axis_cartesian_impedance/controller.py``) AND a box pin
    #: ``tau_lo[i] = tau_hi[i] = tau_hold[i]``. They were first measured as
    #: rival candidates, and the measurement is what showed they are
    #: complementary rather than alternatives -- each fixes the other's one
    #: real defect:
    #:
    #:   COLUMN ZEROING ALONE IS NOT A GUARANTEE. It removes only
    #:   ``J_reduced.T @ wrench`` from the joint; ``tau_damping``,
    #:   ``tau_posture``, ``tau_yz_soft`` and any deviation the QP makes when
    #:   a corridor/CBF row is active all still reach it. Measured at ARM_Q0
    #:   with the corridor rows on: shoulder_pan still moved 0.03 deg at
    #:   dx=-0.06 m and 2.12 deg at dx=+0.12 m -- the corridor rows route
    #:   torque straight into the joint the zeroed column was protecting.
    #:
    #:   THE PIN ALONE IS A GUARANTEE BUT A BAD CONTROLLER. With the column
    #:   left in place, ``H = 2(J_r.T W J_r + reg I)`` still couples the
    #:   pinned coordinate to the free ones, and because ``W`` is dominated by
    #:   ``kp_x`` the QP responds to the pin by re-optimizing the other five
    #:   joints to restore the lost ``j_x . tau`` projection. That projection
    #:   is a FORCE-space quantity, not the acceleration ``j_x M^-1 tau`` that
    #:   actually moves the tool, so the "compensation" over-drives the wrists
    #:   instead of reproducing the motion. Measured at ARM_Q0, pin-only:
    #:   dx=-0.06 m diverged to |qd| = 2.11 rad/s and tripped the orientation
    #:   guard (0.2515 rad) with X-tracking at 0.526; dx=+0.12 m tracked 0.286.
    #:   Raising ``kp_rot`` does not rescue it (kp_rot=200 -> tracking 0.115;
    #:   kp_rot=400 -> 0.140, both still tripping), which is what identified
    #:   the coupling rather than the rotational gains as the cause.
    #:
    #: TOGETHER they have neither defect. Zeroing the column makes
    #: ``H[free, excluded]`` exactly zero (the only surviving entry in that
    #: coordinate is the diagonal Tikhonov ``2*reg``), so the pinned
    #: coordinate no longer couples into the free ones and the pin costs the
    #: free joints nothing -- they keep ``tau_des``, damping and all -- while
    #: ``np.clip`` with ``lo == hi`` makes ``tau_preclip[i] == tau_hold[i]``
    #: bit-for-bit regardless of the rows, the Hessian, or the solver's
    #: iteration budget.
    #:
    #: (The solver's own accuracy under a pin was measured too, since
    #: ``solve_box_qp`` is projected gradient with a fixed 80-iteration
    #: budget: at ARM_Q0's Hessian the free coordinates agree with the exact
    #: eliminated-variable solution to 1.6e-4 Nm over 200 random ``tau_des``,
    #: and on a deliberately ill-conditioned synthetic fixture -- where 80
    #: iterations is NOT enough, ~8.9 Nm out, converging by ~2000 -- it still
    #: captures >99% of the objective improvement. With the column zeroed the
    #: question is moot for the pinned coordinate, which is exact either way.)
    #:
    #: PINNED TO THE BIAS, NOT TO ZERO. ``tau_hold = gravity + tau_posture +
    #: tau_damping``, so the excluded joint keeps gravity compensation and
    #: keeps an ACTIVE spring/damper hold at ``q_rest``. Pinning literally to
    #: 0.0 was measured too and is worse: it makes the joint passively free,
    #: and inertial coupling from the rest of the arm then drags it (see the
    #: design doc's variant table). At ARM_Q0 gravity torque on shoulder_pan
    #: happens to be exactly 0.0 -- its axis is vertical -- but that is a
    #: property of THIS joint at THIS mounting, not a general one, which is
    #: why the pin is expressed against the bias rather than against zero.
    #:
    #: At most 2 indices (4 task rows, 6 joints); ``_parse_task_excluded_joints``
    #: raises otherwise. Whether the REMAINING columns span the task is
    #: pose-dependent and is NOT checked -- at ARM_Q0, dropping shoulder_pan
    #: takes ``cond(J_reduced)`` from 10.1 to 519 (still rank 4). Screen a new
    #: index set against the intended start pose before trusting it.
    #: Per-joint multipliers on the posture spring/damper, length 6, in
    #: JOINT_NAME_ORDER. ``None`` (default) means uniform weight 1.0 for every
    #: joint, which reproduces the previous behaviour EXACTLY -- the term is
    #: then untouched, not multiplied by an array of ones.
    #:
    #: WHY THIS EXISTS. ``kp_posture``/``kd_posture`` are single scalars, so the
    #: only way to hold ONE joint harder was to stiffen all six, which perturbs
    #: every joint and invalidates the validated ``kp_posture = 25``. Concretely:
    #: at Goal 1's pose ``wrist_2 = -90 deg`` is the entire reason the pose works
    #: (cond(J) 7.20 instead of 1395.76, hinge horizontal), yet it drifts 11.6 deg
    #: over a swing-up because posture holds it only as weakly as everything
    #: else. Excluding it from the task instead would hold it, but costs a rank:
    #: with joints {0,4} both excluded the 4x4 (X + 3 orientation) task matrix
    #: drops to rank 3, making one rotational direction uncontrollable -- exactly
    #: where the orientation guard already binds. A per-joint weight holds the
    #: joint WITHOUT removing it from the task, so rank 4 is preserved.
    #: TRUE ENFORCEMENT of joint motion, as opposed to the posture spring.
    #: ``posture_joint_weights`` above only makes a joint STIFFER -- it resists
    #: motion but nothing bounds it, so a large enough disturbance moves the
    #: joint as far as the torque budget allows. These rows put a high-order
    #: control barrier on |q_j - q_j(0)| directly in the QP, so a torque that
    #: would breach the bound is INFEASIBLE rather than merely penalised.
    #:
    #: Implemented by reusing ``_corridor_rows`` unchanged: that function builds
    #: the HOCBF pair for any scalar whose Jacobian row maps qdot to the
    #: scalar's rate, and for joint j that row is simply the unit vector e_j
    #: (qdot_j = e_j . qdot). No new barrier algebra.
    #:
    #: NOTE this can make the QP infeasible if the bound fights the task -- that
    #: is the point of a hard constraint, and it is why the half-width is
    #: explicit rather than defaulted tight.
    joint_corridor_enabled: bool = False
    joint_corridor_joints: tuple[int, ...] = ()
    joint_corridor_half_width_rad: float = 0.05
    joint_corridor_alpha1: float = 20.0
    joint_corridor_alpha2: float = 20.0

    posture_joint_weights: tuple[float, ...] | None = None

    task_excluded_joints: tuple[int, ...] = (0,)

    # --- Which translation axes are TRACKED vs BOUNDED ---------------------
    #: World translation rows of ``J`` that enter the QP objective as task
    #: rows (0 = X, 1 = Y, 2 = Z). Default ``(0,)`` = X only, the behavior
    #: this controller shipped with. The three orientation rows are always
    #: task rows and are not configurable -- there is no corridor formulation
    #: for orientation here.
    #:
    #: Added 2026-08-13, generalizing the previously hardcoded
    #: ``vstack([J[0:1,:], J[3:6,:]])``. Motivation, measured this session and
    #: not hypothetical: with ``task_excluded_joints=[0]`` every large-move
    #: failure at ARM_Q0 trips ``|Z-Z0| > 0.06 m`` -- Z drift is the binding
    #: constraint in both directions, sitting at 94-100% of its corridor
    #: half-width on every large move, while ``shoulder_lift`` sits nearly
    #: idle (0.11-1.33 deg of range). A constrained-IK sweep independently
    #: found continuous, well-conditioned solutions holding Z inside 0.05 m
    #: across the full +-0.20 m X range, so Z is holdable while X moves;
    #: tracking it is how that idle authority gets recruited.
    #:
    #: DELIBERATELY NOT Y. ``docs/status/neg45_y_axis_diagnosis_and_fix_
    #: 2026-08-01.md`` established that no P, D or I gain in this controller
    #: family can hold Y without breaking X-tracking at this pose family --
    #: Y is kinematically coupled to X here (dy ~ 0.975*dx). Y stays a
    #: corridor axis; adding it as a task row would re-litigate a question
    #: already answered three independent ways.
    task_axis_rows: tuple[int, ...] = (0,)
    #: World translation rows bounded by the Y/Z corridor HOCBF instead of
    #: tracked. Default ``(1, 2)`` = Y and Z, the shipped behavior. Must be
    #: DISJOINT from ``task_axis_rows`` (an axis cannot be both an objective
    #: term and a barrier constraint in the same solve) and may only contain
    #: 1 or 2, since the corridor half-widths are the per-axis
    #: ``y_corridor_half_width_m``/``z_corridor_half_width_m`` fields. May be
    #: empty. Validated jointly with ``task_axis_rows`` by
    #: ``_parse_axis_row_sets``.
    #:
    #: These two fields also decide which axes get the low-priority
    #: ``tau_yz_soft`` centering bias: it is applied to the CORRIDOR axes
    #: only. An axis that became a task row gets its authority from the task,
    #: and adding the bias on top would double-count it.
    corridor_axis_rows: tuple[int, ...] = (1, 2)

    #: WHICH FRAME ``task_axis_rows``/``corridor_axis_rows`` INDEX (2026-08-14).
    #:
    #: ``"world"`` (default, and the shipped behavior): rows 0/1/2 are world
    #: X/Y/Z. Nothing is rotated; the byte-identical guarantee for the
    #: default row set is untouched because the whole transform is skipped.
    #:
    #: ``"tool"``: the Jacobian's three POSITION rows are pre-rotated by
    #: ``R_tool^T`` before row selection, so rows 0/1/2 become tool X/Y/Z of
    #: the ``attachment_site`` frame. The three ORIENTATION rows are left
    #: alone -- they are already body-referenced through the quaternion error.
    #:
    #: Why this exists: at ARM_Q0 the pendulum hinge IS the tool Z axis
    #: (measured off the compiled model: tool Z is 89.73 deg from vertical,
    #: tool X 7.29 deg, tool Y 82.72 deg). Pumping quality
    #: ``kappa = |drive_axis x hinge|`` is 1.0000 along tool Y against 0.7165
    #: along world X, i.e. tool Y delivers 1.3958x the pendulum torque per
    #: unit cart motion. In world coordinates "track X, block Y" also fought
    #: the geometry, because the hinge lies on the world X/Y diagonal so the
    #: useful pumping direction had a large blocked-Y component; in tool
    #: coordinates the blocked axis (tool Z) is the one that provably does
    #: nothing to the pendulum (kappa = 0.0000).
    #:
    #: NOTE the row-0 rule in ``_parse_axis_row_sets`` still applies and still
    #: means "row 0 must be tracked" -- under ``"tool"`` that is tool X, not
    #: world X. The transport-axis guard is a ROW index check, not a world-axis
    #: check, so it is satisfied by the intended tool mapping [0, 1].
    #: Rows (world/task indices, same space as ``task_axis_rows``) tracked in
    #: VELOCITY ONLY: the position term ``kp*(pos_des - p)`` is dropped and the
    #: row becomes ``kd*(vel_des - v)``.
    #:
    #: WHY THIS IS A ROW MODE AND NOT A SEPARATE CONTROLLER (2026-08-18). A
    #: swing-up that must flip in one or two strokes needs a stiff, high-
    #: bandwidth inner loop; the obvious build is a joint-velocity PD alongside
    #: the QP, phase-switched. That would be a REGRESSION in the one thing this
    #: pose is short of: a plain velocity PD carries no corridor, orientation or
    #: manipulability CBF row, so drift protection would be weakest during the
    #: single most aggressive phase -- and drift, not actuation, is what has
    #: actually been ending these runs (measured: the LQR catch trips |Y-Y0| at
    #: dX=0.070/dY=0.059 identically for a_max in {9.603, 14, 20, 30}).
    #: Dropping one term from an existing task row keeps every CBF row, the
    #: torque box, the joint exclusion and the posture weighting untouched.
    #:
    #: NO GAIN RE-DERIVATION IS NEEDED, and that is a derivation rather than an
    #: omission: this repo's conversion is ``kp_QP = 400*Lambda`` and
    #: ``kd_QP = 40*Lambda``, where 40 is OSC's VELOCITY gain. So the existing
    #: kd_axis entry already IS the velocity-tracking gain for its row, in the
    #: frame it was fitted in. What changes is its ROLE (damping -> tracking),
    #: which is exactly the substitution AGENTS.md sec.7 warns about -- here the
    #: number survives the role change because both roles read the same 40*Lambda.
    #:
    #: Also removes one integrator from the loop: the drive law produces an
    #: acceleration that is otherwise double-integrated into a position target,
    #: so a velocity row is fed the FIRST integral and carries strictly less lag.
    #:
    #: Must be a subset of ``task_axis_rows`` -- a row that is not tracked has
    #: no position term to drop. Empty (default) => byte-identical behavior.
    # NOTE (2026-08-18): friction_feedforward / friction_ff_coulomb_nm /
    # friction_ff_viscous / friction_ff_qd_deadband are INHERITED from
    # CartesianImpedanceConfig and already parsed by its
    # from_controller_yaml_section (joint-name-keyed dicts for the two arrays).
    # They were present but simply never READ by this controller -- which is
    # what the parent config's header meant by "inherited but NOT wired into
    # this controller". The controller now reads them; redeclaring them here
    # would shadow the parent's parsing, so it deliberately does not.
    #
    # Set friction_ff_viscous explicitly to zeros in a config for this
    # controller: the parent's default (0.4/0.15) cancels the model's viscous
    # damping too, and the ablation that motivated wiring this up zeroed
    # frictionloss ONLY and settled cleanly with viscous damping intact.
    # Viscous damping is passive and stabilising -- there is no evidence this
    # controller needs it cancelled, and real evidence it does not.
    task_velocity_rows: tuple[int, ...] = ()

    task_frame: str = "world"

    #: EXPLICIT constant task basis, 3x3, columns = task axes in world coords
    #: (so ``R.T @ (p - p0)`` resolves displacement along them). ``None`` =
    #: unset, and ``task_frame`` decides. Mutually exclusive with
    #: ``task_frame: "tool"`` -- two different answers to the same question.
    #:
    #: Exists because the drive axis is hardwired to row 0
    #: (``parsing.TRANSPORT_AXIS_ROW``), so the frame determines what row 0
    #: physically IS, and at some poses NEITHER named frame can name the axis
    #: the task needs. Measured at ARM_Q0 (2026-08-16): shoulder_pan sits at
    #: -135.72 deg, so the arm's reachable vertical plane is 44.28 deg from
    #: world X -- world X is not producible, and tool X is near-vertical
    #: (zero hinge authority at the hanging equilibrium). The axis that IS
    #: wanted, the in-plane horizontal, is neither.
    #:
    #: Validated by ``controller_core.safety.validated_task_rotation`` -- the
    #: same checker the drift monitor uses, deliberately shared so a basis that
    #: is legal for the controller cannot be illegal for the guard watching it.
    #: A non-orthonormal basis would rescale measured drift and silently move
    #: the effective threshold.
    task_rotation: tuple[tuple[float, ...], ...] | None = None

    #: HOW ``R_tool`` IS UPDATED when ``task_frame == "tool"``. ``R_tool`` is
    #: time-varying -- the tool rotates as the arm moves, and this controller
    #: deliberately relaxes orientation -- so the choice is a real one and is
    #: resolved by measurement, not assumption:
    #:
    #: * ``"live"``    -- recompute every cycle. Task rows stay exactly aligned
    #:   with the true hinge, but the corridor's own direction rotates
    #:   underneath the barrier, whose bounds were captured at reset.
    #: * ``"frozen"``  -- snapshot once at ``reset_from_state`` and hold it.
    #:   The corridor is a fixed half-space (which is what an HOCBF with fixed
    #:   bounds assumes), but the task frame drifts off the true hinge.
    #: * ``"hybrid"``  -- live for the TRACKED task rows, frozen for the
    #:   CORRIDOR row. Keeps the task aligned with the hinge while leaving the
    #:   barrier a stationary constraint.
    #:
    #: Ignored entirely when ``task_frame == "world"``.
    task_frame_update: str = "frozen"

    # --- Y/Z corridor high-order CBF --------------------------------------
    #: Master switch for the four corridor rows (y_max, y_min, z_max, z_min).
    #: Default off: with this False the controller is a plain reduced-task box
    #: QP and nothing in this block is evaluated.
    yz_corridor_enabled: bool = False
    #: Half-width of the world-Y corridor, in meters, measured from the Y the
    #: end effector had at ``reset_from_state``. The barrier pair is
    #: ``h_max = (y0 + w) - y >= 0`` and ``h_min = y - (y0 - w) >= 0``.
    #:
    #: CALIBRATION (re-verified against this repo 2026-08-13, see the design
    #: doc's calibration section): ``ImpedanceSafetyMonitor``'s drift guards
    #: default to 0.03 m (``controller_core/safety.py``), but the one
    #: validated real dose-response measurement at the -45deg pose (dx=0.06 m)
    #: recorded a NATURAL, self-correcting Y transient peaking at 0.0423 m
    #: (docs/status/neg45_y_axis_diagnosis_and_fix_2026-08-01.md, and the
    #: evidence-scoped 0.05 m tolerance that followed it in
    #: docs/status/neg45_drift_tolerance_validation_2026-08-01.md). A corridor
    #: narrower than that natural transient would put this HOCBF in a fight
    #: with already-validated motion. 0.05 m is ~18% above the measured peak.
    #: That justification is scoped to dx <= 0.06 m -- the transient grows with
    #: displacement and was never measured beyond it, so do NOT assume 0.05 m
    #: still covers a 0.20 m move.
    y_corridor_half_width_m: float = 0.05
    #: Same, for world Z.
    z_corridor_half_width_m: float = 0.05
    #: The two linear class-K gains of the high-order CBF, 1/s. Used exactly
    #: as ``manipulability_cbf.py`` uses its own pair: the condition is
    #: ``hddot + (a1 + a2) hdot + a1 a2 h >= 0``, whose homogeneous solutions
    #: decay as ``exp(-a1 t)``/``exp(-a2 t)``. Bigger = the barrier starts
    #: pushing back sooner and harder (and from h < 0, recovers faster).
    yz_corridor_alpha1: float = 10.0
    yz_corridor_alpha2: float = 10.0

    # --- Manipulability CBF (reused verbatim) ------------------------------
    #: Adds ``controller_core/manipulability_cbf.py``'s singularity-avoidance
    #: row to the SAME QP. Requires a ``jacobian_fn`` on the constructor and a
    #: ``mass_matrix`` on the per-cycle state; both are checked loudly.
    manipulability_cbf: bool = False
    manipulability_cbf_epsilon: float = 1.0e-3
    manipulability_cbf_alpha1: float = 10.0
    manipulability_cbf_alpha2: float = 10.0
    manipulability_cbf_fd_step: float = 1.0e-5
    manipulability_cbf_curvature_step: float = 1.0e-4

    # --- Orientation high-order CBF (2026-08-15) ---------------------------
    #: Master switch for a FIFTH inequality row that bounds
    #: ``||orientation_error||^2`` instead of only tracking it via the 3
    #: rotation task rows in ``wrench_reduced``. Default off, so every
    #: existing config is bit-for-bit unchanged (asserted in
    #: ``tests/unit/test_x_task_yz_corridor_qp.py``).
    #:
    #: WHY THIS EXISTS. A cascade-LQR capture-envelope grid at ARM_Q0
    #: (wrist_2=-90deg, ``pendulum_attachment_realrod.xml``) found: plain OSC
    #: (kp_rot=0) captures 32/117 cells, failing 54 on ``|Y-Y0|`` drift and 31
    #: on orientation; this controller with the Y/Z corridor on (kp_rot=35.12)
    #: captures only 4/117, failing 0 on drift but 113/113 on orientation. The
    #: corridor's own lesson -- BOUNDING an axis beats rigidly TRACKING it,
    #: because rigid tracking of one axis measurably steals authority from
    #: others (the -45deg Y-drift investigation: "no P, D, or I gain ... can
    #: hold Y without breaking X-tracking") -- had never been applied to
    #: orientation itself, even though orientation is now the sole limiter.
    #: This row applies that same idea to orientation: instead of only
    #: pushing ``m_rot = kp_rot*e - kd_rot*omega`` toward zero (task row 2-4),
    #: add a barrier that keeps ``||e|| <= orientation_cbf_max_error_rad``
    #: (default 0.20 rad, INSIDE the 0.25 rad ``max_orientation_error_rad``
    #: guard, so the barrier acts before the guard trips) as a hard
    #: constraint of the SAME QP, one solve, no second filter.
    #:
    #: DERIVATION, ``h = theta_max^2 - e^T e`` (squared norm: smooth at
    #: ``e=0``, matching ``manipulability_cbf``'s own preference for a smooth
    #: barrier over ``theta_max - ||e||``, which is non-differentiable there):
    #:
    #:     hdot  = -2 e^T edot
    #:     hddot = -2 edot^T edot - 2 e^T eddot      (exact, no approximation)
    #:     eddot ~= J_r_ref M^-1 (tau - bias)         (dropping the Jdot term,
    #:                                                  the SAME standing
    #:                                                  approximation the Y/Z
    #:                                                  corridor rows already
    #:                                                  make)
    #:
    #: THE ONE THING THAT IS NOT A GUESS HERE (measured, not assumed):
    #: ``edot`` is NOT ``+-J_r qd`` in the world frame -- it is
    #: ``R_ref^T @ (J_r qd)``, i.e. the world angular velocity ROTATED INTO
    #: the frame of the fixed reference orientation ``self._quat0`` (the same
    #: ``self._R0`` this controller already snapshots at ``reset_from_state``
    #: for ``task_frame=="tool"``). Verified two ways: (1) analytically, from
    #: ``e = 2*vec(conj(q_ref)*q_cur)`` and the standard quaternion kinematics
    #: identity, ``conj(q_ref)*[0,omega_world]*q_ref = [0, R_ref^T omega_world]``
    #: -- the frame ``e`` lives in is q_ref's, not world's; (2) numerically,
    #: central/forward finite-differencing ``e`` over a real 1us step at 10
    #: random (q, qd) states at this pose: ``edot_fd`` matches
    #: ``R_ref^T @ (J_r qd)`` to 1e-7..1e-8 (float roundoff) at ``e=0``, and to
    #: 1-6% relative error (an ``O(||e||^2)`` small-angle residual, the same
    #: class of approximation ``orientation_error_vec_wxyz``'s own docstring
    #: already flags) at ``||e||`` up to 0.21 rad -- while raw ``+J_r qd``
    #: (no rotation) is wrong by 20-60% even AT ``e=0``, and raw ``-J_r qd``
    #: is wrong by 100-200%. Getting only the SIGN right and skipping the
    #: rotation would therefore still have been a real, load-bearing bug here
    #: (this pose's tool orientation is far from identity), not a cosmetic
    #: one -- this is exactly why the derivation is re-verified per-controller
    #: rather than copied from the position corridor rows, which have no such
    #: frame subtlety (world Y/Z are already world-frame quantities).
    #:
    #: Consequently the constraint row uses ``jac_rot_ref = self._R0.T @
    #: jac[3:6, :]`` (rows 3:6 of the FULL, un-rotated Jacobian -- orientation
    #: rows are never remapped by ``task_frame=="tool"`` either, see that
    #: field's docstring) wherever the naive derivation would use ``J_r``
    #: directly, and the resulting HOCBF row is
    #:
    #:     A = 2 e^T jac_rot_ref M^-1                                 (1, 6)
    #:     b = -2 edot^T edot + 2 e^T jac_rot_ref M^-1 bias
    #:         + (a1 + a2) hdot + a1 a2 h
    #:
    #: (the overall sign of A/b flips relative to a naive ``edot=-J_r qd``
    #: derivation too, since that assumption itself has the wrong sign here --
    #: see the controller module docstring's corresponding section for the
    #: fully worked algebra).
    #:
    #: ``self._R0`` is FIXED at ``reset_from_state`` (never updated mid-run,
    #: regardless of ``task_frame_update``), matching ``self._quat0`` -- the
    #: reference ``e`` is measured against -- exactly, so the two never
    #: desync.
    orientation_cbf: bool = False
    #: ``theta_max`` in the barrier above, radians. Default 0.20, strictly
    #: inside ``safety.max_orientation_error_rad`` (0.25 in every config that
    #: sets it, including this controller's own ``ur5e_mujoco_torque_x_task_
    #: yz_corridor_qp_enabled.yaml``) so the barrier has room to act before
    #: the hard e-stop-style guard does.
    orientation_cbf_max_error_rad: float = 0.20
    #: The two linear class-K HOCBF gains, 1/s -- same convention, same
    #: validator (``_parse_manipulability_cbf_alpha``), as
    #: ``yz_corridor_alpha1``/``yz_corridor_alpha2`` and
    #: ``manipulability_cbf_alpha1``/``manipulability_cbf_alpha2``.
    orientation_cbf_alpha1: float = 10.0
    orientation_cbf_alpha2: float = 10.0

    # --- Dual-ascent budget for solve_constrained_box_qp -------------------
    #: Same knobs, same defaults, as ``HardYConstraintQPConfig``. Cost scales
    #: roughly as ``dual_sweeps * m * dual_root_iters`` inner box solves, so
    #: these are the first thing to look at if the measured per-cycle solve
    #: time matters (see the design doc's timing section -- it does).
    dual_sweeps: int = 4
    dual_root_iters: int = 10

    # --- Warm-started QP solve (2026-08-26) --------------------------------
    #: Seed ``solve_constrained_box_qp`` from the PREVIOUS cycle's primal
    #: (``tau``) and dual (``lambda``) instead of a cold start. The control
    #: solution changes slowly cycle-to-cycle, so the dual coordinate-ascent
    #: and every inner projected-gradient box solve start near their answer and
    #: converge in far fewer iterations. Default OFF, so every existing config
    #: is byte-for-byte unchanged: only a config that sets this True takes the
    #: warm path, and the controller resets its warm buffers on
    #: ``reset_from_state``.
    #:
    #: WHY IT PRESERVES THE CONTROL LAW. Projected gradient with a fixed step is
    #: a contraction to the UNIQUE box optimum for the PD Hessian used here, so
    #: the starting iterate changes only the iteration COUNT, never the fixed
    #: point. On the ~99% of cycles where the QP optimum moves slowly the warm
    #: solve is far faster than cold and at least as accurate (up to ~12x more so
    #: at the hard near-wall instances the cold 80-iter solver badly under-
    #: converges).
    #:
    #: NOT UNIFORMLY AT LEAST AS ACCURATE AS COLD -- corrected 2026-08-26. At a
    #: fast transient the QP optimum can JUMP so the previous-cycle warm seed is
    #: FARTHER from the new optimum than a cold analytic start, and the reduced
    #: warm budget cannot recover in its inner iterations: one such cycle measured
    #: 4.80 Nm off a bit-stable reference where cold is 1.7e-4 (the barrier stays
    #: satisfied -- a tracking-torque defect, not a breach). ``qp_warm_fallback_tol``
    #: below is the safeguard: each warm cycle is convergence-gated and the rare
    #: non-converging one is redone cold, so accuracy is >= cold BY CONSTRUCTION.
    #: See the solver docstring, tests/unit/test_constrained_box_qp_warm_start.py,
    #: and tests/mujoco/test_corridor_qp_warm_start_safeguard.py.
    qp_warm_start: bool = False
    #: Inner ``max_iters`` used on warm cycles (a warm start reaches the same
    #: accuracy in ~20 iters that a cold start needs 80 for). Only consulted when
    #: ``qp_warm_start`` is True AND a warm buffer is available (the first cycle
    #: after a reset is still a cold 80-iter solve). Ignored otherwise.
    qp_warm_max_iters: int = 20
    #: CONVERGENCE-GATE THRESHOLD for the warm solve's cold fallback (2026-08-26).
    #: A warm start is fast on slowly-varying cycles, but at a fast transient the
    #: QP optimum can JUMP so the previous-cycle seed is farther from the new
    #: optimum than a cold analytic start, and the reduced warm budget cannot
    #: recover -- producing a large tracking-torque error (measured 4.80 Nm at one
    #: transient vs 1.7e-4 Nm cold; the barrier stays satisfied, so this is a
    #: tracking defect, not a breach). Each warm cycle
    #: ``solve_constrained_box_qp`` computes ONE cheap box-projected fixed-point
    #: residual and, if it exceeds this threshold, REDOES that cycle as a plain
    #: cold 80-iter solve (reliable here). So accuracy is >= cold by construction:
    #: warm keeps the speedup on the ~99% of cycles that pass, and the rare
    #: transient cycles fall back to cold. Passed as ``fallback_tol`` to the
    #: solver; only consulted on warm cycles. ``None`` disables the gate (warm is
    #: then trusted unconditionally -- the pre-gate behavior). Default 1e-3 is
    #: sized so the multi-Nm transient fails it while genuinely-converged warm
    #: cycles (residual ~1e-4 or below) pass; see the solver docstring and
    #: tests/mujoco/test_corridor_qp_warm_start_safeguard.py.
    qp_warm_fallback_tol: float | None = 1.0e-3

    @staticmethod
    def _parse_joint_corridor_joints(raw) -> tuple[int, ...]:
        """Validate joint indices: unique, in range, sorted for determinism."""
        if raw is None:
            return ()
        out = sorted({int(v) for v in raw})
        for j in out:
            if not (0 <= j <= 5):
                raise ValueError(
                    f"joint_corridor_joints entries must be in 0..5, got {j}"
                )
        return tuple(out)

    @staticmethod
    def _parse_posture_joint_weights(raw) -> tuple[float, ...] | None:
        """Validate ``posture_joint_weights``: 6 finite, non-negative floats."""
        if raw is None:
            return None
        vals = list(raw)
        if len(vals) != 6:
            raise ValueError(
                f"posture_joint_weights must have 6 entries (one per joint, "
                f"JOINT_NAME_ORDER), got {len(vals)}"
            )
        out = []
        for i, v in enumerate(vals):
            f = float(v)
            if not np.isfinite(f) or f < 0.0:
                raise ValueError(
                    f"posture_joint_weights[{i}] must be finite and >= 0, got {v!r}"
                )
            out.append(f)
        return tuple(out)

    @classmethod
    def from_controller_yaml_section(cls, ctrl: dict) -> "XTaskYZCorridorQPConfig":
        base = TorqueTaskQPConfig.from_controller_yaml_section(ctrl)
        base_kwargs = {
            f.name: getattr(base, f.name) for f in base.__dataclass_fields__.values()
        }
        gains = ctrl.get("gains", {}) or {}
        # The four soft-centering gains must be read from `gains` directly
        # rather than taken from `base`: TorqueTaskQPConfig's own parser
        # applies CartesianImpedanceConfig's HARD-task defaults (80/15/120/20)
        # when the YAML omits them, which would silently override this class's
        # deliberately-softer defaults for exactly the configs that never
        # mention these fields.
        base_kwargs["kp_y"] = float(gains.get("kp_y", 5.0))
        base_kwargs["kd_y"] = float(gains.get("kd_y", 2.0))
        base_kwargs["kp_z"] = float(gains.get("kp_z", 5.0))
        base_kwargs["kd_z"] = float(gains.get("kd_z", 2.0))
        # The six manipulability_cbf_* fields already exist on
        # CartesianImpedanceConfig (they are shared with the impedance
        # controller), so they are ALREADY keys of base_kwargs and must be
        # overwritten there rather than passed again as keyword arguments.
        # They still have to be re-read from `ctrl`: exactly like
        # nullspace_posture/lambda_regularization in
        # hard_constraint_qp.py::HardYConstraintQPConfig, they are not among
        # the fixed subset TorqueTaskQPConfig.from_controller_yaml_section
        # propagates, so `base` always carries the class default regardless of
        # what the YAML said.
        # Lambda-shaping trio: same "not in the propagated subset" situation as the
        # manipulability_cbf_* fields below -- the dataclass FIELDS exist (inherited
        # from CartesianImpedanceConfig) but nothing re-read them from `ctrl`, so a
        # YAML saying `task_space_inertia_shaping: true` silently produced False and
        # the flag could never be turned on from a config at all. Added 2026-08-15
        # together with the wrench-shaping block in controller.py; defaults keep the
        # historical (unshaped, force-domain) behavior bit-for-bit.
        base_kwargs["task_space_inertia_shaping"] = bool(
            ctrl.get("task_space_inertia_shaping", False)
        )
        base_kwargs["lambda_diagonal_shaping"] = bool(
            ctrl.get("lambda_diagonal_shaping", False)
        )
        _lam_reg = float(ctrl.get("lambda_regularization", base_kwargs.get("lambda_regularization", 1.0e-6)))
        if not math.isfinite(_lam_reg) or _lam_reg <= 0.0:
            raise ValueError(
                f"lambda_regularization must be finite and > 0; got {_lam_reg!r}"
            )
        base_kwargs["lambda_regularization"] = _lam_reg
        base_kwargs["manipulability_cbf"] = bool(ctrl.get("manipulability_cbf", False))
        base_kwargs["manipulability_cbf_epsilon"] = _parse_manipulability_cbf_epsilon(
            ctrl.get("manipulability_cbf_epsilon", 1.0e-3)
        )
        base_kwargs["manipulability_cbf_alpha1"] = _parse_manipulability_cbf_alpha(
            ctrl.get("manipulability_cbf_alpha1", 10.0), "manipulability_cbf_alpha1"
        )
        base_kwargs["manipulability_cbf_alpha2"] = _parse_manipulability_cbf_alpha(
            ctrl.get("manipulability_cbf_alpha2", 10.0), "manipulability_cbf_alpha2"
        )
        base_kwargs["manipulability_cbf_fd_step"] = float(
            ctrl.get("manipulability_cbf_fd_step", 1.0e-5)
        )
        base_kwargs["manipulability_cbf_curvature_step"] = float(
            ctrl.get("manipulability_cbf_curvature_step", 1.0e-4)
        )
        # None (key absent) means "use the class default", which is (0,) --
        # NOT "exclude nothing". A YAML that wants the old unconstrained
        # behavior has to say so explicitly with `task_excluded_joints: []`,
        # because silently reverting a safety-shaped default on an omitted key
        # is exactly the failure this repo keeps writing loud parsers against.
        excluded = _parse_task_excluded_joints(ctrl.get("task_excluded_joints", None))
        if excluded is not None:
            base_kwargs["task_excluded_joints"] = excluded
        # Parsed as a PAIR: the interesting failure is an axis in both sets,
        # which no independent per-field validator can see.
        task_rows, corridor_rows = _parse_axis_row_sets(
            ctrl.get("task_axis_rows", None), ctrl.get("corridor_axis_rows", None)
        )
        if task_rows is not None:
            base_kwargs["task_axis_rows"] = task_rows
        if corridor_rows is not None:
            base_kwargs["corridor_axis_rows"] = corridor_rows
        # FRICTION FEEDFORWARD: same "not in the propagated subset" trap as the
        # manipulability_cbf_* and lambda-shaping fields above. The dataclass
        # fields are inherited from CartesianImpedanceConfig and ITS parser reads
        # them, but TorqueTaskQPConfig.from_controller_yaml_section forwards only
        # a fixed subset, so without this block a YAML saying
        # `friction_feedforward: true` silently produces False and the term can
        # never be enabled from a config at all. Verified by re-parsing, not by
        # reading the file.
        #
        # Array form matches the parent's convention exactly -- joint-name-keyed
        # mappings over JOINT_NAME_ORDER -- so the same YAML block means the same
        # thing to both controllers.
        base_kwargs["friction_feedforward"] = bool(ctrl.get("friction_feedforward", False))
        for _key, _dflt in (
            ("friction_ff_coulomb_nm", (5.0, 5.0, 5.0, 1.0, 1.0, 1.0)),
            ("friction_ff_viscous", (0.4, 0.4, 0.4, 0.15, 0.15, 0.15)),
        ):
            if ctrl.get(_key) is not None:
                _raw = ctrl[_key]
                _vals = ([float(_raw[n]) for n in JOINT_NAME_ORDER]
                         if isinstance(_raw, dict) else [float(v) for v in _raw])
                if len(_vals) != 6:
                    raise ValueError(f"{_key} must have 6 entries, got {len(_vals)}")
                if not all(np.isfinite(_vals)):
                    raise ValueError(f"{_key} contains NaN/Inf: {_vals}")
                if any(v < 0.0 for v in _vals):
                    # A negative entry adds friction-shaped torque ALONG the motion,
                    # i.e. negative damping -- the opposite of compensation.
                    raise ValueError(f"{_key} entries must be >= 0; got {_vals}")
                base_kwargs[_key] = np.asarray(_vals, dtype=np.float64)
            else:
                base_kwargs[_key] = np.asarray(_dflt, dtype=np.float64)
        _db = float(ctrl.get("friction_ff_qd_deadband", 0.05))
        if not (_db > 0.0):
            raise ValueError(f"friction_ff_qd_deadband must be > 0; got {_db}")
        base_kwargs["friction_ff_qd_deadband"] = _db

        raw_vel_rows = ctrl.get("task_velocity_rows", None)
        if raw_vel_rows is not None:
            vel_rows = tuple(sorted({int(r) for r in raw_vel_rows}))
            if any(r < 0 or r > 2 for r in vel_rows):
                raise ValueError(
                    f"task_velocity_rows entries must be in 0..2; got {vel_rows}")
            declared_task = tuple(base_kwargs.get("task_axis_rows", (0,)))
            missing = [r for r in vel_rows if r not in declared_task]
            if missing:
                # A row with no position term to drop is a silent no-op, and a
                # silent no-op here looks exactly like "velocity mode did not
                # help" -- refuse instead.
                raise ValueError(
                    f"task_velocity_rows {vel_rows} must be a subset of "
                    f"task_axis_rows {declared_task}; {missing} are not tracked rows")
            base_kwargs["task_velocity_rows"] = vel_rows

        frame, frame_update = _parse_task_frame(
            ctrl.get("task_frame", None), ctrl.get("task_frame_update", None)
        )
        if frame is not None:
            base_kwargs["task_frame"] = frame
        if frame_update is not None:
            base_kwargs["task_frame_update"] = frame_update
        raw_rot = ctrl.get("task_rotation", None)
        if raw_rot is not None:
            # Validated with the SAME checker the drift monitor uses, so a basis
            # legal here cannot be illegal for the guard watching this run.
            # Local import: importing safety at module scope would add an import
            # cycle for a path most configs never take. (numpy is imported at
            # module scope now -- re-importing it HERE made `np` a function-local
            # name for the whole method, so any earlier use of np in this same
            # method raised UnboundLocalError. The comment that used to sit here
            # claiming the module is numpy-free is no longer true.)
            from controller_core.safety import validated_task_rotation

            if frame == "tool":
                raise ValueError(
                    "task_rotation and task_frame: 'tool' are mutually exclusive -- they "
                    "are two different answers to 'what basis are the task rows in'. Set "
                    "one. (task_rotation is the fixed-basis answer; 'tool' is the "
                    "follow-the-tool answer, and task_frame_update then chooses how it "
                    "tracks.)"
                )
            rot = validated_task_rotation(np.asarray(raw_rot, dtype=np.float64))
            base_kwargs["task_rotation"] = tuple(tuple(float(v) for v in row) for row in rot)
        return cls(
            **base_kwargs,
            posture_joint_weights=cls._parse_posture_joint_weights(
                ctrl.get("posture_joint_weights")
            ),
            joint_corridor_enabled=bool(ctrl.get("joint_corridor_enabled", False)),
            joint_corridor_joints=cls._parse_joint_corridor_joints(
                ctrl.get("joint_corridor_joints", ())
            ),
            joint_corridor_half_width_rad=float(
                ctrl.get("joint_corridor_half_width_rad", 0.05)
            ),
            joint_corridor_alpha1=float(ctrl.get("joint_corridor_alpha1", 20.0)),
            joint_corridor_alpha2=float(ctrl.get("joint_corridor_alpha2", 20.0)),
            yz_corridor_enabled=bool(ctrl.get("yz_corridor_enabled", False)),
            y_corridor_half_width_m=_parse_corridor_half_width(
                ctrl.get("y_corridor_half_width_m", 0.05), "y_corridor_half_width_m"
            ),
            z_corridor_half_width_m=_parse_corridor_half_width(
                ctrl.get("z_corridor_half_width_m", 0.05), "z_corridor_half_width_m"
            ),
            yz_corridor_alpha1=_parse_manipulability_cbf_alpha(
                ctrl.get("yz_corridor_alpha1", 10.0), "yz_corridor_alpha1"
            ),
            yz_corridor_alpha2=_parse_manipulability_cbf_alpha(
                ctrl.get("yz_corridor_alpha2", 10.0), "yz_corridor_alpha2"
            ),
            dual_sweeps=int(ctrl.get("dual_sweeps", 4)),
            dual_root_iters=int(ctrl.get("dual_root_iters", 10)),
            qp_warm_start=bool(ctrl.get("qp_warm_start", False)),
            qp_warm_max_iters=int(ctrl.get("qp_warm_max_iters", 20)),
            qp_warm_fallback_tol=(
                None if ctrl.get("qp_warm_fallback_tol", 1.0e-3) is None
                else float(ctrl.get("qp_warm_fallback_tol", 1.0e-3))
            ),
            orientation_cbf=bool(ctrl.get("orientation_cbf", False)),
            orientation_cbf_max_error_rad=_parse_manipulability_cbf_alpha(
                ctrl.get("orientation_cbf_max_error_rad", 0.20),
                "orientation_cbf_max_error_rad",
            ),
            orientation_cbf_alpha1=_parse_manipulability_cbf_alpha(
                ctrl.get("orientation_cbf_alpha1", 10.0), "orientation_cbf_alpha1"
            ),
            orientation_cbf_alpha2=_parse_manipulability_cbf_alpha(
                ctrl.get("orientation_cbf_alpha2", 10.0), "orientation_cbf_alpha2"
            ),
        )
