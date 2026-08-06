"""The four end-effector orientation/redundancy resolution strategies for
CartesianVelocityController.compute(). Each function here is a standalone,
side-effect-free (except where noted) implementation of one mode -- the
controller orchestrator in ``controller.py`` does the shared setup (position
error, swing-twist orientation error, xd_full), picks exactly one of these
based on ``cfg``, and applies the shared speed clamp to whatever xd_cmd comes
back.

reduced_task_dims (2026-08-03, default on): a first version of this
controller P-held all 3 rotation axes directly (w_cmd = kp_rot * e_rot for
rx/ry/rz together). Sim characterization at the -40deg/wrist_2=0.2-offset
pose found this caps the safe X-transport range at ~0.047m before the SAME
wrist_2=0 kinematic singularity found earlier this session for the torque-
control lane: holding full 3D orientation while translating purely in X
drives wrist_2 back toward zero (cond(J) 29->275 measured over ~0.047m of
travel), a genuine kinematic reachability constraint of this pose/task
combination, independent of control law. The torque-control lane's fix was
reduced_task_dims -- dropping rx/ry from the task entirely so the redundant
DOF is free to absorb whatever motion the kinematics actually need. The
same fix applies here, implemented via real redundancy resolution rather
than just zeroing commanded angular velocity for rx/ry (zeroing wx/wy
outright is NOT equivalent: commanding "hold angular velocity about x/y at
exactly zero" is itself still a rank-6 constraint that would drive wrist_2
the same way full orientation-holding does). Instead:
  1. Build J_task = rows of J(q) for the enabled task dims only (always
     x/y/z, plus rz by default; rx/ry off by default).
  2. xd_task = the same P+feedforward law, restricted to those rows.
  3. qd = pinv(J_task) @ xd_task -- the minimum-norm joint velocity
     achieving the reduced task exactly, letting the spare DOF (here,
     effectively wrist rotation) go wherever the minimum-norm solution
     sends it, unconstrained.
  4. xd_consistent = J(q) @ qd -- project back through the FULL Jacobian so
     the 6D vector actually sent to speedL reflects the real resulting
     motion (including whatever rx/ry velocity falls out of the null-space
     motion), not a fabricated zero. This is what "solve for end-effector
     orientation holding" (rather than just dropping it) means here: rz is
     actively held, rx/ry are left to the kinematics rather than either
     forced to zero or forced to match a P-law.

Pure minimum-norm pinv(J_task) alone was tried first and found insufficient
(2026-08-03): with NOTHING resolving the null-space freedom beyond "smallest
||qd||", rx/ry drift wherever the geometry happens to send them, which
measured as LARGE (tripping the 0.25 rad orientation guard even for a small
dx=0.02m move) -- minimum joint-velocity norm is not the same criterion as
small orientation drift. Fixed by adding a null-space-projected secondary
objective, exactly analogous to XAxisCartesianImpedanceController's
nullspace_posture (controller_core/x_axis_cartesian_impedance.py): a joint-
velocity term pulling q toward q_rest (captured at reset), projected through
N = I - pinv(J_task) @ J_task so it provably cannot perturb the primary
X/Y/Z/rz task (J_task @ N == 0 by construction of the pinv-based projector).
qd = pinv(J_task) @ xd_task + N @ (kp_posture * (q_rest - q)). Since q_rest
is the pose with zero orientation error by construction, pulling toward it
also pulls rx/ry back down -- gently, as a low-priority secondary objective,
not a rigid lock -- which is the actual mechanism by which end-effector
orientation ends up held even though rx/ry are no longer in the primary
task. kp_posture=0.0 reproduces the pre-fix minimum-norm-only behavior
(still available as an explicit opt-out); the class default is nonzero
because reduced_task_dims=True needs it to be practically usable.

pinv_damping (see math_utils.py's _damped_pinv) was ALSO required, not
optional: a (damping=0, kp_posture>0) sweep found qd swinging wildly and
non-monotonically (3.1 -> 11.7 -> 2.5 -> 21.4 rad/s as kp_posture stepped
0.05 -> 0.1 -> 0.2 -> 1.0) near the wrist_2 singularity -- a numerically
ill-conditioned regime, not a real physical requirement. A second sweep
over (pinv_damping, kp_posture) pairs found damping has to stay SMALL
(0.005 -- the class default) to retain enough of the posture correction's
real effect to matter (heavier damping, e.g. 0.01-0.05, bounds qd nicely
but also neuters the correction enough that orientation still trips the
guard); kp_posture in roughly [0.5, 2.0] all work about equally well at
that damping. At the -40deg/wrist_2=0.2-offset pose, dx=0.04m/1.0s:
damping=0.005/kp_posture=1.0 gives max|qd|=1.23 rad/s (was 21.4 undamped)
and orientation=0.178 rad (was tripping the 0.25 rad guard with zero or
heavily-damped posture correction).

posture_reanchor_on_settle (default on, mirrors the torque controller's
flag of the same name): fixes a real bug where the posture term kept
pulling toward the STALE reset-time q_rest even after a move completed and
the arm was correctly holding a new X position -- q_rest is re-captured at
the current q once position error stays under reanchor_pos_tol_m for
reanchor_settle_cycles CONSECUTIVE cycles (not a single instant -- a
single-cycle check falsely fires at t=0, where a min-jerk profile trivially
starts at rest with zero position error, producing a same-as-reset no-op
reanchor that never catches the real post-move settle).

The null-space projector N = I - pinv(J_task) @ J_task deliberately uses
the UNDAMPED pinv even when pinv_damping > 0 for qd_primary: N is a true
orthogonal projector (symmetric, PSD, so the posture feedback loop is
provably stable) only for the exact Moore-Penrose pinv -- with the damped
one, that guarantee is lost.

**Root cause of the original unbounded-hold-phase-divergence bug, found and
fixed 2026-08-03**: the primary task's rz-row used ``kp_rot *
orientation_error_vec_wxyz(...)[2]`` (a small-angle ``2*vec(q_err)``
approximation of the FULL 3D rotation error) as "the yaw error" -- this
does not decompose per-axis for large compound rotations, so once rx/ry
(left free by reduced_task_dims) accumulated real rotation via the null-
space motion, the "yaw correction" itself became contaminated by rx/ry,
closing a genuine positive-feedback loop (confirmed independent of
kp_posture: identical divergence with kp_posture=0.0, i.e. no null-space
term at all -- proving the bug was in the PRIMARY task's own signal, not
the posture correction). Fixed by ``swing_twist_axis_error`` (see
``kinematics_utils.py``): an exact swing-twist decomposition, genuinely
axis-separable for any rotation magnitude, used for all three rows of
w_cmd instead of the coupled small-angle vector (this fix lives in
``controller.py``'s shared setup, since it applies to xd_full for every
mode, not just reduced_task_dims).

**Structural finding, NOT fixable via gain tuning, in reduced_task_dims and
split_base_wrist_task (2026-08-03)**: with the swing-twist fix, orientation
error no longer diverges -- it always converges to SOME finite equilibrium.
But WHICH equilibrium is highly sensitive to the exact displacement, not
smooth in dx: a fine dx sweep at this pose (0.005 to 0.045m, reduced_task_
dims) found settled orientation error jumping between "acceptable"
(~0.18-0.32 rad at dx=0.03-0.04m) and "large" (~0.94-2.0 rad at dx=0.02,
0.025, 0.045m) with no smooth trend -- textbook MULTISTABILITY: minimum-
norm redundancy resolution has multiple basins of attraction, and which
one a given trajectory falls into depends on its exact path through
configuration space, not just its endpoint. Confirmed NOT a gain-tuning
problem: a kp_posture sweep from 0 to 200 left the settled value in each
basin essentially UNCHANGED, and at kp_posture>=20 near the more singular
displacements qd blew up to hundreds-to-thousands of rad/s instead.
split_base_wrist_task (below) was tried as a structural alternative
(exact, non-redundant per-joint-group mapping instead of minimum-norm) --
it DOES eliminate the multistability (smooth, monotonic trend in dx), but
trades it for uniformly poor orientation holding everywhere (~0.6-1.8 rad
at every dx tested) because base-joint motion induces real, uncorrected
rotation (jac[3:6,0:3] has Frobenius norm 1.73 at this pose, a large
fraction of the full rotation block's own norm 2.45) that the wrist-only
correction never accounts for; a follow-up task-priority compensation
attempt (subtracting the position task's induced rotation from the
orientation target) made this WORSE, not better (qd spiked past 20,000
rad/s) -- reported here as a real, verified negative result, not
speculation.

**Actual fix: ik_seeded_resolution (2026-08-03)**. Neither reduced_task_
dims/split_base_wrist_task's rate-integrated null-space walk is path-
independent by construction -- qd is computed from q_CURRENT each cycle,
so the redundancy resolution has memory of how the arm got there. This
mode replaces that with a fresh Newton-Raphson IK solve, seeded from
q_rest EVERY CYCLE (never from q_current), targeting the reduced task
(selected position + orientation dims) via ik_iterations damped-pinv
Newton steps -- producing a joint-space target q_target that is a
deterministic function of ONLY (q_rest, current target pose), with zero
dependence on path history (verified directly: recovering q_target from
two completely different starting q_current values, simulating two
different move histories, gives identical results to 1e-16 -- see
test_ik_seeded_resolution_q_target_is_path_independent). The commanded
velocity is then a plain joint-space P-controller (ik_joint_gain) driving
q_current toward q_target, converted to Cartesian velocity via the FULL
current Jacobian. Measured at the -40deg/wrist_2=0.2-offset pose: smooth,
monotonic, PREDICTABLE orientation error growing from 0.04 rad (dx=0.005m)
to 0.41 rad (dx=0.05m), with near-perfect X-tracking (>99.8%) throughout
-- a precise, repeatable safety boundary at dx~0.029m (0.2432 rad, passes)
vs. dx~0.03m (0.2500 rad, fails), not a chaotic jump between adjacent
values the way reduced_task_dims showed. Compute cost measured negligible:
~0.033ms per fk_jacobian_fn call, ~0.23ms for a full 6-iteration solve --
~3% of the 8ms/125Hz real-time budget.

Requires ``state['fk_jacobian_fn']``: a callable ``q -> (ee_pos, ee_quat,
jacobian)`` usable at ARBITRARY q (not just the current one) -- controller_
core stays simulator-independent (numpy only), so this callable must come
from the caller (hardware/velocity_transport.py and tools/diagnostics/
ur5e_velocity_control_kinematic_sim.py both wire a MuJoCo-backed version;
hardware/local_dynamics.py::LocalMujocoDynamics.fk_and_jacobian is the
real-hardware implementation). Mutually exclusive with reduced_task_dims
and split_base_wrist_task (at most one of the three may be on) -- that
check lives in ``controller.py``'s shared setup.

DO NOT treat reduced_task_dims=True (the current default) or
split_base_wrist_task as validated for real-hardware use across a
displacement range -- both have real, verified failure modes described
above. ik_seeded_resolution is the one mode with a predictable, monotonic
safety boundary measured in sim; it has NOT yet been tried on real
hardware, and the default is still reduced_task_dims=True pending a
decision to promote ik_seeded_resolution (not done unilaterally here).

Requires ``jacobian`` (6x6, world-frame [J_pos; J_rot], mj_jacSite's own
convention) and ``q`` in the state dict. If reduced_task_dims is on and
jacobian is absent, compute() raises rather than silently falling back to
the old full-hold law (a silent behavior change here is worse than an
explicit error). Set reduced_task_dims=False for the original, jacobian-
free, full-3-axis-hold behavior.
"""

from __future__ import annotations

import numpy as np

from ..box_qp import build_weighted_least_squares_qp, solve_box_qp
from ..kinematics_utils import swing_twist_axis_error
from .math_utils import _damped_pinv


def compute_ik_seeded(
    cfg,
    fk_jacobian_fn,
    q_current: np.ndarray,
    p_des: np.ndarray,
    quat0: np.ndarray,
    q_rest: np.ndarray,
) -> np.ndarray:
    """ik_seeded_resolution mode. See this module's docstring ("Actual fix:
    ik_seeded_resolution") for the full design rationale."""
    if fk_jacobian_fn is None:
        raise ValueError(
            "CartesianVelocityConfig.ik_seeded_resolution=True requires "
            "state['fk_jacobian_fn'], a callable q -> (ee_pos, ee_quat, jacobian) "
            "usable at ARBITRARY q, not just the current one -- controller_core "
            "stays simulator-independent, so this must be supplied by the caller "
            "(e.g. a MuJoCo-backed forward-kinematics wrapper), not built in here."
        )
    rot_flags = [cfg.task_dim_rx, cfg.task_dim_ry, cfg.task_dim_rz]
    selected = [0, 1, 2] + [3 + i for i, on in enumerate(rot_flags) if on]

    # Fresh Newton-Raphson IK solve SEEDED FROM q_rest every cycle --
    # NOT from q_current. This is the actual fix for reduced_task_dims'
    # path-dependent multistability (see this module's docstring): q_target
    # is a deterministic function of ONLY (p_des, q_rest) -- the same
    # target position always produces the same q_target, regardless of how
    # the arm got to its current configuration, because the solve never
    # looks at q_current at all. This trades per-cycle compute cost (up to
    # ik_iterations extra forward-kinematics/Jacobian evaluations) for
    # eliminating the redundancy-resolution history-dependence that made
    # reduced_task_dims' and split_base_wrist_task's rate-integrated
    # null-space walks unpredictable.
    # Joint position bounds for the QP step below -- permissive
    # (effectively unconstrained) unless the caller supplied real
    # UR5e limits, since controller_core has no hardware knowledge
    # of its own.
    q_lo = (
        cfg.joint_pos_lower
        if cfg.joint_pos_lower is not None
        else np.full(6, -2.0 * np.pi, dtype=np.float64)
    )
    q_hi = (
        cfg.joint_pos_upper
        if cfg.joint_pos_upper is not None
        else np.full(6, 2.0 * np.pi, dtype=np.float64)
    )
    task_w = max(float(cfg.qp_task_weight), 1.0e-6)
    reg = max(float(cfg.pinv_damping), 1.0e-9) ** 2  # matches _damped_pinv's own d^2 scale

    # ik_posture_activation_joint_dev_rad (added 2026-08-06, replaced an
    # orientation-error-based version the SAME day -- see this module's
    # git history): gates the posture term ON only when the REAL, joint-
    # space deviation ||q_current - q_rest|| already exceeds the
    # threshold. An earlier version gated on real orientation error
    # instead (fixing a first, even earlier bug -- gating on the internal
    # solve's q_rest-relative error, which stays small regardless of real
    # drift). Orientation error genuinely fixed the one real case it was
    # built for (hanging_alpha_0_5's -X failure island), but root-causing
    # a SECOND real failure class (neg40/neg45_wrist2offset's joint_
    # velocity_guard trips) found it structurally can't serve both: at
    # neg40's trip, real orientation error was only 0.053 rad (the
    # failure is q_target's wrist_2 component itself running away, a
    # joint-space phenomenon, not an orientation one) while hanging's own
    # trip needed ~0.25 rad -- a ~5x mismatch no single global threshold
    # can serve. Traced directly: ||q-q_rest|| at the moment of failure is
    # far more comparable across the two cases (0.54 vs 0.39 rad) AND
    # grows at a similar RATE throughout both trajectories (e.g. both
    # reach ~0.2 by roughly the same elapsed time), making it a much
    # better pose-agnostic "something is going wrong" signal -- and it's
    # cheaper too: no extra fk_jacobian_fn call needed, q_current and
    # q_rest are already function arguments. None (default) reproduces
    # ik_posture_gain's unconditional prior behavior unchanged.
    posture_active = cfg.ik_posture_gain != 0.0
    if posture_active and cfg.ik_posture_activation_joint_dev_rad is not None:
        activation_limit = abs(float(cfg.ik_posture_activation_joint_dev_rad))
        posture_active = float(np.linalg.norm(q_current - q_rest)) >= activation_limit

    q_k = q_rest.copy()
    for _ in range(max(int(cfg.ik_iterations), 1)):
        p_k, quat_k, jac_k = fk_jacobian_fn(q_k)
        p_k = np.asarray(p_k, dtype=np.float64).reshape(3)
        quat_k = np.asarray(quat_k, dtype=np.float64).reshape(4)
        jac_k = np.asarray(jac_k, dtype=np.float64).reshape(6, 6)
        pos_err_k = p_des - p_k
        rot_err_k = np.array(
            [swing_twist_axis_error(quat0, quat_k, i) for i in range(3)],
            dtype=np.float64,
        )
        task_err_full_k = np.concatenate([pos_err_k, -rot_err_k])
        j_task_k = jac_k[selected, :]
        task_err_k = task_err_full_k[selected]

        # QP-constrained Newton step, replacing the plain damped
        # pseudoinverse: minimize reg*||dq||^2 + task_w*||J_task@dq
        # - task_err||^2 [+ (ik_posture_gain*task_w)*||dq-(q_rest-q_k)||^2]
        # subject to q_lo <= q_k+dq <= q_hi, via the shared
        # build_weighted_least_squares_qp + solve_box_qp interface
        # (controller_core/box_qp.py). task_w large (default 1e4) makes
        # this closely match the unconstrained damped pinv AWAY from any
        # joint limit (verified: byte-compatible when joint_pos_lower/
        # upper are None, i.e. the box bounds are permissive +-2pi and
        # never bind) while GUARANTEEING q_k+dq never exceeds a real
        # joint limit when one is supplied, unlike the plain Newton step,
        # which has no way to represent that at all.
        #
        # ik_posture_gain (2026-08-06, INTEGRATED into this same QP as of
        # this date -- see this module's own git history for the original
        # solve-then-patch nullspace-projected version, replaced rather
        # than kept alongside this one, per this repo's "reduce ambiguity/
        # redundancy" direction): fixes a real gap -- the reg*||dq||^2
        # term alone is the ONLY thing regularizing whatever's outside the
        # selected task dims (e.g. rx/ry when task_dim_rx/ry are False),
        # and it's a pure minimum-STEP-norm bias, not a pull toward q_rest
        # specifically -- with pinv_damping near zero and qp_task_weight
        # near-infinite (exactly the direction a real gain search pushed
        # toward, since it sharpens task tracking), that leaves the
        # unconstrained axes free to drift. Traced directly: rz (the
        # task-constrained axis) tracked to within 2e-4 rad throughout a
        # failing episode while rx (unconstrained) grew to 0.25 rad and
        # tripped the orientation guard alone.
        #
        # This integrated formulation (one QP, task + posture terms both
        # inside the SAME weighted least-squares objective) replaced an
        # earlier version that solved the task QP alone, THEN added a
        # separately-computed null-space-projected posture correction
        # afterward, then re-clipped the combined step -- that version had
        # an EXACT guarantee of zero task perturbation (via a true
        # orthogonal null-space projector) but validated poorly against a
        # real failing case: at the extreme (pinv_damping, qp_task_weight)
        # a real gain search converged to, solve_box_qp's underlying
        # linear solve was already at the edge of double-precision
        # conditioning, and adding a separately-projected correction on
        # top barely moved the outcome even at posture gains up to 128.
        # This version's posture weight is added DIRECTLY to the same
        # hessian the task term populates, which is itself a regularizer
        # (well-conditioned in every direction, including whatever the
        # task leaves unconstrained) rather than a correction bolted on
        # after an already near-singular solve.
        #
        # ik_posture_gain is a FRACTION OF task_w, not an absolute weight
        # (found and fixed the same day it was added): an absolute weight
        # in a small, easy-to-search range like [0,50] is negligible once
        # qp_task_weight lands anywhere near the 1e8-1e10 range real
        # searches converge toward -- confirmed directly, sweeping the
        # absolute version up to 128 against a real failing case never
        # moved the peak orientation error past 0.2525 rad (still failing
        # the 0.25 rad guard), while sweeping the SAME case with weight
        # comparable to qp_task_weight (i.e. what ik_posture_gain~0.5-5
        # now means) genuinely fixed it (peak orientation error down to
        # 0.04-0.21 rad, guard-free). Scaling by task_w makes
        # ik_posture_gain's effective strength automatically track
        # wherever qp_task_weight ends up, instead of needing a search
        # bound spanning many more orders of magnitude than is practical
        # to sample linearly (0 must remain exactly representable as
        # "off," which a log-scaled field can't do).
        dq_lo = q_lo - q_k
        dq_hi = q_hi - q_k
        terms = [(j_task_k, task_err_k, task_w)]
        if posture_active:
            terms.append((np.eye(6, dtype=np.float64), q_rest - q_k, float(cfg.ik_posture_gain) * task_w))
        hessian_qp, linear_qp = build_weighted_least_squares_qp(terms, reg=reg)
        dq = solve_box_qp(hessian_qp, linear_qp, dq_lo, dq_hi)
        q_k = q_k + dq
    q_target = q_k

    qd_joint = cfg.ik_joint_gain * (q_target - q_current)
    if cfg.joint_vel_limit_radps is not None:
        qd_joint = np.clip(qd_joint, -abs(cfg.joint_vel_limit_radps), abs(cfg.joint_vel_limit_radps))
    _, _, jac_current = fk_jacobian_fn(q_current)
    jac_current = np.asarray(jac_current, dtype=np.float64).reshape(6, 6)
    xd_cmd = (jac_current @ qd_joint).astype(np.float64)
    return xd_cmd


def compute_split_base_wrist(
    cfg,
    jacobian: np.ndarray | None,
    v_cmd: np.ndarray,
    xd_full: np.ndarray,
) -> np.ndarray:
    """split_base_wrist_task mode. See this module's docstring
    ("Structural finding, NOT fixable via gain tuning") for the full design
    rationale, including the rejected task-priority-compensation attempt."""
    if jacobian is None:
        raise ValueError(
            "CartesianVelocityConfig.split_base_wrist_task=True requires "
            "state['jacobian'] (6x6) every cycle."
        )
    jac = np.asarray(jacobian, dtype=np.float64).reshape(6, 6)

    # Position (X/Y/Z) routed through BASE joint columns (shoulder_pan,
    # shoulder_lift, elbow) ONLY -- wrist columns are structurally
    # zeroed, an EXACT 3x3 solve (no redundancy at all, unlike
    # reduced_task_dims' minimum-norm pick over all 6 joints), ported
    # from XAxisCartesianImpedanceController's split_base_wrist_task
    # (same evidence: at the wrist_2=0 pose, cond(3x3 pos-rows x
    # base-cols) ~7.8 (well-conditioned) vs. cond(full 6x6 J) ~7e16
    # (numerically singular) -- position tracking structurally cannot
    # be dragged into the wrist singularity because it never routes
    # through wrist columns at all, regardless of J's conditioning
    # there).
    j_pos_task = np.zeros((3, 6), dtype=np.float64)
    j_pos_task[:, 0:3] = jac[0:3, 0:3]
    qd_pos = _damped_pinv(j_pos_task, cfg.pinv_damping) @ v_cmd

    # Orientation routed through WRIST joint columns (wrist_1/2/3)
    # ONLY -- qd_pos and qd_rot are disjoint in JOINT space (zero
    # overlap: qd_pos always has exactly zero in columns 3:6, qd_rot
    # always has exactly zero in columns 0:3), but that alone does
    # NOT mean the resulting CARTESIAN rotation is decoupled from
    # qd_pos: jac[3:6, 0:3] (rotation induced by BASE joint motion)
    # is NOT structurally zero for a real arm -- measured directly
    # at this pose: Frobenius norm 1.73, a large fraction of the
    # full rotation block's own norm 2.45. A first version of this
    # code solved qd_rot against the RAW target w_cmd and found
    # uniformly poor orientation holding at every dx tested (0.7-1.8
    # rad, vs. the 0.25 rad guard) plus degraded X-tracking and qd
    # blowups at larger dx -- because qd_pos's own base-joint motion
    # was inducing real, uncorrected rotation on top of whatever
    # qd_rot was independently trying to hold. Fixed with a task-
    # PRIORITY sequential solve: qd_rot targets the RESIDUAL
    # rotation error after subtracting what qd_pos's own motion
    # already induces, not the raw w_cmd.
    rot_flags = [cfg.task_dim_rx, cfg.task_dim_ry, cfg.task_dim_rz]
    rot_selected = [3 + i for i, on in enumerate(rot_flags) if on]
    qd_rot = np.zeros(6, dtype=np.float64)
    if rot_selected:
        w_induced_by_pos = jac[3:6, :] @ qd_pos
        j_rot_task = np.zeros((len(rot_selected), 6), dtype=np.float64)
        j_rot_task[:, 3:6] = jac[np.ix_(rot_selected, [3, 4, 5])]
        xd_rot_task = xd_full[rot_selected] - w_induced_by_pos[[i - 3 for i in rot_selected]]
        qd_rot = _damped_pinv(j_rot_task, cfg.pinv_damping) @ xd_rot_task

    qd = qd_pos + qd_rot
    xd_cmd = (jac @ qd).astype(np.float64)
    return xd_cmd


def compute_reduced_task_dims(
    cfg,
    jacobian: np.ndarray | None,
    q: np.ndarray,
    xd_full: np.ndarray,
    pos_err: np.ndarray,
    v_ff: np.ndarray,
    q_rest: np.ndarray,
    reanchored: bool,
    settled_cycles: int,
) -> tuple[np.ndarray, np.ndarray, bool, int]:
    """reduced_task_dims mode (the class default). See this module's
    docstring (top, and "Pure minimum-norm pinv(J_task) alone") for the
    full design rationale.

    Returns (xd_cmd, q_rest, reanchored, settled_cycles) -- q_rest/
    reanchored/settled_cycles may be updated by the settle-based posture
    reanchor below, and the caller (controller.py) is responsible for
    persisting them back onto the controller instance, since a plain
    function can't mutate the caller's attributes."""
    if jacobian is None:
        raise ValueError(
            "CartesianVelocityConfig.reduced_task_dims=True requires "
            "state['jacobian'] (6x6) every cycle -- set reduced_task_dims=False "
            "for the original jacobian-free full-orientation-hold behavior."
        )
    jac = np.asarray(jacobian, dtype=np.float64).reshape(6, 6)
    rot_flags = [cfg.task_dim_rx, cfg.task_dim_ry, cfg.task_dim_rz]
    selected = [0, 1, 2] + [3 + i for i, on in enumerate(rot_flags) if on]
    j_task = jac[selected, :]
    xd_task = xd_full[selected]
    j_task_pinv = _damped_pinv(j_task, cfg.pinv_damping)
    qd_primary = j_task_pinv @ xd_task
    # Deliberately UNDAMPED pinv for the null-space projector, not
    # j_task_pinv above -- real bug found 2026-08-03: N = I - J^+ @ J
    # is a true orthogonal projector (symmetric, positive semi-
    # definite) only for the exact Moore-Penrose pinv; with the
    # damped pinv, N loses that guarantee, and the posture feedback
    # loop dq/dt = N @ kp_posture @ (q_rest - q) -- which is
    # guaranteed stable (all eigenvalues <= 0) for a true PSD
    # projector -- became genuinely UNSTABLE: orientation error grew
    # unboundedly even immediately after freshly reanchoring q_rest
    # to the current q (i.e. even starting from the loop's own
    # equilibrium point). Same split this repo's torque-control lane
    # already validated for an analogous reason (lambda_adaptive_
    # regularization only ever schedules the nullspace projector's
    # eps, wrench_lambda_adaptive_regularization is a SEPARATE
    # scheduler for the wrench-shaping eps -- coupling them was
    # already found to break one or the other).
    j_task_pinv_undamped = np.linalg.pinv(j_task)
    nullspace_proj = np.eye(6, dtype=np.float64) - j_task_pinv_undamped @ j_task

    if cfg.posture_reanchor_on_settle and not reanchored:
        # Requires reanchor_settle_cycles CONSECUTIVE settled cycles,
        # not just one instant -- real bug found 2026-08-03: at t=0
        # a min-jerk profile starts at rest (v_ff=0) with pos_err=0
        # by construction, which is trivially "settled" for exactly
        # one cycle before real motion begins, triggering an
        # immediate no-op reanchor (recapturing the SAME q_rest
        # already set at reset()) instead of ever catching the real
        # post-move settle.
        pos_settled = float(np.linalg.norm(pos_err)) < cfg.reanchor_pos_tol_m
        ff_settled = float(np.linalg.norm(v_ff)) < 1.0e-6
        if pos_settled and ff_settled:
            settled_cycles += 1
        else:
            settled_cycles = 0
        if settled_cycles >= max(int(cfg.reanchor_settle_cycles), 1):
            q_rest = q.copy()
            reanchored = True

    qd_secondary = cfg.kp_posture * (q_rest - q)
    qd = qd_primary + nullspace_proj @ qd_secondary
    xd_cmd = (jac @ qd).astype(np.float64)
    return xd_cmd, q_rest, reanchored, settled_cycles


def compute_full_hold(xd_full: np.ndarray) -> np.ndarray:
    """Trivial default: no Jacobian, no redundancy resolution -- xd_cmd is
    just the raw full-orientation-hold task velocity (P + feedforward for
    position, swing-twist P for all 3 rotation axes). Kept as a standalone
    function for symmetry/consistency with the other three modes."""
    return xd_full
