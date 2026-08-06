"""Resolved-rate Cartesian velocity controller for the UR5e's native RTDE
``speedL`` interface. Still no torque/gravity/mass-matrix dynamics -- but
DOES use the kinematic Jacobian (see reduced_task_dims below), a deliberate
narrowing of the original "no Jacobian at all" design.

Why this exists: UR5e has no native torque interface -- every torque-control
mechanism in this package (x_axis_cartesian_impedance.py, torque_task_qp.py,
hard_constraint_qp.py) exists to fake compliant force behavior on a robot
that is natively position/velocity-controlled, and essentially every
documented Y-drift/orientation bug this repo has fought (see AGENTS.md
section 3) is a consequence of that dynamics modeling, not of the transport
task itself. ``speedL`` resolves a commanded Cartesian velocity to joint
velocities via the Jacobian ON THE ROBOT'S OWN FIRMWARE. The real tradeoff:
this gives zero force compliance, so it is only appropriate for phases
where nothing needs to push back on the end-effector (pure point-to-point
transport / range characterization) -- not for eventual swing-up once a
physical pole is mounted and pole-arm interaction forces matter. See
hardware/velocity_transport.py's module docstring for the real-hardware
wiring.

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

pinv_damping (see _damped_pinv below) was ALSO required, not optional: a
(damping=0, kp_posture>0) sweep found qd swinging wildly and non-
monotonically (3.1 -> 11.7 -> 2.5 -> 21.4 rad/s as kp_posture stepped 0.05
-> 0.1 -> 0.2 -> 1.0) near the wrist_2 singularity -- a numerically
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
w_cmd instead of the coupled small-angle vector.

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
fraction of the full rotation block's own 2.45) that the wrist-only
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
and split_base_wrist_task (at most one of the three may be on).

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

from dataclasses import dataclass

import numpy as np
from typing import Any

from .kinematics_utils import orientation_error_vec_wxyz, swing_twist_axis_error
from .state_types import as_robot_state


def _damped_pinv(j: np.ndarray, damping: float) -> np.ndarray:
    """Tikhonov-damped least-squares pseudoinverse: J^T @ (J J^T + d^2 I)^-1.

    Plain np.linalg.pinv only truncates singular values that are negligible
    relative to the LARGEST one (its rcond default) -- near, but not at, a
    true singularity, moderately small singular values (e.g. ~0.05-0.1, the
    range this repo's wrist_2=0 approach measures) survive that truncation
    and get inverted almost as-is, amplifying any input in that direction by
    a large but finite factor. Measured directly (2026-08-03): with plain
    pinv, a jacobian_singular_cond_max analog wasn't in play here (that flag
    only exists in the torque controller), so this same amplification showed
    up as wildly non-monotonic required joint velocity vs. kp_posture (3.1 ->
    11.7 -> 2.5 -> 21.4 rad/s as kp_posture stepped 0.05 -> 0.1 -> 0.2 ->
    1.0) -- a numerically ill-conditioned regime, not a real physical
    requirement. Damping trades a small amount of task-tracking exactness
    (J @ J^+_damped != I exactly near the singularity) for bounded qd, the
    same tradeoff this repo's torque-control lane already made via
    lambda_regularization for its own Lambda inversion."""
    j = np.asarray(j, dtype=np.float64)
    d2 = float(damping) ** 2
    gram = j @ j.T + d2 * np.eye(j.shape[0], dtype=np.float64)
    return j.T @ np.linalg.inv(gram)


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
        )


class CartesianVelocityController:
    """xd_cmd = feedforward + kp * (target - actual), clamped to configured
    speed ceilings. Holds Y/Z at their reset-time values unless the caller
    supplies a moving target_ee_pos/target_ee_vel for them. Orientation
    holding depends on reduced_task_dims -- see module docstring.

    posture_reanchor_on_settle (default on): fixes a real bug found
    2026-08-03 -- pulling the null-space posture term toward the pose
    captured at reset() is fine WHILE the arm is still at that pose, but
    once a move completes and the arm holds a NEW X position, q_rest (the
    OLD pose) is no longer consistent with the current task -- the posture
    term keeps trying to pull q back toward a configuration the task no
    longer allows, and since the null-space direction relative to a fixed
    target isn't guaranteed to monotonically shrink as the null-space
    projector itself evolves with q, this manifested as UNBOUNDED
    orientation drift during the hold phase (measured: growing from 0 to
    beyond the 0.25 rad guard over ~1s of an otherwise-static hold, for a
    move as small as dx=0.02m). Fix, mirroring
    XAxisCartesianImpedanceController's posture_reanchor_on_settle: once
    the primary task's position error drops under reanchor_pos_tol_m,
    re-capture q_rest at the CURRENT q -- the posture term then has nothing
    left to correct at the settled pose instead of chasing a stale target
    forever."""

    def __init__(self, config: CartesianVelocityConfig) -> None:
        self.cfg = config
        self._initialized = False
        self._p0 = np.zeros(3, dtype=np.float64)
        self._quat0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._q_rest = np.zeros(6, dtype=np.float64)
        self._reanchored = False
        self._settled_cycles = 0

    def reset_from_state(self, state: dict[str, Any]) -> None:
        st = as_robot_state(state)
        self._p0 = np.asarray(st["ee_pos"], dtype=np.float64).reshape(3).copy()
        self._quat0 = np.asarray(st["ee_quat"], dtype=np.float64).reshape(4).copy()
        self._q_rest = np.asarray(st["q"], dtype=np.float64).reshape(6).copy()
        self._reanchored = False
        self._settled_cycles = 0
        self._initialized = True

    @property
    def initialized(self) -> bool:
        return self._initialized

    def compute(self, state: dict[str, Any]) -> np.ndarray:
        """Returns a 6D Cartesian velocity command [vx,vy,vz,wx,wy,wz] in the
        axis-angle/rotvec convention RTDE's speedL/getActualTCPPose use --
        NOT a quaternion, deliberately unlike every torque-path controller
        in this package (hence returning a plain ndarray, not a
        CartesianImpedanceOutput)."""
        if not self._initialized:
            raise RuntimeError("Call reset_from_state() before compute().")
        st = as_robot_state(state)
        p = np.asarray(st["ee_pos"], dtype=np.float64).reshape(3)
        quat = np.asarray(st["ee_quat"], dtype=np.float64).reshape(4)
        p_des = np.asarray(st.get("target_ee_pos", self._p0), dtype=np.float64).reshape(3)
        v_ff = np.asarray(st.get("target_ee_vel", np.zeros(3)), dtype=np.float64).reshape(3)

        pos_err = p_des - p
        kp_lin = np.array([self.cfg.kp_x, self.cfg.kp_y, self.cfg.kp_z], dtype=np.float64)
        v_cmd = v_ff + kp_lin * pos_err

        # Swing-twist per-axis decomposition, NOT orientation_error_vec_wxyz's
        # 2*vec(q_err) -- see swing_twist_axis_error's docstring. That
        # small-angle vector approximation mixes all 3 axes together for a
        # compound rotation and is not exact for large angles; naively using
        # its row 2 as "the rz error" was found (2026-08-03) to create a
        # genuine, reproducible, unbounded-growth instability once the other
        # two axes accumulated real rotation from redundancy-resolution
        # null-space motion -- confirmed independent of kp_posture (identical
        # divergence with kp_posture=0.0, i.e. no null-space term at all),
        # so this is a bug in the task's own orientation-error signal, not a
        # gain-tuning problem.
        w_cmd = self.cfg.kp_rot * np.array(
            [
                swing_twist_axis_error(self._quat0, quat, 0),
                swing_twist_axis_error(self._quat0, quat, 1),
                swing_twist_axis_error(self._quat0, quat, 2),
            ],
            dtype=np.float64,
        )

        xd_full = np.concatenate([v_cmd, w_cmd]).astype(np.float64)

        modes_on = sum([self.cfg.reduced_task_dims, self.cfg.split_base_wrist_task, self.cfg.ik_seeded_resolution])
        if modes_on > 1:
            raise ValueError(
                "reduced_task_dims, split_base_wrist_task, and ik_seeded_resolution "
                "are mutually exclusive -- enable at most one."
            )

        if self.cfg.ik_seeded_resolution:
            fk_jacobian_fn = state.get("fk_jacobian_fn")
            if fk_jacobian_fn is None:
                raise ValueError(
                    "CartesianVelocityConfig.ik_seeded_resolution=True requires "
                    "state['fk_jacobian_fn'], a callable q -> (ee_pos, ee_quat, jacobian) "
                    "usable at ARBITRARY q, not just the current one -- controller_core "
                    "stays simulator-independent, so this must be supplied by the caller "
                    "(e.g. a MuJoCo-backed forward-kinematics wrapper), not built in here."
                )
            q_current = np.asarray(st["q"], dtype=np.float64).reshape(6)
            rot_flags = [self.cfg.task_dim_rx, self.cfg.task_dim_ry, self.cfg.task_dim_rz]
            selected = [0, 1, 2] + [3 + i for i, on in enumerate(rot_flags) if on]

            # Fresh Newton-Raphson IK solve SEEDED FROM q_rest every cycle --
            # NOT from q_current. This is the actual fix for reduced_task_dims'
            # path-dependent multistability (see this class's module
            # docstring): q_target is a deterministic function of ONLY
            # (p_des, q_rest) -- the same target position always produces
            # the same q_target, regardless of how the arm got to its
            # current configuration, because the solve never looks at
            # q_current at all. This trades per-cycle compute cost (up to
            # ik_iterations extra forward-kinematics/Jacobian evaluations)
            # for eliminating the redundancy-resolution history-dependence
            # that made reduced_task_dims' and split_base_wrist_task's rate-
            # integrated null-space walks unpredictable.
            q_k = self._q_rest.copy()
            for _ in range(max(int(self.cfg.ik_iterations), 1)):
                p_k, quat_k, jac_k = fk_jacobian_fn(q_k)
                p_k = np.asarray(p_k, dtype=np.float64).reshape(3)
                quat_k = np.asarray(quat_k, dtype=np.float64).reshape(4)
                jac_k = np.asarray(jac_k, dtype=np.float64).reshape(6, 6)
                pos_err_k = p_des - p_k
                rot_err_k = np.array(
                    [swing_twist_axis_error(self._quat0, quat_k, i) for i in range(3)],
                    dtype=np.float64,
                )
                task_err_full_k = np.concatenate([pos_err_k, -rot_err_k])
                j_task_k = jac_k[selected, :]
                dq = _damped_pinv(j_task_k, self.cfg.pinv_damping) @ task_err_full_k[selected]
                q_k = q_k + dq
            q_target = q_k

            qd_joint = self.cfg.ik_joint_gain * (q_target - q_current)
            _, _, jac_current = fk_jacobian_fn(q_current)
            jac_current = np.asarray(jac_current, dtype=np.float64).reshape(6, 6)
            xd_cmd = (jac_current @ qd_joint).astype(np.float64)
        elif self.cfg.split_base_wrist_task:
            jacobian = st.get("jacobian")
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
            qd_pos = _damped_pinv(j_pos_task, self.cfg.pinv_damping) @ v_cmd

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
            rot_flags = [self.cfg.task_dim_rx, self.cfg.task_dim_ry, self.cfg.task_dim_rz]
            rot_selected = [3 + i for i, on in enumerate(rot_flags) if on]
            qd_rot = np.zeros(6, dtype=np.float64)
            if rot_selected:
                w_induced_by_pos = jac[3:6, :] @ qd_pos
                j_rot_task = np.zeros((len(rot_selected), 6), dtype=np.float64)
                j_rot_task[:, 3:6] = jac[np.ix_(rot_selected, [3, 4, 5])]
                xd_rot_task = xd_full[rot_selected] - w_induced_by_pos[[i - 3 for i in rot_selected]]
                qd_rot = _damped_pinv(j_rot_task, self.cfg.pinv_damping) @ xd_rot_task

            qd = qd_pos + qd_rot
            xd_cmd = (jac @ qd).astype(np.float64)
        elif self.cfg.reduced_task_dims:
            jacobian = st.get("jacobian")
            if jacobian is None:
                raise ValueError(
                    "CartesianVelocityConfig.reduced_task_dims=True requires "
                    "state['jacobian'] (6x6) every cycle -- set reduced_task_dims=False "
                    "for the original jacobian-free full-orientation-hold behavior."
                )
            jac = np.asarray(jacobian, dtype=np.float64).reshape(6, 6)
            q = np.asarray(st["q"], dtype=np.float64).reshape(6)
            rot_flags = [self.cfg.task_dim_rx, self.cfg.task_dim_ry, self.cfg.task_dim_rz]
            selected = [0, 1, 2] + [3 + i for i, on in enumerate(rot_flags) if on]
            j_task = jac[selected, :]
            xd_task = xd_full[selected]
            j_task_pinv = _damped_pinv(j_task, self.cfg.pinv_damping)
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

            if self.cfg.posture_reanchor_on_settle and not self._reanchored:
                # Requires reanchor_settle_cycles CONSECUTIVE settled cycles,
                # not just one instant -- real bug found 2026-08-03: at t=0
                # a min-jerk profile starts at rest (v_ff=0) with pos_err=0
                # by construction, which is trivially "settled" for exactly
                # one cycle before real motion begins, triggering an
                # immediate no-op reanchor (recapturing the SAME q_rest
                # already set at reset()) instead of ever catching the real
                # post-move settle.
                pos_settled = float(np.linalg.norm(pos_err)) < self.cfg.reanchor_pos_tol_m
                ff_settled = float(np.linalg.norm(v_ff)) < 1.0e-6
                if pos_settled and ff_settled:
                    self._settled_cycles += 1
                else:
                    self._settled_cycles = 0
                if self._settled_cycles >= max(int(self.cfg.reanchor_settle_cycles), 1):
                    self._q_rest = q.copy()
                    self._reanchored = True

            qd_secondary = self.cfg.kp_posture * (self._q_rest - q)
            qd = qd_primary + nullspace_proj @ qd_secondary
            xd_cmd = (jac @ qd).astype(np.float64)
        else:
            xd_cmd = xd_full

        v_cmd_out = xd_cmd[:3]
        w_cmd_out = xd_cmd[3:]
        v_norm = float(np.linalg.norm(v_cmd_out))
        if v_norm > self.cfg.max_lin_speed_mps and v_norm > 1.0e-9:
            v_cmd_out = v_cmd_out * (self.cfg.max_lin_speed_mps / v_norm)
        w_norm = float(np.linalg.norm(w_cmd_out))
        if w_norm > self.cfg.max_ang_speed_radps and w_norm > 1.0e-9:
            w_cmd_out = w_cmd_out * (self.cfg.max_ang_speed_radps / w_norm)

        return np.concatenate([v_cmd_out, w_cmd_out]).astype(np.float64)
