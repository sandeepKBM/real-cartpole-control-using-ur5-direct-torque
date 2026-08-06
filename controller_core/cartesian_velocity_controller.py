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

**Remaining, NOT further fixable via gain tuning, structural finding
(2026-08-03)**: with the swing-twist fix, orientation error no longer
diverges -- it always converges to SOME finite equilibrium. But WHICH
equilibrium is highly sensitive to the exact displacement, not smooth in
dx: a fine dx sweep at this pose (0.005 to 0.045m) found settled
orientation error jumping between "acceptable" (~0.18-0.32 rad at
dx=0.03-0.04m) and "large" (~0.94-2.0 rad at dx=0.02, 0.025, 0.045m) with
no smooth trend -- textbook MULTISTABILITY: minimum-norm redundancy
resolution has multiple basins of attraction, and which one a given
trajectory falls into depends on its exact path through configuration
space, not just its endpoint. Confirmed NOT a gain-tuning problem: a
kp_posture sweep from 0 to 200 left the settled value in each basin
essentially UNCHANGED (dx=0.02 stayed ~0.94-0.98 across the entire range),
and at kp_posture>=20 near the more singular displacements (dx=0.045) qd
blew up to hundreds-to-thousands of rad/s instead. No RL or manual gain
search over (kp_posture, pinv_damping, kp_rot, ...) can fix this -- the
issue is WHICH basin the deterministic dynamics fall into, which gain
magnitude does not change. A real fix would need a fundamentally different
redundancy-resolution strategy that is not path-history-dependent -- e.g.
a fresh numerical IK solve toward "nearest to q_rest" each cycle instead
of incrementally-integrated null-space rate control -- not attempted here.
DO NOT treat reduced_task_dims=True (the current default) as validated for
real-hardware use across a displacement range: only specific values (e.g.
dx=0.03-0.04m at this pose) are empirically known to land in a good basin;
others (e.g. dx=0.02m, 0.025m, 0.045m at this same pose) are known to
settle at unsafe orientation error, and there is currently no way to
predict which without simulating that exact displacement first.

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

        if self.cfg.reduced_task_dims:
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
