#!/usr/bin/env python3
"""4-state cartpole LQR (HIGH-LEVEL controller) cascaded onto the torque-lane
OSC (LOW-LEVEL controller), for balancing the pendulum apparatus about its
inverted equilibrium.

ARCHITECTURE (deliberate, do not deviate):

    pole state --> [LQR] --> desired CART ACCELERATION --> [OSC] --> joint torques --> robot

The LQR NEVER touches data.ctrl or joint torques directly -- unlike
pendulum_balance_torque_lqr.py (which writes data.ctrl[:6] from a 14-state
joint-space linearization; kept only as a Riccati-call reference, not a
control-structure template, per this file's own task spec). This LQR is a
textbook 4-state cartpole: state s = [x - x_ref, xdot, phi, phidot] (x = EE
position along world X; phi = pendulum angle measured from the INVERTED
equilibrium), single scalar input u = desired cart acceleration. u is
integrated into target_x/target_x_vel/target_x_accel -- the SAME command
interface pendulum_swingup_multi_kick.py / pendulum_swingup_energy_shaping.py
already drive the OSC through (build_mujoco_state -> adapter.step). Switching
from a kick trajectory to LQR control is a SOURCE SWITCH on those three
fields, not new plumbing. All safety guards (ImpedanceSafetyMonitor, incl.
the |qd| > 3.0 rad/s joint-velocity guard that is this pose's binding cart-
speed ceiling) stay live in the single OSC path throughout.

LINEARIZATION, about phi=0 (inverted). The pendulum's own gravity torque
about the hinge is exactly sinusoidal in angle with amplitude mgr_nm (fitted
by derive_pendulum_constants, not hand-derived -- see that function's
docstring). Measured from the UNSTABLE equilibrium, small-angle gravity
torque is +mgr_nm*phi (destabilizing, the standard inverted-pendulum sign).

The generalized hinge torque produced by a horizontal (world-X) pivot
acceleration a_x is Q = n . (r x (-m*a_x*xhat)) (n = hinge axis, r = COM
minus pivot, both world-frame, m = swinging subtree mass -- the standard
non-inertial-frame pseudo-force derivation). This coupling term is asset-
AND-pose-specific: an EARLIER version of this file assumed the shortcut
Q(0) = -(mgr_nm/g)*a_x (i.e. an implicit "kappa=-1"), ported unmodified from
pendulum_swingup_energy_shaping.py's DIFFERENT (asset, pose) pair. Checked
directly against THIS (realrod asset, OLD_POSE) pair two independent ways --
(1) a closed-loop finite-difference probe of the real OSC path (apply a
constant target_x_accel=+/-0.05 m/s^2 from phi=0, thetadot=0, read the
resulting sign of thetadot after 0.2s) and (2) the cross-product formula
above evaluated on the compiled model -- and found BACKWARDS in sign as well
as wrong in magnitude (true kappa = Q_per_a/(mgr_nm/g) = +2.496 here, not
+/-1). See measure_cart_coupling_nm_per_mps2's docstring for the full
numbers. B's cart-coupling term is now measured fresh off the model every
time, never assumed. Hinge friction (dof_damping) contributes
-damping*phidot. So:

    I_pivot * phiddot = mgr_nm*phi + Q_per_a*u - damping*phidot
    xddot = u   (the OSC is assumed high-bandwidth enough, over the LQR's own
                 timescale, to track a commanded acceleration -- the same
                 assumption pendulum_swingup_energy_shaping.py's a_cmd
                 integration already relies on; see run_lqr_trial's own
                 empirical bandwidth caveat if this assumption matters for a
                 particular result)

State s = [x - x_ref, xdot, phi, phidot], input u = xddot command:

    A = [[0, 1,        0,        0],
         [0, 0,        0,        0],
         [0, 0,        0,        1],
         [0, 0, omega^2, -damping/I_pivot]]
    B = [[0], [1], [0], [Q_per_a/I_pivot]]

omega^2 = mgr_nm / I_pivot = derive_pendulum_constants(...).omega_natural_radps
squared exactly -- the SAME natural frequency that governs small oscillation
about the *hanging* equilibrium also governs the divergence rate at the
*inverted* one (standard pendulum-linearization result; the sign of the
gravity term is the only thing that flips between the two equilibria).

K = -solve_continuous_are(A, B, Q, R)-based full-state feedback gain.
u = clip(-K @ s, -a_max, a_max); the reference velocity target_x_vel is
additionally hard-clipped to +-v_max (measured cart-speed authority at this
pose, ~1.06 m/s with orientation held -- see AGENTS.md) before being
integrated into target_x. This is the "reject gain sets the OSC cannot
deliver" mechanism: any candidate that needs to exceed the clip to work will
show it directly as a failed capture or a guard trip in closed-loop
simulation, not as a hidden linear-model optimism.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import os
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.linalg import solve_continuous_are
from scipy.optimize import differential_evolution

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from simulation.ur5e_pendulum_compose import (  # noqa: E402
    PendulumConstants,
    REALROD_PENDULUM_XML,
)
from simulation.ur5e_mujoco_torque import (  # noqa: E402
    build_initial_state_and_adapter,
    build_mujoco_state,
)
from tools.diagnostics.pendulum_swingup_energy_shaping import (  # noqa: E402
    CONTROL_DT, RATE_HZ,
    PendulumRunContext, add_common_pendulum_args, context_from_args,
    describe_context, load_config, write_output_json,
)

# Measured cart-speed authority at OLD_POSE (see AGENTS.md's OSC/pendulum
# sections): ~1.06 m/s with orientation held before the |qd|>3.0 rad/s joint
# guard trips. A fixed 5% margin, not searched -- the point is to reject any
# LQR gain set that NEEDS to exceed real authority, not to let the search
# quietly raise the ceiling to make a candidate look feasible.
V_MAX_MPS = 1.00

DEFAULT_CONFIG = REPO_ROOT / "config" / "ur5e_mujoco_torque_osc_tuned.yaml"


def _de_workers() -> int:
    import multiprocessing as mp
    try:
        mp.set_start_method("fork")
    except RuntimeError:
        pass
    env = os.environ.get("DE_WORKERS")
    if env:
        return max(1, int(env))
    return max(1, int((os.cpu_count() or 2) * 0.9))


def measure_cart_coupling_nm_per_mps2(model: mujoco.MjModel, arm_q: np.ndarray, inverted_angle: float) -> float:
    """The EXACT hinge generalized torque produced by one unit (1 m/s^2) of
    world-X pivot acceleration, at phi=0 (inverted), via the correct rigid-
    body physics: in the pivot's non-inertial frame a cart acceleration a_x
    applies a pseudo-force F=-m*a_x*xhat at the swinging subtree's COM, and
    the resulting generalized torque about the hinge axis n is
    Q = n . (r x F), r = COM - pivot.

    THIS FUNCTION EXISTS BECAUSE A HAND-DERIVED SHORTCUT WAS WRONG HERE.
    pendulum_swingup_energy_shaping.py derives (and directly measures) this
    same relation for the DEFAULT (pendulum_attachment.xml, local-Z hinge)
    asset/pose as Q(0) = -(mgr_nm/g)*a_x -- i.e. it assumes the coupling
    coefficient equals -mgr_nm/g exactly (implicitly "kappa=-1" in the
    module docstring's notation). Porting that formula's SIGN unmodified to
    THIS asset (realrod, local-X hinge) at OLD_POSE was checked two
    independent ways and found backwards: (1) a closed-loop finite-
    difference probe (apply a constant target_x_accel=+/-0.05 m/s^2 from
    phi=0, thetadot=0 for 0.2s through the real OSC path, read the resulting
    sign of thetadot) measured phiddot growing POSITIVE for positive u; (2)
    this exact cross-product formula, evaluated on the compiled model at
    OLD_POSE, gives Q_per_a = +0.007090 Nm per (m/s^2) -- positive, matching
    (1), and NOT within a simple sign flip of -mgr_nm/g (which would be
    -0.002841 Nm per unit at the other asset/pose; here mgr_nm/g=0.002841 in
    THIS asset too since mgr_nm is nearly identical, but the true coupling
    magnitude is 0.007090, i.e. the implied "kappa" = Q_per_a/(mgr_nm/g) is
    +2.496, not +/-1). Both facts -- the flipped sign and the non-unity
    magnitude -- are real, asset/pose-specific properties of HOW the COM
    offset vector decomposes against this particular hinge axis, and a
    formula written for one (asset, pose) pair must never be assumed to
    carry over unchanged to another; this function computes it fresh from
    the model every time instead."""
    data = mujoco.MjData(model)
    data.qpos[:6] = np.asarray(arm_q, dtype=np.float64).reshape(6)
    pend_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "/pendulum_hinge")
    pend_qpos_adr = model.jnt_qposadr[pend_jid]
    pend_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "/pendulum_hub")
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    data.qpos[pend_qpos_adr] = inverted_angle
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    hinge_pos = np.asarray(data.site_xpos[site_id], dtype=np.float64).copy()
    site_xmat = np.asarray(data.site_xmat[site_id], dtype=np.float64).reshape(3, 3)
    n_axis = site_xmat @ np.array([1.0, 0.0, 0.0])
    n_axis = n_axis / np.linalg.norm(n_axis)
    com = np.asarray(data.subtree_com[pend_body_id], dtype=np.float64).copy()
    m = float(model.body_subtreemass[pend_body_id])
    r = com - hinge_pos
    xhat = np.array([1.0, 0.0, 0.0])
    F = -m * xhat  # pseudo-force for a=1 m/s^2
    Q_per_a = float(np.dot(n_axis, np.cross(r, F)))
    return Q_per_a


def _cartpole_AB(constants: PendulumConstants, damping: float, b32: float) -> tuple[np.ndarray, np.ndarray]:
    omega2 = constants.mgr_nm / constants.i_pivot_kgm2
    A = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, omega2, -damping / constants.i_pivot_kgm2],
    ], dtype=np.float64)
    B = np.array([[0.0], [1.0], [0.0], [b32]], dtype=np.float64)
    return A, B


def linearize_cartpole(
    constants: PendulumConstants, damping: float,
    model: mujoco.MjModel, arm_q: np.ndarray, inverted_angle: float,
) -> tuple[np.ndarray, np.ndarray]:
    """4-state (x, xdot, phi, phidot) linear model about the inverted
    equilibrium, input u = cart acceleration. See module docstring for the
    derivation. ``damping`` is the hinge DOF's ``model.dof_damping`` (read by
    the caller off the compiled model -- this function does not assume its
    value, unlike a hand-cached constant).

    B's cart-coupling term (row 3) is measured EXACTLY off the model via
    measure_cart_coupling_nm_per_mps2, not assumed as ``-omega^2/g``
    (implicitly kappa=1) -- that assumed formula was checked against this
    exact (asset, pose) pair and found backwards in SIGN as well as wrong in
    magnitude (empirically-measured kappa = +2.496 here, not +/-1; see
    measure_cart_coupling_nm_per_mps2's docstring for the two independent
    checks -- a closed-loop finite-difference probe and this cross-product
    derivation -- that caught it)."""
    Q_per_a = measure_cart_coupling_nm_per_mps2(model, arm_q, inverted_angle)
    b32 = Q_per_a / constants.i_pivot_kgm2
    return _cartpole_AB(constants, damping, b32)


def lqr_gain(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)  # 1x4
    return K, P


def hinge_damping(model: mujoco.MjModel) -> float:
    pend_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "/pendulum_hinge")
    return float(model.dof_damping[model.jnt_dofadr[pend_jid]])


def wrap_pi(angle: float) -> float:
    return float(np.mod(angle + np.pi, 2 * np.pi) - np.pi)


def run_lqr_trial(
    model,
    K: np.ndarray,
    *,
    duration_s: float,
    hanging_angle: float,
    inverted_angle: float,
    constants: PendulumConstants,
    initial_phi_rad: float,
    initial_thetadot_radps: float,
    a_max: float = 3.0,
    v_max: float = V_MAX_MPS,
    enforce_guard: bool = True,
    track_history: bool = False,
    config_path: Path | None = None,
    controller_kind: str = "impedance",
    arm_q=None,
    capture_tol_rad: float = 0.12,
    capture_tol_radps: float = 1.0,
    capture_window_s: float = 1.5,
) -> dict:
    """Runs the LQR (K) -> OSC cascade for ``duration_s`` starting the
    pendulum at ``inverted_angle + initial_phi_rad`` with angular rate
    ``initial_thetadot_radps``. Pass ``K = np.zeros((1, 4))`` for the
    zero-control counterfactual (target_x/target_x_vel/target_x_accel stay
    pinned at x_ref/0/0 -- the arm just holds its start pose while the
    pendulum evolves passively under gravity + friction only).

    BANDWIDTH CAVEAT, measured directly (not assumed): the module docstring's
    "xddot=u" assumption -- that the OSC realizes a commanded cart
    acceleration effectively instantly relative to the LQR's own timescale --
    is only APPROXIMATELY true here. A step-response probe of
    config/ur5e_mujoco_torque_osc_tuned.yaml's actual closed-loop X-tracking
    (holding the pendulum fixed, commanding a constant target_x_accel, and
    measuring the REAL resulting EE acceleration) found the achieved
    acceleration only reaches roughly its commanded value after ~150-200ms --
    comparable to, not clearly faster than, the pendulum's own
    1/omega=~92ms instability time constant. This is why naive/large
    hand-picked K values (large Q_phi with a small a_max) were observed to
    diverge explosively even from a tiny few-degree perturbation despite a
    correctly-signed, correctly-derived linear model: the actuator's own lag
    eats the margin a pure kinematic analysis would predict. Practical
    consequence, found by direct closed-loop search rather than linear
    theory alone: only fairly LARGE Q_phi/Q_phidot (pushing K's phi-gain into
    the tens to hundreds) combined with a generous a_max (several m/s^2)
    reliably captures even a few-degree perturbation -- this is a real,
    reported property of the plant as given (config is fixed per this task's
    spec), not a search-tuning shortcoming."""
    config = load_config(config_path) if config_path is not None else load_config()
    arm_q = np.asarray(arm_q, dtype=np.float64).reshape(6)
    data = mujoco.MjData(model)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in [
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
    ]]
    pend_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "/pendulum_hinge")
    pend_qpos_adr = model.jnt_qposadr[pend_jid]
    pend_dof_adr = model.jnt_dofadr[pend_jid]

    data.qpos[:6] = arm_q
    data.qpos[pend_qpos_adr] = wrap_pi(inverted_angle + initial_phi_rad)
    data.qvel[:] = 0.0
    data.qvel[pend_dof_adr] = initial_thetadot_radps
    mujoco.mj_forward(model, data)

    state0, adapter = build_initial_state_and_adapter(
        model, data, site_id, joint_ids,
        controller_cfg=config["controller"],
        transport_axis_index=0,
        target_x_delta=0.0,
        controller_kind=str(controller_kind),
        force_hold_current_pose=False,
        gravity_mode=config["mujoco"].get("gravity_mode", "gravity_comp"),
        gravity_source=config["mujoco"].get("gravity_source", "pinocchio"),
        coriolis_feedforward=bool(config["mujoco"].get("coriolis_feedforward", True)),
        torque_limit_scale=1.0,
    )
    x_ref = float(state0.ee_pos[0])

    n_steps = int(duration_s * RATE_HZ)
    target_x = x_ref
    target_x_vel = 0.0
    guard_fired = False
    guard_reason = None
    first_guard_t = None
    steps_done = 0
    peak_abs_u = 0.0
    peak_abs_target_x_vel = 0.0
    peak_abs_qd = 0.0
    min_capture_window_start = max(0, n_steps - int(capture_window_s * RATE_HZ))
    within_tol_from = None
    history = [] if track_history else None

    K = np.asarray(K, dtype=np.float64).reshape(1, 4)

    for step in range(n_steps):
        t = step * CONTROL_DT
        theta = float(data.qpos[pend_qpos_adr])
        thetadot = float(data.qvel[pend_dof_adr])
        phi = wrap_pi(theta - inverted_angle)

        # Read the ACTUAL cart state (not the reference trajectory) for
        # genuine closed-loop feedback -- build once with the previous target
        # just to get ee_pos/ee_lin_vel/jacobian/gravity/mass consistently
        # (same code path every other pendulum script uses), then patch the
        # target_x* fields below before handing to the OSC.
        state = build_mujoco_state(
            model, data, site_id=site_id, joint_ids=joint_ids,
            time_s=t, dt_s=CONTROL_DT,
            target_x=target_x, target_x_vel=target_x_vel, target_x_accel=0.0,
            reference_quat=state0.ee_quat, transport_axis_index=0,
            gravity_compensation=True,
        )
        x_actual = float(state.ee_pos[0])
        xdot_actual = float(state.ee_lin_vel[0])

        s = np.array([x_actual - x_ref, xdot_actual, phi, thetadot], dtype=np.float64)
        u = float(np.clip(-(K @ s)[0], -a_max, a_max))
        peak_abs_u = max(peak_abs_u, abs(u))

        target_x_vel = float(np.clip(target_x_vel + u * CONTROL_DT, -v_max, v_max))
        target_x = target_x + target_x_vel * CONTROL_DT
        peak_abs_target_x_vel = max(peak_abs_target_x_vel, abs(target_x_vel))

        state.target_x = target_x
        state.target_x_vel = target_x_vel
        state.target_x_accel = u

        tau, diag = adapter.step(state=state)
        step_safety_ok = bool(diag.get("safety_ok", True))
        if not step_safety_ok and first_guard_t is None:
            first_guard_t = t
            guard_reason = str(diag.get("safety_reason", ""))
            guard_fired = True
            if enforce_guard:
                break

        data.ctrl[:6] = np.asarray(tau, dtype=np.float64).reshape(6)
        mujoco.mj_step(model, data)
        steps_done += 1
        peak_abs_qd = max(peak_abs_qd, float(np.max(np.abs(data.qvel[:6]))))

        in_tol = (abs(phi) < capture_tol_rad) and (abs(thetadot) < capture_tol_radps)
        if step >= min_capture_window_start and not in_tol:
            within_tol_from = False  # sentinel: at least one violation seen in window
        elif within_tol_from is None and step >= min_capture_window_start:
            within_tol_from = True

        if track_history:
            history.append({
                "t": t, "phi_deg": float(np.degrees(phi)), "thetadot": thetadot,
                "x": x_actual, "xdot": xdot_actual, "u": u,
                "target_x": target_x, "target_x_vel": target_x_vel,
                "safety_ok": step_safety_ok,
                "qd_max_abs": float(np.max(np.abs(data.qvel[:6]))),
            })

    captured = bool(
        steps_done == n_steps
        and not guard_fired
        and within_tol_from is True
    )
    return {
        "initial_phi_rad": initial_phi_rad, "initial_thetadot_radps": initial_thetadot_radps,
        "duration_s": duration_s, "a_max": a_max, "v_max": v_max,
        "captured": captured,
        "guard_fired": guard_fired, "guard_reason": guard_reason, "first_guard_t": first_guard_t,
        "steps_completed": steps_done, "n_steps": n_steps,
        "peak_abs_cart_accel_cmd_mps2": peak_abs_u,
        "peak_abs_cart_vel_cmd_mps": peak_abs_target_x_vel,
        "peak_abs_joint_vel_radps": peak_abs_qd,
        "final_phi_deg": float(np.degrees(wrap_pi(float(data.qpos[pend_qpos_adr]) - inverted_angle))),
        "history": history,
    }


def default_context() -> PendulumRunContext:
    from simulation.ur5e_pendulum_compose import arm_q_for_pendulum_xml
    pendulum_xml = str(REALROD_PENDULUM_XML)
    return PendulumRunContext(
        pendulum_xml=pendulum_xml,
        arm_q=tuple(float(v) for v in arm_q_for_pendulum_xml(pendulum_xml)),
        config_path=str(DEFAULT_CONFIG),
        controller_kind="impedance",
    )


# ---------------------------------------------------------------------------
# Representative post-swing-up arrival states used to tune Q/R. A swing-up
# arrives near (not exactly at) the inverted angle WITH velocity -- see
# module docstring / task spec: "measure the capture envelope over BOTH
# theta error and thetadot". Symmetric in sign (a real, recurring asymmetry
# elsewhere in this repo -- AGENTS.md Sec.7 -- so both signs are swept, not
# just magnitude).
# ---------------------------------------------------------------------------
# EXTENDED 2026-08-19 TO COVER THE REAL ARRIVAL. The set below used to stop at
# |phi| = 12 deg and |thetadot| = 3.0 rad/s. The measured swing-up handoff
# arrives at phi = -18.0 deg with thetadot = +3.42 rad/s -- OUTSIDE the set on
# BOTH axes -- so the search was fitting K on a set that does not contain its
# own operating point. Consequence, measured: a retune scored BETTER on this
# objective (cost 0.637 vs 0.742) and then DROPPED THE POLE at the real handoff
# (|phi| -> 179.98 deg, lost between +0.5 s and +1 s), because it chose softer
# gains and a_max 6.73 vs 9.07 that the perturbation set never punished.
#
# The added pairs follow the existing anti-diagonal pattern, which is not
# arbitrary: capture is governed by the unstable mode s = thetadot + omega*phi
# (omega = 10.8334), so a catchable state has phi and thetadot of OPPOSITE
# sign. At (-18 deg, +3.42) that gives |s| ~ 0.02. Both signs are swept because
# X-direction asymmetry is real and recurring here (AGENTS.md sec.7).
#
# STILL NOT THE WHOLE STORY, stated so it is not mistaken for one: every state
# here starts with the ARM AT REST, while the real arrival has the arm
# mid-stroke carrying cart velocity and tracking a reference that has been
# running for seconds. That seam is exactly what broke the LQR half before
# (AGENTS.md sec.0). This extension closes the pendulum-state gap, not the
# arm-state gap.
TUNING_STATES = [
    (np.radians(d), w)
    for d, w in [
        (3.0, 0.0), (-3.0, 0.0), (5.0, -1.0), (-5.0, 1.0),
        (8.0, -1.5), (-8.0, 1.5), (12.0, -2.0), (-12.0, 2.0),
        (0.0, -3.0), (0.0, 3.0),
        # --- the real arrival, and a bracket beyond it ---
        (18.0, -3.4), (-18.0, 3.4),
        (22.0, -4.0), (-22.0, 4.0),
    ]
]


def _q_matrix(params) -> np.ndarray:
    log_qx, log_qxdot, log_qphi, log_qphidot = params[:4]
    return np.diag([10.0 ** log_qx, 10.0 ** log_qxdot, 10.0 ** log_qphi, 10.0 ** log_qphidot])


def tuning_objective(params, ctx: PendulumRunContext, duration_s: float = 4.0) -> float:
    a_max = float(np.clip(params[4], 0.3, 10.0))
    Q = _q_matrix(params)
    R = np.array([[1.0]])
    model = ctx.build_model()
    damping = hinge_damping(model)
    A, B = linearize_cartpole(ctx.constants, damping, model, ctx.arm_q_array, ctx.inverted_angle)
    try:
        K, _ = lqr_gain(A, B, Q, R)
    except Exception:
        return 1e3
    if not np.all(np.isfinite(K)):
        return 1e3

    total = 0.0
    for phi0, thetadot0 in TUNING_STATES:
        res = run_lqr_trial(
            model, K, duration_s=duration_s, hanging_angle=ctx.hanging_angle,
            inverted_angle=ctx.inverted_angle, constants=ctx.constants,
            initial_phi_rad=phi0, initial_thetadot_radps=thetadot0,
            a_max=a_max, v_max=V_MAX_MPS,
            config_path=Path(ctx.config_path), controller_kind=ctx.controller_kind,
            arm_q=ctx.arm_q_array,
        )
        cost = abs(np.radians(res["final_phi_deg"]))
        if res["guard_fired"]:
            cost += 5.0
        if not res["captured"]:
            cost += 2.0
        # Discourage gains that ride right at the authority ceiling -- a
        # candidate that only "works" by using ~all of v_max is not a
        # robust margin.
        if res["peak_abs_cart_vel_cmd_mps"] > 0.9 * V_MAX_MPS:
            cost += 1.0
        total += cost
    return total / len(TUNING_STATES)


def search_lqr_gains(ctx: PendulumRunContext, *, maxiter: int, popsize: int, seed: int,
                      duration_s: float, a_max_upper: float = 10.0) -> dict:
    """``a_max_upper`` bounds the cart acceleration the LQR is allowed to ask for.

    IT IS NOT A FREE PARAMETER -- it should come from the pose's MEASURED command
    envelope. The historical default 10.0 has no relation to any measurement, and
    the consequence is concrete: at ARM_Q0/wrist_2=-90 the search returned
    a_max = 9.7238 (97% of this ceiling) and the resulting gain STABILISED THE
    PENDULUM -- four of seven envelope cells ended within 1.3 deg of vertical --
    while tripping |Y-Y0| > 0.03 m in every single one. The pole was held; the arm
    could not execute the motion. The pose-hold ladder
    (tools/diagnostics/pose_hold_orientation_check.py) measured this pose holding
    a sustained command to ~1.0 m/s^2 and tripping by 1.5, so 10.0 authorised
    roughly 5-10x what the arm delivers.

    This is the same failure AGENTS.md already records for the SWING-UP search's
    a_max bound, pointed the other way: there the bound was too tight and hid a
    feasible flip; here it is too loose and produces an infeasible catch. Either
    way an arbitrary bound, not the physics, decided the answer.
    """
    bounds = [
        (-1.0, 4.0),   # log10(qx)
        (-1.0, 4.0),   # log10(qxdot)
        (-1.0, 6.0),   # log10(qphi)
        (-1.0, 5.0),   # log10(qphidot)
        (0.5, float(a_max_upper)),   # a_max -- from the MEASURED envelope
    ]
    res = differential_evolution(
        functools.partial(tuning_objective, ctx=ctx, duration_s=duration_s),
        bounds, maxiter=maxiter, popsize=popsize, tol=1e-4, seed=seed,
        workers=_de_workers(), polish=False,
    )
    Q = _q_matrix(res.x)
    a_max = float(np.clip(res.x[4], 0.3, float(a_max_upper)))
    model = ctx.build_model()
    damping = hinge_damping(model)
    A, B = linearize_cartpole(ctx.constants, damping, model, ctx.arm_q_array, ctx.inverted_angle)
    K, P = lqr_gain(A, B, Q, np.array([[1.0]]))
    return {
        "Q_diag": np.diag(Q).tolist(), "R": 1.0, "a_max": a_max,
        "K": K.reshape(-1).tolist(), "cost": float(res.fun),
        "A": A.tolist(), "B": B.tolist(), "damping": damping,
        "log10_params": res.x.tolist(),
    }


def capture_envelope_grid(
    ctx: PendulumRunContext,
    K: np.ndarray,
    *,
    phi_deg_values,
    thetadot_values,
    a_max: float,
    duration_s: float = 5.0,
    verify_k_zero: bool = True,
) -> list[dict]:
    """Sweeps (phi0, thetadot0) and reports, for each cell, whether K
    captures it -- AND, for every captured cell, whether the K=0
    counterfactual over the SAME initial condition fails (the
    "active control, not passive friction" proof required by the task spec).
    Real friction is tiny here (zeta=0.0194, see the realrod asset's own
    header), so a captured cell whose K=0 run ALSO captures would mean the
    "capture" was actually free -- passive damping, not the LQR -- and must
    be flagged, not reported as a real result."""
    model = ctx.build_model()
    rows = []
    for phi_deg in phi_deg_values:
        for thetadot in thetadot_values:
            phi0 = np.radians(phi_deg)
            res = run_lqr_trial(
                model, K, duration_s=duration_s, hanging_angle=ctx.hanging_angle,
                inverted_angle=ctx.inverted_angle, constants=ctx.constants,
                initial_phi_rad=phi0, initial_thetadot_radps=thetadot,
                a_max=a_max, v_max=V_MAX_MPS,
                config_path=Path(ctx.config_path), controller_kind=ctx.controller_kind,
                arm_q=ctx.arm_q_array,
            )
            row = {
                "phi0_deg": phi_deg, "thetadot0_radps": thetadot,
                "captured": res["captured"], "guard_fired": res["guard_fired"],
                "guard_reason": res["guard_reason"],
                "peak_abs_cart_vel_cmd_mps": res["peak_abs_cart_vel_cmd_mps"],
                "peak_abs_cart_accel_cmd_mps2": res["peak_abs_cart_accel_cmd_mps2"],
                "peak_abs_joint_vel_radps": res["peak_abs_joint_vel_radps"],
                "final_phi_deg": res["final_phi_deg"],
            }
            if verify_k_zero and res["captured"]:
                res0 = run_lqr_trial(
                    model, np.zeros((1, 4)), duration_s=duration_s,
                    hanging_angle=ctx.hanging_angle, inverted_angle=ctx.inverted_angle,
                    constants=ctx.constants, initial_phi_rad=phi0, initial_thetadot_radps=thetadot,
                    a_max=a_max, v_max=V_MAX_MPS,
                    config_path=Path(ctx.config_path), controller_kind=ctx.controller_kind,
                    arm_q=ctx.arm_q_array,
                )
                row["k_zero_also_captured"] = res0["captured"]
                row["k_zero_final_phi_deg"] = res0["final_phi_deg"]
                row["active_control_confirmed"] = res["captured"] and not res0["captured"]
            rows.append(row)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="4-state cartpole LQR (high-level) over the torque OSC (low-level).")
    add_common_pendulum_args(parser, default_config=DEFAULT_CONFIG)
    parser.add_argument("--maxiter", type=int, default=20)
    parser.add_argument("--popsize", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--duration-s", type=float, default=4.0)
    parser.add_argument("--envelope", action="store_true",
                        help="After the gain search, also run the capture-envelope grid "
                             "(phi0 x thetadot0) with the K=0 counterfactual check.")
    parser.add_argument("--envelope-duration-s", type=float, default=5.0)
    parser.add_argument("--a-max-upper", type=float, default=10.0,
                        help="Upper bound on the LQR's commanded cart acceleration. "
                             "Set it from the pose's MEASURED command envelope "
                             "(pose_hold_orientation_check.py), not by habit. Default "
                             "10.0 preserves historical behaviour and is known to "
                             "authorise ~5-10x what the arm delivers at ARM_Q0.")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    ctx = context_from_args(args).resolve()
    print(describe_context(ctx))
    print("=== searching 4-state cartpole LQR Q/R (+a_max) via differential_evolution ===")
    result = search_lqr_gains(ctx, a_max_upper=float(args.a_max_upper),
                              maxiter=args.maxiter, popsize=args.popsize, seed=args.seed,
                               duration_s=args.duration_s)
    print("Q_diag =", result["Q_diag"], "a_max =", result["a_max"])
    print("K =", result["K"])
    print("cost =", result["cost"])

    envelope = None
    if args.envelope:
        print("=== capture envelope grid (phi0 x thetadot0), with K=0 counterfactual on captures ===")
        K = np.asarray(result["K"], dtype=np.float64).reshape(1, 4)
        envelope = capture_envelope_grid(
            ctx, K,
            phi_deg_values=[-30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30],
            thetadot_values=[-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0],
            a_max=result["a_max"], duration_s=args.envelope_duration_s,
        )
        n_cap = sum(1 for r in envelope if r["captured"])
        n_confirmed = sum(1 for r in envelope if r.get("active_control_confirmed"))
        n_k0_also = sum(1 for r in envelope if r.get("k_zero_also_captured"))
        print(f"captured {n_cap}/{len(envelope)} cells; active-control-confirmed {n_confirmed}/{n_cap}; "
              f"K=0-also-captured {n_k0_also}/{n_cap} (should be 0)")

    if args.output_json:
        write_output_json(args.output_json, {
            "context": {"pendulum_xml": ctx.pendulum_xml, "arm_q": list(ctx.arm_q),
                        "config_path": ctx.config_path, "controller_kind": ctx.controller_kind,
                        "hanging_angle": ctx.hanging_angle, "inverted_angle": ctx.inverted_angle,
                        "constants": ctx.constants},
            "lqr": result,
            "envelope": envelope,
        })
        print("wrote", args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
