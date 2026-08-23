#!/usr/bin/env python3
"""Torque/impedance-lane balance controller for the pendulum apparatus,
driven by a full-order LQR instead of the hand-picked PD used in
tools/diagnostics/pendulum_balance_test.py (the velocity-lane version).

Why this exists: the velocity-lane test drives the arm through
ik_seeded_resolution, a POSITION-tracking joint-space follower -- its
"actuator" is a lagged position chase, not a force. This repo's TORQUE
lane has real <motor> actuators (assets/ur5e_torque/ur5e_torque.xml,
gear=1) giving direct force authority, a better structural match for a
fast stabilization problem.

DESIGN HISTORY, kept because it is the reason this version looks the way
it does: the first version of this script hand-derived a classical
2-state cart-pole reduction (cart mass = 1/(Jx@Minv@Jx.T), pendulum m/l/I
from this session's own measurements) and designed a 4-state LQR
(x_err, xdot, theta_err, thetadot) on it. That LQR was verified correct
in isolation -- simulating the pure 4-state linear model directly showed
clean, well-behaved convergence with NO overshoot even for a 0.02 rad
perturbation. But driving the REAL MuJoCo system with the resulting law
diverged badly even at that same tiny perturbation (peak error 0.5-0.6
rad from a 0.02 rad start), and neither a genuine x_err sign bug nor a
genuine gravity/Coriolis double-counting bug (both real, both found and
fixed) resolved it. Root cause: the hand-derived model assumes a clean
1-DOF translating cart, but jx (the Jacobian row for X) has FOUR nonzero
joint components (shoulder_pan/lift/elbow/wrist_1) -- a task-space force
command through this redundant, non-decoupled Jacobian does NOT produce
pure X translation of the pendulum mount; it also rotates and shifts the
wrist in Y/Z, and the pendulum's hinge (whose axis is non-trivially
oriented relative to world frame at this arm pose -- see docs/status/
ur5e_pendulum_cad_model_2026-08-09.md's "settles at 2.786 rad, not 0"
finding) very plausibly responds to that rotation as much as to the
translation, which the 2-state reduction has no way to represent.

Fix: linearize the REAL, FULL nonlinear MuJoCo system numerically via
mujoco.mjd_transitionFD (finite-differenced transition matrices, MuJoCo's
own built-in linearization tool -- no hand-derived reduction, no
opportunity for this class of kinematic-coupling error) and design a
full-order LQR directly on that. The (small) reward for hand-deriving a
reduced model would have been interpretability; the correctness cost
turned out to dominate.

State: the tangent-space perturbation [dq (7, =6 arm + 1 pendulum since
every joint here is a plain hinge, nq=nv so no quaternion subtlety),
dqd (7)] around the equilibrium (ARM_Q0, pendulum at the inverted angle,
qvel=0, ctrl=static gravity torque holding that pose). Q penalizes the
pendulum's angle/rate heavily and the arm's joint deviation/rate lightly
(replacing the earlier version's separate, ad-hoc nullspace-posture term
with a single, jointly-optimal design instead of two controllers that
could fight each other). R penalizes the 6 actuator torques.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.linalg import solve_discrete_are

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from simulation.ur5e_pendulum_compose import (  # noqa: E402
    DEFAULT_ARM_Q,
    DEFAULT_PENDULUM_XML,
    arm_q_for_pendulum_xml,
    compose_ur5e_pendulum_model,
    derive_pendulum_constants,
)

# Fixed 2026-08-12: this used to be a generic, unrelated arm pose ([0, -pi/2,
# pi/2, -pi/2, -pi/2, 0]) with no connection to where the swing-up scripts
# (pendulum_swingup_*.py) actually leave the pendulum. That meant the LQR
# gains computed here were tuned for a completely different arm
# configuration than swing-up's own -- balance and swing-up were two
# disconnected problems, not a real handoff pipeline. Must therefore stay
# identical to pendulum_swingup_energy_shaping.py::ARM_Q0.
#
# Updated again later 2026-08-12, together with the hinge-axis correction to
# local Z: this is now the user's ACTUAL real-hardware UR5e configuration
# (wrist_2 wrapped into the model's valid range from a real 6.2879 rad
# probe). See pendulum_swingup_energy_shaping.py's ARM_Q0 comment for the
# measured geometry that justifies it, including the separate, explicitly
# NOT-conflated caveat that this pose sits on the wrist_2=0 arm singularity
# (cond(J6) = 1396) -- which matters for the LQR designed below, since it
# linearizes the full arm+pendulum system AT this pose.
# Single source of truth is simulation/ur5e_pendulum_compose.py's asset<->pose
# table; value unchanged. Must stay identical to
# pendulum_swingup_energy_shaping.py::ARM_Q0 so balance and swing-up remain a
# real handoff rather than two disconnected problems.
ARM_Q0 = DEFAULT_ARM_Q
TORQUE_LIMIT_NM = np.array([150.0, 150.0, 150.0, 28.0, 28.0, 28.0])
RATE_HZ = 500.0
CONTROL_DT = 1.0 / RATE_HZ
FALL_THRESHOLD_RAD = 2.8  # see pendulum_balance_test.py's own note on why this
# must stay comfortably clear of any tested perturbation.


def find_inverted_angle(model, data, pend_qpos_adr: int) -> float:
    """Finds the TRUE unstable equilibrium directly, via qfrc_bias's own
    zero-crossing + gradient sign, NOT via "release and watch it settle."

    That release-and-settle approach (this function's first two versions)
    was fooled TWICE by this system's own very slow dynamics: releasing
    from 0.3 rad never converged in 8 s (still drifting at qvel~0.0025
    rad/s) -- caught. The fix (release from pi/2, require settled-window
    std < 0.005 over the last 200 steps) then ALSO passed while sitting at
    a genuinely non-equilibrium point (measured directly: qfrc_bias there
    was -0.06, nowhere near 0) -- a short-window std check cannot
    distinguish "truly stationary" from "drifting so slowly the window
    looks flat," exactly the kind of near-threshold slow-dynamics trap
    this session has hit before (see the swing-up friction analysis).
    Direct, robust fix: scan qfrc_bias_pend(theta) for sign changes (its
    true physical zero IS the equilibrium condition, unconditionally, no
    settling time needed), then classify stability by the sign of its
    LOCAL SLOPE -- confirmed empirically against a real, long (16 s),
    large-perturbation (0.15 rad) simulation: the candidate whose distance
    from a perturbed start GREW over time is unstable (slope > 0, pushes
    away); the one whose distance SHRANK is stable (slope < 0, restores).

    BUG FIXED 2026-08-13: this function's ``data`` parameter was silently
    unused -- every internal computation (the qfrc_bias scan, the
    stability-classification stepping, and the gravity-compensation torque
    inside it) hardcoded the module's own ``ARM_Q0`` constant instead of
    reading the arm pose the caller actually set up in ``data.qpos[:6]``.
    That meant this function ALWAYS analyzed pendulum equilibria at the old
    singular ARM_Q0 pose, silently ignoring any other pose a caller passed
    in via ``data`` -- confirmed as the root cause of a real, wrong
    hanging/inverted-angle result at a newly-found, differently-conditioned
    pose (direct check: the label this bug called "hanging" was measurably
    HIGHER in world Z, i.e. higher gravitational PE, than the one it called
    "inverted" -- backwards for a passive pendulum, whose true hanging
    point is always the lower-PE one). Fixed by capturing the caller's
    actual arm pose once at entry and using it everywhere below, instead of
    the hardcoded constant."""
    arm_q = np.asarray(data.qpos[:6], dtype=np.float64).copy()

    def qfrc_bias_pend(theta: float) -> float:
        d = mujoco.MjData(model)
        d.qpos[:6] = arm_q
        d.qpos[pend_qpos_adr] = theta
        d.qvel[:] = 0.0
        mujoco.mj_forward(model, d)
        return float(d.qfrc_bias[pend_qpos_adr])

    thetas = np.linspace(-np.pi, np.pi, 361)
    vals = np.array([qfrc_bias_pend(t) for t in thetas])
    crossings = []
    for i in range(len(thetas) - 1):
        if vals[i] == 0.0 or (vals[i] > 0) != (vals[i + 1] > 0):
            lo, hi = thetas[i], thetas[i + 1]
            for _ in range(60):
                mid = (lo + hi) / 2
                if (qfrc_bias_pend(mid) > 0) == (vals[i] > 0):
                    lo = mid
                else:
                    hi = mid
            crossings.append((lo + hi) / 2)

    # Deduplicate crossings that are the SAME physical equilibrium detected
    # twice near the +-pi periodic boundary (thetas' scan endpoints -pi and
    # +pi are the same point; a spurious extra detection there was found
    # 2026-08-11 after the pendulum rod-length correction shrank peak
    # qfrc_bias by ~6x, 0.174 -> 0.028 Nm -- the same absolute floating-
    # point evaluation noise near the wrap boundary that was negligible at
    # the old torque scale is no longer negligible at the new one. Merge
    # any two raw crossings within 0.05 rad of each other (wraparound-
    # aware) into one, averaged in the wrapped sense, before requiring
    # exactly 2 real equilibria.
    def _wrapped_dist(a: float, b: float) -> float:
        d = abs(a - b)
        return min(d, 2 * np.pi - d)

    deduped: list[float] = []
    for c in crossings:
        merged = False
        for i, existing in enumerate(deduped):
            if _wrapped_dist(c, existing) < 0.05:
                # Average in the wrapped sense: shift c into existing's local
                # branch before averaging, then re-wrap to (-pi, pi].
                c_local = existing + (np.mod(c - existing + np.pi, 2 * np.pi) - np.pi)
                avg = np.mod((existing + c_local) / 2.0 + np.pi, 2 * np.pi) - np.pi
                deduped[i] = float(avg)
                merged = True
                break
        if not merged:
            deduped.append(c)
    crossings = deduped

    if len(crossings) != 2:
        raise RuntimeError(f"expected exactly 2 equilibria, found {len(crossings)}: {crossings}")

    # Classification via qfrc_bias's local slope sign was WRONG -- found
    # 2026-08-09, later the same session, via a direct contradiction: a
    # zero-friction, tiny-perturbation (1e-3 rad) release at what this
    # slope method called "unstable" produced BOUNDED OSCILLATION (theta_err
    # staying within +-0.001 rad over 4 s), never growing -- the textbook
    # signature of a STABLE equilibrium, not an unstable one (a genuinely
    # unstable point under zero friction shows exponential growth, full
    # stop). This directly contradicted this function's own docstring claim
    # of an earlier cross-validation against a long real simulation --
    # that earlier validation was real and was never wrong, but its result
    # was not correctly propagated into the slope-based formula that
    # replaced it. Every torque-lane balance result validated against the
    # old classification was therefore testing recovery to the STABLE
    # point (a trivial, physically-expected outcome for any passive
    # pendulum), not genuine inverted balance -- see docs/status/
    # pendulum_balance_torque_lqr_2026-08-09.md's later corrections.
    #
    # Fixed by classifying via the SAME direct, unambiguous ground-truth
    # method as the original validation: temporarily zero out this joint's
    # damping/frictionloss on a scratch copy, release from a tiny (1e-3 rad)
    # perturbation, and check whether |theta_err| grows (unstable) or stays
    # bounded (stable) over a short window -- no reliance on interpreting
    # any internal MuJoCo sign convention.
    pend_dof_adr = model.jnt_dofadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "/pendulum_hinge")
    ]
    saved_damping = float(model.dof_damping[pend_dof_adr])
    saved_frictionloss = float(model.dof_frictionloss[pend_dof_adr])
    model.dof_damping[pend_dof_adr] = 0.0
    model.dof_frictionloss[pend_dof_adr] = 0.0
    classified = []
    # try/finally added 2026-08-12 (audit): this block mutates the CALLER'S
    # model in place. Every swing-up script calls this on the same model object
    # it then simulates with, so any exception escaping the classification loop
    # (or a KeyboardInterrupt during it -- these loops run 2000 steps per
    # candidate) would leave the pendulum hinge permanently frictionless and
    # undamped for the rest of the process, silently changing the physics of
    # every subsequent trial with no error to connect it to.
    try:
        for c in crossings:
            d = mujoco.MjData(model)
            d.qpos[:6] = arm_q
            d.qpos[pend_qpos_adr] = c + 1e-3
            d.qvel[:] = 0.0
            mujoco.mj_forward(model, d)
            max_dev = 0.0
            for step in range(2000):
                theta = float(d.qpos[pend_qpos_adr])
                dev = abs(float(np.mod(theta - c + np.pi, 2 * np.pi) - np.pi))
                max_dev = max(max_dev, dev)
                # Hold the arm via real gravity-compensating torque (same
                # technique as static_gravity_torque/run_torque_balance_trial
                # elsewhere in this file), NOT a hard qpos/qvel reset every step.
                # Fixed 2026-08-11 after the pendulum rod-length correction
                # shrank the pendulum's own inertia ~15x: the discontinuous
                # per-step reset this loop used to do injects a real momentum
                # artifact through the arm/pendulum mass-matrix coupling on
                # every step (confirmed directly: traced max_dev over time and
                # found large, non-monotonic, non-exponential jumps -- e.g.
                # 0.001->0.66 rad by t=0.2s -- for BOTH candidate equilibria,
                # not the clean bounded-vs-exponential signature this
                # classification depends on). That artifact was negligible
                # relative to the old, ~15x-heavier pendulum's own inertia;
                # it no longer is. A smooth holding torque has no discontinuity
                # to inject.
                tau_gravity = static_gravity_torque(model, np.concatenate([arm_q, [theta]]))
                d.ctrl[:6] = tau_gravity[:6]
                mujoco.mj_step(model, d)
            # Bounded (stable) oscillation stays within a small multiple of the
            # 1e-3 rad initial perturbation; genuine instability grows well
            # beyond it within 4 s (2000 steps at 2 ms).
            classified.append((c, "unstable" if max_dev > 0.02 else "stable"))
    finally:
        model.dof_damping[pend_dof_adr] = saved_damping
        model.dof_frictionloss[pend_dof_adr] = saved_frictionloss

    unstable = [c for c, kind in classified if kind == "unstable"]
    if len(unstable) != 1:
        raise RuntimeError(f"expected exactly 1 unstable equilibrium, classification: {classified}")
    return float(unstable[0])


def static_gravity_torque(model, q_full: np.ndarray) -> np.ndarray:
    """qfrc_bias(q, qd=0) -- static gravity only, NOT the live qfrc_bias(q,qd)
    which also includes the Coriolis/velocity-coupling term. Using the live
    value as "gravity compensation" was a real bug found while debugging
    the first version of this script: it cancels the pendulum's real
    momentum-coupling reaction onto the arm, not just gravity."""
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = q_full
    scratch.qvel[:] = 0.0
    mujoco.mj_forward(model, scratch)
    return scratch.qfrc_bias.copy()


def linearize_and_design_lqr(
    model,
    inverted_angle: float,
    arm_q=None,
    q_pend_angle: float = 200.0,
    q_pend_vel: float = 5.0,
    q_arm_pos: float = 0.05,
    q_arm_vel: float = 0.05,
    r_weight: float = 0.0005,
    zero_hinge_frictionloss_for_linearization: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """``zero_hinge_frictionloss_for_linearization`` (added 2026-08-14, default
    False = historical behavior) temporarily zeroes the PENDULUM HINGE's
    ``frictionloss`` for the ``mjd_transitionFD`` call only, restoring it
    immediately afterwards. The simulated plant is never changed.

    Why it exists: Coulomb ``frictionloss`` (0.001 Nm) completely dominates at
    the microscopic velocities a finite-difference perturbation produces, so the
    linearizer sees a nearly-STUCK hinge and the inverted equilibrium's unstable
    mode is hidden. Measured at ARM_Q0/wrist_2=-90 with
    ``pendulum_attachment_realrod.xml`` (dt=0.002, omega=10.8334 rad/s):

        quantity      as-modelled   frictionloss zeroed   analytic
        A[thd,thd]      0.810829         0.999157         0.99916
        A[thd,th]       0.023741         0.234904         w^2*dt = 0.234725
        max |eig|       1.000252         1.021474         exp(w*dt) = 1.021903

    i.e. the as-modelled linearization reports the pendulum losing 19% of its
    velocity per 2 ms step and being essentially non-divergent. An LQR designed
    against that plant is far too slow: it produced closed-loop time constants of
    ~1-2 s against a pendulum that actually falls with a 0.092 s time constant,
    and FELL at every perturbation from 0.05 rad up. Sweeping ``r_weight`` over
    2e9 barely moved the eigenvalues, which is the signature of a wrong MODEL
    rather than wrong weights.

    Default stays False so previously-derived gains remain reproducible; pass
    True for any new design. Friction is still fully present in the simulated
    rollout either way -- this only corrects the model used for gain synthesis.
    """
    nv = model.nv
    arm_q = ARM_Q0 if arm_q is None else np.asarray(arm_q, dtype=np.float64).reshape(6)
    data = mujoco.MjData(model)
    q_eq = np.concatenate([arm_q, [inverted_angle]])
    data.qpos[:] = q_eq
    data.qvel[:] = 0.0
    tau_eq = static_gravity_torque(model, q_eq)[:6]
    data.ctrl[:6] = tau_eq
    mujoco.mj_forward(model, data)

    A = np.zeros((2 * nv, 2 * nv))
    B = np.zeros((2 * nv, model.nu))
    _pend_dof = int(model.jnt_dofadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "/pendulum_hinge")
    ])
    _saved_friction = float(model.dof_frictionloss[_pend_dof])
    if zero_hinge_frictionloss_for_linearization:
        model.dof_frictionloss[_pend_dof] = 0.0
    try:
        mujoco.mjd_transitionFD(model, data, 1e-6, 1, A, B, None, None)
    finally:
        model.dof_frictionloss[_pend_dof] = _saved_friction

    Q = np.zeros(2 * nv)
    Q[:6] = q_arm_pos
    Q[6] = q_pend_angle
    Q[nv:nv + 6] = q_arm_vel
    Q[nv + 6] = q_pend_vel
    Q = np.diag(Q)
    # R scaled per-actuator by 1/torque_limit^2 (standard LQR practice) --
    # a uniform R let the solver lean almost entirely on wrist_2 (only
    # 28 Nm of authority vs 150 Nm for the shoulder/elbow joints), which
    # then saturated on every cycle regardless of how large r_weight was
    # pushed (demanded torque barely dropped even at 2000x the original
    # r_weight, and eigenvalues drifted TOWARD instability instead of away
    # -- the signature of one weak-authority actuator being asked to do a
    # job only spreading the load across joints can safely do).
    R = np.diag(r_weight / (TORQUE_LIMIT_NM ** 2))

    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
    eigvals = np.linalg.eigvals(A - B @ K)
    diag = {"tau_eq": tau_eq, "eigvals_discrete": eigvals, "A": A, "B": B}
    return K, q_eq, diag


def run_torque_balance_trial(
    model_template,
    K: np.ndarray,
    q_eq: np.ndarray,
    perturbation_rad: float,
    duration_s: float,
    theta_dot_perturbation: float = 0.0,
    arm_q=None,
) -> dict:
    arm_q = ARM_Q0 if arm_q is None else np.asarray(arm_q, dtype=np.float64).reshape(6)
    model = model_template
    data = mujoco.MjData(model)
    pend_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "/pendulum_hinge")
    pend_qpos_adr = model.jnt_qposadr[pend_joint_id]
    pend_dof_adr = model.jnt_dofadr[pend_joint_id]
    inverted_angle = float(q_eq[6])

    data.qpos[:6] = arm_q
    data.qpos[pend_qpos_adr] = inverted_angle + perturbation_rad
    data.qvel[:] = 0.0
    data.qvel[pend_dof_adr] = theta_dot_perturbation
    mujoco.mj_forward(model, data)

    n_steps = int(duration_s * RATE_HZ)
    fell_at = None
    peak_theta_err = 0.0
    theta_err_hist = []
    # Torque-saturation tracking (added 2026-08-10): the pre-clip ("raw")
    # commanded torque is what the LQR law actually WANTS, vs tau (post-clip)
    # which is what the actuator can deliver -- a controller can "survive" a
    # trial (final_theta_err small, never falls) while spending most of the
    # trial with one or more joints clipped hard against their limit, which
    # is a real, load-bearing distinction the old return dict couldn't see at
    # all (tau_extra/tau/etc were local variables, never surfaced). Found the
    # hard way: a new pendulum mount geometry passed every existing pass/fail
    # check yet turned out to saturate shoulder_pan 67.6% of cycles (peak
    # commanded torque 351 Nm against a 150 Nm limit) and wrist_2 to 732 Nm
    # against a 28 Nm limit on brief spikes -- invisible without this.
    overshoot_ratio_hist = []  # per-step: max over joints of max(0, |tau_raw|-limit)/limit
    saturated_joint_frac_hist = []  # per-step: fraction of 6 joints with |tau_raw| > limit

    for step in range(n_steps):
        theta = float(data.qpos[pend_qpos_adr])
        theta_err = float(np.mod(theta - inverted_angle + np.pi, 2 * np.pi) - np.pi)
        theta_err_hist.append(theta_err)
        peak_theta_err = max(peak_theta_err, abs(theta_err))
        if abs(theta_err) > FALL_THRESHOLD_RAD and fell_at is None:
            fell_at = step * CONTROL_DT

        dq = data.qpos.copy()
        dq[6] = theta_err  # wrapped, replaces the raw (possibly-unwrapped) qpos difference
        dq[:6] = data.qpos[:6] - q_eq[:6]
        dqd = data.qvel.copy()
        state = np.concatenate([dq, dqd])

        tau_extra = -K @ state
        tau_gravity = static_gravity_torque(model, data.qpos)[:6]
        tau_raw = tau_extra + tau_gravity
        tau = np.clip(tau_raw, -TORQUE_LIMIT_NM, TORQUE_LIMIT_NM)
        data.ctrl[:6] = tau

        over = np.maximum(0.0, np.abs(tau_raw) - TORQUE_LIMIT_NM) / TORQUE_LIMIT_NM
        overshoot_ratio_hist.append(float(np.max(over)))
        saturated_joint_frac_hist.append(float(np.mean(np.abs(tau_raw) > TORQUE_LIMIT_NM)))

        mujoco.mj_step(model, data)

        if fell_at is not None:
            break

    theta_err_arr = np.array(theta_err_hist)
    return {
        "perturbation_rad": perturbation_rad,
        "theta_dot_perturbation": theta_dot_perturbation,
        "fell_at_s": fell_at,
        "peak_theta_err": peak_theta_err,
        "final_theta_err": float(theta_err_arr[-1]) if len(theta_err_arr) else None,
        "survived_full_duration": fell_at is None,
        "mean_overshoot_ratio": float(np.mean(overshoot_ratio_hist)) if overshoot_ratio_hist else 0.0,
        "peak_overshoot_ratio": float(np.max(overshoot_ratio_hist)) if overshoot_ratio_hist else 0.0,
        "mean_saturated_joint_frac": float(np.mean(saturated_joint_frac_hist)) if saturated_joint_frac_hist else 0.0,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Full-order LQR balance controller for the pendulum apparatus.")
    parser.add_argument(
        "--pendulum-xml", default=str(DEFAULT_PENDULUM_XML),
        help="Pendulum attachment MJCF to compose onto the arm "
             "(default: the real 0.12 m apparatus).")
    parser.add_argument(
        "--start-q-rad", nargs=6, type=float, default=None,
        metavar=("Q1", "Q2", "Q3", "Q4", "Q5", "Q6"),
        help="Arm pose to linearize/balance at, 6 joint angles in radians. "
             "Default: the pose registered for --pendulum-xml in "
             "simulation.ur5e_pendulum_compose.PENDULUM_ASSET_ARM_Q.")
    parser.add_argument("--output-json", default=None,
                        help="Write results to this path as JSON.")
    parser.add_argument("--duration-s", type=float, default=8.0,
                        help="Duration of each balance trial.")
    parser.add_argument("--zero-frictionloss-for-linearization", action="store_true",
                        help="Zero the hinge frictionloss for the mjd_transitionFD call "
                             "ONLY (the simulated plant keeps its friction). Without this, "
                             "Coulomb friction dominates at finite-difference scale and the "
                             "linearizer reports the inverted equilibrium as nearly "
                             "non-divergent (max|eig| 1.000252 vs the analytic 1.021903), "
                             "so the designed LQR is ~10-20x too slow and falls. Default "
                             "off preserves previously-derived gains.")
    parser.add_argument("--r-weight", type=float, default=1e6,
                        help="LQR control-effort weight. MUST stay well above "
                             "linearize_and_design_lqr's own 0.0005 default, "
                             "which predates the per-actuator R scaling fix.")
    parser.add_argument("--perturbations-rad", nargs="+", type=float,
                        default=[0.02, 0.5, 1.0, 1.5, 2.0, 2.3, 2.5, 2.7])
    # NOTE: --controller-kind / --config deliberately absent. This script does
    # not run the Cartesian impedance controller at all -- it designs and
    # applies its own full-order LQR directly on the linearized MuJoCo system,
    # so neither flag has anything to select here.
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    pendulum_xml = str(Path(args.pendulum_xml).resolve())
    arm_q = (np.asarray(args.start_q_rad, dtype=np.float64) if args.start_q_rad is not None
             else arm_q_for_pendulum_xml(pendulum_xml))

    model = compose_ur5e_pendulum_model(pendulum_xml=pendulum_xml)
    pend_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "/pendulum_hinge")
    pend_qpos_adr = model.jnt_qposadr[pend_joint_id]

    constants = derive_pendulum_constants(model, arm_q)
    print(f"pendulum_xml={Path(pendulum_xml).name}  arm_q={np.round(arm_q, 6).tolist()}")
    print(f"derived: m*g*r={constants.mgr_nm:.6f} Nm  I_pivot={constants.i_pivot_kgm2:.6e} kg m^2  "
          f"T_natural={constants.t_natural_s:.4f} s")

    # find_inverted_angle reads the arm pose out of `data` (see its own
    # 2026-08-13 note). Posing it here is REQUIRED, not cosmetic: with a bare
    # MjData the arm sits at all-zeros and the equilibria come back for a
    # completely different configuration.
    data = mujoco.MjData(model)
    data.qpos[:6] = arm_q
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    inverted_angle = find_inverted_angle(model, data, pend_qpos_adr)
    print(f"inverted_angle (composed frame) = {inverted_angle:.4f} rad")

    # r_weight MUST be passed explicitly -- the function's own default
    # (0.0005) is the PRE-FIX value from before the per-actuator R-scaling
    # fix (see this module's docstring, bug #4): at that r_weight, demanded
    # torques saturate every cycle and the closed loop is unreliable.
    K, q_eq, diag = linearize_and_design_lqr(
        model, inverted_angle, arm_q=arm_q, r_weight=args.r_weight,
        zero_hinge_frictionloss_for_linearization=bool(
            args.zero_frictionloss_for_linearization),
    )
    print(f"tau_eq (static gravity torque at equilibrium) = {diag['tau_eq']}")
    n_unstable = int(np.sum(np.abs(diag["eigvals_discrete"]) >= 1.0))
    print(f"discrete closed-loop eigenvalue magnitudes (all must be < 1.0): "
          f"{np.sort(np.abs(diag['eigvals_discrete']))[::-1][:6]}")
    if n_unstable > 0:
        print(f"WARNING: {n_unstable} closed-loop eigenvalues have magnitude >= 1.0 -- unstable per the linear model.")
        return 1

    rows = []
    print(f"\n{'pert (rad)':>10} {'fell_at_s':>10} {'peak_err':>10} {'final_err':>10} {'result':>10}")
    for pert in args.perturbations_rad:
        r = run_torque_balance_trial(model, K, q_eq, pert, duration_s=args.duration_s,
                                     arm_q=arm_q)
        rows.append(r)
        result = "SURVIVED" if r["survived_full_duration"] else "FELL"
        print(f"{pert:10.2f} {str(r['fell_at_s']):>10} {r['peak_theta_err']:10.4f} "
              f"{r['final_theta_err']:10.4f} {result:>10}")

    if args.output_json:
        payload = {
            "pendulum_xml": pendulum_xml,
            "arm_q": [float(v) for v in arm_q],
            "inverted_angle_rad": float(inverted_angle),
            "r_weight": float(args.r_weight),
            "constants": {
                "mgr_nm": constants.mgr_nm, "i_pivot_kgm2": constants.i_pivot_kgm2,
                "m_total_kg": constants.m_total_kg, "r_com_m": constants.r_com_m,
                "omega_natural_radps": constants.omega_natural_radps,
                "t_natural_s": constants.t_natural_s, "e_top_j": constants.e_top_j,
            },
            "max_abs_closed_loop_eig": float(np.max(np.abs(diag["eigvals_discrete"]))),
            "trials": rows,
        }
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float))
        print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
