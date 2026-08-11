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

import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.linalg import solve_discrete_are

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from simulation.ur5e_pendulum_compose import compose_ur5e_pendulum_model  # noqa: E402

ARM_Q0 = np.array([0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0])
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
    away); the one whose distance SHRANK is stable (slope < 0, restores)."""

    def qfrc_bias_pend(theta: float) -> float:
        d = mujoco.MjData(model)
        d.qpos[:6] = ARM_Q0
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
    for c in crossings:
        d = mujoco.MjData(model)
        d.qpos[:6] = ARM_Q0
        d.qpos[pend_qpos_adr] = c + 1e-3
        d.qvel[:] = 0.0
        mujoco.mj_forward(model, d)
        max_dev = 0.0
        for step in range(2000):
            theta = float(d.qpos[pend_qpos_adr])
            dev = abs(float(np.mod(theta - c + np.pi, 2 * np.pi) - np.pi))
            max_dev = max(max_dev, dev)
            d.qpos[:6] = ARM_Q0
            d.qvel[:6] = 0.0
            mujoco.mj_step(model, d)
        # Bounded (stable) oscillation stays within a small multiple of the
        # 1e-3 rad initial perturbation; genuine instability grows well
        # beyond it within 4 s (2000 steps at 2 ms).
        classified.append((c, "unstable" if max_dev > 0.02 else "stable"))
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
    q_pend_angle: float = 200.0,
    q_pend_vel: float = 5.0,
    q_arm_pos: float = 0.05,
    q_arm_vel: float = 0.05,
    r_weight: float = 0.0005,
) -> tuple[np.ndarray, np.ndarray, dict]:
    nv = model.nv
    data = mujoco.MjData(model)
    q_eq = np.concatenate([ARM_Q0, [inverted_angle]])
    data.qpos[:] = q_eq
    data.qvel[:] = 0.0
    tau_eq = static_gravity_torque(model, q_eq)[:6]
    data.ctrl[:6] = tau_eq
    mujoco.mj_forward(model, data)

    A = np.zeros((2 * nv, 2 * nv))
    B = np.zeros((2 * nv, model.nu))
    mujoco.mjd_transitionFD(model, data, 1e-6, 1, A, B, None, None)

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
) -> dict:
    model = model_template
    data = mujoco.MjData(model)
    pend_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "/pendulum_hinge")
    pend_qpos_adr = model.jnt_qposadr[pend_joint_id]
    pend_dof_adr = model.jnt_dofadr[pend_joint_id]
    inverted_angle = float(q_eq[6])

    data.qpos[:6] = ARM_Q0
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


def main() -> int:
    model = compose_ur5e_pendulum_model()
    data = mujoco.MjData(model)
    pend_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "/pendulum_hinge")
    pend_qpos_adr = model.jnt_qposadr[pend_joint_id]

    inverted_angle = find_inverted_angle(model, data, pend_qpos_adr)
    print(f"inverted_angle (composed frame) = {inverted_angle:.4f} rad")

    # r_weight MUST be passed explicitly -- the function's own default
    # (0.0005) is the PRE-FIX value from before the per-actuator R-scaling
    # fix (see this module's docstring, bug #4): at that r_weight, demanded
    # torques saturate every cycle and the closed loop is unreliable (a
    # fresh run without this override showed pert=1.0/1.5 rad FALLING,
    # contradicting the validated 0.02-3.0 rad clean-convergence result --
    # caught by re-running this exact script standalone and finding it did
    # NOT reproduce the validation performed via inline snippets during
    # development, which always passed r_weight=1e6 explicitly).
    K, q_eq, diag = linearize_and_design_lqr(model, inverted_angle, r_weight=1e6)
    print(f"tau_eq (static gravity torque at equilibrium) = {diag['tau_eq']}")
    n_unstable = int(np.sum(np.abs(diag["eigvals_discrete"]) >= 1.0))
    print(f"discrete closed-loop eigenvalue magnitudes (all must be < 1.0): "
          f"{np.sort(np.abs(diag['eigvals_discrete']))[::-1][:6]}")
    if n_unstable > 0:
        print(f"WARNING: {n_unstable} closed-loop eigenvalues have magnitude >= 1.0 -- unstable per the linear model.")
        return 1

    print(f"\n{'pert (rad)':>10} {'fell_at_s':>10} {'peak_err':>10} {'final_err':>10} {'result':>10}")
    for pert in [0.02, 0.5, 1.0, 1.5, 2.0, 2.3, 2.5, 2.7]:
        r = run_torque_balance_trial(model, K, q_eq, pert, duration_s=8.0)
        result = "SURVIVED" if r["survived_full_duration"] else "FELL"
        print(f"{pert:10.2f} {str(r['fell_at_s']):>10} {r['peak_theta_err']:10.4f} "
              f"{r['final_theta_err']:10.4f} {result:>10}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
