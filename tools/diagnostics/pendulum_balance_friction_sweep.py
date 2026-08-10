#!/usr/bin/env python3
"""How does the pendulum's own hinge friction/damping change the balance
picture? Sweeps a multiplier on the pendulum joint's dof_damping/
dof_frictionloss (applied at the COMPILED MODEL level, at runtime -- never
touches assets/ur5e_pendulum/pendulum_attachment.xml, whose placeholder
values the user is checking against the real hardware separately) and at
each level:

  1. Bisects the PASSIVE-only (K=0, pure gravity-hold, no active
     correction at all) recovery envelope boundary -- the same "does
     doing nothing already solve it" baseline this session's whole
     balance investigation kept needing to re-establish before trusting
     any "active control works" claim (see docs/status/
     pendulum_balance_torque_lqr_2026-08-09.md's correction).
  2. At a perturbation just PAST that passive boundary (where passive
     alone is confirmed to fail), tries active LQR configs -- informed by
     this session's most recent finding that weak arm-redundant-direction
     regularization lets the arm swing unnecessarily far and then take a
     long time to settle, so this sweep specifically tries the
     strengthened-regularization region of the design space, not the
     original weak defaults.
  3. Reports whether ANY tried active config extends recovery past the
     passive boundary at that friction level, and re-tunes (tries a small
     grid) rather than a single fixed gain, since the earlier
     investigation found the useful gain region depends on both R and the
     arm-regularization weights together.

Physical context for why this direction (INCREASING friction) is being
swept, not decreasing: the current placeholder values (damping=0.02
Nm*s/rad, frictionloss=0.01 Nm) are unmeasured. If real hardware turns out
to have MORE resistance than modeled, passive recovery should only get
easier (more dissipation); this sweep characterizes exactly how much
easier, and whether active control's usefulness (already found to be
marginal-to-absent at the placeholder value, see the torque_lqr doc's
correction) shrinks further as friction increases -- the expected
direction -- or reveals something unexpected.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import tools.diagnostics.pendulum_balance_torque_lqr as base  # noqa: E402

ARM_Q0 = base.ARM_Q0
TORQUE_LIMIT_NM = base.TORQUE_LIMIT_NM
RATE_HZ = base.RATE_HZ
CONTROL_DT = base.CONTROL_DT
static_gravity_torque = base.static_gravity_torque

BASELINE_DAMPING = 0.02
BASELINE_FRICTIONLOSS = 0.01


def build_model_with_friction(friction_mult: float):
    model = base.compose_ur5e_pendulum_model()
    pend_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "/pendulum_hinge")
    pend_dof_adr = model.jnt_dofadr[pend_joint_id]
    model.dof_damping[pend_dof_adr] = BASELINE_DAMPING * friction_mult
    model.dof_frictionloss[pend_dof_adr] = BASELINE_FRICTIONLOSS * friction_mult
    return model, pend_joint_id, pend_dof_adr


def run_trial(model, K, q_eq, pend_qpos_adr, pend_dof_adr, inverted_angle,
              perturbation_rad, duration_s=8.0, converged_tol_rad=0.3):
    """"Survived" is judged by CONVERGENCE BY THE END OF THE RUN
    (|final_theta_err| < converged_tol_rad), NOT by an instantaneous
    threshold crossing. An earlier version used a fixed
    `fall_threshold_rad` check -- which works fine when perturbations stay
    well below it (see pendulum_balance_torque_lqr.py's own use of this
    pattern), but breaks here because theta_err is wrap-limited to
    (-pi, pi] by construction, and this sweep specifically needs to test
    perturbations approaching pi. A perturbation of e.g. 3.0 rad started
    with |theta_err|=3.0 already -- any fixed threshold below pi trivially
    "fails" it on the very first sample regardless of real dynamics
    (confirmed: three different friction levels gave the IDENTICAL
    "boundary" using that approach, which is physically impossible and
    caught before trusting any result from it). A numerical-blowup guard
    (NaN or extreme joint velocity) is kept separately, since that is a
    genuine failure mode distinct from "hasn't converged yet."""
    data = mujoco.MjData(model)
    data.qpos[:6] = ARM_Q0
    data.qpos[pend_qpos_adr] = inverted_angle + perturbation_rad
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    n_steps = int(duration_s * RATE_HZ)
    fell_at = None
    peak = 0.0
    theta_err = perturbation_rad
    for step in range(n_steps):
        theta = float(data.qpos[pend_qpos_adr])
        theta_err = float(np.mod(theta - inverted_angle + np.pi, 2 * np.pi) - np.pi)
        peak = max(peak, abs(theta_err))
        if not np.all(np.isfinite(data.qpos)) or np.max(np.abs(data.qvel)) > 50.0:
            fell_at = step * CONTROL_DT
            break
        dq = data.qpos.copy()
        dq[6] = theta_err
        dq[:6] = data.qpos[:6] - q_eq[:6]
        dqd = data.qvel.copy()
        state = np.concatenate([dq, dqd])
        tau_extra = -K @ state
        tau_gravity = static_gravity_torque(model, data.qpos)[:6]
        tau = np.clip(tau_extra + tau_gravity, -TORQUE_LIMIT_NM, TORQUE_LIMIT_NM)
        data.ctrl[:6] = tau
        mujoco.mj_step(model, data)
    converged = fell_at is None and abs(theta_err) < converged_tol_rad
    fell_at = fell_at if fell_at is not None else (None if converged else duration_s)
    return {"fell_at_s": fell_at, "peak_theta_err": peak, "final_theta_err": theta_err,
            "survived": converged}


def bisect_passive_boundary(model, K_zero, q_eq, pend_qpos_adr, pend_dof_adr, inverted_angle,
                             lo=1.0, hi=3.13, iters=12):
    # Confirm lo survives and hi fails before bisecting; expand/report if not.
    r_lo = run_trial(model, K_zero, q_eq, pend_qpos_adr, pend_dof_adr, inverted_angle, lo)
    r_hi = run_trial(model, K_zero, q_eq, pend_qpos_adr, pend_dof_adr, inverted_angle, hi)
    if r_lo["survived"] and not r_hi["survived"]:
        for _ in range(iters):
            mid = (lo + hi) / 2
            r = run_trial(model, K_zero, q_eq, pend_qpos_adr, pend_dof_adr, inverted_angle, mid)
            if r["survived"]:
                lo = mid
            else:
                hi = mid
        return lo, hi
    return (lo, None) if r_lo["survived"] else (None, hi)


def main() -> int:
    # Analytical prior computed before choosing these levels: Coulomb
    # frictionloss stops the joint moving under gravity AT ALL, at any
    # angle, once frictionloss > max possible gravity torque
    # (m_total*g*l_com ~= 0.1784*9.81*0.0996 ~= 0.1743 Nm) -- at the
    # baseline frictionloss=0.01 Nm, that's multiplier ~= 17.4. Past that
    # point EVERY starting angle should trivially "survive" (the joint
    # never moves under gravity alone, so it never gets further from
    # wherever it started) -- not real balance, a physical degeneracy
    # where the whole passive-vs-active question stops being meaningful.
    # This sweep brackets that predicted point closely, then keeps pushing
    # well past it looking for the requested "level of trouble" -- ideally
    # confirming the predicted saturation and/or finding a genuine
    # numerical breakdown at extreme values, not just re-confirming the
    # same trivial result forever.
    friction_mults = [1.0, 2.0, 5.0, 10.0, 15.0, 17.0, 17.5, 18.0, 20.0, 25.0,
                       50.0, 100.0, 500.0, 1000.0, 5000.0, 20000.0, 100000.0]
    # Active configs to try, informed by the earlier finding that strong
    # arm-redundant regularization matters -- only run at a SUBSET of
    # friction levels (interesting boundary + a few past it) to keep
    # runtime reasonable across 17 friction levels:
    #   (q_arm_pos, q_arm_vel, q_pend_angle, q_pend_vel, r_weight)
    active_configs = [
        # DE-search-validated best config (docs/status/pendulum_balance_gain_search_2026-08-09.md):
        # clean convergence 0.05-0.4 rad at baseline (1x) friction.
        (2655.206336272164, 1.588066109094743, 20.947598183637, 655.1987898599957, 0.34681947401205965),
        (1008.735572590068, 32.27913994759627, 35.85445272321039, 310.35751752823853, 4.301938997677817),
        (500.0, 50.0, 200.0, 5.0, 10.0),
        (5000.0, 200.0, 500.0, 20.0, 5.0),
    ]
    active_rescue_check_mults = {1.0, 5.0, 15.0, 17.0, 17.5, 18.0, 20.0, 50.0}

    print(f"{'friction_x':>10} {'passive_boundary_rad':>22} {'passive_boundary_deg':>22}")
    results = []
    for mult in friction_mults:
        model, pend_joint_id, pend_dof_adr = build_model_with_friction(mult)
        data0 = mujoco.MjData(model)
        pend_qpos_adr = model.jnt_qposadr[pend_joint_id]
        inverted_angle = base.find_inverted_angle(model, data0, pend_qpos_adr)
        K_dummy, q_eq, _ = base.linearize_and_design_lqr(model, inverted_angle, r_weight=1e6)
        K_zero = np.zeros_like(K_dummy)

        lo, hi = bisect_passive_boundary(model, K_zero, q_eq, pend_qpos_adr, pend_dof_adr, inverted_angle)
        boundary = lo if lo is not None else 0.0
        print(f"{mult:10.1f} {boundary:22.4f} {boundary*180/np.pi:22.2f}")
        results.append({"mult": mult, "model": model, "pend_qpos_adr": pend_qpos_adr,
                         "pend_dof_adr": pend_dof_adr, "inverted_angle": inverted_angle,
                         "q_eq": q_eq, "K_zero": K_zero, "passive_boundary": boundary})

    print("\n=== At each friction level, can any active config rescue a perturbation "
          "just PAST the passive boundary? ===")
    for r in results:
        mult = r["mult"]
        if mult not in active_rescue_check_mults:
            continue
        model, pend_qpos_adr, pend_dof_adr = r["model"], r["pend_qpos_adr"], r["pend_dof_adr"]
        inverted_angle, q_eq = r["inverted_angle"], r["q_eq"]
        test_pert = min(r["passive_boundary"] + 0.15, 3.1)
        r_passive = run_trial(model, r["K_zero"], q_eq, pend_qpos_adr, pend_dof_adr,
                               inverted_angle, test_pert)
        print(f"\n-- friction_x={mult}, test perturbation={test_pert:.3f} rad "
              f"(passive: {'SURVIVED' if r_passive['survived'] else 'FELL'}, "
              f"final={r_passive['final_theta_err']:.4f}) --")
        best = None
        for qap, qav, qpa, qpv, rw in active_configs:
            K, q_eq2, _ = base.linearize_and_design_lqr(
                model, inverted_angle, q_arm_pos=qap, q_arm_vel=qav,
                q_pend_angle=qpa, q_pend_vel=qpv, r_weight=rw,
            )
            r_active = run_trial(model, K, q_eq2, pend_qpos_adr, pend_dof_adr,
                                  inverted_angle, test_pert)
            tag = "RESCUED" if (r_active["survived"] and not r_passive["survived"]) else (
                "same-or-worse" if r_active["survived"] == r_passive["survived"] else "WORSE (broke a passive pass)")
            print(f"   q_arm_pos={qap} q_arm_vel={qav} q_pend_angle={qpa} q_pend_vel={qpv} r={rw}: "
                  f"{'SURVIVED' if r_active['survived'] else 'FELL'} "
                  f"final={r_active['final_theta_err']:.4f} [{tag}]")
            if best is None or (r_active["survived"] and not best[0]):
                best = (r_active["survived"], (qap, qav, qpa, qpv, rw))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
