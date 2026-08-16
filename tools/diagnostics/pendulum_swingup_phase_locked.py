#!/usr/bin/env python3
"""Phase-locked continuous-resonance swing-up -- a 4th, distinct control
structure after the fixed-frequency sinusoid (pendulum_swingup_x_oscillation.py,
failed -- a pendulum's period lengthens with amplitude, so no FIXED
frequency stays resonant through a full swing-up), continuous energy-
shaping (pendulum_swingup_energy_shaping.py), and event-triggered discrete
kicks (pendulum_swingup_multi_kick.py). Ported 2026-08-12 from an earlier
branch's version onto the fully-corrected pendulum model (real 0.12m rod,
real attachment geometry, real ~0.30kg wrist mass) and the now-FIXED
find_hanging_and_inverted_angle (the old version had hanging/inverted
nearly swapped -- see pendulum_swingup_energy_shaping.py's own history).

Motivation for trying this specific strategy now: with the corrected
equilibrium reference, a fresh multi_kick search barely moved the pendulum
at all (num_kicks=1, min_theta_dist~3.03rad, essentially unmoved) --
consistent with this pendulum's documented real Coulomb stiction (a kick
decays to exactly thetadot=0.000 and locks within ~1.6s of free swing), a
regime discrete kicks-with-gaps are structurally bad at: the arm holds
still between kicks while friction eats the pendulum's momentum before it
can trigger another one. This script never lets the pendulum coast at all.

Design: drive continuously, phase-locked to a live period estimate:

    tau = t - t_last_crossing
    E = 0.5*I*thetadot^2 + m*g*r*(1 - cos(phi))          # live pendulum energy
    A = min(A_max, k_a * (E_top - E))                     # shrinks as it nears inversion
    phase = 2*pi/T_est * tau + phase_offset_bias
    target_x       = x0 + dir * A * sin(phase)
    target_x_vel   =       dir * A * omega_est * cos(phase)
    target_x_accel = -     dir * A * omega_est**2 * sin(phase)

`dir` = sign(thetadot) at the most recent zero crossing (push with the
current swing direction); `dir` and `T_est` are re-latched at EVERY zero
crossing (both directions), not held fixed for the trial.

T_est starts at the small-oscillation natural period for the (asset, pose)
pair actually in use -- `constants.t_natural_s`, measured off the compiled
model by simulation.ur5e_pendulum_compose.derive_pendulum_constants, NOT a
module-level literal (which was correct for one asset at one pose only; the
alternate long-rod asset's period is 0.898 s against this one's 0.580 s, so
seeding from the wrong pair detunes the resonant drive by ~35%) -- and the
amplitude law A = k_a*(E_top-E) is nonzero from E=0 at rest (no thetadot
factor) -- so driving begins at step 0 with no seed kick needed and no
passive coasting phase for stiction to kill.

`current_hanging_angle` (the phi=0 reference) is refreshed at every
crossing via find_nearby_equilibrium (imported from
pendulum_swingup_multi_kick.py).

4-parameter search (k_a, A_max, phase_offset_bias, crossing_debounce_s)
via scipy.optimize.differential_evolution, NOT RL (AGENTS.md).

=====================================================================
READ THIS BEFORE SPENDING MORE TIME ON THIS STRATEGY (audit 2026-08-12)
=====================================================================
Two control-law bugs were found and fixed below (the phase_offset_bias
search bound, and R_COM_M in pendulum_swingup_energy_shaping.py which
sets T_NATURAL_S here). Neither makes swing-up work, because the
BLOCKER IS THE PLANT, NOT THE LAW: with the pendulum model as currently
committed, this pendulum is OVERDAMPED and therefore has no resonance
for any phase-locked or energy-shaping law to lock onto.

  critical damping  b_crit = 2*sqrt(I*m*g*r) = 2*sqrt(2.3747e-4 *
                             0.0278696)      = 0.005145 N m s/rad
  modelled damping  b      = 0.02            (pendulum_attachment.xml's
                                              class="pendulum" default)
  damping ratio     zeta   = 3.887           (Q = 0.129)

Measured, not just computed. Released from phi=0.3 rad with the arm held
by real gravity-compensating torque, the pendulum NEVER crosses zero --
it creeps monotonically toward hanging and stiction-locks at phi=0.048
rad. With damping/frictionloss zeroed the same release oscillates 55
times with a 0.584 s period. An open-loop 6 cm sinusoidal pivot drive
swept 0.8 -> 3.0 Hz through this exact controller path produces NO
resonance peak at all as-modelled (phi_max rises monotonically with
frequency, 0.20deg -> 5.35deg, the signature of a pure forced response);
the same sweep at b=0.001 shows a real 20-36deg resonance. This is also
the true explanation of the "phi stops crossing zero" stall that
motivated the STALL_TIMEOUT_PERIODS logic below: the crossing detector
was not broken, it was correctly reporting that an overdamped system has
no crossings. The stall recovery is still worth keeping (it stops a
stale T_est being trusted forever) but it cannot manufacture a
resonance.

Energy budget at the drive amplitude this pose can actually realize
(A <= ~0.072 m, from the pose's validated 0.777 m/s peak at omega=10.83
rad/s), per full cycle at the amplitude phi=pi that inversion requires:
    input at optimal phase   pi*m*r*A*phi*omega^2 = 3.29*A  J   -> 0.24 J
    viscous loss             pi*b*omega*phi^2                -> 6.72 J
    Coulomb loss             4*tau_f*phi                     -> 0.13 J
i.e. losses exceed the maximum possible input by ~28x. Swing-up here
needs roughly 40x less joint damping AND ~10x less frictionloss before
ANY pivot-oscillation strategy can close, at which point A ~ 0.14 m is
marginally sufficient.

BOTH of those numbers (damping="0.02" frictionloss="0.01") are
explicitly labelled UNMEASURED PLACEHOLDERS in
assets/ur5e_pendulum/pendulum_attachment.xml, and they were plausible
for the OLD model: with the 0.30 m rod (I=0.003552, m*g*r=0.174) b=0.02
was zeta=0.40, genuinely underdamped, and every swing-up strategy in
this directory was designed against that. The 2026-08-11/12 rod-length
and mass corrections cut I by ~15x and m*g*r by ~6x, which moved zeta
from 0.40 to 3.887 -- and the damping placeholder was never revisited.
That is a MODEL question (what is the real hinge's damping? the rig has
an MF128ZZ ball bearing, which would not plausibly give zeta=3.9), not a
controller question, and it is deliberately NOT changed here.
"""

from __future__ import annotations

import argparse
import functools
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import differential_evolution

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from simulation.ur5e_pendulum_compose import (  # noqa: E402
    DEFAULT_PENDULUM_XML,
    PendulumConstants,
    compose_ur5e_pendulum_model,
    derive_pendulum_constants,
)
from simulation.ur5e_mujoco_torque import (  # noqa: E402
    build_initial_state_and_adapter,
    build_mujoco_state,
)
from tools.diagnostics.pendulum_swingup_energy_shaping import (  # noqa: E402
    CONFIG_PATH, ARM_Q0, CONTROL_DT, RATE_HZ,
    PendulumRunContext, add_common_pendulum_args, context_from_args,
    default_constants, describe_context, find_hanging_and_inverted_angle,
    load_config, write_output_json,
)
from tools.diagnostics.pendulum_swingup_multi_kick import find_nearby_equilibrium, _de_workers  # noqa: E402


def __getattr__(name: str):
    """T_NATURAL_S -- the small-oscillation natural period the drive seeds
    itself with -- was a module-level literal computed from the DEFAULT
    asset's hand-cached constants. It is a property of the (asset, POSE)
    pair, so as of 2026-08-13 the trial takes it from its own
    ``constants.t_natural_s`` instead. The name survives (PEP 562, lazily
    derived from the default model, cached) only because the test suite and
    the render scripts import it; it is NOT what the trial loop reads."""
    if name == "T_NATURAL_S":
        return default_constants().t_natural_s
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
MIN_CROSSINGS_FOR_LIVE_T_EST = 2  # two most recent crossings (any direction) = one half-period, doubled
# If no real crossing arrives within this many estimated periods, force a
# resync (see the stall-recovery comment at the call site) rather than trust
# an increasingly stale T_est/phase reference indefinitely. 2026-08-12 fix.
STALL_TIMEOUT_PERIODS = 1.5
# How long, after a stall-recovery reset, to ramp drive amplitude back up
# from zero rather than resuming at full A instantly (see the ramp comment
# at the call site for why the instant version tripped a guard).
STALL_RESET_RAMP_S = 0.3


def run_phase_locked_trial(
    model,
    k_a: float,
    a_max: float,
    phase_offset_bias: float,
    crossing_debounce_s: float,
    duration_s: float,
    hanging_angle: float,
    inverted_angle: float,
    enforce_guard: bool = True,
    config_path: Path | None = None,
    controller_kind: str = "impedance",
    arm_q=None,
    constants: PendulumConstants | None = None,
) -> dict:
    """enforce_guard=False: sim-only ceiling-finding mode (never on real
    hardware) -- record the first guard trip but keep simulating.

    config_path/controller_kind (added 2026-08-13): controller_kind was
    HARDCODED to "impedance" at the build_initial_state_and_adapter call
    below, so this strategy alone could not be benchmarked against another
    controller family. Both defaults reproduce the previous behavior exactly.
    arm_q/constants likewise default to the module's ARM_Q0 and the constants
    derived from `model` at it; both must be passed for a non-default
    pendulum asset, since the constants are a property of the (asset, pose)
    PAIR."""
    config = load_config() if config_path is None else load_config(config_path)
    arm_q = ARM_Q0 if arm_q is None else np.asarray(arm_q, dtype=np.float64).reshape(6)
    if constants is None:
        constants = derive_pendulum_constants(model, arm_q)
    t_natural_s = constants.t_natural_s
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
    data.qpos[pend_qpos_adr] = hanging_angle
    data.qvel[:] = 0.0
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
    x0 = float(state0.ee_pos[0])

    n_steps = int(duration_s * RATE_HZ)
    min_theta_dist = np.pi
    guard_fired = False
    guard_reason = None
    first_guard_t = None
    first_guard_reason = None
    steps_done = 0

    current_hanging_angle = hanging_angle
    crossing_ts = []
    # Separate running total, added 2026-08-12: `len(crossing_ts)` is NOT the
    # number of crossings in the trial, because the stall-recovery branch
    # below clears that list. The returned "num_crossings" was therefore only
    # ever the count since the LAST stall reset, and read as 0 for trials that
    # stalled near the end -- actively misleading for exactly the diagnosis
    # (did the phase lock ever engage?) it was added to answer.
    total_crossings = 0
    n_stall_resets = 0
    prev_phi = None

    T_est = t_natural_s
    t_last_crossing = 0.0
    dir_current = 1.0
    t_last_stall_reset = -1e9  # -inf sentinel: no ramp active until the first real reset

    for step in range(n_steps):
        t = step * CONTROL_DT
        theta = float(data.qpos[pend_qpos_adr])
        thetadot = float(data.qvel[pend_dof_adr])
        phi = float(np.mod(theta - current_hanging_angle + np.pi, 2 * np.pi) - np.pi)

        if prev_phi is not None:
            crossed = (prev_phi == 0.0) or ((prev_phi > 0.0) != (phi > 0.0))
            debounced = crossing_ts and (t - crossing_ts[-1]) < crossing_debounce_s
            if crossed and not debounced:
                crossing_ts.append(t)
                total_crossings += 1
                t_last_crossing = t
                dir_current = 1.0 if thetadot >= 0.0 else -1.0
                if len(crossing_ts) >= MIN_CROSSINGS_FOR_LIVE_T_EST:
                    live_T = 2.0 * (crossing_ts[-1] - crossing_ts[-2])
                    if live_T > 1e-6:
                        T_est = live_T
                # Anchor changed 2026-08-12 from `theta` (the pendulum's
                # CURRENT angle) to `current_hanging_angle` (the previous
                # estimate of where the equilibrium is). find_nearby_equilibrium
                # returns the qfrc_bias zero NEAREST its anchor, and this
                # pendulum's two equilibria are exactly pi apart while that
                # function's search window is +-pi/2 -- so anchoring on theta
                # means that once |phi| > pi/2 the INVERTED equilibrium is the
                # nearer one and gets silently returned as "hanging",
                # inverting the sign of phi and of every energy/trigger
                # decision downstream. Anchoring on the previous estimate
                # cannot make that jump: the true equilibrium only drifts
                # slowly, with the arm's posture, which is exactly what this
                # re-estimation exists to track. At a zero crossing phi~=0 so
                # the two anchors agree here anyway -- this is a latent-bug
                # fix for the stall branch and for the multi_kick caller (which
                # calls it at arbitrary phi), kept identical across both call
                # sites so they cannot drift apart.
                current_hanging_angle = find_nearby_equilibrium(
                    model, data.qpos[:6].copy(), pend_qpos_adr, current_hanging_angle
                )
            elif (t - t_last_crossing) > STALL_TIMEOUT_PERIODS * T_est:
                # Fixed 2026-08-12: found via an instrumented render that this
                # loop can get permanently stuck once oscillation amplitude
                # drops -- phi stops crossing zero (drifts to one side instead
                # of oscillating through it), so no new crossing EVER arrives,
                # T_est/t_last_crossing/current_hanging_angle all stay frozen
                # at whatever they were, and `phase = omega_est*(t -
                # t_last_crossing)` keeps cycling through many periods at a
                # frequency that's no longer correlated with the pendulum's
                # real state (t - t_last_crossing grows unboundedly). The
                # resulting "phase-locked" drive degrades into an open-loop
                # oscillation at a stale, uncorrelated phase -- confirmed
                # directly: a full 15s trial showed E/E_top never exceeding
                # 0.11% and phi never exceeding ~4deg despite real torque
                # (peak 45% of limit) being applied throughout, with T_est
                # frozen at 1.92s (4.5x the analytical 0.43s) from t=3s on.
                # Fix: if no real crossing has arrived within
                # STALL_TIMEOUT_PERIODS full estimated periods, don't keep
                # trusting the stale estimate -- force a resync using
                # whatever the pendulum's CURRENT state actually is, giving
                # the loop a fresh chance to re-lock instead of committing to
                # one possibly-wrong estimate for the rest of the trial.
                T_est = t_natural_s
                t_last_crossing = t
                # Also clear crossing history -- found via a second
                # instrumented run that resetting T_est alone isn't enough:
                # the NEXT real crossing after a stall still computed
                # live_T = 2*(new_crossing_t - crossing_ts[-2]), where
                # crossing_ts[-2] was an ANCIENT timestamp from before the
                # stall, producing an even worse period estimate (5.78s,
                # 5.95s -- worse than the original 1.92s bug). Without a
                # cleared history, at least 2 genuinely FRESH crossings must
                # accumulate before live_T is trusted again.
                crossing_ts.clear()
                n_stall_resets += 1
                t_last_stall_reset = t
                # Anchor the equilibrium re-estimate on the PREVIOUS estimate,
                # not on the pendulum's current theta -- see the comment at
                # the crossing-branch call site above for why.
                current_hanging_angle = find_nearby_equilibrium(
                    model, data.qpos[:6].copy(), pend_qpos_adr, current_hanging_angle
                )
        prev_phi = phi

        E = (0.5 * constants.i_pivot_kgm2 * thetadot * thetadot
             + constants.mgr_nm * (1.0 - np.cos(phi)))
        A = float(np.clip(k_a * (constants.e_top_j - E), 0.0, a_max))
        # Ramp A back up over STALL_RESET_RAMP_S after a stall-recovery reset
        # rather than resuming at full amplitude instantly. Found necessary
        # via a direct test: resetting T_est (and thus omega_est, which
        # a_cmd scales with as omega_est**2) abruptly at full amplitude
        # produced a real, sharp acceleration discontinuity that tripped the
        # joint-velocity guard within ~0.2s of the very reset meant to help
        # -- the frequency jump and the amplitude should not both be applied
        # instantaneously in the same step.
        ramp = min(1.0, (t - t_last_stall_reset) / STALL_RESET_RAMP_S)
        A *= ramp
        omega_est = 2.0 * np.pi / T_est
        phase = omega_est * (t - t_last_crossing) + phase_offset_bias
        target_x = x0 + dir_current * A * np.sin(phase)
        target_x_vel = dir_current * A * omega_est * np.cos(phase)
        a_cmd = float(-dir_current * A * omega_est * omega_est * np.sin(phase))

        state = build_mujoco_state(
            model, data, site_id=site_id, joint_ids=joint_ids,
            time_s=t, dt_s=CONTROL_DT,
            target_x=target_x, target_x_vel=target_x_vel, target_x_accel=a_cmd,
            reference_quat=state0.ee_quat, transport_axis_index=0,
            gravity_compensation=True,
        )
        tau, diag = adapter.step(state=state)
        step_safety_ok = bool(diag.get("safety_ok", True))
        if not step_safety_ok and first_guard_t is None:
            first_guard_t = t
            first_guard_reason = str(diag.get("safety_reason", ""))
        if not step_safety_ok:
            guard_fired = True
            guard_reason = first_guard_reason
            if enforce_guard:
                break

        data.ctrl[:6] = np.asarray(tau, dtype=np.float64).reshape(6)
        mujoco.mj_step(model, data)
        steps_done += 1

        dist = abs(float(np.mod(theta - inverted_angle + np.pi, 2 * np.pi) - np.pi))
        min_theta_dist = min(min_theta_dist, dist)

    return {
        "k_a": k_a, "a_max": a_max, "phase_offset_bias": phase_offset_bias,
        "crossing_debounce_s": crossing_debounce_s, "duration_s": duration_s,
        "min_theta_dist_from_inverted_rad": min_theta_dist,
        "flipped": min_theta_dist < 0.35,
        "guard_fired": guard_fired, "guard_reason": guard_reason,
        "first_guard_t": first_guard_t, "first_guard_reason": first_guard_reason,
        "steps_completed": steps_done, "num_crossings": total_crossings,
        "crossings_since_last_stall_reset": len(crossing_ts),
        "num_stall_resets": n_stall_resets,
        "final_T_est_s": T_est,
    }


def default_context() -> PendulumRunContext:
    return PendulumRunContext(
        pendulum_xml=str(DEFAULT_PENDULUM_XML),
        arm_q=tuple(float(v) for v in ARM_Q0),
        config_path=str(CONFIG_PATH),
        controller_kind="impedance",
    )


def objective(x, ctx: PendulumRunContext | None = None, duration_s: float = 8.0):
    k_a, a_max, phase_offset_bias, crossing_debounce_s = x
    if ctx is None:
        ctx = default_context()
    if ctx.constants is None:
        ctx = ctx.resolve()
    model = ctx.build_model()
    result = run_phase_locked_trial(model, k_a, a_max, phase_offset_bias, crossing_debounce_s,
                                     duration_s=duration_s, **ctx.trial_kwargs())
    cost = result["min_theta_dist_from_inverted_rad"]
    if result["guard_fired"]:
        cost += 5.0
    return cost


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase-locked continuous-resonance swing-up search.")
    add_common_pendulum_args(parser)
    parser.add_argument("--maxiter", type=int, default=30)
    parser.add_argument("--popsize", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--duration-s", type=float, default=8.0,
                        help="Trial duration used inside the search objective.")
    parser.add_argument("--final-duration-s", type=float, default=15.0,
                        help="Duration of the final re-validation trial.")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    ctx = context_from_args(args).resolve()
    print(describe_context(ctx))

    # Bounds corrected 2026-08-12 (audit) -- two of the four were unable to
    # express the physically correct answer at all:
    #
    # phase_offset_bias was (-pi/2, +pi/2), which is HALF of this parameter's
    # own natural domain -- it is a phase, so the full circle is (-pi, pi] and
    # nothing outside the old range was reachable at all. That mattered:
    # averaging this law's drive against the measured pivot->hinge coupling
    # Q = -m*r_perp*cos(phi)*a_x over one locked cycle gives
    #     <P> = 0.5 * m*r_perp * A * Phi * omega^3 * sin(phase_offset_bias),
    # i.e. net energy transfer is EXACTLY ZERO at bias=0 -- the old range's
    # centre, and the value the law reduces to nominally -- and extremal at
    # bias=+-pi/2, which were exactly the old range's endpoints, where DE
    # effectively cannot land. So the old bound put a zero-transfer point at
    # the centre of the search and both maxima on the boundary.
    # (The same coupling derivation is what fixed the SIGN of the
    # energy-shaping law in pendulum_swingup_energy_shaping.py; see its
    # a_energy comment for the work-balance measurement that validates the
    # coupling model to within 1%.)
    #
    # Widened to the full circle rather than re-centred on +pi/2, DELIBERATELY.
    # The <P> formula above is a small-angle, perfectly-locked average, and a
    # direct test showed it does NOT predict the ranking in this actual hybrid
    # loop once amplitude is large (at Phi~30deg, with the crossing-triggered
    # re-latching and stall resets both active, bias=0 and bias=-0.225 both
    # outperformed bias=+pi/2).
    #
    # a_max was (0.02, 0.28) m of PIVOT DISPLACEMENT amplitude. At the
    # corrected natural frequency (omega=10.83 rad/s, T=0.580 s) an
    # amplitude of 0.28 m demands a peak pivot speed of A*omega = 3.03 m/s
    # and a peak acceleration of A*omega^2 = 32.8 m/s^2. This pose is
    # validated to 0.777 m/s peak, i.e. A <= 0.072 m; everything above that
    # is unreachable and can only trip guards. Capped at 0.08 m, just past
    # the validated ceiling. NOTE this ceiling was derived for the DEFAULT
    # asset/pose's own omega -- re-derive it before trusting this bound at a
    # different (asset, pose) pair, whose omega differs (measured: 6.99 rad/s
    # for the long-rod asset vs 10.83 here).
    bounds = [
        (0.0, 5.0),                     # k_a
        (0.01, 0.08),                   # a_max (m) -- see note above
        (-np.pi, np.pi),                # phase_offset_bias (rad) -- see note above
        (0.1, 0.5),                     # crossing_debounce_s
    ]
    print("=== searching (k_a, a_max, phase_offset_bias, crossing_debounce_s) via differential_evolution ===")
    res = differential_evolution(
        functools.partial(objective, ctx=ctx, duration_s=args.duration_s),
        bounds, maxiter=args.maxiter, popsize=args.popsize, tol=1e-4,
        seed=args.seed, workers=_de_workers(), polish=False)
    print(f"Best params: k_a={res.x[0]:.4f}, a_max={res.x[1]:.4f}, "
          f"phase_offset_bias={res.x[2]:.4f} ({np.degrees(res.x[2]):.2f}deg), "
          f"crossing_debounce_s={res.x[3]:.4f}")
    print(f"Best cost: {res.fun:.4f}")

    model = ctx.build_model()
    best = run_phase_locked_trial(model, res.x[0], res.x[1], res.x[2], res.x[3],
                                   duration_s=args.final_duration_s, **ctx.trial_kwargs())
    print("Best candidate, re-validated:", best)
    if args.output_json:
        write_output_json(args.output_json, {
            "context": {"pendulum_xml": ctx.pendulum_xml, "arm_q": list(ctx.arm_q),
                        "config_path": ctx.config_path, "controller_kind": ctx.controller_kind,
                        "hanging_angle": ctx.hanging_angle, "inverted_angle": ctx.inverted_angle,
                        "constants": ctx.constants},
            "best_params": {"k_a": res.x[0], "a_max": res.x[1],
                            "phase_offset_bias": res.x[2], "crossing_debounce_s": res.x[3]},
            "best_cost": float(res.fun),
            "best_trial": best,
        })
        print("wrote", args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
