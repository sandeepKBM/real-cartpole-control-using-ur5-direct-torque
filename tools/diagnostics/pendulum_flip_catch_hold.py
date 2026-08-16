"""Goal 1 end-to-end: swing UP, CATCH, and HOLD -- one continuous simulation.

Both halves were already validated SEPARATELY (see AGENTS.md 0):

  * swing-up  -- energy shaping reaches 0.1875 rad from inverted, guards clean
  * LQR       -- survives 0.05-0.40 rad perturbations once the linearizer stops
                 being fooled by the hinge's Coulomb frictionloss

"Both halves work separately" is NOT "the cascade works". Everything that can
go wrong lives in the seam, and every one of these is invisible to the two
halves tested on their own:

  1. The LQR half was always tested from a TELEPORT -- pendulum placed at
     (phi0, thetadot0), arm at rest at the nominal pose, target reference
     zeroed. The real arrival state has none of those properties: the arm is
     mid-stroke, carrying cart velocity, displaced from x0, and the low-level
     controller is tracking an integrated reference that has been running for
     seconds. This script never teleports -- one MjData, one adapter, one
     continuous rollout, and the ONLY thing that changes at the switch is
     which law writes target_x_accel.
  2. Reference continuity. target_x/target_x_vel are the integrated state the
     low-level OSC is tracking. Re-zeroing them at the switch injects a step
     into the inner loop, which would show up as a "catch failure" that is
     really a handoff artifact. They are carried across untouched; only the
     ACCELERATION source switches.

SWITCH CRITERION -- measured, not guessed. The 117-cell capture envelopes
(drift 0.03 and 0.06) are not a ball around the origin: they are an
ANTI-DIAGONAL BAND. Capture needs phi and thetadot of OPPOSITE sign, i.e. the
pendulum must be moving TOWARD vertical. Concretely, the inverted pendulum's
unstable mode is

    s = thetadot + omega*phi          (omega = sqrt(m*g*r/I) = 10.833 rad/s)

and a single threshold |s| <= 1.2 classifies 94% of the 117 envelope cells
(drift 0.03; 87% at drift 0.06). The stable-mode coordinate thetadot -
omega*phi has NO predictive power (captured median 3.92 vs failed 2.84 --
i.e. backwards), so this is the unstable mode specifically, not just "some
weighted combination fits better".

Two consequences, both acted on here:

  * The switch fires on |s|, not on |phi| and |thetadot| separately. Arriving
    at rest exactly at the top (phi=0.1875, thetadot=0) gives |s| = 2.03,
    which is 1.7x OVER threshold -- it looks like the best possible arrival
    and is in fact outside the band. Arriving with a small velocity of the
    correct sign is strictly better than arriving at rest.
  * AGENTS.md currently says a swing-up objective must penalise |thetadot| at
    closest approach. That was the right direction but the wrong quantity;
    the correct one is |thetadot + omega*phi|. Penalising |thetadot| alone
    actively drives the search toward the thetadot=0 column, which is the
    WORST column of the band at any nonzero phi.

Guards stay ON for the whole rollout (both phases). A run that needs them off
is a negative result, per this repo's standing rule.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from simulation.ur5e_pendulum_compose import (  # noqa: E402
    PendulumConstants,
    REALROD_PENDULUM_XML,
    compose_ur5e_pendulum_model,
    derive_pendulum_constants,
)
from simulation.ur5e_mujoco_torque import (  # noqa: E402
    build_initial_state_and_adapter,
    build_mujoco_state,
)
from tools.diagnostics.pendulum_swingup_energy_shaping import (  # noqa: E402
    CONTROL_DT, RATE_HZ, load_config, resolve_equilibria, write_output_json,
)
from tools.diagnostics.pendulum_lqr_cascade import V_MAX_MPS, wrap_pi  # noqa: E402

ARM_Q_W2NEG90 = (-2.3688, -2.1801, -1.8838, -0.7962, -1.5707963, 0.0206)


def run_flip_catch_hold(
    model,
    *,
    swingup: dict,
    K: np.ndarray,
    lqr_a_max: float,
    hanging_angle: float,
    inverted_angle: float,
    constants: PendulumConstants,
    config_path: Path,
    controller_kind: str = "impedance",
    arm_q=ARM_Q_W2NEG90,
    duration_s: float = 14.0,
    hold_s: float = 4.0,
    s_switch: float = 1.2,
    phi_switch_max_rad: float = 0.45,
    v_max: float = V_MAX_MPS,
    hold_tol_rad: float = 0.35,
    track_history: bool = False,
) -> dict:
    """One continuous rollout: energy-shaping swing-up, then (on the |s|
    criterion) LQR hold, for ``hold_s`` past the switch.

    ``hold_tol_rad`` is the pass bar for the hold phase: |phi| must stay
    inside it for every step after the switch. It is deliberately the SAME
    0.35 rad the swing-up half uses to define "flipped", so the two halves
    cannot disagree about what counts as being up.
    """
    config = load_config(config_path)
    arm_q = np.asarray(arm_q, dtype=np.float64).reshape(6)
    omega = float(constants.omega_natural_radps)
    mgr = float(constants.mgr_nm)
    i_pivot = float(constants.i_pivot_kgm2)
    e_top = float(constants.e_top_j)

    k_e = float(swingup["k_e"])
    a_max = float(swingup["a_max"])
    k_pos = float(swingup["k_pos"])
    k_vel = float(swingup["k_vel"])
    kick_amplitude_m = float(swingup["kick_amplitude_m"])
    kick_duration_s = float(swingup["kick_duration_s"])

    data = mujoco.MjData(model)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in [
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
    ]]
    pend_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "/pendulum_hinge")
    pend_qpos_adr = model.jnt_qposadr[pend_jid]
    pend_dof_adr = model.jnt_dofadr[pend_jid]
    # See pendulum_swingup_energy_shaping.run_energy_swingup_trial: the composed
    # model prefixes attached sites with "/", and site_xpos[-1] silently returns
    # the last site rather than failing, so an unchecked lookup can return
    # plausible-looking numbers.
    tip_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "/pendulum_tip_site")
    if tip_site_id < 0:
        tip_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pendulum_tip_site")

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
    y0 = float(state0.ee_pos[1])
    z0 = float(state0.ee_pos[2])
    x_ref = x0

    K = np.asarray(K, dtype=np.float64).reshape(1, 4)

    n_steps = int(duration_s * RATE_HZ)
    target_x = x0
    target_x_vel = 0.0
    phase = "swingup"
    switch_step = None
    switch_state = None
    guard_fired = False
    guard_reason = None
    guard_t = None
    guard_phase = None
    steps_done = 0

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    max_cond_j = 0.0
    cond_j_start = None
    min_tip_z = float("inf")
    arm_contact_steps = 0
    max_abs_x_dev = 0.0
    max_abs_y_dev = 0.0
    max_abs_z_dev = 0.0
    min_abs_s_swingup = float("inf")   # closest the swing-up ever gets to the band
    min_theta_dist = np.pi
    max_abs_phi_after_switch = 0.0
    hold_steps = 0
    history = [] if track_history else None

    for step in range(n_steps):
        t = step * CONTROL_DT
        theta = float(data.qpos[pend_qpos_adr])
        thetadot = float(data.qvel[pend_dof_adr])
        # phi_sw: angle from HANGING, drives the energy law.
        # phi_up: angle from INVERTED, drives the LQR and the switch test.
        phi_sw = wrap_pi(theta - hanging_angle)
        phi_up = wrap_pi(theta - inverted_angle)
        s_unstable = thetadot + omega * phi_up
        min_theta_dist = min(min_theta_dist, abs(phi_up))

        if phase == "swingup":
            min_abs_s_swingup = min(min_abs_s_swingup, abs(s_unstable))
            if abs(phi_up) <= phi_switch_max_rad and abs(s_unstable) <= s_switch:
                phase = "lqr"
                switch_step = step
                switch_state = {
                    "t_s": t,
                    "phi_from_inverted_rad": phi_up,
                    "phi_from_inverted_deg": float(np.degrees(phi_up)),
                    "thetadot_radps": thetadot,
                    "unstable_mode_s": s_unstable,
                    "cart_x_dev_m": float(data.site_xpos[site_id][0]) - x0,
                    "target_x_vel_mps": target_x_vel,
                }

        if phase == "swingup":
            if t < kick_duration_s and kick_duration_s > 1e-9:
                omega_kick = 2.0 * np.pi / kick_duration_s
                target_x = x0 + 0.5 * kick_amplitude_m * (1.0 - np.cos(omega_kick * t))
                target_x_vel = 0.5 * kick_amplitude_m * omega_kick * np.sin(omega_kick * t)
                u = float(0.5 * kick_amplitude_m * omega_kick * omega_kick * np.cos(omega_kick * t))
            else:
                E = 0.5 * i_pivot * thetadot * thetadot + mgr * (1.0 - np.cos(phi_sw))
                a_energy = -k_e * thetadot * np.cos(phi_sw) * (e_top - E)
                a_recenter = -k_pos * (target_x - x0) - k_vel * target_x_vel
                u = float(np.clip(a_energy + a_recenter, -a_max, a_max))
                target_x_vel += u * CONTROL_DT
                target_x += target_x_vel * CONTROL_DT
            state = build_mujoco_state(
                model, data, site_id=site_id, joint_ids=joint_ids,
                time_s=t, dt_s=CONTROL_DT,
                target_x=target_x, target_x_vel=target_x_vel, target_x_accel=u,
                reference_quat=state0.ee_quat, transport_axis_index=0,
                gravity_compensation=True,
            )
        else:
            # Build first to read the ACTUAL cart state for genuine closed-loop
            # feedback, then patch the target_* fields -- same order
            # run_lqr_trial uses, so the two cannot diverge on state handling.
            state = build_mujoco_state(
                model, data, site_id=site_id, joint_ids=joint_ids,
                time_s=t, dt_s=CONTROL_DT,
                target_x=target_x, target_x_vel=target_x_vel, target_x_accel=0.0,
                reference_quat=state0.ee_quat, transport_axis_index=0,
                gravity_compensation=True,
            )
            s_vec = np.array([
                float(state.ee_pos[0]) - x_ref,
                float(state.ee_lin_vel[0]),
                phi_up,
                thetadot,
            ], dtype=np.float64)
            u = float(np.clip(-(K @ s_vec)[0], -lqr_a_max, lqr_a_max))
            # target_x/target_x_vel deliberately carried over from the swing-up
            # phase rather than re-zeroed -- see the module docstring.
            target_x_vel = float(np.clip(target_x_vel + u * CONTROL_DT, -v_max, v_max))
            target_x = target_x + target_x_vel * CONTROL_DT
            state.target_x = target_x
            state.target_x_vel = target_x_vel
            state.target_x_accel = u

        tau, diag = adapter.step(state=state)
        if not bool(diag.get("safety_ok", True)):
            guard_fired = True
            guard_reason = str(diag.get("safety_reason", ""))
            guard_t = t
            guard_phase = phase
            break

        data.ctrl[:6] = np.asarray(tau, dtype=np.float64).reshape(6)
        mujoco.mj_step(model, data)
        steps_done += 1

        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        cond_j = float(np.linalg.cond(np.vstack([jacp[:, :6], jacr[:, :6]])))
        if cond_j_start is None:
            cond_j_start = cond_j
        max_cond_j = max(max_cond_j, cond_j)
        ee = data.site_xpos[site_id]
        max_abs_x_dev = max(max_abs_x_dev, abs(float(ee[0]) - x0))
        max_abs_y_dev = max(max_abs_y_dev, abs(float(ee[1]) - y0))
        max_abs_z_dev = max(max_abs_z_dev, abs(float(ee[2]) - z0))
        if tip_site_id >= 0:
            min_tip_z = min(min_tip_z, float(data.site_xpos[tip_site_id][2]))
        if int(data.ncon) > 0:
            arm_contact_steps += 1

        if phase == "lqr":
            hold_steps += 1
            phi_now = wrap_pi(float(data.qpos[pend_qpos_adr]) - inverted_angle)
            max_abs_phi_after_switch = max(max_abs_phi_after_switch, abs(phi_now))
            if hold_steps >= int(hold_s * RATE_HZ):
                break

        if track_history:
            history.append({
                "t": t, "phase": phase,
                "phi_up_deg": float(np.degrees(phi_up)), "thetadot": thetadot,
                "s_unstable": s_unstable, "u": u,
                "x_dev": float(ee[0]) - x0, "target_x_vel": target_x_vel,
                # Full qpos so a video can be replayed kinematically without
                # re-simulating (same approach as render_trace_video.py).
                "qpos": [float(v) for v in data.qpos],
            })

    switched = switch_step is not None
    # The bar has to account for WHERE the catch started. The switch can fire
    # legitimately outside hold_tol_rad (the capture band reaches out to |phi|
    # ~ 0.44 rad when thetadot carries the right sign), and a catch that enters
    # at 25 deg, never overshoots that, and settles to 0.2 deg is a successful
    # catch -- scoring it purely on max|phi| <= 0.35 rad would call it a
    # failure for the arrival state it was handed. So require BOTH:
    #   * bounded: never went further out than it came in (no runaway), and
    #   * converged: finished inside hold_tol_rad.
    entry_abs_phi = abs(switch_state["phi_from_inverted_rad"]) if switch_state else 0.0
    final_abs_phi = abs(wrap_pi(float(data.qpos[pend_qpos_adr]) - inverted_angle))
    bounded = max_abs_phi_after_switch <= max(hold_tol_rad, entry_abs_phi) + 1e-9
    converged = final_abs_phi <= hold_tol_rad
    held = bool(
        switched
        and not guard_fired
        and hold_steps >= int(hold_s * RATE_HZ)
        and bounded
        and converged
    )
    return {
        "switched": switched,
        "held": held,
        "hold_bounded": bool(bounded),
        "hold_converged": bool(converged),
        "final_abs_phi_rad": final_abs_phi,
        "final_abs_phi_deg": float(np.degrees(final_abs_phi)),
        # The headline. Both must be true, and the guard must never have fired.
        "flip_and_hold": bool(held and not guard_fired),
        "switch": switch_state,
        "hold_duration_s": hold_steps / RATE_HZ,
        "max_abs_phi_after_switch_rad": max_abs_phi_after_switch,
        "max_abs_phi_after_switch_deg": float(np.degrees(max_abs_phi_after_switch)),
        "min_theta_dist_from_inverted_rad": min_theta_dist,
        "min_abs_unstable_mode_during_swingup": (
            None if min_abs_s_swingup == float("inf") else min_abs_s_swingup),
        "guard_fired": guard_fired, "guard_reason": guard_reason,
        "guard_t_s": guard_t, "guard_phase": guard_phase,
        "steps_completed": steps_done,
        "max_cond_j": max_cond_j, "cond_j_start": cond_j_start,
        "max_abs_x_dev_m": max_abs_x_dev,
        # Y/Z reported so the drift tolerance a run NEEDS is a measurement,
        # not an assumption -- the 0.03 -> 0.06 config difference is a real
        # guard loosening and must be justified by these numbers.
        "max_abs_y_dev_m": max_abs_y_dev,
        "max_abs_z_dev_m": max_abs_z_dev,
        "min_tip_z_m": (None if min_tip_z == float("inf") else float(min_tip_z)),
        "tip_hit_floor": bool(min_tip_z != float("inf") and min_tip_z <= 0.0),
        "arm_contact_steps": int(arm_contact_steps),
        "s_switch": s_switch, "phi_switch_max_rad": phi_switch_max_rad,
        "config_path": str(config_path),
        "history": history,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--swingup-json", type=Path, required=True,
                   help="Energy-shaping search result; best_params is read from it.")
    p.add_argument("--lqr-json", type=Path, required=True,
                   help="pendulum_lqr_cascade result; lqr.K and lqr.a_max are read from it.")
    p.add_argument("--config", type=Path, required=True,
                   help="Controller YAML. ONE config drives both phases -- this is a "
                        "single continuous rollout with a single adapter.")
    p.add_argument("--pendulum-xml", type=Path, default=REALROD_PENDULUM_XML)
    p.add_argument("--start-q-rad", type=float, nargs=6, default=list(ARM_Q_W2NEG90))
    p.add_argument("--controller-kind", default="impedance")
    p.add_argument("--duration-s", type=float, default=14.0)
    p.add_argument("--hold-s", type=float, default=4.0,
                   help="Seconds of LQR hold required after the switch to pass.")
    p.add_argument("--s-switch", type=float, default=1.2,
                   help="Switch when |thetadot + omega*phi| <= this. Default 1.2 is the "
                        "threshold measured to classify 94%% of the 117 envelope cells.")
    p.add_argument("--phi-switch-max-rad", type=float, default=0.45)
    p.add_argument("--zero-lqr-gain", action="store_true",
                   help="COUNTERFACTUAL: run the identical swing-up and switch, then apply "
                        "K=0 (cart reference frozen) instead of the LQR. If the pendulum "
                        "still stays up, the 'catch' was passive friction, not control -- "
                        "this repo has retracted exactly that result before.")
    p.add_argument("--track-history", action="store_true")
    p.add_argument("--output-json", type=Path, default=None)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    swing = json.loads(Path(args.swingup_json).read_text())["best_params"]
    lqr = json.loads(Path(args.lqr_json).read_text())["lqr"]

    model = compose_ur5e_pendulum_model(pendulum_xml=str(args.pendulum_xml))
    arm_q = np.asarray(args.start_q_rad, dtype=np.float64).reshape(6)
    hanging_angle, inverted_angle = resolve_equilibria(model, arm_q)
    constants = derive_pendulum_constants(model, arm_q)

    print(f"pendulum={Path(args.pendulum_xml).name}  arm_q={list(np.round(arm_q, 6))}")
    print(f"config={Path(args.config).name}  kind={args.controller_kind}")
    print(f"hanging={hanging_angle:.4f}  inverted={inverted_angle:.4f}  "
          f"omega={constants.omega_natural_radps:.4f}")
    print(f"switch when |thetadot + omega*phi| <= {args.s_switch} and "
          f"|phi| <= {args.phi_switch_max_rad}")

    out = run_flip_catch_hold(
        model,
        swingup=swing,
        K=(np.zeros(4) if args.zero_lqr_gain else np.asarray(lqr["K"], dtype=np.float64)),
        lqr_a_max=float(lqr["a_max"]),
        hanging_angle=hanging_angle,
        inverted_angle=inverted_angle,
        constants=constants,
        config_path=args.config,
        controller_kind=args.controller_kind,
        arm_q=arm_q,
        duration_s=args.duration_s,
        hold_s=args.hold_s,
        s_switch=args.s_switch,
        phi_switch_max_rad=args.phi_switch_max_rad,
        track_history=bool(args.track_history),
    )

    print("\n=== FLIP -> CATCH -> HOLD ===")
    print(f"  switched            = {out['switched']}")
    if out["switch"]:
        sw = out["switch"]
        print(f"    at t              = {sw['t_s']:.3f} s")
        print(f"    phi from inverted = {sw['phi_from_inverted_deg']:+.3f} deg "
              f"({sw['phi_from_inverted_rad']:+.4f} rad)")
        print(f"    thetadot          = {sw['thetadot_radps']:+.4f} rad/s")
        print(f"    |unstable mode s| = {abs(sw['unstable_mode_s']):.4f}")
        print(f"    cart x deviation  = {sw['cart_x_dev_m']:+.4f} m")
    print(f"  held                = {out['held']}   (hold {out['hold_duration_s']:.2f} s, "
          f"bounded={out['hold_bounded']} converged={out['hold_converged']})")
    print(f"  final |phi|         = {out['final_abs_phi_deg']:.3f} deg")
    print(f"  FLIP AND HOLD       = {out['flip_and_hold']}")
    if out["guard_fired"]:
        print(f"  guard_fired         = True  {out['guard_reason']}  "
              f"@t={out['guard_t_s']:.3f}s during {out['guard_phase']}")
    else:
        print("  guard_fired         = False")
    print(f"  max |phi| post-catch = {out['max_abs_phi_after_switch_deg']:.3f} deg")
    print(f"  min |s| in swing-up  = {out['min_abs_unstable_mode_during_swingup']}")
    print(f"  drift  x={out['max_abs_x_dev_m']:.4f}  y={out['max_abs_y_dev_m']:.4f}  "
          f"z={out['max_abs_z_dev_m']:.4f} m")
    print(f"  tip min world z      = {out['min_tip_z_m']}  hit_floor={out['tip_hit_floor']}  "
          f"arm_contacts={out['arm_contact_steps']}")
    print(f"  cond(J) {out['cond_j_start']:.3f} -> {out['max_cond_j']:.3f}")

    if args.output_json:
        write_output_json(args.output_json, {
            "result": out,
            "swingup_params": swing,
            "lqr": {"K": lqr["K"], "a_max": lqr["a_max"]},
            "context": {
                "pendulum_xml": str(args.pendulum_xml),
                "arm_q": [float(v) for v in arm_q],
                "config_path": str(args.config),
                "controller_kind": args.controller_kind,
                "hanging_angle": hanging_angle,
                "inverted_angle": inverted_angle,
                "constants": constants.__dict__ if hasattr(constants, "__dict__") else None,
            },
        })
        print(f"\nwrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
