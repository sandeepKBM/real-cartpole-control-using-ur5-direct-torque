#!/usr/bin/env python3
"""Full swing-up -> LQR-balance pipeline for the pendulum apparatus.

Phase 1 (SWING-UP): the validated event-triggered multi-kick pumping law
from pendulum_swingup_multi_kick.py -- fire a raised-cosine cart-position
pulse every time the pendulum crosses near the bottom of its swing while
already moving, in the same direction as its current motion, pumping energy
in each cycle.

Phase 2 (BALANCE): the 4-state cartpole LQR from pendulum_lqr_cascade.py
(pole state -> desired cart acceleration -> OSC -> joint torques). The LQR
never touches data.ctrl directly.

HANDOFF: a pure SOURCE SWITCH on target_x/target_x_vel/target_x_accel, the
same three fields both phases drive the single OSC path through
(build_mujoco_state -> adapter.step). Triggers the first time the pendulum's
(phi, thetadot) enters the measured LQR capture envelope (see
--handoff-phi-deg / --handoff-thetadot-radps, sized from
pendulum_lqr_cascade.py's --envelope grid results) for
--handoff-confirm-steps consecutive control cycles in a row (debounces a
kick's own zero-crossing through the trigger band, which is not a genuine
arrival). One-way: once switched to LQR, this script never reverts to
kicking. Every OSC safety guard (ImpedanceSafetyMonitor, including the
|qd| > 3.0 rad/s joint-velocity guard) stays continuously active across the
switch -- there is exactly one torque path, never two competing ones.

Writes a full per-step trace (used by render_pendulum_flip_video.py for the
HUD video + companion graphs) and an output JSON summary.
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

from simulation.ur5e_pendulum_compose import REALROD_PENDULUM_XML  # noqa: E402
from simulation.ur5e_mujoco_torque import (  # noqa: E402
    build_initial_state_and_adapter,
    build_mujoco_state,
)
from tools.diagnostics.pendulum_swingup_energy_shaping import (  # noqa: E402
    CONTROL_DT, RATE_HZ,
    PendulumRunContext, add_common_pendulum_args, context_from_args,
    describe_context, load_config, write_output_json,
)
from tools.diagnostics.pendulum_swingup_multi_kick import find_nearby_equilibrium  # noqa: E402
from tools.diagnostics.pendulum_lqr_cascade import (  # noqa: E402
    DEFAULT_CONFIG, V_MAX_MPS, hinge_damping, linearize_cartpole, lqr_gain, wrap_pi,
)

MIN_KICK_GAP_S = 0.15
THETADOT_DEADBAND = 0.02
K_RECENTER = 3.0


def run_handoff_trial(
    model,
    *,
    kick_amplitude_m: float,
    kick_duration_s: float,
    phi_trigger_rad: float,
    K: np.ndarray,
    a_max: float,
    handoff_phi_rad: float,
    handoff_thetadot_radps: float,
    handoff_confirm_steps: int,
    duration_s: float,
    hanging_angle: float,
    inverted_angle: float,
    constants,
    config_path: Path,
    controller_kind: str,
    arm_q,
    max_kicks: int | None = None,
    on_frame=None,
    frame_stride: int = 1,
) -> dict:
    """``on_frame(step, t, data, row_dict)``, if given, is called every
    ``frame_stride`` control steps from INSIDE the single simulation loop
    (after the physics step so the rendered pose matches ``row_dict``'s own
    measurements) -- this is what render_pendulum_flip_and_balance.py uses to
    write video frames, so the video and the trace are guaranteed to come
    from the exact same deterministic rollout rather than two separate runs
    that could silently drift apart."""
    config = load_config(config_path)
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
    # x_ref (the cart position the LQR balances about) is NOT x0 -- it is set
    # to the ACTUAL measured EE x the moment mode switches to "balance"
    # (below), since swing-up's own recentering (K_RECENTER pulling
    # kick_hold_x toward x0 between kicks, MIN_KICK_GAP_S=0.15s) is not
    # guaranteed to have fully settled back to x0 by the time handoff fires.
    # Balancing around a stale x0 would silently eat into the LQR's own
    # actuation margin at the single most authority-constrained instant.
    # None until set at the handoff transition.
    x_ref = None

    n_steps = int(duration_s * RATE_HZ)
    K = np.asarray(K, dtype=np.float64).reshape(1, 4)

    target_x, target_x_vel = x0, 0.0
    kick_active = False
    kick_start_t = 0.0
    kick_sign = 1.0
    kick_hold_x = x0
    last_kick_end_t = -1e9
    num_kicks = 0
    current_hanging_angle = hanging_angle

    mode = "swingup"  # or "balance"
    handoff_t = None
    handoff_confirm_count = 0
    guard_fired = False
    guard_reason = None
    first_guard_t = None
    min_theta_dist_from_inverted = np.pi

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    trace: list[dict] = []

    for step in range(n_steps):
        t = step * CONTROL_DT
        theta = float(data.qpos[pend_qpos_adr])
        thetadot = float(data.qvel[pend_dof_adr])
        ref_for_phi = inverted_angle if mode == "balance" or theta is not None else hanging_angle

        # Two conventions used across this pipeline: phi_hang (from HANGING,
        # used by the kick trigger, matching pendulum_swingup_multi_kick.py
        # exactly) and phi_inv (from INVERTED, used by the LQR + handoff
        # test + HUD "distance from inverted"). Both are always computed;
        # which one drives the ACTIVE control law depends on `mode`.
        phi_hang = wrap_pi(theta - current_hanging_angle)
        phi_inv = wrap_pi(theta - inverted_angle)
        dist_from_inverted = abs(phi_inv)
        min_theta_dist_from_inverted = min(min_theta_dist_from_inverted, dist_from_inverted)

        if mode == "swingup":
            if kick_active and (t - kick_start_t) >= kick_duration_s:
                kick_active = False
                kick_hold_x = target_x
                last_kick_end_t = t
                current_hanging_angle = find_nearby_equilibrium(
                    model, data.qpos[:6].copy(), pend_qpos_adr, current_hanging_angle
                )

            is_bootstrap_kick = (num_kicks == 0 and step == 0)
            can_kick = max_kicks is None or num_kicks < max_kicks
            if can_kick and (is_bootstrap_kick or (
                not kick_active and abs(phi_hang) < phi_trigger_rad
                and abs(thetadot) > THETADOT_DEADBAND
                and (t - last_kick_end_t) >= MIN_KICK_GAP_S
            )):
                kick_active = True
                kick_start_t = t
                kick_sign = 1.0 if is_bootstrap_kick else (1.0 if thetadot >= 0.0 else -1.0)
                num_kicks += 1

            if kick_active:
                tau_local = t - kick_start_t
                omega_kick = 2.0 * np.pi / kick_duration_s
                target_x = kick_hold_x + kick_sign * 0.5 * kick_amplitude_m * (1.0 - np.cos(omega_kick * tau_local))
                target_x_vel = kick_sign * 0.5 * kick_amplitude_m * omega_kick * np.sin(omega_kick * tau_local)
                a_cmd = float(kick_sign * 0.5 * kick_amplitude_m * omega_kick * omega_kick * np.cos(omega_kick * tau_local))
            else:
                a_cmd = float(-K_RECENTER * (kick_hold_x - x0))
                target_x_vel = 0.0
                target_x = kick_hold_x

            # Handoff test (uses the INVERTED-referenced phi, since the LQR
            # capture envelope was measured that way): debounced by
            # requiring handoff_confirm_steps consecutive in-envelope cycles
            # so a kick's own transient pass through the trigger band near
            # phi_inv~=0 (e.g. while swinging THROUGH, not settling AT,
            # inverted on an early low-energy kick) cannot fire it.
            in_envelope = (not kick_active) and (abs(phi_inv) < handoff_phi_rad) and (abs(thetadot) < handoff_thetadot_radps)
            handoff_confirm_count = handoff_confirm_count + 1 if in_envelope else 0
            if handoff_confirm_count >= handoff_confirm_steps:
                mode = "balance"
                handoff_t = t
                # x_ref is the ACTUAL measured EE x RIGHT NOW, not x0 -- see
                # the comment where x_ref is declared above.
                x_ref = float(data.site_xpos[site_id][0])
                target_x = x_ref
                target_x_vel = 0.0  # LQR takes over the trajectory reference from here

        state = build_mujoco_state(
            model, data, site_id=site_id, joint_ids=joint_ids,
            time_s=t, dt_s=CONTROL_DT,
            target_x=target_x, target_x_vel=target_x_vel, target_x_accel=0.0,
            reference_quat=state0.ee_quat, transport_axis_index=0,
            gravity_compensation=True,
        )

        if mode == "balance":
            x_actual = float(state.ee_pos[0])
            xdot_actual = float(state.ee_lin_vel[0])
            s = np.array([x_actual - x_ref, xdot_actual, phi_inv, thetadot], dtype=np.float64)
            a_cmd = float(np.clip(-(K @ s)[0], -a_max, a_max))
            target_x_vel = float(np.clip(target_x_vel + a_cmd * CONTROL_DT, -V_MAX_MPS, V_MAX_MPS))
            target_x = target_x + target_x_vel * CONTROL_DT

        state.target_x = target_x
        state.target_x_vel = target_x_vel
        state.target_x_accel = a_cmd

        tau, diag = adapter.step(state=state)
        step_safety_ok = bool(diag.get("safety_ok", True))
        if not step_safety_ok and first_guard_t is None:
            first_guard_t = t
            guard_reason = str(diag.get("safety_reason", ""))
        if not step_safety_ok:
            guard_fired = True

        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        j6 = np.vstack([jacp[:, :6], jacr[:, :6]])
        cond_j = float(np.linalg.cond(j6))
        ee_pos = np.asarray(data.site_xpos[site_id], dtype=np.float64)
        E = 0.5 * constants.i_pivot_kgm2 * thetadot * thetadot + constants.mgr_nm * (1.0 - np.cos(phi_inv))

        trace.append({
            "step": step, "t": t, "mode": mode,
            "theta_rad": theta, "thetadot_radps": thetadot,
            "phi_inv_rad": phi_inv, "phi_inv_deg": float(np.degrees(phi_inv)),
            "dist_from_inverted_rad": dist_from_inverted,
            "E_over_Etop": float(E / constants.e_top_j),
            "cond_j": cond_j,
            "ee_x": float(ee_pos[0]), "ee_y": float(ee_pos[1]), "ee_z": float(ee_pos[2]),
            "x_dev_m": float(ee_pos[0] - x0), "y_dev_m": float(ee_pos[1] - y0), "z_dev_m": float(ee_pos[2] - z0),
            "qd_max_abs": float(np.max(np.abs(data.qvel[:6]))),
            "tau_applied_clipped": [float(v) for v in diag.get("tau_applied_clipped", [0.0] * 6)],
            "max_abs_tau": float(np.max(np.abs(diag.get("tau_applied_clipped", [0.0] * 6)))),
            "target_x": target_x, "target_x_vel": target_x_vel, "target_x_accel": a_cmd,
            "safety_ok": step_safety_ok, "safety_reason": diag.get("safety_reason"),
            "num_kicks": num_kicks,
        })

        # Safety guards are never disabled here -- any guard trip ends the
        # trial immediately, matching every other real/sim control loop in
        # this repo (AGENTS.md Sec.4). There is no escape hatch.
        if not step_safety_ok:
            break

        data.ctrl[:6] = np.asarray(tau, dtype=np.float64).reshape(6)
        mujoco.mj_step(model, data)

        if on_frame is not None and (step % frame_stride == 0):
            on_frame(step, t, data, trace[-1])

    steps_done = len(trace)
    final = trace[-1] if trace else None
    hold_ok = False
    if handoff_t is not None and not guard_fired:
        tail = [r for r in trace if r["t"] >= handoff_t + 1.0]
        if tail:
            hold_ok = all(abs(r["phi_inv_deg"]) < 15.0 for r in tail[-int(1.0 * RATE_HZ):])

    return {
        "flipped": bool(min_theta_dist_from_inverted < 0.35),
        "min_theta_dist_from_inverted_rad": min_theta_dist_from_inverted,
        "handoff_t": handoff_t, "num_kicks": num_kicks,
        "guard_fired": guard_fired, "guard_reason": guard_reason, "first_guard_t": first_guard_t,
        "steps_completed": steps_done, "n_steps": n_steps,
        "hold_survived": hold_ok,
        "final_mode": final["mode"] if final else None,
        "final_phi_inv_deg": final["phi_inv_deg"] if final else None,
        "x0": x0, "y0": y0, "z0": z0, "x_ref": x_ref,
        "trace": trace,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_pendulum_args(p, default_config=DEFAULT_CONFIG)
    p.add_argument("--kick-amplitude-m", type=float, required=True)
    p.add_argument("--kick-duration-s", type=float, required=True)
    p.add_argument("--phi-trigger-rad", type=float, required=True)
    p.add_argument("--k-gains", type=float, nargs=4, required=True, metavar=("Kx", "Kxdot", "Kphi", "Kphidot"))
    p.add_argument("--a-max", type=float, required=True)
    p.add_argument("--handoff-phi-deg", type=float, required=True)
    p.add_argument("--handoff-thetadot-radps", type=float, required=True)
    p.add_argument("--handoff-confirm-steps", type=int, default=10)
    p.add_argument("--duration-s", type=float, default=15.0)
    p.add_argument("--max-kicks", type=int, default=None)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    ctx = context_from_args(args).resolve()
    print(describe_context(ctx))
    model = ctx.build_model()
    result = run_handoff_trial(
        model,
        kick_amplitude_m=args.kick_amplitude_m, kick_duration_s=args.kick_duration_s,
        phi_trigger_rad=args.phi_trigger_rad,
        K=np.asarray(args.k_gains, dtype=np.float64), a_max=args.a_max,
        handoff_phi_rad=np.radians(args.handoff_phi_deg),
        handoff_thetadot_radps=args.handoff_thetadot_radps,
        handoff_confirm_steps=args.handoff_confirm_steps,
        duration_s=args.duration_s, hanging_angle=ctx.hanging_angle, inverted_angle=ctx.inverted_angle,
        constants=ctx.constants, config_path=Path(ctx.config_path), controller_kind=ctx.controller_kind,
        arm_q=ctx.arm_q_array, max_kicks=args.max_kicks,
    )
    print({k: v for k, v in result.items() if k != "trace"})
    if args.output_json:
        write_output_json(args.output_json, result)
        print("wrote", args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
