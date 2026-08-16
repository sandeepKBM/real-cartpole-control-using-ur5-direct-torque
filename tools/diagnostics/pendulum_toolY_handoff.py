#!/usr/bin/env python3
"""Deliverable 5: swing-up -> LQR handoff, guards enforced continuously
across the switch.

Two phases in ONE closed-loop rollout, both driving the OSC through the
exact same target_x/target_x_vel/target_x_accel interface (rotated tool-Y
frame, pendulum_toolY_common.py):

  1. SWING-UP: energy-shaping control (tools/diagnostics/
     pendulum_toolY_swingup_search.py::run_energy_shaping_trial's law,
     reimplemented inline here so both phases share one continuous MuJoCo
     rollout instead of two separate ones stitched together).
  2. BALANCE: 4-state cartpole LQR (pendulum_toolY_lqr.py), engaged the
     instant phi enters a configurable capture window
     (--handoff-phi-deg / --handoff-thetadot-radps).

x_ref FIX (named defect in the task spec): the LQR's x_ref -- the reference
the "x - x_ref" state component is measured against -- is captured AT THE
INSTANT of handoff (the cart's actual position when mode switches to
"balance"), NOT frozen at trial start. A predecessor script's own comment
claimed this but the code captured x_ref at t=0 regardless. Get this wrong
and the LQR spends its early balance-phase authority fighting a stale
position error left over from the swing-up cart excursion instead of
regulating the pendulum.

Guards stay enforced (ImpedanceSafetyMonitor via the adapter, both phases,
never bypassed) -- there is no `--no-enforce-guard` flag, per this task's
explicit standing instruction.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.diagnostics import pendulum_toolY_common as C  # noqa: E402
from tools.diagnostics.pendulum_toolY_lqr import (  # noqa: E402
    cart_coupling_report, hinge_damping, linearize_cartpole, lqr_gain, V_MAX_MPS,
)


def run_handoff_trial(
    model: mujoco.MjModel,
    *,
    # swing-up (energy-shaping) params
    k_e: float, k_pos: float, a_max_swingup: float, seed_A: float, seed_T: float,
    k_vel: float = 0.0,
    # LQR params
    K: np.ndarray, a_max_balance: float,
    # handoff trigger
    handoff_phi_deg: float = 25.0, handoff_thetadot_radps: float = 3.0,
    swingup_timeout_s: float = 6.0, balance_duration_s: float = 6.0,
    v_max: float = V_MAX_MPS,
    enforce_guard: bool = True,
    track_history: bool = True,
) -> dict:
    data = mujoco.MjData(model)
    data.qpos[:6] = C.ARM_Q0
    data.qvel[:] = 0.0
    pend_jid, hub_bid, site_id = C.hinge_ids(model)
    joint_ids = C.joint_ids_for(model)
    hanging, inverted = C.resolve_equilibria(model)
    data.qpos[model.jnt_qposadr[pend_jid]] = hanging
    pend_qpos_adr = model.jnt_qposadr[pend_jid]
    pend_dof_adr = model.jnt_dofadr[pend_jid]
    mujoco.mj_forward(model, data)

    const = C.default_constants()
    mgr, i_pivot, e_top = const.mgr_nm, const.i_pivot_kgm2, const.e_top_j

    tool_x, tool_y, tool_z = C.tool_frame_world(data, site_id)
    R = C.pumping_rotation_matrix(tool_x, tool_y, tool_z)

    config = C.load_config()
    state0, adapter = C.build_rotated_initial_state_and_adapter(
        model, data, site_id, joint_ids, R=R,
        controller_cfg=config["controller"],
        gravity_mode=config["mujoco"].get("gravity_mode", "gravity_comp"),
    )
    x_ref_swingup = float(state0.ee_pos[0])
    reference_quat = state0.reference_quat if state0.reference_quat is not None else state0.ee_quat

    K = np.asarray(K, dtype=np.float64).reshape(1, 4)
    handoff_phi_rad = np.radians(handoff_phi_deg)

    mode = "swingup"
    x_ref_balance = None  # SET AT THE HANDOFF INSTANT -- the named fix.
    handoff_t = None
    target_x = x_ref_swingup
    target_x_vel = 0.0
    guard_fired = False
    guard_reason = None
    first_guard_t = None
    steps_done = 0
    captured_final = False
    history = [] if track_history else None

    n_steps = int((swingup_timeout_s + balance_duration_s) * C.RATE_HZ)
    for step in range(n_steps):
        t = step * C.CONTROL_DT
        theta = float(data.qpos[pend_qpos_adr])
        thetadot = float(data.qvel[pend_dof_adr])
        phi_from_hanging = C.wrap_pi(theta - hanging)
        phi_from_inverted = C.wrap_pi(theta - inverted)

        if mode == "swingup":
            if (abs(phi_from_inverted) < handoff_phi_rad
                    and abs(thetadot) < handoff_thetadot_radps
                    and t > seed_T):
                mode = "balance"
                handoff_t = t
                x_ref_balance = target_x  # <-- captured AT HANDOFF, not at t=0
            elif t > swingup_timeout_s:
                # Swing-up failed to reach the capture window in time; stay
                # in swing-up (reported as a failed handoff, not silently
                # switched with a bad trigger).
                pass

        if mode == "swingup":
            E = 0.5 * i_pivot * thetadot ** 2 + mgr * (1.0 - np.cos(phi_from_hanging))
            x_rel = target_x - x_ref_swingup
            if t < seed_T:
                u = float(seed_A * np.sin(np.pi * t / seed_T))
            else:
                u = (
                    -k_e * thetadot * np.cos(phi_from_hanging) * (e_top - E)
                    - k_pos * x_rel
                    - k_vel * target_x_vel
                )
            u = float(np.clip(u, -a_max_swingup, a_max_swingup))
        else:
            state_probe, _ = C.build_step_state(
                model, data, site_id=site_id, joint_ids=joint_ids, time_s=t,
                target_x=target_x, target_x_vel=target_x_vel, target_x_accel=0.0,
                reference_quat=reference_quat, R=R,
            )
            x_actual = float(state_probe.ee_pos[0])
            xdot_actual = float(state_probe.ee_lin_vel[0])
            s = np.array([x_actual - x_ref_balance, xdot_actual, phi_from_inverted, thetadot],
                          dtype=np.float64)
            u = float(np.clip(-(K @ s)[0], -a_max_balance, a_max_balance))

        target_x_vel = float(np.clip(target_x_vel + u * C.CONTROL_DT, -v_max, v_max))
        target_x = target_x + target_x_vel * C.CONTROL_DT

        state, ee_world = C.build_step_state(
            model, data, site_id=site_id, joint_ids=joint_ids, time_s=t,
            target_x=target_x, target_x_vel=target_x_vel, target_x_accel=u,
            reference_quat=reference_quat, R=R,
        )
        tau, diag = adapter.step(state=state)
        if not diag["safety_ok"] and not guard_fired:
            guard_fired = True
            guard_reason = diag["safety_reason"]
            first_guard_t = t
            if enforce_guard:
                break
        data.ctrl[:6] = tau
        mujoco.mj_step(model, data)
        steps_done += 1

        if track_history:
            history.append({
                "t": t, "mode": mode, "phi_deg": float(np.degrees(phi_from_inverted)),
                "thetadot": thetadot, "u": u, "target_x": target_x, "target_x_vel": target_x_vel,
                "ee_world": ee_world.tolist(), "safety_ok": bool(diag["safety_ok"]),
                "qd_max_abs": float(np.max(np.abs(data.qvel[:6]))),
                "orientation_error_rad": float(diag["orientation_error_norm"]),
            })

    if mode == "balance" and steps_done == n_steps and not guard_fired:
        final_phi = C.wrap_pi(float(data.qpos[pend_qpos_adr]) - inverted)
        final_thetadot = float(data.qvel[pend_dof_adr])
        captured_final = abs(final_phi) < 0.12 and abs(final_thetadot) < 1.0

    return {
        "handoff_t": handoff_t, "handoff_mode_reached": mode == "balance",
        "captured_final": captured_final,
        "guard_fired": guard_fired, "guard_reason": guard_reason, "first_guard_t": first_guard_t,
        "steps_done": steps_done, "n_steps": n_steps,
        "final_phi_deg": float(np.degrees(C.wrap_pi(float(data.qpos[pend_qpos_adr]) - inverted))),
        "final_thetadot": float(data.qvel[pend_dof_adr]),
        "inverted_angle": inverted, "hanging_angle": hanging,
        "history": history,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--k-e", type=float, default=0.0279)
    p.add_argument("--k-pos", type=float, default=49.4788)
    p.add_argument("--a-max-swingup", type=float, default=9.9304)
    p.add_argument("--seed-a", type=float, default=6.3091)
    p.add_argument("--seed-t", type=float, default=0.2965)
    p.add_argument("--k-vel", type=float, default=0.0)
    p.add_argument("--lqr-gains-json", type=str, default=None,
                    help="Path to a pendulum_toolY_lqr.py --output-json result; "
                         "reads K and a_max from it.")
    p.add_argument("--k-gains", type=float, nargs=4, default=None,
                    help="Explicit K = [k_x, k_xdot, k_phi, k_phidot] (overrides --lqr-gains-json).")
    p.add_argument("--a-max-balance", type=float, default=3.0)
    p.add_argument("--handoff-phi-deg", type=float, default=25.0)
    p.add_argument("--handoff-thetadot-radps", type=float, default=3.0)
    p.add_argument("--swingup-timeout-s", type=float, default=6.0)
    p.add_argument("--balance-duration-s", type=float, default=6.0)
    p.add_argument("--output-json", type=str, default=None)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    model = C.build_model()

    if args.k_gains is not None:
        K = np.asarray(args.k_gains, dtype=np.float64).reshape(1, 4)
        a_max_balance = args.a_max_balance
    elif args.lqr_gains_json is not None:
        with Path(args.lqr_gains_json).open() as fp:
            d = json.load(fp)
        K = np.asarray(d["lqr"]["K"], dtype=np.float64).reshape(1, 4)
        a_max_balance = float(d["lqr"]["a_max"])
    else:
        raise SystemExit("must supply --k-gains or --lqr-gains-json")

    print("K =", K.reshape(-1), "a_max_balance =", a_max_balance)
    res = run_handoff_trial(
        model, k_e=args.k_e, k_pos=args.k_pos, a_max_swingup=args.a_max_swingup,
        seed_A=args.seed_a, seed_T=args.seed_t, k_vel=args.k_vel,
        K=K, a_max_balance=a_max_balance,
        handoff_phi_deg=args.handoff_phi_deg, handoff_thetadot_radps=args.handoff_thetadot_radps,
        swingup_timeout_s=args.swingup_timeout_s, balance_duration_s=args.balance_duration_s,
    )
    print(f"handoff_mode_reached={res['handoff_mode_reached']} handoff_t={res['handoff_t']} "
          f"captured_final={res['captured_final']} guard_fired={res['guard_fired']} "
          f"reason={res['guard_reason']} final_phi_deg={res['final_phi_deg']:.2f} "
          f"final_thetadot={res['final_thetadot']:.3f} steps={res['steps_done']}/{res['n_steps']}")
    if args.output_json:
        with Path(args.output_json).open("w") as fp:
            json.dump(res, fp, indent=2, default=str)
        print("wrote", args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
