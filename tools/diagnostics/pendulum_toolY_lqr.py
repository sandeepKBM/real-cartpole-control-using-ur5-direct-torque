#!/usr/bin/env python3
"""Deliverable 4: 4-state cartpole LQR (HIGH-LEVEL) cascaded onto the
torque-lane OSC (LOW-LEVEL), balancing the pendulum about its inverted
equilibrium, pumping along TOOL Y (see pendulum_toolY_common.py).

ARCHITECTURE (per task spec, do not deviate):

    pole state --> [LQR] --> desired CART ACCELERATION along tool Y
                --> [OSC, unmodified controller_core] --> joint torques

The LQR never touches data.ctrl or joint torques directly. State
s = [x - x_ref, xdot, phi, phidot] (x = EE position along the pumping
direction, in the ROTATED frame pendulum_toolY_common.py sets up; phi =
pendulum angle from the INVERTED equilibrium). u = desired cart
acceleration, integrated into target_x/target_x_vel/target_x_accel -- the
same interface every other script in this pipeline drives the OSC through.

kappa is NEVER hard-coded: B's cart-coupling row is
``measure_cart_coupling_nm_per_mps2(model, arm_q, inverted_angle,
direction_world=tool_y)`` -- the exact rigid-body pseudo-force cross-product,
evaluated fresh off the compiled model at the true inverted equilibrium
angle (not the idealized "amplitude" figure quoted in the task brief, which
is the same physical quantity evaluated as if phi_inverted were exactly 0;
this script's own probe below reports both so the difference -- expected
and small, since the true inverted angle is 7.28 deg, not 0 -- is visible
rather than hidden).

Linearization:
    I_pivot * phiddot = mgr_nm*phi + Q_per_a*u - damping*phidot
    xddot = u   (OSC assumed high-bandwidth over the LQR's own timescale --
                 same assumption every other script here makes; see
                 run_lqr_trial's own note if it matters for a result)

    A = [[0,1,0,0],[0,0,0,0],[0,0,0,1],[0,0,omega^2,-damping/I_pivot]]
    B = [[0],[1],[0],[Q_per_a/I_pivot]]
    omega^2 = mgr_nm/I_pivot

K = -solve_continuous_are(A,B,Q,R)-based full-state feedback. u is clipped
to +-a_max; the reference velocity is additionally clipped to +-v_max
(this task's own measured cart-speed authority at ARM_Q0 pumping along
tool Y -- see pendulum_toolY_speed_sweep.py's tradeoff curve; the "orientation
free" ceiling found there, ~0.20 m/s, not the world-X figure quoted in the
task brief, which does not apply to this direction) -- any candidate K that
needs more than that shows up directly as a failed capture or a guard trip
in closed-loop simulation, not as hidden linear-model optimism.

Every claimed capture is checked against the K=0 counterfactual over the
SAME initial condition (capture_envelope_grid) -- a captured cell whose
K=0 run ALSO captures means the "capture" is free (passive friction/
settling), not the LQR, and is flagged, not reported as a real result.
"""

from __future__ import annotations

import argparse
import functools
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.linalg import solve_continuous_are
from scipy.optimize import differential_evolution

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.diagnostics import pendulum_toolY_common as C  # noqa: E402

# Measured cart-speed authority at ARM_Q0, pumping along TOOL Y, orientation
# guard relaxed (pendulum_toolY_speed_sweep.py): binding guard past
# orientation~0.15rad is the tool-X-role drift guard at ~0.1997 m/s. A fixed
# 5% margin below that, not searched -- reject any LQR gain set that NEEDS
# more than real authority, don't quietly raise the ceiling to make a
# candidate look feasible.
V_MAX_MPS = 0.19

DEFAULT_CONFIG = C.CONFIG_PATH


def _de_workers() -> int:
    try:
        mp.set_start_method("fork")
    except RuntimeError:
        pass
    env = os.environ.get("DE_WORKERS")
    if env:
        return max(1, int(env))
    return max(1, int((os.cpu_count() or 2) * 0.9))


def cart_coupling_report(model: mujoco.MjModel, arm_q: np.ndarray, inverted_angle: float,
                          direction_world: np.ndarray, constants) -> dict:
    """Q_per_a at the TRUE inverted angle (used for B) alongside the
    idealized amplitude figure (sin(angle(hinge_axis, direction)) *
    mgr/g, i.e. what the task brief calls kappa) -- reported together so
    the difference is visible, not silently substituted."""
    pend_jid, hub_bid, site_id = C.hinge_ids(model)
    data = mujoco.MjData(model)
    data.qpos[:6] = arm_q
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    axis = C.hinge_axis_world(model, data, pend_jid, hub_bid)
    d = direction_world / np.linalg.norm(direction_world)
    kappa_amplitude = float(np.linalg.norm(np.cross(axis, d)))
    Q_per_a = C.measure_cart_coupling_nm_per_mps2(model, arm_q, inverted_angle, direction_world)
    kappa_at_equilibrium = Q_per_a / (constants.mgr_nm / constants.g)
    return {
        "Q_per_a_nm_per_mps2": Q_per_a,
        "kappa_amplitude": kappa_amplitude,
        "kappa_at_true_inverted_angle": kappa_at_equilibrium,
        "inverted_angle_rad": inverted_angle,
        "inverted_angle_deg": float(np.degrees(inverted_angle)),
    }


def linearize_cartpole(constants, damping: float, Q_per_a: float) -> tuple[np.ndarray, np.ndarray]:
    omega2 = constants.mgr_nm / constants.i_pivot_kgm2
    b32 = Q_per_a / constants.i_pivot_kgm2
    A = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, omega2, -damping / constants.i_pivot_kgm2],
    ], dtype=np.float64)
    B = np.array([[0.0], [1.0], [0.0], [b32]], dtype=np.float64)
    return A, B


def lqr_gain(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)
    return K, P


def hinge_damping(model: mujoco.MjModel) -> float:
    pend_jid, _, _ = C.hinge_ids(model)
    return float(model.dof_damping[model.jnt_dofadr[pend_jid]])


def run_lqr_trial(
    model: mujoco.MjModel,
    K: np.ndarray,
    *,
    duration_s: float,
    inverted_angle: float,
    hanging_angle: float,
    initial_phi_rad: float,
    initial_thetadot_radps: float,
    a_max: float = 3.0,
    v_max: float = V_MAX_MPS,
    enforce_guard: bool = True,
    track_history: bool = False,
    config_path: Path | None = None,
    capture_tol_rad: float = 0.12,
    capture_tol_radps: float = 1.0,
    capture_window_s: float = 1.5,
) -> dict:
    """BANDWIDTH CAVEAT (kept from the predecessor, re-verified applicable
    here): the "xddot=u" assumption is only approximately true -- the OSC's
    real closed-loop X-tracking lags a commanded acceleration by roughly
    100-200ms, comparable to the pendulum's own ~1/omega=92ms instability
    time constant. Large hand-picked K (big Q_phi, small a_max) can diverge
    even from a correctly-derived linear model for this reason; only
    fairly large Q_phi/Q_phidot with a generous a_max reliably captures a
    real perturbation -- found by direct closed-loop search, not assumed
    from the linear model alone."""
    config = C.load_config(config_path) if config_path is not None else C.load_config()
    data = mujoco.MjData(model)
    data.qpos[:6] = C.ARM_Q0
    data.qvel[:] = 0.0
    pend_jid, hub_bid, site_id = C.hinge_ids(model)
    joint_ids = C.joint_ids_for(model)
    pend_qpos_adr = model.jnt_qposadr[pend_jid]
    pend_dof_adr = model.jnt_dofadr[pend_jid]

    data.qpos[pend_qpos_adr] = C.wrap_pi(inverted_angle + initial_phi_rad)
    data.qvel[:] = 0.0
    data.qvel[pend_dof_adr] = initial_thetadot_radps
    mujoco.mj_forward(model, data)

    tool_x, tool_y, tool_z = C.tool_frame_world(data, site_id)
    R = C.pumping_rotation_matrix(tool_x, tool_y, tool_z)

    state0, adapter = C.build_rotated_initial_state_and_adapter(
        model, data, site_id, joint_ids, R=R,
        controller_cfg=config["controller"],
        gravity_mode=config["mujoco"].get("gravity_mode", "gravity_comp"),
    )
    x_ref = float(state0.ee_pos[0])
    reference_quat = state0.reference_quat if state0.reference_quat is not None else state0.ee_quat

    n_steps = int(duration_s * C.RATE_HZ)
    target_x = x_ref
    target_x_vel = 0.0
    guard_fired = False
    guard_reason = None
    first_guard_t = None
    steps_done = 0
    peak_abs_u = 0.0
    peak_abs_target_x_vel = 0.0
    peak_abs_qd = 0.0
    min_capture_window_start = max(0, n_steps - int(capture_window_s * C.RATE_HZ))
    within_tol_from = None
    history = [] if track_history else None

    K = np.asarray(K, dtype=np.float64).reshape(1, 4)

    for step in range(n_steps):
        t = step * C.CONTROL_DT
        theta = float(data.qpos[pend_qpos_adr])
        thetadot = float(data.qvel[pend_dof_adr])
        phi = C.wrap_pi(theta - inverted_angle)

        state, ee_world = C.build_step_state(
            model, data, site_id=site_id, joint_ids=joint_ids, time_s=t,
            target_x=target_x, target_x_vel=target_x_vel, target_x_accel=0.0,
            reference_quat=reference_quat, R=R,
        )
        x_actual = float(state.ee_pos[0])
        xdot_actual = float(state.ee_lin_vel[0])

        s = np.array([x_actual - x_ref, xdot_actual, phi, thetadot], dtype=np.float64)
        u = float(np.clip(-(K @ s)[0], -a_max, a_max))
        peak_abs_u = max(peak_abs_u, abs(u))

        target_x_vel = float(np.clip(target_x_vel + u * C.CONTROL_DT, -v_max, v_max))
        target_x = target_x + target_x_vel * C.CONTROL_DT
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
            within_tol_from = False
        elif within_tol_from is None and step >= min_capture_window_start:
            within_tol_from = True

        if track_history:
            history.append({
                "t": t, "phi_deg": float(np.degrees(phi)), "thetadot": thetadot,
                "x": x_actual, "xdot": xdot_actual, "u": u,
                "target_x": target_x, "target_x_vel": target_x_vel,
                "safety_ok": step_safety_ok, "ee_world": ee_world.tolist(),
                "qd_max_abs": float(np.max(np.abs(data.qvel[:6]))),
            })

    captured = bool(steps_done == n_steps and not guard_fired and within_tol_from is True)
    return {
        "initial_phi_rad": initial_phi_rad, "initial_thetadot_radps": initial_thetadot_radps,
        "duration_s": duration_s, "a_max": a_max, "v_max": v_max,
        "captured": captured,
        "guard_fired": guard_fired, "guard_reason": guard_reason, "first_guard_t": first_guard_t,
        "steps_completed": steps_done, "n_steps": n_steps,
        "peak_abs_cart_accel_cmd_mps2": peak_abs_u,
        "peak_abs_cart_vel_cmd_mps": peak_abs_target_x_vel,
        "peak_abs_joint_vel_radps": peak_abs_qd,
        "final_phi_deg": float(np.degrees(C.wrap_pi(float(data.qpos[pend_qpos_adr]) - inverted_angle))),
        "history": history,
    }


TUNING_STATES = [
    (np.radians(d), w)
    for d, w in [
        (3.0, 0.0), (-3.0, 0.0), (5.0, -1.0), (-5.0, 1.0),
        (8.0, -1.5), (-8.0, 1.5), (12.0, -2.0), (-12.0, 2.0),
        (0.0, -3.0), (0.0, 3.0),
    ]
]


def _q_matrix(params) -> np.ndarray:
    log_qx, log_qxdot, log_qphi, log_qphidot = params[:4]
    return np.diag([10.0 ** log_qx, 10.0 ** log_qxdot, 10.0 ** log_qphi, 10.0 ** log_qphidot])


def tuning_objective(params, model, constants, damping, Q_per_a, duration_s: float = 4.0) -> float:
    a_max = float(np.clip(params[4], 0.3, 10.0))
    Q = _q_matrix(params)
    R = np.array([[1.0]])
    A, B = linearize_cartpole(constants, damping, Q_per_a)
    try:
        K, _ = lqr_gain(A, B, Q, R)
    except Exception:
        return 1e3
    if not np.all(np.isfinite(K)):
        return 1e3

    hanging_angle, inverted_angle = C.resolve_equilibria(model)[0], C.resolve_equilibria(model)[1]
    total = 0.0
    for phi0, thetadot0 in TUNING_STATES:
        res = run_lqr_trial(
            model, K, duration_s=duration_s, inverted_angle=inverted_angle, hanging_angle=hanging_angle,
            initial_phi_rad=phi0, initial_thetadot_radps=thetadot0, a_max=a_max, v_max=V_MAX_MPS,
        )
        cost = abs(np.radians(res["final_phi_deg"]))
        if res["guard_fired"]:
            cost += 5.0
        if not res["captured"]:
            cost += 2.0
        if res["peak_abs_cart_vel_cmd_mps"] > 0.9 * V_MAX_MPS:
            cost += 1.0
        total += cost
    return total / len(TUNING_STATES)


def search_lqr_gains(model, constants, damping, Q_per_a, *, maxiter: int, popsize: int, seed: int,
                      duration_s: float) -> dict:
    bounds = [
        (-1.0, 4.0), (-1.0, 4.0), (-1.0, 6.0), (-1.0, 5.0), (0.5, 10.0),
    ]
    res = differential_evolution(
        functools.partial(tuning_objective, model=model, constants=constants, damping=damping,
                           Q_per_a=Q_per_a, duration_s=duration_s),
        bounds, maxiter=maxiter, popsize=popsize, tol=1e-4, seed=seed,
        workers=_de_workers(), polish=False,
    )
    Q = _q_matrix(res.x)
    a_max = float(np.clip(res.x[4], 0.3, 10.0))
    A, B = linearize_cartpole(constants, damping, Q_per_a)
    K, P = lqr_gain(A, B, Q, np.array([[1.0]]))
    return {
        "Q_diag": np.diag(Q).tolist(), "R": 1.0, "a_max": a_max,
        "K": K.reshape(-1).tolist(), "cost": float(res.fun),
        "A": A.tolist(), "B": B.tolist(), "damping": damping,
        "log10_params": res.x.tolist(),
    }


def capture_envelope_grid(
    model, K: np.ndarray, *, hanging_angle: float, inverted_angle: float,
    phi_deg_values, thetadot_values, a_max: float, duration_s: float = 5.0,
    verify_k_zero: bool = True,
) -> list[dict]:
    rows = []
    for phi_deg in phi_deg_values:
        for thetadot in thetadot_values:
            phi0 = np.radians(phi_deg)
            res = run_lqr_trial(
                model, K, duration_s=duration_s, inverted_angle=inverted_angle, hanging_angle=hanging_angle,
                initial_phi_rad=phi0, initial_thetadot_radps=thetadot, a_max=a_max, v_max=V_MAX_MPS,
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
                    model, np.zeros((1, 4)), duration_s=duration_s, inverted_angle=inverted_angle,
                    hanging_angle=hanging_angle, initial_phi_rad=phi0, initial_thetadot_radps=thetadot,
                    a_max=a_max, v_max=V_MAX_MPS,
                )
                row["k_zero_also_captured"] = res0["captured"]
                row["k_zero_final_phi_deg"] = res0["final_phi_deg"]
                row["active_control_confirmed"] = res["captured"] and not res0["captured"]
            rows.append(row)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maxiter", type=int, default=20)
    parser.add_argument("--popsize", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--duration-s", type=float, default=4.0)
    parser.add_argument("--envelope", action="store_true")
    parser.add_argument("--envelope-duration-s", type=float, default=5.0)
    parser.add_argument("--output-json", type=str, default=None)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    model = C.build_model()
    constants = C.default_constants()
    damping = hinge_damping(model)
    hanging_angle, inverted_angle = C.resolve_equilibria(model)
    data = mujoco.MjData(model)
    data.qpos[:6] = C.ARM_Q0
    mujoco.mj_forward(model, data)
    _, _, site_id = C.hinge_ids(model)
    tool_x, tool_y, tool_z = C.tool_frame_world(data, site_id)

    coupling = cart_coupling_report(model, C.ARM_Q0, inverted_angle, tool_y, constants)
    Q_per_a = coupling["Q_per_a_nm_per_mps2"]
    print("pendulum constants:", constants)
    print("hinge damping:", damping)
    print("hanging/inverted (deg):", np.degrees(hanging_angle), np.degrees(inverted_angle))
    print("cart coupling report (kappa):", coupling)

    print("=== searching 4-state cartpole LQR Q/R (+a_max) via differential_evolution ===")
    result = search_lqr_gains(model, constants, damping, Q_per_a, maxiter=args.maxiter,
                               popsize=args.popsize, seed=args.seed, duration_s=args.duration_s)
    print("Q_diag =", result["Q_diag"], "a_max =", result["a_max"])
    print("K =", result["K"])
    print("cost =", result["cost"])

    envelope = None
    if args.envelope:
        print("=== capture envelope grid (phi0 x thetadot0), with K=0 counterfactual on captures ===")
        K = np.asarray(result["K"], dtype=np.float64).reshape(1, 4)
        envelope = capture_envelope_grid(
            model, K, hanging_angle=hanging_angle, inverted_angle=inverted_angle,
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
        with Path(args.output_json).open("w") as fp:
            json.dump({
                "constants": vars(constants), "damping": damping,
                "hanging_angle": hanging_angle, "inverted_angle": inverted_angle,
                "coupling": coupling, "lqr": result, "envelope": envelope,
            }, fp, indent=2, default=str)
        print("wrote", args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
