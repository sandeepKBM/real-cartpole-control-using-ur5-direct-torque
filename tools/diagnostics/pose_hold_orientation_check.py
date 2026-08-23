#!/usr/bin/env python3
"""Can the LOW-LEVEL controller alone hold a pose, with the pendulum removed?

WHY THIS EXISTS. The Goal-2 cascade-LQR capture envelopes at the singular
ARM_Q0 all returned 0/117 genuine captures (the single "captured" cell in each
is phi0=0/thetadot0=0, where the K=0 counterfactual also holds -- i.e. no
active control was demonstrated anywhere). Their failure mode is NOT the
pendulum falling: 98% of cells terminate on an ARM-SIDE guard
(``||orientation error|| > 0.25 rad``, or Y/Z drift), often before the catch
has begun. A gain search cannot fix that, and spending search wall-clock on
Q/R/a_max while the objective is dominated by arm-guard trips measures the
wrong thing.

This script removes the pendulum from the question entirely and asks the
prior question directly:

    with NO pendulum attached, can this controller+config hold this pose for
    ``duration_s`` while tracking a PRESCRIBED cart-acceleration command of
    amplitude A, without tripping its own guards?

The command is prescribed (an open-loop sinusoid), never fed back from a pole
state, so nothing here depends on an LQR gain. Sweeping A over a ladder
starting at A=0 separates three hypotheses that the envelope runs confound:

    A = 0 trips           -> the controller cannot even STAND at this pose.
                             No LQR gain is relevant; the low-level controller
                             is the blocker.
    A = 0 holds, A > 0
      trips at small A    -> the pose has real but very limited command
                             authority; the LQR's own a_max is above what the
                             arm can absorb, so the cascade is asking for
                             motion the low level cannot deliver.
    holds to large A      -> the arm is fine and the LQR gains / linearization
                             are the actual problem.

Guards are LEFT ON and are the pass/fail criterion (per this repo's standing
rule that a result needing guards disabled is a negative result), but the
rollout does NOT stop at the first trip by default (``--stop-on-guard`` opts
in): continuing past the trip records how far the quantity actually goes,
which distinguishes "grazed the threshold" from "diverged". The first trip
time and reason are recorded either way.

The pendulum is removed by loading the bare arm scene rather than the composed
arm+pendulum model. That is the honest form of "removed entirely" -- the
composed model's hinge would keep applying reaction torque no matter where it
was initialized. ``--pendulum-xml`` composes it back in (parked at its hanging
equilibrium) for an A/B against the same command ladder.
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

from simulation.ur5e_mujoco_torque import (  # noqa: E402
    build_initial_state_and_adapter,
    build_mujoco_state,
)
from tools.diagnostics.pendulum_swingup_energy_shaping import (  # noqa: E402
    CONTROL_DT,
    RATE_HZ,
    load_config,
)

BARE_ARM_SCENE = REPO_ROOT / "assets" / "ur5e_torque" / "scene.xml"

JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)

# The cart-speed clip the cascade itself applies before integrating u into the
# position reference (pendulum_lqr_cascade.V_MAX_MPS). Mirrored here so the
# command ladder drives the controller through the SAME reference-generation
# path the envelope runs used -- an unclipped sinusoid would be a different
# input than anything the LQR can actually emit.
V_MAX_MPS = 1.00


def build_model(pendulum_xml: str | None) -> mujoco.MjModel:
    """Bare arm when ``pendulum_xml`` is None, else the composed arm+pendulum."""
    if pendulum_xml is None:
        return mujoco.MjModel.from_xml_path(str(BARE_ARM_SCENE))
    from simulation.ur5e_pendulum_compose import compose_ur5e_pendulum_model

    return compose_ur5e_pendulum_model(pendulum_xml=pendulum_xml)


def run_hold_trial(
    model: mujoco.MjModel,
    *,
    arm_q: np.ndarray,
    accel_amplitude_mps2: float,
    accel_freq_hz: float,
    duration_s: float,
    config_path: Path | None,
    controller_kind: str,
    transport_axis_index: int = 0,
    stop_on_guard: bool = False,
) -> dict:
    """Holds ``arm_q`` while commanding u(t) = A*cos(2*pi*f*t) as the cart
    acceleration reference. A=0 is a pure hold. Returns the peak of every
    quantity the ImpedanceSafetyMonitor gates on, plus the first trip.

    COSINE, NOT SINE -- and the difference is the whole validity of the test.
    The reference is built by integrating u twice, from rest. With sin, the
    velocity is (A/w)*(1 - cos wt): strictly non-negative, mean A/w, so the
    "oscillation" is really a RAMP that transports the arm A/w * T metres
    (0.24 m at A=0.25, f=1, T=6 s). That measures the pose's transport range,
    which is a known-limited and entirely different quantity. With cos, the
    velocity is (A/w)*sin(wt) -- zero mean -- and the position oscillates
    inside [0, 2A/w^2] with no net travel, which is what "can it hold this pose
    while being shaken" actually means, and is also the shape of the LQR's own
    zero-mean feedback command. ``commanded_excursion_m`` is reported so the
    size of the demand is never left implicit again."""
    config = load_config(config_path) if config_path is not None else load_config()
    arm_q = np.asarray(arm_q, dtype=np.float64).reshape(6)

    data = mujoco.MjData(model)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    joint_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES
    ]

    data.qpos[:6] = arm_q
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    state0, adapter = build_initial_state_and_adapter(
        model,
        data,
        site_id,
        joint_ids,
        controller_cfg=config["controller"],
        transport_axis_index=transport_axis_index,
        target_x_delta=0.0,
        controller_kind=str(controller_kind),
        force_hold_current_pose=False,
        gravity_mode=config["mujoco"].get("gravity_mode", "gravity_comp"),
        gravity_source=config["mujoco"].get("gravity_source", "pinocchio"),
        coriolis_feedforward=bool(config["mujoco"].get("coriolis_feedforward", True)),
        torque_limit_scale=1.0,
    )
    ee0 = np.asarray(state0.ee_pos, dtype=np.float64).copy()
    x_ref = float(ee0[transport_axis_index])

    n_steps = int(duration_s * RATE_HZ)
    target_x = x_ref
    target_x_vel = 0.0

    guard_fired = False
    guard_reason = None
    first_guard_t = None
    steps_done = 0

    max_orient = 0.0
    max_dy = 0.0
    max_dz = 0.0
    max_dx = 0.0
    max_qd = 0.0

    omega_cmd = 2.0 * np.pi * accel_freq_hz

    for step in range(n_steps):
        t = step * CONTROL_DT
        u = float(accel_amplitude_mps2 * np.cos(omega_cmd * t))

        target_x_vel = float(
            np.clip(target_x_vel + u * CONTROL_DT, -V_MAX_MPS, V_MAX_MPS)
        )
        target_x = target_x + target_x_vel * CONTROL_DT

        state = build_mujoco_state(
            model,
            data,
            site_id=site_id,
            joint_ids=joint_ids,
            time_s=t,
            dt_s=CONTROL_DT,
            target_x=target_x,
            target_x_vel=target_x_vel,
            target_x_accel=u,
            reference_quat=state0.ee_quat,
            transport_axis_index=transport_axis_index,
            gravity_compensation=True,
        )

        tau, diag = adapter.step(state=state)

        if not bool(diag.get("safety_ok", True)) and first_guard_t is None:
            first_guard_t = t
            guard_reason = str(diag.get("safety_reason", ""))
            guard_fired = True
            if stop_on_guard:
                break

        max_orient = max(max_orient, float(diag.get("orientation_error_norm", 0.0)))
        ee = np.asarray(state.ee_pos, dtype=np.float64)
        # Ask the MONITOR for its own drift vector rather than differencing
        # against a locally-remembered start pose. Two things make the local
        # version wrong: the guard resolves drift in its captured frame (world
        # or task, depending on whether a task rotation was supplied), and its
        # reference _pos0 is captured inside the adapter's init path, which can
        # re-anchor -- so a locally-anchored world difference can exceed a
        # threshold while the guard, measuring from its own reference, does not
        # fire. Reporting a number that cannot be compared against the limit
        # printed next to it is how a frame/anchor error survives review.
        drift = np.asarray(
            adapter.safety_monitor.drift_vector(ee), dtype=np.float64
        ).reshape(3)
        max_dx = max(max_dx, abs(float(drift[0])))
        max_dy = max(max_dy, abs(float(drift[1])))
        max_dz = max(max_dz, abs(float(drift[2])))

        data.ctrl[:6] = np.asarray(tau, dtype=np.float64).reshape(6)
        mujoco.mj_step(model, data)
        steps_done += 1
        max_qd = max(max_qd, float(np.max(np.abs(data.qvel[:6]))))

    return {
        "accel_amplitude_mps2": accel_amplitude_mps2,
        "accel_freq_hz": accel_freq_hz,
        # Peak-to-peak of the commanded reference: 2A/w^2. Reported so a trip
        # can be read against how far the arm was actually asked to move.
        "commanded_excursion_m": (2.0 * accel_amplitude_mps2 / (omega_cmd ** 2)) if omega_cmd else 0.0,
        "held": bool(not guard_fired and steps_done == n_steps),
        "guard_fired": guard_fired,
        "guard_reason": guard_reason,
        "first_guard_t": first_guard_t,
        "steps_completed": steps_done,
        "n_steps": n_steps,
        "max_orientation_error_rad": max_orient,
        # WORLD-FRAME deviations, reported as diagnostics only. They are NOT
        # the quantity ImpedanceSafetyMonitor gates on here: when a task
        # rotation is set (which the corridor-QP configs do), the monitor
        # checks TASK-frame drift, rot.T @ (ee - pos0), per component with the
        # commanded axis exempt, against max_abs_orthogonal_drift_m -- see
        # controller_core/safety.py. Only in the no-rotation branch does it
        # check world Y/Z. So a world |dy| above max_abs_y_drift_m with
        # held=True is not a contradiction, and comparing these columns
        # against the config's drift limits is a frame error. The authoritative
        # pass/fail is `held` / `guard_reason`, which come from the monitor.
        "max_abs_guard_drift_axis0_m": max_dx,
        "max_abs_guard_drift_axis1_m": max_dy,
        "max_abs_guard_drift_axis2_m": max_dz,
        "max_abs_joint_vel_radps": max_qd,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--controller-kind", default="impedance")
    p.add_argument("--start-q-rad", type=float, nargs=6, required=True)
    p.add_argument("--pendulum-xml", default=None,
                   help="Omit for the BARE ARM (pendulum removed entirely).")
    p.add_argument("--duration-s", type=float, default=6.0)
    p.add_argument("--accel-amplitudes", type=float, nargs="+",
                   default=[0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
    p.add_argument("--accel-freq-hz", type=float, default=1.0)
    p.add_argument("--transport-axis-index", type=int, default=0)
    p.add_argument("--stop-on-guard", action="store_true")
    p.add_argument("--output-json", type=Path, default=None)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    model = build_model(args.pendulum_xml)
    arm_q = np.asarray(args.start_q_rad, dtype=np.float64)

    print(f"model            = {'BARE ARM (no pendulum)' if args.pendulum_xml is None else args.pendulum_xml}")
    print(f"controller_kind  = {args.controller_kind}")
    print(f"config           = {args.config}")
    print(f"arm_q            = {np.round(arm_q, 6).tolist()}")
    w = 2.0 * np.pi * float(args.accel_freq_hz)
    print(f"command          = u(t) = A*cos(2*pi*{args.accel_freq_hz}*t) m/s^2, {args.duration_s}s "
          f"(zero-mean velocity; commanded excursion = 2A/w^2 = A*{2.0 / w**2:.4f} m)")
    print()
    header = (f"{'A m/s^2':>9} {'cmd_m':>7} {'held':>6} {'orient':>8} {'drift0':>7} {'drift1':>7} "
              f"{'drift2':>7} {'|qd|':>7}  first trip")
    print(header)
    print("-" * len(header))

    results = []
    for amp in args.accel_amplitudes:
        r = run_hold_trial(
            model,
            arm_q=arm_q,
            accel_amplitude_mps2=float(amp),
            accel_freq_hz=float(args.accel_freq_hz),
            duration_s=float(args.duration_s),
            config_path=args.config,
            controller_kind=str(args.controller_kind),
            transport_axis_index=int(args.transport_axis_index),
            stop_on_guard=bool(args.stop_on_guard),
        )
        results.append(r)
        trip = "-" if not r["guard_fired"] else f"{r['first_guard_t']:.3f}s {r['guard_reason']}"
        print(f"{amp:9.3f} {r['commanded_excursion_m']:7.4f} {str(r['held']):>6} "
              f"{r['max_orientation_error_rad']:8.4f} "
              f"{r['max_abs_guard_drift_axis0_m']:7.4f} {r['max_abs_guard_drift_axis1_m']:7.4f} "
              f"{r['max_abs_guard_drift_axis2_m']:7.4f} {r['max_abs_joint_vel_radps']:7.3f}  {trip}")

    payload = {
        "pendulum_xml": args.pendulum_xml,
        "controller_kind": args.controller_kind,
        "config": str(args.config) if args.config else None,
        "arm_q": arm_q.tolist(),
        "duration_s": args.duration_s,
        "accel_freq_hz": args.accel_freq_hz,
        "trials": results,
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
