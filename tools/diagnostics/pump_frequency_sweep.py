#!/usr/bin/env python3
"""How much energy can a pivot oscillation inject per unit of GUARD BUDGET?

THE QUESTION. Every swing-up search in this repo has maximised energy, or
``min_theta_dist_from_inverted``, subject to guards as a pass/fail afterthought.
The operative constraint is the other way round: the guards bound DISPLACEMENT
(off-axis drift 0.06 m, orientation 0.25 rad, |qd| 3.0 rad/s), and the pose-hold
ladder (pose_hold_orientation_check.py) measured where that bites -- the
singular ARM_Q0 trips in A in (1.0, 1.5] m/s^2 at 1 Hz, wrist_2=-90 in
(2.0, 3.0]. So the real design quantity is energy gained PER UNIT of guard
budget spent, and the free lever nobody has pulled is DRIVE FREQUENCY.

WHY FREQUENCY IS THE LEVER. Integrating u(t) = A*cos(w t) twice from rest gives
a reference that oscillates within 2A/w^2 with no net travel. At a FIXED
displacement budget the admissible amplitude therefore scales as

    A = excursion * w^2 / 2      i.e.   A  proportional to  f^2

so raising f buys amplitude quadratically without spending more displacement --
and the pendulum's own natural frequency (omega = 10.8334 rad/s, f = 1.724 Hz
for the 0.12 m rod) is where that amplitude also couples coherently into the
pendulum. Every ladder run so far used 1 Hz, i.e. off-resonance and
displacement-expensive.

This sweep holds the commanded excursion CONSTANT and varies f, so every row
spends the same displacement budget and the only thing that changes is how much
energy comes back. Guards stay ON and are reported as a UTILISATION fraction
(max over drift/orientation/velocity, each against its own limit), so "held" is
not the only output -- a row that holds at 0.95 utilisation is not the same
result as one that holds at 0.30.

THE AXIS TRAP, WHICH THIS SCRIPT REFUSES TO WALK INTO. The drive is ONE scalar
axis, and pivot acceleration along the HINGE exerts no torque at all: AGENTS.md
records that every world-X run at ARM_Q0 spends ~70% of its motion along the
hinge, and that a tool-frame run once drove an axis 7.3 deg from vertical and
was worthless in a way invisible to the config, the gains and the logs. So
before sweeping anything this script MEASURES the pivot coupling c0 = Q/a (hinge
generalized force per unit pivot acceleration, at the hanging equilibrium) for
world X, Y and Z, prints all three, and refuses to guess: pick the axis with
--transport-axis-index, or pass --auto-axis to take the largest |c0| and have it
say so. c0 is also exactly the coefficient EnergyMonitor needs, so the axis
check and the energy accounting come from the same measured number rather than
two independently-assumed ones.
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

from controller_core.config_provenance import check_config_pose, describe_provenance  # noqa: E402
from simulation.ur5e_pendulum_compose import (  # noqa: E402
    compose_ur5e_pendulum_model,
    derive_pendulum_constants,
)
from simulation.ur5e_mujoco_torque import (  # noqa: E402
    build_initial_state_and_adapter,
    build_mujoco_state,
)
from tools.diagnostics.pendulum_energy_monitor import EnergyMonitor  # noqa: E402
from tools.diagnostics.pendulum_swingup_energy_shaping import (  # noqa: E402
    CONTROL_DT,
    RATE_HZ,
    load_config,
    measure_pivot_coupling,
    resolve_equilibria,
)

JOINT_NAMES = (
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
)

# Same clip the cascade applies before integrating u into the reference
# (pendulum_lqr_cascade.V_MAX_MPS), so the command is one the LQR could emit.
V_MAX_MPS = 1.00

AXIS_NAMES = ("world_X", "world_Y", "world_Z")


def measure_axis_couplings(model, arm_q, hanging_angle: float) -> dict[str, float]:
    """c0 = Q/a at the hanging equilibrium for each world axis.

    |c0| is the drive authority of that axis; near-zero means the axis is
    (anti)parallel to the hinge and pumping along it does nothing.
    """
    out = {}
    for idx, name in enumerate(AXIS_NAMES):
        axis = np.zeros(3)
        axis[idx] = 1.0
        out[name] = float(measure_pivot_coupling(model, arm_q, hanging_angle, axis))
    return out


def run_pump_trial(
    model,
    *,
    arm_q: np.ndarray,
    freq_hz: float,
    excursion_m: float,
    duration_s: float,
    config_path: Path,
    controller_kind: str,
    hanging_angle: float,
    constants,
    coupling_c0: float,
    transport_axis_index: int = 0,
) -> dict:
    """One open-loop pivot oscillation at ``freq_hz``, sized so the commanded
    excursion is ``excursion_m`` regardless of frequency."""
    config = load_config(config_path)
    arm_q = np.asarray(arm_q, dtype=np.float64).reshape(6)

    data = mujoco.MjData(model)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES]
    pend_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "/pendulum_hinge")
    pend_qpos = model.jnt_qposadr[pend_jid]
    pend_dof = model.jnt_dofadr[pend_jid]

    data.qpos[:6] = arm_q
    data.qpos[pend_qpos] = hanging_angle
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    state0, adapter = build_initial_state_and_adapter(
        model, data, site_id, joint_ids,
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
    sm = adapter.safety_monitor
    drift_limit = float(sm.cfg.max_abs_orthogonal_drift_m)
    orient_limit = float(sm.cfg.max_orientation_error_rad)
    qd_limit = float(sm.cfg.max_joint_velocity_radps)

    w = 2.0 * np.pi * float(freq_hz)
    amplitude = float(excursion_m) * w * w / 2.0

    mon = EnergyMonitor(
        i_pivot_kgm2=float(constants.i_pivot_kgm2),
        mgr_nm=float(constants.mgr_nm),
        e_top_j=float(constants.e_top_j),
        coupling_c0=float(coupling_c0),
        hinge_damping=float(model.dof_damping[pend_dof]),
    )

    target = float(state0.ee_pos[transport_axis_index])
    target_vel = 0.0
    guard_fired = False
    guard_reason = None
    first_guard_t = None
    util = 0.0
    max_abs_phi = 0.0

    for step in range(int(duration_s * RATE_HZ)):
        t = step * CONTROL_DT
        u = float(amplitude * np.cos(w * t))
        target_vel = float(np.clip(target_vel + u * CONTROL_DT, -V_MAX_MPS, V_MAX_MPS))
        target = target + target_vel * CONTROL_DT

        state = build_mujoco_state(
            model, data, site_id=site_id, joint_ids=joint_ids,
            time_s=t, dt_s=CONTROL_DT,
            target_x=target, target_x_vel=target_vel, target_x_accel=u,
            reference_quat=state0.ee_quat,
            transport_axis_index=transport_axis_index,
            gravity_compensation=True,
        )
        tau, diag = adapter.step(state=state)

        if not bool(diag.get("safety_ok", True)) and first_guard_t is None:
            first_guard_t = t
            guard_reason = str(diag.get("safety_reason", ""))
            guard_fired = True

        # Guard UTILISATION, from the monitor's own drift vector and limits --
        # never a locally-recomputed difference (see pose_hold_orientation_check).
        drift = np.asarray(sm.drift_vector(np.asarray(state.ee_pos)), dtype=np.float64)
        exempt = sm._tracked_axes if sm._tracked_axes is not None else frozenset({transport_axis_index})
        off_axis = [abs(float(drift[i])) for i in range(3) if i not in exempt]
        util = max(
            util,
            (max(off_axis) / drift_limit) if off_axis else 0.0,
            float(diag.get("orientation_error_norm", 0.0)) / orient_limit,
            float(np.max(np.abs(data.qvel[:6]))) / qd_limit,
        )

        theta = float(data.qpos[pend_qpos])
        thetadot = float(data.qvel[pend_dof])
        phi = theta - hanging_angle
        max_abs_phi = max(max_abs_phi, abs(phi))
        mon.step(thetadot=thetadot, phi=phi, drive_accel=u, dt=CONTROL_DT)

        data.ctrl[:6] = np.asarray(tau, dtype=np.float64).reshape(6)
        mujoco.mj_step(model, data)

    b = mon.budget()
    e_top = float(constants.e_top_j)
    return {
        "freq_hz": float(freq_hz),
        "amplitude_mps2": amplitude,
        "excursion_m": float(excursion_m),
        "held": bool(not guard_fired),
        "guard_reason": guard_reason,
        "first_guard_t": first_guard_t,
        "guard_utilisation": float(util),
        "e_peak_over_e_top": float(b.e_peak_j / e_top) if e_top else None,
        "delta_e_j": float(b.e_peak_j - b.e_initial_j),
        "work_by_drive_j": float(b.work_by_drive_j),
        "positive_work_fraction": float(b.positive_work_fraction),
        "max_abs_phi_deg": float(np.degrees(max_abs_phi)),
        # The headline: energy bought per unit of guard budget spent.
        "e_gain_per_util": (float((b.e_peak_j - b.e_initial_j) / util) if util > 0 else None),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pendulum-xml", required=True)
    p.add_argument("--start-q-rad", type=float, nargs=6, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--controller-kind", default="impedance")
    p.add_argument("--freqs-hz", type=float, nargs="+",
                   default=[0.5, 0.75, 1.0, 1.25, 1.5, 1.724, 2.0, 2.5, 3.0])
    p.add_argument("--excursion-m", type=float, default=0.05,
                   help="Commanded peak-to-peak reference travel, held CONSTANT "
                        "across frequencies so every row spends the same "
                        "displacement budget. Default 0.05 m, just inside the "
                        "0.06 m drift guard.")
    p.add_argument("--duration-s", type=float, default=8.0)
    p.add_argument("--transport-axis-index", type=int, default=None,
                   help="0/1/2 = world X/Y/Z. Required unless --auto-axis.")
    p.add_argument("--auto-axis", action="store_true",
                   help="Use the world axis with the largest |c0|, and say so.")
    p.add_argument("--allow-pose-mismatch", action="store_true")
    p.add_argument("--output-json", type=Path, default=None)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    arm_q = np.asarray(args.start_q_rad, dtype=np.float64)

    provenance = check_config_pose(
        load_config(Path(args.config)), arm_q, args.pendulum_xml,
        config_name=Path(args.config).name,
        allow_mismatch=bool(args.allow_pose_mismatch),
    )

    model = compose_ur5e_pendulum_model(pendulum_xml=str(args.pendulum_xml))
    hanging_angle, inverted_angle = resolve_equilibria(model, arm_q)
    constants = derive_pendulum_constants(model, arm_q)

    couplings = measure_axis_couplings(model, arm_q, hanging_angle)
    best_axis = int(np.argmax([abs(couplings[n]) for n in AXIS_NAMES]))

    print(f"pendulum={Path(args.pendulum_xml).name}  arm_q={np.round(arm_q, 6).tolist()}")
    print(f"config={Path(args.config).name}  kind={args.controller_kind}")
    print(describe_provenance(provenance))
    print(f"omega={constants.omega_natural_radps:.4f} rad/s  "
          f"f_natural={constants.omega_natural_radps / (2 * np.pi):.4f} Hz  "
          f"E_top={constants.e_top_j:.6f} J")
    print("\nPIVOT COUPLING c0 = Q/a at hanging (drive authority per axis):")
    for i, n in enumerate(AXIS_NAMES):
        mark = "  <-- largest" if i == best_axis else ""
        print(f"  {n}: {couplings[n]:+.6f} Nm per m/s^2{mark}")

    if args.transport_axis_index is None:
        if not args.auto_axis:
            print("\nREFUSING TO GUESS THE DRIVE AXIS. Pass --transport-axis-index "
                  "(0/1/2) or --auto-axis. A pump along the hinge exerts no torque "
                  "and the run would be worthless in a way the logs do not show.")
            return 2
        axis = best_axis
        print(f"\n--auto-axis -> using {AXIS_NAMES[axis]} (largest |c0|)")
    else:
        axis = int(args.transport_axis_index)
        print(f"\nusing {AXIS_NAMES[axis]} (explicitly requested)")

    c0 = couplings[AXIS_NAMES[axis]]
    print(f"\nexcursion held constant at {args.excursion_m:.4f} m; A = excursion*w^2/2\n")
    hdr = (f"{'f Hz':>7} {'A m/s^2':>9} {'held':>6} {'util':>6} {'Epk/Etop':>9} "
           f"{'dE mJ':>8} {'pos%':>6} {'phi_max':>8} {'dE/util':>9}  guard")
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for f in args.freqs_hz:
        r = run_pump_trial(
            model, arm_q=arm_q, freq_hz=float(f), excursion_m=float(args.excursion_m),
            duration_s=float(args.duration_s), config_path=Path(args.config),
            controller_kind=str(args.controller_kind), hanging_angle=hanging_angle,
            constants=constants, coupling_c0=c0, transport_axis_index=axis,
        )
        rows.append(r)
        g = "-" if r["held"] else f"{r['first_guard_t']:.3f}s {r['guard_reason']}"
        dpu = "-" if r["e_gain_per_util"] is None else f"{1000 * r['e_gain_per_util']:9.2f}"
        print(f"{r['freq_hz']:7.3f} {r['amplitude_mps2']:9.3f} {str(r['held']):>6} "
              f"{r['guard_utilisation']:6.3f} {r['e_peak_over_e_top']:9.4f} "
              f"{1000 * r['delta_e_j']:8.3f} {100 * r['positive_work_fraction']:6.1f} "
              f"{r['max_abs_phi_deg']:8.2f} {dpu}  {g}")

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps({
            "pendulum_xml": str(args.pendulum_xml),
            "arm_q": arm_q.tolist(),
            "config": str(args.config),
            "controller_kind": args.controller_kind,
            "axis_couplings_nm_per_mps2": couplings,
            "transport_axis_index": axis,
            "excursion_m": args.excursion_m,
            "duration_s": args.duration_s,
            "omega_natural_radps": float(constants.omega_natural_radps),
            "e_top_j": float(constants.e_top_j),
            "provenance": provenance.as_dict(),
            "rows": rows,
        }, indent=2))
        print(f"\nwrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
