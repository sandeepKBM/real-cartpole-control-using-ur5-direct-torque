#!/usr/bin/env python3
"""The minimum drive needed to beat hinge losses -- derived, then measured.

WHY THIS EXISTS. Every swing-up parameter in this repo has been found by search,
so nothing said which regions of the search space were PHYSICALLY DEAD. They
exist: below a definite acceleration the Coulomb term at the hinge wins at every
phase and no amount of time, velocity or tuning produces net energy. A search
that spends candidates there is not exploring, it is sampling zeros -- the DE
search over the energy schedule had `a_slow` bounds starting at 0.2 m/s^2, which
this file shows is below the floor.

THE DERIVATION. With the phase-locked drive a = A*sign(c0*cos(phi)*thetadot),
which is the pointwise-optimal choice subject to |a| <= A, the hinge power
balance is

    dE/dt = A*|c0|*|cos(phi)|*|thetadot|  -  b*thetadot^2  -  f*|thetadot|

where c0 = Q/a is the MEASURED pivot coupling (a property of pose, asset and
drive axis -- see measure_pivot_coupling), b is the hinge viscous damping and f
its Coulomb frictionloss. Dividing by |thetadot| (nonzero) gives

    A*|c0|*|cos(phi)|  >  f + b*|thetadot|                                  (1)

Three consequences, and the first is the one that matters most:

 1. VELOCITY DROPS OUT of the leading term. Net energy gain is a condition on
    ACCELERATION, not on speed -- there is no "minimum velocity" to overcome
    Coulomb friction. What velocity does is set the viscous penalty on the
    right-hand side, i.e. a CEILING, not a floor.
 2. BREAKAWAY FLOOR. At the hanging equilibrium (|cos phi| = 1) and small
    speed, (1) becomes A > f/|c0|. Below that the drive cannot add energy at
    the one phase where it has maximum authority, so it cannot anywhere.
 3. VISCOUS SPEED CAP. Rearranging (1): |thetadot| < (A*|c0|*|cos phi| - f)/b.
    Exceed it and the same drive starts REMOVING energy.

Integrating (1) over a full swing (int |cos phi| dphi = 2 over 0..pi, while the
loss terms integrate over the angle itself) gives the cost of reaching the top:

    A*|c0|*2  >  (f + b*<|thetadot|>)*pi                                    (2)

MEASURED (10 s, Goal-1 pose, realrod, friction_ff config). The dead zone is
real and sharp, and the CYCLE-AVERAGED threshold (2) is the one that predicts it:

    A = 0.25 .. 0.80  ->  E_peak/E_top = 0.0001, positive_work ~0.40
    A = 1.20          ->  E_peak/E_top = 0.0020, positive_work  0.389
    A = 1.60          ->  E_peak/E_top = 0.0585, positive_work  0.984   <-- onset
    A = 2.40          ->  E_peak/E_top = 0.3732, positive_work  0.988
    A = 4.20          ->  E_peak/E_top = 0.6677, positive_work  0.989

against a predicted (2) of 1.6418 m/s^2 -- a 2.6% match. Note which prediction
FAILED: the instantaneous breakaway f/|c0| = 0.502 shows no transition at all.
The drive must pay the loss integrated over the whole swing, not merely at the
single phase where its authority peaks, so (2) is the operative floor and (1) at
|cos phi| = 1 is optimistic by ~3.3x.

WHERE THE ENERGY GOES -- checked, not assumed. An earlier version of this
docstring claimed the shortfall between commanded work and delivered energy was
the arm failing to track. That was WRONG and the arithmetic refutes it: at
A = 4.2, work_by_drive = 267.0 mJ and the pendulum gains 37.2 mJ, leaving
229.8 mJ; hinge dissipation over the same run is b*int(thetadot^2)dt ~ 100 mJ
plus f*int|thetadot|dt ~ 100 mJ ~ 200 mJ, which accounts for essentially all of
it. The budget closes AT THE HINGE. The pump is friction-limited, and the
E_peak/E_top ~ 0.65 plateau is the amplitude at which dissipation per cycle
equals input per cycle for that A -- not an arm-tracking ceiling.

That is why a CONSTANT amplitude cannot reach the top however long it runs, and
why the energy-scheduled law's a_sharp = 12.5 near the top does (measured
E_peak/E_top = 2.54): raising A raises the amplitude at which the balance
closes. `dissipated_j` is reported per row so this budget can be closed exactly
rather than estimated.
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
from tools.diagnostics.pendulum_swingup_energy_shaping import (  # noqa: E402
    load_config,
    measure_pivot_coupling,
    resolve_equilibria,
)
from tools.diagnostics.pendulum_two_phase_swingup import (  # noqa: E402
    EnergyScheduleParams,
    run_energy_scheduled_trial,
)

HINGE_NAMES = ("/pendulum_hinge", "pendulum_hinge")


def hinge_friction(model) -> tuple[float, float]:
    """(viscous damping b, Coulomb frictionloss f) of the pendulum hinge."""
    jid = -1
    for name in HINGE_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            break
    if jid < 0:
        raise ValueError("pendulum hinge joint not found")
    dof = model.jnt_dofadr[jid]
    return float(model.dof_damping[dof]), float(model.dof_frictionloss[dof])


def derived_thresholds(c0: float, b: float, f: float,
                       thetadot_typ: float) -> dict:
    """Closed-form floors and ceilings from the power balance."""
    a_breakaway = f / abs(c0)
    # Cycle-averaged cost of a full 0..pi swing: int|cos| = 2 vs loss over pi.
    a_to_top = (np.pi / 2.0) * (f + b * thetadot_typ) / abs(c0)
    return {
        "a_breakaway_mps2": float(a_breakaway),
        "a_to_top_mps2": float(a_to_top),
        "thetadot_typ_radps": float(thetadot_typ),
        "note": "velocity sets a CEILING (viscous), never a floor; the floor is on acceleration",
    }


def viscous_speed_cap(A: float, c0: float, b: float, f: float,
                      cos_phi: float = 1.0) -> float:
    return (A * abs(c0) * cos_phi - f) / b


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pendulum-xml", required=True)
    p.add_argument("--start-q-rad", type=float, nargs=6, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--controller-kind", default="impedance")
    p.add_argument("--transport-axis-index", type=int, default=0)
    p.add_argument("--amplitudes", type=float, nargs="+",
                   default=[0.25, 0.40, 0.50, 0.60, 0.80, 1.20, 1.60, 2.40, 4.20])
    p.add_argument("--duration-s", type=float, default=10.0)
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
    hanging, inverted = resolve_equilibria(model, arm_q)
    constants = derive_pendulum_constants(model, arm_q)
    b, f = hinge_friction(model)
    axis = int(args.transport_axis_index)
    drive_axis = np.zeros(3)
    drive_axis[axis] = 1.0
    c0 = float(measure_pivot_coupling(model, arm_q, hanging, drive_axis))

    # Representative speed over a full swing: half the bottom-of-swing speed at
    # E_top, which is where the viscous term is actually paid.
    thetadot_top = float(np.sqrt(2.0 * constants.e_top_j / constants.i_pivot_kgm2))
    th = derived_thresholds(c0, b, f, thetadot_top / 2.0)

    print(f"pendulum={Path(args.pendulum_xml).name}  arm_q={np.round(arm_q,6).tolist()}")
    print(describe_provenance(provenance))
    print(f"\nMEASURED MODEL CONSTANTS")
    print(f"  |c0| (pivot coupling) = {abs(c0):.6g} N.m per m/s^2")
    print(f"  b    (viscous)        = {b:.6g} N.m.s/rad")
    print(f"  f    (Coulomb)        = {f:.6g} N.m")
    print(f"  I_pivot={constants.i_pivot_kgm2:.6g}  omega={constants.omega_natural_radps:.4f}  "
          f"E_top={constants.e_top_j:.6f} J")
    print(f"  thetadot at E_top (bottom of swing) = {thetadot_top:.3f} rad/s")
    print(f"\nDERIVED THRESHOLDS (acceleration floors; velocity is a ceiling, not a floor)")
    print(f"  breakaway  A_min = f/|c0|           = {th['a_breakaway_mps2']:.4f} m/s^2")
    print(f"  reach top  A     = (pi/2)(f+b<td>)/|c0| = {th['a_to_top_mps2']:.4f} m/s^2")
    print(f"  viscous cap at A=1.0: thetadot < {viscous_speed_cap(1.0, c0, b, f):.2f} rad/s")

    hdr = (f"\n{'A m/s^2':>9} {'vs floor':>9} {'E_pk/E_top':>11} {'pos_work':>9} "
           f"{'dE mJ':>8} {'W_drive mJ':>11} {'delivered':>10}  guard")
    print(hdr)
    print("-" * (len(hdr) - 1))

    rows = []
    for A in args.amplitudes:
        params = EnergyScheduleParams(a_slow=float(A), a_sharp=float(A),
                                      e_center=0.5, e_width=0.1,
                                      db_slow=0.005, db_sharp=0.005,
                                      a_seed=max(float(A), 0.6), t_seed_s=0.25)
        r = run_energy_scheduled_trial(
            model, params, arm_q=arm_q, hanging_angle=hanging,
            inverted_angle=inverted, constants=constants, coupling_c0=c0,
            config_path=Path(args.config), controller_kind=str(args.controller_kind),
            transport_axis_index=axis, duration_s=float(args.duration_s),
        )
        e_top = float(constants.e_top_j)
        d_e = float(r["e_peak_over_e_top"]) * e_top
        w = float(r["work_by_drive_j"])
        # Fraction of the COMMANDED work that shows up as energy in the pendulum.
        # ~1 would mean the arm delivered the command; well below 1 means the
        # shortfall is in the arm, not the hinge.
        delivered = (d_e / w) if abs(w) > 1e-12 else float("nan")
        diss = float(r.get("dissipated_j", float("nan")))
        # Budget residual: W_drive - dE - dissipated. Near zero => the energy is
        # fully accounted for at the hinge and nothing is unexplained.
        residual = (float(r.get("e_final_j", float("nan")))
                    - float(r.get("e_initial_j", float("nan")))) - (w - diss)
        rows.append({**r, "amplitude_mps2": float(A),
                     "delivered_fraction": delivered,
                     "budget_residual_j": residual})
        print(f"{A:9.3f} {A/th['a_breakaway_mps2']:8.2f}x {r['e_peak_over_e_top']:11.4f} "
              f"{r['positive_work_fraction']:9.3f} {1000*d_e:8.3f} {1000*w:11.3f} "
              f"{delivered:10.3f}  {'-' if not r['guard_fired'] else r['guard_reason']}")

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "arm_q": arm_q.tolist(), "pendulum_xml": str(args.pendulum_xml),
            "config": str(args.config), "coupling_c0": c0,
            "viscous_damping": b, "coulomb_frictionloss": f,
            "i_pivot_kgm2": float(constants.i_pivot_kgm2),
            "omega_natural_radps": float(constants.omega_natural_radps),
            "e_top_j": float(constants.e_top_j),
            "thetadot_at_e_top_radps": thetadot_top,
            "derived": th,
            "rows": [{k: v for k, v in r.items() if k != "history"} for r in rows],
            "provenance": provenance.as_dict(),
        }
        args.output_json.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
