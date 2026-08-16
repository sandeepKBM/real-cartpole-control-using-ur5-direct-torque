#!/usr/bin/env python3
"""Deliverable 2: max clean cart speed along TOOL Y at ARM_Q0, and the
speed-vs-orientation-guard tradeoff curve.

At ARM_Q0 the wrist is essentially at the classic UR wrist_2=0 singularity
(cond(J6)=1395.76). This kills ORIENTATION authority (a full-rank task tries
to hold 6 DOF through a near-singular Jacobian column), not X/Z position
authority -- so the achievable cart speed along the pumping direction is
gated by how tightly orientation error is enforced. This script bisects, for
each of several ``max_orientation_error_rad`` guard thresholds, the maximum
peak cart speed (a trapezoidal ramp: accelerate at a fixed a_max, hold at
v_peak) that completes ``duration_s`` with NO guard trip, and records which
guard bound the ceiling.

Uses the SAME rotated-frame wrapper as every other script in this pipeline
(tools/diagnostics/pendulum_toolY_common.py) -- motion is commanded along
tool Y, not world X, per this task's spec.
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


def run_speed_trial(
    model: mujoco.MjModel,
    *,
    v_peak: float,
    a_max: float,
    duration_s: float,
    max_orientation_error_rad: float,
    start_at_hanging: bool = True,
) -> dict:
    data = mujoco.MjData(model)
    data.qpos[:6] = C.ARM_Q0
    data.qvel[:] = 0.0
    pend_jid, hub_bid, site_id = C.hinge_ids(model)
    joint_ids = C.joint_ids_for(model)
    if start_at_hanging:
        hanging, _inverted = C.resolve_equilibria(model)
        data.qpos[model.jnt_qposadr[pend_jid]] = hanging
    mujoco.mj_forward(model, data)

    tool_x, tool_y, tool_z = C.tool_frame_world(data, site_id)
    R = C.pumping_rotation_matrix(tool_x, tool_y, tool_z)

    config = C.load_config()
    ctrl_cfg = dict(config["controller"])
    safety_cfg = dict(ctrl_cfg.get("safety", {}))
    safety_cfg["max_orientation_error_rad"] = float(max_orientation_error_rad)
    ctrl_cfg["safety"] = safety_cfg

    state0, adapter = C.build_rotated_initial_state_and_adapter(
        model, data, site_id, joint_ids, R=R,
        controller_cfg=ctrl_cfg,
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
    peak_qd = 0.0
    peak_orient_err = 0.0

    for step in range(n_steps):
        t = step * C.CONTROL_DT
        target_x_vel = float(np.clip(target_x_vel + a_max * C.CONTROL_DT, -v_peak, v_peak))
        target_x = target_x + target_x_vel * C.CONTROL_DT
        u = a_max if abs(target_x_vel) < v_peak - 1e-9 else 0.0
        state, ee_world = C.build_step_state(
            model, data, site_id=site_id, joint_ids=joint_ids, time_s=t,
            target_x=target_x, target_x_vel=target_x_vel, target_x_accel=u,
            reference_quat=reference_quat, R=R,
        )
        tau, diag = adapter.step(state=state)
        peak_orient_err = max(peak_orient_err, float(diag["orientation_error_norm"]))
        if not diag["safety_ok"]:
            guard_fired = True
            guard_reason = diag["safety_reason"]
            first_guard_t = t
            break
        data.ctrl[:6] = tau
        mujoco.mj_step(model, data)
        steps_done += 1
        peak_qd = max(peak_qd, float(np.max(np.abs(data.qvel[:6]))))

    return {
        "v_peak": v_peak, "a_max": a_max, "duration_s": duration_s,
        "max_orientation_error_rad": max_orientation_error_rad,
        "guard_fired": guard_fired, "guard_reason": guard_reason, "first_guard_t": first_guard_t,
        "steps_done": steps_done, "n_steps": n_steps,
        "peak_abs_qd": peak_qd, "peak_orientation_error_rad": peak_orient_err,
    }


def bisect_max_clean_speed(
    model, *, max_orientation_error_rad: float, a_max: float, duration_s: float,
    v_lo: float = 0.0005, v_hi: float = 1.5, iters: int = 14,
) -> dict:
    lo_result = run_speed_trial(model, v_peak=v_lo, a_max=a_max, duration_s=duration_s,
                                 max_orientation_error_rad=max_orientation_error_rad)
    if lo_result["guard_fired"]:
        return {"max_clean_speed_mps": 0.0, "binding_guard": lo_result["guard_reason"], "lo_trial": lo_result}

    best_clean = lo_result
    lo, hi = v_lo, v_hi
    last_dirty = None
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        res = run_speed_trial(model, v_peak=mid, a_max=a_max, duration_s=duration_s,
                               max_orientation_error_rad=max_orientation_error_rad)
        if res["guard_fired"]:
            hi = mid
            last_dirty = res
        else:
            lo = mid
            best_clean = res
    return {
        "max_clean_speed_mps": lo,
        "binding_guard": last_dirty["guard_reason"] if last_dirty else None,
        "best_clean_trial": best_clean,
        "first_dirty_trial": last_dirty,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a-max", type=float, default=5.0)
    parser.add_argument("--duration-s", type=float, default=1.2)
    parser.add_argument("--iters", type=int, default=14)
    parser.add_argument(
        "--thresholds", type=float, nargs="+",
        default=[0.01, 0.02, 0.05, 0.1, 0.15, 0.25, 0.35, 0.5, 1.0, 3.0],
        help="max_orientation_error_rad values to sweep (rad).",
    )
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args(argv)

    model = C.build_model()
    rows = []
    print(f"{'thresh(rad)':>12} {'max_clean_v(m/s)':>18} {'binding_guard':>40}")
    for thresh in args.thresholds:
        res = bisect_max_clean_speed(
            model, max_orientation_error_rad=thresh, a_max=args.a_max,
            duration_s=args.duration_s, iters=args.iters,
        )
        rows.append({"threshold_rad": thresh, **res})
        print(f"{thresh:>12.4f} {res['max_clean_speed_mps']:>18.4f} {str(res['binding_guard']):>40}")

    if args.output_json:
        # Trim history-heavy sub-dicts (none here, but keep consistent).
        with Path(args.output_json).open("w") as fp:
            json.dump(rows, fp, indent=2, default=str)
        print("wrote", args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
