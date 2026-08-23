#!/usr/bin/env python3
"""Cheap kinematic feasibility pre-check for an X-transport task with a
bounded Y/Z corridor and near-fixed orientation, on the REAL UR5e model.

Before building a real torque-level controller that enforces this task
(``x_task_yz_corridor_qp``), this script answers a much cheaper geometric
question first: is there even a CONTINUOUS path of joint configurations that
satisfies

    p_x(q) = x_d                                   (swept over --x-range)
    y0 - hw_y <= p_y(q) <= y0 + hw_y                (Y corridor)
    z0 - hw_z <= p_z(q) <= z0 + hw_z                (Z corridor)
    ||orientation_error(q, R0)|| <= --orientation-tol-rad
    q_min <= q <= q_max                             (real UR5e joint limits)

at the given start pose? Each point is solved with ``scipy.optimize.minimize``
(SLSQP), WARM-STARTED from the previous solution (continuation) -- this is
deliberately not a pointwise IK feasibility check: continuation is what
exposes a path that is only feasible as a set of disconnected per-point
solutions (e.g. across a branch cut / IK multiplicity) that a real
closed-loop controller could never actually follow continuously.

Pure kinematics -- no dynamics, no controller, no MuJoCo stepping. Two
independent continuation branches run from x_d=0 (the start pose, which is
feasible by construction): one ascending through +x-range, one descending
through -x-range, each warm-started from the x_d=0 solution.

At every solved point this records (not just pass/fail):
    sigma_min(J(q))              -- smallest singular value of the full 6x6
                                     Jacobian, i.e. conditioning ALONG the
                                     path, not just at the endpoints.
    joint-limit margins           -- min(q - q_min), min(q_max - q), which
                                     joint, using the REAL jointrange values
                                     compiled into assets/ur5e_torque/ur5e_torque.xml
                                     (do not guess these).
    gravity-torque proxy          -- |tau_gravity(q)| / tau_limit per joint.
                                     Explicitly a CHEAP PROXY for torque
                                     margin, not a dynamic-feasibility
                                     guarantee -- it ignores inertial/
                                     Coriolis/actuation-direction effects a
                                     real controller must also handle.
    SLSQP convergence/feasibility  -- solver success flag AND a direct
                                     re-check of every constraint against the
                                     solved q (SLSQP can report "success"
                                     with a small residual constraint
                                     violation; both are recorded).

Example
-------
    python tools/diagnostics/x_task_yz_corridor_ik_feasibility_check.py \\
        --start-q-rad -2.3688 -2.1801 -1.8838 -0.7962 0.004714693 0.0206 \\
        --x-range -0.06 0.06 --y-corridor-half-width-m 0.05 \\
        --z-corridor-half-width-m 0.05 --json /tmp/x_task_yz_corridor_ik_feasibility.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from controller_core.kinematics_utils import (  # noqa: E402
    orientation_error_vec_wxyz,
    rotmat_to_quat,
)
from mujoco_ur5e_tools import UR5E_JOINT_ORDER, compute_gravity_torque, torque_limit_vector  # noqa: E402
from simulation.ur5e_mujoco_torque import load_model, make_mujoco_jacobian_fn  # noqa: E402

SCENE_XML = REPO_ROOT / "assets" / "ur5e_torque" / "scene.xml"

#: Real deployment pose. ``wrist_2 ~= 0.0206 rad`` sits close to the UR5e's
#: classic wrist singularity (cond(J) ~= 1396 at this exact q, measured
#: below at x_d=0).
ARM_Q0 = np.array(
    [-2.3688, -2.1801, -1.8838, -0.7962, 0.004714693, 0.0206],
    dtype=np.float64,
)


def fk(model: mujoco.MjModel, data: mujoco.MjData, site_id: int, joint_ids: list[int], q: np.ndarray):
    """Forward kinematics only (no velocities) -- position + quaternion."""
    q = np.asarray(q, dtype=np.float64).reshape(6)
    for idx, jid in enumerate(joint_ids):
        data.qpos[int(model.jnt_qposadr[jid])] = float(q[idx])
    mujoco.mj_forward(model, data)
    pos = np.asarray(data.site_xpos[site_id], dtype=np.float64).copy()
    rotmat = np.asarray(data.site_xmat[site_id], dtype=np.float64).reshape(3, 3).copy()
    quat = rotmat_to_quat(rotmat)
    return pos, quat


@dataclass
class PointResult:
    x_d: float
    target_delta_x_m: float
    success: bool
    feasible: bool
    message: str
    q_sol: list[float] = field(default_factory=list)
    p_x: float = float("nan")
    p_y: float = float("nan")
    p_z: float = float("nan")
    y_violation_m: float = 0.0
    z_violation_m: float = 0.0
    orientation_error_rad: float = float("nan")
    orientation_violation_rad: float = 0.0
    sigma_min_j: float = float("nan")
    sigma_max_j: float = float("nan")
    cond_j: float = float("nan")
    min_lower_margin_rad: float = float("nan")
    min_lower_margin_joint: str = ""
    min_upper_margin_rad: float = float("nan")
    min_upper_margin_joint: str = ""
    tau_gravity_nm: list[float] = field(default_factory=list)
    tau_gravity_frac_of_limit: list[float] = field(default_factory=list)
    max_tau_gravity_frac: float = float("nan")
    max_tau_gravity_frac_joint: str = ""


def solve_point(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_id: int,
    joint_ids: list[int],
    *,
    q_prev: np.ndarray,
    x_d: float,
    y0: float,
    z0: float,
    quat0: np.ndarray,
    y_half_width: float,
    z_half_width: float,
    orientation_tol_rad: float,
    q_min: np.ndarray,
    q_max: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    def px(q: np.ndarray) -> float:
        pos, _ = fk(model, data, site_id, joint_ids, q)
        return float(pos[0])

    def py(q: np.ndarray) -> float:
        pos, _ = fk(model, data, site_id, joint_ids, q)
        return float(pos[1])

    def pz(q: np.ndarray) -> float:
        pos, _ = fk(model, data, site_id, joint_ids, q)
        return float(pos[2])

    def orient_err_norm(q: np.ndarray) -> float:
        _, quat = fk(model, data, site_id, joint_ids, q)
        return float(np.linalg.norm(orientation_error_vec_wxyz(quat0, quat)))

    objective = lambda q: float(np.sum((q - q_prev) ** 2))  # noqa: E731
    objective_grad = lambda q: 2.0 * (q - q_prev)  # noqa: E731

    constraints = [
        {"type": "eq", "fun": lambda q: px(q) - float(x_d)},
        {"type": "ineq", "fun": lambda q: py(q) - (y0 - y_half_width)},
        {"type": "ineq", "fun": lambda q: (y0 + y_half_width) - py(q)},
        {"type": "ineq", "fun": lambda q: pz(q) - (z0 - z_half_width)},
        {"type": "ineq", "fun": lambda q: (z0 + z_half_width) - pz(q)},
        {"type": "ineq", "fun": lambda q: orientation_tol_rad - orient_err_norm(q)},
    ]
    bounds = list(zip(q_min.tolist(), q_max.tolist()))

    res = minimize(
        objective,
        q_prev,
        jac=objective_grad,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 300, "ftol": 1e-10},
    )
    q_sol = np.asarray(res.x, dtype=np.float64).reshape(6)

    pos, quat = fk(model, data, site_id, joint_ids, q_sol)
    y_violation = max(0.0, (y0 - y_half_width) - pos[1], pos[1] - (y0 + y_half_width))
    z_violation = max(0.0, (z0 - z_half_width) - pos[2], pos[2] - (z0 + z_half_width))
    orient_err = float(np.linalg.norm(orientation_error_vec_wxyz(quat0, quat)))
    orient_violation = max(0.0, orient_err - orientation_tol_rad)
    x_violation = abs(pos[0] - float(x_d))
    feasible = bool(
        res.success
        and x_violation < 1e-6
        and y_violation < 1e-6
        and z_violation < 1e-6
        and orient_violation < 1e-6
    )

    info = {
        "success": bool(res.success),
        "feasible": feasible,
        "message": str(res.message),
        "pos": pos,
        "quat": quat,
        "y_violation_m": float(y_violation),
        "z_violation_m": float(z_violation),
        "orientation_error_rad": orient_err,
        "orientation_violation_rad": float(orient_violation),
    }
    return q_sol, info


def run_branch(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_id: int,
    joint_ids: list[int],
    jac_fn,
    *,
    q_seed: np.ndarray,
    x_deltas: list[float],
    x0: float,
    y0: float,
    z0: float,
    quat0: np.ndarray,
    y_half_width: float,
    z_half_width: float,
    orientation_tol_rad: float,
    q_min: np.ndarray,
    q_max: np.ndarray,
) -> list[PointResult]:
    q_prev = q_seed.copy()
    results: list[PointResult] = []
    for x_delta in x_deltas:
        x_d = x0 + float(x_delta)
        q_sol, info = solve_point(
            model, data, site_id, joint_ids,
            q_prev=q_prev, x_d=x_d, y0=y0, z0=z0, quat0=quat0,
            y_half_width=y_half_width, z_half_width=z_half_width,
            orientation_tol_rad=orientation_tol_rad, q_min=q_min, q_max=q_max,
        )

        jac = jac_fn(q_sol)
        singular_values = np.linalg.svd(jac, compute_uv=False)
        sigma_min = float(singular_values[-1])
        sigma_max = float(singular_values[0])
        cond_j = float(sigma_max / sigma_min) if sigma_min > 0.0 else float("inf")

        lower_margin = q_sol - q_min
        upper_margin = q_max - q_sol
        lo_idx = int(np.argmin(lower_margin))
        hi_idx = int(np.argmin(upper_margin))

        tau_g = compute_gravity_torque(model, q_sol, joint_ids)
        tau_limits = torque_limit_vector()
        tau_frac = np.abs(tau_g) / tau_limits
        frac_idx = int(np.argmax(tau_frac))

        pr = PointResult(
            x_d=float(x_d),
            target_delta_x_m=float(x_delta),
            success=info["success"],
            feasible=info["feasible"],
            message=info["message"],
            q_sol=q_sol.tolist(),
            p_x=float(info["pos"][0]),
            p_y=float(info["pos"][1]),
            p_z=float(info["pos"][2]),
            y_violation_m=info["y_violation_m"],
            z_violation_m=info["z_violation_m"],
            orientation_error_rad=info["orientation_error_rad"],
            orientation_violation_rad=info["orientation_violation_rad"],
            sigma_min_j=sigma_min,
            sigma_max_j=sigma_max,
            cond_j=cond_j,
            min_lower_margin_rad=float(lower_margin[lo_idx]),
            min_lower_margin_joint=UR5E_JOINT_ORDER[lo_idx],
            min_upper_margin_rad=float(upper_margin[hi_idx]),
            min_upper_margin_joint=UR5E_JOINT_ORDER[hi_idx],
            tau_gravity_nm=tau_g.tolist(),
            tau_gravity_frac_of_limit=tau_frac.tolist(),
            max_tau_gravity_frac=float(tau_frac[frac_idx]),
            max_tau_gravity_frac_joint=UR5E_JOINT_ORDER[frac_idx],
        )
        results.append(pr)
        # Continuation: always warm-start the next point from THIS solution,
        # even if this one failed -- that is what continuation means, and it
        # is what would expose a solver getting stuck vs. recovering.
        q_prev = q_sol
    return results


def build_x_deltas(x_range: tuple[float, float], step_m: float) -> list[float]:
    xmin, xmax = float(x_range[0]), float(x_range[1])
    if xmin > 0.0 or xmax < 0.0:
        raise ValueError(f"--x-range must straddle 0 (the start pose); got {x_range!r}")
    n_pos = int(round(xmax / step_m)) if xmax > 0.0 else 0
    n_neg = int(round(-xmin / step_m)) if xmin < 0.0 else 0
    pos_deltas = [round(i * step_m, 10) for i in range(1, n_pos + 1)]
    neg_deltas = [round(-i * step_m, 10) for i in range(1, n_neg + 1)]
    return neg_deltas, pos_deltas


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start-q-rad", nargs=6, type=float, default=list(ARM_Q0))
    parser.add_argument("--x-range", nargs=2, type=float, default=[-0.06, 0.06], metavar=("XMIN", "XMAX"))
    parser.add_argument("--y-corridor-half-width-m", type=float, default=0.05)
    parser.add_argument("--z-corridor-half-width-m", type=float, default=0.05)
    parser.add_argument("--step-m", type=float, default=0.005)
    parser.add_argument(
        "--orientation-tol-rad", type=float, default=0.05,
        help="Max allowed ||orientation_error_vec_wxyz|| relative to the start orientation.",
    )
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)

    model, data, site_id, joint_ids, _actuator_ids = load_model(SCENE_XML)
    jac_fn = make_mujoco_jacobian_fn(model, site_id, joint_ids)

    q_start = np.asarray(args.start_q_rad, dtype=np.float64).reshape(6)
    q_min = np.array([float(model.jnt_range[jid, 0]) for jid in joint_ids], dtype=np.float64)
    q_max = np.array([float(model.jnt_range[jid, 1]) for jid in joint_ids], dtype=np.float64)

    pos0, quat0 = fk(model, data, site_id, joint_ids, q_start)
    x0, y0, z0 = float(pos0[0]), float(pos0[1]), float(pos0[2])
    jac0 = jac_fn(q_start)
    sv0 = np.linalg.svd(jac0, compute_uv=False)
    print(f"Start pose: x0={x0:.6f} y0={y0:.6f} z0={z0:.6f} m")
    print(f"Start Jacobian: sigma_min={sv0[-1]:.6e} sigma_max={sv0[0]:.6e} cond(J)={sv0[0]/sv0[-1]:.4e}")

    neg_deltas, pos_deltas = build_x_deltas(tuple(args.x_range), float(args.step_m))

    common_kwargs = dict(
        model=model, data=data, site_id=site_id, joint_ids=joint_ids, jac_fn=jac_fn,
        x0=x0, y0=y0, z0=z0, quat0=quat0,
        y_half_width=float(args.y_corridor_half_width_m),
        z_half_width=float(args.z_corridor_half_width_m),
        orientation_tol_rad=float(args.orientation_tol_rad),
        q_min=q_min, q_max=q_max,
    )

    pos_results = run_branch(q_seed=q_start, x_deltas=pos_deltas, **common_kwargs)
    neg_results = run_branch(q_seed=q_start, x_deltas=neg_deltas, **common_kwargs)

    # Full path, ordered by x_delta ascending, with the x_delta=0 start point
    # included explicitly (feasible by construction).
    start_point = PointResult(
        x_d=x0, target_delta_x_m=0.0, success=True, feasible=True, message="start pose",
        q_sol=q_start.tolist(), p_x=x0, p_y=y0, p_z=z0,
        orientation_error_rad=0.0, sigma_min_j=float(sv0[-1]), sigma_max_j=float(sv0[0]),
        cond_j=float(sv0[0] / sv0[-1]),
    )
    ordered = list(reversed(neg_results)) + [start_point] + pos_results

    print(f"\n{'x_delta':>9} {'success':>8} {'feasible':>9} {'sigma_min':>12} {'cond(J)':>10} "
          f"{'y_viol':>9} {'z_viol':>9} {'orient_err':>11} {'lo_margin':>10} {'hi_margin':>10} {'max_g_frac':>11}")
    for r in ordered:
        print(
            f"{r.target_delta_x_m:>9.4f} {str(r.success):>8} {str(r.feasible):>9} "
            f"{r.sigma_min_j:>12.4e} {r.cond_j:>10.3e} {r.y_violation_m:>9.5f} {r.z_violation_m:>9.5f} "
            f"{r.orientation_error_rad:>11.5f} {r.min_lower_margin_rad:>10.4f} {r.min_upper_margin_rad:>10.4f} "
            f"{r.max_tau_gravity_frac:>11.4f}"
        )

    all_feasible = all(r.feasible for r in ordered)
    first_break_pos = next((r for r in pos_results if not r.feasible), None)
    first_break_neg = next((r for r in neg_results if not r.feasible), None)
    sigma_mins = [r.sigma_min_j for r in ordered if np.isfinite(r.sigma_min_j)]
    lower_margins = [r.min_lower_margin_rad for r in ordered if np.isfinite(r.min_lower_margin_rad)]
    upper_margins = [r.min_upper_margin_rad for r in ordered if np.isfinite(r.min_upper_margin_rad)]
    g_fracs = [r.max_tau_gravity_frac for r in ordered if np.isfinite(r.max_tau_gravity_frac)]

    summary = {
        "start_q_rad": q_start.tolist(),
        "start_pos_xyz_m": [x0, y0, z0],
        "start_sigma_min_j": float(sv0[-1]),
        "start_cond_j": float(sv0[0] / sv0[-1]),
        "x_range_m": list(args.x_range),
        "step_m": float(args.step_m),
        "y_corridor_half_width_m": float(args.y_corridor_half_width_m),
        "z_corridor_half_width_m": float(args.z_corridor_half_width_m),
        "orientation_tol_rad": float(args.orientation_tol_rad),
        "n_points": len(ordered),
        "all_feasible": bool(all_feasible),
        "first_infeasible_x_delta_positive_branch": (
            None if first_break_pos is None else first_break_pos.target_delta_x_m
        ),
        "first_infeasible_x_delta_negative_branch": (
            None if first_break_neg is None else first_break_neg.target_delta_x_m
        ),
        "sigma_min_j_min": min(sigma_mins) if sigma_mins else None,
        "sigma_min_j_max": max(sigma_mins) if sigma_mins else None,
        "sigma_min_j_mean": float(np.mean(sigma_mins)) if sigma_mins else None,
        "min_lower_margin_rad_overall": min(lower_margins) if lower_margins else None,
        "min_upper_margin_rad_overall": min(upper_margins) if upper_margins else None,
        "max_gravity_torque_frac_of_limit_overall": max(g_fracs) if g_fracs else None,
    }
    print("\n=== summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    if args.json:
        payload = {
            "summary": summary,
            "points": [asdict(r) for r in ordered],
        }
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\nwrote {args.json}")

    return 0 if all_feasible else 1


if __name__ == "__main__":
    raise SystemExit(main())
