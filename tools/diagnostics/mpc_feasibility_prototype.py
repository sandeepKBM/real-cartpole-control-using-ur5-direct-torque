"""Standalone feasibility prototype: short-horizon (MPC-style) extension of
``controller_core.cartesian_velocity_controller``'s ``ik_seeded_resolution``
solve.

SCOPE: this is a THROWAWAY MEASUREMENT SCRIPT for a go/no-go decision. It
deliberately does NOT touch ``modes.py`` or any production file -- it
re-implements the same QP structure in a horizon form so the two can be run
side by side against the same kinematic-only sim harness
(``velocity_gain_tuning/``) and the same guards.

What ``compute_ik_seeded`` does today (per control cycle):

    q_k <- q_rest
    repeat ik_iterations times:
        J_k, err_k <- FK/Jacobian at q_k against the CURRENT commanded pose
        dq        <- argmin reg*||dq||^2 + task_w*||J_task_k dq - err_k||^2
        (optional null-space deviation clip)
        q_k <- q_k + dq
    q_target <- q_k;  qd = ik_joint_gain * (q_target - q_current)

That is a single-point solve: the only configuration whose Jacobian ever
enters the cost is the one the Newton iterate happens to be sitting on, and
the only pose target is the instantaneous one. It has no way to express "this
solution is fine now but routes me through a near-singular configuration."

What this prototype adds -- a CONTINUATION HORIZON. Instead of one target it
solves for a whole joint-space path

    q_0 = q_rest,  q_{i+1} = q_i + u_i,  i = 0 .. N-1

whose stage targets p_1 .. p_N interpolate linearly in Cartesian space from
FK(q_rest) to the commanded p_des (so stage N still hits the real commanded
target exactly, and the intermediate stages are genuine approximations of the
configurations the arm passes through on its way there -- the arm really does
travel along that straight Cartesian line during a min-jerk X move). Cost:

    sum_i  task_w*||J_task(q_i) (q_i - q_i^nom) - task_err_i||^2      (task)
         + reg*||u_i||^2                                              (step size)
         + w_cond * max(0, sigma_floor - sigma_min(J_full(q_i)))^2    (NEW)

The third term is the genuinely new ingredient: a hinge barrier on the
smallest singular value of the FULL 6x6 Jacobian at every configuration along
the path. ``sigma_min(J_full)`` is the right quantity to penalise here (not
the reduced task Jacobian's conditioning) because the documented failure is a
blowup of ``pinv(J_full) @ xd_cmd`` at ``wrist_2 -> 0`` -- both in the
evaluation env's joint-velocity guard and, on real hardware, inside the
robot's own ``speedL`` resolution.

Because ``q_i`` depends on ``u_0 .. u_{i-1}``, the stage costs are genuinely
coupled and the problem is nonlinear (J re-linearised at each stage each
outer iteration) -- solved SQP-style: linearise the whole horizon, solve one
box-constrained QP over the stacked ``(u_0..u_{N-1})`` with a trust-region
box, update the nominal path, repeat ``sqp_iterations`` times. With N=1 and
w_cond=0 this reduces EXACTLY to today's Newton loop (asserted by
``--mode selfcheck``), which is what makes the comparison apples-to-apples.

Subcommands
-----------
  profile     wall-clock cost of the baseline solve and of the horizon solve
              vs. (N, sqp_iterations, w_cond)
  validate    run the two documented failure cases through the same kinematic
              sim, baseline vs. horizon, and report guard outcomes
  selfcheck   assert the N=1/w_cond=0 horizon solve reproduces the production
              ``compute_ik_seeded`` bit-for-bit-ish

Nothing here is imported by any production module.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, replace

import numpy as np

from controller_core.box_qp import build_weighted_least_squares_qp, solve_box_qp
from controller_core.cartesian_velocity_controller import (
    CartesianVelocityConfig,
    CartesianVelocityController,
)
from controller_core.cartesian_velocity_controller.math_utils import _damped_pinv
from controller_core.cartesian_velocity_controller.modes import compute_ik_seeded
from controller_core.kinematics_utils import null_space_basis, swing_twist_axis_error
from hardware.local_dynamics import LocalMujocoDynamics
from simulation.ur5e_mujoco_torque import x_profile_target
from velocity_gain_tuning.envs.velocity_transport_env import (
    VelocityTransportEnv,
    VelocityTransportEnvConfig,
    action_to_gains,
)
from velocity_gain_tuning.optimize import FAST_MOVE_DURATION_S, run_episode
from velocity_gain_tuning.poses import scenario_by_name

# This lane's reproducible fixed-gain best (outputs/velocity_gain_tuning/
# search_result_nullspace_v2_20260806_194402.json), 104/128 on the standard
# grid -- the same vector docs/status/task_priority_orientation_hanging_
# 2026-08-06.md benchmarks against.
BASELINE_ACTION = np.array(
    [
        -0.5452930656195676,
        -0.31201103390079576,
        0.19603435480606923,
        -0.40319481871903273,
        0.6634521666673519,
        -0.29877165734428546,
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class HorizonParams:
    n_horizon: int = 3
    sqp_iterations: int = 6
    task_w: float = 1.0e4
    reg: float = 1.0e-6
    cond_weight: float = 0.0
    sigma_floor: float = 0.05
    trust_radius_rad: float = np.inf
    max_joint_deviation_rad: float | None = None
    fd_eps: float = 1.0e-5


def _sigma_min(jac: np.ndarray) -> float:
    return float(np.linalg.svd(jac, compute_uv=False)[-1])


def horizon_ik_solve(
    fk_jacobian_fn,
    p_des: np.ndarray,
    quat0: np.ndarray,
    q_rest: np.ndarray,
    selected: list[int],
    hp: HorizonParams,
) -> tuple[np.ndarray, dict]:
    """Returns (q_target, diagnostics). ``q_target`` is the horizon's TERMINAL
    state q_N, i.e. the configuration that achieves the commanded pose -- the
    same object today's solve returns, so the downstream joint-space P law is
    untouched. The horizon's only job is to pick WHICH of the (redundant) IK
    solutions achieving that pose to return, informed by the conditioning of
    every configuration on the interpolated path leading to it."""
    n = int(hp.n_horizon)
    q_rest = np.asarray(q_rest, dtype=np.float64).reshape(6)
    p_des = np.asarray(p_des, dtype=np.float64).reshape(3)

    q_lo = np.full(6, -2.0 * np.pi, dtype=np.float64)
    q_hi = np.full(6, 2.0 * np.pi, dtype=np.float64)

    p_start, _, _ = fk_jacobian_fn(q_rest)
    p_start = np.asarray(p_start, dtype=np.float64).reshape(3)

    # Stage targets: linear Cartesian interpolation rest -> commanded, with
    # the terminal stage exactly on the commanded target. Orientation target
    # is quat0 at every stage (the task is "hold the reset orientation").
    stage_p = [p_start + (float(i + 1) / n) * (p_des - p_start) for i in range(n)]

    u = np.zeros((n, 6), dtype=np.float64)
    fk_calls = 0
    diag: dict = {}

    for _ in range(max(int(hp.sqp_iterations), 1)):
        # --- forward pass: nominal path + per-stage linearisation ---
        q_nom = np.zeros((n + 1, 6), dtype=np.float64)
        q_nom[0] = q_rest
        for i in range(n):
            q_nom[i + 1] = q_nom[i] + u[i]

        jacs: list[np.ndarray] = []
        errs: list[np.ndarray] = []
        cond_rows: list[tuple[np.ndarray, float]] = []
        for i in range(n):
            qi = q_nom[i + 1]
            p_k, quat_k, jac_k = fk_jacobian_fn(qi)
            fk_calls += 1
            p_k = np.asarray(p_k, dtype=np.float64).reshape(3)
            quat_k = np.asarray(quat_k, dtype=np.float64).reshape(4)
            jac_k = np.asarray(jac_k, dtype=np.float64).reshape(6, 6)
            rot_err = np.array(
                [swing_twist_axis_error(quat0, quat_k, a) for a in range(3)], dtype=np.float64
            )
            task_err_full = np.concatenate([stage_p[i] - p_k, -rot_err])
            jacs.append(jac_k)
            errs.append(task_err_full[selected])

            if hp.cond_weight > 0.0:
                s0 = _sigma_min(jac_k)
                slack = hp.sigma_floor - s0
                if slack > 0.0:
                    # d(slack)/dq = -d(sigma_min)/dq, forward differences.
                    grad = np.zeros(6, dtype=np.float64)
                    for k in range(6):
                        qp = qi.copy()
                        qp[k] += hp.fd_eps
                        _, _, jp = fk_jacobian_fn(qp)
                        fk_calls += 1
                        grad[k] = -(_sigma_min(np.asarray(jp).reshape(6, 6)) - s0) / hp.fd_eps
                    cond_rows.append((grad, slack))
                else:
                    cond_rows.append((np.zeros(6), 0.0))
            else:
                cond_rows.append((np.zeros(6), 0.0))

        # --- build the stacked QP over delta = (du_0 .. du_{N-1}) ---
        # q_i - q_i^nom = sum_{j<i} du_j  =>  selector S_i = [I I .. I 0 .. 0]
        dim = 6 * n
        terms: list[tuple[np.ndarray, np.ndarray, float]] = []
        for i in range(n):
            sel = np.zeros((6, dim), dtype=np.float64)
            for j in range(i + 1):
                sel[:, 6 * j : 6 * (j + 1)] = np.eye(6)
            a_task = jacs[i][selected, :] @ sel
            terms.append((a_task, errs[i], hp.task_w))
            grad, slack = cond_rows[i]
            if hp.cond_weight > 0.0 and slack > 0.0:
                a_cond = (grad @ sel).reshape(1, dim)
                terms.append((a_cond, np.array([-slack]), hp.cond_weight))

        # reg acts on the SQP step (du), exactly as production's reg acts on
        # the Newton step dq -- this is what makes N=1 an EXACT reduction to
        # today's solve (verified by --mode selfcheck). Penalising the
        # cumulative u_j instead was tried first and is a real semantic
        # change: it drives the CUMULATIVE null-space component to zero
        # rather than each step's, selecting a different (equally valid, but
        # not identical) redundant branch -- measured as a 1.5e-2 rad q_target
        # difference at N=1, which would have confounded the whole
        # horizon-vs-no-horizon comparison.
        hess, lin = build_weighted_least_squares_qp(terms, reg=hp.reg, n=dim)
        # Box = production's own joint-limit box (permissive +-2pi, matching
        # the caller-supplies-nothing case), optionally tightened by an SQP
        # trust region. The exact bound VALUES matter more than they look:
        # once the task residual converges, the Hessian's null-space
        # directions are regularised only by reg/task_w ~ 5e-19, so
        # solve_box_qp's initial -H^-1 f blows up there and the answer is
        # decided by where it gets clipped -- with a different box, N=1
        # stops reproducing production after ~3 iterations (measured: exact
        # to 4.4e-16 for 1-3 iterations, 6.3e-2 rad apart by iteration 6).
        # That is itself a confirmation of the documented "redundant
        # component is set by little more than the linear solver's own
        # rounding" finding.
        tr = float(hp.trust_radius_rad)
        lo_box = np.concatenate([np.maximum(-tr, q_lo - q_nom[i + 1]) for i in range(n)])
        hi_box = np.concatenate([np.minimum(tr, q_hi - q_nom[i + 1]) for i in range(n)])
        delta = solve_box_qp(hess, lin, lo_box, hi_box).reshape(n, 6)

        # --- null-space deviation clip, same construction as production ---
        # Applied to the SQP step, per stage, against that stage's own
        # null-space basis and referenced to q_rest -- at N=1 this is
        # literally production's clip on dq (verified by --mode selfcheck).
        if hp.max_joint_deviation_rad is not None:
            max_dev = abs(float(hp.max_joint_deviation_rad))
            for i in range(n):
                basis = null_space_basis(jacs[i][selected, :])
                if basis.shape[1] == 0:
                    continue
                d_prefix = delta[: i + 1].sum(axis=0)  # state change at stage i+1
                c_current = basis.T @ (q_nom[i + 1] - q_rest)
                c_prop = basis.T @ (q_nom[i + 1] + d_prefix - q_rest)
                c_clip = np.clip(c_prop, -max_dev, max_dev)
                row_d = d_prefix - basis @ (basis.T @ d_prefix)
                corrected = row_d + basis @ (c_clip - c_current)
                delta[i] += corrected - d_prefix

        u = u + delta

    q_target = q_rest + u.sum(axis=0)
    _, _, jac_t = fk_jacobian_fn(q_target)
    fk_calls += 1
    diag["fk_calls"] = fk_calls
    diag["sigma_min_terminal"] = _sigma_min(np.asarray(jac_t).reshape(6, 6))
    diag["sigma_min_path"] = float(
        min(_sigma_min(j) for j in jacs) if jacs else np.nan
    )
    return q_target, diag


# --------------------------------------------------------------------------
# episode runner -- mirrors VelocityTransportEnv.step() exactly, but with the
# q_target source swappable. Validated against the real env in --mode validate
# (the "baseline_env" row must match "baseline_replica").
# --------------------------------------------------------------------------


@dataclass
class EpisodeOut:
    guard_reason: str | None
    achieved_x_delta_m: float
    orientation_error: float
    max_abs_qd_radps: float
    min_sigma_min: float
    min_abs_wrist2: float
    steps: int
    solve_ms_mean: float


def run_episode_custom(
    dyn: LocalMujocoDynamics,
    env_cfg: VelocityTransportEnvConfig,
    gains: dict,
    scenario_name: str,
    target_x_delta_m: float,
    move_duration_s: float,
    solver,
) -> EpisodeOut:
    scenario = scenario_by_name(scenario_name)
    q = scenario.q0.copy()
    p0, quat0, _ = dyn.fk_and_jacobian(q)
    p0 = np.asarray(p0, dtype=np.float64).reshape(3).copy()
    quat0 = np.asarray(quat0, dtype=np.float64).reshape(4).copy()
    x0 = float(p0[0])
    q_rest = q.copy()
    dt = 1.0 / env_cfg.rate_hz

    rot_flags = [env_cfg.task_dim_rx, env_cfg.task_dim_ry, env_cfg.task_dim_rz]
    selected = [0, 1, 2] + [3 + i for i, on in enumerate(rot_flags) if on]

    t_s = 0.0
    settled = 0
    guard_reason = None
    max_abs_qd = 0.0
    min_sig = np.inf
    min_w2 = np.inf
    steps = 0
    solve_times: list[float] = []
    orientation_error = 0.0
    achieved = 0.0

    while True:
        target_x, target_x_vel = x_profile_target(
            "min_jerk_move_hold", x0, target_x_delta_m, t_s, env_cfg.duration_s,
            move_duration_s=move_duration_s,
        )
        p, quat, jac = dyn.fk_and_jacobian(q)
        p = np.asarray(p, dtype=np.float64).reshape(3)
        jac = np.asarray(jac, dtype=np.float64).reshape(6, 6)
        p_des = p0.copy()
        p_des[0] = float(target_x)

        t0 = time.perf_counter()
        q_target = solver(p_des, quat0, q_rest, selected, gains)
        solve_times.append((time.perf_counter() - t0) * 1e3)

        qd_joint = gains["ik_joint_gain"] * (q_target - q)
        xd_cmd = jac @ qd_joint
        v, w = xd_cmd[:3], xd_cmd[3:]
        vn = float(np.linalg.norm(v))
        if vn > env_cfg.max_lin_speed_mps and vn > 1e-9:
            v = v * (env_cfg.max_lin_speed_mps / vn)
        wn = float(np.linalg.norm(w))
        if wn > env_cfg.max_ang_speed_radps and wn > 1e-9:
            w = w * (env_cfg.max_ang_speed_radps / wn)
        xd_cmd = np.concatenate([v, w])

        qd = _damped_pinv(jac, env_cfg.qd_estimate_damping) @ xd_cmd
        qd_max = float(np.max(np.abs(qd)))
        max_abs_qd = max(max_abs_qd, qd_max)
        min_sig = min(min_sig, _sigma_min(jac))
        min_w2 = min(min_w2, abs(float(q[4])))

        y_drift = abs(float(p[1] - p0[1]))
        z_drift = abs(float(p[2] - p0[2]))
        ortho = max(y_drift, z_drift)
        orientation_error = float(
            np.linalg.norm([swing_twist_axis_error(quat0, quat, i) for i in range(3)])
        )
        x_error = float(target_x - p[0])
        achieved = float(p[0] - x0)
        steps += 1

        if qd_max > env_cfg.max_joint_velocity_radps:
            guard_reason = f"joint_velocity_guard: {qd_max:.4f} > {env_cfg.max_joint_velocity_radps}"
        elif ortho > env_cfg.max_abs_orthogonal_drift_m:
            guard_reason = f"orthogonal_drift_guard: {ortho:.4f} > {env_cfg.max_abs_orthogonal_drift_m}"
        elif orientation_error > env_cfg.max_orientation_error_rad:
            guard_reason = f"orientation_guard: {orientation_error:.4f} > {env_cfg.max_orientation_error_rad}"
        if guard_reason is not None:
            break

        q = q + qd * dt
        t_s += dt
        settled = settled + 1 if abs(x_error) < env_cfg.terminal_success_tol_m else 0
        if (t_s >= move_duration_s and settled >= env_cfg.settle_cycles_for_early_stop) or (
            t_s >= env_cfg.duration_s - 1e-12
        ):
            break

    return EpisodeOut(
        guard_reason=guard_reason,
        achieved_x_delta_m=achieved,
        orientation_error=orientation_error,
        max_abs_qd_radps=max_abs_qd,
        min_sigma_min=float(min_sig),
        min_abs_wrist2=float(min_w2),
        steps=steps,
        solve_ms_mean=float(np.mean(solve_times)),
    )


def _baseline_solver(dyn: LocalMujocoDynamics, gains: dict, env_cfg: VelocityTransportEnvConfig):
    cfg = CartesianVelocityConfig(
        reduced_task_dims=False,
        split_base_wrist_task=False,
        ik_seeded_resolution=True,
        ik_iterations=env_cfg.ik_iterations,
        task_dim_rx=env_cfg.task_dim_rx,
        task_dim_ry=env_cfg.task_dim_ry,
        task_dim_rz=env_cfg.task_dim_rz,
        kp_x=gains["kp_x"], kp_y=gains["kp_x"], kp_z=gains["kp_x"],
        kp_rot=gains["kp_rot"],
        ik_joint_gain=gains["ik_joint_gain"],
        pinv_damping=gains["pinv_damping"],
        qp_task_weight=gains["qp_task_weight"],
        ik_max_joint_deviation_rad=gains["ik_max_joint_deviation_rad"],
    )

    def solve(p_des, quat0, q_rest, selected, _g):
        # compute_ik_seeded returns xd_cmd; recover q_target by re-running the
        # same solve with a unit joint gain against q_current=0.
        from controller_core.cartesian_velocity_controller.modes import _ik_newton_solve

        q_lo = np.full(6, -2.0 * np.pi)
        q_hi = np.full(6, 2.0 * np.pi)
        task_w = max(float(cfg.qp_task_weight), 1e-6)
        reg = max(float(cfg.pinv_damping), 1e-9) ** 2
        return _ik_newton_solve(
            cfg, dyn.fk_and_jacobian, p_des, quat0, q_rest, selected,
            q_lo, q_hi, task_w, reg, extra_rot=[], extra_w=0.0,
        )

    return solve


def _horizon_solver(dyn: LocalMujocoDynamics, hp: HorizonParams):
    def solve(p_des, quat0, q_rest, selected, _g):
        q_target, _ = horizon_ik_solve(dyn.fk_and_jacobian, p_des, quat0, q_rest, selected, hp)
        return q_target

    return solve


def _hp_from_gains(gains: dict, **overrides) -> HorizonParams:
    base = HorizonParams(
        task_w=max(float(gains["qp_task_weight"]), 1e-6),
        reg=max(float(gains["pinv_damping"]), 1e-9) ** 2,
        max_joint_deviation_rad=gains["ik_max_joint_deviation_rad"],
    )
    return replace(base, **overrides)


# --------------------------------------------------------------------------
# subcommands
# --------------------------------------------------------------------------

CASES = (
    # (label, scenario, dx, move_duration_s, action_override_idx5)
    ("wrist_sing_neg45", "neg45_wrist2offset", -0.029, 1.0, 1.0),
    ("hanging_-0.296_slow", "hanging_alpha_0_5", -0.296, 1.0, None),
    ("hanging_-0.296_fast", "hanging_alpha_0_5", -0.296, FAST_MOVE_DURATION_S, None),
    ("hanging_-0.370_slow", "hanging_alpha_0_5", -0.370, 1.0, None),
)


def _action_for(case_override) -> np.ndarray:
    a = BASELINE_ACTION.copy()
    if case_override is not None:
        a[5] = float(case_override)
    return a


def cmd_selfcheck(args) -> None:
    dyn = LocalMujocoDynamics()
    env_cfg = VelocityTransportEnvConfig()
    for label, scen, dx, _md, ov in CASES:
        gains = action_to_gains(_action_for(ov))
        scenario = scenario_by_name(scen)
        q_rest = scenario.q0.copy()
        p0, quat0, _ = dyn.fk_and_jacobian(q_rest)
        p_des = np.asarray(p0).reshape(3).copy()
        p_des[0] += dx * 0.5
        selected = [0, 1, 2, 5]
        q_base = _baseline_solver(dyn, gains, env_cfg)(p_des, quat0, q_rest, selected, gains)
        hp = _hp_from_gains(gains, n_horizon=1, sqp_iterations=env_cfg.ik_iterations, cond_weight=0.0)
        q_h, _ = horizon_ik_solve(dyn.fk_and_jacobian, p_des, quat0, q_rest, selected, hp)
        print(f"{label:<24} max|q_horizon(N=1) - q_baseline| = {np.max(np.abs(q_h - q_base)):.3e}")


def cmd_profile(args) -> None:
    dyn = LocalMujocoDynamics()
    env_cfg = VelocityTransportEnvConfig()
    gains = action_to_gains(BASELINE_ACTION)
    scenario = scenario_by_name("neg45_wrist2offset")
    q_rest = scenario.q0.copy()
    p0, quat0, _ = dyn.fk_and_jacobian(q_rest)
    p_des = np.asarray(p0).reshape(3).copy()
    p_des[0] -= 0.015
    selected = [0, 1, 2, 5]
    reps = int(args.reps)

    def bench(fn, warmup=20):
        for _ in range(warmup):
            fn()
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            ts.append((time.perf_counter() - t0) * 1e3)
        a = np.array(ts)
        return dict(mean_ms=float(a.mean()), p50_ms=float(np.percentile(a, 50)),
                    p95_ms=float(np.percentile(a, 95)), p99_ms=float(np.percentile(a, 99)))

    out: dict = {"reps": reps, "baseline": {}, "horizon": {}, "fk_call": {}}

    out["fk_call"] = bench(lambda: dyn.fk_and_jacobian(q_rest))
    out["svd_6x6"] = bench(lambda: np.linalg.svd(dyn.fk_and_jacobian(q_rest)[2], compute_uv=False))

    for iters in (1, 3, 6, 10):
        cfg_g = dict(gains)
        env2 = replace(env_cfg, ik_iterations=iters)
        solve = _baseline_solver(dyn, cfg_g, env2)
        out["baseline"][f"ik_iterations={iters}"] = bench(
            lambda s=solve: s(p_des, quat0, q_rest, selected, cfg_g)
        )
    # orientation_priority costs a second identical solve
    out["baseline"]["ik_iterations=6_x2_orientation_priority"] = {
        k: v * 2 for k, v in out["baseline"]["ik_iterations=6"].items()
    }

    for n in (1, 2, 3, 5):
        for sqp in (2, 3, 6):
            for wc in (0.0, 1.0e6):
                hp = _hp_from_gains(gains, n_horizon=n, sqp_iterations=sqp, cond_weight=wc,
                                    sigma_floor=args.sigma_floor)
                key = f"N={n},sqp={sqp},cond={'on' if wc > 0 else 'off'}"
                out["horizon"][key] = bench(
                    lambda h=hp: horizon_ik_solve(dyn.fk_and_jacobian, p_des, quat0, q_rest, selected, h)
                )
    print(json.dumps(out, indent=2))
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(out, f, indent=2)


def cmd_validate(args) -> None:
    dyn = LocalMujocoDynamics()
    env_cfg = VelocityTransportEnvConfig()
    env = VelocityTransportEnv(env_cfg, seed=0)
    rows = []

    variants = [("baseline_env", None), ("baseline_replica", None)]
    for n in (1, 2, 3, 5):
        for wc in (0.0, args.cond_weight):
            variants.append((f"mpc N={n} cond={'on' if wc > 0 else 'off'}", (n, wc)))

    for label, scen, dx, md, ov in CASES:
        action = _action_for(ov)
        gains = action_to_gains(action)
        for vname, vparam in variants:
            if vname == "baseline_env":
                r = run_episode(env, action, scenario=scenario_by_name(scen),
                                target_x_delta_m=dx, move_duration_s=md)
                rows.append(dict(case=label, variant=vname, guard=r.guard_reason,
                                 achieved=r.achieved_x_delta_m, ori=r.orientation_error,
                                 qd=r.max_abs_qd_radps, sigma_min=None, wrist2=None,
                                 solve_ms=None))
                continue
            if vname == "baseline_replica":
                solver = _baseline_solver(dyn, gains, env_cfg)
            else:
                n, wc = vparam
                hp = _hp_from_gains(gains, n_horizon=n, sqp_iterations=args.sqp_iterations,
                                    cond_weight=wc, sigma_floor=args.sigma_floor,
                                    trust_radius_rad=args.trust_radius)
                solver = _horizon_solver(dyn, hp)
            o = run_episode_custom(dyn, env_cfg, gains, scen, dx, md, solver)
            rows.append(dict(case=label, variant=vname, guard=o.guard_reason,
                             achieved=o.achieved_x_delta_m, ori=o.orientation_error,
                             qd=o.max_abs_qd_radps, sigma_min=o.min_sigma_min,
                             wrist2=o.min_abs_wrist2, solve_ms=o.solve_ms_mean))
            print(f"{label:<20} {vname:<22} {str(rows[-1]['guard'])[:44]:<45} "
                  f"ach={o.achieved_x_delta_m:+.4f} ori={o.orientation_error:.4f} "
                  f"qd={o.max_abs_qd_radps:7.3f} smin={o.min_sigma_min:.5f} "
                  f"|w2|min={o.min_abs_wrist2:.5f} solve={o.solve_ms_mean:.3f}ms", flush=True)
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(rows, f, indent=2)


def cmd_mechanism(args) -> None:
    """The two load-bearing probes behind this study's verdict.

    (1) What actually trips the wrist case's joint-velocity guard: a Jacobian
        blowup, or a cycle-to-cycle JUMP in q_target? Reports the q_target
        jump distribution alongside the gap that alone saturates the guard.
    (2) Does the recovered -X band need the HORIZON, or just the conditioning
        barrier (which N=1 can express at a fraction of the cost)?
    """
    dyn = LocalMujocoDynamics()
    env_cfg = VelocityTransportEnvConfig()
    a = BASELINE_ACTION.copy()
    a[5] = 1.0  # tight ik_max_joint_deviation_rad = 0.01, the documented case
    gains = action_to_gains(a)
    kp = gains["ik_joint_gain"]
    sc = scenario_by_name("neg45_wrist2offset")
    q_rest = sc.q0.copy()
    p0, quat0, _ = dyn.fk_and_jacobian(q_rest)
    p0 = np.asarray(p0).reshape(3)
    x0 = float(p0[0])
    sel = [0, 1, 2, 5]

    print(f"(1) ik_joint_gain={kp:.2f}: a q_target jump of {3.0/kp:.4f} rad alone "
          f"saturates the 3.0 rad/s guard")

    def roll(solver, label):
        q = q_rest.copy()
        t = 0.0
        dt = 1.0 / env_cfg.rate_hz
        prev = None
        jumps = []
        res = "complete"
        while t < env_cfg.duration_s:
            tx, _ = x_profile_target("min_jerk_move_hold", x0, -0.029, t,
                                     env_cfg.duration_s, move_duration_s=1.0)
            p, quat, jac = dyn.fk_and_jacobian(q)
            pd = p0.copy()
            pd[0] = tx
            qt = solver(pd, quat0, q_rest, sel, gains)
            if prev is not None:
                jumps.append(float(np.max(np.abs(qt - prev))))
            prev = qt.copy()
            xd = np.asarray(jac).reshape(6, 6) @ (kp * (qt - q))
            v, w = xd[:3], xd[3:]
            vn, wn = np.linalg.norm(v), np.linalg.norm(w)
            if vn > env_cfg.max_lin_speed_mps:
                v = v * (env_cfg.max_lin_speed_mps / vn)
            if wn > env_cfg.max_ang_speed_radps:
                w = w * (env_cfg.max_ang_speed_radps / wn)
            qd = _damped_pinv(np.asarray(jac).reshape(6, 6), env_cfg.qd_estimate_damping) @ np.concatenate([v, w])
            m = float(np.max(np.abs(qd)))
            if m > env_cfg.max_joint_velocity_radps:
                res = f"TRIP qd={m:.3f} at t={t:.3f}"
                break
            q = q + qd * dt
            t += dt
        j = np.array(jumps) if jumps else np.array([0.0])
        ach = float(dyn.fk_and_jacobian(q)[0][0] - x0)
        print(f"    {label:<30} {res:<26} ach={ach:+.4f}  q_target jump "
              f"p50={np.percentile(j,50):.5f} p99={np.percentile(j,99):.5f} max={j.max():.5f} rad")

    roll(_baseline_solver(dyn, gains, env_cfg), "baseline (production solve)")
    for n in (2, 3, 5):
        hp = _hp_from_gains(gains, n_horizon=n, sqp_iterations=4,
                            cond_weight=args.cond_weight, sigma_floor=args.sigma_floor)
        roll(_horizon_solver(dyn, hp), f"horizon N={n} barrier on sqp=4")

    print(f"\n(2) barrier w={args.cond_weight:.0e} sigma_floor={args.sigma_floor} sqp=4, "
          f"neg45_wrist2offset, move=1.0 s")
    cands = [("baseline", None)] + [(f"N={n}", n) for n in (1, 2, 3, 4)]
    print(f"    {'dx (m)':>9} " + " ".join(f"{nm:>22}" for nm, _ in cands))
    for dx in (-0.0203, -0.0261, -0.0290, -0.0319, -0.0377, -0.0464):
        cells = []
        for _nm, n in cands:
            solver = (_baseline_solver(dyn, gains, env_cfg) if n is None else _horizon_solver(
                dyn, _hp_from_gains(gains, n_horizon=n, sqp_iterations=4,
                                    cond_weight=args.cond_weight, sigma_floor=args.sigma_floor)))
            o = run_episode_custom(dyn, env_cfg, gains, "neg45_wrist2offset", dx, 1.0, solver)
            tag = "PASS" if o.guard_reason is None else o.guard_reason.split("_guard")[0][:9]
            cells.append(f"{tag:>9} ach={o.achieved_x_delta_m:+.4f}")
        print(f"    {dx:>9.4f} " + " ".join(f"{c:>22}" for c in cells))


def cmd_breakdown(args) -> None:
    """Where the baseline solve's wall-clock actually goes -- the docstring in
    modes.py quotes ~0.23 ms for a 6-iteration solve, which predates
    ``ik_max_joint_deviation_rad``; this re-measures it."""
    import cProfile
    import io
    import pstats

    from controller_core.box_qp import build_weighted_least_squares_qp as _bwls
    from controller_core.cartesian_velocity_controller.modes import _ik_newton_solve

    dyn = LocalMujocoDynamics()
    gains = action_to_gains(BASELINE_ACTION)
    sc = scenario_by_name("neg45_wrist2offset")
    q_rest = sc.q0.copy()
    p0, quat0, jac0 = dyn.fk_and_jacobian(q_rest)
    p_des = np.asarray(p0).copy()
    p_des[0] -= 0.015
    sel = [0, 1, 2, 5]
    q_lo, q_hi = np.full(6, -2 * np.pi), np.full(6, 2 * np.pi)
    tw = float(gains["qp_task_weight"])
    rg = float(gains["pinv_damping"]) ** 2

    def bench(fn, reps=2000, warm=100):
        for _ in range(warm):
            fn()
        t = time.perf_counter()
        for _ in range(reps):
            fn()
        return (time.perf_counter() - t) / reps * 1e3

    jt = jac0[sel, :]
    hh, ll = _bwls([(jt, np.full(4, 0.01), tw)], reg=rg)
    print(f"fk_and_jacobian        {bench(lambda: dyn.fk_and_jacobian(q_rest)):.4f} ms")
    print(f"swing_twist_axis_error {bench(lambda: swing_twist_axis_error(quat0, quat0, 1)):.4f} ms")
    print(f"null_space_basis       {bench(lambda: null_space_basis(jt)):.4f} ms")
    print(f"build_wls_qp           {bench(lambda: _bwls([(jt, np.full(4, 0.01), tw)], reg=rg)):.4f} ms")
    print(f"solve_box_qp (n=6)     {bench(lambda: solve_box_qp(hh, ll, q_lo, q_hi)):.4f} ms")
    print(f"svd 6x6 sigma_min      {bench(lambda: np.linalg.svd(jac0, compute_uv=False)):.4f} ms")

    for dev in (gains["ik_max_joint_deviation_rad"], None):
        cfg = CartesianVelocityConfig(
            ik_seeded_resolution=True, reduced_task_dims=False, ik_iterations=6,
            pinv_damping=gains["pinv_damping"], qp_task_weight=gains["qp_task_weight"],
            ik_max_joint_deviation_rad=dev,
        )
        f = lambda c=cfg: _ik_newton_solve(  # noqa: E731
            c, dyn.fk_and_jacobian, p_des, quat0, q_rest, sel, q_lo, q_hi, tw, rg,
            extra_rot=[], extra_w=0.0,
        )
        print(f"_ik_newton_solve 6it, clip={dev}: {bench(f, reps=500):.4f} ms")
        if dev is not None:
            pr = cProfile.Profile()
            pr.enable()
            for _ in range(500):
                f()
            pr.disable()
            s = io.StringIO()
            pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(12)
            print(s.getvalue())


def cmd_sweep(args) -> None:
    """Barrier-hyperparameter sweep on ONE case -- guards against the
    'you just didn't tune the new term' objection."""
    dyn = LocalMujocoDynamics()
    env_cfg = VelocityTransportEnvConfig()
    label, scen, dx, md, ov = [c for c in CASES if c[0] == args.case][0]
    gains = action_to_gains(_action_for(ov))
    rows = []
    for n in (2, 3, 5, 8):
        for wc in (1.0e4, 1.0e6, 1.0e8, 1.0e10):
            for sf in (0.03, 0.05, 0.10, 0.20):
                hp = _hp_from_gains(gains, n_horizon=n, sqp_iterations=args.sqp_iterations,
                                    cond_weight=wc, sigma_floor=sf)
                o = run_episode_custom(dyn, env_cfg, gains, scen, dx, md, _horizon_solver(dyn, hp))
                rows.append(dict(case=label, n=n, cond_weight=wc, sigma_floor=sf,
                                 guard=o.guard_reason, achieved=o.achieved_x_delta_m,
                                 ori=o.orientation_error, qd=o.max_abs_qd_radps,
                                 sigma_min=o.min_sigma_min, wrist2=o.min_abs_wrist2))
                print(f"N={n} w={wc:.0e} sf={sf:.2f} {str(o.guard_reason)[:40]:<41} "
                      f"ach={o.achieved_x_delta_m:+.4f} ori={o.orientation_error:.4f} "
                      f"qd={o.max_abs_qd_radps:8.3f} smin={o.min_sigma_min:.5f} "
                      f"|w2|min={o.min_abs_wrist2:.5f}", flush=True)
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(rows, f, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("selfcheck", "profile", "validate", "sweep", "breakdown", "mechanism"), required=True)
    ap.add_argument("--case", default="wrist_sing_neg45")
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--sqp-iterations", type=int, default=6)
    ap.add_argument("--cond-weight", type=float, default=1.0e6)
    ap.add_argument("--sigma-floor", type=float, default=0.05)
    ap.add_argument("--trust-radius", type=float, default=float("inf"))
    ap.add_argument("--output-json", default=None)
    args = ap.parse_args()
    {"selfcheck": cmd_selfcheck, "profile": cmd_profile, "validate": cmd_validate,
     "sweep": cmd_sweep, "breakdown": cmd_breakdown,
     "mechanism": cmd_mechanism}[args.mode](args)


if __name__ == "__main__":
    main()
