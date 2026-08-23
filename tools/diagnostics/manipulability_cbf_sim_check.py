#!/usr/bin/env python3
"""Manipulability CBF on / off, closed loop, on the REAL UR5e model.

``controller_core/manipulability_cbf.py`` implements an OSCBF-style
(arXiv:2503.06736, Sec V-B1) control-barrier-function constraint on the
Yoshikawa manipulability ``mu(q) = prod_i sigma_i(J(q))``, solved as a
per-cycle QP filter on the commanded torque. Unlike every pre-existing
singularity mechanism in this repo (``jacobian_singular_cond_max``,
``svd_singularity_filtering``, offline pose pre-filtering), it constrains the
arm's MOTION toward the singular set rather than shrinking authority once the
arm is already there.

This script measures that difference on ``assets/ur5e_torque/scene.xml`` --
never a toy plant -- two ways:

  ``--mode profile`` (instant)
      ``mu(q)`` and ``cond(J)`` along a one-joint sweep, so the barrier's scale
      at a given pose family can be read directly before choosing an epsilon.

  ``--mode rollout`` (default)
      Real closed-loop move+hold through the same adapter pipeline every sim
      tool in this repo uses (``build_initial_state_and_adapter`` /
      ``build_mujoco_state`` / ``adapter.step()`` / ``mujoco.mj_step``), run
      once with the CBF and once without, everything else byte-identical.
      Sweeps +X and -X by default (AGENTS.md sec 7).

Examples
--------
    python tools/diagnostics/manipulability_cbf_sim_check.py --mode profile
    python tools/diagnostics/manipulability_cbf_sim_check.py \
        --deltas 0.20 -0.20 --epsilon 1e-3 --json out.json
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
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from controller_core.manipulability_cbf import manipulability  # noqa: E402
from hardware.poses import (  # noqa: E402
    HEIGHT_ALPHA_0_5_CLEARANCE_Q,
    HEIGHT_ALPHA_0_5_Q,
    HEIGHT_ALPHA_0_5_WRIST2_OFFSET_Q,
    MEGA_SEARCH_WINNER_Q,
)
from simulation.ur5e_mujoco_torque import (  # noqa: E402
    build_initial_state_and_adapter,
    build_mujoco_state,
    load_model,
    make_mujoco_jacobian_fn,
    x_profile_target,
)

DEFAULT_CONFIG = "config/ur5e_mujoco_torque_osc_tuned.yaml"

#: ``height_alpha_0_5`` with wrist_2 = 0.10 rad -- close enough to the wrist
#: singularity that a world-Y transport walks the arm INTO it (measured
#: 2026-08-13: wrist_2 0.100 -> 0.0057 rad with the tuned config, mu down to
#: 1.7e-4), which is what makes it the reproducible wrist_2 -> 0 case for this
#: mechanism. HEIGHT_ALPHA_0_5_WRIST2_OFFSET_Q (wrist_2 = 0.20) is too far out
#: for the same move to reach it.
HEIGHT_ALPHA_0_5_WRIST2_0_10_Q = HEIGHT_ALPHA_0_5_Q.copy()
HEIGHT_ALPHA_0_5_WRIST2_0_10_Q[4] = 0.10

POSES: dict[str, np.ndarray] = {
    "height_alpha_0_5": HEIGHT_ALPHA_0_5_Q,
    "height_alpha_0_5_clearance": HEIGHT_ALPHA_0_5_CLEARANCE_Q,
    "height_alpha_0_5_wrist2_offset": HEIGHT_ALPHA_0_5_WRIST2_OFFSET_Q,
    "height_alpha_0_5_wrist2_0_10": HEIGHT_ALPHA_0_5_WRIST2_0_10_Q,
    "mega_search_winner": MEGA_SEARCH_WINNER_Q,
}


SCENE_XML = REPO_ROOT / "assets" / "ur5e_torque" / "scene.xml"


def _seed_model(q_start: np.ndarray):
    model, data, site_id, joint_ids, _ = load_model(SCENE_XML)
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    q_start = np.asarray(q_start, dtype=np.float64).reshape(6)
    for idx, jid in enumerate(joint_ids):
        data.qpos[int(model.jnt_qposadr[jid])] = float(q_start[idx])
    mujoco.mj_forward(model, data)
    return model, data, site_id, joint_ids


def load_controller_config(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


# --------------------------------------------------------------------------- #
@dataclass
class ProfileResult:
    pose: str
    joint_index: int
    values: list[float] = field(default_factory=list)
    manipulability: list[float] = field(default_factory=list)
    cond_j: list[float] = field(default_factory=list)


def run_profile(
    q_start: np.ndarray,
    *,
    joint_index: int = 4,
    values: list[float] | None = None,
    pose_label: str = "custom",
) -> ProfileResult:
    """``mu(q)`` / ``cond(J)`` along a single-joint sweep on the real model."""
    model, _data, site_id, joint_ids = _seed_model(q_start)
    jac_fn = make_mujoco_jacobian_fn(model, site_id, joint_ids)
    if values is None:
        values = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.2]
    res = ProfileResult(pose=pose_label, joint_index=int(joint_index))
    for value in values:
        q = np.asarray(q_start, dtype=np.float64).copy()
        q[int(joint_index)] = float(value)
        jac = jac_fn(q)
        res.values.append(float(value))
        res.manipulability.append(float(manipulability(jac)))
        res.cond_j.append(float(np.linalg.cond(jac)))
    return res


# --------------------------------------------------------------------------- #
@dataclass
class RolloutResult:
    pose: str
    config: str
    cbf: bool
    epsilon: float
    alpha1: float
    alpha2: float
    target_delta_m: float
    move_duration_s: float
    hold_duration_s: float
    steps: int = 0
    # Barrier trajectory.
    min_manipulability: float = float("inf")
    final_manipulability: float = float("nan")
    min_abs_wrist_2_rad: float = float("inf")
    max_cond_j: float = 0.0
    # CBF activity (all zero when cbf is False).
    cbf_active_steps: int = 0
    cbf_infeasible_steps: int = 0
    max_cbf_delta_tau_nm: float = 0.0
    min_cbf_h: float = float("nan")
    # Task / safety.
    achieved_delta_m: float = 0.0
    tracking_fraction: float = float("nan")
    max_abs_qd_radps: float = 0.0
    max_abs_tau_nm: float = 0.0
    max_orientation_error_rad: float = 0.0
    max_abs_y_drift_m: float = 0.0
    max_abs_z_drift_m: float = 0.0
    guard_tripped: bool = False
    guard_reason: str = ""
    guard_time_s: float = float("nan")


def run_rollout(
    q_start: np.ndarray,
    *,
    cbf: bool,
    config_path: str | Path = DEFAULT_CONFIG,
    epsilon: float = 1.0e-3,
    alpha1: float = 10.0,
    alpha2: float = 10.0,
    target_delta_m: float = 0.20,
    move_duration_s: float = 1.5,
    hold_duration_s: float = 1.0,
    pose_label: str = "custom",
    transport_axis: int = 0,
    gravity_source: str | None = "mujoco_qfrc",
    coriolis_feedforward: bool | None = False,
    controller_overrides: dict[str, Any] | None = None,
) -> RolloutResult:
    """One real closed-loop move+hold with the manipulability CBF on or off.

    Everything except ``manipulability_cbf`` (and its epsilon/alpha) is
    byte-identical between the two runs, so any difference is attributable to
    the flag alone.

    ``gravity_source``/``coriolis_feedforward`` default to MuJoCo's own
    ``qfrc_bias`` gravity and no Coriolis feedforward, overriding the config's
    ``mujoco:`` section -- the tuned configs select ``pinocchio``, an optional
    dependency, and the two are parity-checked to <1e-8 Nm (AGENTS.md sec 3).
    Pass ``None`` to use whatever the config says instead.
    """
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    cfg = load_controller_config(config_path)
    ctrl_cfg = dict(cfg["controller"])
    ctrl_cfg.update(controller_overrides or {})
    if cbf:
        ctrl_cfg["manipulability_cbf"] = True
        ctrl_cfg["manipulability_cbf_epsilon"] = float(epsilon)
        ctrl_cfg["manipulability_cbf_alpha1"] = float(alpha1)
        ctrl_cfg["manipulability_cbf_alpha2"] = float(alpha2)
    mj_cfg = cfg.get("mujoco", {}) or {}

    model, data, site_id, joint_ids = _seed_model(q_start)
    state0, adapter = build_initial_state_and_adapter(
        model, data, site_id, joint_ids,
        controller_cfg=ctrl_cfg,
        transport_axis_index=int(transport_axis),
        target_x_delta=float(target_delta_m),
        controller_kind="impedance",
        force_hold_current_pose=False,
        gravity_mode=str(mj_cfg.get("gravity_mode", "gravity_comp")),
        gravity_source=str(
            mj_cfg.get("gravity_source", "mujoco_qfrc") if gravity_source is None else gravity_source
        ),
        coriolis_feedforward=bool(
            mj_cfg.get("coriolis_feedforward", False)
            if coriolis_feedforward is None
            else coriolis_feedforward
        ),
        torque_limit_scale=1.0,
    )

    start_ee = np.asarray(state0.ee_pos, dtype=np.float64).copy()
    axis0 = float(start_ee[transport_axis])
    dt = float(model.opt.timestep)
    duration_s = float(move_duration_s) + float(hold_duration_s)
    steps = max(1, int(np.ceil(duration_s / max(dt, 1e-9))))

    res = RolloutResult(
        pose=pose_label,
        config=str(config_path.relative_to(REPO_ROOT)),
        cbf=bool(cbf),
        epsilon=float(epsilon),
        alpha1=float(alpha1),
        alpha2=float(alpha2),
        target_delta_m=float(target_delta_m),
        move_duration_s=float(move_duration_s),
        hold_duration_s=float(hold_duration_s),
    )
    gravity_scratch = mujoco.MjData(model)
    ee_now = start_ee.copy()

    for _ in range(steps):
        t_s = float(data.time)
        target_now, target_vel_now = x_profile_target(
            "min_jerk_move_hold", axis0, float(target_delta_m), t_s, duration_s,
            move_duration_s=float(move_duration_s),
        )
        target_ee_pos = start_ee.copy()
        target_ee_pos[transport_axis] = target_now
        target_ee_vel = np.zeros(3, dtype=np.float64)
        target_ee_vel[transport_axis] = target_vel_now

        state = build_mujoco_state(
            model, data, site_id=site_id, joint_ids=joint_ids, time_s=t_s, dt_s=dt,
            target_x=float(target_now), target_x_vel=float(target_vel_now), target_x_accel=0.0,
            target_axis=float(target_now), target_axis_vel=float(target_vel_now),
            target_ee_pos=target_ee_pos, target_ee_vel=target_ee_vel,
            reference_quat=state0.reference_quat, hold_current_pose=False,
            transport_axis_index=int(transport_axis),
            gravity_compensation=bool(str(mj_cfg.get("gravity_mode", "gravity_comp")) == "gravity_comp"),
            gravity_scratch_data=gravity_scratch,
        )
        tau, diag = adapter.step(state=state)

        jac = np.asarray(state.jacobian, dtype=np.float64)
        mu = float(manipulability(jac))
        res.min_manipulability = min(res.min_manipulability, mu)
        res.final_manipulability = mu
        res.min_abs_wrist_2_rad = min(res.min_abs_wrist_2_rad, abs(float(state.q[4])))
        cond_j = float(np.linalg.cond(jac))
        if np.isfinite(cond_j):
            res.max_cond_j = max(res.max_cond_j, cond_j)

        out = diag.get("controller_output", {}) or {}
        if out.get("manipulability_cbf_active"):
            res.cbf_active_steps += 1
        if out.get("manipulability_cbf_feasible") is False:
            res.cbf_infeasible_steps += 1
        res.max_cbf_delta_tau_nm = max(
            res.max_cbf_delta_tau_nm, float(out.get("manipulability_cbf_delta_tau_norm", 0.0) or 0.0)
        )
        h_val = out.get("manipulability_cbf_h")
        if h_val is not None:
            res.min_cbf_h = h_val if not np.isfinite(res.min_cbf_h) else min(res.min_cbf_h, float(h_val))

        res.max_abs_qd_radps = max(res.max_abs_qd_radps, float(np.max(np.abs(state.qd))))
        res.max_abs_tau_nm = max(res.max_abs_tau_nm, float(np.max(np.abs(np.asarray(tau)))))
        res.max_orientation_error_rad = max(
            res.max_orientation_error_rad, float(diag.get("orientation_error_norm", 0.0))
        )

        ee_now = np.asarray(state.ee_pos, dtype=np.float64).copy()
        held = [i for i in (0, 1, 2) if i != transport_axis]
        drifts = {i: abs(float(ee_now[i] - start_ee[i])) for i in held}
        if 1 in drifts:
            res.max_abs_y_drift_m = max(res.max_abs_y_drift_m, drifts[1])
        if 2 in drifts:
            res.max_abs_z_drift_m = max(res.max_abs_z_drift_m, drifts[2])

        if not bool(diag.get("safety_ok", True)):
            res.guard_tripped = True
            res.guard_reason = str(diag.get("safety_reason", "") or "unknown")
            res.guard_time_s = t_s
            break

        data.ctrl[:6] = np.asarray(tau, dtype=np.float64).reshape(6)
        mujoco.mj_step(model, data)
        res.steps += 1

    res.achieved_delta_m = float(ee_now[transport_axis] - start_ee[transport_axis])
    if abs(float(target_delta_m)) > 0.0:
        res.tracking_fraction = res.achieved_delta_m / float(target_delta_m)
    return res


# --------------------------------------------------------------------------- #
def _print_profile(r: ProfileResult) -> None:
    print(f"\n=== mu / cond(J) profile: pose={r.pose} joint={r.joint_index} ===")
    print(f"{'q_joint':>10} {'mu':>14} {'cond(J)':>14}")
    for v, mu, c in zip(r.values, r.manipulability, r.cond_j):
        print(f"{v:>10.4f} {mu:>14.6e} {c:>14.4e}")


def _print_rollouts(rows: list[RolloutResult]) -> None:
    print(f"\n=== closed-loop rollouts (config={rows[0].config if rows else '-'}) ===")
    header = (
        f"{'pose':>32} {'dx':>7} {'cbf':>5} {'min_mu':>11} {'min|w2|':>9} "
        f"{'max_cond':>10} {'act':>5} {'dtau':>8} {'track':>7} {'guard':>22}"
    )
    print(header)
    for r in rows:
        print(
            f"{r.pose:>32} {r.target_delta_m:>7.3f} {str(r.cbf):>5} "
            f"{r.min_manipulability:>11.4e} {r.min_abs_wrist_2_rad:>9.5f} "
            f"{r.max_cond_j:>10.3e} {r.cbf_active_steps:>5} {r.max_cbf_delta_tau_nm:>8.3f} "
            f"{r.tracking_fraction:>7.3f} {(r.guard_reason or '-'):>22}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["rollout", "profile"], default="rollout")
    parser.add_argument("--poses", nargs="+", default=["height_alpha_0_5_wrist2_offset"], choices=sorted(POSES))
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--deltas", nargs="+", type=float, default=[0.20, -0.20])
    parser.add_argument("--epsilon", type=float, default=1.0e-3)
    parser.add_argument("--alpha1", type=float, default=10.0)
    parser.add_argument("--alpha2", type=float, default=10.0)
    parser.add_argument("--move-duration", type=float, default=1.5)
    parser.add_argument("--hold-duration", type=float, default=1.0)
    parser.add_argument("--transport-axis", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)

    payload: dict[str, Any] = {"mode": args.mode}
    if args.mode == "profile":
        results = [run_profile(POSES[p], pose_label=p) for p in args.poses]
        for r in results:
            _print_profile(r)
        payload["profiles"] = [asdict(r) for r in results]
    else:
        rows: list[RolloutResult] = []
        for pose in args.poses:
            for delta in args.deltas:
                for cbf in (False, True):
                    rows.append(
                        run_rollout(
                            POSES[pose], cbf=cbf, config_path=args.config,
                            epsilon=args.epsilon, alpha1=args.alpha1, alpha2=args.alpha2,
                            target_delta_m=float(delta),
                            move_duration_s=args.move_duration,
                            hold_duration_s=args.hold_duration,
                            pose_label=pose, transport_axis=int(args.transport_axis),
                        )
                    )
        _print_rollouts(rows)
        payload["rollouts"] = [asdict(r) for r in rows]

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
