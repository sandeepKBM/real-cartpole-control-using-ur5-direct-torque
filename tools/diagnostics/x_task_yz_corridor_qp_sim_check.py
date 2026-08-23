#!/usr/bin/env python3
"""Reduced-task (X + orientation) QP with a Y/Z corridor, on the REAL UR5e
model -- against the tuned OSC controller, at the real deployment pose.

``controller_core/x_task_yz_corridor_qp/`` builds a 4-row task
(``J[0:1,:]`` + ``J[3:6,:]``) and lets Y/Z move freely inside a bounded
corridor enforced as high-order-CBF inequality rows on the same QP, composed
with ``controller_core/manipulability_cbf.py``'s singularity row. See
``docs/status/x_task_yz_corridor_qp_2026-08-13.md``.

Two modes:

  ``--mode profile`` (instant, no closed loop)
      (a) ``mu(q)`` / ``cond(J)`` / ``sigma_min(J)`` along a single-joint sweep
          from the start pose, so the CBF epsilon can be sized against real
          numbers; and
      (b) an ISOLATED microbenchmark of ``compute()`` at the start pose with
          0, 1 (manipulability only), 4 (corridor only) and 5 (both)
          inequality rows -- the per-cycle cost that decides whether this
          controller can ever run at 500 Hz on hardware. Measured, never
          estimated.

  ``--mode rollout`` (default)
      Real closed-loop move+hold through the SAME adapter pipeline every sim
      tool in this repo uses (``build_initial_state_and_adapter`` /
      ``build_mujoco_state`` / ``adapter.step()`` / ``mujoco.mj_step``), for
      the new controller (flags off and flags on) and for
      ``config/ur5e_mujoco_torque_osc_tuned.yaml``, at the same dx. Sweeps
      +X and -X by default (AGENTS.md sec 7), and records real per-cycle
      ``qp_solve_time_s`` statistics from the closed loop as well.

The default dx set is {0.02, 0.06} -- the range the corridor half-width's
calibration evidence actually covers -- plus 0.12, explicitly flagged
UNVALIDATED REGIME, informational only.

Examples
--------
    python tools/diagnostics/x_task_yz_corridor_qp_sim_check.py --mode profile
    python tools/diagnostics/x_task_yz_corridor_qp_sim_check.py \\
        --deltas 0.02 -0.02 --json /tmp/corridor.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
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
from simulation.ur5e_mujoco_torque import (  # noqa: E402
    build_initial_state_and_adapter,
    build_mujoco_state,
    load_model,
    make_mujoco_jacobian_fn,
    x_profile_target,
)

SCENE_XML = REPO_ROOT / "assets" / "ur5e_torque" / "scene.xml"

NEW_CONFIG = "config/ur5e_mujoco_torque_x_task_yz_corridor_qp.yaml"
NEW_CONFIG_ENABLED = "config/ur5e_mujoco_torque_x_task_yz_corridor_qp_enabled.yaml"
OSC_CONFIG = "config/ur5e_mujoco_torque_osc_tuned.yaml"

#: The 500 Hz ``direct_torque`` real-time budget, in seconds. Every timing
#: number this script prints is measured against this.
DIRECT_TORQUE_BUDGET_S = 2.0e-3


def _load_arm_q0() -> np.ndarray:
    """``ARM_Q0`` from the IK feasibility pre-check, imported rather than
    re-typed so the two scripts provably describe the same pose."""
    path = REPO_ROOT / "tools" / "diagnostics" / "x_task_yz_corridor_ik_feasibility_check.py"
    spec = importlib.util.spec_from_file_location("x_task_yz_corridor_ik_feasibility_check", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return np.asarray(module.ARM_Q0, dtype=np.float64).reshape(6)


ARM_Q0 = _load_arm_q0()

POSES: dict[str, np.ndarray] = {"arm_q0": ARM_Q0}


def _seed_model(q_start: np.ndarray):
    model, data, site_id, joint_ids, _ = load_model(SCENE_XML)
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    q_start = np.asarray(q_start, dtype=np.float64).reshape(6)
    for idx, jid in enumerate(joint_ids):
        data.qpos[int(model.jnt_qposadr[jid])] = float(q_start[idx])
    mujoco.mj_forward(model, data)
    return model, data, site_id, joint_ids


def load_controller_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


# --------------------------------------------------------------------------- #
# profile mode
# --------------------------------------------------------------------------- #
@dataclass
class ProfileResult:
    pose: str
    joint_index: int
    values: list[float] = field(default_factory=list)
    manipulability: list[float] = field(default_factory=list)
    cond_j: list[float] = field(default_factory=list)
    sigma_min_j: list[float] = field(default_factory=list)


def run_profile(
    q_start: np.ndarray,
    *,
    joint_index: int = 4,
    values: list[float] | None = None,
    pose_label: str = "arm_q0",
) -> ProfileResult:
    model, _data, site_id, joint_ids = _seed_model(q_start)
    jac_fn = make_mujoco_jacobian_fn(model, site_id, joint_ids)
    if values is None:
        # 0.004714693 is ARM_Q0's OWN wrist_2 value (joint index 4). Note that
        # 0.0206 is ARM_Q0's wrist_3, not its wrist_2 -- easy to transpose,
        # and the two give very different conditioning.
        values = [0.0, 0.004714693, 0.0206, 0.05, 0.1, 0.2, 0.4, 0.8]
    res = ProfileResult(pose=pose_label, joint_index=int(joint_index))
    for value in values:
        q = np.asarray(q_start, dtype=np.float64).copy()
        q[int(joint_index)] = float(value)
        jac = jac_fn(q)
        sv = np.linalg.svd(jac, compute_uv=False)
        res.values.append(float(value))
        res.manipulability.append(float(manipulability(jac)))
        res.cond_j.append(float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf"))
        res.sigma_min_j.append(float(sv[-1]))
    return res


@dataclass
class TimingResult:
    label: str
    n_ineq_rows: int
    n_calls: int
    mean_total_s: float
    p95_total_s: float
    max_total_s: float
    mean_qp_s: float
    p95_qp_s: float
    max_qp_s: float
    over_budget_fraction: float


def _timing_stats(label: str, rows: int, totals: list[float], qps: list[float]) -> TimingResult:
    t = np.asarray(totals, dtype=np.float64)
    q = np.asarray(qps, dtype=np.float64)
    return TimingResult(
        label=label,
        n_ineq_rows=int(rows),
        n_calls=int(t.size),
        mean_total_s=float(np.mean(t)),
        p95_total_s=float(np.percentile(t, 95)),
        max_total_s=float(np.max(t)),
        mean_qp_s=float(np.mean(q)),
        p95_qp_s=float(np.percentile(q, 95)),
        max_qp_s=float(np.max(q)),
        over_budget_fraction=float(np.mean(t > DIRECT_TORQUE_BUDGET_S)),
    )


def run_timing_profile(
    q_start: np.ndarray,
    *,
    n_calls: int = 200,
    config_path: str | Path = NEW_CONFIG_ENABLED,
) -> list[TimingResult]:
    """Isolated per-cycle cost of ``compute()`` at ``q_start``, for each row count.

    Deliberately isolated (one fixed state, called repeatedly) rather than
    read off a rollout: a rollout's own numbers are also reported, but they
    mix in states where the corridor is trivially satisfied, which is exactly
    the cheap case. This measures the controller as a real-time component.

    ``qd`` is set to a small non-zero value so the manipulability CBF's
    directional-curvature term (2 extra Jacobian evaluations) is really paid
    for -- at ``qd == 0`` it short-circuits to 0.0 and the measurement would
    flatter the mechanism.
    """
    from controller_core.x_task_yz_corridor_qp import (
        XTaskYZCorridorQPConfig,
        XTaskYZCorridorQPController,
    )

    model, data, site_id, joint_ids = _seed_model(q_start)
    jac_fn = make_mujoco_jacobian_fn(model, site_id, joint_ids)
    cfg_yaml = load_controller_config(config_path)
    ctrl_yaml = dict(cfg_yaml["controller"])

    state = build_mujoco_state(
        model, data, site_id=site_id, joint_ids=joint_ids,
        time_s=0.0, dt_s=float(model.opt.timestep),
        target_x=float(data.site_xpos[site_id][0]) + 0.02,
        target_x_vel=0.0,
        transport_axis_index=0,
        gravity_compensation=False,
    ).as_robot_state()
    # A small, realistic joint velocity: the curvature term is a quadratic
    # form in qd and is exactly zero at rest (see manipulability_cbf.py).
    state["qd"] = np.full(6, 0.05, dtype=np.float64)

    cases = [
        ("no_rows", False, False, 0),
        ("manipulability_only", False, True, 1),
        ("corridor_only", True, False, 4),
        ("corridor_and_manipulability", True, True, 5),
    ]
    out: list[TimingResult] = []
    for label, corridor, cbf, rows in cases:
        ctrl = dict(ctrl_yaml)
        ctrl["yz_corridor_enabled"] = bool(corridor)
        ctrl["manipulability_cbf"] = bool(cbf)
        controller = XTaskYZCorridorQPController(
            XTaskYZCorridorQPConfig.from_controller_yaml_section(ctrl), jacobian_fn=jac_fn
        )
        controller.reset_from_state(state)
        # Move the corridor walls in tight so the rows are genuinely ACTIVE
        # (an inactive row costs one g(0) probe and then stops; the expensive
        # case is the bisection, and that is the case that has to fit in the
        # budget).
        if corridor:
            controller.cfg.y_corridor_half_width_m = 1.0e-4
            controller.cfg.z_corridor_half_width_m = 1.0e-4
        controller.compute(state)  # warm-up (BLAS/first-touch)
        totals: list[float] = []
        qps: list[float] = []
        n_rows_seen = 0
        for _ in range(int(n_calls)):
            t0 = time.perf_counter()
            res = controller.compute(state)
            totals.append(time.perf_counter() - t0)
            qps.append(float(res.qp_solve_time_s))
            n_rows_seen = int(res.qp_num_ineq_rows)
        assert n_rows_seen == rows, f"{label}: expected {rows} rows, got {n_rows_seen}"
        out.append(_timing_stats(label, rows, totals, qps))
    return out


# --------------------------------------------------------------------------- #
# rollout mode
# --------------------------------------------------------------------------- #
@dataclass
class RolloutResult:
    pose: str
    label: str
    config: str
    controller_kind: str
    corridor: bool
    cbf: bool
    target_delta_m: float
    move_duration_s: float
    hold_duration_s: float
    validated_regime: bool = True
    steps: int = 0
    # Task.
    achieved_delta_m: float = 0.0
    tracking_fraction: float = float("nan")
    final_abs_x_error_m: float = float("nan")
    # Joint-space behavior of the joints the task is NOT allowed to drive.
    # Added 2026-08-13 with `task_excluded_joints`: the bug that motivated that
    # mechanism (shoulder_pan swinging 4-13 deg during ordinary X transport at a
    # pose that pins it for real wall clearance) is INVISIBLE in every other
    # field here -- the tracking fraction, the drift numbers and the orientation
    # error were all reported and none of them named the joint.
    shoulder_pan_range_rad: float = 0.0
    max_abs_shoulder_pan_dev_rad: float = 0.0
    max_abs_shoulder_pan_tau_nm: float = 0.0
    #: Cycles on which an excluded joint's PRE-CLIP torque was not bit-exactly
    #: its `tau_hold`. The pin is a structural guarantee, so the only honest
    #: expectation is ZERO -- reported as a count rather than asserted here so
    #: a violation shows up as a number in a rollout table instead of an
    #: exception in the middle of a sweep.
    excluded_joint_pin_violations: int = 0
    # Off-axis behavior.
    max_abs_y_drift_m: float = 0.0
    max_abs_z_drift_m: float = 0.0
    final_abs_y_drift_m: float = float("nan")
    final_abs_z_drift_m: float = float("nan")
    max_orientation_error_rad: float = 0.0
    # Conditioning.
    min_manipulability: float = float("inf")
    max_cond_j: float = 0.0
    # Effort / safety.
    max_abs_qd_radps: float = 0.0
    max_abs_tau_nm: float = 0.0
    guard_tripped: bool = False
    guard_reason: str = ""
    guard_time_s: float = float("nan")
    # Mechanism activity (all zero for the OSC baseline).
    corridor_active_steps: int = 0
    corridor_row_active_steps: tuple[int, int, int, int] = (0, 0, 0, 0)
    cbf_active_steps: int = 0
    infeasible_steps: int = 0
    # Real per-cycle QP cost from the closed loop.
    qp_mean_s: float = float("nan")
    qp_p95_s: float = float("nan")
    qp_max_s: float = float("nan")
    qp_rows: int = 0


def run_rollout(
    q_start: np.ndarray,
    *,
    controller_kind: str = "x_task_yz_corridor_qp",
    config_path: str | Path = NEW_CONFIG,
    corridor: bool | None = None,
    cbf: bool | None = None,
    target_delta_m: float = 0.06,
    move_duration_s: float = 1.5,
    hold_duration_s: float = 1.0,
    pose_label: str = "arm_q0",
    label: str = "",
    safety_overrides: dict[str, float] | None = None,
    gravity_source: str | None = "mujoco_qfrc",
    coriolis_feedforward: bool | None = False,
    validated_regime: bool = True,
) -> RolloutResult:
    """One closed-loop move+hold, identical pipeline for both controllers.

    ``gravity_source``/``coriolis_feedforward`` default to MuJoCo's own
    ``qfrc_bias`` and no Coriolis feedforward, overriding the config's
    ``mujoco:`` section, for the same reason the manipulability-CBF check does
    it: the tuned configs select ``pinocchio``, an optional dependency, and
    the two are parity-checked to <1e-8 Nm (AGENTS.md sec 3). Pass ``None`` to
    use whatever the config says.
    """
    cfg = load_controller_config(config_path)
    ctrl_cfg = dict(cfg["controller"])
    if corridor is not None:
        ctrl_cfg["yz_corridor_enabled"] = bool(corridor)
    if cbf is not None:
        ctrl_cfg["manipulability_cbf"] = bool(cbf)
    if safety_overrides:
        ctrl_cfg["safety"] = {**(ctrl_cfg.get("safety", {}) or {}), **safety_overrides}
    mj_cfg = cfg.get("mujoco", {}) or {}

    model, data, site_id, joint_ids = _seed_model(q_start)
    state0, adapter = build_initial_state_and_adapter(
        model, data, site_id, joint_ids,
        controller_cfg=ctrl_cfg,
        transport_axis_index=0,
        target_x_delta=float(target_delta_m),
        controller_kind=str(controller_kind),
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
    x0 = float(start_ee[0])
    dt = float(model.opt.timestep)
    duration_s = float(move_duration_s) + float(hold_duration_s)
    steps = max(1, int(np.ceil(duration_s / max(dt, 1e-9))))

    res = RolloutResult(
        pose=pose_label,
        label=label or ("corridor+cbf" if (corridor and cbf) else str(controller_kind)),
        config=str(Path(config_path)),
        controller_kind=str(controller_kind),
        corridor=bool(ctrl_cfg.get("yz_corridor_enabled", False)),
        cbf=bool(ctrl_cfg.get("manipulability_cbf", False)),
        target_delta_m=float(target_delta_m),
        move_duration_s=float(move_duration_s),
        hold_duration_s=float(hold_duration_s),
        validated_regime=bool(validated_regime),
    )
    gravity_scratch = mujoco.MjData(model)
    ee_now = start_ee.copy()
    q_pan: list[float] = []
    qp_times: list[float] = []
    row_active = [0, 0, 0, 0]
    x_err_final = float("nan")

    for _ in range(steps):
        t_s = float(data.time)
        target_now, target_vel_now = x_profile_target(
            "min_jerk_move_hold", x0, float(target_delta_m), t_s, duration_s,
            move_duration_s=float(move_duration_s),
        )
        target_ee_pos = start_ee.copy()
        target_ee_pos[0] = target_now
        target_ee_vel = np.zeros(3, dtype=np.float64)
        target_ee_vel[0] = target_vel_now

        state = build_mujoco_state(
            model, data, site_id=site_id, joint_ids=joint_ids, time_s=t_s, dt_s=dt,
            target_x=float(target_now), target_x_vel=float(target_vel_now), target_x_accel=0.0,
            target_axis=float(target_now), target_axis_vel=float(target_vel_now),
            target_ee_pos=target_ee_pos, target_ee_vel=target_ee_vel,
            reference_quat=state0.reference_quat, hold_current_pose=False,
            transport_axis_index=0,
            gravity_compensation=bool(str(mj_cfg.get("gravity_mode", "gravity_comp")) == "gravity_comp"),
            gravity_scratch_data=gravity_scratch,
        )
        tau, diag = adapter.step(state=state)

        jac = np.asarray(state.jacobian, dtype=np.float64)
        res.min_manipulability = min(res.min_manipulability, float(manipulability(jac)))
        cond_j = float(np.linalg.cond(jac))
        if np.isfinite(cond_j):
            res.max_cond_j = max(res.max_cond_j, cond_j)

        out = diag.get("controller_output", {}) or {}
        rows = out.get("yz_corridor_active_rows")
        if rows is not None:
            rows = tuple(bool(r) for r in rows)
            if any(rows):
                res.corridor_active_steps += 1
            for i in range(4):
                row_active[i] += int(rows[i])
        if out.get("manipulability_cbf_active"):
            res.cbf_active_steps += 1
        if out.get("yz_corridor_feasible") is False or out.get("manipulability_cbf_feasible") is False:
            res.infeasible_steps += 1
        if "qp_solve_time_s" in out:
            qp_times.append(float(out["qp_solve_time_s"]))
            res.qp_rows = int(out.get("qp_num_ineq_rows", 0))
        if "x_error" in out:
            x_err_final = abs(float(out["x_error"]))

        res.max_abs_qd_radps = max(res.max_abs_qd_radps, float(np.max(np.abs(state.qd))))
        res.max_abs_tau_nm = max(res.max_abs_tau_nm, float(np.max(np.abs(np.asarray(tau)))))
        q_pan.append(float(np.asarray(state.q, dtype=np.float64)[0]))
        excluded = out.get("task_excluded_joints") or ()
        if excluded is not None and len(excluded) and "tau_preclip" in out and "tau_hold" in out:
            pre = np.asarray(out["tau_preclip"], dtype=np.float64)
            hold = np.asarray(out["tau_hold"], dtype=np.float64)
            idx = [int(i) for i in excluded]
            if not np.array_equal(pre[idx], hold[idx]):
                res.excluded_joint_pin_violations += 1
        res.max_abs_shoulder_pan_tau_nm = max(
            res.max_abs_shoulder_pan_tau_nm, abs(float(np.asarray(tau)[0]))
        )
        res.max_orientation_error_rad = max(
            res.max_orientation_error_rad, float(diag.get("orientation_error_norm", 0.0))
        )

        ee_now = np.asarray(state.ee_pos, dtype=np.float64).copy()
        res.max_abs_y_drift_m = max(res.max_abs_y_drift_m, abs(float(ee_now[1] - start_ee[1])))
        res.max_abs_z_drift_m = max(res.max_abs_z_drift_m, abs(float(ee_now[2] - start_ee[2])))

        if not bool(diag.get("safety_ok", True)):
            res.guard_tripped = True
            res.guard_reason = str(diag.get("safety_reason", "") or "unknown")
            res.guard_time_s = t_s
            break

        data.ctrl[:6] = np.asarray(tau, dtype=np.float64).reshape(6)
        mujoco.mj_step(model, data)
        res.steps += 1

    if q_pan:
        pan = np.asarray(q_pan, dtype=np.float64)
        res.shoulder_pan_range_rad = float(pan.max() - pan.min())
        res.max_abs_shoulder_pan_dev_rad = float(
            np.max(np.abs(pan - float(np.asarray(q_start, dtype=np.float64).reshape(6)[0])))
        )
    res.achieved_delta_m = float(ee_now[0] - start_ee[0])
    if abs(float(target_delta_m)) > 0.0:
        res.tracking_fraction = res.achieved_delta_m / float(target_delta_m)
    res.final_abs_x_error_m = x_err_final
    res.final_abs_y_drift_m = abs(float(ee_now[1] - start_ee[1]))
    res.final_abs_z_drift_m = abs(float(ee_now[2] - start_ee[2]))
    res.corridor_row_active_steps = tuple(row_active)  # type: ignore[assignment]
    if qp_times:
        arr = np.asarray(qp_times, dtype=np.float64)
        res.qp_mean_s = float(np.mean(arr))
        res.qp_p95_s = float(np.percentile(arr, 95))
        res.qp_max_s = float(np.max(arr))
    return res


# --------------------------------------------------------------------------- #
def _print_profile(r: ProfileResult) -> None:
    print(f"\n=== mu / cond(J) / sigma_min profile: pose={r.pose} joint={r.joint_index} ===")
    print(f"{'q_joint':>10} {'mu':>14} {'cond(J)':>13} {'sigma_min':>13}")
    for v, mu, c, s in zip(r.values, r.manipulability, r.cond_j, r.sigma_min_j):
        print(f"{v:>10.4f} {mu:>14.6e} {c:>13.4e} {s:>13.4e}")


def _print_timing(rows: list[TimingResult]) -> None:
    print(f"\n=== isolated compute() cost at the start pose "
          f"(budget = {DIRECT_TORQUE_BUDGET_S*1e3:.1f} ms/cycle @ 500 Hz) ===")
    print(f"{'case':>28} {'rows':>5} {'mean_ms':>9} {'p95_ms':>9} {'max_ms':>9} "
          f"{'qp_mean_ms':>11} {'qp_p95_ms':>10} {'over_budget':>12}")
    for r in rows:
        print(
            f"{r.label:>28} {r.n_ineq_rows:>5} {r.mean_total_s*1e3:>9.3f} {r.p95_total_s*1e3:>9.3f} "
            f"{r.max_total_s*1e3:>9.3f} {r.mean_qp_s*1e3:>11.3f} {r.p95_qp_s*1e3:>10.3f} "
            f"{r.over_budget_fraction:>12.2f}"
        )


def _print_rollouts(rows: list[RolloutResult]) -> None:
    print("\n=== closed-loop move+hold at the real deployment pose ===")
    header = (
        f"{'label':>22} {'dx':>7} {'val':>4} {'track':>7} {'maxY':>8} {'maxZ':>8} "
        f"{'ori':>7} {'panDeg':>7} {'min_mu':>10} {'|qd|':>6} {'tau':>7} {'corr':>6} {'cbf':>5} "
        f"{'qp_ms':>7} {'guard':>24}"
    )
    print(header)
    for r in rows:
        print(
            f"{r.label:>22} {r.target_delta_m:>7.3f} {('Y' if r.validated_regime else 'N'):>4} "
            f"{r.tracking_fraction:>7.3f} {r.max_abs_y_drift_m:>8.4f} {r.max_abs_z_drift_m:>8.4f} "
            f"{r.max_orientation_error_rad:>7.4f} "
            f"{np.degrees(r.shoulder_pan_range_rad):>7.2f} {r.min_manipulability:>10.3e} "
            f"{r.max_abs_qd_radps:>6.3f} {r.max_abs_tau_nm:>7.2f} {r.corridor_active_steps:>6} "
            f"{r.cbf_active_steps:>5} "
            f"{(r.qp_mean_s*1e3 if np.isfinite(r.qp_mean_s) else float('nan')):>7.3f} "
            f"{(r.guard_reason or '-'):>24}"
        )
    print("  val = dx inside the validated (<=0.06 m) corridor-calibration regime; "
          "N rows are informational only.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mode", choices=["rollout", "profile"], default="rollout")
    parser.add_argument(
        "--deltas", nargs="+", type=float, default=[0.02, -0.02, 0.06, -0.06, 0.12, -0.12],
        help="X displacements in m. |dx| > 0.06 is an UNVALIDATED regime (informational).",
    )
    parser.add_argument("--move-duration", type=float, default=1.5)
    parser.add_argument("--hold-duration", type=float, default=1.0)
    parser.add_argument("--timing-calls", type=int, default=200)
    parser.add_argument(
        "--osc-widened-guards", action="store_true",
        help="Additionally run the OSC baseline with the same widened drift guards the "
             "enabled config uses, so the comparison is not confounded by the guard "
             "threshold difference alone.",
    )
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)

    payload: dict[str, Any] = {"mode": args.mode, "arm_q0": ARM_Q0.tolist()}
    if args.mode == "profile":
        prof = run_profile(ARM_Q0)
        _print_profile(prof)
        timing = run_timing_profile(ARM_Q0, n_calls=int(args.timing_calls))
        _print_timing(timing)
        payload["profile"] = asdict(prof)
        payload["timing"] = [asdict(t) for t in timing]
    else:
        widened = {
            "max_abs_y_drift_m": 0.06,
            "max_abs_z_drift_m": 0.06,
            "max_abs_orthogonal_drift_m": 0.06,
        }
        rows: list[RolloutResult] = []
        for delta in args.deltas:
            validated = abs(float(delta)) <= 0.06 + 1e-12
            common = dict(
                target_delta_m=float(delta),
                move_duration_s=float(args.move_duration),
                hold_duration_s=float(args.hold_duration),
                validated_regime=validated,
            )
            rows.append(run_rollout(
                ARM_Q0, controller_kind="impedance", config_path=OSC_CONFIG,
                label="osc_tuned", **common,
            ))
            if args.osc_widened_guards:
                rows.append(run_rollout(
                    ARM_Q0, controller_kind="impedance", config_path=OSC_CONFIG,
                    label="osc_tuned_wide", safety_overrides=widened, **common,
                ))
            # NOTE: every "new_*" row below reads NEW_CONFIG_ENABLED and
            # differs ONLY in the two mechanism flags. Using NEW_CONFIG for
            # the flags-off row instead would ALSO change the drift guards
            # (0.03 vs 0.06 m), so a flags-off run would stop earlier for a
            # reason that has nothing to do with the flags -- measured, not
            # hypothetical: at dx=0.06 m that confound alone moved the
            # tracking fraction from 0.706 to 0.852.
            rows.append(run_rollout(
                ARM_Q0, config_path=NEW_CONFIG_ENABLED, corridor=False, cbf=False,
                label="new_flags_off", **common,
            ))
            # The shipped default-off config, guards and all, run once per dx
            # so the file that is meant to be the safe baseline is actually
            # exercised rather than merely described.
            rows.append(run_rollout(
                ARM_Q0, config_path=NEW_CONFIG,
                label="new_default_cfg", **common,
            ))
            rows.append(run_rollout(
                ARM_Q0, config_path=NEW_CONFIG_ENABLED, corridor=True, cbf=False,
                label="new_corridor_only", **common,
            ))
            rows.append(run_rollout(
                ARM_Q0, config_path=NEW_CONFIG_ENABLED, corridor=False, cbf=True,
                label="new_cbf_only", **common,
            ))
            rows.append(run_rollout(
                ARM_Q0, config_path=NEW_CONFIG_ENABLED, corridor=True, cbf=True,
                label="new_corridor_cbf", **common,
            ))
        _print_rollouts(rows)
        payload["rollouts"] = [asdict(r) for r in rows]

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
