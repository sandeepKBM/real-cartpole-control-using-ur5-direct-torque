#!/usr/bin/env python3
"""Uniform damping vs. singularity-consistent inversion (SCI) on the REAL UR5e.

``controller_core/x_axis_cartesian_impedance``'s two pre-existing
near-singularity mechanisms are both UNIFORM across task directions: the scalar
``lambda_regularization * I`` inside ``Lambda = (J M^-1 J^T + eps I)^-1``, and
``singular_scale`` (one scalar multiplying the whole wrench). At a UR wrist
singularity exactly ONE task direction is genuinely lost, so uniform damping
also throws away authority in the five directions the arm still has.
``svd_singularity_filtering`` replaces that with a per-direction damped inverse.

This script measures the difference on the real model
(``assets/ur5e_torque/scene.xml``) -- never a toy plant -- two ways:

  ``--mode authority`` (default, instant)
      Static, per-singular-direction analysis at a pose: the task-space singular
      values of ``J M^-1 J^T``, the fraction of the ideal undamped response each
      direction keeps under uniform eps vs. under SCI, and the task acceleration
      each scheme actually delivers for a pure transport-axis command
      (``a = J M^-1 tau``, i.e. what the arm would really do).

  ``--mode rollout``
      Real closed-loop move+hold through the same adapter pipeline every sim
      tool in this repo uses (``build_initial_state_and_adapter`` /
      ``build_mujoco_state`` / ``adapter.step()`` / ``mujoco.mj_step``), run once
      per scheme so tracking, drift, guard trips and peak torque are directly
      comparable. Sweeps +X and -X by default (AGENTS.md sec 7).

Examples
--------
    python tools/diagnostics/svd_singularity_filtering_sim_check.py
    python tools/diagnostics/svd_singularity_filtering_sim_check.py \
        --mode rollout --deltas 0.02 -0.02 --json out.json
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
    x_profile_target,
)

SCENE_XML = REPO_ROOT / "assets" / "ur5e_torque" / "scene.xml"
DEFAULT_CONFIG = "config/ur5e_mujoco_torque_osc_tuned.yaml"

#: pose name -> joint vector. ``height_alpha_0_5`` is the motivating case: a
#: genuine UR wrist singularity (wrist_2 == 0, cond(J) = 7.3e16). The
#: ``_wrist2_offset`` variant is the same pose nudged OFF the singularity, and
#: ``mega_search_winner`` is a well-conditioned pose -- both included so a
#: reader can see that SCI does nothing surprising away from a singularity.
POSES: dict[str, np.ndarray] = {
    "height_alpha_0_5": HEIGHT_ALPHA_0_5_Q,
    "height_alpha_0_5_wrist2_offset": HEIGHT_ALPHA_0_5_WRIST2_OFFSET_Q,
    "mega_search_winner": MEGA_SEARCH_WINNER_Q,
    # The -45 deg base-rotation clearance pose behind this repo's open,
    # gain-search-exhausted Y-drift finding (AGENTS.md sec 3). Included purely
    # as an observation surface: the per-direction table shows how much Y
    # authority today's uniform eps is discarding at that pose. Not a claimed
    # fix for it -- see this file's report/AGENTS.md before drawing conclusions.
    "height_alpha_0_5_clearance_neg45": HEIGHT_ALPHA_0_5_CLEARANCE_Q,
}

AXIS_NAMES = ("X", "Y", "Z", "Rx", "Ry", "Rz")


def load_controller_config(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp) or {}


def _seed_model(q_start: np.ndarray):
    model, data, site_id, joint_ids, _ = load_model(SCENE_XML)
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    q_start = np.asarray(q_start, dtype=np.float64).reshape(6)
    for idx, jid in enumerate(joint_ids):
        data.qpos[int(model.jnt_qposadr[jid])] = float(q_start[idx])
    mujoco.mj_forward(model, data)
    return model, data, site_id, joint_ids


# --------------------------------------------------------------------------- #
# Mode 1: static per-direction authority analysis.
# --------------------------------------------------------------------------- #
@dataclass
class AuthorityResult:
    pose: str
    config: str
    cond_j: float = 0.0
    #: Mass-weighted task-space singular values, sqrt(eigh(J M^-1 J^T)), ascending.
    sigma: list[float] = field(default_factory=list)
    #: Fraction of the ideal undamped response each direction keeps.
    attenuation_uniform: list[float] = field(default_factory=list)
    attenuation_sci: list[float] = field(default_factory=list)
    #: Dominant world axis of each singular direction (largest |component|).
    direction_dominant_axis: list[str] = field(default_factory=list)
    lambda_regularization: float = 0.0
    svd_sigma_threshold: float = 0.0
    svd_lambda_max: float = 0.0
    #: For a pure transport-axis command: task acceleration actually delivered,
    #: a = J M^-1 tau_task, under each scheme, against what was commanded.
    commanded_task_accel: float = 0.0
    delivered_task_accel_uniform: float = 0.0
    delivered_task_accel_sci: float = 0.0
    #: Norm of the SPURIOUS acceleration on the five non-commanded task rows.
    cross_axis_leak_uniform: float = 0.0
    cross_axis_leak_sci: float = 0.0
    tau_task_norm_uniform: float = 0.0
    tau_task_norm_sci: float = 0.0


def run_authority_analysis(
    q_start: np.ndarray,
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    pose_label: str = "custom",
    probe_delta_m: float = 0.01,
    transport_axis: int = 0,
) -> AuthorityResult:
    """Per-direction authority at one pose, uniform eps vs. SCI.

    Runs the REAL controller twice on the REAL model's Jacobian/mass matrix
    (identical state, only the flag differs) and reports what task acceleration
    each scheme's commanded torque would actually produce.
    """
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    cfg = load_controller_config(config_path)
    ctrl_cfg = dict(cfg["controller"])
    mj_cfg = cfg.get("mujoco", {}) or {}

    model, data, site_id, joint_ids = _seed_model(q_start)
    state = build_mujoco_state(
        model, data, site_id=site_id, joint_ids=joint_ids, time_s=0.0,
        dt_s=float(model.opt.timestep), target_x=0.0,
        gravity_compensation=bool(str(mj_cfg.get("gravity_mode", "gravity_comp")) == "gravity_comp"),
    )
    st = state.as_robot_state()
    J = np.asarray(st["jacobian"], dtype=np.float64).reshape(6, 6)
    M = np.asarray(st["mass_matrix"], dtype=np.float64).reshape(6, 6)
    m_inv = np.linalg.inv(M)

    a_mat = J @ m_inv @ J.T
    eigvals, eigvecs = np.linalg.eigh(a_mat)
    sigma = np.sqrt(np.maximum(eigvals, 0.0))

    # Import here so the module stays importable without touching controller_core
    # at argparse time.
    from controller_core.x_axis_cartesian_impedance import (
        CartesianImpedanceConfig,
        XAxisCartesianImpedanceController,
    )

    def _build(**overrides):
        merged = dict(ctrl_cfg)
        merged.update(overrides)
        c = CartesianImpedanceConfig.from_controller_yaml_section(merged)
        ctl = XAxisCartesianImpedanceController(c)
        ctl.reset_from_state(st)
        return ctl

    uniform_ctl = _build()
    sci_ctl = _build(svd_singularity_filtering=True)

    probe = dict(st)
    probe["target_x"] = float(np.asarray(st["ee_pos"])[transport_axis]) + float(probe_delta_m)
    probe["target_axis"] = probe["target_x"]
    probe["transport_axis_index"] = int(transport_axis)

    out_u = uniform_ctl.compute(probe)
    out_s = sci_ctl.compute(probe)

    eps = float(uniform_ctl.cfg.lambda_regularization)
    a_u = J @ m_inv @ np.asarray(out_u.tau_task_nominal, dtype=np.float64)
    a_s = J @ m_inv @ np.asarray(out_s.tau_task_nominal, dtype=np.float64)
    other = [i for i in range(6) if i != transport_axis]

    return AuthorityResult(
        pose=pose_label,
        config=str(config_path.relative_to(REPO_ROOT)),
        cond_j=float(np.linalg.cond(J)),
        sigma=[float(x) for x in sigma],
        attenuation_uniform=[float(s * s / (s * s + eps)) for s in sigma],
        attenuation_sci=[float(x) for x in np.asarray(out_s.svd_direction_attenuation)],
        direction_dominant_axis=[AXIS_NAMES[int(np.argmax(np.abs(eigvecs[:, i])))] for i in range(6)],
        lambda_regularization=eps,
        svd_sigma_threshold=float(sci_ctl.cfg.svd_sigma_threshold),
        svd_lambda_max=float(sci_ctl.cfg.svd_lambda_max),
        commanded_task_accel=float(out_u.wrench[transport_axis]),
        delivered_task_accel_uniform=float(a_u[transport_axis]),
        delivered_task_accel_sci=float(a_s[transport_axis]),
        cross_axis_leak_uniform=float(np.linalg.norm(a_u[other])),
        cross_axis_leak_sci=float(np.linalg.norm(a_s[other])),
        tau_task_norm_uniform=float(np.linalg.norm(out_u.tau_task_nominal)),
        tau_task_norm_sci=float(np.linalg.norm(out_s.tau_task_nominal)),
    )


# --------------------------------------------------------------------------- #
# Mode 2: real closed-loop move+hold rollout.
# --------------------------------------------------------------------------- #
@dataclass
class RolloutResult:
    pose: str
    config: str
    scheme: str
    target_delta_m: float
    move_duration_s: float
    hold_duration_s: float
    steps: int = 0
    achieved_delta_m: float = 0.0
    tracking_fraction: float = 0.0
    final_axis_error_m: float = 0.0
    max_abs_y_drift_m: float = 0.0
    max_abs_z_drift_m: float = 0.0
    max_orientation_error_rad: float = 0.0
    max_abs_qd_radps: float = 0.0
    max_abs_tau_nm: float = 0.0
    max_torque_clip_fraction: float = 0.0
    max_cond_j: float = 0.0
    guard_tripped: bool = False
    guard_reason: str = ""
    guard_time_s: float | None = None


def run_rollout(
    q_start: np.ndarray,
    *,
    scheme: str,
    config_path: str | Path = DEFAULT_CONFIG,
    target_delta_m: float = 0.02,
    move_duration_s: float = 1.5,
    hold_duration_s: float = 1.0,
    pose_label: str = "custom",
    transport_axis: int = 0,
    gravity_source: str | None = None,
    coriolis_feedforward: bool | None = None,
) -> RolloutResult:
    """One real closed-loop move+hold, with SCI either on (``scheme='sci'``) or
    off (``scheme='uniform'``). Everything else is byte-identical between the
    two runs, so any difference is attributable to the flag alone.

    ``gravity_source``/``coriolis_feedforward`` override the config's own
    ``mujoco:`` section. Both default to None (use the config). The only reason
    to override is that the tuned config selects ``pinocchio``, which is an
    optional dependency; MuJoCo's own ``qfrc_bias`` gravity is parity-checked
    against it to <1e-8 Nm (AGENTS.md sec 3), so swapping it does not affect
    what this script measures."""
    if scheme not in ("uniform", "sci"):
        raise ValueError(f"scheme must be 'uniform' or 'sci'; got {scheme!r}")
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    cfg = load_controller_config(config_path)
    ctrl_cfg = dict(cfg["controller"])
    if scheme == "sci":
        ctrl_cfg["svd_singularity_filtering"] = True
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
        scheme=scheme,
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

        cond_j = float(np.linalg.cond(np.asarray(state.jacobian, dtype=np.float64)))
        if np.isfinite(cond_j):
            res.max_cond_j = max(res.max_cond_j, cond_j)
        res.max_torque_clip_fraction = max(
            res.max_torque_clip_fraction, float(diag.get("torque_clip_fraction", 0.0))
        )
        res.max_orientation_error_rad = max(
            res.max_orientation_error_rad, float(diag.get("orientation_error_norm", 0.0))
        )
        res.max_abs_qd_radps = max(res.max_abs_qd_radps, float(np.max(np.abs(state.qd))))
        res.max_abs_tau_nm = max(res.max_abs_tau_nm, float(np.max(np.abs(np.asarray(tau)))))
        res.final_axis_error_m = float(diag.get("axis_error", 0.0))

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
def _print_authority(r: AuthorityResult) -> None:
    print(f"\n=== per-direction authority: pose={r.pose} config={r.config} ===")
    print(f"cond(J) = {r.cond_j:.4e}   eps(uniform) = {r.lambda_regularization}   "
          f"SCI: sigma_threshold={r.svd_sigma_threshold} lambda_max={r.svd_lambda_max:.6f}")
    print(f"{'dir':>4} {'sigma':>13} {'dom.axis':>9} {'keep(uniform)':>14} {'keep(SCI)':>11}")
    for i, s in enumerate(r.sigma):
        print(f"{i:>4} {s:>13.6e} {r.direction_dominant_axis[i]:>9} "
              f"{r.attenuation_uniform[i]:>14.4f} {r.attenuation_sci[i]:>11.4f}")
    cmd = r.commanded_task_accel
    print(f"\ncommanded task accel   : {cmd:.6f} m/s^2")
    frac_u = r.delivered_task_accel_uniform / cmd if cmd else float("nan")
    frac_s = r.delivered_task_accel_sci / cmd if cmd else float("nan")
    print(f"delivered (uniform eps): {r.delivered_task_accel_uniform:.6f}  ({100 * frac_u:.2f}% of commanded)")
    print(f"delivered (SCI)        : {r.delivered_task_accel_sci:.6f}  ({100 * frac_s:.2f}% of commanded)")
    print(f"cross-axis leak norm   : uniform {r.cross_axis_leak_uniform:.6e}   SCI {r.cross_axis_leak_sci:.6e}")
    print(f"|tau_task|             : uniform {r.tau_task_norm_uniform:.6f}   SCI {r.tau_task_norm_sci:.6f}")


def _print_rollouts(rows: list[RolloutResult]) -> None:
    print(f"\n=== closed-loop move+hold ({rows[0].pose}, {rows[0].config}) ===")
    hdr = (f"{'dx[m]':>8} {'scheme':>8} {'track%':>8} {'|Ydrift|':>10} {'|Zdrift|':>10} "
           f"{'ori[rad]':>9} {'max|qd|':>9} {'max|tau|':>9} {'guard':>28}")
    print(hdr)
    for r in rows:
        guard = f"{r.guard_reason}@{r.guard_time_s:.2f}s" if r.guard_tripped else "-"
        print(f"{r.target_delta_m:>8.3f} {r.scheme:>8} {100 * r.tracking_fraction:>8.2f} "
              f"{r.max_abs_y_drift_m:>10.5f} {r.max_abs_z_drift_m:>10.5f} "
              f"{r.max_orientation_error_rad:>9.4f} {r.max_abs_qd_radps:>9.4f} "
              f"{r.max_abs_tau_nm:>9.3f} {guard:>28}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["authority", "rollout", "both"], default="authority")
    ap.add_argument("--pose", choices=sorted(POSES), default="height_alpha_0_5")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--transport-axis", type=int, default=0, choices=[0, 1, 2])
    ap.add_argument("--probe-delta-m", type=float, default=0.01)
    # Both directions by default -- AGENTS.md sec 7.
    ap.add_argument("--deltas", type=float, nargs="+", default=[0.02, -0.02])
    ap.add_argument("--move-duration", type=float, default=1.5)
    ap.add_argument("--hold-duration", type=float, default=1.0)
    ap.add_argument("--json", default=None, help="write results to this JSON path")
    ap.add_argument("--gravity-source", default=None, choices=["mujoco_qfrc", "pinocchio"],
                    help="override the config's mujoco.gravity_source (rollout mode)")
    ap.add_argument("--no-coriolis", action="store_true",
                    help="force coriolis_feedforward off (rollout mode)")
    ap.add_argument("--start-q-rad", type=float, nargs=6, default=None,
                    metavar=("Q1", "Q2", "Q3", "Q4", "Q5", "Q6"),
                    help="Override --pose with an arbitrary 6-joint start pose in radians "
                         "(e.g. a live real-robot probe) -- takes precedence over --pose.")
    args = ap.parse_args(argv)

    if args.start_q_rad is not None:
        q_start = np.asarray(args.start_q_rad, dtype=np.float64)
        pose_label = "custom_start_q"
    else:
        q_start = POSES[args.pose]
        pose_label = args.pose
    payload: dict[str, Any] = {}

    if args.mode in ("authority", "both"):
        r = run_authority_analysis(
            q_start, config_path=args.config, pose_label=pose_label,
            probe_delta_m=args.probe_delta_m, transport_axis=args.transport_axis,
        )
        _print_authority(r)
        payload["authority"] = asdict(r)

    if args.mode in ("rollout", "both"):
        rows: list[RolloutResult] = []
        for delta in args.deltas:
            for scheme in ("uniform", "sci"):
                rows.append(run_rollout(
                    q_start, scheme=scheme, config_path=args.config,
                    target_delta_m=float(delta), move_duration_s=args.move_duration,
                    hold_duration_s=args.hold_duration, pose_label=pose_label,
                    transport_axis=args.transport_axis,
                    gravity_source=args.gravity_source,
                    coriolis_feedforward=False if args.no_coriolis else None,
                ))
        _print_rollouts(rows)
        payload["rollouts"] = [asdict(r) for r in rows]

    if args.json:
        Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
