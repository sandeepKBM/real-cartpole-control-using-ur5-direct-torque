#!/usr/bin/env python3
"""Row+column restriction of the split translation task, on the REAL UR5e model.

``split_base_wrist_task`` restricts the translation task to a subset of joint
COLUMNS but always uses all three position ROWS. That is unusable for a joint
set whose position sub-Jacobian is structurally rank-2 -- notably
``{shoulder_lift, elbow, wrist_1}`` (the UR planar sub-chain: three parallel
axes, so 3D linear velocity is out of reach at every pose). But a front/back
transport task is ONE-dimensional, and world X lies inside the 2D subspace
those joints do span. ``split_base_wrist_task_dims`` (2026-08-12) selects the
task rows too, turning an unusable 3x3 into a well-posed 1x3.

This script measures that on ``assets/ur5e_torque/scene.xml`` -- never a toy
plant -- two ways:

  ``--mode authority`` (default, instant)
      Static kinematics at a pose: for each candidate (rows x columns) block,
      its rank, cond (or norm, for a single row), and singular values. This is
      the screen that says whether a row/column selection is usable AT ALL
      before any rollout is worth running.

  ``--mode rollout``
      Real closed-loop move+hold through the same adapter pipeline every sim
      tool in this repo uses (``build_initial_state_and_adapter`` /
      ``build_mujoco_state`` / ``adapter.step()`` / ``mujoco.mj_step``), with
      arbitrary controller-config overrides, so a row-reduced task can be
      compared directly against the 3-row task and against a
      posture-disabled control. Sweeps +X and -X by default (AGENTS.md sec 7).

Examples
--------
    python tools/diagnostics/split_base_wrist_task_dims_sim_check.py
    python tools/diagnostics/split_base_wrist_task_dims_sim_check.py \
        --mode rollout --deltas 0.02 -0.02
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import mujoco
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hardware.poses import (  # noqa: E402
    HEIGHT_ALPHA_0_5_CLEARANCE_Q,
    HEIGHT_ALPHA_0_5_Q,
    MEGA_SEARCH_WINNER_Q,
)
from simulation.ur5e_mujoco_torque import (  # noqa: E402
    build_initial_state_and_adapter,
    build_mujoco_state,
    load_model,
    x_profile_target,
)

SCENE_XML = REPO_ROOT / "assets" / "ur5e_torque" / "scene.xml"
DEFAULT_CONFIG = "config/ur5e_mujoco_torque_osc_tuned_split_base_wrist_lift_elbow_wrist1_x_only.yaml"

#: The real robot pose this feature was built for (measured 2026-08-12, wrist_2
#: already wrapped into the model's valid range): shoulder_pan held for base
#: clearance, wrist_2 held away from its physical singularity limit, leaving
#: shoulder_lift/elbow/wrist_1 to do a 1D front/back transport.
USER_REAL_POSE_Q = np.array(
    [-2.3688, -2.1801, -1.8838, -0.7962, 0.004714693, 0.0206], dtype=np.float64
)

POSES: dict[str, np.ndarray] = {
    "user_real_pose": USER_REAL_POSE_Q,
    "height_alpha_0_5": HEIGHT_ALPHA_0_5_Q,
    "height_alpha_0_5_clearance_neg45": HEIGHT_ALPHA_0_5_CLEARANCE_Q,
    "mega_search_winner": MEGA_SEARCH_WINNER_Q,
}

ROW_NAMES = ("X", "Y", "Z")
JOINT_SHORT = ("pan", "lift", "elbow", "w1", "w2", "w3")

#: The motivating selection: shoulder_lift/elbow/wrist_1 columns, world-X row.
LIFT_ELBOW_WRIST1 = (1, 2, 3)
X_ONLY = (0,)


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
    with Path(path).open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp) or {}


def model_state(q_start: np.ndarray) -> dict[str, Any]:
    """Raw controller state dict at a pose (real J and mass matrix)."""
    model, data, site_id, joint_ids = _seed_model(q_start)
    return dict(
        build_mujoco_state(
            model, data, site_id=site_id, joint_ids=joint_ids, time_s=0.0,
            dt_s=float(model.opt.timestep), target_x=0.0,
        ).as_robot_state()
    )


# --------------------------------------------------------------------------- #
# Mode 1: static block-conditioning screen.
# --------------------------------------------------------------------------- #
@dataclass
class BlockResult:
    pose: str
    rows: tuple[int, ...]
    cols: tuple[int, ...]
    shape: tuple[int, int]
    rank: int
    #: cond() for >=2 rows, the block NORM for a single row -- matching what
    #: the controller itself reports as jacobian_cond in each case.
    cond_or_norm: float
    singular_values: list[float] = field(default_factory=list)
    usable: bool = False


def analyze_block(q_start: np.ndarray, rows: Sequence[int], cols: Sequence[int],
                  *, pose_label: str = "custom") -> BlockResult:
    st = model_state(q_start)
    J = np.asarray(st["jacobian"], dtype=np.float64).reshape(6, 6)
    rows = tuple(int(r) for r in rows)
    cols = tuple(int(c) for c in cols)
    block = J[np.ix_(rows, cols)]
    sv = np.linalg.svd(block, compute_uv=False)
    rank = int(np.linalg.matrix_rank(block))
    metric = float(np.linalg.cond(block)) if len(rows) > 1 else float(np.linalg.norm(block))
    return BlockResult(
        pose=pose_label, rows=rows, cols=cols, shape=(len(rows), len(cols)),
        rank=rank, cond_or_norm=metric,
        singular_values=[float(s) for s in sv],
        # "Usable" = the task rows are actually spanned: full row rank, and for
        # a single row a non-negligible norm.
        usable=bool(rank == len(rows) and float(sv.min()) > 1e-9),
    )


def run_authority_analysis(q_start: np.ndarray, *, pose_label: str = "custom") -> list[BlockResult]:
    """Screen the interesting (rows, cols) combinations at one pose."""
    combos: list[tuple[tuple[int, ...], tuple[int, ...]]] = [
        ((0, 1, 2), (0, 1, 2)),            # historical default: 3 rows, base joints
        ((0, 1, 2), LIFT_ELBOW_WRIST1),    # 3 rows, the planar sub-chain -> rank 2, unusable
        (X_ONLY, LIFT_ELBOW_WRIST1),       # the new case: 1 row, planar sub-chain
        ((0, 2), LIFT_ELBOW_WRIST1),       # 2 rows (X,Z) over the same columns
        ((0, 1), LIFT_ELBOW_WRIST1),       # 2 rows (X,Y) over the same columns
        (X_ONLY, (0, 1, 2)),               # 1 row over the base joints, for scale
        ((0, 1, 2), (2, 3, 4)),            # the pan-free 3-row set (needs wrist_2 != 0)
    ]
    return [analyze_block(q_start, r, c, pose_label=pose_label) for r, c in combos]


# --------------------------------------------------------------------------- #
# Mode 2: real closed-loop move+hold rollout.
# --------------------------------------------------------------------------- #
@dataclass
class RolloutResult:
    pose: str
    config: str
    label: str
    target_delta_m: float
    move_duration_s: float
    hold_duration_s: float
    steps: int = 0
    achieved_delta_m: float = 0.0
    tracking_fraction: float = 0.0
    final_axis_error_m: float = 0.0
    max_abs_y_drift_m: float = 0.0
    max_abs_z_drift_m: float = 0.0
    final_abs_y_drift_m: float = 0.0
    final_abs_z_drift_m: float = 0.0
    max_orientation_error_rad: float = 0.0
    max_abs_qd_radps: float = 0.0
    max_abs_tau_nm: float = 0.0
    max_torque_clip_fraction: float = 0.0
    #: Max |q - q_start| over the joints the task is NOT allowed to drive.
    max_held_joint_excursion_rad: float = 0.0
    #: Max |tau_task| on those same joints -- must stay exactly 0.
    max_held_joint_task_torque_nm: float = 0.0
    #: What the controller reported it actually ran with (read back from its
    #: own output, not from the config we passed in).
    task_dims_used: tuple[int, ...] = ()
    active_joints_used: tuple[int, ...] = ()
    guard_tripped: bool = False
    guard_reason: str = ""
    guard_time_s: float | None = None


def run_rollout(
    q_start: np.ndarray,
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    ctrl_overrides: dict[str, Any] | None = None,
    target_delta_m: float = 0.02,
    move_duration_s: float = 1.5,
    hold_duration_s: float = 1.0,
    pose_label: str = "custom",
    label: str = "",
    transport_axis: int = 0,
    gravity_source: str | None = "mujoco_qfrc",
    coriolis_feedforward: bool | None = False,
) -> RolloutResult:
    """One real closed-loop move+hold with the given controller-config overrides.

    ``gravity_source``/``coriolis_feedforward`` default to MuJoCo's own
    ``qfrc_bias`` and off, so the rollout does not depend on the optional
    pinocchio package; that gravity source is parity-checked against pinocchio
    to <1e-8 Nm (AGENTS.md sec 3). Pass ``None`` to use the config's own.
    """
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    cfg = load_controller_config(config_path)
    ctrl_cfg = dict(cfg["controller"])
    ctrl_cfg.update(ctrl_overrides or {})
    mj_cfg = cfg.get("mujoco", {}) or {}

    active = ctrl_cfg.get("split_base_wrist_active_joints", None)
    held_joints = (
        [j for j in range(6) if j not in set(int(a) for a in active)]
        if (active is not None and bool(ctrl_cfg.get("split_base_wrist_task", False)))
        else []
    )

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
    q0 = np.asarray(state0.q, dtype=np.float64).copy()
    axis0 = float(start_ee[transport_axis])
    dt = float(model.opt.timestep)
    duration_s = float(move_duration_s) + float(hold_duration_s)
    steps = max(1, int(np.ceil(duration_s / max(dt, 1e-9))))

    res = RolloutResult(
        pose=pose_label,
        config=str(config_path.relative_to(REPO_ROOT)),
        label=label or "default",
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

        res.max_torque_clip_fraction = max(
            res.max_torque_clip_fraction, float(diag.get("torque_clip_fraction", 0.0))
        )
        res.max_orientation_error_rad = max(
            res.max_orientation_error_rad, float(diag.get("orientation_error_norm", 0.0))
        )
        res.max_abs_qd_radps = max(res.max_abs_qd_radps, float(np.max(np.abs(state.qd))))
        res.max_abs_tau_nm = max(res.max_abs_tau_nm, float(np.max(np.abs(np.asarray(tau)))))
        res.final_axis_error_m = float(diag.get("axis_error", 0.0))
        # The controller's own structured output rides along in the adapter's
        # diagnostics (simulation/ur5e_mujoco_torque.py::_controller_step).
        ctl_out = diag.get("controller_output", {}) or {}
        if res.steps == 0:
            res.task_dims_used = tuple(ctl_out.get("split_base_wrist_task_dims") or ())
            res.active_joints_used = tuple(ctl_out.get("split_base_wrist_active_joints") or ())
        if held_joints:
            q_now = np.asarray(state.q, dtype=np.float64)
            res.max_held_joint_excursion_rad = max(
                res.max_held_joint_excursion_rad,
                float(np.max(np.abs(q_now[held_joints] - q0[held_joints]))),
            )
            tau_task = np.asarray(
                ctl_out.get("tau_task_nominal", np.zeros(6)), dtype=np.float64
            ).ravel()
            if tau_task.size == 6:
                res.max_held_joint_task_torque_nm = max(
                    res.max_held_joint_task_torque_nm,
                    float(np.max(np.abs(tau_task[held_joints]))),
                )

        ee_now = np.asarray(state.ee_pos, dtype=np.float64).copy()
        res.final_abs_y_drift_m = abs(float(ee_now[1] - start_ee[1]))
        res.final_abs_z_drift_m = abs(float(ee_now[2] - start_ee[2]))
        res.max_abs_y_drift_m = max(res.max_abs_y_drift_m, res.final_abs_y_drift_m)
        res.max_abs_z_drift_m = max(res.max_abs_z_drift_m, res.final_abs_z_drift_m)

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


#: The rollout variants the report below (and the pytest layer over this file)
#: compare. Every one uses the SAME config file; only these overrides differ.
ROLLOUT_VARIANTS: dict[str, dict[str, Any]] = {
    # The new capability: 1 task row (world X) x 3 joint columns.
    "x_only_1x3": {},
    # Same columns, all three task rows -- the pre-existing mechanism, which
    # this joint set makes structurally singular (rank 2 of 3).
    "three_row_3x3": {"split_base_wrist_task_dims": None},
    # 1x3 with the posture spring switched off, to show what actually holds
    # the two unselected (Y, Z) task rows. The override itself is built in
    # _variant_overrides() below, since it has to start from the config file's
    # own gains block.
    "x_only_no_posture": {},
}


def _variant_overrides(name: str, ctrl_cfg: dict[str, Any]) -> dict[str, Any]:
    if name == "x_only_no_posture":
        gains = dict(ctrl_cfg.get("gains", {}))
        gains["kp_posture"] = 0.0
        gains["kd_posture"] = 0.0
        return {"gains": gains}
    return dict(ROLLOUT_VARIANTS[name])


def run_variant(
    name: str, q_start: np.ndarray, *, config_path: str | Path = DEFAULT_CONFIG, **kwargs
) -> RolloutResult:
    path = Path(config_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    ctrl_cfg = dict(load_controller_config(path)["controller"])
    return run_rollout(
        q_start, config_path=path, ctrl_overrides=_variant_overrides(name, ctrl_cfg),
        label=name, **kwargs,
    )


# --------------------------------------------------------------------------- #
def _print_blocks(rows: list[BlockResult]) -> None:
    print(f"\n=== task-block conditioning screen: pose={rows[0].pose} ===")
    print(f"{'rows':>12} {'cols':>22} {'shape':>7} {'rank':>5} {'cond|norm':>13} "
          f"{'sigma_min':>12} {'usable':>7}")
    for r in rows:
        rn = "".join(ROW_NAMES[i] for i in r.rows)
        cn = ",".join(JOINT_SHORT[i] for i in r.cols)
        print(f"{rn:>12} {cn:>22} {r.shape[0]}x{r.shape[1]:<5} {r.rank:>5} "
              f"{r.cond_or_norm:>13.4e} {min(r.singular_values):>12.4e} {str(r.usable):>7}")
    print("  (a single-row block reports its NORM, not cond -- cond of any nonzero 1xN "
          "matrix is exactly 1.0; that is what the controller reports as jacobian_cond too)")


def _print_rollouts(rows: list[RolloutResult]) -> None:
    print(f"\n=== closed-loop move+hold ({rows[0].pose}, {rows[0].config}) ===")
    print(f"{'dx[m]':>8} {'variant':>20} {'track%':>8} {'|Ydrift|':>10} {'|Zdrift|':>10} "
          f"{'ori[rad]':>9} {'max|qd|':>9} {'max|tau|':>9} {'heldJtau':>9} {'guard':>26}")
    for r in rows:
        guard = f"{r.guard_reason}@{r.guard_time_s:.2f}s" if r.guard_tripped else "-"
        print(f"{r.target_delta_m:>8.3f} {r.label:>20} {100 * r.tracking_fraction:>8.2f} "
              f"{r.max_abs_y_drift_m:>10.5f} {r.max_abs_z_drift_m:>10.5f} "
              f"{r.max_orientation_error_rad:>9.4f} {r.max_abs_qd_radps:>9.4f} "
              f"{r.max_abs_tau_nm:>9.3f} {r.max_held_joint_task_torque_nm:>9.2e} {guard:>26}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["authority", "rollout", "both"], default="authority")
    ap.add_argument("--pose", choices=sorted(POSES), default="user_real_pose")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--variants", nargs="+", default=sorted(ROLLOUT_VARIANTS),
                    choices=sorted(ROLLOUT_VARIANTS))
    # Both directions by default -- AGENTS.md sec 7.
    ap.add_argument("--deltas", type=float, nargs="+", default=[0.02, -0.02])
    ap.add_argument("--move-duration", type=float, default=1.5)
    ap.add_argument("--hold-duration", type=float, default=1.0)
    ap.add_argument("--start-q-rad", type=float, nargs=6, default=None,
                    metavar=("Q1", "Q2", "Q3", "Q4", "Q5", "Q6"),
                    help="Override --pose with an arbitrary 6-joint start pose in radians.")
    ap.add_argument("--json", default=None, help="write results to this JSON path")
    args = ap.parse_args(argv)

    if args.start_q_rad is not None:
        q_start = np.asarray(args.start_q_rad, dtype=np.float64)
        pose_label = "custom_start_q"
    else:
        q_start = POSES[args.pose]
        pose_label = args.pose
    payload: dict[str, Any] = {}

    if args.mode in ("authority", "both"):
        blocks = run_authority_analysis(q_start, pose_label=pose_label)
        _print_blocks(blocks)
        payload["blocks"] = [asdict(b) for b in blocks]

    if args.mode in ("rollout", "both"):
        rows: list[RolloutResult] = []
        for delta in args.deltas:
            for variant in args.variants:
                rows.append(run_variant(
                    variant, q_start, config_path=args.config,
                    target_delta_m=float(delta), move_duration_s=args.move_duration,
                    hold_duration_s=args.hold_duration, pose_label=pose_label,
                ))
        _print_rollouts(rows)
        payload["rollouts"] = [asdict(r) for r in rows]

    if args.json:
        Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
