#!/usr/bin/env python3
"""Closed-loop MuJoCo validation of ``transport_axis_index`` 0/1/2.

The axis-generic transport work (``controller_core/x_axis_cartesian_impedance``
+ the ``hardware/`` transport loops) was validated with pure-numpy unit tests
against a synthetic point-mass plant plus mocked-RTDE plumbing tests. Neither
can see anything that depends on the real UR5e kinematics: the Jacobian's
per-axis conditioning, the wrist singularity, gravity acting along world Z but
not X/Y, or joint friction. This script closes that gap by driving the SAME
adapter pipeline every sim tool in this repo already uses
(``simulation/ur5e_mujoco_torque.py``'s ``build_initial_state_and_adapter`` /
``build_mujoco_state`` / ``adapter.step()`` / ``mujoco.mj_step``, loaded from
``assets/ur5e_torque/scene.xml``) with the transport axis set to Y or Z, and
measuring whether the commanded axis converges, whether the two held axes stay
held, and whether any safety guard trips.

Deliberately hand-rolls the per-step loop rather than shelling out to
``tools/ur5e_mujoco_torque_experiments.py`` -- that keeps this an INDEPENDENT
cross-check of the same adapter/controller contract rather than a re-run of the
experiment runner's own target-generation code (which is where the axis bug
this script first surfaced actually lived). The trajectory shape is the repo's
standard ``min_jerk_move_hold`` profile via
``simulation.ur5e_mujoco_torque.x_profile_target``, seeded from the START value
of the SELECTED axis -- the same thing ``hardware/position_transport.py`` does
with ``x0 = start_pose[transport_axis_index]``.

Examples
--------
    python tools/diagnostics/axis_generic_transport_sim_check.py \
        --pose mega_search_winner --axis 1 --deltas 0.05 -0.05

    python tools/diagnostics/axis_generic_transport_sim_check.py \
        --pose hanging_alpha_0_5 --axis 0 1 2 --deltas 0.05 -0.05 --json out.json
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
    HANGING_ALPHA_0_5_Q,
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

#: pose name -> (joint vector, the config this repo already validated AT that pose).
#: Every entry is an already-defined, already-referenced pose from hardware/poses.py
#: paired with the config its own X-axis characterization used, so an axis=1/2 result
#: is directly comparable to that pose's documented axis=0 numbers.
POSES: dict[str, tuple[np.ndarray, str]] = {
    "mega_search_winner": (
        MEGA_SEARCH_WINNER_Q,
        "config/ur5e_mujoco_torque_osc_mega_search_winner.yaml",
    ),
    "hanging_alpha_0_5": (
        HANGING_ALPHA_0_5_Q,
        "config/ur5e_mujoco_torque_osc_hanging_pose_friction_ff.yaml",
    ),
    "height_alpha_0_5": (
        HEIGHT_ALPHA_0_5_Q,
        "config/ur5e_mujoco_torque_osc_tuned.yaml",
    ),
    "height_alpha_0_5_wrist2_offset": (
        HEIGHT_ALPHA_0_5_WRIST2_OFFSET_Q,
        "config/ur5e_mujoco_torque_osc_tuned.yaml",
    ),
}

AXIS_NAMES = ("X", "Y", "Z")


@dataclass
class AxisTrialResult:
    """Everything a caller needs to judge one closed-loop move+hold trial."""

    pose: str
    config: str
    axis: int
    axis_name: str
    target_delta_m: float
    move_duration_s: float
    hold_duration_s: float
    steps: int

    start_ee_pos: list[float] = field(default_factory=list)
    final_ee_pos: list[float] = field(default_factory=list)

    #: Displacement actually achieved along the TRANSPORT axis (signed, metres).
    achieved_delta_m: float = 0.0
    #: achieved / commanded, as a fraction (1.0 == perfect).
    tracking_fraction: float = 0.0
    #: |target - actual| on the transport axis at the final step.
    final_axis_error_m: float = 0.0
    #: Largest |target - actual| on the transport axis at any step.
    max_abs_axis_error_m: float = 0.0

    #: Max |pos - start| on each of the two NON-transport axes (the "held" axes).
    max_abs_held_drift_m: dict[str, float] = field(default_factory=dict)
    #: The single worst held-axis drift -- the quantity ImpedanceSafetyMonitor's
    #: max_abs_orthogonal_drift_m guard actually bounds.
    max_abs_orthogonal_drift_m: float = 0.0

    max_orientation_error_rad: float = 0.0
    max_abs_qd_radps: float = 0.0
    min_cond_j: float = 0.0
    max_cond_j: float = 0.0
    max_torque_clip_fraction: float = 0.0

    guard_tripped: bool = False
    guard_reason: str | None = None
    guard_time_s: float | None = None

    def summary_line(self) -> str:
        status = f"TRIP({self.guard_reason})" if self.guard_tripped else "ok"
        return (
            f"{self.pose:32s} axis={self.axis}({self.axis_name}) "
            f"dx={self.target_delta_m:+.3f} -> achieved={self.achieved_delta_m:+.4f} "
            f"({100.0 * self.tracking_fraction:5.1f}%) held_drift={self.max_abs_orthogonal_drift_m:.4f} "
            f"orient={self.max_orientation_error_rad:.4f} cond=[{self.min_cond_j:.3g},{self.max_cond_j:.3g}] "
            f"|qd|max={self.max_abs_qd_radps:.3f} {status}"
        )


def load_controller_config(config_path: Path) -> dict[str, Any]:
    with Path(config_path).open() as fp:
        return yaml.safe_load(fp)


def run_axis_transport_trial(
    q_start: np.ndarray,
    *,
    config_path: Path | str,
    axis: int,
    target_delta_m: float,
    move_duration_s: float = 1.0,
    hold_duration_s: float = 1.0,
    scene_xml: Path | str = SCENE_XML,
    pose_label: str = "custom",
    stop_on_guard: bool = True,
) -> AxisTrialResult:
    """Run one real closed-loop move+hold along ``axis`` and measure the outcome.

    ``axis`` is the world Cartesian axis to transport along (0=X, 1=Y, 2=Z).
    The two other axes are expected to be HELD at their start values by the
    controller's hold terms; how well they actually are is the main thing this
    function measures (that, and whether the transport axis converges at all).
    """
    axis = int(axis)
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1, or 2; got {axis}")
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    cfg = load_controller_config(config_path)
    mj_cfg = cfg.get("mujoco", {}) or {}

    model, data, site_id, joint_ids, _ = load_model(scene_xml)
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    q_start = np.asarray(q_start, dtype=np.float64).reshape(6)
    for idx, jid in enumerate(joint_ids):
        data.qpos[int(model.jnt_qposadr[jid])] = float(q_start[idx])
    mujoco.mj_forward(model, data)

    state0, adapter = build_initial_state_and_adapter(
        model,
        data,
        site_id,
        joint_ids,
        controller_cfg=cfg["controller"],
        transport_axis_index=axis,
        target_x_delta=float(target_delta_m),
        controller_kind="impedance",
        force_hold_current_pose=False,
        gravity_mode=str(mj_cfg.get("gravity_mode", "gravity_comp")),
        gravity_source=str(mj_cfg.get("gravity_source", "mujoco_qfrc")),
        coriolis_feedforward=bool(mj_cfg.get("coriolis_feedforward", False)),
        torque_limit_scale=1.0,
    )

    start_ee = np.asarray(state0.ee_pos, dtype=np.float64).copy()
    axis0 = float(start_ee[axis])
    held_axes = [i for i in (0, 1, 2) if i != axis]

    dt = float(model.opt.timestep)
    duration_s = float(move_duration_s) + float(hold_duration_s)
    steps = max(1, int(np.ceil(duration_s / max(dt, 1e-9))))

    result = AxisTrialResult(
        pose=pose_label,
        config=str(config_path.relative_to(REPO_ROOT)) if config_path.is_relative_to(REPO_ROOT) else str(config_path),
        axis=axis,
        axis_name=AXIS_NAMES[axis],
        target_delta_m=float(target_delta_m),
        move_duration_s=float(move_duration_s),
        hold_duration_s=float(hold_duration_s),
        steps=0,
        start_ee_pos=start_ee.tolist(),
    )
    result.min_cond_j = float("inf")

    gravity_scratch = mujoco.MjData(model)
    ee_now = start_ee.copy()

    for _ in range(steps):
        t_s = float(data.time)
        # Same profile + same seeding convention as hardware/position_transport.py:
        # the reference is generated from the START value of the SELECTED axis.
        target_axis_now, target_axis_vel_now = x_profile_target(
            "min_jerk_move_hold",
            axis0,
            float(target_delta_m),
            t_s,
            duration_s,
            move_duration_s=float(move_duration_s),
        )
        target_ee_pos = start_ee.copy()
        target_ee_pos[axis] = target_axis_now
        target_ee_vel = np.zeros(3, dtype=np.float64)
        target_ee_vel[axis] = target_axis_vel_now

        state = build_mujoco_state(
            model,
            data,
            site_id=site_id,
            joint_ids=joint_ids,
            time_s=t_s,
            dt_s=dt,
            # target_x carries the SELECTED axis' target, matching what the
            # real-hardware state builder (hardware/direct_torque_link.py::
            # compose_robot_state) puts there; target_axis/_vel is the explicit
            # axis-aware key the controller prefers when axis != 0.
            target_x=float(target_axis_now),
            target_x_vel=float(target_axis_vel_now),
            target_x_accel=0.0,
            target_axis=float(target_axis_now),
            target_axis_vel=float(target_axis_vel_now),
            target_ee_pos=target_ee_pos,
            target_ee_vel=target_ee_vel,
            reference_quat=state0.reference_quat,
            hold_current_pose=False,
            transport_axis_index=axis,
            gravity_compensation=bool(str(mj_cfg.get("gravity_mode", "gravity_comp")) == "gravity_comp"),
            gravity_scratch_data=gravity_scratch,
        )

        tau, diag = adapter.step(state=state)

        cond_j = float(np.linalg.cond(np.asarray(state.jacobian, dtype=np.float64)))
        if np.isfinite(cond_j):
            result.min_cond_j = min(result.min_cond_j, cond_j)
            result.max_cond_j = max(result.max_cond_j, cond_j)
        result.max_torque_clip_fraction = max(
            result.max_torque_clip_fraction, float(diag.get("torque_clip_fraction", 0.0))
        )
        result.max_orientation_error_rad = max(
            result.max_orientation_error_rad, float(diag.get("orientation_error_norm", 0.0))
        )
        result.max_abs_axis_error_m = max(result.max_abs_axis_error_m, abs(float(diag.get("axis_error", 0.0))))
        result.max_abs_qd_radps = max(result.max_abs_qd_radps, float(np.max(np.abs(state.qd))))

        ee_now = np.asarray(state.ee_pos, dtype=np.float64).copy()
        for held in held_axes:
            name = AXIS_NAMES[held]
            drift = abs(float(ee_now[held] - start_ee[held]))
            result.max_abs_held_drift_m[name] = max(result.max_abs_held_drift_m.get(name, 0.0), drift)
        result.max_abs_orthogonal_drift_m = max(result.max_abs_held_drift_m.values(), default=0.0)

        if not bool(diag.get("safety_ok", True)):
            result.guard_tripped = True
            result.guard_reason = str(diag.get("safety_reason", "") or "unknown")
            result.guard_time_s = t_s
            if stop_on_guard:
                break

        data.ctrl[:6] = np.asarray(tau, dtype=np.float64).reshape(6)
        mujoco.mj_step(model, data)
        result.steps += 1

    mujoco.mj_forward(model, data)
    final_ee = np.asarray(data.site_xpos[site_id], dtype=np.float64).copy()
    result.final_ee_pos = final_ee.tolist()
    result.achieved_delta_m = float(final_ee[axis] - start_ee[axis])
    result.final_axis_error_m = abs(float((axis0 + float(target_delta_m)) - final_ee[axis]))
    denom = abs(float(target_delta_m))
    result.tracking_fraction = float(result.achieved_delta_m / target_delta_m) if denom > 1e-12 else 0.0
    for held in held_axes:
        name = AXIS_NAMES[held]
        result.max_abs_held_drift_m[name] = max(
            result.max_abs_held_drift_m.get(name, 0.0), abs(float(final_ee[held] - start_ee[held]))
        )
    result.max_abs_orthogonal_drift_m = max(result.max_abs_held_drift_m.values(), default=0.0)
    if not np.isfinite(result.min_cond_j):
        result.min_cond_j = 0.0
    return result


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--pose",
        nargs="+",
        default=["mega_search_winner"],
        choices=sorted(POSES) + ["all"],
        help="named pose(s) from hardware/poses.py to test",
    )
    p.add_argument(
        "--axis",
        nargs="+",
        type=int,
        default=[0, 1],
        choices=[0, 1, 2],
        help="world Cartesian transport axis/axes (0=X, 1=Y, 2=Z)",
    )
    p.add_argument(
        "--deltas",
        nargs="+",
        type=float,
        default=[0.05, -0.05],
        help="commanded displacements in metres; always sweep BOTH signs (AGENTS.md sec 7)",
    )
    p.add_argument("--move-duration", type=float, default=1.0)
    p.add_argument("--hold-duration", type=float, default=1.0)
    p.add_argument("--config", type=str, default=None, help="override the per-pose default config")
    p.add_argument("--scene", type=str, default=str(SCENE_XML))
    p.add_argument("--json", type=str, default=None, help="write all results to this JSON path")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pose_names = sorted(POSES) if "all" in args.pose else list(args.pose)

    results: list[AxisTrialResult] = []
    for pose_name in pose_names:
        q_start, default_cfg = POSES[pose_name]
        config_path = args.config or default_cfg
        for axis in args.axis:
            for delta in args.deltas:
                res = run_axis_transport_trial(
                    q_start,
                    config_path=config_path,
                    axis=int(axis),
                    target_delta_m=float(delta),
                    move_duration_s=float(args.move_duration),
                    hold_duration_s=float(args.hold_duration),
                    scene_xml=args.scene,
                    pose_label=pose_name,
                )
                results.append(res)
                print(res.summary_line(), flush=True)

    if args.json:
        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as fp:
            json.dump([asdict(r) for r in results], fp, indent=2)
        print(f"\nwrote {len(results)} results to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
