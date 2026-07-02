#!/usr/bin/env python3
"""MuJoCo UR5e gravity-torque and actuator-mapping audit.

Simulation-only. This script stays inside MuJoCo and the reusable controller
core. It verifies the torque-motor mapping, the gravity-compensation sign, the
relation between static gravity torque and live MuJoCo bias force, and basic
hold quality on the existing transport start poses.

No RL, hardware, RTDE, URScript, or CoppeliaSim code is imported.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import mujoco
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from controller_core.kinematics_utils import orientation_error_vec_wxyz, rotmat_to_quat  # noqa: E402
from controller_core.logging_utils import JsonlTraceWriter  # noqa: E402
from mujoco_ur5e_tools import (  # noqa: E402
    UR5E_JOINT_ORDER,
    compute_gravity_torque,
    get_compiled_ur5e_torque_model_diagnostics,
    validate_compiled_ur5e_torque_model,
    validate_ur5e_torque_xml_source_tree,
)
from simulation.ur5e_mujoco_torque import (  # noqa: E402
    MujocoUR5eState,
    MujocoUR5eTorqueAdapter,
    MujocoUR5eTorqueAdapterConfig,
    build_controller,
    build_mujoco_state,
    compute_joint_limit_proximity,
    load_model,
)
from transport_metrics import controller_gain_summary, summarize_residual_torque_trace  # noqa: E402


POSE_LIBRARY: dict[str, np.ndarray] = {
    "active_origin": np.array([0.0, -1.5707963267948966, 0.0, -1.5707963267948966, 0.0, 0.0], dtype=np.float64),
    # Copied from the earlier transport smoke / test fixture so the audit can
    # probe a second numerically different posture without inventing a new one.
    "low_z": np.array(
        [
            0.0,
            -0.1133064268431449,
            -0.664621645801302,
            4.921777393344012,
            -6.283185307179586,
            5.280928640069786,
        ],
        dtype=np.float64,
    ),
}

GRAVITY_SIGN_VARIANTS = (
    "raw_zero",
    "plus_gravity_comp",
    "minus_gravity_comp",
    "direct_bias_force",
    "inverse_dynamics_hold",
)

HOLD_VARIANTS = ("gravity_comp_hold", "residual_impedance_hold")

_POSES_DEFAULT = ("active_origin", "low_z")
_DURATIONS_DEFAULT = (1.0, 3.0, 5.0)
_SAFE_QD_LIMIT_RADPS = 3.0


@dataclass(frozen=True)
class PoseSpec:
    name: str
    q: np.ndarray


def _load_yaml(path: Path) -> dict[str, Any]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def _resolve_path(path: str | Path, *, base: Path = REPO_ROOT) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (base / p).resolve()


def _default_output_root() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "outputs" / "ur5e_mujoco_torque_transport" / f"gravity_torque_audit_{stamp}"


def _fmt_token(value: float | str) -> str:
    if isinstance(value, str):
        return value.replace(" ", "_")
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "config" / "ur5e_mujoco_torque_transport.yaml",
        help="Base MuJoCo transport config YAML.",
    )
    p.add_argument("--scene", type=Path, default=None, help="Optional scene XML override.")
    p.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Audit output directory. Defaults under outputs/ur5e_mujoco_torque_transport/.",
    )
    p.add_argument(
        "--poses",
        nargs="+",
        default=list(_POSES_DEFAULT),
        choices=tuple(POSE_LIBRARY.keys()),
        help="Named start poses to audit.",
    )
    p.add_argument(
        "--durations",
        nargs="+",
        type=float,
        default=list(_DURATIONS_DEFAULT),
        help="Durations in seconds to audit for each pose.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-plot", action="store_true", help="Skip plot generation.")
    return p.parse_args()


def compute_gravity_torque_static(
    model: mujoco.MjModel,
    q: Sequence[float],
    joint_ids: Sequence[int],
    *,
    scratch_data: mujoco.MjData | None = None,
) -> np.ndarray:
    """Return the current compensation torque for zero-velocity, zero-accel hold.

    This is the same sign convention used by the residual-torque transport lane:
    it returns the torque that should be added to the actuator command to hold
    the present configuration.
    """

    tau = compute_gravity_torque(model, np.asarray(q, dtype=np.float64).reshape(6), joint_ids, scratch_data=scratch_data)
    return np.asarray(tau, dtype=np.float64).reshape(6)


def compute_bias_torque_static(
    model: mujoco.MjModel,
    q: Sequence[float],
    joint_ids: Sequence[int],
    *,
    scratch_data: mujoco.MjData | None = None,
) -> np.ndarray:
    """Return MuJoCo's static bias-force torque for q, qd=0, qacc=0."""

    return compute_gravity_torque_static(model, q, joint_ids, scratch_data=scratch_data)


def compute_bias_torque_live(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_ids: Sequence[int],
) -> np.ndarray:
    """Return the live MuJoCo bias force at the current state."""

    del model
    out = np.zeros(len(joint_ids), dtype=np.float64)
    for idx, jid in enumerate(joint_ids):
        dof_adr = int(data.model.jnt_dofadr[int(jid)])
        out[idx] = float(data.qfrc_bias[dof_adr])
    return out


def _joint_dof_ids(model: mujoco.MjModel, joint_ids: Sequence[int]) -> np.ndarray:
    return np.asarray([int(model.jnt_dofadr[int(jid)]) for jid in joint_ids], dtype=np.int32)


def _joint_vector_from_data(data: mujoco.MjData, dof_ids: Sequence[int], source: str) -> np.ndarray:
    arr = np.asarray(getattr(data, source), dtype=np.float64).reshape(-1)
    return arr[np.asarray(dof_ids, dtype=np.int32)].copy()


def _pose_spec(name: str) -> PoseSpec:
    if name not in POSE_LIBRARY:
        raise KeyError(f"Unknown audit pose: {name!r}")
    return PoseSpec(name=name, q=np.asarray(POSE_LIBRARY[name], dtype=np.float64).reshape(6).copy())


def _load_pose_start(model: mujoco.MjModel, pose: PoseSpec) -> np.ndarray:
    q_start = np.asarray(pose.q, dtype=np.float64).reshape(6)
    if np.any(~np.isfinite(q_start)):
        raise ValueError(f"Pose {pose.name!r} contains non-finite values")
    qmin = np.asarray(model.jnt_range[:6, 0], dtype=np.float64)
    qmax = np.asarray(model.jnt_range[:6, 1], dtype=np.float64)
    if np.any(q_start < qmin - 1e-6) or np.any(q_start > qmax + 1e-6):
        raise ValueError(f"Pose {pose.name!r} is outside the joint limits")
    return q_start


def _reset_model_to_pose(model: mujoco.MjModel, data: mujoco.MjData, q_start: Sequence[float]) -> None:
    q_start = np.asarray(q_start, dtype=np.float64).reshape(6)
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    if hasattr(data, "qacc"):
        data.qacc[:] = 0.0
    if hasattr(data, "ctrl"):
        data.ctrl[:] = 0.0
    if hasattr(data, "qfrc_applied"):
        data.qfrc_applied[:] = 0.0
    if hasattr(data, "xfrc_applied"):
        data.xfrc_applied[:] = 0.0
    data.qpos[: q_start.shape[0]] = q_start
    mujoco.mj_forward(model, data)


def _make_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    site_id: int,
    joint_ids: Sequence[int],
    time_s: float,
    dt_s: float,
    target_x: float,
    gravity_compensation: bool,
    reference_quat: np.ndarray | None = None,
    target_ee_pos: np.ndarray | None = None,
) -> MujocoUR5eState:
    return build_mujoco_state(
        model,
        data,
        site_id=site_id,
        joint_ids=joint_ids,
        time_s=float(time_s),
        dt_s=float(dt_s),
        target_x=float(target_x),
        target_ee_pos=target_ee_pos,
        reference_quat=reference_quat,
        transport_axis_index=0,
        gravity_compensation=bool(gravity_compensation),
    )


def _hold_validity_metrics(summary: Mapping[str, Any]) -> dict[str, Any]:
    max_abs_qd = float(summary.get("max_abs_qd_radps", 0.0))
    max_abs_y = float(summary.get("max_abs_y_drift_m", 0.0))
    max_abs_z = float(summary.get("max_abs_z_drift_m", 0.0))
    max_abs_orient = float(summary.get("max_abs_orientation_error_rad", 0.0))
    normal = bool(max_abs_y <= 0.03 and max_abs_z <= 0.03 and max_abs_orient <= 0.25 and max_abs_qd <= _SAFE_QD_LIMIT_RADPS)
    strict = bool(max_abs_y <= 0.01 and max_abs_z <= 0.01 and max_abs_orient <= 0.10 and max_abs_qd <= _SAFE_QD_LIMIT_RADPS)
    return {
        "hold_valid_normal": normal,
        "hold_valid_strict": strict,
    }


def _hold_quality_score(summary: Mapping[str, Any]) -> float:
    final_q_drift_norm = float(summary.get("final_q_drift_norm", 0.0))
    final_ee_drift_m = float(summary.get("final_ee_drift_m", 0.0))
    max_abs_y = float(summary.get("max_abs_y_drift_m", 0.0))
    max_abs_z = float(summary.get("max_abs_z_drift_m", 0.0))
    max_abs_orient = float(summary.get("max_abs_orientation_error_rad", 0.0))
    max_abs_qd = float(summary.get("max_abs_qd_radps", 0.0))
    return 1.0 / (
        1.0
        + final_q_drift_norm
        + final_ee_drift_m
        + max_abs_y / 0.03
        + max_abs_z / 0.03
        + max_abs_orient / 0.25
        + max_abs_qd / _SAFE_QD_LIMIT_RADPS
    )


def _summarize_trace(
    trace_rows: Sequence[Mapping[str, Any]],
    *,
    start_q: Sequence[float],
    start_ee: Sequence[float],
    start_quat: Sequence[float],
    variant_name: str,
    pose_name: str,
    duration_s: float,
) -> dict[str, Any]:
    start_q = np.asarray(start_q, dtype=np.float64).reshape(6)
    start_ee = np.asarray(start_ee, dtype=np.float64).reshape(3)
    start_quat = np.asarray(start_quat, dtype=np.float64).reshape(4)

    if not trace_rows:
        return {
            "variant_name": variant_name,
            "pose_name": pose_name,
            "duration_s": float(duration_s),
            "success": False,
            "termination_reason": "no_trace_rows",
        }

    q_all = np.asarray([row["q"] for row in trace_rows], dtype=np.float64)
    qd_all = np.asarray([row["qd"] for row in trace_rows], dtype=np.float64)
    ee_all = np.asarray([row["ee_pos"] for row in trace_rows], dtype=np.float64)
    quat_all = np.asarray([row["ee_quat"] for row in trace_rows], dtype=np.float64)
    tau_cmd_all = np.asarray([row["tau_requested"] for row in trace_rows], dtype=np.float64)
    tau_applied_all = np.asarray([row["tau_applied"] for row in trace_rows], dtype=np.float64)
    tau_gravity_all = np.asarray([row.get("tau_gravity_static", row["tau_applied"]) for row in trace_rows], dtype=np.float64)
    tau_gravity_existing_all = np.asarray([row.get("tau_gravity_existing", row.get("tau_gravity_static", row["tau_applied"])) for row in trace_rows], dtype=np.float64)
    tau_bias_live_all = np.asarray([row.get("tau_bias_live", row["tau_applied"]) for row in trace_rows], dtype=np.float64)
    tau_bias_static_all = np.asarray([row.get("tau_bias_static", row["tau_applied"]) for row in trace_rows], dtype=np.float64)
    qfrc_actuator_all = np.asarray([row["qfrc_actuator"] for row in trace_rows], dtype=np.float64)
    qfrc_bias_all = np.asarray([row["qfrc_bias"] for row in trace_rows], dtype=np.float64)

    final_q = q_all[-1]
    final_ee = ee_all[-1]
    final_quat = quat_all[-1]
    final_q_drift = final_q - start_q
    ee_drift_all = ee_all - start_ee.reshape(1, 3)
    quat_err_all = np.asarray([orientation_error_vec_wxyz(start_quat, quat) for quat in quat_all], dtype=np.float64)
    final_quat_err = float(np.linalg.norm(orientation_error_vec_wxyz(start_quat, final_quat)))

    qfrc_actuator_err = qfrc_actuator_all - tau_applied_all
    requested_vs_applied_err = tau_applied_all - tau_cmd_all
    gravity_static_vs_existing_err = tau_gravity_all - tau_gravity_existing_all
    live_bias_vs_existing_err = tau_bias_live_all - tau_gravity_existing_all

    summary = {
        "variant_name": variant_name,
        "pose_name": pose_name,
        "duration_s": float(duration_s),
        "steps": int(len(trace_rows)),
        "sim_time_s": float(trace_rows[-1]["time_s"]),
        "success": bool(trace_rows[-1].get("termination_reason", "") == "duration_complete"),
        "termination_reason": str(trace_rows[-1].get("termination_reason", "")),
        "start_q_rad": start_q.tolist(),
        "start_ee_pos": start_ee.tolist(),
        "start_quat_wxyz": start_quat.tolist(),
        "final_q_rad": final_q.tolist(),
        "final_ee_pos": final_ee.tolist(),
        "final_quat_wxyz": final_quat.tolist(),
        "final_q_drift_rad": final_q_drift.tolist(),
        "final_q_drift_norm": float(np.linalg.norm(final_q_drift)),
        "final_ee_drift_m": float(np.linalg.norm(final_ee - start_ee)),
        "final_x_drift_m": float(final_ee[0] - start_ee[0]),
        "final_y_drift_m": float(final_ee[1] - start_ee[1]),
        "final_z_drift_m": float(final_ee[2] - start_ee[2]),
        "final_orientation_error_rad": final_quat_err,
        "max_abs_x_drift_m": float(np.max(np.abs(ee_drift_all[:, 0]))),
        "max_abs_y_drift_m": float(np.max(np.abs(ee_drift_all[:, 1]))),
        "max_abs_z_drift_m": float(np.max(np.abs(ee_drift_all[:, 2]))),
        "max_abs_ee_drift_m": float(np.max(np.linalg.norm(ee_drift_all, axis=1))),
        "max_abs_qd_radps": float(np.max(np.abs(qd_all))),
        "max_abs_q_drift_rad": float(np.max(np.abs(q_all - start_q.reshape(1, 6)))),
        "max_abs_orientation_error_rad": float(np.max(np.abs(quat_err_all))),
        "mean_abs_tau_requested_nm": float(np.mean(np.abs(tau_cmd_all))),
        "max_abs_tau_requested_nm": float(np.max(np.abs(tau_cmd_all))),
        "mean_abs_tau_applied_nm": float(np.mean(np.abs(tau_applied_all))),
        "max_abs_tau_applied_nm": float(np.max(np.abs(tau_applied_all))),
        "mean_abs_tau_gravity_static_nm": float(np.mean(np.abs(tau_gravity_all))),
        "max_abs_tau_gravity_static_nm": float(np.max(np.abs(tau_gravity_all))),
        "mean_abs_tau_bias_live_nm": float(np.mean(np.abs(tau_bias_live_all))),
        "max_abs_tau_bias_live_nm": float(np.max(np.abs(tau_bias_live_all))),
        "mean_abs_tau_bias_static_nm": float(np.mean(np.abs(tau_bias_static_all))),
        "max_abs_tau_bias_static_nm": float(np.max(np.abs(tau_bias_static_all))),
        "max_abs_qfrc_actuator_nm": float(np.max(np.abs(qfrc_actuator_all))),
        "max_abs_qfrc_bias_nm": float(np.max(np.abs(qfrc_bias_all))),
        "max_abs_mapping_error_nm": float(np.max(np.abs(qfrc_actuator_err))),
        "mean_abs_mapping_error_nm": float(np.mean(np.abs(qfrc_actuator_err))),
        "max_abs_requested_vs_applied_error_nm": float(np.max(np.abs(requested_vs_applied_err))),
        "mean_abs_requested_vs_applied_error_nm": float(np.mean(np.abs(requested_vs_applied_err))),
        "max_abs_gravity_static_minus_existing_nm": float(np.max(np.abs(gravity_static_vs_existing_err))),
        "mean_abs_gravity_static_minus_existing_nm": float(np.mean(np.abs(gravity_static_vs_existing_err))),
        "max_abs_live_bias_minus_existing_nm": float(np.max(np.abs(live_bias_vs_existing_err))),
        "mean_abs_live_bias_minus_existing_nm": float(np.mean(np.abs(live_bias_vs_existing_err))),
        "hold_quality_score": float(_hold_quality_score(
            {
                "final_q_drift_norm": float(np.linalg.norm(final_q_drift)),
                "final_ee_drift_m": float(np.linalg.norm(final_ee - start_ee)),
                "max_abs_y_drift_m": float(np.max(np.abs(ee_drift_all[:, 1]))),
                "max_abs_z_drift_m": float(np.max(np.abs(ee_drift_all[:, 2]))),
                "max_abs_orientation_error_rad": float(np.max(np.abs(quat_err_all))),
                "max_abs_qd_radps": float(np.max(np.abs(qd_all))),
            }
        )),
    }
    summary.update(_hold_validity_metrics(summary))
    summary["better_than_raw_zero"] = False
    summary["worse_than_raw_zero"] = False
    return summary


def audit_actuator_mapping_case(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    joint_ids: Sequence[int],
    tau_cmd: Sequence[float],
    site_id: int | None = None,
) -> dict[str, Any]:
    """Map a commanded torque vector through MuJoCo and report the error."""

    del site_id
    dof_ids = _joint_dof_ids(model, joint_ids)
    tau_cmd = np.asarray(tau_cmd, dtype=np.float64).reshape(6)
    data.ctrl[:6] = tau_cmd
    mujoco.mj_forward(model, data)
    qfrc_actuator = _joint_vector_from_data(data, dof_ids, "qfrc_actuator")
    actuator_force = _joint_vector_from_data(data, dof_ids, "actuator_force")
    qfrc_bias = _joint_vector_from_data(data, dof_ids, "qfrc_bias")
    return {
        "tau_cmd": tau_cmd.tolist(),
        "qfrc_actuator_joint": qfrc_actuator.tolist(),
        "actuator_force_joint": actuator_force.tolist(),
        "qfrc_bias_joint": qfrc_bias.tolist(),
        "max_abs_mapping_error": float(np.max(np.abs(qfrc_actuator - tau_cmd))),
        "mean_abs_mapping_error": float(np.mean(np.abs(qfrc_actuator - tau_cmd))),
        "max_abs_actuator_force_error": float(np.max(np.abs(actuator_force - tau_cmd))),
        "mean_abs_actuator_force_error": float(np.mean(np.abs(actuator_force - tau_cmd))),
        "joint_dof_ids": dof_ids.tolist(),
        "actuator_names": [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid) for aid in range(min(6, int(model.nu)))],
        "gear": np.asarray(model.actuator_gear[:6, 0], dtype=np.float64).tolist(),
        "ctrlrange": np.asarray(model.actuator_ctrlrange[:6], dtype=np.float64).tolist(),
        "forcerange": np.asarray(model.actuator_forcerange[:6], dtype=np.float64).tolist(),
    }


def _run_variant_rollout(
    *,
    model: mujoco.MjModel,
    site_id: int,
    joint_ids: Sequence[int],
    pose: PoseSpec,
    duration_s: float,
    variant_name: str,
    controller_cfg: dict[str, Any],
    gravity_mode: str,
    run_dir: Path,
    no_plot: bool,
) -> dict[str, Any]:
    data = mujoco.MjData(model)
    _reset_model_to_pose(model, data, pose.q)
    dt = float(model.opt.timestep)
    steps = max(1, int(np.ceil(float(duration_s) / max(dt, 1e-9))))
    trace_path = run_dir / "trace.jsonl"
    trace_rows: list[dict[str, Any]] = []
    start_state = _make_state(
        model,
        data,
        site_id=site_id,
        joint_ids=joint_ids,
        time_s=float(data.time),
        dt_s=dt,
        target_x=float(data.site_xpos[site_id][0]),
        gravity_compensation=True,
        reference_quat=rotmat_to_quat(np.asarray(data.site_xmat[site_id], dtype=np.float64).reshape(3, 3)),
        target_ee_pos=np.asarray(data.site_xpos[site_id], dtype=np.float64).copy(),
    )
    start_q = np.asarray(start_state.q, dtype=np.float64).copy()
    start_ee = np.asarray(start_state.ee_pos, dtype=np.float64).copy()
    start_quat = np.asarray(start_state.ee_quat, dtype=np.float64).copy()
    scratch = mujoco.MjData(model)
    controller = build_controller("impedance", controller_cfg)
    adapter = MujocoUR5eTorqueAdapter(
        model=model,
        site_id=site_id,
        joint_ids=joint_ids,
        controller=controller,
        config=MujocoUR5eTorqueAdapterConfig(
            controller_kind="impedance",
            torque_limit_nm=np.asarray([150.0, 150.0, 150.0, 28.0, 28.0, 28.0], dtype=np.float64),
            torque_limit_scale=1.0,
            rate_limit_nm_per_sec=np.asarray([800.0, 800.0, 800.0, 160.0, 160.0, 160.0], dtype=np.float64),
            lowpass_alpha=1.0,
            gravity_mode=str(gravity_mode),
            transport_axis_index=0,
        ),
    )
    adapter.reset(start_state)

    termination_reason = "duration_complete"
    max_abs_tau_controller = 0.0
    max_abs_tau_gravity = 0.0
    max_abs_tau_applied = 0.0
    max_abs_qd = 0.0
    max_abs_mapping_error = 0.0
    max_abs_requested_vs_applied_error = 0.0
    with JsonlTraceWriter(trace_path) as trace_writer:
        for step_idx in range(steps):
            state = _make_state(
                model,
                data,
                site_id=site_id,
                joint_ids=joint_ids,
                time_s=float(data.time),
                dt_s=dt,
                target_x=float(start_state.target_x),
                gravity_compensation=True,
                reference_quat=start_quat,
                target_ee_pos=start_ee,
            )
            q = np.asarray(state.q, dtype=np.float64).copy()
            qd = np.asarray(state.qd, dtype=np.float64).copy()
            ee_pos = np.asarray(state.ee_pos, dtype=np.float64).copy()
            ee_quat = np.asarray(state.ee_quat, dtype=np.float64).copy()
            qfrc_bias_live = compute_bias_torque_live(model, data, joint_ids)
            tau_gravity_static = compute_gravity_torque_static(model, q, joint_ids, scratch_data=scratch)
            tau_bias_static = tau_gravity_static.copy()

            if variant_name == "raw_zero":
                tau_controller = np.zeros(6, dtype=np.float64)
                tau_gravity = np.zeros(6, dtype=np.float64)
                tau_applied = np.zeros(6, dtype=np.float64)
                tau_requested = tau_applied.copy()
                controller_kind = "zero_torque"
                gravity_mode_used = "raw"
            elif variant_name == "plus_gravity_comp":
                tau_controller = np.zeros(6, dtype=np.float64)
                tau_gravity = tau_gravity_static
                tau_applied = tau_gravity.copy()
                tau_requested = tau_applied.copy()
                controller_kind = "gravity_sign_audit"
                gravity_mode_used = "gravity_comp"
            elif variant_name == "minus_gravity_comp":
                tau_controller = np.zeros(6, dtype=np.float64)
                tau_gravity = -tau_gravity_static
                tau_applied = tau_gravity.copy()
                tau_requested = tau_applied.copy()
                controller_kind = "gravity_sign_audit"
                gravity_mode_used = "gravity_comp"
            elif variant_name == "direct_bias_force":
                tau_controller = np.zeros(6, dtype=np.float64)
                tau_gravity = qfrc_bias_live.copy()
                tau_applied = tau_gravity.copy()
                tau_requested = tau_applied.copy()
                controller_kind = "gravity_bias_audit"
                gravity_mode_used = "raw"
            elif variant_name == "inverse_dynamics_hold":
                tau_controller = np.zeros(6, dtype=np.float64)
                tau_gravity = tau_bias_static.copy()
                tau_applied = tau_gravity.copy()
                tau_requested = tau_applied.copy()
                controller_kind = "gravity_bias_audit"
                gravity_mode_used = "raw"
            elif variant_name == "residual_impedance_hold":
                tau_cmd, diag = adapter.step(state=state)
                tau_requested = np.asarray(diag.get("tau_raw", tau_cmd), dtype=np.float64).reshape(6)
                tau_controller = np.asarray(diag.get("tau_controller", tau_cmd), dtype=np.float64).reshape(6)
                tau_gravity = np.asarray(diag.get("tau_gravity", np.zeros(6, dtype=np.float64)), dtype=np.float64).reshape(6)
                tau_applied = np.asarray(diag.get("tau_applied", tau_cmd), dtype=np.float64).reshape(6)
                controller_kind = str(diag.get("controller_kind", "impedance"))
                gravity_mode_used = str(diag.get("gravity_mode_used", gravity_mode))
            else:
                raise ValueError(f"Unsupported variant: {variant_name!r}")

            data.ctrl[:6] = np.asarray(tau_applied, dtype=np.float64).reshape(6)
            mujoco.mj_forward(model, data)
            qfrc_actuator = _joint_vector_from_data(data, _joint_dof_ids(model, joint_ids), "qfrc_actuator")
            qfrc_bias = _joint_vector_from_data(data, _joint_dof_ids(model, joint_ids), "qfrc_bias")
            actuator_force = _joint_vector_from_data(data, _joint_dof_ids(model, joint_ids), "actuator_force")
            mapping_error = qfrc_actuator - tau_requested
            requested_vs_applied_error = tau_applied - tau_requested
            max_abs_mapping_error = max(max_abs_mapping_error, float(np.max(np.abs(mapping_error))))
            max_abs_requested_vs_applied_error = max(max_abs_requested_vs_applied_error, float(np.max(np.abs(requested_vs_applied_error))))
            max_abs_tau_controller = max(max_abs_tau_controller, float(np.max(np.abs(tau_controller))))
            max_abs_tau_gravity = max(max_abs_tau_gravity, float(np.max(np.abs(tau_gravity))))
            max_abs_tau_applied = max(max_abs_tau_applied, float(np.max(np.abs(tau_applied))))
            max_abs_qd = max(max_abs_qd, float(np.max(np.abs(qd))))
            row = {
                "step": int(step_idx),
                "time_s": float(data.time),
                "dt_s": dt,
                "pose_name": pose.name,
                "variant_name": variant_name,
                "controller_kind": controller_kind,
                "gravity_mode": "gravity_comp" if variant_name in {"plus_gravity_comp", "minus_gravity_comp", "residual_impedance_hold"} else "raw",
                "gravity_mode_used": gravity_mode_used,
                "gravity_compensation_active": bool(variant_name in {"plus_gravity_comp", "minus_gravity_comp", "residual_impedance_hold"}),
                "raw_mode_used": bool(variant_name in {"raw_zero", "direct_bias_force", "inverse_dynamics_hold"}),
                "q": q.tolist(),
                "qd": qd.tolist(),
                "ee_pos": ee_pos.tolist(),
                "ee_quat": ee_quat.tolist(),
                "tau_requested": np.asarray(tau_requested, dtype=np.float64).tolist(),
                "tau_controller": np.asarray(tau_controller, dtype=np.float64).tolist(),
                "tau_gravity_static": np.asarray(tau_gravity, dtype=np.float64).tolist(),
                "tau_gravity_existing": np.asarray(tau_gravity_static, dtype=np.float64).tolist(),
                "tau_bias_live": np.asarray(qfrc_bias_live, dtype=np.float64).tolist(),
                "tau_bias_static": np.asarray(tau_bias_static, dtype=np.float64).tolist(),
                "tau_applied": np.asarray(tau_applied, dtype=np.float64).tolist(),
                "qfrc_actuator": qfrc_actuator.tolist(),
                "qfrc_bias": qfrc_bias.tolist(),
                "actuator_force": actuator_force.tolist(),
                "max_abs_mapping_error_nm": float(np.max(np.abs(mapping_error))),
                "max_abs_requested_vs_applied_error_nm": float(np.max(np.abs(requested_vs_applied_error))),
                "x_error": float(start_ee[0] - ee_pos[0]),
                "orientation_error_norm": float(np.linalg.norm(orientation_error_vec_wxyz(start_quat, ee_quat))),
                "joint_limit_min_fraction": float(min(compute_joint_limit_proximity(model, q, joint_ids).values(), default=0.0)),
                "termination_reason": "",
            }
            trace_rows.append(row)
            trace_writer.write_row(row)
            mujoco.mj_step(model, data)
            post_qd = np.asarray(data.qvel[:6], dtype=np.float64)
            if not np.all(np.isfinite(post_qd)) or np.max(np.abs(post_qd)) > 1.0e4:
                termination_reason = "numerical_instability"
                trace_rows[-1]["termination_reason"] = termination_reason
                break

    if trace_rows and termination_reason == "duration_complete":
        trace_rows[-1]["termination_reason"] = "duration_complete"
    elif trace_rows:
        trace_rows[-1]["termination_reason"] = termination_reason

    summary = _summarize_trace(
        trace_rows,
        start_q=start_q,
        start_ee=start_ee,
        start_quat=start_quat,
        variant_name=variant_name,
        pose_name=pose.name,
        duration_s=float(duration_s),
    )
    summary.update(
        {
            "controller_kind": controller_kind,
            "gravity_mode_used": trace_rows[-1].get("gravity_mode_used", "raw") if trace_rows else "raw",
            "gravity_compensation_active": bool(variant_name in {"plus_gravity_comp", "minus_gravity_comp", "residual_impedance_hold"}),
            "raw_mode_used": bool(variant_name in {"raw_zero", "direct_bias_force", "inverse_dynamics_hold"}),
            "max_abs_tau_controller_nm": float(max_abs_tau_controller),
            "max_abs_tau_gravity_nm": float(max_abs_tau_gravity),
            "max_abs_tau_applied_nm": float(max_abs_tau_applied),
            "max_abs_qd_radps": float(max_abs_qd),
            "max_abs_mapping_error_nm": float(max_abs_mapping_error),
            "max_abs_requested_vs_applied_error_nm": float(max_abs_requested_vs_applied_error),
            "mean_abs_tau_controller_nm": float(np.mean(np.abs(np.asarray([r["tau_controller"] for r in trace_rows], dtype=np.float64)))) if trace_rows else 0.0,
            "mean_abs_tau_gravity_nm": float(np.mean(np.abs(np.asarray([r["tau_gravity_static"] for r in trace_rows], dtype=np.float64)))) if trace_rows else 0.0,
            "mean_abs_tau_applied_nm": float(np.mean(np.abs(np.asarray([r["tau_applied"] for r in trace_rows], dtype=np.float64)))) if trace_rows else 0.0,
            "mean_abs_mapping_error_nm": float(np.mean(np.abs(np.asarray([r["qfrc_actuator"] for r in trace_rows], dtype=np.float64) - np.asarray([r["tau_requested"] for r in trace_rows], dtype=np.float64)))) if trace_rows else 0.0,
            "mean_abs_requested_vs_applied_error_nm": float(np.mean(np.abs(np.asarray([r["tau_applied"] for r in trace_rows], dtype=np.float64) - np.asarray([r["tau_requested"] for r in trace_rows], dtype=np.float64)))) if trace_rows else 0.0,
        }
    )
    summary.update(_hold_validity_metrics(summary))
    summary["hold_quality_score"] = float(_hold_quality_score(summary))
    return summary


def _clamping_case_results(
    *,
    model: mujoco.MjModel,
    site_id: int,
    joint_ids: Sequence[int],
    pose: PoseSpec,
    tau_cmd: Sequence[float],
) -> dict[str, Any]:
    data = mujoco.MjData(model)
    _reset_model_to_pose(model, data, pose.q)
    dof_ids = _joint_dof_ids(model, joint_ids)
    tau_cmd = np.asarray(tau_cmd, dtype=np.float64).reshape(6)
    controller_clip_limit = np.asarray([150.0, 150.0, 150.0, 28.0, 28.0, 28.0], dtype=np.float64)
    controller_clipped = np.clip(tau_cmd, -controller_clip_limit, +controller_clip_limit)
    data.ctrl[:6] = tau_cmd
    mujoco.mj_forward(model, data)
    qfrc_actuator = _joint_vector_from_data(data, dof_ids, "qfrc_actuator")
    actuator_force = _joint_vector_from_data(data, dof_ids, "actuator_force")
    return {
        "pose_name": pose.name,
        "tau_requested": tau_cmd.tolist(),
        "controller_level_clipped_tau": controller_clipped.tolist(),
        "controller_level_clip_fraction": float(np.mean(np.abs(controller_clipped - tau_cmd) > 1e-9)),
        "data_ctrl_after_assignment": np.asarray(data.ctrl[:6], dtype=np.float64).tolist(),
        "qfrc_actuator_joint": qfrc_actuator.tolist(),
        "actuator_force_joint": actuator_force.tolist(),
        "max_abs_ctrl_vs_qfrc_error": float(np.max(np.abs(qfrc_actuator - tau_cmd))),
        "mean_abs_ctrl_vs_qfrc_error": float(np.mean(np.abs(qfrc_actuator - tau_cmd))),
        "max_abs_ctrl_vs_actuator_force_error": float(np.max(np.abs(actuator_force - tau_cmd))),
        "mean_abs_ctrl_vs_actuator_force_error": float(np.mean(np.abs(actuator_force - tau_cmd))),
        "mujo_co_ctrlrange": np.asarray(model.actuator_ctrlrange[:6], dtype=np.float64).tolist(),
        "mujo_co_forcerange": np.asarray(model.actuator_forcerange[:6], dtype=np.float64).tolist(),
        "gear": np.asarray(model.actuator_gear[:6, 0], dtype=np.float64).tolist(),
        "actuator_names": [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid) for aid in range(min(6, int(model.nu)))],
        "joint_dof_ids": dof_ids.tolist(),
    }


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    preferred = [
        "section",
        "pose_name",
        "variant_name",
        "case_name",
        "duration_s",
        "success",
        "termination_reason",
        "hold_valid_normal",
        "hold_valid_strict",
        "hold_quality_score",
        "final_q_drift_norm",
        "final_ee_drift_m",
        "final_x_drift_m",
        "final_y_drift_m",
        "final_z_drift_m",
        "final_orientation_error_rad",
        "max_abs_x_drift_m",
        "max_abs_y_drift_m",
        "max_abs_z_drift_m",
        "max_abs_ee_drift_m",
        "max_abs_orientation_error_rad",
        "max_abs_qd_radps",
        "mean_abs_tau_requested_nm",
        "mean_abs_tau_applied_nm",
        "mean_abs_tau_controller_nm",
        "mean_abs_tau_gravity_nm",
        "mean_abs_tau_gravity_static_nm",
        "mean_abs_tau_bias_live_nm",
        "mean_abs_tau_bias_static_nm",
        "max_abs_tau_requested_nm",
        "max_abs_tau_applied_nm",
        "max_abs_tau_controller_nm",
        "max_abs_tau_gravity_nm",
        "max_abs_mapping_error_nm",
        "mean_abs_mapping_error_nm",
        "max_abs_requested_vs_applied_error_nm",
        "mean_abs_requested_vs_applied_error_nm",
        "controller_level_clip_fraction",
        "max_abs_ctrl_vs_qfrc_error",
        "mean_abs_ctrl_vs_qfrc_error",
        "max_abs_ctrl_vs_actuator_force_error",
        "mean_abs_ctrl_vs_actuator_force_error",
        "gravity_compensation_active",
        "raw_mode_used",
        "gravity_mode",
        "gravity_mode_used",
        "controller_kind",
        "trace_path",
        "summary_path",
    ]
    gain_fields = [name for name in ("kp_x", "kd_x", "kp_y", "kd_y", "kp_z", "kd_z", "kp_rot", "kd_rot", "kp_posture", "kd_posture", "kd_joint") if name not in preferred]
    remaining = sorted({key for row in rows for key in row.keys() if key not in preferred and key not in gain_fields})
    fieldnames = preferred + gain_fields + remaining
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _variant_better_than_raw(variant: Mapping[str, Any], raw: Mapping[str, Any]) -> bool:
    return float(variant.get("max_abs_ee_drift_m", np.inf)) + 1e-12 < float(raw.get("max_abs_ee_drift_m", np.inf))


def _variant_worse_than_raw(variant: Mapping[str, Any], raw: Mapping[str, Any]) -> bool:
    return float(variant.get("max_abs_ee_drift_m", np.inf)) > float(raw.get("max_abs_ee_drift_m", np.inf)) + 1e-12


def _aggregate_variant_rows(rows: list[dict[str, Any]], *, key_fields: Sequence[str]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(k) for k in key_fields)].append(row)
    return grouped


def _best_row(rows: Iterable[Mapping[str, Any]], *, key: Callable[[Mapping[str, Any]], tuple[Any, ...]]) -> dict[str, Any] | None:
    rows = list(rows)
    if not rows:
        return None
    return dict(sorted(rows, key=key)[0])


def _variant_rank_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        0 if bool(row.get("hold_valid_normal", False)) else 1,
        0 if bool(row.get("success", False)) else 1,
        float(row.get("max_abs_ee_drift_m", np.inf)),
        float(row.get("max_abs_orientation_error_rad", np.inf)),
        float(row.get("max_abs_qd_radps", np.inf)),
        float(row.get("max_abs_tau_applied_nm", np.inf)),
    )


def _run_audit() -> int:
    args = parse_args()
    np.random.seed(int(args.seed))

    cfg = _load_yaml(args.config)
    mujoco_cfg = cfg["mujoco"]
    ctrl_cfg = cfg["controller"]
    scene_xml = _resolve_path(args.scene if args.scene is not None else mujoco_cfg["scene_xml"])
    output_root = _resolve_path(args.output_root if args.output_root is not None else _default_output_root())
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "per_run_traces").mkdir(parents=True, exist_ok=True)
    (output_root / "plots").mkdir(parents=True, exist_ok=True)

    model, data, site_id, joint_ids, actuator_ids = load_model(scene_xml)
    summary = {
        "output_root": str(output_root),
        "config_path": str(args.config),
        "scene_xml": str(scene_xml),
        "poses_requested": list(args.poses),
        "durations_requested": [float(v) for v in args.durations],
        "controller_gains": controller_gain_summary(ctrl_cfg)["controller_gains"],
        "true_torque_verified": True,
        "source_validation": validate_ur5e_torque_xml_source_tree(scene_xml),
        "compiled_model_diagnostics": get_compiled_ur5e_torque_model_diagnostics(model, site_name=str(mujoco_cfg.get("site_name", "attachment_site"))),
        "joint_ids": list(joint_ids),
        "actuator_ids": list(actuator_ids),
        "joint_names": [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) for jid in joint_ids],
        "actuator_names": [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid) for aid in actuator_ids],
        "rows": [],
    }
    summary["source_validation_ok"] = bool(summary["source_validation"].get("source_tree_ok", False))
    validate_compiled_ur5e_torque_model(model, site_name=str(mujoco_cfg.get("site_name", "attachment_site")))

    rows: list[dict[str, Any]] = []
    sign_audit: dict[str, Any] = {"poses": {}, "durations": [float(v) for v in args.durations], "variants": list(GRAVITY_SIGN_VARIANTS)}
    hold_audit: dict[str, Any] = {"poses": {}, "durations": [float(v) for v in args.durations], "variants": list(HOLD_VARIANTS)}
    mapping_audit: dict[str, Any] = {"poses": {}, "cases": []}
    clamping_audit: dict[str, Any] = {"poses": {}, "cases": []}

    pose_specs = [_pose_spec(name) for name in args.poses]
    base_output_dir = output_root / "per_run_traces"

    # Actuator mapping audit on the first pose only, since it is a static MuJoCo
    # forward-pass check rather than a time rollout.
    mapping_pose = pose_specs[0]
    _reset_model_to_pose(model, data, mapping_pose.q)
    mapping_cases = [
        ("zero", np.zeros(6, dtype=np.float64)),
        ("joint0_pos", np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)),
        ("joint0_neg", np.array([-0.5, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)),
        ("random_small", np.array([0.31, -0.27, 0.14, -0.11, 0.08, -0.04], dtype=np.float64)),
        ("gravity_comp", compute_gravity_torque_static(model, mapping_pose.q, joint_ids, scratch_data=mujoco.MjData(model))),
    ]
    for case_name, tau_cmd in mapping_cases:
        case = audit_actuator_mapping_case(model, data, joint_ids=joint_ids, tau_cmd=tau_cmd, site_id=site_id)
        case.update({"case_name": case_name, "pose_name": mapping_pose.name, "section": "actuator_mapping"})
        mapping_audit["cases"].append(case)
        rows.append(case)
    mapping_audit["max_abs_mapping_error"] = float(max(case["max_abs_mapping_error"] for case in mapping_audit["cases"]))
    mapping_audit["max_abs_actuator_force_error"] = float(max(case["max_abs_actuator_force_error"] for case in mapping_audit["cases"]))
    mapping_audit["actuator_names"] = summary["actuator_names"]
    mapping_audit["joint_dof_ids"] = _joint_dof_ids(model, joint_ids).tolist()
    mapping_audit["gear"] = np.asarray(model.actuator_gear[:6, 0], dtype=np.float64).tolist()
    mapping_audit["ctrlrange"] = np.asarray(model.actuator_ctrlrange[:6], dtype=np.float64).tolist()
    mapping_audit["forcerange"] = np.asarray(model.actuator_forcerange[:6], dtype=np.float64).tolist()
    mapping_audit["one_to_one"] = bool(mapping_audit["max_abs_mapping_error"] < 1e-9)

    # Clamping audit.
    clamp_cases = [
        ("within_limit", np.array([0.1, -0.2, 0.15, 0.02, -0.03, 0.04], dtype=np.float64)),
        ("shoulder_slight_over", np.array([151.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)),
        ("wrist_slight_over", np.array([0.0, 0.0, 0.0, 0.0, 0.0, 28.5], dtype=np.float64)),
        ("very_large", np.full(6, 1.0e3, dtype=np.float64)),
    ]
    for case_name, tau_cmd in clamp_cases:
        case = _clamping_case_results(model=model, site_id=site_id, joint_ids=joint_ids, pose=mapping_pose, tau_cmd=tau_cmd)
        case.update({"case_name": case_name, "section": "clamping"})
        clamping_audit["cases"].append(case)
        rows.append(case)
    clamping_audit["max_abs_ctrl_vs_qfrc_error"] = float(max(case["max_abs_ctrl_vs_qfrc_error"] for case in clamping_audit["cases"]))
    clamping_audit["max_abs_ctrl_vs_actuator_force_error"] = float(max(case["max_abs_ctrl_vs_actuator_force_error"] for case in clamping_audit["cases"]))
    clamping_audit["controller_level_clip_appears"] = bool(any(float(case["controller_level_clip_fraction"]) > 0.0 for case in clamping_audit["cases"]))
    clamping_audit["mujo_co_clip_appears"] = bool(any(float(case["max_abs_ctrl_vs_qfrc_error"]) > 1e-9 for case in clamping_audit["cases"]))
    clamping_audit["controller_clip_note"] = "No separate controller-stage clip is active in this direct-torque lane; the explicit clamp is the adapter limit when used."

    # Time rollouts.
    raw_zero_baselines: dict[tuple[str, float], dict[str, Any]] = {}
    for pose in pose_specs:
        pose_q = _load_pose_start(model, pose)
        _reset_model_to_pose(model, data, pose_q)
        start_state = _make_state(
            model,
            data,
            site_id=site_id,
            joint_ids=joint_ids,
            time_s=float(data.time),
            dt_s=float(model.opt.timestep),
            target_x=float(data.site_xpos[site_id][0]),
            gravity_compensation=True,
            reference_quat=rotmat_to_quat(np.asarray(data.site_xmat[site_id], dtype=np.float64).reshape(3, 3)),
            target_ee_pos=np.asarray(data.site_xpos[site_id], dtype=np.float64).copy(),
        )
        pose_key = pose.name
        sign_audit["poses"][pose_key] = {}
        hold_audit["poses"][pose_key] = {}
        for duration_s in args.durations:
            duration_s = float(duration_s)
            variant_results: dict[str, dict[str, Any]] = {}
            for variant_name in GRAVITY_SIGN_VARIANTS:
                run_label = f"{pose.name}_{variant_name}_dur{_fmt_token(duration_s)}"
                run_dir = base_output_dir / run_label
                run_dir.mkdir(parents=True, exist_ok=True)
                summary_variant = _run_variant_rollout(
                    model=model,
                    site_id=site_id,
                    joint_ids=joint_ids,
                    pose=pose,
                    duration_s=duration_s,
                    variant_name=variant_name,
                    controller_cfg=ctrl_cfg,
                    gravity_mode=str(mujoco_cfg.get("gravity_mode", "gravity_comp")),
                    run_dir=run_dir,
                    no_plot=bool(args.no_plot),
                )
                summary_variant.update(
                    {
                        "section": "gravity_sign",
                        "run_label": run_label,
                        "trace_path": str(run_dir / "trace.jsonl"),
                        "summary_path": str(run_dir / "summary.json"),
                    }
                )
                if variant_name == "raw_zero":
                    raw_zero_baselines[(pose.name, duration_s)] = summary_variant
                variant_results[variant_name] = summary_variant
                rows.append(summary_variant)
            sign_audit["poses"][pose_key][str(duration_s)] = variant_results

            # Hold audit variants.
            hold_results: dict[str, dict[str, Any]] = {}
            for variant_name in HOLD_VARIANTS:
                run_label = f"{pose.name}_{variant_name}_dur{_fmt_token(duration_s)}"
                run_dir = base_output_dir / run_label
                run_dir.mkdir(parents=True, exist_ok=True)
                if variant_name == "gravity_comp_hold":
                    data_hold = mujoco.MjData(model)
                    _reset_model_to_pose(model, data_hold, pose_q)
                    # Simulate gravity compensation only by applying the
                    # static compensation torque each step.
                    trace_rows: list[dict[str, Any]] = []
                    trace_path = run_dir / "trace.jsonl"
                    dt = float(model.opt.timestep)
                    steps = max(1, int(np.ceil(duration_s / max(dt, 1e-9))))
                    start_quat = np.asarray(start_state.ee_quat, dtype=np.float64).copy()
                    start_ee = np.asarray(start_state.ee_pos, dtype=np.float64).copy()
                    scratch = mujoco.MjData(model)
                    with JsonlTraceWriter(trace_path) as trace_writer:
                        for step_idx in range(steps):
                            state = _make_state(
                                model,
                                data_hold,
                                site_id=site_id,
                                joint_ids=joint_ids,
                                time_s=float(data_hold.time),
                                dt_s=dt,
                                target_x=float(start_ee[0]),
                                gravity_compensation=True,
                                reference_quat=start_quat,
                                target_ee_pos=start_ee,
                            )
                            tau_gravity = compute_gravity_torque_static(model, state.q, joint_ids, scratch_data=scratch)
                            tau_controller = np.zeros(6, dtype=np.float64)
                            tau_applied = tau_gravity.copy()
                            data_hold.ctrl[:6] = tau_applied
                            mujoco.mj_forward(model, data_hold)
                            qfrc_actuator = _joint_vector_from_data(data_hold, _joint_dof_ids(model, joint_ids), "qfrc_actuator")
                            qfrc_bias = _joint_vector_from_data(data_hold, _joint_dof_ids(model, joint_ids), "qfrc_bias")
                            actuator_force = _joint_vector_from_data(data_hold, _joint_dof_ids(model, joint_ids), "actuator_force")
                            row = {
                                "step": int(step_idx),
                                "time_s": float(data_hold.time),
                                "dt_s": dt,
                                "pose_name": pose.name,
                                "variant_name": variant_name,
                                "controller_kind": "zero_torque",
                                "gravity_mode": "gravity_comp",
                                "gravity_mode_used": "gravity_comp",
                                "gravity_compensation_active": True,
                                "raw_mode_used": False,
                                "q": np.asarray(state.q, dtype=np.float64).tolist(),
                                "qd": np.asarray(state.qd, dtype=np.float64).tolist(),
                                "ee_pos": np.asarray(state.ee_pos, dtype=np.float64).tolist(),
                                "ee_quat": np.asarray(state.ee_quat, dtype=np.float64).tolist(),
                                "tau_requested": tau_applied.tolist(),
                                "tau_controller": tau_controller.tolist(),
                                "tau_gravity_static": tau_gravity.tolist(),
                                "tau_gravity_existing": tau_gravity.tolist(),
                                "tau_bias_live": compute_bias_torque_live(model, data_hold, joint_ids).tolist(),
                "tau_bias_static": tau_gravity.tolist(),
                                "tau_applied": tau_applied.tolist(),
                                "qfrc_actuator": qfrc_actuator.tolist(),
                                "qfrc_bias": qfrc_bias.tolist(),
                                "actuator_force": actuator_force.tolist(),
                                "x_error": float(start_ee[0] - state.ee_pos[0]),
                                "orientation_error_norm": float(np.linalg.norm(orientation_error_vec_wxyz(start_quat, state.ee_quat))),
                                "joint_limit_min_fraction": float(min(compute_joint_limit_proximity(model, state.q, joint_ids).values(), default=0.0)),
                                "termination_reason": "",
                            }
                            trace_rows.append(row)
                            trace_writer.write_row(row)
                            data_hold.qfrc_applied[:] = 0.0
                            mujoco.mj_step(model, data_hold)
                    summary_variant = _summarize_trace(
                        trace_rows,
                        start_q=pose_q,
                        start_ee=start_ee,
                        start_quat=start_quat,
                        variant_name=variant_name,
                        pose_name=pose.name,
                        duration_s=duration_s,
                    )
                    summary_variant.update(
                        {
                            "section": "hold_quality",
                            "run_label": run_label,
                            "controller_kind": "zero_torque",
                            "gravity_mode_used": "gravity_comp",
                            "gravity_compensation_active": True,
                            "raw_mode_used": False,
                            "trace_path": str(trace_path),
                            "summary_path": str(run_dir / "summary.json"),
                        }
                    )
                    summary_variant.update(_hold_validity_metrics(summary_variant))
                    summary_variant["hold_quality_score"] = float(_hold_quality_score(summary_variant))
                    summary_variant["mean_abs_tau_controller_nm"] = 0.0
                    summary_variant["max_abs_tau_controller_nm"] = 0.0
                    summary_variant["mean_abs_tau_gravity_nm"] = float(np.mean(np.abs(np.asarray([r["tau_gravity_static"] for r in trace_rows], dtype=np.float64))))
                    summary_variant["max_abs_tau_gravity_nm"] = float(np.max(np.abs(np.asarray([r["tau_gravity_static"] for r in trace_rows], dtype=np.float64))))
                    summary_variant["mean_abs_tau_applied_nm"] = float(np.mean(np.abs(np.asarray([r["tau_applied"] for r in trace_rows], dtype=np.float64))))
                    summary_variant["max_abs_tau_applied_nm"] = float(np.max(np.abs(np.asarray([r["tau_applied"] for r in trace_rows], dtype=np.float64))))
                else:
                    summary_variant = _run_variant_rollout(
                        model=model,
                        site_id=site_id,
                        joint_ids=joint_ids,
                        pose=pose,
                        duration_s=duration_s,
                        variant_name=variant_name,
                        controller_cfg=ctrl_cfg,
                        gravity_mode=str(mujoco_cfg.get("gravity_mode", "gravity_comp")),
                        run_dir=run_dir,
                        no_plot=bool(args.no_plot),
                    )
                    summary_variant.update(
                        {
                            "section": "hold_quality",
                            "run_label": run_label,
                            "trace_path": str(run_dir / "trace.jsonl"),
                            "summary_path": str(run_dir / "summary.json"),
                        }
                    )
                hold_results[variant_name] = summary_variant
                rows.append(summary_variant)
            hold_audit["poses"][pose_key][str(duration_s)] = hold_results

    # Cross-run comparisons.
    for (pose_name, duration_s), raw_row in raw_zero_baselines.items():
        sign_rows = sign_audit["poses"][pose_name][str(duration_s)]
        for variant_name, row in sign_rows.items():
            row["better_than_raw_zero"] = _variant_better_than_raw(row, raw_row)
            row["worse_than_raw_zero"] = _variant_worse_than_raw(row, raw_row)
        hold_rows = hold_audit["poses"][pose_name][str(duration_s)]
        for variant_name, row in hold_rows.items():
            row["better_than_raw_zero"] = _variant_better_than_raw(row, raw_row)
            row["worse_than_raw_zero"] = _variant_worse_than_raw(row, raw_row)

    # Aggregate summaries.
    sign_flat = [row for pose_rows in sign_audit["poses"].values() for dur_rows in pose_rows.values() for row in dur_rows.values()]
    hold_flat = [row for pose_rows in hold_audit["poses"].values() for dur_rows in pose_rows.values() for row in dur_rows.values()]
    mapping_flat = mapping_audit["cases"]
    clamp_flat = clamping_audit["cases"]
    all_rows = rows

    sign_by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sign_flat:
        sign_by_variant[str(row["variant_name"])].append(row)
    hold_by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in hold_flat:
        hold_by_variant[str(row["variant_name"])].append(row)

    def _avg_metric(rows_in: Sequence[Mapping[str, Any]], field: str) -> float:
        return float(np.mean([float(row.get(field, 0.0)) for row in rows_in])) if rows_in else 0.0

    sign_variant_summary = {
        name: {
            "mean_max_abs_ee_drift_m": _avg_metric(rows_in, "max_abs_ee_drift_m"),
            "mean_max_abs_orientation_error_rad": _avg_metric(rows_in, "max_abs_orientation_error_rad"),
            "mean_max_abs_qd_radps": _avg_metric(rows_in, "max_abs_qd_radps"),
            "mean_max_abs_tau_applied_nm": _avg_metric(rows_in, "max_abs_tau_applied_nm"),
            "mean_final_ee_drift_m": _avg_metric(rows_in, "final_ee_drift_m"),
            "mean_final_q_drift_norm": _avg_metric(rows_in, "final_q_drift_norm"),
            "mean_better_than_raw_zero": float(np.mean([bool(row.get("better_than_raw_zero", False)) for row in rows_in])) if rows_in else 0.0,
            "mean_worse_than_raw_zero": float(np.mean([bool(row.get("worse_than_raw_zero", False)) for row in rows_in])) if rows_in else 0.0,
        }
        for name, rows_in in sign_by_variant.items()
    }
    hold_variant_summary = {
        name: {
            "mean_hold_quality_score": _avg_metric(rows_in, "hold_quality_score"),
            "mean_hold_valid_normal": float(np.mean([bool(row.get("hold_valid_normal", False)) for row in rows_in])) if rows_in else 0.0,
            "mean_hold_valid_strict": float(np.mean([bool(row.get("hold_valid_strict", False)) for row in rows_in])) if rows_in else 0.0,
            "mean_max_abs_ee_drift_m": _avg_metric(rows_in, "max_abs_ee_drift_m"),
            "mean_max_abs_orientation_error_rad": _avg_metric(rows_in, "max_abs_orientation_error_rad"),
            "mean_max_abs_qd_radps": _avg_metric(rows_in, "max_abs_qd_radps"),
        }
        for name, rows_in in hold_by_variant.items()
    }

    best_sign_variant = min(
        (
            {
                "variant_name": name,
                **stats,
            }
            for name, stats in sign_variant_summary.items()
        ),
        key=lambda row: (
            -float(row.get("mean_better_than_raw_zero", 0.0)),
            float(row.get("mean_max_abs_ee_drift_m", np.inf)),
            float(row.get("mean_max_abs_orientation_error_rad", np.inf)),
            float(row.get("mean_max_abs_qd_radps", np.inf)),
        ),
        default=None,
    )
    best_hold_variant = min(
        (
            {
                "variant_name": name,
                **stats,
            }
            for name, stats in hold_variant_summary.items()
        ),
        key=lambda row: (
            -float(row.get("mean_hold_valid_normal", 0.0)),
            -float(row.get("mean_hold_valid_strict", 0.0)),
            -float(row.get("mean_hold_quality_score", 0.0)),
            float(row.get("mean_max_abs_ee_drift_m", np.inf)),
            float(row.get("mean_max_abs_orientation_error_rad", np.inf)),
        ),
        default=None,
    )

    best_transport_by_duration: dict[str, dict[str, Any] | None] = {}
    for duration_s in args.durations:
        duration_rows = [row for row in hold_flat if float(row.get("duration_s", -1.0)) == float(duration_s)]
        valid_rows = [row for row in duration_rows if bool(row.get("hold_valid_normal", False))]
        best_transport_by_duration[str(duration_s)] = _best_row(valid_rows, key=_variant_rank_key) if valid_rows else None

    gravity_comp_hold_by_duration = {
        str(duration_s): _best_row(
            [row for row in hold_flat if row["variant_name"] == "gravity_comp_hold" and float(row["duration_s"]) == float(duration_s)],
            key=_variant_rank_key,
        )
        for duration_s in args.durations
    }
    residual_hold_by_duration = {
        str(duration_s): _best_row(
            [row for row in hold_flat if row["variant_name"] == "residual_impedance_hold" and float(row["duration_s"]) == float(duration_s)],
            key=_variant_rank_key,
        )
        for duration_s in args.durations
    }

    gravity_comp_can_hold = {duration: bool(row and row.get("hold_valid_normal", False)) for duration, row in gravity_comp_hold_by_duration.items()}
    residual_can_hold = {duration: bool(row and row.get("hold_valid_normal", False)) for duration, row in residual_hold_by_duration.items()}

    likely_failure = "controller_limitation"
    if mapping_audit["max_abs_mapping_error"] > 1e-6:
        likely_failure = "actuator_mapping_bug"
    elif clamping_audit["max_abs_ctrl_vs_qfrc_error"] > 1e-6 and not clamping_audit["mujo_co_clip_appears"]:
        likely_failure = "clipping_issue"
    elif best_sign_variant and float(best_sign_variant.get("mean_better_than_raw_zero", 0.0)) < 0.5:
        likely_failure = "gravity_comp_sign_bug"

    summary.update(
        {
            "num_runs": int(len(all_rows)),
            "num_sign_runs": int(len(sign_flat)),
            "num_hold_runs": int(len(hold_flat)),
            "num_mapping_cases": int(len(mapping_flat)),
            "num_clamping_cases": int(len(clamp_flat)),
            "gravity_sign_best_variant": best_sign_variant,
            "hold_quality_best_variant": best_hold_variant,
            "gravity_comp_hold_can_hold_by_duration": gravity_comp_can_hold,
            "residual_impedance_hold_can_hold_by_duration": residual_can_hold,
            "likely_failure_mode": likely_failure,
            "recommendation": (
                "Gravity compensation sign and actuator mapping look consistent; if hold still fails at 3-5 s, "
                "the remaining limiter is controller structure/tuning rather than a MuJoCo torque sign bug."
                if likely_failure == "controller_limitation"
                else "Investigate the identified MuJoCo torque-path issue before further controller tuning."
            ),
            "gravity_sign_audit": {
                "pose_count": len(pose_specs),
                "durations": [float(v) for v in args.durations],
                "variants": list(GRAVITY_SIGN_VARIANTS),
                "best_variant_summary": best_sign_variant,
                "variant_summary": sign_variant_summary,
                "poses": sign_audit["poses"],
            },
            "hold_quality_audit": {
                "pose_count": len(pose_specs),
                "durations": [float(v) for v in args.durations],
                "variants": list(HOLD_VARIANTS),
                "best_variant_summary": best_hold_variant,
                "variant_summary": hold_variant_summary,
                "poses": hold_audit["poses"],
            },
            "actuator_mapping_audit": {
                **mapping_audit,
            },
            "clamping_audit": {
                **clamping_audit,
            },
        }
    )
    summary["gravity_compensation_sign_correct"] = bool(
        best_sign_variant is not None and float(best_sign_variant.get("mean_better_than_raw_zero", 0.0)) >= 0.5 and best_sign_variant["variant_name"] == "plus_gravity_comp"
    )
    summary["gravity_compensation_used_static_state"] = True
    summary["gravity_compensation_uses_live_qd_terms"] = bool(
        max(float(v.get("max_abs_live_bias_minus_existing_nm", 0.0)) for v in sign_flat) > 1e-8
    )

    summary_path = output_root / "summary.json"
    gravity_sign_path = output_root / "gravity_sign_audit.json"
    actuator_mapping_path = output_root / "actuator_mapping_audit.json"
    clamping_path = output_root / "clamping_audit.json"
    hold_quality_path = output_root / "hold_quality_audit.json"
    csv_path = output_root / "summary.csv"
    for path, payload in (
        (summary_path, summary),
        (gravity_sign_path, summary["gravity_sign_audit"]),
        (actuator_mapping_path, summary["actuator_mapping_audit"]),
        (clamping_path, summary["clamping_audit"]),
        (hold_quality_path, summary["hold_quality_audit"]),
    ):
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_csv(all_rows, csv_path)

    readme = output_root / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# UR5e MuJoCo Gravity-Torque Audit",
                "",
                f"- Output root: `{output_root}`",
                f"- Poses: {', '.join(args.poses)}",
                f"- Durations: {', '.join(str(float(v)) for v in args.durations)}",
                f"- Gravity compensation sign correct: `{summary['gravity_compensation_sign_correct']}`",
                f"- Actuator mapping one-to-one: `{summary['actuator_mapping_audit']['one_to_one']}`",
                f"- Max actuator mapping error: `{summary['actuator_mapping_audit']['max_abs_mapping_error']:.6g} Nm`",
                f"- MuJoCo ctrlrange clipping appears: `{summary['clamping_audit']['mujo_co_clip_appears']}`",
                f"- Controller-stage clip is separate in this lane: `{summary['clamping_audit']['controller_level_clip_appears']}`",
                f"- Likely failure mode: `{summary['likely_failure_mode']}`",
                f"- Recommendation: {summary['recommendation']}",
            ]
        ),
        encoding="utf-8",
    )

    if not args.no_plot:
        _write_plots(output_root, sign_flat, hold_flat, mapping_flat, clamp_flat)

    print(json.dumps(summary, indent=2))
    return 0


def _write_plots(
    output_root: Path,
    sign_rows: Sequence[Mapping[str, Any]],
    hold_rows: Sequence[Mapping[str, Any]],
    mapping_rows: Sequence[Mapping[str, Any]],
    clamp_rows: Sequence[Mapping[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plots_dir = output_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Drift / torque traces for one representative sign-audit pose-duration.
    if sign_rows:
        rep_rows = [row for row in sign_rows if row["pose_name"] == "active_origin"]
        if rep_rows:
            rep_variant = "plus_gravity_comp" if any(row["variant_name"] == "plus_gravity_comp" for row in rep_rows) else rep_rows[0]["variant_name"]
            rep = [row for row in rep_rows if row["variant_name"] == rep_variant]
            if rep and Path(rep[0].get("trace_path", "")).exists():
                trace_path = Path(rep[0]["trace_path"])
                trace_rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
                if trace_rows:
                    t = np.array([float(r["time_s"]) for r in trace_rows], dtype=np.float64)
                    ee = np.array([r["ee_pos"] for r in trace_rows], dtype=np.float64)
                    q = np.array([r["q"] for r in trace_rows], dtype=np.float64)
                    qd = np.array([r["qd"] for r in trace_rows], dtype=np.float64)
                    tau_g = np.array([r.get("tau_gravity_static", [0.0] * 6) for r in trace_rows], dtype=np.float64)
                    tau_a = np.array([r.get("tau_applied", [0.0] * 6) for r in trace_rows], dtype=np.float64)
                    fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True)
                    axes[0].plot(t, ee[:, 0], label="x")
                    axes[0].plot(t, ee[:, 1], label="y")
                    axes[0].plot(t, ee[:, 2], label="z")
                    axes[0].set_ylabel("EE pos [m]")
                    axes[0].legend(ncol=3, fontsize=8)
                    axes[1].plot(t, q)
                    axes[1].set_ylabel("q [rad]")
                    axes[2].plot(t, qd)
                    axes[2].set_ylabel("qd [rad/s]")
                    axes[3].plot(t, tau_g)
                    axes[3].set_ylabel("tau_g [Nm]")
                    axes[4].plot(t, tau_a)
                    axes[4].set_ylabel("tau_applied [Nm]")
                    axes[4].set_xlabel("time [s]")
                    fig.tight_layout()
                    fig.savefig(plots_dir / "sign_audit_rep_trace.png", dpi=150)
                    plt.close(fig)

    # Mapping / clamping comparison plots.
    if mapping_rows:
        fig, ax = plt.subplots(figsize=(10, 4))
        labels = [str(row.get("case_name", f"case_{i}")) for i, row in enumerate(mapping_rows)]
        vals = [float(row.get("max_abs_mapping_error", 0.0)) for row in mapping_rows]
        ax.bar(labels, vals)
        ax.set_ylabel("max abs mapping error [Nm]")
        ax.set_title("Actuator mapping audit")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(plots_dir / "actuator_mapping_errors.png", dpi=150)
        plt.close(fig)
    if clamp_rows:
        fig, ax = plt.subplots(figsize=(10, 4))
        labels = [str(row.get("case_name", f"case_{i}")) for i, row in enumerate(clamp_rows)]
        vals = [float(row.get("max_abs_ctrl_vs_qfrc_error", 0.0)) for row in clamp_rows]
        ax.bar(labels, vals)
        ax.set_ylabel("max abs ctrl vs qfrc error [Nm]")
        ax.set_title("Clamping audit")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(plots_dir / "clamping_errors.png", dpi=150)
        plt.close(fig)

    # Hold quality comparison across poses / durations.
    if hold_rows:
        fig, ax = plt.subplots(figsize=(11, 5))
        labels = [f"{row['pose_name']}|{row['variant_name']}|{row['duration_s']:g}" for row in hold_rows]
        vals = [float(row.get("max_abs_ee_drift_m", 0.0)) for row in hold_rows]
        ax.bar(labels, vals)
        ax.set_ylabel("max abs EE drift [m]")
        ax.set_title("Hold-quality comparison")
        ax.tick_params(axis="x", rotation=90)
        fig.tight_layout()
        fig.savefig(plots_dir / "hold_quality_comparison.png", dpi=150)
        plt.close(fig)


def run() -> int:
    return _run_audit()


if __name__ == "__main__":
    raise SystemExit(run())
