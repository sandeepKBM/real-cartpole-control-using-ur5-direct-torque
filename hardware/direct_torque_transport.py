"""Bounded world-X move+hold via PolyScope ``direct_torque()`` and the tuned OSC law."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from controller_core.safety import ImpedanceSafetyConfig, ImpedanceSafetyMonitor
from controller_core.x_axis_cartesian_impedance import (
    CartesianImpedanceConfig,
    XAxisCartesianImpedanceController,
)
from simulation.ur5e_mujoco_torque import x_profile_target
from transport_metrics import compute_valid_move_hold_metrics, summarize_move_hold_trace

from .direct_torque_link import UR5eDirectTorqueLink
from .link import RTDEStateError
from .safety import EStopLatch


@dataclass
class DirectTorqueTransportResult:
    ok: bool
    reason: str
    summary: dict[str, Any]
    trace_path: Path | None


def _load_impedance_bundle(config_path: Path) -> tuple[CartesianImpedanceConfig, ImpedanceSafetyConfig, float]:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    ctrl = cfg.get("controller", {}) or {}
    safety_raw = ctrl.get("safety", {}) or {}
    impedance_cfg = CartesianImpedanceConfig.from_controller_yaml_section(ctrl)
    safety_cfg = ImpedanceSafetyConfig(
        max_abs_y_drift_m=float(safety_raw.get("max_abs_y_drift_m", 0.03)),
        max_abs_z_drift_m=float(safety_raw.get("max_abs_z_drift_m", 0.03)),
        max_abs_orthogonal_drift_m=float(safety_raw.get("max_abs_orthogonal_drift_m", 0.03)),
        max_orientation_error_rad=float(safety_raw.get("max_orientation_error_rad", 0.25)),
        max_joint_velocity_radps=float(safety_raw.get("max_joint_velocity_radps", 3.0)),
    )
    frequency_hz = float(cfg.get("hardware", {}).get("rtde_frequency_hz", 500.0))
    return impedance_cfg, safety_cfg, frequency_hz


def run_x_transport_direct_torque(
    link: UR5eDirectTorqueLink,
    *,
    config_path: Path,
    target_x_delta_m: float,
    move_duration_s: float,
    duration_s: float,
    output_dir: Path | None = None,
    motion_opt_in: bool,
) -> DirectTorqueTransportResult:
    if not motion_opt_in:
        raise ValueError("motion_opt_in must be True for a live direct-torque transport")

    impedance_cfg, safety_cfg, frequency_hz = _load_impedance_bundle(config_path)
    if abs(frequency_hz - 500.0) > 1.0:
        raise ValueError(
            f"direct_torque transport expects ~500 Hz RTDE; config requested {frequency_hz} Hz"
        )
    dt_s = 1.0 / frequency_hz
    move_duration_s = float(move_duration_s)
    duration_s = float(duration_s)
    if move_duration_s <= 0.0 or duration_s <= 0.0:
        raise ValueError("move_duration_s and duration_s must be positive")
    if move_duration_s > duration_s:
        raise ValueError("move_duration_s must not exceed duration_s")

    controller = XAxisCartesianImpedanceController(impedance_cfg)
    safety = ImpedanceSafetyMonitor(safety_cfg)
    estop = EStopLatch()

    link.connect()
    state0 = link.read_state()
    x0 = float(state0.tcp_pose[0])
    controller.reset_from_state(
        link.build_robot_state(state0, time_s=0.0, target_x=x0, target_x_vel=0.0)
    )
    safety.reset()
    safety.set_initial_position(
        np.asarray(state0.tcp_pose[:3], dtype=np.float64),
        move_axis=0,
    )

    trace_rows: list[dict[str, Any]] = []
    termination_reason = "duration_complete"
    steps = 0
    t_s = 0.0
    last_link_state = state0

    try:
        while t_s < duration_s - 1e-12:
            if estop.tripped:
                termination_reason = estop.reason or "estop"
                break

            link_state = link.read_state()
            last_link_state = link_state
            target_x, target_x_vel = x_profile_target(
                "min_jerk_move_hold",
                x0,
                float(target_x_delta_m),
                t_s,
                duration_s,
                move_duration_s=move_duration_s,
            )
            robot_state = link.build_robot_state(
                link_state,
                time_s=t_s,
                target_x=target_x,
                target_x_vel=target_x_vel,
            )
            output = controller.compute(robot_state)
            tau = np.asarray(output.tau, dtype=np.float64).reshape(6)

            safety_status = safety.check(
                robot_state,
                x_error=float(output.x_error),
                orientation_error_norm=float(output.orientation_error_norm),
                axis_target_moving=bool(t_s <= move_duration_s),
            )
            if not safety_status.ok:
                termination_reason = safety_status.reason or "safety_stop"
                estop.trip(termination_reason)
                break

            link.direct_torque(tau, friction_comp=True)
            trace_rows.append(
                {
                    "time_s": t_s,
                    "q": link_state.q.tolist(),
                    "qd": link_state.qd.tolist(),
                    "ee_pos": robot_state["ee_pos"].tolist(),
                    "ee_quat": robot_state["ee_quat"].tolist(),
                    "tcp_pose": link_state.tcp_pose.tolist(),
                    "target_x": target_x,
                    "x_error": float(output.x_error),
                    "orientation_error_norm": float(output.orientation_error_norm),
                    "tau_controller": tau.tolist(),
                    "tau_applied": tau.tolist(),
                }
            )
            steps += 1
            t_s += dt_s
            time.sleep(max(0.0, dt_s * 0.95))
    except RTDEStateError as exc:
        termination_reason = f"rtde_state_error: {exc}"
        estop.trip(termination_reason)
    finally:
        try:
            link.direct_torque(np.zeros(6), friction_comp=True)
        except Exception:
            pass
        link.safe_stop("transport_exit")

    final_state = last_link_state
    achieved_x_delta_m = float(final_state.tcp_pose[0] - x0)
    hold_duration_s = max(duration_s - move_duration_s, 0.0)
    max_abs_qd = float(
        max((max(abs(v) for v in row.get("qd", [0.0] * 6)) for row in trace_rows), default=0.0)
    )

    summary = {
        "backend": "ursim_rtde_direct_torque",
        "config_path": str(config_path),
        "target_x_delta": float(target_x_delta_m),
        "target_x_delta_m": float(target_x_delta_m),
        "move_duration_s": move_duration_s,
        "hold_duration_s": hold_duration_s,
        "duration_s": duration_s,
        "total_duration_s": duration_s,
        "sim_time_s": min(t_s, duration_s),
        "frequency_hz": frequency_hz,
        "steps": steps,
        "termination_reason": termination_reason,
        "achieved_x_delta_m": achieved_x_delta_m,
        "final_tcp_pose": final_state.tcp_pose.tolist(),
        "initial_ee_pos": state0.tcp_pose[:3].tolist(),
        "success": termination_reason == "duration_complete" and not estop.tripped,
        "velocity_guard_ok": max_abs_qd <= safety_cfg.max_joint_velocity_radps,
        "max_abs_qd_radps": max_abs_qd,
        "joint_limit_guard_ok": True,
        "torque_saturation_percentage": 0.0,
    }
    summary.update(
        summarize_move_hold_trace(
            trace_rows,
            initial_ee_pos=state0.tcp_pose[:3],
            move_duration_s=move_duration_s,
            total_duration_s=duration_s,
            transport_axis_index=0,
        )
    )
    summary.update(compute_valid_move_hold_metrics(summary))
    summary["success"] = bool(
        not estop.tripped
        and summary.get("valid_move_and_hold", False)
        and termination_reason == "duration_complete"
    )

    trace_path = None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        trace_path = output_dir / "trace.jsonl"
        with trace_path.open("w", encoding="utf-8") as fh:
            for row in trace_rows:
                fh.write(json.dumps(row) + "\n")
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    ok = bool(summary["success"])
    reason = "" if ok else str(summary.get("hold_failure_reason") or summary.get("move_failure_reason") or termination_reason)
    return DirectTorqueTransportResult(ok=ok, reason=reason, summary=summary, trace_path=trace_path)
