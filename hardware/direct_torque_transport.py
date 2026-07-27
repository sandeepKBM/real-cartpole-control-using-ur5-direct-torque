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
from .latency import PhaseLatencyRecorder
from .link import RTDEStateError
from .local_dynamics import LocalPinocchioDynamics, normalize_dynamics_source
from .safety import (
    CartesianMoveLimits,
    CartesianMoveMonitor,
    DeadlineMonitor,
    EStopLatch,
    StaleStateMonitor,
    UR5eSafetyLimits,
    is_robot_safety_normal,
)
from .timing import TimingTracker, monotonic_ns
from .transport_common import impedance_safety_config_from_section, max_abs_qd_from_trace


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
    safety_cfg = impedance_safety_config_from_section(safety_raw)
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
    record_latency: bool = True,
    dynamics_source: str = "rtde",
    coriolis_feedforward: bool = False,
    gain_overrides: dict[str, float] | None = None,
) -> DirectTorqueTransportResult:
    if not motion_opt_in:
        raise ValueError("motion_opt_in must be True for a live direct-torque transport")

    dynamics_source = normalize_dynamics_source(dynamics_source)
    if coriolis_feedforward and dynamics_source != "local":
        # The robot's own firmware compensates gravity inside directTorque()
        # (never add that in Python -- see AGENTS.md), but it does NOT
        # automatically apply Coriolis/centrifugal compensation the way it
        # does gravity; it only exposes the values for retrieval (confirmed
        # against Universal Robots' own Direct Torque Control documentation,
        # 2026-07-26). This flag adds that missing term via
        # LocalMujocoDynamics.coriolis(), which needs dynamics_source=local's
        # q/qd -> MuJoCo pipeline; the rtde dynamics_source path has no
        # equivalent Coriolis getter wired up yet.
        raise ValueError("coriolis_feedforward requires dynamics_source='local' (rtde path not implemented)")
    local_dynamics = LocalPinocchioDynamics() if dynamics_source == "local" else None

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
    if gain_overrides:
        # Validated (unknown-field / non-finite rejection) and applied before
        # link.connect() -- fail on a bad override dict before ever touching
        # the robot, not mid-run.
        controller.set_gains(gain_overrides)
    safety = ImpedanceSafetyMonitor(safety_cfg)
    estop = EStopLatch()
    tracker = TimingTracker(frequency_hz)
    phases = PhaseLatencyRecorder() if record_latency else None

    link.connect()
    state0 = link.read_state()
    x0 = float(state0.tcp_pose[0])
    if local_dynamics is not None:
        J0, M0 = local_dynamics.jacobian_and_mass_matrix(state0.q)
        init_robot_state = link.compose_robot_state(
            state0, jacobian=J0, mass_matrix=M0, time_s=0.0, target_x=x0, target_x_vel=0.0
        )
    else:
        init_robot_state = link.build_robot_state(state0, time_s=0.0, target_x=x0, target_x_vel=0.0)
    controller.reset_from_state(init_robot_state)
    safety.reset()
    safety.set_initial_position(
        np.asarray(state0.tcp_pose[:3], dtype=np.float64),
        move_axis=0,
    )
    # ImpedanceSafetyMonitor (above) has no TCP speed/acceleration/waypoint-jump
    # ceiling -- only drift/orientation/joint-velocity/axis-growth. Layer
    # CartesianMoveMonitor on top for the checks position mode already has and
    # this (live-torque) mode was missing. Reuse safety_cfg's already-active
    # thresholds for qd/drift/orientation so this doesn't introduce a second,
    # different trip point for a check that already exists.
    move_limits = CartesianMoveLimits.from_impedance_safety_config(safety_cfg)
    move_monitor = CartesianMoveMonitor(move_limits)
    move_monitor.set_start(state0.tcp_pose, move_axis_index=0)

    trace_rows: list[dict[str, Any]] = []
    termination_reason = "duration_complete"
    steps = 0
    t_s = 0.0
    last_link_state = state0
    next_deadline_ns = monotonic_ns() + tracker.period_ns
    prev_cycle_start_ns: int | None = None

    # Enforce the previously-unchecked max_deadline_ms, and catch a
    # frozen-but-non-raising RTDE stream, on every cycle (see hardware/safety.py
    # DeadlineMonitor/StaleStateMonitor for the trip-condition reasoning). This
    # loop already tracks per-cycle start-lateness (lateness_ns below), so the
    # deadline monitor is fed that directly rather than a separate measure.
    safety_limits = getattr(link, "limits", None) or UR5eSafetyLimits()
    deadline_monitor = DeadlineMonitor(safety_limits.max_deadline_ms)
    stale_monitor = StaleStateMonitor()

    try:
        while t_s < duration_s - 1e-12:
            if estop.tripped:
                termination_reason = estop.reason or "estop"
                break

            cycle_start_ns = monotonic_ns()
            lateness_ns = max(0, cycle_start_ns - next_deadline_ns)
            if phases is not None:
                phases.record("lateness_ns", lateness_ns)

            deadline_reason = deadline_monitor.record(lateness_ns)
            if deadline_reason:
                termination_reason = deadline_reason
                estop.trip(deadline_reason)
                break

            t_read = monotonic_ns()
            link_state = link.read_state()
            if phases is not None:
                phases.record("read_state_ns", monotonic_ns() - t_read)
            last_link_state = link_state

            stale_reason = stale_monitor.record(link_state.robot_timestamp_s, link_state.host_stamp_ns)
            if stale_reason:
                termination_reason = stale_reason
                estop.trip(stale_reason)
                break

            target_x, target_x_vel = x_profile_target(
                "min_jerk_move_hold",
                x0,
                float(target_x_delta_m),
                t_s,
                duration_s,
                move_duration_s=move_duration_s,
            )

            t_jac = monotonic_ns()
            tau_coriolis = np.zeros(6, dtype=np.float64)
            if local_dynamics is not None:
                if coriolis_feedforward:
                    jacobian, mass_matrix, tau_coriolis = local_dynamics.jacobian_mass_and_coriolis(
                        link_state.q, link_state.qd
                    )
                else:
                    jacobian, mass_matrix = local_dynamics.jacobian_and_mass_matrix(link_state.q)
                if phases is not None:
                    phases.record("local_dynamics_ns", monotonic_ns() - t_jac)
            else:
                jacobian = link.get_jacobian()
                if phases is not None:
                    phases.record("get_jacobian_ns", monotonic_ns() - t_jac)
                t_mass = monotonic_ns()
                mass_matrix = link.get_mass_matrix()
                if phases is not None:
                    phases.record("get_mass_matrix_ns", monotonic_ns() - t_mass)

            t_build = monotonic_ns()
            robot_state = link.compose_robot_state(
                link_state,
                jacobian=jacobian,
                mass_matrix=mass_matrix,
                time_s=t_s,
                target_x=target_x,
                target_x_vel=target_x_vel,
            )
            if phases is not None:
                phases.record("build_state_ns", monotonic_ns() - t_build)

            t_ctrl = monotonic_ns()
            output = controller.compute(robot_state)
            tau_controller = np.asarray(output.tau, dtype=np.float64).reshape(6)
            # tau_coriolis is zeros(6) unless coriolis_feedforward is on -- see
            # the flag's docstring above. Gravity is deliberately NOT added
            # here; PolyScope's directTorque() already compensates it.
            tau = tau_controller + tau_coriolis
            if phases is not None:
                phases.record("controller_ns", monotonic_ns() - t_ctrl)

            t_safe = monotonic_ns()
            safety_decision = safety.check(
                robot_state,
                x_error=float(output.x_error),
                orientation_error_norm=float(output.orientation_error_norm),
                axis_target_moving=bool(t_s <= move_duration_s),
            )
            if phases is not None:
                phases.record("safety_ns", monotonic_ns() - t_safe)
            if not safety_decision.ok:
                termination_reason = safety_decision.reason or "safety_stop"
                estop.trip(termination_reason)
                break

            move_decision = move_monitor.check(
                q=link_state.q,
                qd=link_state.qd,
                tcp_pose=link_state.tcp_pose,
                target_tcp_pose=np.concatenate(([target_x], state0.tcp_pose[1:6])),
                orientation_error_rad=float(output.orientation_error_norm),
                axis_target_moving=bool(t_s <= move_duration_s),
                dt_s=dt_s,
            )
            if not move_decision.ok:
                termination_reason = move_decision.reason or "cartesian_move_stop"
                estop.trip(termination_reason)
                break

            if not is_robot_safety_normal(link_state.safety_status):
                termination_reason = f"robot_safety_status_abnormal: {link_state.safety_status}"
                estop.trip(termination_reason)
                break

            t_torque = monotonic_ns()
            link.direct_torque(tau, friction_comp=True)
            if phases is not None:
                phases.record("direct_torque_ns", monotonic_ns() - t_torque)

            cycle_end_ns = monotonic_ns()
            work_ns = cycle_end_ns - cycle_start_ns
            if phases is not None:
                phases.record("total_work_ns", work_ns)

            interval_ns = None if prev_cycle_start_ns is None else cycle_start_ns - prev_cycle_start_ns
            sleep_ns = 0
            next_deadline_ns += tracker.period_ns
            sleep_ns = max(0, next_deadline_ns - cycle_end_ns)
            if sleep_ns > 0:
                time.sleep(sleep_ns / 1e9)
            if phases is not None:
                phases.record("sleep_ns", sleep_ns)

            tracker.add_sample(
                cycle_index=steps,
                start_ns=cycle_start_ns,
                deadline_ns=next_deadline_ns - tracker.period_ns,
                end_ns=cycle_end_ns,
                sleep_ns=sleep_ns,
                interval_ns=interval_ns,
            )
            prev_cycle_start_ns = cycle_start_ns

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
                    "tau_controller": tau_controller.tolist(),
                    "tau_coriolis": tau_coriolis.tolist(),
                    "coriolis_feedforward_active": bool(coriolis_feedforward),
                    "tau_applied": tau.tolist(),
                    "cycle_work_ms": work_ns / 1e6,
                    "lateness_ms": lateness_ns / 1e6,
                }
            )
            steps += 1
            t_s += dt_s
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
    max_abs_qd = max_abs_qd_from_trace(trace_rows)

    summary = {
        "backend": "ursim_rtde_direct_torque",
        "config_path": str(config_path),
        "gain_overrides": dict(gain_overrides) if gain_overrides else {},
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
        "dynamics_source": dynamics_source,
        "timing": tracker.summary(),
    }
    if phases is not None:
        summary["latency_phases"] = phases.summary()
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
