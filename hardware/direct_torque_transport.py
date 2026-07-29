"""Bounded world-X move+hold via PolyScope ``direct_torque()`` and the tuned OSC law."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from controller_core.dynamics_residual import joint_acceleration_residual, predict_joint_acceleration
from controller_core.model_dynamics import PinocchioUR5eDynamics
from controller_core.safety import ImpedanceSafetyConfig, ImpedanceSafetyMonitor
from controller_core.x_axis_cartesian_impedance import (
    CartesianImpedanceConfig,
    XAxisCartesianImpedanceController,
)
from simulation.ur5e_mujoco_torque import x_profile_target
from transport_metrics import compute_valid_move_hold_metrics, summarize_move_hold_trace

from .direct_torque_link import UR5eDirectTorqueLink
from .joint_accel_estimator import JointAccelEstimator
from .latency import PhaseLatencyRecorder
from .link import RTDEStateError
from .local_dynamics import LocalPinocchioDynamics, LocalPinocchioFastDynamics, normalize_dynamics_source
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
    max_tcp_accel_mps2_override: float | None = None,
    accel_gap_cycles_override: int | None = None,
    speed_lowpass_alpha_override: float | None = None,
    enable_residual_observer: bool = True,
    residual_qdd_gap_cycles: int = 1,
    residual_qdd_lowpass_alpha: float = 1.0,
) -> DirectTorqueTransportResult:
    if not motion_opt_in:
        raise ValueError("motion_opt_in must be True for a live direct-torque transport")

    dynamics_source = normalize_dynamics_source(dynamics_source)
    if coriolis_feedforward and dynamics_source not in ("local", "local_pinocchio"):
        # The robot's own firmware compensates gravity inside directTorque()
        # (never add that in Python -- see AGENTS.md), but it does NOT
        # automatically apply Coriolis/centrifugal compensation the way it
        # does gravity; it only exposes the values for retrieval (confirmed
        # against Universal Robots' own Direct Torque Control documentation,
        # 2026-07-26). This flag adds that missing term via the selected
        # local dynamics provider's coriolis(), which needs dynamics_source
        # in {local, local_pinocchio}'s q/qd -> J/M/C pipeline; the rtde
        # dynamics_source path has no equivalent Coriolis getter wired up yet.
        raise ValueError(
            "coriolis_feedforward requires dynamics_source in {'local', 'local_pinocchio'} "
            "(rtde path not implemented)"
        )
    if dynamics_source == "local":
        local_dynamics = LocalPinocchioDynamics()
    elif dynamics_source == "local_pinocchio":
        # Opt-in fast path (2026-07-29): a genuinely Pinocchio-backed J(q)/M(q)/
        # Coriolis provider, ~10x lower per-call latency than the MuJoCo-backed
        # LocalPinocchioDynamics/LocalMujocoDynamics default -- see
        # docs/status/local_dynamics_speedup_investigation_2026-07-29.md.
        # Default behavior (dynamics_source="local" or "rtde") is unchanged.
        local_dynamics = LocalPinocchioFastDynamics()
    else:
        local_dynamics = None

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

    # Diagnostic-only dynamics residual observer (2026-07-29, direct_torque
    # only -- see docs/status/direct_torque_residual_observer_2026-07-29.md).
    # Predicts qdd from known rigid-body dynamics + the true total commanded
    # torque and compares it to a noise-robust qdd estimated from consecutive
    # qd samples; logged to trace_rows for post-hoc analysis ONLY. Built
    # unconditionally from a dedicated PinocchioUR5eDynamics instance
    # (independent of `dynamics_source`, which only affects the CONTROLLER's
    # own J/M source) so the residual model is identical regardless of which
    # dynamics_source this run uses. Constructed before link.connect(), same
    # as gain_overrides above, so a bad gap_cycles/lowpass_alpha value fails
    # before ever touching the robot.
    residual_dynamics = None
    residual_accel_estimator = None
    if enable_residual_observer:
        try:
            residual_dynamics = PinocchioUR5eDynamics()
            residual_accel_estimator = JointAccelEstimator(
                gap_cycles=residual_qdd_gap_cycles, lowpass_alpha=residual_qdd_lowpass_alpha
            )
        except Exception as exc:  # noqa: BLE001 - diagnostic-only feature must never block real motion
            print(
                f"[direct_torque_transport] WARNING: residual observer disabled, failed to "
                f"initialize PinocchioUR5eDynamics ({type(exc).__name__}: {exc}). This is a "
                f"diagnostic-only feature (see "
                f"docs/status/direct_torque_residual_observer_2026-07-29.md) -- transport will "
                f"proceed without it rather than block real hardware operation over a missing/"
                f"broken diagnostic dependency.",
                flush=True,
            )
            residual_dynamics = None
            residual_accel_estimator = None

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
    if residual_accel_estimator is not None:
        residual_accel_estimator.reset(state0.qd)
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
    if max_tcp_accel_mps2_override is not None:
        # See the identical override in hardware/position_transport.py for
        # the full rationale (real-hardware noise-amplification finding,
        # 2026-07-28). Direct-torque runs at 500 Hz vs position mode's
        # 125 Hz, so the same one-step finite-difference accel estimate's
        # noise amplification (~1/dt^2) is ~16x worse here -- expect this
        # override to matter more, not less, in this mode.
        move_limits = replace(move_limits, max_tcp_accel_mps2=float(max_tcp_accel_mps2_override))
    accel_overrides: dict[str, float] = {}
    if accel_gap_cycles_override is not None:
        accel_overrides["accel_gap_cycles"] = int(accel_gap_cycles_override)
    if speed_lowpass_alpha_override is not None:
        accel_overrides["speed_lowpass_alpha"] = float(speed_lowpass_alpha_override)
    if accel_overrides:
        move_limits = replace(move_limits, **accel_overrides)
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
    # Period-relative deadline cap (2026-07-29, see
    # docs/status/deadline_monitor_period_relative_fix_2026-07-29.md): the flat
    # max_deadline_ms default (3.0 ms) was calibrated for the three 125 Hz
    # loops (8 ms period) and is too loose for this loop's own 500 Hz/2 ms
    # period -- a real ~2 ms overrun on real hardware sat under it undetected.
    # min() makes this a strict no-op for any loop whose period is >= 6 ms
    # (0.5 * period_ms >= max_deadline_ms there), so this only ever tightens
    # the threshold for dt_s well below that -- exactly this loop's 2 ms case.
    effective_deadline_ms = min(
        safety_limits.max_deadline_ms,
        safety_limits.max_deadline_fraction_of_period * dt_s * 1000.0,
    )
    deadline_monitor = DeadlineMonitor(effective_deadline_ms)
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

            # Diagnostic-only dynamics residual observer (see the setup
            # comment above and docs/status/direct_torque_residual_observer_2026-07-29.md).
            # Reuses this cycle's already-computed mass_matrix (from whichever
            # dynamics_source is active) rather than recomputing it. gravity(q)
            # is added back to `tau` (the Python-side commanded torque) to
            # reconstruct the TRUE total physical torque, since PolyScope's
            # directTorque() auto-adds gravity compensation that Python never
            # sends (AGENTS.md: never add gravity twice) -- bias(q, qd)
            # subtracts an equal g(q) term back out, so qdd_pred is
            # insensitive to any residual mismatch between this Pinocchio
            # model's gravity(q) and PolyScope's own internal one, as long as
            # both are evaluated consistently here.
            qdd_pred = qdd_measured = qdd_residual = None
            if residual_dynamics is not None and residual_accel_estimator is not None:
                t_residual = monotonic_ns()
                tau_true_total = tau + residual_dynamics.gravity(link_state.q)
                bias = residual_dynamics.bias(link_state.q, link_state.qd)
                qdd_pred = predict_joint_acceleration(mass_matrix, tau_true_total, bias)
                real_dt_s = dt_s if interval_ns is None else max(dt_s, interval_ns / 1e9)
                qdd_measured = residual_accel_estimator.update(link_state.qd, real_dt_s)
                if qdd_measured is not None:
                    qdd_residual = joint_acceleration_residual(qdd_measured, qdd_pred)
                if phases is not None:
                    phases.record("residual_observer_ns", monotonic_ns() - t_residual)

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
                    # Controller-internal diagnostics already computed inside
                    # controller.compute() (see CartesianImpedanceOutput) but
                    # previously discarded every cycle -- found blocking the
                    # 2026-07-28 velocity-overshoot investigation (see
                    # docs/status/clock_timing_late_cycles_2026-07-28.md).
                    "jacobian_cond": float(output.jacobian_cond),
                    "singular_scale": float(output.singular_scale),
                    "task_scale": float(output.task_scale),
                    "task_backtrack_iters": int(output.task_backtrack_iters),
                    "tau_controller": tau_controller.tolist(),
                    "tau_coriolis": tau_coriolis.tolist(),
                    "coriolis_feedforward_active": bool(coriolis_feedforward),
                    "tau_applied": tau.tolist(),
                    # Diagnostic-only, never read by ImpedanceSafetyMonitor or
                    # CartesianMoveMonitor -- see the computation above and
                    # docs/status/direct_torque_residual_observer_2026-07-29.md.
                    # qdd_measured/qdd_residual are None for the first
                    # residual_qdd_gap_cycles cycles (estimator still filling
                    # its gap window) or whenever enable_residual_observer is
                    # False.
                    "qdd_pred": None if qdd_pred is None else qdd_pred.tolist(),
                    "qdd_measured": None if qdd_measured is None else qdd_measured.tolist(),
                    "qdd_residual": None if qdd_residual is None else qdd_residual.tolist(),
                    "qdd_residual_norm": (
                        None if qdd_residual is None else float(np.linalg.norm(qdd_residual))
                    ),
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
