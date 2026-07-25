"""Deploy and supervise the on-robot URScript OSC inner loop."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from controller_core.safety import ImpedanceSafetyConfig, ImpedanceSafetyMonitor

from .link import RTDELinkError, RTDEStateError, UR5eState, _load_rtde_classes
from .poses import HEIGHT_ALPHA_0_5_Q
from .safety import (
    CartesianMoveLimits,
    CartesianMoveMonitor,
    DeadlineMonitor,
    StaleStateMonitor,
    UR5eSafetyLimits,
    is_robot_safety_normal,
)
from .transport_common import impedance_safety_config_from_section
from .urscript_gen import (
    DEFAULT_CONFIG,
    UrscriptOscParams,
    load_params_from_yaml,
    write_generated_script,
)


def _set_stop_register(control: Any, reg: int, value: int) -> None:
    """Set the on-robot stop register under whichever setter name this
    ur_rtde build exposes. Every call site that signals a stop must go
    through this helper, not call setInputIntRegister directly -- a build
    exposing only setInputIntegerRegister would otherwise silently fail to
    stop the robot on a real fault (setup already tried both names; the
    fault paths previously did not)."""
    if hasattr(control, "setInputIntRegister"):
        control.setInputIntRegister(reg, value)
    elif hasattr(control, "setInputIntegerRegister"):
        control.setInputIntegerRegister(reg, value)


@dataclass
class UrscriptTransportResult:
    ok: bool
    reason: str
    summary: dict[str, Any]
    script_path: Path | None


def _load_safety_cfg(config_path: Path) -> ImpedanceSafetyConfig:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    safety_raw = (cfg.get("controller", {}) or {}).get("safety", {}) or {}
    return impedance_safety_config_from_section(safety_raw)


def _read_state_from_receive(receive: Any) -> UR5eState:
    """Read q/qd/tcp_pose and raise RTDEStateError on any NaN/Inf -- matching
    hardware.link.UR5eLink.read_state()'s guarantee. Without this, a corrupt
    reading defeats the drift/orientation checks silently: abs(NaN) > thr is
    False, so a NaN tcp_pose would pass every CartesianMoveMonitor/
    ImpedanceSafetyMonitor check rather than failing loudly."""
    host_stamp_ns = time.monotonic_ns()
    try:
        q = np.asarray(receive.getActualQ(), dtype=np.float64).reshape(6)
        qd = np.asarray(receive.getActualQd(), dtype=np.float64).reshape(6)
        tcp_pose = np.asarray(receive.getActualTCPPose(), dtype=np.float64).reshape(6)
    except Exception as exc:
        raise RTDEStateError(f"RTDE state read failed: {exc}") from exc
    if not (np.all(np.isfinite(q)) and np.all(np.isfinite(qd)) and np.all(np.isfinite(tcp_pose))):
        raise RTDEStateError("NaN/Inf in q, qd, or tcp_pose")
    # Populate the robot clock too (was previously always None), matching
    # hardware.link.UR5eLink.read_state() -- StaleStateMonitor needs it to
    # detect a frozen-but-non-raising stream during motion.
    robot_timestamp_s = None
    get_timestamp = getattr(receive, "getTimestamp", None)
    if get_timestamp is not None:
        try:
            robot_timestamp_s = float(get_timestamp())
        except Exception:
            robot_timestamp_s = None
    safety_status = None
    get_safety = getattr(receive, "getSafetyStatusBits", None) or getattr(receive, "getSafetyStatus", None)
    if get_safety is not None:
        try:
            safety_status = int(get_safety())
        except Exception:
            safety_status = None
    return UR5eState(
        q=q,
        qd=qd,
        tcp_pose=tcp_pose,
        host_stamp_ns=host_stamp_ns,
        robot_timestamp_s=robot_timestamp_s,
        safety_status=safety_status,
    )


def run_urscript_x_transport(
    *,
    robot_ip: str,
    config_path: Path = DEFAULT_CONFIG,
    target_x_delta_m: float,
    move_duration_s: float,
    duration_s: float,
    output_dir: Path | None = None,
    motion_opt_in: bool,
    joint_target_q: np.ndarray | None = None,
    skip_joint_move: bool = False,
    monitor_hz: float = 125.0,
    use_lambda: bool | None = None,
) -> UrscriptTransportResult:
    """Generate URScript, run it on PolyScope, supervise from Python at monitor_hz."""
    if not motion_opt_in:
        raise ValueError("motion_opt_in must be True for live URScript transport")

    params = load_params_from_yaml(
        config_path,
        target_x_delta_m=target_x_delta_m,
        move_duration_s=move_duration_s,
        duration_s=duration_s,
        use_lambda=use_lambda,
    )
    script_path = None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        script_path = output_dir / "x_axis_osc_inner.script"
        write_generated_script(params, script_path)
        script_text = script_path.read_text(encoding="utf-8")
    else:
        from .urscript_gen import render_urscript

        script_text = render_urscript(params)

    control_cls, receive_cls = _load_rtde_classes()
    safety_cfg = _load_safety_cfg(config_path)
    safety = ImpedanceSafetyMonitor(safety_cfg)

    control = control_cls(robot_ip, 500.0)
    receive = receive_cls(robot_ip, 500.0)

    trace_rows: list[dict[str, Any]] = []
    termination_reason = "duration_complete"
    monitor_fault: list[str] = []
    stop_monitor = threading.Event()

    try:
        if not skip_joint_move:
            target_q = np.asarray(
                HEIGHT_ALPHA_0_5_Q if joint_target_q is None else joint_target_q,
                dtype=np.float64,
            ).reshape(6)
            try:
                control.moveJ(target_q.tolist(), 0.5, 0.5)
            except Exception as exc:
                return UrscriptTransportResult(
                    ok=False,
                    reason=f"moveJ failed: {exc}",
                    summary={"success": False, "termination_reason": f"moveJ_failed: {exc}"},
                    script_path=script_path,
                )
            deadline = time.monotonic() + 30.0
            settled = False
            while time.monotonic() < deadline:
                st = _read_state_from_receive(receive)
                if float(np.max(np.abs(st.q - target_q))) <= 0.03:
                    settled = True
                    break
                time.sleep(0.05)
            if not settled:
                return UrscriptTransportResult(
                    ok=False,
                    reason="joint move did not settle",
                    summary={"success": False, "termination_reason": "joint_move_timeout"},
                    script_path=script_path,
                )

        state0 = _read_state_from_receive(receive)
        x0 = float(state0.tcp_pose[0])
        safety.reset()
        safety.set_initial_position(np.asarray(state0.tcp_pose[:3], dtype=np.float64), move_axis=0)

        # ImpedanceSafetyMonitor (above) has no TCP speed/acceleration/waypoint-jump
        # ceiling -- only drift/orientation/joint-velocity/axis-growth. Layer
        # CartesianMoveMonitor on top for the checks position mode already has and
        # this mode was missing. Reuse safety_cfg's already-active thresholds for
        # qd/drift/orientation (so this doesn't introduce a second, different trip
        # point for a check that already exists) and CartesianMoveLimits' own
        # conservative defaults for the genuinely new speed/accel/jump checks.
        move_limits = CartesianMoveLimits.from_impedance_safety_config(safety_cfg)
        move_monitor = CartesianMoveMonitor(move_limits)
        move_monitor.set_start(state0.tcp_pose, move_axis_index=0)
        target_tcp_pose = state0.tcp_pose.copy()
        target_tcp_pose[0] = x0 + float(target_x_delta_m)

        stop_reg = int(params.stop_input_int_reg)
        _set_stop_register(control, stop_reg, 0)

        stop_monitor.clear()
        dt_monitor = 1.0 / float(monitor_hz)

        # Enforce the previously-unchecked max_deadline_ms, and catch a
        # frozen-but-non-raising RTDE stream, on every supervisor cycle (see
        # hardware/safety.py for the trip-condition reasoning). The control law
        # itself runs on-robot in URScript; this Python supervisor is the only
        # place that can notice the telemetry feeding its safety checks has
        # stalled or that its own loop can no longer keep its budget.
        deadline_monitor = DeadlineMonitor(UR5eSafetyLimits().max_deadline_ms)
        stale_monitor = StaleStateMonitor()

        def _supervisor() -> None:
            t0 = time.monotonic()
            while not stop_monitor.is_set():
                cycle_start = time.monotonic()
                try:
                    st = _read_state_from_receive(receive)
                except Exception as exc:
                    monitor_fault.append(f"monitor_read_failed: {exc}")
                    _set_stop_register(control, stop_reg, 1)
                    return
                stale_reason = stale_monitor.record(st.robot_timestamp_s, st.host_stamp_ns)
                if stale_reason:
                    monitor_fault.append(stale_reason)
                    _set_stop_register(control, stop_reg, 1)
                    return
                elapsed = time.monotonic() - t0
                trace_rows.append(
                    {
                        "time_s": elapsed,
                        "q": st.q.tolist(),
                        "qd": st.qd.tolist(),
                        "tcp_pose": st.tcp_pose.tolist(),
                        "achieved_x_delta_m": float(st.tcp_pose[0] - x0),
                    }
                )
                robot_state = {
                    "q": st.q,
                    "qd": st.qd,
                    "ee_pos": st.tcp_pose[:3],
                    "ee_quat": np.array([1.0, 0.0, 0.0, 0.0]),
                    "target_x": x0 + float(target_x_delta_m),
                    "transport_axis_index": 0,
                }
                orientation_error_rad = float(np.linalg.norm(st.tcp_pose[3:] - state0.tcp_pose[3:]))
                axis_target_moving = elapsed <= float(move_duration_s)
                decision = safety.check(
                    robot_state,
                    x_error=float((x0 + float(target_x_delta_m)) - st.tcp_pose[0]),
                    orientation_error_norm=orientation_error_rad,
                    axis_target_moving=axis_target_moving,
                )
                if not decision.ok:
                    monitor_fault.append(decision.reason or "safety_stop")
                    _set_stop_register(control, stop_reg, 1)
                    return
                move_decision = move_monitor.check(
                    q=st.q,
                    qd=st.qd,
                    tcp_pose=st.tcp_pose,
                    target_tcp_pose=target_tcp_pose,
                    orientation_error_rad=orientation_error_rad,
                    axis_target_moving=axis_target_moving,
                    dt_s=dt_monitor,
                )
                if not move_decision.ok:
                    monitor_fault.append(move_decision.reason or "cartesian_move_stop")
                    _set_stop_register(control, stop_reg, 1)
                    return
                if not is_robot_safety_normal(st.safety_status):
                    monitor_fault.append(f"robot_safety_status_abnormal: {st.safety_status}")
                    _set_stop_register(control, stop_reg, 1)
                    return
                if elapsed > float(duration_s) + 2.0:
                    monitor_fault.append("supervisor_timeout")
                    _set_stop_register(control, stop_reg, 1)
                    return
                # Overrun = supervisor work (read + all checks) past its period
                # budget. A stalled supervisor read that still returns is the
                # case this catches; the fixed sleep below never subtracts work,
                # so this is the only place lateness is acted on.
                overrun_ns = int(max(0.0, (time.monotonic() - cycle_start) - dt_monitor) * 1e9)
                deadline_reason = deadline_monitor.record(overrun_ns)
                if deadline_reason:
                    monitor_fault.append(deadline_reason)
                    _set_stop_register(control, stop_reg, 1)
                    return
                time.sleep(max(0.0, dt_monitor - 0.001))

        supervisor = threading.Thread(target=_supervisor, name="urscript-supervisor", daemon=True)
        supervisor.start()

        send_ok = False
        if hasattr(control, "sendCustomScript"):
            send_ok = bool(control.sendCustomScript(script_text))
        elif hasattr(control, "sendCustomScriptFile") and script_path is not None:
            send_ok = bool(control.sendCustomScriptFile(str(script_path)))
        else:
            raise RTDELinkError("ur_rtde RTDEControlInterface has no sendCustomScript()")

        stop_monitor.set()
        supervisor.join(timeout=5.0)

        if monitor_fault:
            termination_reason = monitor_fault[0]
        elif not send_ok:
            termination_reason = "sendCustomScript_failed"
        else:
            termination_reason = "duration_complete"

    except RTDEStateError as exc:
        termination_reason = f"rtde_error: {exc}"
    except Exception as exc:
        termination_reason = f"transport_error: {exc}"
    finally:
        stop_monitor.set()
        try:
            _set_stop_register(control, int(params.stop_input_int_reg), 1)
        except Exception:
            pass
        for obj in (control, receive):
            try:
                obj.disconnect()
            except Exception:
                pass

    final_state = trace_rows[-1] if trace_rows else {}
    achieved = float(final_state.get("achieved_x_delta_m", 0.0)) if final_state else 0.0
    summary: dict[str, Any] = {
        "backend": "urscript_inner_loop",
        "config_path": str(config_path),
        "target_x_delta_m": float(target_x_delta_m),
        "move_duration_s": float(move_duration_s),
        "duration_s": float(duration_s),
        "termination_reason": termination_reason,
        "achieved_x_delta_m": achieved,
        "monitor_samples": len(trace_rows),
        "use_lambda": bool(params.use_lambda),
        "direct_torque_friction": "v2_per_joint",
        "success": termination_reason == "duration_complete" and not monitor_fault,
    }
    if output_dir is not None:
        if trace_rows:
            trace_path = output_dir / "supervisor_trace.jsonl"
            with trace_path.open("w", encoding="utf-8") as fh:
                for row in trace_rows:
                    fh.write(json.dumps(row) + "\n")
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    ok = bool(summary["success"])
    reason = "" if ok else str(termination_reason)
    return UrscriptTransportResult(ok=ok, reason=reason, summary=summary, script_path=script_path)
