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
from .urscript_gen import (
    DEFAULT_CONFIG,
    UrscriptOscParams,
    load_params_from_yaml,
    write_generated_script,
)


@dataclass
class UrscriptTransportResult:
    ok: bool
    reason: str
    summary: dict[str, Any]
    script_path: Path | None


def _load_safety_cfg(config_path: Path) -> ImpedanceSafetyConfig:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    safety_raw = (cfg.get("controller", {}) or {}).get("safety", {}) or {}
    return ImpedanceSafetyConfig(
        max_abs_y_drift_m=float(safety_raw.get("max_abs_y_drift_m", 0.03)),
        max_abs_z_drift_m=float(safety_raw.get("max_abs_z_drift_m", 0.03)),
        max_abs_orthogonal_drift_m=float(safety_raw.get("max_abs_orthogonal_drift_m", 0.03)),
        max_orientation_error_rad=float(safety_raw.get("max_orientation_error_rad", 0.25)),
        max_joint_velocity_radps=float(safety_raw.get("max_joint_velocity_radps", 3.0)),
    )


def _read_state_from_receive(receive: Any) -> UR5eState:
    host_stamp_ns = time.monotonic_ns()
    q = np.asarray(receive.getActualQ(), dtype=np.float64).reshape(6)
    qd = np.asarray(receive.getActualQd(), dtype=np.float64).reshape(6)
    tcp_pose = np.asarray(receive.getActualTCPPose(), dtype=np.float64).reshape(6)
    return UR5eState(
        q=q,
        qd=qd,
        tcp_pose=tcp_pose,
        host_stamp_ns=host_stamp_ns,
        robot_timestamp_s=None,
        safety_status=None,
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

        stop_reg = int(params.stop_input_int_reg)
        if hasattr(control, "setInputIntRegister"):
            control.setInputIntRegister(stop_reg, 0)
        elif hasattr(control, "setInputIntegerRegister"):
            control.setInputIntegerRegister(stop_reg, 0)

        stop_monitor.clear()
        dt_monitor = 1.0 / float(monitor_hz)

        def _supervisor() -> None:
            t0 = time.monotonic()
            while not stop_monitor.is_set():
                try:
                    st = _read_state_from_receive(receive)
                except Exception as exc:
                    monitor_fault.append(f"monitor_read_failed: {exc}")
                    if hasattr(control, "setInputIntRegister"):
                        control.setInputIntRegister(stop_reg, 1)
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
                decision = safety.check(
                    robot_state,
                    x_error=float((x0 + float(target_x_delta_m)) - st.tcp_pose[0]),
                    orientation_error_norm=float(np.linalg.norm(st.tcp_pose[3:] - state0.tcp_pose[3:])),
                    axis_target_moving=elapsed <= float(move_duration_s),
                )
                if not decision.ok:
                    monitor_fault.append(decision.reason or "safety_stop")
                    if hasattr(control, "setInputIntRegister"):
                        control.setInputIntRegister(stop_reg, 1)
                    return
                if elapsed > float(duration_s) + 2.0:
                    monitor_fault.append("supervisor_timeout")
                    if hasattr(control, "setInputIntRegister"):
                        control.setInputIntRegister(stop_reg, 1)
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
            if hasattr(control, "setInputIntRegister"):
                control.setInputIntRegister(int(params.stop_input_int_reg), 1)
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
