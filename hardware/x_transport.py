"""Dispatch X transport to position, direct-torque, or URScript inner loop."""

from __future__ import annotations

import time

import numpy as np

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .control_mode import normalize_control_mode
from .direct_torque_link import UR5eDirectTorqueLink
from .direct_torque_transport import DirectTorqueTransportResult, run_x_transport_direct_torque
from .link import UR5eLink
from .position_transport import PositionTransportResult, run_x_transport_position
from .safety import UR5eSafetyLimits
from .urscript_transport import UrscriptTransportResult, run_urscript_x_transport


def _validate_start_q_rad(start_q_rad: np.ndarray) -> np.ndarray:
    """Sanity-check a caller-supplied start pose before it ever reaches the
    robot: shape, finiteness, and the same absolute q_lower/q_upper ceiling
    ``check_joint_state`` enforces post-move. Catches gross typos (e.g.
    degrees instead of radians) before a moveJ is ever issued, rather than
    relying solely on the robot's own firmware check or the post-move
    ``verify_joint_pose`` -- the hardcoded HEIGHT_ALPHA_0_5_Q default never
    needed this because it's a fixed, already-known-good constant; a
    free-form CLI value does.
    """
    q = np.asarray(start_q_rad, dtype=np.float64).reshape(-1)
    if q.shape[0] != 6:
        raise ValueError(f"start_q_rad must have exactly 6 elements; got {q.shape[0]}")
    if not np.all(np.isfinite(q)):
        raise ValueError("start_q_rad contains NaN/Inf")
    limits = UR5eSafetyLimits()
    if np.any(q < limits.q_lower) or np.any(q > limits.q_upper):
        raise ValueError(
            f"start_q_rad {q.tolist()} exceeds absolute joint limits "
            f"[{limits.q_lower.tolist()}, {limits.q_upper.tolist()}]"
        )
    return q


@dataclass
class XTransportResult:
    ok: bool
    reason: str
    summary: dict[str, Any]
    trace_path: Path | None
    control_mode: str


class _HasDisconnect(Protocol):
    def safe_stop(self, reason: str) -> None: ...


def run_x_transport(
    *,
    control_mode: str,
    robot_ip: str,
    config_path: Path,
    target_x_delta_m: float,
    move_duration_s: float,
    duration_s: float,
    output_dir: Path | None,
    motion_opt_in: bool,
    dynamics_source: str = "local",
    shadow_osc: bool = True,
    skip_joint_move: bool = False,
    record_latency: bool = True,
    start_q_rad: np.ndarray | None = None,
    coriolis_feedforward: bool = False,
    gain_overrides: dict[str, float] | None = None,
    max_tcp_accel_mps2_override: float | None = None,
    accel_gap_cycles_override: int | None = None,
    speed_lowpass_alpha_override: float | None = None,
    accel_max_consecutive_violations_override: int | None = None,
    accel_hard_multiple_override: float | None = None,
    speed_max_consecutive_violations_override: int | None = None,
    speed_hard_multiple_override: float | None = None,
    enable_residual_observer: bool = True,
    residual_observer_async: bool = False,
) -> XTransportResult:
    mode = normalize_control_mode(control_mode)
    if start_q_rad is not None:
        start_q_rad = _validate_start_q_rad(start_q_rad)

    if mode == "urscript":
        raw: UrscriptTransportResult = run_urscript_x_transport(
            robot_ip=robot_ip,
            config_path=config_path,
            target_x_delta_m=target_x_delta_m,
            move_duration_s=move_duration_s,
            duration_s=duration_s,
            output_dir=output_dir,
            motion_opt_in=motion_opt_in,
            skip_joint_move=skip_joint_move,
            joint_target_q=start_q_rad,
            max_tcp_accel_mps2_override=max_tcp_accel_mps2_override,
            accel_gap_cycles_override=accel_gap_cycles_override,
            speed_lowpass_alpha_override=speed_lowpass_alpha_override,
            accel_max_consecutive_violations_override=accel_max_consecutive_violations_override,
            accel_hard_multiple_override=accel_hard_multiple_override,
            speed_max_consecutive_violations_override=speed_max_consecutive_violations_override,
            speed_hard_multiple_override=speed_hard_multiple_override,
        )
        return XTransportResult(
            ok=raw.ok,
            reason=raw.reason,
            summary=raw.summary,
            trace_path=output_dir / "supervisor_trace.jsonl" if output_dir and raw.summary.get("monitor_samples") else None,
            control_mode=mode,
        )

    if mode == "position":
        link = UR5eLink(robot_ip, frequency_hz=125.0)
        if not skip_joint_move:
            _joint_move_ur5e_link(link, motion_opt_in=motion_opt_in, target_q_rad=start_q_rad)
        raw_pos: PositionTransportResult = run_x_transport_position(
            link,
            config_path=config_path,
            target_x_delta_m=target_x_delta_m,
            move_duration_s=move_duration_s,
            duration_s=duration_s,
            output_dir=output_dir,
            motion_opt_in=motion_opt_in,
            shadow_osc=shadow_osc,
            max_tcp_accel_mps2_override=max_tcp_accel_mps2_override,
            accel_gap_cycles_override=accel_gap_cycles_override,
            speed_lowpass_alpha_override=speed_lowpass_alpha_override,
            accel_max_consecutive_violations_override=accel_max_consecutive_violations_override,
            accel_hard_multiple_override=accel_hard_multiple_override,
            speed_max_consecutive_violations_override=speed_max_consecutive_violations_override,
            speed_hard_multiple_override=speed_hard_multiple_override,
        )
        return XTransportResult(
            ok=raw_pos.ok,
            reason=raw_pos.reason,
            summary=raw_pos.summary,
            trace_path=raw_pos.trace_path,
            control_mode=mode,
        )

    link = UR5eDirectTorqueLink(robot_ip, frequency_hz=500.0)
    if not motion_opt_in:
        # Mirror run_x_transport_position's / run_x_transport_direct_torque's
        # own raise-before-connect check (hardware/position_transport.py,
        # hardware/direct_torque_transport.py) -- previously this branch
        # called link.connect() (which unconditionally opens BOTH the RTDE
        # receive and control interfaces) before any opt-in check ran,
        # unlike the position branch a few lines above, which never opens
        # control unless motion_opt_in is True.
        raise ValueError("motion_opt_in must be True for a live direct-torque transport")
    link.connect(with_control=motion_opt_in)
    if not skip_joint_move:
        from .joint_motion import move_joints_to_pose, verify_joint_pose
        from .poses import HEIGHT_ALPHA_0_5_CLEARANCE_Q
        from .safety import EStopLatch

        target_q_rad = HEIGHT_ALPHA_0_5_CLEARANCE_Q if start_q_rad is None else start_q_rad
        estop = EStopLatch()
        jres = move_joints_to_pose(
            link,
            estop,
            target_q_rad=target_q_rad,
            motion_opt_in=motion_opt_in,
        )
        ok_v, reason_v, _ = verify_joint_pose(link, target_q_rad=target_q_rad)
        if not jres.ok or not ok_v:
            link.safe_stop("joint_move_failed")
            return XTransportResult(
                ok=False,
                reason=jres.reason or reason_v or "joint_move_failed",
                summary={"success": False, "termination_reason": jres.reason or reason_v},
                trace_path=None,
                control_mode=mode,
            )

    raw_dt: DirectTorqueTransportResult = run_x_transport_direct_torque(
        link,
        config_path=config_path,
        target_x_delta_m=target_x_delta_m,
        move_duration_s=move_duration_s,
        duration_s=duration_s,
        output_dir=output_dir,
        motion_opt_in=motion_opt_in,
        record_latency=record_latency,
        dynamics_source=dynamics_source,
        coriolis_feedforward=coriolis_feedforward,
        gain_overrides=gain_overrides,
        max_tcp_accel_mps2_override=max_tcp_accel_mps2_override,
        accel_gap_cycles_override=accel_gap_cycles_override,
        speed_lowpass_alpha_override=speed_lowpass_alpha_override,
        accel_max_consecutive_violations_override=accel_max_consecutive_violations_override,
        accel_hard_multiple_override=accel_hard_multiple_override,
        speed_max_consecutive_violations_override=speed_max_consecutive_violations_override,
        speed_hard_multiple_override=speed_hard_multiple_override,
        enable_residual_observer=enable_residual_observer,
        residual_observer_async=residual_observer_async,
    )
    return XTransportResult(
        ok=raw_dt.ok,
        reason=raw_dt.reason,
        summary=raw_dt.summary,
        trace_path=raw_dt.trace_path,
        control_mode=mode,
    )


def _joint_move_ur5e_link(
    link: UR5eLink, *, motion_opt_in: bool, target_q_rad: np.ndarray | None = None
) -> None:
    from .poses import HEIGHT_ALPHA_0_5_CLEARANCE_Q

    if not motion_opt_in:
        raise ValueError("motion_opt_in required")
    target_q = (
        HEIGHT_ALPHA_0_5_CLEARANCE_Q if target_q_rad is None else np.asarray(target_q_rad, dtype=np.float64).reshape(6)
    )
    link.connect(with_control=True)
    link.move_j(target_q, speed_rad_s=0.5, acceleration_rad_s2=0.5)
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        st = link.read_state()
        if float(np.max(np.abs(st.q - target_q))) <= 0.03:
            return
        time.sleep(0.05)
    raise RuntimeError("joint move did not settle within tolerance")
