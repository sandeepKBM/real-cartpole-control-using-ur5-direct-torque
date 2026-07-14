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
from .urscript_transport import UrscriptTransportResult, run_urscript_x_transport


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
) -> XTransportResult:
    mode = normalize_control_mode(control_mode)

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
            _joint_move_ur5e_link(link, motion_opt_in=motion_opt_in)
        raw_pos: PositionTransportResult = run_x_transport_position(
            link,
            config_path=config_path,
            target_x_delta_m=target_x_delta_m,
            move_duration_s=move_duration_s,
            duration_s=duration_s,
            output_dir=output_dir,
            motion_opt_in=motion_opt_in,
            shadow_osc=shadow_osc,
        )
        return XTransportResult(
            ok=raw_pos.ok,
            reason=raw_pos.reason,
            summary=raw_pos.summary,
            trace_path=raw_pos.trace_path,
            control_mode=mode,
        )

    link = UR5eDirectTorqueLink(robot_ip, frequency_hz=500.0)
    link.connect()
    if not skip_joint_move:
        from .joint_motion import move_joints_to_pose, verify_joint_pose
        from .poses import HEIGHT_ALPHA_0_5_Q
        from .safety import EStopLatch

        estop = EStopLatch()
        jres = move_joints_to_pose(
            link,
            estop,
            target_q_rad=HEIGHT_ALPHA_0_5_Q,
            motion_opt_in=motion_opt_in,
        )
        ok_v, reason_v, _ = verify_joint_pose(link, target_q_rad=HEIGHT_ALPHA_0_5_Q)
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
    )
    return XTransportResult(
        ok=raw_dt.ok,
        reason=raw_dt.reason,
        summary=raw_dt.summary,
        trace_path=raw_dt.trace_path,
        control_mode=mode,
    )


def _joint_move_ur5e_link(link: UR5eLink, *, motion_opt_in: bool) -> None:
    from .poses import HEIGHT_ALPHA_0_5_Q

    if not motion_opt_in:
        raise ValueError("motion_opt_in required")
    link.connect(with_control=True)
    link.move_j(HEIGHT_ALPHA_0_5_Q, speed_rad_s=0.5, acceleration_rad_s2=0.5)
    deadline = time.monotonic() + 30.0
    target_q = HEIGHT_ALPHA_0_5_Q
    while time.monotonic() < deadline:
        st = link.read_state()
        if float(np.max(np.abs(st.q - target_q))) <= 0.03:
            return
        time.sleep(0.05)
    raise RuntimeError("joint move did not settle within tolerance")
