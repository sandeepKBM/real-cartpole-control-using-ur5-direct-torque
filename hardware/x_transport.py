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
from .transport_common import validate_transport_axis_index
from .urscript_transport import UrscriptTransportResult, run_urscript_x_transport
from .velocity_transport import VelocityTransportResult, run_x_transport_velocity


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
    transport_axis_index: int = 0,
    dynamics_source: str = "local",
    shadow_osc: bool = True,
    skip_joint_move: bool = False,
    record_latency: bool = True,
    start_q_rad: np.ndarray | None = None,
    coriolis_feedforward: bool = False,
    gain_overrides: dict[str, float] | None = None,
    max_tcp_accel_mps2_override: float | None = None,
    max_tcp_speed_mps_override: float | None = None,
    accel_gap_cycles_override: int | None = None,
    speed_lowpass_alpha_override: float | None = None,
    speed_limit_gap_cycles_override: int | None = None,
    speed_limit_lowpass_alpha_override: float | None = None,
    accel_max_consecutive_violations_override: int | None = None,
    accel_hard_multiple_override: float | None = None,
    speed_max_consecutive_violations_override: int | None = None,
    speed_hard_multiple_override: float | None = None,
    accel_variable_tolerance_override: bool | None = None,
    speed_variable_tolerance_override: bool | None = None,
    enable_residual_observer: bool = True,
    residual_observer_async: bool = False,
    trajectory_profile: str = "min_jerk_move_hold",
    target_accel_mps2: float | None = None,
    telemetry_gap_bridge: bool = False,
    telemetry_gap_bridge_max_cycles: int = 2,
    # velocity mode only -- forwarded to run_x_transport_velocity's own
    # rate_hz/speed_l_acceleration params (defaults match that function's
    # defaults, so omitting these preserves prior behavior exactly). Every
    # other mode ignores them.
    rate_hz: float = 125.0,
    speed_l_acceleration: float = 1.2,
) -> XTransportResult:
    mode = normalize_control_mode(control_mode)
    transport_axis_index = validate_transport_axis_index(transport_axis_index)
    if transport_axis_index != 0 and mode != "position":
        # Deliberate fail-fast, added with the axis plumbing itself (see the
        # three transport modules' `transport_axis_index` parameter). The
        # guards, the commanded waypoint and the reported metrics are now all
        # axis-generic, but the INNER CONTROL LAW of every mode except
        # `position` still regulates world-X and nothing else:
        #   * direct_torque -> controller_core/x_axis_cartesian_impedance/
        #     controller.py::compute() reads `ee_pos[0]` / `target_x` literally
        #     (`x_err = x_des - p[0]`); there is no axis field on
        #     CartesianImpedanceConfig or on its RobotState contract.
        #   * urscript      -> assets/urscript/x_axis_osc_inner.script.template
        #     computes `x0 = tcp0[0]` / `x_err = x_des - tcp[0]` on-robot.
        #   * velocity      -> hardware/velocity_transport.py is not plumbed
        #     for a transport axis at all (still hardcoded index 0).
        # Allowing a non-zero axis there would command motion along X while
        # guarding (and scoring) along Y/Z -- i.e. the robot pushes off-axis
        # until the 0.03 m orthogonal-drift guard trips. `position` mode is
        # exempt because its commanded motion IS the waypoint this function
        # writes (servoL), which is genuinely axis-generic; only its optional
        # `shadow_osc` diagnostic remains X-only (never sent to the robot).
        # Lifting this requires an axis-aware control law, not more plumbing.
        raise ValueError(
            f"transport_axis_index={transport_axis_index} is only supported for "
            f"control_mode='position'; control_mode={mode!r}'s inner control law regulates "
            "world-X only (see hardware/x_transport.py for the per-mode detail), so a "
            "non-zero axis would command X while guarding the selected axis"
        )
    if start_q_rad is not None:
        start_q_rad = _validate_start_q_rad(start_q_rad)
    if trajectory_profile != "min_jerk_move_hold" and mode != "direct_torque":
        # accel/duration profiles are wired for the direct_torque OSC loop
        # only tonight -- position/urscript have their own separate
        # trajectory-generation paths not touched here.
        raise ValueError(
            f"trajectory_profile={trajectory_profile!r} is only supported for control_mode='direct_torque'"
        )

    if mode == "urscript":
        raw: UrscriptTransportResult = run_urscript_x_transport(
            robot_ip=robot_ip,
            config_path=config_path,
            target_x_delta_m=target_x_delta_m,
            move_duration_s=move_duration_s,
            duration_s=duration_s,
            output_dir=output_dir,
            motion_opt_in=motion_opt_in,
            transport_axis_index=transport_axis_index,
            skip_joint_move=skip_joint_move,
            joint_target_q=start_q_rad,
            max_tcp_accel_mps2_override=max_tcp_accel_mps2_override,
            max_tcp_speed_mps_override=max_tcp_speed_mps_override,
            accel_gap_cycles_override=accel_gap_cycles_override,
            speed_lowpass_alpha_override=speed_lowpass_alpha_override,
            speed_limit_gap_cycles_override=speed_limit_gap_cycles_override,
            speed_limit_lowpass_alpha_override=speed_limit_lowpass_alpha_override,
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
            transport_axis_index=transport_axis_index,
            shadow_osc=shadow_osc,
            max_tcp_accel_mps2_override=max_tcp_accel_mps2_override,
            max_tcp_speed_mps_override=max_tcp_speed_mps_override,
            accel_gap_cycles_override=accel_gap_cycles_override,
            speed_lowpass_alpha_override=speed_lowpass_alpha_override,
            speed_limit_gap_cycles_override=speed_limit_gap_cycles_override,
            speed_limit_lowpass_alpha_override=speed_limit_lowpass_alpha_override,
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

    if mode == "velocity":
        # frequency_hz must match the loop's own rate_hz -- the link's
        # internal ConnectionHealth staleness budget and the RTDE
        # receive/control interfaces are constructed against this value, so a
        # mismatch (e.g. --rate-hz 250 against a hardcoded 125.0 link) is a
        # real desync, not just cosmetic. Default 125.0 matches
        # run_x_transport_velocity's own default, preserving prior behavior
        # for every caller that doesn't pass rate_hz.
        link_v = UR5eLink(robot_ip, frequency_hz=rate_hz)
        if not skip_joint_move:
            _joint_move_ur5e_link(link_v, motion_opt_in=motion_opt_in, target_q_rad=start_q_rad)
        raw_vel: VelocityTransportResult = run_x_transport_velocity(
            link_v,
            config_path=config_path,
            target_x_delta_m=target_x_delta_m,
            move_duration_s=move_duration_s,
            duration_s=duration_s,
            output_dir=output_dir,
            motion_opt_in=motion_opt_in,
            max_tcp_accel_mps2_override=max_tcp_accel_mps2_override,
            max_tcp_speed_mps_override=max_tcp_speed_mps_override,
            accel_gap_cycles_override=accel_gap_cycles_override,
            speed_lowpass_alpha_override=speed_lowpass_alpha_override,
            speed_limit_gap_cycles_override=speed_limit_gap_cycles_override,
            speed_limit_lowpass_alpha_override=speed_limit_lowpass_alpha_override,
            accel_max_consecutive_violations_override=accel_max_consecutive_violations_override,
            accel_hard_multiple_override=accel_hard_multiple_override,
            speed_max_consecutive_violations_override=speed_max_consecutive_violations_override,
            speed_hard_multiple_override=speed_hard_multiple_override,
            rate_hz=rate_hz,
            speed_l_acceleration=speed_l_acceleration,
        )
        return XTransportResult(
            ok=raw_vel.ok,
            reason=raw_vel.reason,
            summary=raw_vel.summary,
            trace_path=raw_vel.trace_path,
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
        transport_axis_index=transport_axis_index,
        record_latency=record_latency,
        dynamics_source=dynamics_source,
        coriolis_feedforward=coriolis_feedforward,
        gain_overrides=gain_overrides,
        max_tcp_accel_mps2_override=max_tcp_accel_mps2_override,
        max_tcp_speed_mps_override=max_tcp_speed_mps_override,
        accel_gap_cycles_override=accel_gap_cycles_override,
        speed_lowpass_alpha_override=speed_lowpass_alpha_override,
        speed_limit_gap_cycles_override=speed_limit_gap_cycles_override,
        speed_limit_lowpass_alpha_override=speed_limit_lowpass_alpha_override,
        accel_max_consecutive_violations_override=accel_max_consecutive_violations_override,
        accel_hard_multiple_override=accel_hard_multiple_override,
        speed_max_consecutive_violations_override=speed_max_consecutive_violations_override,
        speed_hard_multiple_override=speed_hard_multiple_override,
        accel_variable_tolerance_override=accel_variable_tolerance_override,
        speed_variable_tolerance_override=speed_variable_tolerance_override,
        enable_residual_observer=enable_residual_observer,
        trajectory_profile=trajectory_profile,
        target_accel_mps2=target_accel_mps2,
        residual_observer_async=residual_observer_async,
        telemetry_gap_bridge=telemetry_gap_bridge,
        telemetry_gap_bridge_max_cycles=telemetry_gap_bridge_max_cycles,
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
