"""Staged UR5e hardware test helpers.

All stages default to no motion unless the caller explicitly opts in for the
servoJ motion stages. These helpers are designed for direct use by the scripts
under ``tools/``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .logging import write_json
from .ros_topics import AsyncRosVisualizer, RosTopicSample
from .safety_limits import MotionCommandGuard, UR5eSafetyLimits, UR5eStateGuard
from .timing import TimingTracker, monotonic_ns
from .ur5e_rtde_bridge import UR5eRTDEBridge, UR5eState, _sample_row, _summarize_exception


@dataclass
class StageResult:
    ok: bool
    stage: str
    reason: str = ""
    report: dict[str, Any] = field(default_factory=dict)


def _print_summary(stage: str, result: StageResult) -> None:
    status = "PASS" if result.ok else "FAIL"
    print(f"[{status}] {stage}: {result.reason or 'ok'}")
    if result.report:
        timing = result.report.get("timing", {})
        if timing:
            print(
                "  timing: "
                f"mean={timing.get('cycle_interval', {}).get('mean_ms')} ms, "
                f"p95={timing.get('cycle_interval', {}).get('p95_ms')} ms, "
                f"p99={timing.get('cycle_interval', {}).get('p99_ms')} ms, "
                f"max={timing.get('cycle_interval', {}).get('max_ms')} ms"
            )


def _build_report(
    *,
    stage: str,
    bridge: UR5eRTDEBridge,
    timing: TimingTracker,
    samples: list[dict[str, Any]],
    exceptions: list[dict[str, Any]],
    robot_connected: bool,
    motion_attempted: bool,
    stop_errors: list[str],
    status: str,
    reason: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "stage": stage,
        "status": status,
        "reason": reason,
        "bridge": bridge.report(),
        "timing": timing.summary(),
        "samples": samples,
        "exceptions": exceptions,
        "robot": {
            "robot_ip": bridge.robot_ip,
            "robot_connected": bool(robot_connected),
            "motion_attempted": bool(motion_attempted),
            "stop_errors": stop_errors,
        },
    }
    if extra:
        report.update(extra)
    return report


def _finalize(
    *,
    stage: str,
    ok: bool,
    reason: str,
    bridge: UR5eRTDEBridge,
    timing: TimingTracker,
    samples: list[dict[str, Any]],
    exceptions: list[dict[str, Any]],
    robot_connected: bool,
    motion_attempted: bool,
    stop_errors: list[str],
    output_path: str | Path,
    extra: dict[str, Any] | None = None,
) -> StageResult:
    report = _build_report(
        stage=stage,
        bridge=bridge,
        timing=timing,
        samples=samples,
        exceptions=exceptions,
        robot_connected=robot_connected,
        motion_attempted=motion_attempted,
        stop_errors=stop_errors,
        status="PASS" if ok else "FAIL",
        reason=reason,
        extra=extra,
    )
    write_json(output_path, report)
    return StageResult(ok=ok, stage=stage, reason=reason, report=report)


def _start_visualizer(
    publish_ros_topics: bool,
    ros_prefix: str,
) -> AsyncRosVisualizer | None:
    if not publish_ros_topics:
        return None
    vis = AsyncRosVisualizer(ros_prefix=ros_prefix)
    if not vis.start():
        return None
    return vis


def _submit_ros_sample(
    vis: AsyncRosVisualizer | None,
    *,
    state: UR5eState,
    cmd_q: np.ndarray | None,
    cmd_qd: np.ndarray | None,
) -> None:
    if vis is None:
        return
    vis.submit(
        RosTopicSample(
            stamp_ns=state.host_stamp_ns,
            q_real=state.q,
            qd_real=state.qd,
            q_desired=cmd_q,
            qd_desired=cmd_qd,
            tcp_pose_real=state.tcp_pose,
            tcp_pose_desired=state.tcp_pose if cmd_q is None else None,
        )
    )


def run_receive_only(
    *,
    robot_ip: str,
    frequency: float,
    duration: float,
    output: str | Path,
    max_deadline_ms: float = 3.0,
    publish_ros_topics: bool = False,
    ros_prefix: str = "/ur5e",
    dry_run: bool = False,
    limits: UR5eSafetyLimits | None = None,
) -> StageResult:
    stage = "receive-only"
    if dry_run:
        timing = TimingTracker(frequency)
        samples: list[dict[str, Any]] = []
        exceptions: list[dict[str, Any]] = []
        start_ns = monotonic_ns()
        deadline_ns = start_ns + timing.period_ns
        end_ns = start_ns + int(round(duration * 1e9))
        cycle_index = 0
        prev_start_ns: int | None = None
        while monotonic_ns() < end_ns:
            now_ns = monotonic_ns()
            if now_ns < deadline_ns:
                time.sleep((deadline_ns - now_ns) / 1e9)
                continue
            cycle_start_ns = monotonic_ns()
            work_start_ns = monotonic_ns()
            _ = cycle_index * 0
            work_end_ns = monotonic_ns()
            interval_ns = None if prev_start_ns is None else cycle_start_ns - prev_start_ns
            timing.add_sample(
                cycle_index=cycle_index,
                start_ns=cycle_start_ns,
                deadline_ns=deadline_ns,
                end_ns=work_end_ns,
                sleep_ns=0,
                interval_ns=interval_ns,
            )
            samples.append(
                {
                    "cycle_index": cycle_index,
                    "host_stamp_ns": int(cycle_start_ns),
                    "robot_timestamp_s": None,
                    "q": None,
                    "qd": None,
                    "tcp_pose": None,
                    "timing": timing.samples[-1].__dict__,
                }
            )
            prev_start_ns = cycle_start_ns
            cycle_index += 1
            deadline_ns += timing.period_ns
        bridge = UR5eRTDEBridge(robot_ip=robot_ip, frequency=frequency, limits=limits)
        result = _finalize(
            stage=stage,
            ok=True,
            reason="dry-run completed without robot I/O",
            bridge=bridge,
            timing=timing,
            samples=samples,
            exceptions=exceptions,
            robot_connected=False,
            motion_attempted=False,
            stop_errors=[],
            output_path=output,
        )
        _print_summary(stage, result)
        return result

    bridge = UR5eRTDEBridge(robot_ip=robot_ip, frequency=frequency, limits=limits)
    vis = _start_visualizer(publish_ros_topics, ros_prefix)
    timing = TimingTracker(frequency, overrun_threshold_s=max_deadline_ms / 1000.0)
    deadline_limit_ns = int(round(max_deadline_ms * 1e6))
    samples: list[dict[str, Any]] = []
    exceptions: list[dict[str, Any]] = []
    robot_connected = False
    motion_attempted = False
    stop_errors: list[str] = []
    reason = ""
    ok = False
    state_guard = UR5eStateGuard(bridge.limits)
    try:
        bridge.connect_receive_only()
        robot_connected = True
        start_ns = monotonic_ns()
        deadline_ns = start_ns + timing.period_ns
        end_ns = start_ns + int(round(duration * 1e9))
        cycle_index = 0
        prev_start_ns: int | None = None
        while monotonic_ns() < end_ns:
            now_ns = monotonic_ns()
            if now_ns < deadline_ns:
                time.sleep((deadline_ns - now_ns) / 1e9)
                continue
            cycle_start_ns = monotonic_ns()
            cycle_lateness_ns = max(0, cycle_start_ns - deadline_ns)
            if cycle_lateness_ns > deadline_limit_ns:
                reason = f"deadline miss {cycle_lateness_ns / 1e6:.3f} ms > {timing.overrun_threshold_s * 1e3:.3f} ms"
                break
            state = bridge.read_state()
            if state.tcp_pose is None:
                reason = "RTDE receive interface did not expose actual_TCP_pose"
                break
            state_decision = state_guard.check(
                q=state.q,
                qd=state.qd,
                tcp_pose=state.tcp_pose,
                host_stamp_ns=state.host_stamp_ns,
                robot_stamp_s=state.robot_timestamp_s,
            )
            if not state_decision.ok:
                reason = state_decision.reason
                break
            cycle_end_ns = monotonic_ns()
            if cycle_end_ns - cycle_start_ns > deadline_limit_ns:
                reason = (
                    f"cycle duration {(cycle_end_ns - cycle_start_ns) / 1e6:.3f} ms "
                    f"> {timing.overrun_threshold_s * 1e3:.3f} ms"
                )
                break
            timing.add_sample(
                cycle_index=cycle_index,
                start_ns=cycle_start_ns,
                deadline_ns=deadline_ns,
                end_ns=cycle_end_ns,
                sleep_ns=0,
                interval_ns=None if prev_start_ns is None else cycle_start_ns - prev_start_ns,
            )
            samples.append(
                _sample_row(
                    phase=stage,
                    cycle_index=cycle_index,
                    state=state,
                    cmd_q=None,
                    cmd_qd=None,
                    timing=timing.samples[-1].__dict__,
                    extra={"state_ok": True},
                )
            )
            _submit_ros_sample(vis, state=state, cmd_q=None, cmd_qd=None)
            prev_start_ns = cycle_start_ns
            cycle_index += 1
            deadline_ns += timing.period_ns
        else:
            ok = True
            reason = "completed receive-only sampling"
    except Exception as exc:
        exceptions.append(_summarize_exception(exc))
        reason = f"{type(exc).__name__}: {exc}"
    finally:
        stop_errors.extend(bridge.safe_stop("receive-only exit"))
        if vis is not None:
            vis.stop()
    result = _finalize(
        stage=stage,
        ok=ok,
        reason=reason,
        bridge=bridge,
        timing=timing,
        samples=samples,
        exceptions=exceptions,
        robot_connected=robot_connected,
        motion_attempted=motion_attempted,
        stop_errors=stop_errors,
        output_path=output,
    )
    _print_summary(stage, result)
    return result


def _motion_loop_common(
    *,
    stage: str,
    bridge: UR5eRTDEBridge,
    duration: float,
    gain: float,
    lookahead_time: float,
    velocity: float,
    acceleration: float,
    max_deadline_ms: float,
    output: str | Path,
    publish_ros_topics: bool = False,
    ros_prefix: str = "/ur5e",
) -> StageResult:
    vis = _start_visualizer(publish_ros_topics, ros_prefix)
    timing = TimingTracker(bridge.frequency, overrun_threshold_s=max_deadline_ms / 1000.0)
    samples: list[dict[str, Any]] = []
    exceptions: list[dict[str, Any]] = []
    robot_connected = bridge.receive is not None or bridge.control is not None
    motion_attempted = True
    stop_errors: list[str] = []
    reason = ""
    ok = False
    state_guard = UR5eStateGuard(bridge.limits)
    try:
        start_ns = monotonic_ns()
        deadline_ns = start_ns + timing.period_ns
        end_ns = start_ns + int(round(duration * 1e9))
        cycle_index = 0
        prev_start_ns: int | None = None
        while monotonic_ns() < end_ns:
            now_ns = monotonic_ns()
            if now_ns < deadline_ns:
                time.sleep((deadline_ns - now_ns) / 1e9)
                continue
            cycle_start_ns = monotonic_ns()
            state = bridge.read_state()
            state_decision = state_guard.check(
                q=state.q,
                qd=state.qd,
                tcp_pose=state.tcp_pose,
                host_stamp_ns=state.host_stamp_ns,
                robot_stamp_s=state.robot_timestamp_s,
            )
            if not state_decision.ok:
                reason = state_decision.reason
                break
            timing.add_sample(
                cycle_index=cycle_index,
                start_ns=cycle_start_ns,
                deadline_ns=deadline_ns,
                end_ns=monotonic_ns(),
                sleep_ns=0,
                interval_ns=None if prev_start_ns is None else cycle_start_ns - prev_start_ns,
            )
            prev_start_ns = cycle_start_ns
            cycle_index += 1
            deadline_ns += timing.period_ns
        else:
            ok = True
            reason = f"completed {stage}"
    except Exception as exc:
        exceptions.append(_summarize_exception(exc))
        reason = f"{type(exc).__name__}: {exc}"
    finally:
        stop_errors.extend(bridge.safe_stop(f"{stage} exit"))
        if vis is not None:
            vis.stop()
    result = _finalize(
        stage=stage,
        ok=ok,
        reason=reason,
        bridge=bridge,
        timing=timing,
        samples=samples,
        exceptions=exceptions,
        robot_connected=robot_connected,
        motion_attempted=motion_attempted,
        stop_errors=stop_errors,
        output_path=output,
    )
    _print_summary(stage, result)
    return result


def run_servoj_zero_hold(
    *,
    robot_ip: str,
    frequency: float,
    duration: float,
    gain: float,
    lookahead_time: float,
    velocity: float,
    acceleration: float,
    max_deadline_ms: float,
    motion_opt_in: bool,
    output: str | Path,
    publish_ros_topics: bool = False,
    ros_prefix: str = "/ur5e",
    limits: UR5eSafetyLimits | None = None,
) -> StageResult:
    stage = "servoj-zero-hold"
    if not motion_opt_in:
        raise SystemExit(
            "--i-understand-this-moves-the-robot motion opt-in is required for zero-hold servoJ"
        )
    bridge = UR5eRTDEBridge(robot_ip=robot_ip, frequency=frequency, limits=limits)
    timing = TimingTracker(frequency, overrun_threshold_s=max_deadline_ms / 1000.0)
    deadline_limit_ns = int(round(max_deadline_ms * 1e6))
    vis = _start_visualizer(publish_ros_topics, ros_prefix)
    samples: list[dict[str, Any]] = []
    exceptions: list[dict[str, Any]] = []
    robot_connected = False
    motion_attempted = False
    stop_errors: list[str] = []
    state_guard = UR5eStateGuard(bridge.limits)
    command_guard = MotionCommandGuard(bridge.limits)
    reason = ""
    ok = False
    hold_q: np.ndarray | None = None
    prev_error_q: np.ndarray | None = None
    prev_start_ns: int | None = None
    try:
        bridge.connect_receive_only()
        bridge.connect_control()
        robot_connected = True
        motion_attempted = True
        initial = bridge.read_state()
        if initial.tcp_pose is None:
            raise RuntimeError("RTDE receive interface did not expose actual_TCP_pose")
        initial_decision = state_guard.check(
            q=initial.q,
            qd=initial.qd,
            tcp_pose=initial.tcp_pose,
            host_stamp_ns=initial.host_stamp_ns,
            robot_stamp_s=initial.robot_timestamp_s,
        )
        if not initial_decision.ok:
            raise RuntimeError(initial_decision.reason)
        hold_q = initial.q.copy()
        command_guard.check(hold_q, stamp_ns=initial.host_stamp_ns, period_s=1.0 / frequency)
        start_ns = monotonic_ns()
        deadline_ns = start_ns + timing.period_ns
        end_ns = start_ns + int(round(duration * 1e9))
        cycle_index = 0
        while monotonic_ns() < end_ns:
            now_ns = monotonic_ns()
            if now_ns < deadline_ns:
                time.sleep((deadline_ns - now_ns) / 1e9)
                continue
            cycle_start_ns = monotonic_ns()
            cycle_lateness_ns = max(0, cycle_start_ns - deadline_ns)
            if cycle_lateness_ns > deadline_limit_ns:
                reason = f"deadline miss {cycle_lateness_ns / 1e6:.3f} ms > {max_deadline_ms:.3f} ms"
                break
            state = bridge.read_state()
            if state.tcp_pose is None:
                reason = "RTDE receive interface did not expose actual_TCP_pose"
                break
            state_decision = state_guard.check(
                q=state.q,
                qd=state.qd,
                tcp_pose=state.tcp_pose,
                host_stamp_ns=state.host_stamp_ns,
                robot_stamp_s=state.robot_timestamp_s,
            )
            if not state_decision.ok:
                reason = state_decision.reason
                break
            assert hold_q is not None
            cmd_q = hold_q.copy()
            cmd_decision = command_guard.check(
                cmd_q,
                stamp_ns=cycle_start_ns,
                period_s=timing.period_s,
            )
            if not cmd_decision.ok:
                reason = cmd_decision.reason
                break
            period_token = bridge.begin_period()
            bridge._call_servoj(
                cmd_q,
                gain=gain,
                lookahead_time=lookahead_time,
                velocity=velocity,
                acceleration=acceleration,
                period_s=timing.period_s,
            )
            bridge.wait_period(period_token)
            post_state = bridge.read_state()
            if post_state.tcp_pose is None:
                reason = "RTDE receive interface did not expose actual_TCP_pose"
                break
            error_q = post_state.q - cmd_q
            if prev_error_q is not None:
                _ = float(np.max(np.abs(error_q - prev_error_q)))
            cycle_end_ns = monotonic_ns()
            if cycle_end_ns - cycle_start_ns > deadline_limit_ns:
                reason = f"cycle duration {(cycle_end_ns - cycle_start_ns) / 1e6:.3f} ms > {max_deadline_ms:.3f} ms"
                break
            timing.add_sample(
                cycle_index=cycle_index,
                start_ns=cycle_start_ns,
                deadline_ns=deadline_ns,
                end_ns=cycle_end_ns,
                sleep_ns=0,
                interval_ns=None if prev_start_ns is None else cycle_start_ns - prev_start_ns,
            )
            samples.append(
                _sample_row(
                    phase=stage,
                    cycle_index=cycle_index,
                    state=post_state,
                    cmd_q=cmd_q,
                    cmd_qd=np.zeros(6, dtype=np.float64),
                    timing=timing.samples[-1].__dict__,
                    error_q=error_q,
                    extra={
                        "gain": float(gain),
                        "lookahead_time": float(lookahead_time),
                        "velocity": float(velocity),
                        "acceleration": float(acceleration),
                        "command_ok": True,
                    },
                )
            )
            _submit_ros_sample(vis, state=post_state, cmd_q=cmd_q, cmd_qd=np.zeros(6, dtype=np.float64))
            prev_error_q = error_q
            prev_start_ns = cycle_start_ns
            cycle_index += 1
            deadline_ns += timing.period_ns
        else:
            ok = True
            reason = "completed zero-hold servoJ"
    except Exception as exc:
        exceptions.append(_summarize_exception(exc))
        reason = f"{type(exc).__name__}: {exc}"
    finally:
        stop_errors.extend(bridge.safe_stop("zero-hold exit"))
        if vis is not None:
            vis.stop()
    result = _finalize(
        stage=stage,
        ok=ok,
        reason=reason,
        bridge=bridge,
        timing=timing,
        samples=samples,
        exceptions=exceptions,
        robot_connected=robot_connected,
        motion_attempted=motion_attempted,
        stop_errors=stop_errors,
        output_path=output,
        extra={"hold_q": hold_q},
    )
    _print_summary(stage, result)
    return result


def run_servoj_tiny_motion(
    *,
    robot_ip: str,
    frequency: float,
    duration: float,
    joint_index: int,
    amplitude_rad: float,
    max_amplitude_rad: float,
    gain: float,
    lookahead_time: float,
    velocity: float,
    acceleration: float,
    max_deadline_ms: float,
    motion_opt_in: bool,
    output: str | Path,
    publish_ros_topics: bool = False,
    ros_prefix: str = "/ur5e",
    limits: UR5eSafetyLimits | None = None,
) -> StageResult:
    stage = "servoj-tiny-motion"
    if not motion_opt_in:
        raise SystemExit(
            "--i-understand-this-moves-the-robot motion opt-in is required for tiny-motion servoJ"
        )
    if amplitude_rad > max_amplitude_rad + 1e-12:
        raise SystemExit(
            f"amplitude_rad {amplitude_rad} exceeds max-amplitude-rad {max_amplitude_rad}"
        )
    if joint_index < 0 or joint_index > 5:
        raise SystemExit("joint-index must be between 0 and 5")

    bridge = UR5eRTDEBridge(robot_ip=robot_ip, frequency=frequency, limits=limits)
    timing = TimingTracker(frequency, overrun_threshold_s=max_deadline_ms / 1000.0)
    deadline_limit_ns = int(round(max_deadline_ms * 1e6))
    vis = _start_visualizer(publish_ros_topics, ros_prefix)
    samples: list[dict[str, Any]] = []
    exceptions: list[dict[str, Any]] = []
    robot_connected = False
    motion_attempted = False
    stop_errors: list[str] = []
    state_guard = UR5eStateGuard(bridge.limits)
    command_guard = MotionCommandGuard(bridge.limits)
    reason = ""
    ok = False
    hold_q: np.ndarray | None = None
    prev_cmd_q: np.ndarray | None = None
    prev_start_ns: int | None = None
    try:
        bridge.connect_receive_only()
        bridge.connect_control()
        robot_connected = True
        motion_attempted = True
        initial = bridge.read_state()
        if initial.tcp_pose is None:
            raise RuntimeError("RTDE receive interface did not expose actual_TCP_pose")
        initial_decision = state_guard.check(
            q=initial.q,
            qd=initial.qd,
            tcp_pose=initial.tcp_pose,
            host_stamp_ns=initial.host_stamp_ns,
            robot_stamp_s=initial.robot_timestamp_s,
        )
        if not initial_decision.ok:
            raise RuntimeError(initial_decision.reason)
        hold_q = initial.q.copy()
        start_ns = monotonic_ns()
        deadline_ns = start_ns + timing.period_ns
        end_ns = start_ns + int(round(duration * 1e9))
        cycle_index = 0
        while monotonic_ns() < end_ns:
            now_ns = monotonic_ns()
            if now_ns < deadline_ns:
                time.sleep((deadline_ns - now_ns) / 1e9)
                continue
            cycle_start_ns = monotonic_ns()
            cycle_lateness_ns = max(0, cycle_start_ns - deadline_ns)
            if cycle_lateness_ns > deadline_limit_ns:
                reason = f"deadline miss {cycle_lateness_ns / 1e6:.3f} ms > {max_deadline_ms:.3f} ms"
                break
            state = bridge.read_state()
            if state.tcp_pose is None:
                reason = "RTDE receive interface did not expose actual_TCP_pose"
                break
            state_decision = state_guard.check(
                q=state.q,
                qd=state.qd,
                tcp_pose=state.tcp_pose,
                host_stamp_ns=state.host_stamp_ns,
                robot_stamp_s=state.robot_timestamp_s,
            )
            if not state_decision.ok:
                reason = state_decision.reason
                break
            assert hold_q is not None
            phase = min(1.0, (cycle_start_ns - start_ns) / max(1.0, duration * 1e9))
            offset = amplitude_rad * math.sin(2.0 * math.pi * phase)
            cmd_q = hold_q.copy()
            cmd_q[joint_index] += offset
            cmd_decision = command_guard.check(
                cmd_q,
                stamp_ns=cycle_start_ns,
                period_s=timing.period_s,
            )
            if not cmd_decision.ok:
                reason = cmd_decision.reason
                break
            if prev_cmd_q is not None:
                delta_prev = cmd_q - prev_cmd_q
                if np.any(np.abs(delta_prev) > bridge.limits.max_command_jump_rad + 1e-12):
                    reason = "command jump exceeds limit"
                    break
            period_token = bridge.begin_period()
            bridge._call_servoj(
                cmd_q,
                gain=gain,
                lookahead_time=lookahead_time,
                velocity=velocity,
                acceleration=acceleration,
                period_s=timing.period_s,
            )
            bridge.wait_period(period_token)
            post_state = bridge.read_state()
            if post_state.tcp_pose is None:
                reason = "RTDE receive interface did not expose actual_TCP_pose"
                break
            error_q = post_state.q - cmd_q
            cycle_end_ns = monotonic_ns()
            if cycle_end_ns - cycle_start_ns > deadline_limit_ns:
                reason = f"cycle duration {(cycle_end_ns - cycle_start_ns) / 1e6:.3f} ms > {max_deadline_ms:.3f} ms"
                break
            timing.add_sample(
                cycle_index=cycle_index,
                start_ns=cycle_start_ns,
                deadline_ns=deadline_ns,
                end_ns=cycle_end_ns,
                sleep_ns=0,
                interval_ns=None if prev_start_ns is None else cycle_start_ns - prev_start_ns,
            )
            samples.append(
                _sample_row(
                    phase=stage,
                    cycle_index=cycle_index,
                    state=post_state,
                    cmd_q=cmd_q,
                    cmd_qd=np.zeros(6, dtype=np.float64),
                    timing=timing.samples[-1].__dict__,
                    error_q=error_q,
                    extra={
                        "gain": float(gain),
                        "lookahead_time": float(lookahead_time),
                        "velocity": float(velocity),
                        "acceleration": float(acceleration),
                        "joint_index": int(joint_index),
                        "amplitude_rad": float(amplitude_rad),
                        "command_ok": True,
                    },
                )
            )
            _submit_ros_sample(vis, state=post_state, cmd_q=cmd_q, cmd_qd=np.zeros(6, dtype=np.float64))
            prev_cmd_q = cmd_q
            prev_start_ns = cycle_start_ns
            cycle_index += 1
            deadline_ns += timing.period_ns
        else:
            ok = True
            reason = "completed tiny servoJ motion"
    except Exception as exc:
        exceptions.append(_summarize_exception(exc))
        reason = f"{type(exc).__name__}: {exc}"
    finally:
        stop_errors.extend(bridge.safe_stop("tiny-motion exit"))
        if vis is not None:
            vis.stop()
    result = _finalize(
        stage=stage,
        ok=ok,
        reason=reason,
        bridge=bridge,
        timing=timing,
        samples=samples,
        exceptions=exceptions,
        robot_connected=robot_connected,
        motion_attempted=motion_attempted,
        stop_errors=stop_errors,
        output_path=output,
        extra={"hold_q": hold_q, "joint_index": joint_index, "amplitude_rad": amplitude_rad},
    )
    _print_summary(stage, result)
    return result


def run_direct_torque_probe(
    *,
    robot_ip: str,
    frequency: float,
    duration: float,
    max_torque_nm: float,
    zero_only: bool,
    understand_danger: bool,
    supervisor_present: bool,
    enable_nonzero_torque: bool,
    output: str | Path,
    limits: UR5eSafetyLimits | None = None,
) -> StageResult:
    stage = "direct-torque-probe"
    bridge = UR5eRTDEBridge(robot_ip=robot_ip, frequency=frequency, limits=limits)
    timing = TimingTracker(frequency)
    samples: list[dict[str, Any]] = []
    exceptions: list[dict[str, Any]] = []
    stop_errors: list[str] = []
    reason = ""
    ok = False
    if not zero_only:
        if not (understand_danger and supervisor_present and enable_nonzero_torque):
            raise SystemExit(
                "Nonzero direct torque requires --i-understand-direct-torque-is-dangerous, "
                "--i-am-with-trained-supervisor, and --enable-nonzero-torque"
            )
        if duration > 2.0:
            raise SystemExit("Nonzero direct torque probe must be <= 2 seconds")
        if max_torque_nm > 1.0:
            raise SystemExit("Nonzero direct torque probe must keep max-torque-nm <= 1.0")
    try:
        bridge.connect_receive_only()
        bridge.connect_control()
        state = bridge.read_state()
        if state.tcp_pose is None:
            raise RuntimeError("RTDE receive interface did not expose actual_TCP_pose")
        state_decision = bridge.validate_live_state(state)
        if not state_decision.ok:
            raise RuntimeError(state_decision.reason)
        capability = bridge.probe_direct_torque_capability()
        if not bridge.supports_direct_torque():
            reason = "direct torque is not supported by the available RTDE control interface"
            samples.append(
                {
                    "stage": stage,
                    "zero_only": bool(zero_only),
                    "status": "unsupported",
                    "robot_ip": robot_ip,
                    "supports_direct_torque": False,
                    "capability": capability,
                    "state": state.as_dict(),
                }
            )
        else:
            reason = (
                "direct torque support detected but guarded nonzero execution is not "
                "enabled in this patch"
            )
            samples.append(
                {
                    "stage": stage,
                    "zero_only": bool(zero_only),
                    "status": "blocked",
                    "robot_ip": robot_ip,
                    "supports_direct_torque": True,
                    "capability": capability,
                    "state": state.as_dict(),
                }
            )
        ok = False
    except Exception as exc:
        exceptions.append(_summarize_exception(exc))
        reason = f"{type(exc).__name__}: {exc}"
    finally:
        stop_errors.extend(bridge.safe_stop("direct-torque probe exit"))
    report = _build_report(
        stage=stage,
        bridge=bridge,
        timing=timing,
        samples=samples,
        exceptions=exceptions,
        robot_connected=bridge.receive is not None or bridge.control is not None,
        motion_attempted=not zero_only,
        stop_errors=stop_errors,
        status="PASS" if ok else "FAIL",
        reason=reason,
        extra={
            "zero_only": bool(zero_only),
            "max_torque_nm": float(max_torque_nm),
            "understand_danger": bool(understand_danger),
            "supervisor_present": bool(supervisor_present),
            "enable_nonzero_torque": bool(enable_nonzero_torque),
        },
    )
    write_json(output, report)
    result = StageResult(ok=ok, stage=stage, reason=reason, report=report)
    _print_summary(stage, result)
    return result
