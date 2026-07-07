"""Shared UR5e connection / actuator session helpers.

The goal is to centralize the safe-by-default hardware session logic for the
three staged experiments we care about:

1. connection smoke
2. bounded actuator motion
3. direct torque probing / future direct torque control

This module deliberately stays free of ROS imports so it can be reused by the
standalone hardware scripts and by a thin ROS wrapper later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .safety_limits import MotionCommandGuard, UR5eSafetyLimits, UR5eStateGuard
from .ur5e_rtde_bridge import UR5eRTDEBridge, UR5eState


def _as_finite_vector(name: str, value: Any, length: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.shape[0] != length:
        raise ValueError(f"{name} must have length {length}; got shape {np.asarray(value).shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN/Inf")
    return arr


@dataclass
class UR5eHardwareSessionConfig:
    """Safe-by-default hardware session configuration."""

    robot_ip: str
    frequency_hz: float
    motion_opt_in: bool = False
    allow_nonzero_direct_torque: bool = False
    direct_torque_zero_only: bool = True

    def validate(self) -> None:
        if float(self.frequency_hz) <= 0.0 or not np.isfinite(float(self.frequency_hz)):
            raise ValueError("frequency_hz must be positive and finite")
        if not self.robot_ip:
            raise ValueError("robot_ip is required")


@dataclass
class UR5eCommandResult:
    """Structured result of a requested actuator action."""

    kind: str
    accepted: bool
    blocked: bool
    reason: str
    requested: dict[str, Any] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "accepted": bool(self.accepted),
            "blocked": bool(self.blocked),
            "reason": self.reason,
            "requested": self.requested,
            "details": self.details,
        }


@dataclass
class UR5eConnectionSnapshot:
    """One-shot diagnostic snapshot of the current connection state."""

    bridge: dict[str, Any]
    state: dict[str, Any] | None = None
    state_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        out = {"bridge": self.bridge}
        if self.state is not None:
            out["state"] = self.state
        if self.state_reason is not None:
            out["state_reason"] = self.state_reason
        return out


class UR5eHardwareSession:
    """Reusable receive/control session for staged UR5e hardware work."""

    def __init__(
        self,
        config: UR5eHardwareSessionConfig,
        *,
        limits: UR5eSafetyLimits | None = None,
        control_factory: Callable[..., Any] | None = None,
        receive_factory: Callable[..., Any] | None = None,
    ) -> None:
        config.validate()
        self.cfg = config
        self.bridge = UR5eRTDEBridge(
            robot_ip=config.robot_ip,
            frequency=float(config.frequency_hz),
            limits=limits,
            control_factory=control_factory,
            receive_factory=receive_factory,
        )
        self._state_guard = UR5eStateGuard(self.bridge.limits)
        self._command_guard = MotionCommandGuard(self.bridge.limits)
        self._last_state: UR5eState | None = None

    def connect_receive_only(self) -> Any:
        return self.bridge.connect_receive_only()

    def connect_control(self) -> Any:
        return self.bridge.connect_control()

    def safe_stop(self, reason: str) -> list[str]:
        return self.bridge.safe_stop(reason)

    def read_state(self) -> UR5eState:
        state = self.bridge.read_state()
        self._last_state = state
        return state

    def capture_snapshot(self, *, connect_control: bool = False, include_state: bool = True) -> UR5eConnectionSnapshot:
        self.connect_receive_only()
        if connect_control:
            self.connect_control()
        state_dict: dict[str, Any] | None = None
        state_reason: str | None = None
        if include_state:
            try:
                state = self.read_state()
                state_decision = self._state_guard.check(
                    q=state.q,
                    qd=state.qd,
                    tcp_pose=state.tcp_pose,
                    host_stamp_ns=state.host_stamp_ns,
                    robot_stamp_s=state.robot_timestamp_s,
                )
                state_dict = {
                    **state.as_dict(),
                    "state_ok": bool(state_decision.ok),
                    "state_reason": state_decision.reason,
                }
            except Exception as exc:
                state_reason = f"{type(exc).__name__}: {exc}"
        return UR5eConnectionSnapshot(
            bridge=self.bridge.report(),
            state=state_dict,
            state_reason=state_reason,
        )

    def request_servoj_hold(
        self,
        q_ref: Any,
        *,
        gain: float = 100.0,
        lookahead_time: float = 0.1,
        velocity: float = 0.05,
        acceleration: float = 0.05,
        period_s: float | None = None,
    ) -> UR5eCommandResult:
        q_ref_arr = _as_finite_vector("q_ref", q_ref, 6)
        requested = {
            "q_ref": q_ref_arr.tolist(),
            "gain": float(gain),
            "lookahead_time": float(lookahead_time),
            "velocity": float(velocity),
            "acceleration": float(acceleration),
            "period_s": None if period_s is None else float(period_s),
        }
        if not self.cfg.motion_opt_in:
            return UR5eCommandResult(
                kind="servoj_hold",
                accepted=False,
                blocked=True,
                reason="servoJ motion is blocked until motion_opt_in is enabled",
                requested=requested,
                details={"motion_opt_in": False},
            )
        command_decision = self._command_guard.check(
            q_ref_arr,
            period_s=period_s or (1.0 / max(float(self.cfg.frequency_hz), 1.0)),
        )
        if not command_decision.ok:
            return UR5eCommandResult(
                kind="servoj_hold",
                accepted=False,
                blocked=True,
                reason=command_decision.reason,
                requested=requested,
                details={"motion_opt_in": True, "command_guard": command_decision.__dict__},
            )
        self.connect_control()
        period_token = self.bridge.begin_period()
        try:
            self.bridge.send_servoj(
                q_ref_arr,
                gain=float(gain),
                lookahead_time=float(lookahead_time),
                velocity=float(velocity),
                acceleration=float(acceleration),
                period_s=period_s,
            )
        finally:
            self.bridge.wait_period(period_token)
        return UR5eCommandResult(
            kind="servoj_hold",
            accepted=True,
            blocked=False,
            reason="servoJ hold command issued",
            requested=requested,
            details={"motion_opt_in": True},
        )

    def request_servoj_tiny_motion(
        self,
        q_hold: Any,
        *,
        joint_index: int = 0,
        amplitude_rad: float = 0.005,
        phase: float = 0.0,
        gain: float = 100.0,
        lookahead_time: float = 0.1,
        velocity: float = 0.05,
        acceleration: float = 0.05,
        period_s: float | None = None,
        max_amplitude_rad: float = 0.01,
    ) -> UR5eCommandResult:
        q_hold_arr = _as_finite_vector("q_hold", q_hold, 6)
        requested = {
            "q_hold": q_hold_arr.tolist(),
            "joint_index": int(joint_index),
            "amplitude_rad": float(amplitude_rad),
            "phase": float(phase),
            "gain": float(gain),
            "lookahead_time": float(lookahead_time),
            "velocity": float(velocity),
            "acceleration": float(acceleration),
            "period_s": None if period_s is None else float(period_s),
        }
        if not self.cfg.motion_opt_in:
            return UR5eCommandResult(
                kind="servoj_tiny_motion",
                accepted=False,
                blocked=True,
                reason="servoJ motion is blocked until motion_opt_in is enabled",
                requested=requested,
                details={"motion_opt_in": False},
            )
        if joint_index < 0 or joint_index > 5:
            return UR5eCommandResult(
                kind="servoj_tiny_motion",
                accepted=False,
                blocked=True,
                reason="joint_index must be in [0, 5]",
                requested=requested,
                details={"motion_opt_in": True},
            )
        if abs(float(amplitude_rad)) > abs(float(max_amplitude_rad)) + 1e-12:
            return UR5eCommandResult(
                kind="servoj_tiny_motion",
                accepted=False,
                blocked=True,
                reason="requested amplitude exceeds max_amplitude_rad",
                requested=requested,
                details={"motion_opt_in": True},
            )
        cmd_q = q_hold_arr.copy()
        cmd_q[int(joint_index)] += float(amplitude_rad) * float(np.sin(float(phase)))
        command_decision = self._command_guard.check(
            cmd_q,
            period_s=period_s or (1.0 / max(float(self.cfg.frequency_hz), 1.0)),
        )
        if not command_decision.ok:
            return UR5eCommandResult(
                kind="servoj_tiny_motion",
                accepted=False,
                blocked=True,
                reason=command_decision.reason,
                requested=requested,
                details={
                    "motion_opt_in": True,
                    "cmd_q": cmd_q.tolist(),
                    "command_guard": command_decision.__dict__,
                },
            )
        self.connect_control()
        period_token = self.bridge.begin_period()
        try:
            self.bridge.send_servoj(
                cmd_q,
                gain=float(gain),
                lookahead_time=float(lookahead_time),
                velocity=float(velocity),
                acceleration=float(acceleration),
                period_s=period_s,
            )
        finally:
            self.bridge.wait_period(period_token)
        return UR5eCommandResult(
            kind="servoj_tiny_motion",
            accepted=True,
            blocked=False,
            reason="tiny servoJ motion command issued",
            requested=requested,
            details={"cmd_q": cmd_q.tolist(), "motion_opt_in": True},
        )

    def request_direct_torque(
        self,
        tau_nm: Any,
        *,
        zero_only: bool | None = None,
        allow_nonzero: bool | None = None,
    ) -> UR5eCommandResult:
        tau_arr = _as_finite_vector("tau_nm", tau_nm, 6)
        zero_only_eff = self.cfg.direct_torque_zero_only if zero_only is None else bool(zero_only)
        allow_nonzero_eff = (
            self.cfg.allow_nonzero_direct_torque if allow_nonzero is None else bool(allow_nonzero)
        )
        requested = {
            "tau_nm": tau_arr.tolist(),
            "zero_only": bool(zero_only_eff),
            "allow_nonzero": bool(allow_nonzero_eff),
        }
        self.connect_control()
        backend = self.bridge.probe_direct_torque_capability()
        details = {
            "motion_opt_in": bool(self.cfg.motion_opt_in),
            "bridge": backend,
        }
        if np.any(np.abs(tau_arr) > 1e-12) and zero_only_eff:
            return UR5eCommandResult(
                kind="direct_torque",
                accepted=False,
                blocked=True,
                reason="nonzero direct torque is blocked by default",
                requested=requested,
                details=details,
            )
        if np.any(np.abs(tau_arr) > 1e-12) and not allow_nonzero_eff:
            return UR5eCommandResult(
                kind="direct_torque",
                accepted=False,
                blocked=True,
                reason="explicit nonzero direct torque opt-in is disabled",
                requested=requested,
                details=details,
            )
        command_result = self.bridge.send_joint_torque(
            tau_arr,
            allow_nonzero=allow_nonzero_eff,
            zero_only=zero_only_eff,
        )
        details["command_result"] = command_result
        return UR5eCommandResult(
            kind="direct_torque",
            accepted=bool(command_result.get("accepted", False)),
            blocked=bool(command_result.get("blocked", True)),
            reason=str(command_result.get("reason", "direct torque probe recorded")),
            requested=requested,
            details=details,
        )
