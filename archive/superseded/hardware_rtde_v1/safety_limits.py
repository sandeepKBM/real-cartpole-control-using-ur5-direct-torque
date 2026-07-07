"""Safety gates for the staged UR5e hardware bring-up scripts."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from controller_core.safety_utils import UR5_MANUFACTURER_QD_MAX_RAD_S


def _as_vector(value: Any, name: str, length: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.shape[0] != length:
        raise ValueError(f"{name} must have length {length}; got shape {np.asarray(value).shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN/Inf")
    return arr


@dataclass
class UR5eSafetyLimits:
    """Conservative limits for the first UR5e hardware checks."""

    q_lower: np.ndarray = field(
        default_factory=lambda: np.full(6, -2.0 * np.pi, dtype=np.float64)
    )
    q_upper: np.ndarray = field(
        default_factory=lambda: np.full(6, 2.0 * np.pi, dtype=np.float64)
    )
    qd_max_radps: np.ndarray = field(
        default_factory=lambda: UR5_MANUFACTURER_QD_MAX_RAD_S.copy()
    )
    qdd_max_radps2: np.ndarray = field(
        default_factory=lambda: np.full(6, 5.0, dtype=np.float64)
    )
    tcp_speed_max_mps: float = 0.25
    tcp_jump_max_m: float = 0.05
    max_command_jump_rad: float = 0.01
    max_command_velocity_radps: float = 0.5
    max_command_acceleration_radps2: float = 5.0
    state_stale_max_s: float = 0.1
    max_deadline_ms: float = 3.0

    def validate(self) -> None:
        for name in ("q_lower", "q_upper", "qd_max_radps", "qdd_max_radps2"):
            value = np.asarray(getattr(self, name), dtype=np.float64).reshape(-1)
            if value.shape[0] != 6:
                raise ValueError(f"{name} must have length 6")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite")
        if not math.isfinite(self.tcp_speed_max_mps) or self.tcp_speed_max_mps <= 0.0:
            raise ValueError("tcp_speed_max_mps must be positive and finite")
        if not math.isfinite(self.tcp_jump_max_m) or self.tcp_jump_max_m <= 0.0:
            raise ValueError("tcp_jump_max_m must be positive and finite")
        if not math.isfinite(self.max_command_jump_rad) or self.max_command_jump_rad < 0.0:
            raise ValueError("max_command_jump_rad must be finite and non-negative")
        if not math.isfinite(self.max_command_velocity_radps) or self.max_command_velocity_radps <= 0.0:
            raise ValueError("max_command_velocity_radps must be positive and finite")
        if not math.isfinite(self.max_command_acceleration_radps2) or self.max_command_acceleration_radps2 <= 0.0:
            raise ValueError("max_command_acceleration_radps2 must be positive and finite")
        if not math.isfinite(self.state_stale_max_s) or self.state_stale_max_s <= 0.0:
            raise ValueError("state_stale_max_s must be positive and finite")
        if not math.isfinite(self.max_deadline_ms) or self.max_deadline_ms <= 0.0:
            raise ValueError("max_deadline_ms must be positive and finite")


@dataclass
class SafetyDecision:
    ok: bool
    reason: str = ""
    reasons: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def add(self, reason: str) -> None:
        self.reasons.append(reason)
        self.ok = False
        self.reason = "; ".join(self.reasons)


def check_finite_array(name: str, value: Any, length: int) -> np.ndarray:
    return _as_vector(value, name, length)


def check_joint_state(
    q: Any,
    qd: Any,
    limits: UR5eSafetyLimits | None = None,
    *,
    host_stamp_ns: int | None = None,
    robot_stamp_s: float | None = None,
    prev_host_stamp_ns: int | None = None,
    prev_robot_stamp_s: float | None = None,
) -> SafetyDecision:
    limits = limits if limits is not None else UR5eSafetyLimits()
    limits.validate()
    q_arr = _as_vector(q, "q", 6)
    qd_arr = _as_vector(qd, "qd", 6)

    decision = SafetyDecision(ok=True)
    if np.any(q_arr < limits.q_lower) or np.any(q_arr > limits.q_upper):
        decision.add("joint limit violated")
    if np.any(np.abs(qd_arr) > limits.qd_max_radps + 1e-9):
        decision.add(f"|qd| exceeds {limits.qd_max_radps.tolist()} rad/s")

    if host_stamp_ns is not None and prev_host_stamp_ns is not None:
        dt_s = max(0.0, float(host_stamp_ns - prev_host_stamp_ns) / 1e9)
        if dt_s > limits.state_stale_max_s:
            decision.add(f"host state gap {dt_s:.3f}s exceeds stale threshold")
    if robot_stamp_s is not None and prev_robot_stamp_s is not None:
        if float(robot_stamp_s) <= float(prev_robot_stamp_s) + 1e-12:
            decision.add("robot timestamp did not advance")
    return decision


def check_tcp_pose(pose: Any, limits: UR5eSafetyLimits | None = None) -> SafetyDecision:
    limits = limits if limits is not None else UR5eSafetyLimits()
    limits.validate()
    pose_arr = _as_vector(pose, "tcp_pose", 6)
    decision = SafetyDecision(ok=True)
    if np.any(~np.isfinite(pose_arr)):
        decision.add("NaN/Inf in tcp pose")
    if abs(float(pose_arr[2])) > 5.0:
        decision.add("tcp z is implausible for a UR5e")
    if np.linalg.norm(pose_arr[:3]) > 10.0:
        decision.add("tcp translation is implausible")
    if np.linalg.norm(pose_arr[3:]) > np.pi * 4.0:
        decision.add("tcp orientation vector is implausible")
    return decision


class UR5eStateGuard:
    """State freshness and kinematics guard for receive-only / motion stages."""

    def __init__(self, limits: UR5eSafetyLimits | None = None) -> None:
        self.limits = limits if limits is not None else UR5eSafetyLimits()
        self.limits.validate()
        self._prev_q: np.ndarray | None = None
        self._prev_qd: np.ndarray | None = None
        self._prev_host_stamp_ns: int | None = None
        self._prev_robot_stamp_s: float | None = None

    def reset(self) -> None:
        self._prev_q = None
        self._prev_qd = None
        self._prev_host_stamp_ns = None
        self._prev_robot_stamp_s = None

    def check(
        self,
        *,
        q: Any,
        qd: Any,
        tcp_pose: Any | None = None,
        host_stamp_ns: int | None = None,
        robot_stamp_s: float | None = None,
    ) -> SafetyDecision:
        q_arr = _as_vector(q, "q", 6)
        qd_arr = _as_vector(qd, "qd", 6)
        decision = check_joint_state(
            q_arr,
            qd_arr,
            self.limits,
            host_stamp_ns=host_stamp_ns,
            robot_stamp_s=robot_stamp_s,
            prev_host_stamp_ns=self._prev_host_stamp_ns,
            prev_robot_stamp_s=self._prev_robot_stamp_s,
        )
        if tcp_pose is not None:
            tcp_decision = check_tcp_pose(tcp_pose, self.limits)
            if not tcp_decision.ok:
                for reason in tcp_decision.reasons:
                    decision.add(reason)

        if self._prev_q is not None and host_stamp_ns is not None and self._prev_host_stamp_ns is not None:
            dt_s = max(1e-6, float(host_stamp_ns - self._prev_host_stamp_ns) / 1e9)
            qd_est = (q_arr - self._prev_q) / dt_s
            if np.any(np.abs(qd_est) > self.limits.qd_max_radps + 1e-9):
                decision.add("estimated joint velocity exceeds limit")
            if self._prev_qd is not None:
                qdd_est = (qd_arr - self._prev_qd) / dt_s
                if np.any(np.abs(qdd_est) > self.limits.qdd_max_radps2 + 1e-9):
                    decision.add("estimated joint acceleration exceeds limit")

        self._prev_q = q_arr.copy()
        self._prev_qd = qd_arr.copy()
        if host_stamp_ns is not None:
            self._prev_host_stamp_ns = int(host_stamp_ns)
        if robot_stamp_s is not None:
            self._prev_robot_stamp_s = float(robot_stamp_s)
        return decision


class MotionCommandGuard:
    """Refuses discontinuous joint commands and implausible command rates."""

    def __init__(self, limits: UR5eSafetyLimits | None = None) -> None:
        self.limits = limits if limits is not None else UR5eSafetyLimits()
        self.limits.validate()
        self._prev_cmd: np.ndarray | None = None
        self._prev_vel: np.ndarray | None = None
        self._prev_stamp_ns: int | None = None

    def reset(self) -> None:
        self._prev_cmd = None
        self._prev_vel = None
        self._prev_stamp_ns = None

    def check(self, cmd_q: Any, *, stamp_ns: int | None = None, period_s: float | None = None) -> SafetyDecision:
        cmd_arr = _as_vector(cmd_q, "cmd_q", 6)
        decision = SafetyDecision(ok=True)
        if self._prev_cmd is None:
            self._prev_cmd = cmd_arr.copy()
            self._prev_stamp_ns = None if stamp_ns is None else int(stamp_ns)
            return decision

        if period_s is None:
            if stamp_ns is not None and self._prev_stamp_ns is not None:
                period_s = max(1e-6, float(stamp_ns - self._prev_stamp_ns) / 1e9)
            else:
                period_s = 1.0 / 500.0
        else:
            period_s = max(1e-6, float(period_s))

        delta = cmd_arr - self._prev_cmd
        if np.any(np.abs(delta) > self.limits.max_command_jump_rad + 1e-12):
            decision.add("command jump exceeds limit")
        cmd_vel = delta / period_s
        if np.any(np.abs(cmd_vel) > self.limits.max_command_velocity_radps + 1e-9):
            decision.add("command velocity exceeds limit")
        if self._prev_vel is not None:
            cmd_acc = (cmd_vel - self._prev_vel) / period_s
            if np.any(np.abs(cmd_acc) > self.limits.max_command_acceleration_radps2 + 1e-9):
                decision.add("command acceleration exceeds limit")

        self._prev_cmd = cmd_arr.copy()
        self._prev_vel = cmd_vel.copy()
        if stamp_ns is not None:
            self._prev_stamp_ns = int(stamp_ns)
        return decision
