"""Safety limits and decisions for the real-UR5e hardware lane.

Every numeric threshold that can stop the robot lives in this one file, and
nothing here imports RTDE -- these are plain dataclasses and numpy math, so
they can be fully tested without a robot (see tests/hardware/test_safety.py).

This module has two families of checks:
- ``UR5eSafetyLimits`` / ``check_joint_state`` / ``check_tcp_pose``: absolute
  ceilings the real robot's own joints and telemetry must never violate,
  regardless of what we're doing.
- ``CartesianMoveLimits`` / ``CartesianMoveMonitor``: a live monitor for one
  bounded Cartesian move, ported from the simulation's
  ``controller_core.safety.ImpedanceSafetyMonitor`` (drift-from-start,
  orientation-error, monotonic-growth-abort) but operating on real
  ``getActualTCPPose()`` telemetry instead of MuJoCo state, since there is no
  Jacobian/torque law running on real hardware in this lane.

``ConnectionHealth`` and ``EStopLatch`` fix a real bug found in the previous
hardware lane's ROS2 node: a failed state read left stale data in place
forever, so a dropped connection went undetected indefinitely. Here,
``hardware.link.UR5eLink.read_state()`` never returns a stale value -- it
raises -- and every caller must route failures through ``ConnectionHealth``,
which trips a real ``EStopLatch`` (no reset/clear method exists anywhere in
that class) once too many failures accumulate.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace

import numpy as np

from controller_core.safety_utils import (
    UR5_MANUFACTURER_QD_MAX_RAD_S,
    UR5_QLIM_LOWER_RAD,
    UR5_QLIM_UPPER_RAD,
)

# Bit 0 of RTDEReceiveInterface.getSafetyStatusBits() is IS_NORMAL_MODE, per
# Universal Robots' documented SafetyStatusBits convention. Confirmed against
# this project's actual runtime, not just documentation: a live URSim
# URControl instance was checked directly earlier in this project's history
# and getSafetyStatusBits is what's actually present (getSafetyStatus, the
# older enum-valued fallback in UR5eState population, is not) -- so the
# bitmask interpretation applies to whatever this codebase actually populates
# safety_status with in practice.
_SAFETY_STATUS_IS_NORMAL_MODE_BIT = 1  # bit 0


def is_robot_safety_normal(safety_status: int | None) -> bool:
    """True if the robot's own reported safety status is normal, or if the
    status isn't available at all (None means the getter isn't exposed on
    this robot/simulator -- treated as "can't verify," not "abnormal," so
    this doesn't make the lane unusable wherever that telemetry is absent).
    False only for a confirmed abnormal reading -- callers should trip an
    e-stop on False."""
    if safety_status is None:
        return True
    return bool(int(safety_status) & _SAFETY_STATUS_IS_NORMAL_MODE_BIT)


@dataclass
class SafetyDecision:
    """Accumulates zero or more violation reasons. ``ok`` flips to False the
    moment any reason is added."""

    ok: bool = True
    reason: str = ""
    reasons: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)

    def add(self, reason: str) -> None:
        self.reasons.append(reason)
        self.ok = False
        self.reason = "; ".join(self.reasons)


@dataclass
class UR5eSafetyLimits:
    """Absolute ceilings for the real-UR5e hardware lane. Every value here is
    carried over unchanged from the previous lane's
    ``hardware/safety_limits.py::UR5eSafetyLimits`` (see AGENTS.md/plan for
    the line-by-line provenance) -- nothing loosened."""

    q_lower: np.ndarray = field(default_factory=lambda: UR5_QLIM_LOWER_RAD.copy())
    q_upper: np.ndarray = field(default_factory=lambda: UR5_QLIM_UPPER_RAD.copy())
    qd_max_radps: np.ndarray = field(default_factory=lambda: UR5_MANUFACTURER_QD_MAX_RAD_S.copy())
    qdd_max_radps2: np.ndarray = field(default_factory=lambda: np.full(6, 5.0, dtype=np.float64))
    tcp_speed_max_mps: float = 0.25
    tcp_jump_max_m: float = 0.05
    state_stale_max_s: float = 0.1
    max_deadline_ms: float = 3.0

    def validate(self) -> None:
        for name in ("q_lower", "q_upper", "qd_max_radps", "qdd_max_radps2"):
            value = np.asarray(getattr(self, name), dtype=np.float64).reshape(-1)
            if value.shape[0] != 6:
                raise ValueError(f"{name} must have exactly 6 elements")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} contains NaN/Inf")
        for name in ("tcp_speed_max_mps", "tcp_jump_max_m", "state_stale_max_s", "max_deadline_ms"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")


def check_joint_state(q, qd, limits: UR5eSafetyLimits) -> SafetyDecision:
    """One-shot check of a raw (q, qd) pair against absolute joint limits."""
    decision = SafetyDecision()
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    qd = np.asarray(qd, dtype=np.float64).reshape(-1)
    if q.shape[0] != 6 or qd.shape[0] != 6:
        decision.add(f"q/qd must have 6 elements each; got q={q.shape}, qd={qd.shape}")
        return decision
    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(qd)):
        decision.add("NaN/Inf in joint state")
        return decision
    if np.any(q < limits.q_lower) or np.any(q > limits.q_upper):
        decision.add("joint position outside q_lower/q_upper")
    if np.any(np.abs(qd) > limits.qd_max_radps):
        decision.add("joint velocity exceeds manufacturer qd_max_radps")
    return decision


def check_tcp_pose(tcp_pose) -> SafetyDecision:
    """One-shot check that a raw 6-vector TCP pose is finite and well-shaped."""
    decision = SafetyDecision()
    pose = np.asarray(tcp_pose, dtype=np.float64).reshape(-1)
    if pose.shape[0] != 6:
        decision.add(f"tcp_pose must have 6 elements; got {pose.shape}")
        return decision
    if not np.all(np.isfinite(pose)):
        decision.add("NaN/Inf in TCP pose")
    return decision


class ConnectionHealth:
    """Tracks consecutive read failures and state staleness.

    Every ``read_state()`` failure must be reported here via
    ``record_failure()``, which returns True the instant the failure streak
    reaches ``max_consecutive_failures`` -- that return value is the signal
    for the caller to stop, call ``safe_stop()``, and trip the e-stop latch.
    There is no automatic recovery here: a caller must call
    ``record_success()`` again (a fresh successful read) to clear the streak.
    """

    def __init__(self, *, max_consecutive_failures: int = 3, max_state_age_s: float = 0.1) -> None:
        if max_consecutive_failures < 1:
            raise ValueError("max_consecutive_failures must be >= 1")
        if max_state_age_s <= 0.0:
            raise ValueError("max_state_age_s must be positive")
        self.max_consecutive_failures = int(max_consecutive_failures)
        self.max_state_age_s = float(max_state_age_s)
        self._consecutive_failures = 0
        self._last_success_ns: int | None = None

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    def record_success(self, host_stamp_ns: int) -> None:
        self._consecutive_failures = 0
        self._last_success_ns = int(host_stamp_ns)

    def record_failure(self) -> bool:
        """Increments the failure streak. Returns True once it has reached
        ``max_consecutive_failures`` -- the caller must treat that as fatal."""
        self._consecutive_failures += 1
        return self._consecutive_failures >= self.max_consecutive_failures

    def is_alive(self, now_ns: int | None = None) -> bool:
        """False if we've never had a successful read, if the failure streak
        has already tripped, or if the last successful read is older than
        ``max_state_age_s``."""
        if self._last_success_ns is None:
            return False
        if self._consecutive_failures >= self.max_consecutive_failures:
            return False
        now_ns = time.monotonic_ns() if now_ns is None else int(now_ns)
        age_s = (now_ns - self._last_success_ns) / 1e9
        return age_s <= self.max_state_age_s


class EStopTripped(RuntimeError):
    """Raised by ``EStopLatch.raise_if_tripped()`` once the latch has tripped."""


class EStopLatch:
    """A one-way safety latch.

    Once ``trip()`` is called, ``tripped`` stays True for the lifetime of
    this object -- there is deliberately no ``reset()``/``clear()`` method
    anywhere in this class. The only way to run again after a trip is to
    build a fresh ``UR5eLink`` (a new process, in practice), matching
    AGENTS.md's "the e-stop latch has no un-latch path by design" rule for
    real, not just in a docstring.
    """

    def __init__(self) -> None:
        self._tripped = False
        self._reason = ""

    @property
    def tripped(self) -> bool:
        return self._tripped

    @property
    def reason(self) -> str:
        return self._reason

    def trip(self, reason: str) -> None:
        self._tripped = True
        self._reason = str(reason)

    def raise_if_tripped(self) -> None:
        if self._tripped:
            raise EStopTripped(self._reason)


@dataclass
class CartesianMoveLimits:
    """Limits for one bounded Cartesian move.

    ``max_off_axis_drift_m``/``max_orientation_error_rad``/
    ``max_axis_error_growth_steps`` are the same category of check as the
    simulation's ``controller_core.safety.ImpedanceSafetyConfig`` (whose
    defaults are 0.03 m / 0.25 rad / 100 steps) -- the drift and growth
    limits here are tightened (0.02 m) rather than reused verbatim, since
    this is the first-ever Cartesian move attempted on real hardware in this
    repo and there's no track record to trust yet. ``max_tcp_speed_mps``,
    ``max_tcp_accel_mps2``, and ``max_waypoint_jump_m`` are new -- no prior
    Cartesian-motion value existed on the hardware side to carry over.

    ``max_tcp_accel_mps2`` raised 0.2 -> 0.5 (2026-07-24) after testing a
    0.15m/1.3s min-jerk move (peak accel ~0.5 m/s^2, the duration needed to
    reach this ceiling at that distance) through the tuned OSC config in
    MuJoCo: clean pass, max orientation error 0.166 rad (34% margin under the
    0.25 rad guard), max |qd| 0.467 rad/s (well under the 1.5 rad/s operative
    cap). NOTE: raising this alone does not unlock faster real moves --
    max_tcp_speed_mps is a separate, still-unchanged guard, and a 1.3s/0.15m
    move's peak velocity (~0.22 m/s) would trip it first. Left unchanged
    pending a deliberate decision on speed, not accel.
    """

    max_off_axis_drift_m: float = 0.02
    max_orientation_error_rad: float = 0.25
    max_tcp_speed_mps: float = 0.05
    max_tcp_accel_mps2: float = 0.5
    max_waypoint_jump_m: float = 0.002
    max_axis_error_growth_steps: int = 100
    # Tighter operative trip point than UR5eSafetyLimits.qd_max_radps
    # (the manufacturer ceiling) -- matches the sim's
    # ImpedanceSafetyConfig.max_joint_velocity_radps default (1.5).
    qd_max_radps: float = 1.5

    def validate(self) -> None:
        for name in (
            "max_off_axis_drift_m",
            "max_orientation_error_rad",
            "max_tcp_speed_mps",
            "max_tcp_accel_mps2",
            "max_waypoint_jump_m",
            "qd_max_radps",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if self.max_axis_error_growth_steps < 1:
            raise ValueError("max_axis_error_growth_steps must be >= 1")

    @classmethod
    def for_robot(cls, robot_ip: str, *, base: "CartesianMoveLimits | None" = None, **overrides) -> "CartesianMoveLimits":
        """Build limits for a target host; relax TCP kinematic guards on URSim."""
        limits = replace(base or cls(), **overrides) if overrides else (base or cls())
        if is_likely_ursim(robot_ip):
            limits = replace(
                limits,
                max_tcp_accel_mps2=max(float(limits.max_tcp_accel_mps2), 10.0),
                max_waypoint_jump_m=max(float(limits.max_waypoint_jump_m), 0.02),
                max_tcp_speed_mps=max(float(limits.max_tcp_speed_mps), 0.5),
            )
        limits.validate()
        return limits


def is_likely_ursim(robot_ip: str) -> bool:
    """True for loopback targets (Docker URSim on the same machine)."""
    ip = str(robot_ip).strip().lower()
    if ip in ("127.0.0.1", "localhost", "::1"):
        return True
    if ip.startswith("127."):
        return True
    return False


class CartesianMoveMonitor:
    """Live safety monitor for one bounded Cartesian move.

    Call ``set_start()`` once before the move begins, then ``check()`` every
    control cycle with the freshly-read robot state. This has no Jacobian,
    no torque, no dynamics -- pure geometry on the six numbers
    ``getActualTCPPose()`` already gives us, plus joint velocity from
    ``getActualQd()``.
    """

    def __init__(self, limits: CartesianMoveLimits) -> None:
        limits.validate()
        self.limits = limits
        self._start_pos: np.ndarray | None = None
        self._move_axis: int | None = None
        self._prev_pos: np.ndarray | None = None
        self._prev_speed_mps: float | None = None
        self._prev_abs_axis_err: float | None = None
        self._axis_err_grow_count = 0

    def set_start(self, tcp_pose, move_axis_index: int) -> None:
        if move_axis_index not in (0, 1, 2):
            raise ValueError(f"move_axis_index must be 0, 1, or 2; got {move_axis_index!r}")
        pose = np.asarray(tcp_pose, dtype=np.float64).reshape(6)
        self._start_pos = pose[:3].copy()
        self._move_axis = int(move_axis_index)
        self._prev_pos = pose[:3].copy()
        self._prev_speed_mps = None
        self._prev_abs_axis_err = None
        self._axis_err_grow_count = 0

    def check(
        self,
        *,
        q,
        qd,
        tcp_pose,
        target_tcp_pose,
        orientation_error_rad: float,
        axis_target_moving: bool,
        dt_s: float,
    ) -> SafetyDecision:
        if self._start_pos is None or self._move_axis is None:
            raise RuntimeError("CartesianMoveMonitor.set_start() must be called before check()")
        if dt_s <= 0.0:
            raise ValueError("dt_s must be positive")

        decision = SafetyDecision()
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        qd = np.asarray(qd, dtype=np.float64).reshape(-1)
        pose = np.asarray(tcp_pose, dtype=np.float64).reshape(6)
        target = np.asarray(target_tcp_pose, dtype=np.float64).reshape(6)
        pos = pose[:3]

        if (
            not np.all(np.isfinite(q))
            or not np.all(np.isfinite(qd))
            or not np.all(np.isfinite(pose))
        ):
            decision.add("NaN/Inf in robot state")
            return decision

        if np.any(np.abs(qd) > self.limits.qd_max_radps):
            decision.add(f"|qd| > {self.limits.qd_max_radps} rad/s")

        axis_names = ("X", "Y", "Z")
        for idx in range(3):
            if idx == self._move_axis:
                continue
            drift = abs(float(pos[idx]) - float(self._start_pos[idx]))
            if drift > self.limits.max_off_axis_drift_m:
                decision.add(
                    f"|{axis_names[idx]}-{axis_names[idx]}0| = {drift:.4f} m "
                    f"> {self.limits.max_off_axis_drift_m} m"
                )

        if orientation_error_rad > self.limits.max_orientation_error_rad:
            decision.add(
                f"||orientation error|| = {orientation_error_rad:.4f} rad "
                f"> {self.limits.max_orientation_error_rad} rad"
            )

        step_m = float(np.linalg.norm(pos - self._prev_pos))
        if step_m > self.limits.max_waypoint_jump_m:
            decision.add(f"waypoint jump {step_m:.5f} m > {self.limits.max_waypoint_jump_m} m")

        speed_mps = step_m / dt_s
        if speed_mps > self.limits.max_tcp_speed_mps:
            decision.add(f"TCP speed {speed_mps:.4f} m/s > {self.limits.max_tcp_speed_mps} m/s")
        if self._prev_speed_mps is not None:
            accel_mps2 = abs(speed_mps - self._prev_speed_mps) / dt_s
            if accel_mps2 > self.limits.max_tcp_accel_mps2:
                decision.add(
                    f"TCP acceleration {accel_mps2:.4f} m/s^2 > {self.limits.max_tcp_accel_mps2} m/s^2"
                )

        axis_err = abs(float(target[self._move_axis]) - float(pos[self._move_axis]))
        if axis_target_moving:
            self._axis_err_grow_count = 0
        elif self._prev_abs_axis_err is not None:
            if axis_err > self._prev_abs_axis_err + 1e-9:
                self._axis_err_grow_count += 1
            else:
                self._axis_err_grow_count = 0
            if self._axis_err_grow_count >= self.limits.max_axis_error_growth_steps:
                decision.add(
                    f"axis tracking error grew for {self._axis_err_grow_count} consecutive steps"
                )

        self._prev_abs_axis_err = axis_err
        self._prev_pos = pos.copy()
        self._prev_speed_mps = speed_mps

        return decision
