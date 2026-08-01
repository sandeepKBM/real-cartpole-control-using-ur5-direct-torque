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
from collections import deque
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
    # Period-relative cap layered on top of max_deadline_ms (2026-07-29, see
    # docs/status/deadline_monitor_period_relative_fix_2026-07-29.md). The flat
    # 3.0 ms max_deadline_ms default is fine for the three 125 Hz loops (8 ms
    # period -- 3.0 ms is a real fraction of it) but tolerates up to ~250% of
    # a 500 Hz/2 ms loop's own period before DeadlineMonitor even starts
    # counting an overrun (test_deadline_monitor_ignores_clean_cycles asserts
    # this is intentional at 3.0 ms in isolation) -- too loose for that loop's
    # own budget, and a real ~2 ms overrun on real hardware slipped under it
    # undetected. Only hardware/direct_torque_transport.py's 500 Hz loop
    # currently applies this via min(max_deadline_ms, max_deadline_fraction_of_period
    # * dt_s * 1000.0); the other three loops are unaffected.
    max_deadline_fraction_of_period: float = 0.5

    def validate(self) -> None:
        for name in ("q_lower", "q_upper", "qd_max_radps", "qdd_max_radps2"):
            value = np.asarray(getattr(self, name), dtype=np.float64).reshape(-1)
            if value.shape[0] != 6:
                raise ValueError(f"{name} must have exactly 6 elements")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} contains NaN/Inf")
        for name in (
            "tcp_speed_max_mps",
            "tcp_jump_max_m",
            "state_stale_max_s",
            "max_deadline_ms",
            "max_deadline_fraction_of_period",
        ):
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


class DeadlineMonitor:
    """Aborts a control loop that can't hold its real-time period.

    Each cycle reports how far it overran its control-period budget
    (nanoseconds *beyond* the period -- 0 if it finished on time; for a loop
    that schedules against an absolute deadline this is the start-lateness, for
    a relative-sleep loop it is ``work - period``). This enforces
    ``UR5eSafetyLimits.max_deadline_ms`` (3.0 ms), which was previously
    validated-but-never-checked: an overrun just ran late instead of aborting.

    A single transient overrun (an OS scheduling hiccup, a GC pause) on an
    otherwise-healthy loop must NOT abort a physical robot mid-motion -- a
    spurious e-stop is itself a hazard -- so a lone spike is tolerated. There
    are two independent trip conditions:

    * ``max_consecutive_overruns`` cycles *in a row* each overran the deadline
      by more than ``max_deadline_ms``. Consecutive (not cumulative) because a
      loop that overruns once, recovers, and then runs thousands of clean
      cycles is healthy; it is *sustained* lateness that means the control law
      is now acting on stale state and commanding torques timed to a clock it
      cannot meet. N defaults to 3, matching this module's existing
      ``ConnectionHealth.max_consecutive_failures`` convention for "how many
      bad cycles before fatal."
    * a *single* cycle overran by more than ``hard_overrun_multiple`` x
      ``max_deadline_ms``. A lateness that large in one cycle is not jitter but
      a stall -- the loop blocked for many control periods at once and the
      robot ran with no fresh command that whole window. Waiting for two more
      equally-bad cycles just prolongs an already-unsafe open-loop interval, so
      this trips immediately. 5x (15 ms at the 3 ms default) sits well above
      any plausible single-cycle scheduling jitter yet is a small fraction of a
      human-noticeable delay.

    ``record()`` returns an abort-reason string the first time either
    condition trips, else ``None``. The caller must treat a non-None return as
    fatal (safe_stop + trip the e-stop latch), exactly like a monitor
    ``SafetyDecision`` that isn't ``ok``.
    """

    def __init__(
        self,
        max_deadline_ms: float,
        *,
        max_consecutive_overruns: int = 3,
        hard_overrun_multiple: float = 5.0,
    ) -> None:
        max_deadline_ms = float(max_deadline_ms)
        if not np.isfinite(max_deadline_ms) or max_deadline_ms <= 0.0:
            raise ValueError("max_deadline_ms must be positive and finite")
        if max_consecutive_overruns < 1:
            raise ValueError("max_consecutive_overruns must be >= 1")
        if not np.isfinite(hard_overrun_multiple) or hard_overrun_multiple < 1.0:
            raise ValueError("hard_overrun_multiple must be >= 1.0")
        self.max_deadline_ms = max_deadline_ms
        self.max_consecutive_overruns = int(max_consecutive_overruns)
        self.hard_overrun_multiple = float(hard_overrun_multiple)
        self._max_deadline_ns = int(round(max_deadline_ms * 1e6))
        self._hard_overrun_ns = int(round(max_deadline_ms * hard_overrun_multiple * 1e6))
        self._consecutive_overruns = 0

    @property
    def consecutive_overruns(self) -> int:
        return self._consecutive_overruns

    def record(self, overrun_ns: int) -> str | None:
        overrun_ns = max(0, int(overrun_ns))
        overrun_ms = overrun_ns / 1e6
        # A single catastrophic overrun trips at once -- see class docstring.
        if overrun_ns >= self._hard_overrun_ns:
            self._consecutive_overruns += 1
            return (
                f"deadline_overrun: single cycle late by {overrun_ms:.2f} ms > "
                f"{self.hard_overrun_multiple:g}x max_deadline_ms "
                f"({self.max_deadline_ms:g} ms)"
            )
        if overrun_ns > self._max_deadline_ns:
            self._consecutive_overruns += 1
            if self._consecutive_overruns >= self.max_consecutive_overruns:
                return (
                    f"deadline_overrun: {self._consecutive_overruns} consecutive cycles "
                    f"late by > max_deadline_ms ({self.max_deadline_ms:g} ms); "
                    f"latest {overrun_ms:.2f} ms"
                )
        else:
            self._consecutive_overruns = 0
        return None


class StaleStateMonitor:
    """Detects a frozen-but-non-erroring RTDE stream during motion.

    ``ur_rtde``'s receive interface can return the *last buffered* value
    without raising when the underlying stream stalls, so ``read_state()``'s
    "never returns stale data" guarantee only covers the raise-on-exception
    case -- not a stream that keeps handing back the same frozen sample. This
    monitor compares the robot's own clock (``getTimestamp()`` ->
    ``UR5eState.robot_timestamp_s``) across cycles: if it stops advancing for
    more than ``max_frozen_cycles`` consecutive reads *while the host clock
    keeps advancing*, the stream has stalled and the loop must abort.

    Why not trip on a single repeated timestamp: when the control/monitor loop
    polls faster than the robot's RTDE publish rate, two consecutive reads can
    legitimately return the same robot timestamp (we read between robot
    updates). ``max_frozen_cycles`` defaults to 5 -- large enough that ordinary
    poll-faster-than-publish duplicates never trip it, small enough that a
    genuine stall is caught within a bounded window (~10 ms at 500 Hz, ~40 ms
    at 125 Hz). At the 125 Hz position/monitor loops the robot clock advances
    ~4x per read, so a duplicate never occurs normally and 5-in-a-row is
    unambiguously a stall.

    ``robot_timestamp_s=None`` (``getTimestamp`` not exposed on this
    robot/simulator) is treated as "can't verify," never as stale -- consistent
    with ``is_robot_safety_normal(None)``. ``record()`` returns an abort-reason
    string the first time it trips, else ``None``.
    """

    def __init__(self, *, max_frozen_cycles: int = 5) -> None:
        if max_frozen_cycles < 1:
            raise ValueError("max_frozen_cycles must be >= 1")
        self.max_frozen_cycles = int(max_frozen_cycles)
        self._prev_robot_ts: float | None = None
        self._prev_host_ns: int | None = None
        self._frozen_count = 0

    @property
    def frozen_count(self) -> int:
        return self._frozen_count

    def record(self, robot_timestamp_s: float | None, host_stamp_ns: int) -> str | None:
        host_ns = int(host_stamp_ns)
        if robot_timestamp_s is None:
            # Can't verify -- drop any prior baseline so a stream that starts
            # exposing a clock later begins counting cleanly from that point.
            self._prev_robot_ts = None
            self._prev_host_ns = host_ns
            self._frozen_count = 0
            return None
        ts = float(robot_timestamp_s)
        if self._prev_robot_ts is not None and self._prev_host_ns is not None:
            host_advanced = host_ns > self._prev_host_ns
            robot_advanced = ts > self._prev_robot_ts + 1e-9
            if host_advanced and not robot_advanced:
                self._frozen_count += 1
                if self._frozen_count >= self.max_frozen_cycles:
                    return (
                        f"stale_state: robot timestamp frozen at {ts:.6f} s for "
                        f"{self._frozen_count} consecutive cycles while host clock "
                        f"advanced -- RTDE stream stalled"
                    )
            else:
                self._frozen_count = 0
        self._prev_robot_ts = ts
        self._prev_host_ns = host_ns
        return None


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
    # Noise-robust acceleration estimation (added 2026-07-28, real-hardware
    # finding: tools/analyze_state_noise_capture.py measured the accel
    # estimate's own noise floor from a 10s stationary real-RTDE capture --
    # MEDIAN 1.74 m/s^2 at rest, already ~3.5x the 0.5 default -- because
    # accel = |delta speed|/dt is a double finite-difference of raw position,
    # which amplifies real sensor noise by ~1/dt^2. Defaults below (1, 1.0)
    # reproduce the exact old single-cycle, unfiltered behavior -- opt-in
    # only. See CartesianMoveMonitor for the mechanism.
    #   accel_gap_cycles: use position from N cycles back (not 1) to form
    #     each speed sample fed into the acceleration estimate. Real position
    #     noise doesn't grow with N (still ~sqrt(2)*sigma between any two
    #     independent samples), but the time gap in the denominator grows by
    #     N, so speed noise shrinks by ~1/N and the accel estimate (a
    #     difference of two such speed samples) shrinks by ~1/N^2.
    #   speed_lowpass_alpha: EMA smoothing applied to that gap-windowed speed
    #     before differencing for accel (1.0 = no filtering).
    accel_gap_cycles: int = 1
    speed_lowpass_alpha: float = 1.0

    # Noise-robust TCP SPEED-LIMIT estimation (added 2026-08-01, real-hardware
    # finding). Unlike the accel estimate above, the speed-limit check was
    # deliberately kept as a raw, single-cycle, unfiltered
    # step_m/real_dt_s -- see the comment at its use site in check() -- on the
    # grounds that single-differenced position noise is far smaller than the
    # double-differenced accel noise. That's still true at rest (measured
    # tonight from real traces: stationary/near-static p100 <= ~0.030 m/s,
    # consistent with the original documented 0.036 m/s capture). But a real
    # trip tonight (direct_torque_20260801_193232, split_base_wrist_task
    # config, accel_duration_scurve, target_accel=0.02) was root-caused by
    # pulling the raw per-cycle speed trend for the 140ms before the trip:
    # the underlying smoothed trend climbed genuinely and monotonically
    # (~0.020 -> ~0.051 m/s, tracking a real, separately-confirmed
    # orientation-error growth), while raw single-cycle noise on top of that
    # trend (residual std ~0.007-0.009 m/s during real motion, vs ~0.006 at
    # rest -- meaningfully noisier moving than stationary, unlike the
    # documented rest-only capture the original no-smoothing decision was
    # based on) meant the *exact* 3 consecutive cycles that tripped
    # speed_max_consecutive_violations were partly noise-timing-dependent
    # rather than a clean threshold crossing. The real average had genuinely
    # already crossed 0.05 m/s -- smoothing does not prevent this class of
    # trip -- but it removes noise-driven jitter in exactly *when* it trips,
    # the same rationale already applied to the accel estimate. Separate
    # field names from accel_gap_cycles/speed_lowpass_alpha above (not
    # reused) because those are documented as feeding ONLY the accel
    # estimate, not the speed-limit decision -- conflating the two would
    # silently change what an existing --noise-robust-guards run does.
    # Defaults (1, 1.0) reproduce the exact old raw single-cycle behavior --
    # opt-in only, not folded into NOISE_ROBUST_GUARD_OVERRIDES below (that
    # preset is already validated on real hardware as-is; extending it is a
    # separate, deliberate decision, not bundled silently here).
    speed_limit_gap_cycles: int = 1
    speed_limit_lowpass_alpha: float = 1.0

    # DeadlineMonitor-style graduated tolerance (2026-07-30). Real-hardware
    # evidence (docs/status/safety_envelope_backtest_2026-07-30.md,
    # experiments/safety-envelope-study branch): of 21 real guard trips
    # across this project's history, 15 (71%) were single-cycle TCP
    # speed/accel noise spikes on an otherwise-clean, bounded-qd move -- not
    # real divergence -- while the genuinely real divergences found were
    # multi-cycle, escalating trends (qd roughly doubling cycle over cycle).
    # A falsifiable backtest of two different smooth/state-conditioned
    # envelope redesigns against this same real data both failed for real
    # structural reasons (one missed a genuine catch despite a better
    # aggregate score; the other nuisance-tripped at this project's own
    # validated wrist_2=0 transport pose) -- see that doc. What DID hold up
    # in the same investigation is this project's own DeadlineMonitor,
    # which already solves the identical shape of problem (a real, isolated
    # transient overrun must not abort a physical robot mid-motion) with
    # two independent, rigid trip conditions instead of a smooth/continuous
    # bound: N consecutive over-threshold cycles trips (tolerates one noise
    # spike, still catches a sustained real trend fast since real
    # divergences escalate quickly), and a single cycle over
    # `hard_multiple` x the base threshold trips immediately regardless
    # (catches a genuine one-shot catastrophic event without ever waiting
    # for N more, and is proven by this repo's own real trip data --
    # direct_torque_20260728_162206's single-cycle deadline spike -- to
    # catch what a purely graduated/consecutive-only rule would miss by
    # design). Applying that exact pattern to the TCP speed/accel checks
    # instead of a lookalike-but-untested "funnel" is not a compromise --
    # it is a validated pattern already proven correct in this codebase.
    #
    # Defaults (1, 5.0) are a no-op: max_consecutive_violations=1 means
    # every check() call passes straight through the hard-multiple branch's
    # exact old message/behavior, reproducing today's single-cycle instant
    # trip exactly. Opt-in only, same convention as accel_gap_cycles/
    # speed_lowpass_alpha above -- enabling this for a real run is a
    # deliberate, separate decision, not a silent default change.
    accel_max_consecutive_violations: int = 1
    accel_hard_multiple: float = 5.0
    speed_max_consecutive_violations: int = 1
    speed_hard_multiple: float = 5.0

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
        if int(self.accel_gap_cycles) < 1:
            raise ValueError("accel_gap_cycles must be >= 1")
        if not (0.0 < float(self.speed_lowpass_alpha) <= 1.0):
            raise ValueError("speed_lowpass_alpha must be in (0.0, 1.0]")
        if int(self.speed_limit_gap_cycles) < 1:
            raise ValueError("speed_limit_gap_cycles must be >= 1")
        if not (0.0 < float(self.speed_limit_lowpass_alpha) <= 1.0):
            raise ValueError("speed_limit_lowpass_alpha must be in (0.0, 1.0]")
        if int(self.accel_max_consecutive_violations) < 1:
            raise ValueError("accel_max_consecutive_violations must be >= 1")
        if not np.isfinite(self.accel_hard_multiple) or self.accel_hard_multiple < 1.0:
            raise ValueError("accel_hard_multiple must be >= 1.0")
        if int(self.speed_max_consecutive_violations) < 1:
            raise ValueError("speed_max_consecutive_violations must be >= 1")
        if not np.isfinite(self.speed_hard_multiple) or self.speed_hard_multiple < 1.0:
            raise ValueError("speed_hard_multiple must be >= 1.0")

    @classmethod
    def from_impedance_safety_config(cls, safety_cfg) -> "CartesianMoveLimits":
        """Layer Cartesian speed/accel/waypoint-jump guards on top of an
        already-active ``ImpedanceSafetyConfig`` (duck-typed on its
        ``max_joint_velocity_radps``/``max_abs_y_drift_m``/``max_abs_z_drift_m``/
        ``max_orientation_error_rad`` attributes).

        The qd/drift/orientation trip points are reused from ``safety_cfg`` so
        this doesn't introduce a second, different trip point for a check that
        already exists; the genuinely new speed/accel/waypoint-jump ceilings come
        from this class's own defaults. Byte-identical to the construction that
        was duplicated in ``direct_torque_transport`` and ``urscript_transport``.
        """
        return cls(
            qd_max_radps=safety_cfg.max_joint_velocity_radps,
            max_off_axis_drift_m=min(safety_cfg.max_abs_y_drift_m, safety_cfg.max_abs_z_drift_m),
            max_orientation_error_rad=safety_cfg.max_orientation_error_rad,
        )

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


# Validated 6-parameter combination that closes the real-hardware
# noise-driven-spurious-trip gap the graduated tolerance above does NOT close
# on its own (2026-07-30 backtest: docs/status/safety_envelope_backtest_2026-07-30.md
# section 9, experiments/safety-envelope-study branch). Using real measured
# RTDE noise magnitudes, the graduated-tolerance fields alone
# (accel_max_consecutive_violations=3/accel_hard_multiple=5.0, same for speed)
# still spuriously tripped 30/30 seeds on a profile constructed to be
# genuinely physically clean; only combining them with the older, separately
# existing accel_gap_cycles/speed_lowpass_alpha filtering (a 2026-07-28 fix,
# unrelated to the graduated-tolerance work) closed the gap: 0/30 spurious
# trips, while still correctly catching the real genuine-catch case (the
# documented -0.20m/1.0s min-jerk move, theoretical peak accel 1.1547 m/s^2)
# 30/30. Exposed via --noise-robust-guards in tools/ur5e_move.py and
# tools/ur5e_direct_torque_x_transport.py: apply this preset first, then let
# any explicit individual override flag win for that specific field.
NOISE_ROBUST_GUARD_OVERRIDES: dict[str, float | int] = {
    "accel_max_consecutive_violations": 3,
    "accel_hard_multiple": 5.0,
    "speed_max_consecutive_violations": 3,
    "speed_hard_multiple": 5.0,
    "accel_gap_cycles": 5,
    "speed_lowpass_alpha": 0.2,
}


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
        # Real elapsed wall-clock time between set_start()/check() calls,
        # used for the speed/accel estimate instead of the caller-supplied
        # dt_s. Added 2026-07-28: on real hardware, real elapsed time between
        # set_start() (called once, before any per-cycle setup work -- e.g.
        # direct_torque_transport.py's local_dynamics.jacobian_and_mass_matrix()
        # call, itself real wall-clock time) and the first check() call can be
        # meaningfully longer than the nominal control period. Dividing a
        # (tiny but real) position delta by the WRONG, too-small assumed dt_s
        # inflates the computed speed, and squares that error again for
        # accel -- reproduced twice on real hardware (13.9 m/s^2 on a cycle
        # with qd<=0.0001 rad/s and tau<=0.002 Nm; nothing physically moved).
        # Using real elapsed time fixes this for every cycle, not just the
        # first -- in steady-state control-loop operation, real elapsed time
        # is already close to dt_s, so this doesn't change behavior for a
        # real fast-motion event (confirmed separately: a genuine joint-
        # velocity divergence near the wrist singularity was still caught
        # correctly the same day, before this fix, at real elapsed times
        # close to nominal).
        self._prev_check_ns: int | None = None
        # Ring buffer of (pos, corrected_clock_s) pairs, up to
        # accel_gap_cycles entries, used ONLY to form the gap-windowed speed
        # sample fed into the accel estimate (see check()). corrected_clock_s
        # is a running sum of each cycle's own real_dt_s (the same
        # max(dt_s, measured) value used for the single-cycle speed check),
        # NOT raw wall-clock timestamps -- summing already-corrected
        # per-cycle values keeps this exact for synthetic/test call
        # sequences with no real sleep between check() calls, the same way
        # the single-cycle real_dt_s fix does. At accel_gap_cycles=1 this
        # reduces to exactly the original single-cycle behavior.
        self._gap = max(int(limits.accel_gap_cycles), 1)
        self._corrected_clock_s = 0.0
        self._pos_history: deque[tuple[np.ndarray, float]] = deque(maxlen=self._gap)
        # Separate gap-windowed ring buffer + EMA state for the SPEED-LIMIT
        # check specifically (see CartesianMoveLimits.speed_limit_gap_cycles/
        # speed_limit_lowpass_alpha docstring) -- independent from
        # _pos_history/_prev_speed_mps above, which feed only the
        # acceleration estimate.
        self._speed_limit_gap = max(int(limits.speed_limit_gap_cycles), 1)
        self._speed_limit_corrected_clock_s = 0.0
        self._speed_limit_pos_history: deque[tuple[np.ndarray, float]] = deque(
            maxlen=self._speed_limit_gap
        )
        self._prev_speed_limit_ema: float | None = None
        # DeadlineMonitor-style consecutive-violation counters -- see
        # CartesianMoveLimits.accel_max_consecutive_violations' docstring.
        self._speed_consecutive_violations = 0
        self._accel_consecutive_violations = 0

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
        self._prev_check_ns = time.monotonic_ns()
        self._corrected_clock_s = 0.0
        self._pos_history.clear()
        self._pos_history.append((pose[:3].copy(), 0.0))
        self._speed_limit_corrected_clock_s = 0.0
        self._speed_limit_pos_history.clear()
        self._speed_limit_pos_history.append((pose[:3].copy(), 0.0))
        self._prev_speed_limit_ema = None
        self._speed_consecutive_violations = 0
        self._accel_consecutive_violations = 0

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

        # Use whichever is LONGER: the caller-supplied nominal dt_s, or the
        # real measured elapsed time since the previous check()/set_start()
        # call -- never shorter than either. See __init__'s _prev_check_ns
        # comment for the real-hardware bug this fixes (the gap between
        # set_start() and the first check() can be meaningfully longer than
        # one nominal control period, e.g. from real setup work in between,
        # and dividing a real position delta by too-small an assumed dt_s
        # inflates speed/accel). Taking the max (not just measured time)
        # keeps this a no-op for synthetic/test call sequences that invoke
        # check() back-to-back with no real sleep -- there, measured elapsed
        # time is always far smaller than the test's own intended dt_s, so
        # max() always resolves to dt_s, unchanged from before this fix.
        now_ns = time.monotonic_ns()
        measured_dt_s = (now_ns - self._prev_check_ns) / 1e9 if self._prev_check_ns is not None else float(dt_s)
        real_dt_s = max(float(dt_s), measured_dt_s)

        # Speed used for the speed-limit check. At speed_limit_gap_cycles=1,
        # speed_limit_lowpass_alpha=1.0 (the defaults), this reduces exactly
        # to the original immediate single-cycle step_m/real_dt_s -- see
        # CartesianMoveLimits.speed_limit_gap_cycles/speed_limit_lowpass_alpha
        # docstring for the real-hardware finding motivating the opt-in
        # smoothing path. Mirrors the accel estimate's gap-windowed/EMA
        # mechanism below, with its own independent ring buffer/EMA state so
        # enabling this cannot change the (separately opt-in) accel
        # behavior.
        speed_mps = step_m / real_dt_s
        speed_limit_new_clock_s = self._speed_limit_corrected_clock_s + real_dt_s
        speed_mps_for_limit = speed_mps
        if len(self._speed_limit_pos_history) >= self._speed_limit_gap:
            limit_gap_pos, limit_gap_clock_s = self._speed_limit_pos_history[0]
            limit_gap_step_m = float(np.linalg.norm(pos - limit_gap_pos))
            limit_gap_dt_s = max(speed_limit_new_clock_s - limit_gap_clock_s, 1e-9)
            raw_limit_gap_speed_mps = limit_gap_step_m / limit_gap_dt_s

            limit_alpha = float(self.limits.speed_limit_lowpass_alpha)
            if self._prev_speed_limit_ema is None:
                speed_mps_for_limit = raw_limit_gap_speed_mps
            else:
                speed_mps_for_limit = (
                    limit_alpha * raw_limit_gap_speed_mps
                    + (1.0 - limit_alpha) * self._prev_speed_limit_ema
                )
            self._prev_speed_limit_ema = speed_mps_for_limit

        if speed_mps_for_limit > self.limits.max_tcp_speed_mps:
            speed_hard_ceiling = self.limits.max_tcp_speed_mps * self.limits.speed_hard_multiple
            if self.limits.speed_max_consecutive_violations <= 1 or speed_mps_for_limit >= speed_hard_ceiling:
                decision.add(f"TCP speed {speed_mps_for_limit:.4f} m/s > {self.limits.max_tcp_speed_mps} m/s")
                self._speed_consecutive_violations = 0
            else:
                self._speed_consecutive_violations += 1
                if self._speed_consecutive_violations >= self.limits.speed_max_consecutive_violations:
                    decision.add(
                        f"TCP speed {speed_mps_for_limit:.4f} m/s > {self.limits.max_tcp_speed_mps} m/s "
                        f"for {self._speed_consecutive_violations} consecutive cycles"
                    )
        else:
            self._speed_consecutive_violations = 0
        self._speed_limit_corrected_clock_s = speed_limit_new_clock_s
        self._speed_limit_pos_history.append((pos.copy(), speed_limit_new_clock_s))

        # Gap-windowed, optionally low-pass-filtered speed sample, used ONLY
        # to feed the acceleration estimate (see CartesianMoveLimits'
        # accel_gap_cycles/speed_lowpass_alpha docstring for the mechanism
        # and the real-hardware finding motivating it). At the defaults
        # (gap=1, alpha=1.0) this reproduces the original single-cycle,
        # unfiltered accel_mps2 computation exactly.
        new_clock_s = self._corrected_clock_s + real_dt_s
        if len(self._pos_history) >= self._gap:
            gap_pos, gap_clock_s = self._pos_history[0]
            gap_step_m = float(np.linalg.norm(pos - gap_pos))
            gap_dt_s = max(new_clock_s - gap_clock_s, 1e-9)
            raw_gap_speed_mps = gap_step_m / gap_dt_s

            alpha = float(self.limits.speed_lowpass_alpha)
            if self._prev_speed_mps is None:
                gap_speed_mps = raw_gap_speed_mps
            else:
                gap_speed_mps = alpha * raw_gap_speed_mps + (1.0 - alpha) * self._prev_speed_mps

            if self._prev_speed_mps is not None:
                accel_mps2 = abs(gap_speed_mps - self._prev_speed_mps) / real_dt_s
                if accel_mps2 > self.limits.max_tcp_accel_mps2:
                    accel_hard_ceiling = self.limits.max_tcp_accel_mps2 * self.limits.accel_hard_multiple
                    if self.limits.accel_max_consecutive_violations <= 1 or accel_mps2 >= accel_hard_ceiling:
                        decision.add(
                            f"TCP acceleration {accel_mps2:.4f} m/s^2 > {self.limits.max_tcp_accel_mps2} m/s^2"
                        )
                        self._accel_consecutive_violations = 0
                    else:
                        self._accel_consecutive_violations += 1
                        if self._accel_consecutive_violations >= self.limits.accel_max_consecutive_violations:
                            decision.add(
                                f"TCP acceleration {accel_mps2:.4f} m/s^2 > {self.limits.max_tcp_accel_mps2} m/s^2 "
                                f"for {self._accel_consecutive_violations} consecutive cycles"
                            )
                else:
                    self._accel_consecutive_violations = 0

            self._prev_speed_mps = gap_speed_mps

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
        self._prev_check_ns = now_ns
        self._corrected_clock_s = new_clock_s
        self._pos_history.append((pos.copy(), new_clock_s))

        return decision
