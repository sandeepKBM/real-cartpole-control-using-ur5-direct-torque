"""Bridges isolated, short RTDE telemetry dropouts (duplicate reads) for the
Cartesian safety guard's derivative estimate ONLY.

Background: ``StaleStateMonitor`` (hardware/safety.py) already catches a
genuine RTDE stall -- 5+ consecutive frozen robot timestamps while the host
clock keeps advancing -- and e-stops. That threshold is deliberately not 1,
because at a control loop polling faster than the robot's own RTDE publish
rate, a single repeated timestamp can be entirely normal (see that class's
own docstring). What it does NOT catch: an isolated single- or double-cycle
freeze, too short to trip the 5-consecutive threshold, that still corrupts
``CartesianMoveMonitor``'s finite-difference speed/accel estimate -- a frozen
position followed by a real jump on the next fresh frame reads as a spurious
one-cycle speed/accel spike. Real-hardware evidence for this: two runs on
2026-08-02 (direct_torque_20260802_164116, _164659) both had ~17% isolated
single-cycle duplicate tcp_pose frames, evenly spread, uncorrelated with this
process's own loop timing (lateness_ms/cycle_work_ms stayed <0.5ms in every
run, duplicate or not -- ruling out this process as the cause). One of those
runs tripped the TCP-speed guard on exactly this signature.

This module bridges ONLY that narrow gap: on a detected duplicate, bounded to
``max_bridge_cycles`` consecutive cycles, it produces a forward-dynamics
prediction of (q, qd, tcp position) to feed the guard's derivative estimate
in place of the frozen raw reading. Beyond that bound, or with no prior real
state to anchor from, it always falls back to the raw reading unchanged --
StaleStateMonitor's 5-consecutive e-stop remains the real backstop and is
never touched by this class.

Explicit, deliberate scope limits -- do not widen without a fresh decision:
  * This NEVER feeds the torque-control path. The controller still reads
    live RTDE state every cycle, unchanged, even during a bridged cycle for
    the guard. This is the same principle that motivated ``read_state()``
    raising ``RTDEStateError`` instead of ever returning cached state (see
    hardware/link.py's module docstring, the fix for "the old ROS2 bug") --
    fabricated state must never enter the control loop. This class only ever
    feeds a SEPARATE, informational estimate to a safety guard.
  * Orientation is frozen at its last real value during a bridged cycle, not
    predicted -- only TCP position and joint state are forward-integrated.
    The guard's orientation-error check is unaffected by the failure mode
    this module exists to fix (a position/velocity finite-difference spike),
    so extrapolating it would be scope creep with no evidence behind it.
  * This makes ``predict_joint_acceleration`` (controller_core/dynamics_residual.py)
    feed a safety decision for the first time -- that module's own docstring
    states nothing it computes feeds a trip condition today. That is a real,
    deliberate policy change, scoped narrowly to this one bridging use case;
    it does not change dynamics_residual.py's diagnostic-only status anywhere
    else it's used (e.g. hardware/direct_torque_transport.py's post-hoc
    residual logging is untouched).

Pure numpy, no RTDE/hardware imports -- all physics inputs (mass matrix,
Coriolis term, Jacobian, commanded torque) are computed by the caller once
per cycle (already required for other purposes in the direct_torque loop)
and passed in, so this class is a small, independently unit-testable
arithmetic/bookkeeping layer, not a dynamics provider itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from controller_core.dynamics_residual import predict_joint_acceleration


@dataclass
class TelemetryGapBridgeResult:
    q: np.ndarray
    qd: np.ndarray
    tcp_pose: np.ndarray
    bridged: bool
    consecutive_duplicates: int


class TelemetryGapBridge:
    """Detects consecutive-duplicate RTDE reads and bridges up to
    ``max_bridge_cycles`` of them with a forward-dynamics position/velocity
    prediction, for feeding ``CartesianMoveMonitor.check()`` only.
    """

    def __init__(self, *, max_bridge_cycles: int = 2) -> None:
        if max_bridge_cycles < 1:
            raise ValueError("max_bridge_cycles must be >= 1")
        self.max_bridge_cycles = int(max_bridge_cycles)
        self.reset()

    def reset(self) -> None:
        self._last_real_q: np.ndarray | None = None
        self._last_real_qd: np.ndarray | None = None
        self._last_real_tcp_pose: np.ndarray | None = None
        self._last_real_timestamp: float | None = None
        self._consecutive_duplicates = 0

    def _is_duplicate(
        self,
        q: np.ndarray,
        tcp_pose: np.ndarray,
        robot_timestamp_s: float | None,
    ) -> bool:
        if self._last_real_q is None:
            return False
        if robot_timestamp_s is not None and self._last_real_timestamp is not None:
            return bool(robot_timestamp_s == self._last_real_timestamp)
        # No robot clock available -- fall back to raw-value equality on the
        # two signals CartesianMoveMonitor actually derives from (position,
        # joint state). Matches the informal duplicate-frame analysis this
        # module's evidence is based on.
        return bool(np.array_equal(tcp_pose, self._last_real_tcp_pose)) and bool(
            np.array_equal(q, self._last_real_q)
        )

    def process(
        self,
        *,
        q: np.ndarray,
        qd: np.ndarray,
        tcp_pose: np.ndarray,
        robot_timestamp_s: float | None,
        tau_applied: np.ndarray,
        mass_matrix: np.ndarray,
        coriolis_term: np.ndarray,
        jacobian: np.ndarray,
        dt_s: float,
    ) -> TelemetryGapBridgeResult:
        """One cycle. ``jacobian`` is the full 6x6 (or >=3x6) spatial
        Jacobian in the same [position; rotation] row convention as
        ``hardware/local_dynamics.py`` -- only the first 3 rows (linear
        velocity) are used.

        Returns the (possibly bridged) q/qd/tcp_pose to feed the guard.
        ``bridged=False`` means: use the raw reading unchanged, either
        because it wasn't a duplicate, or because the duplicate run exceeded
        ``max_bridge_cycles`` (defer to StaleStateMonitor), or because there
        is no prior real state to predict from yet.
        """
        q = np.asarray(q, dtype=np.float64).reshape(6)
        qd = np.asarray(qd, dtype=np.float64).reshape(6)
        tcp_pose = np.asarray(tcp_pose, dtype=np.float64).reshape(6)
        if dt_s <= 0.0:
            raise ValueError("dt_s must be positive")

        is_dup = self._is_duplicate(q, tcp_pose, robot_timestamp_s)

        if not is_dup:
            self._last_real_q = q.copy()
            self._last_real_qd = qd.copy()
            self._last_real_tcp_pose = tcp_pose.copy()
            self._last_real_timestamp = robot_timestamp_s
            self._consecutive_duplicates = 0
            return TelemetryGapBridgeResult(q, qd, tcp_pose, bridged=False, consecutive_duplicates=0)

        self._consecutive_duplicates += 1
        if self._consecutive_duplicates > self.max_bridge_cycles:
            # Beyond the bridge window -- raw (stale) reading passes through
            # unchanged. StaleStateMonitor's 5-consecutive e-stop is the real
            # backstop for a genuine stall; this class does not extend itself
            # to cover that case.
            return TelemetryGapBridgeResult(
                q, qd, tcp_pose, bridged=False, consecutive_duplicates=self._consecutive_duplicates
            )

        elapsed = self._consecutive_duplicates * float(dt_s)
        mass_matrix = np.asarray(mass_matrix, dtype=np.float64).reshape(6, 6)
        coriolis_term = np.asarray(coriolis_term, dtype=np.float64).reshape(6)
        tau_applied = np.asarray(tau_applied, dtype=np.float64).reshape(6)
        jacobian = np.asarray(jacobian, dtype=np.float64)

        qdd_pred = predict_joint_acceleration(mass_matrix, tau_applied, coriolis_term)
        qd_pred = self._last_real_qd + qdd_pred * elapsed
        q_pred = (
            self._last_real_q
            + self._last_real_qd * elapsed
            + 0.5 * qdd_pred * elapsed * elapsed
        )

        jac_pos = jacobian[:3, :6]
        qd_avg = 0.5 * (self._last_real_qd + qd_pred)
        tcp_vel_pred = jac_pos @ qd_avg
        tcp_pos_pred = self._last_real_tcp_pose[:3] + tcp_vel_pred * elapsed
        # Orientation frozen at the last real value -- see module docstring.
        tcp_pose_pred = np.concatenate([tcp_pos_pred, self._last_real_tcp_pose[3:6]])

        return TelemetryGapBridgeResult(
            q_pred,
            qd_pred,
            tcp_pose_pred,
            bridged=True,
            consecutive_duplicates=self._consecutive_duplicates,
        )
