"""
Simulator-independent safety monitor for torque-mode X-axis control.

All checks operate purely on ``RobotState`` snapshots plus a local view of the
commanded torque. There is no coupling to MuJoCo or CoppeliaSim, so the same
monitor is reused by the MuJoCo comparison script and the ROS 2 controller
node.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .state_types import RobotState


# UR5 manufacturer-style upper bounds, matching the project's
# ``SERVO_FORCE_LIMIT`` (from ``ur5e.xml``). These are the *eventual* caps.
# Stage 5 says to start with much smaller test limits.
UR5_MANUFACTURER_TAU_MAX_NM = np.array(
    [150.0, 150.0, 150.0, 28.0, 28.0, 28.0], dtype=np.float64
)

UR5_CONSERVATIVE_TEST_TAU_MAX_NM = np.array(
    [10.0, 10.0, 10.0, 3.0, 3.0, 3.0], dtype=np.float64
)

UR5_MANUFACTURER_QD_MAX_RAD_S = np.array(
    [3.15, 3.15, 3.15, 3.20, 3.20, 3.20], dtype=np.float64
)

# UR5 joint limits in radians. Check against your URDF / MJCF before relying
# on these; they are the commonly quoted "+- 2*pi" on UR5 revolute joints.
UR5_QLIM_LOWER_RAD = np.array(
    [-2.0 * np.pi] * 6, dtype=np.float64
)
UR5_QLIM_UPPER_RAD = np.array(
    [+2.0 * np.pi] * 6, dtype=np.float64
)


@dataclass
class SafetyConfig:
    """Safety thresholds used by ``SafetyMonitor``.

    The defaults are the **conservative test** limits from Stage 5.
    Increase only after stable behavior is confirmed.
    """

    tau_max: np.ndarray = field(
        default_factory=lambda: UR5_CONSERVATIVE_TEST_TAU_MAX_NM.copy()
    )
    qd_max: np.ndarray = field(
        default_factory=lambda: UR5_MANUFACTURER_QD_MAX_RAD_S.copy()
    )
    q_lower: np.ndarray = field(
        default_factory=lambda: UR5_QLIM_LOWER_RAD.copy()
    )
    q_upper: np.ndarray = field(
        default_factory=lambda: UR5_QLIM_UPPER_RAD.copy()
    )
    # Maximum allowed single-step end-effector jump (world frame, meters).
    ee_jump_max_m: float = 0.05
    # Maximum allowed Y/Z drift from the initial EE pose (meters).
    yz_drift_max_m: float = 0.05
    # Abort if |x_error| grows monotonically for this many seconds.
    x_error_growth_abort_s: float = 3.0
    # Deadman: if no torque command arrives for this long, hold zero torque.
    watchdog_timeout_s: float = 0.25


@dataclass
class SafetyStatus:
    ok: bool
    reason: str = ""
    # Number of checks tripped this cycle (for logging, mostly informational).
    tripped: int = 0


class SafetyMonitor:
    """Stateful safety monitor with basic anomaly detectors.

    Call ``check(state, tau)`` each cycle. On the first cycle it records an
    initial EE pose, which is used for Y/Z drift checks.
    """

    def __init__(self, config: SafetyConfig | None = None) -> None:
        self.config = config if config is not None else SafetyConfig()
        self._initial_ee_pos: np.ndarray | None = None
        self._prev_ee_pos: np.ndarray | None = None
        self._last_cmd_wall_time_s: float | None = None
        self._x_error_growing_since_s: float | None = None
        self._prev_abs_x_error: float | None = None

    def reset(self) -> None:
        self._initial_ee_pos = None
        self._prev_ee_pos = None
        self._last_cmd_wall_time_s = None
        self._x_error_growing_since_s = None
        self._prev_abs_x_error = None

    def touch(self) -> None:
        """Call whenever a fresh torque command is accepted (deadman reset)."""
        self._last_cmd_wall_time_s = time.monotonic()

    def watchdog_elapsed_s(self) -> float:
        if self._last_cmd_wall_time_s is None:
            return 0.0
        return float(time.monotonic() - self._last_cmd_wall_time_s)

    def check(self, state: RobotState, tau: np.ndarray) -> SafetyStatus:
        reasons: list[str] = []

        tau = np.asarray(tau, dtype=np.float64).reshape(-1)
        q = np.asarray(state["q"], dtype=np.float64).reshape(-1)
        qd = np.asarray(state["qd"], dtype=np.float64).reshape(-1)
        ee = np.asarray(state["ee_pos"], dtype=np.float64).reshape(-1)

        if np.any(~np.isfinite(tau)) or np.any(~np.isfinite(q)) or np.any(~np.isfinite(qd)):
            reasons.append("NaN/Inf in tau or joint state")

        if np.any(np.abs(tau) > self.config.tau_max + 1e-9):
            reasons.append(
                f"tau saturation: {np.abs(tau)} > {self.config.tau_max}"
            )

        if np.any(np.abs(qd) > self.config.qd_max + 1e-6):
            reasons.append(f"qd over limit: {np.abs(qd)} > {self.config.qd_max}")

        if np.any(q < self.config.q_lower) or np.any(q > self.config.q_upper):
            reasons.append("joint limit violated")

        if self._initial_ee_pos is None:
            self._initial_ee_pos = ee.copy()
        yz_now = np.linalg.norm(ee[1:3] - self._initial_ee_pos[1:3])
        if yz_now > self.config.yz_drift_max_m:
            reasons.append(f"Y/Z drift {yz_now:.3f} m > {self.config.yz_drift_max_m} m")

        if self._prev_ee_pos is not None:
            jump = float(np.linalg.norm(ee - self._prev_ee_pos))
            if jump > self.config.ee_jump_max_m:
                reasons.append(
                    f"EE jump {jump*1000:.1f} mm > {self.config.ee_jump_max_m*1000:.1f} mm"
                )
        self._prev_ee_pos = ee.copy()

        # X-error monotonic growth watchdog (decoupled from wall time -- uses
        # ``time`` field so it also works when replayed at non-real-time).
        abs_err = abs(float(state["target_x"]) - float(ee[0]))
        t_now = float(state["time"])
        if self._prev_abs_x_error is None or abs_err < self._prev_abs_x_error:
            self._x_error_growing_since_s = None
        else:
            if self._x_error_growing_since_s is None:
                self._x_error_growing_since_s = t_now
            elif t_now - self._x_error_growing_since_s > self.config.x_error_growth_abort_s:
                reasons.append(
                    f"x_error growing for >{self.config.x_error_growth_abort_s:.1f}s"
                )
        self._prev_abs_x_error = abs_err

        # Watchdog / deadman.
        if self._last_cmd_wall_time_s is not None:
            dt = self.watchdog_elapsed_s()
            if dt > self.config.watchdog_timeout_s:
                reasons.append(f"watchdog elapsed {dt:.3f}s > {self.config.watchdog_timeout_s:.3f}s")

        ok = len(reasons) == 0
        return SafetyStatus(ok=ok, reason="; ".join(reasons), tripped=len(reasons))
