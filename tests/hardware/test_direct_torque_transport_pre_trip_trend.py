"""``pre_trip_trend`` diagnostic capture in ``run_x_transport_direct_torque``
(2026-07-31). Real motivation: diagnosing a real guard trip on the real
UR5e required manually re-parsing trace.jsonl by hand to see the trend of
qd/x_error/tau/orientation error in the ~60 cycles before the trip. This
should now be captured automatically, for free, whenever a guard trips --
and be a complete no-op (key absent/None) for a clean run.

Two cases:
- a clean (no-trip) run has ``pre_trip_trend`` present and None;
- a synthetic run with monotonically increasing qd trips a guard and
  produces a ``pre_trip_trend`` whose qd_max_radps series is real and
  classified "rising" -- known ground truth, since qd is fed in strictly
  increasing.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from hardware.direct_torque_link import UR5eDirectTorqueLink
from hardware.direct_torque_transport import (
    PRE_TRIP_TREND_WINDOW_CYCLES,
    _classify_trend,
    run_x_transport_direct_torque,
)
from hardware.link import UR5eState
from hardware.poses import HEIGHT_ALPHA_0_5_Q
from hardware.safety import UR5eSafetyLimits

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / "config" / "ur5e_mujoco_torque_osc_tuned.yaml"


class _MockDTLink:
    """Same clean-run mock as test_direct_torque_transport_diagnostics.py --
    starts off-target so kp_x produces nonzero Fx, stays well inside every
    guard for the short duration used here."""

    def __init__(self) -> None:
        self._tcp_x = 0.35
        self.limits = UR5eSafetyLimits()
        self.connect_calls = 0

    def connect(self) -> None:
        self.connect_calls += 1

    def read_state(self) -> UR5eState:
        return UR5eState(
            q=HEIGHT_ALPHA_0_5_Q.copy(),
            qd=np.zeros(6),
            tcp_pose=np.array([self._tcp_x, -0.2, 0.3, 0.0, 3.14, 0.0]),
            host_stamp_ns=time.monotonic_ns(),
            robot_timestamp_s=None,
            safety_status=None,
        )

    def get_jacobian(self) -> np.ndarray:
        return np.eye(6)

    def get_mass_matrix(self) -> np.ndarray:
        return np.eye(6)

    def direct_torque(self, tau_nm: np.ndarray, *, friction_comp: bool = True) -> None:
        self._tcp_x += float(tau_nm[0]) * 1e-6

    @staticmethod
    def compose_robot_state(link_state, *, jacobian, mass_matrix, time_s, target_x, target_x_vel, dt_s=None, target_x_accel=None):
        return UR5eDirectTorqueLink.compose_robot_state(
            link_state, jacobian=jacobian, mass_matrix=mass_matrix,
            time_s=time_s, target_x=target_x, target_x_vel=target_x_vel, dt_s=dt_s,
            target_x_accel=target_x_accel,
        )

    def build_robot_state(self, link_state, *, time_s, target_x, target_x_vel, dt_s=None):
        return self.compose_robot_state(
            link_state, jacobian=self.get_jacobian(), mass_matrix=self.get_mass_matrix(),
            time_s=time_s, target_x=target_x, target_x_vel=target_x_vel, dt_s=dt_s,
        )

    def safe_stop(self, reason: str) -> None:
        pass


class _MockStationaryLink(_MockDTLink):
    """Perfectly still: tcp_pose never moves (direct_torque is a no-op) and
    the target delta is zero, so x_error/orientation/qd/speed/accel all stay
    at zero for the whole run -- a genuinely clean, guard-free run. (The
    diagnostics-test mock this class inherits from is off-target by design,
    which under this task's short/tight tolerances can itself trip the accel
    guard -- not suitable for a "definitely clean" test.)"""

    def direct_torque(self, tau_nm: np.ndarray, *, friction_comp: bool = True) -> None:
        pass


class _MockRisingQdLink(_MockDTLink):
    """Feeds a strictly-increasing |qd| every cycle (same value on all six
    joints, so the joint-velocity guard trips deterministically) -- known
    ground truth for the "rising" trend classification. TCP pose is held
    exactly fixed so no Cartesian speed/waypoint guard fires first for an
    unrelated reason; the joint-velocity ceiling
    (config safety.max_joint_velocity_radps: 3.0) is the only thing designed
    to trip here."""

    def __init__(self, qd_start: float = 0.1, qd_step: float = 0.1) -> None:
        super().__init__()
        self._qd_val = qd_start - qd_step
        self._qd_step = qd_step

    def read_state(self) -> UR5eState:
        self._qd_val += self._qd_step
        return UR5eState(
            q=HEIGHT_ALPHA_0_5_Q.copy(),
            qd=np.full(6, self._qd_val),
            tcp_pose=np.array([0.35, -0.2, 0.3, 0.0, 3.14, 0.0]),
            host_stamp_ns=time.monotonic_ns(),
            robot_timestamp_s=None,
            safety_status=None,
        )


@pytest.mark.hardware
def test_clean_run_has_no_pre_trip_trend(tmp_path: Path) -> None:
    link = _MockStationaryLink()
    result = run_x_transport_direct_torque(
        link,  # type: ignore[arg-type]
        config_path=CONFIG, target_x_delta_m=0.0,
        move_duration_s=0.02, duration_s=0.03, output_dir=tmp_path,
        motion_opt_in=True, record_latency=False, dynamics_source="local",
    )
    assert result.summary["termination_reason"] == "duration_complete"
    assert "pre_trip_trend" in result.summary
    assert result.summary["pre_trip_trend"] is None


@pytest.mark.hardware
def test_guard_trip_produces_rising_pre_trip_trend(tmp_path: Path) -> None:
    link = _MockRisingQdLink(qd_start=0.1, qd_step=0.1)
    result = run_x_transport_direct_torque(
        link,  # type: ignore[arg-type]
        config_path=CONFIG, target_x_delta_m=0.0,
        move_duration_s=0.1, duration_s=1.0, output_dir=tmp_path,
        motion_opt_in=True, record_latency=False, dynamics_source="local",
        enable_residual_observer=False,
    )
    summary = result.summary
    assert summary["termination_reason"] != "duration_complete"

    trend = summary["pre_trip_trend"]
    assert trend is not None
    assert 0 < trend["window_cycles"] <= PRE_TRIP_TREND_WINDOW_CYCLES

    for key in (
        "qd_max_radps",
        "tcp_speed_mps",
        "x_error_m",
        "tau_controller_l1",
        "orientation_error_norm_rad",
        "y_drift_m",
        "z_drift_m",
    ):
        assert key in trend
        assert "values" in trend[key]
        assert "trend" in trend[key]
        assert len(trend[key]["values"]) == trend["window_cycles"]

    # TCP pose was held exactly fixed every cycle (see _MockRisingQdLink),
    # so Y/Z drift relative to the initial pose is identically zero -- a
    # real, known "stable" case, same reasoning as tcp_speed_mps below.
    assert all(v == pytest.approx(0.0, abs=1e-9) for v in trend["y_drift_m"]["values"])
    assert all(v == pytest.approx(0.0, abs=1e-9) for v in trend["z_drift_m"]["values"])

    # Ground truth: qd was fed in strictly increasing every cycle, so its
    # window must be monotonically increasing and classified "rising".
    qd_values = trend["qd_max_radps"]["values"]
    assert all(b >= a for a, b in zip(qd_values, qd_values[1:]))
    assert qd_values[-1] > qd_values[0]
    assert trend["qd_max_radps"]["trend"] == "rising"

    # TCP pose was held exactly fixed every cycle, so the position-delta
    # speed signal is identically zero -- a real, known "stable" case.
    assert all(v == pytest.approx(0.0, abs=1e-9) for v in trend["tcp_speed_mps"]["values"])
    assert trend["tcp_speed_mps"]["trend"] == "stable"


def test_classify_trend_edge_cases() -> None:
    assert _classify_trend([]) == "insufficient_data"
    assert _classify_trend([1.0]) == "insufficient_data"
    assert _classify_trend([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) == "stable"
    assert _classify_trend([0.0, 0.0, 0.0, 5.0, 5.0, 5.0]) == "rising"
    assert _classify_trend([5.0, 5.0, 5.0, 0.0, 0.0, 0.0]) == "falling"


def test_classify_trend_near_zero_noise_is_stable_not_rising() -> None:
    """Regression for a real bug found 2026-08-01 via a Kalman-filtering
    follow-up: a signal hovering near zero with no true trend (e.g.
    y_drift_m/z_drift_m during a clean segment) used to get misclassified as
    rising/falling almost every time, because the old relative-only deadband
    collapses when the mean is near zero -- a tiny absolute noise wiggle is a
    huge fraction of an already-tiny mean. Fixed via an added absolute
    noise-floor deadband (2x the window's own std)."""
    # A small, non-monotonic wiggle around zero -- no real drift, well
    # within a plausible real RTDE position-noise floor (~1e-5 m).
    noisy_near_zero = [1e-6, -2e-6, 3e-6, -1e-6, 2e-6, -3e-6]
    assert _classify_trend(noisy_near_zero) == "stable"

    # A genuine small-magnitude but real, monotonic drift (change well above
    # the window's own noise floor) must still be classified correctly --
    # the fix must not make the detector blind to real slow trends.
    real_slow_drift = [-0.00301, -0.00302, -0.00303, -0.00336, -0.00338, -0.00339]
    assert _classify_trend(real_slow_drift) == "falling"
