"""Unit tests for controller_core.safety.ImpedanceSafetyMonitor.

No prior test file existed for this class. Covers the pre-existing
consecutive-growth behavior (regression baseline) and the axis_target_moving
fix: tracking error growing while chasing an actively-moving target (e.g. a
min-jerk move profile) must not trip the growth guard, since that's expected
dynamics, not divergence -- only sustained growth once the target has
settled should trip it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from controller_core.safety import ImpedanceSafetyConfig, ImpedanceSafetyMonitor  # noqa: E402


def _state(*, q=None, qd=None, ee_pos=(0.0, 0.0, 0.5)) -> dict:
    return {
        "q": np.zeros(6) if q is None else np.asarray(q, dtype=np.float64),
        "qd": np.zeros(6) if qd is None else np.asarray(qd, dtype=np.float64),
        "ee_pos": np.asarray(ee_pos, dtype=np.float64),
    }


def _monitor(max_axis_error_growth_steps: int = 5) -> ImpedanceSafetyMonitor:
    cfg = ImpedanceSafetyConfig(max_axis_error_growth_steps=max_axis_error_growth_steps)
    mon = ImpedanceSafetyMonitor(cfg)
    mon.reset()
    mon.set_initial_position(np.array([0.0, 0.0, 0.5]), move_axis=0)
    return mon


def test_static_target_growth_trips_after_threshold():
    mon = _monitor(max_axis_error_growth_steps=5)
    growing_errors = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006]
    last_ok = True
    for err in growing_errors:
        status = mon.check(_state(), axis_error=err, orientation_error_norm=0.0, axis_target_moving=False)
        last_ok = status.ok
    assert last_ok is False


def test_static_target_non_monotonic_growth_never_trips():
    mon = _monitor(max_axis_error_growth_steps=5)
    # Oscillating error: never grows for more than 1 consecutive step.
    errors = [0.001, 0.002, 0.001, 0.002, 0.001, 0.002, 0.001, 0.002]
    for err in errors:
        status = mon.check(_state(), axis_error=err, orientation_error_norm=0.0, axis_target_moving=False)
        assert status.ok is True


def test_moving_target_growth_never_trips_regardless_of_duration():
    mon = _monitor(max_axis_error_growth_steps=5)
    # Same monotonically growing error sequence that trips the static case
    # above, but with axis_target_moving=True throughout (e.g. tracking lag
    # during a min-jerk move ramp) -- must never trip.
    growing_errors = [0.001 * i for i in range(1, 51)]
    for err in growing_errors:
        status = mon.check(_state(), axis_error=err, orientation_error_norm=0.0, axis_target_moving=True)
        assert status.ok is True


def test_growth_trips_once_target_settles_after_a_move():
    mon = _monitor(max_axis_error_growth_steps=5)
    # Move phase: error grows while chasing the target -- must not trip.
    for err in (0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007):
        status = mon.check(_state(), axis_error=err, orientation_error_norm=0.0, axis_target_moving=True)
        assert status.ok is True
    # Target settles (axis_target_moving=False from here on); if the error
    # keeps growing now, that IS genuine divergence and must trip.
    last_ok = True
    for err in (0.008, 0.009, 0.010, 0.011, 0.012, 0.013):
        status = mon.check(_state(), axis_error=err, orientation_error_norm=0.0, axis_target_moving=False)
        last_ok = status.ok
    assert last_ok is False


def test_other_checks_unaffected_by_axis_target_moving():
    # Orientation/velocity/drift checks must still fire regardless of the
    # axis_target_moving flag -- it only gates the axis-error growth streak.
    mon = _monitor()
    status = mon.check(_state(qd=np.full(6, 5.0)), axis_error=0.0, orientation_error_norm=0.0, axis_target_moving=True)
    assert status.ok is False
    assert "rad/s" in status.reason

    mon2 = _monitor()
    status2 = mon2.check(_state(), axis_error=0.0, orientation_error_norm=1.0, axis_target_moving=True)
    assert status2.ok is False
    assert "orientation" in status2.reason


if __name__ == "__main__":
    test_static_target_growth_trips_after_threshold()
    test_static_target_non_monotonic_growth_never_trips()
    test_moving_target_growth_never_trips_regardless_of_duration()
    test_growth_trips_once_target_settles_after_a_move()
    test_other_checks_unaffected_by_axis_target_moving()
    print("safety tests OK")
