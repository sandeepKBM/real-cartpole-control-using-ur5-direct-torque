"""Tests for ``CartesianImpedanceConfig.second_task_axis_enabled`` -- see that
field's docstring in ``controller_core/x_axis_cartesian_impedance/config.py``
for the motivation (driving TWO translational axes simultaneously, needed for
the tool-Y-pumping pendulum pipeline in tools/diagnostics/pendulum_toolY_*.py
to test combined tool-X/tool-Y trajectories). These tests pin down:
  1. Default (flag off) is byte-identical to the pre-existing behavior.
  2. When enabled, the y-role axis tracks a real target_y/target_y_vel
     instead of the frozen reset-time value.
  3. Absent target_y (flag on, key missing) falls back to the historical
     p0[y_axis] hold -- not a crash, not a silent zero.
  4. Mutual exclusion with y_coupling_feedforward / corridor / integral is
     enforced (raises), matching this file's own documented rationale.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from controller_core.x_axis_cartesian_impedance import (  # noqa: E402
    CartesianImpedanceConfig,
    XAxisCartesianImpedanceController,
)


def _state(pos, vel=(0.0, 0.0, 0.0), target_x=0.0, target_x_vel=0.0,
           target_y=None, target_y_vel=None):
    st = {
        "time": 0.0,
        "q": np.zeros(6, dtype=np.float64),
        "qd": np.zeros(6, dtype=np.float64),
        "ee_pos": np.asarray(pos, dtype=np.float64),
        "ee_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        "ee_lin_vel": np.asarray(vel, dtype=np.float64),
        "ee_ang_vel": np.zeros(3, dtype=np.float64),
        "target_x": float(target_x),
        "target_x_vel": float(target_x_vel),
        "jacobian": np.eye(6, dtype=np.float64),
    }
    if target_y is not None:
        st["target_y"] = float(target_y)
    if target_y_vel is not None:
        st["target_y_vel"] = float(target_y_vel)
    return st


def _cfg(**overrides) -> CartesianImpedanceConfig:
    base = dict(
        kp_x=0.0, kd_x=0.0, kp_y=100.0, kd_y=10.0, kp_z=0.0, kd_z=0.0,
        kp_rot=0.0, kd_rot=0.0, kp_posture=0.0, kd_posture=0.0, kd_joint=0.0,
        tau_max_nm=np.array([100.0] * 6, dtype=np.float64),
    )
    base.update(overrides)
    return CartesianImpedanceConfig(**base)


def _run_once(cfg, start, state):
    ctrl = XAxisCartesianImpedanceController(cfg)
    ctrl.reset_from_state(start)
    return ctrl.compute(state)


def test_default_is_disabled():
    assert _cfg().second_task_axis_enabled is False


def test_disabled_default_matches_pre_existing_hold_behavior():
    """Flag off: y-role axis holds at the frozen reset value, exactly as
    before this feature existed -- even if a caller (mistakenly or not)
    supplies target_y, it must be ignored when the flag is off."""
    cfg = _cfg()
    start = _state((0.0, 0.0, 0.0))
    state = _state((0.0, 0.05, 0.0), target_y=999.0)  # moved +5cm in Y; target_y ignored
    out = _run_once(cfg, start, state)
    # y_err = y_des(=0, frozen) - p[1](=0.05) = -0.05 -> Fy = kp_y*(-0.05) = -5.0
    assert out.wrench[1] == pytest.approx(-5.0, abs=1e-9)


def test_enabled_tracks_moving_target_y():
    cfg = _cfg(second_task_axis_enabled=True)
    start = _state((0.0, 0.0, 0.0))
    # EE sits at y=0.0; commanded target_y=0.02 -> y_err=+0.02 -> Fy=kp_y*0.02=2.0
    state = _state((0.0, 0.0, 0.0), target_y=0.02)
    out = _run_once(cfg, start, state)
    assert out.wrench[1] == pytest.approx(2.0, abs=1e-9)


def test_enabled_uses_target_y_vel_in_damping_term():
    cfg = _cfg(kp_y=0.0, kd_y=10.0, second_task_axis_enabled=True)
    start = _state((0.0, 0.0, 0.0))
    # y_err=0 (target_y==p[1]), but target_y_vel=0.5 with actual v[1]=0.0
    # -> Fy = kd_y*(y_vel_des - v[1]) = 10*(0.5-0.0) = 5.0
    state = _state((0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), target_y=0.0, target_y_vel=0.5)
    out = _run_once(cfg, start, state)
    assert out.wrench[1] == pytest.approx(5.0, abs=1e-9)


def test_enabled_without_target_y_key_falls_back_to_frozen_hold():
    """The flag alone does not change behavior for a caller that never
    supplies target_y -- the safe, non-silently-different fallback."""
    cfg = _cfg(second_task_axis_enabled=True)
    start = _state((0.0, 0.0, 0.0))
    state = _state((0.0, 0.05, 0.0))  # no target_y key at all
    out = _run_once(cfg, start, state)
    assert out.wrench[1] == pytest.approx(-5.0, abs=1e-9)


@pytest.mark.parametrize("conflicting_kwargs", [
    dict(y_coupling_feedforward=True),
    dict(y_control_mode="corridor"),
    dict(y_integral_action=True),
])
def test_mutually_exclusive_with_other_y_mechanisms(conflicting_kwargs):
    cfg = _cfg(second_task_axis_enabled=True, **conflicting_kwargs)
    start = _state((0.0, 0.0, 0.0))
    state = _state((0.0, 0.0, 0.0), target_y=0.02)
    with pytest.raises(ValueError):
        _run_once(cfg, start, state)


def test_hold_current_pose_unaffected_by_flag():
    """second_task_axis_enabled only applies in the non-hold-current-pose
    branch -- hold_current_pose must still pin y at the settle reference
    regardless, matching this feature's own scoped design."""
    cfg = _cfg(second_task_axis_enabled=True)
    ctrl = XAxisCartesianImpedanceController(cfg)
    start = _state((0.0, 0.0, 0.0))
    ctrl.reset_from_state(start)
    state = _state((0.0, 0.05, 0.0), target_y=999.0)
    state["hold_current_pose"] = True
    out = ctrl.compute(state)
    # First hold_current_pose call captures (0,0.05,0) as the new reference
    # (see controller.py's own settle-capture semantics), so y_err=0 here.
    assert out.wrench[1] == pytest.approx(0.0, abs=1e-9)
