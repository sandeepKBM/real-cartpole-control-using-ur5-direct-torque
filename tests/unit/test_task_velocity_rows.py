"""Velocity-tracked task rows: drop kp*(pos_des - p), keep kd*(vel_des - v).

Built for a swing-up that must flip in one or two strokes. The obvious build is
a separate joint-velocity PD, phase-switched against the QP -- rejected because
a plain velocity PD carries NO corridor, orientation or manipulability CBF row,
so drift protection would be weakest during the most aggressive phase, and
drift is what has actually been ending these runs (the LQR catch trips
|Y-Y0| at dX=0.070/dY=0.059 identically for a_max in {9.603, 14, 20, 30}, i.e.
the corridor, not the actuation). Dropping one term from an existing task row
keeps every CBF, the torque box, the joint exclusion and posture weighting.

These tests assert the EFFECT -- that position error stops acting and velocity
error still does -- not that a flag parsed.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import yaml

from controller_core.x_task_yz_corridor_qp.config import XTaskYZCorridorQPConfig
from controller_core.x_task_yz_corridor_qp.controller import XTaskYZCorridorQPController

CONFIG = "config/ur5e_mujoco_torque_x_task_yz_corridor_qp_inplane_drive.yaml"

from tests.unit.test_x_task_yz_corridor_qp import make_state  # noqa: E402


def _ctrl(**over):
    c = copy.deepcopy(yaml.safe_load(open(CONFIG))["controller"])
    # manipulability_cbf needs a jacobian_fn (grad_mu wants J at perturbed q),
    # which only the MuJoCo adapter supplies. Off here because these tests
    # target the task ROW, not that barrier -- and at this pose cond(J)=7.20
    # keeps the row inactive anyway, so it is not what makes the assertions
    # below true. Its wiring is covered by its own tests.
    c["manipulability_cbf"] = False
    c.update(over)
    cfg = XTaskYZCorridorQPConfig.from_controller_yaml_section(c)
    return XTaskYZCorridorQPController(cfg)


def test_off_by_default_everywhere():
    assert _ctrl().task_velocity_rows == ()
    cfg = XTaskYZCorridorQPConfig.from_controller_yaml_section(
        yaml.safe_load(open(CONFIG))["controller"])
    assert cfg.task_velocity_rows == ()


def test_position_error_stops_acting_on_a_velocity_row():
    """The defining behaviour. Same state, same position error, two modes."""
    st = make_state(target_ee_pos=(0.55, -0.2, 0.3))  # 0.15 m of X error
    pos_ctrl, vel_ctrl = _ctrl(), _ctrl(task_velocity_rows=[0])
    for c in (pos_ctrl, vel_ctrl):
        c.reset_from_state(st)
    tau_pos = np.asarray(pos_ctrl.compute(copy.deepcopy(st)).tau, dtype=np.float64)
    tau_vel = np.asarray(vel_ctrl.compute(copy.deepcopy(st)).tau, dtype=np.float64)
    # A 0.15 m error at kp_x=1587 is a large wrench; ignoring it must change tau.
    assert not np.allclose(tau_pos, tau_vel, atol=1e-6)
    assert np.linalg.norm(tau_vel) < np.linalg.norm(tau_pos)


def test_velocity_row_is_insensitive_to_position_error_magnitude():
    """Stronger than the above: doubling the position error must change nothing."""
    a = make_state(target_ee_pos=(0.55, -0.2, 0.3))
    b = make_state(target_ee_pos=(0.70, -0.2, 0.3))
    c = _ctrl(task_velocity_rows=[0])
    c.reset_from_state(a)
    tau_a = np.asarray(c.compute(copy.deepcopy(a)).tau, dtype=np.float64)
    c2 = _ctrl(task_velocity_rows=[0])
    c2.reset_from_state(b)
    tau_b = np.asarray(c2.compute(copy.deepcopy(b)).tau, dtype=np.float64)
    np.testing.assert_allclose(tau_a, tau_b, atol=1e-9)


def test_velocity_error_still_acts_on_a_velocity_row():
    """Dropping kp must not accidentally drop kd -- that would be a dead row."""
    c = _ctrl(task_velocity_rows=[0])
    st = make_state()
    c.reset_from_state(st)
    tau_zero = np.asarray(c.compute(copy.deepcopy(st)).tau, dtype=np.float64)

    fast = make_state()
    fast["target_ee_vel"] = np.array([0.8, 0.0, 0.0], dtype=np.float64)
    c2 = _ctrl(task_velocity_rows=[0])
    c2.reset_from_state(fast)
    tau_fast = np.asarray(c2.compute(copy.deepcopy(fast)).tau, dtype=np.float64)
    assert not np.allclose(tau_zero, tau_fast, atol=1e-6)


def test_runtime_toggle_switches_mode_on_one_instance():
    """The phase switch must not require rebuilding the controller -- that is
    what keeps the handoff a source switch rather than a controller swap."""
    st = make_state(target_ee_pos=(0.55, -0.2, 0.3))
    c = _ctrl()
    c.reset_from_state(st)
    tau_pos = np.asarray(c.compute(copy.deepcopy(st)).tau, dtype=np.float64)
    c.task_velocity_rows = (0,)
    tau_vel = np.asarray(c.compute(copy.deepcopy(st)).tau, dtype=np.float64)
    c.task_velocity_rows = ()
    tau_back = np.asarray(c.compute(copy.deepcopy(st)).tau, dtype=np.float64)
    assert not np.allclose(tau_pos, tau_vel, atol=1e-6)
    np.testing.assert_allclose(tau_pos, tau_back, atol=1e-9)  # and it is reversible


def test_empty_rows_is_bit_identical_to_the_unpatched_path():
    """Every existing config must be untouched."""
    st = make_state(target_ee_pos=(0.55, -0.2, 0.3))
    a, b = _ctrl(), _ctrl(task_velocity_rows=[])
    for c in (a, b):
        c.reset_from_state(st)
    np.testing.assert_allclose(
        np.asarray(a.compute(copy.deepcopy(st)).tau, dtype=np.float64),
        np.asarray(b.compute(copy.deepcopy(st)).tau, dtype=np.float64), atol=0.0)


def test_untracked_row_is_refused_rather_than_silently_ignored():
    c = copy.deepcopy(yaml.safe_load(open(CONFIG))["controller"])
    c["task_velocity_rows"] = [1]          # task_axis_rows is (0,)
    with pytest.raises(ValueError, match="subset of task_axis_rows"):
        XTaskYZCorridorQPConfig.from_controller_yaml_section(c)


def test_out_of_range_row_is_refused():
    c = copy.deepcopy(yaml.safe_load(open(CONFIG))["controller"])
    c["task_velocity_rows"] = [5]
    with pytest.raises(ValueError, match="must be in 0..2"):
        XTaskYZCorridorQPConfig.from_controller_yaml_section(c)
