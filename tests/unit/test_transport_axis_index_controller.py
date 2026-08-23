"""Tests for the selectable transport (task) axis -- see
``CartesianImpedanceConfig.transport_axis_index``'s docstring in
``controller_core/x_axis_cartesian_impedance/config.py`` for the full axis->gain
mapping and the two things deliberately left un-generalized.

Motivation, briefly: ``compute()`` used to read ``p[0]``/``v[0]`` and write the
resulting task force into ``wrench[0]`` unconditionally. The
``transport_axis_index`` that already existed on the state contract
(``controller_core/state_types.py``) and in the MuJoCo adapter therefore only
changed what CALLERS computed as a target and which axis
``ImpedanceSafetyMonitor`` treated as "orthogonal" -- so selecting axis 1
produced a genuinely broken configuration: a target derived from Y compared
against, and driving, world X. These tests pin down that the error, the
velocity term, and the wrench row now all follow the selected axis together,
and that axis 0 remains exactly the historical behavior.

The class name ``XAxisCartesianImpedanceController`` and the ``kp_x``/``kd_x``/
``ki_x`` gain names are historical: those are the TASK-axis gains, not
literally world X.
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

JOINT_YAML_NAMES = (
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
)


def _state(pos, vel=(0.0, 0.0, 0.0), target=0.0, target_vel=0.0, axis=None, dt_s=None):
    """Synthetic state. ``target`` is the TASK-axis target (target_x is the
    task-axis reference regardless of which world axis is selected)."""
    st = {
        "time": 0.0,
        "q": np.zeros(6, dtype=np.float64),
        "qd": np.zeros(6, dtype=np.float64),
        "ee_pos": np.asarray(pos, dtype=np.float64),
        "ee_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        "ee_lin_vel": np.asarray(vel, dtype=np.float64),
        "ee_ang_vel": np.zeros(3, dtype=np.float64),
        "target_x": float(target),
        "target_x_vel": float(target_vel),
        "jacobian": np.eye(6, dtype=np.float64),
    }
    if axis is not None:
        st["transport_axis_index"] = int(axis)
    if dt_s is not None:
        st["dt_s"] = float(dt_s)
    return st


def _cfg(**overrides) -> CartesianImpedanceConfig:
    base = dict(
        kp_x=0.0, kd_x=0.0, kp_y=0.0, kd_y=0.0, kp_z=0.0, kd_z=0.0,
        kp_rot=0.0, kd_rot=0.0, kp_posture=0.0, kd_posture=0.0, kd_joint=0.0,
        tau_max_nm=np.array([100.0] * 6, dtype=np.float64),
    )
    base.update(overrides)
    return CartesianImpedanceConfig(**base)


def _run_once(cfg, start, state):
    ctrl = XAxisCartesianImpedanceController(cfg)
    ctrl.reset_from_state(start)
    return ctrl.compute(state)


# ---------------------------------------------------------------------------
# 1. Default / regression: axis 0 is untouched
# ---------------------------------------------------------------------------


def test_transport_axis_index_defaults_to_zero():
    assert _cfg().transport_axis_index == 0


def test_output_reports_axis_zero_by_default():
    cfg = _cfg(kp_x=100.0)
    out = _run_once(cfg, _state((0.3, -0.1, 0.5), target=0.3),
                    _state((0.3, -0.1, 0.5), target=0.35))
    assert out.transport_axis_index == 0


def test_absent_state_key_is_identical_to_explicit_axis_zero():
    """Every existing caller omits the key entirely; that path must be exactly
    the explicit axis-0 path, not merely similar."""
    cfg = _cfg(kp_x=400.0, kd_x=40.0, kp_y=300.0, kd_y=30.0, kp_z=120.0, kd_z=20.0)
    start = _state((0.3, -0.1, 0.5), target=0.3)
    moved = dict(pos=(0.32, -0.09, 0.51), vel=(0.05, -0.01, 0.02), target=0.35)
    out_absent = _run_once(cfg, start, _state(**moved))
    out_explicit = _run_once(cfg, start, _state(**moved, axis=0))
    np.testing.assert_array_equal(out_absent.wrench, out_explicit.wrench)
    np.testing.assert_array_equal(out_absent.tau, out_explicit.tau)
    assert out_absent.x_error == out_explicit.x_error


# ---------------------------------------------------------------------------
# 2. The task error is measured against the SELECTED axis
# ---------------------------------------------------------------------------


def test_task_error_follows_selected_axis():
    """Same state, same target value, different selected axis -> genuinely
    different task error, each read off its own axis."""
    cfg = _cfg(kp_x=100.0)
    pos = (0.30, -0.10, 0.50)
    target = 0.05

    out0 = _run_once(cfg, _state(pos, target=target), _state(pos, target=target, axis=0))
    out1 = _run_once(cfg, _state(pos, target=target), _state(pos, target=target, axis=1))
    out2 = _run_once(cfg, _state(pos, target=target), _state(pos, target=target, axis=2))

    assert out0.x_error == pytest.approx(0.05 - 0.30)   # against world X
    assert out1.x_error == pytest.approx(0.05 - (-0.10))  # against world Y
    assert out2.x_error == pytest.approx(0.05 - 0.50)   # against world Z
    assert out1.transport_axis_index == 1
    assert out2.transport_axis_index == 2


def test_task_force_lands_in_the_selected_wrench_row():
    """The other half of the fix: computing the error against the right axis is
    useless if the force still lands in row 0."""
    cfg = _cfg(kp_x=100.0)
    for axis in (0, 1, 2):
        pos = [0.30, -0.10, 0.50]
        target = pos[axis] + 0.02  # +2cm on the selected axis only
        out = _run_once(cfg, _state(pos, target=pos[axis]),
                        _state(pos, target=target, axis=axis))
        assert out.wrench[axis] == pytest.approx(100.0 * 0.02), f"axis={axis}"
        # No error on the two held axes (they sit exactly at their start), so
        # their rows must be exactly zero -- proving the task force did not
        # leak into them.
        for other in {0, 1, 2} - {axis}:
            assert out.wrench[other] == 0.0, f"axis={axis} leaked into row {other}"


def test_force_points_to_reduce_the_selected_axis_error():
    cfg = _cfg(kp_x=100.0)
    pos = (0.30, -0.10, 0.50)
    # Y target above current Y -> positive (increasing-Y) force in row 1.
    out_pos = _run_once(cfg, _state(pos, target=-0.10),
                        _state(pos, target=-0.06, axis=1))
    # Y target below current Y -> negative force in row 1.
    out_neg = _run_once(cfg, _state(pos, target=-0.10),
                        _state(pos, target=-0.14, axis=1))
    assert out_pos.wrench[1] > 0.0
    assert out_neg.wrench[1] < 0.0
    assert out_pos.wrench[0] == 0.0 and out_neg.wrench[0] == 0.0


def test_task_damping_uses_the_selected_axis_velocity():
    """kd_x must damp the selected axis' velocity, not world X's."""
    cfg = _cfg(kd_x=10.0)
    pos = (0.30, -0.10, 0.50)
    out = _run_once(cfg, _state(pos, target=-0.10),
                    _state(pos, vel=(0.7, 0.2, -0.3), target=-0.10, axis=1))
    # Zero task error, so the whole task force is -kd_x * v[1].
    assert out.wrench[1] == pytest.approx(-10.0 * 0.2)


# ---------------------------------------------------------------------------
# 3. Role swap: the axis that used to be driven becomes a HELD axis
# ---------------------------------------------------------------------------


def test_axis1_holds_world_x_with_the_kp_y_kd_y_gains():
    """With axis=1, world X is held by the kp_y/kd_y (hold-role) gains -- the
    documented transposition. Task gain is zeroed here to isolate the hold."""
    cfg = _cfg(kp_x=0.0, kd_x=0.0, kp_y=250.0, kd_y=12.0, kp_z=999.0, kd_z=999.0)
    start = _state((0.30, -0.10, 0.50), target=-0.10)
    # Push world X 3cm past its captured start, with an X velocity.
    out = _run_once(cfg, start,
                    _state((0.33, -0.10, 0.50), vel=(0.4, 0.0, 0.0),
                           target=-0.10, axis=1))
    # Restoring force back toward x0: kp_y*(x0 - x) - kd_y*v_x
    assert out.wrench[0] == pytest.approx(250.0 * (-0.03) - 12.0 * 0.4)
    assert out.wrench[0] < 0.0, "held X must be pulled back toward its start"
    # kp_z/kd_z (999) must NOT be what holds X -- that is the mapping bug this
    # test exists to catch.
    assert out.wrench[0] != pytest.approx(999.0 * (-0.03) - 999.0 * 0.4)


def test_axis2_gain_roles_are_the_documented_transposition():
    """axis=2: Z<-kp_x/kd_x (task), Y keeps kp_y/kd_y, X<-kp_z/kd_z."""
    cfg = _cfg(kp_x=100.0, kp_y=250.0, kp_z=40.0)
    start = _state((0.30, -0.10, 0.50), target=0.50)
    out = _run_once(cfg, start,
                    _state((0.33, -0.14, 0.50), target=0.52, axis=2))
    # Task row: kp_x * (target - current). Hold rows: kp * (start - current),
    # i.e. restoring toward the captured start value.
    assert out.wrench[2] == pytest.approx(100.0 * 0.02)   # Z driven by kp_x
    assert out.wrench[1] == pytest.approx(250.0 * 0.04)   # Y keeps kp_y
    assert out.wrench[0] == pytest.approx(40.0 * -0.03)   # X takes kp_z


# ---------------------------------------------------------------------------
# 4. Permutation equivalence -- the general proof, not a spot check
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("axis", [1, 2])
def test_axis_selection_is_exactly_a_coordinate_transposition(axis):
    """Selecting axis N on a state whose world coordinates have been swapped
    (0<->N) must reproduce the axis-0 wrench with the same rows swapped back.
    This proves generality across the WHOLE translational pipeline at once
    rather than checking one gain at a time."""
    cfg = _cfg(kp_x=400.0, kd_x=40.0, kp_y=300.0, kd_y=30.0, kp_z=120.0, kd_z=20.0)

    def swap(vec):
        out = list(vec)
        out[0], out[axis] = out[axis], out[0]
        return tuple(out)

    start_pos = (0.30, -0.10, 0.50)
    now_pos = (0.33, -0.13, 0.54)
    now_vel = (0.05, -0.02, 0.03)
    target = 0.36  # task-axis target, same number in both frames

    out_ref = _run_once(cfg, _state(start_pos, target=start_pos[0]),
                        _state(now_pos, vel=now_vel, target=target, axis=0))
    out_swapped = _run_once(cfg, _state(swap(start_pos), target=start_pos[0]),
                            _state(swap(now_pos), vel=swap(now_vel),
                                   target=target, axis=axis))

    np.testing.assert_allclose(out_swapped.wrench[0:3], swap(out_ref.wrench[0:3]), atol=1e-12)
    assert out_swapped.x_error == pytest.approx(out_ref.x_error)


# ---------------------------------------------------------------------------
# 5. Closed-loop rollout: the selected axis actually converges
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_closed_loop_rollout_drives_selected_axis_and_holds_the_others(axis):
    """Short numpy-only closed loop over a trivial 3-DOF point-mass plant
    driven by the controller's own translational wrench. The selected axis must
    converge toward its target; the two held axes must stay pinned near their
    start values. Not a plant-fidelity claim -- it checks the qualitative
    closed-loop behavior the axis fix is supposed to deliver."""
    cfg = _cfg(kp_x=400.0, kd_x=60.0, kp_y=400.0, kd_y=60.0, kp_z=400.0, kd_z=60.0)
    ctrl = XAxisCartesianImpedanceController(cfg)

    p = np.array([0.30, -0.10, 0.50])
    p0 = p.copy()
    v = np.zeros(3)
    target = p0[axis] + 0.05
    dt, mass = 0.002, 5.0

    ctrl.reset_from_state(_state(tuple(p), target=p0[axis], axis=axis))
    err_first = None
    for _ in range(1500):
        out = ctrl.compute(_state(tuple(p), vel=tuple(v), target=target,
                                  axis=axis, dt_s=dt))
        if err_first is None:
            err_first = abs(target - p[axis])
        v = v + (np.asarray(out.wrench[0:3]) / mass) * dt
        p = p + v * dt

    err_final = abs(target - p[axis])
    assert err_final < 0.1 * err_first, (
        f"axis={axis} did not converge: {err_first:.4f} -> {err_final:.4f}"
    )
    for other in {0, 1, 2} - {axis}:
        assert abs(p[other] - p0[other]) < 5e-3, (
            f"held axis {other} drifted {p[other] - p0[other]:.4f} m while axis={axis} moved"
        )


def test_axis1_rollout_moves_y_not_x():
    """The specific regression the fix targets: with axis=1 selected, a Y
    target must move Y. Before the fix the same request moved X instead."""
    cfg = _cfg(kp_x=400.0, kd_x=60.0, kp_y=400.0, kd_y=60.0, kp_z=400.0, kd_z=60.0)
    ctrl = XAxisCartesianImpedanceController(cfg)
    p = np.array([0.30, -0.10, 0.50])
    p0 = p.copy()
    v = np.zeros(3)
    target = p0[1] + 0.05
    dt, mass = 0.002, 5.0
    ctrl.reset_from_state(_state(tuple(p), target=p0[1], axis=1))
    for _ in range(1500):
        out = ctrl.compute(_state(tuple(p), vel=tuple(v), target=target, axis=1, dt_s=dt))
        v = v + (np.asarray(out.wrench[0:3]) / mass) * dt
        p = p + v * dt
    assert p[1] - p0[1] > 0.045, "Y should have travelled essentially the full 5cm"
    assert abs(p[0] - p0[0]) < 5e-3, "X must NOT have been driven"


# ---------------------------------------------------------------------------
# 6. Precedence, validation, YAML parsing
# ---------------------------------------------------------------------------


def test_state_key_overrides_config_field():
    """Documented precedence: the per-cycle state wins, so the transport loop
    that owns the guards and the target generator stays in control of the
    axis."""
    cfg = _cfg(kp_x=100.0, transport_axis_index=2)
    pos = (0.30, -0.10, 0.50)
    out = _run_once(cfg, _state(pos, target=-0.10), _state(pos, target=-0.06, axis=1))
    assert out.transport_axis_index == 1
    assert out.wrench[1] != 0.0 and out.wrench[2] == 0.0


def test_config_field_used_when_state_omits_the_key():
    cfg = _cfg(kp_x=100.0, transport_axis_index=1)
    pos = (0.30, -0.10, 0.50)
    out = _run_once(cfg, _state(pos, target=-0.10), _state(pos, target=-0.06))
    assert out.transport_axis_index == 1
    assert out.wrench[1] == pytest.approx(100.0 * 0.04)


@pytest.mark.parametrize("bad", [3, -1, 99])
def test_invalid_axis_raises(bad):
    cfg = _cfg(kp_x=100.0)
    pos = (0.30, -0.10, 0.50)
    ctrl = XAxisCartesianImpedanceController(cfg)
    with pytest.raises(ValueError, match="transport_axis_index"):
        ctrl.reset_from_state(_state(pos, target=0.0, axis=bad))


def test_yaml_parsing_roundtrip_and_default():
    limits = {name: 50.0 for name in JOINT_YAML_NAMES}
    cfg = CartesianImpedanceConfig.from_controller_yaml_section(
        {"gains": {}, "torque_limits_initial": limits, "transport_axis_index": 2}
    )
    assert cfg.transport_axis_index == 2
    cfg_default = CartesianImpedanceConfig.from_controller_yaml_section(
        {"gains": {}, "torque_limits_initial": limits}
    )
    assert cfg_default.transport_axis_index == 0


@pytest.mark.parametrize("bad", [3, -1, "x", True, None, 1.5])
def test_yaml_parsing_rejects_bad_axis_loudly(bad):
    """Must RAISE, never silently fall back to 0: a quiet fallback would run a
    Y transport as an X one, under the guards and targets of the axis the
    caller believed it selected."""
    limits = {name: 50.0 for name in JOINT_YAML_NAMES}
    with pytest.raises(ValueError, match="transport_axis_index"):
        CartesianImpedanceConfig.from_controller_yaml_section(
            {"gains": {}, "torque_limits_initial": limits, "transport_axis_index": bad}
        )


def test_yaml_parsing_accepts_integral_floats():
    """Plain YAML scalars routinely deserialize as floats; ``1.0`` names axis 1
    unambiguously, unlike ``1.5`` (rejected above rather than truncated)."""
    limits = {name: 50.0 for name in JOINT_YAML_NAMES}
    cfg = CartesianImpedanceConfig.from_controller_yaml_section(
        {"gains": {}, "torque_limits_initial": limits, "transport_axis_index": 1.0}
    )
    assert cfg.transport_axis_index == 1


def test_acceleration_feedforward_refuses_non_zero_axis():
    """Deliberately un-generalized: target_x_accel/target_y_accel/target_z_accel
    are named for physical axes while target_x is the task-axis reference, so
    the conventions collide. Raising beats silently corrupting."""
    cfg = _cfg(kp_x=100.0, acceleration_feedforward=True)
    pos = (0.30, -0.10, 0.50)
    ctrl = XAxisCartesianImpedanceController(cfg)
    ctrl.reset_from_state(_state(pos, target=-0.10, axis=1))
    with pytest.raises(ValueError, match="acceleration_feedforward"):
        ctrl.compute(_state(pos, target=-0.06, axis=1))
