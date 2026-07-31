"""Smoke tests for Cartesian impedance core."""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from controller_core.state_types import as_impedance_robot_state  # noqa: E402
from controller_core.x_axis_cartesian_impedance import (  # noqa: E402
    CartesianImpedanceConfig,
    XAxisCartesianImpedanceController,
)
from transport_metrics import GAIN_FIELDS  # noqa: E402


def _state(t, x, vx, y, vy, z, vz, quat, wx, wy, wz, q, qd, target_x, J):
    return {
        "time": t,
        "q": np.asarray(q, dtype=np.float64),
        "qd": np.asarray(qd, dtype=np.float64),
        "ee_pos": np.array([x, y, z], dtype=np.float64),
        "ee_quat": np.asarray(quat, dtype=np.float64),
        "ee_lin_vel": np.array([vx, vy, vz], dtype=np.float64),
        "ee_ang_vel": np.array([wx, wy, wz], dtype=np.float64),
        "target_x": float(target_x),
        "jacobian": np.asarray(J, dtype=np.float64),
    }


def test_hold_at_goal_zero_wrench_components() -> None:
    cfg = CartesianImpedanceConfig(
        kp_x=25.0,
        kd_x=8.0,
        kp_y=80.0,
        kd_y=15.0,
        kp_z=120.0,
        kd_z=20.0,
        kp_rot=20.0,
        kd_rot=5.0,
        kp_posture=2.0,
        kd_posture=0.5,
        kd_joint=0.8,
        tau_max_nm=np.array([50.0] * 6),
    )
    ctrl = XAxisCartesianImpedanceController(cfg)
    J = np.eye(6)
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    q0 = np.zeros(6)
    st0 = _state(0.0, 0.1, 0.0, 0.0, 0.0, 0.5, 0.0, quat, 0, 0, 0, q0, np.zeros(6), 0.1, J)
    ctrl.reset_from_state(st0)
    out = ctrl.compute(st0)
    assert abs(out.x_error) < 1e-9
    assert np.linalg.norm(out.wrench[:3]) < 1e-6


def test_as_impedance_robot_state_passes_through_dt_s() -> None:
    """as_impedance_robot_state() is the function compute() calls internally
    (`st = as_impedance_robot_state(state)`) -- this is the layer-2 fix:
    dt_s must survive that normalization so a future compute() could read
    it via state.get("dt_s")."""
    J = np.eye(6)
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    raw = _state(0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, quat, 0, 0, 0, np.zeros(6), np.zeros(6), 0.0, J)
    raw["dt_s"] = 0.002
    normalized = as_impedance_robot_state(raw)
    assert normalized["dt_s"] == 0.002


def test_as_impedance_robot_state_omits_dt_s_when_absent() -> None:
    J = np.eye(6)
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    raw = _state(0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, quat, 0, 0, 0, np.zeros(6), np.zeros(6), 0.0, J)
    normalized = as_impedance_robot_state(raw)
    assert "dt_s" not in normalized


def _assert_outputs_identical(a, b) -> None:
    for field in dataclasses.fields(a):
        av = getattr(a, field.name)
        bv = getattr(b, field.name)
        if isinstance(av, np.ndarray) or isinstance(bv, np.ndarray):
            np.testing.assert_array_equal(av, bv, err_msg=f"field {field.name!r} differs")
        else:
            assert av == bv, f"field {field.name!r} differs: {av!r} != {bv!r}"


def test_compute_output_unaffected_by_dt_s_presence() -> None:
    """Purely additive plumbing: compute() does not read dt_s yet, so its
    output must be byte-identical whether or not the caller's state dict
    carries a dt_s key. Regression guard for this state-contract change."""
    cfg = CartesianImpedanceConfig(
        kp_x=25.0,
        kd_x=8.0,
        kp_y=80.0,
        kd_y=15.0,
        kp_z=120.0,
        kd_z=20.0,
        kp_rot=20.0,
        kd_rot=5.0,
        kp_posture=2.0,
        kd_posture=0.5,
        kd_joint=0.8,
        tau_max_nm=np.array([50.0] * 6),
    )
    J = np.eye(6)
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    q0 = np.zeros(6)

    ctrl_without = XAxisCartesianImpedanceController(cfg)
    st0 = _state(0.0, 0.1, 0.0, 0.0, 0.0, 0.5, 0.0, quat, 0, 0, 0, q0, np.zeros(6), 0.1, J)
    ctrl_without.reset_from_state(st0)
    st1 = _state(0.01, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, quat, 0, 0, 0, q0, np.zeros(6), 0.05, J)
    out_without = ctrl_without.compute(st1)

    ctrl_with = XAxisCartesianImpedanceController(cfg)
    st0_dt = dict(st0)
    st0_dt["dt_s"] = 0.002
    ctrl_with.reset_from_state(st0_dt)
    st1_dt = dict(st1)
    st1_dt["dt_s"] = 0.002
    out_with = ctrl_with.compute(st1_dt)

    _assert_outputs_identical(out_without, out_with)


def test_x_error_produces_positive_fx() -> None:
    cfg = CartesianImpedanceConfig(tau_max_nm=np.array([100.0] * 6))
    ctrl = XAxisCartesianImpedanceController(cfg)
    J = np.eye(6)
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    st0 = _state(0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, quat, 0, 0, 0, np.zeros(6), np.zeros(6), 0.0, J)
    ctrl.reset_from_state(st0)
    st1 = _state(0.01, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, quat, 0, 0, 0, np.zeros(6), np.zeros(6), 0.05, J)
    out = ctrl.compute(st1)
    assert out.x_error > 0
    assert out.wrench[0] > 0


def test_torque_backtracking_shrinks_task_scale_under_tight_limits() -> None:
    cfg = CartesianImpedanceConfig(
        kp_x=25.0,
        kd_x=0.0,
        kp_y=0.0,
        kd_y=0.0,
        kp_z=0.0,
        kd_z=0.0,
        kp_rot=0.0,
        kd_rot=0.0,
        kp_posture=0.0,
        kd_posture=0.0,
        kd_joint=0.0,
        tau_max_nm=np.array([0.5] * 6, dtype=np.float64),
        torque_headroom=0.9,
        task_resample_factor=0.5,
        task_resample_min_scale=1.0 / 64.0,
        task_resample_max_iters=8,
    )
    ctrl = XAxisCartesianImpedanceController(cfg)
    J = np.eye(6)
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    q0 = np.zeros(6)
    st0 = _state(0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, quat, 0.0, 0.0, 0.0, q0, np.zeros(6), 0.0, J)
    ctrl.reset_from_state(st0)
    st1 = _state(0.01, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, quat, 0.0, 0.0, 0.0, q0, np.zeros(6), 1.0, J)
    out = ctrl.compute(st1)
    assert out.task_backtrack_iters >= 1
    assert out.task_backtrack_scale < 1.0
    assert out.task_scale < 1.0
    assert out.task_feasible
    assert np.isclose(out.tau_task[0], out.tau_task_nominal[0] * out.task_backtrack_scale)
    assert np.max(np.abs(out.tau_preclip)) <= 0.5 * 0.9 + 1e-9
    assert np.max(np.abs(out.tau)) <= 0.5 + 1e-9


def test_bias_only_saturation_backtracks_full_torque_candidate() -> None:
    cfg = CartesianImpedanceConfig(
        kp_x=0.0,
        kd_x=0.0,
        kp_y=0.0,
        kd_y=0.0,
        kp_z=0.0,
        kd_z=0.0,
        kp_rot=0.0,
        kd_rot=0.0,
        kp_posture=2.0,
        kd_posture=0.5,
        kd_joint=0.8,
        tau_max_nm=np.array([0.5] * 6, dtype=np.float64),
        torque_headroom=0.9,
        task_resample_factor=0.5,
        task_resample_min_scale=1.0 / 64.0,
        task_resample_max_iters=8,
    )
    ctrl = XAxisCartesianImpedanceController(cfg)
    J = np.eye(6)
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    q0 = np.zeros(6)
    st0 = _state(0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, quat, 0.0, 0.0, 0.0, q0, np.zeros(6), 0.0, J)
    ctrl.reset_from_state(st0)
    q1 = np.array([0.0, 1.5, 1.5, 1.5, 1.5, 1.5], dtype=np.float64)
    st1 = _state(0.01, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, quat, 0.0, 0.0, 0.0, q1, np.zeros(6), 0.0, J)
    out = ctrl.compute(st1)
    assert out.task_backtrack_iters >= 1
    assert out.task_backtrack_scale < 1.0
    assert out.task_feasible
    assert np.allclose(
        out.tau_preclip,
        out.tau_task + out.tau_damping + out.tau_posture + out.tau_gravity,
    )
    assert np.max(np.abs(out.tau_preclip)) <= 0.5 * 0.9 + 1e-9
    assert np.max(np.abs(out.tau)) <= 0.5 + 1e-9


def test_hold_current_pose_reanchors_controller_state() -> None:
    cfg = CartesianImpedanceConfig(
        kp_x=0.0,
        kd_x=0.0,
        kp_y=0.0,
        kd_y=0.0,
        kp_z=0.0,
        kd_z=0.0,
        kp_rot=0.0,
        kd_rot=0.0,
        kp_posture=0.0,
        kd_posture=0.0,
        kd_joint=0.0,
        tau_max_nm=np.array([10.0] * 6, dtype=np.float64),
    )
    ctrl = XAxisCartesianImpedanceController(cfg)
    J = np.eye(6)
    quat0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    st0 = _state(0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, quat0, 0.0, 0.0, 0.0, np.zeros(6), np.zeros(6), 0.0, J)
    ctrl.reset_from_state(st0)
    q1 = np.array([0.25, -0.35, 0.45, -0.55, 0.65, -0.75], dtype=np.float64)
    quat1 = np.array([0.70710678, 0.0, 0.70710678, 0.0], dtype=np.float64)
    st1 = _state(0.1, 0.2, -0.1, 0.3, 0.0, 0.7, 0.0, quat1, 0.0, 0.0, 0.0, q1, np.zeros(6), 1.0, J)
    st1["hold_current_pose"] = True
    out = ctrl.compute(st1)
    assert np.allclose(out.wrench, np.zeros(6))
    assert np.allclose(out.tau_posture, np.zeros(6))
    assert np.allclose(out.tau_damping, np.zeros(6))
    assert np.allclose(ctrl._q_rest, q1)
    assert np.isclose(ctrl._x0, 0.2)
    assert np.isclose(ctrl._y0, 0.3)
    assert np.isclose(ctrl._z0, 0.7)
    assert np.allclose(ctrl._quat0, quat1)
    st2 = _state(
        0.2,
        0.4,
        0.0,
        0.6,
        0.0,
        0.8,
        0.0,
        quat0,
        0.0,
        0.0,
        0.0,
        np.zeros(6),
        np.zeros(6),
        2.0,
        J,
    )
    st2["hold_current_pose"] = True
    _ = ctrl.compute(st2)
    assert np.isclose(ctrl._x0, 0.2)
    assert np.isclose(ctrl._y0, 0.3)
    assert np.isclose(ctrl._z0, 0.7)
    assert np.allclose(ctrl._q_rest, q1)
    assert np.allclose(ctrl._quat0, quat1)


def _default_controller() -> XAxisCartesianImpedanceController:
    cfg = CartesianImpedanceConfig(tau_max_nm=np.array([1.0e6] * 6))
    ctrl = XAxisCartesianImpedanceController(cfg)
    J = np.eye(6)
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    st0 = _state(0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, quat, 0, 0, 0, np.zeros(6), np.zeros(6), 0.0, J)
    ctrl.reset_from_state(st0)
    return ctrl


def test_set_gains_updates_only_selected_fields() -> None:
    ctrl = _default_controller()
    before = {name: getattr(ctrl.cfg, name) for name in ctrl._SCHEDULABLE_GAIN_FIELDS}
    ctrl.set_gains({"kp_x": 999.0})
    assert ctrl.cfg.kp_x == 999.0
    for name in ctrl._SCHEDULABLE_GAIN_FIELDS:
        if name == "kp_x":
            continue
        assert getattr(ctrl.cfg, name) == before[name]
    # Non-gain fields untouched.
    assert np.array_equal(ctrl.cfg.tau_max_nm, np.array([1.0e6] * 6))
    assert ctrl.cfg.task_space_inertia_shaping is False
    assert ctrl.cfg.nullspace_posture is False


def test_set_gains_rejects_unknown_key() -> None:
    ctrl = _default_controller()
    for bad_key in ("tau_max_nm", "bogus_field"):
        try:
            ctrl.set_gains({bad_key: 1.0})
            assert False, f"expected ValueError for {bad_key!r}"
        except ValueError:
            pass


def test_set_gains_rejects_non_finite_atomically() -> None:
    ctrl = _default_controller()
    before = {name: getattr(ctrl.cfg, name) for name in ctrl._SCHEDULABLE_GAIN_FIELDS}
    for bad_value in (float("nan"), float("inf"), float("-inf")):
        try:
            ctrl.set_gains({"kp_x": 1.0, "kp_y": bad_value})
            assert False, f"expected ValueError for {bad_value!r}"
        except ValueError:
            pass
        # No partial apply: kp_x must NOT have been updated to 1.0 either.
        for name in ctrl._SCHEDULABLE_GAIN_FIELDS:
            assert getattr(ctrl.cfg, name) == before[name]


def test_set_gains_changes_next_compute_and_preserves_hold_state() -> None:
    ctrl = _default_controller()
    J = np.eye(6)
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    q1 = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6], dtype=np.float64)
    st_hold = _state(0.1, 0.05, 0.0, 0.0, 0.0, 0.5, 0.0, quat, 0, 0, 0, q1, np.zeros(6), 0.0, J)
    st_hold["hold_current_pose"] = True
    ctrl.compute(st_hold)
    q_rest_before = ctrl._q_rest.copy()
    hold_ref_before = ctrl._hold_reference_initialized

    st_move = _state(0.2, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, quat, 0, 0, 0, np.zeros(6), np.zeros(6), 1.0, J)
    baseline = ctrl.compute(st_move)
    ctrl.set_gains({"kp_x": ctrl.cfg.kp_x * 4.0})
    scaled = ctrl.compute(st_move)
    assert np.isclose(scaled.wrench[0], baseline.wrench[0] * 4.0)
    # Hold/anchor instance state must survive the gain change untouched.
    assert np.allclose(ctrl._q_rest, q_rest_before)
    assert ctrl._hold_reference_initialized == hold_ref_before


def test_gain_field_tuple_matches_transport_metrics() -> None:
    assert set(XAxisCartesianImpedanceController._SCHEDULABLE_GAIN_FIELDS) == set(GAIN_FIELDS)


if __name__ == "__main__":
    test_hold_at_goal_zero_wrench_components()
    test_x_error_produces_positive_fx()
    test_torque_backtracking_shrinks_task_scale_under_tight_limits()
    test_bias_only_saturation_backtracks_full_torque_candidate()
    test_hold_current_pose_reanchors_controller_state()
    test_set_gains_updates_only_selected_fields()
    test_set_gains_rejects_unknown_key()
    test_set_gains_rejects_non_finite_atomically()
    test_set_gains_changes_next_compute_and_preserves_hold_state()
    test_gain_field_tuple_matches_transport_metrics()
    print("impedance tests OK")
