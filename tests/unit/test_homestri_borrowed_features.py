"""Tests for the three opt-in mechanisms adapted from ian-chuang/homestri-ur5e-rl
(studied 2026-08-05): posture_inertia_scaling, task_velocity_saturation, and
orientation_error_exact_axis_angle.

The load-bearing test in this file is the first one: with every new flag left at
its default, the controller's output must be bit-identical to the pre-change
behavior. Everything else here is only safe because that holds.
"""

from __future__ import annotations

import numpy as np
import pytest

from controller_core.kinematics_utils import (
    orientation_error_vec_axis_angle_wxyz,
    orientation_error_vec_wxyz,
)
from controller_core.x_axis_cartesian_impedance import (
    CartesianImpedanceConfig,
    XAxisCartesianImpedanceController,
)


def _state(mass: bool = True) -> dict:
    """A representative non-trivial state; values chosen to exercise every term
    (nonzero velocity, off-axis position error, non-identity orientation)."""
    rng = np.random.default_rng(20260805)
    m = rng.normal(size=(6, 6))
    jac = rng.normal(size=(6, 6))
    st = {
        "time": 0.0,
        "q": np.array([-0.70, -0.84, -1.20, -0.99, 0.20, 0.0], dtype=np.float64),
        "qd": np.array([0.02, -0.03, 0.01, 0.004, -0.002, 0.001], dtype=np.float64),
        "ee_pos": np.array([-0.121, -0.234, 0.928], dtype=np.float64),
        "ee_quat": np.array([0.9962, 0.0872, 0.0, 0.0], dtype=np.float64),  # ~10 deg about x
        "ee_lin_vel": np.array([0.01, -0.002, 0.003], dtype=np.float64),
        "ee_ang_vel": np.array([0.004, 0.001, -0.002], dtype=np.float64),
        "target_x": -0.101,  # 2 cm of X error
        "jacobian": jac,
    }
    if mass:
        st["mass_matrix"] = m @ m.T + 6.0 * np.eye(6)  # SPD, well conditioned
    return st


def _osc_cfg(**over) -> CartesianImpedanceConfig:
    """Tuned-OSC-like config: inertia shaping + nullspace posture on, kp_rot=0."""
    base = dict(
        tau_max_nm=np.full(6, 1e6),  # keep clipping/backtracking out of the way
        kp_x=400.0,
        kd_x=40.0,
        kp_rot=0.0,
        kd_rot=10.0,
        kp_posture=25.0,
        kd_posture=6.0,
        kd_joint=4.0,
        lambda_regularization=0.1,
        task_space_inertia_shaping=True,
        nullspace_posture=True,
    )
    base.update(over)
    return CartesianImpedanceConfig(**base)


def _run(cfg: CartesianImpedanceConfig, st: dict):
    ctrl = XAxisCartesianImpedanceController(cfg)
    ctrl.reset_from_state(st)
    return ctrl.compute(st)


_LIMITS = {
    "torque_limits_initial": {
        "shoulder_pan_joint": 150.0,
        "shoulder_lift_joint": 150.0,
        "elbow_joint": 150.0,
        "wrist_1_joint": 28.0,
        "wrist_2_joint": 28.0,
        "wrist_3_joint": 28.0,
    }
}


# --- the guarantee everything else rests on -------------------------------


def test_all_new_flags_default_off():
    cfg = CartesianImpedanceConfig()
    assert cfg.posture_inertia_scaling is False
    assert cfg.task_velocity_saturation is False
    assert cfg.orientation_error_exact_axis_angle is False
    assert cfg.task_vel_sat_linear == 0.0
    assert cfg.task_vel_sat_angular == 0.0


def test_defaults_are_bit_identical_to_explicitly_disabled():
    """Setting every new flag to its default explicitly must change nothing."""
    st = _state()
    out_default = _run(_osc_cfg(), st)
    out_explicit = _run(
        _osc_cfg(
            posture_inertia_scaling=False,
            task_velocity_saturation=False,
            orientation_error_exact_axis_angle=False,
        ),
        st,
    )
    np.testing.assert_array_equal(out_default.tau, out_explicit.tau)
    np.testing.assert_array_equal(out_default.tau_posture, out_explicit.tau_posture)
    assert out_default.orientation_error_norm == out_explicit.orientation_error_norm


# --- orientation error ----------------------------------------------------


def test_axis_angle_matches_linearized_for_small_angles():
    """The two forms agree to O(theta^3), so they must be close when theta is small."""
    half = 0.001 / 2.0
    quat_cur = np.array([np.cos(half), np.sin(half), 0.0, 0.0])
    quat_des = np.array([1.0, 0.0, 0.0, 0.0])
    lin = orientation_error_vec_wxyz(quat_des, quat_cur)
    exact = orientation_error_vec_axis_angle_wxyz(quat_des, quat_cur)
    np.testing.assert_allclose(lin, exact, atol=1e-9)


@pytest.mark.parametrize("theta", [0.25, 0.5, 1.0, 2.0])
def test_axis_angle_recovers_true_angle_where_linearized_undershoots(theta):
    """Exact form returns theta; the linearized form returns 2*sin(theta/2)."""
    half = theta / 2.0
    quat_cur = np.array([np.cos(half), np.sin(half), 0.0, 0.0])
    quat_des = np.array([1.0, 0.0, 0.0, 0.0])
    exact_norm = float(np.linalg.norm(orientation_error_vec_axis_angle_wxyz(quat_des, quat_cur)))
    lin_norm = float(np.linalg.norm(orientation_error_vec_wxyz(quat_des, quat_cur)))
    assert exact_norm == pytest.approx(theta, abs=1e-9)
    assert lin_norm == pytest.approx(2.0 * np.sin(half), abs=1e-9)
    # The linearized form always understates a real rotation.
    assert lin_norm <= exact_norm + 1e-12


def test_axis_angle_is_zero_and_finite_at_zero_rotation():
    q = np.array([1.0, 0.0, 0.0, 0.0])
    out = orientation_error_vec_axis_angle_wxyz(q, q)
    assert np.all(np.isfinite(out))
    np.testing.assert_array_equal(out, np.zeros(3))


def test_axis_angle_stays_finite_near_pi():
    """acos-based extraction loses precision here; atan2-based must not."""
    half = (np.pi - 1e-7) / 2.0
    quat_cur = np.array([np.cos(half), np.sin(half), 0.0, 0.0])
    out = orientation_error_vec_axis_angle_wxyz(np.array([1.0, 0.0, 0.0, 0.0]), quat_cur)
    assert np.all(np.isfinite(out))
    assert float(np.linalg.norm(out)) == pytest.approx(np.pi, abs=1e-5)


def test_exact_axis_angle_does_not_change_torque_when_kp_rot_is_zero():
    """The tuned configs run kp_rot=0 and kp_rot_wrist=0, so e_rot is multiplied
    by zero everywhere it feeds torque -- only the reported metric may move."""
    st = _state()
    out_lin = _run(_osc_cfg(), st)
    out_exact = _run(_osc_cfg(orientation_error_exact_axis_angle=True), st)
    np.testing.assert_allclose(out_lin.tau, out_exact.tau, atol=1e-12)


def test_exact_axis_angle_does_change_torque_when_wrist_gain_is_nonzero():
    """Where e_rot is actually consumed with gain, the representation matters.

    The orientation reference is captured at reset, so the arm must actually be
    rotated away from it for the error to be nonzero at all.
    """
    st = _state()
    theta = 0.6  # rad; large enough that 2*sin(theta/2) and theta visibly differ
    rotated = dict(st)
    rotated["ee_quat"] = np.array([np.cos(theta / 2.0), np.sin(theta / 2.0), 0.0, 0.0])
    kw = dict(wrist_orientation_task=True, kp_rot_wrist=30.0, kd_rot_wrist=5.0)

    def _run_rotated(cfg):
        ctrl = XAxisCartesianImpedanceController(cfg)
        ctrl.reset_from_state(st)  # reference = unrotated orientation
        return ctrl.compute(rotated)

    out_lin = _run_rotated(_osc_cfg(**kw))
    out_exact = _run_rotated(_osc_cfg(orientation_error_exact_axis_angle=True, **kw))
    # sanity: the error really is nonzero, else this test proves nothing
    assert out_lin.orientation_error_norm > 0.1
    assert not np.allclose(out_lin.tau, out_exact.tau, atol=1e-12)
    # and the exact form reports the true angle while the linearized one undershoots
    assert out_exact.orientation_error_norm > out_lin.orientation_error_norm


# --- posture inertia scaling ---------------------------------------------


def test_posture_inertia_scaling_applies_mass_matrix_before_projection():
    """tau_posture must equal the projector applied to M @ (plain PD)."""
    st = _state()
    cfg = _osc_cfg(posture_inertia_scaling=True, nullspace_posture=False)
    out = _run(cfg, st)
    q = np.asarray(st["q"], dtype=np.float64)
    qd = np.asarray(st["qd"], dtype=np.float64)
    plain = cfg.kp_posture * (q - q) - cfg.kd_posture * qd  # q_rest == q after reset_from_state
    expected = np.asarray(st["mass_matrix"], dtype=np.float64) @ plain
    np.testing.assert_allclose(out.tau_posture, expected, atol=1e-9)


def test_posture_inertia_scaling_changes_posture_torque():
    st = _state()
    out_off = _run(_osc_cfg(), st)
    out_on = _run(_osc_cfg(posture_inertia_scaling=True), st)
    assert not np.allclose(out_off.tau_posture, out_on.tau_posture, atol=1e-9)


def test_posture_inertia_scaling_is_inert_without_a_mass_matrix():
    """No mass matrix available => fall back to the plain PD rather than crash."""
    st = _state(mass=False)
    cfg = _osc_cfg(posture_inertia_scaling=True, task_space_inertia_shaping=False, nullspace_posture=False)
    cfg_off = _osc_cfg(task_space_inertia_shaping=False, nullspace_posture=False)
    out_on = _run(cfg, st)
    out_off = _run(cfg_off, st)
    np.testing.assert_allclose(out_on.tau_posture, out_off.tau_posture, atol=1e-12)


# --- task velocity saturation --------------------------------------------


def test_saturation_is_inert_below_threshold():
    """Sub-threshold behavior must be bit-identical to the unsaturated law."""
    st = _state()
    out_off = _run(_osc_cfg(), st)
    out_on = _run(
        _osc_cfg(
            task_velocity_saturation=True,
            task_vel_sat_linear=1e9,
            task_vel_sat_angular=1e9,
        ),
        st,
    )
    np.testing.assert_array_equal(out_off.tau, out_on.tau)


def test_saturation_caps_the_linear_block_and_preserves_direction():
    st = _state()
    out_off = _run(_osc_cfg(), st)
    cap = 1.0
    out_on = _run(
        _osc_cfg(task_velocity_saturation=True, task_vel_sat_linear=cap),
        st,
    )
    w_off = np.asarray(out_off.wrench, dtype=np.float64)
    w_on = np.asarray(out_on.wrench, dtype=np.float64)
    off_norm = float(np.linalg.norm(w_off[:3]))
    assert off_norm > cap, "test precondition: unsaturated wrench must exceed the cap"
    assert float(np.linalg.norm(w_on[:3])) == pytest.approx(cap, rel=1e-9)
    # direction preserved
    np.testing.assert_allclose(
        w_on[:3] / np.linalg.norm(w_on[:3]), w_off[:3] / off_norm, atol=1e-9
    )


def test_zero_threshold_disables_that_block_rather_than_zeroing_it():
    """A 0.0 threshold means 'off', not 'saturate to zero' -- guards a footgun."""
    st = _state()
    out_off = _run(_osc_cfg(), st)
    out_on = _run(
        _osc_cfg(task_velocity_saturation=True, task_vel_sat_linear=0.0, task_vel_sat_angular=0.0),
        st,
    )
    np.testing.assert_array_equal(out_off.tau, out_on.tau)


# --- yaml plumbing --------------------------------------------------------


def test_flags_round_trip_through_controller_yaml_section():
    cfg = CartesianImpedanceConfig.from_controller_yaml_section(
        {
            **_LIMITS,
            "posture_inertia_scaling": True,
            "task_velocity_saturation": True,
            "task_vel_sat_linear": 12.5,
            "task_vel_sat_angular": 3.5,
            "orientation_error_exact_axis_angle": True,
        }
    )
    assert cfg.posture_inertia_scaling is True
    assert cfg.task_velocity_saturation is True
    assert cfg.task_vel_sat_linear == 12.5
    assert cfg.task_vel_sat_angular == 3.5
    assert cfg.orientation_error_exact_axis_angle is True


def test_yaml_defaults_are_off_when_keys_absent():
    cfg = CartesianImpedanceConfig.from_controller_yaml_section(dict(_LIMITS))
    assert cfg.posture_inertia_scaling is False
    assert cfg.task_velocity_saturation is False
    assert cfg.orientation_error_exact_axis_angle is False
