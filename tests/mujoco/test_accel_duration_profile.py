"""Tests for the accel/duration-driven trajectory profiles (2026-07-31) --
`x_profile_target(profile="accel_duration_triangular" | "accel_duration_scurve", ...)`
in simulation/ur5e_mujoco_torque.py. Pure closed-form kinematics; no MuJoCo
model is loaded, but the module itself imports mujoco at load time, hence
this lives under tests/mujoco/ per this repo's directory-based marker
convention (see tests/conftest.py), not tests/unit/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from simulation.ur5e_mujoco_torque import (  # noqa: E402
    accel_duration_displacement,
    x_profile_accel,
    x_profile_target,
)

PROFILES = ("accel_duration_triangular", "accel_duration_scurve")


@pytest.mark.mujoco
@pytest.mark.parametrize("profile", PROFILES)
def test_starts_and_ends_at_rest(profile):
    accel, T = 0.4, 2.0
    x_start, v_start = x_profile_target(profile, 0.0, 0.0, 0.0, T, move_duration_s=T, target_accel_mps2=accel)
    assert x_start == pytest.approx(0.0, abs=1e-9)
    assert v_start == pytest.approx(0.0, abs=1e-9)
    x_end, v_end = x_profile_target(profile, 0.0, 0.0, T, T, move_duration_s=T, target_accel_mps2=accel)
    assert v_end == pytest.approx(0.0, abs=1e-6)
    expected_delta = accel_duration_displacement(profile, accel, T)
    assert x_end == pytest.approx(expected_delta, rel=1e-6)


@pytest.mark.mujoco
@pytest.mark.parametrize("profile", PROFILES)
def test_holds_after_move_duration(profile):
    accel, T, total = 0.4, 2.0, 5.0
    x_end, v_end = x_profile_target(profile, 1.0, 0.0, T, total, move_duration_s=T, target_accel_mps2=accel)
    x_hold, v_hold = x_profile_target(profile, 1.0, 0.0, 4.0, total, move_duration_s=T, target_accel_mps2=accel)
    assert x_hold == pytest.approx(x_end, abs=1e-9)
    assert v_hold == 0.0
    assert v_end == pytest.approx(0.0, abs=1e-6)


@pytest.mark.mujoco
def test_triangular_matches_closed_form_kinematics():
    accel, T = 0.5, 4.0
    # Peak velocity at the midpoint: v = a*T/2.
    _, v_mid = x_profile_target("accel_duration_triangular", 0.0, 0.0, T / 2.0, T, move_duration_s=T, target_accel_mps2=accel)
    assert v_mid == pytest.approx(accel * T / 2.0, rel=1e-9)
    # Displacement at the midpoint: x = a*(T/2)^2/2 = a*T^2/8.
    x_mid, _ = x_profile_target("accel_duration_triangular", 0.0, 0.0, T / 2.0, T, move_duration_s=T, target_accel_mps2=accel)
    assert x_mid == pytest.approx(accel * T * T / 8.0, rel=1e-9)
    # Total displacement: x = a*T^2/4.
    x_end, _ = x_profile_target("accel_duration_triangular", 0.0, 0.0, T, T, move_duration_s=T, target_accel_mps2=accel)
    assert x_end == pytest.approx(accel * T * T / 4.0, rel=1e-9)
    assert accel_duration_displacement("accel_duration_triangular", accel, T) == pytest.approx(accel * T * T / 4.0)


@pytest.mark.mujoco
def test_scurve_matches_closed_form_kinematics():
    accel, T = 0.5, 4.0
    # Peak velocity at the midpoint: v = accel*(1-cos(pi)) = 2*accel.
    _, v_mid = x_profile_target("accel_duration_scurve", 0.0, 0.0, T / 2.0, T, move_duration_s=T, target_accel_mps2=accel)
    assert v_mid == pytest.approx(2.0 * accel, rel=1e-9)
    # Total displacement: x = accel*T^2/(2*pi).
    x_end, _ = x_profile_target("accel_duration_scurve", 0.0, 0.0, T, T, move_duration_s=T, target_accel_mps2=accel)
    expected = accel * T * T / (2.0 * np.pi)
    assert x_end == pytest.approx(expected, rel=1e-9)
    assert accel_duration_displacement("accel_duration_scurve", accel, T) == pytest.approx(expected)


@pytest.mark.mujoco
def test_scurve_has_no_velocity_discontinuity_at_move_duration_boundary():
    """Unlike the triangular profile (whose acceleration jumps discontinuously
    at t=0, T/2, T), the s-curve's velocity derivative (acceleration) is
    continuous at the move->hold transition: a(T) = accel*sin(2*pi) = 0, so
    velocity approaches zero smoothly rather than the triangular profile's
    abrupt accel-to-zero switch at exactly t=T."""
    accel, T = 0.5, 4.0
    eps = 1e-4
    _, v_before = x_profile_target("accel_duration_scurve", 0.0, 0.0, T - eps, T, move_duration_s=T, target_accel_mps2=accel)
    _, v_after = x_profile_target("accel_duration_scurve", 0.0, 0.0, T + eps, T, move_duration_s=T, target_accel_mps2=accel)
    assert abs(v_before - v_after) < 1e-3


@pytest.mark.mujoco
@pytest.mark.parametrize("profile", PROFILES)
def test_negative_accel_moves_in_negative_direction(profile):
    accel, T = -0.3, 2.0
    x_end, _ = x_profile_target(profile, 0.0, 0.0, T, T, move_duration_s=T, target_accel_mps2=accel)
    assert x_end < 0.0
    assert x_end == pytest.approx(accel_duration_displacement(profile, accel, T), rel=1e-6)


@pytest.mark.mujoco
@pytest.mark.parametrize("profile", PROFILES)
def test_requires_target_accel_mps2(profile):
    with pytest.raises(ValueError, match="target_accel_mps2"):
        x_profile_target(profile, 0.0, 0.0, 0.0, 2.0, move_duration_s=2.0)


@pytest.mark.mujoco
@pytest.mark.parametrize("profile", PROFILES)
def test_requires_move_duration_s(profile):
    with pytest.raises(ValueError, match="move_duration_s"):
        x_profile_target(profile, 0.0, 0.0, 0.0, 2.0, target_accel_mps2=0.3)


@pytest.mark.mujoco
@pytest.mark.parametrize("profile", PROFILES)
def test_move_duration_cannot_exceed_total_duration(profile):
    with pytest.raises(ValueError, match="move_duration_s"):
        x_profile_target(profile, 0.0, 0.0, 0.0, 1.0, move_duration_s=2.0, target_accel_mps2=0.3)


def test_accel_duration_displacement_rejects_unknown_profile():
    with pytest.raises(ValueError, match="Unsupported"):
        accel_duration_displacement("not_a_real_profile", 0.3, 2.0)


# ---------------------------------------------------------------------------
# x_profile_accel (2026-08-01) -- closed-form reference acceleration for
# controller_core's acceleration_feedforward flag. See that function's own
# docstring for why it is defined as the derivative of the VELOCITY
# x_profile_target returns (not a fresh re-derivation from position), and for
# the scurve profile specifically where those two are not the same thing.
# ---------------------------------------------------------------------------


def _finite_diff_accel(profile, x0, target_x_delta, t_s, duration_s, *, move_duration_s=None, target_accel_mps2=None, eps=1e-6):
    """Central-difference derivative of x_profile_target's own velocity, as an
    independent cross-check of x_profile_accel's closed form."""
    _, v_plus = x_profile_target(profile, x0, target_x_delta, t_s + eps, duration_s, move_duration_s=move_duration_s, target_accel_mps2=target_accel_mps2)
    _, v_minus = x_profile_target(profile, x0, target_x_delta, t_s - eps, duration_s, move_duration_s=move_duration_s, target_accel_mps2=target_accel_mps2)
    return (v_plus - v_minus) / (2.0 * eps)


@pytest.mark.mujoco
def test_accel_step_and_ramp_are_zero():
    assert x_profile_accel("step", 0.05, 0.5, 2.0) == 0.0
    assert x_profile_accel("ramp", 0.05, 0.5, 2.0) == 0.0


@pytest.mark.mujoco
def test_accel_min_jerk_matches_finite_difference():
    delta, T = 0.05, 2.0
    for t in (0.3, 0.9, 1.4, 1.8):
        expected = _finite_diff_accel("min_jerk", 0.0, delta, t, T)
        actual = x_profile_accel("min_jerk", delta, t, T)
        assert actual == pytest.approx(expected, rel=1e-3, abs=1e-6)
    # Outside [0, duration_s): zero, matching x_profile_target's own velocity
    # boundary treatment (0 outside [0, duration_s)).
    assert x_profile_accel("min_jerk", delta, T, T) == 0.0
    assert x_profile_accel("min_jerk", delta, T + 0.5, T) == 0.0


@pytest.mark.mujoco
def test_accel_min_jerk_move_hold_matches_finite_difference_and_holds_zero():
    delta, move_T, total = 0.05, 0.5, 1.0
    for t in (0.1, 0.3, 0.45):
        expected = _finite_diff_accel("min_jerk_move_hold", 0.0, delta, t, total, move_duration_s=move_T)
        actual = x_profile_accel("min_jerk_move_hold", delta, t, total, move_duration_s=move_T)
        assert actual == pytest.approx(expected, rel=1e-3, abs=1e-6)
    # During the hold phase (t_s >= move_duration_s): zero.
    assert x_profile_accel("min_jerk_move_hold", delta, move_T, total, move_duration_s=move_T) == 0.0
    assert x_profile_accel("min_jerk_move_hold", delta, 0.9, total, move_duration_s=move_T) == 0.0


@pytest.mark.mujoco
def test_accel_triangular_matches_bang_bang_sign():
    accel, T = 0.5, 4.0
    half_t = T / 2.0
    assert x_profile_accel("accel_duration_triangular", 0.0, half_t - 0.5, T, move_duration_s=T, target_accel_mps2=accel) == pytest.approx(accel)
    assert x_profile_accel("accel_duration_triangular", 0.0, half_t + 0.5, T, move_duration_s=T, target_accel_mps2=accel) == pytest.approx(-accel)
    # Hold phase (strictly after move_duration_s): zero.
    assert x_profile_accel("accel_duration_triangular", 0.0, T + 0.1, T, move_duration_s=T, target_accel_mps2=accel) == 0.0
    # Cross-check against a finite-difference of x_profile_target's own
    # velocity away from the t=0/half_t/T kinks (where a central difference
    # straddling a discontinuity would be misleading).
    for t in (0.5, 1.5, 2.5, 3.5):
        expected = _finite_diff_accel("accel_duration_triangular", 0.0, 0.0, t, T, move_duration_s=T, target_accel_mps2=accel)
        actual = x_profile_accel("accel_duration_triangular", 0.0, t, T, move_duration_s=T, target_accel_mps2=accel)
        assert actual == pytest.approx(expected, rel=1e-3, abs=1e-4)


@pytest.mark.mujoco
def test_accel_scurve_matches_finite_difference_and_is_continuous_at_boundary():
    accel, T = 0.5, 4.0
    omega = 2.0 * np.pi / T
    # Closed form: a(t) = accel*omega*sin(omega*t) -- see x_profile_accel's
    # own docstring for why this (not the simpler accel*sin(2*pi*t/T)) is the
    # form consistent with the velocity x_profile_target actually returns.
    for t in (0.3, 1.0, 2.0, 3.0, 3.9):
        expected_closed_form = accel * omega * np.sin(omega * t)
        actual = x_profile_accel("accel_duration_scurve", 0.0, t, T, move_duration_s=T, target_accel_mps2=accel)
        assert actual == pytest.approx(expected_closed_form, rel=1e-9)
        expected_fd = _finite_diff_accel("accel_duration_scurve", 0.0, 0.0, t, T, move_duration_s=T, target_accel_mps2=accel)
        assert actual == pytest.approx(expected_fd, rel=1e-3, abs=1e-4)
    # a(T) = accel*omega*sin(2*pi) = 0, matching the hold phase's own zero --
    # i.e. acceleration (not just velocity) is continuous across the
    # move->hold boundary for this profile, unlike the triangular one above.
    assert x_profile_accel("accel_duration_scurve", 0.0, T, T, move_duration_s=T, target_accel_mps2=accel) == pytest.approx(0.0, abs=1e-9)
    assert x_profile_accel("accel_duration_scurve", 0.0, T + 0.1, T, move_duration_s=T, target_accel_mps2=accel) == 0.0


@pytest.mark.mujoco
@pytest.mark.parametrize("profile", PROFILES)
def test_accel_negative_target_accel_flips_sign(profile):
    accel, T = -0.4, 3.0
    # Sample away from the triangular midpoint/boundary kinks.
    t = T * 0.2
    value = x_profile_accel(profile, 0.0, t, T, move_duration_s=T, target_accel_mps2=accel)
    assert value < 0.0
    flipped = x_profile_accel(profile, 0.0, t, T, move_duration_s=T, target_accel_mps2=-accel)
    assert flipped == pytest.approx(-value)


@pytest.mark.mujoco
@pytest.mark.parametrize("profile", PROFILES)
def test_accel_requires_target_accel_mps2(profile):
    with pytest.raises(ValueError, match="target_accel_mps2"):
        x_profile_accel(profile, 0.0, 0.0, 2.0, move_duration_s=2.0)


@pytest.mark.mujoco
@pytest.mark.parametrize("profile", PROFILES)
def test_accel_requires_move_duration_s(profile):
    with pytest.raises(ValueError, match="move_duration_s"):
        x_profile_accel(profile, 0.0, 0.0, 2.0, target_accel_mps2=0.3)


def test_accel_min_jerk_move_hold_requires_move_duration_s():
    with pytest.raises(ValueError, match="move_duration_s"):
        x_profile_accel("min_jerk_move_hold", 0.05, 0.0, 2.0)


def test_accel_rejects_unknown_profile():
    with pytest.raises(ValueError, match="Unsupported"):
        x_profile_accel("not_a_real_profile", 0.05, 0.0, 2.0)
