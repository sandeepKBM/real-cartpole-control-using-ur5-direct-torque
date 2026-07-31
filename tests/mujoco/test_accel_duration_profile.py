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

from simulation.ur5e_mujoco_torque import accel_duration_displacement, x_profile_target  # noqa: E402

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
