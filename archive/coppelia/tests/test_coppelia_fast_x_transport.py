"""Unit tests for limit-aware fast X transport planning."""

from __future__ import annotations

import math

import numpy as np

from simulation.coppelia_fast_x_transport import (
    minimum_fast_run_duration_s,
    recommend_fast_point_to_point_limits,
)


def _dummy_capability(v_axis: float = 0.25, max_qdot: float = 2.0) -> dict:
    qdot = np.array([1.0, 0.5, 0.2, 0.1, 0.05, 0.02], dtype=np.float64)
    qdot = qdot / max(np.max(np.abs(qdot)), 1.0e-9) * max_qdot
    return {
        "finite": True,
        "qdot_for_unit_transport_axis": qdot.tolist(),
        "predicted_axis_velocity_mps_for_unit_cmd": float(v_axis),
        "max_abs_qdot_for_unit_cmd": float(max_qdot),
    }


def test_recommend_fast_limits_respects_joint_velocity_cap() -> None:
    a_max, v_max, diag = recommend_fast_point_to_point_limits(
        0.05,
        _dummy_capability(v_axis=0.30, max_qdot=2.0),
        max_joint_velocity_radps=4.0,
        max_acceleration_mps2=1.0,
        joint_speed_fraction=0.5,
        accel_fraction=1.0,
    )
    assert diag["finite"] is True
    assert v_max > 0.0
    assert a_max > 0.0
    assert v_max <= 0.30 * (0.5 * 4.0 / 2.0) + 1.0e-6


def test_recommend_fast_limits_distance_limited_reduces_velocity() -> None:
    a_max, v_max, diag = recommend_fast_point_to_point_limits(
        0.002,
        _dummy_capability(v_axis=0.40, max_qdot=1.5),
        max_joint_velocity_radps=6.0,
        max_acceleration_mps2=0.08,
        joint_speed_fraction=0.9,
        accel_fraction=0.9,
    )
    assert diag["profile"] in {"accel_or_distance_limited", "velocity_limited_short_move"}
    assert v_max <= math.sqrt(a_max * 0.002) + 0.02
    assert a_max <= 0.08 + 1.0e-9


def test_minimum_fast_run_duration_includes_settle() -> None:
    duration = minimum_fast_run_duration_s(
        0.04,
        a_abs_m_s2=0.20,
        v_abs_m_s=0.10,
        settle_duration_s=2.0,
        tail_hold_s=0.25,
    )
    assert duration >= 2.25
