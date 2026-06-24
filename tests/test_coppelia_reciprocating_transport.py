"""Unit tests for CoppeliaSim reciprocating transport references."""

from __future__ import annotations

import math

import pytest

from simulation.coppelia_reciprocating_transport import (
    build_reciprocating_plan,
    minimum_run_duration_s,
    point_to_point_accel_reference,
    reciprocating_axis_reference,
    reciprocating_ik_task_weights,
    slew_axis_reference,
)


def test_point_to_point_reaches_target_and_stops() -> None:
    offset, vel, acc, total = point_to_point_accel_reference(
        t_move=1.0e9,
        target_dx=0.04,
        a_abs=0.2,
        v_abs=0.1,
    )
    assert total > 0.0
    assert math.isclose(offset, 0.04, abs_tol=1.0e-9)
    assert math.isclose(vel, 0.0, abs_tol=1.0e-9)
    assert math.isclose(acc, 0.0, abs_tol=1.0e-9)


def test_reciprocating_plan_visits_endpoints() -> None:
    plan = build_reciprocating_plan(
        stroke_m=0.03,
        a_abs_m_s2=0.12,
        v_abs_m_s=0.06,
        hold_s=0.1,
    )
    assert plan.motion_duration_s > 0.0
    samples = [
        reciprocating_axis_reference(t, plan)
        for t in [0.0, plan.motion_duration_s * 0.2, plan.motion_duration_s * 0.5]
    ]
    assert samples[0][0] == pytest.approx(0.0, abs=1.0e-6)
    max_offset = max(abs(s[0]) for s in samples)
    assert max_offset > 0.01


def test_reciprocating_returns_to_origin() -> None:
    plan = build_reciprocating_plan(
        stroke_m=0.02,
        a_abs_m_s2=0.15,
        v_abs_m_s=0.08,
        hold_s=0.0,
    )
    offset, vel, acc, phase, completed = reciprocating_axis_reference(
        plan.motion_duration_s,
        plan,
    )
    assert completed
    assert math.isclose(offset, 0.0, abs_tol=1.0e-6)
    assert math.isclose(vel, 0.0, abs_tol=1.0e-6)
    assert phase in {"return_to_origin", "hold_origin"}


def test_slew_axis_reference_limits_step() -> None:
    out = slew_axis_reference(0.0, 0.05, dt=0.05, max_step_m=0.002, max_velocity_mps=1.0)
    assert out == pytest.approx(0.002, abs=1.0e-9)


def test_reciprocating_ik_task_weights_positive() -> None:
    weights = reciprocating_ik_task_weights()
    assert weights["hold_axis_weight"] > weights["move_axis_weight"]


def test_minimum_run_duration_includes_settle() -> None:
    plan = build_reciprocating_plan(
        stroke_m=0.03,
        a_abs_m_s2=0.12,
        v_abs_m_s=0.06,
        hold_s=0.25,
    )
    duration = minimum_run_duration_s(plan, settle_duration_s=1.0, tail_hold_s=0.5)
    assert duration >= 1.0 + plan.motion_duration_s
