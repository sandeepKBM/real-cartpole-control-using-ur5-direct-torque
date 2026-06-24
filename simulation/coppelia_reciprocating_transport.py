"""Acceleration-limited reciprocating end-effector transport along one world axis."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


SegmentKind = Literal["move", "hold"]


@dataclass(frozen=True)
class ReciprocatingSegment:
    kind: SegmentKind
    t_start: float
    t_end: float
    start_offset_m: float
    end_offset_m: float
    distance_m: float
    phase: str


@dataclass(frozen=True)
class ReciprocatingPlan:
    stroke_m: float
    hold_s: float
    a_abs_m_s2: float
    v_abs_m_s: float
    segments: tuple[ReciprocatingSegment, ...]

    @property
    def motion_duration_s(self) -> float:
        if not self.segments:
            return 0.0
        return float(self.segments[-1].t_end)

    @property
    def expected_span_m(self) -> float:
        return 2.0 * abs(float(self.stroke_m))


def point_to_point_accel_reference(
    t_move: float,
    target_dx: float,
    a_abs: float,
    v_abs: float,
) -> tuple[float, float, float, float]:
    """Return (offset, velocity, accel, total_time) for a bounded 1D move."""
    distance = abs(float(target_dx))
    if distance <= 0.0:
        return 0.0, 0.0, 0.0, 0.0
    direction = 1.0 if target_dx >= 0.0 else -1.0
    a = max(abs(float(a_abs)), 1.0e-9)
    v_cap = max(abs(float(v_abs)), 1.0e-9)
    t_accel_cap = v_cap / a
    d_accel_cap = 0.5 * a * t_accel_cap * t_accel_cap
    if 2.0 * d_accel_cap >= distance:
        t_accel = math.sqrt(distance / a)
        v_peak = a * t_accel
        t_flat = 0.0
        d_accel = 0.5 * a * t_accel * t_accel
    else:
        t_accel = t_accel_cap
        v_peak = v_cap
        d_accel = d_accel_cap
        t_flat = (distance - 2.0 * d_accel) / v_peak
    total = 2.0 * t_accel + t_flat
    t = max(float(t_move), 0.0)
    if t <= 0.0:
        s, v, acc = 0.0, 0.0, a
    elif t < t_accel:
        s = 0.5 * a * t * t
        v = a * t
        acc = a
    elif t < t_accel + t_flat:
        tau = t - t_accel
        s = d_accel + v_peak * tau
        v = v_peak
        acc = 0.0
    elif t < total:
        tau = total - t
        s = distance - 0.5 * a * tau * tau
        v = a * tau
        acc = -a
    else:
        s, v, acc = distance, 0.0, 0.0
    return direction * s, direction * v, direction * acc, total


def build_reciprocating_plan(
    *,
    stroke_m: float,
    a_abs_m_s2: float,
    v_abs_m_s: float,
    hold_s: float = 0.25,
) -> ReciprocatingPlan:
    """
    Plan origin -> +stroke -> -stroke -> origin with optional endpoint holds.

    Offsets are relative to the transport start position on the selected axis.
    """
    stroke = abs(float(stroke_m))
    hold = max(float(hold_s), 0.0)
    a_abs = max(abs(float(a_abs_m_s2)), 1.0e-9)
    v_abs = max(abs(float(v_abs_m_s)), 1.0e-9)

    waypoints: list[tuple[float, str]] = [
        (+stroke, "to_positive_end"),
        (+stroke, "hold_positive_end"),
        (-stroke, "to_negative_end"),
        (-stroke, "hold_negative_end"),
        (0.0, "return_to_origin"),
        (0.0, "hold_origin"),
    ]

    segments: list[ReciprocatingSegment] = []
    t_cursor = 0.0
    pos = 0.0
    for target_pos, phase in waypoints:
        if abs(target_pos - pos) <= 1.0e-12:
            if hold <= 0.0:
                continue
            segments.append(
                ReciprocatingSegment(
                    kind="hold",
                    t_start=t_cursor,
                    t_end=t_cursor + hold,
                    start_offset_m=pos,
                    end_offset_m=pos,
                    distance_m=0.0,
                    phase=phase,
                )
            )
            t_cursor += hold
            continue

        _, _, _, move_time = point_to_point_accel_reference(
            1.0e9,
            target_pos - pos,
            a_abs,
            v_abs,
        )
        segments.append(
            ReciprocatingSegment(
                kind="move",
                t_start=t_cursor,
                t_end=t_cursor + move_time,
                start_offset_m=pos,
                end_offset_m=target_pos,
                distance_m=abs(target_pos - pos),
                phase=phase,
            )
        )
        t_cursor += move_time
        pos = target_pos

    return ReciprocatingPlan(
        stroke_m=stroke,
        hold_s=hold,
        a_abs_m_s2=a_abs,
        v_abs_m_s=v_abs,
        segments=tuple(segments),
    )


def reciprocating_axis_reference(
    t_move: float,
    plan: ReciprocatingPlan,
) -> tuple[float, float, float, str, bool]:
    """
    Sample the reciprocating plan at ``t_move`` seconds after motion start.

    Returns (offset_m, velocity_mps, accel_mps2, phase_name, completed).
    """
    t = max(float(t_move), 0.0)
    if not plan.segments:
        return 0.0, 0.0, 0.0, "idle", True

    for segment in plan.segments:
        if t < segment.t_start:
            break
        if t <= segment.t_end:
            if segment.kind == "hold":
                return (
                    float(segment.end_offset_m),
                    0.0,
                    0.0,
                    segment.phase,
                    t >= plan.motion_duration_s,
                )
            t_seg = t - segment.t_start
            direction = 1.0 if segment.end_offset_m >= segment.start_offset_m else -1.0
            offset, vel, acc, _ = point_to_point_accel_reference(
                t_seg,
                direction * segment.distance_m,
                plan.a_abs_m_s2,
                plan.v_abs_m_s,
            )
            return (
                float(segment.start_offset_m + offset),
                float(vel),
                float(acc),
                segment.phase,
                t >= plan.motion_duration_s,
            )

    last = plan.segments[-1]
    completed = t >= plan.motion_duration_s
    return float(last.end_offset_m), 0.0, 0.0, last.phase, completed


def minimum_run_duration_s(
    plan: ReciprocatingPlan,
    *,
    settle_duration_s: float,
    tail_hold_s: float = 0.5,
) -> float:
    return float(settle_duration_s) + plan.motion_duration_s + max(float(tail_hold_s), 0.0)


def reciprocating_ik_task_weights() -> dict[str, float]:
    """Stronger orthogonal/orientation weights for reciprocating IK transport."""
    return {
        "move_axis_weight": 90.0,
        "hold_axis_weight": 320.0,
        "orientation_weight": 160.0,
        "hold_axis_gain": 22.0,
    }


def slew_axis_reference(
    prev_offset_m: float,
    target_offset_m: float,
    *,
    dt: float,
    max_step_m: float,
    max_velocity_mps: float,
) -> float:
    """Rate-limit a scalar axis position reference."""
    dt = max(float(dt), 1.0e-6)
    delta = float(target_offset_m) - float(prev_offset_m)
    max_delta = min(abs(float(max_step_m)), abs(float(max_velocity_mps)) * dt)
    if abs(delta) <= max_delta:
        return float(target_offset_m)
    return float(prev_offset_m + math.copysign(max_delta, delta))
