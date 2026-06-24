"""Compute feasible fast transport limits from Jacobian capability and safety caps."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from simulation.coppelia_reciprocating_transport import point_to_point_accel_reference


def damped_axis_velocity_capability(
    jacobian: np.ndarray,
    transport_axis_index: int,
    *,
    max_joint_velocity_radps: float,
    joint_speed_fraction: float = 0.82,
    damping_ratio: float = 0.05,
) -> dict[str, Any]:
    """Estimate axis speed from a damped least-squares joint-velocity solve."""
    J = np.asarray(jacobian, dtype=np.float64).reshape(6, 6)
    axis_idx = int(transport_axis_index)
    qd_limit = float(joint_speed_fraction) * float(max_joint_velocity_radps)
    twist = np.zeros(6, dtype=np.float64)
    twist[axis_idx] = 1.0
    jjt = J @ J.T
    damp = float(damping_ratio) * float(np.trace(jjt) / 6.0)
    qdot_unit = J.T @ np.linalg.solve(jjt + damp * np.eye(6), twist)
    predicted = J @ qdot_unit
    max_abs_qdot = float(np.max(np.abs(qdot_unit)))
    v_axis_unit = float(predicted[axis_idx])
    finite = bool(
        np.all(np.isfinite(qdot_unit))
        and np.all(np.isfinite(predicted))
        and max_abs_qdot > 1.0e-9
        and abs(v_axis_unit) > 1.0e-9
    )
    joint_scale = qd_limit / max_abs_qdot if finite else 0.0
    return {
        "finite": finite,
        "qdot_for_unit_transport_axis": qdot_unit.tolist(),
        "predicted_axis_velocity_mps_for_unit_cmd": v_axis_unit,
        "max_abs_qdot_for_unit_cmd": max_abs_qdot,
        "joint_velocity_scale": float(joint_scale),
        "damping": float(damp),
        "source": "damped_jacobian",
    }


def recommend_fast_point_to_point_limits(
    distance_m: float,
    capability: dict[str, Any],
    *,
    max_joint_velocity_radps: float,
    max_acceleration_mps2: float,
    joint_speed_fraction: float = 0.82,
    accel_fraction: float = 0.78,
    min_v_mps: float = 0.01,
    min_a_mps2: float = 0.02,
    jacobian: np.ndarray | None = None,
    transport_axis_index: int = 0,
) -> tuple[float, float, dict[str, Any]]:
    """
  Return ``(a_max, v_max, diagnostics)`` for a bang-bang / trapezoidal point-to-point move.

  When ``jacobian`` is provided, limits are derived from a damped least-squares solve
  instead of the raw capability ``qdot`` vector (which can be ill-conditioned with
  numerical Jacobians).
    """
    if jacobian is not None:
        capability = damped_axis_velocity_capability(
            jacobian,
            transport_axis_index,
            max_joint_velocity_radps=max_joint_velocity_radps,
            joint_speed_fraction=joint_speed_fraction,
        )
        J = np.asarray(jacobian, dtype=np.float64).reshape(6, 6)
        capability["task_condition_number"] = float(np.linalg.cond(J))

    distance = abs(float(distance_m))
    max_joint_velocity_radps = max(float(max_joint_velocity_radps), 1.0e-6)
    max_acceleration_mps2 = max(float(max_acceleration_mps2), 1.0e-6)
    joint_speed_fraction = float(np.clip(joint_speed_fraction, 0.05, 1.0))
    accel_fraction = float(np.clip(accel_fraction, 0.05, 1.0))

    qdot_unit = np.asarray(
        capability.get("qdot_for_unit_transport_axis", np.zeros(6)),
        dtype=np.float64,
    ).reshape(6)
    max_abs_qdot = float(capability.get("max_abs_qdot_for_unit_cmd", 0.0))
    v_axis_unit = float(capability.get("predicted_axis_velocity_mps_for_unit_cmd", 0.0))
    finite = bool(capability.get("finite", False))

    diag: dict[str, Any] = {
        "distance_m": distance,
        "max_abs_qdot_for_unit_cmd": max_abs_qdot,
        "predicted_axis_velocity_mps_for_unit_cmd": v_axis_unit,
        "finite": finite,
        "joint_speed_fraction": joint_speed_fraction,
        "accel_fraction": accel_fraction,
        "capability_source": capability.get("source", "estimate_axis_transport_capability"),
    }

    if not finite or max_abs_qdot <= 1.0e-9 or abs(v_axis_unit) <= 1.0e-9:
        diag["reason"] = "degenerate_capability"
        return float(min_a_mps2), float(min_v_mps), diag

    joint_scale = joint_speed_fraction * max_joint_velocity_radps / max_abs_qdot
    v_max = max(float(min_v_mps), abs(v_axis_unit) * joint_scale)
    cond = float(capability.get("task_condition_number", 0.0))
    if cond > 5000.0:
        cond_scale = min(1.0, math.sqrt(5000.0 / cond))
        v_max *= cond_scale
        diag["condition_number_scale"] = float(cond_scale)
    diag["task_condition_number"] = cond if cond > 0.0 else None
    diag["joint_velocity_scale"] = float(joint_scale)

    if distance <= 1.0e-6:
        a_max = min(max_acceleration_mps2, max(float(min_a_mps2), 0.5 * v_max))
        diag["profile"] = "velocity_limited_short_move"
        return float(a_max), float(v_max), diag

    a_for_full_speed = (v_max * v_max) / distance
    a_max = min(max_acceleration_mps2, max(float(min_a_mps2), accel_fraction * a_for_full_speed))
    if a_max < a_for_full_speed:
        v_max = max(float(min_v_mps), math.sqrt(max(a_max, 1.0e-9) * distance) * 0.98)
        diag["profile"] = "accel_or_distance_limited"
    else:
        diag["profile"] = "velocity_limited"
    diag["a_for_full_speed_mps2"] = float(a_for_full_speed)
    diag["recommended_a_max_mps2"] = float(a_max)
    diag["recommended_v_max_mps"] = float(v_max)
    _, _, _, move_time = point_to_point_accel_reference(
        0.0,
        distance,
        a_max,
        v_max,
    )
    diag["estimated_move_time_s"] = float(move_time)
    return float(a_max), float(v_max), diag


def minimum_fast_run_duration_s(
    distance_m: float,
    a_abs_m_s2: float,
    v_abs_m_s: float,
    *,
    settle_duration_s: float,
    tail_hold_s: float = 0.35,
) -> float:
    _, _, _, move_time = point_to_point_accel_reference(
        0.0,
        abs(float(distance_m)),
        abs(float(a_abs_m_s2)),
        abs(float(v_abs_m_s)),
    )
    return float(settle_duration_s) + float(move_time) + max(float(tail_hold_s), 0.0)
