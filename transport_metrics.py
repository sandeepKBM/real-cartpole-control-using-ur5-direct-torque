"""Shared scoring and ranking helpers for MuJoCo UR5e transport runs."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from controller_core.kinematics_utils import orientation_error_vec_wxyz

GAIN_FIELDS: tuple[str, ...] = (
    "kp_x",
    "kd_x",
    "kp_y",
    "kd_y",
    "kp_z",
    "kd_z",
    "kp_rot",
    "kd_rot",
    "kp_posture",
    "kd_posture",
    "kd_joint",
)

_DEFAULT_GAIN_VALUES: dict[str, float] = {
    "kp_x": 25.0,
    "kd_x": 8.0,
    "kp_y": 80.0,
    "kd_y": 15.0,
    "kp_z": 120.0,
    "kd_z": 20.0,
    "kp_rot": 20.0,
    "kd_rot": 5.0,
    "kp_posture": 2.0,
    "kd_posture": 0.5,
    "kd_joint": 0.8,
}

_NORMAL_X_TOL_ABS = 0.005
_NORMAL_X_TOL_REL = 0.25
_STRICT_X_TOL_ABS = 0.003
_STRICT_X_TOL_REL = 0.15

_NORMAL_ORIENTATION_TOL = 0.25
_STRICT_ORIENTATION_TOL = 0.15

_NORMAL_YZ_DRIFT_TOL = 0.03
_STRICT_YZ_DRIFT_TOL = 0.02

_TORQUE_SAT_TOL_PERCENT = 5.0
_STRICT_TORQUE_SAT_TOL_PERCENT = 2.0

_QD_LIMIT_DEFAULT = 3.0


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off", ""}:
            return False
    return bool(default)


def _as_array(value: Any, *, shape: tuple[int, ...] | None = None, default: float = 0.0) -> np.ndarray:
    if value is None:
        arr = np.asarray([], dtype=np.float64)
    else:
        arr = np.asarray(value, dtype=np.float64)
    if arr.size == 0:
        arr = np.full(shape if shape is not None else (0,), float(default), dtype=np.float64)
    if shape is not None:
        arr = arr.reshape(shape)
    return arr.astype(np.float64, copy=False)


def _load_trace_rows(trace_path: str | Path | None) -> list[dict[str, Any]]:
    if trace_path is None:
        return []
    path = Path(trace_path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            row = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def controller_gain_summary(ctrl_cfg: Mapping[str, Any] | None) -> dict[str, Any]:
    gains = (ctrl_cfg or {}).get("gains", {}) or {}
    summary = {name: _as_float(gains.get(name, _DEFAULT_GAIN_VALUES[name])) for name in GAIN_FIELDS}
    return {"controller_gains": summary, **summary}


def summarize_transport_trace(
    trace_rows: Sequence[Mapping[str, Any]],
    *,
    initial_ee_pos: Sequence[float] | None = None,
    transport_axis_index: int = 0,
) -> dict[str, Any]:
    if not trace_rows:
        return {
            "final_x_error_m": 0.0,
            "achieved_x_delta_m": 0.0,
            "final_x_displacement_m": 0.0,
            "max_abs_x_error_m": 0.0,
            "max_abs_y_drift_m": 0.0,
            "max_abs_z_drift_m": 0.0,
            "max_abs_orthogonal_drift_m": 0.0,
            "final_orientation_error_rad": 0.0,
        }

    first_row = trace_rows[0]
    start_pos = (
        _as_array(initial_ee_pos, shape=(3,), default=0.0)
        if initial_ee_pos is not None
        else _as_array(first_row.get("ee_pos", [0.0, 0.0, 0.0]), shape=(3,), default=0.0)
    )
    final_row = trace_rows[-1]
    final_pos = _as_array(final_row.get("ee_pos", start_pos), shape=(3,), default=0.0)
    qd_all = _as_array([row.get("qd", [0.0] * 6) for row in trace_rows], shape=(len(trace_rows), 6), default=0.0)
    x_err_all = _as_array([row.get("x_error", 0.0) for row in trace_rows], shape=(len(trace_rows),), default=0.0)
    orient_all = _as_array(
        [row.get("orientation_error_norm", 0.0) for row in trace_rows],
        shape=(len(trace_rows),),
        default=0.0,
    )
    ee_all = _as_array([row.get("ee_pos", start_pos.tolist()) for row in trace_rows], shape=(len(trace_rows), 3), default=0.0)
    transport_axis_index = int(np.clip(int(transport_axis_index), 0, 2))

    max_abs_y_drift_m = float(np.max(np.abs(ee_all[:, 1] - float(start_pos[1]))))
    max_abs_z_drift_m = float(np.max(np.abs(ee_all[:, 2] - float(start_pos[2]))))
    max_abs_orthogonal_drift_m = float(max(max_abs_y_drift_m, max_abs_z_drift_m))
    final_orientation_error_rad = _as_float(final_row.get("orientation_error_norm", orient_all[-1] if orient_all.size else 0.0))
    final_target = _as_array(final_row.get("target_ee_pos", final_pos), shape=(3,), default=0.0)
    if transport_axis_index == 0:
        final_target_axis = _as_float(final_row.get("target_x", final_target[0]))
    else:
        final_target_axis = _as_float(final_row.get("target_axis", final_target[transport_axis_index]))
    final_axis_value = float(final_pos[transport_axis_index])

    return {
        "initial_ee_pos": start_pos.tolist(),
        "final_ee_pos": final_pos.tolist(),
        "achieved_x_delta_m": float(final_pos[0] - float(start_pos[0])),
        "final_x_displacement_m": float(final_pos[0] - float(start_pos[0])),
        "final_x_error_m": float(final_target_axis - final_axis_value),
        "max_abs_x_error_m": float(np.max(np.abs(x_err_all))) if x_err_all.size else 0.0,
        "max_abs_y_drift_m": max_abs_y_drift_m,
        "max_abs_z_drift_m": max_abs_z_drift_m,
        "max_abs_orthogonal_drift_m": max_abs_orthogonal_drift_m,
        "final_orientation_error_rad": final_orientation_error_rad,
        "max_abs_orientation_error_rad": float(np.max(np.abs(orient_all))) if orient_all.size else final_orientation_error_rad,
        "max_abs_qd_radps": float(np.max(np.abs(qd_all))) if qd_all.size else 0.0,
    }


def _ensure_transport_fields(run_summary: Mapping[str, Any]) -> dict[str, Any]:
    data = dict(run_summary)
    trace_rows = _load_trace_rows(data.get("trace_path"))
    if trace_rows:
        derived = summarize_transport_trace(
            trace_rows,
            initial_ee_pos=data.get("initial_ee_pos"),
            transport_axis_index=int(data.get("transport_axis_index", 0)),
        )
        for key, value in derived.items():
            data.setdefault(key, value)
        residual = summarize_residual_torque_trace(trace_rows)
        for key, value in residual.items():
            data.setdefault(key, value)
    else:
        data.setdefault("max_abs_y_drift_m", _as_float(data.get("max_abs_y_drift_m", 0.0)))
        data.setdefault("max_abs_z_drift_m", _as_float(data.get("max_abs_z_drift_m", 0.0)))
        data.setdefault(
            "max_abs_orthogonal_drift_m",
            max(_as_float(data.get("max_abs_y_drift_m", 0.0)), _as_float(data.get("max_abs_z_drift_m", 0.0))),
        )
        data.setdefault("final_orientation_error_rad", _as_float(data.get("final_orientation_error_rad", data.get("max_abs_orientation_error_rad", 0.0))))
    return data


def _termination_has_reason(text: str, *needles: str) -> bool:
    low = text.lower()
    return any(needle in low for needle in needles)


def _target_x_tolerance(run_summary: Mapping[str, Any], *, strict: bool) -> float:
    target_x_delta = abs(_as_float(run_summary.get("target_x_delta", 0.0)))
    if strict:
        return max(_STRICT_X_TOL_ABS, _STRICT_X_TOL_REL * target_x_delta)
    return max(_NORMAL_X_TOL_ABS, _NORMAL_X_TOL_REL * target_x_delta)


def _orientation_tolerance(strict: bool) -> float:
    return _STRICT_ORIENTATION_TOL if strict else _NORMAL_ORIENTATION_TOL


def _drift_tolerance(strict: bool) -> float:
    return _STRICT_YZ_DRIFT_TOL if strict else _NORMAL_YZ_DRIFT_TOL


def _torque_saturation_tolerance(strict: bool) -> float:
    return _STRICT_TORQUE_SAT_TOL_PERCENT if strict else _TORQUE_SAT_TOL_PERCENT


def _qd_limit(run_summary: Mapping[str, Any]) -> float:
    value = run_summary.get("velocity_guard_margin_radps")
    if value is None:
        return _QD_LIMIT_DEFAULT
    max_qd = _as_float(run_summary.get("max_abs_qd_radps", 0.0))
    margin = _as_float(value)
    if margin <= 0.0 and max_qd > 0.0:
        return max(max_qd, _QD_LIMIT_DEFAULT)
    return max(max_qd + margin, _QD_LIMIT_DEFAULT)


def _transport_metrics_internal(run_summary: Mapping[str, Any], *, strict: bool) -> dict[str, Any]:
    data = _ensure_transport_fields(run_summary)
    success = _as_bool(data.get("success", False))
    termination_reason = str(data.get("termination_reason", "") or "")
    failure_reason = str(data.get("failure_reason", "") or "")
    safety_reason = str(data.get("safety_reason", "") or "")
    target_x_delta = _as_float(data.get("target_x_delta", 0.0))
    final_x_error_m = _as_float(data.get("final_x_error_m", 0.0))
    achieved_x_delta_m = _as_float(data.get("achieved_x_delta_m", data.get("final_x_displacement_m", 0.0)))
    max_abs_y_drift_m = _as_float(data.get("max_abs_y_drift_m", 0.0))
    max_abs_z_drift_m = _as_float(data.get("max_abs_z_drift_m", 0.0))
    max_abs_orthogonal_drift_m = _as_float(
        data.get("max_abs_orthogonal_drift_m", max(max_abs_y_drift_m, max_abs_z_drift_m))
    )
    max_abs_orientation_error_rad = _as_float(data.get("max_abs_orientation_error_rad", data.get("final_orientation_error_rad", 0.0)))
    final_orientation_error_rad = _as_float(data.get("final_orientation_error_rad", max_abs_orientation_error_rad))
    max_abs_qd_radps = _as_float(data.get("max_abs_qd_radps", 0.0))
    torque_saturation_percentage = _as_float(data.get("torque_saturation_percentage", 0.0))
    velocity_guard_ok = _as_bool(data.get("velocity_guard_ok", True))
    joint_limit_guard_ok = _as_bool(data.get("joint_limit_guard_ok", True))
    qd_limit = _qd_limit(data)

    x_tol = _target_x_tolerance(data, strict=strict)
    y_tol = _drift_tolerance(strict)
    z_tol = _drift_tolerance(strict)
    orientation_tol = _orientation_tolerance(strict)
    torque_sat_tol = _torque_saturation_tolerance(strict)

    x_tracking_pass = bool(
        success
        and (termination_reason == "duration_complete" or _termination_has_reason(termination_reason, "duration_complete"))
        and abs(final_x_error_m) <= x_tol
        and abs(achieved_x_delta_m - target_x_delta) <= x_tol
    )
    orthogonal_drift_pass = bool(max_abs_y_drift_m <= y_tol and max_abs_z_drift_m <= z_tol)
    orientation_pass = bool(max_abs_orientation_error_rad <= orientation_tol)
    duration_pass = bool(
        success
        and (termination_reason == "duration_complete" or _termination_has_reason(termination_reason, "duration_complete"))
    )
    no_growth_termination = not _termination_has_reason(
        " ".join([termination_reason, failure_reason, safety_reason]),
        "axis_error",
        "x_error",
        "grew for",
        "consecutive steps",
    )
    safety_pass = bool(
        success
        and velocity_guard_ok
        and joint_limit_guard_ok
        and torque_saturation_percentage <= torque_sat_tol
        and no_growth_termination
        and not _termination_has_reason(" ".join([termination_reason, failure_reason, safety_reason]), "velocity", "joint limit")
    )

    tracking_score = 1.0 / (
        1.0
        + (abs(final_x_error_m) / max(x_tol, 1.0e-12))
        + (abs(achieved_x_delta_m - target_x_delta) / max(x_tol, 1.0e-12))
    )
    transport_quality_score = tracking_score / (
        1.0
        + (max_abs_y_drift_m / max(y_tol, 1.0e-12))
        + (max_abs_z_drift_m / max(z_tol, 1.0e-12))
        + (max_abs_orientation_error_rad / max(orientation_tol, 1.0e-12))
        + (torque_saturation_percentage / max(torque_sat_tol, 1.0e-12))
        + (max_abs_qd_radps / max(qd_limit, 1.0e-12))
    )

    valid_transport = bool(success and duration_pass and x_tracking_pass and orthogonal_drift_pass and orientation_pass and safety_pass)

    strict_x_tol = _target_x_tolerance(data, strict=True)
    strict_orientation_tol = _orientation_tolerance(True)
    strict_drift_tol = _drift_tolerance(True)
    strict_torque_sat_tol = _torque_saturation_tolerance(True)
    strict_x_tracking_pass = bool(
        success
        and (termination_reason == "duration_complete" or _termination_has_reason(termination_reason, "duration_complete"))
        and abs(final_x_error_m) <= strict_x_tol
        and abs(achieved_x_delta_m - target_x_delta) <= strict_x_tol
    )
    strict_orthogonal_drift_pass = bool(max_abs_y_drift_m <= strict_drift_tol and max_abs_z_drift_m <= strict_drift_tol)
    strict_orientation_pass = bool(max_abs_orientation_error_rad <= strict_orientation_tol)
    strict_duration_pass = duration_pass
    strict_safety_pass = bool(
        success
        and velocity_guard_ok
        and joint_limit_guard_ok
        and torque_saturation_percentage <= strict_torque_sat_tol
        and no_growth_termination
        and not _termination_has_reason(" ".join([termination_reason, failure_reason, safety_reason]), "velocity", "joint limit")
    )
    strict_tracking_score = 1.0 / (
        1.0
        + (abs(final_x_error_m) / max(strict_x_tol, 1.0e-12))
        + (abs(achieved_x_delta_m - target_x_delta) / max(strict_x_tol, 1.0e-12))
    )
    strict_transport_quality_score = strict_tracking_score / (
        1.0
        + (max_abs_y_drift_m / max(strict_drift_tol, 1.0e-12))
        + (max_abs_z_drift_m / max(strict_drift_tol, 1.0e-12))
        + (max_abs_orientation_error_rad / max(strict_orientation_tol, 1.0e-12))
        + (torque_saturation_percentage / max(strict_torque_sat_tol, 1.0e-12))
        + (max_abs_qd_radps / max(qd_limit, 1.0e-12))
    )
    strict_valid_transport = bool(
        success
        and strict_duration_pass
        and strict_x_tracking_pass
        and strict_orthogonal_drift_pass
        and strict_orientation_pass
        and strict_safety_pass
    )

    metrics = {
        "valid_transport": valid_transport,
        "x_tracking_pass": x_tracking_pass,
        "orthogonal_drift_pass": orthogonal_drift_pass,
        "orientation_pass": orientation_pass,
        "duration_pass": duration_pass,
        "safety_pass": safety_pass,
        "tracking_score": float(tracking_score),
        "transport_quality_score": float(transport_quality_score),
        "strict_valid_transport": strict_valid_transport,
        "strict_x_tracking_pass": strict_x_tracking_pass,
        "strict_orthogonal_drift_pass": strict_orthogonal_drift_pass,
        "strict_orientation_pass": strict_orientation_pass,
        "strict_duration_pass": strict_duration_pass,
        "strict_safety_pass": strict_safety_pass,
        "strict_tracking_score": float(strict_tracking_score),
        "strict_transport_quality_score": float(strict_transport_quality_score),
        "final_x_error_m": final_x_error_m,
        "achieved_x_delta_m": achieved_x_delta_m,
        "max_abs_y_drift_m": max_abs_y_drift_m,
        "max_abs_z_drift_m": max_abs_z_drift_m,
        "max_abs_orthogonal_drift_m": max_abs_orthogonal_drift_m,
        "final_orientation_error_rad": final_orientation_error_rad,
        "max_abs_orientation_error_rad": max_abs_orientation_error_rad,
        "max_abs_qd_radps": max_abs_qd_radps,
        "torque_saturation_percentage": torque_saturation_percentage,
        "velocity_guard_ok": velocity_guard_ok,
        "joint_limit_guard_ok": joint_limit_guard_ok,
    }

    if strict:
        metrics["valid_transport"] = strict_valid_transport
        metrics["x_tracking_pass"] = strict_x_tracking_pass
        metrics["orthogonal_drift_pass"] = strict_orthogonal_drift_pass
        metrics["orientation_pass"] = strict_orientation_pass
        metrics["duration_pass"] = strict_duration_pass
        metrics["safety_pass"] = strict_safety_pass
        metrics["tracking_score"] = float(strict_tracking_score)
        metrics["transport_quality_score"] = float(strict_transport_quality_score)
    return metrics


def compute_valid_transport_metrics(run_summary: Mapping[str, Any], strict: bool = False) -> dict[str, Any]:
    """Compute validity and scoring fields for a transport run summary."""

    return _transport_metrics_internal(run_summary, strict=strict)


def summarize_residual_torque_trace(
    trace_rows: Sequence[Mapping[str, Any]],
    *,
    torque_limit_nm: Sequence[float] | None = None,
) -> dict[str, Any]:
    if not trace_rows:
        return {
            "gravity_mode_used": "raw",
            "gravity_compensation_active": False,
            "raw_mode_used": True,
            "max_abs_tau_controller_nm": 0.0,
            "max_abs_tau_gravity_nm": 0.0,
            "max_abs_tau_applied_nm": 0.0,
            "mean_abs_tau_controller_nm": 0.0,
            "mean_abs_tau_gravity_nm": 0.0,
            "mean_abs_tau_applied_nm": 0.0,
            "max_abs_tau_controller_clipped_nm": 0.0,
            "mean_abs_tau_controller_clipped_nm": 0.0,
            "max_abs_tau_applied_clipped_nm": 0.0,
            "mean_abs_tau_applied_clipped_nm": 0.0,
            "controller_torque_clip_fraction": 0.0,
            "applied_torque_clip_fraction": 0.0,
            "controller_torque_clip_percentage": 0.0,
            "applied_torque_clip_percentage": 0.0,
            "controller_torque_clip_count": 0,
            "applied_torque_clip_count": 0,
            "gravity_torque_fraction": 0.0,
            "controller_torque_fraction": 0.0,
        }

    def _series(key: str, fallback_key: str | None = None, default: Any = 0.0) -> np.ndarray:
        values: list[Any] = []
        for row in trace_rows:
            value = row.get(key, None)
            if value is None and fallback_key is not None:
                value = row.get(fallback_key, None)
            if value is None:
                value = default
            values.append(value)
        if isinstance(default, (list, tuple, np.ndarray)):
            shape = (len(trace_rows), len(np.asarray(default, dtype=np.float64).reshape(-1)))
            return _as_array(values, shape=shape, default=0.0)
        return _as_array(values, shape=(len(trace_rows),), default=float(default))

    gravity_modes = [str(row.get("gravity_mode", "") or "") for row in trace_rows]
    gravity_mode_used = gravity_modes[0] if gravity_modes else "raw"
    gravity_compensation_active = bool(any(mode == "gravity_comp" for mode in gravity_modes))

    tau_controller_all = _series("tau_controller", "tau_raw", default=[0.0] * 6)
    tau_gravity_all = _series("tau_gravity", default=[0.0] * 6)
    tau_applied_all = _series("tau_applied", "tau", default=[0.0] * 6)

    if torque_limit_nm is not None:
        torque_limit_vec = _as_array(torque_limit_nm, shape=(6,), default=0.0)
        tau_controller_clipped_all = np.clip(tau_controller_all, -torque_limit_vec, +torque_limit_vec)
        tau_applied_clipped_all = np.clip(tau_applied_all, -torque_limit_vec, +torque_limit_vec)
    else:
        tau_controller_clipped_all = _series("tau_controller_clipped", "tau_controller", default=[0.0] * 6)
        tau_applied_clipped_all = _series("tau_clipped", "tau_applied", default=[0.0] * 6)

    controller_clip_fraction_series = _series("tau_controller_clip_fraction", "controller_torque_clip_fraction", default=0.0)
    if "tau_controller_clip_fraction" not in trace_rows[0] and "controller_torque_clip_fraction" not in trace_rows[0]:
        controller_clip_fraction_series = np.mean(np.abs(tau_controller_clipped_all - tau_controller_all) > 1.0e-9, axis=1)
    applied_clip_fraction_series = _series("torque_clip_fraction", "tau_applied_clip_fraction", default=0.0)
    if "torque_clip_fraction" not in trace_rows[0] and "tau_applied_clip_fraction" not in trace_rows[0]:
        applied_clip_fraction_series = np.mean(np.abs(tau_applied_clipped_all - tau_applied_all) > 1.0e-9, axis=1)

    mean_abs_tau_controller_nm = float(np.mean(np.abs(tau_controller_all)))
    mean_abs_tau_gravity_nm = float(np.mean(np.abs(tau_gravity_all)))
    mean_abs_tau_applied_nm = float(np.mean(np.abs(tau_applied_all)))
    max_abs_tau_controller_nm = float(np.max(np.abs(tau_controller_all)))
    max_abs_tau_gravity_nm = float(np.max(np.abs(tau_gravity_all)))
    max_abs_tau_applied_nm = float(np.max(np.abs(tau_applied_all)))

    mean_abs_tau_controller_clipped_nm = float(np.mean(np.abs(tau_controller_clipped_all)))
    max_abs_tau_controller_clipped_nm = float(np.max(np.abs(tau_controller_clipped_all)))
    mean_abs_tau_applied_clipped_nm = float(np.mean(np.abs(tau_applied_clipped_all)))
    max_abs_tau_applied_clipped_nm = float(np.max(np.abs(tau_applied_clipped_all)))

    controller_torque_clip_fraction = float(np.mean(controller_clip_fraction_series))
    applied_torque_clip_fraction = float(np.mean(applied_clip_fraction_series))
    eps = 1.0e-12
    controller_torque_fraction = float(mean_abs_tau_controller_nm / max(mean_abs_tau_applied_nm, eps))
    gravity_torque_fraction = float(mean_abs_tau_gravity_nm / max(mean_abs_tau_applied_nm, eps))

    return {
        "gravity_mode_used": gravity_mode_used,
        "gravity_compensation_active": gravity_compensation_active,
        "raw_mode_used": bool(gravity_mode_used == "raw"),
        "max_abs_tau_controller_nm": max_abs_tau_controller_nm,
        "max_abs_tau_gravity_nm": max_abs_tau_gravity_nm,
        "max_abs_tau_applied_nm": max_abs_tau_applied_nm,
        "mean_abs_tau_controller_nm": mean_abs_tau_controller_nm,
        "mean_abs_tau_gravity_nm": mean_abs_tau_gravity_nm,
        "mean_abs_tau_applied_nm": mean_abs_tau_applied_nm,
        "max_abs_tau_controller_clipped_nm": max_abs_tau_controller_clipped_nm,
        "mean_abs_tau_controller_clipped_nm": mean_abs_tau_controller_clipped_nm,
        "max_abs_tau_applied_clipped_nm": max_abs_tau_applied_clipped_nm,
        "mean_abs_tau_applied_clipped_nm": mean_abs_tau_applied_clipped_nm,
        "controller_torque_clip_fraction": controller_torque_clip_fraction,
        "applied_torque_clip_fraction": applied_torque_clip_fraction,
        "controller_torque_clip_percentage": float(100.0 * controller_torque_clip_fraction),
        "applied_torque_clip_percentage": float(100.0 * applied_torque_clip_fraction),
        "controller_torque_clip_count": int(np.count_nonzero(np.asarray(controller_clip_fraction_series) > 1.0e-12)),
        "applied_torque_clip_count": int(np.count_nonzero(np.asarray(applied_clip_fraction_series) > 1.0e-12)),
        "gravity_torque_fraction": gravity_torque_fraction,
        "controller_torque_fraction": controller_torque_fraction,
    }


def summarize_move_hold_trace(
    trace_rows: Sequence[Mapping[str, Any]],
    *,
    initial_ee_pos: Sequence[float] | None = None,
    move_duration_s: float | None = None,
    total_duration_s: float | None = None,
    transport_axis_index: int = 0,
) -> dict[str, Any]:
    if not trace_rows:
        move_duration = max(_as_float(move_duration_s, 0.0), 0.0)
        total_duration = max(_as_float(total_duration_s, move_duration), move_duration)
        hold_duration = max(total_duration - move_duration, 0.0)
        return {
            "move_duration_s": float(move_duration),
            "hold_duration_s": float(hold_duration),
            "total_duration_s": float(total_duration),
            "move_phase_final_x_error_m": 0.0,
            "move_phase_achieved_x_delta_m": 0.0,
            "move_phase_max_abs_x_error_m": 0.0,
            "move_phase_max_abs_y_drift_m": 0.0,
            "move_phase_max_abs_z_drift_m": 0.0,
            "move_phase_max_abs_orthogonal_drift_m": 0.0,
            "move_phase_max_abs_orientation_error_rad": 0.0,
            "move_phase_max_abs_qd_radps": 0.0,
            "move_phase_max_abs_tau_controller_nm": 0.0,
            "move_phase_max_abs_tau_applied_nm": 0.0,
            "hold_phase_final_x_error_m": 0.0,
            "hold_phase_max_abs_x_error_m": 0.0,
            "hold_phase_x_drift_from_hold_start_m": 0.0,
            "hold_phase_max_abs_y_drift_m": 0.0,
            "hold_phase_max_abs_z_drift_m": 0.0,
            "hold_phase_max_abs_orthogonal_drift_m": 0.0,
            "hold_phase_max_abs_orientation_error_rad": 0.0,
            "hold_phase_max_abs_qd_radps": 0.0,
            "hold_phase_max_abs_tau_controller_nm": 0.0,
            "hold_phase_max_abs_tau_applied_nm": 0.0,
            "move_phase_complete": False,
            "hold_phase_complete": False,
        }

    start_pos = (
        _as_array(initial_ee_pos, shape=(3,), default=0.0)
        if initial_ee_pos is not None
        else _as_array(trace_rows[0].get("ee_pos", [0.0, 0.0, 0.0]), shape=(3,), default=0.0)
    )
    times = np.asarray([_as_float(row.get("time_s", 0.0)) for row in trace_rows], dtype=np.float64)
    ee_all = _as_array([row.get("ee_pos", start_pos.tolist()) for row in trace_rows], shape=(len(trace_rows), 3), default=0.0)
    quat_all = _as_array([row.get("ee_quat", [1.0, 0.0, 0.0, 0.0]) for row in trace_rows], shape=(len(trace_rows), 4), default=0.0)
    qd_all = _as_array([row.get("qd", [0.0] * 6) for row in trace_rows], shape=(len(trace_rows), 6), default=0.0)
    orient_all = _as_array([row.get("orientation_error_norm", 0.0) for row in trace_rows], shape=(len(trace_rows),), default=0.0)
    x_err_all = _as_array([row.get("x_error", 0.0) for row in trace_rows], shape=(len(trace_rows),), default=0.0)
    tau_controller_all = _as_array([row.get("tau_controller", row.get("tau_raw", [0.0] * 6)) for row in trace_rows], shape=(len(trace_rows), 6), default=0.0)
    tau_applied_all = _as_array([row.get("tau_applied", row.get("tau", [0.0] * 6)) for row in trace_rows], shape=(len(trace_rows), 6), default=0.0)

    move_duration = max(_as_float(move_duration_s, 0.0), 0.0)
    total_duration = max(_as_float(total_duration_s, float(times[-1]) if times.size else move_duration), move_duration)
    hold_duration = max(total_duration - move_duration, 0.0)

    move_indices = np.flatnonzero(times <= move_duration + 1.0e-9)
    hold_indices = np.flatnonzero(times >= move_duration - 1.0e-9)
    if move_indices.size == 0:
        move_indices = np.array([0], dtype=np.int64)
    if hold_indices.size == 0:
        hold_indices = np.array([len(trace_rows) - 1], dtype=np.int64)
    move_end_index = int(hold_indices[0])
    hold_start_index = int(hold_indices[0])
    final_index = len(trace_rows) - 1

    move_end_row = trace_rows[move_end_index]
    hold_start_row = trace_rows[hold_start_index]
    final_row = trace_rows[final_index]
    move_end_pos = ee_all[move_end_index]
    hold_start_pos = ee_all[hold_start_index]
    final_pos = ee_all[final_index]
    hold_start_quat = quat_all[hold_start_index]

    move_phase_rows = ee_all[move_indices]
    hold_phase_rows = ee_all[hold_indices]
    hold_phase_orient_rel = np.asarray(
        [
            float(np.linalg.norm(orientation_error_vec_wxyz(hold_start_quat, quat_all[int(idx)])))
            for idx in hold_indices.tolist()
        ],
        dtype=np.float64,
    )
    if hold_phase_orient_rel.size == 0:
        hold_phase_orient_rel = np.array([0.0], dtype=np.float64)

    move_phase_max_abs_y_drift_m = float(np.max(np.abs(move_phase_rows[:, 1] - float(start_pos[1]))))
    move_phase_max_abs_z_drift_m = float(np.max(np.abs(move_phase_rows[:, 2] - float(start_pos[2]))))
    move_phase_max_abs_orthogonal_drift_m = float(max(move_phase_max_abs_y_drift_m, move_phase_max_abs_z_drift_m))
    hold_phase_max_abs_y_drift_m = float(np.max(np.abs(hold_phase_rows[:, 1] - float(hold_start_pos[1]))))
    hold_phase_max_abs_z_drift_m = float(np.max(np.abs(hold_phase_rows[:, 2] - float(hold_start_pos[2]))))
    hold_phase_max_abs_orthogonal_drift_m = float(max(hold_phase_max_abs_y_drift_m, hold_phase_max_abs_z_drift_m))

    move_target_x = _as_float(move_end_row.get("target_x", final_row.get("target_x", start_pos[0])))
    final_target_x = _as_float(final_row.get("target_x", move_target_x))
    move_phase_final_x_error_m = float(move_target_x - float(move_end_pos[0]))
    move_phase_achieved_x_delta_m = float(move_end_pos[0] - float(start_pos[0]))
    move_phase_max_abs_x_error_m = float(np.max(np.abs(x_err_all[move_indices])))
    hold_phase_final_x_error_m = float(final_target_x - float(final_pos[0]))
    hold_phase_max_abs_x_error_m = float(np.max(np.abs(x_err_all[hold_indices])))
    hold_phase_x_drift_from_hold_start_m = float(np.max(np.abs(hold_phase_rows[:, 0] - float(hold_start_pos[0]))))

    return {
        "move_duration_s": float(move_duration),
        "hold_duration_s": float(hold_duration),
        "total_duration_s": float(total_duration),
        "move_phase_final_x_error_m": move_phase_final_x_error_m,
        "move_phase_achieved_x_delta_m": move_phase_achieved_x_delta_m,
        "move_phase_max_abs_x_error_m": move_phase_max_abs_x_error_m,
        "move_phase_max_abs_y_drift_m": move_phase_max_abs_y_drift_m,
        "move_phase_max_abs_z_drift_m": move_phase_max_abs_z_drift_m,
        "move_phase_max_abs_orthogonal_drift_m": move_phase_max_abs_orthogonal_drift_m,
        "move_phase_max_abs_orientation_error_rad": float(np.max(np.abs(orient_all[move_indices]))),
        "move_phase_max_abs_qd_radps": float(np.max(np.abs(qd_all[move_indices]))),
        "move_phase_max_abs_tau_controller_nm": float(np.max(np.abs(tau_controller_all[move_indices]))),
        "move_phase_max_abs_tau_applied_nm": float(np.max(np.abs(tau_applied_all[move_indices]))),
        "hold_phase_final_x_error_m": hold_phase_final_x_error_m,
        "hold_phase_max_abs_x_error_m": hold_phase_max_abs_x_error_m,
        "hold_phase_x_drift_from_hold_start_m": hold_phase_x_drift_from_hold_start_m,
        "hold_phase_max_abs_y_drift_m": hold_phase_max_abs_y_drift_m,
        "hold_phase_max_abs_z_drift_m": hold_phase_max_abs_z_drift_m,
        "hold_phase_max_abs_orthogonal_drift_m": hold_phase_max_abs_orthogonal_drift_m,
        "hold_phase_max_abs_orientation_error_rad": float(np.max(hold_phase_orient_rel)),
        "hold_phase_max_abs_qd_radps": float(np.max(np.abs(qd_all[hold_indices]))),
        "hold_phase_max_abs_tau_controller_nm": float(np.max(np.abs(tau_controller_all[hold_indices]))),
        "hold_phase_max_abs_tau_applied_nm": float(np.max(np.abs(tau_applied_all[hold_indices]))),
        "move_phase_complete": bool(len(move_indices) > 0 and float(times[move_end_index]) >= move_duration - 1.0e-9),
        "hold_phase_complete": bool(len(hold_indices) > 0 and float(times[final_index]) >= total_duration - 1.0e-9),
    }


def _move_hold_tolerances(run_summary: Mapping[str, Any], *, strict: bool) -> dict[str, float]:
    target_x_delta = abs(_as_float(run_summary.get("target_x_delta", 0.0)))
    move_x_tol = _target_x_tolerance(run_summary, strict=strict)
    hold_x_drift_tol = max(0.003, 0.15 * target_x_delta)
    if strict:
        yz_tol = _STRICT_YZ_DRIFT_TOL
        orientation_tol = _STRICT_ORIENTATION_TOL
        torque_sat_tol = _STRICT_TORQUE_SAT_TOL_PERCENT
    else:
        yz_tol = _NORMAL_YZ_DRIFT_TOL
        orientation_tol = _NORMAL_ORIENTATION_TOL
        torque_sat_tol = _TORQUE_SAT_TOL_PERCENT
    return {
        "move_x_tol": float(move_x_tol),
        "hold_x_drift_tol": float(hold_x_drift_tol),
        "yz_tol": float(yz_tol),
        "orientation_tol": float(orientation_tol),
        "torque_sat_tol": float(torque_sat_tol),
    }


def _ensure_move_hold_fields(run_summary: Mapping[str, Any]) -> dict[str, Any]:
    data = dict(run_summary)
    trace_rows = _load_trace_rows(data.get("trace_path"))
    if trace_rows:
        derived = summarize_move_hold_trace(
            trace_rows,
            initial_ee_pos=data.get("initial_ee_pos"),
            move_duration_s=data.get("move_duration_s", data.get("move_duration")),
            total_duration_s=data.get("total_duration_s", data.get("duration_s")),
            transport_axis_index=int(data.get("transport_axis_index", 0)),
        )
        for key, value in derived.items():
            data.setdefault(key, value)
    else:
        data.setdefault("move_duration_s", _as_float(data.get("move_duration_s", data.get("move_duration", 0.0))))
        data.setdefault("hold_duration_s", _as_float(data.get("hold_duration_s", max(_as_float(data.get("duration_s", 0.0)) - _as_float(data.get("move_duration_s", data.get("move_duration", 0.0))), 0.0))))
        data.setdefault("total_duration_s", _as_float(data.get("total_duration_s", data.get("duration_s", 0.0))))
        for key in (
            "move_phase_final_x_error_m",
            "move_phase_achieved_x_delta_m",
            "move_phase_max_abs_x_error_m",
            "move_phase_max_abs_y_drift_m",
            "move_phase_max_abs_z_drift_m",
            "move_phase_max_abs_orthogonal_drift_m",
            "move_phase_max_abs_orientation_error_rad",
            "move_phase_max_abs_qd_radps",
            "move_phase_max_abs_tau_controller_nm",
            "move_phase_max_abs_tau_applied_nm",
            "hold_phase_final_x_error_m",
            "hold_phase_max_abs_x_error_m",
            "hold_phase_x_drift_from_hold_start_m",
            "hold_phase_max_abs_y_drift_m",
            "hold_phase_max_abs_z_drift_m",
            "hold_phase_max_abs_orthogonal_drift_m",
            "hold_phase_max_abs_orientation_error_rad",
            "hold_phase_max_abs_qd_radps",
            "hold_phase_max_abs_tau_controller_nm",
            "hold_phase_max_abs_tau_applied_nm",
        ):
            data.setdefault(key, 0.0)
        data.setdefault("move_phase_complete", False)
        data.setdefault("hold_phase_complete", False)
    return data


def compute_valid_move_hold_metrics(run_summary: Mapping[str, Any], strict: bool = False) -> dict[str, Any]:
    data = _ensure_transport_fields(run_summary)
    data = _ensure_move_hold_fields(data)
    thresholds = _move_hold_tolerances(data, strict=strict)
    success = _as_bool(data.get("success", False))
    termination_reason = str(data.get("termination_reason", "") or "")
    failure_reason = str(data.get("failure_reason", "") or "")
    safety_reason = str(data.get("safety_reason", "") or "")
    combined_reason = " ".join([termination_reason, failure_reason, safety_reason]).strip()
    target_x_delta = abs(_as_float(data.get("target_x_delta", 0.0)))
    move_duration_s = max(_as_float(data.get("move_duration_s", data.get("move_duration", 0.0))), 0.0)
    total_duration_s = max(_as_float(data.get("total_duration_s", data.get("duration_s", move_duration_s))), move_duration_s)
    hold_duration_s = max(_as_float(data.get("hold_duration_s", total_duration_s - move_duration_s)), 0.0)
    sim_time_s = _as_float(data.get("sim_time_s", data.get("duration_s", total_duration_s)), total_duration_s)
    failure_time_s = _as_float(data.get("failure_time_s", sim_time_s), sim_time_s)
    velocity_guard_ok = _as_bool(data.get("velocity_guard_ok", True))
    joint_limit_guard_ok = _as_bool(data.get("joint_limit_guard_ok", True))
    torque_saturation_percentage = _as_float(data.get("torque_saturation_percentage", 0.0))
    max_abs_qd_radps = _as_float(data.get("max_abs_qd_radps", 0.0))
    qd_limit = _qd_limit(data)
    no_growth_termination = not _termination_has_reason(combined_reason, "axis_error", "x_error", "grew for", "consecutive steps")
    overall_safety_pass = bool(
        success
        and velocity_guard_ok
        and joint_limit_guard_ok
        and torque_saturation_percentage <= thresholds["torque_sat_tol"]
        and no_growth_termination
        and not _termination_has_reason(combined_reason, "velocity", "joint limit")
    )
    duration_pass = bool(
        success
        and (termination_reason == "duration_complete" or _termination_has_reason(termination_reason, "duration_complete"))
        and sim_time_s >= total_duration_s - 1.0e-9
    )
    move_phase_duration_pass = bool(sim_time_s >= move_duration_s - 1.0e-9)
    hold_phase_duration_pass = bool(sim_time_s >= total_duration_s - 1.0e-9)
    move_phase_safety_pass = bool(overall_safety_pass or failure_time_s >= move_duration_s - 1.0e-9)
    hold_phase_safety_pass = bool(overall_safety_pass or failure_time_s >= total_duration_s - 1.0e-9)

    move_x_pass = bool(
        abs(_as_float(data.get("move_phase_final_x_error_m", 0.0))) <= thresholds["move_x_tol"]
        and abs(_as_float(data.get("move_phase_achieved_x_delta_m", 0.0)) - target_x_delta) <= thresholds["move_x_tol"]
    )
    move_y_pass = bool(_as_float(data.get("move_phase_max_abs_y_drift_m", 0.0)) <= thresholds["yz_tol"])
    move_z_pass = bool(_as_float(data.get("move_phase_max_abs_z_drift_m", 0.0)) <= thresholds["yz_tol"])
    move_orientation_pass = bool(_as_float(data.get("move_phase_max_abs_orientation_error_rad", 0.0)) <= thresholds["orientation_tol"])
    move_phase_pass = bool(
        move_phase_duration_pass
        and move_phase_safety_pass
        and move_x_pass
        and move_y_pass
        and move_z_pass
        and move_orientation_pass
    )

    hold_x_pass = bool(
        abs(_as_float(data.get("hold_phase_final_x_error_m", 0.0))) <= thresholds["move_x_tol"]
        and _as_float(data.get("hold_phase_x_drift_from_hold_start_m", 0.0)) <= thresholds["hold_x_drift_tol"]
    )
    hold_y_pass = bool(_as_float(data.get("hold_phase_max_abs_y_drift_m", 0.0)) <= thresholds["yz_tol"])
    hold_z_pass = bool(_as_float(data.get("hold_phase_max_abs_z_drift_m", 0.0)) <= thresholds["yz_tol"])
    hold_orientation_pass = bool(_as_float(data.get("hold_phase_max_abs_orientation_error_rad", 0.0)) <= thresholds["orientation_tol"])
    hold_phase_pass = bool(
        hold_phase_duration_pass
        and hold_phase_safety_pass
        and hold_x_pass
        and hold_y_pass
        and hold_z_pass
        and hold_orientation_pass
    )

    move_reason = "none"
    if not move_phase_duration_pass:
        move_reason = "move_phase_incomplete"
    elif not move_x_pass:
        move_reason = "move_phase_target_tracking"
    elif not move_y_pass:
        move_reason = "move_phase_y_drift"
    elif not move_z_pass:
        move_reason = "move_phase_z_drift"
    elif not move_orientation_pass:
        move_reason = "move_phase_orientation"
    elif not move_phase_safety_pass:
        if not velocity_guard_ok:
            move_reason = "move_phase_velocity_guard"
        elif not joint_limit_guard_ok:
            move_reason = "move_phase_joint_limit"
        elif not no_growth_termination:
            move_reason = "move_phase_axis_growth"
        elif torque_saturation_percentage > thresholds["torque_sat_tol"]:
            move_reason = "move_phase_torque_saturation"
        else:
            move_reason = "move_phase_safety"

    hold_reason = "none"
    if not hold_phase_duration_pass:
        hold_reason = "hold_phase_incomplete"
    elif not hold_x_pass:
        hold_reason = "hold_phase_target_tracking"
    elif not hold_y_pass:
        hold_reason = "hold_phase_y_drift"
    elif not hold_z_pass:
        hold_reason = "hold_phase_z_drift"
    elif not hold_orientation_pass:
        hold_reason = "hold_phase_orientation"
    elif not hold_phase_safety_pass:
        if not velocity_guard_ok:
            hold_reason = "hold_phase_velocity_guard"
        elif not joint_limit_guard_ok:
            hold_reason = "hold_phase_joint_limit"
        elif not no_growth_termination:
            hold_reason = "hold_phase_axis_growth"
        elif torque_saturation_percentage > thresholds["torque_sat_tol"]:
            hold_reason = "hold_phase_torque_saturation"
        else:
            hold_reason = "hold_phase_safety"

    move_tracking_score = 1.0 / (
        1.0
        + (abs(_as_float(data.get("move_phase_final_x_error_m", 0.0))) / max(thresholds["move_x_tol"], 1.0e-12))
        + (abs(_as_float(data.get("move_phase_achieved_x_delta_m", 0.0)) - target_x_delta) / max(thresholds["move_x_tol"], 1.0e-12))
        + (_as_float(data.get("hold_phase_x_drift_from_hold_start_m", 0.0)) / max(thresholds["hold_x_drift_tol"], 1.0e-12))
    )
    move_hold_quality_score = move_tracking_score / (
        1.0
        + (_as_float(data.get("move_phase_max_abs_y_drift_m", 0.0)) / max(thresholds["yz_tol"], 1.0e-12))
        + (_as_float(data.get("move_phase_max_abs_z_drift_m", 0.0)) / max(thresholds["yz_tol"], 1.0e-12))
        + (_as_float(data.get("hold_phase_max_abs_y_drift_m", 0.0)) / max(thresholds["yz_tol"], 1.0e-12))
        + (_as_float(data.get("hold_phase_max_abs_z_drift_m", 0.0)) / max(thresholds["yz_tol"], 1.0e-12))
        + (_as_float(data.get("move_phase_max_abs_orientation_error_rad", 0.0)) / max(thresholds["orientation_tol"], 1.0e-12))
        + (_as_float(data.get("hold_phase_max_abs_orientation_error_rad", 0.0)) / max(thresholds["orientation_tol"], 1.0e-12))
        + (torque_saturation_percentage / max(thresholds["torque_sat_tol"], 1.0e-12))
        + (max_abs_qd_radps / max(qd_limit, 1.0e-12))
    )

    valid_move_phase = bool(move_phase_pass)
    valid_hold_phase = bool(hold_phase_pass)
    valid_move_and_hold = bool(duration_pass and valid_move_phase and valid_hold_phase and overall_safety_pass)

    strict_result = None
    if strict:
        strict_result = True

    return {
        "move_duration_s": float(move_duration_s),
        "hold_duration_s": float(hold_duration_s),
        "total_duration_s": float(total_duration_s),
        "move_phase_final_x_error_m": float(_as_float(data.get("move_phase_final_x_error_m", 0.0))),
        "move_phase_achieved_x_delta_m": float(_as_float(data.get("move_phase_achieved_x_delta_m", 0.0))),
        "move_phase_max_abs_x_error_m": float(_as_float(data.get("move_phase_max_abs_x_error_m", 0.0))),
        "move_phase_max_abs_y_drift_m": float(_as_float(data.get("move_phase_max_abs_y_drift_m", 0.0))),
        "move_phase_max_abs_z_drift_m": float(_as_float(data.get("move_phase_max_abs_z_drift_m", 0.0))),
        "move_phase_max_abs_orthogonal_drift_m": float(_as_float(data.get("move_phase_max_abs_orthogonal_drift_m", max(_as_float(data.get("move_phase_max_abs_y_drift_m", 0.0)), _as_float(data.get("move_phase_max_abs_z_drift_m", 0.0)))))),
        "move_phase_max_abs_orientation_error_rad": float(_as_float(data.get("move_phase_max_abs_orientation_error_rad", 0.0))),
        "move_phase_max_abs_qd_radps": float(_as_float(data.get("move_phase_max_abs_qd_radps", 0.0))),
        "move_phase_max_abs_tau_controller_nm": float(_as_float(data.get("move_phase_max_abs_tau_controller_nm", 0.0))),
        "move_phase_max_abs_tau_applied_nm": float(_as_float(data.get("move_phase_max_abs_tau_applied_nm", 0.0))),
        "hold_phase_final_x_error_m": float(_as_float(data.get("hold_phase_final_x_error_m", 0.0))),
        "hold_phase_max_abs_x_error_m": float(_as_float(data.get("hold_phase_max_abs_x_error_m", 0.0))),
        "hold_phase_x_drift_from_hold_start_m": float(_as_float(data.get("hold_phase_x_drift_from_hold_start_m", 0.0))),
        "hold_phase_max_abs_y_drift_m": float(_as_float(data.get("hold_phase_max_abs_y_drift_m", 0.0))),
        "hold_phase_max_abs_z_drift_m": float(_as_float(data.get("hold_phase_max_abs_z_drift_m", 0.0))),
        "hold_phase_max_abs_orthogonal_drift_m": float(_as_float(data.get("hold_phase_max_abs_orthogonal_drift_m", max(_as_float(data.get("hold_phase_max_abs_y_drift_m", 0.0)), _as_float(data.get("hold_phase_max_abs_z_drift_m", 0.0)))))),
        "hold_phase_max_abs_orientation_error_rad": float(_as_float(data.get("hold_phase_max_abs_orientation_error_rad", 0.0))),
        "hold_phase_max_abs_qd_radps": float(_as_float(data.get("hold_phase_max_abs_qd_radps", 0.0))),
        "hold_phase_max_abs_tau_controller_nm": float(_as_float(data.get("hold_phase_max_abs_tau_controller_nm", 0.0))),
        "hold_phase_max_abs_tau_applied_nm": float(_as_float(data.get("hold_phase_max_abs_tau_applied_nm", 0.0))),
        "move_phase_complete": move_phase_duration_pass,
        "hold_phase_complete": hold_phase_duration_pass,
        "valid_move_phase": valid_move_phase,
        "valid_hold_phase": valid_hold_phase,
        "valid_move_and_hold": valid_move_and_hold,
        "move_failure_reason": move_reason,
        "hold_failure_reason": hold_reason,
        "move_tracking_score": float(move_tracking_score),
        "move_hold_quality_score": float(move_hold_quality_score),
        "move_hold_tracking_score": float(move_tracking_score),
        "duration_pass": duration_pass,
        "safety_pass": overall_safety_pass,
        "velocity_guard_ok": velocity_guard_ok,
        "joint_limit_guard_ok": joint_limit_guard_ok,
        "torque_saturation_percentage": float(torque_saturation_percentage),
        "max_abs_qd_radps": float(max_abs_qd_radps),
    }


def move_hold_ranking_key(run_summary: Mapping[str, Any], *, strict: bool = False) -> tuple[Any, ...]:
    metrics = compute_valid_move_hold_metrics(run_summary, strict=strict)
    target_x_delta = abs(_as_float(run_summary.get("target_x_delta", 0.0)))
    return (
        1 if bool(metrics["valid_move_and_hold"]) else 0,
        target_x_delta,
        -abs(_as_float(metrics.get("hold_phase_final_x_error_m", 0.0))),
        -float(metrics.get("hold_phase_x_drift_from_hold_start_m", 0.0)),
        -abs(_as_float(metrics.get("move_phase_final_x_error_m", 0.0))),
        -float(metrics.get("move_phase_max_abs_y_drift_m", 0.0)),
        -float(metrics.get("move_phase_max_abs_z_drift_m", 0.0)),
        -float(metrics.get("hold_phase_max_abs_y_drift_m", 0.0)),
        -float(metrics.get("hold_phase_max_abs_z_drift_m", 0.0)),
        -float(metrics.get("move_phase_max_abs_orientation_error_rad", 0.0)),
        -float(metrics.get("hold_phase_max_abs_orientation_error_rad", 0.0)),
        -float(metrics.get("torque_saturation_percentage", 0.0)),
        -float(metrics.get("max_abs_qd_radps", 0.0)),
        float(metrics.get("move_hold_quality_score", 0.0)),
        float(metrics.get("move_hold_tracking_score", 0.0)),
    )


def transport_ranking_key(run_summary: Mapping[str, Any], *, strict: bool = False) -> tuple[Any, ...]:
    metrics = compute_valid_transport_metrics(run_summary, strict=strict)
    return (
        1 if bool(metrics["valid_transport"]) else 0,
        abs(_as_float(run_summary.get("target_x_delta", 0.0))),
        -abs(_as_float(metrics.get("final_x_error_m", 0.0))),
        -float(metrics.get("max_abs_y_drift_m", 0.0)),
        -float(metrics.get("max_abs_z_drift_m", 0.0)),
        -float(metrics.get("max_abs_orientation_error_rad", 0.0)),
        -float(metrics.get("torque_saturation_percentage", 0.0)),
        -float(metrics.get("max_abs_qd_radps", 0.0)),
        float(metrics.get("transport_quality_score", 0.0)),
        float(metrics.get("tracking_score", 0.0)),
    )


def tracking_ranking_key(run_summary: Mapping[str, Any], *, strict: bool = False) -> tuple[Any, ...]:
    metrics = compute_valid_transport_metrics(run_summary, strict=strict)
    target_x_delta = _as_float(run_summary.get("target_x_delta", 0.0))
    return (
        float(metrics.get("tracking_score", 0.0)),
        float(metrics.get("transport_quality_score", 0.0)),
        1 if bool(metrics["valid_transport"]) else 0,
        -abs(_as_float(metrics.get("final_x_error_m", 0.0))),
        -abs(_as_float(metrics.get("achieved_x_delta_m", 0.0)) - target_x_delta),
        -float(metrics.get("max_abs_y_drift_m", 0.0)),
        -float(metrics.get("max_abs_z_drift_m", 0.0)),
        -float(metrics.get("max_abs_orientation_error_rad", 0.0)),
        -float(metrics.get("torque_saturation_percentage", 0.0)),
        -float(metrics.get("max_abs_qd_radps", 0.0)),
    )


def raw_motion_ranking_key(run_summary: Mapping[str, Any]) -> tuple[Any, ...]:
    metrics = compute_valid_transport_metrics(run_summary, strict=False)
    return (
        float(_as_float(run_summary.get("achieved_x_delta_m", metrics.get("achieved_x_delta_m", 0.0)))),
        -abs(_as_float(metrics.get("final_x_error_m", 0.0))),
        -float(metrics.get("max_abs_y_drift_m", 0.0)),
        -float(metrics.get("max_abs_z_drift_m", 0.0)),
        -float(metrics.get("max_abs_orientation_error_rad", 0.0)),
        -float(metrics.get("torque_saturation_percentage", 0.0)),
        -float(metrics.get("max_abs_qd_radps", 0.0)),
    )


def transport_failure_category(run_summary: Mapping[str, Any], strict: bool = False) -> str:
    metrics = compute_valid_transport_metrics(run_summary, strict=strict)
    if bool(metrics["valid_transport"]):
        return "valid"

    termination_reason = str(run_summary.get("termination_reason", "") or "").lower()
    failure_reason = str(run_summary.get("failure_reason", "") or "").lower()
    safety_reason = str(run_summary.get("safety_reason", "") or "").lower()
    combined = " ".join([termination_reason, failure_reason, safety_reason])

    if "velocity" in combined or not _as_bool(run_summary.get("velocity_guard_ok", True)):
        return "velocity"
    if "joint limit" in combined or "joint_limit" in combined or not _as_bool(run_summary.get("joint_limit_guard_ok", True)):
        return "joint_limit"
    if not bool(metrics["orthogonal_drift_pass"]):
        return "y_drift" if metrics["max_abs_y_drift_m"] >= metrics["max_abs_z_drift_m"] else "z_drift"
    if not bool(metrics["orientation_pass"]):
        return "orientation"
    if not bool(metrics["x_tracking_pass"]):
        if "axis_error" in combined or "x_error" in combined:
            return "target_tracking"
        return "target_tracking"
    if not bool(metrics["duration_pass"]):
        return "duration"
    if _as_float(metrics.get("torque_saturation_percentage", 0.0)) > _torque_saturation_tolerance(strict):
        return "torque_saturation"
    return "other"


def failure_category_counts(rows: Sequence[Mapping[str, Any]], strict: bool = False) -> dict[str, int]:
    counts = Counter(transport_failure_category(row, strict=strict) for row in rows)
    return dict(counts)
