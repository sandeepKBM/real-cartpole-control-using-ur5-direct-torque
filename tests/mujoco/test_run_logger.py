"""Unit tests for observability.run_logger using synthetic trace/summary fixtures.

No MuJoCo required — everything is driven from hand-built trace rows.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from controller_core.x_axis_cartesian_impedance import JOINT_NAME_ORDER  # noqa: E402
from observability.run_logger import (  # noqa: E402
    RunLogger,
    _find_first_safety_violation,
    _infer_phase_at_failure,
    _per_joint_qd_stats,
    _per_joint_torque_stats,
)

N = len(JOINT_NAME_ORDER)


def _make_row(
    step: int,
    time_s: float,
    *,
    qd: list[float] | None = None,
    tau_controller: list[float] | None = None,
    tau_applied: list[float] | None = None,
    saturated: list[bool] | None = None,
    ee_pos: list[float] | None = None,
    orientation_error_norm: float = 0.0,
    safety_ok: bool = True,
    safety_reason: str = "",
) -> dict:
    return {
        "step": step,
        "time_s": time_s,
        "qd": qd if qd is not None else [0.0] * N,
        "tau_controller": tau_controller if tau_controller is not None else [1.0] * N,
        "tau_applied": tau_applied if tau_applied is not None else [2.0] * N,
        "tau_controller_saturated": saturated if saturated is not None else [False] * N,
        "ee_pos": ee_pos if ee_pos is not None else [0.4, 0.1, 0.5],
        "orientation_error_norm": orientation_error_norm,
        "safety_ok": safety_ok,
        "safety_reason": safety_reason,
    }


def _write_trace(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def _base_summary(run_dir: Path, **overrides) -> dict:
    summary = {
        "success": True,
        "termination_reason": "duration_complete",
        "steps": 3,
        "dt_s": 0.002,
        "sim_time_s": 0.006,
        "initial_ee_pos": [0.4, 0.1, 0.5],
        "trace_path": str(run_dir / "trace.jsonl"),
        "summary_path": str(run_dir / "summary.json"),
        "torque_limit_nm": [150.0, 150.0, 150.0, 28.0, 28.0, 28.0],
        "target_x_delta": 0.02,
        "achieved_x_delta_m": 0.019,
        "final_x_error_m": 0.001,
        "max_abs_x_error_m": 0.002,
        "max_abs_y_drift_m": 0.001,
        "max_abs_z_drift_m": 0.001,
        "max_abs_orientation_error_rad": 0.01,
        "final_orientation_error_rad": 0.005,
        "max_abs_qd_radps": 0.4,
        "torque_saturation_percentage": 0.0,
        "controller_gains": {"kp_x": 40.0, "kd_x": 12.0},
        "gravity_mode": "gravity_comp",
    }
    summary.update(overrides)
    return summary


def test_per_joint_clip_counting() -> None:
    sat_j2 = [False] * N
    sat_j2[2] = True
    rows = [
        _make_row(0, 0.0),
        _make_row(1, 0.002, saturated=sat_j2, tau_controller=[0, 0, 99.0, 0, 0, 0]),
        _make_row(2, 0.004, saturated=sat_j2),
    ]
    stats = _per_joint_torque_stats(rows, [150.0] * N)
    elbow = JOINT_NAME_ORDER[2]
    assert stats["per_joint_num_controller_clips"][elbow] == 2
    assert stats["per_joint_first_controller_clip_step"][elbow] == 1
    assert stats["per_joint_first_controller_clip_time_s"][elbow] == 0.002
    assert stats["per_joint_num_controller_clips"][JOINT_NAME_ORDER[0]] == 0
    assert stats["per_joint_first_controller_clip_step"][JOINT_NAME_ORDER[0]] is None
    assert stats["per_joint_max_abs_tau_controller_nm"][elbow] == 99.0


def test_torque_usage_fraction_uses_applied_and_limit() -> None:
    rows = [_make_row(0, 0.0, tau_applied=[75.0, 0, 0, 0, 0, 0])]
    stats = _per_joint_torque_stats(rows, [150.0] * N)
    assert abs(stats["per_joint_torque_usage_fraction"][JOINT_NAME_ORDER[0]] - 0.5) < 1e-12
    stats_no_limit = _per_joint_torque_stats(rows, None)
    assert stats_no_limit["per_joint_torque_usage_fraction"][JOINT_NAME_ORDER[0]] is None


def test_per_joint_qd_time_to_limit() -> None:
    fast = [0.0] * N
    fast[4] = 2.0  # exceeds 1.5 rad/s default
    rows = [
        _make_row(0, 0.0),
        _make_row(1, 0.1, qd=fast),
        _make_row(2, 0.2, qd=fast),
    ]
    qd_max, ttl = _per_joint_qd_stats(rows, 1.5)
    wrist2 = JOINT_NAME_ORDER[4]
    assert qd_max[wrist2] == 2.0
    assert ttl[wrist2] == 0.1
    assert ttl[JOINT_NAME_ORDER[0]] is None


def test_safety_violation_from_trace_rows() -> None:
    rows = [
        _make_row(0, 0.0),
        _make_row(1, 0.1, safety_ok=False, safety_reason="|Y-Y0| > 0.03 m"),
    ]
    fired, reason, t, step = _find_first_safety_violation(rows, {})
    assert fired and reason == "|Y-Y0| > 0.03 m" and t == 0.1 and step == 1


def test_safety_violation_summary_fallback() -> None:
    rows = [_make_row(0, 0.0)]
    summary = {"termination_reason": "safety_stop", "failure_time_s": 1.25}
    fired, reason, t, _ = _find_first_safety_violation(rows, summary)
    assert fired and reason == "safety_stop" and t == 1.25
    fired, reason, _, _ = _find_first_safety_violation(rows, {"termination_reason": "duration_complete"})
    assert not fired and reason is None


def test_phase_at_failure_branches() -> None:
    assert _infer_phase_at_failure({"move_failure_reason": "vel"}) == "move"
    assert _infer_phase_at_failure({"hold_failure_reason": "drift"}) == "hold"
    assert _infer_phase_at_failure({"steps": 0}) == "pre_motion"
    assert _infer_phase_at_failure({"steps": 10, "hold_valid_normal": False}) == "hold"
    assert (
        _infer_phase_at_failure(
            {"steps": 10, "success": False, "move_duration_s": 1.0, "failure_time_s": 0.5}
        )
        == "move"
    )
    assert (
        _infer_phase_at_failure(
            {"steps": 10, "success": False, "move_duration_s": 1.0, "failure_time_s": 2.5}
        )
        == "hold"
    )


def test_full_record_roundtrip_and_csv(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_a"
    run_dir.mkdir()
    rows = [
        _make_row(0, 0.0),
        _make_row(1, 0.002, ee_pos=[0.4, 0.15, 0.5]),  # +0.05 Y drift > 0.03 limit
        _make_row(2, 0.004, safety_ok=False, safety_reason="|Y-Y0| > 0.03 m"),
    ]
    _write_trace(run_dir / "trace.jsonl", rows)
    summary = _base_summary(
        run_dir,
        success=False,
        termination_reason="safety_stop: |Y-Y0| > 0.03 m",
        max_abs_y_drift_m=0.05,
        failure_time_s=0.004,
    )

    logger = RunLogger(output_root=tmp_path, source_script="test_source.py")
    record = logger.log_run(summary, run_dir=run_dir, seed=7, config_path="cfg.yaml")

    # per-run record file
    on_disk = json.loads((run_dir / "run_record.json").read_text(encoding="utf-8"))
    assert on_disk["backend"] == "mujoco"
    assert on_disk["outcome"] == "failure"
    assert on_disk["safety_guard_fired"] is True
    assert on_disk["safety_guard_reason"] == "|Y-Y0| > 0.03 m"
    assert on_disk["safety_guard_fired_step"] == 2
    assert on_disk["seed"] == 7
    assert set(on_disk["per_joint_max_abs_qd_radps"].keys()) == set(JOINT_NAME_ORDER)
    # y drift crossed the 0.03 limit at t=0.002
    assert on_disk["y_drift_time_to_limit_s"] == 0.002
    assert on_disk["z_drift_time_to_limit_s"] is None
    assert on_disk["failure_category"] in ("y_drift", "z_drift", "velocity", "other", "target_tracking", "duration")

    # sweep log: append a second (successful) run
    run_dir_b = tmp_path / "run_b"
    run_dir_b.mkdir()
    _write_trace(run_dir_b / "trace.jsonl", [_make_row(0, 0.0), _make_row(1, 0.002)])
    logger.log_run(_base_summary(run_dir_b), run_dir=run_dir_b, seed=8, config_path="cfg.yaml")

    log_rows = [
        json.loads(line)
        for line in (tmp_path / "run_log.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(log_rows) == 2
    assert log_rows[0]["outcome"] == "failure" and log_rows[1]["outcome"] == "success"
    assert log_rows[0]["run_id"] != log_rows[1]["run_id"]

    # CSV flattening: per-joint dicts become <field>__<joint> columns
    csv_path = logger.write_sweep_csv_snapshot()
    assert csv_path is not None and csv_path.exists()
    with csv_path.open(encoding="utf-8", newline="") as fp:
        csv_rows = list(csv.DictReader(fp))
    assert len(csv_rows) == 2
    col = f"per_joint_num_controller_clips__{JOINT_NAME_ORDER[0]}"
    assert col in csv_rows[0]
    assert csv_rows[1]["outcome"] == "success"
    assert record.run_label == "run_a"


def test_missing_trace_is_tolerated(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_no_trace"
    run_dir.mkdir()
    summary = _base_summary(run_dir)  # trace file never written
    logger = RunLogger(output_root=tmp_path, source_script="x.py")
    record = logger.build_record_from_summary(summary, run_dir=run_dir, seed=0, config_path="c.yaml")
    assert record.outcome == "success"
    assert record.per_joint_qd_time_to_limit_s[JOINT_NAME_ORDER[0]] is None
    assert not record.safety_guard_fired


def test_hold_variant_without_success_fields_infers_from_hold_validity(tmp_path: Path) -> None:
    """Audit hold variants write neither `success` nor `termination_reason`."""

    run_dir = tmp_path / "hold_variant"
    run_dir.mkdir()
    _write_trace(run_dir / "trace.jsonl", [_make_row(0, 0.0), _make_row(1, 0.002)])
    summary = _base_summary(run_dir, hold_valid_normal=True)
    del summary["success"]
    del summary["termination_reason"]
    logger = RunLogger(output_root=tmp_path, source_script="x.py")
    rec = logger.build_record_from_summary(summary, run_dir=run_dir, seed=0, config_path="c")
    assert rec.outcome == "success"
    assert rec.failure_category == "valid"
    assert rec.phase_at_failure == "none"
    summary_fail = _base_summary(run_dir, hold_valid_normal=False)
    del summary_fail["success"]
    del summary_fail["termination_reason"]
    rec = logger.build_record_from_summary(summary_fail, run_dir=run_dir, seed=0, config_path="c")
    assert rec.outcome == "failure"
    assert rec.phase_at_failure == "hold"


def test_gravity_hold_status(tmp_path: Path) -> None:
    run_dir = tmp_path / "hold"
    run_dir.mkdir()
    _write_trace(run_dir / "trace.jsonl", [_make_row(0, 0.0)])
    logger = RunLogger(output_root=tmp_path, source_script="x.py")
    rec = logger.build_record_from_summary(
        _base_summary(run_dir, hold_valid_normal=True), run_dir=run_dir, seed=0, config_path="c"
    )
    assert rec.gravity_hold_status == "pass"
    rec = logger.build_record_from_summary(
        _base_summary(run_dir, hold_valid_normal=False), run_dir=run_dir, seed=0, config_path="c"
    )
    assert rec.gravity_hold_status == "fail"
    rec = logger.build_record_from_summary(
        _base_summary(run_dir), run_dir=run_dir, seed=0, config_path="c"
    )
    assert rec.gravity_hold_status == "not_applicable"
