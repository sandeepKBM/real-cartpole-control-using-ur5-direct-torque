"""Unified per-run observability records for the MuJoCo torque lane.

`RunLogger` is a *post-hoc reader*: it builds one `RunRecord` per run from the
`summary.json` dict and `trace.jsonl` rows that the existing entrypoints
(`tools/ur5e_mujoco_torque_experiments.py` via `tools/ur5e_move_hold_transport.py`,
and `tools/audit_ur5e_mujoco_gravity_torque.py`) already write. It never hooks
into the control loop.

Outputs per run: `run_record.json` (sibling of the run's existing summary).
Outputs per sweep: `run_log.jsonl` (crash-safe incremental append) and a
flattened `run_log.csv` snapshot where per-joint dicts become
`<field>__<joint>` columns.

The `backend` field is always present ("mujoco" for now) so a future
CoppeliaSim wiring needs no schema break.
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from controller_core.logging_utils import json_dumps_safe
from controller_core.safety import ImpedanceSafetyConfig
from controller_core.x_axis_cartesian_impedance import JOINT_NAME_ORDER as UR5E_JOINT_ORDER
from transport_metrics import transport_failure_category


# ---------------------------------------------------------------------------
# Trace helpers
# ---------------------------------------------------------------------------


def _read_jsonl(path: str | Path | None) -> list[dict[str, Any]]:
    """Tolerant JSONL reader: skips blank/corrupt lines, returns [] if missing."""

    if path is None:
        return []
    p = Path(path)
    if not p.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
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


def _row_vec(row: Mapping[str, Any], key: str, n: int = 6) -> np.ndarray:
    """Extract a length-``n`` float vector from a trace row; NaN-fill on mismatch."""

    value = row.get(key)
    if value is None:
        return np.full(n, np.nan, dtype=np.float64)
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.shape[0] != n:
        return np.full(n, np.nan, dtype=np.float64)
    return arr


def _as_float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _per_joint_torque_stats(
    trace_rows: Sequence[Mapping[str, Any]],
    torque_limit_nm: Sequence[float] | None,
    joint_names: Sequence[str] = UR5E_JOINT_ORDER,
) -> dict[str, dict[str, Any]]:
    """Per-joint commanded/applied torque maxima, usage fraction, clip counts."""

    n = len(joint_names)
    max_abs_controller = np.zeros(n, dtype=np.float64)
    max_abs_applied = np.zeros(n, dtype=np.float64)
    num_clips = np.zeros(n, dtype=np.int64)
    first_clip_step: list[int | None] = [None] * n
    first_clip_time: list[float | None] = [None] * n
    saw_any = False

    for row in trace_rows:
        tau_controller = _row_vec(row, "tau_controller", n)
        tau_applied = _row_vec(row, "tau_applied", n)
        if np.all(np.isnan(tau_controller)) and np.all(np.isnan(tau_applied)):
            continue
        saw_any = True
        max_abs_controller = np.fmax(max_abs_controller, np.abs(tau_controller))
        max_abs_applied = np.fmax(max_abs_applied, np.abs(tau_applied))
        saturated = row.get("tau_controller_saturated")
        if saturated is not None:
            sat = np.asarray(saturated, dtype=bool).reshape(-1)
            if sat.shape[0] == n:
                for j in range(n):
                    if sat[j]:
                        num_clips[j] += 1
                        if first_clip_step[j] is None:
                            first_clip_step[j] = int(row.get("step", -1))
                            first_clip_time[j] = _as_float_or_none(row.get("time_s"))

    limits = None
    if torque_limit_nm is not None:
        lim = np.asarray(torque_limit_nm, dtype=np.float64).reshape(-1)
        if lim.shape[0] == n:
            limits = lim

    usage: dict[str, float | None] = {}
    for j, name in enumerate(joint_names):
        if not saw_any or limits is None or limits[j] <= 0.0:
            usage[name] = None
        else:
            usage[name] = float(max_abs_applied[j] / limits[j])

    def _f(vec: np.ndarray, j: int) -> float:
        v = float(vec[j])
        return v if np.isfinite(v) else float("nan")

    return {
        "per_joint_max_abs_tau_controller_nm": {
            name: _f(max_abs_controller, j) for j, name in enumerate(joint_names)
        },
        "per_joint_max_abs_tau_applied_nm": {
            name: _f(max_abs_applied, j) for j, name in enumerate(joint_names)
        },
        "per_joint_torque_usage_fraction": usage,
        "per_joint_num_controller_clips": {
            name: int(num_clips[j]) for j, name in enumerate(joint_names)
        },
        "per_joint_first_controller_clip_step": {
            name: first_clip_step[j] for j, name in enumerate(joint_names)
        },
        "per_joint_first_controller_clip_time_s": {
            name: first_clip_time[j] for j, name in enumerate(joint_names)
        },
    }


def _per_joint_qd_stats(
    trace_rows: Sequence[Mapping[str, Any]],
    limit_radps: float,
    joint_names: Sequence[str] = UR5E_JOINT_ORDER,
) -> tuple[dict[str, float], dict[str, float | None]]:
    """Per-joint |qd| maxima and first time each joint exceeded the limit."""

    n = len(joint_names)
    max_abs_qd = np.zeros(n, dtype=np.float64)
    time_to_limit: list[float | None] = [None] * n
    for row in trace_rows:
        qd = _row_vec(row, "qd", n)
        if np.all(np.isnan(qd)):
            continue
        max_abs_qd = np.fmax(max_abs_qd, np.abs(qd))
        t = _as_float_or_none(row.get("time_s"))
        for j in range(n):
            if time_to_limit[j] is None and abs(float(qd[j])) > limit_radps:
                time_to_limit[j] = t
    return (
        {name: float(max_abs_qd[j]) for j, name in enumerate(joint_names)},
        {name: time_to_limit[j] for j, name in enumerate(joint_names)},
    )


def _first_time_exceeding(
    trace_rows: Sequence[Mapping[str, Any]],
    value_fn: Any,
    threshold: float,
) -> float | None:
    """First trace time at which ``value_fn(row)`` exceeds ``threshold``."""

    for row in trace_rows:
        value = value_fn(row)
        if value is None:
            continue
        if abs(float(value)) > threshold:
            return _as_float_or_none(row.get("time_s"))
    return None


def _drift_time_fn(axis_index: int, initial: float | None):
    def _fn(row: Mapping[str, Any]) -> float | None:
        ee = row.get("ee_pos")
        if ee is None or initial is None:
            return None
        arr = np.asarray(ee, dtype=np.float64).reshape(-1)
        if arr.shape[0] != 3:
            return None
        return float(arr[axis_index]) - float(initial)

    return _fn


def _find_first_safety_violation(
    trace_rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> tuple[bool, str | None, float | None, int | None]:
    """(fired, reason, time_s, step) from per-row safety fields, with summary fallback."""

    for row in trace_rows:
        if row.get("safety_ok") is False:
            return (
                True,
                str(row.get("safety_reason", "")) or None,
                _as_float_or_none(row.get("time_s")),
                int(row.get("step", -1)),
            )
    termination = str(summary.get("termination_reason", "") or "")
    if termination and termination not in ("duration_complete",):
        t = _as_float_or_none(summary.get("failure_time_s"))
        if t is None and trace_rows:
            t = _as_float_or_none(trace_rows[-1].get("time_s"))
        step = int(trace_rows[-1].get("step", -1)) if trace_rows else None
        return True, termination, t, step
    return False, None, None, None


def _infer_phase_at_failure(summary: Mapping[str, Any]) -> str:
    """"pre_motion" | "move" | "hold" | "none" from summary-level evidence."""

    if summary.get("move_failure_reason"):
        return "move"
    if summary.get("hold_failure_reason"):
        return "hold"
    if int(summary.get("steps", -1)) == 0:
        return "pre_motion"
    if summary.get("hold_valid_normal") is False:
        return "hold"
    termination = str(summary.get("termination_reason", "") or "")
    success = bool(summary.get("success", termination == "duration_complete"))
    if success:
        return "none"
    move_duration = _as_float_or_none(summary.get("move_duration_s"))
    failure_time = _as_float_or_none(summary.get("failure_time_s"))
    if move_duration is not None and failure_time is not None:
        return "move" if failure_time <= move_duration else "hold"
    return "move"


def _infer_gravity_hold_status(summary: Mapping[str, Any]) -> str:
    value = summary.get("hold_valid_normal")
    if value is None:
        return "not_applicable"
    return "pass" if bool(value) else "fail"


# ---------------------------------------------------------------------------
# Record schema
# ---------------------------------------------------------------------------


@dataclass
class RunRecord:
    # identity / provenance
    run_id: str
    created_at_utc: str
    backend: str
    source_script: str
    run_label: str
    seed: int
    config_path: str
    trace_path: str
    summary_path: str
    # configuration
    controller_kind: str
    gravity_mode: str
    trajectory_profile: str | None
    controller_gains: dict[str, float]
    torque_limit_nm: list[float] | None
    torque_limit_scale: float | None
    # motion
    target_x_delta_m: float
    achieved_x_delta_m: float
    final_x_error_m: float
    max_abs_x_error_m: float
    # drift
    max_abs_y_drift_m: float
    y_drift_time_to_limit_s: float | None
    max_abs_z_drift_m: float
    z_drift_time_to_limit_s: float | None
    # orientation
    final_orientation_error_rad: float
    max_abs_orientation_error_rad: float
    orientation_error_time_to_limit_s: float | None
    # joint velocity
    max_abs_qd_radps: float
    per_joint_max_abs_qd_radps: dict[str, float]
    per_joint_qd_time_to_limit_s: dict[str, float | None]
    # per-joint torque
    per_joint_max_abs_tau_controller_nm: dict[str, float]
    per_joint_max_abs_tau_applied_nm: dict[str, float]
    per_joint_torque_usage_fraction: dict[str, float | None]
    per_joint_num_controller_clips: dict[str, int]
    per_joint_first_controller_clip_step: dict[str, int | None]
    per_joint_first_controller_clip_time_s: dict[str, float | None]
    torque_saturation_percentage: float | None
    # safety / outcome
    safety_guard_fired: bool
    safety_guard_reason: str | None
    safety_guard_fired_time_s: float | None
    safety_guard_fired_step: int | None
    gravity_hold_status: str
    phase_at_failure: str
    outcome: str
    failure_category: str
    termination_reason: str
    # timing
    steps: int
    dt_s: float
    sim_time_s: float
    duration_s: float | None = None
    move_duration_s: float | None = None
    hold_duration_s: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------


class RunLogger:
    """Builds and persists `RunRecord`s for a sweep rooted at ``output_root``."""

    def __init__(
        self,
        *,
        output_root: str | Path,
        source_script: str,
        backend: str = "mujoco",
        sweep_log_name: str = "run_log.jsonl",
        safety_cfg: ImpedanceSafetyConfig | None = None,
    ) -> None:
        self.output_root = Path(output_root)
        self.source_script = str(source_script)
        self.backend = str(backend)
        self.sweep_log_path = self.output_root / sweep_log_name
        self.safety_cfg = safety_cfg or ImpedanceSafetyConfig()

    # -- record construction -------------------------------------------------

    def build_record_from_summary(
        self,
        summary: Mapping[str, Any],
        *,
        run_dir: str | Path,
        seed: int,
        config_path: str | Path,
        run_label: str | None = None,
    ) -> RunRecord:
        run_dir = Path(run_dir)
        trace_path = summary.get("trace_path") or (run_dir / "trace.jsonl")
        trace_rows = _read_jsonl(trace_path)
        cfg = self.safety_cfg

        initial_ee = summary.get("initial_ee_pos")
        y0 = float(initial_ee[1]) if initial_ee is not None else None
        z0 = float(initial_ee[2]) if initial_ee is not None else None

        torque_limit_nm = summary.get("torque_limit_nm")
        torque_stats = _per_joint_torque_stats(trace_rows, torque_limit_nm)
        qd_max, qd_ttl = _per_joint_qd_stats(trace_rows, cfg.max_joint_velocity_radps)
        fired, reason, fired_t, fired_step = _find_first_safety_violation(trace_rows, summary)

        termination = str(summary.get("termination_reason", "") or "")
        if "success" in summary:
            success = bool(summary["success"])
        elif termination:
            success = termination == "duration_complete"
        elif summary.get("hold_valid_normal") is not None:
            success = bool(summary["hold_valid_normal"])
        else:
            # No failure evidence anywhere in the summary: the run ran to completion.
            success = True

        gains = summary.get("controller_gains")
        if not isinstance(gains, Mapping):
            gains = {}

        orientation_fn = lambda row: row.get("orientation_error_norm")  # noqa: E731

        record = RunRecord(
            run_id=uuid.uuid4().hex[:12],
            created_at_utc=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            backend=self.backend,
            source_script=self.source_script,
            run_label=str(run_label or summary.get("run_label", run_dir.name)),
            seed=int(seed),
            config_path=str(config_path),
            trace_path=str(trace_path),
            summary_path=str(summary.get("summary_path", run_dir / "summary.json")),
            controller_kind=str(summary.get("controller_kind", summary.get("controller", "impedance"))),
            gravity_mode=str(summary.get("gravity_mode", summary.get("gravity_mode_used", ""))),
            trajectory_profile=(
                str(summary["trajectory_profile"]) if summary.get("trajectory_profile") else None
            ),
            controller_gains={k: float(v) for k, v in gains.items() if isinstance(v, (int, float))},
            torque_limit_nm=(
                [float(v) for v in torque_limit_nm] if torque_limit_nm is not None else None
            ),
            torque_limit_scale=_as_float_or_none(summary.get("torque_limit_scale")),
            target_x_delta_m=float(summary.get("target_x_delta", summary.get("target_x_delta_m", 0.0)) or 0.0),
            achieved_x_delta_m=float(summary.get("achieved_x_delta_m", 0.0) or 0.0),
            final_x_error_m=float(summary.get("final_x_error_m", 0.0) or 0.0),
            max_abs_x_error_m=float(summary.get("max_abs_x_error_m", 0.0) or 0.0),
            max_abs_y_drift_m=float(summary.get("max_abs_y_drift_m", 0.0) or 0.0),
            y_drift_time_to_limit_s=_first_time_exceeding(
                trace_rows, _drift_time_fn(1, y0), cfg.max_abs_y_drift_m
            ),
            max_abs_z_drift_m=float(summary.get("max_abs_z_drift_m", 0.0) or 0.0),
            z_drift_time_to_limit_s=_first_time_exceeding(
                trace_rows, _drift_time_fn(2, z0), cfg.max_abs_z_drift_m
            ),
            final_orientation_error_rad=float(summary.get("final_orientation_error_rad", 0.0) or 0.0),
            max_abs_orientation_error_rad=float(
                summary.get("max_abs_orientation_error_rad", 0.0) or 0.0
            ),
            orientation_error_time_to_limit_s=_first_time_exceeding(
                trace_rows, orientation_fn, cfg.max_orientation_error_rad
            ),
            max_abs_qd_radps=float(summary.get("max_abs_qd_radps", 0.0) or 0.0),
            per_joint_max_abs_qd_radps=qd_max,
            per_joint_qd_time_to_limit_s=qd_ttl,
            **torque_stats,
            torque_saturation_percentage=_as_float_or_none(
                summary.get("torque_saturation_percentage")
            ),
            safety_guard_fired=fired,
            safety_guard_reason=reason,
            safety_guard_fired_time_s=fired_t,
            safety_guard_fired_step=fired_step,
            gravity_hold_status=_infer_gravity_hold_status(summary),
            phase_at_failure="none" if success else _infer_phase_at_failure(summary),
            outcome="success" if success else "failure",
            failure_category="valid" if success else transport_failure_category(summary),
            termination_reason=termination,
            steps=int(summary.get("steps", len(trace_rows))),
            dt_s=float(summary.get("dt_s", 0.0) or 0.0),
            sim_time_s=float(summary.get("sim_time_s", 0.0) or 0.0),
            duration_s=_as_float_or_none(summary.get("duration_s")),
            move_duration_s=_as_float_or_none(summary.get("move_duration_s")),
            hold_duration_s=_as_float_or_none(summary.get("hold_duration_s")),
        )
        return record

    # -- persistence ----------------------------------------------------------

    def write_run_record(
        self,
        record: RunRecord,
        *,
        run_dir: str | Path,
        filename: str = "run_record.json",
    ) -> Path:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / filename
        path.write_text(json_dumps_safe(asdict(record), indent=2), encoding="utf-8")
        return path

    def append_to_sweep(self, record: RunRecord) -> Path:
        self.sweep_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.sweep_log_path.open("a", encoding="utf-8") as fp:
            fp.write(json_dumps_safe(asdict(record)) + "\n")
        return self.sweep_log_path

    def log_run(
        self,
        summary: Mapping[str, Any],
        *,
        run_dir: str | Path,
        seed: int,
        config_path: str | Path,
        run_label: str | None = None,
    ) -> RunRecord:
        """Convenience: build, write per-run record, append to sweep log."""

        record = self.build_record_from_summary(
            summary, run_dir=run_dir, seed=seed, config_path=config_path, run_label=run_label
        )
        self.write_run_record(record, run_dir=run_dir)
        self.append_to_sweep(record)
        return record

    def write_sweep_csv_snapshot(self, csv_path: str | Path | None = None) -> Path | None:
        """Flatten run_log.jsonl into a CSV; per-joint dicts become <field>__<joint>."""

        rows = _read_jsonl(self.sweep_log_path)
        if not rows:
            return None
        csv_path = Path(csv_path) if csv_path is not None else self.sweep_log_path.with_suffix(".csv")

        flat_rows: list[dict[str, Any]] = []
        for row in rows:
            flat: dict[str, Any] = {}
            for key, value in row.items():
                if key == "extra":
                    continue
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        flat[f"{key}__{sub_key}"] = sub_value
                elif isinstance(value, list):
                    flat[key] = json.dumps(value)
                else:
                    flat[key] = value
            flat_rows.append(flat)

        fieldnames: list[str] = []
        for flat in flat_rows:
            for key in flat:
                if key not in fieldnames:
                    fieldnames.append(key)

        with csv_path.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            for flat in flat_rows:
                writer.writerow(flat)
        return csv_path
