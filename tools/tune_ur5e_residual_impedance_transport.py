#!/usr/bin/env python3
"""Stage residual-torque impedance gains for MuJoCo UR5e transport.

Simulation-only. This runner keeps the true-torque validation path intact,
uses gravity-compensated MuJoCo torque motors as the primary development
environment, and ranks results by clean X transport rather than raw motion.

No RL, hardware, RTDE, URScript, or CoppeliaSim code is imported.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

EXPERIMENT_SCRIPT = REPO_ROOT / "tools" / "ur5e_mujoco_torque_experiments.py"

from tools.tune_ur5e_impedance_transport import (  # noqa: E402
    _candidate_config_payload,
    _candidate_config_path,
    _candidate_label,
    _best_validation_rows,
    _fmt_token,
    _final_run_dir,
    _gain_dict,
    _load_yaml,
    _plot_candidate_rates,
    _plot_metrics,
    _plot_valid_heatmap,
    _run_output_root,
    _write_yaml,
)
from transport_metrics import (  # noqa: E402
    GAIN_FIELDS,
    compute_valid_transport_metrics,
    controller_gain_summary,
    failure_category_counts,
    raw_motion_ranking_key,
    tracking_ranking_key,
    transport_failure_category,
    transport_ranking_key,
)


BASE_STAGE_A_VARIANTS = (
    ("x_kp40_kd12", {"kp_x": 40.0, "kd_x": 12.0}),
    ("x_kp60_kd16", {"kp_x": 60.0, "kd_x": 16.0}),
    ("x_kp80_kd20", {"kp_x": 80.0, "kd_x": 20.0}),
    ("y_kp60_kd15", {"kp_y": 60.0, "kd_y": 15.0}),
    ("y_kp80_kd20", {"kp_y": 80.0, "kd_y": 20.0}),
    ("y_kp100_kd25", {"kp_y": 100.0, "kd_y": 25.0}),
    ("y_kp130_kd25", {"kp_y": 130.0, "kd_y": 25.0}),
    ("z_kp120_kd20", {"kp_z": 120.0, "kd_z": 20.0}),
    ("z_kp140_kd25", {"kp_z": 140.0, "kd_z": 25.0}),
    ("z_kp160_kd30", {"kp_z": 160.0, "kd_z": 30.0}),
    ("z_kp200_kd35", {"kp_z": 200.0, "kd_z": 35.0}),
)

BASE_STAGE_B_VARIANTS = (
    ("rot10_post025", {"kp_rot": 10.0, "kd_rot": 3.0, "kp_posture": 0.25, "kd_posture": 0.2}),
    ("rot15_post05", {"kp_rot": 15.0, "kd_rot": 5.0, "kp_posture": 0.5, "kd_posture": 0.5}),
    ("rot20_post10", {"kp_rot": 20.0, "kd_rot": 8.0, "kp_posture": 1.0, "kd_posture": 0.5}),
    ("rot30_post20", {"kp_rot": 30.0, "kd_rot": 8.0, "kp_posture": 2.0, "kd_posture": 0.8}),
)


def _default_output_roots() -> tuple[Path, Path]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = REPO_ROOT / "outputs" / "ur5e_mujoco_torque_transport"
    tuning_root = base_dir / f"residual_impedance_tuning_{stamp}"
    diagnostics_root = base_dir / f"residual_torque_diagnostics_{stamp}"
    return tuning_root, diagnostics_root


def _parse_float_list(values: list[float] | None, default: list[float]) -> list[float]:
    return [float(v) for v in (values if values is not None else default)]


def _normalize_gain_overrides(raw: str) -> dict[str, float]:
    path = Path(raw)
    if path.exists():
        text = path.read_text(encoding="utf-8")
    else:
        text = raw
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("--gain-overrides-json must decode to a JSON object")
    overrides: dict[str, float] = {}
    for key, value in payload.items():
        if key in GAIN_FIELDS:
            overrides[key] = float(value)
    return overrides


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "config" / "ur5e_mujoco_torque_transport.yaml",
        help="Base MuJoCo transport config YAML.",
    )
    p.add_argument("--scene", type=Path, default=None, help="Optional scene XML override.")
    p.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Tuning output directory. Defaults under outputs/ur5e_mujoco_torque_transport/.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--gravity-mode",
        choices=("raw", "gravity_comp"),
        default="gravity_comp",
        help="Gravity application mode for all runs.",
    )
    p.add_argument(
        "--profile",
        choices=("step", "ramp", "min_jerk"),
        default="min_jerk",
        help="Trajectory profile for transport runs.",
    )
    p.add_argument(
        "--hold-durations",
        nargs="+",
        type=float,
        default=[1.0, 3.0, 5.0],
        help="Durations for the hold diagnostics and Stage A.",
    )
    p.add_argument(
        "--small-target-x-deltas",
        nargs="+",
        type=float,
        default=[0.005, 0.01, 0.02],
        help="Stage B target X deltas.",
    )
    p.add_argument(
        "--small-durations",
        nargs="+",
        type=float,
        default=[1.0, 3.0],
        help="Stage B durations.",
    )
    p.add_argument(
        "--validation-target-x-deltas",
        nargs="+",
        type=float,
        default=[0.03, 0.04, 0.05, 0.06],
        help="Stage C target X deltas.",
    )
    p.add_argument(
        "--validation-durations",
        nargs="+",
        type=float,
        default=[1.0, 3.0, 5.0],
        help="Stage C durations.",
    )
    p.add_argument(
        "--torque-limit-scales",
        nargs="+",
        type=float,
        default=[0.5, 0.75, 1.0],
        help="Torque-limit scaling factors for validation.",
    )
    p.add_argument("--stage-a-top-k", type=int, default=5, help="How many Stage A candidates advance.")
    p.add_argument("--stage-b-top-k", type=int, default=3, help="How many Stage B candidates advance.")
    p.add_argument("--stage-c-top-k", type=int, default=3, help="How many Stage C candidates advance.")
    p.add_argument(
        "--gain-overrides-json",
        default=None,
        help="Inline JSON object of gain overrides for a single-candidate smoke run.",
    )
    p.add_argument(
        "--target-x-deltas",
        nargs="+",
        type=float,
        default=None,
        help="Smoke-run target X deltas when using --gain-overrides-json.",
    )
    p.add_argument(
        "--durations",
        nargs="+",
        type=float,
        default=None,
        help="Smoke-run durations when using --gain-overrides-json.",
    )
    p.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip top-level plots. Child experiment runs are always executed with --no-plot.",
    )
    return p.parse_args()


def _load_base_cfg(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, float]]:
    cfg = _load_yaml(args.config)
    base_controller_cfg = cfg["controller"]
    base_gains = controller_gain_summary(base_controller_cfg)["controller_gains"]
    return cfg, base_gains


def _candidate_payload(base_cfg: dict[str, Any], gains: dict[str, float]) -> dict[str, Any]:
    payload = _candidate_config_payload(base_cfg, gains)
    payload.setdefault("mujoco", {})
    payload["mujoco"]["default_controller"] = "impedance"
    payload["mujoco"]["gravity_mode"] = "gravity_comp"
    return payload


def _run_child_experiment(
    *,
    root_dir: Path,
    stage: str,
    candidate_label: str,
    candidate_config_path: Path,
    run_label: str,
    args: argparse.Namespace,
    mode: str,
    controller_kind: str | None,
    gravity_mode: str,
    profile: str,
    target_x_delta: float,
    duration_s: float,
    torque_limit_scale: float,
    candidate_gains: dict[str, float],
) -> dict[str, Any]:
    candidate_run_root = _run_output_root(root_dir, stage, candidate_label)
    candidate_run_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(EXPERIMENT_SCRIPT),
        "--mode",
        mode,
        "--gravity-mode",
        gravity_mode,
        "--trajectory-profile",
        profile,
        "--target-x-delta",
        str(float(target_x_delta)),
        "--duration",
        str(float(duration_s)),
        "--torque-limit-scale",
        str(float(torque_limit_scale)),
        "--config",
        str(candidate_config_path),
        "--output-dir",
        str(candidate_run_root),
        "--seed",
        str(int(args.seed)),
    ]
    if controller_kind is not None:
        cmd.extend(["--controller-kind", controller_kind])
    if args.scene is not None:
        cmd.extend(["--scene", str(args.scene)])
    if args.no_plot:
        cmd.append("--no-plot")

    completed = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False, text=True, capture_output=True)
    child_dirs = sorted([p for p in candidate_run_root.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime)
    if not child_dirs:
        raise RuntimeError(
            f"Child experiment produced no run directory for {run_label!r}. "
            f"Return code={completed.returncode}. stdout={completed.stdout!r}. stderr={completed.stderr!r}"
        )
    child_dir = child_dirs[-1]
    final_dir = _final_run_dir(root_dir, stage, candidate_label, run_label)
    if final_dir.exists():
        shutil.rmtree(final_dir)
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(child_dir), str(final_dir))
    if not any(candidate_run_root.iterdir()):
        candidate_run_root.rmdir()

    summary_path = final_dir / "summary.json"
    if not summary_path.exists():
        raise RuntimeError(
            f"Child experiment did not write summary.json for {run_label!r}. "
            f"Return code={completed.returncode}. stdout={completed.stdout!r}. stderr={completed.stderr!r}"
        )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["subprocess_returncode"] = int(completed.returncode)
    summary["stdout"] = completed.stdout
    summary["stderr"] = completed.stderr
    summary["stage"] = stage
    summary["candidate_label"] = candidate_label
    summary["candidate_config_path"] = str(candidate_config_path)
    summary["run_label"] = run_label
    summary["envelope_run_dir"] = str(final_dir)
    summary["summary_path"] = str(summary_path)
    summary["trace_path"] = str(final_dir / Path(summary.get("trace_path", "trace.jsonl")).name)
    for key in ("plot_path", "diagnostics_plot_path"):
        value = summary.get(key)
        if value:
            summary[key] = str(final_dir / Path(value).name)
    summary.update(controller_gain_summary({"gains": candidate_gains}))
    summary.update(compute_valid_transport_metrics(summary))
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _run_grid(
    *,
    root_dir: Path,
    stage: str,
    candidate_label: str,
    candidate_config_path: Path,
    args: argparse.Namespace,
    mode: str,
    controller_kind: str | None,
    gravity_mode: str,
    profile: str,
    target_x_deltas: list[float],
    durations: list[float],
    torque_limit_scales: list[float],
    candidate_gains: dict[str, float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for duration in durations:
        for torque_limit_scale in torque_limit_scales:
            for target_x_delta in target_x_deltas:
                run_label = (
                    f"{candidate_label}_dx{_fmt_token(target_x_delta)}_dur{_fmt_token(duration)}_scale{_fmt_token(torque_limit_scale)}"
                )
                summary = _run_child_experiment(
                    root_dir=root_dir,
                    stage=stage,
                    candidate_label=candidate_label,
                    candidate_config_path=candidate_config_path,
                    run_label=run_label,
                    args=args,
                    mode=mode,
                    controller_kind=controller_kind,
                    gravity_mode=gravity_mode,
                    profile=profile,
                    target_x_delta=float(target_x_delta),
                    duration_s=float(duration),
                    torque_limit_scale=float(torque_limit_scale),
                    candidate_gains=candidate_gains,
                )
                row = {
                    "stage": stage,
                    "candidate_label": candidate_label,
                    "parent_candidate_label": None,
                    "candidate_config_path": str(candidate_config_path),
                    "run_label": run_label,
                    "mode": mode,
                    "controller_kind": controller_kind,
                    "gravity_mode": gravity_mode,
                    "trajectory_profile": profile,
                    "target_x_delta": float(target_x_delta),
                    "duration_s": float(duration),
                    "torque_limit_scale": float(torque_limit_scale),
                    "success": bool(summary.get("success", False)),
                    "valid_transport": bool(summary.get("valid_transport", False)),
                    "strict_valid_transport": bool(summary.get("strict_valid_transport", False)),
                    "x_tracking_pass": bool(summary.get("x_tracking_pass", False)),
                    "orthogonal_drift_pass": bool(summary.get("orthogonal_drift_pass", False)),
                    "orientation_pass": bool(summary.get("orientation_pass", False)),
                    "duration_pass": bool(summary.get("duration_pass", False)),
                    "safety_pass": bool(summary.get("safety_pass", False)),
                    "tracking_score": float(summary.get("tracking_score", 0.0)),
                    "transport_quality_score": float(summary.get("transport_quality_score", 0.0)),
                    "strict_tracking_score": float(summary.get("strict_tracking_score", 0.0)),
                    "strict_transport_quality_score": float(summary.get("strict_transport_quality_score", 0.0)),
                    "achieved_x_delta_m": float(summary.get("achieved_x_delta_m", summary.get("final_x_displacement_m", 0.0))),
                    "final_x_error_m": float(summary.get("final_x_error_m", 0.0)),
                    "max_abs_x_error_m": float(summary.get("max_abs_x_error_m", 0.0)),
                    "max_abs_y_drift_m": float(summary.get("max_abs_y_drift_m", 0.0)),
                    "max_abs_z_drift_m": float(summary.get("max_abs_z_drift_m", 0.0)),
                    "max_abs_orthogonal_drift_m": float(summary.get("max_abs_orthogonal_drift_m", 0.0)),
                    "final_orientation_error_rad": float(summary.get("final_orientation_error_rad", 0.0)),
                    "max_abs_orientation_error_rad": float(summary.get("max_abs_orientation_error_rad", 0.0)),
                    "max_abs_qd_radps": float(summary.get("max_abs_qd_radps", 0.0)),
                    "max_abs_tau_controller_nm": float(summary.get("max_abs_tau_controller_nm", 0.0)),
                    "max_abs_tau_gravity_nm": float(summary.get("max_abs_tau_gravity_nm", 0.0)),
                    "max_abs_tau_applied_nm": float(summary.get("max_abs_tau_applied_nm", 0.0)),
                    "mean_abs_tau_controller_nm": float(summary.get("mean_abs_tau_controller_nm", 0.0)),
                    "mean_abs_tau_gravity_nm": float(summary.get("mean_abs_tau_gravity_nm", 0.0)),
                    "mean_abs_tau_applied_nm": float(summary.get("mean_abs_tau_applied_nm", 0.0)),
                    "gravity_torque_fraction": float(summary.get("gravity_torque_fraction", 0.0)),
                    "controller_torque_fraction": float(summary.get("controller_torque_fraction", 0.0)),
                    "controller_torque_clip_fraction": float(summary.get("controller_torque_clip_fraction", 0.0)),
                    "applied_torque_clip_fraction": float(summary.get("applied_torque_clip_fraction", summary.get("torque_clip_fraction", 0.0))),
                    "gravity_compensation_active": bool(summary.get("gravity_compensation_active", False)),
                    "gravity_mode_used": str(summary.get("gravity_mode_used", summary.get("gravity_mode", gravity_mode))),
                    "raw_mode_used": bool(summary.get("raw_mode_used", gravity_mode == "raw")),
                    "max_abs_tau_nm": float(summary.get("max_abs_tau_nm", summary.get("max_abs_tau_applied_nm", 0.0))),
                    "mean_abs_tau_nm": float(summary.get("mean_abs_tau_nm", summary.get("mean_abs_tau_applied_nm", 0.0))),
                    "torque_saturation_percentage": float(summary.get("torque_saturation_percentage", 0.0)),
                    "clipping_count": int(summary.get("clipping_count", 0)),
                    "joint_limit_margin_fraction": float(summary.get("joint_limit_margin_fraction", summary.get("joint_limit_min_fraction", 0.0))),
                    "velocity_guard_margin_radps": float(summary.get("velocity_guard_margin_radps", 0.0)),
                    "timestep_count": int(summary.get("timestep_count", summary.get("steps", 0))),
                    "failure_time_s": summary.get("failure_time_s"),
                    "termination_reason": str(summary.get("termination_reason", "")),
                    "trace_path": str(summary.get("trace_path", "")),
                    "summary_path": str(summary.get("summary_path", "")),
                    "envelope_run_dir": str(summary.get("envelope_run_dir", "")),
                    "subprocess_returncode": int(summary.get("subprocess_returncode", 0)),
                }
                row.update({name: float(summary.get(name, candidate_gains.get(name, 0.0))) for name in GAIN_FIELDS})
                row["controller_gains"] = {name: float(summary.get(name, candidate_gains.get(name, 0.0))) for name in GAIN_FIELDS}
                row["failure_category"] = transport_failure_category(row)
                rows.append(row)
    return rows


def _candidate_summary(
    *,
    stage: str,
    candidate_label: str,
    candidate_config_path: Path,
    candidate_gains: dict[str, float],
    rows: list[dict[str, Any]],
    parent_label: str | None = None,
) -> dict[str, Any]:
    valid_rows = [r for r in rows if bool(r.get("valid_transport", False))]
    best_valid = max(valid_rows, key=transport_ranking_key) if valid_rows else None
    best_tracking = max(rows, key=tracking_ranking_key) if rows else None
    best_raw = max(rows, key=raw_motion_ranking_key) if rows else None
    best_row = best_valid or best_tracking or best_raw
    selection_score = (
        1 if best_valid is not None else 0,
        len(valid_rows),
    )
    if best_valid is not None:
        selection_score += transport_ranking_key(best_valid)
    elif best_tracking is not None:
        selection_score += tracking_ranking_key(best_tracking)
    elif best_raw is not None:
        selection_score += raw_motion_ranking_key(best_raw)

    failure_rows = [r for r in rows if not bool(r.get("valid_transport", False))]
    failure_counts = Counter(str(r.get("termination_reason", "")) for r in failure_rows)
    category_counts = failure_category_counts(failure_rows)
    dominant_failure_mode = None
    if category_counts:
        dominant_failure_mode = max(category_counts.items(), key=lambda item: (item[1], item[0]))[0]

    return {
        "stage": stage,
        "candidate_label": candidate_label,
        "candidate_config_path": str(candidate_config_path),
        "parent_candidate_label": parent_label,
        "candidate_gains": {name: float(candidate_gains[name]) for name in GAIN_FIELDS},
        "candidate_gains_flat": {name: float(candidate_gains[name]) for name in GAIN_FIELDS},
        "rows": rows,
        "best_valid_result": best_valid,
        "best_tracking_result": best_tracking,
        "best_raw_motion_result": best_raw,
        "best_row": best_row,
        "selection_score": selection_score,
        "num_runs": len(rows),
        "num_success": sum(1 for r in rows if bool(r.get("success", False))),
        "num_valid_transport": len(valid_rows),
        "num_valid_transport_strict": sum(1 for r in rows if bool(r.get("strict_valid_transport", False))),
        "failure_category_counts": category_counts,
        "common_failure_modes": [
            {"termination_reason": reason, "count": count}
            for reason, count in failure_counts.most_common()
        ],
        "dominant_failure_mode": dominant_failure_mode,
    }


def _snapshot(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    gains = {name: float(row.get(name, 0.0)) for name in GAIN_FIELDS}
    return {
        "stage": row.get("stage"),
        "candidate_label": row.get("candidate_label"),
        "candidate_config_path": row.get("candidate_config_path"),
        "parent_candidate_label": row.get("parent_candidate_label"),
        "mode": row.get("mode"),
        "controller_kind": row.get("controller_kind"),
        "gravity_mode": row.get("gravity_mode"),
        "gravity_mode_used": row.get("gravity_mode_used"),
        "gravity_compensation_active": bool(row.get("gravity_compensation_active", False)),
        "raw_mode_used": bool(row.get("raw_mode_used", False)),
        "trajectory_profile": row.get("trajectory_profile"),
        "target_x_delta": float(row.get("target_x_delta", 0.0)),
        "duration_s": float(row.get("duration_s", 0.0)),
        "torque_limit_scale": float(row.get("torque_limit_scale", 0.0)),
        "achieved_x_delta_m": float(row.get("achieved_x_delta_m", 0.0)),
        "final_x_error_m": float(row.get("final_x_error_m", 0.0)),
        "max_abs_x_error_m": float(row.get("max_abs_x_error_m", 0.0)),
        "max_abs_y_drift_m": float(row.get("max_abs_y_drift_m", 0.0)),
        "max_abs_z_drift_m": float(row.get("max_abs_z_drift_m", 0.0)),
        "max_abs_orthogonal_drift_m": float(row.get("max_abs_orthogonal_drift_m", 0.0)),
        "final_orientation_error_rad": float(row.get("final_orientation_error_rad", 0.0)),
        "max_abs_orientation_error_rad": float(row.get("max_abs_orientation_error_rad", 0.0)),
        "max_abs_qd_radps": float(row.get("max_abs_qd_radps", 0.0)),
        "max_abs_tau_controller_nm": float(row.get("max_abs_tau_controller_nm", 0.0)),
        "max_abs_tau_gravity_nm": float(row.get("max_abs_tau_gravity_nm", 0.0)),
        "max_abs_tau_applied_nm": float(row.get("max_abs_tau_applied_nm", 0.0)),
        "mean_abs_tau_controller_nm": float(row.get("mean_abs_tau_controller_nm", 0.0)),
        "mean_abs_tau_gravity_nm": float(row.get("mean_abs_tau_gravity_nm", 0.0)),
        "mean_abs_tau_applied_nm": float(row.get("mean_abs_tau_applied_nm", 0.0)),
        "gravity_torque_fraction": float(row.get("gravity_torque_fraction", 0.0)),
        "controller_torque_fraction": float(row.get("controller_torque_fraction", 0.0)),
        "controller_torque_clip_fraction": float(row.get("controller_torque_clip_fraction", 0.0)),
        "applied_torque_clip_fraction": float(row.get("applied_torque_clip_fraction", row.get("torque_clip_fraction", 0.0))),
        "max_abs_tau_nm": float(row.get("max_abs_tau_nm", 0.0)),
        "mean_abs_tau_nm": float(row.get("mean_abs_tau_nm", 0.0)),
        "torque_saturation_percentage": float(row.get("torque_saturation_percentage", 0.0)),
        "valid_transport": bool(row.get("valid_transport", False)),
        "strict_valid_transport": bool(row.get("strict_valid_transport", False)),
        "tracking_score": float(row.get("tracking_score", 0.0)),
        "transport_quality_score": float(row.get("transport_quality_score", 0.0)),
        "termination_reason": str(row.get("termination_reason", "")),
        "trace_path": row.get("trace_path"),
        "summary_path": row.get("summary_path"),
        "envelope_run_label": row.get("envelope_run_label"),
        "envelope_run_dir": row.get("envelope_run_dir"),
        "controller_gains": gains,
        **gains,
    }


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "stage",
        "candidate_label",
        "parent_candidate_label",
        "candidate_config_path",
        "run_label",
        "mode",
        "controller_kind",
        "gravity_mode",
        "gravity_mode_used",
        "gravity_compensation_active",
        "raw_mode_used",
        "trajectory_profile",
        "target_x_delta",
        "duration_s",
        "torque_limit_scale",
        "success",
        "valid_transport",
        "strict_valid_transport",
        "x_tracking_pass",
        "orthogonal_drift_pass",
        "orientation_pass",
        "duration_pass",
        "safety_pass",
        "tracking_score",
        "transport_quality_score",
        "strict_tracking_score",
        "strict_transport_quality_score",
        "achieved_x_delta_m",
        "final_x_error_m",
        "max_abs_x_error_m",
        "max_abs_y_drift_m",
        "max_abs_z_drift_m",
        "max_abs_orthogonal_drift_m",
        "final_orientation_error_rad",
        "max_abs_orientation_error_rad",
        "max_abs_qd_radps",
        "max_abs_tau_controller_nm",
        "max_abs_tau_gravity_nm",
        "max_abs_tau_applied_nm",
        "mean_abs_tau_controller_nm",
        "mean_abs_tau_gravity_nm",
        "mean_abs_tau_applied_nm",
        "gravity_torque_fraction",
        "controller_torque_fraction",
        "controller_torque_clip_fraction",
        "applied_torque_clip_fraction",
        "max_abs_tau_nm",
        "mean_abs_tau_nm",
        "torque_saturation_percentage",
        "clipping_count",
        "joint_limit_margin_fraction",
        "velocity_guard_margin_radps",
        "timestep_count",
        "failure_time_s",
        "termination_reason",
        "trace_path",
        "summary_path",
        "envelope_run_dir",
        "subprocess_returncode",
    ] + list(GAIN_FIELDS)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _stage_a_candidates(base_gains: dict[str, float]) -> list[tuple[str, dict[str, float]]]:
    candidates = [("baseline", dict(base_gains))]
    for label, overrides in BASE_STAGE_A_VARIANTS:
        candidates.append((label, _gain_dict(base_gains, overrides)))
    return candidates


def _stage_b_candidates(parent_gains: dict[str, float]) -> list[tuple[str, dict[str, float]]]:
    candidates: list[tuple[str, dict[str, float]]] = []
    for label, overrides in BASE_STAGE_B_VARIANTS:
        candidates.append((label, _gain_dict(parent_gains, overrides)))
    return candidates


def _run_diagnostics(
    *,
    diagnostics_root: Path,
    args: argparse.Namespace,
    base_cfg: dict[str, Any],
    base_gains: dict[str, float],
) -> list[dict[str, Any]]:
    diag_rows: list[dict[str, Any]] = []
    diagnostics_root.mkdir(parents=True, exist_ok=True)
    (diagnostics_root / "per_run_traces").mkdir(parents=True, exist_ok=True)
    (diagnostics_root / "candidate_configs").mkdir(parents=True, exist_ok=True)

    diag_plan = [
        ("hold_long", "gravity-comp-hold-long", None, "step", 0.0, args.hold_durations, args.torque_limit_scales),
        ("residual_hold", "residual-impedance-hold", "impedance", "step", 0.0, args.hold_durations, args.torque_limit_scales),
        (
            "x_transport",
            "x-transport-minjerk",
            "impedance",
            "min_jerk",
            float(args.small_target_x_deltas[0]),
            [float(args.small_durations[0])],
            [float(args.torque_limit_scales[0])],
        ),
    ]

    for candidate_label, mode, controller_kind, profile, target_x_delta, durations, torque_scales in diag_plan:
        candidate_config_path = _candidate_config_path(diagnostics_root, "diagnostics", candidate_label)
        _write_yaml(candidate_config_path, _candidate_payload(base_cfg, base_gains))
        rows = _run_grid(
            root_dir=diagnostics_root,
            stage="diagnostics",
            candidate_label=candidate_label,
            candidate_config_path=candidate_config_path,
            args=args,
            mode=mode,
            controller_kind=controller_kind,
            gravity_mode="gravity_comp" if mode != "gravity-comp-hold-long" else "gravity_comp",
            profile=profile,
            target_x_deltas=[target_x_delta],
            durations=_parse_float_list(list(durations), [float(durations[0])]),
            torque_limit_scales=_parse_float_list(list(torque_scales), [float(torque_scales[0])]),
            candidate_gains=base_gains,
        )
        diag_rows.extend(rows)
    return diag_rows


def _build_best_settings(rows: list[dict[str, Any]], *, stage_a_rows: list[dict[str, Any]], stage_b_rows: list[dict[str, Any]], stage_c_rows: list[dict[str, Any]], diagnostics_rows: list[dict[str, Any]], tuning_root: Path, diagnostics_root: Path) -> dict[str, Any]:
    transport_rows = [r for r in rows if float(r.get("target_x_delta", 0.0)) > 0.0]
    hold_rows = [r for r in rows if float(r.get("target_x_delta", 0.0)) <= 0.0]
    valid_rows = [r for r in transport_rows if bool(r.get("valid_transport", False))]
    best_overall_valid_transport = max(valid_rows, key=transport_ranking_key) if valid_rows else None
    best_raw_motion_result = max(transport_rows or rows, key=raw_motion_ranking_key) if (transport_rows or rows) else None
    best_tracking_result = max(transport_rows or rows, key=tracking_ranking_key) if (transport_rows or rows) else None

    best_hold_candidates = [r for r in hold_rows if bool(r.get("valid_transport", False))]
    if not best_hold_candidates and stage_a_rows:
        best_hold_candidates = list(stage_a_rows)
    best_hold_row = max(best_hold_candidates, key=transport_ranking_key) if best_hold_candidates else None
    best_transport_1s = max([r for r in transport_rows if float(r.get("duration_s", 0.0)) == 1.0 and bool(r.get("valid_transport", False))], key=transport_ranking_key) if any(float(r.get("duration_s", 0.0)) == 1.0 and bool(r.get("valid_transport", False)) for r in transport_rows) else None
    best_transport_3s = max([r for r in transport_rows if float(r.get("duration_s", 0.0)) == 3.0 and bool(r.get("valid_transport", False))], key=transport_ranking_key) if any(float(r.get("duration_s", 0.0)) == 3.0 and bool(r.get("valid_transport", False)) for r in transport_rows) else None
    best_transport_5s = max([r for r in transport_rows if float(r.get("duration_s", 0.0)) == 5.0 and bool(r.get("valid_transport", False))], key=transport_ranking_key) if any(float(r.get("duration_s", 0.0)) == 5.0 and bool(r.get("valid_transport", False)) for r in transport_rows) else None

    durations = sorted({float(r["duration_s"]) for r in transport_rows})
    best_by_duration: dict[str, dict[str, Any] | None] = {}
    largest_valid_target_x_delta_by_duration: dict[str, float | None] = {}
    largest_valid_achieved_x_delta_by_duration: dict[str, float | None] = {}
    for duration in durations:
        duration_rows = [r for r in transport_rows if float(r["duration_s"]) == float(duration) and bool(r.get("valid_transport", False))]
        best_by_duration[f"{duration:g}"] = max(duration_rows, key=transport_ranking_key) if duration_rows else None
        largest_valid_target_x_delta_by_duration[f"{duration:g}"] = max((float(r["target_x_delta"]) for r in duration_rows), default=None)
        largest_valid_achieved_x_delta_by_duration[f"{duration:g}"] = max((float(r["achieved_x_delta_m"]) for r in duration_rows), default=None)

    failure_rows = [r for r in rows if not bool(r.get("valid_transport", False))]
    failure_counts = Counter(str(r.get("termination_reason", "")) for r in failure_rows)
    category_counts = failure_category_counts(failure_rows)
    dominant_failure_mode = None
    dominant_failure_mode_share = 0.0
    if category_counts:
        dominant_failure_mode, dominant_count = max(category_counts.items(), key=lambda item: (item[1], item[0]))
        dominant_failure_mode_share = dominant_count / max(sum(category_counts.values()), 1)

    def _stage_snapshot(row: dict[str, Any] | None) -> dict[str, Any] | None:
        return _snapshot(row)

    return {
        "output_root": str(tuning_root),
        "diagnostics_root": str(diagnostics_root),
        "analysis_source": "stage_c" if stage_c_rows else ("stage_b" if stage_b_rows else ("stage_a" if stage_a_rows else "all_rows")),
        "total_rows": len(rows),
        "diagnostics_rows": len(diagnostics_rows),
        "stage_a_rows": len(stage_a_rows),
        "stage_b_rows": len(stage_b_rows),
        "stage_c_rows": len(stage_c_rows),
        "num_valid_transport": len(valid_rows),
        "num_valid_transport_strict": sum(1 for r in transport_rows if bool(r.get("strict_valid_transport", False))),
        "best_hold_stability_gain_set": _stage_snapshot(best_hold_row) or {},
        "best_1s_transport_gain_set": _stage_snapshot(best_transport_1s) or {},
        "best_3s_transport_gain_set": _stage_snapshot(best_transport_3s) or {},
        "best_5s_transport_gain_set": _stage_snapshot(best_transport_5s) or {},
        "best_overall": _stage_snapshot(best_overall_valid_transport) or {},
        "best_overall_valid_transport": _stage_snapshot(best_overall_valid_transport) or {},
        "best_gain_set_overall": _stage_snapshot(best_overall_valid_transport) or {},
        "best_valid_transport_by_duration": {duration: _stage_snapshot(row) for duration, row in best_by_duration.items()},
        "largest_valid_target_x_delta_by_duration": largest_valid_target_x_delta_by_duration,
        "largest_valid_achieved_x_delta_by_duration": largest_valid_achieved_x_delta_by_duration,
        "largest_valid_target_x_delta_overall": max((float(r["target_x_delta"]) for r in valid_rows), default=None),
        "largest_valid_achieved_x_delta_overall": max((float(r["achieved_x_delta_m"]) for r in valid_rows), default=None),
        "best_raw_motion_result": _stage_snapshot(best_raw_motion_result) or {},
        "best_tracking_result": _stage_snapshot(best_tracking_result) or {},
        "common_failure_modes": [{"termination_reason": reason, "count": count} for reason, count in failure_counts.most_common()],
        "failure_category_counts": category_counts,
        "dominant_failure_mode": dominant_failure_mode,
        "dominant_failure_mode_share": dominant_failure_mode_share,
        "z_drift_is_dominant": bool(category_counts.get("z_drift", 0) >= max(category_counts.values(), default=0)),
        "y_drift_is_dominant": bool(category_counts.get("y_drift", 0) >= max(category_counts.values(), default=0)),
        "orientation_is_dominant": bool(category_counts.get("orientation", 0) >= max(category_counts.values(), default=0)),
        "rows": rows,
        "diagnostics_rows_payload": diagnostics_rows,
        "best_by_duration": {duration: _stage_snapshot(row) for duration, row in best_by_duration.items()},
        "best_by_duration_achieved": {
            duration: (
                _stage_snapshot(max([r for r in transport_rows if float(r["duration_s"]) == float(duration) and bool(r.get("valid_transport", False))], key=raw_motion_ranking_key))
                if any(float(r["duration_s"]) == float(duration) and bool(r.get("valid_transport", False)) for r in transport_rows)
                else None
            )
            for duration in best_by_duration
        },
    }


def _write_readme(path: Path, *, tuning_root: Path, diagnostics_root: Path, args: argparse.Namespace, rows: list[dict[str, Any]], best_settings: dict[str, Any]) -> None:
    lines = [
        "# UR5e MuJoCo Residual Impedance Transport Tuning",
        "",
        f"- tuning_root: {tuning_root}",
        f"- diagnostics_root: {diagnostics_root}",
        f"- gravity_mode: {args.gravity_mode}",
        f"- profile: {args.profile}",
        f"- total_rows: {len(rows)}",
        f"- valid_rows: {sum(1 for r in rows if float(r.get('target_x_delta', 0.0)) > 0.0 and bool(r.get('valid_transport', False)))}",
        f"- dominant_failure_mode: {best_settings.get('dominant_failure_mode') or 'none'}",
        "",
        "See `summary.csv`, `summary.json`, `best_settings.json`, `candidate_configs/`, `per_run_traces/`, `diagnostics/`, and `plots/`.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run() -> int:
    args = parse_args()
    base_cfg, base_gains = _load_base_cfg(args)

    if args.output_root is None:
        tuning_root, diagnostics_root = _default_output_roots()
    else:
        tuning_root = args.output_root.expanduser().resolve()
        diagnostics_root = tuning_root.parent / f"residual_torque_diagnostics_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    tuning_root.mkdir(parents=True, exist_ok=True)
    diagnostics_root.mkdir(parents=True, exist_ok=True)
    (tuning_root / "candidate_configs").mkdir(parents=True, exist_ok=True)
    (tuning_root / "per_run_traces").mkdir(parents=True, exist_ok=True)
    (tuning_root / "plots").mkdir(parents=True, exist_ok=True)
    (tuning_root / "diagnostics").mkdir(parents=True, exist_ok=True)

    if args.gain_overrides_json is not None:
        gains = _gain_dict(base_gains, _normalize_gain_overrides(args.gain_overrides_json))
        candidate_label = "single_001"
        candidate_config_path = _candidate_config_path(tuning_root, "single", candidate_label)
        _write_yaml(candidate_config_path, _candidate_payload(base_cfg, gains))
        rows = _run_grid(
            root_dir=tuning_root,
            stage="single",
            candidate_label=candidate_label,
            candidate_config_path=candidate_config_path,
            args=args,
            mode="x-transport-minjerk",
            controller_kind="impedance",
            gravity_mode=args.gravity_mode,
            profile=args.profile,
            target_x_deltas=_parse_float_list(args.target_x_deltas, _parse_float_list(args.small_target_x_deltas, [0.005])),
            durations=_parse_float_list(args.durations, _parse_float_list(args.small_durations, [0.05])),
            torque_limit_scales=_parse_float_list(args.torque_limit_scales, [0.5]),
            candidate_gains=gains,
        )
        stage_a_rows: list[dict[str, Any]] = []
        stage_b_rows: list[dict[str, Any]] = []
        stage_c_rows: list[dict[str, Any]] = rows
        diagnostics_rows: list[dict[str, Any]] = []
    else:
        diagnostics_rows = _run_diagnostics(
            diagnostics_root=diagnostics_root,
            args=args,
            base_cfg=base_cfg,
            base_gains=base_gains,
        )
        diag_hold_rows = [r for r in diagnostics_rows if float(r.get("target_x_delta", 0.0)) <= 0.0]
        diag_transport_rows = [r for r in diagnostics_rows if float(r.get("target_x_delta", 0.0)) > 0.0]
        diagnostics_summary = {
            "diagnostics_root": str(diagnostics_root),
            "config_path": str(args.config),
            "scene_xml": str(args.scene) if args.scene is not None else None,
            "gravity_mode": "gravity_comp",
            "profile": args.profile,
            "num_runs": len(diagnostics_rows),
            "num_valid_transport": sum(1 for r in diag_transport_rows if bool(r.get("valid_transport", False))),
            "num_valid_transport_strict": sum(1 for r in diag_transport_rows if bool(r.get("strict_valid_transport", False))),
            "rows": diagnostics_rows,
        }
        diagnostics_summary_path = diagnostics_root / "summary.json"
        diagnostics_csv_path = diagnostics_root / "summary.csv"
        diagnostics_best_path = diagnostics_root / "best_settings.json"
        diagnostics_readme_path = diagnostics_root / "README.md"
        _write_csv(diagnostics_rows, diagnostics_csv_path)
        diagnostics_summary_path.write_text(json.dumps(diagnostics_summary, indent=2), encoding="utf-8")
        diagnostics_best = _build_best_settings(
            diagnostics_rows,
            stage_a_rows=diag_hold_rows,
            stage_b_rows=[],
            stage_c_rows=diag_transport_rows,
            diagnostics_rows=diagnostics_rows,
            tuning_root=diagnostics_root,
            diagnostics_root=diagnostics_root,
        )
        diagnostics_best_path.write_text(json.dumps(diagnostics_best, indent=2), encoding="utf-8")
        _write_readme(
            diagnostics_readme_path,
            tuning_root=diagnostics_root,
            diagnostics_root=diagnostics_root,
            args=args,
            rows=diagnostics_rows,
            best_settings=diagnostics_best,
        )
        for name in ("summary.csv", "summary.json", "best_settings.json", "README.md"):
            shutil.copy2(diagnostics_root / name, tuning_root / "diagnostics" / name)

        rows: list[dict[str, Any]] = []
        stage_a_candidates = _stage_a_candidates(base_gains)
        stage_a_results: list[dict[str, Any]] = []
        for idx, (label, gains) in enumerate(stage_a_candidates, start=1):
            candidate_label = f"{idx:03d}_{label}"
            candidate_config_path = _candidate_config_path(tuning_root, "stage_a", candidate_label)
            _write_yaml(candidate_config_path, _candidate_payload(base_cfg, gains))
            stage_rows = _run_grid(
                root_dir=tuning_root,
                stage="stage_a",
                candidate_label=candidate_label,
                candidate_config_path=candidate_config_path,
                args=args,
                mode="residual-impedance-hold",
                controller_kind="impedance",
                gravity_mode=args.gravity_mode,
                profile="step",
                target_x_deltas=[0.0],
                durations=_parse_float_list(args.hold_durations, [1.0, 3.0, 5.0]),
                torque_limit_scales=_parse_float_list(args.torque_limit_scales, [0.5, 0.75, 1.0]),
                candidate_gains=gains,
            )
            rows.extend(stage_rows)
            stage_a_results.append(
                _candidate_summary(
                    stage="stage_a",
                    candidate_label=candidate_label,
                    candidate_config_path=candidate_config_path,
                    candidate_gains=gains,
                    rows=stage_rows,
                )
            )

        stage_a_results.sort(key=lambda r: r["selection_score"], reverse=True)
        stage_a_top = stage_a_results[: max(int(args.stage_a_top_k), 0)]

        stage_b_results: list[dict[str, Any]] = []
        for parent_idx, parent_result in enumerate(stage_a_top, start=1):
            parent_gains = parent_result["candidate_gains"]
            for child_idx, (label, gains) in enumerate(_stage_b_candidates(parent_gains), start=1):
                candidate_label = f"{parent_idx:03d}_{child_idx:02d}_{label}"
                candidate_config_path = _candidate_config_path(tuning_root, "stage_b", candidate_label)
                _write_yaml(candidate_config_path, _candidate_payload(base_cfg, gains))
                stage_rows = _run_grid(
                    root_dir=tuning_root,
                    stage="stage_b",
                    candidate_label=candidate_label,
                    candidate_config_path=candidate_config_path,
                    args=args,
                    mode="x-transport-minjerk",
                    controller_kind="impedance",
                    gravity_mode=args.gravity_mode,
                    profile="min_jerk",
                    target_x_deltas=_parse_float_list(args.small_target_x_deltas, [0.005, 0.01, 0.02]),
                    durations=_parse_float_list(args.small_durations, [1.0, 3.0]),
                    torque_limit_scales=_parse_float_list(args.torque_limit_scales, [0.5, 0.75, 1.0]),
                    candidate_gains=gains,
                )
                rows.extend(stage_rows)
                stage_b_results.append(
                    _candidate_summary(
                        stage="stage_b",
                        candidate_label=candidate_label,
                        candidate_config_path=candidate_config_path,
                        candidate_gains=gains,
                        rows=stage_rows,
                        parent_label=parent_result["candidate_label"],
                    )
                )

        stage_b_results.sort(key=lambda r: r["selection_score"], reverse=True)
        stage_b_top = stage_b_results[: max(int(args.stage_b_top_k), 0)]

        stage_c_results: list[dict[str, Any]] = []
        for parent_idx, parent_result in enumerate(stage_b_top, start=1):
            gains = parent_result["candidate_gains"]
            candidate_label = f"{parent_idx:03d}_{parent_result['candidate_label']}"
            candidate_config_path = _candidate_config_path(tuning_root, "stage_c", candidate_label)
            _write_yaml(candidate_config_path, _candidate_payload(base_cfg, gains))
            stage_rows = _run_grid(
                root_dir=tuning_root,
                stage="stage_c",
                candidate_label=candidate_label,
                candidate_config_path=candidate_config_path,
                args=args,
                mode="x-transport-minjerk",
                controller_kind="impedance",
                gravity_mode=args.gravity_mode,
                profile="min_jerk",
                target_x_deltas=_parse_float_list(args.validation_target_x_deltas, [0.03, 0.04, 0.05, 0.06]),
                durations=_parse_float_list(args.validation_durations, [1.0, 3.0, 5.0]),
                torque_limit_scales=_parse_float_list(args.torque_limit_scales, [0.5, 0.75, 1.0]),
                candidate_gains=gains,
            )
            rows.extend(stage_rows)
            stage_c_results.append(
                _candidate_summary(
                    stage="stage_c",
                    candidate_label=candidate_label,
                    candidate_config_path=candidate_config_path,
                    candidate_gains=gains,
                    rows=stage_rows,
                    parent_label=parent_result["candidate_label"],
                )
            )

        stage_a_rows = [r for candidate in stage_a_results for r in candidate["rows"]]
        stage_b_rows = [r for candidate in stage_b_results for r in candidate["rows"]]
        stage_c_rows = [r for candidate in stage_c_results for r in candidate["rows"]]

    summary_csv_path = tuning_root / "summary.csv"
    summary_json_path = tuning_root / "summary.json"
    best_path = tuning_root / "best_settings.json"
    readme_path = tuning_root / "README.md"

    _write_csv(rows, summary_csv_path)
    summary = {
        "output_root": str(tuning_root),
        "diagnostics_root": str(diagnostics_root),
        "config_path": str(args.config),
        "scene_xml": str(args.scene) if args.scene is not None else None,
        "gravity_mode": args.gravity_mode,
        "profile": args.profile,
        "hold_durations": args.hold_durations,
        "small_target_x_deltas": args.small_target_x_deltas,
        "small_durations": args.small_durations,
        "validation_target_x_deltas": args.validation_target_x_deltas,
        "validation_durations": args.validation_durations,
        "validation_torque_limit_scales": args.torque_limit_scales,
        "num_runs": len(rows),
        "num_diagnostics_runs": len(diagnostics_rows),
        "num_valid_transport": sum(1 for r in rows if float(r.get("target_x_delta", 0.0)) > 0.0 and bool(r.get("valid_transport", False))),
        "num_valid_transport_strict": sum(1 for r in rows if float(r.get("target_x_delta", 0.0)) > 0.0 and bool(r.get("strict_valid_transport", False))),
        "rows": rows,
        "diagnostics_rows": diagnostics_rows,
        "candidate_configs_dir": str(tuning_root / "candidate_configs"),
        "per_run_traces_dir": str(tuning_root / "per_run_traces"),
        "plots_dir": str(tuning_root / "plots"),
        "diagnostics_dir": str(tuning_root / "diagnostics"),
        "best_settings_path": str(best_path),
    }
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    best_settings = _build_best_settings(
        rows,
        stage_a_rows=stage_a_rows,
        stage_b_rows=stage_b_rows,
        stage_c_rows=stage_c_rows,
        diagnostics_rows=diagnostics_rows,
        tuning_root=tuning_root,
        diagnostics_root=diagnostics_root,
    )
    best_path.write_text(json.dumps(best_settings, indent=2), encoding="utf-8")

    diagnostics_summary_path = tuning_root / "diagnostics" / "diagnostics_summary.json"
    diagnostics_summary_path.write_text(json.dumps({"diagnostics_root": str(diagnostics_root), "rows": diagnostics_rows}, indent=2), encoding="utf-8")
    _write_readme(readme_path, tuning_root=tuning_root, diagnostics_root=diagnostics_root, args=args, rows=rows, best_settings=best_settings)

    if not args.no_plot and rows:
        _plot_valid_heatmap(rows, tuning_root / "plots" / "valid_transport_heatmap.png", title="Valid transport by target and duration")
        _plot_candidate_rates(rows, tuning_root / "plots" / "valid_transport_rate_by_candidate.png", title="Valid transport rate by candidate")
        _plot_metrics(rows, tuning_root / "plots" / "transport_metrics.png", title="Residual impedance transport metrics")
        for mode in sorted({str(r["gravity_mode"]) for r in rows}):
            gravity_rows = [r for r in rows if str(r["gravity_mode"]) == mode]
            _plot_valid_heatmap(
                gravity_rows,
                tuning_root / "plots" / f"valid_transport_heatmap_{mode}.png",
                title=f"Valid transport by target and duration ({mode})",
            )

    print(
        json.dumps(
            {
                "tuning_root": str(tuning_root),
                "diagnostics_root": str(diagnostics_root),
                "csv_path": str(summary_csv_path),
                "json_path": str(summary_json_path),
                "best_settings_path": str(best_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
