#!/usr/bin/env python3
"""Stage impedance gains for MuJoCo UR5e X-frame transport.

Simulation-only. This runner does not import RL, hardware, RTDE, URScript, or
CoppeliaSim code. It generates temporary candidate YAML configs under the
output directory, runs the existing MuJoCo torque experiment runner, and ranks
results by clean transport rather than raw X displacement.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

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

EXPERIMENT_SCRIPT = REPO_ROOT / "tools" / "ur5e_mujoco_torque_experiments.py"

X_GAIN_GRID = {
    "kp_x": [25.0, 40.0, 60.0, 80.0],
    "kd_x": [8.0, 12.0, 16.0],
    "kp_y": [40.0, 60.0, 80.0, 100.0],
    "kp_z": [80.0, 120.0, 160.0],
}
ROT_GAIN_GRID = {
    "kp_rot": [10.0, 20.0, 30.0],
    "kd_rot": [3.0, 5.0, 8.0],
    "kp_posture": [0.5, 1.0, 2.0],
    "kd_posture": [0.2, 0.5],
}


def _default_output_root() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "outputs" / "ur5e_mujoco_torque_transport" / f"impedance_tuning_{stamp}"


def _parse_float_list(values: list[float] | None, default: list[float]) -> list[float]:
    return [float(v) for v in (values if values is not None else default)]


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
        help="Trajectory profile for all runs.",
    )
    p.add_argument(
        "--target-x-deltas",
        nargs="+",
        type=float,
        default=[0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
        help="Validation target X deltas.",
    )
    p.add_argument(
        "--durations",
        nargs="+",
        type=float,
        default=[1.0, 3.0, 5.0],
        help="Validation durations in seconds.",
    )
    p.add_argument(
        "--torque-limit-scales",
        nargs="+",
        type=float,
        default=[0.5, 0.75, 1.0],
        help="Validation torque-limit scales.",
    )
    p.add_argument(
        "--probe-target-x-deltas",
        nargs="+",
        type=float,
        default=[0.005],
        help="Stage A/B probe target deltas.",
    )
    p.add_argument(
        "--probe-durations",
        nargs="+",
        type=float,
        default=[0.1],
        help="Stage A/B probe durations.",
    )
    p.add_argument(
        "--probe-torque-limit-scales",
        nargs="+",
        type=float,
        default=[0.5],
        help="Stage A/B probe torque-limit scales.",
    )
    p.add_argument("--stage-a-top-k", type=int, default=2, help="How many Stage A candidates advance.")
    p.add_argument("--stage-b-top-k", type=int, default=1, help="How many Stage B candidates advance.")
    p.add_argument("--stage-c-top-k", type=int, default=1, help="How many Stage C candidates advance.")
    p.add_argument(
        "--gain-overrides-json",
        default=None,
        help="Inline JSON object of gain overrides for a single-candidate smoke run.",
    )
    p.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip top-level plots. Child experiment runs are always executed with --no-plot.",
    )
    return p.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        yaml.safe_dump(payload, fp, sort_keys=False)


def _fmt_token(value: float | str) -> str:
    if isinstance(value, str):
        return value.replace(" ", "_")
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def _candidate_label(stage: str, index: int, parent: str | None = None) -> str:
    if parent is None:
        return f"{stage}_{index:03d}"
    return f"{stage}_{parent}_{index:03d}"


def _candidate_config_path(output_root: Path, stage: str, label: str) -> Path:
    return output_root / "candidate_configs" / stage / f"{label}.yaml"


def _run_output_root(output_root: Path, stage: str, candidate_label: str) -> Path:
    return output_root / "_runs" / stage / candidate_label


def _final_run_dir(output_root: Path, stage: str, candidate_label: str, run_label: str) -> Path:
    return output_root / "per_run_traces" / stage / candidate_label / run_label


def _gain_dict(base_gains: dict[str, float], overrides: dict[str, float]) -> dict[str, float]:
    gains = dict(base_gains)
    for key, value in overrides.items():
        if key in GAIN_FIELDS:
            gains[key] = float(value)
    return gains


def _candidate_config_payload(base_cfg: dict[str, Any], gains: dict[str, float]) -> dict[str, Any]:
    payload = copy.deepcopy(base_cfg)
    payload.setdefault("controller", {})
    payload["controller"].setdefault("gains", {})
    payload["controller"]["gains"].update({name: float(gains[name]) for name in GAIN_FIELDS})
    return payload


def _gain_candidate_grid(
    base_gains: dict[str, float],
    *,
    x_grid: dict[str, list[float]],
    rot_grid: dict[str, list[float]] | None = None,
    fixed_overrides: dict[str, float] | None = None,
) -> list[dict[str, float]]:
    fixed_overrides = fixed_overrides or {}
    rot_grid = rot_grid or {}
    candidates: list[dict[str, float]] = []
    x_keys = ("kp_x", "kd_x", "kp_y", "kp_z")
    x_values = [x_grid.get(key, [base_gains[key]]) for key in x_keys]
    rot_keys = tuple(rot_grid.keys())
    rot_values = [rot_grid[key] for key in rot_keys]
    if rot_keys:
        for x_combo in product(*x_values):
            x_overrides = dict(zip(x_keys, x_combo, strict=True))
            for rot_combo in product(*rot_values):
                rot_overrides = dict(zip(rot_keys, rot_combo, strict=True))
                gains = _gain_dict(base_gains, {**fixed_overrides, **x_overrides, **rot_overrides})
                candidates.append(gains)
    else:
        for x_combo in product(*x_values):
            x_overrides = dict(zip(x_keys, x_combo, strict=True))
            gains = _gain_dict(base_gains, {**fixed_overrides, **x_overrides})
            candidates.append(gains)
    return candidates


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
        if key not in GAIN_FIELDS:
            continue
        overrides[key] = float(value)
    return overrides


def _call_experiment(
    *,
    output_root: Path,
    stage: str,
    candidate_label: str,
    candidate_config_path: Path,
    candidate_run_root: Path,
    run_label: str,
    args: argparse.Namespace,
    target_x_delta: float,
    duration_s: float,
    torque_limit_scale: float,
    candidate_gains: dict[str, float],
) -> dict[str, Any]:
    candidate_run_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(EXPERIMENT_SCRIPT),
        "--mode",
        "controller-rollout",
        "--controller-kind",
        "impedance",
        "--gravity-mode",
        args.gravity_mode,
        "--trajectory-profile",
        args.profile,
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
    if args.scene is not None:
        cmd.extend(["--scene", str(args.scene)])
    cmd.append("--no-plot")

    completed = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False, text=True, capture_output=True)
    child_dirs = sorted([p for p in candidate_run_root.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime)
    if not child_dirs:
        raise RuntimeError(
            f"Child experiment produced no run directory for {run_label!r}. "
            f"Return code={completed.returncode}. stdout={completed.stdout!r}. stderr={completed.stderr!r}"
        )
    child_dir = child_dirs[-1]
    final_dir = _final_run_dir(output_root, stage, candidate_label, run_label)
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


def _run_candidate_grid(
    *,
    output_root: Path,
    stage: str,
    candidate_label: str,
    candidate_config_path: Path,
    candidate_gains: dict[str, float],
    args: argparse.Namespace,
    target_x_deltas: Iterable[float],
    durations: Iterable[float],
    torque_limit_scales: Iterable[float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    candidate_run_root = _run_output_root(output_root, stage, candidate_label)
    run_counter = 0
    for duration in durations:
        for torque_limit_scale in torque_limit_scales:
            for target_x_delta in target_x_deltas:
                run_counter += 1
                run_label = (
                    f"{candidate_label}_dx{_fmt_token(target_x_delta)}_dur{_fmt_token(duration)}_scale{_fmt_token(torque_limit_scale)}"
                )
                summary = _call_experiment(
                    output_root=output_root,
                    stage=stage,
                    candidate_label=candidate_label,
                    candidate_config_path=candidate_config_path,
                    candidate_run_root=candidate_run_root,
                    run_label=run_label,
                    args=args,
                    target_x_delta=float(target_x_delta),
                    duration_s=float(duration),
                    torque_limit_scale=float(torque_limit_scale),
                    candidate_gains=candidate_gains,
                )
                row = {
                    "stage": stage,
                    "candidate_label": candidate_label,
                    "candidate_config_path": str(candidate_config_path),
                    "run_label": run_label,
                    "target_x_delta": float(target_x_delta),
                    "duration_s": float(duration),
                    "torque_limit_scale": float(torque_limit_scale),
                    "gravity_mode": args.gravity_mode,
                    "trajectory_profile": args.profile,
                    "controller_kind": "impedance",
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
                    "achieved_x_delta_m": float(summary.get("achieved_x_delta_m", 0.0)),
                    "final_x_error_m": float(summary.get("final_x_error_m", 0.0)),
                    "max_abs_x_error_m": float(summary.get("max_abs_x_error_m", 0.0)),
                    "max_abs_y_drift_m": float(summary.get("max_abs_y_drift_m", 0.0)),
                    "max_abs_z_drift_m": float(summary.get("max_abs_z_drift_m", 0.0)),
                    "max_abs_orthogonal_drift_m": float(summary.get("max_abs_orthogonal_drift_m", 0.0)),
                    "final_orientation_error_rad": float(summary.get("final_orientation_error_rad", 0.0)),
                    "max_abs_orientation_error_rad": float(summary.get("max_abs_orientation_error_rad", 0.0)),
                    "max_abs_qd_radps": float(summary.get("max_abs_qd_radps", 0.0)),
                    "max_abs_tau_applied_nm": float(summary.get("max_abs_tau_applied_nm", 0.0)),
                    "torque_saturation_percentage": float(summary.get("torque_saturation_percentage", 0.0)),
                    "termination_reason": str(summary.get("termination_reason", "")),
                    "failure_time_s": summary.get("failure_time_s"),
                    "subprocess_returncode": int(summary.get("subprocess_returncode", 0)),
                    "trace_path": str(summary.get("trace_path", "")),
                    "summary_path": str(summary.get("summary_path", "")),
                    "envelope_run_dir": str(summary.get("envelope_run_dir", "")),
                }
                row.update({name: float(summary.get(name, candidate_gains[name])) for name in GAIN_FIELDS})
                row["controller_gains"] = {name: float(summary.get(name, candidate_gains[name])) for name in GAIN_FIELDS}
                row["failure_category"] = transport_failure_category(row)
                rows.append(row)
    return rows


def _candidate_result(
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

    result = {
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
    return result


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "stage",
        "candidate_label",
        "candidate_config_path",
        "run_label",
        "controller_kind",
        "gravity_mode",
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
        "max_abs_tau_applied_nm",
        "torque_saturation_percentage",
        "termination_reason",
        "failure_time_s",
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


def _plot_valid_heatmap(rows: list[dict[str, Any]], output_path: Path, *, title: str) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    x_vals = sorted({float(r["target_x_delta"]) for r in rows})
    y_vals = sorted({float(r["duration_s"]) for r in rows})
    matrix = np.full((len(y_vals), len(x_vals)), np.nan, dtype=np.float64)
    for r in rows:
        xi = x_vals.index(float(r["target_x_delta"]))
        yi = y_vals.index(float(r["duration_s"]))
        value = 1.0 if bool(r.get("valid_transport", False)) else 0.0
        matrix[yi, xi] = value if np.isnan(matrix[yi, xi]) else max(matrix[yi, xi], value)

    fig, ax = plt.subplots(figsize=(max(6.0, 0.9 * len(x_vals)), max(4.0, 0.7 * len(y_vals))))
    cmap = plt.get_cmap("RdYlGn")
    im = ax.imshow(matrix, origin="lower", aspect="auto", vmin=0.0, vmax=1.0, cmap=cmap)
    ax.set_xticks(range(len(x_vals)), [f"{x:g}" for x in x_vals])
    ax.set_yticks(range(len(y_vals)), [f"{y:g}" for y in y_vals])
    ax.set_xlabel("target_x_delta [m]")
    ax.set_ylabel("duration [s]")
    ax.set_title(title)
    for yi, dur in enumerate(y_vals):
        for xi, delta in enumerate(x_vals):
            value = matrix[yi, xi]
            if np.isnan(value):
                continue
            label = "PASS" if value >= 0.5 else "FAIL"
            ax.text(xi, yi, label, ha="center", va="center", color="black", fontsize=8, fontweight="bold")
    fig.colorbar(im, ax=ax, label="valid transport")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _plot_candidate_rates(rows: list[dict[str, Any]], output_path: Path, *, title: str) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_candidate[str(row["candidate_label"])].append(row)
    labels = sorted(by_candidate)
    rates = [sum(1 for r in by_candidate[label] if bool(r.get("valid_transport", False))) / max(len(by_candidate[label]), 1) for label in labels]
    counts = [len(by_candidate[label]) for label in labels]
    fig, ax = plt.subplots(figsize=(max(8.0, 0.55 * len(labels)), 4.5))
    ax.bar(labels, rates, color="tab:blue")
    for idx, (count, rate) in enumerate(zip(counts, rates, strict=True)):
        ax.text(idx, rate + 0.02, f"{count} runs", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("valid transport rate")
    ax.set_title(title)
    ax.tick_params(axis="x", labelrotation=45)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _plot_metrics(rows: list[dict[str, Any]], output_path: Path, *, title: str) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    durations = sorted({float(r["duration_s"]) for r in rows})
    metrics = [
        ("final_x_error_m", "Final X error [m]"),
        ("achieved_x_delta_m", "Achieved X delta [m]"),
        ("max_abs_y_drift_m", "Max |Y drift| [m]"),
        ("max_abs_z_drift_m", "Max |Z drift| [m]"),
        ("max_abs_orientation_error_rad", "Max orientation error [rad]"),
        ("max_abs_qd_radps", "Max |qd| [rad/s]"),
        ("torque_saturation_percentage", "Torque saturation [%]"),
    ]
    fig, axes = plt.subplots(4, 2, figsize=(15, 14), sharex=True)
    for ax, (metric_key, ylabel) in zip(axes.flat, metrics, strict=True):
        for duration in durations:
            subset = sorted(
                [r for r in rows if float(r["duration_s"]) == float(duration)],
                key=lambda r: float(r["target_x_delta"]),
            )
            if not subset:
                continue
            xs = [float(r["target_x_delta"]) for r in subset]
            ys = [float(r.get(metric_key, 0.0)) for r in subset]
            ax.plot(xs, ys, marker="o", label=f"{duration:g}s")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    for ax in axes.flat[len(metrics):]:
        ax.axis("off")
    for ax in axes[-1, :]:
        ax.set_xlabel("target_x_delta [m]")
    axes[0, 0].legend(loc="best")
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _best_validation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stage_c_rows = [r for r in rows if str(r.get("stage", "")) == "stage_c"]
    if stage_c_rows:
        return stage_c_rows
    single_rows = [r for r in rows if str(r.get("stage", "")) == "single"]
    if single_rows:
        return single_rows
    return rows


def run() -> int:
    args = parse_args()
    output_root = args.output_root if args.output_root is not None else _default_output_root()
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "candidate_configs").mkdir(parents=True, exist_ok=True)
    (output_root / "per_run_traces").mkdir(parents=True, exist_ok=True)
    (output_root / "plots").mkdir(parents=True, exist_ok=True)

    base_cfg = _load_yaml(args.config)
    base_controller_cfg = base_cfg["controller"]
    base_gains = controller_gain_summary(base_controller_cfg)["controller_gains"]

    rows: list[dict[str, Any]] = []
    stage_candidates: list[dict[str, Any]] = []
    probe_target_x_deltas = _parse_float_list(args.probe_target_x_deltas, [0.005])
    probe_durations = _parse_float_list(args.probe_durations, [0.1])
    probe_torque_limit_scales = _parse_float_list(args.probe_torque_limit_scales, [0.5])
    validation_target_x_deltas = _parse_float_list(args.target_x_deltas, [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
    validation_durations = _parse_float_list(args.durations, [1.0, 3.0, 5.0])
    validation_torque_limit_scales = _parse_float_list(args.torque_limit_scales, [0.5, 0.75, 1.0])

    if args.gain_overrides_json is not None:
        gains = _gain_dict(base_gains, _normalize_gain_overrides(args.gain_overrides_json))
        candidate_label = "single_001"
        candidate_config_path = _candidate_config_path(output_root, "single", candidate_label)
        _write_yaml(candidate_config_path, _candidate_config_payload(base_cfg, gains))
        run_rows = _run_candidate_grid(
            output_root=output_root,
            stage="single",
            candidate_label=candidate_label,
            candidate_config_path=candidate_config_path,
            candidate_gains=gains,
            args=args,
            target_x_deltas=validation_target_x_deltas,
            durations=validation_durations,
            torque_limit_scales=validation_torque_limit_scales,
        )
        rows.extend(run_rows)
        stage_candidates.append(
            _candidate_result(
                stage="single",
                candidate_label=candidate_label,
                candidate_config_path=candidate_config_path,
                candidate_gains=gains,
                rows=run_rows,
            )
        )
    else:
        stage_a_defs = []
        for idx, gains in enumerate(
            _gain_candidate_grid(base_gains, x_grid=X_GAIN_GRID),
            start=1,
        ):
            label = _candidate_label("stage_a", idx)
            stage_a_defs.append((label, gains, None))

        stage_a_results: list[dict[str, Any]] = []
        for idx, (label, gains, parent) in enumerate(stage_a_defs, start=1):
            config_path = _candidate_config_path(output_root, "stage_a", label)
            _write_yaml(config_path, _candidate_config_payload(base_cfg, gains))
            run_rows = _run_candidate_grid(
                output_root=output_root,
                stage="stage_a",
                candidate_label=label,
                candidate_config_path=config_path,
                candidate_gains=gains,
                args=args,
                target_x_deltas=probe_target_x_deltas,
                durations=probe_durations,
                torque_limit_scales=probe_torque_limit_scales,
            )
            rows.extend(run_rows)
            stage_a_results.append(
                _candidate_result(
                    stage="stage_a",
                    candidate_label=label,
                    candidate_config_path=config_path,
                    candidate_gains=gains,
                    rows=run_rows,
                    parent_label=parent,
                )
            )

        stage_a_results.sort(key=lambda r: r["selection_score"], reverse=True)
        stage_b_parent_results = stage_a_results[: max(int(args.stage_a_top_k), 0)]

        stage_b_results: list[dict[str, Any]] = []
        for parent_idx, parent_result in enumerate(stage_b_parent_results, start=1):
            parent_gains = parent_result["candidate_gains"]
            for idx, gains in enumerate(
                _gain_candidate_grid(
                    parent_gains,
                    x_grid={},
                    rot_grid=ROT_GAIN_GRID,
                ),
                start=1,
            ):
                # Merge the parent x-gains with the new orientation/posture sweep.
                merged_gains = _gain_dict(parent_gains, gains)
                label = _candidate_label("stage_b", idx, parent=f"{parent_idx:03d}")
                config_path = _candidate_config_path(output_root, "stage_b", label)
                _write_yaml(config_path, _candidate_config_payload(base_cfg, merged_gains))
                run_rows = _run_candidate_grid(
                    output_root=output_root,
                    stage="stage_b",
                    candidate_label=label,
                    candidate_config_path=config_path,
                    candidate_gains=merged_gains,
                    args=args,
                    target_x_deltas=probe_target_x_deltas,
                    durations=probe_durations,
                    torque_limit_scales=probe_torque_limit_scales,
                )
                rows.extend(run_rows)
                stage_b_results.append(
                    _candidate_result(
                        stage="stage_b",
                        candidate_label=label,
                        candidate_config_path=config_path,
                        candidate_gains=merged_gains,
                        rows=run_rows,
                        parent_label=parent_result["candidate_label"],
                    )
                )

        stage_b_results.sort(key=lambda r: r["selection_score"], reverse=True)
        stage_c_parent_results = stage_b_results[: max(int(args.stage_b_top_k), 0)]

        for parent_idx, parent_result in enumerate(stage_c_parent_results, start=1):
            gains = parent_result["candidate_gains"]
            label = _candidate_label("stage_c", parent_idx, parent=parent_result["candidate_label"])
            config_path = _candidate_config_path(output_root, "stage_c", label)
            _write_yaml(config_path, _candidate_config_payload(base_cfg, gains))
            run_rows = _run_candidate_grid(
                output_root=output_root,
                stage="stage_c",
                candidate_label=label,
                candidate_config_path=config_path,
                candidate_gains=gains,
                args=args,
                target_x_deltas=validation_target_x_deltas,
                durations=validation_durations,
                torque_limit_scales=validation_torque_limit_scales,
            )
            rows.extend(run_rows)
            stage_candidates.append(
                _candidate_result(
                    stage="stage_c",
                    candidate_label=label,
                    candidate_config_path=config_path,
                    candidate_gains=gains,
                    rows=run_rows,
                    parent_label=parent_result["candidate_label"],
                )
            )

        if stage_candidates:
            stage_candidates.sort(key=lambda r: r["selection_score"], reverse=True)
            stage_candidates = stage_candidates[: max(int(args.stage_c_top_k), 0)]

    summary_path = output_root / "summary.json"
    csv_path = output_root / "summary.csv"
    best_path = output_root / "best_settings.json"
    readme_path = output_root / "README.md"

    _write_csv(rows, csv_path)
    analysis_rows = _best_validation_rows(rows)
    summary = {
        "output_root": str(output_root),
        "config_path": str(args.config),
        "scene_xml": str(args.scene) if args.scene is not None else None,
        "gravity_mode": args.gravity_mode,
        "profile": args.profile,
        "validation_target_x_deltas": validation_target_x_deltas,
        "validation_durations": validation_durations,
        "validation_torque_limit_scales": validation_torque_limit_scales,
        "probe_target_x_deltas": probe_target_x_deltas,
        "probe_durations": probe_durations,
        "probe_torque_limit_scales": probe_torque_limit_scales,
        "num_runs": len(rows),
        "num_validation_runs": len(analysis_rows),
        "num_valid_transport": sum(1 for r in analysis_rows if bool(r.get("valid_transport", False))),
        "num_valid_transport_strict": sum(1 for r in analysis_rows if bool(r.get("strict_valid_transport", False))),
        "rows": rows,
        "analysis_rows": analysis_rows,
        "candidate_configs_dir": str(output_root / "candidate_configs"),
        "per_run_traces_dir": str(output_root / "per_run_traces"),
        "plots_dir": str(output_root / "plots"),
        "best_settings_path": str(best_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    best_rows = analysis_rows
    valid_rows = [r for r in best_rows if bool(r.get("valid_transport", False))]
    best_overall_valid_transport = max(valid_rows, key=transport_ranking_key) if valid_rows else None
    best_raw_motion_result = max(best_rows, key=raw_motion_ranking_key) if best_rows else None
    best_tracking_result = max(best_rows, key=tracking_ranking_key) if best_rows else None

    best_valid_transport_by_duration: dict[str, dict[str, Any] | None] = {}
    largest_valid_target_x_delta_by_duration: dict[str, float | None] = {}
    largest_valid_achieved_x_delta_by_duration: dict[str, float | None] = {}
    for duration in validation_durations:
        duration_rows = [r for r in best_rows if float(r["duration_s"]) == float(duration) and bool(r.get("valid_transport", False))]
        best_valid_transport_by_duration[f"{duration:g}"] = max(duration_rows, key=transport_ranking_key) if duration_rows else None
        largest_valid_target_x_delta_by_duration[f"{duration:g}"] = (
            max((float(r["target_x_delta"]) for r in duration_rows), default=None)
        )
        largest_valid_achieved_x_delta_by_duration[f"{duration:g}"] = (
            max((float(r["achieved_x_delta_m"]) for r in duration_rows), default=None)
        )

    failure_counts = Counter(str(r.get("termination_reason", "")) for r in best_rows if not bool(r.get("valid_transport", False)))
    category_counts = failure_category_counts([r for r in best_rows if not bool(r.get("valid_transport", False))])
    dominant_failure_mode = None
    dominant_failure_mode_share = 0.0
    if category_counts:
        dominant_failure_mode, dominant_count = max(category_counts.items(), key=lambda item: (item[1], item[0]))
        dominant_failure_mode_share = dominant_count / max(sum(category_counts.values()), 1)

    def _snapshot(row: dict[str, Any] | None) -> dict[str, Any] | None:
        if row is None:
            return None
        gains = {name: float(row.get(name, 0.0)) for name in GAIN_FIELDS}
        return {
            "stage": row.get("stage"),
            "candidate_label": row.get("candidate_label"),
            "candidate_config_path": row.get("candidate_config_path"),
            "parent_candidate_label": row.get("parent_candidate_label"),
            "controller_kind": row.get("controller_kind"),
            "gravity_mode": row.get("gravity_mode"),
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
            "max_abs_tau_applied_nm": float(row.get("max_abs_tau_applied_nm", 0.0)),
            "torque_saturation_percentage": float(row.get("torque_saturation_percentage", 0.0)),
            "valid_transport": bool(row.get("valid_transport", False)),
            "strict_valid_transport": bool(row.get("strict_valid_transport", False)),
            "tracking_score": float(row.get("tracking_score", 0.0)),
            "transport_quality_score": float(row.get("transport_quality_score", 0.0)),
            "termination_reason": str(row.get("termination_reason", "")),
            "failure_category": str(row.get("failure_category", "")),
            "num_runs": int(row.get("num_runs", 0)),
            "num_valid_transport": int(row.get("num_valid_transport", 0)),
            "num_valid_transport_strict": int(row.get("num_valid_transport_strict", 0)),
            "controller_gains": gains,
            **gains,
        }

    def _candidate_short_snapshot(candidate: dict[str, Any]) -> dict[str, Any]:
        return {
            "stage": candidate.get("stage"),
            "candidate_label": candidate.get("candidate_label"),
            "candidate_config_path": candidate.get("candidate_config_path"),
            "parent_candidate_label": candidate.get("parent_candidate_label"),
            "selection_score": list(candidate.get("selection_score", ())),
            "num_runs": int(candidate.get("num_runs", 0)),
            "num_valid_transport": int(candidate.get("num_valid_transport", 0)),
            "num_valid_transport_strict": int(candidate.get("num_valid_transport_strict", 0)),
            "dominant_failure_mode": candidate.get("dominant_failure_mode"),
            "failure_category_counts": candidate.get("failure_category_counts", {}),
            "best_row": _snapshot(candidate.get("best_row")),
        }

    best_settings = {
        "output_root": str(output_root),
        "analysis_source": (
            "stage_c"
            if any(str(r.get("stage", "")) == "stage_c" for r in rows)
            else ("single" if any(str(r.get("stage", "")) == "single" for r in rows) else "all_rows")
        ),
        "analysis_rows": len(best_rows),
        "total_rows": len(rows),
        "num_valid_transport": sum(1 for r in best_rows if bool(r.get("valid_transport", False))),
        "num_valid_transport_strict": sum(1 for r in best_rows if bool(r.get("strict_valid_transport", False))),
        "top_validation_candidates": [_candidate_short_snapshot(candidate) for candidate in stage_candidates],
        "best_overall": _snapshot(best_overall_valid_transport) or {},
        "best_overall_valid_transport": _snapshot(best_overall_valid_transport) or {},
        "best_gain_set_overall": _snapshot(best_overall_valid_transport) or {},
        "best_valid_transport_by_duration": {
            duration: (_snapshot(row) if row is not None else None) for duration, row in best_valid_transport_by_duration.items()
        },
        "largest_valid_target_x_delta_by_duration": largest_valid_target_x_delta_by_duration,
        "largest_valid_achieved_x_delta_by_duration": largest_valid_achieved_x_delta_by_duration,
        "largest_valid_target_x_delta_overall": max((float(r["target_x_delta"]) for r in valid_rows), default=None),
        "largest_valid_achieved_x_delta_overall": max((float(r["achieved_x_delta_m"]) for r in valid_rows), default=None),
        "best_raw_motion_result": _snapshot(best_raw_motion_result) or {},
        "best_tracking_result": _snapshot(best_tracking_result) or {},
        "best_overall_achieved": _snapshot(best_raw_motion_result) or {},
        "best_by_duration": {
            duration: (_snapshot(row) if row is not None else None) for duration, row in best_valid_transport_by_duration.items()
        },
        "best_by_duration_achieved": {
            duration: (
                _snapshot(
                    max(
                        [r for r in best_rows if float(r["duration_s"]) == float(duration) and bool(r.get("valid_transport", False))],
                        key=raw_motion_ranking_key,
                    )
                )
                if any(float(r["duration_s"]) == float(duration) and bool(r.get("valid_transport", False)) for r in best_rows)
                else None
            )
            for duration in [f"{d:g}" for d in validation_durations]
        },
        "common_failure_modes": [{"termination_reason": reason, "count": count} for reason, count in failure_counts.most_common()],
        "failure_category_counts": category_counts,
        "dominant_failure_mode": dominant_failure_mode,
        "dominant_failure_mode_share": dominant_failure_mode_share,
    }
    best_path.write_text(json.dumps(best_settings, indent=2), encoding="utf-8")

    readme_path.write_text(
        "\n".join(
            [
                "# UR5e MuJoCo Impedance Transport Tuning",
                "",
                f"- output_root: {output_root}",
                f"- gravity_mode: {args.gravity_mode}",
                f"- profile: {args.profile}",
                f"- total_rows: {len(rows)}",
                f"- valid_rows: {sum(1 for r in best_rows if bool(r.get('valid_transport', False)))}",
                f"- dominant_failure_mode: {dominant_failure_mode or 'none'}",
                "",
                "See `summary.csv`, `summary.json`, `best_settings.json`, `candidate_configs/`, `per_run_traces/`, and `plots/`.",
            ]
        ),
        encoding="utf-8",
    )

    if not args.no_plot and best_rows:
        _plot_valid_heatmap(best_rows, output_root / "plots" / "valid_transport_heatmap.png", title="Valid transport by duration")
        _plot_candidate_rates(best_rows, output_root / "plots" / "valid_transport_rate_by_candidate.png", title="Valid transport rate by candidate")
        _plot_metrics(best_rows, output_root / "plots" / "transport_metrics.png", title="Transport metrics by duration")

    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "csv_path": str(csv_path),
                "summary_path": str(summary_path),
                "best_settings_path": str(best_path),
                "num_runs": len(rows),
                "num_valid_transport": sum(1 for r in best_rows if bool(r.get("valid_transport", False))),
                "dominant_failure_mode": dominant_failure_mode,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
