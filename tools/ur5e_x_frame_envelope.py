#!/usr/bin/env python3
"""MuJoCo UR5e X-frame envelope study.

Simulation-only. This runner shells out to
``tools/ur5e_mujoco_torque_experiments.py`` for each sweep point, then
collects the per-run summary JSON files into a comparison directory with
CSV, JSON, per-run traces, plots, and a best-settings report.

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

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
EXPERIMENT_SCRIPT = REPO_ROOT / "tools" / "ur5e_mujoco_torque_experiments.py"
from transport_metrics import (  # noqa: E402
    GAIN_FIELDS,
    compute_valid_transport_metrics,
    failure_category_counts,
    raw_motion_ranking_key,
    tracking_ranking_key,
    transport_failure_category,
    transport_ranking_key,
)


def _fmt_token(value: float | str) -> str:
    if isinstance(value, str):
        return value.replace(" ", "_")
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def _default_output_root() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "outputs" / "ur5e_mujoco_torque" / f"x_frame_envelope_{stamp}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--controllers",
        nargs="+",
        default=["impedance"],
        choices=("torque_qp", "impedance"),
        help="Controller kinds to compare.",
    )
    p.add_argument(
        "--gravity-modes",
        nargs="+",
        default=["gravity_comp"],
        choices=("raw", "gravity_comp"),
        help="Gravity application modes to compare.",
    )
    p.add_argument(
        "--profiles",
        nargs="+",
        default=["min_jerk"],
        choices=("step", "ramp", "min_jerk"),
        help="X-frame target profiles to compare.",
    )
    p.add_argument(
        "--target-x-deltas",
        nargs="+",
        type=float,
        default=[0.005, 0.01, 0.02],
        help="Target X displacements to sweep.",
    )
    p.add_argument(
        "--durations",
        nargs="+",
        type=float,
        default=[1.0, 3.0],
        help="Durations to sweep.",
    )
    p.add_argument(
        "--torque-limit-scales",
        nargs="+",
        type=float,
        default=[0.5, 0.75, 1.0],
        help="Torque-limit scaling factors to sweep.",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Envelope output directory. Defaults under outputs/ur5e_mujoco_torque/.",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "config" / "ur5e_mujoco_torque.yaml",
        help="Experiment config YAML.",
    )
    p.add_argument(
        "--start-q-rad",
        nargs=6,
        type=float,
        default=None,
        metavar=("Q1", "Q2", "Q3", "Q4", "Q5", "Q6"),
        help="Optional six-joint start pose in radians forwarded to each child run.",
    )
    p.add_argument("--scene", type=Path, default=None, help="Optional scene override for the experiment runner.")
    p.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable per-run plots in the child experiment runs and skip envelope plots.",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _run_child_experiment(
    *,
    envelope_dir: Path,
    controller: str,
    gravity_mode: str,
    profile: str,
    target_x_delta: float,
    duration: float,
    torque_limit_scale: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    run_label = (
        f"{controller}_{gravity_mode}_{profile}_dx{_fmt_token(target_x_delta)}_dur{_fmt_token(duration)}_scale{_fmt_token(torque_limit_scale)}"
    )
    run_root = envelope_dir / "_runs" / run_label
    run_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(EXPERIMENT_SCRIPT),
        "--mode",
        "controller-rollout",
        "--controller-kind",
        controller,
        "--gravity-mode",
        gravity_mode,
        "--trajectory-profile",
        profile,
        "--target-x-delta",
        str(float(target_x_delta)),
        "--duration",
        str(float(duration)),
        "--torque-limit-scale",
        str(float(torque_limit_scale)),
        "--output-dir",
        str(run_root),
        "--seed",
        str(int(args.seed)),
    ]
    if args.scene is not None:
        cmd.extend(["--scene", str(args.scene)])
    if args.config is not None:
        cmd.extend(["--config", str(args.config)])
    if args.start_q_rad is not None:
        cmd.extend(["--start-q-rad", *[str(float(v)) for v in args.start_q_rad]])
    if args.no_plot:
        cmd.append("--no-plot")

    completed = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False, text=True, capture_output=True)
    child_dirs = sorted([p for p in run_root.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime)
    if not child_dirs:
        raise RuntimeError(
            f"Child experiment produced no run directory for {run_label!r}. "
            f"Return code={completed.returncode}. stdout={completed.stdout!r}. stderr={completed.stderr!r}"
        )
    child_dir = child_dirs[-1]
    final_dir = envelope_dir / "per_run_traces" / run_label
    if final_dir.exists():
        shutil.rmtree(final_dir)
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(child_dir), str(final_dir))
    if not any(run_root.iterdir()):
        run_root.rmdir()
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
    summary["envelope_run_label"] = run_label
    summary["envelope_run_dir"] = str(final_dir)
    summary["controller_kind"] = controller
    summary["gravity_mode"] = gravity_mode
    summary["trajectory_profile"] = profile
    summary["target_x_delta"] = float(target_x_delta)
    summary["duration_s"] = float(duration)
    summary["torque_limit_scale"] = float(torque_limit_scale)
    summary.update(compute_valid_transport_metrics(summary))
    summary["summary_path"] = str(final_dir / "summary.json")
    for key in ("trace_path", "plot_path", "diagnostics_plot_path"):
        value = summary.get(key)
        if value:
            summary[key] = str(final_dir / Path(value).name)
    (final_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "envelope_run_label",
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
        "failure_time_s",
        "termination_reason",
        "achieved_x_delta_m",
        "final_x_error_m",
        "max_abs_x_error_m",
        "max_abs_y_drift_m",
        "max_abs_z_drift_m",
        "max_abs_orthogonal_drift_m",
        "final_orientation_error_rad",
        "max_abs_orientation_error_rad",
        "max_abs_q_rad",
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
        "gravity_compensation_active",
        "gravity_mode_used",
        "raw_mode_used",
        "max_abs_tau_nm",
        "mean_abs_tau_nm",
        "torque_saturation_percentage",
        "clipping_count",
        "joint_limit_margin_fraction",
        "velocity_guard_margin_radps",
        "timestep_count",
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


def _plot_success_heatmap(rows: list[dict[str, Any]], output_path: Path, *, title: str) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    x_vals = sorted({float(r["target_x_delta"]) for r in rows})
    y_vals = sorted({float(r["duration_s"]) for r in rows})
    matrix = np.full((len(y_vals), len(x_vals)), np.nan, dtype=np.float64)
    for r in rows:
        xi = x_vals.index(float(r["target_x_delta"]))
        yi = y_vals.index(float(r["duration_s"]))
        matrix[yi, xi] = 1.0 if bool(r["success"]) else 0.0

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
    fig.colorbar(im, ax=ax, label="success")
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
        ("achieved_x_delta_m", "Achieved X delta [m]"),
        ("max_abs_x_error_m", "Max |X error| [m]"),
        ("max_abs_orientation_error_rad", "Max orientation error [rad]"),
        ("max_abs_qd_radps", "Max |qd| [rad/s]"),
        ("max_abs_tau_applied_nm", "Max |tau_applied| [Nm]"),
        ("torque_saturation_percentage", "Torque saturation [%]"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True)
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
    for ax in axes[-1, :]:
        ax.set_xlabel("target_x_delta [m]")
    axes[0, 0].legend(loc="best")
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _plot_success_by_gravity_mode(rows: list[dict[str, Any]], output_path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    gravity_modes = sorted({str(r["gravity_mode"]) for r in rows})
    success_rates = []
    counts = []
    for mode in gravity_modes:
        subset = [r for r in rows if str(r["gravity_mode"]) == mode]
        counts.append(len(subset))
        success_rates.append(sum(1 for r in subset if bool(r["success"])) / max(len(subset), 1))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(gravity_modes, success_rates, color=["tab:blue", "tab:orange"][: len(gravity_modes)])
    for idx, (count, rate) in enumerate(zip(counts, success_rates, strict=True)):
        ax.text(idx, rate + 0.02, f"{count} runs", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("success rate")
    ax.set_title("Success rate by gravity mode")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _build_best_settings(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def _snapshot(row: dict[str, Any] | None) -> dict[str, Any] | None:
        if row is None:
            return None
        gains = {name: float(row.get(name, 0.0)) for name in GAIN_FIELDS}
        return {
            "controller_kind": row.get("controller_kind"),
            "gravity_mode": row.get("gravity_mode"),
            "trajectory_profile": row.get("trajectory_profile"),
            "torque_limit_scale": float(row.get("torque_limit_scale", 0.0)),
            "target_x_delta": float(row.get("target_x_delta", 0.0)),
            "duration_s": float(row.get("duration_s", 0.0)),
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
            "mean_abs_tau_controller_nm": float(row.get("mean_abs_tau_controller_nm", 0.0)),
            "mean_abs_tau_gravity_nm": float(row.get("mean_abs_tau_gravity_nm", 0.0)),
            "mean_abs_tau_applied_nm": float(row.get("mean_abs_tau_applied_nm", 0.0)),
            "gravity_torque_fraction": float(row.get("gravity_torque_fraction", 0.0)),
            "controller_torque_fraction": float(row.get("controller_torque_fraction", 0.0)),
            "controller_torque_clip_fraction": float(row.get("controller_torque_clip_fraction", 0.0)),
            "applied_torque_clip_fraction": float(row.get("applied_torque_clip_fraction", row.get("torque_clip_fraction", 0.0))),
            "gravity_compensation_active": bool(row.get("gravity_compensation_active", False)),
            "gravity_mode_used": row.get("gravity_mode_used"),
            "raw_mode_used": bool(row.get("raw_mode_used", False)),
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

    durations = sorted({float(r["duration_s"]) for r in rows})
    valid_rows = [r for r in rows if bool(r.get("valid_transport", False))]
    success_rows = [r for r in rows if bool(r.get("success", False))]
    failure_rows = [r for r in rows if not bool(r.get("valid_transport", False))]
    termination_counts = Counter(str(r.get("termination_reason", "")) for r in failure_rows)
    category_counts = failure_category_counts(failure_rows)
    dominant_failure_mode = None
    if category_counts:
        dominant_failure_mode = max(category_counts.items(), key=lambda item: (item[1], item[0]))[0]

    def _pick_best(subset: list[dict[str, Any]], key_fn) -> dict[str, Any] | None:
        return max(subset, key=key_fn) if subset else None

    def _duration_map(subset: list[dict[str, Any]], key_fn) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for duration in durations:
            duration_rows = [r for r in subset if float(r["duration_s"]) == float(duration)]
            best = _pick_best(duration_rows, key_fn)
            out[f"{duration:g}"] = _snapshot(best)
        return out

    best_overall_valid_transport = _snapshot(_pick_best(valid_rows, transport_ranking_key))
    best_valid_transport_by_duration = _duration_map(valid_rows, transport_ranking_key)
    largest_valid_target_x_delta_by_duration = _duration_map(
        valid_rows,
        lambda r: (
            float(r.get("target_x_delta", 0.0)),
            -abs(float(r.get("final_x_error_m", 0.0))),
            -float(r.get("max_abs_y_drift_m", 0.0)),
            -float(r.get("max_abs_z_drift_m", 0.0)),
            -float(r.get("max_abs_orientation_error_rad", 0.0)),
            -float(r.get("torque_saturation_percentage", 0.0)),
            -float(r.get("max_abs_qd_radps", 0.0)),
        ),
    )
    largest_valid_achieved_x_delta_by_duration = _duration_map(
        valid_rows,
        lambda r: (
            float(r.get("achieved_x_delta_m", 0.0)),
            -abs(float(r.get("final_x_error_m", 0.0))),
            -float(r.get("max_abs_y_drift_m", 0.0)),
            -float(r.get("max_abs_z_drift_m", 0.0)),
            -float(r.get("max_abs_orientation_error_rad", 0.0)),
            -float(r.get("torque_saturation_percentage", 0.0)),
            -float(r.get("max_abs_qd_radps", 0.0)),
        ),
    )

    best_raw_motion_result = _snapshot(_pick_best(rows, raw_motion_ranking_key))
    best_tracking_result = _snapshot(_pick_best(rows, tracking_ranking_key))

    largest_valid_target_overall = None
    largest_valid_achieved_overall = None
    if valid_rows:
        largest_valid_target_overall = _snapshot(
            _pick_best(
                valid_rows,
                lambda r: (
                    float(r.get("target_x_delta", 0.0)),
                    -abs(float(r.get("final_x_error_m", 0.0))),
                    -float(r.get("max_abs_y_drift_m", 0.0)),
                    -float(r.get("max_abs_z_drift_m", 0.0)),
                    -float(r.get("max_abs_orientation_error_rad", 0.0)),
                    -float(r.get("torque_saturation_percentage", 0.0)),
                    -float(r.get("max_abs_qd_radps", 0.0)),
                ),
            )
        )
        largest_valid_achieved_overall = _snapshot(
            _pick_best(
                valid_rows,
                lambda r: (
                    float(r.get("achieved_x_delta_m", 0.0)),
                    -abs(float(r.get("final_x_error_m", 0.0))),
                    -float(r.get("max_abs_y_drift_m", 0.0)),
                    -float(r.get("max_abs_z_drift_m", 0.0)),
                    -float(r.get("max_abs_orientation_error_rad", 0.0)),
                    -float(r.get("torque_saturation_percentage", 0.0)),
                    -float(r.get("max_abs_qd_radps", 0.0)),
                ),
            )
        )

    best_overall_dict = best_overall_valid_transport or {}
    best_by_duration_dict = best_valid_transport_by_duration
    best_by_duration_achieved_dict = largest_valid_achieved_x_delta_by_duration

    return {
        "best_overall": best_overall_dict,
        "best_overall_valid_transport": best_overall_valid_transport or {},
        "best_valid_transport_by_duration": best_valid_transport_by_duration,
        "largest_valid_target_x_delta_by_duration": {
            duration: details["target_x_delta"] if details is not None else None
            for duration, details in largest_valid_target_x_delta_by_duration.items()
        },
        "largest_valid_achieved_x_delta_by_duration": {
            duration: details["achieved_x_delta_m"] if details is not None else None
            for duration, details in largest_valid_achieved_x_delta_by_duration.items()
        },
        "largest_valid_target_x_delta_overall": largest_valid_target_overall["target_x_delta"] if largest_valid_target_overall else None,
        "largest_valid_achieved_x_delta_overall": largest_valid_achieved_overall["achieved_x_delta_m"] if largest_valid_achieved_overall else None,
        "best_raw_motion_result": best_raw_motion_result or {},
        "best_tracking_result": best_tracking_result or {},
        "best_overall_achieved": best_raw_motion_result or {},
        "best_by_duration": best_by_duration_dict,
        "best_by_duration_achieved": best_by_duration_achieved_dict,
        "common_failure_modes": [
            {"termination_reason": reason, "count": count}
            for reason, count in termination_counts.most_common()
        ],
        "failure_category_counts": category_counts,
        "dominant_failure_mode": dominant_failure_mode,
        "num_valid_transport": len(valid_rows),
        "num_valid_transport_strict": sum(1 for r in rows if bool(r.get("strict_valid_transport", False))),
    }


def run() -> int:
    args = parse_args()
    envelope_dir = args.output_root if args.output_root is not None else _default_output_root()
    envelope_dir = envelope_dir.expanduser().resolve()
    envelope_dir.mkdir(parents=True, exist_ok=True)
    (envelope_dir / "per_run_traces").mkdir(parents=True, exist_ok=True)
    (envelope_dir / "plots").mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for duration in args.durations:
        for torque_limit_scale in args.torque_limit_scales:
            for gravity_mode in args.gravity_modes:
                for profile in args.profiles:
                    for controller in args.controllers:
                        for target_x_delta in args.target_x_deltas:
                            summary = _run_child_experiment(
                                envelope_dir=envelope_dir,
                                controller=controller,
                                gravity_mode=gravity_mode,
                                profile=profile,
                                target_x_delta=float(target_x_delta),
                                duration=float(duration),
                                torque_limit_scale=float(torque_limit_scale),
                                args=args,
                            )
                            row = {
                                "envelope_run_label": summary["envelope_run_label"],
                                "controller_kind": controller,
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
                                "failure_time_s": summary.get("failure_time_s"),
                                "termination_reason": str(summary.get("termination_reason", "")),
                                "achieved_x_delta_m": float(summary.get("achieved_x_delta_m", summary.get("final_x_displacement_m", 0.0))),
                                "final_x_error_m": float(summary.get("final_x_error_m", 0.0)),
                                "max_abs_x_error_m": float(summary.get("max_abs_x_error_m", 0.0)),
                                "max_abs_y_drift_m": float(summary.get("max_abs_y_drift_m", 0.0)),
                                "max_abs_z_drift_m": float(summary.get("max_abs_z_drift_m", 0.0)),
                                "max_abs_orthogonal_drift_m": float(summary.get("max_abs_orthogonal_drift_m", 0.0)),
                                "final_orientation_error_rad": float(summary.get("final_orientation_error_rad", summary.get("max_abs_orientation_error_rad", 0.0))),
                                "max_abs_orientation_error_rad": float(summary.get("max_abs_orientation_error_rad", 0.0)),
                                "max_abs_q_rad": float(summary.get("max_abs_q_rad", 0.0)),
                                "max_abs_qd_radps": float(summary.get("max_abs_qd_radps", 0.0)),
                                "max_abs_tau_controller_nm": float(summary.get("max_abs_tau_controller_nm", 0.0)),
                                "max_abs_tau_gravity_nm": float(summary.get("max_abs_tau_gravity_nm", 0.0)),
                                "max_abs_tau_applied_nm": float(summary.get("max_abs_tau_applied_nm", summary.get("max_abs_tau_nm", 0.0))),
                                "mean_abs_tau_controller_nm": float(summary.get("mean_abs_tau_controller_nm", 0.0)),
                                "mean_abs_tau_gravity_nm": float(summary.get("mean_abs_tau_gravity_nm", 0.0)),
                                "mean_abs_tau_applied_nm": float(summary.get("mean_abs_tau_applied_nm", summary.get("mean_abs_tau_nm", 0.0))),
                                "gravity_torque_fraction": float(summary.get("gravity_torque_fraction", 0.0)),
                                "controller_torque_fraction": float(summary.get("controller_torque_fraction", 0.0)),
                                "controller_torque_clip_fraction": float(summary.get("controller_torque_clip_fraction", 0.0)),
                                "applied_torque_clip_fraction": float(summary.get("applied_torque_clip_fraction", summary.get("torque_clip_fraction", 0.0))),
                                "gravity_compensation_active": bool(summary.get("gravity_compensation_active", False)),
                                "gravity_mode_used": str(summary.get("gravity_mode_used", summary.get("gravity_mode", ""))),
                                "raw_mode_used": bool(summary.get("raw_mode_used", False)),
                                "max_abs_tau_nm": float(summary.get("max_abs_tau_nm", summary.get("max_abs_tau_applied_nm", 0.0))),
                                "mean_abs_tau_nm": float(summary.get("mean_abs_tau_nm", summary.get("mean_abs_tau_applied_nm", 0.0))),
                                "torque_saturation_percentage": float(summary.get("torque_saturation_percentage", 0.0)),
                                "clipping_count": int(summary.get("clipping_count", 0)),
                                "joint_limit_margin_fraction": float(summary.get("joint_limit_margin_fraction", summary.get("joint_limit_min_fraction", 0.0))),
                                "velocity_guard_margin_radps": float(summary.get("velocity_guard_margin_radps", 0.0)),
                                "timestep_count": int(summary.get("timestep_count", summary.get("steps", 0))),
                                "trace_path": str(summary.get("trace_path", "")),
                                "summary_path": str(summary.get("summary_path", "")),
                                "envelope_run_dir": str(summary.get("envelope_run_dir", "")),
                                "subprocess_returncode": int(summary.get("subprocess_returncode", 0)),
                            }
                            row.update({name: float(summary.get(name, 0.0)) for name in GAIN_FIELDS})
                            row["controller_gains"] = {name: float(summary.get(name, 0.0)) for name in GAIN_FIELDS}
                            row["failure_category"] = transport_failure_category(row)
                            rows.append(row)

    summary_path = envelope_dir / "summary.json"
    csv_path = envelope_dir / "summary.csv"
    best_path = envelope_dir / "best_settings.json"
    readme_path = envelope_dir / "README.md"

    _write_csv(rows, csv_path)
    transport_rows = [r for r in rows if float(r.get("target_x_delta", 0.0)) > 0.0]
    summary = {
        "envelope_dir": str(envelope_dir),
        "num_runs": len(rows),
        "num_success": sum(1 for r in rows if bool(r["success"])),
        "num_failure": sum(1 for r in rows if not bool(r["success"])),
        "num_valid_transport": sum(1 for r in transport_rows if bool(r.get("valid_transport", False))),
        "num_valid_transport_strict": sum(1 for r in transport_rows if bool(r.get("strict_valid_transport", False))),
        "controllers": args.controllers,
        "gravity_modes": args.gravity_modes,
        "profiles": args.profiles,
        "target_x_deltas": args.target_x_deltas,
        "durations": args.durations,
        "torque_limit_scales": args.torque_limit_scales,
        "runs": rows,
        "best_settings_path": str(best_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    best_settings = _build_best_settings(rows)
    best_path.write_text(json.dumps(best_settings, indent=2), encoding="utf-8")
    readme_path.write_text(
        "\n".join(
            [
                "# UR5e MuJoCo X-Frame Envelope",
                "",
                f"- controllers: {', '.join(args.controllers)}",
                f"- gravity modes: {', '.join(args.gravity_modes)}",
                f"- profiles: {', '.join(args.profiles)}",
                f"- target_x_deltas: {', '.join(str(x) for x in args.target_x_deltas)}",
                f"- durations: {', '.join(str(x) for x in args.durations)}",
                f"- torque_limit_scales: {', '.join(str(x) for x in args.torque_limit_scales)}",
                "",
                "See `summary.csv`, `summary.json`, `best_settings.json`, `per_run_traces/`, and `plots/`.",
            ]
        ),
        encoding="utf-8",
    )

    if not args.no_plot and rows:
        for gravity_mode in sorted({str(r["gravity_mode"]) for r in rows}):
            gravity_rows = [r for r in rows if str(r["gravity_mode"]) == gravity_mode]
            _plot_success_by_gravity_mode(gravity_rows, envelope_dir / "plots" / f"success_by_gravity_mode_{gravity_mode}.png")
        for controller in sorted({str(r["controller_kind"]) for r in rows}):
            for gravity_mode in sorted({str(r["gravity_mode"]) for r in rows}):
                for profile in sorted({str(r["trajectory_profile"]) for r in rows}):
                    for torque_scale in sorted({float(r["torque_limit_scale"]) for r in rows}):
                        group_rows = [
                            r
                            for r in rows
                            if str(r["controller_kind"]) == controller
                            and str(r["gravity_mode"]) == gravity_mode
                            and str(r["trajectory_profile"]) == profile
                            and float(r["torque_limit_scale"]) == float(torque_scale)
                        ]
                        if not group_rows:
                            continue
                        title = (
                            f"{controller} | {gravity_mode} | {profile} | scale={torque_scale:g}"
                        )
                        _plot_success_heatmap(
                            group_rows,
                            envelope_dir
                            / "plots"
                            / f"heatmap_{controller}_{gravity_mode}_{profile}_scale{_fmt_token(torque_scale)}.png",
                            title=title,
                        )
                        _plot_metrics(
                            group_rows,
                            envelope_dir
                            / "plots"
                            / f"metrics_{controller}_{gravity_mode}_{profile}_scale{_fmt_token(torque_scale)}.png",
                            title=title,
                        )

    print(
        json.dumps(
            {
                "envelope_dir": str(envelope_dir),
                "csv_path": str(csv_path),
                "json_path": str(summary_path),
                "best_settings_path": str(best_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
