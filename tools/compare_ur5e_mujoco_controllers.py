#!/usr/bin/env python3
"""Compare non-RL MuJoCo UR5e torque controllers.

This is simulation-only. It shells out to ``tools/ur5e_mujoco_torque_experiments.py``
for each sweep point, collects the per-run summary JSON files, and produces a
comparison directory with CSV, JSON, per-run traces, and plots.

No hardware, RTDE, URScript, RL, or CoppeliaSim code is imported.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_SCRIPT = REPO_ROOT / "tools" / "ur5e_mujoco_torque_experiments.py"


def _fmt_token(value: float | str) -> str:
    if isinstance(value, str):
        return value.replace(" ", "_")
    text = f"{float(value):g}"
    text = text.replace("-", "m").replace(".", "p")
    return text


def _default_output_root() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "outputs" / "ur5e_mujoco_torque" / f"comparison_{stamp}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--controllers",
        nargs="+",
        default=["torque_qp", "impedance"],
        choices=("torque_qp", "impedance"),
        help="Controller kinds to compare.",
    )
    p.add_argument(
        "--target-x-deltas",
        nargs="+",
        type=float,
        default=[0.0025, 0.005, 0.01, 0.02, 0.04],
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
        default=[0.25, 0.5, 1.0],
        help="Torque-limit scaling factors to sweep.",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Comparison output directory. Defaults under outputs/ur5e_mujoco_torque/.",
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
        "--no-run-plots",
        action="store_true",
        help="Disable per-run plots in the child experiment runs for faster sweeps.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed forwarded to the child experiment runs.",
    )
    return p.parse_args()


def _run_child_experiment(
    *,
    comparison_dir: Path,
    controller: str,
    target_x_delta: float,
    duration: float,
    torque_limit_scale: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    run_label = (
        f"{controller}_dx{_fmt_token(target_x_delta)}_dur{_fmt_token(duration)}_scale{_fmt_token(torque_limit_scale)}"
    )
    run_root = comparison_dir / "_runs" / run_label
    run_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(EXPERIMENT_SCRIPT),
        "--mode",
        "controller-rollout",
        "--controller-kind",
        controller,
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
    if args.start_q_rad is not None:
        cmd.extend(["--start-q-rad", *[str(float(v)) for v in args.start_q_rad]])
    if args.no_run_plots:
        cmd.append("--no-plot")

    completed = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False, text=True, capture_output=True)
    child_dirs = sorted([p for p in run_root.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime)
    if not child_dirs:
        raise RuntimeError(
            f"Child experiment produced no run directory for {run_label!r}. "
            f"Return code={completed.returncode}. stdout={completed.stdout!r}. stderr={completed.stderr!r}"
        )
    child_dir = child_dirs[-1]
    final_dir = comparison_dir / "per_run_traces" / run_label
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
    summary["comparison_run_label"] = run_label
    summary["comparison_run_dir"] = str(final_dir)
    summary["output_dir"] = str(final_dir)
    summary["controller_kind"] = controller
    summary["target_x_delta"] = float(target_x_delta)
    summary["duration_s"] = float(duration)
    summary["torque_limit_scale"] = float(torque_limit_scale)
    summary["summary_path"] = str(final_dir / "summary.json")
    for key in ("trace_path", "plot_path", "diagnostics_plot_path"):
        value = summary.get(key)
        if value:
            summary[key] = str(final_dir / Path(value).name)
    (final_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "comparison_run_label",
        "controller_kind",
        "target_x_delta",
        "duration_s",
        "torque_limit_scale",
        "success",
        "termination_reason",
        "final_x_error_m",
        "final_ee_error_norm_m",
        "final_x_displacement_m",
        "max_abs_tau_nm",
        "mean_abs_tau_nm",
        "torque_saturation_percentage",
        "clipping_count",
        "max_abs_q_rad",
        "max_abs_qd_radps",
        "joint_limit_min_fraction",
        "velocity_guard_ok",
        "joint_limit_guard_ok",
        "steps",
        "trace_path",
        "summary_path",
        "comparison_run_dir",
        "subprocess_returncode",
    ]
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _plot_group(rows: list[dict[str, Any]], output_path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    controllers = sorted({str(r["controller_kind"]) for r in rows})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    metrics = [
        ("final_x_error_m", "Final X error [m]"),
        ("max_abs_tau_nm", "Max |tau| [Nm]"),
        ("torque_saturation_percentage", "Torque saturation [%]"),
        ("max_abs_qd_radps", "Max |qd| [rad/s]"),
    ]
    for ax, (metric_key, ylabel) in zip(axes.flat, metrics, strict=True):
        for controller in controllers:
            subset = sorted(
                [r for r in rows if str(r["controller_kind"]) == controller],
                key=lambda r: float(r["target_x_delta"]),
            )
            if not subset:
                continue
            xs = [float(r["target_x_delta"]) for r in subset]
            ys = [float(r.get(metric_key, 0.0)) for r in subset]
            ax.plot(xs, ys, marker="o", label=controller)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    for ax in axes[-1, :]:
        ax.set_xlabel("target_x_delta [m]")
    axes[0, 0].legend(loc="best")
    fig.suptitle(
        f"UR5e MuJoCo controller comparison | duration={rows[0]['duration_s']} s | torque_scale={rows[0]['torque_limit_scale']}"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def run() -> int:
    args = parse_args()
    comparison_dir = args.output_root if args.output_root is not None else _default_output_root()
    comparison_dir = comparison_dir.expanduser().resolve()
    comparison_dir.mkdir(parents=True, exist_ok=True)
    (comparison_dir / "per_run_traces").mkdir(parents=True, exist_ok=True)
    (comparison_dir / "plots").mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    sweep_summaries: list[dict[str, Any]] = []
    for duration in args.durations:
        for torque_limit_scale in args.torque_limit_scales:
            group_rows: list[dict[str, Any]] = []
            for controller in args.controllers:
                for target_x_delta in args.target_x_deltas:
                    summary = _run_child_experiment(
                        comparison_dir=comparison_dir,
                        controller=controller,
                        target_x_delta=float(target_x_delta),
                        duration=float(duration),
                        torque_limit_scale=float(torque_limit_scale),
                        args=args,
                    )
                    row = {
                        "comparison_run_label": summary["comparison_run_label"],
                        "controller_kind": controller,
                        "target_x_delta": float(target_x_delta),
                        "duration_s": float(duration),
                        "torque_limit_scale": float(torque_limit_scale),
                        "success": bool(summary.get("success", False)),
                        "termination_reason": str(summary.get("termination_reason", "")),
                        "final_x_error_m": float(summary.get("final_x_error_m", 0.0)),
                        "final_ee_error_norm_m": float(summary.get("final_ee_error_norm_m", 0.0)),
                        "final_x_displacement_m": float(summary.get("final_x_displacement_m", 0.0)),
                        "max_abs_tau_nm": float(summary.get("max_abs_tau_nm", 0.0)),
                        "mean_abs_tau_nm": float(summary.get("mean_abs_tau_nm", 0.0)),
                        "torque_saturation_percentage": float(summary.get("torque_saturation_percentage", 0.0)),
                        "clipping_count": int(summary.get("clipping_count", 0)),
                        "max_abs_q_rad": float(summary.get("max_abs_q_rad", 0.0)),
                        "max_abs_qd_radps": float(summary.get("max_abs_qd_radps", 0.0)),
                        "joint_limit_min_fraction": float(summary.get("joint_limit_min_fraction", 0.0)),
                        "velocity_guard_ok": bool(summary.get("velocity_guard_ok", True)),
                        "joint_limit_guard_ok": bool(summary.get("joint_limit_guard_ok", True)),
                        "steps": int(summary.get("steps", 0)),
                        "trace_path": str(summary.get("trace_path", "")),
                        "summary_path": str(summary.get("summary_path", "")),
                        "comparison_run_dir": str(summary.get("comparison_run_dir", "")),
                        "subprocess_returncode": int(summary.get("subprocess_returncode", 0)),
                    }
                    rows.append(row)
                    group_rows.append(row)
                    sweep_summaries.append(summary)

            plot_path = comparison_dir / "plots" / f"comparison_duration{_fmt_token(duration)}_scale{_fmt_token(torque_limit_scale)}.png"
            if group_rows and not args.no_run_plots:
                _plot_group(group_rows, plot_path)

    csv_path = comparison_dir / "summary.csv"
    _write_csv(rows, csv_path)
    json_path = comparison_dir / "summary.json"
    json_path.write_text(
        json.dumps(
            {
                "comparison_dir": str(comparison_dir),
                "controllers": list(args.controllers),
                "target_x_deltas": [float(x) for x in args.target_x_deltas],
                "durations": [float(x) for x in args.durations],
                "torque_limit_scales": [float(x) for x in args.torque_limit_scales],
                "num_runs": int(len(rows)),
                "num_success": int(sum(1 for row in rows if row["success"])),
                "num_failure": int(sum(1 for row in rows if not row["success"])),
                "runs": rows,
                "child_summaries": sweep_summaries,
                "csv_path": str(csv_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    readme_path = comparison_dir / "README.md"
    readme_path.write_text(
        "\n".join(
            [
                "# UR5e MuJoCo Torque Controller Comparison",
                "",
                "This directory was generated by `tools/compare_ur5e_mujoco_controllers.py`.",
                "",
                f"- controllers: {', '.join(args.controllers)}",
                f"- target_x_deltas: {', '.join(str(x) for x in args.target_x_deltas)}",
                f"- durations: {', '.join(str(x) for x in args.durations)}",
                f"- torque_limit_scales: {', '.join(str(x) for x in args.torque_limit_scales)}",
                "",
                "Inspect `summary.csv` for the per-run comparison table and `plots/` for metric overlays.",
            ]
        ),
        encoding="utf-8",
    )

    print(json.dumps({"comparison_dir": str(comparison_dir), "csv_path": str(csv_path), "json_path": str(json_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
