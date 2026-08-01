#!/usr/bin/env python3
"""Residual-torque move-and-hold transport sweeps for MuJoCo UR5e.

Simulation-only. This runner keeps the true-torque validation lane intact,
uses gravity-compensated residual torque as the development environment, and
evaluates whether the impedance controller can move to an X target and hold
the transported pose cleanly.

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

BASELINE_GAINS: dict[str, float] = {
    "kp_x": 80.0,
    "kd_x": 20.0,
    "kp_y": 80.0,
    "kd_y": 15.0,
    "kp_z": 120.0,
    "kd_z": 20.0,
    "kp_rot": 30.0,
    "kd_rot": 8.0,
    "kp_posture": 2.0,
    "kd_posture": 0.8,
    "kd_joint": 0.8,
}

from observability.run_logger import RunLogger  # noqa: E402
from simulation.ur5e_mujoco_torque import build_safety_config  # noqa: E402
from tools.tuning_common import (  # noqa: E402
    _candidate_config_payload,
    _candidate_config_path,
    _fmt_token,
    _load_yaml,
    _write_yaml,
)
from transport_metrics import (  # noqa: E402
    GAIN_FIELDS,
    compute_valid_move_hold_metrics,
    controller_gain_summary,
    move_hold_ranking_key,
)


def _default_output_root() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "outputs" / "ur5e_mujoco_torque_transport" / f"move_hold_transport_{stamp}"


def _parse_float_list(values: list[float] | None, default: list[float]) -> list[float]:
    return [float(v) for v in (values if values is not None else default)]


def _normalize_gain_overrides(raw: str) -> dict[str, float]:
    path = Path(raw)
    text = path.read_text(encoding="utf-8") if path.exists() else raw
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
        help="Move-and-hold output directory. Defaults under outputs/ur5e_mujoco_torque_transport/.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--gravity-source",
        choices=("mujoco_qfrc", "pinocchio"),
        default=None,
        help="Forwarded to the child experiment (default: child's own default).",
    )
    p.add_argument(
        "--coriolis-feedforward",
        action="store_true",
        help="Forwarded to the child experiment: add C(q,qd)qd feedforward.",
    )
    p.add_argument(
        "--asymmetric-coulomb-friction",
        action="store_true",
        help=(
            "Forwarded to the child experiment: enable opt-in PLANT-side "
            "asymmetric backdrive Coulomb friction (Clochiatti et al. 2024, "
            "see simulation.ur5e_mujoco_torque.AsymmetricCoulombFrictionConfig)."
        ),
    )
    p.add_argument(
        "--gravity-mode",
        choices=("raw", "gravity_comp"),
        default="gravity_comp",
        help="Gravity application mode for all runs.",
    )
    p.add_argument(
        "--target-x-deltas",
        nargs="+",
        type=float,
        default=[0.01, 0.02, 0.03, 0.04],
        help="Target X deltas for the move-and-hold sweep.",
    )
    p.add_argument(
        "--move-durations",
        nargs="+",
        type=float,
        default=[0.5, 1.0, 1.5, 2.0],
        help="Move durations in seconds.",
    )
    p.add_argument(
        "--hold-durations",
        nargs="+",
        type=float,
        default=[1.0, 2.0, 4.0],
        help="Hold durations in seconds.",
    )
    p.add_argument(
        "--torque-limit-scales",
        nargs="+",
        type=float,
        default=[0.5, 0.75, 1.0],
        help="Torque-limit scaling factors.",
    )
    p.add_argument(
        "--gain-overrides-json",
        default=None,
        help="Optional JSON object of gain overrides for a single-candidate smoke run.",
    )
    p.add_argument(
        "--use-legacy-baseline-gains",
        action="store_true",
        help=(
            "Substitute the hardcoded BASELINE_GAINS constant for the named --config's own "
            "gains before applying --gain-overrides-json. Off by default: without this flag, "
            "the config's own controller.gains are used as-is (the sane, expected behavior "
            "for validating a named config). This flag exists only for reproducing pre-2026-07-30 "
            "sweep behavior, which silently used BASELINE_GAINS regardless of --config unless "
            "every field was re-specified via --gain-overrides-json -- a real, silent-wrong-"
            "results footgun found during the height_alpha=0.2/0.3 validation sweep (see "
            "docs/status/bug_audit_2026-07-29.md and tools/ur5e_pose_sweep_transport.py's "
            "workaround, which is no longer needed after this fix but is left in place as "
            "defensive belt-and-suspenders)."
        ),
    )
    p.add_argument(
        "--start-q-rad",
        nargs=6,
        type=float,
        default=None,
        metavar=("Q1", "Q2", "Q3", "Q4", "Q5", "Q6"),
        help="Optional six-joint start pose in radians.",
    )
    p.add_argument("--no-plot", action="store_true", help="Skip plot generation.")
    return p.parse_args()


def _candidate_label(index: int) -> str:
    return f"baseline_{index:03d}"


def _run_label(target_x_delta: float, move_duration: float, hold_duration: float, torque_limit_scale: float) -> str:
    total_duration = float(move_duration + hold_duration)
    return (
        f"dx{_fmt_token(target_x_delta)}_move{_fmt_token(move_duration)}_hold{_fmt_token(hold_duration)}_"
        f"tot{_fmt_token(total_duration)}_scale{_fmt_token(torque_limit_scale)}"
    )


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    preferred = [
        "stage",
        "candidate_label",
        "candidate_config_path",
        "run_label",
        "target_x_delta",
        "move_duration_s",
        "hold_duration_s",
        "total_duration_s",
        "torque_limit_scale",
        "success",
        "valid_move_phase",
        "valid_hold_phase",
        "valid_move_and_hold",
        "move_failure_reason",
        "hold_failure_reason",
        "move_phase_final_x_error_m",
        "move_phase_achieved_x_delta_m",
        "move_phase_max_abs_x_error_m",
        "move_phase_max_abs_y_drift_m",
        "move_phase_max_abs_z_drift_m",
        "move_phase_max_abs_orientation_error_rad",
        "move_phase_max_abs_qd_radps",
        "move_phase_max_abs_tau_controller_nm",
        "move_phase_max_abs_tau_applied_nm",
        "hold_phase_final_x_error_m",
        "hold_phase_max_abs_x_error_m",
        "hold_phase_x_drift_from_hold_start_m",
        "hold_phase_max_abs_y_drift_m",
        "hold_phase_max_abs_z_drift_m",
        "hold_phase_max_abs_orientation_error_rad",
        "hold_phase_max_abs_qd_radps",
        "hold_phase_max_abs_tau_controller_nm",
        "hold_phase_max_abs_tau_applied_nm",
        "max_abs_qd_radps",
        "torque_saturation_percentage",
        "gravity_mode",
        "gravity_mode_used",
        "gravity_compensation_active",
        "raw_mode_used",
        "termination_reason",
        "failure_reason",
        "trace_path",
        "summary_path",
    ]
    gain_fields = [field for field in GAIN_FIELDS if field not in preferred]
    remaining = sorted(
        {
            key
            for row in rows
            for key in row.keys()
            if key not in preferred and key not in gain_fields
        }
    )
    fieldnames = preferred + gain_fields + remaining
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _row_snapshot(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    gains = {name: float(row.get(name, 0.0)) for name in GAIN_FIELDS}
    return {
        "stage": row.get("stage"),
        "candidate_label": row.get("candidate_label"),
        "candidate_config_path": row.get("candidate_config_path"),
        "run_label": row.get("run_label"),
        "controller_kind": row.get("controller_kind"),
        "gravity_mode": row.get("gravity_mode"),
        "trajectory_profile": row.get("trajectory_profile"),
        "target_x_delta": float(row.get("target_x_delta", 0.0)),
        "move_duration_s": float(row.get("move_duration_s", 0.0)),
        "hold_duration_s": float(row.get("hold_duration_s", 0.0)),
        "total_duration_s": float(row.get("total_duration_s", 0.0)),
        "torque_limit_scale": float(row.get("torque_limit_scale", 0.0)),
        "valid_move_phase": bool(row.get("valid_move_phase", False)),
        "valid_hold_phase": bool(row.get("valid_hold_phase", False)),
        "valid_move_and_hold": bool(row.get("valid_move_and_hold", False)),
        "move_phase_final_x_error_m": float(row.get("move_phase_final_x_error_m", 0.0)),
        "move_phase_achieved_x_delta_m": float(row.get("move_phase_achieved_x_delta_m", 0.0)),
        "move_phase_max_abs_x_error_m": float(row.get("move_phase_max_abs_x_error_m", 0.0)),
        "move_phase_max_abs_y_drift_m": float(row.get("move_phase_max_abs_y_drift_m", 0.0)),
        "move_phase_max_abs_z_drift_m": float(row.get("move_phase_max_abs_z_drift_m", 0.0)),
        "move_phase_max_abs_orientation_error_rad": float(row.get("move_phase_max_abs_orientation_error_rad", 0.0)),
        "hold_phase_final_x_error_m": float(row.get("hold_phase_final_x_error_m", 0.0)),
        "hold_phase_max_abs_x_error_m": float(row.get("hold_phase_max_abs_x_error_m", 0.0)),
        "hold_phase_x_drift_from_hold_start_m": float(row.get("hold_phase_x_drift_from_hold_start_m", 0.0)),
        "hold_phase_max_abs_y_drift_m": float(row.get("hold_phase_max_abs_y_drift_m", 0.0)),
        "hold_phase_max_abs_z_drift_m": float(row.get("hold_phase_max_abs_z_drift_m", 0.0)),
        "hold_phase_max_abs_orientation_error_rad": float(row.get("hold_phase_max_abs_orientation_error_rad", 0.0)),
        "max_abs_qd_radps": float(row.get("max_abs_qd_radps", 0.0)),
        "torque_saturation_percentage": float(row.get("torque_saturation_percentage", 0.0)),
        "termination_reason": str(row.get("termination_reason", "")),
        "failure_reason": str(row.get("failure_reason", "")),
        "move_failure_reason": str(row.get("move_failure_reason", "")),
        "hold_failure_reason": str(row.get("hold_failure_reason", "")),
        "controller_gains": gains,
        **gains,
    }


def _load_base_cfg(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, float]]:
    cfg = _load_yaml(args.config)
    base_controller_cfg = cfg["controller"]
    base_gains = controller_gain_summary(base_controller_cfg)["controller_gains"]
    return cfg, base_gains


def _resolve_candidate_gains(
    base_gains: dict[str, float],
    *,
    use_legacy_baseline_gains: bool,
    gain_overrides_json: str | None,
) -> dict[str, float]:
    """Default: the named --config's own gains, untouched. --use-legacy-baseline-gains
    substitutes the hardcoded BASELINE_GAINS constant first (pre-2026-07-30 behavior,
    kept only for reproducing old sweeps). --gain-overrides-json always applies last,
    on top of whichever base was selected."""
    candidate_gains = dict(base_gains)
    if use_legacy_baseline_gains:
        candidate_gains.update(BASELINE_GAINS)
    if gain_overrides_json is not None:
        candidate_gains.update(_normalize_gain_overrides(gain_overrides_json))
    return candidate_gains


def _candidate_payload(base_cfg: dict[str, Any], gains: dict[str, float], *, gravity_mode: str) -> dict[str, Any]:
    payload = _candidate_config_payload(base_cfg, gains)
    payload.setdefault("mujoco", {})
    payload["mujoco"]["default_controller"] = "impedance"
    payload["mujoco"]["gravity_mode"] = str(gravity_mode)
    return payload


def _run_child_experiment(
    *,
    output_root: Path,
    candidate_label: str,
    candidate_config_path: Path,
    run_label: str,
    args: argparse.Namespace,
    target_x_delta: float,
    move_duration: float,
    hold_duration: float,
    torque_limit_scale: float,
) -> dict[str, Any]:
    candidate_run_root = output_root / "_runs" / candidate_label
    candidate_run_root.mkdir(parents=True, exist_ok=True)
    total_duration = float(move_duration + hold_duration)
    cmd = [
        sys.executable,
        str(EXPERIMENT_SCRIPT),
        "--mode",
        "controller-rollout",
        "--controller-kind",
        "impedance",
        "--gravity-mode",
        "gravity_comp" if args.gravity_mode == "gravity_comp" else "raw",
        "--trajectory-profile",
        "min_jerk_move_hold",
        "--target-x-delta",
        str(float(target_x_delta)),
        "--move-duration",
        str(float(move_duration)),
        "--duration",
        str(float(total_duration)),
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
    if args.start_q_rad is not None:
        cmd.extend(["--start-q-rad", *[str(float(v)) for v in args.start_q_rad]])
    if getattr(args, "gravity_source", None):
        cmd.extend(["--gravity-source", str(args.gravity_source)])
    if getattr(args, "coriolis_feedforward", False):
        cmd.append("--coriolis-feedforward")
    if getattr(args, "asymmetric_coulomb_friction", False):
        cmd.append("--asymmetric-coulomb-friction")
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
    final_dir = output_root / "per_run_traces" / candidate_label / run_label
    if final_dir.exists():
        shutil.rmtree(final_dir)
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(child_dir), str(final_dir))
    if candidate_run_root.exists() and not any(candidate_run_root.iterdir()):
        candidate_run_root.rmdir()

    summary_path = final_dir / "summary.json"
    if not summary_path.exists():
        raise RuntimeError(
            f"Child experiment did not write summary.json for {run_label!r}. "
            f"Return code={completed.returncode}. stdout={completed.stdout!r}. stderr={completed.stderr!r}"
        )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(compute_valid_move_hold_metrics(summary))
    summary["subprocess_returncode"] = int(completed.returncode)
    summary["candidate_label"] = candidate_label
    summary["candidate_config_path"] = str(candidate_config_path)
    summary["run_label"] = run_label
    summary["stage"] = "baseline"
    summary["controller_kind"] = "impedance"
    summary["gravity_mode"] = "gravity_comp" if args.gravity_mode == "gravity_comp" else "raw"
    summary["trajectory_profile"] = "min_jerk_move_hold"
    summary["target_x_delta"] = float(target_x_delta)
    summary["move_duration_s"] = float(move_duration)
    summary["hold_duration_s"] = float(hold_duration)
    summary["total_duration_s"] = float(total_duration)
    summary["torque_limit_scale"] = float(torque_limit_scale)
    summary["summary_path"] = str(summary_path)
    for key in ("trace_path", "plot_path", "diagnostics_plot_path"):
        value = summary.get(key)
        if value:
            summary[key] = str(final_dir / Path(value).name)
    (final_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _plot_valid_heatmap(rows: list[dict[str, Any]], output_path: Path, *, title: str) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    target_vals = sorted({float(r["target_x_delta"]) for r in rows})
    hold_vals = sorted({float(r["hold_duration_s"]) for r in rows})
    matrix = np.full((len(hold_vals), len(target_vals)), np.nan, dtype=np.float64)
    for row in rows:
        xi = target_vals.index(float(row["target_x_delta"]))
        yi = hold_vals.index(float(row["hold_duration_s"]))
        value = 1.0 if bool(row.get("valid_move_and_hold", False)) else 0.0
        matrix[yi, xi] = value if np.isnan(matrix[yi, xi]) else max(matrix[yi, xi], value)

    fig, ax = plt.subplots(figsize=(max(6.5, 0.9 * len(target_vals)), max(4.0, 0.8 * len(hold_vals))))
    cmap = plt.get_cmap("RdYlGn")
    im = ax.imshow(matrix, origin="lower", aspect="auto", vmin=0.0, vmax=1.0, cmap=cmap)
    ax.set_xticks(range(len(target_vals)), [f"{x:g}" for x in target_vals])
    ax.set_yticks(range(len(hold_vals)), [f"{y:g}" for y in hold_vals])
    ax.set_xlabel("target_x_delta [m]")
    ax.set_ylabel("hold_duration [s]")
    ax.set_title(title)
    for yi, hold_duration in enumerate(hold_vals):
        for xi, target_x_delta in enumerate(target_vals):
            value = matrix[yi, xi]
            if np.isnan(value):
                continue
            ax.text(xi, yi, "PASS" if value >= 0.5 else "FAIL", ha="center", va="center", fontsize=8, fontweight="bold")
    fig.colorbar(im, ax=ax, label="valid move-and-hold")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _plot_metrics(rows: list[dict[str, Any]], output_path: Path, *, title: str) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    hold_vals = sorted({float(r["hold_duration_s"]) for r in rows})
    metrics = [
        ("final_x_error_m", "Final X error [m]"),
        ("hold_phase_x_drift_from_hold_start_m", "Hold X drift [m]"),
        ("max_abs_y_drift_m", "Max |Y drift| [m]"),
        ("max_abs_z_drift_m", "Max |Z drift| [m]"),
        ("max_abs_orientation_error_rad", "Max orientation error [rad]"),
        ("max_abs_qd_radps", "Max |qd| [rad/s]"),
        ("max_abs_tau_controller_nm", "Max |tau_controller| [Nm]"),
        ("max_abs_tau_applied_nm", "Max |tau_applied| [Nm]"),
    ]
    fig, axes = plt.subplots(4, 2, figsize=(16, 14), sharex=False)
    for ax, (metric_key, ylabel) in zip(axes.flat, metrics, strict=True):
        for hold_duration in hold_vals:
            subset = sorted(
                [r for r in rows if float(r["hold_duration_s"]) == float(hold_duration)],
                key=lambda r: float(r["target_x_delta"]),
            )
            if not subset:
                continue
            xs = [float(r["target_x_delta"]) for r in subset]
            ys = [float(r.get(metric_key, 0.0)) for r in subset]
            ax.plot(xs, ys, marker="o", label=f"{hold_duration:g}s hold")
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


def _plot_torque_metrics(rows: list[dict[str, Any]], output_path: Path, *, title: str) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    plots = [
        ("max_abs_tau_controller_nm", "Max |tau_controller| [Nm]"),
        ("max_abs_tau_gravity_nm", "Max |tau_gravity| [Nm]"),
        ("max_abs_tau_applied_nm", "Max |tau_applied| [Nm]"),
        ("controller_torque_fraction", "Controller / applied torque fraction"),
    ]
    for ax, (metric_key, ylabel) in zip(axes.flat, plots, strict=True):
        for hold_duration in sorted({float(r["hold_duration_s"]) for r in rows}):
            subset = sorted(
                [r for r in rows if float(r["hold_duration_s"]) == float(hold_duration)],
                key=lambda r: float(r["target_x_delta"]),
            )
            xs = [float(r["target_x_delta"]) for r in subset]
            ys = [float(r.get(metric_key, 0.0)) for r in subset]
            ax.plot(xs, ys, marker="o", label=f"{hold_duration:g}s hold")
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


def _plot_failure_counts(rows: list[dict[str, Any]], output_path: Path, *, title: str) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    move_counts = Counter(str(r.get("move_failure_reason", "none")) for r in rows if not bool(r.get("valid_move_phase", False)))
    hold_counts = Counter(str(r.get("hold_failure_reason", "none")) for r in rows if not bool(r.get("valid_hold_phase", False)))
    move_counts.pop("none", None)
    hold_counts.pop("none", None)

    move_top = move_counts.most_common(6)
    hold_top = hold_counts.most_common(6)
    move_labels = [label for label, _ in move_top] or ["none"]
    move_values = [count for _, count in move_top] or [0]
    hold_labels = [label for label, _ in hold_top] or ["none"]
    hold_values = [count for _, count in hold_top] or [0]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].bar(move_labels, move_values, color="tab:orange")
    axes[0].set_title("Move-phase failures")
    axes[0].tick_params(axis="x", labelrotation=30)
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[1].bar(hold_labels, hold_values, color="tab:blue")
    axes[1].set_title("Hold-phase failures")
    axes[1].tick_params(axis="x", labelrotation=30)
    axes[1].grid(True, axis="y", alpha=0.25)
    for ax in axes:
        ax.set_ylabel("count")
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _write_readme(output_root: Path, *, rows: list[dict[str, Any]], best_settings: dict[str, Any]) -> None:
    lines = [
        "# UR5e MuJoCo Residual Impedance Move-and-Hold Tuning",
        "",
        f"- output_root: {output_root}",
        f"- total_rows: {len(rows)}",
        f"- valid_move_and_hold: {sum(1 for r in rows if bool(r.get('valid_move_and_hold', False)))}",
        f"- dominant_failure_phase: {best_settings.get('dominant_failure_phase', 'none')}",
        "",
        "See `summary.csv`, `summary.json`, `best_settings.json`, `per_run_traces/`, and `plots/`.",
    ]
    (output_root / "README.md").write_text("\n".join(lines), encoding="utf-8")


def run() -> int:
    args = parse_args()
    output_root = args.output_root if args.output_root is not None else _default_output_root()
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "candidate_configs").mkdir(parents=True, exist_ok=True)
    (output_root / "per_run_traces").mkdir(parents=True, exist_ok=True)
    (output_root / "plots").mkdir(parents=True, exist_ok=True)

    base_cfg, base_gains = _load_base_cfg(args)
    # Real safety_cfg for this sweep's own config, not RunLogger's default --
    # see observability/run_logger.py's RunLogger docstring / 2026-07-30 fix
    # for why the default silently produced wrong time-to-limit values.
    run_logger = RunLogger(
        output_root=output_root, source_script=Path(__file__).name,
        safety_cfg=build_safety_config(base_cfg["controller"]),
    )
    candidate_gains = _resolve_candidate_gains(
        base_gains,
        use_legacy_baseline_gains=args.use_legacy_baseline_gains,
        gain_overrides_json=args.gain_overrides_json,
    )

    candidate_label = _candidate_label(1)
    candidate_config_path = _candidate_config_path(output_root, "baseline", candidate_label)
    _write_yaml(candidate_config_path, _candidate_payload(base_cfg, candidate_gains, gravity_mode=args.gravity_mode))

    target_x_deltas = _parse_float_list(args.target_x_deltas, [0.01, 0.02, 0.03, 0.04])
    move_durations = _parse_float_list(args.move_durations, [0.5, 1.0, 1.5, 2.0])
    hold_durations = _parse_float_list(args.hold_durations, [1.0, 2.0, 4.0])
    torque_limit_scales = _parse_float_list(args.torque_limit_scales, [0.5, 0.75, 1.0])

    rows: list[dict[str, Any]] = []
    for target_x_delta in target_x_deltas:
        for move_duration in move_durations:
            for hold_duration in hold_durations:
                for torque_limit_scale in torque_limit_scales:
                    run_label = _run_label(target_x_delta, move_duration, hold_duration, torque_limit_scale)
                    summary = _run_child_experiment(
                        output_root=output_root,
                        candidate_label=candidate_label,
                        candidate_config_path=candidate_config_path,
                        run_label=run_label,
                        args=args,
                        target_x_delta=float(target_x_delta),
                        move_duration=float(move_duration),
                        hold_duration=float(hold_duration),
                        torque_limit_scale=float(torque_limit_scale),
                    )
                    rows.append(summary)
                    run_logger.log_run(
                        summary,
                        run_dir=Path(summary["summary_path"]).parent,
                        seed=int(args.seed),
                        config_path=candidate_config_path,
                        run_label=run_label,
                    )

    csv_path = output_root / "summary.csv"
    summary_path = output_root / "summary.json"
    best_path = output_root / "best_settings.json"
    _write_csv(rows, csv_path)
    run_logger.write_sweep_csv_snapshot()

    valid_rows = [row for row in rows if bool(row.get("valid_move_and_hold", False))]
    strict_valid_rows = [row for row in rows if bool(row.get("valid_move_and_hold", False)) and bool(compute_valid_move_hold_metrics(row, strict=True)["valid_move_and_hold"])]
    best_valid_overall = max(valid_rows, key=move_hold_ranking_key) if valid_rows else None

    best_by_hold_duration: dict[str, dict[str, Any] | None] = {}
    best_by_total_duration: dict[str, dict[str, Any] | None] = {}
    largest_valid_target_x_delta_by_hold_duration: dict[str, float | None] = {}
    largest_valid_target_x_delta_by_total_duration: dict[str, float | None] = {}
    for hold_duration in sorted({float(r["hold_duration_s"]) for r in rows}):
        hold_rows = [row for row in valid_rows if float(row["hold_duration_s"]) == float(hold_duration)]
        best_by_hold_duration[f"{hold_duration:g}"] = max(hold_rows, key=move_hold_ranking_key) if hold_rows else None
        largest_valid_target_x_delta_by_hold_duration[f"{hold_duration:g}"] = (
            max((float(row["target_x_delta"]) for row in hold_rows), default=None)
        )
    for total_duration in sorted({float(r["total_duration_s"]) for r in rows}):
        total_rows = [row for row in valid_rows if float(row["total_duration_s"]) == float(total_duration)]
        best_by_total_duration[f"{total_duration:g}"] = max(total_rows, key=move_hold_ranking_key) if total_rows else None
        largest_valid_target_x_delta_by_total_duration[f"{total_duration:g}"] = (
            max((float(row["target_x_delta"]) for row in total_rows), default=None)
        )

    move_failure_counts = Counter(str(row.get("move_failure_reason", "")) for row in rows if not bool(row.get("valid_move_phase", False)))
    hold_failure_counts = Counter(str(row.get("hold_failure_reason", "")) for row in rows if not bool(row.get("valid_hold_phase", False)))
    move_failure_counts.pop("none", None)
    hold_failure_counts.pop("none", None)

    move_failure_total = sum(move_failure_counts.values())
    hold_failure_total = sum(hold_failure_counts.values())
    dominant_failure_phase = None
    if move_failure_total > hold_failure_total:
        dominant_failure_phase = "move"
    elif hold_failure_total > move_failure_total:
        dominant_failure_phase = "hold"

    target_tracking_related = sum(
        count
        for reason, count in move_failure_counts.items()
        if "target_tracking" in reason
    ) + sum(count for reason, count in hold_failure_counts.items() if "target_tracking" in reason)
    drift_related = sum(
        count
        for reason, count in move_failure_counts.items()
        if "drift" in reason or "orientation" in reason
    ) + sum(count for reason, count in hold_failure_counts.items() if "drift" in reason or "orientation" in reason)
    torque_saturation_appears = bool(any(float(row.get("torque_saturation_percentage", 0.0)) > 0.0 for row in rows))

    valid_controller_ratio = float(
        np.mean(
            [
                float(row.get("controller_torque_fraction", 0.0)) / max(float(row.get("gravity_torque_fraction", 0.0)), 1.0e-12)
                for row in valid_rows
            ]
        )
        if valid_rows
        else 0.0
    )
    valid_gravity_fraction = float(np.mean([float(row.get("gravity_torque_fraction", 0.0)) for row in valid_rows])) if valid_rows else 0.0
    valid_controller_fraction = float(np.mean([float(row.get("controller_torque_fraction", 0.0)) for row in valid_rows])) if valid_rows else 0.0

    top_failure_counts = {
        "move": move_failure_counts.most_common(),
        "hold": hold_failure_counts.most_common(),
    }
    top_failure_counts = {
        phase: [{"reason": reason, "count": int(count)} for reason, count in counts]
        for phase, counts in top_failure_counts.items()
    }

    summary = {
        "output_root": str(output_root),
        "config_path": str(args.config),
        "scene_xml": str(args.scene) if args.scene is not None else None,
        "gravity_mode": args.gravity_mode,
        "trajectory_profile": "min_jerk_move_hold",
        "target_x_deltas": target_x_deltas,
        "move_durations": move_durations,
        "hold_durations": hold_durations,
        "torque_limit_scales": torque_limit_scales,
        "baseline_gains": candidate_gains,
        "num_runs": len(rows),
        "num_valid_move_and_hold": len(valid_rows),
        "num_valid_move_and_hold_strict": len(strict_valid_rows),
        "rows": rows,
        "candidate_configs_dir": str(output_root / "candidate_configs"),
        "per_run_traces_dir": str(output_root / "per_run_traces"),
        "plots_dir": str(output_root / "plots"),
        "summary_path": str(summary_path),
        "best_settings_path": str(best_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    best_settings = {
        "output_root": str(output_root),
        "analysis_rows": len(rows),
        "total_rows": len(rows),
        "num_valid_move_and_hold": len(valid_rows),
        "num_valid_move_and_hold_strict": len(strict_valid_rows),
        "best_valid_move_and_hold_overall": _row_snapshot(best_valid_overall) or {},
        "best_valid_move_and_hold_by_hold_duration": {
            duration: (_row_snapshot(row) if row is not None else None) for duration, row in best_by_hold_duration.items()
        },
        "best_valid_move_and_hold_by_total_duration": {
            duration: (_row_snapshot(row) if row is not None else None) for duration, row in best_by_total_duration.items()
        },
        "largest_valid_target_x_delta_by_hold_duration": largest_valid_target_x_delta_by_hold_duration,
        "largest_valid_target_x_delta_by_total_duration": largest_valid_target_x_delta_by_total_duration,
        "largest_valid_target_x_delta_overall": max((float(row["target_x_delta"]) for row in valid_rows), default=None),
        "largest_valid_hold_duration_with_valid_transport": max((float(row["hold_duration_s"]) for row in valid_rows), default=None),
        "dominant_failure_phase": dominant_failure_phase,
        "common_move_failure_reasons": top_failure_counts["move"],
        "common_hold_failure_reasons": top_failure_counts["hold"],
        "move_failure_total": int(move_failure_total),
        "hold_failure_total": int(hold_failure_total),
        "target_tracking_remains_dominant": bool(target_tracking_related >= drift_related),
        "drift_returns_as_bottleneck": bool(drift_related > 0 and drift_related >= target_tracking_related),
        "torque_saturation_appears": torque_saturation_appears,
        "residual_torque_ratio_summary": {
            "valid_controller_to_gravity_ratio_mean": valid_controller_ratio,
            "valid_gravity_fraction_mean": valid_gravity_fraction,
            "valid_controller_fraction_mean": valid_controller_fraction,
        },
        "best_gain_set_overall": _row_snapshot(best_valid_overall) or {},
        "best_overall_valid_move_and_hold": _row_snapshot(best_valid_overall) or {},
        "best_by_hold_duration": {
            duration: (_row_snapshot(row) if row is not None else None) for duration, row in best_by_hold_duration.items()
        },
        "best_by_total_duration": {
            duration: (_row_snapshot(row) if row is not None else None) for duration, row in best_by_total_duration.items()
        },
        "candidate_config_path": str(candidate_config_path),
        "baseline_gain_set": candidate_gains,
    }
    best_path.write_text(json.dumps(best_settings, indent=2), encoding="utf-8")

    _write_readme(output_root, rows=rows, best_settings=best_settings)

    if not args.no_plot and rows:
        _plot_valid_heatmap(rows, output_root / "plots" / "valid_move_hold_heatmap.png", title="Valid move-and-hold by target and hold duration")
        _plot_metrics(rows, output_root / "plots" / "move_hold_metrics.png", title="Move/hold transport metrics by target delta")
        _plot_torque_metrics(rows, output_root / "plots" / "move_hold_torque_metrics.png", title="Residual-torque move/hold metrics")
        _plot_failure_counts(rows, output_root / "plots" / "move_hold_failure_counts.png", title="Move vs hold failure counts")

    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "summary_path": str(summary_path),
                "best_settings_path": str(best_path),
                "num_runs": len(rows),
                "num_valid_move_and_hold": len(valid_rows),
                "dominant_failure_phase": dominant_failure_phase,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
