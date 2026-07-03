"""Shared helpers for the UR5e impedance-transport tuning drivers.

Extracted verbatim from tools/tune_ur5e_impedance_transport.py (now in
archive/superseded/) so the active residual tuner has no dependency on the
superseded driver.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from transport_metrics import GAIN_FIELDS  # noqa: E402


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


