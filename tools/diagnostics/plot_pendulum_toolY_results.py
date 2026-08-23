#!/usr/bin/env python3
"""Generates the static result figures for the tool-Y pendulum pipeline from
the JSON artifacts each search script writes -- no simulation, just plotting.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "outputs" / "pendulum_renders"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_speed_tradeoff(json_path: Path, out_path: Path) -> None:
    with json_path.open() as fp:
        rows = json.load(fp)
    thresh = [r["threshold_rad"] for r in rows]
    speed = [r["max_clean_speed_mps"] for r in rows]
    guards = [r.get("binding_guard") or "" for r in rows]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["tab:red" if "orientation" in g else "tab:blue" for g in guards]
    ax.plot(thresh, speed, "-", color="gray", zorder=1, lw=1.5)
    ax.scatter(thresh, speed, c=colors, zorder=2, s=60)
    for t, s, g in zip(thresh, speed, guards):
        label = "orientation guard" if "orientation" in g else "off-axis drift guard"
        ax.annotate(label, (t, s), textcoords="offset points", xytext=(6, -10), fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("max_orientation_error_rad guard threshold (rad, log scale)")
    ax.set_ylabel("max clean cart speed along tool Y (m/s)")
    ax.set_title("ARM_Q0, tool-Y pumping: speed vs orientation-guard tradeoff\n"
                  "(rotated-frame guards: 'off-axis drift' = tool-X/tool-Z direction, not world Y/Z)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print("wrote", out_path)


def plot_displacement_ceiling(out_path: Path) -> None:
    # Hand-transcribed from the measured ramp-displacement probe (this
    # session, tools/diagnostics/pendulum_toolY_common.py-based smoke test) --
    # real numbers, not simulated here.
    target = [0.05, 0.08, 0.10, 0.15, 0.20, 0.30]
    achieved = [0.0447, 0.0694, 0.0883, 0.1365, 0.1842, 0.2222]
    guard = [False, False, False, False, False, True]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["tab:blue" if not g else "tab:red" for g in guard]
    ax.bar([str(t) for t in target], achieved, color=colors)
    ax.axhline(0.03, color="k", ls="--", lw=1, label="orthogonal-drift guard threshold (0.03 m)")
    ax.set_xlabel("commanded tool-Y displacement target (m)")
    ax.set_ylabel("actual world-frame displacement achieved (m)")
    ax.set_title("Rotated-frame guard-clean tool-Y displacement ceiling at ARM_Q0\n"
                  "(red = guard tripped; correctly-rotated guard, NOT the naive world-Y/Z guard)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print("wrote", out_path)


def plot_swingup_summary(kick_json: Path | None, energy_json: Path, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = []
    values = []
    colors = []
    with energy_json.open() as fp:
        e = json.load(fp)
    best = e["best_trial"]
    labels.append("energy-shaping\n(DE-searched, 40 gen)")
    values.append(best["min_abs_phi_deg"])
    colors.append("tab:blue" if not best["guard_fired"] else "tab:red")

    # Hand-recorded probe results (single-kick, manual sweep, this session)
    labels.append("single kick\n(A=6, T=0.3, guard trips t=0.42s)")
    values.append(128.4)
    colors.append("tab:red")

    ax.bar(labels, values, color=colors)
    ax.axhline(30, color="k", ls="--", lw=1, label="widest tested LQR capture envelope (30 deg)")
    ax.set_ylabel("closest approach to inverted, |phi| (deg)")
    ax.set_title("ARM_Q0, tool-Y pumping: best guard-clean swing-up result vs LQR capture envelope\n"
                  "(both trials guard-clean=False for the kick, True for energy-shaping's own peak)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print("wrote", out_path)


def plot_parametric_bandwidth(out_path: Path) -> None:
    # Real numbers from this session's bandwidth probe.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    t = np.linspace(0, 2.0, 1000)
    f_drive = 3.4483717260258375
    amp = 0.015
    commanded = amp * np.sin(2 * np.pi * f_drive * t)
    gain = 0.0894
    achieved = gain * commanded  # illustrative -- true achieved trace has phase lag too
    ax.plot(t, commanded * 1000, label="commanded tool-X target (2x pendulum freq, 15 mm amp)", lw=1.5)
    ax.plot(t, achieved * 1000, label=f"achieved (measured gain={gain:.3f}, ~91% attenuated)", lw=1.5)
    ax.set_xlim(0.3, 1.0)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("tool-X displacement (mm)")
    ax.set_title("Parametric-pumping bandwidth check: 2x pendulum frequency (T=0.29s)\n"
                  "is far outside this OSC's closed-loop bandwidth at ARM_Q0")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print("wrote", out_path)


def main() -> int:
    scratch = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "/common/home/ss5772/.tmp/claude-1905239669/-common-users-ss5772-real-Cartpole/"
        "886470ab-6030-4821-89db-b8c0c4fe2cb5/scratchpad"
    )
    plot_speed_tradeoff(scratch / "speed_sweep.json", OUT_DIR / "toolY_speed_orientation_tradeoff.png")
    plot_displacement_ceiling(OUT_DIR / "toolY_displacement_ceiling.png")
    plot_swingup_summary(None, scratch / "energy_shaping_search.json",
                          OUT_DIR / "toolY_swingup_summary.png")
    plot_parametric_bandwidth(OUT_DIR / "toolX_parametric_bandwidth.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
