#!/usr/bin/env python3
"""
Plot a JSONL trace from the ROS controller (one JSON object per line).

Columns expected match ``controller_node`` trace rows (see README).

Usage::

    python simulation/plot_coppeliasim_trace.py outputs/traces/run.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--out", type=Path, default=None, help="PNG output prefix")
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("Install matplotlib: pip install matplotlib") from exc

    rows: list[dict] = []
    with args.jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise SystemExit("empty trace")

    t = np.array([r["time"] for r in rows], dtype=np.float64)
    ee = np.array([r["ee_pos"] for r in rows], dtype=np.float64)
    tx = np.array([r["target_x"] for r in rows], dtype=np.float64)
    x_err = np.array([r["x_error"] for r in rows], dtype=np.float64)
    y_err = np.array([r["y_error"] for r in rows], dtype=np.float64)
    z_err = np.array([r["z_error"] for r in rows], dtype=np.float64)
    ori = np.array([r["orientation_error_norm"] for r in rows], dtype=np.float64)
    q = np.array([r["q"] for r in rows], dtype=np.float64)
    tau = np.array([r["tau_final"] for r in rows], dtype=np.float64)
    sat = np.array([r["tau_saturated"] for r in rows], dtype=np.float64)

    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    axes[0].plot(t, ee[:, 0], label="ee_x")
    axes[0].plot(t, tx, "--", label="target_x")
    axes[0].set_ylabel("X (m)")
    axes[0].legend(loc="upper right")
    axes[0].grid(True)

    axes[1].plot(t, y_err * 1000, label="y_err (mm)")
    axes[1].plot(t, z_err * 1000, label="z_err (mm)")
    axes[1].set_ylabel("YZ err (mm)")
    axes[1].legend(loc="upper right")
    axes[1].grid(True)

    axes[2].plot(t, ori, label="||ori err|| (rad)")
    axes[2].set_ylabel("orientation")
    axes[2].legend(loc="upper right")
    axes[2].grid(True)

    axes[3].plot(t, tau[:, 0], label="tau0")
    axes[3].plot(t, tau[:, 1], label="tau1")
    axes[3].plot(t, tau[:, 2], label="tau2")
    axes[3].set_ylabel("tau (Nm)")
    axes[3].set_xlabel("time (s)")
    axes[3].legend(loc="upper right")
    axes[3].grid(True)

    fig.tight_layout()
    if args.out:
        fig.savefig(str(args.out) + "_overview.png", dpi=150)
        print(f"Wrote {args.out}_overview.png")
    else:
        plt.show()


if __name__ == "__main__":
    main()
