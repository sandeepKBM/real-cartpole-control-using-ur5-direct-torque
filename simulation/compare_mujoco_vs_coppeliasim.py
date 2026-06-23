#!/usr/bin/env python3
"""
Compare torque-mode UR5 X-axis runs between MuJoCo and CoppeliaSim.

Each side produces a JSON trace with a common schema (see
``run_x_torque_transport_mujoco.py`` for the MuJoCo producer and the
``coppeliasim_bridge_node`` README for how to record the CoppeliaSim side).
This script aligns the two traces on time, prints a tabular summary of key
metrics, and optionally writes a CSV with the combined time series so the user
can plot offline.

Trace JSON schema (compatible with the MuJoCo producer):

    {
      "simulator": "mujoco" | "coppeliasim",
      "dt": 0.002,
      "duration": 4.0,
      "target_x": 0.05,
      "initial_ee_pos": [x0, y0, z0],
      "controller": {...},
      "trace": [
        {
          "time": t,
          "q": [6],
          "qd": [6],
          "ee_pos": [3],
          "ee_lin_vel": [3],
          "target_x": float,
          "Fx": float,
          "x_error": float,
          "tau": [6],
          "safety_ok": bool,
          ...
        },
        ...
      ]
    }

Run:

    python simulation/compare_mujoco_vs_coppeliasim.py \\
        --mujoco outputs/control_runs/x_torque_mujoco_gravcomp.json \\
        --coppelia outputs/control_runs/x_torque_coppelia.json \\
        --csv outputs/control_runs/compare.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class TraceSummary:
    name: str
    source_path: Path
    t: np.ndarray
    ee_pos: np.ndarray  # (N, 3)
    ee_vel: np.ndarray  # (N, 3), possibly zero-filled if missing
    q: np.ndarray  # (N, nj)
    qd: np.ndarray  # (N, nj)
    tau: np.ndarray  # (N, nj)
    x_error: np.ndarray  # (N,)
    target_x: float
    initial_ee_pos: np.ndarray
    safety_trips: int
    final_x_error: float
    yz_drift: np.ndarray  # (N,) distance of ee from initial ee in YZ plane
    peak_tau: np.ndarray  # (nj,)

    @staticmethod
    def _column(trace: list[dict[str, Any]], key: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
        vals = [row.get(key, None) for row in trace]
        arr = np.array([v if v is not None else np.zeros(shape or ()) for v in vals], dtype=np.float64)
        return arr

    @classmethod
    def from_file(cls, path: Path) -> "TraceSummary":
        data = json.loads(Path(path).read_text())
        trace = data["trace"]
        if not trace:
            raise ValueError(f"{path}: empty trace")
        t = np.array([row["time"] for row in trace], dtype=np.float64)
        ee_pos = np.array([row["ee_pos"] for row in trace], dtype=np.float64)
        # Some traces may not carry ee_lin_vel; default to zeros.
        if "ee_lin_vel" in trace[0]:
            ee_vel = np.array([row["ee_lin_vel"] for row in trace], dtype=np.float64)
        else:
            ee_vel = np.zeros_like(ee_pos)
        q = np.array([row["q"] for row in trace], dtype=np.float64)
        qd = np.array([row["qd"] for row in trace], dtype=np.float64)
        tau = np.array([row["tau"] for row in trace], dtype=np.float64)
        x_err = np.array([row.get("x_error", 0.0) for row in trace], dtype=np.float64)
        target_x = float(data.get("target_x", trace[-1].get("target_x", 0.0)))
        initial = np.array(data.get("initial_ee_pos", ee_pos[0]), dtype=np.float64)

        safety_trips = int(sum(1 for row in trace if not row.get("safety_ok", True)))

        yz = ee_pos[:, 1:3] - initial[None, 1:3]
        yz_drift = np.linalg.norm(yz, axis=1)

        return cls(
            name=str(data.get("simulator", Path(path).stem)),
            source_path=Path(path),
            t=t,
            ee_pos=ee_pos,
            ee_vel=ee_vel,
            q=q,
            qd=qd,
            tau=tau,
            x_error=x_err,
            target_x=target_x,
            initial_ee_pos=initial,
            safety_trips=safety_trips,
            final_x_error=float(x_err[-1]),
            yz_drift=yz_drift,
            peak_tau=np.max(np.abs(tau), axis=0),
        )


def _fmt(x: float, unit: str = "") -> str:
    if unit == "mm":
        return f"{x*1000:+.2f} mm"
    if unit == "nm":
        return f"{x:+.3f} Nm"
    return f"{x:+.6f}"


def print_summary(a: TraceSummary, b: TraceSummary) -> None:
    width = 30
    print(f"{'metric':{width}}  {a.name:>18}  {b.name:>18}")
    print("-" * (width + 2 + 20 + 2 + 20))
    rows: list[tuple[str, float, float, str]] = [
        ("initial EE x (m)", a.initial_ee_pos[0], b.initial_ee_pos[0], ""),
        ("initial EE y (m)", a.initial_ee_pos[1], b.initial_ee_pos[1], ""),
        ("initial EE z (m)", a.initial_ee_pos[2], b.initial_ee_pos[2], ""),
        ("target_x (m)", a.target_x, b.target_x, ""),
        ("final EE x (m)", a.ee_pos[-1, 0], b.ee_pos[-1, 0], ""),
        ("final x_error", a.final_x_error, b.final_x_error, ""),
        ("max |x_error|", float(np.max(np.abs(a.x_error))), float(np.max(np.abs(b.x_error))), ""),
        ("peak YZ drift (mm)", float(np.max(a.yz_drift)), float(np.max(b.yz_drift)), "mm"),
        ("final YZ drift (mm)", float(a.yz_drift[-1]), float(b.yz_drift[-1]), "mm"),
        ("peak |qd| (rad/s)", float(np.max(np.abs(a.qd))), float(np.max(np.abs(b.qd))), ""),
        ("safety trips", float(a.safety_trips), float(b.safety_trips), ""),
        ("duration (s)", float(a.t[-1]), float(b.t[-1]), ""),
    ]
    for name, av, bv, unit in rows:
        print(f"{name:{width}}  {_fmt(av, unit):>18}  {_fmt(bv, unit):>18}")
    print()
    print(f"{'peak |tau| per joint (Nm)':{width}}")
    for i in range(a.peak_tau.shape[0]):
        print(
            f"  joint[{i}]                   "
            f"  {a.peak_tau[i]:>15.3f}    {b.peak_tau[i]:>15.3f}"
        )

    # Qualitative verdict per Stage 7 acceptance criteria.
    verdict = []
    if np.sign(a.ee_pos[-1, 0] - a.initial_ee_pos[0]) == np.sign(b.ee_pos[-1, 0] - b.initial_ee_pos[0]) != 0:
        verdict.append("X direction matches")
    if np.max(np.abs(b.x_error)) <= np.max(np.abs(a.x_error)) * 2.0:
        verdict.append("X error bounded vs MuJoCo reference")
    if np.max(b.yz_drift) < 0.05:
        verdict.append("YZ drift < 50 mm")
    if b.safety_trips == 0:
        verdict.append("no safety trips in CoppeliaSim run")
    print("\nQualitative verdict:")
    if not verdict:
        print("  (no acceptance criteria satisfied yet)")
    for v in verdict:
        print(f"  - {v}")


def write_csv(out: Path, a: TraceSummary, b: TraceSummary) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    # Use MuJoCo timebase; nearest-neighbor resample B onto A.
    idx = np.searchsorted(b.t, a.t)
    idx = np.clip(idx, 0, b.t.shape[0] - 1)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "t",
                f"{a.name}_ee_x",
                f"{a.name}_ee_y",
                f"{a.name}_ee_z",
                f"{a.name}_x_error",
                f"{a.name}_tau0",
                f"{b.name}_ee_x",
                f"{b.name}_ee_y",
                f"{b.name}_ee_z",
                f"{b.name}_x_error",
                f"{b.name}_tau0",
            ]
        )
        for i, t in enumerate(a.t):
            j = idx[i]
            w.writerow(
                [
                    f"{t:.6f}",
                    f"{a.ee_pos[i, 0]:.6f}",
                    f"{a.ee_pos[i, 1]:.6f}",
                    f"{a.ee_pos[i, 2]:.6f}",
                    f"{a.x_error[i]:.6f}",
                    f"{a.tau[i, 0]:.4f}",
                    f"{b.ee_pos[j, 0]:.6f}",
                    f"{b.ee_pos[j, 1]:.6f}",
                    f"{b.ee_pos[j, 2]:.6f}",
                    f"{b.x_error[j]:.6f}",
                    f"{b.tau[j, 0]:.4f}",
                ]
            )
    print(f"Wrote CSV: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mujoco", required=True, type=Path)
    parser.add_argument("--coppelia", required=True, type=Path)
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()

    a = TraceSummary.from_file(args.mujoco)
    b = TraceSummary.from_file(args.coppelia)

    if a.tau.shape[1] != b.tau.shape[1]:
        raise ValueError(
            f"Joint count mismatch: {a.tau.shape[1]} (mujoco) vs {b.tau.shape[1]} (coppelia)"
        )

    print_summary(a, b)
    if args.csv is not None:
        write_csv(args.csv, a, b)


if __name__ == "__main__":
    main()
