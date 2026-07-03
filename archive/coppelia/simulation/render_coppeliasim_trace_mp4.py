#!/usr/bin/env python3
"""Render an MP4 overview of a CoppeliaSim controller JSONL trace."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np


def load_trace(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"empty trace: {path}")
    return rows


def pick_tau(row: dict) -> np.ndarray:
    for key in ("tau_cmd", "tau_final", "tau_final_sent"):
        if key in row and row[key] is not None:
            return np.asarray(row[key], dtype=np.float64).reshape(6)
    return np.zeros(6, dtype=np.float64)


def render_mp4(rows: list[dict], out_path: Path, *, fps: int = 25) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import animation

    t = np.array([float(r["time"]) for r in rows], dtype=np.float64)
    ee = np.array([r["ee_pos"] for r in rows], dtype=np.float64)
    target_axis = np.array(
        [float(r.get("target_axis", r.get("target_x", ee[i, 0]))) for i, r in enumerate(rows)],
        dtype=np.float64,
    )
    y_err = ee[:, 1] - ee[0, 1]
    z_err = ee[:, 2] - ee[0, 2]
    tau = np.vstack([pick_tau(r) for r in rows])
    vx = np.array([float(r.get("ee_lin_vel", [0, 0, 0])[0]) for r in rows], dtype=np.float64)

    fig = plt.figure(figsize=(11, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0], width_ratios=[1.2, 1.0])
    ax_path = fig.add_subplot(gs[:, 0])
    ax_x = fig.add_subplot(gs[0, 1])
    ax_yz = fig.add_subplot(gs[1, 1])

    ax_path.plot(ee[:, 0], ee[:, 2], color="#9aa0a6", linewidth=1.0, alpha=0.6, label="path")
    (ee_dot,) = ax_path.plot([ee[0, 0]], [ee[0, 2]], "o", color="#1a73e8", markersize=8, label="EE")
    (target_line,) = ax_path.plot([], [], "--", color="#ea4335", linewidth=1.5, label="target X")
    ax_path.set_xlabel("world X (m)")
    ax_path.set_ylabel("world Z (m)")
    ax_path.set_title("End-effector transport (X–Z)")
    ax_path.grid(True, alpha=0.3)
    ax_path.legend(loc="upper right")
    pad = 0.05
    ax_path.set_xlim(float(np.min(ee[:, 0]) - pad), float(np.max(ee[:, 0]) + pad))
    ax_path.set_ylim(float(np.min(ee[:, 2]) - pad), float(np.max(ee[:, 2]) + pad))

    (x_now_line,) = ax_x.plot([t[0]], [ee[0, 0]], "o", color="#1a73e8")
    ax_x.plot(t, ee[:, 0], color="#9aa0a6", linewidth=1.0)
    ax_x.plot(t, target_axis, "--", color="#ea4335", linewidth=1.0)
    ax_x.set_ylabel("X (m)")
    ax_x.set_title("X tracking")
    ax_x.grid(True, alpha=0.3)

    (yz_now_line,) = ax_yz.plot([t[0]], [y_err[0] * 1000.0], "o", color="#1a73e8")
    ax_yz.plot(t, y_err * 1000.0, label="Y drift (mm)")
    ax_yz.plot(t, z_err * 1000.0, label="Z drift (mm)")
    ax_yz.set_xlabel("time (s)")
    ax_yz.set_ylabel("drift (mm)")
    ax_yz.set_title("Orthogonal drift")
    ax_yz.grid(True, alpha=0.3)
    ax_yz.legend(loc="upper right")

    time_text = fig.text(0.02, 0.97, "", fontsize=11, family="monospace")
    fig.suptitle("CoppeliaSim fast_x torque transport (trace replay)", fontsize=13)

    def update(frame_idx: int):
        ee_dot.set_data([ee[frame_idx, 0]], [ee[frame_idx, 2]])
        target_line.set_data(
            [target_axis[frame_idx], target_axis[frame_idx]],
            ax_path.get_ylim(),
        )
        x_now_line.set_data([t[frame_idx]], [ee[frame_idx, 0]])
        yz_now_line.set_data([t[frame_idx]], [y_err[frame_idx] * 1000.0])
        time_text.set_text(
            f"t={t[frame_idx]:.2f}s  vx={vx[frame_idx]:+.3f} m/s  "
            f"tau_max={np.max(np.abs(tau[frame_idx])):.1f} Nm"
        )
        return ee_dot, target_line, x_now_line, yz_now_line, time_text

    interval_ms = int(round(1000.0 / max(int(fps), 1)))
    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(rows),
        interval=interval_ms,
        blit=False,
        repeat=False,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_bin = os.environ.get("FFMPEG_BIN", "ffmpeg")
    writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
    if shutil.which(ffmpeg_bin) is None and ffmpeg_bin == "ffmpeg":
        raise RuntimeError("ffmpeg not found; set FFMPEG_BIN")
    try:
        anim.save(str(out_path), writer=writer)
    except Exception:
        # Matplotlib may call bare 'ffmpeg' even when FFMPEG_BIN is set.
        if ffmpeg_bin != "ffmpeg":
            old_path = os.environ.get("PATH", "")
            os.environ["PATH"] = f"{Path(ffmpeg_bin).parent}:{old_path}"
        anim.save(str(out_path), writer=writer)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("demonstration_videos/ur5e_coppeliasim/coppelia_fast_x_transport.mp4"),
    )
    parser.add_argument("--fps", type=int, default=25)
    args = parser.parse_args()
    rows = load_trace(args.jsonl)
    render_mp4(rows, args.out, fps=args.fps)
    print(f"Wrote trace replay video: {args.out}")


if __name__ == "__main__":
    main()
