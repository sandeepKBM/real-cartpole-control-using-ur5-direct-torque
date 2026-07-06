#!/usr/bin/env python3
"""Render an MP4 from a recorded UR5e MuJoCo torque-lane trace.jsonl.

Kinematic replay only (sets data.qpos from the recorded q and calls
mj_forward per frame) -- this does not step the simulator, so it is not a
second per-step control loop. tools/ur5e_mujoco_torque_experiments.py
remains the sole owner of the per-step rollout loop.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace", type=Path, required=True, help="Path to a trace.jsonl file.")
    p.add_argument("--scene", type=Path, default=REPO_ROOT / "assets/ur5e_torque/scene.xml")
    p.add_argument("--output", type=Path, required=True, help="Output .mp4 path.")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--azimuth", type=float, default=90.0, help="Camera azimuth (90 = side view, X motion reads left-right).")
    p.add_argument("--elevation", type=float, default=-10.0)
    p.add_argument("--distance", type=float, default=2.6, help="Camera distance from lookat, in meters.")
    p.add_argument(
        "--lookat",
        type=float,
        nargs=3,
        default=[0.1, -0.2, 0.7],
        metavar=("X", "Y", "Z"),
        help="Camera lookat point. Default is tuned for the active-origin transport pose "
        "([0, -pi/2, 0, -pi/2, 0, 0]) used throughout this repo's move-hold configs -- "
        "override for other start poses.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rows = [json.loads(line) for line in args.trace.open()]
    if not rows:
        raise RuntimeError(f"No rows in {args.trace}")

    model = mujoco.MjModel.from_xml_path(str(args.scene))
    # Grow the offscreen framebuffer if the requested resolution exceeds the
    # model's default (this is a runtime rendering setting, not a physics/
    # model change -- the on-disk scene.xml is left untouched).
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), int(args.width))
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), int(args.height))
    data = mujoco.MjData(model)

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.azimuth = float(args.azimuth)
    cam.elevation = float(args.elevation)
    cam.distance = float(args.distance)
    cam.lookat[:] = np.asarray(args.lookat, dtype=np.float64)

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)

    dt_s = float(rows[1]["time_s"] - rows[0]["time_s"]) if len(rows) > 1 else float(model.opt.timestep)
    frame_stride = max(1, round(1.0 / (args.fps * max(dt_s, 1e-9))))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with imageio.get_writer(str(args.output), fps=args.fps, macro_block_size=1) as writer:
        for i, row in enumerate(rows):
            if i % frame_stride != 0:
                continue
            data.qpos[: len(row["q"])] = np.asarray(row["q"], dtype=np.float64)
            data.qvel[: len(row["qd"])] = np.asarray(row["qd"], dtype=np.float64)
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=cam)
            writer.append_data(renderer.render())
            written += 1

    print(f"Wrote {written} frames ({written / args.fps:.1f}s at {args.fps} fps) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
