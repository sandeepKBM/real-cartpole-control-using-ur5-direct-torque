#!/usr/bin/env python3
"""
Render a 360-degree orbit video of a saved UR5e pose.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "glfw"

import mediapy
import mujoco
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
SCENE_XML = BASE_DIR / "mujoco_menagerie" / "universal_robots_ur5e" / "scene_ur5e_cartpole.xml"
VIDEO_OUTPUT_DIR = BASE_DIR / "demonstration_videos" / "ur5e_cartpole"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-json", type=Path, required=True, help="Run summary JSON containing the pose.")
    parser.add_argument(
        "--pose-key",
        type=str,
        default="final_q",
        choices=("final_q", "origin_target_q"),
        help="Which joint vector in the summary to render.",
    )
    parser.add_argument("--duration", type=float, default=10.0, help="Video duration in seconds.")
    parser.add_argument("--fps", type=int, default=50, help="Video frame rate.")
    parser.add_argument("--width", type=int, default=960, help="Requested render width.")
    parser.add_argument("--height", type=int, default=720, help="Requested render height.")
    parser.add_argument("--azimuth-start", type=float, default=12.0, help="Starting camera azimuth.")
    parser.add_argument("--azimuth-sweep", type=float, default=360.0, help="Total azimuth sweep over the clip.")
    parser.add_argument("--elevation", type=float, default=-18.0, help="Camera elevation.")
    parser.add_argument("--distance", type=float, default=2.05, help="Camera distance.")
    parser.add_argument(
        "--video-name",
        type=str,
        default=None,
        help="Optional output filename. Defaults to <summary-stem>_orbit.mp4.",
    )
    return parser.parse_args()


def make_camera(lookat: np.ndarray, azimuth: float, elevation: float, distance: float) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = np.asarray(lookat, dtype=np.float64)
    camera.azimuth = float(azimuth)
    camera.elevation = float(elevation)
    camera.distance = float(distance)
    return camera


def main() -> None:
    args = parse_args()
    report = json.loads(args.summary_json.read_text())
    q = np.asarray(report[args.pose_key], dtype=np.float64)

    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[: model.nu] = q
    data.ctrl[:] = q
    mujoco.mj_forward(model, data)

    lookat = np.asarray(
        report.get("target_origin_site_world", report.get("final_origin_site_world", [0.0, -0.1, 0.75])),
        dtype=np.float64,
    )
    lookat[2] = max(float(lookat[2]), 0.55)

    render_width = min(args.width, int(model.vis.global_.offwidth))
    render_height = min(args.height, int(model.vis.global_.offheight))
    if render_width != args.width or render_height != args.height:
        print(
            "Requested render size "
            f"{args.width}x{args.height} exceeds MuJoCo framebuffer "
            f"{int(model.vis.global_.offwidth)}x{int(model.vis.global_.offheight)}; "
            f"using {render_width}x{render_height}."
        )

    renderer = mujoco.Renderer(model, height=render_height, width=render_width)
    n_frames = int(round(args.duration * args.fps))
    frames: list[np.ndarray] = []
    for frame_idx in range(n_frames):
        phase = frame_idx / max(1, n_frames - 1)
        azimuth = args.azimuth_start + args.azimuth_sweep * phase
        camera = make_camera(
            lookat=lookat,
            azimuth=azimuth,
            elevation=args.elevation,
            distance=args.distance,
        )
        renderer.update_scene(data, camera)
        frames.append(renderer.render().copy())

    stem = args.summary_json.stem + "_orbit"
    video_path = VIDEO_OUTPUT_DIR / (args.video_name or f"{stem}.mp4")
    mediapy.write_video(video_path, frames, fps=args.fps)
    print(f"Rendered pose key: {args.pose_key}")
    print(f"Pose q: {np.array2string(q, precision=6, separator=', ')}")
    print(f"Saved orbit video: {video_path}")


if __name__ == "__main__":
    main()
