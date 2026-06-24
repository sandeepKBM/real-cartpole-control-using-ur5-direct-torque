#!/usr/bin/env python3
"""Render a 3D UR5 MuJoCo replay MP4 from a CoppeliaSim controller JSONL trace."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE = (
    REPO_ROOT / "mujoco_menagerie" / "universal_robots_ur5e" / "scene_ur5e_cartpole.xml"
)
FALLBACK_SCENE = REPO_ROOT / "mujoco_menagerie" / "universal_robots_ur5e" / "scene.xml"


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


def pick_scene(path: Path | None) -> Path:
    if path is not None:
        return path
    if DEFAULT_SCENE.exists():
        return DEFAULT_SCENE
    return FALLBACK_SCENE


def write_mp4_ffmpeg(frames: list[np.ndarray], out_path: Path, fps: int) -> None:
    if not frames:
        raise ValueError("no frames to encode")
    h, w, _ = frames[0].shape
    ffmpeg_bin = os.environ.get("FFMPEG_BIN", "ffmpeg")
    if shutil.which(ffmpeg_bin) is None and ffmpeg_bin == "ffmpeg":
        raise RuntimeError("ffmpeg not found; set FFMPEG_BIN")
    cmd = [
        ffmpeg_bin,
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s:v",
        f"{w}x{h}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(out_path),
    ]
    data = b"".join(np.asarray(frame, dtype=np.uint8).tobytes() for frame in frames)
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    _out, err = proc.communicate(input=data)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed ({proc.returncode}): "
            f"{err.decode('utf-8', errors='replace') if err else ''}"
        )


def render_trace(
    rows: list[dict],
    out_path: Path,
    *,
    scene_path: Path,
    fps: int,
    width: int,
    height: int,
) -> dict:
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = np.array([0.35, -0.15, 0.45], dtype=np.float64)
    camera.distance = 2.1
    camera.azimuth = 118.0
    camera.elevation = -18.0

    scene_option = mujoco.MjvOption()
    frames: list[np.ndarray] = []
    ee_path: list[np.ndarray] = []
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    if site_id < 0:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")

    for row in rows:
        q = np.asarray(row["q"], dtype=np.float64).reshape(-1)
        n_arm = min(q.size, 6)
        data.qpos[:n_arm] = q[:n_arm]
        if model.nq > 6 and n_arm == 6:
            data.qpos[6] = 0.0
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=camera, scene_option=scene_option)
        frames.append(renderer.render().copy())
        if site_id >= 0:
            ee_path.append(np.array(data.site_xpos[site_id], dtype=np.float64))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_mp4_ffmpeg(frames, out_path, fps=fps)
    renderer.close()
    ee_path_arr = np.vstack(ee_path) if ee_path else np.zeros((0, 3))
    return {
        "frames": len(frames),
        "duration_s": float(rows[-1]["time"] - rows[0]["time"]) if len(rows) > 1 else 0.0,
        "scene_path": str(scene_path),
        "ee_x_span_m": float(np.max(ee_path_arr[:, 0]) - np.min(ee_path_arr[:, 0]))
        if ee_path_arr.size
        else None,
        "output_path": str(out_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("demonstration_videos/ur5e_coppeliasim/coppelia_fast_x_transport.mp4"),
    )
    parser.add_argument("--scene", type=Path, default=None)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    args = parser.parse_args()

    rows = load_trace(args.jsonl)
    meta = render_trace(
        rows,
        args.out,
        scene_path=pick_scene(args.scene),
        fps=args.fps,
        width=args.width,
        height=args.height,
    )
    print(json.dumps(meta, indent=2))
    print(f"Wrote UR5 simulation replay video: {args.out}")


if __name__ == "__main__":
    main()
