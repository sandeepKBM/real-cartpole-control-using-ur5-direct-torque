#!/usr/bin/env python3
"""
Headless CoppeliaSim video smoke test for the official UR5 model.

This script assumes CoppeliaSim is already running with the ZMQ Remote API
enabled. It loads the default scene, inserts the official UR5 model, advances
the simulation for a short duration, and writes an MP4 plus a summary JSON.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COPPELIA_ROOT = (
    REPO_ROOT
    / "third_party"
    / "coppelia_runtime"
    / "CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu24_04"
)
DEFAULT_VIDEO_DIR = REPO_ROOT / "demonstration_videos" / "ur5e_coppeliasim"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "control_runs"


def require_remote_api(host: str, port: int) -> object:
    last_exc: Exception | None = None
    for attempt in range(1, 31):
        try:
            client = RemoteAPIClient(host, port)
            return client.require("sim")
        except Exception as exc:
            last_exc = exc
            print(
                f"Waiting for CoppeliaSim RPC at {host}:{port} "
                f"({attempt}/30): {exc}",
                flush=True,
            )
            time.sleep(1.0)
    raise RuntimeError(f"Failed connecting to CoppeliaSim at {host}:{port}") from last_exc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--coppelia-root", type=Path, default=DEFAULT_COPPELIA_ROOT)
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=23000)
    p.add_argument("--duration", type=float, default=2.0)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=360)
    p.add_argument(
        "--video-name",
        type=str,
        default="coppeliasim_ur5_video_smoke.mp4",
    )
    p.add_argument(
        "--summary-name",
        type=str,
        default="coppeliasim_ur5_video_smoke_summary.json",
    )
    return p.parse_args()


def decode_rgb_bytes(buffer: bytes, resolution: list[int]) -> np.ndarray:
    width, height = int(resolution[0]), int(resolution[1])
    img = np.frombuffer(buffer, dtype=np.uint8).reshape(height, width, 3)
    return np.flipud(img)


def write_video_ffmpeg(path: Path, frames: list[np.ndarray], fps: int) -> None:
    if not frames:
        raise RuntimeError("No frames captured for video output.")
    height, width, _ = frames[0].shape
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s:v",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        assert proc.stdin is not None
        for frame in frames:
            proc.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())
        proc.stdin.close()
        _, stderr = proc.communicate()
    finally:
        if proc.stdin is not None and not proc.stdin.closed:
            proc.stdin.close()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed writing {path}: {stderr.decode('utf-8', errors='replace')}")


def make_vision_sensor(sim: object, width: int, height: int) -> int:
    options = 2 | 4
    int_params = [int(width), int(height), 0, 0]
    float_params = [
        0.02,
        6.0,
        np.deg2rad(58.0),
        0.1,
        0.0,
        0.0,
        0.82,
        0.86,
        0.92,
        0.0,
        0.0,
    ]
    sensor = int(sim.createVisionSensor(options, int_params, float_params))
    sim.setObjectAlias(sensor, "SmokeVideoCamera")
    return sensor


def camera_pose(step_idx: int, total_steps: int) -> list[float]:
    progress = 0.0 if total_steps <= 1 else float(step_idx) / float(total_steps - 1)
    yaw = math.radians(-48.0 + 18.0 * progress)
    pitch = math.radians(22.0)
    radius = 1.95
    target = np.array([0.0, 0.0, 0.62], dtype=np.float64)

    cam_pos = np.array(
        [
            radius * math.cos(yaw),
            radius * math.sin(yaw),
            target[2] + 0.34,
        ],
        dtype=np.float64,
    )

    forward = target - cam_pos
    forward /= np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    # Vision sensor local +Z looks forward.
    rot = np.column_stack((right, up, forward))
    quat_xyzw = rotation_matrix_to_quat_xyzw(rot)
    return [
        float(cam_pos[0]),
        float(cam_pos[1]),
        float(cam_pos[2]),
        float(quat_xyzw[3]),
        float(quat_xyzw[0]),
        float(quat_xyzw[1]),
        float(quat_xyzw[2]),
    ]


def rotation_matrix_to_quat_xyzw(rot: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rot))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rot[2, 1] - rot[1, 2]) / s
        qy = (rot[0, 2] - rot[2, 0]) / s
        qz = (rot[1, 0] - rot[0, 1]) / s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
        qw = (rot[2, 1] - rot[1, 2]) / s
        qx = 0.25 * s
        qy = (rot[0, 1] + rot[1, 0]) / s
        qz = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
        qw = (rot[0, 2] - rot[2, 0]) / s
        qx = (rot[0, 1] + rot[1, 0]) / s
        qy = 0.25 * s
        qz = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
        qw = (rot[1, 0] - rot[0, 1]) / s
        qx = (rot[0, 2] + rot[2, 0]) / s
        qy = (rot[1, 2] + rot[2, 1]) / s
        qz = 0.25 * s
    quat = np.array([qx, qy, qz, qw], dtype=np.float64)
    quat /= np.linalg.norm(quat)
    return quat


def set_ur5_pose(sim: object) -> None:
    joint_paths = (
        "/UR5/joint",
        "/UR5/link/joint",
        "/UR5/link/link/joint",
        "/UR5/link/link/link/joint",
        "/UR5/link/link/link/link/joint",
        "/UR5/link/link/link/link/link/joint",
    )
    joint_targets = (0.2, -1.15, 1.55, -1.8, -1.45, 0.35)
    for joint_path, target in zip(joint_paths, joint_targets):
        joint_handle = int(sim.getObject(joint_path))
        sim.setJointPosition(joint_handle, float(target))


def main() -> None:
    args = parse_args()
    video_path = DEFAULT_VIDEO_DIR / args.video_name
    summary_path = DEFAULT_OUTPUT_DIR / args.summary_name
    video_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    scene_path = args.coppelia_root / "system" / "dfltscn.ttt"
    model_path = args.coppelia_root / "models" / "robots" / "non-mobile" / "UR5.ttm"
    if not scene_path.exists():
        raise FileNotFoundError(f"Missing default scene: {scene_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing official UR5 model: {model_path}")

    print(f"Connecting to CoppeliaSim at {args.host}:{args.port}", flush=True)
    sim = require_remote_api(args.host, args.port)

    print(f"Loading scene: {scene_path}", flush=True)
    sim.loadScene(str(scene_path))
    print(f"Loading model: {model_path}", flush=True)
    ur5_handle = int(sim.loadModel(str(model_path)))
    set_ur5_pose(sim)
    vision_sensor = make_vision_sensor(sim, args.width, args.height)

    sim.setStepping(True)
    print("Starting simulation", flush=True)
    sim.startSimulation()

    sim_dt = float(sim.getSimulationTimeStep())
    if sim_dt <= 0.0:
        sim_dt = 0.01
    total_steps = max(1, int(round(args.duration / sim_dt)))
    frame_every = max(1, int(round(1.0 / (args.fps * sim_dt))))

    frames: list[np.ndarray] = []
    captured = 0
    print(f"Capturing {args.duration:.2f}s of video at {args.fps} fps", flush=True)
    for step_idx in range(total_steps):
        pose = camera_pose(step_idx, total_steps)
        sim.setObjectPose(vision_sensor + sim.handleflag_wxyzquat, pose, sim.handle_world)
        sim.step()
        if step_idx % frame_every == 0 or step_idx == total_steps - 1:
            sim.handleVisionSensor(vision_sensor)
            img_bytes, resolution = sim.getVisionSensorImg(vision_sensor)
            frames.append(decode_rgb_bytes(img_bytes, resolution))
            captured += 1

    sim.stopSimulation()
    print(f"Writing video to {video_path}", flush=True)
    write_video_ffmpeg(video_path, frames, fps=args.fps)

    summary = {
        "scene_path": str(scene_path),
        "model_path": str(model_path),
        "video_path": str(video_path),
        "duration_s": args.duration,
        "fps": args.fps,
        "width": args.width,
        "height": args.height,
        "sim_dt_s": sim_dt,
        "ur5_handle": ur5_handle,
        "frames_written": captured,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
