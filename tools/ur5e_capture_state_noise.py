#!/usr/bin/env python3
"""Capture raw RTDE state at full rate while the robot sits still, for
characterizing real sensor noise (q/qd/tcp_pose jitter) against the
sim's noise-injection parameters.

Receive-only, like ``tools/ur5e_connect.py`` -- this script never imports
``hardware.motion`` and is physically incapable of commanding any robot
motion regardless of what flags you pass it. No guardrails are needed or
touched: nothing is asked to move, so there is nothing for a Cartesian
safety guard to evaluate. Run this with the robot already parked wherever
you want the noise characterized (it does not move it there for you).

Writes one JSON row per sample to --output (default: a timestamped file
under outputs/hardware_state_noise/), and prints per-axis/per-joint
standard deviation over the capture at the end -- directly comparable to
config/rl_gain_scheduling_noise_smoke.yaml's q_noise_std_rad/
qd_noise_std_radps or tools/ur5e_mujoco_torque_experiments.py's
--q-noise-std-rad/--qd-noise-std-radps.

Example:
  python tools/ur5e_capture_state_noise.py --robot-ip 172.16.71.77 \
    --duration-s 10.0 --frequency-hz 500.0
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from _bootstrap import ensure_repo_root

ensure_repo_root()

from hardware.link import RTDEStateError, UR5eLink  # noqa: E402
from hardware.safety import UR5eSafetyLimits  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--robot-ip", required=True, help="UR5e robot IP address. No default -- always explicit.")
    p.add_argument("--duration-s", type=float, default=10.0)
    p.add_argument("--frequency-hz", type=float, default=500.0, help="Sample rate. 500Hz matches direct_torque mode.")
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def _default_output() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "outputs" / "hardware_state_noise" / f"capture_{stamp}.jsonl"


def compute_noise_stats(rows: list[dict]) -> dict:
    """Per-axis/per-joint std over a stationary capture. Pure function,
    testable without a real (or fake) RTDE link -- rows are plain dicts
    with q/qd/tcp_pose lists, exactly what gets written to the JSONL file.
    """
    q = np.array([r["q"] for r in rows], dtype=np.float64)
    qd = np.array([r["qd"] for r in rows], dtype=np.float64)
    tcp = np.array([r["tcp_pose"] for r in rows], dtype=np.float64)
    q_std = np.std(q, axis=0)
    qd_std = np.std(qd, axis=0)
    return {
        "q_std_rad_per_joint": q_std.tolist(),
        "q_std_rad_max": float(np.max(q_std)),
        "qd_std_radps_per_joint": qd_std.tolist(),
        "qd_std_radps_max": float(np.max(qd_std)),
        "tcp_pos_std_m": np.std(tcp[:, :3], axis=0).tolist(),
        "tcp_rot_std_rad": np.std(tcp[:, 3:], axis=0).tolist(),
    }


def main() -> int:
    args = parse_args()
    output_path = args.output or _default_output()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    limits = UR5eSafetyLimits()
    link = UR5eLink(args.robot_ip, args.frequency_hz, limits=limits)
    link.connect(with_control=False)
    print(f"[connected] {args.robot_ip} (receive-only) -- capturing {args.duration_s}s at {args.frequency_hz} Hz")

    period_s = 1.0 / args.frequency_hz
    rows: list[dict] = []
    t_start = time.monotonic()
    try:
        with output_path.open("w", encoding="utf-8") as f:
            while (time.monotonic() - t_start) < args.duration_s:
                cycle_start = time.monotonic()
                try:
                    state = link.read_state()
                except RTDEStateError as exc:
                    print(f"[read failed, aborting capture] {exc}")
                    break
                row = {
                    "t_s": time.monotonic() - t_start,
                    "q": state.q.tolist(),
                    "qd": state.qd.tolist(),
                    "tcp_pose": state.tcp_pose.tolist(),
                    "robot_timestamp_s": state.robot_timestamp_s,
                }
                rows.append(row)
                f.write(json.dumps(row) + "\n")
                sleep_s = period_s - (time.monotonic() - cycle_start)
                if sleep_s > 0:
                    time.sleep(sleep_s)
    finally:
        link.disconnect()

    print(f"[captured] {len(rows)} samples over {args.duration_s}s -> {output_path}")
    if len(rows) < 2:
        print("[warning] too few samples for noise statistics")
        return 0

    stats = compute_noise_stats(rows)
    print("\n[noise statistics -- std over the capture, robot held still]")
    print(f"  q_std_rad (per joint):     {[round(v, 8) for v in stats['q_std_rad_per_joint']]}")
    print(f"  q_std_rad (max over joints): {stats['q_std_rad_max']:.8f}")
    print(f"  qd_std_radps (per joint):  {[round(v, 8) for v in stats['qd_std_radps_per_joint']]}")
    print(f"  qd_std_radps (max over joints): {stats['qd_std_radps_max']:.8f}")
    print(f"  tcp_pos_std_m (x,y,z):     {[round(v, 8) for v in stats['tcp_pos_std_m']]}")
    print(f"  tcp_rot_std_rad (rx,ry,rz): {[round(v, 8) for v in stats['tcp_rot_std_rad']]}")
    print(
        "\nCompare q_std_rad/qd_std_radps above against "
        "config/rl_gain_scheduling_noise_smoke.yaml's q_noise_std_rad/qd_noise_std_radps "
        "and tools/ur5e_mujoco_torque_experiments.py's --q-noise-std-rad/--qd-noise-std-radps."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
