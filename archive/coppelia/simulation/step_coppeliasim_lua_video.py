#!/usr/bin/env python3
"""Advance a CoppeliaSim stepped run while a Lua add-on owns control/capture."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from coppeliasim_zmqremoteapi_client import RemoteAPIClient


def count_frames(frame_dir: Path) -> int:
    return sum(1 for _ in frame_dir.glob("frame_*.png"))


def connect(host: str, port: int, timeout_s: float):
    deadline = time.monotonic() + timeout_s
    last_exc: Exception | None = None
    attempt = 0
    while time.monotonic() < deadline:
        attempt += 1
        try:
            client = RemoteAPIClient(host=host, port=port)
            return client, client.require("sim")
        except Exception as exc:  # pragma: no cover - exercised by live Coppelia
            last_exc = exc
            print(
                f"Waiting for CoppeliaSim RPC at {host}:{port} "
                f"({attempt}): {exc}",
                flush=True,
            )
            time.sleep(0.5)
    raise RuntimeError(f"Failed connecting to CoppeliaSim at {host}:{port}") from last_exc


def wait_for_file(path: Path, timeout_s: float, label: str) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.1)
    raise TimeoutError(f"Timed out waiting for {label}: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--frame-dir", type=Path, required=True)
    parser.add_argument("--done-marker", type=Path, required=True)
    parser.add_argument("--frame-count", type=int, required=True)
    parser.add_argument("--release-marker", type=Path)
    parser.add_argument("--ready-marker", type=Path)
    parser.add_argument("--configured-marker", type=Path)
    parser.add_argument("--connect-timeout", type=float, default=30.0)
    parser.add_argument("--timeout", type=float, default=180.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.release_marker is not None:
        args.release_marker.parent.mkdir(parents=True, exist_ok=True)
        args.release_marker.write_text("release\n", encoding="utf-8")
    if args.configured_marker is not None:
        wait_for_file(args.configured_marker, args.connect_timeout, "Lua configured marker")
    if args.ready_marker is not None:
        wait_for_file(args.ready_marker, args.connect_timeout, "Lua ready marker")

    client, sim = connect(args.host, args.port, args.connect_timeout)
    deadline = time.monotonic() + args.timeout
    steps = 0
    last_report_t = 0.0

    try:
        sim.setStepping(True)
        start_deadline = time.monotonic() + min(args.timeout, 30.0)
        while sim.getSimulationState() == sim.simulation_stopped:
            if time.monotonic() >= start_deadline:
                print("Simulation did not start after ready marker; requesting start", flush=True)
                sim.startSimulation()
                break
            time.sleep(0.05)

        while time.monotonic() < deadline:
            frames = count_frames(args.frame_dir)
            if args.done_marker.exists() or frames >= args.frame_count:
                print(f"Lua video step pump complete: frames={frames} steps={steps}", flush=True)
                return 0

            try:
                sim.step()
            except Exception as exc:
                frames = count_frames(args.frame_dir)
                if args.done_marker.exists() or frames >= args.frame_count:
                    print(
                        f"Lua video step pump complete after simulator exit: "
                        f"frames={frames} steps={steps}",
                        flush=True,
                    )
                    return 0
                print(
                    f"CoppeliaSim step failed after {steps} steps and "
                    f"{frames} frames: {exc}",
                    flush=True,
                )
                return 1

            steps += 1
            now = time.monotonic()
            if now - last_report_t >= 5.0:
                last_report_t = now
                print(f"Lua video step pump progress: frames={frames} steps={steps}", flush=True)

        frames = count_frames(args.frame_dir)
        print(f"Timed out stepping Lua video: frames={frames} steps={steps}", flush=True)
        return 2
    finally:
        try:
            sim.setStepping(False)
        except Exception:
            pass
        # Keep a reference alive until after the final remote call.
        _ = client


if __name__ == "__main__":
    raise SystemExit(main())
