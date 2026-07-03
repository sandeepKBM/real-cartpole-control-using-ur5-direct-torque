#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import zmq

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "ros2_ws" / "src" / "ur5_x_axis_controller_ros"))

DEFAULT_COPPELIA_ROOT = (
    REPO_ROOT
    / "third_party"
    / "coppelia_runtime"
    / "CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu24_04"
)
DEFAULT_COPPELIA_PYDEPS = REPO_ROOT / "third_party" / "coppelia_pydeps"
DEFAULT_UR5_MODEL = (
    DEFAULT_COPPELIA_ROOT
    / "models"
    / "robots"
    / "non-mobile"
    / "UR5.ttm"
)
_bootstrap_root = Path(os.environ.get("COPPELIA_ROOT", str(DEFAULT_COPPELIA_ROOT)))
_bootstrap_pydeps = Path(os.environ.get("COPPELIA_PYDEPS", str(DEFAULT_COPPELIA_PYDEPS)))
for candidate in (
    _bootstrap_root / "programming" / "zmqRemoteApi" / "clients" / "python" / "src",
    _bootstrap_pydeps,
):
    if candidate.exists():
        sys.path.insert(0, str(candidate))

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

from external_zmq_controller_common import (
    SINGLE_JOINT_TORQUE_PROBE_FAMILY,
    UR5_JOINT_NAMES,
    close_remote_api_client,
    probe_ownership_metadata,
    write_json_summary,
)
from ur5_x_axis_controller_ros.coppeliasim_adapter import (
    CoppeliaSimConfig,
    CoppeliaSimURAdapter,
)


DEFAULT_SUMMARY = (
    REPO_ROOT
    / "outputs"
    / "control_runs"
    / "external_zmq_single_joint_torque"
    / "summary.json"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=23000)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--active-steps", type=int, default=50)
    p.add_argument("--torque", type=float, default=0.05)
    p.add_argument("--min-abs-displacement-rad", type=float, default=1e-5)
    p.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY)
    p.add_argument(
        "--joint-name",
        action="append",
        default=None,
        help=(
            "Optional ordered list of six UR5 joint names or object paths. "
            "If omitted, the built-in UR5 joint list is used."
        ),
    )
    return p.parse_args()


def _build_joint_name_map(joint_names: list[str] | None) -> dict[str, str]:
    if not joint_names:
        return {}
    if len(joint_names) != 6:
        raise ValueError("--joint-name must be provided exactly six times or omitted")
    return {canonical: provided for canonical, provided in zip(UR5_JOINT_NAMES, joint_names)}


def _connect(host: str, port: int) -> tuple[Any, Any]:
    client = None
    last_exc: Exception | None = None
    for attempt in range(1, 61):
        try:
            print(f"[probe-single-joint] RPC attempt {attempt}/60", flush=True)
            client = RemoteAPIClient(host=host, port=port)
            first_timeout_ms = int(os.environ.get("REAL_CARTPOLE_RPC_FIRST_RCVTIMEO_MS", "20000") or 20000)
            client.socket.setsockopt(zmq.RCVTIMEO, first_timeout_ms)
            sim = client.require("sim")
            sim.getSimulationState()
            client.socket.setsockopt(zmq.RCVTIMEO, 10 * 60 * 1000)
            return client, sim
        except Exception as exc:
            last_exc = exc
            print(
                f"[probe-single-joint] connect failure on attempt {attempt}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            if client is not None:
                close_remote_api_client(client)
            time.sleep(0.25)
    raise RuntimeError(f"Could not connect to CoppeliaSim ZMQ RPC at {host}:{port}") from last_exc


def _ensure_ur5_model_loaded(sim: Any) -> None:
    try:
        sim.getObject("/UR5")
    except Exception:
        sim.loadModel(str(DEFAULT_UR5_MODEL))


def main() -> int:
    args = parse_args()
    print("[probe-single-joint] starting", flush=True)
    joint_names = list(args.joint_name) if args.joint_name else list(UR5_JOINT_NAMES)
    joint_name_map = _build_joint_name_map(list(args.joint_name) if args.joint_name else None)
    summary: dict[str, Any] = {
        "success": False,
        **probe_ownership_metadata(SINGLE_JOINT_TORQUE_PROBE_FAMILY),
        "steps_requested": int(args.steps),
        "steps_completed": 0,
        "active_steps_requested": int(args.active_steps),
        "torque_nm": float(args.torque),
        "min_abs_displacement_rad": float(args.min_abs_displacement_rad),
        "joint_0_start_rad": None,
        "joint_0_end_rad": None,
        "joint_0_displacement_rad": None,
        "joint_0_displacement_nonzero": False,
        "joint_handles_resolved": False,
        "joint_names": joint_names,
        "joint_handles": [],
        "error": None,
        "summary_json": str(args.summary_json),
    }

    client = None
    sim = None
    adapter = None
    try:
        print(f"[probe-single-joint] connecting to {args.host}:{args.port}", flush=True)
        client, sim = _connect(args.host, args.port)
        print("[probe-single-joint] connected; ensuring UR5 model", flush=True)
        _ensure_ur5_model_loaded(sim)
        print("[probe-single-joint] configuring adapter", flush=True)
        cfg = CoppeliaSimConfig(
            zmq_host=args.host,
            zmq_port=args.port,
            joint_name_map=joint_name_map,
            stepping=True,
        )
        adapter = CoppeliaSimURAdapter(cfg)
        adapter.connect_with_existing_client(client)
        adapter.configure_force_torque_mode()

        print("[probe-single-joint] enabling stepping and starting simulation", flush=True)
        sim.setStepping(True)
        sim.startSimulation()
        sim_time_start = float(sim.getSimulationTime())
        print(f"[probe-single-joint] simulation time start={sim_time_start:.6f}", flush=True)
        q_start, _ = adapter.read_joint_state()
        summary["joint_handles"] = [row["handle"] for row in adapter.resolved_joint_handles()]
        summary["joint_handles_resolved"] = len(summary["joint_handles"]) == 6
        summary["joint_0_start_rad"] = float(q_start[0])

        q_last = np.asarray(q_start, dtype=np.float64)
        steps_completed = 0
        for step in range(int(args.steps)):
            if step == 0:
                print("[probe-single-joint] stepping loop started", flush=True)
            tau = np.zeros(6, dtype=np.float64)
            if step < int(args.active_steps):
                tau[0] = float(args.torque)
            adapter.apply_torque(tau)
            sim.step()
            steps_completed += 1
            summary["steps_completed"] = steps_completed
            q_last, _ = adapter.read_joint_state()

        summary["joint_0_end_rad"] = float(q_last[0])
        displacement = float(summary["joint_0_end_rad"] - float(summary["joint_0_start_rad"]))
        summary["joint_0_displacement_rad"] = displacement
        summary["joint_0_displacement_nonzero"] = bool(abs(displacement) >= float(args.min_abs_displacement_rad))
        summary["success"] = bool(summary["joint_handles_resolved"] and summary["joint_0_displacement_nonzero"])
        if not summary["joint_handles_resolved"]:
            summary["error"] = "not all six UR5 joint handles resolved"
        elif not summary["joint_0_displacement_nonzero"]:
            summary["error"] = (
                "joint_0 displacement below threshold: "
                f"|{displacement:.6e}| < {float(args.min_abs_displacement_rad):.6e}"
            )
        summary["sim_time_start_s"] = sim_time_start
        summary["sim_time_end_s"] = float(sim.getSimulationTime())
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            if sim is not None:
                sim.stopSimulation()
        except Exception:
            pass
        if client is not None:
            close_remote_api_client(client)

    write_json_summary(args.summary_json, summary)
    print(json.dumps(summary, indent=2))
    return 0 if summary["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
