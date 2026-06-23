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
    ZERO_TORQUE_PROBE_FAMILY,
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
    / "external_zmq_handshake"
    / "summary.json"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=23000)
    p.add_argument("--duration-steps", type=int, default=20)
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
            print(f"[probe-handshake] RPC attempt {attempt}/60", flush=True)
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
                f"[probe-handshake] connect failure on attempt {attempt}: "
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


def _write_failure_summary(path: Path, payload: dict[str, Any], error: str) -> None:
    payload["success"] = False
    payload["error"] = error
    write_json_summary(path, payload)
    print(json.dumps(payload, indent=2))


def main() -> int:
    args = parse_args()
    print("[probe-handshake] starting", flush=True)
    joint_names = list(args.joint_name) if args.joint_name else list(UR5_JOINT_NAMES)
    joint_name_map = _build_joint_name_map(list(args.joint_name) if args.joint_name else None)
    summary: dict[str, Any] = {
        "success": False,
        **probe_ownership_metadata(ZERO_TORQUE_PROBE_FAMILY),
        "steps_requested": int(args.duration_steps),
        "steps_completed": 0,
        "sim_time_start_s": None,
        "sim_time_end_s": None,
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
        print(f"[probe-handshake] connecting to {args.host}:{args.port}", flush=True)
        client, sim = _connect(args.host, args.port)
        print("[probe-handshake] connected; ensuring UR5 model", flush=True)
        _ensure_ur5_model_loaded(sim)
        print("[probe-handshake] configuring adapter", flush=True)
        cfg = CoppeliaSimConfig(
            zmq_host=args.host,
            zmq_port=args.port,
            joint_name_map=joint_name_map,
            stepping=True,
        )
        adapter = CoppeliaSimURAdapter(cfg)
        adapter.connect_with_existing_client(client)
        adapter.configure_force_torque_mode()

        print("[probe-handshake] enabling stepping and starting simulation", flush=True)
        sim.setStepping(True)
        sim.startSimulation()
        sim_time_start = float(sim.getSimulationTime())
        print(f"[probe-handshake] simulation time start={sim_time_start:.6f}", flush=True)
        summary["sim_time_start_s"] = sim_time_start
        summary["joint_handles"] = [row["handle"] for row in adapter.resolved_joint_handles()]
        summary["joint_handles_resolved"] = len(summary["joint_handles"]) == 6

        progressed = False
        steps_completed = 0
        for _ in range(int(args.duration_steps)):
            print(f"[probe-handshake] stepping {steps_completed + 1}/{int(args.duration_steps)}", flush=True)
            q, _ = adapter.read_joint_state()
            _ = q  # ensure the read happens every step
            adapter.apply_torque(np.zeros(6, dtype=np.float64))
            sim.step()
            steps_completed += 1
            summary["steps_completed"] = steps_completed
            if float(sim.getSimulationTime()) > sim_time_start + 1e-12:
                progressed = True
        summary["sim_time_end_s"] = float(sim.getSimulationTime())
        summary["success"] = bool(progressed and summary["joint_handles_resolved"])
        if not progressed:
            summary["error"] = "simulation time did not advance during stepping"
        elif not summary["joint_handles_resolved"]:
            summary["error"] = "not all six UR5 joint handles resolved"
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        _write_failure_summary(args.summary_json, summary, summary["error"])
        try:
            if sim is not None:
                sim.stopSimulation()
        except Exception:
            pass
        if client is not None:
            close_remote_api_client(client)
        return 1
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
