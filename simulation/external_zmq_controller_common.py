from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import zmq


UR5_JOINT_NAMES: tuple[str, ...] = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)

LIVE_CONTROLLER_FAMILY = "python_zmq_external_cartesian_impedance"
ZERO_TORQUE_PROBE_FAMILY = "python_zmq_external_zero_torque_probe"
SINGLE_JOINT_TORQUE_PROBE_FAMILY = "python_zmq_external_single_joint_torque_probe"
STEPPING_OWNER = "python_zmq"
SIMULATION_STARTED_BY_PYTHON = "python"


def controller_ownership_metadata(*, legacy_marker_handoff: bool = False) -> dict[str, Any]:
    """Common ownership metadata for live external ZMQ controller summaries."""
    return {
        "controller_family": LIVE_CONTROLLER_FAMILY,
        "uses_direct_torque_control": True,
        "stepping_owner": STEPPING_OWNER,
        "simulation_started_by": (
            "coppeliasim_lua_legacy_bootstrap" if legacy_marker_handoff else SIMULATION_STARTED_BY_PYTHON
        ),
        "lua_motion_enabled": False,
        "legacy_marker_handoff": bool(legacy_marker_handoff),
    }


def probe_ownership_metadata(controller_family: str) -> dict[str, Any]:
    """Common ownership metadata for the minimal external ZMQ probe summaries."""
    return {
        "controller_family": controller_family,
        "uses_direct_torque_control": True,
        "stepping_owner": STEPPING_OWNER,
        "simulation_started_by": SIMULATION_STARTED_BY_PYTHON,
        "lua_motion_enabled": False,
    }


def write_json_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def close_remote_api_client(client: Any) -> None:
    """Best-effort close of a RemoteAPIClient without lingering sockets."""
    try:
        client.socket.setsockopt(zmq.LINGER, 0)
        client.socket.close()
        client.context.term()
    except Exception:
        pass


def startup_banner_lines(*, legacy_marker_handoff: bool) -> list[str]:
    lines = [
        "LIVE_EXTERNAL_CONTROLLER=1",
        f"controller_family={LIVE_CONTROLLER_FAMILY}",
        f"stepping_owner={STEPPING_OWNER}",
        f"simulation_started_by={SIMULATION_STARTED_BY_PYTHON}",
        "lua_motion_enabled=false",
        f"legacy_marker_handoff={'true' if legacy_marker_handoff else 'false'}",
    ]
    return lines
