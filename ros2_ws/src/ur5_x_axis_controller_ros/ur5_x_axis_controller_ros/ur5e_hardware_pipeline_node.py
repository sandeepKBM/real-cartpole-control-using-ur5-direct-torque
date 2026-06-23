#!/usr/bin/env python3
"""ROS 2 wrapper for the staged UR5e hardware pipeline.

This node is intentionally conservative:

- connection smoke only by default
- bounded servoJ motion requires explicit motion opt-in
- direct torque is blocked by default unless an explicit nonzero opt-in is set

The node does not assume that direct torque is available on the current RTDE
control library. It records capability metadata and only sends motion when the
session and caller both allow it.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _ensure_repo_root() -> Path:
    env_root = os.environ.get("REAL_CARTPOLE_REPO_ROOT", "").strip()
    if env_root:
        repo_root = Path(env_root)
        if repo_root.is_dir():
            repo_root_str = str(repo_root)
            if repo_root_str not in sys.path:
                sys.path.insert(0, repo_root_str)
            return repo_root
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "hardware").is_dir() and (parent / "ros2_ws").is_dir():
            parent_str = str(parent)
            if parent_str not in sys.path:
                sys.path.insert(0, parent_str)
            return parent
    fallback = here.parents[2]
    fallback_str = str(fallback)
    if fallback_str not in sys.path:
        sys.path.insert(0, fallback_str)
    return fallback


REPO_ROOT = _ensure_repo_root()

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String

from hardware.logging import json_dumps_safe, write_json
from hardware.ur5e_control_session import (
    UR5eCommandResult,
    UR5eConnectionSnapshot,
    UR5eHardwareSession,
    UR5eHardwareSessionConfig,
)


class UR5eHardwarePipelineNode(Node):
    """Run the staged connection / actuator / torque pipeline."""

    def __init__(self) -> None:
        super().__init__("ur5e_hardware_pipeline_node")
        self.declare_parameter("robot_ip", "")
        self.declare_parameter("frequency_hz", 500.0)
        self.declare_parameter("stage", "connection_smoke")
        self.declare_parameter("motion_opt_in", False)
        self.declare_parameter("allow_nonzero_direct_torque", False)
        self.declare_parameter("direct_torque_zero_only", True)
        self.declare_parameter("joint_index", 0)
        self.declare_parameter("amplitude_rad", 0.005)
        self.declare_parameter("max_amplitude_rad", 0.01)
        self.declare_parameter("gain", 100.0)
        self.declare_parameter("lookahead_time", 0.1)
        self.declare_parameter("velocity", 0.05)
        self.declare_parameter("acceleration", 0.05)
        self.declare_parameter("output_path", "outputs/control_runs/ur5e_hardware_pipeline_summary.json")
        self.declare_parameter("publish_status_hz", 5.0)
        self.declare_parameter("direct_torque_topic", "/ur5e/direct_torque_command")

        robot_ip = str(self.get_parameter("robot_ip").value or "")
        if not robot_ip:
            raise RuntimeError("robot_ip is required")
        self._frequency_hz = float(self.get_parameter("frequency_hz").value or 500.0)
        self._stage = str(self.get_parameter("stage").value or "connection_smoke").strip().lower()
        self._motion_opt_in = bool(self.get_parameter("motion_opt_in").value)
        self._allow_nonzero_direct_torque = bool(self.get_parameter("allow_nonzero_direct_torque").value)
        self._direct_torque_zero_only = bool(self.get_parameter("direct_torque_zero_only").value)
        self._joint_index = int(self.get_parameter("joint_index").value)
        self._amplitude_rad = float(self.get_parameter("amplitude_rad").value)
        self._max_amplitude_rad = float(self.get_parameter("max_amplitude_rad").value)
        self._gain = float(self.get_parameter("gain").value)
        self._lookahead_time = float(self.get_parameter("lookahead_time").value)
        self._velocity = float(self.get_parameter("velocity").value)
        self._acceleration = float(self.get_parameter("acceleration").value)
        self._output_path = Path(str(self.get_parameter("output_path").value or "outputs/control_runs/ur5e_hardware_pipeline_summary.json"))
        self._publish_status_hz = float(self.get_parameter("publish_status_hz").value or 5.0)
        self._direct_torque_topic = str(self.get_parameter("direct_torque_topic").value or "/ur5e/direct_torque_command")

        self._session = UR5eHardwareSession(
            UR5eHardwareSessionConfig(
                robot_ip=robot_ip,
                frequency_hz=self._frequency_hz,
                motion_opt_in=self._motion_opt_in,
                allow_nonzero_direct_torque=self._allow_nonzero_direct_torque,
                direct_torque_zero_only=self._direct_torque_zero_only,
            )
        )
        self._status_pub = self.create_publisher(String, "/ur5e/hardware_pipeline/status", 10)
        self._connection_pub = self.create_publisher(String, "/ur5e/hardware_pipeline/connection", 10)
        self._actuator_pub = self.create_publisher(String, "/ur5e/hardware_pipeline/actuator", 10)
        self._torque_sub = self.create_subscription(Float64MultiArray, self._direct_torque_topic, self._on_direct_torque, 10)

        self._snapshot: UR5eConnectionSnapshot | None = None
        self._latest_state: dict[str, Any] | None = None
        self._last_command: UR5eCommandResult | None = None
        self._pending_tau = np.zeros(6, dtype=np.float64)
        self._hold_q: np.ndarray | None = None
        self._cycles = 0
        self._accepted_commands = 0
        self._blocked_commands = 0
        self._state_failures = 0
        self._status_counter = 0
        self._status_period_cycles = max(1, int(round(self._frequency_hz / max(self._publish_status_hz, 1.0))))
        self._reason = "initialized"

        self._initialize_session()
        self._timer = self.create_timer(1.0 / max(self._frequency_hz, 1.0), self._tick)

    def _initialize_session(self) -> None:
        connect_control = self._stage in {"basic_servoj_hold", "basic_servoj_tiny", "direct_torque_probe"}
        self._snapshot = self._session.capture_snapshot(connect_control=connect_control, include_state=True)
        self._latest_state = self._snapshot.state
        if self._latest_state and "q" in self._latest_state:
            self._hold_q = np.asarray(self._latest_state["q"], dtype=np.float64).reshape(6)
        self._publish_snapshot()
        self._reason = "session initialized"

    def _publish_snapshot(self) -> None:
        if self._snapshot is None:
            return
        payload = json_dumps_safe(self._snapshot.as_dict())
        self._connection_pub.publish(String(data=payload))

    def _publish_status(self) -> None:
        payload = {
            "stage": self._stage,
            "reason": self._reason,
            "cycles": int(self._cycles),
            "accepted_commands": int(self._accepted_commands),
            "blocked_commands": int(self._blocked_commands),
            "state_failures": int(self._state_failures),
            "motion_opt_in": bool(self._motion_opt_in),
            "allow_nonzero_direct_torque": bool(self._allow_nonzero_direct_torque),
            "direct_torque_zero_only": bool(self._direct_torque_zero_only),
            "snapshot": None if self._snapshot is None else self._snapshot.as_dict(),
            "last_command": None if self._last_command is None else self._last_command.as_dict(),
            "pending_tau_nm": self._pending_tau.tolist(),
        }
        self._status_pub.publish(String(data=json_dumps_safe(payload)))

    def _on_direct_torque(self, msg: Float64MultiArray) -> None:
        tau = np.asarray(msg.data, dtype=np.float64).reshape(-1)
        if tau.shape[0] != 6 or not np.all(np.isfinite(tau)):
            return
        self._pending_tau = tau.copy()

    def _update_state(self) -> None:
        try:
            state = self._session.read_state()
        except Exception as exc:
            self._state_failures += 1
            self._reason = f"state read failed: {type(exc).__name__}: {exc}"
            return
        self._latest_state = state.as_dict()
        if self._hold_q is None and "q" in self._latest_state:
            self._hold_q = np.asarray(self._latest_state["q"], dtype=np.float64).reshape(6)
        self._reason = "state ok"

    def _tick(self) -> None:
        self._cycles += 1
        self._update_state()
        if self._latest_state is None:
            if self._cycles % self._status_period_cycles == 0:
                self._publish_status()
            return

        if self._stage == "connection_smoke":
            if self._cycles % self._status_period_cycles == 0:
                self._publish_status()
            return

        if self._stage == "basic_servoj_hold":
            if self._hold_q is None:
                self._reason = "no hold_q available yet"
                if self._cycles % self._status_period_cycles == 0:
                    self._publish_status()
                return
            self._last_command = self._session.request_servoj_hold(
                self._hold_q,
                gain=self._gain,
                lookahead_time=self._lookahead_time,
                velocity=self._velocity,
                acceleration=self._acceleration,
                period_s=1.0 / max(self._frequency_hz, 1.0),
            )
        elif self._stage == "basic_servoj_tiny":
            if self._hold_q is None:
                self._reason = "no hold_q available yet"
                if self._cycles % self._status_period_cycles == 0:
                    self._publish_status()
                return
            phase = 2.0 * np.pi * (self._cycles / max(self._frequency_hz, 1.0))
            self._last_command = self._session.request_servoj_tiny_motion(
                self._hold_q,
                joint_index=self._joint_index,
                amplitude_rad=self._amplitude_rad,
                phase=phase,
                gain=self._gain,
                lookahead_time=self._lookahead_time,
                velocity=self._velocity,
                acceleration=self._acceleration,
                period_s=1.0 / max(self._frequency_hz, 1.0),
                max_amplitude_rad=self._max_amplitude_rad,
            )
        elif self._stage == "direct_torque_probe":
            self._last_command = self._session.request_direct_torque(
                self._pending_tau,
                zero_only=self._direct_torque_zero_only,
                allow_nonzero=self._allow_nonzero_direct_torque,
            )
        else:
            self._reason = f"unknown stage: {self._stage}"
            if self._cycles % self._status_period_cycles == 0:
                self._publish_status()
            return

        if self._last_command is not None:
            if self._last_command.accepted:
                self._accepted_commands += 1
            if self._last_command.blocked:
                self._blocked_commands += 1
            self._reason = self._last_command.reason
            self._actuator_pub.publish(String(data=json_dumps_safe(self._last_command.as_dict())))

        if self._cycles % self._status_period_cycles == 0:
            self._publish_status()

    def destroy_node(self) -> bool:
        try:
            summary = {
                "stage": self._stage,
                "reason": self._reason,
                "cycles": int(self._cycles),
                "accepted_commands": int(self._accepted_commands),
                "blocked_commands": int(self._blocked_commands),
                "state_failures": int(self._state_failures),
                "motion_opt_in": bool(self._motion_opt_in),
                "allow_nonzero_direct_torque": bool(self._allow_nonzero_direct_torque),
                "direct_torque_zero_only": bool(self._direct_torque_zero_only),
                "snapshot": None if self._snapshot is None else self._snapshot.as_dict(),
                "last_command": None if self._last_command is None else self._last_command.as_dict(),
                "pending_tau_nm": self._pending_tau.tolist(),
            }
            write_json(self._output_path, summary)
        except Exception:
            pass
        try:
            self._session.safe_stop(f"{self._stage} exit")
        except Exception:
            pass
        return super().destroy_node()


def main(args: Any = None) -> None:
    rclpy.init(args=args)
    node: UR5eHardwarePipelineNode | None = None
    try:
        node = UR5eHardwarePipelineNode()
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
