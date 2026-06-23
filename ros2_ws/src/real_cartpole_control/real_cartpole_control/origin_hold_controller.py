#!/usr/bin/env python3
"""
Minimal real-UR5e control node for holding the measured origin pose.

This node is intentionally simple:

- subscribe to joint states
- reorder them into the UR5e joint vector
- compute a bounded incremental target toward `origin_peak`
- optionally publish that target as a short JointTrajectory

The default mode is safe-by-default: it computes the command but does not
publish to hardware until `publish_commands` is explicitly enabled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from builtin_interfaces.msg import Duration
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


DEFAULT_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

DEFAULT_ORIGIN_Q = [
    0.0,
    -1.570784,
    -0.000007,
    -2.356201,
    -1.570784,
    0.0,
]


@dataclass
class ControllerConfig:
    joint_names: list[str]
    origin_q: list[float]
    joint_state_topic: str
    trajectory_topic: str
    publish_commands: bool
    control_period_s: float
    trajectory_duration_s: float
    max_step_rad: float
    position_tolerance_rad: float


def _read_string_list(node: Node, name: str, default: Sequence[str]) -> list[str]:
    node.declare_parameter(name, list(default))
    return list(node.get_parameter(name).value)


def _read_float_list(node: Node, name: str, default: Sequence[float]) -> list[float]:
    node.declare_parameter(name, list(default))
    return [float(v) for v in node.get_parameter(name).value]


class OriginHoldController(Node):
    def __init__(self) -> None:
        super().__init__("real_cartpole_origin_hold_controller")
        self.config = ControllerConfig(
            joint_names=_read_string_list(self, "joint_names", DEFAULT_JOINT_NAMES),
            origin_q=_read_float_list(self, "origin_q", DEFAULT_ORIGIN_Q),
            joint_state_topic=self._read_string_param("joint_state_topic", "/joint_states"),
            trajectory_topic=self._read_string_param(
                "trajectory_topic", "/scaled_joint_trajectory_controller/joint_trajectory"
            ),
            publish_commands=self._read_bool_param("publish_commands", False),
            control_period_s=self._read_float_param("control_period_s", 0.2),
            trajectory_duration_s=self._read_float_param("trajectory_duration_s", 0.6),
            max_step_rad=self._read_float_param("max_step_rad", 0.08),
            position_tolerance_rad=self._read_float_param("position_tolerance_rad", 0.02),
        )

        if len(self.config.joint_names) != len(self.config.origin_q):
            raise ValueError("`joint_names` and `origin_q` must have the same length.")

        self._latest_joint_state: JointState | None = None
        self._name_to_index: dict[str, int] | None = None
        self._warned_missing_joints = False
        self._joint_state_version = 0
        self._last_processed_version = -1

        self._joint_state_sub = self.create_subscription(
            JointState,
            self.config.joint_state_topic,
            self._on_joint_state,
            10,
        )
        self._trajectory_pub = self.create_publisher(
            JointTrajectory,
            self.config.trajectory_topic,
            10,
        )
        self._timer = self.create_timer(self.config.control_period_s, self._on_timer)

        mode = "LIVE publish enabled" if self.config.publish_commands else "dry-run only"
        self.get_logger().info(
            f"Origin hold controller ready. Mode: {mode}. "
            f"Listening on {self.config.joint_state_topic}, target topic {self.config.trajectory_topic}"
        )

    def _read_string_param(self, name: str, default: str) -> str:
        self.declare_parameter(name, default)
        return str(self.get_parameter(name).value)

    def _read_bool_param(self, name: str, default: bool) -> bool:
        self.declare_parameter(name, default)
        return bool(self.get_parameter(name).value)

    def _read_float_param(self, name: str, default: float) -> float:
        self.declare_parameter(name, default)
        return float(self.get_parameter(name).value)

    def _on_joint_state(self, msg: JointState) -> None:
        self._latest_joint_state = msg
        self._joint_state_version += 1
        if self._name_to_index is None:
            self._name_to_index = {name: idx for idx, name in enumerate(msg.name)}

    def _ordered_positions(self) -> list[float] | None:
        if self._latest_joint_state is None or self._name_to_index is None:
            return None

        positions = []
        missing = []
        for joint_name in self.config.joint_names:
            idx = self._name_to_index.get(joint_name)
            if idx is None or idx >= len(self._latest_joint_state.position):
                missing.append(joint_name)
            else:
                positions.append(float(self._latest_joint_state.position[idx]))

        if missing:
            if not self._warned_missing_joints:
                self.get_logger().warning(
                    "JointState does not contain the expected UR5e joints: " + ", ".join(missing)
                )
                self._warned_missing_joints = True
            return None

        return positions

    def _next_target(self, current_q: Sequence[float]) -> list[float]:
        bounded_target = []
        for q_now, q_goal in zip(current_q, self.config.origin_q):
            error = q_goal - q_now
            if error > self.config.max_step_rad:
                error = self.config.max_step_rad
            elif error < -self.config.max_step_rad:
                error = -self.config.max_step_rad
            bounded_target.append(q_now + error)
        return bounded_target

    def _max_error(self, current_q: Sequence[float]) -> float:
        return max(abs(q_goal - q_now) for q_now, q_goal in zip(current_q, self.config.origin_q))

    def _on_timer(self) -> None:
        if self._joint_state_version == self._last_processed_version:
            return

        current_q = self._ordered_positions()
        if current_q is None:
            return
        self._last_processed_version = self._joint_state_version

        max_error = self._max_error(current_q)
        if max_error <= self.config.position_tolerance_rad:
            self.get_logger().debug("Origin already held within tolerance.")
            return

        next_target = self._next_target(current_q)
        if not self.config.publish_commands:
            self.get_logger().info(
                f"Dry-run target computed. max_error={max_error:.4f} rad, next_target={next_target}"
            )
            return

        traj = JointTrajectory()
        traj.joint_names = list(self.config.joint_names)
        point = JointTrajectoryPoint()
        point.positions = list(next_target)

        duration = max(self.config.trajectory_duration_s, 1e-3)
        sec = int(duration)
        nanosec = int(round((duration - sec) * 1e9))
        point.time_from_start = Duration(sec=sec, nanosec=nanosec)
        traj.points = [point]
        self._trajectory_pub.publish(traj)
        self.get_logger().info(
            f"Published bounded origin-hold command. max_error={max_error:.4f} rad"
        )


def main() -> None:
    rclpy.init()
    node = OriginHoldController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
