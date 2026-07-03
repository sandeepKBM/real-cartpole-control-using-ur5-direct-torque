#!/usr/bin/env python3
"""CoppeliaSim ZMQ bridge: state publishers + torque subscriber."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from collections.abc import Sequence
from typing import Any, Callable

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from .config_loader import load_yaml_config
from .coppeliasim_adapter import CANONICAL_JOINT_ORDER, CoppeliaSimConfig, CoppeliaSimURAdapter
from .messages import UR5_JOINT_NAMES, float_array_from


def _startup_joint_positions_rad(raw: object) -> tuple[float, ...]:
    if raw is None or raw == "":
        return ()
    arr = np.asarray(raw, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return ()
    if arr.size != len(CANONICAL_JOINT_ORDER):
        raise ValueError(
            "coppeliasim.startup_joint_positions_rad must contain "
            f"{len(CANONICAL_JOINT_ORDER)} joint values"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("coppeliasim.startup_joint_positions_rad must be finite")
    return tuple(float(v) for v in arr.tolist())


def seed_startup_joint_positions(
    adapter: object,
    startup_joint_positions_rad: Sequence[float],
    log_fn: Callable[[str], None] | None = None,
) -> None:
    seed = np.asarray(startup_joint_positions_rad, dtype=np.float64).reshape(-1)
    if seed.size == 0:
        return
    if seed.size != len(CANONICAL_JOINT_ORDER):
        raise ValueError("startup_joint_positions_rad must contain 6 joint values")
    if not np.all(np.isfinite(seed)):
        raise ValueError("startup_joint_positions_rad must be finite")
    if log_fn is not None:
        log_fn(f"Seeding startup joint pose before simulation start: {seed.tolist()}")
    adapter.set_joint_positions(seed)  # type: ignore[attr-defined]
    q_read, _ = adapter.read_joint_state()  # type: ignore[attr-defined]
    q_read_arr = np.asarray(q_read, dtype=np.float64).reshape(-1)
    wrapped_err = (q_read_arr - seed + np.pi) % (2.0 * np.pi) - np.pi
    max_err = float(np.max(np.abs(wrapped_err)))
    if log_fn is not None:
        log_fn(
            "Startup joint pose readback max_abs_wrapped_err="
            f"{max_err:.3e} rad"
        )
    if max_err > 1e-6 and log_fn is not None:
        log_fn(
            "Startup joint pose readback differs from the requested pose "
            f"(max_abs_wrapped_err={max_err:.3e} rad); continuing because "
            "this CoppeliaSim build does not expose a reliable kinematic refresh "
            "after setJointPosition."
        )


def _default_config_path() -> str:
    try:
        from ament_index_python.packages import get_package_share_directory

        return os.path.join(
            get_package_share_directory("ur5_x_axis_controller_ros"),
            "config",
            "controller.yaml",
        )
    except Exception:
        return str(
            Path(__file__).resolve().parents[2] / "config" / "controller.yaml"
        )


class CoppeliaSimBridgeNode(Node):
    def __init__(self) -> None:
        super().__init__("coppeliasim_ur5_bridge")
        self.declare_parameter("config_path", "")
        cfg_path = self.get_parameter("config_path").value or _default_config_path()
        y = load_yaml_config(cfg_path)
        self._cop = y.get("coppeliasim", {}) or {}
        self._topics = y.get("topics", {}) or {}

        jm = self._cop.get("joint_name_map", {}) or {}
        if isinstance(jm, dict):
            joint_map = {str(k): str(v) for k, v in jm.items()}
        else:
            joint_map = {}

        for name in CANONICAL_JOINT_ORDER:
            if name not in joint_map:
                self.get_logger().fatal(f"coppeliasim.joint_name_map missing {name!r}")
                raise RuntimeError("Invalid joint_name_map")

        ta = self._cop.get("torque_application", {}) or {}
        jac = self._cop.get("jacobian", {}) or {}

        self._adapter = CoppeliaSimURAdapter(
            CoppeliaSimConfig(
                zmq_host=str(self._cop.get("host", "127.0.0.1")),
                zmq_port=int(self._cop.get("port", 23000)),
                joint_name_map=joint_map,
                startup_joint_positions_rad=_startup_joint_positions_rad(
                    self._cop.get("startup_joint_positions_rad", ())
                ),
                ee_object_name=str(self._cop.get("ee_object_name", "/UR5/UR5_connection")),
                ee_object_name_alternates=tuple(
                    str(p) for p in (self._cop.get("ee_object_name_alternates", []) or [])
                ),
                stepping=bool(self._cop.get("step_simulation_manually", False)),
                prefer_signed_target_force=bool(
                    ta.get("prefer_signed_target_force", True)
                ),
                fallback_large_velocity_rad_s=float(
                    ta.get("fallback_large_velocity_rad_s", 10.0)
                ),
                jacobian_source=str(jac.get("source", "auto")),
                numerical_epsilon=float(jac.get("numerical_epsilon", 1e-5)),
            )
        )
        self._adapter.set_logger(lambda m: self.get_logger().info(m))

        self._joint_topic = str(self._topics.get("joint_state_topic", "/joint_states"))
        self._pose_topic = str(self._topics.get("ee_pose_topic", "/ur5/ee_pose"))
        self._twist_topic = str(self._topics.get("ee_twist_topic", "/ur5/ee_twist"))
        self._jac_topic = str(self._topics.get("jacobian_topic", "/ur5/jacobian"))
        self._tau_topic = str(self._topics.get("torque_command_topic", "/ur5/torque_command"))

        self._pub_hz = float(self._cop.get("publish_rate_hz", 100.0))
        self._step_manual = bool(self._cop.get("step_simulation_manually", False))

        self._joint_pub = self.create_publisher(JointState, self._joint_topic, 10)
        self._pose_pub = self.create_publisher(PoseStamped, self._pose_topic, 10)
        self._twist_pub = self.create_publisher(TwistStamped, self._twist_topic, 10)
        self._jac_pub = self.create_publisher(Float64MultiArray, self._jac_topic, 10)
        self.create_subscription(Float64MultiArray, self._tau_topic, self._on_tau, 10)

        self._last_tau = np.zeros(6, dtype=np.float64)

        self._bringup()
        period = 1.0 / max(self._pub_hz, 1.0)
        self.create_timer(period, self._tick)

    def _bringup(self) -> None:
        try:
            self._adapter.connect()
        except Exception as exc:
            self.get_logger().error(f"CoppeliaSim connect failed: {exc}")
            raise
        try:
            sim_state = self._adapter.read_simulation_state()
            sim = getattr(self._adapter, "_sim", None)
            stopped_value = None
            if sim is not None and hasattr(sim, "simulation_stopped"):
                stopped_value = int(sim.simulation_stopped)
            if (
                sim_state is not None
                and stopped_value is not None
                and sim_state != stopped_value
            ):
                self.get_logger().info(
                    f"Stopping existing simulation before seeding (state={sim_state})"
                )
                self._adapter.stop_simulation()
                deadline = time.monotonic() + 10.0
                while time.monotonic() < deadline:
                    current_state = self._adapter.read_simulation_state()
                    if current_state == stopped_value:
                        break
                    time.sleep(0.05)
        except Exception as exc:
            self.get_logger().warn(f"Could not confirm stopped simulation before seeding: {exc}")
        self._adapter.print_scene_summary()
        seed_startup_joint_positions(
            self._adapter,
            self._adapter.config.startup_joint_positions_rad,
            log_fn=self.get_logger().info,
        )
        self._adapter.configure_force_torque_mode()
        self._adapter.start_simulation()
        self.get_logger().warn("Simulation started; bridge publishing state.")

    def _on_tau(self, msg: Float64MultiArray) -> None:
        tau = np.asarray(msg.data, dtype=np.float64).reshape(-1)
        if tau.size != 6:
            return
        if not np.all(np.isfinite(tau)):
            return
        self._last_tau = tau
        try:
            self._adapter.apply_torque(tau)
        except Exception as exc:
            self.get_logger().warn(f"apply_torque: {exc}")

    def _tick(self) -> None:
        try:
            q, qd = self._adapter.read_joint_state()
            ee_pos, ee_quat, lin, ang = self._adapter.read_ee_pose_twist()
            j_pos, j_rot = self._adapter.read_jacobian()
        except Exception as exc:
            self.get_logger().warn(f"read failed: {exc}")
            return
        if self._step_manual:
            self._adapter.step()

        now = self.get_clock().now().to_msg()
        js = JointState()
        js.header.stamp = now
        js.name = list(UR5_JOINT_NAMES)
        js.position = q.tolist()
        js.velocity = qd.tolist()
        js.effort = self._last_tau.tolist()
        self._joint_pub.publish(js)

        pose = PoseStamped()
        pose.header.stamp = now
        pose.header.frame_id = "world"
        pose.pose.position.x = float(ee_pos[0])
        pose.pose.position.y = float(ee_pos[1])
        pose.pose.position.z = float(ee_pos[2])
        pose.pose.orientation.w = float(ee_quat[0])
        pose.pose.orientation.x = float(ee_quat[1])
        pose.pose.orientation.y = float(ee_quat[2])
        pose.pose.orientation.z = float(ee_quat[3])
        self._pose_pub.publish(pose)

        tw = TwistStamped()
        tw.header.stamp = now
        tw.header.frame_id = "world"
        tw.twist.linear.x = float(lin[0])
        tw.twist.linear.y = float(lin[1])
        tw.twist.linear.z = float(lin[2])
        tw.twist.angular.x = float(ang[0])
        tw.twist.angular.y = float(ang[1])
        tw.twist.angular.z = float(ang[2])
        self._twist_pub.publish(tw)

        J6 = np.vstack([j_pos, j_rot]).reshape(-1)
        self._jac_pub.publish(float_array_from(J6, label="jacobian_6x6"))

    def destroy_node(self) -> bool:
        try:
            self._adapter.stop_simulation()
        except Exception:
            pass
        return super().destroy_node()


def main(args: object = None) -> None:
    rclpy.init(args=args)
    try:
        node = CoppeliaSimBridgeNode()
    except Exception as exc:
        print(f"[coppeliasim_bridge_node] {exc}", file=sys.stderr)
        rclpy.shutdown()
        raise SystemExit(1)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
