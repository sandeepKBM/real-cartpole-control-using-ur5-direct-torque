"""Optional non-blocking ROS 2 publishers for hardware visualization topics."""

from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class RosTopicSample:
    stamp_ns: int
    q_real: np.ndarray
    qd_real: np.ndarray
    q_desired: np.ndarray | None = None
    qd_desired: np.ndarray | None = None
    tcp_pose_real: np.ndarray | None = None
    tcp_pose_desired: np.ndarray | None = None


class AsyncRosVisualizer:
    """Publish joint-state and trajectory topics from a background thread.

    The main loop only calls ``submit``. Publishing happens on a queue so the
    RTDE/control loop never blocks on ROS 2 I/O.
    """

    def __init__(
        self,
        *,
        ros_prefix: str = "/ur5e",
        joint_names: list[str] | tuple[str, ...] | None = None,
        queue_size: int = 256,
        publish_hz: float = 50.0,
    ) -> None:
        self.ros_prefix = ros_prefix.rstrip("/")
        self.joint_names = list(joint_names) if joint_names is not None else [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        self.queue_size = int(queue_size)
        self.publish_hz = float(publish_hz)
        self._queue: queue.Queue[RosTopicSample] = queue.Queue(maxsize=self.queue_size)
        self._thread: threading.Thread | None = None
        self._stop_evt = threading.Event()
        self._dropped = 0
        self._enabled = False
        self._node = None
        self._pubs: dict[str, Any] = {}

    @property
    def dropped_samples(self) -> int:
        return int(self._dropped)

    def _topic(self, suffix: str) -> str:
        return f"{self.ros_prefix}/{suffix}".replace("//", "/")

    def start(self) -> bool:
        if self._thread is not None:
            return self._enabled
        try:
            import rclpy
            from builtin_interfaces.msg import Duration
            from builtin_interfaces.msg import Time
            from sensor_msgs.msg import JointState
            from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
        except Exception:
            return False

        self._enabled = True
        self._rclpy = rclpy  # type: ignore[attr-defined]
        self._Duration = Duration  # type: ignore[attr-defined]
        self._Time = Time  # type: ignore[attr-defined]
        self._JointState = JointState  # type: ignore[attr-defined]
        self._JointTrajectory = JointTrajectory  # type: ignore[attr-defined]
        self._JointTrajectoryPoint = JointTrajectoryPoint  # type: ignore[attr-defined]
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._run, name="ur5e_ros_viz", daemon=True)
        self._thread.start()
        return True

    def stop(self, timeout_s: float = 2.0) -> None:
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)
            self._thread = None
        if self._node is not None:
            try:
                self._node.destroy_node()
            except Exception:
                pass
            self._node = None
        if self._enabled:
            try:
                self._rclpy.shutdown()  # type: ignore[attr-defined]
            except Exception:
                pass
        self._enabled = False

    def submit(self, sample: RosTopicSample) -> bool:
        if not self._enabled:
            return False
        try:
            self._queue.put_nowait(sample)
            return True
        except queue.Full:
            try:
                _ = self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(sample)
                self._dropped += 1
                return True
            except queue.Full:
                self._dropped += 1
                return False

    def _make_stamp(self, stamp_ns: int):
        sec = int(stamp_ns // 1_000_000_000)
        nsec = int(stamp_ns % 1_000_000_000)
        return self._Time(sec=sec, nanosec=nsec)

    def _publish_joint_state(self, topic_key: str, q: np.ndarray, qd: np.ndarray, stamp_ns: int) -> None:
        msg = self._JointState()
        msg.header.stamp = self._make_stamp(stamp_ns)
        msg.header.frame_id = "base"
        msg.name = list(self.joint_names)
        msg.position = np.asarray(q, dtype=np.float64).reshape(-1).tolist()
        msg.velocity = np.asarray(qd, dtype=np.float64).reshape(-1).tolist()
        msg.effort = []
        self._pubs[topic_key].publish(msg)

    def _publish_joint_trajectory(self, topic_key: str, q: np.ndarray, qd: np.ndarray, stamp_ns: int) -> None:
        traj = self._JointTrajectory()
        traj.header.stamp = self._make_stamp(stamp_ns)
        traj.header.frame_id = "base"
        traj.joint_names = list(self.joint_names)
        pt = self._JointTrajectoryPoint()
        pt.positions = np.asarray(q, dtype=np.float64).reshape(-1).tolist()
        pt.velocities = np.asarray(qd, dtype=np.float64).reshape(-1).tolist()
        pt.time_from_start = self._Duration(sec=0, nanosec=0)
        traj.points = [pt]
        self._pubs[topic_key].publish(traj)

    def _run(self) -> None:
        self._rclpy.init(args=None)  # type: ignore[attr-defined]
        self._node = self._rclpy.create_node("ur5e_hardware_visualizer")  # type: ignore[attr-defined]
        self._pubs = {
            "joints_real": self._node.create_publisher(self._JointState, self._topic("joints/real"), 10),
            "joints_desired": self._node.create_publisher(self._JointState, self._topic("joints/desired"), 10),
            "trajectory_real": self._node.create_publisher(self._JointTrajectory, self._topic("trajectory/real"), 10),
            "trajectory_desired": self._node.create_publisher(self._JointTrajectory, self._topic("trajectory/desired"), 10),
        }
        publish_period_s = 1.0 / max(self.publish_hz, 1.0)
        while not self._stop_evt.is_set():
            try:
                sample = self._queue.get(timeout=publish_period_s)
            except queue.Empty:
                continue

            q_real = np.asarray(sample.q_real, dtype=np.float64).reshape(6)
            qd_real = np.asarray(sample.qd_real, dtype=np.float64).reshape(6)
            self._publish_joint_state("joints_real", q_real, qd_real, sample.stamp_ns)
            self._publish_joint_trajectory("trajectory_real", q_real, qd_real, sample.stamp_ns)

            if sample.q_desired is not None:
                q_des = np.asarray(sample.q_desired, dtype=np.float64).reshape(6)
            else:
                q_des = q_real
            if sample.qd_desired is not None:
                qd_des = np.asarray(sample.qd_desired, dtype=np.float64).reshape(6)
            else:
                qd_des = qd_real
            self._publish_joint_state("joints_desired", q_des, qd_des, sample.stamp_ns)
            self._publish_joint_trajectory("trajectory_desired", q_des, qd_des, sample.stamp_ns)
            self._rclpy.spin_once(self._node, timeout_sec=0.0)  # type: ignore[attr-defined]
            time.sleep(0.0)


@dataclass
class GuardrailStatusSample:
    stamp_ns: int
    state: str
    frame: str
    margin_m: float
    message: str
    boundary_name: str | None = None
    violated_boundary_names: list[str] | None = None
    near_boundary_names: list[str] | None = None
    decision: dict[str, Any] | None = None


class AsyncGuardrailPublisher:
    """Publish guardrail config and status on a background ROS 2 thread."""

    def __init__(
        self,
        *,
        ros_prefix: str = "/ur5e",
        queue_size: int = 256,
        publish_hz: float = 20.0,
    ) -> None:
        self.ros_prefix = ros_prefix.rstrip("/")
        self.queue_size = int(queue_size)
        self.publish_hz = float(publish_hz)
        self._queue: queue.Queue[GuardrailStatusSample] = queue.Queue(maxsize=self.queue_size)
        self._thread: threading.Thread | None = None
        self._stop_evt = threading.Event()
        self._dropped = 0
        self._enabled = False
        self._node = None
        self._pubs: dict[str, Any] = {}
        self._config_json: str | None = None
        self._config_published = False

    @property
    def dropped_samples(self) -> int:
        return int(self._dropped)

    def _topic(self, suffix: str) -> str:
        return f"{self.ros_prefix}/{suffix}".replace("//", "/")

    def set_config(self, config: dict[str, Any] | str) -> None:
        self._config_json = config if isinstance(config, str) else json.dumps(config, separators=(",", ":"))
        self._config_published = False

    def start(self) -> bool:
        if self._thread is not None:
            return self._enabled
        try:
            import rclpy
            from std_msgs.msg import String
        except Exception:
            return False

        self._enabled = True
        self._rclpy = rclpy  # type: ignore[attr-defined]
        self._String = String  # type: ignore[attr-defined]
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._run, name="ur5e_guardrail_ros", daemon=True)
        self._thread.start()
        return True

    def stop(self, timeout_s: float = 2.0) -> None:
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)
            self._thread = None
        if self._node is not None:
            try:
                self._node.destroy_node()
            except Exception:
                pass
            self._node = None
        if self._enabled:
            try:
                self._rclpy.shutdown()  # type: ignore[attr-defined]
            except Exception:
                pass
        self._enabled = False

    def submit(self, sample: GuardrailStatusSample) -> bool:
        if not self._enabled:
            return False
        try:
            self._queue.put_nowait(sample)
            return True
        except queue.Full:
            try:
                _ = self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(sample)
                self._dropped += 1
                return True
            except queue.Full:
                self._dropped += 1
                return False

    def _publish_string(self, topic_key: str, payload: str, stamp_ns: int | None = None) -> None:
        msg = self._String()
        msg.data = payload
        self._pubs[topic_key].publish(msg)

    def _run(self) -> None:
        self._rclpy.init(args=None)  # type: ignore[attr-defined]
        self._node = self._rclpy.create_node("ur5e_workspace_guardrails")  # type: ignore[attr-defined]
        self._pubs = {
            "guardrails": self._node.create_publisher(self._String, self._topic("workspace_guardrails"), 10),
            "status": self._node.create_publisher(self._String, self._topic("workspace_guardrail_status"), 10),
        }
        publish_period_s = 1.0 / max(self.publish_hz, 1.0)
        while not self._stop_evt.is_set():
            if self._config_json is not None and not self._config_published:
                self._publish_string("guardrails", self._config_json)
                self._config_published = True
            try:
                sample = self._queue.get(timeout=publish_period_s)
            except queue.Empty:
                continue
            payload = json.dumps(
                {
                    "stamp_ns": int(sample.stamp_ns),
                    "state": sample.state,
                    "frame": sample.frame,
                    "margin_m": float(sample.margin_m),
                    "message": sample.message,
                    "boundary_name": sample.boundary_name,
                    "violated_boundary_names": sample.violated_boundary_names or [],
                    "near_boundary_names": sample.near_boundary_names or [],
                    "decision": sample.decision or {},
                    "diagnostic_only": True,
                },
                separators=(",", ":"),
            )
            self._publish_string("status", payload)
            self._rclpy.spin_once(self._node, timeout_sec=0.0)  # type: ignore[attr-defined]
            time.sleep(0.0)
