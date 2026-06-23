#!/usr/bin/env python3
"""ROS 2 node: UR5 X motion control with selectable torque/controller families."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, Float64MultiArray, String

from controller_core.filters import TorqueCommandFilter
from controller_core.logging_utils import json_dumps_safe
from controller_core.safety import ImpedanceSafetyConfig, ImpedanceSafetyMonitor
from controller_core.kinematics_utils import orientation_error_vec_wxyz, quat_to_rotmat
from controller_core.x_axis_cartesian_impedance import (
    CartesianImpedanceConfig,
    JOINT_NAME_ORDER,
    XAxisCartesianImpedanceController,
)

from .config_loader import load_yaml_config
from .jacobian_provider import JacobianProvider
from .messages import (
    UR5_JOINT_NAMES,
    float_array_from,
    joint_state_to_arrays,
    jacobian_from_multiarray,
    pose_to_pos_quat,
    twist_to_lin_ang,
)

try:
    from simulation.controller import differential_ik_xz_transport_controller
except Exception:  # pragma: no cover - import path is validated in live smoke tests
    differential_ik_xz_transport_controller = None  # type: ignore[assignment]


LEGACY_XZ_TRANSPORT_FAMILY = "legacy_xz_transport_pd"


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


def _joint_pd_torque(
    q: np.ndarray,
    qd: np.ndarray,
    q_ref: np.ndarray,
    *,
    kp: float,
    kd: float,
    tau_limit: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    q = np.asarray(q, dtype=np.float64).reshape(6)
    qd = np.asarray(qd, dtype=np.float64).reshape(6)
    q_ref = np.asarray(q_ref, dtype=np.float64).reshape(6)
    tau_limit = np.asarray(tau_limit, dtype=np.float64).reshape(6)
    raw = float(kp) * (q_ref - q) - float(kd) * qd
    clipped = np.clip(raw, -tau_limit, +tau_limit)
    return clipped, {
        "tau_raw": raw.tolist(),
        "tau_clipped": clipped.tolist(),
        "tau_saturated": (np.abs(raw - clipped) > 1e-9).astype(np.float64).tolist(),
        "max_abs_tau_raw_nm": float(np.max(np.abs(raw))),
        "max_abs_tau_cmd_nm": float(np.max(np.abs(clipped))),
    }


class ControllerNode(Node):
    def __init__(self) -> None:
        super().__init__("ur5_x_axis_controller")
        self.declare_parameter("config_path", "")
        cfg_path = self.get_parameter("config_path").value or _default_config_path()
        self._yaml = load_yaml_config(cfg_path)
        self._ctrl_y = self._yaml.get("controller", {}) or {}
        self._safe_y = self._yaml.get("safety", {}) or {}
        self._log_y = self._yaml.get("logging", {}) or {}
        self._topics = self._yaml.get("topics", {}) or {}

        self._joint_state_topic = str(
            self._topics.get("joint_state_topic", "/joint_states")
        )
        self._ee_pose_topic = str(self._topics.get("ee_pose_topic", "/ur5/ee_pose"))
        self._ee_twist_topic = str(self._topics.get("ee_twist_topic", "/ur5/ee_twist"))
        self._jac_topic = str(self._topics.get("jacobian_topic", "/ur5/jacobian"))
        self._target_topic = str(self._topics.get("target_x_topic", "/target_x"))
        self._torque_topic = str(
            self._topics.get("torque_command_topic", "/ur5/torque_command")
        )
        self._debug_topic = str(
            self._topics.get("controller_debug_topic", "/ur5/controller_debug")
        )
        self._safety_topic = str(
            self._topics.get("safety_status_topic", "/ur5/safety_status")
        )
        self._grav_topic = str(
            self._topics.get("gravity_torque_topic", "/ur5/gravity_torque")
        )

        self._rate_hz = float(self._ctrl_y.get("control_rate_hz", 100.0))
        self._step_max = float(self._ctrl_y.get("target_x_step_max_m", 0.02))
        self._vx_cap = float(self._ctrl_y.get("target_x_velocity_limit_mps", 0.02))
        self._use_grav = bool(self._ctrl_y.get("use_gravity_compensation", False))
        self._controller_family = str(self._ctrl_y.get("family", "") or "").strip().lower()
        self._legacy_joint_kp = float(self._ctrl_y.get("legacy_joint_kp", 45.0))
        self._legacy_joint_kd = float(self._ctrl_y.get("legacy_joint_kd", 9.0))
        self._legacy_tau_scale = float(self._ctrl_y.get("legacy_torque_scale", 1.0))
        self._legacy_x_sign = float(self._ctrl_y.get("legacy_x_sign", -1.0))

        self._imp_cfg = CartesianImpedanceConfig.from_controller_yaml_section(self._ctrl_y)
        self._imp = XAxisCartesianImpedanceController(self._imp_cfg)

        rate_dict = self._ctrl_y.get("torque_rate_limit_nm_per_sec", {}) or {}
        rate_list = [float(rate_dict[n]) for n in JOINT_NAME_ORDER]
        self._filt = TorqueCommandFilter(
            num_joints=6,
            lowpass_alpha=float(self._ctrl_y.get("torque_lowpass_alpha", 0.15)),
            rate_limit_nm_per_sec=np.asarray(rate_list, dtype=np.float64),
        )

        self._safe = ImpedanceSafetyMonitor(
            ImpedanceSafetyConfig(
                max_abs_y_drift_m=float(self._safe_y.get("max_abs_y_drift_m", 0.03)),
                max_abs_z_drift_m=float(self._safe_y.get("max_abs_z_drift_m", 0.03)),
                max_orientation_error_rad=float(
                    self._safe_y.get("max_orientation_error_rad", 0.25)
                ),
                max_joint_velocity_radps=float(
                    self._safe_y.get("max_joint_velocity_radps", 1.5)
                ),
                max_x_error_growth_steps=int(
                    self._safe_y.get("max_x_error_growth_steps", 100)
                ),
                emergency_stop_on_nan=bool(
                    self._safe_y.get("emergency_stop_on_nan", True)
                ),
                emergency_stop_on_joint_limit=bool(
                    self._safe_y.get("emergency_stop_on_joint_limit", True)
                ),
            )
        )

        self._q = self._qd = self._ee_pos = self._ee_quat = None
        self._ee_lin = self._ee_ang = None
        self._grav: Optional[np.ndarray] = None
        self._raw_target_x: Optional[float] = None
        self._cmd_target_x: Optional[float] = None
        self._prev_ctrl_time: Optional[float] = None
        self._initialized = False
        self._estop = False
        self._jac = JacobianProvider(num_joints=6)
        self._legacy_ctrl_prev: Optional[np.ndarray] = None
        self._legacy_q_rest: Optional[np.ndarray] = None
        self._legacy_target_rot: Optional[np.ndarray] = None
        self._legacy_target_quat: Optional[np.ndarray] = None
        self._legacy_x_hold: Optional[float] = None
        self._legacy_y_hold: Optional[float] = None
        self._legacy_z_hold: Optional[float] = None
        self._legacy_pan_target: Optional[float] = None

        self._trace_path = str(self._log_y.get("trace_jsonl_path", "") or "")
        self._trace_fp: Any = None
        if self._trace_path:
            Path(self._trace_path).parent.mkdir(parents=True, exist_ok=True)
            self._trace_fp = open(self._trace_path, "w", encoding="utf-8")

        self._debug_hz = float(self._log_y.get("log_controller_debug_hz", 0.0))
        self._debug_counter = 0

        self.create_subscription(JointState, self._joint_state_topic, self._on_js, 10)
        self.create_subscription(PoseStamped, self._ee_pose_topic, self._on_pose, 10)
        self.create_subscription(TwistStamped, self._ee_twist_topic, self._on_twist, 10)
        self.create_subscription(Float64MultiArray, self._jac_topic, self._on_jac, 10)
        self.create_subscription(Float64, self._target_topic, self._on_target, 10)
        self.create_subscription(Float64MultiArray, self._grav_topic, self._on_grav, 10)

        self._tau_pub = self.create_publisher(Float64MultiArray, self._torque_topic, 10)
        self._dbg_pub = self.create_publisher(String, self._debug_topic, 10)
        self._safe_pub = self.create_publisher(String, self._safety_topic, 10)

        period = 1.0 / max(self._rate_hz, 1.0)
        self.create_timer(period, self._tick)
        self.get_logger().info(f"Loaded config: {cfg_path}")
        self.get_logger().info(
            f"Controller family: {self._controller_family or 'cartesian_impedance'}"
        )
        self.get_logger().info(f"Control @ {self._rate_hz} Hz; topics from YAML.")

    def destroy_node(self) -> bool:
        if self._trace_fp is not None:
            self._trace_fp.close()
            self._trace_fp = None
        return super().destroy_node()

    def _on_js(self, msg: JointState) -> None:
        try:
            self._q, self._qd = joint_state_to_arrays(msg, UR5_JOINT_NAMES)
        except (KeyError, ValueError) as exc:
            self.get_logger().warn(f"joint_states: {exc}")

    def _on_pose(self, msg: PoseStamped) -> None:
        self._ee_pos, self._ee_quat = pose_to_pos_quat(msg)

    def _on_twist(self, msg: TwistStamped) -> None:
        self._ee_lin, self._ee_ang = twist_to_lin_ang(msg)

    def _on_jac(self, msg: Float64MultiArray) -> None:
        try:
            J = jacobian_from_multiarray(msg, num_joints=6)
            if J.shape[0] == 3:
                J = np.vstack([J, np.zeros((3, 6), dtype=np.float64)])
            now_ns = self.get_clock().now().nanoseconds
            self._jac.update(J.reshape(-1).tolist(), stamp_ns=now_ns)
        except ValueError as exc:
            self.get_logger().warn(f"jacobian: {exc}")

    def _on_target(self, msg: Float64) -> None:
        self._raw_target_x = float(msg.data)

    def _on_grav(self, msg: Float64MultiArray) -> None:
        a = np.asarray(msg.data, dtype=np.float64).reshape(-1)
        if a.size == 6:
            self._grav = a

    def _limit_target_x(self, t_now: float) -> float:
        if self._raw_target_x is None or self._cmd_target_x is None:
            return float(self._cmd_target_x or 0.0)
        dt = 0.02
        if self._prev_ctrl_time is not None:
            dt = max(float(t_now - self._prev_ctrl_time), 1e-4)
        raw = float(self._raw_target_x)
        lim = float(self._cmd_target_x)
        delta = raw - lim
        step = float(np.clip(delta, -self._step_max, self._step_max))
        max_dr = self._vx_cap * dt
        if abs(step) > max_dr:
            step = float(np.sign(step) * max_dr)
        return lim + step

    def _tick(self) -> None:
        if self._estop:
            self._tau_pub.publish(float_array_from(np.zeros(6), label="tau_zero"))
            return
        if any(
            x is None
            for x in (
                self._q,
                self._qd,
                self._ee_pos,
                self._ee_quat,
                self._ee_lin,
                self._ee_ang,
            )
        ):
            return
        J = self._jac.get_matrix6()
        if J is None:
            return

        t_now = self.get_clock().now().nanoseconds * 1e-9
        if not self._initialized:
            self._raw_target_x = float(self._ee_pos[0])
            self._cmd_target_x = float(self._ee_pos[0])
            if self._controller_family == LEGACY_XZ_TRANSPORT_FAMILY:
                self._legacy_ctrl_prev = np.asarray(self._q, dtype=np.float64).copy()
                self._legacy_q_rest = np.asarray(self._q, dtype=np.float64).copy()
                self._legacy_target_quat = np.asarray(self._ee_quat, dtype=np.float64).copy()
                self._legacy_target_rot = quat_to_rotmat(self._ee_quat)
                self._legacy_x_hold = float(self._ee_pos[0])
                self._legacy_y_hold = float(self._ee_pos[1])
                self._legacy_z_hold = float(self._ee_pos[2])
                self._legacy_pan_target = float(self._q[0])
                self._safe.set_initial_yz(float(self._ee_pos[1]), float(self._ee_pos[2]))
                self.get_logger().info(
                    "First valid state: reset legacy differential-IK transport hold."
                )
            else:
                self._imp.reset_from_state(
                    {
                        "time": t_now,
                        "q": self._q,
                        "qd": self._qd,
                        "ee_pos": self._ee_pos,
                        "ee_quat": self._ee_quat,
                        "ee_lin_vel": self._ee_lin,
                        "ee_ang_vel": self._ee_ang,
                        "target_x": float(self._ee_pos[0]),
                        "jacobian": J,
                    }
                )
                self._safe.set_initial_yz(float(self._ee_pos[1]), float(self._ee_pos[2]))
                self.get_logger().info(
                    "First valid state: reset impedance (hold Y,Z,orient,posture)."
                )
            self._initialized = True

        self._cmd_target_x = self._limit_target_x(t_now)
        self._prev_ctrl_time = t_now

        dt = 1.0 / max(self._rate_hz, 1.0)
        if self._controller_family == LEGACY_XZ_TRANSPORT_FAMILY:
            if differential_ik_xz_transport_controller is None:
                self.get_logger().error(
                    "simulation.controller import failed; legacy transport family unavailable"
                )
                self._estop = True
                self._tau_pub.publish(float_array_from(np.zeros(6), label="tau_zero"))
                return
            assert self._legacy_ctrl_prev is not None
            assert self._legacy_q_rest is not None
            assert self._legacy_target_rot is not None
            assert self._legacy_target_quat is not None
            assert self._legacy_x_hold is not None
            assert self._legacy_y_hold is not None
            assert self._legacy_z_hold is not None
            assert self._legacy_pan_target is not None

            legacy_x_target = float(
                self._legacy_x_hold
                + self._legacy_x_sign * (float(self._cmd_target_x) - self._legacy_x_hold)
            )
            q_des = differential_ik_xz_transport_controller(
                q=np.asarray(self._q, dtype=np.float64),
                ctrl_prev=np.asarray(self._legacy_ctrl_prev, dtype=np.float64),
                ctrl_lower=np.asarray(self._safe.cfg.q_lower, dtype=np.float64),
                ctrl_upper=np.asarray(self._safe.cfg.q_upper, dtype=np.float64),
                qvel=np.asarray(self._qd, dtype=np.float64),
                tool_pos=np.asarray(self._ee_pos, dtype=np.float64),
                x_target=legacy_x_target,
                z_target=float(self._legacy_z_hold),
                tool_jacobian_pos=np.asarray(J[0:3, :], dtype=np.float64),
                tool_rot=quat_to_rotmat(self._ee_quat),
                target_tool_rot=np.asarray(self._legacy_target_rot, dtype=np.float64),
                tool_jacobian_rot=np.asarray(J[3:6, :], dtype=np.float64),
                pan_target=float(self._legacy_pan_target),
                posture_target=np.asarray(self._legacy_q_rest, dtype=np.float64),
            )
            self._legacy_ctrl_prev = np.asarray(q_des, dtype=np.float64).copy()
            tau_limit = np.asarray(self._imp_cfg.tau_max_nm, dtype=np.float64) * float(
                np.clip(self._legacy_tau_scale, 0.0, 1.0)
            )
            tau_pre_filter, pd_diag = _joint_pd_torque(
                q=self._q,
                qd=self._qd,
                q_ref=q_des,
                kp=self._legacy_joint_kp,
                kd=self._legacy_joint_kd,
                tau_limit=tau_limit,
            )
            if self._use_grav and self._grav is not None:
                tau_pre_filter = np.asarray(tau_pre_filter, dtype=np.float64) + self._grav
            tau_cmd = self._filt.apply(tau_pre_filter, dt)
            x_error = float(self._cmd_target_x - float(self._ee_pos[0]))
            y_error = float(self._ee_pos[1] - self._legacy_y_hold)
            z_error = float(float(self._ee_pos[2]) - float(self._legacy_z_hold))
            orientation_error_vec = orientation_error_vec_wxyz(self._legacy_target_quat, self._ee_quat)
            orientation_error_norm = float(np.linalg.norm(orientation_error_vec))
            jacobian_cond = float(np.linalg.cond(J))
            safe_st = self._safe.check(
                state={
                    "time": t_now,
                    "q": self._q,
                    "qd": self._qd,
                    "ee_pos": self._ee_pos,
                    "ee_quat": self._ee_quat,
                },
                x_error=x_error,
                orientation_error_norm=orientation_error_norm,
            )
            self._safe_pub.publish(
                String(data=json.dumps({"ok": safe_st.ok, "reason": safe_st.reason}))
            )
            if not safe_st.ok:
                self.get_logger().error(f"SAFETY: {safe_st.reason} -> E-STOP latch")
                self._estop = True
                self._filt.reset()
                self._tau_pub.publish(float_array_from(np.zeros(6), label="tau_zero"))
                return
            if not np.all(np.isfinite(tau_cmd)):
                self.get_logger().error("NaN in tau_cmd -> zero")
                self._tau_pub.publish(float_array_from(np.zeros(6), label="tau_zero"))
                return
            self._tau_pub.publish(float_array_from(tau_cmd, label="tau"))

            if self._trace_fp is not None:
                row = {
                    "time": t_now,
                    "q": self._q,
                    "qd": self._qd,
                    "ee_pos": self._ee_pos,
                    "ee_quat": self._ee_quat,
                    "ee_lin_vel": self._ee_lin,
                    "ee_ang_vel": self._ee_ang,
                    "target_x": float(self._cmd_target_x),
                    "x_error": x_error,
                    "y_error": y_error,
                    "z_error": z_error,
                    "orientation_error_norm": orientation_error_norm,
                    "q_des": q_des,
                    "tau_raw": tau_pre_filter,
                    "tau_final": tau_cmd,
                    "tau_saturated": pd_diag["tau_saturated"],
                    "jacobian_condition_number": jacobian_cond,
                    "safety_ok": safe_st.ok,
                    "controller_family": self._controller_family,
                }
                self._trace_fp.write(json_dumps_safe(row) + "\n")

            if self._debug_hz > 0:
                self._debug_counter += 1
                if self._debug_counter % max(1, int(round(self._rate_hz / self._debug_hz))) == 0:
                    dbg = {
                        "time": t_now,
                        "controller_family": self._controller_family,
                        "x_error": x_error,
                        "q_des": np.asarray(q_des, dtype=np.float64).tolist(),
                        "tau_cmd": tau_cmd.tolist(),
                        "pd_diag": pd_diag,
                        "cond": jacobian_cond,
                    }
                    self._dbg_pub.publish(String(data=json_dumps_safe(dbg)))
            return

        state: dict[str, Any] = {
            "time": t_now,
            "q": self._q,
            "qd": self._qd,
            "ee_pos": self._ee_pos,
            "ee_quat": self._ee_quat,
            "ee_lin_vel": self._ee_lin,
            "ee_ang_vel": self._ee_ang,
            "target_x": float(self._cmd_target_x),
            "jacobian": J,
        }
        if self._use_grav and self._grav is not None:
            state["gravity_torque"] = self._grav

        out = self._imp.compute(state)
        tau_raw = out.tau
        tau_cmd = self._filt.apply(tau_raw, dt)

        st_check = state  # same dict satisfies RobotState keys for safety
        safe_st = self._safe.check(
            st_check,
            x_error=out.x_error,
            orientation_error_norm=out.orientation_error_norm,
        )
        self._safe_pub.publish(String(data=json.dumps({"ok": safe_st.ok, "reason": safe_st.reason})))
        if not safe_st.ok:
            self.get_logger().error(f"SAFETY: {safe_st.reason} -> E-STOP latch")
            self._estop = True
            self._filt.reset()
            self._tau_pub.publish(float_array_from(np.zeros(6), label="tau_zero"))
            return

        if not np.all(np.isfinite(tau_cmd)):
            self.get_logger().error("NaN in tau_cmd -> zero")
            self._tau_pub.publish(float_array_from(np.zeros(6), label="tau_zero"))
            return

        self._tau_pub.publish(float_array_from(tau_cmd, label="tau"))

        if self._trace_fp is not None:
            row = {
                "time": t_now,
                "q": self._q,
                "qd": self._qd,
                "ee_pos": self._ee_pos,
                "ee_quat": self._ee_quat,
                "ee_lin_vel": self._ee_lin,
                "ee_ang_vel": self._ee_ang,
                "target_x": float(self._cmd_target_x),
                "x_error": out.x_error,
                "y_error": out.y_error,
                "z_error": out.z_error,
                "orientation_error_norm": out.orientation_error_norm,
                "tau_task": out.tau_task,
                "tau_posture": out.tau_posture,
                "tau_damping": out.tau_damping,
                "tau_final": tau_cmd,
                "tau_saturated": out.tau_saturated,
                "jacobian_condition_number": out.jacobian_cond,
                "safety_ok": safe_st.ok,
                "controller_family": self._controller_family,
            }
            self._trace_fp.write(json_dumps_safe(row) + "\n")

        if self._debug_hz > 0:
            self._debug_counter += 1
            if self._debug_counter % max(1, int(round(self._rate_hz / self._debug_hz))) == 0:
                dbg = {
                    "time": t_now,
                    "x_error": out.x_error,
                    "wrench": out.wrench.tolist(),
                    "tau_cmd": tau_cmd.tolist(),
                    "cond": out.jacobian_cond,
                    "singular_scale": out.singular_scale,
                    "controller_family": self._controller_family,
                }
                self._dbg_pub.publish(String(data=json_dumps_safe(dbg)))


def main(args: Any = None) -> None:
    rclpy.init(args=args)
    node = ControllerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
