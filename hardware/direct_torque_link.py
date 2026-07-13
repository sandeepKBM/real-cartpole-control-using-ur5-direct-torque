"""RTDE connection specialized for PolyScope ``direct_torque()`` control.

Unlike ``hardware.link.UR5eLink`` (``servoL`` position streaming), this opens
the control socket and streams joint torques via ``ur_rtde``'s
``RTDEControlInterface.directTorque()``. PolyScope compensates gravity inside
``direct_torque()`` -- callers must not add gravity compensation on top.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np

from controller_core.kinematics_utils import rotvec_to_quat_wxyz

from .link import RTDELinkError, RTDEStateError, UR5eState, _load_rtde_classes
from .safety import ConnectionHealth, UR5eSafetyLimits


def _reshape_square_matrix(flat: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(flat, dtype=np.float64).reshape(-1)
    n = int(round(np.sqrt(arr.size)))
    if n * n != arr.size:
        raise RTDEStateError(f"{name} returned {arr.size} elements; expected a square matrix")
    return arr.reshape(n, n)


class UR5eDirectTorqueLink:
    """Receive + direct-torque control over RTDE."""

    def __init__(
        self,
        robot_ip: str,
        frequency_hz: float = 500.0,
        *,
        limits: UR5eSafetyLimits | None = None,
        control_factory: Callable[[str, float], Any] | None = None,
        receive_factory: Callable[[str, float], Any] | None = None,
    ) -> None:
        if not robot_ip:
            raise ValueError("robot_ip must be a non-empty string")
        if frequency_hz <= 0.0:
            raise ValueError("frequency_hz must be positive")
        self.robot_ip = str(robot_ip)
        self.frequency_hz = float(frequency_hz)
        self.limits = limits or UR5eSafetyLimits()
        self.limits.validate()
        self.health = ConnectionHealth(max_state_age_s=self.limits.state_stale_max_s)
        self._control_factory = control_factory
        self._receive_factory = receive_factory
        self._control: Any | None = None
        self._receive: Any | None = None

    def connect(self) -> None:
        if self._receive is None:
            if self._receive_factory is not None:
                self._receive = self._receive_factory(self.robot_ip, self.frequency_hz)
            else:
                _, receive_cls = _load_rtde_classes()
                try:
                    self._receive = receive_cls(self.robot_ip, self.frequency_hz)
                except Exception as exc:
                    raise RTDELinkError(f"Failed to open RTDE receive interface: {exc}") from exc

        if self._control is None:
            if self._control_factory is not None:
                self._control = self._control_factory(self.robot_ip, self.frequency_hz)
            else:
                control_cls, _ = _load_rtde_classes()
                try:
                    self._control = control_cls(self.robot_ip, self.frequency_hz)
                except Exception as exc:
                    raise RTDELinkError(f"Failed to open RTDE control interface: {exc}") from exc
            if not callable(getattr(self._control, "directTorque", None)):
                raise RTDELinkError(
                    "Connected RTDE control interface has no directTorque() method. "
                    "Upgrade ur_rtde to >=1.6 and ensure PolyScope >=5.23 with remote control enabled."
                )

    def read_state(self) -> UR5eState:
        if self._receive is None:
            raise RTDEStateError("read_state() called before connect()")
        host_stamp_ns = time.monotonic_ns()
        try:
            q = self._receive.getActualQ()
            qd = self._receive.getActualQd()
            tcp_pose = self._receive.getActualTCPPose()
        except Exception as exc:
            raise RTDEStateError(f"RTDE state read failed: {exc}") from exc

        q_arr = np.asarray(q, dtype=np.float64).reshape(-1)
        qd_arr = np.asarray(qd, dtype=np.float64).reshape(-1)
        pose_arr = np.asarray(tcp_pose, dtype=np.float64).reshape(-1)
        if q_arr.shape[0] != 6 or qd_arr.shape[0] != 6 or pose_arr.shape[0] != 6:
            raise RTDEStateError(
                f"unexpected shapes: q={q_arr.shape}, qd={qd_arr.shape}, tcp_pose={pose_arr.shape}"
            )
        if not np.all(np.isfinite(q_arr + qd_arr + pose_arr)):
            raise RTDEStateError("NaN/Inf in q, qd, or tcp_pose")

        robot_timestamp_s = None
        get_timestamp = getattr(self._receive, "getTimestamp", None)
        if get_timestamp is not None:
            try:
                robot_timestamp_s = float(get_timestamp())
            except Exception:
                robot_timestamp_s = None

        safety_status = None
        get_safety = getattr(self._receive, "getSafetyStatusBits", None) or getattr(
            self._receive, "getSafetyStatus", None
        )
        if get_safety is not None:
            try:
                safety_status = int(get_safety())
            except Exception:
                safety_status = None

        state = UR5eState(
            q=q_arr,
            qd=qd_arr,
            tcp_pose=pose_arr,
            host_stamp_ns=host_stamp_ns,
            robot_timestamp_s=robot_timestamp_s,
            safety_status=safety_status,
        )
        self.health.record_success(host_stamp_ns)
        return state

    def get_jacobian(self) -> np.ndarray:
        if self._control is None:
            raise RTDEStateError("get_jacobian() called before connect()")
        try:
            flat = self._control.getJacobian()
        except Exception as exc:
            raise RTDEStateError(f"getJacobian() failed: {exc}") from exc
        return _reshape_square_matrix(flat, name="getJacobian")

    def get_mass_matrix(self) -> np.ndarray:
        if self._control is None:
            raise RTDEStateError("get_mass_matrix() called before connect()")
        try:
            flat = self._control.getMassMatrix()
        except Exception as exc:
            raise RTDEStateError(f"getMassMatrix() failed: {exc}") from exc
        return _reshape_square_matrix(flat, name="getMassMatrix")

    def direct_torque(self, tau_nm: np.ndarray, *, friction_comp: bool = True) -> None:
        if self._control is None:
            raise RTDEStateError("direct_torque() called before connect()")
        tau = np.asarray(tau_nm, dtype=np.float64).reshape(6)
        if not np.all(np.isfinite(tau)):
            raise RTDEStateError("direct_torque() got non-finite torques")
        try:
            result = self._control.directTorque(tau.tolist(), friction_comp)
        except Exception as exc:
            raise RTDEStateError(f"directTorque() failed: {exc}") from exc
        if result is False:
            raise RTDEStateError("directTorque() returned false")

    def build_robot_state(
        self,
        link_state: UR5eState,
        *,
        time_s: float,
        target_x: float,
        target_x_vel: float,
    ) -> dict[str, np.ndarray | float | bool]:
        """Map RTDE telemetry into the controller_core RobotState contract."""
        tcp = np.asarray(link_state.tcp_pose, dtype=np.float64).reshape(6)
        ee_pos = tcp[:3].copy()
        ee_quat = rotvec_to_quat_wxyz(tcp[3:6])
        q = np.asarray(link_state.q, dtype=np.float64).reshape(6)
        qd = np.asarray(link_state.qd, dtype=np.float64).reshape(6)
        jacobian = self.get_jacobian()
        mass_matrix = self.get_mass_matrix()
        twist = jacobian @ qd
        return {
            "time": float(time_s),
            "q": q,
            "qd": qd,
            "ee_pos": ee_pos,
            "ee_quat": ee_quat,
            "ee_lin_vel": np.asarray(twist[:3], dtype=np.float64),
            "ee_ang_vel": np.asarray(twist[3:6], dtype=np.float64),
            "jacobian": jacobian,
            "mass_matrix": mass_matrix,
            "target_x": float(target_x),
            "target_x_vel": float(target_x_vel),
            "transport_axis_index": 0,
        }

    def safe_stop(self, reason: str) -> None:
        if self._control is not None:
            for method_name in ("stopJ", "stopj", "servoStop", "stopScript"):
                method = getattr(self._control, method_name, None)
                if method is None:
                    continue
                try:
                    if method_name in ("stopJ", "stopj"):
                        method(2.0)
                    else:
                        method()
                except Exception:
                    pass
        self.disconnect()

    def disconnect(self) -> None:
        for attr in ("_control", "_receive"):
            obj = getattr(self, attr)
            if obj is not None:
                disconnect_fn = getattr(obj, "disconnect", None)
                if disconnect_fn is not None:
                    try:
                        disconnect_fn()
                    except Exception:
                        pass
            setattr(self, attr, None)
