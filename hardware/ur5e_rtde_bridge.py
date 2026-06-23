"""Thin, safe-by-default wrapper around the UR RTDE control / receive APIs."""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .safety_limits import (
    MotionCommandGuard,
    SafetyDecision,
    UR5eSafetyLimits,
    UR5eStateGuard,
    check_joint_state,
    check_tcp_pose,
)


def _load_rtde_classes() -> tuple[type[Any], type[Any]]:
    try:
        from rtde_control import RTDEControlInterface
        from rtde_receive import RTDEReceiveInterface
    except Exception as exc:  # pragma: no cover - exercised in environments without RTDE libs
        raise RuntimeError(
            "RTDE Python bindings are not available. Install rtde_control / rtde_receive "
            "before attempting a live UR5e connection."
        ) from exc
    return RTDEControlInterface, RTDEReceiveInterface


def _maybe_call(obj: Any, name: str, *args: Any, default: Any = None, **kwargs: Any) -> Any:
    fn = getattr(obj, name, None)
    if fn is None:
        return default
    try:
        return fn(*args, **kwargs)
    except TypeError:
        return default


def _as_vec(value: Any, length: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.shape[0] != length:
        raise ValueError(f"{name} must have length {length}; got shape {np.asarray(value).shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN/Inf")
    return arr


CONTROL_METHOD_CANDIDATES: tuple[str, ...] = (
    "initPeriod",
    "waitPeriod",
    "servoJ",
    "speedJ",
    "stopJ",
    "servoStop",
    "stopScript",
    "setJointTorque",
    "setJointTorques",
    "setJointTargetTorque",
    "set_torque",
    "setTorques",
)

RECEIVE_METHOD_CANDIDATES: tuple[str, ...] = (
    "getActualQ",
    "getActualQd",
    "getActualTCPPose",
    "getTimestamp",
    "getRobotMode",
    "getSafetyStatus",
)


@dataclass
class UR5eState:
    q: np.ndarray
    qd: np.ndarray
    tcp_pose: np.ndarray | None
    host_stamp_ns: int
    robot_timestamp_s: float | None
    robot_mode: int | None
    safety_status: int | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "q": self.q,
            "qd": self.qd,
            "tcp_pose": self.tcp_pose,
            "host_stamp_ns": int(self.host_stamp_ns),
            "robot_timestamp_s": self.robot_timestamp_s,
            "robot_mode": self.robot_mode,
            "safety_status": self.safety_status,
        }


class UR5eRTDEBridge:
    """Safe wrapper around RTDE receive/control connections.

    The class does not send motion by default. Scripts must explicitly call the
    motion helpers after they have validated the current state and confirmed
    the motion opt-in flags.
    """

    def __init__(
        self,
        robot_ip: str,
        frequency: float,
        *,
        limits: UR5eSafetyLimits | None = None,
        control_factory: Callable[..., Any] | None = None,
        receive_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.robot_ip = str(robot_ip)
        self.frequency = float(frequency)
        self.period_s = 1.0 / max(self.frequency, 1.0)
        self.limits = limits if limits is not None else UR5eSafetyLimits()
        self.limits.validate()
        self._control_factory = control_factory
        self._receive_factory = receive_factory
        self._control: Any | None = None
        self._receive: Any | None = None
        self.state_guard = UR5eStateGuard(self.limits)
        self.command_guard = MotionCommandGuard(self.limits)
        self.last_stop_reason: str | None = None

    @property
    def control(self) -> Any | None:
        return self._control

    @property
    def receive(self) -> Any | None:
        return self._receive

    def _make_receive(self) -> Any:
        if self._receive_factory is not None:
            return self._receive_factory(self.robot_ip, self.frequency)
        _, receive_cls = _load_rtde_classes()
        return receive_cls(self.robot_ip, self.frequency)

    def _make_control(self) -> Any:
        if self._control_factory is not None:
            return self._control_factory(self.robot_ip, self.frequency)
        control_cls, _ = _load_rtde_classes()
        return control_cls(self.robot_ip, self.frequency)

    def connect_receive_only(self) -> Any:
        if self._receive is None:
            self._receive = self._make_receive()
        return self._receive

    def connect_control(self) -> Any:
        if self._control is None:
            self._control = self._make_control()
        return self._control

    def disconnect(self) -> None:
        for obj in (self._control, self._receive):
            if obj is None:
                continue
            try:
                disconnect = getattr(obj, "disconnect", None)
                if disconnect is not None:
                    disconnect()
            except Exception:
                pass
        self._control = None
        self._receive = None

    def safe_stop(self, reason: str) -> list[str]:
        """Best-effort stop sequence.

        This is deliberately defensive: every step is attempted independently so
        one failure does not suppress later stop commands.
        """

        errors: list[str] = []
        self.last_stop_reason = str(reason)
        control = self._control
        if control is not None:
            for method_name, args in (
                ("servoStop", ()),
                ("stopJ", (2.0,)),
                ("stopScript", ()),
            ):
                fn = getattr(control, method_name, None)
                if fn is None:
                    continue
                try:
                    fn(*args)
                except Exception as exc:  # pragma: no cover - defensive best-effort cleanup
                    errors.append(f"{method_name}: {type(exc).__name__}: {exc}")
        self.disconnect()
        return errors

    def supports_direct_torque(self) -> bool:
        """Return ``True`` only if a distinct torque API is visible.

        The RTDE Python control interface used in this repo does not expose a
        real direct-torque motion API, so this will normally be ``False``.
        """

        control = self._control
        if control is None:
            return False
        for name in ("setJointTorque", "setJointTorques", "setJointTargetTorque", "set_torque", "setTorques"):
            if hasattr(control, name):
                return True
        return False

    def available_control_methods(self) -> dict[str, bool]:
        control = self._control
        return {
            name: bool(control is not None and hasattr(control, name))
            for name in CONTROL_METHOD_CANDIDATES
        }

    def available_receive_methods(self) -> dict[str, bool]:
        receive = self._receive
        return {
            name: bool(receive is not None and hasattr(receive, name))
            for name in RECEIVE_METHOD_CANDIDATES
        }

    def probe_direct_torque_capability(self) -> dict[str, Any]:
        control = self._control
        method_map = self.available_control_methods()
        support = self.supports_direct_torque()
        return {
            "control_connected": control is not None,
            "supports_direct_torque": bool(support),
            "direct_torque_method_map": method_map,
            "direct_torque_method_names": [
                name for name, present in method_map.items() if present and name.startswith("set")
            ],
            "blocked_by_default": True,
            "reason": (
                "direct torque API unavailable on the connected RTDE control interface"
                if not support
                else "direct torque capability detected; explicit nonzero execution remains blocked by default"
            ),
        }

    def send_servoj(
        self,
        q: np.ndarray,
        *,
        gain: float,
        lookahead_time: float,
        velocity: float,
        acceleration: float,
        period_s: float | None = None,
    ) -> None:
        self._call_servoj(
            q,
            gain=gain,
            lookahead_time=lookahead_time,
            velocity=velocity,
            acceleration=acceleration,
            period_s=period_s,
        )

    def send_speedj(self, qd: np.ndarray, *, acceleration: float, duration_s: float) -> None:
        self._call_speedj(qd, acceleration=acceleration, duration_s=duration_s)

    def send_joint_torque(
        self,
        tau: np.ndarray,
        *,
        allow_nonzero: bool = False,
        zero_only: bool = True,
    ) -> dict[str, Any]:
        tau_arr = _as_vec(tau, 6, "tau")
        result: dict[str, Any] = {
            "requested_tau_nm": tau_arr.tolist(),
            "allow_nonzero": bool(allow_nonzero),
            "zero_only": bool(zero_only),
            "control_connected": self._control is not None,
            "supports_direct_torque": bool(self.supports_direct_torque()),
            "available_control_methods": self.available_control_methods(),
            "accepted": False,
            "blocked": True,
        }
        if not np.all(np.isfinite(tau_arr)):
            result["reason"] = "non-finite torque command"
            return result
        if zero_only and np.any(np.abs(tau_arr) > 1e-12):
            result["reason"] = "nonzero direct torque is blocked by default"
            return result
        if not allow_nonzero and np.any(np.abs(tau_arr) > 1e-12):
            result["reason"] = "explicit nonzero direct torque opt-in is disabled"
            return result
        if not self.supports_direct_torque():
            result["reason"] = "direct torque API unavailable on the connected RTDE control interface"
            return result

        control = self._control
        if control is None:
            result["reason"] = "connect_control() must be called before direct torque"
            return result

        tau_list = tau_arr.tolist()
        fn_names = [
            "setJointTorque",
            "setJointTorques",
            "setJointTargetTorque",
            "set_torque",
            "setTorques",
        ]
        attempts = [(tau_list,), (tau_list,)]
        last_exc: Exception | None = None
        for name in fn_names:
            fn = getattr(control, name, None)
            if fn is None:
                continue
            for args in attempts:
                try:
                    fn(*args)
                    result["accepted"] = True
                    result["blocked"] = False
                    result["reason"] = f"direct torque sent via {name}"
                    result["method"] = name
                    return result
                except TypeError as exc:
                    last_exc = exc
                    continue
                except Exception as exc:  # pragma: no cover - defensive future backend hook
                    last_exc = exc
                    continue
        result["reason"] = (
            "direct torque API detected but the connected control interface does not accept "
            "the candidate method signatures"
        )
        if last_exc is not None:
            result["error"] = f"{type(last_exc).__name__}: {last_exc}"
        return result

    def begin_period(self) -> Any:
        control = self._control
        if control is None:
            raise RuntimeError("connect_control() must be called before begin_period()")
        init_period = getattr(control, "initPeriod", None)
        if init_period is None:
            return time.monotonic()
        return init_period()

    def wait_period(self, period_token: Any) -> None:
        control = self._control
        if control is None:
            raise RuntimeError("connect_control() must be called before wait_period()")
        wait_period = getattr(control, "waitPeriod", None)
        if wait_period is None:
            time.sleep(self.period_s)
            return
        wait_period(period_token)

    def read_state(self) -> UR5eState:
        receive = self._receive
        if receive is None:
            raise RuntimeError("connect_receive_only() must be called before read_state()")
        host_stamp_ns = time.monotonic_ns()
        q = _as_vec(_maybe_call(receive, "getActualQ"), 6, "actual_q")
        qd = _as_vec(_maybe_call(receive, "getActualQd"), 6, "actual_qd")
        tcp_pose_raw = _maybe_call(receive, "getActualTCPPose", default=None)
        tcp_pose = None if tcp_pose_raw is None else _as_vec(tcp_pose_raw, 6, "actual_TCP_pose")
        robot_timestamp_s = _maybe_call(receive, "getTimestamp", default=None)
        robot_mode = _maybe_call(receive, "getRobotMode", default=None)
        safety_status = _maybe_call(receive, "getSafetyStatus", default=None)
        return UR5eState(
            q=q,
            qd=qd,
            tcp_pose=tcp_pose,
            host_stamp_ns=int(host_stamp_ns),
            robot_timestamp_s=None if robot_timestamp_s is None else float(robot_timestamp_s),
            robot_mode=None if robot_mode is None else int(robot_mode),
            safety_status=None if safety_status is None else int(safety_status),
        )

    def get_joint_state(self) -> dict[str, Any]:
        state = self.read_state()
        return {
            "q": state.q,
            "qd": state.qd,
            "host_stamp_ns": state.host_stamp_ns,
            "robot_timestamp_s": state.robot_timestamp_s,
            "robot_mode": state.robot_mode,
            "safety_status": state.safety_status,
        }

    def get_tcp_pose(self) -> np.ndarray:
        state = self.read_state()
        if state.tcp_pose is None:
            raise RuntimeError("RTDE receive interface did not expose actual_TCP_pose")
        return state.tcp_pose

    def validate_live_state(self, state: UR5eState) -> SafetyDecision:
        decision = check_joint_state(
            state.q,
            state.qd,
            self.limits,
            host_stamp_ns=state.host_stamp_ns,
            robot_stamp_s=state.robot_timestamp_s,
        )
        if state.tcp_pose is not None:
            pose_decision = check_tcp_pose(state.tcp_pose, self.limits)
            for reason in pose_decision.reasons:
                decision.add(reason)
        return decision

    def _call_servoj(
        self,
        q: np.ndarray,
        *,
        gain: float,
        lookahead_time: float,
        velocity: float,
        acceleration: float,
        period_s: float | None = None,
    ) -> None:
        control = self._control
        if control is None:
            raise RuntimeError("connect_control() must be called before servoJ")
        fn = getattr(control, "servoJ", None)
        if fn is None:
            raise RuntimeError("rtde_control does not expose servoJ")
        period_s = self.period_s if period_s is None else float(period_s)
        q_list = np.asarray(q, dtype=np.float64).reshape(6).tolist()
        attempts = [
            (q_list, float(acceleration), float(velocity), float(period_s), float(lookahead_time), float(gain)),
            (q_list, float(velocity), float(acceleration), float(period_s), float(lookahead_time), float(gain)),
            (q_list, float(period_s), float(lookahead_time), float(gain)),
            (q_list, float(lookahead_time), float(gain)),
            (q_list,),
        ]
        last_exc: Exception | None = None
        for args in attempts:
            try:
                fn(*args)
                return
            except TypeError as exc:
                last_exc = exc
        raise RuntimeError(f"servoJ invocation failed: {last_exc}") from last_exc

    def _call_speedj(self, qd: np.ndarray, *, acceleration: float, duration_s: float) -> None:
        control = self._control
        if control is None:
            raise RuntimeError("connect_control() must be called before speedJ")
        fn = getattr(control, "speedJ", None)
        if fn is None:
            raise RuntimeError("rtde_control does not expose speedJ")
        qd_list = np.asarray(qd, dtype=np.float64).reshape(6).tolist()
        attempts = [
            (qd_list, float(acceleration), float(duration_s)),
            (qd_list, float(acceleration)),
            (qd_list,),
        ]
        last_exc: Exception | None = None
        for args in attempts:
            try:
                fn(*args)
                return
            except TypeError as exc:
                last_exc = exc
        raise RuntimeError(f"speedJ invocation failed: {last_exc}") from last_exc

    def report(self) -> dict[str, Any]:
        return {
            "robot_ip": self.robot_ip,
            "frequency_hz": float(self.frequency),
            "supports_direct_torque": bool(self.supports_direct_torque()),
            "control_connected": self._control is not None,
            "receive_connected": self._receive is not None,
            "supports_servoj": bool(self._control is not None and hasattr(self._control, "servoJ")),
            "supports_speedj": bool(self._control is not None and hasattr(self._control, "speedJ")),
            "control_methods": self.available_control_methods(),
            "receive_methods": self.available_receive_methods(),
            "last_stop_reason": self.last_stop_reason,
        }


def _summarize_exception(exc: BaseException) -> dict[str, str]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }


def _sample_row(
    *,
    phase: str,
    cycle_index: int,
    state: UR5eState,
    cmd_q: np.ndarray | None,
    cmd_qd: np.ndarray | None,
    timing: dict[str, Any],
    error_q: np.ndarray | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "phase": phase,
        "cycle_index": int(cycle_index),
        "host_stamp_ns": int(state.host_stamp_ns),
        "robot_timestamp_s": state.robot_timestamp_s,
        "robot_mode": state.robot_mode,
        "safety_status": state.safety_status,
        "q": state.q,
        "qd": state.qd,
        "tcp_pose": state.tcp_pose,
        "timing": timing,
    }
    if cmd_q is not None:
        row["cmd_q"] = np.asarray(cmd_q, dtype=np.float64).reshape(6)
    if cmd_qd is not None:
        row["cmd_qd"] = np.asarray(cmd_qd, dtype=np.float64).reshape(6)
    if error_q is not None:
        row["error_q"] = np.asarray(error_q, dtype=np.float64).reshape(6)
        row["max_abs_error_q_rad"] = float(np.max(np.abs(row["error_q"])))
    if extra:
        row.update(extra)
    return row
