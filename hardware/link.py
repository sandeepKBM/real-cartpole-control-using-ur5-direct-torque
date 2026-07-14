"""Connection and live state for the real UR5e -- and nothing else.

``UR5eLink`` opens the RTDE receive/control sockets, reads state, streams
``servoL`` waypoints, and can stop the robot. It never returns stale data:
``read_state()`` raises ``RTDEStateError`` on any problem instead of caching
or defaulting -- this is the direct fix for the bug found in the previous
hardware lane's ROS2 node, where a failed read silently left old state in
place forever.

Reconnect policy is intentionally NOT decided here. ``UR5eLink`` only updates
its own ``health`` (a ``hardware.safety.ConnectionHealth``) on a *successful*
read; on failure, ``read_state()`` raises and the caller must decide what to
do (retry with backoff while idle in ``tools/ur5e_connect.py --watch``, or
abort immediately mid-motion in ``hardware/motion.py``) -- those two contexts
need different policies, so the policy lives with the caller, not here.
"""

from __future__ import annotations

import inspect
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .safety import ConnectionHealth, UR5eSafetyLimits


class RTDELinkError(RuntimeError):
    """Connection setup failed: RTDE bindings unavailable, socket couldn't
    reach the robot, or the connected control interface's API doesn't match
    what this code expects (e.g. servoL's argument order)."""


class RTDEStateError(RuntimeError):
    """A state read failed or returned bad data. Never swallowed -- every
    caller must catch this explicitly and route it through
    ``hardware.safety.ConnectionHealth``."""


# The exact servoL(pose, speed, acceleration, time, lookahead_time, gain)
# signature this code was written against (ur_rtde's documented API). Verified
# at connect() time via introspection instead of guessed via TypeError trial
# and error (the previous lane's bug: a different library version could
# silently swap velocity/acceleration positionally with no error at all).
_EXPECTED_SERVOL_PARAMS = ("pose", "speed", "acceleration", "time", "lookahead_time", "gain")


def _load_rtde_classes() -> tuple[type[Any], type[Any]]:
    try:
        from rtde_control import RTDEControlInterface
        from rtde_receive import RTDEReceiveInterface
    except Exception as exc:  # pragma: no cover - exercised only without rtde installed
        raise RTDELinkError(
            "RTDE Python bindings are not available. Install rtde_control / "
            "rtde_receive (the `ur_rtde` package) before connecting to a real robot."
        ) from exc
    return RTDEControlInterface, RTDEReceiveInterface


@dataclass
class UR5eState:
    q: np.ndarray
    qd: np.ndarray
    tcp_pose: np.ndarray
    host_stamp_ns: int
    robot_timestamp_s: float | None
    safety_status: int | None


class UR5eLink:
    """Thin, honest wrapper around one RTDE connection to a real UR5e.

    ``control_factory``/``receive_factory`` let tests inject fake RTDE
    objects (same pattern the previous lane's tests used) -- production code
    just calls ``UR5eLink(robot_ip, frequency_hz)`` and leaves them None.
    """

    def __init__(
        self,
        robot_ip: str,
        frequency_hz: float,
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

    @property
    def has_control(self) -> bool:
        return self._control is not None

    def connect(self, *, with_control: bool) -> None:
        """Open the receive socket, and the control socket too if
        ``with_control`` -- the receive-only default from the previous lane
        is preserved: nothing here ever opens control unless explicitly
        asked."""
        if self._receive is None:
            if self._receive_factory is not None:
                self._receive = self._receive_factory(self.robot_ip, self.frequency_hz)
            else:
                _, receive_cls = _load_rtde_classes()
                try:
                    self._receive = receive_cls(self.robot_ip, self.frequency_hz)
                except Exception as exc:
                    raise RTDELinkError(f"Failed to open RTDE receive interface: {exc}") from exc

        if with_control and self._control is None:
            if self._control_factory is not None:
                self._control = self._control_factory(self.robot_ip, self.frequency_hz)
            else:
                control_cls, _ = _load_rtde_classes()
                try:
                    self._control = control_cls(self.robot_ip, self.frequency_hz)
                except Exception as exc:
                    raise RTDELinkError(f"Failed to open RTDE control interface: {exc}") from exc
            self._verify_servol_signature()

    def _verify_servol_signature(self) -> None:
        servo_l = getattr(self._control, "servoL", None)
        if servo_l is None:
            raise RTDELinkError(
                "Connected RTDE control interface has no servoL method -- cannot "
                "perform Cartesian motion with this library/robot combination."
            )
        try:
            sig = inspect.signature(servo_l)
        except (TypeError, ValueError):
            # C-extension bindings (and some test doubles) aren't
            # introspectable -- accept rather than block on an unverifiable
            # check, but never silently guess argument order elsewhere.
            return
        params = list(sig.parameters)
        if params and params[0] == "self":
            params = params[1:]
        if tuple(params[: len(_EXPECTED_SERVOL_PARAMS)]) != _EXPECTED_SERVOL_PARAMS:
            raise RTDELinkError(
                "servoL's signature does not match the expected "
                f"{_EXPECTED_SERVOL_PARAMS}; found parameters {tuple(params)}. Refusing "
                "to guess argument order -- update hardware/link.py's "
                "_EXPECTED_SERVOL_PARAMS after confirming the real signature for this "
                "rtde_control version."
            )

    def read_state(self) -> UR5eState:
        """Raises ``RTDEStateError`` immediately on any problem -- never
        returns a cached or default value. On success, also records the read
        with ``self.health`` (see ``ConnectionHealth``)."""
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
        if (
            not np.all(np.isfinite(q_arr))
            or not np.all(np.isfinite(qd_arr))
            or not np.all(np.isfinite(pose_arr))
        ):
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

    def is_alive(self) -> bool:
        return self.health.is_alive()

    def servo_l(
        self,
        pose,
        *,
        speed: float,
        acceleration: float,
        time_s: float,
        lookahead_time: float,
        gain: float,
    ) -> None:
        if self._control is None:
            raise RTDEStateError("servo_l() called before connect(with_control=True)")
        pose_arr = np.asarray(pose, dtype=np.float64).reshape(6)
        self._control.servoL(pose_arr.tolist(), speed, acceleration, time_s, lookahead_time, gain)

    def move_j(
        self,
        q_rad: np.ndarray,
        *,
        speed_rad_s: float = 0.5,
        acceleration_rad_s2: float = 0.5,
    ) -> None:
        """Blocking joint-space move via RTDE ``moveJ``."""
        if self._control is None:
            raise RTDEStateError("move_j() called before connect(with_control=True)")
        q = np.asarray(q_rad, dtype=np.float64).reshape(6)
        if not np.all(np.isfinite(q)):
            raise RTDEStateError("move_j() got non-finite joint targets")
        move_j = getattr(self._control, "moveJ", None)
        if move_j is None:
            raise RTDELinkError("Connected RTDE control interface has no moveJ() method")
        try:
            result = move_j(q.tolist(), float(speed_rad_s), float(acceleration_rad_s2))
        except Exception as exc:
            raise RTDEStateError(f"moveJ() failed: {exc}") from exc
        if result is False:
            raise RTDEStateError("moveJ() returned false")

    def servo_stop(self) -> None:
        """Cleanly end a servoL streaming session (best-effort)."""
        if self._control is None:
            return
        method = getattr(self._control, "servoStop", None)
        if method is not None:
            try:
                method()
            except Exception:
                pass

    def safe_stop(self, reason: str) -> None:
        """Best-effort stop: try servoStop, then stopScript, independently --
        one failing must never prevent the other from being attempted. Then
        disconnect. ``reason`` is accepted for logging symmetry with
        ``EStopLatch.trip(reason)`` (callers should trip the latch
        separately with the same reason)."""
        if self._control is not None:
            for method_name in ("servoStop", "stopScript"):
                method = getattr(self._control, method_name, None)
                if method is None:
                    continue
                try:
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
