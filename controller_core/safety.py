"""Safety checks for Cartesian-impedance torque control (simulator-independent)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .state_types import RobotState


@dataclass
class ImpedanceSafetyConfig:
    max_abs_y_drift_m: float = 0.03
    max_abs_z_drift_m: float = 0.03
    max_abs_orthogonal_drift_m: float = 0.03
    max_orientation_error_rad: float = 0.25
    max_joint_velocity_radps: float = 1.5
    max_x_error_growth_steps: int = 100
    max_axis_error_growth_steps: int = 100
    emergency_stop_on_nan: bool = True
    emergency_stop_on_joint_limit: bool = True
    q_lower: np.ndarray = field(
        default_factory=lambda: np.full(6, -2.0 * np.pi, dtype=np.float64)
    )
    q_upper: np.ndarray = field(
        default_factory=lambda: np.full(6, 2.0 * np.pi, dtype=np.float64)
    )


@dataclass
class ImpedanceSafetyStatus:
    ok: bool
    reason: str = ""


class ImpedanceSafetyMonitor:
    """Tracks drift from an initial pose and monotonic transport-axis error growth."""

    def __init__(self, cfg: ImpedanceSafetyConfig) -> None:
        self.cfg = cfg
        self._y0: float | None = None
        self._z0: float | None = None
        self._prev_abs_x_err: float | None = None
        self._x_err_grow_count: int = 0
        self._pos0: np.ndarray | None = None
        self._move_axis: int | None = None
        self._prev_abs_axis_err: float | None = None
        self._axis_err_grow_count: int = 0

    def reset(self) -> None:
        self._y0 = None
        self._z0 = None
        self._prev_abs_x_err = None
        self._x_err_grow_count = 0
        self._pos0 = None
        self._move_axis = None
        self._prev_abs_axis_err = None
        self._axis_err_grow_count = 0

    def set_initial_yz(self, y0: float, z0: float) -> None:
        self._y0 = float(y0)
        self._z0 = float(z0)

    def set_initial_position(self, position: np.ndarray, move_axis: int) -> None:
        self._pos0 = np.asarray(position, dtype=np.float64).reshape(3)
        self._move_axis = int(move_axis)

    def check(
        self,
        state: RobotState,
        *,
        axis_error: float | None = None,
        x_error: float | None = None,
        orientation_error_norm: float,
    ) -> ImpedanceSafetyStatus:
        reasons: list[str] = []
        q = np.asarray(state["q"], dtype=np.float64).reshape(-1)
        qd = np.asarray(state["qd"], dtype=np.float64).reshape(-1)
        ee = np.asarray(state["ee_pos"], dtype=np.float64).reshape(-1)
        axis_names = ("X", "Y", "Z")

        if self.cfg.emergency_stop_on_nan:
            if not np.all(np.isfinite(q)) or not np.all(np.isfinite(qd)):
                reasons.append("NaN/Inf in joint state")

        if self.cfg.emergency_stop_on_joint_limit:
            if np.any(q < self.cfg.q_lower) or np.any(q > self.cfg.q_upper):
                reasons.append("joint limit violated")

        if np.any(np.abs(qd) > self.cfg.max_joint_velocity_radps + 1e-9):
            reasons.append(f"|qd| > {self.cfg.max_joint_velocity_radps} rad/s")

        if self._pos0 is not None and self._move_axis is not None:
            for idx in range(3):
                if idx == self._move_axis:
                    continue
                if abs(float(ee[idx]) - float(self._pos0[idx])) > self.cfg.max_abs_orthogonal_drift_m:
                    reasons.append(
                        f"|{axis_names[idx]}-{axis_names[idx]}0| > {self.cfg.max_abs_orthogonal_drift_m} m"
                    )
        else:
            if self._y0 is not None:
                if abs(float(ee[1]) - self._y0) > self.cfg.max_abs_y_drift_m:
                    reasons.append(f"|Y-Y0| > {self.cfg.max_abs_y_drift_m} m")
            if self._z0 is not None:
                if abs(float(ee[2]) - self._z0) > self.cfg.max_abs_z_drift_m:
                    reasons.append(f"|Z-Z0| > {self.cfg.max_abs_z_drift_m} m")

        if orientation_error_norm > self.cfg.max_orientation_error_rad:
            reasons.append(
                f"||orientation error|| > {self.cfg.max_orientation_error_rad} rad"
            )

        axis_err = axis_error if axis_error is not None else x_error
        if axis_err is not None:
            abs_axis_err = abs(float(axis_err))
            if self._move_axis is not None:
                if self._prev_abs_axis_err is not None:
                    if abs_axis_err > self._prev_abs_axis_err + 1e-9:
                        self._axis_err_grow_count += 1
                    else:
                        self._axis_err_grow_count = 0
                    if self._axis_err_grow_count >= self.cfg.max_axis_error_growth_steps:
                        reasons.append(
                            f"|axis_error| grew for {self._axis_err_grow_count} consecutive steps"
                        )
                self._prev_abs_axis_err = abs_axis_err
            else:
                if self._prev_abs_x_err is not None:
                    if abs_axis_err > self._prev_abs_x_err + 1e-9:
                        self._x_err_grow_count += 1
                    else:
                        self._x_err_grow_count = 0
                    if self._x_err_grow_count >= self.cfg.max_x_error_growth_steps:
                        reasons.append(
                            f"|x_error| grew for {self._x_err_grow_count} consecutive steps"
                        )
                self._prev_abs_x_err = abs_axis_err

        ok = len(reasons) == 0
        return ImpedanceSafetyStatus(ok=ok, reason="; ".join(reasons))
