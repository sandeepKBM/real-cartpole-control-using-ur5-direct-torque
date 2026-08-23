"""Safety checks for Cartesian-impedance torque control (simulator-independent)."""

from __future__ import annotations

from collections.abc import Sequence
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


#: Tolerance on ``R.T @ R - I`` for a supplied task-frame rotation. A basis
#: that is not orthonormal would rescale measured drift -- i.e. silently
#: weaken (or tighten) a safety threshold -- so it is rejected rather than
#: normalized. Reflections (det = -1) ARE accepted: a column permutation such
#: as ``[tool_y | tool_x | tool_z]`` (used to put the transported direction in
#: the skipped column) is a legitimate basis and preserves every length.
TASK_ROTATION_ORTHONORMAL_TOL = 1.0e-6


def validated_task_rotation(rotation: np.ndarray) -> np.ndarray:
    """Return ``rotation`` as a checked 3x3 orthonormal world->task basis.

    Columns are the task axes expressed in world coordinates, so
    ``R.T @ (p - p0)`` is the displacement resolved along the task axes.
    """
    rot = np.asarray(rotation, dtype=np.float64)
    if rot.shape != (3, 3):
        raise ValueError(f"task_rotation must be 3x3; got shape {rot.shape}")
    if not np.all(np.isfinite(rot)):
        raise ValueError("task_rotation contains NaN/Inf")
    ortho_err = float(np.max(np.abs(rot.T @ rot - np.eye(3))))
    if ortho_err > TASK_ROTATION_ORTHONORMAL_TOL:
        raise ValueError(
            "task_rotation columns are not orthonormal "
            f"(max|R^T R - I| = {ortho_err:.3e} > {TASK_ROTATION_ORTHONORMAL_TOL:.1e}); "
            "a non-orthonormal basis would rescale measured drift and therefore "
            "silently change the effective drift threshold"
        )
    return rot


def validated_tracked_axes(
    tracked_axes: Sequence[int] | None, move_axis: int
) -> frozenset[int]:
    """Return the set of axes exempt from the drift check.

    ``None`` -> ``{move_axis}``, the historical single-axis behavior.

    Rejects, loudly rather than silently:
      * indices outside 0..2, which would exempt nothing and give a false
        sense that an axis was excluded;
      * a set that omits ``move_axis``, which would charge the commanded
        transport direction itself as drift;
      * all three axes, which would leave NOTHING checked -- that is a guard
        disabled by configuration, and this module never permits that.
    """
    if tracked_axes is None:
        return frozenset({int(move_axis)})
    axes = frozenset(int(a) for a in tracked_axes)
    bad = sorted(a for a in axes if a not in (0, 1, 2))
    if bad:
        raise ValueError(f"tracked_axes entries must be in 0..2; got {bad}")
    if int(move_axis) not in axes:
        raise ValueError(
            f"tracked_axes {sorted(axes)} does not contain move_axis {int(move_axis)}; "
            "the commanded transport direction would be charged as drift"
        )
    if len(axes) == 3:
        raise ValueError(
            "tracked_axes cannot contain all three axes -- no component would be "
            "left for the drift check, which silently disables the guard"
        )
    return axes


class ImpedanceSafetyMonitor:
    """Tracks drift from an initial pose and monotonic transport-axis error growth.

    DRIFT FRAME (opt-in, added 2026-08-14). By default the two non-transport
    drift components are the WORLD axes other than ``move_axis`` -- the exact
    historical behavior, and the only behavior any caller gets unless it opts
    in. That default is correct only while the commanded task direction IS a
    world axis. It is not for a tool-frame task: with the tool-Y direction at
    ``[0.7103, 0.6924, 0.1268]`` in world, 0.7039 of every metre of INTENDED
    travel lands in the world-orthogonal components, so the guard counts
    commanded motion as a fault and caps clean displacement at
    ``tolerance / 0.7039`` regardless of controller or gains.

    A caller that drives a non-world-axis task may therefore supply a
    ``task_rotation`` -- a 3x3 orthonormal matrix whose columns are the task
    axes in world coordinates, with the TRANSPORTED direction in column
    ``move_axis``. Drift is then measured as ``R.T @ (ee - pos0)``, i.e. along
    the two directions genuinely perpendicular to the commanded one. The
    thresholds are untouched: the same ``max_abs_orthogonal_drift_m`` is
    applied to the same number of components, just resolved on the right axes.

    Supply it once via :meth:`set_initial_position` for a frame frozen at
    reset, and/or per cycle via :meth:`check` for a live (rotating) frame; the
    per-cycle value wins when both are given. Omit both for world behavior.
    """

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
        self._task_rotation: np.ndarray | None = None
        self._tracked_axes: frozenset[int] | None = None

    def reset(self) -> None:
        self._y0 = None
        self._z0 = None
        self._prev_abs_x_err = None
        self._x_err_grow_count = 0
        self._pos0 = None
        self._move_axis = None
        self._prev_abs_axis_err = None
        self._axis_err_grow_count = 0
        self._task_rotation = None
        self._tracked_axes = None

    def set_initial_yz(self, y0: float, z0: float) -> None:
        self._y0 = float(y0)
        self._z0 = float(z0)

    def set_initial_position(
        self,
        position: np.ndarray,
        move_axis: int,
        *,
        task_rotation: np.ndarray | None = None,
        tracked_axes: Sequence[int] | None = None,
    ) -> None:
        """Capture the drift reference pose.

        ``tracked_axes`` (opt-in) lists EVERY axis the controller actively
        commands, so drift is measured only on the ones it does not. Default
        ``None`` means ``{move_axis}`` -- the historical single-axis behavior,
        bit-for-bit.

        This exists because a multi-axis task otherwise has its own commanded
        motion counted as drift: a controller tracking tool X AND tool Y with
        ``move_axis=1`` would have every metre of intended tool-X travel
        charged against ``max_abs_orthogonal_drift_m``. That is the same class
        of error as measuring tool-frame drift on world axes, one level up.
        """
        self._pos0 = np.asarray(position, dtype=np.float64).reshape(3)
        self._move_axis = int(move_axis)
        self._task_rotation = (
            None if task_rotation is None else validated_task_rotation(task_rotation)
        )
        self._tracked_axes = validated_tracked_axes(tracked_axes, self._move_axis)

    @property
    def task_rotation(self) -> np.ndarray | None:
        """The frozen task-frame basis, or ``None`` for world-frame drift."""
        return None if self._task_rotation is None else self._task_rotation.copy()

    def drift_vector(
        self,
        ee_pos: np.ndarray,
        *,
        task_rotation: np.ndarray | None = None,
    ) -> np.ndarray:
        """Drift from the captured start pose, in the frame :meth:`check` uses.

        Exposed so a harness can log/report the same three numbers the guard
        compares, instead of re-deriving the rotation and risking a different
        convention.
        """
        if self._pos0 is None:
            raise RuntimeError("set_initial_position() must be called first")
        delta = np.asarray(ee_pos, dtype=np.float64).reshape(3) - self._pos0
        rot = self._task_rotation if task_rotation is None else validated_task_rotation(task_rotation)
        return delta if rot is None else rot.T @ delta

    def check(
        self,
        state: RobotState,
        *,
        axis_error: float | None = None,
        x_error: float | None = None,
        orientation_error_norm: float,
        axis_target_moving: bool = False,
        task_rotation: np.ndarray | None = None,
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
            rot = (
                self._task_rotation
                if task_rotation is None
                else validated_task_rotation(task_rotation)
            )
            # Axes the controller actively commands are exempt from the drift
            # check. Defaults to {move_axis}, i.e. the historical behavior.
            exempt = (
                self._tracked_axes
                if self._tracked_axes is not None
                else frozenset({self._move_axis})
            )
            if rot is None:
                # WORLD FRAME -- the historical path, deliberately left
                # expression-for-expression as it was so it stays
                # byte-identical rather than merely numerically close (no
                # identity matmul, no rewritten subtraction).
                for idx in range(3):
                    if idx in exempt:
                        continue
                    if abs(float(ee[idx]) - float(self._pos0[idx])) > self.cfg.max_abs_orthogonal_drift_m:
                        reasons.append(
                            f"|{axis_names[idx]}-{axis_names[idx]}0| > {self.cfg.max_abs_orthogonal_drift_m} m"
                        )
            else:
                # TASK FRAME -- same threshold, same component count, resolved
                # on the axes the task is actually driven along. Column
                # `move_axis` is the commanded direction and is skipped, so
                # what is checked is genuinely off-task motion.
                drift = rot.T @ (ee - self._pos0)
                for idx in range(3):
                    if idx in exempt:
                        continue
                    if abs(float(drift[idx])) > self.cfg.max_abs_orthogonal_drift_m:
                        reasons.append(
                            f"|task {axis_names[idx]} drift| > {self.cfg.max_abs_orthogonal_drift_m} m"
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
                if axis_target_moving:
                    # The transport target is actively ramping (e.g. a
                    # min-jerk move profile): the tracking error growing
                    # while chasing a moving target is expected dynamics,
                    # not divergence. Don't accumulate the growth streak
                    # here, but keep the reference current so growth from
                    # this point is detected once the target settles.
                    self._axis_err_grow_count = 0
                elif self._prev_abs_axis_err is not None:
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
                if axis_target_moving:
                    self._x_err_grow_count = 0
                elif self._prev_abs_x_err is not None:
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
