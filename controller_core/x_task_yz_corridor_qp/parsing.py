"""
YAML-value parsing helpers for ``XTaskYZCorridorQPConfig``.

Only ONE genuinely new validator lives here
(``_parse_corridor_half_width``); the manipulability-CBF epsilon/alpha
validators are IMPORTED from
``controller_core.x_axis_cartesian_impedance.parsing`` and re-exported, not
copied -- the fields mean exactly the same thing in both controllers, and
``_parse_manipulability_cbf_alpha`` already takes a ``field_name`` argument,
so it works unmodified for ``yz_corridor_alpha1``/``yz_corridor_alpha2`` too.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..x_axis_cartesian_impedance.parsing import (
    _parse_manipulability_cbf_alpha,
    _parse_manipulability_cbf_epsilon,
)

__all__ = [
    "_parse_corridor_half_width",
    "_parse_manipulability_cbf_alpha",
    "_parse_manipulability_cbf_epsilon",
    "_parse_task_excluded_joints",
]

#: The reduced task is 4 rows (world X + 3 orientation). A 6-joint arm
#: therefore has exactly 2 redundant DOF, so at most 2 joints can be held out
#: of the task before the remaining columns can no longer span it. See
#: ``_parse_task_excluded_joints``.
MAX_TASK_EXCLUDED_JOINTS = 2


def _parse_corridor_half_width(raw: Any, field_name: str) -> float:
    """Parse/validate a Y or Z corridor half-width, in meters.

    Must be finite and strictly > 0, and this is deliberately LOUD rather
    than permissive (the same call ``_parse_transport_axis_index`` and
    ``_parse_manipulability_cbf_epsilon`` make in the sibling package):

      - ``width <= 0`` makes ``y_min >= y_max``, i.e. the two barrier
        functions ``h_max = y_max - y`` and ``h_min = y - y_min`` cannot both
        be non-negative anywhere. The QP would then be permanently infeasible
        -- reported as infeasible every cycle, but only after the dual ascent
        has burned its whole iteration budget hunting for a solution that
        provably does not exist.
      - ``NaN`` makes every corridor comparison silently False and every
        constraint row NaN, which propagates into the solve as a silently
        disabled (worse: silently poisoned) safety mechanism.
      - ``inf`` is not accepted either: an infinite half-width is a request
        to disable the corridor, and the honest way to spell that is
        ``yz_corridor_enabled: false``, which skips building the rows at all
        rather than paying for four vacuous ones every cycle.

    Validated once at config-construction time; the value cannot change
    during a run.
    """
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{field_name} must be a finite float > 0 (meters); got {raw!r}"
        ) from exc
    if not (value > 0.0) or value == float("inf"):
        raise ValueError(f"{field_name} must be finite and > 0 (meters); got {value!r}")
    return value


def _parse_task_excluded_joints(raw: Any) -> tuple[int, ...]:
    """Parse/validate ``task_excluded_joints`` -- the joints this controller is
    forbidden to spend TASK torque on.

    Accepts ``None`` (meaning "the class default applies", returned as-is by
    the caller), a list/tuple of ints, or an empty list/tuple (meaning "exclude
    nothing", i.e. the pre-2026-08-13 behavior). Returns a sorted tuple of
    distinct indices.

    Loud, not permissive, for the same reasons
    ``x_axis_cartesian_impedance.parsing._parse_split_base_wrist_active_joints``
    is loud about its own index set:

      - an out-of-range index would silently index nothing (or, with a
        negative int, silently exclude a DIFFERENT joint than the one named --
        ``-1`` reads as wrist_3, not "no joint");
      - a duplicate index is always a typo, never a stronger exclusion;
      - MORE THAN 2 exclusions is a dimensional error, not a preference. The
        reduced task is 4 rows (world X + 3 orientation) on 6 joints, so there
        are exactly 2 redundant DOF. Excluding 3 joints leaves 3 columns for a
        4-row task: the QP would still "run" and still return a torque, but it
        would be solving an underdetermined task whose residual nothing here
        reports. That is exactly the class of silent mismatch this repo's
        other index parsers exist to prevent.

    NOT checked here, and deliberately so (it cannot be): whether the columns
    that REMAIN actually span the 4-row task at the pose you intend to run.
    That is a property of a POSE, not of an index set -- the same caveat
    ``split_base_wrist_active_joints`` documents at length. Screen your
    selection against the intended start pose (measure
    ``cond(J_reduced[:, remaining])``) before trusting it.
    """
    if raw is None:
        return None  # type: ignore[return-value]
    if isinstance(raw, (int, np.integer)) and not isinstance(raw, bool):
        raw = [raw]
    try:
        values = [int(v) for v in raw]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "task_excluded_joints must be a list/tuple of joint indices in [0, 5] "
            f"(or [] for none); got {raw!r}"
        ) from exc
    if any(v < 0 or v > 5 for v in values):
        raise ValueError(
            "task_excluded_joints entries must be indices into JOINT_NAME_ORDER, i.e. "
            f"in [0, 5]; got {values!r}"
        )
    if len(set(values)) != len(values):
        raise ValueError(f"task_excluded_joints has duplicate indices: {values!r}")
    if len(values) > MAX_TASK_EXCLUDED_JOINTS:
        raise ValueError(
            f"task_excluded_joints={values!r} excludes {len(values)} joints, but the reduced "
            f"task has 4 rows on 6 joints -- at most {MAX_TASK_EXCLUDED_JOINTS} joints can be "
            "held out before the remaining columns cannot span it and the QP silently solves "
            "an underdetermined task."
        )
    return tuple(sorted(values))


#: The transport axis. It MUST be a tracked task row -- see
#: ``_parse_axis_row_sets``.
TRANSPORT_AXIS_ROW = 0


def _parse_axis_row_sets(raw_task: Any, raw_corridor: Any) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Parse/validate ``task_axis_rows`` and ``corridor_axis_rows`` TOGETHER.

    They are validated as a pair rather than independently because the only
    interesting failure mode is a translation axis appearing in both -- which
    would put the same axis in the QP's objective AND in a barrier constraint
    on the same solve, i.e. the controller fighting itself with two
    mechanisms that were designed as alternatives. Independent per-field
    validators cannot see that.

    Indices are into the WORLD translation rows of ``J``: 0 = X, 1 = Y,
    2 = Z. (Rows 3-5, the orientation rows, are always task rows and are not
    configurable -- there is no corridor formulation for orientation here.)

    ``None`` for either means "use the class default" and is returned as-is.

    Rules, all loud:
      - every index in [0, 2];
      - no duplicates within a set;
      - the two sets DISJOINT;
      - ``task_axis_rows`` non-empty -- an empty task is not a controller;
      - ``TRANSPORT_AXIS_ROW`` (world X) must be IN ``task_axis_rows``. This
        controller's ``_desired_task``/``transport_axis_index`` plumbing, its
        tolerances and its callers all describe X as the transported axis;
        bounding X with a corridor instead of tracking it would mean the arm
        never drives toward its target while every target and metric still
        said it should. The same class of silent mismatch ``compute()``
        already refuses ``transport_axis_index != 0`` for.
      - ``corridor_axis_rows`` may be empty (no corridor at all), but every
        entry must be 1 or 2, because the corridor half-widths are per-axis
        fields (``y_corridor_half_width_m``/``z_corridor_half_width_m``) and
        there is no X one. This is implied by the rule above but is checked
        explicitly so the error message names the real reason.
    """
    def _one(raw: Any, field: str) -> tuple[int, ...] | None:
        if raw is None:
            return None
        if isinstance(raw, (int, np.integer)) and not isinstance(raw, bool):
            raw = [raw]
        try:
            values = [int(v) for v in raw]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{field} must be a list of world translation row indices in [0, 2] "
                f"(0=X, 1=Y, 2=Z); got {raw!r}"
            ) from exc
        if any(v < 0 or v > 2 for v in values):
            raise ValueError(
                f"{field} entries must be world translation row indices in [0, 2] "
                f"(0=X, 1=Y, 2=Z); got {values!r}"
            )
        if len(set(values)) != len(values):
            raise ValueError(f"{field} has duplicate indices: {values!r}")
        return tuple(sorted(values))

    task = _one(raw_task, "task_axis_rows")
    corridor = _one(raw_corridor, "corridor_axis_rows")
    if task is None and corridor is None:
        return None, None  # type: ignore[return-value]
    task_eff = (TRANSPORT_AXIS_ROW,) if task is None else task
    corridor_eff = (1, 2) if corridor is None else corridor

    if not task_eff:
        raise ValueError("task_axis_rows is empty -- a task with no tracked axis is not a task.")
    if TRANSPORT_AXIS_ROW not in task_eff:
        raise ValueError(
            f"task_axis_rows={task_eff} does not contain the transport axis "
            f"{TRANSPORT_AXIS_ROW} (world X). This controller transports along X: its targets, "
            "tolerances and guards all describe X, so leaving X untracked would mean the arm "
            "never drives toward its target while every metric still said it should."
        )
    overlap = sorted(set(task_eff) & set(corridor_eff))
    if overlap:
        raise ValueError(
            f"task_axis_rows={task_eff} and corridor_axis_rows={corridor_eff} both contain "
            f"{overlap} -- an axis cannot be tracked in the QP objective AND bounded by a "
            "barrier row in the same solve; those are alternatives, not layers."
        )
    bad_corridor = [a for a in corridor_eff if a not in (1, 2)]
    if bad_corridor:
        raise ValueError(
            f"corridor_axis_rows={corridor_eff} contains {bad_corridor}, but corridor "
            "half-widths are per-axis fields and only y_corridor_half_width_m / "
            "z_corridor_half_width_m exist (axes 1 and 2)."
        )
    return task, corridor


#: Valid values for ``XTaskYZCorridorQPConfig.task_frame`` /
#: ``task_frame_update``. Kept as module constants so tests and configs can
#: assert against the same source rather than restating string literals.
TASK_FRAMES = ("world", "tool")
TASK_FRAME_UPDATES = ("live", "frozen", "hybrid")


def _parse_task_frame(raw_frame: Any, raw_update: Any) -> tuple[str | None, str | None]:
    """Parse/validate ``task_frame`` and ``task_frame_update``.

    Parsed as a PAIR for the same reason the axis row sets are: the pairing
    carries meaning that neither field shows alone. ``task_frame_update`` is
    only consulted when ``task_frame == "tool"``, so a config that sets an
    update mode while leaving the frame at ``"world"`` has written something
    that provably does nothing -- that is a mistake worth surfacing loudly
    rather than silently ignoring, exactly like the axis-overlap case.

    Returns ``(None, None)`` for absent fields so the caller can leave the
    dataclass defaults (``"world"``/``"frozen"``) in place -- the same
    "absent means default, do not disable" convention ``task_excluded_joints``
    uses, which is what keeps the byte-identical default guarantee.
    """
    frame = None
    if raw_frame is not None:
        if not isinstance(raw_frame, str):
            raise ValueError(f"task_frame must be a string, got {type(raw_frame).__name__}")
        frame = raw_frame.strip().lower()
        if frame not in TASK_FRAMES:
            raise ValueError(
                f"task_frame={raw_frame!r} is not one of {TASK_FRAMES}. "
                '"world" indexes world X/Y/Z; "tool" indexes the attachment_site '
                "frame's X/Y/Z after pre-rotating the Jacobian's position rows."
            )

    update = None
    if raw_update is not None:
        if not isinstance(raw_update, str):
            raise ValueError(
                f"task_frame_update must be a string, got {type(raw_update).__name__}"
            )
        update = raw_update.strip().lower()
        if update not in TASK_FRAME_UPDATES:
            raise ValueError(
                f"task_frame_update={raw_update!r} is not one of {TASK_FRAME_UPDATES}."
            )

    if update is not None and (frame or "world") != "tool":
        raise ValueError(
            f"task_frame_update={update!r} was set while task_frame is "
            f"{(frame or 'world')!r}. The update mode only governs how R_tool is "
            "refreshed and is ignored entirely in the world frame, so this config "
            "asks for something that provably has no effect."
        )
    return frame, update
