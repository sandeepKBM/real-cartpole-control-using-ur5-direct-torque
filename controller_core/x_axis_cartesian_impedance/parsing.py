"""
Small, permissive YAML-value parsing helpers for CartesianImpedanceConfig.

Split out of the former single-file ``x_axis_cartesian_impedance.py`` module
(pure structural refactor; see the package ``__init__.py`` for the original
module docstring).
"""

from __future__ import annotations

from typing import Any, Literal


def _parse_y_control_mode(raw: Any) -> Literal["tight", "corridor"]:
    """Case-insensitive parse of the ``y_control_mode`` YAML value, matching
    _parse_friction_model's permissive convention: an unrecognized value
    falls back to "tight" (the historical default) rather than raising."""
    value = str(raw).lower()
    if value == "corridor":
        return "corridor"
    return "tight"


def _parse_transport_axis_index(raw: Any) -> int:
    """Parse the ``transport_axis_index`` YAML value into 0, 1, or 2.

    Deliberately NOT permissive, unlike the two string parsers in this module:
    an unparseable or out-of-range value RAISES instead of silently falling
    back to 0. Falling back would mean a caller that asked for a Y or Z
    transport quietly got a world-X one instead -- with the guards, targets,
    and drift tolerances of whatever the caller believed it selected. That is
    exactly the silent axis mismatch this field exists to remove, so it must
    be loud. Mirrors ``hardware/transport_common.py::validate_transport_axis_index``
    (bools rejected there too -- ``True`` is an ``int`` in Python and would
    otherwise parse as axis 1).

    Non-integral values are rejected for the same "be loud" reason: bare
    ``int(1.5)`` truncates to 1, which would hand the caller axis 1 for a value
    that names no axis at all. Integral floats (``1.0``) ARE accepted, since
    plain YAML scalars routinely deserialize as floats and ``1.0`` is
    unambiguous about which axis it means.
    """
    if isinstance(raw, bool):
        raise ValueError(f"transport_axis_index must be 0, 1, or 2; got {raw!r}")
    try:
        idx = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"transport_axis_index must be an int 0, 1, or 2; got {raw!r}"
        ) from exc
    if isinstance(raw, float) and not float(raw).is_integer():
        raise ValueError(
            f"transport_axis_index must be a whole number 0, 1, or 2; got {raw!r}"
        )
    if idx not in (0, 1, 2):
        raise ValueError(f"transport_axis_index must be 0, 1, or 2; got {idx}")
    return idx


def _parse_split_base_wrist_active_joints(raw: Any) -> tuple[int, ...] | None:
    """Parse/validate the ``split_base_wrist_active_joints`` value.

    ``None`` (the default) means "use the historical hardcoded base-joint set
    ``(0, 1, 2)`` = shoulder_pan/shoulder_lift/elbow" and is returned as-is --
    the caller, not this parser, substitutes that default, so a config that
    never mentions the field is provably the pre-existing code path.

    Otherwise: exactly 3 joint indices, each in ``[0, 5]`` (positions in
    ``constants.JOINT_NAME_ORDER``), no duplicates. Accepts any sequence
    (YAML list, tuple), returns a tuple of plain ``int``.

    Deliberately NOT permissive, for the same reason
    ``_parse_transport_axis_index`` above isn't: silently falling back to
    ``(0, 1, 2)`` on a malformed value would hand the caller the base-joint
    split while it believed it had selected a different joint set -- the same
    class of silent mismatch that field's docstring exists to rule out. Bools
    are rejected explicitly (``True`` is an ``int`` in Python and would
    otherwise parse as joint 1); integral floats (``1.0``) are accepted since
    plain YAML scalars routinely deserialize as floats, while non-integral
    ones are rejected rather than truncated.

    Exactly 3 is required (not "any subset"): the split task is the 3
    position rows of the Jacobian, so 3 active columns is what makes the
    reduced task square and the historical case a strict special case of it.
    Fewer would leave the translation task underdetermined; more has never
    been evaluated against the Lambda/nullspace math downstream. See
    ``CartesianImpedanceConfig.split_base_wrist_active_joints``.
    """
    if raw is None:
        return None
    if isinstance(raw, (str, bytes)) or not hasattr(raw, "__iter__"):
        raise ValueError(
            "split_base_wrist_active_joints must be a sequence of exactly 3 joint "
            f"indices in [0, 5]; got {raw!r}"
        )
    items = list(raw)
    if len(items) != 3:
        raise ValueError(
            "split_base_wrist_active_joints must have exactly 3 joint indices "
            f"(one per position-task row); got {len(items)}: {raw!r}"
        )
    parsed: list[int] = []
    for item in items:
        if isinstance(item, bool):
            raise ValueError(
                f"split_base_wrist_active_joints entries must be ints in [0, 5]; got {item!r}"
            )
        try:
            idx = int(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"split_base_wrist_active_joints entries must be ints in [0, 5]; got {item!r}"
            ) from exc
        if isinstance(item, float) and not float(item).is_integer():
            raise ValueError(
                "split_base_wrist_active_joints entries must be whole numbers in "
                f"[0, 5]; got {item!r}"
            )
        if not 0 <= idx <= 5:
            raise ValueError(
                f"split_base_wrist_active_joints entries must be in [0, 5]; got {idx}"
            )
        parsed.append(idx)
    if len(set(parsed)) != len(parsed):
        raise ValueError(
            "split_base_wrist_active_joints entries must be distinct joints; got "
            f"{tuple(parsed)}"
        )
    return tuple(parsed)


def _parse_split_base_wrist_task_dims(raw: Any) -> tuple[int, ...] | None:
    """Parse/validate the ``split_base_wrist_task_dims`` value.

    ``None`` (the default) means "use all 3 translation rows (0, 1, 2) = the
    historical behavior of ``split_base_wrist_task``" and is returned as-is --
    the caller, not this parser, substitutes that default, so a config that
    never mentions the field is provably the pre-existing code path (same
    contract as ``_parse_split_base_wrist_active_joints`` above).

    Otherwise: 1 to 3 distinct TRANSLATION row indices, each in ``[0, 2]``
    (0 = world X, 1 = world Y, 2 = world Z). Rotation rows are deliberately
    not selectable here: ``split_base_wrist_task`` drops the rotational wrench
    from this pipeline entirely by design (the wrist-only rotation
    sub-Jacobian is exactly singular at the poses this mechanism exists for --
    see ``CartesianImpedanceConfig.split_base_wrist_task``), so offering a
    rotation row would be offering something the mechanism does not implement.
    ``reduced_task_dims`` is the flag for arbitrary 6-row selection.

    Deliberately NOT permissive, for exactly the reason
    ``_parse_split_base_wrist_active_joints`` above isn't: silently falling
    back to all three rows on a malformed value would hand the caller a full
    3D translation task while it believed it had selected a 1D one -- and at
    the joint sets this field exists to enable (e.g. the structurally rank-2
    ``shoulder_lift/elbow/wrist_1`` planar sub-chain) that difference is the
    difference between a solvable task and a singular one. Bools are rejected
    explicitly (``True`` is an ``int`` in Python and would otherwise parse as
    row 1); integral floats (``1.0``) are accepted since plain YAML scalars
    routinely deserialize as floats, while non-integral ones are rejected
    rather than truncated.
    """
    if raw is None:
        return None
    if isinstance(raw, (str, bytes)) or not hasattr(raw, "__iter__"):
        raise ValueError(
            "split_base_wrist_task_dims must be a sequence of 1-3 distinct translation "
            f"row indices in [0, 2] (0=X, 1=Y, 2=Z); got {raw!r}"
        )
    items = list(raw)
    if not 1 <= len(items) <= 3:
        raise ValueError(
            "split_base_wrist_task_dims must have between 1 and 3 translation row "
            f"indices; got {len(items)}: {raw!r}"
        )
    parsed: list[int] = []
    for item in items:
        if isinstance(item, bool):
            raise ValueError(
                f"split_base_wrist_task_dims entries must be ints in [0, 2]; got {item!r}"
            )
        try:
            idx = int(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"split_base_wrist_task_dims entries must be ints in [0, 2]; got {item!r}"
            ) from exc
        if isinstance(item, float) and not float(item).is_integer():
            raise ValueError(
                "split_base_wrist_task_dims entries must be whole numbers in "
                f"[0, 2]; got {item!r}"
            )
        if not 0 <= idx <= 2:
            raise ValueError(
                "split_base_wrist_task_dims entries must be translation rows in [0, 2] "
                f"(0=X, 1=Y, 2=Z); got {idx}. Rotation rows are not selectable here -- "
                "split_base_wrist_task drops the rotational wrench by design; use "
                "reduced_task_dims for arbitrary 6-row selection."
            )
        parsed.append(idx)
    if len(set(parsed)) != len(parsed):
        raise ValueError(
            "split_base_wrist_task_dims entries must be distinct rows; got "
            f"{tuple(parsed)}"
        )
    return tuple(parsed)


def _parse_manipulability_cbf_epsilon(raw: Any) -> float:
    """Parse/validate ``manipulability_cbf_epsilon``: finite and strictly > 0.

    Deliberately NOT permissive, for the same reason
    ``_parse_transport_axis_index`` above isn't. ``epsilon <= 0`` puts the
    barrier's own boundary AT (or past) the singular set, where ``mu`` is not
    differentiable and the central-difference gradient straddles a kink -- the
    constraint row would then be built from a meaningless derivative while
    still looking perfectly healthy in a trace. A NaN would silently make
    every comparison in the filter False, i.e. a silently disabled safety
    mechanism. Both are exactly the "looks enabled, provably isn't" class this
    module's other parsers exist to reject.

    Only validated here, at config-construction time, rather than per cycle:
    the value never changes during a run and ``set_gains()`` cannot reach it.
    """
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"manipulability_cbf_epsilon must be a finite float > 0; got {raw!r}"
        ) from exc
    if not (value > 0.0) or not float(value) == float(value) or value == float("inf"):
        raise ValueError(
            f"manipulability_cbf_epsilon must be finite and > 0; got {value!r}"
        )
    return value


def _parse_manipulability_cbf_alpha(raw: Any, field_name: str) -> float:
    """Parse/validate one of the two high-order-CBF class-K gains.

    Must be finite and strictly > 0: a linear class-K function needs a
    positive slope to BE class-K, and ``alpha == 0`` collapses the second-order
    condition ``hddot + (a1+a2) hdot + a1 a2 h >= 0`` to ``hddot + a hdot >= 0``,
    which no longer references ``h`` at all -- the barrier would stop caring
    where it is, only how fast it is moving. Negative values invert the
    correction's sign and would actively drive the arm INTO the singularity.
    Same "be loud" call as ``_parse_manipulability_cbf_epsilon`` above.
    """
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite float > 0; got {raw!r}") from exc
    if not (value > 0.0) or value == float("inf"):
        raise ValueError(f"{field_name} must be finite and > 0; got {value!r}")
    return value


def _parse_friction_model(raw: Any) -> Literal["static", "lugre", "karnopp"]:
    """Case-insensitive parse of the ``friction_model`` YAML value.

    Matches the pre-existing (2026-08-01) permissive convention: any value that
    isn't a recognized model name silently falls back to ``"static"`` rather
    than raising, so a typo in a config never breaks loading -- it just leaves
    the historical default behavior in place. Extended 2026-08-02 to recognize
    ``"karnopp"`` alongside the pre-existing ``"lugre"``.
    """
    value = str(raw).lower()
    if value == "lugre":
        return "lugre"
    if value == "karnopp":
        return "karnopp"
    return "static"
