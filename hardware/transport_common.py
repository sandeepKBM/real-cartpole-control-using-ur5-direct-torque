"""Shared parsing/summary helpers for the X-transport drivers.

These are pure extractions of code that was duplicated verbatim across
``direct_torque_transport.py``, ``urscript_transport.py``, and
``position_transport.py``. Nothing here changes any numeric default, check
ordering, or control-loop behaviour -- each helper reproduces exactly what the
call sites did inline.
"""

from __future__ import annotations

from typing import Any

from controller_core.safety import ImpedanceSafetyConfig


def impedance_safety_config_from_section(safety_raw: dict[str, Any] | None) -> ImpedanceSafetyConfig:
    """Build an ``ImpedanceSafetyConfig`` from a ``controller.safety`` YAML
    section, using the hardware lane's own defaults.

    NOTE the ``max_joint_velocity_radps`` default is 3.0 here -- deliberately
    the value both original loaders passed, NOT ``ImpedanceSafetyConfig``'s own
    dataclass default of 1.5. Every field/default below is byte-identical to the
    inline construction previously duplicated in
    ``direct_torque_transport._load_impedance_bundle`` and
    ``urscript_transport._load_safety_cfg``.
    """
    safety_raw = safety_raw or {}
    return ImpedanceSafetyConfig(
        max_abs_y_drift_m=float(safety_raw.get("max_abs_y_drift_m", 0.03)),
        max_abs_z_drift_m=float(safety_raw.get("max_abs_z_drift_m", 0.03)),
        max_abs_orthogonal_drift_m=float(safety_raw.get("max_abs_orthogonal_drift_m", 0.03)),
        max_orientation_error_rad=float(safety_raw.get("max_orientation_error_rad", 0.25)),
        max_joint_velocity_radps=float(safety_raw.get("max_joint_velocity_radps", 3.0)),
    )


def max_abs_qd_from_trace(trace_rows: list[dict[str, Any]]) -> float:
    """Largest ``|qd|`` seen across every trace row's ``"qd"`` list.

    Byte-identical to the expression previously duplicated in
    ``position_transport`` and ``direct_torque_transport`` summary building:
    a missing ``"qd"`` key falls back to six zeros, and an empty trace yields
    ``0.0``.
    """
    return float(
        max(
            (max(abs(v) for v in row.get("qd", [0.0] * 6)) for row in trace_rows),
            default=0.0,
        )
    )
