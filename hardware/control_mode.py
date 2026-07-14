"""Hardware control mode selection for X transport drivers."""

from __future__ import annotations

HARDWARE_CONTROL_MODES = frozenset({"position", "direct_torque", "urscript"})


def normalize_control_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode not in HARDWARE_CONTROL_MODES:
        raise ValueError(
            f"control_mode must be one of {sorted(HARDWARE_CONTROL_MODES)}; got {value!r}"
        )
    return mode
