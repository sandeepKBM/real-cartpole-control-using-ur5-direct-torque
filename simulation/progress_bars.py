"""
Standard tqdm setup for MuJoCo / rendering loops that run for many frames.

Import `simulation_progress` anywhere a long-running `for` loop would otherwise
run silent or with a minimal bar.
"""

from __future__ import annotations

from typing import Any

from tqdm import tqdm


def simulation_progress(
    total: int,
    desc: str,
    *,
    unit: str = "frame",
    mininterval: float = 0.2,
    **kwargs: Any,
) -> tqdm:
    """
    Progress bar with ETA and rate; postfix is updated by the caller each step.
    """
    return tqdm(
        total=total,
        desc=desc,
        unit=unit,
        dynamic_ncols=True,
        mininterval=mininterval,
        smoothing=0.05,
        leave=True,
        **kwargs,
    )
