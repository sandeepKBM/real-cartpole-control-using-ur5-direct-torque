from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "simulation"))

from simulation.coppelia_lqr_transport import (  # noqa: E402
    build_coppelia_lqr_transport,
    compute_coppelia_lqr_outer_command,
)


def test_coppelia_lqr_transport_helper_builds_goal_and_command() -> None:
    controller, safety_filter, target_x = build_coppelia_lqr_transport(
        x_start=0.0,
        target_dx=0.10,
        dt_s=0.05,
        q_x=60.0,
        q_xdot=8.0,
        r_weight=1.0,
        accel_limit=0.20,
        velocity_limit=0.12,
        command_change_limit=0.20,
        guardrail_margin_m=0.05,
    )

    assert np.isfinite(target_x)
    assert abs(target_x - 0.10) < 1e-12

    accel_cmd, diag = compute_coppelia_lqr_outer_command(
        controller,
        safety_filter,
        x_now=0.0,
        x_dot_now=0.0,
        time_s=0.0,
        dt_s=0.05,
        target_x=target_x,
    )

    assert np.isfinite(accel_cmd)
    assert accel_cmd > 0.0
    assert abs(accel_cmd) <= 0.20 + 1e-12
    assert diag["severity"] in ("ok", "warning", "clipped")
    assert diag["raw_command"]["metadata"]["controller"] == "fixed_x_transport_lqr"
