"""Controller-internal diagnostics (jacobian_cond, task_backtrack_iters,
task_scale, singular_scale) must reach ``trace_rows`` -- previously computed
by ``controller.compute()`` every cycle but silently discarded, which blocked
the 2026-07-28 velocity-overshoot investigation (see
docs/status/clock_timing_late_cycles_2026-07-28.md)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from hardware.direct_torque_link import UR5eDirectTorqueLink
from hardware.direct_torque_transport import run_x_transport_direct_torque
from hardware.link import UR5eState
from hardware.poses import HEIGHT_ALPHA_0_5_Q
from hardware.safety import UR5eSafetyLimits

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / "config" / "ur5e_mujoco_torque_osc_tuned.yaml"


class _MockDTLink:
    def __init__(self) -> None:
        self._tcp_x = 0.35  # off target so kp_x actually produces a nonzero Fx
        self.limits = UR5eSafetyLimits()
        self.connect_calls = 0

    def connect(self) -> None:
        self.connect_calls += 1

    def read_state(self) -> UR5eState:
        return UR5eState(
            q=HEIGHT_ALPHA_0_5_Q.copy(),
            qd=np.zeros(6),
            tcp_pose=np.array([self._tcp_x, -0.2, 0.3, 0.0, 3.14, 0.0]),
            host_stamp_ns=time.monotonic_ns(),
            robot_timestamp_s=None,
            safety_status=None,
        )

    def get_jacobian(self) -> np.ndarray:
        return np.eye(6)

    def get_mass_matrix(self) -> np.ndarray:
        return np.eye(6)

    def direct_torque(self, tau_nm: np.ndarray, *, friction_comp: bool = True) -> None:
        self._tcp_x += float(tau_nm[0]) * 1e-6

    @staticmethod
    def compose_robot_state(link_state, *, jacobian, mass_matrix, time_s, target_x, target_x_vel, dt_s=None, target_x_accel=None):
        return UR5eDirectTorqueLink.compose_robot_state(
            link_state, jacobian=jacobian, mass_matrix=mass_matrix,
            time_s=time_s, target_x=target_x, target_x_vel=target_x_vel, dt_s=dt_s,
            target_x_accel=target_x_accel,
        )

    def build_robot_state(self, link_state, *, time_s, target_x, target_x_vel, dt_s=None):
        return self.compose_robot_state(
            link_state, jacobian=self.get_jacobian(), mass_matrix=self.get_mass_matrix(),
            time_s=time_s, target_x=target_x, target_x_vel=target_x_vel, dt_s=dt_s,
        )

    def safe_stop(self, reason: str) -> None:
        pass


@pytest.mark.hardware
def test_trace_rows_include_controller_diagnostics(tmp_path: Path) -> None:
    link = _MockDTLink()
    result = run_x_transport_direct_torque(
        link,  # type: ignore[arg-type]
        config_path=CONFIG, target_x_delta_m=0.01,
        move_duration_s=0.02, duration_s=0.03, output_dir=tmp_path,
        motion_opt_in=True, record_latency=False, dynamics_source="local",
    )
    assert result.trace_path is not None
    rows = [json.loads(line) for line in result.trace_path.read_text().splitlines()]
    assert rows, "expected at least one trace row"

    diagnostic_fields = ("jacobian_cond", "singular_scale", "task_scale", "task_backtrack_iters")
    row = rows[0]
    for field_name in diagnostic_fields:
        assert field_name in row, f"trace row missing controller diagnostic field {field_name!r}"
        assert row[field_name] is not None, f"trace row field {field_name!r} is None, not populated"

    # jacobian_cond/singular_scale/task_scale are floats, task_backtrack_iters
    # is an int -- confirm they came through as real numeric values, not
    # placeholders or NaN.
    assert np.isfinite(row["jacobian_cond"])
    assert np.isfinite(row["singular_scale"])
    assert np.isfinite(row["task_scale"])
    assert isinstance(row["task_backtrack_iters"], int)
    assert row["task_backtrack_iters"] >= 0
