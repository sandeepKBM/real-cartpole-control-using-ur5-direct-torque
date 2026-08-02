"""Wiring tests for the diagnostic-only direct_torque dynamics residual
observer (2026-07-29) -- see
docs/status/direct_torque_residual_observer_2026-07-29.md.

These tests only check that the new trace_rows fields are populated with the
right shapes/types and that `enable_residual_observer=False` disables them
cleanly -- the actual physical validation (near-zero residual for clean
motion, detectable residual under an injected disturbance) uses real MuJoCo
dynamics and lives in tests/mujoco/test_direct_torque_residual_observer.py,
since this module's mock link has no real physics behind it (identity mass
matrix, kinematics-free state updates)."""

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
        self._tcp_x = 0.35
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
def test_trace_rows_include_residual_fields_by_default(tmp_path: Path) -> None:
    link = _MockDTLink()
    result = run_x_transport_direct_torque(
        link,  # type: ignore[arg-type]
        config_path=CONFIG, target_x_delta_m=0.01,
        move_duration_s=0.02, duration_s=0.03, output_dir=tmp_path,
        motion_opt_in=True, record_latency=False, dynamics_source="local",
        residual_qdd_gap_cycles=2,
    )
    assert result.trace_path is not None
    rows = [json.loads(line) for line in result.trace_path.read_text().splitlines()]
    assert len(rows) >= 3, "need at least gap_cycles+1 rows to exercise a populated residual"

    for field_name in ("qdd_pred", "qdd_measured", "qdd_residual", "qdd_residual_norm"):
        assert field_name in rows[0], f"trace row missing residual field {field_name!r}"

    # qdd_pred is always populated (needs no history).
    for row in rows:
        assert row["qdd_pred"] is not None
        assert len(row["qdd_pred"]) == 6
        assert all(np.isfinite(v) for v in row["qdd_pred"])

    # qdd_measured/qdd_residual are None while the gap window fills, then
    # populated -- with gap_cycles=2, the first 2 rows are None.
    assert rows[0]["qdd_measured"] is None
    assert rows[0]["qdd_residual"] is None
    assert rows[0]["qdd_residual_norm"] is None
    later_populated = [r for r in rows if r["qdd_measured"] is not None]
    assert later_populated, "expected qdd_measured to become populated once the gap window fills"
    for row in later_populated:
        assert len(row["qdd_measured"]) == 6
        assert len(row["qdd_residual"]) == 6
        assert isinstance(row["qdd_residual_norm"], float)
        assert np.isfinite(row["qdd_residual_norm"])
        np.testing.assert_allclose(
            row["qdd_residual_norm"],
            float(np.linalg.norm(np.asarray(row["qdd_residual"]) )),
            atol=1e-9,
        )


@pytest.mark.hardware
def test_residual_observer_disabled_flag_yields_none_fields(tmp_path: Path) -> None:
    link = _MockDTLink()
    result = run_x_transport_direct_torque(
        link,  # type: ignore[arg-type]
        config_path=CONFIG, target_x_delta_m=0.01,
        move_duration_s=0.02, duration_s=0.03, output_dir=tmp_path,
        motion_opt_in=True, record_latency=False, dynamics_source="local",
        enable_residual_observer=False,
    )
    assert result.trace_path is not None
    rows = [json.loads(line) for line in result.trace_path.read_text().splitlines()]
    assert rows
    for row in rows:
        assert row["qdd_pred"] is None
        assert row["qdd_measured"] is None
        assert row["qdd_residual"] is None
        assert row["qdd_residual_norm"] is None
