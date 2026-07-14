"""Deadline-scheduled direct-torque transport timing (mocked RTDE)."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from hardware.direct_torque_transport import run_x_transport_direct_torque
from hardware.link import UR5eState
from hardware.poses import HEIGHT_ALPHA_0_5_Q

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / "config" / "ur5e_mujoco_torque_osc_tuned.yaml"


class _FastMockLink:
    def __init__(self) -> None:
        self._tcp_x = 0.4

    def connect(self) -> None:
        return None

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
    def compose_robot_state(link_state, *, jacobian, mass_matrix, time_s, target_x, target_x_vel):
        from hardware.direct_torque_link import UR5eDirectTorqueLink

        return UR5eDirectTorqueLink.compose_robot_state(
            link_state,
            jacobian=jacobian,
            mass_matrix=mass_matrix,
            time_s=time_s,
            target_x=target_x,
            target_x_vel=target_x_vel,
        )

    def build_robot_state(self, link_state, *, time_s, target_x, target_x_vel):
        return self.compose_robot_state(
            link_state,
            jacobian=self.get_jacobian(),
            mass_matrix=self.get_mass_matrix(),
            time_s=time_s,
            target_x=target_x,
            target_x_vel=target_x_vel,
        )

    def safe_stop(self, reason: str) -> None:
        return None


@pytest.mark.hardware
def test_transport_records_timing_and_deadline_loop(tmp_path: Path) -> None:
    result = run_x_transport_direct_torque(
        _FastMockLink(),  # type: ignore[arg-type]
        config_path=CONFIG,
        target_x_delta_m=0.01,
        move_duration_s=0.1,
        duration_s=0.2,
        output_dir=tmp_path,
        motion_opt_in=True,
        record_latency=True,
        dynamics_source="local",
    )
    assert result.summary["dynamics_source"] == "local"
    timing = result.summary["timing"]
    phases = result.summary["latency_phases"]
    assert timing["cycle_count"] > 0
    assert phases["phase_count"] == timing["cycle_count"]
    assert "work_duration" in timing
    assert timing["work_duration"]["mean_ms"] is not None
    assert phases.get("controller_mean_ms") is not None
    assert phases.get("dominant_phase") in {
        "read_state",
        "get_jacobian",
        "get_mass_matrix",
        "build_state",
        "controller",
        "safety",
        "direct_torque",
    }
