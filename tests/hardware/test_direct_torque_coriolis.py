"""Tests for coriolis_feedforward on the direct-torque hardware transport loop.

Universal Robots' own Direct Torque Control docs confirm the firmware
auto-compensates gravity inside directTorque() but only *exposes* (does not
apply) Coriolis/centrifugal values -- so unlike gravity, Python has to add
this term itself if it wants it. Default off (never validated on real
hardware); mocked RTDE only, real MuJoCo math for the Coriolis computation
itself (dynamics_source='local').
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from hardware.direct_torque_link import UR5eDirectTorqueLink  # noqa: E402
from hardware.direct_torque_transport import run_x_transport_direct_torque  # noqa: E402
from hardware.link import UR5eState  # noqa: E402
from hardware.poses import HEIGHT_ALPHA_0_5_Q  # noqa: E402
from hardware.safety import UR5eSafetyLimits  # noqa: E402

CONFIG = REPO_ROOT / "config" / "ur5e_mujoco_torque_osc_tuned.yaml"


class _MockDTLinkNonzeroVelocity:
    """Mock UR5eDirectTorqueLink reporting a nonzero qd throughout, so a
    Coriolis/centrifugal term is actually nontrivial to compute -- the whole
    point of these tests is confirming that term shows up (or doesn't) in the
    trace and the commanded torque, not just that the flag is accepted."""

    def __init__(self) -> None:
        self._tcp_x = 0.4
        self._qd = np.array([0.3, -0.2, 0.4, 0.1, -0.3, 0.2])
        self.limits = UR5eSafetyLimits()

    def connect(self) -> None:
        pass

    def read_state(self) -> UR5eState:
        return UR5eState(
            q=HEIGHT_ALPHA_0_5_Q.copy(),
            qd=self._qd.copy(),
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
    def compose_robot_state(link_state, *, jacobian, mass_matrix, time_s, target_x, target_x_vel, dt_s=None, target_x_accel=None, transport_axis_index=0):
        return UR5eDirectTorqueLink.compose_robot_state(
            link_state, jacobian=jacobian, mass_matrix=mass_matrix,
            time_s=time_s, target_x=target_x, target_x_vel=target_x_vel, dt_s=dt_s,
            target_x_accel=target_x_accel, transport_axis_index=transport_axis_index,
        )

    def build_robot_state(self, link_state, *, time_s, target_x, target_x_vel, dt_s=None, transport_axis_index=0):
        return self.compose_robot_state(
            link_state, jacobian=self.get_jacobian(), mass_matrix=self.get_mass_matrix(),
            time_s=time_s, target_x=target_x, target_x_vel=target_x_vel, dt_s=dt_s,
            transport_axis_index=transport_axis_index,
        )

    def safe_stop(self, reason: str) -> None:
        pass


@pytest.mark.hardware
def test_coriolis_feedforward_off_by_default(tmp_path: Path) -> None:
    link = _MockDTLinkNonzeroVelocity()
    result = run_x_transport_direct_torque(
        link,  # type: ignore[arg-type]
        config_path=CONFIG, target_x_delta_m=0.01,
        move_duration_s=0.02, duration_s=0.03, output_dir=tmp_path,
        motion_opt_in=True, record_latency=False, dynamics_source="local",
    )
    assert result.trace_path is not None
    rows = [__import__("json").loads(line) for line in result.trace_path.read_text().splitlines()]
    assert rows, "expected at least one trace row"
    for row in rows:
        assert row["coriolis_feedforward_active"] is False
        np.testing.assert_allclose(row["tau_coriolis"], np.zeros(6), atol=1e-12)
        np.testing.assert_allclose(row["tau_applied"], row["tau_controller"], atol=1e-12)


@pytest.mark.hardware
def test_coriolis_feedforward_on_adds_nonzero_term(tmp_path: Path) -> None:
    link = _MockDTLinkNonzeroVelocity()
    result = run_x_transport_direct_torque(
        link,  # type: ignore[arg-type]
        config_path=CONFIG, target_x_delta_m=0.01,
        move_duration_s=0.02, duration_s=0.03, output_dir=tmp_path,
        motion_opt_in=True, record_latency=False, dynamics_source="local",
        coriolis_feedforward=True,
    )
    assert result.trace_path is not None
    rows = [__import__("json").loads(line) for line in result.trace_path.read_text().splitlines()]
    assert rows, "expected at least one trace row"
    saw_nonzero = False
    for row in rows:
        assert row["coriolis_feedforward_active"] is True
        tau_controller = np.asarray(row["tau_controller"])
        tau_coriolis = np.asarray(row["tau_coriolis"])
        tau_applied = np.asarray(row["tau_applied"])
        np.testing.assert_allclose(tau_applied, tau_controller + tau_coriolis, atol=1e-9)
        if float(np.max(np.abs(tau_coriolis))) > 1e-6:
            saw_nonzero = True
    assert saw_nonzero, "expected a nonzero Coriolis/centrifugal term at this nonzero qd"


@pytest.mark.hardware
def test_coriolis_feedforward_requires_local_dynamics_source(tmp_path: Path) -> None:
    link = _MockDTLinkNonzeroVelocity()
    with pytest.raises(ValueError, match="dynamics_source"):
        run_x_transport_direct_torque(
            link,  # type: ignore[arg-type]
            config_path=CONFIG, target_x_delta_m=0.01,
            move_duration_s=0.02, duration_s=0.03, output_dir=tmp_path,
            motion_opt_in=True, record_latency=False, dynamics_source="rtde",
            coriolis_feedforward=True,
        )
