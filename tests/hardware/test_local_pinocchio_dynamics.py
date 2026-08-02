"""Parity + wiring tests for the opt-in Pinocchio-backed local dynamics fast
path (`LocalPinocchioFastDynamics`, `dynamics_source="local_pinocchio"`).

See docs/status/local_dynamics_speedup_investigation_2026-07-29.md for the
benchmark and the world-frame correction this required. These tests confirm:
  1. LocalPinocchioFastDynamics's J/M/Coriolis match LocalMujocoDynamics
     (the existing, validated hot-path implementation) within tolerance.
  2. The legacy `LocalPinocchioDynamics` alias is untouched -- still
     MuJoCo-backed, still `is LocalMujocoDynamics`.
  3. `dynamics_source="local_pinocchio"` is accepted by normalize/DYNAMICS_SOURCES
     and actually wires up through `run_x_transport_direct_torque`.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

mujoco = pytest.importorskip("mujoco")
pin = pytest.importorskip("pinocchio")

from hardware.local_dynamics import (  # noqa: E402
    DEFAULT_SCENE_XML,
    DYNAMICS_SOURCES,
    LocalMujocoDynamics,
    LocalPinocchioDynamics,
    LocalPinocchioFastDynamics,
    normalize_dynamics_source,
)

CONFIG = REPO_ROOT / "config" / "ur5e_mujoco_torque_osc_tuned.yaml"


def test_local_pinocchio_dynamics_alias_unchanged() -> None:
    # Must remain a MuJoCo-backed alias -- do not repurpose this name.
    assert LocalPinocchioDynamics is LocalMujocoDynamics


def test_dynamics_sources_includes_local_pinocchio() -> None:
    assert DYNAMICS_SOURCES == frozenset({"rtde", "local", "local_pinocchio"})
    assert normalize_dynamics_source("local_pinocchio") == "local_pinocchio"
    assert normalize_dynamics_source("LOCAL_PINOCCHIO") == "local_pinocchio"


@pytest.fixture(scope="module")
def engines():
    mj = LocalMujocoDynamics(scene_xml=DEFAULT_SCENE_XML)
    fast = LocalPinocchioFastDynamics()
    return mj, fast


@pytest.mark.hardware
def test_fast_jacobian_and_mass_match_mujoco(engines) -> None:
    mj, fast = engines
    rng = np.random.default_rng(0)
    lo = mj.model.jnt_range[:, 0].copy()
    hi = mj.model.jnt_range[:, 1].copy()
    unlimited = lo >= hi
    lo[unlimited], hi[unlimited] = -np.pi, np.pi

    for _ in range(40):
        q = rng.uniform(lo[:6], hi[:6])
        J_mj, M_mj = mj.jacobian_and_mass_matrix(q)
        J_fast, M_fast = fast.jacobian_and_mass_matrix(q)
        np.testing.assert_allclose(J_fast, J_mj, rtol=0, atol=1e-6)
        np.testing.assert_allclose(M_fast, M_mj, rtol=0, atol=1e-6)


@pytest.mark.hardware
def test_fast_coriolis_matches_mujoco(engines) -> None:
    mj, fast = engines
    rng = np.random.default_rng(1)
    saw_nonzero = False
    for _ in range(20):
        q = rng.uniform(-1.0, 1.0, size=6)
        qd = rng.uniform(-1.5, 1.5, size=6)
        c_mj = mj.coriolis(q, qd)
        c_fast = fast.coriolis(q, qd)
        np.testing.assert_allclose(c_fast, c_mj, atol=1e-5)
        if float(np.max(np.abs(c_fast))) > 1e-4:
            saw_nonzero = True
    assert saw_nonzero, "expected nonzero Coriolis torque for some sample"


@pytest.mark.hardware
def test_fast_jacobian_mass_and_coriolis_matches_separate_calls(engines) -> None:
    _, fast = engines
    rng = np.random.default_rng(2)
    for _ in range(10):
        q = rng.uniform(-1.0, 1.0, size=6)
        qd = rng.uniform(-1.0, 1.0, size=6)
        J_combined, M_combined, C_combined = fast.jacobian_mass_and_coriolis(q, qd)
        J_separate, M_separate = fast.jacobian_and_mass_matrix(q)
        C_separate = fast.coriolis(q, qd)
        np.testing.assert_allclose(J_combined, J_separate)
        np.testing.assert_allclose(M_combined, M_separate)
        np.testing.assert_allclose(C_combined, C_separate)


class _MockDTLinkNonzeroVelocity:
    """Same mock shape as tests/hardware/test_direct_torque_coriolis.py's --
    duplicated locally (small, self-contained) rather than imported, to keep
    this file independent of that one's internal test fixtures."""

    def __init__(self) -> None:
        from hardware.poses import HEIGHT_ALPHA_0_5_Q
        from hardware.safety import UR5eSafetyLimits

        self._tcp_x = 0.4
        self._qd = np.array([0.3, -0.2, 0.4, 0.1, -0.3, 0.2])
        self._q = HEIGHT_ALPHA_0_5_Q.copy()
        self.limits = UR5eSafetyLimits()

    def connect(self) -> None:
        pass

    def read_state(self):
        from hardware.link import UR5eState

        return UR5eState(
            q=self._q.copy(),
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
    def compose_robot_state(link_state, *, jacobian, mass_matrix, time_s, target_x, target_x_vel, dt_s=None, target_x_accel=None):
        from hardware.direct_torque_link import UR5eDirectTorqueLink

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
def test_direct_torque_transport_accepts_local_pinocchio(tmp_path: Path) -> None:
    from hardware.direct_torque_transport import run_x_transport_direct_torque

    link = _MockDTLinkNonzeroVelocity()
    result = run_x_transport_direct_torque(
        link,  # type: ignore[arg-type]
        config_path=CONFIG, target_x_delta_m=0.01,
        move_duration_s=0.02, duration_s=0.03, output_dir=tmp_path,
        motion_opt_in=True, record_latency=False, dynamics_source="local_pinocchio",
        coriolis_feedforward=True,
    )
    assert result.trace_path is not None
    rows = [json.loads(line) for line in result.trace_path.read_text().splitlines()]
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
    assert json.loads((tmp_path / "summary.json").read_text())["dynamics_source"] == "local_pinocchio"
