"""Tests for hardware direct-torque helpers."""

from __future__ import annotations

import time

import numpy as np

from controller_core.kinematics_utils import rotvec_to_quat_wxyz
from controller_core.state_types import as_impedance_robot_state
from hardware.direct_torque_link import UR5eDirectTorqueLink
from hardware.link import UR5eState


def test_rotvec_zero_is_identity_quat() -> None:
    q = rotvec_to_quat_wxyz(np.zeros(3))
    np.testing.assert_allclose(q, [1.0, 0.0, 0.0, 0.0], atol=1e-12)


def _link_state() -> UR5eState:
    return UR5eState(
        q=np.zeros(6),
        qd=np.zeros(6),
        tcp_pose=np.array([0.4, -0.2, 0.3, 0.0, 3.14, 0.0]),
        host_stamp_ns=time.monotonic_ns(),
        robot_timestamp_s=None,
        safety_status=None,
    )


def test_compose_robot_state_includes_real_dt_s_when_provided() -> None:
    """This is the real hardware layer-3 fix: compose_robot_state() must put
    the caller-supplied real per-cycle dt into the dict it returns, and that
    value must survive as_impedance_robot_state()'s normalization (the same
    layer-2 pass-through fixed in controller_core/state_types.py) so
    compute() could read it via state.get("dt_s")."""
    state = UR5eDirectTorqueLink.compose_robot_state(
        _link_state(),
        jacobian=np.eye(6),
        mass_matrix=np.eye(6),
        time_s=0.0,
        target_x=0.4,
        target_x_vel=0.0,
        dt_s=0.002,
    )
    assert state["dt_s"] == 0.002

    normalized = as_impedance_robot_state(state)
    assert normalized["dt_s"] == 0.002


def test_build_robot_state_forwards_dt_s() -> None:
    class _MockLink(UR5eDirectTorqueLink):
        def __init__(self) -> None:  # pragma: no cover - trivial
            pass

        def get_jacobian(self) -> np.ndarray:
            return np.eye(6)

        def get_mass_matrix(self) -> np.ndarray:
            return np.eye(6)

    state = _MockLink().build_robot_state(
        _link_state(), time_s=0.0, target_x=0.4, target_x_vel=0.0, dt_s=0.0021
    )
    assert state["dt_s"] == 0.0021


def test_compose_robot_state_dt_s_omitted_stays_none_and_is_dropped() -> None:
    """No caller-supplied dt_s (the pre-existing default) must not surface a
    ``dt_s`` key downstream -- byte-identical to before this plumbing was
    added, for any caller that doesn't yet pass one."""
    state = UR5eDirectTorqueLink.compose_robot_state(
        _link_state(),
        jacobian=np.eye(6),
        mass_matrix=np.eye(6),
        time_s=0.0,
        target_x=0.4,
        target_x_vel=0.0,
    )
    assert state["dt_s"] is None

    normalized = as_impedance_robot_state(state)
    assert "dt_s" not in normalized
