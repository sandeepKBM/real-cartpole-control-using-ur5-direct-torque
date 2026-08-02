"""Tests for hardware/telemetry_gap_bridge.py -- pure numpy, no RTDE dependency."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from hardware.telemetry_gap_bridge import TelemetryGapBridge  # noqa: E402

DT_S = 0.002  # 500 Hz, matches direct_torque's loop rate


def _identity_mass_matrix() -> np.ndarray:
    return np.eye(6, dtype=np.float64)


def _zero6() -> np.ndarray:
    return np.zeros(6, dtype=np.float64)


def _base_pose() -> np.ndarray:
    return np.array([0.1, 0.2, 0.9, 0.0, 3.14, 0.0], dtype=np.float64)


def _base_q() -> np.ndarray:
    return np.array([0.0, -0.8, -1.2, -1.0, 0.2, 0.0], dtype=np.float64)


def _identity_jacobian() -> np.ndarray:
    # First 3 rows = identity position Jacobian (unit velocity gain per
    # joint), last 3 rows = zeros -- simplest jacobian that still exercises
    # the position-integration path deterministically.
    jac = np.zeros((6, 6), dtype=np.float64)
    jac[:3, :3] = np.eye(3)
    return jac


def test_first_cycle_never_bridges_even_if_it_looks_like_a_duplicate():
    bridge = TelemetryGapBridge(max_bridge_cycles=2)
    q, qd, pose = _base_q(), _zero6(), _base_pose()
    result = bridge.process(
        q=q,
        qd=qd,
        tcp_pose=pose,
        robot_timestamp_s=1.0,
        tau_applied=_zero6(),
        mass_matrix=_identity_mass_matrix(),
        coriolis_term=_zero6(),
        jacobian=_identity_jacobian(),
        dt_s=DT_S,
    )
    assert result.bridged is False
    assert result.consecutive_duplicates == 0
    np.testing.assert_array_equal(result.q, q)
    np.testing.assert_array_equal(result.tcp_pose, pose)


def test_non_duplicate_reads_never_bridge_and_update_the_anchor():
    bridge = TelemetryGapBridge(max_bridge_cycles=2)
    q0, pose0 = _base_q(), _base_pose()
    bridge.process(
        q=q0, qd=_zero6(), tcp_pose=pose0, robot_timestamp_s=1.0,
        tau_applied=_zero6(), mass_matrix=_identity_mass_matrix(),
        coriolis_term=_zero6(), jacobian=_identity_jacobian(), dt_s=DT_S,
    )
    q1 = q0 + 0.01
    pose1 = pose0.copy()
    pose1[0] += 0.001
    result = bridge.process(
        q=q1, qd=_zero6(), tcp_pose=pose1, robot_timestamp_s=1.002,
        tau_applied=_zero6(), mass_matrix=_identity_mass_matrix(),
        coriolis_term=_zero6(), jacobian=_identity_jacobian(), dt_s=DT_S,
    )
    assert result.bridged is False
    np.testing.assert_array_equal(result.q, q1)
    np.testing.assert_array_equal(result.tcp_pose, pose1)


def test_single_duplicate_cycle_is_bridged_with_forward_dynamics_prediction():
    bridge = TelemetryGapBridge(max_bridge_cycles=2)
    q0, pose0 = _base_q(), _base_pose()
    qd0 = np.array([0.0, 0.1, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    bridge.process(
        q=q0, qd=qd0, tcp_pose=pose0, robot_timestamp_s=1.0,
        tau_applied=_zero6(), mass_matrix=_identity_mass_matrix(),
        coriolis_term=_zero6(), jacobian=_identity_jacobian(), dt_s=DT_S,
    )
    # Duplicate: identical q and tcp_pose, same timestamp.
    tau = np.array([0.0, 2.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)  # accelerates joint 1
    result = bridge.process(
        q=q0, qd=qd0, tcp_pose=pose0, robot_timestamp_s=1.0,
        tau_applied=tau, mass_matrix=_identity_mass_matrix(),
        coriolis_term=_zero6(), jacobian=_identity_jacobian(), dt_s=DT_S,
    )
    assert result.bridged is True
    assert result.consecutive_duplicates == 1
    # With M=I, coriolis=0: qdd_pred = tau = [0,2,0,0,0,0].
    expected_qd = qd0 + np.array([0.0, 2.0, 0.0, 0.0, 0.0, 0.0]) * DT_S
    np.testing.assert_allclose(result.qd, expected_qd, atol=1e-12)
    expected_q = q0 + qd0 * DT_S + 0.5 * np.array([0.0, 2.0, 0.0, 0.0, 0.0, 0.0]) * DT_S**2
    np.testing.assert_allclose(result.q, expected_q, atol=1e-12)
    # Position rows of jacobian are identity on joints 0:3 -- joint 1's
    # velocity feeds tcp_pose[1] (Y), not X, and orientation stays frozen.
    assert result.tcp_pose[3:6] == pytest.approx(pose0[3:6])


def test_bridging_stops_beyond_max_bridge_cycles_and_defers_to_raw():
    bridge = TelemetryGapBridge(max_bridge_cycles=2)
    q0, pose0, qd0 = _base_q(), _base_pose(), _zero6()
    common = dict(
        q=q0, qd=qd0, tcp_pose=pose0, robot_timestamp_s=1.0,
        tau_applied=np.array([1.0, 0, 0, 0, 0, 0]), mass_matrix=_identity_mass_matrix(),
        coriolis_term=_zero6(), jacobian=_identity_jacobian(), dt_s=DT_S,
    )
    bridge.process(**common)  # cycle 0: real, sets anchor
    r1 = bridge.process(**common)  # cycle 1: duplicate #1, bridged
    r2 = bridge.process(**common)  # cycle 2: duplicate #2, bridged (== max_bridge_cycles)
    r3 = bridge.process(**common)  # cycle 3: duplicate #3, exceeds max_bridge_cycles
    assert r1.bridged is True
    assert r2.bridged is True
    assert r3.bridged is False
    # Beyond the window, the raw (frozen) reading passes through unchanged --
    # this is deliberate: StaleStateMonitor is the real backstop from here.
    np.testing.assert_array_equal(r3.q, q0)
    np.testing.assert_array_equal(r3.tcp_pose, pose0)
    assert r3.consecutive_duplicates == 3


def test_a_real_reading_after_a_bridged_run_resets_the_duplicate_counter():
    bridge = TelemetryGapBridge(max_bridge_cycles=2)
    q0, pose0, qd0 = _base_q(), _base_pose(), _zero6()
    dup_kwargs = dict(
        q=q0, qd=qd0, tcp_pose=pose0, robot_timestamp_s=1.0,
        tau_applied=_zero6(), mass_matrix=_identity_mass_matrix(),
        coriolis_term=_zero6(), jacobian=_identity_jacobian(), dt_s=DT_S,
    )
    bridge.process(**dup_kwargs)
    bridge.process(**dup_kwargs)  # bridged, consecutive_duplicates=1
    fresh = dict(dup_kwargs)
    fresh["q"] = q0 + 0.05
    fresh["robot_timestamp_s"] = 1.002
    result = bridge.process(**fresh)
    assert result.bridged is False
    assert result.consecutive_duplicates == 0
    # And a subsequent duplicate of THIS new reading bridges again from cycle 1.
    dup2 = dict(fresh)
    result2 = bridge.process(**dup2)
    assert result2.bridged is True
    assert result2.consecutive_duplicates == 1


def test_falls_back_to_value_equality_when_no_robot_timestamp_available():
    bridge = TelemetryGapBridge(max_bridge_cycles=2)
    q0, pose0, qd0 = _base_q(), _base_pose(), _zero6()
    bridge.process(
        q=q0, qd=qd0, tcp_pose=pose0, robot_timestamp_s=None,
        tau_applied=_zero6(), mass_matrix=_identity_mass_matrix(),
        coriolis_term=_zero6(), jacobian=_identity_jacobian(), dt_s=DT_S,
    )
    result = bridge.process(
        q=q0, qd=qd0, tcp_pose=pose0, robot_timestamp_s=None,
        tau_applied=np.array([1.0, 0, 0, 0, 0, 0]), mass_matrix=_identity_mass_matrix(),
        coriolis_term=_zero6(), jacobian=_identity_jacobian(), dt_s=DT_S,
    )
    assert result.bridged is True


def test_non_matching_values_are_not_treated_as_duplicate_without_timestamp():
    bridge = TelemetryGapBridge(max_bridge_cycles=2)
    q0, pose0, qd0 = _base_q(), _base_pose(), _zero6()
    bridge.process(
        q=q0, qd=qd0, tcp_pose=pose0, robot_timestamp_s=None,
        tau_applied=_zero6(), mass_matrix=_identity_mass_matrix(),
        coriolis_term=_zero6(), jacobian=_identity_jacobian(), dt_s=DT_S,
    )
    q1 = q0.copy()
    q1[0] += 1e-3
    result = bridge.process(
        q=q1, qd=qd0, tcp_pose=pose0, robot_timestamp_s=None,
        tau_applied=_zero6(), mass_matrix=_identity_mass_matrix(),
        coriolis_term=_zero6(), jacobian=_identity_jacobian(), dt_s=DT_S,
    )
    assert result.bridged is False


def test_reset_clears_anchor_and_duplicate_counter():
    bridge = TelemetryGapBridge(max_bridge_cycles=2)
    q0, pose0, qd0 = _base_q(), _base_pose(), _zero6()
    bridge.process(
        q=q0, qd=qd0, tcp_pose=pose0, robot_timestamp_s=1.0,
        tau_applied=_zero6(), mass_matrix=_identity_mass_matrix(),
        coriolis_term=_zero6(), jacobian=_identity_jacobian(), dt_s=DT_S,
    )
    bridge.reset()
    result = bridge.process(
        q=q0, qd=qd0, tcp_pose=pose0, robot_timestamp_s=1.0,
        tau_applied=_zero6(), mass_matrix=_identity_mass_matrix(),
        coriolis_term=_zero6(), jacobian=_identity_jacobian(), dt_s=DT_S,
    )
    # Immediately after reset, there is no anchor -- first read never bridges.
    assert result.bridged is False


def test_rejects_non_positive_dt():
    bridge = TelemetryGapBridge(max_bridge_cycles=2)
    with pytest.raises(ValueError):
        bridge.process(
            q=_base_q(), qd=_zero6(), tcp_pose=_base_pose(), robot_timestamp_s=1.0,
            tau_applied=_zero6(), mass_matrix=_identity_mass_matrix(),
            coriolis_term=_zero6(), jacobian=_identity_jacobian(), dt_s=0.0,
        )


def test_rejects_invalid_max_bridge_cycles():
    with pytest.raises(ValueError):
        TelemetryGapBridge(max_bridge_cycles=0)
