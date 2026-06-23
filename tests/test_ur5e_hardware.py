from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from hardware.ros_topics import AsyncRosVisualizer, RosTopicSample
from hardware.safety_limits import MotionCommandGuard, check_joint_state
from hardware.ur5e_stages import (
    run_direct_torque_probe,
    run_receive_only,
    run_servoj_tiny_motion,
    run_servoj_zero_hold,
)


def test_nan_joint_state_is_refused() -> None:
    q = np.zeros(6, dtype=np.float64)
    q[2] = np.nan
    try:
        check_joint_state(q=q, qd=np.zeros(6, dtype=np.float64))
    except ValueError as exc:
        assert "NaN/Inf" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("NaN joint state should be refused")


def test_command_jump_is_refused() -> None:
    guard = MotionCommandGuard()
    first = guard.check(np.zeros(6, dtype=np.float64))
    assert first.ok
    second = guard.check(np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64), period_s=0.002)
    assert not second.ok
    assert "command jump" in second.reason


def test_receive_only_dry_run_writes_report(tmp_path: Path) -> None:
    output = tmp_path / "receive_only.json"
    result = run_receive_only(
        robot_ip="127.0.0.1",
        frequency=500.0,
        duration=0.01,
        output=output,
        dry_run=True,
    )
    assert result.ok
    assert output.exists()
    assert result.report["status"] == "PASS"
    assert result.report["robot"]["motion_attempted"] is False


def test_zero_hold_requires_motion_flag(tmp_path: Path) -> None:
    output = tmp_path / "zero_hold.json"
    try:
        run_servoj_zero_hold(
            robot_ip="127.0.0.1",
            frequency=500.0,
            duration=0.01,
            gain=100.0,
            lookahead_time=0.1,
            velocity=0.05,
            acceleration=0.05,
            max_deadline_ms=3.0,
            motion_opt_in=False,
            output=output,
        )
    except SystemExit as exc:
        assert "motion opt-in" in str(exc).lower()
    else:  # pragma: no cover - defensive
        raise AssertionError("motion flag should be required")


def test_tiny_motion_refuses_large_amplitude(tmp_path: Path) -> None:
    output = tmp_path / "tiny_motion.json"
    try:
        run_servoj_tiny_motion(
            robot_ip="127.0.0.1",
            frequency=500.0,
            duration=0.01,
            joint_index=0,
            amplitude_rad=0.02,
            max_amplitude_rad=0.01,
            gain=100.0,
            lookahead_time=0.1,
            velocity=0.05,
            acceleration=0.05,
            max_deadline_ms=3.0,
            motion_opt_in=True,
            output=output,
        )
    except SystemExit as exc:
        assert "exceeds" in str(exc).lower()
    else:  # pragma: no cover - defensive
        raise AssertionError("amplitude cap should be enforced")


def test_direct_torque_nonzero_requires_all_flags(tmp_path: Path) -> None:
    output = tmp_path / "direct_torque.json"
    try:
        run_direct_torque_probe(
            robot_ip="127.0.0.1",
            frequency=500.0,
            duration=0.5,
            max_torque_nm=0.05,
            zero_only=False,
            understand_danger=False,
            supervisor_present=False,
            enable_nonzero_torque=False,
            output=output,
        )
    except SystemExit as exc:
        assert "nonzero direct torque" in str(exc).lower()
    else:  # pragma: no cover - defensive
        raise AssertionError("nonzero direct torque must be gated")


def test_visualization_queue_does_not_block() -> None:
    vis = AsyncRosVisualizer(queue_size=1)
    vis._enabled = True  # type: ignore[attr-defined]
    sample = RosTopicSample(
        stamp_ns=123,
        q_real=np.zeros(6, dtype=np.float64),
        qd_real=np.zeros(6, dtype=np.float64),
        q_desired=np.zeros(6, dtype=np.float64),
        qd_desired=np.zeros(6, dtype=np.float64),
    )
    assert vis.submit(sample) is True
    assert vis.submit(sample) is True
    assert vis.dropped_samples == 1
