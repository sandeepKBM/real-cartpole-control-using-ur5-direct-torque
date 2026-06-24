from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "simulation"))

from controller import acceleration_transport_controller  # noqa: E402


def _run_allocator(joint_speed_limit_scale: float) -> tuple[np.ndarray, dict]:
    q = np.zeros(6, dtype=np.float64)
    qvel = np.zeros(6, dtype=np.float64)
    ctrl_prev = np.zeros(6, dtype=np.float64)
    ctrl_lower = np.full(6, -2.0 * np.pi, dtype=np.float64)
    ctrl_upper = np.full(6, 2.0 * np.pi, dtype=np.float64)
    tool_pos = np.zeros(3, dtype=np.float64)
    tool_rot = np.eye(3, dtype=np.float64)
    tool_jacobian_pos = np.zeros((3, 6), dtype=np.float64)
    tool_jacobian_rot = np.zeros((3, 6), dtype=np.float64)
    tool_jacobian_pos[0, 1:] = np.array([1.0, 0.8, 0.6, 0.4, 0.2], dtype=np.float64)

    ctrl, diag = acceleration_transport_controller(
        q=q,
        qvel=qvel,
        ctrl_prev=ctrl_prev,
        ctrl_lower=ctrl_lower,
        ctrl_upper=ctrl_upper,
        tool_pos=tool_pos,
        tool_rot=tool_rot,
        tool_jacobian_pos=tool_jacobian_pos,
        tool_jacobian_rot=tool_jacobian_rot,
        a_axis_cmd=1000.0,
        axis_state=0.0,
        transport_axis="x",
        fixed_position=np.zeros(3, dtype=np.float64),
        target_tool_rot=np.eye(3, dtype=np.float64),
        dt=0.1,
        a_axis_max_m_s2=1000.0,
        v_axis_max_m_s=1000.0,
        torque_headroom=1000.0,
        joint_speed_limit_scale=joint_speed_limit_scale,
        move_axis_weight=500.0,
        hold_axis_weight=1.0,
        orientation_weight=1.0,
        hold_axis_gain=0.0,
        posture_target=np.zeros(6, dtype=np.float64),
    )
    return ctrl, diag


def test_lqr_joint_speed_limit_scale_reduces_joint_command() -> None:
    _, diag_full = _run_allocator(1.0)
    _, diag_soft = _run_allocator(0.2)

    qdot_full = np.asarray(diag_full["q_dot_des"], dtype=np.float64)
    qdot_soft = np.asarray(diag_soft["q_dot_des"], dtype=np.float64)

    assert diag_full["joint_speed_limit_scale"] == 1.0
    assert diag_soft["joint_speed_limit_scale"] == 0.2
    assert np.max(np.abs(qdot_soft)) < np.max(np.abs(qdot_full))
