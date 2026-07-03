from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "ros2_ws" / "src"))
sys.path.insert(0, str(REPO_ROOT / "ros2_ws" / "src" / "ur5_x_axis_controller_ros"))

from ur5_x_axis_controller_ros.config_loader import load_yaml_config
from ur5_x_axis_controller_ros.controller_node import _joint_pd_torque
from simulation.controller import _apply_joint_limit_guardrails


def test_joint_pd_torque_clips_to_limits() -> None:
    q = np.zeros(6, dtype=np.float64)
    qd = np.zeros(6, dtype=np.float64)
    q_ref = np.array([1.0, -1.0, 0.5, 0.25, -0.25, 0.1], dtype=np.float64)
    tau_limit = np.array([0.4, 0.4, 0.4, 0.2, 0.2, 0.1], dtype=np.float64)

    tau, diag = _joint_pd_torque(q=q, qd=qd, q_ref=q_ref, kp=10.0, kd=1.0, tau_limit=tau_limit)

    np.testing.assert_allclose(tau, tau_limit * np.sign(q_ref))
    assert diag["max_abs_tau_cmd_nm"] == float(np.max(np.abs(tau_limit)))
    assert any(bool(flag) for flag in diag["tau_saturated"])


def test_legacy_transport_config_declares_family() -> None:
    cfg = load_yaml_config(
        REPO_ROOT
        / "ros2_ws"
        / "src"
        / "ur5_x_axis_controller_ros"
        / "config"
        / "controller_coppelia_legacy_xz_transport.yaml"
    )
    assert cfg["controller"]["family"] == "legacy_xz_transport_pd"
    assert cfg["controller"]["legacy_x_sign"] == -1.0
    assert cfg["coppeliasim"]["task_frame"]["mode"] == "mujoco_attachment_dummy"
    assert cfg["safety"]["emergency_stop_on_joint_limit"] is False


def test_reduced_chain_joint_limit_guardrails_accept_five_dofs() -> None:
    prev = np.zeros(5, dtype=np.float64)
    delta = np.array([0.5, -0.5, 0.25, -0.25, 0.1], dtype=np.float64)
    lower = np.full(5, -0.1, dtype=np.float64)
    upper = np.full(5, 0.1, dtype=np.float64)

    clipped = _apply_joint_limit_guardrails(prev, delta, lower, upper)

    assert clipped.shape == (5,)
    assert np.all(clipped <= 0.1 + 1e-12)
    assert np.all(clipped >= -0.1 - 1e-12)
