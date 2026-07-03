from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "simulation"))
sys.path.insert(0, str(REPO_ROOT / "ros2_ws" / "src" / "ur5_x_axis_controller_ros"))

from controller_core.filters import TorqueCommandFilter
from coppelia_torque_diagnostics import (
    CoppeliaTorqueDiagnostics,
    CoppeliaTorqueDiagnosticsConfig,
    smooth_scalar_reference,
    smooth_vector_reference,
)
from ur5_x_axis_controller_ros.coppeliasim_adapter import CoppeliaSimURAdapter, CoppeliaSimConfig


def test_smooth_scalar_reference_clamps_step() -> None:
    out = smooth_scalar_reference(0.0, 1.0, dt=0.01, max_step=0.02, max_velocity=1.0)
    assert out == 0.01


def test_torque_filter_apply_with_diagnostics_reports_rate_clip() -> None:
    filt = TorqueCommandFilter(6, lowpass_alpha=1.0, rate_limit_nm_per_sec=np.full(6, 10.0))
    filt.apply(np.zeros(6), 0.01)
    tau, diag = filt.apply_with_diagnostics(np.full(6, 5.0), 0.01)
    assert tau.shape == (6,)
    assert any(diag["torque_rate_clipped"])


def test_diagnostics_record_and_summary(tmp_path: Path) -> None:
    cfg = CoppeliaTorqueDiagnosticsConfig(
        enable_coppelia_torque_diagnostics=True,
        save_controller_logs=True,
        diagnostics_output_dir=tmp_path,
        diagnostics_mode="hold_soft",
    )
    diag = CoppeliaTorqueDiagnostics(
        cfg,
        tau_limit_nm=np.full(6, 8.0),
        rate_limit_nm_per_sec=np.full(6, 80.0),
        controller_mode="cartesian_impedance_hold",
        prefer_signed_target_force=True,
        run_label="unit_test",
    )
    tau_lim = 8.0
    raw = np.full(6, 9.0)
    sat = np.clip(raw, -tau_lim, tau_lim)
    diag.record_step(
        step_idx=0,
        timestamp=0.0,
        dt=0.01,
        q=np.zeros(6),
        qd=np.zeros(6),
        q_des=np.zeros(6),
        qd_des=np.zeros(6),
        cart_position=None,
        cart_velocity=None,
        pole_angle=None,
        pole_angular_velocity=None,
        tau_impedance_p=np.ones(6),
        tau_impedance_d=np.zeros(6),
        tau_feedforward=None,
        tau_gravity=None,
        tau_raw_before_safety=raw,
        tau_after_saturation=sat,
        tau_final_sent=sat,
        filter_diag=None,
        torque_api_modes=["setJointTargetForce"] * 6,
        safety_flags={},
    )
    summary = diag.build_summary(duration_s=1.0)
    assert summary["per_joint"]["shoulder_pan_joint"]["num_torque_clips"] == 1
    assert summary["first_joint_torque_limit_hit"] == "shoulder_pan_joint"
    trace_path, summary_path = diag.write_logs(tmp_path / "unused.jsonl", summary)
    assert trace_path.exists()
    assert summary_path.exists()
    row = json.loads(trace_path.read_text(encoding="utf-8").strip())
    assert row["coppelia_api_per_joint"][0] == "setJointTargetForce"


def test_diagnostics_generate_plots(tmp_path: Path) -> None:
    cfg = CoppeliaTorqueDiagnosticsConfig(
        enable_coppelia_torque_diagnostics=True,
        save_controller_plots=True,
        diagnostics_output_dir=tmp_path,
    )
    diag = CoppeliaTorqueDiagnostics(
        cfg,
        tau_limit_nm=np.full(6, 8.0),
        rate_limit_nm_per_sec=np.full(6, 80.0),
        controller_mode="hold",
        prefer_signed_target_force=True,
        run_label="plot_test",
    )
    for i in range(20):
        diag.record_step(
            step_idx=i,
            timestamp=i * 0.01,
            dt=0.01,
            q=np.zeros(6),
            qd=np.zeros(6),
            q_des=np.zeros(6),
            qd_des=np.zeros(6),
            cart_position=None,
            cart_velocity=None,
            pole_angle=None,
            pole_angular_velocity=None,
            tau_impedance_p=np.ones(6) * i * 0.1,
            tau_impedance_d=np.zeros(6),
            tau_feedforward=None,
            tau_gravity=None,
            tau_raw_before_safety=np.ones(6) * i * 0.2,
            tau_after_saturation=np.ones(6) * i * 0.15,
            tau_final_sent=np.ones(6) * i * 0.1,
            filter_diag=None,
            torque_api_modes=["setJointTargetForce"] * 6,
            safety_flags={},
        )
    plots = diag.generate_plots(tmp_path / "plot_test")
    assert len(plots) == 13


def test_build_diagnostics_config_cli_overrides() -> None:
    import argparse
    from run_coppeliasim_x_axis_headless import build_diagnostics_config

    args = argparse.Namespace(
        enable_coppelia_torque_diagnostics=True,
        save_controller_logs=None,
        save_controller_plots=None,
        diagnostics_output_dir=None,
        impedance_gain_scale=0.25,
        reference_smoothing_enabled=True,
        max_reference_step=None,
        max_reference_velocity=None,
        torque_diagnostics_mode="hold_soft",
        torque_diagnostics_joint_index=3,
    )
    cfg = build_diagnostics_config(args, {})
    assert cfg.enable_coppelia_torque_diagnostics is True
    assert cfg.save_controller_logs is True
    assert cfg.save_controller_plots is True
    assert cfg.impedance_gain_scale == 0.25
    assert cfg.reference_smoothing_enabled is True
    assert cfg.diagnostics_mode == "hold_soft"
    assert cfg.sinusoid_joint_index == 3
    class _Sim:
        def setJointTargetForce(self, handle: int, tau: float, signed: bool) -> None:
            self.last = (handle, tau, signed)

        def setJointTargetVelocity(self, handle: int, v: float) -> None:
            raise AssertionError("should not use velocity fallback")

        def setJointMaxForce(self, handle: int, mag: float) -> None:
            raise AssertionError("should not use velocity fallback")

    adapter = CoppeliaSimURAdapter(CoppeliaSimConfig(prefer_signed_target_force=True))
    adapter._sim = _Sim()  # type: ignore[assignment]
    adapter._joint_handles = [
        type("JH", (), {"name": "j0", "handle": 1, "resolved_path": ""})(),
    ]
    adapter._last_torque_api_modes = ["unknown"]
    adapter.apply_torque([1.5])
    assert adapter.last_torque_api_modes() == ["setJointTargetForce"]
