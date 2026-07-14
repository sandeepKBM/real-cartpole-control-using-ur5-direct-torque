"""Tests for URScript generation."""

from __future__ import annotations

from pathlib import Path

from hardware.urscript_gen import (
    DEFAULT_CONFIG,
    DEFAULT_TEMPLATE,
    UrscriptOscParams,
    load_params_from_yaml,
    render_urscript,
)


def test_render_urscript_from_tuned_yaml() -> None:
    params = load_params_from_yaml(
        DEFAULT_CONFIG,
        target_x_delta_m=0.02,
        move_duration_s=1.0,
        duration_s=3.0,
    )
    script = render_urscript(params, template_path=DEFAULT_TEMPLATE)
    assert "direct_torque(tau, viscous_scale=viscous, coulomb_scale=coulomb)" in script
    assert "kp_x = 400" in script
    assert "use_lambda = 1" in script
    assert "{{" not in script


def test_render_kinematic_mode() -> None:
    params = UrscriptOscParams(
        target_x_delta_m=0.01,
        move_duration_s=0.5,
        duration_s=1.0,
        kp_x=100.0,
        kd_x=10.0,
        kp_y=80.0,
        kd_y=15.0,
        kp_z=120.0,
        kd_z=20.0,
        kd_rot=10.0,
        kp_posture=25.0,
        kd_posture=6.0,
        kd_joint=4.0,
        lambda_regularization=0.1,
        use_lambda=False,
        torque_headroom=0.9,
        reanchor_x_tol_m=0.002,
        reanchor_qd_tol_radps=0.05,
        tau_limits_nm=(150.0, 150.0, 150.0, 28.0, 28.0, 28.0),
    )
    script = render_urscript(params)
    assert "use_lambda = 0" in script
