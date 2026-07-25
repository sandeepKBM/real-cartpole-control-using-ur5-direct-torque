"""Tests for URScript generation."""

from __future__ import annotations

from pathlib import Path

import pytest

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


def test_tuned_config_has_zero_kp_rot() -> None:
    # The URScript lane is only valid for kp_rot=0 (damping-only rotation), and
    # the tuned config must satisfy that -- otherwise it would raise below.
    params = load_params_from_yaml(
        DEFAULT_CONFIG, target_x_delta_m=0.02, move_duration_s=1.0, duration_s=3.0
    )
    assert params.kp_rot == 0.0
    # renders fine
    render_urscript(params, template_path=DEFAULT_TEMPLATE)


def test_nonzero_kp_rot_is_rejected_not_silently_dropped() -> None:
    # Regression for gap 4: previously kp_rot was not a field at all, so a config
    # with rotational stiffness would be silently ignored on the real arm. The
    # generator must now refuse it loudly.
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
        kp_rot=5.0,
    )
    with pytest.raises(ValueError, match="kp_rot"):
        render_urscript(params)
