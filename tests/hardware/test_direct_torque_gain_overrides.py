"""Tests for gain_overrides on the direct-torque hardware transport loop --
live retuning between real-hardware trials without editing --config."""

from __future__ import annotations

import json
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
# HEIGHT_ALPHA_0_5_Q sits exactly on the wrist_2=0 singularity (all interpolated
# alpha values do). Before 2026-07-30, the default tuned config's global
# singular_scale (cond_max=1e5) collapsed task torque to near-zero there
# regardless of gains, so the kp_x-response test below needed a separate
# no_singular_scale config to actually see a gain effect. That config was
# promoted to be the default (jacobian_singular_cond_max=1.0e18 in
# ur5e_mujoco_torque_osc_tuned.yaml itself now -- see that file's header,
# docs/status/disable_global_singular_scale_validation_2026-07-30.md), so
# CONFIG alone is sufficient now; the old singular_scale-enabled behavior is
# preserved at config/ur5e_mujoco_torque_osc_tuned_singular_scale_enabled.yaml
# if ever needed again.


class _MockDTLink:
    def __init__(self) -> None:
        self._tcp_x = 0.35  # off target so kp_x actually produces a nonzero Fx
        self.limits = UR5eSafetyLimits()
        self.connect_calls = 0

    def connect(self) -> None:
        self.connect_calls += 1

    def read_state(self) -> UR5eState:
        return UR5eState(
            q=HEIGHT_ALPHA_0_5_Q.copy(),
            qd=np.zeros(6),
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


def _tau_controller_trace(result) -> np.ndarray:
    rows = [json.loads(line) for line in result.trace_path.read_text().splitlines()]
    return np.asarray([r["tau_controller"] for r in rows], dtype=np.float64)


@pytest.mark.hardware
def test_no_override_matches_config_gains(tmp_path: Path) -> None:
    link = _MockDTLink()
    result = run_x_transport_direct_torque(
        link,  # type: ignore[arg-type]
        config_path=CONFIG, target_x_delta_m=0.01,
        move_duration_s=0.02, duration_s=0.03, output_dir=tmp_path,
        motion_opt_in=True, record_latency=False, dynamics_source="local",
    )
    assert result.summary["gain_overrides"] == {}


@pytest.mark.hardware
def test_gain_override_changes_commanded_torque(tmp_path: Path) -> None:
    link_default = _MockDTLink()
    result_default = run_x_transport_direct_torque(
        link_default,  # type: ignore[arg-type]
        config_path=CONFIG, target_x_delta_m=0.01,
        move_duration_s=0.02, duration_s=0.03, output_dir=tmp_path / "default",
        motion_opt_in=True, record_latency=False, dynamics_source="local",
    )
    link_boosted = _MockDTLink()
    result_boosted = run_x_transport_direct_torque(
        link_boosted,  # type: ignore[arg-type]
        config_path=CONFIG, target_x_delta_m=0.01,
        move_duration_s=0.02, duration_s=0.03, output_dir=tmp_path / "boosted",
        motion_opt_in=True, record_latency=False, dynamics_source="local",
        gain_overrides={"kp_x": 4000.0},
    )
    assert result_boosted.summary["gain_overrides"] == {"kp_x": 4000.0}
    # tau_controller mixes in kd_x/posture/damping terms that don't move with
    # kp_x, and large-enough kp_x deltas can engage geometric backtracking
    # (non-monotonic with kp_x) -- rather than asserting a fragile directional
    # inequality on a single scalar, confirm the override actually changed the
    # full commanded-torque trajectory at all, which is what the mechanism
    # needs to guarantee.
    trace_default = _tau_controller_trace(result_default)
    trace_boosted = _tau_controller_trace(result_boosted)
    assert not np.allclose(trace_default, trace_boosted, atol=1e-6), (
        "gain_overrides={'kp_x': 4000.0} produced an identical tau_controller "
        "trace to the unmodified config -- the override doesn't appear to be "
        "reaching the controller"
    )


@pytest.mark.hardware
def test_bad_gain_override_fails_before_connecting(tmp_path: Path) -> None:
    link = _MockDTLink()
    with pytest.raises(ValueError, match="unknown gain field"):
        run_x_transport_direct_torque(
            link,  # type: ignore[arg-type]
            config_path=CONFIG, target_x_delta_m=0.01,
            move_duration_s=0.02, duration_s=0.03, output_dir=tmp_path,
            motion_opt_in=True, record_latency=False, dynamics_source="local",
            gain_overrides={"kp_bogus": 5.0},
        )
    assert link.connect_calls == 0, "must validate gain_overrides before ever calling link.connect()"


@pytest.mark.hardware
def test_normalize_gain_overrides_cli_helper() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "ur5e_direct_torque_x_transport", REPO_ROOT / "tools" / "ur5e_direct_torque_x_transport.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    finally:
        sys.path.pop(0)
    overrides = module._normalize_gain_overrides('{"kp_x": 500.0, "not_a_gain": 1.0}')
    assert overrides == {"kp_x": 500.0}
