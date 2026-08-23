"""Smoke coverage for the pendulum-free pose-hold ladder.

Kept deliberately short (0.4 s trials): the point is that the rollout, the
guard plumbing and the bare-arm model path all work, not to re-measure the
ladder -- that is the diagnostic's own job and it takes minutes.
"""

from __future__ import annotations

import numpy as np
import pytest

from tools.diagnostics.pose_hold_orientation_check import (
    BARE_ARM_SCENE,
    build_model,
    build_parser,
    run_hold_trial,
)

SINGULAR_Q = np.array([-2.3688, -2.1801, -1.8838, -0.7962, 0.004714693, 0.0206])
REALROD = "assets/ur5e_pendulum/pendulum_attachment_realrod.xml"


@pytest.fixture(scope="module")
def bare_model():
    return build_model(None)


def test_bare_model_has_no_pendulum(bare_model):
    """The whole point of the diagnostic is that the hinge is absent, not just
    parked -- a parked hinge still applies reaction torque."""
    assert bare_model.nq == 6
    assert bare_model.nu == 6
    assert BARE_ARM_SCENE.exists()


def test_composed_model_is_bigger_than_the_bare_one(bare_model):
    assert build_model(REALROD).nq == bare_model.nq + 1


def test_zero_command_hold_reports_every_guarded_quantity(bare_model):
    r = run_hold_trial(
        bare_model,
        arm_q=SINGULAR_Q,
        accel_amplitude_mps2=0.0,
        accel_freq_hz=1.0,
        duration_s=0.4,
        config_path=None,
        controller_kind="impedance",
    )
    for key in (
        "max_orientation_error_rad",
        "max_abs_guard_drift_axis0_m",
        "max_abs_guard_drift_axis1_m",
        "max_abs_guard_drift_axis2_m",
        "max_abs_joint_vel_radps",
    ):
        assert key in r and np.isfinite(r[key]), key
    assert r["n_steps"] > 0
    assert r["accel_amplitude_mps2"] == 0.0


def test_a_nonzero_command_actually_moves_the_arm(bare_model):
    """Guards against the silent no-op: a ladder where every rung reports the
    same numbers would look like a clean pass while testing nothing."""
    kwargs = dict(
        arm_q=SINGULAR_Q,
        accel_freq_hz=1.0,
        duration_s=0.4,
        config_path=None,
        controller_kind="impedance",
    )
    still = run_hold_trial(bare_model, accel_amplitude_mps2=0.0, **kwargs)
    driven = run_hold_trial(bare_model, accel_amplitude_mps2=4.0, **kwargs)
    assert driven["max_abs_guard_drift_axis0_m"] > still["max_abs_guard_drift_axis0_m"]
    assert driven["max_abs_joint_vel_radps"] > still["max_abs_joint_vel_radps"]


def test_rollout_continues_past_the_first_guard_trip_by_default(bare_model):
    """Peak excursions must distinguish 'grazed the threshold' from 'diverged',
    which stopping at the trip would hide."""
    # 1 Hz, not 4: the commanded excursion is 2A/w^2, so raising the FREQUENCY
    # shrinks the demand as 1/f^2. At 4 Hz even A=40 only asks for 0.13 m and
    # the trip was not reliable, which made this test skip -- i.e. assert
    # nothing. At 1 Hz the same amplitude commands ~2 m and the guard must fire.
    kwargs = dict(
        arm_q=SINGULAR_Q,
        accel_amplitude_mps2=40.0,   # far past anything the arm can absorb
        accel_freq_hz=1.0,
        duration_s=1.0,
        config_path=None,
        controller_kind="impedance",
    )
    r = run_hold_trial(bare_model, **kwargs)
    assert r["guard_fired"], "setup no longer trips a guard; the test asserts nothing"
    assert r["steps_completed"] == r["n_steps"]
    assert not r["held"]
    stopped = run_hold_trial(bare_model, stop_on_guard=True, **kwargs)
    assert stopped["steps_completed"] < stopped["n_steps"]


def test_parser_defaults_start_the_ladder_at_zero_and_omit_the_pendulum():
    args = build_parser().parse_args(["--start-q-rad", *map(str, SINGULAR_Q)])
    assert args.pendulum_xml is None, "default must be the BARE arm"
    assert args.accel_amplitudes[0] == 0.0, "ladder must start at the pure hold"
    assert args.stop_on_guard is False
