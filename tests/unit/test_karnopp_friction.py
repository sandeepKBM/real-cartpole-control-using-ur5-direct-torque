"""Tests for the opt-in Karnopp stick-slip friction feedforward model
(2026-08-02) -- see ``CartesianImpedanceConfig.karnopp_qd_stick_enter_radps``'s
docstring in ``controller_core/x_axis_cartesian_impedance.py`` and
``docs/status/karnopp_stiction_friction_model_2026-08-02.md`` for the full
design rationale and real-hardware evidence.

Motivation, briefly: a real UR5e trace found wrist_1 stationary
(``|qd| < 1.2e-4 rad/s``) for an entire ~6.4s run while the residual observer's
``qdd_residual`` grew to several rad/s^2 in magnitude, strongly correlated with
elapsed time (not velocity) -- a growing, unmodeled static-holding torque. The
already-existing LuGre option (``friction_model="lugre"``, landed 2026-08-01)
cannot represent this: its ODE has ``dz/dt == 0`` whenever ``qd == 0`` exactly
(proven in ``tests/unit/test_lugre_friction.py``), so a joint sitting at
machine-precision-zero velocity never builds bristle deflection regardless of
tuning. Karnopp (1985) is a velocity-hysteresis switching model instead: below
a "stuck" band vs. above a "moving" threshold, with a hysteresis dead zone
between them to prevent chatter.

CORRECTED 2026-08-02: the stuck branch originally tried to "cancel" the net
driving torque already computed that cycle (task + damping + posture +
orient_wrist + gravity) by re-adding a clipped copy of it as feedforward --
but those same terms are ALSO summed into the controller's own tau_bias
separately, so this roughly DOUBLED the real commanded torque on any stuck
joint (confirmed by direct measurement: full-controller output at exactly 2x
the "static" model's for an identical stuck state). The stuck branch now
contributes zero feedforward -- see ``_karnopp_step``'s docstring in
``controller_core/x_axis_cartesian_impedance.py`` for the full reasoning,
including why a simple sign flip is ALSO wrong (real static friction already
self-adjusts to hold a stuck joint on its own; the qdd_residual gap this
feature targets reflects an incomplete dynamics MODEL, which sending more
commanded torque cannot fix). With this fix, Karnopp's stuck-regime behavior
is honestly equivalent to the static model's at qd==0 -- this file's tests
reflect that, not the original (incorrect) "closes the gap" claim. The
sliding-regime branch is unaffected and still provides real value (see
``test_karnopp_sliding_branch_matches_kinetic_coulomb_plus_viscous``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from controller_core.x_axis_cartesian_impedance import (  # noqa: E402
    JOINT_NAME_ORDER,
    CartesianImpedanceConfig,
    XAxisCartesianImpedanceController,
)


def _state(q, qd, target_x=0.0, gravity_torque=None, dt_s=None):
    st = {
        "time": 0.0,
        "q": np.asarray(q, dtype=np.float64),
        "qd": np.asarray(qd, dtype=np.float64),
        "ee_pos": np.array([0.0, 0.0, 0.5], dtype=np.float64),
        "ee_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        "ee_lin_vel": np.zeros(3, dtype=np.float64),
        "ee_ang_vel": np.zeros(3, dtype=np.float64),
        "target_x": float(target_x),
        "jacobian": np.eye(6, dtype=np.float64),
    }
    if gravity_torque is not None:
        st["gravity_torque"] = np.asarray(gravity_torque, dtype=np.float64)
    if dt_s is not None:
        st["dt_s"] = float(dt_s)
    return st


def _zero_gain_config(**overrides) -> CartesianImpedanceConfig:
    """All PD/posture/damping gains zeroed so tau_friction_ff (and, via
    gravity_torque, driving_torque) can be read/controlled in isolation from
    the rest of the wrench pipeline -- same pattern as
    test_friction_feedforward.py / test_lugre_friction.py."""
    base = dict(
        kp_x=0.0, kd_x=0.0, kp_y=0.0, kd_y=0.0, kp_z=0.0, kd_z=0.0,
        kp_rot=0.0, kd_rot=0.0, kp_posture=0.0, kd_posture=0.0, kd_joint=0.0,
        tau_max_nm=np.array([100.0] * 6, dtype=np.float64),
    )
    base.update(overrides)
    return CartesianImpedanceConfig(**base)


# ---------------------------------------------------------------------------
# 1. Flag-off / default regression
# ---------------------------------------------------------------------------


def test_friction_model_default_is_not_karnopp():
    cfg = _zero_gain_config()
    assert cfg.friction_model == "static"


def test_karnopp_off_by_default_even_with_friction_feedforward_on():
    """friction_model defaults to 'static' -- turning on friction_feedforward
    alone (the karnopp_* fields are present with real defaults, but unused)
    must still produce the byte-identical static tanh/viscous law."""
    coulomb = np.array([5.0, 5.0, 5.0, 1.0, 1.0, 1.0], dtype=np.float64)
    viscous = np.array([0.4, 0.4, 0.4, 0.15, 0.15, 0.15], dtype=np.float64)
    deadband = 0.02
    cfg = _zero_gain_config(
        friction_feedforward=True,
        friction_ff_coulomb_nm=coulomb,
        friction_ff_viscous=viscous,
        friction_ff_qd_deadband=deadband,
    )
    assert cfg.friction_model == "static"
    ctrl = XAxisCartesianImpedanceController(cfg)
    q0 = np.zeros(6)
    ctrl.reset_from_state(_state(q0, np.zeros(6)))
    qd = np.full(6, 0.3, dtype=np.float64)
    out = ctrl.compute(_state(q0, qd))
    expected = coulomb * np.tanh(qd / deadband) + viscous * qd
    np.testing.assert_allclose(out.tau_friction_ff, expected, atol=1e-10)
    assert out.friction_model_used == "static"
    # friction_karnopp_stuck is diagnostic-only, always exposed from the
    # internal hysteresis latch (init True/stuck) regardless of which
    # friction_model is active -- it is simply never consulted or updated
    # here since the karnopp branch never runs when friction_model=='static'.
    np.testing.assert_array_equal(out.friction_karnopp_stuck, np.ones(6))


def test_gravity_reorder_does_not_change_static_model_output():
    """The karnopp addition required moving the gravity-torque extraction
    above the friction-feedforward block (a pure reordering of two
    independent computations, so driving_torque can include gravity). This
    regression-checks that reordering: with gravity_torque supplied and
    friction_model='static' (or friction_feedforward off entirely), tau must
    match the closed-form static formula plus gravity, exactly as before the
    reorder -- gravity must still land in tau_bias/tau unchanged."""
    q0 = np.zeros(6)
    qd = np.array([0.5, -0.3, 0.2, 0.0, 0.1, -0.1], dtype=np.float64)
    gravity = np.array([1.0, -2.0, 3.0, -0.5, 0.25, -0.1], dtype=np.float64)
    coulomb = np.array([5.0, 5.0, 5.0, 1.0, 1.0, 1.0], dtype=np.float64)
    viscous = np.array([0.4, 0.4, 0.4, 0.15, 0.15, 0.15], dtype=np.float64)
    deadband = 0.02
    cfg = _zero_gain_config(
        friction_feedforward=True,
        friction_ff_coulomb_nm=coulomb,
        friction_ff_viscous=viscous,
        friction_ff_qd_deadband=deadband,
    )
    ctrl = XAxisCartesianImpedanceController(cfg)
    ctrl.reset_from_state(_state(q0, np.zeros(6)))
    out = ctrl.compute(_state(q0, qd, gravity_torque=gravity))
    expected_friction = coulomb * np.tanh(qd / deadband) + viscous * qd
    expected_tau = expected_friction + gravity
    np.testing.assert_allclose(out.tau, expected_tau, atol=1e-10)

    cfg_off = _zero_gain_config(friction_feedforward=False)
    ctrl_off = XAxisCartesianImpedanceController(cfg_off)
    ctrl_off.reset_from_state(_state(q0, np.zeros(6)))
    out_off = ctrl_off.compute(_state(q0, qd, gravity_torque=gravity))
    np.testing.assert_allclose(out_off.tau, gravity, atol=1e-10)


# ---------------------------------------------------------------------------
# 2. Directional correctness
# ---------------------------------------------------------------------------


def test_karnopp_stuck_branch_contributes_zero_feedforward():
    """CORRECTED 2026-08-02: the stuck branch used to try to "cancel" the net
    driving torque by re-adding a clipped copy of it here, but that torque was
    ALREADY summed into tau_bias by the caller separately, so it roughly
    doubled the commanded torque on any stuck joint (see _karnopp_step's
    docstring and docs/status/karnopp_stiction_friction_model_2026-08-02.md).
    The stuck branch now always contributes zero, regardless of driving
    torque magnitude -- this test locks that in."""
    fs = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float64)
    fc = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    cfg = _zero_gain_config(
        friction_feedforward=True,
        friction_model="karnopp",
        lugre_fc_nm=fc,
        lugre_fs_nm=fs,
    )
    ctrl = XAxisCartesianImpedanceController(cfg)
    q0 = np.zeros(6)
    ctrl.reset_from_state(_state(q0, np.zeros(6)))

    for gravity in (
        np.array([0.5, -0.5, 1.0, -1.0, 0.2, -0.2], dtype=np.float64),
        np.array([5.0, -5.0, 3.0, -3.0, 2.5, -2.5], dtype=np.float64),  # well beyond fs too
    ):
        out = ctrl.compute(_state(q0, np.zeros(6), gravity_torque=gravity))
        np.testing.assert_allclose(out.tau_friction_ff, np.zeros(6), atol=1e-12)
        assert out.friction_model_used == "karnopp"
        np.testing.assert_array_equal(out.friction_karnopp_stuck, np.ones(6))


def test_karnopp_sliding_branch_matches_kinetic_coulomb_plus_viscous():
    """Once a joint is latched free (|qd| above the exit threshold), the
    feedforward must match fc*sign(qd) + viscous*qd -- the same physical
    quantities/sign convention as the static model's sliding-regime term."""
    fc = np.array([5.0, 5.0, 5.0, 1.0, 1.0, 1.0], dtype=np.float64)
    fs = np.array([6.5, 6.5, 6.5, 1.3, 1.3, 1.3], dtype=np.float64)
    viscous = np.array([0.4, 0.4, 0.4, 0.15, 0.15, 0.15], dtype=np.float64)
    cfg = _zero_gain_config(
        friction_feedforward=True,
        friction_model="karnopp",
        lugre_fc_nm=fc,
        lugre_fs_nm=fs,
        friction_ff_viscous=viscous,
    )
    ctrl = XAxisCartesianImpedanceController(cfg)
    q0 = np.zeros(6)
    ctrl.reset_from_state(_state(q0, np.zeros(6)))

    # First call with a large qd (well above the default exit threshold,
    # 0.02 rad/s) flips the latch from its initial "stuck" state to "free".
    qd = np.full(6, 0.3, dtype=np.float64)
    ctrl.compute(_state(q0, qd))
    out = ctrl.compute(_state(q0, qd))
    expected = fc * np.sign(qd) + viscous * qd
    np.testing.assert_allclose(out.tau_friction_ff, expected, atol=1e-10)
    np.testing.assert_array_equal(out.friction_karnopp_stuck, np.zeros(6))


def test_karnopp_odd_symmetry_in_sliding_regime():
    cfg = _zero_gain_config(friction_feedforward=True, friction_model="karnopp")
    q0 = np.zeros(6)

    ctrl_pos = XAxisCartesianImpedanceController(cfg)
    ctrl_pos.reset_from_state(_state(q0, np.zeros(6)))
    qd_pos = np.full(6, 0.3, dtype=np.float64)
    ctrl_pos.compute(_state(q0, qd_pos))
    out_pos = ctrl_pos.compute(_state(q0, qd_pos))

    ctrl_neg = XAxisCartesianImpedanceController(cfg)
    ctrl_neg.reset_from_state(_state(q0, np.zeros(6)))
    qd_neg = -qd_pos
    ctrl_neg.compute(_state(q0, qd_neg))
    out_neg = ctrl_neg.compute(_state(q0, qd_neg))

    np.testing.assert_allclose(out_neg.tau_friction_ff, -out_pos.tau_friction_ff, atol=1e-10)


# ---------------------------------------------------------------------------
# 3. Hysteresis / no-chatter
# ---------------------------------------------------------------------------


def test_karnopp_hysteresis_holds_previous_state_in_dead_zone():
    """A qd magnitude strictly between enter and exit thresholds must not
    flip the latch either way -- it holds whatever state it was already in.
    This is the standard fix for single-threshold chatter."""
    enter = np.full(6, 0.01, dtype=np.float64)
    exit_ = np.full(6, 0.03, dtype=np.float64)
    mid = np.full(6, 0.02, dtype=np.float64)  # strictly between enter and exit
    cfg = _zero_gain_config(
        friction_feedforward=True,
        friction_model="karnopp",
        karnopp_qd_stick_enter_radps=enter,
        karnopp_qd_stick_exit_radps=exit_,
    )
    q0 = np.zeros(6)

    # Starts stuck (reset_from_state default); a "mid" velocity must not
    # flip it to free (mid < exit).
    ctrl = XAxisCartesianImpedanceController(cfg)
    ctrl.reset_from_state(_state(q0, np.zeros(6)))
    out = ctrl.compute(_state(q0, mid))
    np.testing.assert_array_equal(out.friction_karnopp_stuck, np.ones(6))

    # Drive it free with a large velocity, then bring it back to "mid" --
    # must stay free (mid > enter), not re-latch to stuck.
    large = np.full(6, 0.5, dtype=np.float64)
    ctrl.compute(_state(q0, large))
    ctrl.compute(_state(q0, large))
    out2 = ctrl.compute(_state(q0, mid))
    np.testing.assert_array_equal(out2.friction_karnopp_stuck, np.zeros(6))


def test_karnopp_stuck_state_resets_via_reset_from_state():
    cfg = _zero_gain_config(friction_feedforward=True, friction_model="karnopp")
    ctrl = XAxisCartesianImpedanceController(cfg)
    q0 = np.zeros(6)
    ctrl.reset_from_state(_state(q0, np.zeros(6)))
    np.testing.assert_array_equal(ctrl._karnopp_stuck, np.ones(6, dtype=bool))

    large = np.full(6, 0.5, dtype=np.float64)
    ctrl.compute(_state(q0, large))
    ctrl.compute(_state(q0, large))
    assert not np.all(ctrl._karnopp_stuck), "latch should have flipped to free"

    ctrl.reset_from_state(_state(q0, np.zeros(6)))
    np.testing.assert_array_equal(ctrl._karnopp_stuck, np.ones(6, dtype=bool))


# ---------------------------------------------------------------------------
# 4. Core claim: less steady residual than the static model for a joint held
#    near a small non-zero driving torque at qd == 0 (the real evidence this
#    model targets -- see the module docstring).
# ---------------------------------------------------------------------------


def test_karnopp_stuck_matches_static_zero_velocity_output_exactly():
    """CORRECTED 2026-08-02: this test used to claim Karnopp closes the held-
    torque gap the static model leaves open at qd==0. That claim relied on the
    stuck branch re-adding a clipped copy of driving_torque -- the exact
    mechanism found to roughly double the controller's real commanded torque
    (see _karnopp_step's docstring). With that removed, Karnopp's stuck
    branch now contributes exactly the same (zero) as the static model's
    tanh(qd/deadband) term at qd==0 -- no better, no worse. This test locks
    in that honest equivalence instead of a false improvement claim. Closing
    this gap for real needs a separately-validated design, not this feature
    as currently built -- see docs/status/karnopp_stiction_friction_model_2026-08-02.md."""
    joint = 3
    fc = np.full(6, 1.0, dtype=np.float64)
    fs = np.full(6, 5.0, dtype=np.float64)
    cfg_static = _zero_gain_config(
        friction_feedforward=True, friction_model="static",
        lugre_fc_nm=fc, lugre_fs_nm=fs,
    )
    cfg_karnopp = _zero_gain_config(
        friction_feedforward=True, friction_model="karnopp",
        lugre_fc_nm=fc, lugre_fs_nm=fs,
    )
    q0 = np.zeros(6)
    ctrl_static = XAxisCartesianImpedanceController(cfg_static)
    ctrl_karnopp = XAxisCartesianImpedanceController(cfg_karnopp)
    ctrl_static.reset_from_state(_state(q0, np.zeros(6)))
    ctrl_karnopp.reset_from_state(_state(q0, np.zeros(6)))

    ramp = np.linspace(0.0, 3.0, 50)
    for driving_mag in ramp:
        gravity = np.zeros(6, dtype=np.float64)
        gravity[joint] = driving_mag
        qd = np.zeros(6, dtype=np.float64)  # exactly stationary, like real wrist_1

        out_s = ctrl_static.compute(_state(q0, qd, gravity_torque=gravity, dt_s=0.002))
        out_k = ctrl_karnopp.compute(_state(q0, qd, gravity_torque=gravity, dt_s=0.002))

        assert out_s.tau_friction_ff[joint] == 0.0
        assert out_k.tau_friction_ff[joint] == 0.0
        np.testing.assert_allclose(out_k.tau, out_s.tau, atol=1e-10)


def test_karnopp_stuck_branch_ignores_fs_now_that_it_contributes_zero():
    """CORRECTED 2026-08-02: with the stuck branch fixed to contribute zero
    feedforward, lugre_fs_nm no longer affects its output at all -- confirm
    this explicitly (previously this scenario exercised the saturation-clip
    path; that path no longer exists for the stuck branch)."""
    joint = 3
    fc = np.full(6, 1.0, dtype=np.float64)
    fs = np.full(6, 2.0, dtype=np.float64)
    cfg_karnopp = _zero_gain_config(
        friction_feedforward=True, friction_model="karnopp",
        lugre_fc_nm=fc, lugre_fs_nm=fs,
    )
    q0 = np.zeros(6)
    ctrl = XAxisCartesianImpedanceController(cfg_karnopp)
    ctrl.reset_from_state(_state(q0, np.zeros(6)))

    gravity = np.zeros(6, dtype=np.float64)
    gravity[joint] = 10.0  # far beyond fs=2.0 -- irrelevant now, stuck always yields 0
    out = ctrl.compute(_state(q0, np.zeros(6), gravity_torque=gravity))
    assert out.tau_friction_ff[joint] == 0.0


# ---------------------------------------------------------------------------
# 5. YAML parsing
# ---------------------------------------------------------------------------


def test_from_controller_yaml_section_parses_karnopp_fields():
    ctrl_yaml = {
        "gains": {},
        "friction_feedforward": True,
        "friction_model": "KARNOPP",  # case-insensitive
        "karnopp_qd_stick_enter_radps": {name: 0.01 for name in JOINT_NAME_ORDER},
        "karnopp_qd_stick_exit_radps": {name: 0.04 for name in JOINT_NAME_ORDER},
        "lugre_fc_nm": {name: float(i + 1) for i, name in enumerate(JOINT_NAME_ORDER)},
        "lugre_fs_nm": {name: float(i + 1) * 1.3 for i, name in enumerate(JOINT_NAME_ORDER)},
        "torque_limits_initial": {name: 50.0 for name in JOINT_NAME_ORDER},
    }
    cfg = CartesianImpedanceConfig.from_controller_yaml_section(ctrl_yaml)
    assert cfg.friction_model == "karnopp"
    np.testing.assert_allclose(cfg.karnopp_qd_stick_enter_radps, [0.01] * 6)
    np.testing.assert_allclose(cfg.karnopp_qd_stick_exit_radps, [0.04] * 6)


def test_from_controller_yaml_section_defaults_karnopp_thresholds():
    ctrl_yaml = {"gains": {}, "torque_limits_initial": {name: 50.0 for name in JOINT_NAME_ORDER}}
    cfg = CartesianImpedanceConfig.from_controller_yaml_section(ctrl_yaml)
    assert cfg.friction_model == "static"
    np.testing.assert_allclose(cfg.karnopp_qd_stick_enter_radps, [0.005] * 6)
    np.testing.assert_allclose(cfg.karnopp_qd_stick_exit_radps, [0.02] * 6)


def test_from_controller_yaml_section_unknown_friction_model_falls_back_to_static():
    ctrl_yaml = {
        "gains": {},
        "friction_model": "not_a_real_model",
        "torque_limits_initial": {name: 50.0 for name in JOINT_NAME_ORDER},
    }
    cfg = CartesianImpedanceConfig.from_controller_yaml_section(ctrl_yaml)
    assert cfg.friction_model == "static"


# ---------------------------------------------------------------------------
# 6. Full-controller no-double-counting regression (2026-08-02 fix)
# ---------------------------------------------------------------------------
#
# Every test above uses _zero_gain_config(), which zeroes kp_x/kd_x/kp_posture/
# kd_posture/kd_joint -- exactly the terms whose duplication caused the
# original bug. That fixture structurally could not have caught it (with those
# terms zeroed, tau_task_nominal/tau_damping/tau_posture are always zero, so
# there was nothing to double). These tests use REALISTIC, non-zeroed gains
# instead, matching a real deployed config, and check the FULL controller
# output (out.tau), not just out.tau_friction_ff in isolation.


def _realistic_gain_config(**overrides) -> CartesianImpedanceConfig:
    base = dict(
        kp_x=400.0, kd_x=40.0, kp_y=100.0, kd_y=20.0, kp_z=100.0, kd_z=20.0,
        kp_rot=0.0, kd_rot=10.0, kp_posture=25.0, kd_posture=6.0, kd_joint=4.0,
        tau_max_nm=np.array([150.0, 150.0, 150.0, 28.0, 28.0, 28.0], dtype=np.float64),
        torque_headroom=0.9,
    )
    base.update(overrides)
    return CartesianImpedanceConfig(**base)


def _realistic_state(q, qd, gravity_torque):
    return {
        "time": 0.0,
        "q": np.asarray(q, dtype=np.float64),
        "qd": np.asarray(qd, dtype=np.float64),
        "ee_pos": np.array([-0.12, -0.34, 0.93], dtype=np.float64),
        "ee_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        "ee_lin_vel": np.zeros(3, dtype=np.float64),
        "ee_ang_vel": np.zeros(3, dtype=np.float64),
        "target_x": -0.12,
        "jacobian": np.eye(6, dtype=np.float64) * 0.5 + 0.05,
        "gravity_torque": np.asarray(gravity_torque, dtype=np.float64),
        "dt_s": 0.002,
    }


def test_karnopp_stuck_produces_identical_full_controller_output_to_static():
    """The core regression: with realistic, non-zeroed gains, a stuck joint
    (qd ~ 0) must produce EXACTLY the same total commanded torque (out.tau)
    under friction_model="karnopp" as under "static" -- not double it. This
    is the precise scenario (real gains, a genuinely stuck joint, non-zero
    gravity/posture/damping contributions) that reproduced the original bug
    via direct measurement: karnopp's out.tau came out at exactly 2x static's."""
    q0 = np.array([0.0, -0.8, -1.2, -1.0, 0.2, 0.0], dtype=np.float64)
    qd_stuck = np.array([0.0001, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    g = np.array([0.5, -2.0, -1.0, 0.1, 0.01, 0.1], dtype=np.float64)

    cfg_static = _realistic_gain_config()
    cfg_karnopp = _realistic_gain_config(
        friction_feedforward=True, friction_model="karnopp",
        karnopp_qd_stick_enter_radps=np.full(6, 0.005),
        karnopp_qd_stick_exit_radps=np.full(6, 0.02),
        lugre_fc_nm=np.full(6, 5.0), lugre_fs_nm=np.full(6, 50.0),
        friction_ff_viscous=np.full(6, 0.4),
    )
    ctrl_static = XAxisCartesianImpedanceController(cfg_static)
    ctrl_karnopp = XAxisCartesianImpedanceController(cfg_karnopp)
    ctrl_static.reset_from_state(_realistic_state(q0, np.zeros(6), g))
    ctrl_karnopp.reset_from_state(_realistic_state(q0, np.zeros(6), g))

    out_static = ctrl_static.compute(_realistic_state(q0, qd_stuck, g))
    out_karnopp = ctrl_karnopp.compute(_realistic_state(q0, qd_stuck, g))

    assert out_karnopp.friction_model_used == "karnopp"
    np.testing.assert_array_equal(out_karnopp.friction_karnopp_stuck, np.ones(6))
    np.testing.assert_allclose(out_karnopp.tau, out_static.tau, atol=1e-10)
    # Explicitly guard against the exact regression found: karnopp == 2x static.
    doubled = 2.0 * out_static.tau
    assert not np.allclose(out_karnopp.tau, doubled, atol=1e-6), (
        "karnopp output matches 2x static -- the double-counting bug is back"
    )
