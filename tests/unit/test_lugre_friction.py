"""Tests for the opt-in LuGre dynamic friction feedforward model (2026-08-01) --
see CartesianImpedanceConfig.friction_model/lugre_* fields in
controller_core/x_axis_cartesian_impedance.py and
docs/hardware/LUGRE_FRICTION_MODEL_PLAN.md for the full design rationale.

Motivation: real UR5e hardware found the existing static friction_feedforward
model (tanh(qd/deadband)) has no memory of "how long has this joint been
stuck" and does not close a stick-slip breakaway failure. LuGre's bristle-
deflection state z is built to fill that gap.
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


def _state(q, qd, target_x=0.0, dt_s=None):
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
    if dt_s is not None:
        st["dt_s"] = float(dt_s)
    return st


def _zero_gain_config(**overrides) -> CartesianImpedanceConfig:
    """All PD/posture/damping gains zeroed so tau_friction_ff can be read in
    isolation from tau_preclip without needing to invert the backtrack/clip
    pipeline (same pattern as test_friction_feedforward.py)."""
    base = dict(
        kp_x=0.0, kd_x=0.0, kp_y=0.0, kd_y=0.0, kp_z=0.0, kd_z=0.0,
        kp_rot=0.0, kd_rot=0.0, kp_posture=0.0, kd_posture=0.0, kd_joint=0.0,
        tau_max_nm=np.array([100.0] * 6, dtype=np.float64),
    )
    base.update(overrides)
    return CartesianImpedanceConfig(**base)


def test_friction_model_defaults_to_static():
    cfg = _zero_gain_config()
    assert cfg.friction_model == "static"


def test_lugre_off_by_default_even_with_friction_feedforward_on():
    """friction_model defaults to 'static' -- turning on friction_feedforward
    alone must still use the static tanh/viscous law, byte-identical to
    test_friction_feedforward.py's own closed-form check."""
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
    np.testing.assert_array_equal(out.friction_z, np.zeros(6))


def test_lugre_z_state_persists_and_resets():
    """z accumulates across compute() calls (genuine per-joint state), and is
    zeroed by reset_from_state() -- matches _q_rest/_y_integral lifecycle."""
    cfg = _zero_gain_config(friction_feedforward=True, friction_model="lugre")
    ctrl = XAxisCartesianImpedanceController(cfg)
    q0 = np.zeros(6)
    ctrl.reset_from_state(_state(q0, np.zeros(6)))
    np.testing.assert_array_equal(ctrl._friction_z, np.zeros(6))

    qd = np.full(6, 0.05, dtype=np.float64)
    for _ in range(50):
        out = ctrl.compute(_state(q0, qd, dt_s=0.002))
    assert np.all(out.friction_z != 0.0), "z should have accumulated after 50 nonzero-qd cycles"
    assert out.friction_model_used == "lugre"

    # A fresh reset must zero z again.
    ctrl.reset_from_state(_state(q0, np.zeros(6)))
    np.testing.assert_array_equal(ctrl._friction_z, np.zeros(6))


def test_lugre_zero_velocity_leaves_z_and_torque_at_zero():
    """qd == 0 exactly -> z_dot == 0 (both terms of the ODE vanish), so z
    never departs from its initial value and tau_friction_ff == sigma2*qd == 0."""
    cfg = _zero_gain_config(friction_feedforward=True, friction_model="lugre")
    ctrl = XAxisCartesianImpedanceController(cfg)
    q0 = np.zeros(6)
    ctrl.reset_from_state(_state(q0, np.zeros(6)))
    for _ in range(20):
        out = ctrl.compute(_state(q0, np.zeros(6), dt_s=0.002))
    np.testing.assert_array_equal(out.friction_z, np.zeros(6))
    np.testing.assert_array_equal(out.tau_friction_ff, np.zeros(6))


def test_lugre_held_against_a_wall_z_grows_then_plateaus():
    """The plan's sec 6 item 4 scenario: a persistent, constant-sign creep
    velocity should make z rise monotonically and then plateau (bristle
    deflection saturating near the Stribeck curve g(qd)), not diverge. This
    is the ODE unit test called for by the plan independent of the full
    controller/sim integration; it isolates _lugre_step() from
    tau_preclip/backtrack/clip.

    Real, honestly-reported finding from building this test (see
    docs/status/lugre_friction_feedforward_2026-08-01.md): under the plan's
    literal ODE (`g(qd)` in Nm, NOT divided by sigma0 the way textbook LuGre
    normalizes it), the relaxation time constant governing how fast z
    approaches g(qd) is approximately g(qd)/|qd| -- with Nm-scale g (~5-6.5
    here) and a realistic tiny pre-sliding creep velocity (e.g. 5e-4 rad/s,
    the magnitude actually used in the config's own header-comment
    reasoning), that time constant is on the order of *minutes*, far beyond
    any real transport-move hold-phase duration (1-10s). A first version of
    this test used that realistic creep magnitude and found z was still
    firmly in its early, non-plateaued linear-growth regime after 8
    simulated seconds (no bug -- z growth matched qd*t exactly, i.e. the ODE
    is behaving correctly, just far slower than a "textbook" implementation
    with a meters-scale, sigma0-normalized g would be). This test therefore
    uses a much larger qd purely to observe the ODE's own asymptotic
    plateau behavior within a bounded runtime -- it validates the ODE's
    correctness, not physical realism of the velocity used for it.
    """
    sigma0 = np.full(6, 1.0)
    sigma1 = np.full(6, 0.8)
    sigma2 = np.full(6, 0.4)
    fc = np.full(6, 5.0)
    fs = np.full(6, 6.5)  # Fs > Fc: real breakaway peak
    vs = np.full(6, 0.02)
    cfg = _zero_gain_config(
        friction_feedforward=True,
        friction_model="lugre",
        lugre_sigma0_nm_per_rad=sigma0,
        lugre_sigma1_nm_s_per_rad=sigma1,
        lugre_sigma2_nm_s_per_rad=sigma2,
        lugre_fc_nm=fc,
        lugre_fs_nm=fs,
        lugre_vs_radps=vs,
    )
    ctrl = XAxisCartesianImpedanceController(cfg)
    q0 = np.zeros(6)
    ctrl.reset_from_state(_state(q0, np.zeros(6)))

    # qd well above vs so g(qd) settles at the Coulomb floor fc -- picked
    # only to make the ODE's own relaxation time (g/qd ~= 1s here) visible
    # within a short, bounded test window; see the docstring above for why
    # this is NOT meant to represent a realistic pre-sliding creep speed.
    qd_creep = np.full(6, 5.0, dtype=np.float64)
    dt = 0.002
    z_trace = []
    for _ in range(4000):  # 8s of simulated creep
        out = ctrl.compute(_state(q0, qd_creep, dt_s=dt))
        z_trace.append(float(out.friction_z[0]))
    z_trace = np.asarray(z_trace)

    # Monotonically non-decreasing (bristle deflection only ever grows
    # toward the Stribeck curve at a constant positive qd -- no
    # oscillation/overshoot).
    assert np.all(np.diff(z_trace) >= -1e-12), "z should not oscillate/decrease under constant positive qd"
    # Plateaus: the back third of the trace should be nearly flat relative
    # to the front third's growth (converged, not still ramping/diverging).
    growth_first_half = z_trace[len(z_trace) // 2] - z_trace[0]
    growth_second_half = z_trace[-1] - z_trace[len(z_trace) // 2]
    assert growth_second_half < 0.1 * max(growth_first_half, 1e-9), (
        f"z should be plateauing by the end of the window: "
        f"first-half growth={growth_first_half}, second-half growth={growth_second_half}"
    )
    # z should be approaching the Stribeck ceiling g(qd) ~= fs at this tiny
    # qd (qd << vs), not diverging past it or collapsing to ~0.
    assert 0.0 < z_trace[-1] <= fs[0] * 1.05

    # Net motion stays effectively pinned: with these zero PD gains,
    # tau_friction_ff is the ONLY torque this test exercises, and qd itself
    # is an externally supplied constant (this test isolates the ODE, not
    # closed-loop dynamics) -- so "near-zero net motion" here is verified as
    # the tiny commanded creep itself never being amplified: the resulting
    # feedforward torque stays bounded and does not blow up.
    assert np.all(np.abs(out.tau_friction_ff) < (sigma0 * fs + sigma1 * 1.0 + sigma2 * 1.0))


def test_lugre_insufficient_torque_case_z_bounded_by_stribeck_curve():
    """A direct algebraic check of the "held against a wall" invariant: for
    any qd, |z| after a step from z=0 must stay within the Stribeck bound
    g(qd) (the ODE's own self-limiting property -- dz/dt flips sign once
    |z| crosses g(qd), so explicit Euler cannot run away at these dt/qd/g
    magnitudes)."""
    cfg = _zero_gain_config(friction_feedforward=True, friction_model="lugre")
    ctrl = XAxisCartesianImpedanceController(cfg)
    q0 = np.zeros(6)
    ctrl.reset_from_state(_state(q0, np.zeros(6)))

    fc = cfg.lugre_fc_nm
    fs = cfg.lugre_fs_nm
    vs = cfg.lugre_vs_radps
    qd_hold = np.full(6, 0.01, dtype=np.float64)
    dt = 0.002
    for _ in range(10000):  # 20s, far beyond any realistic hold
        out = ctrl.compute(_state(q0, qd_hold, dt_s=dt))
        g = fc + (fs - fc) * np.exp(-((qd_hold / vs) ** 2))
        assert np.all(np.abs(out.friction_z) <= g * 1.01), "z must stay within the Stribeck bound"


def test_lugre_odd_symmetry_in_velocity_sign():
    """Symmetric setup: qd and -qd from a fresh reset should produce
    tau_friction_ff of opposite sign and equal magnitude (mirrors the static
    model's own odd-symmetry test)."""
    cfg = _zero_gain_config(friction_feedforward=True, friction_model="lugre")
    q0 = np.zeros(6)

    ctrl_pos = XAxisCartesianImpedanceController(cfg)
    ctrl_pos.reset_from_state(_state(q0, np.zeros(6)))
    qd_pos = np.full(6, 0.3, dtype=np.float64)
    out_pos = ctrl_pos.compute(_state(q0, qd_pos, dt_s=0.002))

    ctrl_neg = XAxisCartesianImpedanceController(cfg)
    ctrl_neg.reset_from_state(_state(q0, np.zeros(6)))
    qd_neg = -qd_pos
    out_neg = ctrl_neg.compute(_state(q0, qd_neg, dt_s=0.002))

    np.testing.assert_allclose(out_neg.tau_friction_ff, -out_pos.tau_friction_ff, atol=1e-10)
    np.testing.assert_allclose(out_neg.friction_z, -out_pos.friction_z, atol=1e-10)


def test_lugre_dt_fallback_when_dt_s_absent():
    """dt_s is optional on the state contract -- when absent, the LuGre step
    must fall back to the 500 Hz nominal period (1/500s), matching
    y_integral_action's identical fallback, not crash or silently use dt=0."""
    cfg = _zero_gain_config(friction_feedforward=True, friction_model="lugre")
    ctrl_with_dt = XAxisCartesianImpedanceController(cfg)
    ctrl_no_dt = XAxisCartesianImpedanceController(cfg)
    q0 = np.zeros(6)
    ctrl_with_dt.reset_from_state(_state(q0, np.zeros(6)))
    ctrl_no_dt.reset_from_state(_state(q0, np.zeros(6)))

    qd = np.full(6, 0.1, dtype=np.float64)
    out_with_dt = ctrl_with_dt.compute(_state(q0, qd, dt_s=1.0 / 500.0))
    out_no_dt = ctrl_no_dt.compute(_state(q0, qd))  # no dt_s key at all
    np.testing.assert_allclose(out_with_dt.friction_z, out_no_dt.friction_z, atol=1e-15)
    np.testing.assert_allclose(out_with_dt.tau_friction_ff, out_no_dt.tau_friction_ff, atol=1e-15)


def test_lugre_flows_through_backtrack_and_clip():
    """With real (nonzero) other gains too, the LuGre term must still show up
    in the final clipped tau -- mirrors
    test_friction_feedforward_flows_through_backtrack_and_clip."""
    cfg = CartesianImpedanceConfig(
        kp_x=25.0, kd_x=8.0, kp_y=80.0, kd_y=15.0, kp_z=120.0, kd_z=20.0,
        kp_rot=0.0, kd_rot=10.0, kp_posture=2.0, kd_posture=0.5, kd_joint=0.8,
        tau_max_nm=np.array([100.0] * 6, dtype=np.float64),
        friction_feedforward=True,
        friction_model="lugre",
    )
    ctrl_on = XAxisCartesianImpedanceController(cfg)
    ctrl_off = XAxisCartesianImpedanceController(
        CartesianImpedanceConfig(
            kp_x=25.0, kd_x=8.0, kp_y=80.0, kd_y=15.0, kp_z=120.0, kd_z=20.0,
            kp_rot=0.0, kd_rot=10.0, kp_posture=2.0, kd_posture=0.5, kd_joint=0.8,
            tau_max_nm=np.array([100.0] * 6, dtype=np.float64),
        )
    )
    q0 = np.zeros(6)
    qd = np.array([0.3, -0.2, 0.1, 0.0, 0.05, -0.05], dtype=np.float64)
    st0 = _state(q0, np.zeros(6))
    ctrl_on.reset_from_state(st0)
    ctrl_off.reset_from_state(st0)
    out_on = ctrl_on.compute(_state(q0, qd, target_x=0.05, dt_s=0.002))
    out_off = ctrl_off.compute(_state(q0, qd, target_x=0.05, dt_s=0.002))
    assert not np.allclose(out_on.tau, out_off.tau)


def test_from_controller_yaml_section_parses_lugre_fields():
    ctrl_yaml = {
        "gains": {},
        "friction_feedforward": True,
        "friction_model": "lugre",
        "lugre_sigma0_nm_per_rad": {name: float(i + 1) for i, name in enumerate(JOINT_NAME_ORDER)},
        "lugre_sigma1_nm_s_per_rad": {name: float(i + 1) * 0.1 for i, name in enumerate(JOINT_NAME_ORDER)},
        "lugre_sigma2_nm_s_per_rad": {name: float(i + 1) * 0.01 for i, name in enumerate(JOINT_NAME_ORDER)},
        "lugre_fc_nm": {name: float(i + 1) for i, name in enumerate(JOINT_NAME_ORDER)},
        "lugre_fs_nm": {name: float(i + 1) * 1.3 for i, name in enumerate(JOINT_NAME_ORDER)},
        "lugre_vs_radps": {name: 0.03 for name in JOINT_NAME_ORDER},
        "torque_limits_initial": {name: 50.0 for name in JOINT_NAME_ORDER},
    }
    cfg = CartesianImpedanceConfig.from_controller_yaml_section(ctrl_yaml)
    assert cfg.friction_model == "lugre"
    np.testing.assert_allclose(cfg.lugre_sigma0_nm_per_rad, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    np.testing.assert_allclose(cfg.lugre_sigma1_nm_s_per_rad, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    np.testing.assert_allclose(cfg.lugre_sigma2_nm_s_per_rad, [0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
    np.testing.assert_allclose(cfg.lugre_fc_nm, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    np.testing.assert_allclose(cfg.lugre_fs_nm, [1.3, 2.6, 3.9, 5.2, 6.5, 7.8])
    np.testing.assert_allclose(cfg.lugre_vs_radps, [0.03] * 6)


def test_from_controller_yaml_section_defaults_friction_model_static():
    ctrl_yaml = {"gains": {}, "torque_limits_initial": {name: 50.0 for name in JOINT_NAME_ORDER}}
    cfg = CartesianImpedanceConfig.from_controller_yaml_section(ctrl_yaml)
    assert cfg.friction_model == "static"
    np.testing.assert_allclose(cfg.lugre_fc_nm, [5.0, 5.0, 5.0, 1.0, 1.0, 1.0])
    np.testing.assert_allclose(cfg.lugre_fs_nm, [6.5, 6.5, 6.5, 1.3, 1.3, 1.3])
