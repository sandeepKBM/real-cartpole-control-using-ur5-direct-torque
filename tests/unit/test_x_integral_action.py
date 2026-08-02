"""Tests for the opt-in X-axis integral action (2026-08-02) -- see
``CartesianImpedanceConfig.x_integral_action``'s docstring in
``controller_core/x_axis_cartesian_impedance.py`` for the full design
rationale and real-hardware evidence.

Motivation, briefly: direct_torque_20260802_190759 (real UR5e, wrist2-offset
pose, split_base_wrist_task) showed shoulder_lift torque settle to a flat
-6.08 Nm and x_error plateau at 0.0082m out of a 0.02m target for an entire
2s hold, completely flat -- the textbook signature of proportional control
settling at a stable equilibrium once its own error-proportional output
drops below the joint's real static-friction breakaway threshold. Integral
action is the classical, model-free fix: it needs no friction model, only
that x_error stays nonzero, and directly mirrors the already-existing
y_integral_action pattern (built 2026-08-01 for a different, structural
Y-drift problem).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from controller_core.x_axis_cartesian_impedance import (  # noqa: E402
    CartesianImpedanceConfig,
    XAxisCartesianImpedanceController,
)


def _state(x, y=0.0, z=0.5, target_x=0.0, target_x_vel=0.0, dt_s=None):
    st = {
        "time": 0.0,
        "q": np.zeros(6, dtype=np.float64),
        "qd": np.zeros(6, dtype=np.float64),
        "ee_pos": np.array([x, y, z], dtype=np.float64),
        "ee_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        "ee_lin_vel": np.zeros(3, dtype=np.float64),
        "ee_ang_vel": np.zeros(3, dtype=np.float64),
        "target_x": float(target_x),
        "target_x_vel": float(target_x_vel),
        "jacobian": np.eye(6, dtype=np.float64),
    }
    if dt_s is not None:
        st["dt_s"] = float(dt_s)
    return st


def _zero_gain_config(**overrides) -> CartesianImpedanceConfig:
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


def test_x_integral_action_defaults_off():
    cfg = _zero_gain_config()
    assert cfg.x_integral_action is False
    assert cfg.ki_x == 0.0


def test_flag_off_output_identical_regardless_of_persistent_x_error():
    """With x_integral_action off, repeated nonzero x_error across many
    cycles must never change Fx/tau -- byte-identical to a single-cycle call,
    proving the integral state (even if some code path touched it) never
    leaks into the output when disabled."""
    cfg = _zero_gain_config(kp_x=50.0)
    ctrl = XAxisCartesianImpedanceController(cfg)
    ctrl.reset_from_state(_state(0.0, target_x=0.0))
    first = None
    for _ in range(20):
        out = ctrl.compute(_state(0.0, target_x=0.1, dt_s=0.002))
        if first is None:
            first = out.tau.copy()
        else:
            np.testing.assert_allclose(out.tau, first, atol=1e-12)
        assert out.x_integral_action_active is False
        assert out.x_integral_value == 0.0


def test_x_integral_does_not_accumulate_while_target_is_actively_moving():
    """CORRECTED 2026-08-02, found by direct sim testing before real hardware:
    accumulating during an active move (nonzero target_x_vel) winds the
    integral up against the large, transient move-phase tracking error --
    nothing to do with friction -- and that windup then overshoots the
    target once holding begins. This feature exists to close a HOLD-phase
    steady-state gap; accumulation must stay off whenever the target is
    still moving, and resume (not reset) once it stops."""
    cfg = _zero_gain_config(x_integral_action=True, ki_x=0.0, x_integral_limit_m_s=10.0)
    ctrl = XAxisCartesianImpedanceController(cfg)
    ctrl.reset_from_state(_state(0.0, target_x=0.0))

    # "Moving": large x_err (as an active move would have) AND nonzero
    # target_x_vel -- integral must not move.
    for _ in range(50):
        out = ctrl.compute(_state(0.0, target_x=0.05, target_x_vel=0.03, dt_s=0.002))
    assert out.x_integral_value == 0.0

    # "Holding": target_x_vel drops to zero -- accumulation resumes from
    # wherever it was (0.0 here), not reset to some other baseline.
    for _ in range(50):
        out = ctrl.compute(_state(0.0, target_x=0.05, target_x_vel=0.0, dt_s=0.002))
    assert out.x_integral_value > 0.0


# ---------------------------------------------------------------------------
# 2. Accumulation / clamping correctness
# ---------------------------------------------------------------------------


def test_x_integral_accumulates_linearly_with_constant_error():
    cfg = _zero_gain_config(x_integral_action=True, ki_x=0.0, x_integral_limit_m_s=10.0)
    ctrl = XAxisCartesianImpedanceController(cfg)
    ctrl.reset_from_state(_state(0.0, target_x=0.0))
    dt = 0.002
    x_err = 0.05  # target_x=0.05, ee at x=0
    n = 100
    for _ in range(n):
        out = ctrl.compute(_state(0.0, target_x=x_err, dt_s=dt))
    expected = x_err * dt * n
    assert out.x_integral_value == pytest.approx(expected, rel=1e-9)


def test_x_integral_clamps_at_limit():
    cfg = _zero_gain_config(x_integral_action=True, ki_x=0.0, x_integral_limit_m_s=0.01)
    ctrl = XAxisCartesianImpedanceController(cfg)
    ctrl.reset_from_state(_state(0.0, target_x=0.0))
    for _ in range(2000):
        out = ctrl.compute(_state(0.0, target_x=1.0, dt_s=0.01))
    assert abs(out.x_integral_value) <= 0.01 + 1e-9
    assert abs(out.x_integral_value - 0.01) < 1e-6  # actually saturated, not just under


def test_x_integral_direction_matches_error_sign():
    cfg = _zero_gain_config(x_integral_action=True, ki_x=0.0, x_integral_limit_m_s=10.0)

    ctrl_pos = XAxisCartesianImpedanceController(cfg)
    ctrl_pos.reset_from_state(_state(0.0, target_x=0.0))
    for _ in range(10):
        out_pos = ctrl_pos.compute(_state(0.0, target_x=0.05, dt_s=0.002))

    ctrl_neg = XAxisCartesianImpedanceController(cfg)
    ctrl_neg.reset_from_state(_state(0.0, target_x=0.0))
    for _ in range(10):
        out_neg = ctrl_neg.compute(_state(0.0, target_x=-0.05, dt_s=0.002))

    assert out_pos.x_integral_value > 0.0
    assert out_neg.x_integral_value < 0.0
    assert abs(out_pos.x_integral_value + out_neg.x_integral_value) < 1e-10


# ---------------------------------------------------------------------------
# 3. Reset lifecycle
# ---------------------------------------------------------------------------


def test_x_integral_resets_via_reset_from_state():
    cfg = _zero_gain_config(x_integral_action=True, ki_x=0.0, x_integral_limit_m_s=10.0)
    ctrl = XAxisCartesianImpedanceController(cfg)
    ctrl.reset_from_state(_state(0.0, target_x=0.0))
    for _ in range(50):
        ctrl.compute(_state(0.0, target_x=0.05, dt_s=0.002))
    assert ctrl._x_integral != 0.0

    ctrl.reset_from_state(_state(0.0, target_x=0.0))
    assert ctrl._x_integral == 0.0
    out = ctrl.compute(_state(0.0, target_x=0.0, dt_s=0.002))
    assert out.x_integral_value == 0.0


def test_x_integral_not_reset_by_set_gains():
    """Matches y_integral_action / friction_z / karnopp_stuck's documented
    contract: a live gain change must never wipe accumulated integral state
    mid-episode."""
    cfg = _zero_gain_config(x_integral_action=True, ki_x=1.0, x_integral_limit_m_s=10.0)
    ctrl = XAxisCartesianImpedanceController(cfg)
    ctrl.reset_from_state(_state(0.0, target_x=0.0))
    for _ in range(20):
        ctrl.compute(_state(0.0, target_x=0.05, dt_s=0.002))
    accumulated = ctrl._x_integral
    assert accumulated != 0.0
    ctrl.set_gains({"kp_x": 99.0})
    assert ctrl._x_integral == accumulated


# ---------------------------------------------------------------------------
# 4. Core claim: closes a steady-state gap a P-only controller cannot
# ---------------------------------------------------------------------------


def test_x_integral_eventually_overcomes_a_simulated_breakaway_threshold():
    """Synthetic scenario modeled on the real evidence: a constant
    'friction-like' opposing force (a fixed disturbance subtracted from Fx,
    standing in for real static friction resisting up to some breakaway
    magnitude) exactly cancels a P-only controller's steady-state output once
    x_error shrinks enough -- the plateau signature. With ki_x=0, the
    achieved position converges to a fixed value strictly short of target and
    stays there. With ki_x>0, the accumulating integral eventually drives net
    Fx (position P-term + integral, disturbance-adjusted) back above the
    disturbance and the residual x_error keeps shrinking rather than
    plateauing -- this test checks the qualitative claim (integral case makes
    genuine additional progress a P-only case does not), not a specific
    convergence trajectory, since this simple synthetic loop has no real
    plant dynamics."""
    disturbance = 0.4  # constant opposing "force" in the same units as Fx

    def run(ki_x, steps=3000, dt=0.002, kp_x=50.0):
        cfg = _zero_gain_config(
            kp_x=kp_x, x_integral_action=(ki_x != 0.0), ki_x=ki_x, x_integral_limit_m_s=10.0,
        )
        ctrl = XAxisCartesianImpedanceController(cfg)
        x = 0.0
        target = 1.0
        ctrl.reset_from_state(_state(x, target_x=target))
        for _ in range(steps):
            out = ctrl.compute(_state(x, target_x=target, dt_s=dt))
            x_err = target - x
            fx = kp_x * x_err + (ki_x * out.x_integral_value if ki_x else 0.0)
            fx_effective = fx - disturbance  # constant opposing disturbance
            if fx_effective <= 0.0:
                fx_effective = 0.0  # "stuck": disturbance fully cancels sub-threshold push
            x += fx_effective * dt  # trivial single-integrator plant, dt small enough to be stable
        return x, target - x

    x_p_only, residual_p_only = run(ki_x=0.0)
    x_with_integral, residual_with_integral = run(ki_x=5.0)

    # P-only equilibrium is exactly disturbance/kp_x (Fx == 0 there): 0.4/50 = 0.008.
    assert residual_p_only == pytest.approx(0.008, rel=1e-6), (
        "sanity check: the P-only case must plateau at exactly disturbance/kp_x"
    )
    assert residual_with_integral < residual_p_only, (
        "integral action must make real additional progress toward target vs. P-only"
    )


# ---------------------------------------------------------------------------
# 5. YAML parsing
# ---------------------------------------------------------------------------


def test_from_controller_yaml_section_parses_x_integral_fields():
    ctrl_yaml = {
        "gains": {"ki_x": 2.5},
        "x_integral_action": True,
        "x_integral_limit_m_s": 0.03,
        "torque_limits_initial": {
            name: 50.0
            for name in (
                "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
            )
        },
    }
    cfg = CartesianImpedanceConfig.from_controller_yaml_section(ctrl_yaml)
    assert cfg.x_integral_action is True
    assert cfg.ki_x == 2.5
    assert cfg.x_integral_limit_m_s == 0.03


def test_from_controller_yaml_section_defaults_x_integral_off():
    ctrl_yaml = {
        "gains": {},
        "torque_limits_initial": {
            name: 50.0
            for name in (
                "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
            )
        },
    }
    cfg = CartesianImpedanceConfig.from_controller_yaml_section(ctrl_yaml)
    assert cfg.x_integral_action is False
    assert cfg.ki_x == 0.0
    assert cfg.x_integral_limit_m_s == 0.02
