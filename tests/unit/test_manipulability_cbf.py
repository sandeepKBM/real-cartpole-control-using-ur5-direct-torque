"""Unit tests for the manipulability CBF (controller_core/manipulability_cbf.py).

These check the MATH in isolation -- every step of the derivation in that
module's docstring gets its own numerical check against a closed-form case, so
a sign error or a dropped term cannot hide behind a plausible-looking closed
loop. The closed-loop proof that the mechanism actually works on the real
robot model lives in tests/mujoco/test_manipulability_cbf_closed_loop.py; this
file deliberately does NOT try to prove anything about robot behavior from an
analytic Jacobian (this repo has been burned by exactly that, see AGENTS.md).

The analytic Jacobians used below are chosen so that mu(q) has a hand-writable
closed form (a product of the diagonal entries), which is what makes the
gradient and the directional curvature independently checkable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from controller_core.manipulability_cbf import (  # noqa: E402
    manipulability,
    manipulability_cbf_constraint_row,
    manipulability_cbf_filter,
    manipulability_directional_curvature,
    manipulability_gradient,
)
from controller_core.x_axis_cartesian_impedance import (  # noqa: E402
    CartesianImpedanceConfig,
    XAxisCartesianImpedanceController,
)


# --------------------------------------------------------------------------- #
# Analytic test kinematics. mu(q) = |sin(q[4])| exactly, so
#   grad_mu = sign(sin q4) * cos(q4) * e_4
#   d^2 mu/dq4^2 = -|sin(q4)|      (away from the kink at sin q4 == 0)
# --------------------------------------------------------------------------- #
def analytic_jacobian(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(6)
    return np.diag([1.0, 1.0, 1.0, 1.0, float(np.sin(q[4])), 1.0])


def analytic_mu(q: np.ndarray) -> float:
    return float(abs(np.sin(np.asarray(q, dtype=np.float64).reshape(6)[4])))


# --------------------------------------------------------------------------- #
# 1. mu itself
# --------------------------------------------------------------------------- #
def test_manipulability_equals_abs_det_for_square_jacobian():
    rng = np.random.default_rng(11)
    for _ in range(20):
        jac = rng.normal(0.0, 1.0, (6, 6))
        assert manipulability(jac) == pytest.approx(abs(np.linalg.det(jac)), rel=1e-10)


def test_manipulability_equals_sqrt_det_jjt():
    rng = np.random.default_rng(12)
    jac = rng.normal(0.0, 1.0, (6, 6))
    assert manipulability(jac) == pytest.approx(np.sqrt(np.linalg.det(jac @ jac.T)), rel=1e-9)


def test_manipulability_is_zero_at_a_rank_deficient_jacobian():
    jac = np.eye(6)
    jac[:, 4] = jac[:, 3]  # two identical columns -> rank 5
    assert manipulability(jac) == pytest.approx(0.0, abs=1e-14)


def test_manipulability_is_never_negative_even_for_negative_determinant():
    jac = np.eye(6)
    jac[0, 0] = -3.0  # det < 0
    assert np.linalg.det(jac) < 0.0
    assert manipulability(jac) > 0.0


# --------------------------------------------------------------------------- #
# 2. grad_mu
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("q4", [0.35, 0.9, 1.4, -0.7])
def test_gradient_matches_closed_form(q4):
    q = np.array([0.1, -0.2, 0.3, -0.4, q4, 0.6])
    grad = manipulability_gradient(analytic_jacobian, q)
    expected = np.zeros(6)
    expected[4] = float(np.sign(np.sin(q4)) * np.cos(q4))
    assert np.allclose(grad, expected, atol=1e-8)


def test_gradient_is_zero_in_joints_mu_does_not_depend_on():
    q = np.array([0.1, -0.2, 0.3, -0.4, 0.8, 0.6])
    grad = manipulability_gradient(analytic_jacobian, q)
    assert np.allclose(np.delete(grad, 4), 0.0, atol=1e-12)


def test_gradient_rejects_nonpositive_step():
    q = np.zeros(6)
    with pytest.raises(ValueError, match="step > 0"):
        manipulability_gradient(analytic_jacobian, q, step=0.0)


# --------------------------------------------------------------------------- #
# 3. directional curvature qd^T H_mu qd
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("q4", [0.5, 1.0, 1.3])
@pytest.mark.parametrize("qd4", [0.3, -1.7])
def test_directional_curvature_matches_closed_form(q4, qd4):
    q = np.array([0.0, 0.0, 0.0, 0.0, q4, 0.0])
    qd = np.zeros(6)
    qd[4] = qd4
    got = manipulability_directional_curvature(analytic_jacobian, q, qd)
    # d^2/dq4^2 |sin q4| = -|sin q4| away from the kink; the quadratic form
    # picks up qd4^2.
    expected = qd4 * qd4 * (-abs(np.sin(q4)))
    assert got == pytest.approx(expected, abs=1e-6)


def test_directional_curvature_is_exactly_zero_at_rest():
    q = np.array([0.0, 0.0, 0.0, 0.0, 0.7, 0.0])
    assert manipulability_directional_curvature(analytic_jacobian, q, np.zeros(6)) == 0.0


def test_directional_curvature_is_a_quadratic_form_not_a_linear_one():
    """Doubling qd must quadruple the term -- the property that distinguishes
    qd^T H qd from a first-derivative-shaped mistake."""
    q = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    qd = np.array([0.0, 0.0, 0.0, 0.0, 0.4, 0.0])
    one = manipulability_directional_curvature(analytic_jacobian, q, qd)
    two = manipulability_directional_curvature(analytic_jacobian, q, 2.0 * qd)
    assert two == pytest.approx(4.0 * one, rel=1e-4)


# --------------------------------------------------------------------------- #
# 4. the constraint row itself
# --------------------------------------------------------------------------- #
def test_constraint_row_matches_hand_computed_case():
    grad = np.array([0.0, 1.0, 0.0, 0.0, 2.0, 0.0])
    m_inv = np.diag([1.0, 2.0, 1.0, 1.0, 0.5, 1.0])
    bias = np.array([0.0, 3.0, 0.0, 0.0, -1.0, 0.0])
    qd = np.array([0.0, 0.5, 0.0, 0.0, -0.25, 0.0])
    mu, eps, a1, a2, curv = 0.004, 0.001, 5.0, 7.0, -0.002

    a_row, b_val = manipulability_cbf_constraint_row(
        grad_mu=grad, m_inv=m_inv, bias=bias, qd=qd, mu=mu,
        curvature=curv, epsilon=eps, alpha1=a1, alpha2=a2,
    )
    lie = grad @ m_inv                       # [0, 2, 0, 0, 1, 0]
    assert np.allclose(a_row, -lie[None, :])
    h_dot = float(grad @ qd)                 # 0.5*1 + (-0.25)*2 = 0.0
    expected_b = (
        -float(lie @ bias)                   # -(2*3 + 1*(-1)) = -5.0
        + curv
        + (a1 + a2) * h_dot
        + a1 * a2 * (mu - eps)
    )
    assert b_val == pytest.approx(expected_b, rel=1e-12)


def test_constraint_row_direction_is_the_one_that_increases_mu():
    """A torque along +grad_mu^T M^-1 pushes mu UP, so it must SATISFY the row
    (A @ tau <= b with A = -grad^T M^-1); a torque the other way must violate
    it once the barrier is tight. This is the sign check that a derivation
    error would fail."""
    grad = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    m_inv = np.eye(6)
    a_row, b_val = manipulability_cbf_constraint_row(
        grad_mu=grad, m_inv=m_inv, bias=np.zeros(6), qd=np.zeros(6),
        mu=0.0, curvature=0.0, epsilon=1.0e-3, alpha1=10.0, alpha2=10.0,
    )
    # h = -1e-3 < 0, hdot = 0, curvature = 0  =>  b = 100 * (-1e-3) = -0.1
    assert b_val == pytest.approx(-0.1, rel=1e-12)
    tau_toward_safety = np.zeros(6)
    tau_toward_safety[4] = +1.0
    tau_toward_singularity = -tau_toward_safety
    assert float(a_row[0] @ tau_toward_safety) <= b_val
    assert float(a_row[0] @ tau_toward_singularity) > b_val


# --------------------------------------------------------------------------- #
# 5. the filter: activation, no-op, and the constraint actually being met
# --------------------------------------------------------------------------- #
def _filter(q, qd, tau_nominal, *, epsilon=1.0e-3, alpha1=10.0, alpha2=10.0, tau_max=50.0):
    return manipulability_cbf_filter(
        tau_nominal=np.asarray(tau_nominal, dtype=np.float64),
        jacobian=analytic_jacobian(q),
        jacobian_fn=analytic_jacobian,
        q=np.asarray(q, dtype=np.float64),
        qd=np.asarray(qd, dtype=np.float64),
        m_inv=np.eye(6),
        bias=np.zeros(6),
        tau_lower=np.full(6, -tau_max),
        tau_upper=np.full(6, +tau_max),
        epsilon=epsilon, alpha1=alpha1, alpha2=alpha2,
    )


def test_filter_is_an_exact_no_op_far_from_any_singularity():
    q = np.array([0.0, 0.0, 0.0, 0.0, 1.2, 0.0])   # mu = sin(1.2) = 0.932 >> eps
    tau = np.array([3.0, -2.0, 1.0, 0.5, -0.25, 0.1])
    res = _filter(q, np.zeros(6), tau)
    assert res.active is False
    assert res.delta_norm == 0.0
    # Byte-identical, not merely close: the short-circuit returns the same
    # values rather than a solver approximation of them.
    assert np.array_equal(res.tau, tau)
    assert res.manipulability == pytest.approx(float(np.sin(1.2)), rel=1e-12)
    assert res.h == pytest.approx(float(np.sin(1.2)) - 1.0e-3, rel=1e-12)


def test_filter_is_still_a_no_op_when_moving_AWAY_from_the_singularity():
    """Approach RATE, not position: mu below eps but increasing fast must not
    trigger a correction. This is the property that distinguishes a CBF from a
    threshold."""
    q = np.array([0.0, 0.0, 0.0, 0.0, 5.0e-4, 0.0])   # mu ~ 5e-4 < eps
    qd = np.zeros(6)
    qd[4] = +2.0                                      # moving away, fast
    res = _filter(q, qd, np.zeros(6))
    assert res.h < 0.0
    assert res.h_dot > 0.0
    assert res.active is False


def test_filter_activates_when_driven_toward_the_singularity():
    q = np.array([0.0, 0.0, 0.0, 0.0, 1.2e-3, 0.0])   # mu ~ 1.2e-3, just above eps
    qd = np.zeros(6)
    qd[4] = -0.05                                     # closing on the barrier
    tau = np.zeros(6)
    tau[4] = -20.0                                    # and being pushed further in
    res = _filter(q, qd, tau)
    assert res.active is True
    assert res.slack_at_nominal < 0.0
    assert res.feasible is True
    assert res.delta_norm > 0.0
    # The correction opposes the motion into the singularity.
    assert res.tau[4] > tau[4]


def test_filter_output_satisfies_the_constraint_it_was_given():
    q = np.array([0.0, 0.0, 0.0, 0.0, 1.2e-3, 0.0])
    qd = np.zeros(6)
    qd[4] = -0.05
    tau = np.zeros(6)
    tau[4] = -20.0
    res = _filter(q, qd, tau)
    grad = manipulability_gradient(analytic_jacobian, q)
    curv = manipulability_directional_curvature(analytic_jacobian, q, qd)
    a_row, b_val = manipulability_cbf_constraint_row(
        grad_mu=grad, m_inv=np.eye(6), bias=np.zeros(6), qd=qd,
        mu=manipulability(analytic_jacobian(q)), curvature=curv,
        epsilon=1.0e-3, alpha1=10.0, alpha2=10.0,
    )
    assert float(a_row[0] @ res.tau) <= b_val + 1e-6


def test_filter_stays_inside_the_torque_box_and_reports_infeasibility():
    """With the box tightened to almost nothing, the row becomes unreachable.
    The filter must say so rather than return a point outside the box."""
    q = np.array([0.0, 0.0, 0.0, 0.0, 1.2e-3, 0.0])
    qd = np.zeros(6)
    qd[4] = -5.0                                   # closing hard: hdot ~ -5
    tau = np.zeros(6)
    tau[4] = -1.0e-4
    res = _filter(q, qd, tau, tau_max=1.0e-4)      # ~1e-4 Nm of authority
    assert res.active is True
    assert res.feasible is False
    assert np.all(np.abs(res.tau) <= 1.0e-4 + 1e-9)


def test_curvature_is_meaningless_within_a_step_of_the_singular_kink():
    """Documented limitation, asserted so it cannot silently change.

    mu behaves like |sin q4| ~ c|q4| near the singular set, so a second
    difference whose step straddles q4 == 0 measures the KINK, not a curvature:
    it comes back enormous and positive, which makes the CBF row trivially
    satisfiable (b is huge) and therefore INACTIVE. That is the failure
    direction to be aware of -- it degrades to "no constraint", not to a wrong
    correction -- and it is exactly why manipulability_cbf_epsilon must keep
    the state orders of magnitude further out than
    manipulability_cbf_curvature_step. Compare the well-behaved case below,
    one decade further from the kink, where the closed-form value is
    recovered.
    """
    qd = np.zeros(6)
    qd[4] = -5.0

    q_at_kink = np.array([0.0, 0.0, 0.0, 0.0, 1.0e-6, 0.0])   # << 1e-4 FD step
    curv_kink = manipulability_directional_curvature(analytic_jacobian, q_at_kink, qd)
    assert curv_kink > 1.0e4                                   # nonsense, and positive
    res = _filter(q_at_kink, qd, np.array([0.0, 0.0, 0.0, 0.0, -1.0e-4, 0.0]))
    assert res.active is False                                 # degrades to "no constraint"

    q_clear = np.array([0.0, 0.0, 0.0, 0.0, 1.0e-3, 0.0])      # 10x the FD step
    curv_clear = manipulability_directional_curvature(analytic_jacobian, q_clear, qd)
    assert curv_clear == pytest.approx(25.0 * (-np.sin(1.0e-3)), abs=1e-3)


def test_filter_reports_inactive_on_a_gradient_plateau():
    """A configuration-independent Jacobian has grad_mu == 0: there is no
    direction to constrain in, and the filter must not emit a degenerate
    0 @ tau <= b row."""
    res = manipulability_cbf_filter(
        tau_nominal=np.ones(6),
        jacobian=np.eye(6),
        jacobian_fn=lambda q: np.eye(6),
        q=np.zeros(6), qd=np.ones(6), m_inv=np.eye(6), bias=np.zeros(6),
        tau_lower=np.full(6, -10.0), tau_upper=np.full(6, 10.0),
        epsilon=1.0e-3, alpha1=10.0, alpha2=10.0,
    )
    assert res.grad_norm == 0.0
    assert res.active is False
    assert np.array_equal(res.tau, np.ones(6))


# --------------------------------------------------------------------------- #
# 6. config / parsing / controller wiring
# --------------------------------------------------------------------------- #
def _base_cfg_kwargs():
    return dict(
        kp_x=400.0, kd_x=40.0, kp_y=80.0, kd_y=15.0, kp_z=120.0, kd_z=20.0,
        kp_rot=0.0, kd_rot=10.0, kp_posture=25.0, kd_posture=6.0, kd_joint=4.0,
        tau_max_nm=np.array([150.0, 150.0, 150.0, 28.0, 28.0, 28.0], dtype=np.float64),
    )


def test_flag_defaults_off_with_documented_default_values():
    cfg = CartesianImpedanceConfig(**_base_cfg_kwargs())
    assert cfg.manipulability_cbf is False
    assert cfg.manipulability_cbf_epsilon == 1.0e-3
    assert cfg.manipulability_cbf_alpha1 == 10.0
    assert cfg.manipulability_cbf_alpha2 == 10.0


def _yaml_ctrl(**extra):
    ctrl = {
        "gains": {"kp_x": 400.0, "kd_x": 40.0, "kp_posture": 25.0, "kd_posture": 6.0, "kd_joint": 4.0},
        "torque_limits_initial": {
            "shoulder_pan_joint": 150.0, "shoulder_lift_joint": 150.0, "elbow_joint": 150.0,
            "wrist_1_joint": 28.0, "wrist_2_joint": 28.0, "wrist_3_joint": 28.0,
        },
    }
    ctrl.update(extra)
    return ctrl


def test_yaml_roundtrip_reads_every_field():
    cfg = CartesianImpedanceConfig.from_controller_yaml_section(
        _yaml_ctrl(
            manipulability_cbf=True,
            manipulability_cbf_epsilon=2.5e-3,
            manipulability_cbf_alpha1=4.0,
            manipulability_cbf_alpha2=6.0,
            manipulability_cbf_fd_step=1.0e-6,
            manipulability_cbf_curvature_step=5.0e-4,
        )
    )
    assert cfg.manipulability_cbf is True
    assert cfg.manipulability_cbf_epsilon == pytest.approx(2.5e-3)
    assert cfg.manipulability_cbf_alpha1 == pytest.approx(4.0)
    assert cfg.manipulability_cbf_alpha2 == pytest.approx(6.0)
    assert cfg.manipulability_cbf_fd_step == pytest.approx(1.0e-6)
    assert cfg.manipulability_cbf_curvature_step == pytest.approx(5.0e-4)


def test_yaml_default_is_off():
    cfg = CartesianImpedanceConfig.from_controller_yaml_section(_yaml_ctrl())
    assert cfg.manipulability_cbf is False


@pytest.mark.parametrize("bad", [0.0, -1.0e-3, float("nan"), float("inf"), "abc"])
def test_bad_epsilon_raises_rather_than_falling_back(bad):
    with pytest.raises(ValueError, match="manipulability_cbf_epsilon"):
        CartesianImpedanceConfig.from_controller_yaml_section(
            _yaml_ctrl(manipulability_cbf_epsilon=bad)
        )


@pytest.mark.parametrize("field", ["manipulability_cbf_alpha1", "manipulability_cbf_alpha2"])
@pytest.mark.parametrize("bad", [0.0, -5.0, float("nan"), float("inf")])
def test_bad_alpha_raises(field, bad):
    with pytest.raises(ValueError, match=field):
        CartesianImpedanceConfig.from_controller_yaml_section(_yaml_ctrl(**{field: bad}))


def _state(q, *, mass_matrix=True, jacobian=None):
    st = {
        "time": 0.0,
        "q": np.asarray(q, dtype=np.float64),
        "qd": np.zeros(6, dtype=np.float64),
        "ee_pos": np.array([0.4, 0.0, 0.3]),
        "ee_quat": np.array([1.0, 0.0, 0.0, 0.0]),
        "ee_lin_vel": np.zeros(3),
        "ee_ang_vel": np.zeros(3),
        "target_x": 0.45,
        "target_x_vel": 0.0,
        "jacobian": analytic_jacobian(q) if jacobian is None else jacobian,
    }
    if mass_matrix:
        st["mass_matrix"] = np.eye(6) * 2.0
    return st


def test_controller_raises_without_a_jacobian_fn():
    cfg = CartesianImpedanceConfig(**_base_cfg_kwargs(), manipulability_cbf=True)
    ctrl = XAxisCartesianImpedanceController(cfg)
    st = _state(np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0]))
    ctrl.reset_from_state(st)
    with pytest.raises(ValueError, match="jacobian_fn"):
        ctrl.compute(st)


def test_controller_raises_without_a_mass_matrix():
    cfg = CartesianImpedanceConfig(**_base_cfg_kwargs(), manipulability_cbf=True)
    ctrl = XAxisCartesianImpedanceController(cfg, jacobian_fn=analytic_jacobian)
    st = _state(np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0]), mass_matrix=False)
    ctrl.reset_from_state(st)
    with pytest.raises(ValueError, match="mass_matrix"):
        ctrl.compute(st)


def test_controller_output_is_identical_to_flag_off_far_from_a_singularity():
    q = np.array([0.0, 0.0, 0.0, 0.0, 1.2, 0.0])
    st = _state(q)
    off = XAxisCartesianImpedanceController(CartesianImpedanceConfig(**_base_cfg_kwargs()))
    on = XAxisCartesianImpedanceController(
        CartesianImpedanceConfig(**_base_cfg_kwargs(), manipulability_cbf=True),
        jacobian_fn=analytic_jacobian,
    )
    off.reset_from_state(st)
    on.reset_from_state(st)
    out_off = off.compute(st)
    out_on = on.compute(st)
    assert np.array_equal(out_off.tau, out_on.tau)
    assert out_on.manipulability_cbf_active is False
    assert out_on.manipulability_cbf_delta_tau_norm == 0.0
    # ... and the diagnostics are inert with the flag off.
    assert out_off.manipulability is None
    assert out_off.manipulability_cbf_h is None


def test_controller_diagnostics_populated_when_the_cbf_engages():
    q = np.array([0.0, 0.0, 0.0, 0.0, 1.05e-3, 0.0])
    st = _state(q)
    st["qd"] = np.array([0.0, 0.0, 0.0, 0.0, -0.2, 0.0])
    cfg = CartesianImpedanceConfig(**_base_cfg_kwargs(), manipulability_cbf=True)
    ctrl = XAxisCartesianImpedanceController(cfg, jacobian_fn=analytic_jacobian)
    ctrl.reset_from_state(st)
    out = ctrl.compute(st)
    assert out.manipulability == pytest.approx(float(np.sin(1.05e-3)), rel=1e-9)
    assert out.manipulability_cbf_h is not None
    assert out.manipulability_cbf_h_dot is not None
    assert out.manipulability_cbf_slack is not None
    if out.manipulability_cbf_active:
        assert out.manipulability_cbf_delta_tau_norm > 0.0


def test_torque_stays_inside_the_hard_limit_when_the_cbf_is_active():
    q = np.array([0.0, 0.0, 0.0, 0.0, 1.05e-3, 0.0])
    st = _state(q)
    st["qd"] = np.array([0.0, 0.0, 0.0, 0.0, -1.0, 0.0])
    cfg = CartesianImpedanceConfig(**_base_cfg_kwargs(), manipulability_cbf=True)
    ctrl = XAxisCartesianImpedanceController(cfg, jacobian_fn=analytic_jacobian)
    ctrl.reset_from_state(st)
    out = ctrl.compute(st)
    assert np.all(np.abs(out.tau) <= np.asarray(cfg.tau_max_nm) + 1e-9)
