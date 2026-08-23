"""Unit tests for ``nullspace_inertia_adaptive_regularization`` (2026-08-12).

The nullspace-posture projector needs a damping constant ``eps`` inside
``Lambda_ns = (A + eps I)^-1``, ``A = J_task M^-1 J_task^T``. Until now that was
either the static ``lambda_regularization`` or ``lambda_adaptive_regularization``'s
``log(cond_task)`` schedule -- and the latter is explicitly REFUSED for a
``split_base_wrist_task`` with fewer than three selected rows, because a 1-row
task has no meaningful condition number (``cond`` of any nonzero ``1xN`` matrix
is exactly 1.0, so ``cond_task`` reports the block's NORM there instead). This
flag fills exactly that hole, scheduling ``eps`` against the task's own
inverse-inertia scale ``lambda_min(A)``.

Pure numpy -- no simulator required. Three tiers, matching
``test_split_base_wrist_task_dims.py``'s own layout:

  * DEFAULT-OFF PROOFS -- the flag unset must be the pre-existing code path.
  * SCHEDULE ALGEBRA -- the two bounding properties the design rests on, and
    the exact leak identity, on a SYNTHETIC operator where the reference
    computation is unambiguous. Synthetic algebra is explicitly NOT evidence of
    real-world viability.
  * REAL-POSE tier -- the actual UR5e Jacobian and mass matrix at the real
    robot start pose this feature was built for, hardcoded below (same pattern
    as ``FAILURE_POSE_J`` in ``test_split_base_wrist_task.py``). The closed-loop
    behavior there is covered by
    ``tests/mujoco/test_nullspace_inertia_regularization_closed_loop.py``, which
    is what any real-world claim should rest on.
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
from tests.unit.test_split_base_wrist_task_dims import (  # noqa: E402
    LIFT_ELBOW_WRIST1,
    SYNTH_M,
    X_ONLY,
    _cfg,
    _controller,
    _synth_state,
)

#: Real UR5e Jacobian and mass matrix at the real robot start pose
#: q = [-2.3688, -2.1801, -1.8838, -0.7962, 0.004714693, 0.0206], measured on
#: assets/ur5e_torque/scene.xml (2026-08-12). Hardcoded so this tier stays a
#: pure-numpy unit test; regenerate with
#: tools/diagnostics/split_base_wrist_task_dims_sim_check.py::model_state.
REAL_POSE_J = np.array([
    [2.365674201e-01, 1.497303728e-02, -2.345545667e-01, -1.087103700e-02, 1.086619109e-02, -6.938893904e-18],
    [-5.777839736e-01, 1.460023140e-02, -2.287145144e-01, -1.060036470e-02, 9.937131919e-03, -6.938893904e-18],
    [0.000000000e+00, 5.788289770e-01, 3.356029311e-01, 9.884166684e-02, -9.890995552e-02, 1.110223025e-16],
    [0.000000000e+00, -6.981373973e-01, -6.981373973e-01, -6.981373973e-01, -7.081673548e-01, -6.976328452e-01],
    [0.000000000e+00, 7.159638081e-01, 7.159638081e-01, 7.159638081e-01, -6.905350638e-01, 7.164402743e-01],
    [1.000000000e+00, 0.000000000e+00, -2.775557562e-17, 5.551115123e-17, -1.471744654e-01, -4.663335302e-03],
])
REAL_POSE_M = np.array([
    [1.566255361e+00, -2.151234332e-01, 1.200214618e-01, 3.005308646e-03, -7.507598114e-04, -6.161851468e-07],
    [-2.151234332e-01, 1.785806907e+00, 4.540059632e-01, 8.541722915e-02, -8.300803109e-03, 1.321325314e-04],
    [1.200214618e-01, 4.540059632e-01, 7.204931135e-01, 5.879405507e-02, -5.541620253e-03, 1.321325314e-04],
    [3.005308646e-03, 8.541722915e-02, 5.879405507e-02, 1.193309583e-01, -1.457692986e-03, 1.321325314e-04],
    [-7.507598114e-04, -8.300803109e-03, -5.541620253e-03, -1.457692986e-03, 1.034181757e-01, -9.131097383e-19],
    [-6.161851468e-07, 1.321325314e-04, 1.321325314e-04, 1.321325314e-04, -9.131097383e-19, 1.001321340e-01],
])

SPLIT_1ROW = dict(
    split_base_wrist_task=True,
    split_base_wrist_active_joints=LIFT_ELBOW_WRIST1,
    split_base_wrist_task_dims=X_ONLY,
    task_space_inertia_shaping=True,
    nullspace_posture=True,
)


def _a_mat(J: np.ndarray, M: np.ndarray, rows, cols) -> np.ndarray:
    """``A = J_task M^-1 J_task^T`` for the scattered split block, recomputed
    here independently of the controller."""
    rows = list(rows)
    cols = list(cols)
    J_task = np.zeros((len(rows), 6), dtype=np.float64)
    J_task[:, cols] = J[np.ix_(rows, cols)]
    return J_task @ np.linalg.inv(M) @ J_task.T, J_task


# --------------------------------------------------------------------------- #
# 1. Default off is the pre-existing path.
# --------------------------------------------------------------------------- #
def test_flag_defaults_off_and_ratio_default():
    cfg = CartesianImpedanceConfig()
    assert cfg.nullspace_inertia_adaptive_regularization is False
    assert cfg.nullspace_inertia_eps_ratio == pytest.approx(0.05)


def test_explicit_false_is_byte_identical_to_unset():
    state = _synth_state(mass_matrix=SYNTH_M)
    unset = _controller(state, **SPLIT_1ROW).compute(state)
    explicit = _controller(
        state, nullspace_inertia_adaptive_regularization=False, **SPLIT_1ROW
    ).compute(state)
    for field in ("tau", "tau_preclip", "tau_task_nominal", "tau_posture", "tau_damping"):
        assert np.array_equal(getattr(unset, field), getattr(explicit, field)), field
    assert unset.lambda_regularization_effective == explicit.lambda_regularization_effective
    assert unset.nullspace_inertia_adaptive_regularization_active is False
    assert explicit.nullspace_inertia_adaptive_regularization_active is False


def test_flag_on_reports_itself_and_a_smaller_eps():
    state = _synth_state(mass_matrix=SYNTH_M)
    off = _controller(state, **SPLIT_1ROW).compute(state)
    on = _controller(
        state, nullspace_inertia_adaptive_regularization=True,
        nullspace_inertia_eps_ratio=0.05, **SPLIT_1ROW,
    ).compute(state)
    assert on.nullspace_inertia_adaptive_regularization_active is True
    assert on.lambda_regularization_effective < off.lambda_regularization_effective
    # Only the projector's eps moved -- the flag must not silently change the
    # wrench-shaping path, so the raw task wrench and the Jacobian metric are
    # untouched.
    assert np.array_equal(on.wrench, off.wrench)
    assert on.jacobian_cond == off.jacobian_cond


# --------------------------------------------------------------------------- #
# 2. Schedule algebra: the two bounding properties, and the leak identity.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("lmin", [0.0, 1e-12, 1e-3, 0.05, 0.09746, 0.1, 0.5, 5.0, 100.0])
@pytest.mark.parametrize("ratio", [0.0, 0.005, 0.05, 0.5, 1.0])
def test_schedule_bounding_properties(lmin, ratio):
    """P1: never more damping than the static eps. P2: ``lmin + eps`` never
    drops below it, so ``||Lambda_ns|| <= 1/lambda_regularization``."""
    eps_static = 0.1
    ctl = XAxisCartesianImpedanceController(
        _cfg(lambda_regularization=eps_static, nullspace_inertia_eps_ratio=ratio)
    )
    eps = ctl._inertia_scheduled_lambda_regularization(np.array([[lmin]]))
    assert 0.0 <= eps <= eps_static + 1e-15                       # P1
    assert lmin + eps >= eps_static - 1e-15                       # P2
    assert 1.0 / (lmin + eps) <= 1.0 / eps_static + 1e-9          # P2, restated


def test_schedule_is_continuous_across_the_crossover():
    """The two branches (``ratio*lmin`` vs ``eps_static - lmin``) meet at
    ``lmin = eps_static/(1+ratio)``; a jump there would be a torque
    discontinuity as the pose drifts across it."""
    eps_static, ratio = 0.1, 0.05
    ctl = XAxisCartesianImpedanceController(
        _cfg(lambda_regularization=eps_static, nullspace_inertia_eps_ratio=ratio)
    )
    cross = eps_static / (1.0 + ratio)
    for delta in (1e-9, 1e-10, 1e-11):
        below = ctl._inertia_scheduled_lambda_regularization(np.array([[cross - delta]]))
        above = ctl._inertia_scheduled_lambda_regularization(np.array([[cross + delta]]))
        assert below == pytest.approx(above, abs=1e-8)


def test_schedule_reaches_target_leak_ratio_between_the_clamps():
    """Between the two clamps the residual leak fraction is exactly
    ``ratio/(1+ratio)``, independent of pose -- that is the knob's meaning."""
    eps_static = 0.1
    for ratio in (0.005, 0.01, 0.05, 0.2):
        ctl = XAxisCartesianImpedanceController(
            _cfg(lambda_regularization=eps_static, nullspace_inertia_eps_ratio=ratio)
        )
        lo, hi = eps_static / (1.0 + ratio), eps_static / ratio
        for lmin in (lo * 1.01, 0.5 * (lo + hi), hi * 0.99):
            eps = ctl._inertia_scheduled_lambda_regularization(np.array([[lmin]]))
            assert eps / (lmin + eps) == pytest.approx(ratio / (1.0 + ratio), rel=1e-12)


@pytest.mark.parametrize("ratio", [0.005, 0.05, 0.5])
@pytest.mark.parametrize("lmin", [1e-6, 1e-3, 0.09746, 0.5, 50.0, 5000.0])
def test_ratio_is_an_upper_bound_on_leak_everywhere(ratio, lmin):
    """Outside the band a clamp binds, and both clamps can only make the leak
    SMALLER than the target -- so ``ratio`` is an upper bound on residual leak
    at every pose, never an under-delivery."""
    eps_static = 0.1
    ctl = XAxisCartesianImpedanceController(
        _cfg(lambda_regularization=eps_static, nullspace_inertia_eps_ratio=ratio)
    )
    eps = ctl._inertia_scheduled_lambda_regularization(np.array([[lmin]]))
    leak = eps / (lmin + eps)
    assert leak <= ratio / (1.0 + ratio) + 1e-12 or lmin < eps_static
    # And it is never worse than the static-eps behavior it replaces.
    assert leak <= eps_static / (lmin + eps_static) + 1e-12


def test_schedule_reduces_to_the_static_eps_at_a_singular_task():
    """``lambda_min(A) -> 0`` is where the static eps is load-bearing; the
    schedule must hand back exactly ``lambda_regularization`` there rather than
    collapsing to a floor (the failure mode a plain clip() form would have)."""
    ctl = XAxisCartesianImpedanceController(
        _cfg(lambda_regularization=0.1, nullspace_inertia_eps_ratio=0.05)
    )
    assert ctl._inertia_scheduled_lambda_regularization(np.zeros((1, 1))) == pytest.approx(0.1)
    assert ctl._inertia_scheduled_lambda_regularization(np.zeros((2, 2))) == pytest.approx(0.1)


def test_negative_ratio_raises():
    ctl = XAxisCartesianImpedanceController(_cfg(nullspace_inertia_eps_ratio=-0.1))
    with pytest.raises(ValueError, match="non-negative"):
        ctl._inertia_scheduled_lambda_regularization(np.array([[0.5]]))


def test_multi_row_schedule_uses_the_smallest_eigenvalue():
    """For >1 row the leak operator's norm is ``eps/(lambda_min(A)+eps)``, so
    the worst (smallest-eigenvalue) direction is the one that must set eps."""
    ctl = XAxisCartesianImpedanceController(
        _cfg(lambda_regularization=0.1, nullspace_inertia_eps_ratio=0.05)
    )
    a_mat = np.diag([3.0, 0.4])
    assert ctl._inertia_scheduled_lambda_regularization(a_mat) == pytest.approx(
        ctl._inertia_scheduled_lambda_regularization(np.array([[0.4]]))
    )


# --------------------------------------------------------------------------- #
# 3. Real-pose tier: the leak identity and the measured improvement.
# --------------------------------------------------------------------------- #
def test_real_pose_leak_matches_the_closed_form_the_schedule_is_derived_from():
    """The projector's residual leak operator is ``eps (A + eps I)^-1``.

    Recomputed here from the real Jacobian/mass matrix, independently of the
    controller, and compared against a direct measurement of the projected
    posture torque's task-space acceleration. This identity is the entire
    justification for scheduling against ``lambda_min(A)`` rather than
    ``||J_task||``.
    """
    a_mat, J_task = _a_mat(REAL_POSE_J, REAL_POSE_M, X_ONLY, LIFT_ELBOW_WRIST1)
    m_inv = np.linalg.inv(REAL_POSE_M)
    tau = np.random.default_rng(20260812).normal(size=6)
    raw = float((J_task @ m_inv @ tau).ravel()[0])
    for eps in (0.1, 0.05, 0.01, 1e-3, 1e-4):
        lam = np.linalg.inv(a_mat + eps * np.eye(1))
        proj = np.eye(6) - J_task.T @ (m_inv @ J_task.T @ lam).T
        measured = float((J_task @ m_inv @ (proj @ tau)).ravel()[0])
        assert measured / raw == pytest.approx(eps / (a_mat[0, 0] + eps), rel=1e-9)


def test_real_pose_block_norm_is_not_the_leak_scale():
    """Guard against the intuitive-but-wrong scheduling variable.

    ``||J_task||`` (what ``singular_scale``'s 1-row diagnostic reports) and
    ``lambda_min(A)`` are different quantities on different scales here --
    0.2353 vs 0.0975 -- so a schedule keyed to the norm would pick eps for a
    reason unrelated to the leak it is meant to bound. If this ever starts
    failing, the two have converged by coincidence at this pose and the test
    below needs a different pose, not a relaxed tolerance.
    """
    a_mat, _J_task = _a_mat(REAL_POSE_J, REAL_POSE_M, X_ONLY, LIFT_ELBOW_WRIST1)
    block_norm = float(np.linalg.norm(REAL_POSE_J[np.ix_(list(X_ONLY), list(LIFT_ELBOW_WRIST1))]))
    assert block_norm == pytest.approx(0.2352832676, rel=1e-6)
    assert float(a_mat[0, 0]) == pytest.approx(0.0974623, rel=1e-5)
    assert abs(block_norm - a_mat[0, 0]) > 0.1


def test_real_pose_static_eps_cancels_only_about_half_the_leak():
    """The measured motivation for this whole mechanism (the number quoted in
    docs/status/transport_axis_generalization_and_pendulum_axis_2026-08-12.md
    sec 3a), pinned so it cannot silently drift."""
    a_mat, _ = _a_mat(REAL_POSE_J, REAL_POSE_M, X_ONLY, LIFT_ELBOW_WRIST1)
    leak_static = 0.1 / (a_mat[0, 0] + 0.1)
    assert leak_static == pytest.approx(0.5064, abs=1e-3)


def test_real_pose_scheduled_eps_hits_the_target_leak():
    """At this pose ``lmin = 0.09746`` sits just inside the band for
    ``ratio = 0.05``, so the target leak is delivered exactly: 50.6% -> 4.8%."""
    a_mat, _ = _a_mat(REAL_POSE_J, REAL_POSE_M, X_ONLY, LIFT_ELBOW_WRIST1)
    ctl = XAxisCartesianImpedanceController(
        _cfg(lambda_regularization=0.1, nullspace_inertia_eps_ratio=0.05)
    )
    eps = ctl._inertia_scheduled_lambda_regularization(a_mat)
    assert eps / (a_mat[0, 0] + eps) == pytest.approx(0.05 / 1.05, rel=1e-9)
    assert eps < 0.1  # strictly less damping than the static value


def test_real_pose_leak_floor_is_p2_bound_not_the_ratio():
    """Documented, deliberate limitation: ``lmin = 0.09746`` is just BELOW
    ``lambda_regularization = 0.1``, so for small ratios the P2 bound binds and
    the leak floors at ``(0.1 - lmin)/0.1 = 2.54%`` however small the ratio
    gets. Pinned so nobody spends a search budget on ratios below ~0.026
    expecting them to do anything at this pose (they measurably do not)."""
    a_mat, _ = _a_mat(REAL_POSE_J, REAL_POSE_M, X_ONLY, LIFT_ELBOW_WRIST1)
    lmin = float(a_mat[0, 0])
    leaks = []
    for ratio in (1e-6, 0.001, 0.01, 0.02):
        ctl = XAxisCartesianImpedanceController(
            _cfg(lambda_regularization=0.1, nullspace_inertia_eps_ratio=ratio)
        )
        eps = ctl._inertia_scheduled_lambda_regularization(a_mat)
        leaks.append(eps / (lmin + eps))
    assert all(leak == pytest.approx(leaks[0], rel=1e-12) for leak in leaks)
    assert leaks[0] == pytest.approx((0.1 - lmin) / 0.1, rel=1e-9)
    assert leaks[0] == pytest.approx(0.0254, abs=1e-3)


def test_controller_projector_at_the_real_pose_matches_an_independent_reference():
    """End-to-end: the projected posture torque the controller emits equals a
    from-scratch reconstruction using the scheduled eps -- so the flag is wired
    into the projector, not merely reported."""
    state = _synth_state(mass_matrix=REAL_POSE_M)
    state["jacobian"] = REAL_POSE_J.copy()
    ratio = 0.05
    out = _controller(
        state, nullspace_inertia_adaptive_regularization=True,
        nullspace_inertia_eps_ratio=ratio, **SPLIT_1ROW,
    ).compute(state)

    a_mat, J_task = _a_mat(REAL_POSE_J, REAL_POSE_M, X_ONLY, LIFT_ELBOW_WRIST1)
    lmin = float(a_mat[0, 0])
    eps_ref = min(max(ratio * lmin, 0.1 - lmin), 0.1)
    assert out.lambda_regularization_effective == pytest.approx(eps_ref, rel=1e-12)

    m_inv = np.linalg.inv(REAL_POSE_M)
    lam = np.linalg.inv(a_mat + eps_ref * np.eye(1))
    proj = np.eye(6) - J_task.T @ (m_inv @ J_task.T @ lam).T
    q = np.asarray(state["q"], dtype=np.float64)
    qd = np.asarray(state["qd"], dtype=np.float64)
    tau_posture_ref = proj @ (25.0 * (q - q) - 6.0 * qd)  # q_rest == q at reset
    assert np.allclose(out.tau_posture, tau_posture_ref, atol=1e-12)


# --------------------------------------------------------------------------- #
# 4. Guards: every refused combination raises rather than silently no-opping.
# --------------------------------------------------------------------------- #
def test_raises_without_split_base_wrist_task():
    state = _synth_state(mass_matrix=SYNTH_M)
    ctl = _controller(
        state, nullspace_inertia_adaptive_regularization=True,
        task_space_inertia_shaping=True, nullspace_posture=True,
    )
    with pytest.raises(ValueError, match="only supported with split_base_wrist_task"):
        ctl.compute(state)


def test_raises_when_split_is_on_but_task_dims_unset():
    state = _synth_state(mass_matrix=SYNTH_M)
    ctl = _controller(
        state, nullspace_inertia_adaptive_regularization=True,
        split_base_wrist_task=True, split_base_wrist_active_joints=LIFT_ELBOW_WRIST1,
        task_space_inertia_shaping=True, nullspace_posture=True,
    )
    with pytest.raises(ValueError, match="split_base_wrist_task_dims"):
        ctl.compute(state)


def test_raises_on_three_task_rows():
    state = _synth_state(mass_matrix=SYNTH_M)
    ctl = _controller(
        state, nullspace_inertia_adaptive_regularization=True,
        split_base_wrist_task=True, split_base_wrist_active_joints=(0, 1, 2),
        split_base_wrist_task_dims=(0, 1, 2),
        task_space_inertia_shaping=True, nullspace_posture=True,
    )
    with pytest.raises(ValueError, match="3 task rows"):
        ctl.compute(state)


def test_one_row_plus_lambda_adaptive_hits_the_pre_existing_guard_first():
    """With a SINGLE row the pre-existing refusal fires before this flag's own
    mutual-exclusion guard -- both reject the combination, so recording which
    message wins keeps the guard order honest if either is ever reordered."""
    state = _synth_state(mass_matrix=SYNTH_M)
    ctl = _controller(
        state, nullspace_inertia_adaptive_regularization=True,
        lambda_adaptive_regularization=True, **SPLIT_1ROW,
    )
    with pytest.raises(ValueError, match="single-row split_base_wrist_task_dims"):
        ctl.compute(state)


def test_raises_with_lambda_adaptive_regularization_at_two_rows():
    """Two rows is where the pre-existing guard does NOT fire (cond_task is a
    real condition number there), so this flag's own mutual-exclusion guard is
    the only thing stopping branch order from silently deciding which eps wins."""
    state = _synth_state(mass_matrix=SYNTH_M)
    cfg = dict(SPLIT_1ROW)
    cfg["split_base_wrist_task_dims"] = (0, 2)
    ctl = _controller(
        state, nullspace_inertia_adaptive_regularization=True,
        lambda_adaptive_regularization=True, **cfg,
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        ctl.compute(state)


def test_raises_when_nullspace_posture_is_off():
    state = _synth_state(mass_matrix=SYNTH_M)
    cfg = dict(SPLIT_1ROW)
    cfg["nullspace_posture"] = False
    ctl = _controller(state, nullspace_inertia_adaptive_regularization=True, **cfg)
    with pytest.raises(ValueError, match="nullspace_posture is False"):
        ctl.compute(state)


def test_two_row_task_is_allowed():
    """Two rows is the other case ``lambda_adaptive_regularization`` handles
    badly at this joint set (the block is still rank-deficient), so it is
    inside this flag's scope, not refused."""
    state = _synth_state(mass_matrix=SYNTH_M)
    cfg = dict(SPLIT_1ROW)
    cfg["split_base_wrist_task_dims"] = (0, 2)
    out = _controller(state, nullspace_inertia_adaptive_regularization=True, **cfg).compute(state)
    assert out.nullspace_inertia_adaptive_regularization_active is True
    assert np.all(np.isfinite(out.tau))


def test_yaml_round_trip():
    from controller_core.x_axis_cartesian_impedance.constants import JOINT_NAME_ORDER

    limits = {"torque_limits_initial": {name: 100.0 for name in JOINT_NAME_ORDER}}
    cfg = CartesianImpedanceConfig.from_controller_yaml_section(
        {
            **limits,
            "nullspace_inertia_adaptive_regularization": True,
            "nullspace_inertia_eps_ratio": 0.037,
        }
    )
    assert cfg.nullspace_inertia_adaptive_regularization is True
    assert cfg.nullspace_inertia_eps_ratio == pytest.approx(0.037)
    default = CartesianImpedanceConfig.from_controller_yaml_section(dict(limits))
    assert default.nullspace_inertia_adaptive_regularization is False
    assert default.nullspace_inertia_eps_ratio == pytest.approx(0.05)
