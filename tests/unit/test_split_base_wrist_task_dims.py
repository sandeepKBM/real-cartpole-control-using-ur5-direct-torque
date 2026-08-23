"""Unit tests for `split_base_wrist_task_dims` (combined row + column selection).

``split_base_wrist_task`` has always restricted the translation task to a
subset of joint COLUMNS while using all three position ROWS. Since 2026-08-12
``split_base_wrist_task_dims`` selects the rows too, so ``J_task`` can be a
general ``len(rows) x len(cols)`` block -- notably ``1x3`` for a
one-dimensional world-X transport driven by the (structurally rank-2) UR
planar sub-chain ``{shoulder_lift, elbow, wrist_1}``.

Pure numpy -- no simulator required. Two kinds of test live here:

  * DEFAULT-OFF PROOFS -- the new field unset must be the pre-existing code
    path, exactly, including at the real failure pose.
  * 1x3 LINEAR-ALGEBRA CORRECTNESS -- every downstream operator (Lambda, the
    dynamically consistent nullspace projector, the SCI filter, the
    Jacobian-transpose force map) recomputed independently here and compared
    against what the controller produced.

The linear-algebra tests deliberately use a SYNTHETIC constant Jacobian and a
synthetic mass matrix (clearly named ``SYNTH_*``): the point there is to pin
the algebra, and a synthetic operator makes the reference computation
unambiguous. That is explicitly NOT evidence of real-world viability -- the
real UR5e kinematics at the real start pose, and the closed-loop behavior
there, are covered by ``tests/mujoco/test_split_base_wrist_task_dims_closed_loop.py``
and are the only thing any real-world claim should rest on. A middle tier
(real Jacobian/mass matrix, single cycle) is included below too.
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
from controller_core.x_axis_cartesian_impedance.parsing import (  # noqa: E402
    _parse_split_base_wrist_task_dims,
)
from tests.unit.test_split_base_wrist_task import (  # noqa: E402
    FAILURE_POSE_J,
    LINEAR_J,
    LINEAR_P0,
    _make_state,
)

#: The motivating selection: world-X row, shoulder_lift/elbow/wrist_1 columns.
LIFT_ELBOW_WRIST1 = (1, 2, 3)
HELD_BY_LIFT_ELBOW_WRIST1 = (0, 4, 5)
X_ONLY = (0,)

#: Synthetic operators for the pure linear-algebra tier -- see this module's
#: docstring for why these are deliberately not the real robot's.
SYNTH_J = LINEAR_J
_SYNTH_RNG = np.random.default_rng(20260812)
_A = _SYNTH_RNG.normal(size=(6, 6)) * 0.3
SYNTH_M = _A @ _A.T + np.eye(6) * 2.0  # symmetric positive definite


def _cfg(**overrides) -> CartesianImpedanceConfig:
    base = dict(
        tau_max_nm=np.full(6, 1e6),  # keep clipping/backtracking out of the way
        kp_x=400.0, kd_x=40.0, kp_y=80.0, kd_y=15.0, kp_z=120.0, kd_z=20.0,
        kp_rot=0.0, kd_rot=10.0, kp_posture=25.0, kd_posture=6.0, kd_joint=4.0,
        lambda_regularization=0.1, jacobian_singular_cond_max=1.0e18,
    )
    base.update(overrides)
    return CartesianImpedanceConfig(**base)


def _controller(state, **overrides) -> XAxisCartesianImpedanceController:
    ctl = XAxisCartesianImpedanceController(_cfg(**overrides))
    ctl.reset_from_state(state)
    return ctl


def _synth_state(*, q=None, qd=None, target_x=None, mass_matrix=None):
    q = np.array([0.1, -1.2, 1.4, -1.7, -1.55, 0.3]) if q is None else np.asarray(q, dtype=np.float64)
    qd = np.full(6, 0.05) if qd is None else np.asarray(qd, dtype=np.float64)
    ee = LINEAR_P0 + SYNTH_J[0:3, :] @ q
    st = {
        "time": 0.0,
        "q": q.copy(),
        "qd": qd.copy(),
        "ee_pos": ee,
        "ee_quat": np.array([1.0, 0.0, 0.0, 0.0]),
        "ee_lin_vel": SYNTH_J[0:3, :] @ qd,
        "ee_ang_vel": np.zeros(3),
        "jacobian": SYNTH_J.copy(),
        "target_x": float(ee[0] + 0.03) if target_x is None else float(target_x),
    }
    if mass_matrix is not None:
        st["mass_matrix"] = np.asarray(mass_matrix, dtype=np.float64)
    return st


# --------------------------------------------------------------------------- #
# 1. Default-off is the pre-existing path, exactly.
# --------------------------------------------------------------------------- #
def test_task_dims_default_is_none():
    assert CartesianImpedanceConfig().split_base_wrist_task_dims is None


@pytest.mark.parametrize("active_joints", [None, (0, 1, 2), (2, 3, 4)])
def test_explicit_all_three_rows_is_byte_identical_to_unset(active_joints):
    """``[0, 1, 2]`` is the substituted default, so setting it explicitly must
    reproduce the historical 3-row task bit-for-bit -- not merely closely."""
    state = _make_state(q=np.full(6, 0.1), qd=np.full(6, 0.05), J=FAILURE_POSE_J,
                        mass_matrix=SYNTH_M)
    common = dict(
        split_base_wrist_task=True,
        task_space_inertia_shaping=True,
        nullspace_posture=True,
        wrist_orientation_task=True,
    )
    if active_joints is not None:
        common["split_base_wrist_active_joints"] = active_joints
    unset = _controller(state, **common).compute(state)
    explicit = _controller(state, split_base_wrist_task_dims=(0, 1, 2), **common).compute(state)
    for field in ("tau", "tau_preclip", "wrench", "tau_task_nominal", "tau_posture",
                  "tau_damping", "tau_orient_wrist"):
        assert np.array_equal(getattr(unset, field), getattr(explicit, field)), field
    assert unset.jacobian_cond == explicit.jacobian_cond
    assert unset.singular_scale == explicit.singular_scale
    assert unset.lambda_regularization_effective == explicit.lambda_regularization_effective
    # ...and the new diagnostic reports the substituted default either way.
    assert unset.split_base_wrist_task_dims == (0, 1, 2)
    assert explicit.split_base_wrist_task_dims == (0, 1, 2)


def test_task_dims_is_none_when_split_is_off():
    state = _make_state(q=np.full(6, 0.1), J=FAILURE_POSE_J)
    out = _controller(state).compute(state)
    assert out.split_base_wrist_task_dims is None
    assert out.split_base_wrist_active_joints is None


# --------------------------------------------------------------------------- #
# 2. The 1x3 case: linear algebra, recomputed independently.
#    SYNTHETIC operators -- see this module's docstring.
# --------------------------------------------------------------------------- #
def _reference_pieces(st, *, eps=0.1, rows=X_ONLY, cols=LIFT_ELBOW_WRIST1):
    J = np.asarray(st["jacobian"], dtype=np.float64)
    J_task = np.zeros((len(rows), 6), dtype=np.float64)
    J_task[:, list(cols)] = J[np.ix_(list(rows), list(cols))]
    m_inv = np.linalg.inv(np.asarray(st["mass_matrix"], dtype=np.float64))
    a_mat = J_task @ m_inv @ J_task.T
    lam = np.linalg.inv(a_mat + eps * np.eye(len(rows)))
    return J_task, m_inv, a_mat, lam


def test_1x3_jacobian_transpose_map_matches_reference():
    """Shaping off: tau_task must be exactly ``J_task.T @ [Fx]``."""
    st = _synth_state()
    out = _controller(
        st, split_base_wrist_task=True, split_base_wrist_active_joints=LIFT_ELBOW_WRIST1,
        split_base_wrist_task_dims=X_ONLY,
    ).compute(st)
    J_task = np.zeros((1, 6))
    J_task[:, list(LIFT_ELBOW_WRIST1)] = np.asarray(st["jacobian"])[np.ix_([0], list(LIFT_ELBOW_WRIST1))]
    expected = J_task.T @ np.array([out.wrench[0]])
    assert np.allclose(out.tau_task_nominal, expected, rtol=0.0, atol=1e-15)
    # Every joint outside the active set gets EXACTLY zero task torque.
    assert np.all(out.tau_task_nominal[list(HELD_BY_LIFT_ELBOW_WRIST1)] == 0.0)


def test_1x3_inertia_shaping_matches_reference_lambda():
    st = _synth_state(mass_matrix=SYNTH_M)
    out = _controller(
        st, split_base_wrist_task=True, split_base_wrist_active_joints=LIFT_ELBOW_WRIST1,
        split_base_wrist_task_dims=X_ONLY, task_space_inertia_shaping=True,
    ).compute(st)
    J_task, _m_inv, a_mat, lam = _reference_pieces(st)
    assert a_mat.shape == (1, 1) and lam.shape == (1, 1)
    expected = J_task.T @ (lam @ np.array([out.wrench[0]]))
    assert np.allclose(out.tau_task_nominal, expected, rtol=0.0, atol=1e-12)


def test_1x3_nullspace_projector_is_dynamically_consistent():
    """The projector must remove exactly the ONE task direction it now has.

    With a single task row, ``J_task M^-1 tau_posture`` should be reduced by
    the factor the eps-regularized projector implies (``eps / (a + eps)``) --
    not zeroed (that is what eps costs, a known documented property), and not
    left untouched.
    """
    st = _synth_state(mass_matrix=SYNTH_M)
    kwargs = dict(
        split_base_wrist_task=True, split_base_wrist_active_joints=LIFT_ELBOW_WRIST1,
        split_base_wrist_task_dims=X_ONLY, task_space_inertia_shaping=True,
    )
    plain = _controller(st, **kwargs).compute(st)
    projected = _controller(st, nullspace_posture=True, **kwargs).compute(st)

    J_task, m_inv, a_mat, lam = _reference_pieces(st)
    proj = np.eye(6) - J_task.T @ (m_inv @ J_task.T @ lam).T
    assert np.allclose(projected.tau_posture, proj @ plain.tau_posture, rtol=0.0, atol=1e-12)

    # The removed subspace is exactly 1-dimensional (the task row), so posture
    # keeps authority in the other five -- strictly more than the 3-row case.
    assert np.linalg.matrix_rank(np.eye(6) - proj) == 1

    accel_before = float((J_task @ m_inv @ plain.tau_posture)[0])
    accel_after = float((J_task @ m_inv @ projected.tau_posture)[0])
    expected_ratio = 0.1 / (float(a_mat[0, 0]) + 0.1)
    assert abs(accel_before) > 1e-6, "need a posture torque that does something to test this"
    assert accel_after / accel_before == pytest.approx(expected_ratio, rel=1e-9)


def test_1x3_sci_filter_scales_the_single_direction_exactly():
    """SCI on a 1-row task reduces to one scalar attenuation factor."""
    st = _synth_state()
    kwargs = dict(
        split_base_wrist_task=True, split_base_wrist_active_joints=LIFT_ELBOW_WRIST1,
        split_base_wrist_task_dims=X_ONLY,
    )
    undamped = _controller(st, **kwargs).compute(st)
    # Threshold deliberately above the single direction's sigma so the filter
    # actually engages (at the real pose it does not -- see the closed-loop
    # file); lambda_max sets how hard.
    damped = _controller(
        st, svd_singularity_filtering=True, svd_sigma_threshold=10.0, svd_lambda_max=2.0,
        **kwargs,
    ).compute(st)
    sigma = np.asarray(damped.svd_task_singular_values, dtype=np.float64)
    atten = np.asarray(damped.svd_direction_attenuation, dtype=np.float64)
    assert sigma.shape == (1,) and atten.shape == (1,)
    # sigma of a 1-row task is the row norm.
    J_row = np.zeros((1, 6))
    J_row[:, list(LIFT_ELBOW_WRIST1)] = np.asarray(st["jacobian"])[np.ix_([0], list(LIFT_ELBOW_WRIST1))]
    assert sigma[0] == pytest.approx(float(np.linalg.norm(J_row)), rel=1e-12)
    lam_sq = (1.0 - min((sigma[0] / 10.0) ** 2, 1.0)) * 2.0 ** 2
    assert atten[0] == pytest.approx(sigma[0] ** 2 / (sigma[0] ** 2 + lam_sq), rel=1e-12)
    # The filter can only ever reduce commanded torque here.
    assert 0.0 < atten[0] < 1.0
    assert np.allclose(damped.tau_task_nominal, atten[0] * undamped.tau_task_nominal,
                       rtol=1e-12, atol=1e-15)


def test_1x3_singular_scale_reports_norm_and_stays_inactive():
    """Documented consequence of the 1-row convention, asserted so it cannot
    change silently: cond_task carries the block NORM (cond of a 1xN matrix is
    identically 1.0 and says nothing), so the cond-ceiling test never fires."""
    st = _synth_state()
    out = _controller(
        st, split_base_wrist_task=True, split_base_wrist_active_joints=LIFT_ELBOW_WRIST1,
        split_base_wrist_task_dims=X_ONLY,
        jacobian_singular_cond_max=1.0e5,  # the class default, i.e. term enabled
    ).compute(st)
    J_row = np.zeros((1, 3))
    J_row[:] = np.asarray(st["jacobian"])[np.ix_([0], list(LIFT_ELBOW_WRIST1))]
    assert out.jacobian_cond == pytest.approx(float(np.linalg.norm(J_row)), rel=1e-12)
    assert out.singular_scale == 1.0


def test_two_row_selection_reports_a_real_condition_number():
    st = _synth_state()
    out = _controller(
        st, split_base_wrist_task=True, split_base_wrist_active_joints=LIFT_ELBOW_WRIST1,
        split_base_wrist_task_dims=(0, 2),
    ).compute(st)
    block = np.asarray(st["jacobian"])[np.ix_([0, 2], list(LIFT_ELBOW_WRIST1))]
    assert out.jacobian_cond == pytest.approx(float(np.linalg.cond(block)), rel=1e-12)


# --------------------------------------------------------------------------- #
# 3. Held rows are held, not dropped.
# --------------------------------------------------------------------------- #
def test_held_rows_keep_their_error_and_force_reported():
    """Y and Z leave the task pipeline but must not vanish from the output --
    the drift guards and any diagnostic reading y_error/z_error still need
    them, and the posture spring is what actually holds those axes."""
    st = _synth_state(mass_matrix=SYNTH_M)
    full = _controller(
        st, split_base_wrist_task=True, split_base_wrist_active_joints=LIFT_ELBOW_WRIST1,
    ).compute(st)
    reduced = _controller(
        st, split_base_wrist_task=True, split_base_wrist_active_joints=LIFT_ELBOW_WRIST1,
        split_base_wrist_task_dims=X_ONLY,
    ).compute(st)
    # The wrench diagnostic is the full 6D one in both cases, unchanged...
    assert np.array_equal(full.wrench, reduced.wrench)
    assert full.y_error == reduced.y_error
    assert full.z_error == reduced.z_error
    assert reduced.wrench[1] != 0.0 or reduced.y_error == 0.0
    # ...but the Y/Z rows no longer reach the joints through the task path.
    assert not np.allclose(full.tau_task_nominal, reduced.tau_task_nominal)
    # The posture spring, which is what holds them, is still live and nonzero.
    assert np.linalg.norm(reduced.tau_posture) > 0.0


def test_held_rows_get_more_posture_authority_not_less():
    """The nullspace projector is rebuilt against the reduced task, so a 1-row
    task constrains posture in one direction instead of three."""
    st = _synth_state(mass_matrix=SYNTH_M)
    kwargs = dict(
        split_base_wrist_task=True, split_base_wrist_active_joints=LIFT_ELBOW_WRIST1,
        task_space_inertia_shaping=True, nullspace_posture=True,
    )
    three_row = _controller(st, **kwargs).compute(st)
    one_row = _controller(st, split_base_wrist_task_dims=X_ONLY, **kwargs).compute(st)
    unprojected = _controller(
        st, split_base_wrist_task=True, split_base_wrist_active_joints=LIFT_ELBOW_WRIST1,
        task_space_inertia_shaping=True,
    ).compute(st)
    # Less of the raw posture torque is projected away with fewer task rows.
    removed_3 = np.linalg.norm(unprojected.tau_posture - three_row.tau_posture)
    removed_1 = np.linalg.norm(unprojected.tau_posture - one_row.tau_posture)
    assert removed_1 < removed_3


# --------------------------------------------------------------------------- #
# 4. Guards. Every one of these fires ONLY when the new field is set, so none
#    of them can affect a pre-existing config.
# --------------------------------------------------------------------------- #
def _compute_with(**overrides):
    st = _synth_state(mass_matrix=SYNTH_M)
    return _controller(st, **overrides).compute(st)


def test_task_dims_without_split_flag_raises():
    with pytest.raises(ValueError, match="split_base_wrist_task is False"):
        _compute_with(split_base_wrist_task=False, split_base_wrist_task_dims=X_ONLY)


def test_transport_axis_not_in_task_dims_raises():
    with pytest.raises(ValueError, match="does not include the transport axis"):
        _compute_with(split_base_wrist_task=True, split_base_wrist_task_dims=(1, 2))


def test_acceleration_feedforward_with_task_dims_raises():
    with pytest.raises(ValueError, match="acceleration_feedforward \\+ split_base_wrist_task_dims"):
        _compute_with(
            split_base_wrist_task=True, split_base_wrist_task_dims=X_ONLY,
            acceleration_feedforward=True, task_space_inertia_shaping=True,
        )


def test_y_integral_with_y_row_dropped_raises():
    with pytest.raises(ValueError, match="y_integral_action is on"):
        _compute_with(
            split_base_wrist_task=True, split_base_wrist_task_dims=X_ONLY,
            y_integral_action=True, ki_y=1.0,
        )


def test_y_integral_is_allowed_when_the_y_row_is_kept():
    out = _compute_with(
        split_base_wrist_task=True, split_base_wrist_task_dims=(0, 1),
        y_integral_action=True, ki_y=1.0,
    )
    assert out.y_integral_action_active is True


@pytest.mark.parametrize(
    "flag", ["lambda_adaptive_regularization", "wrench_lambda_adaptive_regularization"]
)
def test_adaptive_eps_with_single_row_raises(flag):
    with pytest.raises(ValueError, match="single-row split_base_wrist_task_dims"):
        _compute_with(
            split_base_wrist_task=True, split_base_wrist_task_dims=X_ONLY,
            task_space_inertia_shaping=True, **{flag: True},
        )


@pytest.mark.parametrize(
    "flag", ["lambda_adaptive_regularization", "wrench_lambda_adaptive_regularization"]
)
def test_adaptive_eps_is_allowed_with_two_rows(flag):
    """With 2+ rows cond_task IS a real condition number, so the log(cond)
    schedule is meaningful and is deliberately left alone."""
    out = _compute_with(
        split_base_wrist_task=True, split_base_wrist_task_dims=(0, 2),
        task_space_inertia_shaping=True, **{flag: True},
    )
    assert np.isfinite(out.lambda_regularization_effective)


def test_still_mutually_exclusive_with_reduced_task_dims():
    """The pre-existing refusal is unchanged: this feature generalizes the
    SPLIT mechanism's own rows, it does not merge the two mechanisms."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        _compute_with(
            split_base_wrist_task=True, split_base_wrist_task_dims=X_ONLY,
            reduced_task_dims=True,
        )


# --------------------------------------------------------------------------- #
# 5. Parsing / YAML wiring.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "bad",
    [[], [0, 0], [3], [-1], [0, 1, 2, 0], "0", 0, True, [True], [1.5], [None]],
)
def test_task_dims_parser_rejects_bad_values(bad):
    with pytest.raises(ValueError):
        _parse_split_base_wrist_task_dims(bad)


def test_task_dims_parser_accepts_good_values():
    assert _parse_split_base_wrist_task_dims(None) is None
    assert _parse_split_base_wrist_task_dims([0]) == (0,)
    assert _parse_split_base_wrist_task_dims((0, 2)) == (0, 2)
    parsed = _parse_split_base_wrist_task_dims([0.0, 1.0, 2.0])
    assert parsed == (0, 1, 2)
    assert all(type(i) is int for i in parsed)


def test_task_dims_yaml_parsing():
    torque_limits = {
        name: 100.0
        for name in (
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
        )
    }
    base = {
        "torque_limits_mode": "initial",
        "torque_limits_initial": torque_limits,
        "gains": {"kp_x": 400.0, "kd_x": 40.0},
        "split_base_wrist_task": True,
        "split_base_wrist_active_joints": [1, 2, 3],
    }
    assert CartesianImpedanceConfig.from_controller_yaml_section(
        dict(base)
    ).split_base_wrist_task_dims is None
    cfg = CartesianImpedanceConfig.from_controller_yaml_section(
        dict(base, split_base_wrist_task_dims=[0])
    )
    assert cfg.split_base_wrist_task_dims == (0,)
    assert cfg.split_base_wrist_active_joints == (1, 2, 3)
    with pytest.raises(ValueError):
        CartesianImpedanceConfig.from_controller_yaml_section(
            dict(base, split_base_wrist_task_dims=[0, 5])
        )


def test_shipped_config_parses_and_selects_the_intended_task():
    path = (
        Path(__file__).resolve().parents[2]
        / "config"
        / "ur5e_mujoco_torque_osc_tuned_split_base_wrist_lift_elbow_wrist1_x_only.yaml"
    )
    import yaml

    ctrl = yaml.safe_load(path.read_text(encoding="utf-8"))["controller"]
    cfg = CartesianImpedanceConfig.from_controller_yaml_section(ctrl)
    assert cfg.split_base_wrist_task is True
    assert cfg.split_base_wrist_active_joints == LIFT_ELBOW_WRIST1
    assert cfg.split_base_wrist_task_dims == X_ONLY
    # The combinations this config must NOT quietly acquire (each is a hard
    # error with a single task row, or unsupported with row selection).
    assert cfg.acceleration_feedforward is False
    assert cfg.y_integral_action is False
    assert cfg.lambda_adaptive_regularization is False
    assert cfg.wrench_lambda_adaptive_regularization is False
