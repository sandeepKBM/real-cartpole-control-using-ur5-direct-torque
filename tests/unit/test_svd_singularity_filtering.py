"""Unit tests for ``svd_singularity_filtering`` -- singularity-consistent
inversion (SCI) in ``controller_core/x_axis_cartesian_impedance``.

Pure numpy, no simulator required. See the flag's docstring on
``CartesianImpedanceConfig`` for the full derivation; the short version is that
the two pre-existing near-singularity mechanisms (a scalar ``eps*I`` inside
Lambda, and ``singular_scale``) are both UNIFORM across task directions, so at a
UR wrist singularity -- where exactly ONE task direction is genuinely lost --
they throw away authority in the five directions the robot still has.

Two classes of evidence here, deliberately:
  * synthetic matrices with a hand-placed small singular value, to isolate the
    linear algebra; and
  * the REAL UR5e Jacobian and mass matrix at ``HEIGHT_ALPHA_0_5_Q``
    (``wrist_2 == 0``, cond(J) = 7.3e16), captured from
    ``assets/ur5e_torque/scene.xml``. A synthetic well-conditioned constant
    Jacobian on its own would not exercise this feature at all.
``tests/mujoco/test_svd_singularity_filtering_closed_loop.py`` re-derives the
real matrices below straight from the model and asserts they still match, so
these literals cannot silently drift away from the robot they claim to describe.
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

# ---------------------------------------------------------------------------
# Real UR5e data at the genuine wrist singularity (wrist_2 == 0).
# q = HEIGHT_ALPHA_0_5_Q (hardware/poses.py), model assets/ur5e_torque/scene.xml.
# cond(J) = 7.28e16; sigma(J) = [2.117, 1.433, 0.573, 0.198, 0.119, 2.9e-17].
# Mass-weighted sigma = sqrt(eigh(J M^-1 J^T)) = [4.350, 3.356, 0.531, 0.315,
# 0.1997, ~0]: ONE dead direction, seven decades below the next one up.
# ---------------------------------------------------------------------------
REAL_Q = np.array(
    [0.0, -0.8353981633974483, -1.2, -0.9853981633974482, 0.0, 0.0], dtype=np.float64
)
REAL_EE_POS = np.array(
    [-0.12153311115561133, -0.23399999999999996, 0.927883883722642], dtype=np.float64
)
REAL_EE_QUAT = np.array(
    [-0.04268198965140774, -0.04268198965140774, -0.7058174323147571, 0.7058174323147574],
    dtype=np.float64,
)
REAL_JACOBIAN = np.array(
    [
        [0.23399999999999999, -0.76488388372264171, -0.4497193149003888,
         -0.099271299103758759, 0.099271299103758842, 0.0],
        [-0.12153311115561133, -0.0, 0.0, 0.0, 1.3877787807814457e-17, 0.0],
        [0.0, -0.12153311115561127, 0.16359193958367962,
         -0.012050276936736649, 0.012050276936736663, 0.0],
        [0.0, 0.0, 0.0, 0.0, -0.12050276936736663, 0.0],
        [0.0, -0.99999999999999956, -0.99999999999999956,
         -0.99999999999999911, 0.0, -0.99999999999999933],
        [1.0, 0.0, 0.0, 0.0, 0.99271299103758848, 0.0],
    ],
    dtype=np.float64,
)
REAL_MASS_MATRIX = np.array(
    [
        [0.69595602214919505, -0.45450345624265137, -0.15143374002517992,
         -0.020175471156186127, 0.0053323637967015265, -2.7531647349795699e-18],
        [-0.45450345624265137, 2.6841319428012977, 0.90970553759982753,
         0.11421709286904874, -0.011282016701919655, 0.00013213400000002421],
        [-0.15143374002517992, 0.90970553759982753, 0.73356722614835568,
         0.065331099274178336, -0.0062204667353832656, 0.00013213400000001982],
        [-0.020175471156186127, 0.11421709286904874, 0.065331099274178336,
         0.11933093400000017, -0.0014577091869999999, 0.00013213400000001491],
        [0.0053323637967015265, -0.011282016701919655, -0.0062204667353832656,
         -0.0014577091869999999, 0.10341817569855172, -8.8702992622759588e-19],
        [-2.7531647349795699e-18, 0.00013213400000002421, 0.00013213400000001982,
         0.00013213400000001491, -8.8702992622759588e-19, 0.10013213400000003],
    ],
    dtype=np.float64,
)

# The tuned OSC gain set (config/ur5e_mujoco_torque_osc_tuned.yaml), which is
# what any real config enabling this flag would build on.
TUNED_GAINS = dict(
    kp_x=400.0, kd_x=40.0, kp_y=300.0, kd_y=30.0, kp_z=300.0, kd_z=30.0,
    kp_rot=0.0, kd_rot=10.0, kp_posture=25.0, kd_posture=6.0, kd_joint=4.0,
    lambda_regularization=0.1,
)


def _state(*, J, M=None, target_x=None, q=None, ee_pos=None, wrench_probe=None):
    ee_pos = REAL_EE_POS.copy() if ee_pos is None else np.asarray(ee_pos, dtype=np.float64)
    st = {
        "time": 0.0,
        "dt_s": 0.002,
        "q": REAL_Q.copy() if q is None else np.asarray(q, dtype=np.float64),
        "qd": np.zeros(6),
        "ee_pos": ee_pos,
        "ee_quat": REAL_EE_QUAT.copy(),
        "ee_lin_vel": np.zeros(3),
        "ee_ang_vel": np.zeros(3),
        "target_x": float(ee_pos[0]) if target_x is None else float(target_x),
        "target_x_vel": 0.0,
        "jacobian": np.asarray(J, dtype=np.float64),
    }
    if M is not None:
        st["mass_matrix"] = np.asarray(M, dtype=np.float64)
    del wrench_probe
    return st


def _controller(**overrides) -> XAxisCartesianImpedanceController:
    kwargs = dict(TUNED_GAINS)
    # Keep clipping/backtracking out of the way unless a test asks for it: this
    # module is about the wrench-shaping math, and the torque box is a separate,
    # deliberately untouched safety stage.
    kwargs.setdefault("tau_max_nm", np.full(6, 1e6))
    kwargs.update(overrides)
    cfg = CartesianImpedanceConfig(**kwargs)
    ctl = XAxisCartesianImpedanceController(cfg)
    ctl.reset_from_state(_state(J=REAL_JACOBIAN, M=REAL_MASS_MATRIX))
    return ctl


def _task_accel(J, M, tau):
    """The task-space acceleration a commanded joint torque actually produces."""
    return np.asarray(J) @ np.linalg.inv(np.asarray(M)) @ np.asarray(tau)


# ===========================================================================
# (c) Flag off == today's behavior. The full byte-identity proof is the golden
# scripted rollout across 16 flag combinations; these lock the property into
# the suite.
# ===========================================================================


def test_flag_off_is_bit_identical_to_never_setting_it():
    for extra in (
        dict(task_space_inertia_shaping=True, nullspace_posture=True),
        dict(task_space_inertia_shaping=True, jacobian_singular_cond_max=1.0e5),
        dict(nullspace_posture=True),
        dict(),  # plain Jacobian-transpose PD, no Lambda at all
    ):
        ref = _controller(**extra)
        off = _controller(svd_singularity_filtering=False, **extra)
        st = _state(J=REAL_JACOBIAN, M=REAL_MASS_MATRIX, target_x=REAL_EE_POS[0] + 0.01)
        out_ref, out_off = ref.compute(st), off.compute(st)
        assert np.array_equal(out_ref.tau, out_off.tau)
        assert np.array_equal(out_ref.tau_preclip, out_off.tau_preclip)
        assert np.array_equal(out_ref.tau_task_nominal, out_off.tau_task_nominal)
        assert np.array_equal(out_ref.tau_posture, out_off.tau_posture)
        assert out_ref.singular_scale == out_off.singular_scale


def test_flag_off_leaves_diagnostics_none():
    out = _controller(task_space_inertia_shaping=True).compute(
        _state(J=REAL_JACOBIAN, M=REAL_MASS_MATRIX, target_x=REAL_EE_POS[0] + 0.01)
    )
    assert out.svd_singularity_filtering_active is False
    assert out.svd_task_singular_values is None
    assert out.svd_damping_lambda is None
    assert out.svd_direction_attenuation is None


# ===========================================================================
# Linear-algebra correctness on synthetic matrices with a hand-placed small
# singular value (isolation only -- the real-robot tests below are the ones
# that matter for the motivating case).
# ===========================================================================


def test_synthetic_undamped_directions_are_inverted_exactly():
    """Directions at/above the threshold get the EXACT analytic inverse.

    That is what makes this "singularity-consistent" rather than "uniform":
    Lambda_SCI must agree with the undamped Lambda = A^-1 on the
    well-conditioned subspace to full precision, not merely approximately.
    """
    rng = np.random.default_rng(7)
    basis, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    sigma = np.array([1.7, 0.9, 1e-9])  # one deliberately dead direction
    a_mat = basis @ np.diag(sigma**2) @ basis.T
    # Reproduce the controller's construction directly.
    ctl = _controller(svd_singularity_filtering=True)
    eigvals, eigvecs = np.linalg.eigh(a_mat)
    sig = np.sqrt(np.maximum(eigvals, 0.0))
    lam, lam_sq, atten = ctl._sci_direction_damping(sig)
    lambda_sci = (eigvecs * (1.0 / (sig**2 + lam_sq))) @ eigvecs.T

    well = sig >= ctl.cfg.svd_sigma_threshold
    assert well.sum() == 2
    assert np.allclose(lam[well], 0.0, atol=0.0)
    assert np.allclose(atten[well], 1.0, atol=0.0)
    # Exact inverse restricted to the well-conditioned subspace.
    for i in np.flatnonzero(well):
        u = eigvecs[:, i]
        assert lambda_sci @ u == pytest.approx(u / sig[i] ** 2, rel=1e-12)


def test_synthetic_near_singular_direction_is_attenuated_not_amplified():
    ctl = _controller(svd_singularity_filtering=True)
    sig = np.array([2.0, 0.5, 1e-3, 1e-9, 0.0])
    lam, lam_sq, atten = ctl._sci_direction_damping(sig)
    assert np.all(np.isfinite(lam))
    assert np.all(np.isfinite(atten))
    assert np.all(atten >= 0.0) and np.all(atten <= 1.0)
    # Monotone: smaller sigma -> less surviving response.
    assert atten[0] == 1.0 and atten[1] == 1.0
    assert atten[2] > atten[3] > atten[4]
    assert atten[4] == 0.0
    # The damped reciprocal (the actual per-direction torque gain) also stays
    # bounded and goes to zero, which is the whole point of the back-off.
    gain = sig / np.maximum(sig**2 + lam_sq, np.finfo(np.float64).tiny)
    assert np.all(np.isfinite(gain))
    assert gain[4] == 0.0
    assert gain[3] < gain[2]


def test_damping_is_continuous_at_the_threshold():
    """No torque step as a direction crosses into or out of the damped set.

    The quantity that actually multiplies the response is the attenuation
    a_i = sigma^2/(sigma^2 + lambda^2), and lambda^2 (not lambda) is what the
    scheme's bracket makes vanish linearly at the threshold -- lambda itself
    approaches 0 like a square root, which is continuous but with unbounded
    slope. Both are asserted, at the tolerance each actually supports.
    """
    ctl = _controller(svd_singularity_filtering=True)
    thr = ctl.cfg.svd_sigma_threshold
    below = ctl._sci_direction_damping(np.array([thr * (1 - 1e-12)]))
    at = ctl._sci_direction_damping(np.array([thr]))
    above = ctl._sci_direction_damping(np.array([thr * (1 + 1e-12)]))
    # lambda^2 -> 0 from below; exactly 0 at and above the threshold.
    assert below[1][0] == pytest.approx(0.0, abs=1e-12)
    assert at[1][0] == 0.0
    assert above[1][0] == 0.0
    # lambda itself is continuous too, just with a sqrt-shaped approach.
    assert below[0][0] == pytest.approx(0.0, abs=1e-5)
    # The surviving fraction is continuous to full precision across the seam.
    assert below[2][0] == pytest.approx(1.0, rel=1e-9)
    assert at[2][0] == 1.0
    assert above[2][0] == 1.0
    # Sweeping across the threshold produces no JUMP: refining the grid shrinks
    # the largest step proportionally, which a genuine discontinuity would not
    # do. (The transition is continuous but deliberately steep -- with the
    # defaults svd_lambda_max^2 / svd_sigma_threshold^2 == 40, so most of the
    # attenuation ramp is squeezed into the last ~0.1% below the threshold. See
    # the flag's docstring; this is a real tuning characteristic, not a bug.)
    steps = []
    for n in (4001, 40001, 400001):
        sweep = np.linspace(thr * 0.5, thr * 1.5, n)
        steps.append(float(np.max(np.abs(np.diff(ctl._sci_direction_damping(sweep)[2])))))
    assert steps[1] < steps[0] / 5.0
    assert steps[2] < steps[1] / 5.0


def test_lambda_max_reproduces_todays_damping_in_a_fully_lost_direction():
    """At sigma == 0 the SCI damping equals the historical uniform eps.

    svd_lambda_max defaults to sqrt(0.1) so lambda_max^2 == the tuned configs'
    lambda_regularization: SCI is never MORE aggressive than today in the
    direction that is actually lost, it only stops punishing the others.
    """
    ctl = _controller(svd_singularity_filtering=True)
    _lam, lam_sq, _atten = ctl._sci_direction_damping(np.array([0.0]))
    assert lam_sq[0] == pytest.approx(ctl.cfg.lambda_regularization, rel=1e-12)


# ===========================================================================
# (a) + (b) The real robot at the real singularity.
# ===========================================================================


def test_real_pose_has_exactly_one_lost_direction():
    """Guards the premise of the whole feature against model drift."""
    a_mat = REAL_JACOBIAN @ np.linalg.inv(REAL_MASS_MATRIX) @ REAL_JACOBIAN.T
    sig = np.sqrt(np.maximum(np.linalg.eigvalsh(a_mat), 0.0))
    assert np.linalg.cond(REAL_JACOBIAN) > 1e10
    assert (sig < 1e-6).sum() == 1, "expected exactly one genuinely lost direction"
    assert sig[sig >= 1e-6].min() > 0.19, "the surviving directions are well-conditioned"


def test_real_pose_sci_preserves_tracking_where_uniform_damping_destroys_it():
    """(a) Well-conditioned response is close to the undamped ideal.

    The controller is commanded a pure task-axis (world X) acceleration at the
    real wrist singularity. X is NOT the lost direction here -- the lost one is
    a Y/Rx mix -- so an ideal controller should deliver the commanded X
    acceleration in full.
    """
    st = _state(J=REAL_JACOBIAN, M=REAL_MASS_MATRIX, target_x=REAL_EE_POS[0] + 0.01)
    common = dict(task_space_inertia_shaping=True, nullspace_posture=True,
                  jacobian_singular_cond_max=1.0e18)

    uniform = _controller(**common).compute(st)
    sci = _controller(svd_singularity_filtering=True, **common).compute(st)

    commanded_ax = uniform.wrench[0]
    assert commanded_ax == pytest.approx(400.0 * 0.01, rel=1e-9)

    a_uniform = _task_accel(REAL_JACOBIAN, REAL_MASS_MATRIX, uniform.tau_task_nominal)
    a_sci = _task_accel(REAL_JACOBIAN, REAL_MASS_MATRIX, sci.tau_task_nominal)

    # SCI delivers the commanded task acceleration essentially exactly.
    assert a_sci[0] / commanded_ax == pytest.approx(1.0, rel=1e-9)
    # Today's uniform damping delivers well under 80% of it -- collateral damage
    # from eps sized for a direction X has nothing to do with.
    assert a_uniform[0] / commanded_ax < 0.8
    assert a_sci[0] > a_uniform[0]


def test_real_pose_sci_removes_the_uniform_damping_cross_axis_leak():
    """Uniform eps couples the X command into Y/Z/rotation; SCI does not."""
    st = _state(J=REAL_JACOBIAN, M=REAL_MASS_MATRIX, target_x=REAL_EE_POS[0] + 0.01)
    common = dict(task_space_inertia_shaping=True, nullspace_posture=True,
                  jacobian_singular_cond_max=1.0e18)
    a_uniform = _task_accel(
        REAL_JACOBIAN, REAL_MASS_MATRIX, _controller(**common).compute(st).tau_task_nominal
    )
    a_sci = _task_accel(
        REAL_JACOBIAN,
        REAL_MASS_MATRIX,
        _controller(svd_singularity_filtering=True, **common).compute(st).tau_task_nominal,
    )
    leak_uniform = float(np.linalg.norm(a_uniform[1:]))
    leak_sci = float(np.linalg.norm(a_sci[1:]))
    assert leak_uniform > 0.1, "the uniform-eps leak this test is about must be present"
    assert leak_sci < 1e-9
    assert leak_sci < leak_uniform / 1e6


def test_real_pose_command_along_the_lost_direction_is_attenuated_and_finite():
    """(b) Along the genuinely lost direction the response goes toward zero.

    Rather than blowing up (undamped 1/sigma^2 with sigma ~ 1e-8) or emitting
    NaN/inf. Driven through the controller's own Lambda by commanding a task
    acceleration aligned with the lost singular direction.
    """
    a_mat = REAL_JACOBIAN @ np.linalg.inv(REAL_MASS_MATRIX) @ REAL_JACOBIAN.T
    eigvals, eigvecs = np.linalg.eigh(a_mat)
    lost = eigvecs[:, int(np.argmin(eigvals))]

    ctl = _controller(
        svd_singularity_filtering=True,
        task_space_inertia_shaping=True,
        jacobian_singular_cond_max=1.0e18,
    )
    st = _state(J=REAL_JACOBIAN, M=REAL_MASS_MATRIX)
    ctl.compute(st)  # prime
    # Reproduce the wrench-shaping stage on a unit command along `lost`.
    out = ctl.compute(st)
    sig = out.svd_task_singular_values
    lam_sq = out.svd_damping_lambda ** 2
    lambda_sci = (eigvecs * (1.0 / (sig**2 + lam_sq))) @ eigvecs.T

    force_lost = lambda_sci @ lost
    tau_lost = REAL_JACOBIAN.T @ force_lost
    assert np.all(np.isfinite(force_lost))
    assert np.all(np.isfinite(tau_lost))
    # The joint torque commanded on account of the lost direction is negligible.
    assert float(np.linalg.norm(tau_lost)) < 1e-6
    # ...whereas a well-conditioned direction still commands real torque.
    well = eigvecs[:, int(np.argmax(eigvals))]
    assert float(np.linalg.norm(REAL_JACOBIAN.T @ (lambda_sci @ well))) > 1e-3
    # And the reported attenuation identifies exactly one backed-off direction.
    assert (out.svd_direction_attenuation < 0.5).sum() == 1
    assert np.all(out.svd_direction_attenuation[out.svd_direction_attenuation > 0.5] == 1.0)


def test_real_pose_produces_no_nan_or_inf_anywhere():
    for extra in (
        dict(task_space_inertia_shaping=True, nullspace_posture=True),
        dict(task_space_inertia_shaping=True),
        dict(nullspace_posture=True),
        dict(),
    ):
        out = _controller(svd_singularity_filtering=True, **extra).compute(
            _state(J=REAL_JACOBIAN, M=REAL_MASS_MATRIX, target_x=REAL_EE_POS[0] + 0.05)
        )
        for name in ("tau", "tau_preclip", "tau_task_nominal", "tau_posture", "wrench"):
            arr = np.asarray(getattr(out, name), dtype=np.float64)
            assert np.all(np.isfinite(arr)), f"{name} not finite for {extra}"


def test_real_pose_reported_attenuation_beats_uniform_in_every_direction():
    """The headline claim, as a direct per-direction comparison."""
    out = _controller(
        svd_singularity_filtering=True, task_space_inertia_shaping=True,
        jacobian_singular_cond_max=1.0e18,
    ).compute(_state(J=REAL_JACOBIAN, M=REAL_MASS_MATRIX, target_x=REAL_EE_POS[0] + 0.01))
    sig = out.svd_task_singular_values
    eps = 0.1  # the tuned uniform lambda_regularization
    uniform_atten = sig**2 / (sig**2 + eps)
    sci_atten = out.svd_direction_attenuation
    assert np.all(sci_atten >= uniform_atten - 1e-15)
    # And it is a large difference, not a rounding-level one, for the
    # second-smallest (still perfectly usable) direction.
    order = np.argsort(sig)
    worst_well_conditioned = order[1]
    assert uniform_atten[worst_well_conditioned] < 0.35
    assert sci_atten[worst_well_conditioned] == 1.0


# ===========================================================================
# The shaping-off branch (plain Jacobian-transpose force law).
# ===========================================================================


def test_shaping_off_branch_never_amplifies_commanded_torque():
    st = _state(J=REAL_JACOBIAN, M=REAL_MASS_MATRIX, target_x=REAL_EE_POS[0] + 0.02)
    plain = _controller(jacobian_singular_cond_max=1.0e18).compute(st)
    sci = _controller(
        svd_singularity_filtering=True, jacobian_singular_cond_max=1.0e18
    ).compute(st)
    assert np.linalg.norm(sci.wrench_accel_ff) == 0.0
    assert float(np.linalg.norm(sci.tau_task_nominal)) <= float(
        np.linalg.norm(plain.tau_task_nominal)
    ) + 1e-12
    assert sci.svd_task_singular_values is not None
    assert np.all(sci.svd_direction_attenuation <= 1.0 + 1e-15)


def test_shaping_off_sci_replaces_singular_scale_instead_of_collapsing():
    """With the class-default cond_max, singular_scale collapses the whole
    wrench at this pose (the documented freeze). SCI keeps the well-conditioned
    directions instead of scaling everything to ~1e-12."""
    st = _state(J=REAL_JACOBIAN, M=REAL_MASS_MATRIX, target_x=REAL_EE_POS[0] + 0.02)
    collapsed = _controller(jacobian_singular_cond_max=1.0e5).compute(st)
    sci = _controller(
        svd_singularity_filtering=True, jacobian_singular_cond_max=1.0e5
    ).compute(st)
    assert collapsed.singular_scale < 1e-10
    assert float(np.linalg.norm(collapsed.tau_task_nominal)) < 1e-8
    # SCI bypasses singular_scale entirely (jacobian_singular_cond_max itself
    # is untouched -- it still governs the default path).
    assert sci.singular_scale == 1.0
    assert float(np.linalg.norm(sci.tau_task_nominal)) > 1.0


# ===========================================================================
# Scope guarantees and misconfiguration.
# ===========================================================================


def test_nullspace_projector_is_not_affected_by_sci():
    """SCI is scoped to the wrench-shaping Lambda only; the dynamically
    consistent posture projector keeps its uniform-eps Lambda."""
    st = _state(J=REAL_JACOBIAN, M=REAL_MASS_MATRIX, target_x=REAL_EE_POS[0] + 0.01,
                q=REAL_Q + 0.05)
    common = dict(task_space_inertia_shaping=True, nullspace_posture=True,
                  jacobian_singular_cond_max=1.0e18)
    a = _controller(**common).compute(st)
    b = _controller(svd_singularity_filtering=True, **common).compute(st)
    # Same q_rest, same gains, same projector -> byte-identical posture torque
    # (the task torque, by contrast, must differ).
    assert np.array_equal(a.tau_posture, b.tau_posture)
    assert not np.array_equal(a.tau_task_nominal, b.tau_task_nominal)


def test_conflicts_with_lambda_diagonal_shaping():
    ctl = _controller(
        svd_singularity_filtering=True, lambda_diagonal_shaping=True,
        task_space_inertia_shaping=True,
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        ctl.compute(_state(J=REAL_JACOBIAN, M=REAL_MASS_MATRIX))


def test_conflicts_with_wrench_lambda_adaptive_regularization():
    ctl = _controller(
        svd_singularity_filtering=True, wrench_lambda_adaptive_regularization=True,
        task_space_inertia_shaping=True,
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        ctl.compute(_state(J=REAL_JACOBIAN, M=REAL_MASS_MATRIX))


def test_coexists_with_nullspace_only_adaptive_regularization():
    """lambda_adaptive_regularization schedules the NULLSPACE eps, which SCI
    does not touch -- these two must remain compatible."""
    out = _controller(
        svd_singularity_filtering=True, lambda_adaptive_regularization=True,
        task_space_inertia_shaping=True, nullspace_posture=True,
    ).compute(_state(J=REAL_JACOBIAN, M=REAL_MASS_MATRIX, target_x=REAL_EE_POS[0] + 0.01))
    assert np.all(np.isfinite(out.tau))
    assert out.lambda_adaptive_regularization_active is True
    assert out.svd_singularity_filtering_active is True


@pytest.mark.parametrize(
    "bad", [dict(svd_sigma_threshold=0.0), dict(svd_sigma_threshold=-1.0),
            dict(svd_lambda_max=0.0), dict(svd_lambda_max=-0.1),
            dict(svd_lambda_max=float("nan"))]
)
def test_degenerate_parameters_raise_rather_than_emitting_inf(bad):
    ctl = _controller(svd_singularity_filtering=True, task_space_inertia_shaping=True, **bad)
    with pytest.raises(ValueError):
        ctl.compute(_state(J=REAL_JACOBIAN, M=REAL_MASS_MATRIX))


_YAML_TORQUE_LIMITS = {
    "torque_limits_initial": {
        "shoulder_pan_joint": 8.0, "shoulder_lift_joint": 8.0, "elbow_joint": 8.0,
        "wrist_1_joint": 2.5, "wrist_2_joint": 2.5, "wrist_3_joint": 2.5,
    }
}


def test_yaml_section_roundtrip():
    cfg = CartesianImpedanceConfig.from_controller_yaml_section(
        {
            "gains": {},
            **_YAML_TORQUE_LIMITS,
            "svd_singularity_filtering": True,
            "svd_sigma_threshold": 0.07,
            "svd_lambda_max": 0.25,
        }
    )
    assert cfg.svd_singularity_filtering is True
    assert cfg.svd_sigma_threshold == pytest.approx(0.07)
    assert cfg.svd_lambda_max == pytest.approx(0.25)
    default = CartesianImpedanceConfig.from_controller_yaml_section(
        {"gains": {}, **_YAML_TORQUE_LIMITS}
    )
    assert default.svd_singularity_filtering is False
    assert default.svd_sigma_threshold == pytest.approx(0.05)
    assert default.svd_lambda_max == pytest.approx(np.sqrt(0.1))


@pytest.mark.parametrize(
    "label,extra",
    [
        ("wrist_orientation_task",
         dict(wrist_orientation_task=True, kp_rot_wrist=5.0, kd_rot_wrist=1.0)),
        ("task_lock_wrist_2", dict(task_lock_wrist_2=True)),
        ("task_lock_whole_wrist",
         dict(task_lock_wrist_1=True, task_lock_wrist_2=True, task_lock_wrist_3=True)),
        ("acceleration_feedforward", dict(acceleration_feedforward=True)),
        ("posture_per_joint_gains",
         dict(posture_kp_by_joint=np.full(6, 25.0), posture_kd_by_joint=np.full(6, 6.0))),
        ("friction_feedforward", dict(friction_feedforward=True)),
    ],
)
def test_coexists_with_the_other_optional_mechanisms(label, extra):
    """None of these share the wrench-shaping Lambda, so none should conflict --
    asserted rather than assumed. ``task_lock_*`` in particular zeroes Jacobian
    columns and so manufactures ADDITIONAL near-zero singular directions, which
    SCI must attenuate rather than divide by."""
    st = _state(J=REAL_JACOBIAN, M=REAL_MASS_MATRIX, target_x=REAL_EE_POS[0] + 0.01)
    st["target_x_accel"] = 0.5
    out = _controller(
        svd_singularity_filtering=True, task_space_inertia_shaping=True,
        nullspace_posture=True, jacobian_singular_cond_max=1.0e18, **extra
    ).compute(st)
    assert np.all(np.isfinite(out.tau)), label
    assert np.all(np.isfinite(out.svd_direction_attenuation)), label
    atten = out.svd_direction_attenuation
    assert np.all(atten >= 0.0) and np.all(atten <= 1.0), label
    if label == "task_lock_whole_wrist":
        # Three locked wrist joints remove three task directions; all three must
        # be backed off, not inverted.
        assert (atten < 1e-6).sum() >= 3


def test_mass_matrix_absent_falls_back_to_the_kinematic_filter():
    """With no mass_matrix the Lambda block falls back to M = I, so SCI filters
    J J^T -- still well-defined, still one lost direction at this pose."""
    out = _controller(
        svd_singularity_filtering=True, task_space_inertia_shaping=True,
        jacobian_singular_cond_max=1.0e18,
    ).compute(_state(J=REAL_JACOBIAN, M=None, target_x=REAL_EE_POS[0] + 0.01))
    assert out.mass_matrix_provided is False
    assert np.all(np.isfinite(out.tau))
    assert (out.svd_direction_attenuation < 1e-6).sum() == 1
    # sigma then matches the RAW Jacobian's singular values, not mass-weighted.
    got = np.sort(out.svd_task_singular_values)
    want = np.sort(np.linalg.svd(REAL_JACOBIAN, compute_uv=False))
    # Every resolvable singular value agrees closely...
    assert np.allclose(got[1:], want[1:], rtol=1e-8)
    # ...but the smallest does NOT, and deliberately so: going through
    # eigh(J J^T) squares the conditioning, so a singular value that a direct
    # SVD resolves at 2.9e-17 comes back as ~1.3e-8 (about sqrt(eps)). This is
    # the classic argument against forming J J^T -- and it is harmless HERE
    # precisely because SCI thresholds rather than divides: anything below
    # svd_sigma_threshold (0.05) is treated as lost either way, and the
    # resulting attenuation is identical to ~1e-15.
    assert got[0] < 1e-6 and want[0] < 1e-6
    assert out.svd_direction_attenuation.min() < 1e-12


def test_works_with_reduced_task_dims_and_split_base_wrist():
    """Both reshape J_task; SCI must handle a non-square/reduced task."""
    for extra in (
        dict(split_base_wrist_task=True),
        dict(reduced_task_dims=True, task_dim_rx=False, task_dim_ry=False, task_dim_rz=False),
    ):
        out = _controller(
            svd_singularity_filtering=True, task_space_inertia_shaping=True,
            nullspace_posture=True, jacobian_singular_cond_max=1.0e18, **extra
        ).compute(_state(J=REAL_JACOBIAN, M=REAL_MASS_MATRIX, target_x=REAL_EE_POS[0] + 0.01))
        assert np.all(np.isfinite(out.tau))
        assert out.svd_task_singular_values.shape == (3,)
