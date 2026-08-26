"""Pure-numpy unit tests for controller_core/damped_least_squares.py.

Per that module's docstring: away from a singularity this must reduce to
the ordinary Jacobian inverse (lambda ~= 0); NEAR a singularity the point is
a BOUNDED joint velocity, not accurate Cartesian tracking, so the
near-singular tests assert boundedness, not qd ~= J^-1 xd.
"""

from __future__ import annotations

import numpy as np
import pytest

from controller_core.damped_least_squares import (
    DampedLeastSquaresConfig,
    DampedLeastSquaresResult,
    damped_least_squares_qd,
)

# The exact singular ARM_Q0 pose named in this repo's AGENTS.md and in the
# task that created this module.
ARM_Q0_SINGULAR = np.array(
    [-2.3688, -2.1801, -1.8838, -0.7962, 0.004714693, 0.0206], dtype=np.float64
)


def _well_conditioned_jacobian() -> np.ndarray:
    """A hand-built 6x6 Jacobian with singular values well clear of any
    plausible sigma0 (all > 1.0) -- deterministic, no robot/mujoco needed."""
    rng = np.random.default_rng(0)
    # Random orthogonal-ish matrix scaled so its smallest singular value is
    # comfortably above typical sigma0 values (~0.05).
    A = rng.normal(size=(6, 6))
    U, _, Vt = np.linalg.svd(A)
    s = np.array([5.0, 4.5, 4.0, 3.5, 3.0, 2.5], dtype=np.float64)
    return U @ np.diag(s) @ Vt


def _near_singular_jacobian(sigma_min: float) -> np.ndarray:
    """A 6x6 Jacobian whose smallest singular value is exactly ``sigma_min``,
    everything else well-conditioned."""
    rng = np.random.default_rng(1)
    A = rng.normal(size=(6, 6))
    U, _, Vt = np.linalg.svd(A)
    s = np.array([5.0, 4.5, 4.0, 3.5, 3.0, sigma_min], dtype=np.float64)
    return U @ np.diag(s) @ Vt


class TestDampedLeastSquaresConfig:
    def test_valid_config_passes_validate(self) -> None:
        DampedLeastSquaresConfig(lambda_max=0.05, sigma0=0.05).validate()

    @pytest.mark.parametrize("lambda_max", [0.0, -1.0, float("nan"), float("inf")])
    def test_invalid_lambda_max_rejected(self, lambda_max: float) -> None:
        with pytest.raises(ValueError, match="lambda_max"):
            DampedLeastSquaresConfig(lambda_max=lambda_max, sigma0=0.05).validate()

    @pytest.mark.parametrize("sigma0", [0.0, -1.0, float("nan"), float("inf")])
    def test_invalid_sigma0_rejected(self, sigma0: float) -> None:
        with pytest.raises(ValueError, match="sigma0"):
            DampedLeastSquaresConfig(lambda_max=0.05, sigma0=sigma0).validate()


class TestAwayFromSingularity:
    def test_reduces_to_ordinary_inverse_when_well_conditioned(self) -> None:
        J = _well_conditioned_jacobian()
        xd = np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        result = damped_least_squares_qd(J, xd, lambda_max=0.05, sigma0=0.05)

        assert isinstance(result, DampedLeastSquaresResult)
        assert result.lambda_used == pytest.approx(0.0, abs=1e-9)
        qd_expected = np.linalg.solve(J, xd)
        np.testing.assert_allclose(result.qd, qd_expected, atol=1e-8)
        assert result.qd_norm == pytest.approx(float(np.linalg.norm(qd_expected)), rel=1e-6)

    def test_sigma_min_matches_numpy_svd(self) -> None:
        J = _well_conditioned_jacobian()
        xd = np.array([0.0, 0.03, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        result = damped_least_squares_qd(J, xd, lambda_max=0.05, sigma0=0.05)
        expected_sigma_min = float(np.linalg.svd(J, compute_uv=False)[-1])
        assert result.sigma_min == pytest.approx(expected_sigma_min, rel=1e-9)


class TestNearSingularity:
    """The whole point of variable damping: boundedness, not tracking
    accuracy. Do NOT assert qd ~= J^-1 xd here."""

    def test_qd_stays_bounded_as_sigma_min_shrinks(self) -> None:
        xd = np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        lambda_max, sigma0 = 0.05, 0.05

        norms = []
        for sigma_min in (0.05, 0.01, 1e-3, 1e-6, 0.0):
            J = _near_singular_jacobian(sigma_min)
            result = damped_least_squares_qd(J, xd, lambda_max=lambda_max, sigma0=sigma0)
            assert np.all(np.isfinite(result.qd)), f"qd not finite at sigma_min={sigma_min}"
            norms.append(result.qd_norm)

        # The theoretical ceiling of the damped term along the singular
        # direction is |xd| / (2*lambda) as sigma_min -> 0 (the damped gain
        # sigma/(sigma^2+lambda^2) is maximized at sigma=lambda, value
        # 1/(2*lambda)) -- confirm boundedness against a generous multiple of
        # that, not against the undamped pseudoinverse (which diverges).
        theoretical_ceiling = float(np.linalg.norm(xd)) / (2.0 * lambda_max) * 5.0
        assert all(n <= theoretical_ceiling for n in norms), norms

    def test_undamped_pseudoinverse_would_diverge_but_dls_does_not(self) -> None:
        """Directly contrasts the plain pseudoinverse (which blows up) with
        DLS (which stays bounded) at the same near-zero sigma_min -- this is
        the property that matters for the real firmware-IK-protective-stop
        failure this module exists to fix."""
        xd = np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        J = _near_singular_jacobian(1e-8)

        plain_pinv_qd = np.linalg.pinv(J) @ xd
        result = damped_least_squares_qd(J, xd, lambda_max=0.05, sigma0=0.05)

        assert np.linalg.norm(plain_pinv_qd) > 1.0e4  # genuinely blows up
        assert result.qd_norm < 10.0  # DLS stays sane
        assert result.lambda_used > 0.0

    def test_lambda_used_ramps_up_monotonically_as_sigma_min_shrinks(self) -> None:
        xd = np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        sigma_mins = [0.05, 0.04, 0.03, 0.02, 0.01, 0.0]
        lambdas = []
        for sigma_min in sigma_mins:
            J = _near_singular_jacobian(sigma_min)
            result = damped_least_squares_qd(J, xd, lambda_max=0.05, sigma0=0.05)
            lambdas.append(result.lambda_used)
        # Non-decreasing as sigma_min shrinks (sigma_min=0.05 == sigma0 -> lambda
        # ~= 0; loose tolerance since reconstructing J via U@diag(s)@Vt and then
        # re-decomposing it can perturb the exact boundary singular value by a
        # few ULPs, e.g. 0.049999996 instead of exactly 0.05).
        assert lambdas[0] == pytest.approx(0.0, abs=1e-4)
        for a, b in zip(lambdas, lambdas[1:]):
            assert b >= a - 1e-12
        assert lambdas[-1] == pytest.approx(0.05, rel=1e-6)  # lambda_max at sigma_min=0


class TestInputValidation:
    def test_rejects_bad_lambda_max(self) -> None:
        J = _well_conditioned_jacobian()
        xd = np.zeros(6)
        with pytest.raises(ValueError, match="lambda_max"):
            damped_least_squares_qd(J, xd, lambda_max=0.0, sigma0=0.05)

    def test_rejects_bad_sigma0(self) -> None:
        J = _well_conditioned_jacobian()
        xd = np.zeros(6)
        with pytest.raises(ValueError, match="sigma0"):
            damped_least_squares_qd(J, xd, lambda_max=0.05, sigma0=-1.0)

    def test_rejects_non_2d_jacobian(self) -> None:
        with pytest.raises(ValueError, match="2D"):
            damped_least_squares_qd(np.zeros(6), np.zeros(6), lambda_max=0.05, sigma0=0.05)

    def test_rejects_shape_mismatch(self) -> None:
        J = _well_conditioned_jacobian()
        xd_wrong = np.zeros(3)
        with pytest.raises(ValueError, match="does not match"):
            damped_least_squares_qd(J, xd_wrong, lambda_max=0.05, sigma0=0.05)

    def test_rejects_nan_jacobian(self) -> None:
        J = _well_conditioned_jacobian()
        J[0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN/Inf"):
            damped_least_squares_qd(J, np.zeros(6), lambda_max=0.05, sigma0=0.05)

    def test_rejects_nan_xd(self) -> None:
        J = _well_conditioned_jacobian()
        xd = np.zeros(6)
        xd[0] = np.inf
        with pytest.raises(ValueError, match="NaN/Inf"):
            damped_least_squares_qd(J, xd, lambda_max=0.05, sigma0=0.05)


class TestArmQ0Pose:
    """This module was built specifically to handle ARM_Q0's documented
    wrist singularity (AGENTS.md: cond(J)=1395.76, sigma_min=1.485e-3). Uses
    a real UR5e Jacobian if mujoco is importable; otherwise skips (no
    mujoco dependency for the rest of this test module)."""

    def test_bounded_qd_at_arm_q0_singularity(self) -> None:
        mujoco = pytest.importorskip("mujoco")
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[2]
        scene_xml = repo_root / "assets" / "ur5e_torque" / "scene.xml"
        if not scene_xml.exists():
            pytest.skip("assets/ur5e_torque/scene.xml not present in this checkout")

        model = mujoco.MjModel.from_xml_path(str(scene_xml))
        data = mujoco.MjData(model)
        site_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site"))
        data.qpos[:6] = ARM_Q0_SINGULAR
        mujoco.mj_forward(model, data)
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        J = np.vstack([jacp[:, :6], jacr[:, :6]])

        # A -X Cartesian velocity command -- the direction real-hardware
        # testing found the firmware's own speedL IK protective-stops on
        # (see hardware/joint_velocity_transport.py's module docstring).
        xd_minus_x = np.array([-0.02, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        result = damped_least_squares_qd(J, xd_minus_x, lambda_max=0.05, sigma0=0.05)

        assert np.all(np.isfinite(result.qd))
        assert result.qd_norm < 50.0  # bounded, not exploding through the singularity
        assert result.sigma_min < 0.05  # confirms this pose IS near-singular for DLS's threshold
