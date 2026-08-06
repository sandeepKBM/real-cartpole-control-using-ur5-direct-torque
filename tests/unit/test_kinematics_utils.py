"""Tests for controller_core/kinematics_utils.py's null_space_basis --
the SVD-based null-space basis used by compute_ik_seeded's
ik_max_joint_deviation_rad to hard-bound redundant joint motion without
blocking task-necessary motion."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from controller_core.kinematics_utils import null_space_basis  # noqa: E402


def test_basis_columns_are_orthonormal():
    rng = np.random.default_rng(0)
    jac = rng.standard_normal((4, 6))
    basis = null_space_basis(jac)
    assert basis.shape == (6, 2)  # 4x6 full-row-rank -> 2-dim null space
    np.testing.assert_allclose(basis.T @ basis, np.eye(basis.shape[1]), atol=1e-10)


def test_basis_spans_the_actual_null_space():
    rng = np.random.default_rng(1)
    jac = rng.standard_normal((4, 6))
    basis = null_space_basis(jac)
    # Every basis column, mapped through jacobian, must vanish.
    np.testing.assert_allclose(jac @ basis, np.zeros((4, basis.shape[1])), atol=1e-10)


def test_full_column_rank_has_empty_null_space():
    rng = np.random.default_rng(2)
    jac = rng.standard_normal((6, 6))  # square, generically full rank
    basis = null_space_basis(jac)
    assert basis.shape == (6, 0)


def test_zero_row_rank_gives_full_null_space():
    jac = np.zeros((3, 6))
    basis = null_space_basis(jac)
    assert basis.shape == (6, 6)
    np.testing.assert_allclose(basis.T @ basis, np.eye(6), atol=1e-10)


def test_near_singular_row_widens_the_detected_null_space():
    """A jacobian row that's nearly (but not exactly) linearly dependent on
    another must still be treated as part of the null space -- otherwise a
    tiny, physically-meaningless singular value would be mistaken for a
    genuine independent task direction, letting a supposedly-bounded
    "redundant" step actually leak into what's really a near-singular task
    row."""
    jac = np.eye(3, 6)
    jac[2, :] = jac[1, :] * (1.0 + 1e-12)  # numerically indistinguishable from rank-2
    basis = null_space_basis(jac, rank_tol=1e-9)
    assert basis.shape == (6, 4)  # rank ~2, not 3


def test_task_image_of_null_space_combination_is_exactly_zero_for_real_ur5e_style_jacobian():
    """A more realistic (non-square, coupled) 4x6 task jacobian -- confirms
    the exactness property compute_ik_seeded's null-space clipping relies
    on: ANY linear combination of basis columns maps to (numerically) zero
    under the task jacobian, not just approximately."""
    rng = np.random.default_rng(3)
    jac = np.eye(4, 6) + 0.3 * rng.standard_normal((4, 6))
    basis = null_space_basis(jac)
    coeffs = rng.standard_normal(basis.shape[1])
    step = basis @ coeffs
    np.testing.assert_allclose(jac @ step, np.zeros(4), atol=1e-9)


def test_rejects_non_2d_input():
    with pytest.raises(ValueError):
        null_space_basis(np.zeros(6))
