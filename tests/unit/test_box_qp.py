"""Tests for controller_core/box_qp.py -- the box-constrained QP solver and
build_weighted_least_squares_qp, the shared interface for building a
Tikhonov-regularized multi-term weighted-least-squares QP objective."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from controller_core.box_qp import build_weighted_least_squares_qp, solve_box_qp  # noqa: E402


def test_single_term_matches_ordinary_weighted_normal_equations():
    # A must have full column rank (rows >= cols) for A'A to be invertible
    # and the reference weighted-normal-equations solution to be unique --
    # an underdetermined A (fewer rows than columns) has a singular A'A
    # and infinitely many solutions, not a well-posed comparison.
    rng = np.random.default_rng(0)
    a = rng.standard_normal((8, 6))
    b = rng.standard_normal(8)
    hessian, linear = build_weighted_least_squares_qp([(a, b, 2.5)])
    x = -np.linalg.solve(hessian, linear)
    # Ordinary weighted normal equations: (A'A) x = A'b (weight cancels).
    x_expected = np.linalg.solve(a.T @ a, a.T @ b)
    np.testing.assert_allclose(x, x_expected, atol=1e-8)


def test_reg_only_pulls_solution_toward_zero():
    a = np.eye(6)
    b = np.full(6, 0.05)
    hessian_noreg, linear_noreg = build_weighted_least_squares_qp([(a, b, 1.0)], reg=0.0)
    hessian_reg, linear_reg = build_weighted_least_squares_qp([(a, b, 1.0)], reg=10.0)
    x_noreg = -np.linalg.solve(hessian_noreg, linear_noreg)
    x_reg = -np.linalg.solve(hessian_reg, linear_reg)
    np.testing.assert_allclose(x_noreg, b, atol=1e-10)  # A=I, so unregularized solution is exactly b
    assert np.linalg.norm(x_reg) < np.linalg.norm(x_noreg)  # regularization shrinks it toward 0


def test_two_terms_trade_off_by_relative_weight():
    """A task term pulling toward b_task and a posture term pulling toward
    b_posture, both A=I -- the combined solution must land strictly
    between the two targets, closer to whichever has more weight."""
    a = np.eye(3)
    b_task = np.array([1.0, 0.0, 0.0])
    b_posture = np.array([0.0, 0.0, 0.0])

    hessian_eq, linear_eq = build_weighted_least_squares_qp([(a, b_task, 1.0), (a, b_posture, 1.0)])
    x_eq = -np.linalg.solve(hessian_eq, linear_eq)
    np.testing.assert_allclose(x_eq, [0.5, 0.0, 0.0], atol=1e-10)  # equal weights -> exact midpoint

    hessian_task_heavy, linear_task_heavy = build_weighted_least_squares_qp(
        [(a, b_task, 100.0), (a, b_posture, 1.0)]
    )
    x_task_heavy = -np.linalg.solve(hessian_task_heavy, linear_task_heavy)
    assert x_task_heavy[0] > x_eq[0]  # heavier task weight pulls closer to b_task
    assert x_task_heavy[0] < 1.0  # but posture term still has SOME pull, not fully ignored


def test_high_task_weight_relative_to_posture_approaches_exact_task_solution():
    """The regime this repo's real gain searches converge toward (task_w
    >> posture_w by 2+ orders of magnitude) -- confirms the posture term's
    perturbation of the task solution becomes negligible, not that it's
    exactly zero (that exactness was the OLD nullspace-projected
    mechanism's guarantee; this integrated formulation trades an exact
    guarantee for better numerical conditioning, approaching exactness
    only in the task_w >> posture_w limit)."""
    a_task = np.array([[1.0, 0.15, 0.0]])
    b_task = np.array([0.05])
    a_posture = np.eye(3)
    b_posture = np.zeros(3)
    hessian, linear = build_weighted_least_squares_qp(
        [(a_task, b_task, 1.0e6), (a_posture, b_posture, 1.0)], reg=1e-6
    )
    x = -np.linalg.solve(hessian, linear)
    task_residual = float((a_task @ x - b_task)[0])
    assert abs(task_residual) < 1e-4


def test_build_weighted_least_squares_qp_rejects_empty_terms():
    with pytest.raises(ValueError):
        build_weighted_least_squares_qp([])


def test_output_feeds_directly_into_solve_box_qp():
    """The two functions are meant to compose directly: build the
    objective, then solve it subject to bounds."""
    a = np.eye(3)
    b = np.array([5.0, -5.0, 0.2])
    hessian, linear = build_weighted_least_squares_qp([(a, b, 1.0)])
    x = solve_box_qp(hessian, linear, np.full(3, -1.0), np.full(3, 1.0))
    np.testing.assert_allclose(x, [1.0, -1.0, 0.2], atol=1e-6)  # clipped to the box, unclipped dim exact


def test_damped_pinv_style_single_axis_regularization_matches_known_closed_form():
    # 1D sanity check against the textbook ridge-regression closed form:
    # x = a*b / (a^2 + reg), for a single scalar A=[a], b=[b].
    a = np.array([[3.0]])
    b = np.array([2.0])
    hessian, linear = build_weighted_least_squares_qp([(a, b, 1.0)], reg=0.5)
    x = -np.linalg.solve(hessian, linear)
    expected = (3.0 * 2.0) / (3.0**2 + 0.5)
    assert x[0] == pytest.approx(expected, rel=1e-9)
