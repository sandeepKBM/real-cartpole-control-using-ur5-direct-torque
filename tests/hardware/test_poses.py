"""Tests for hardware.poses."""

from __future__ import annotations

import numpy as np
import pytest

from hardware.poses import HEIGHT_ALPHA_0_5_Q, q_for_height_alpha


def test_height_alpha_0_5_matches_fixed_gain_test() -> None:
    # Shell script uses rounded literals; pose module uses exact pi/2 interpolation.
    expected = np.array([0.0, -0.835098168, -1.2, -0.985398163, 0.0, 0.0], dtype=np.float64)
    np.testing.assert_allclose(HEIGHT_ALPHA_0_5_Q, expected, rtol=0, atol=5e-4)
    np.testing.assert_allclose(q_for_height_alpha(0.5), HEIGHT_ALPHA_0_5_Q)


def test_height_alpha_endpoints() -> None:
    from hardware.poses import ACTIVE_ORIGIN_Q, LOWER_B_Q

    np.testing.assert_allclose(q_for_height_alpha(0.0), ACTIVE_ORIGIN_Q)
    np.testing.assert_allclose(q_for_height_alpha(1.0), LOWER_B_Q)


# --- "Hanging"/elbow-down pose family (added 2026-08-01) ---------------------------


def test_hanging_pose_family_does_not_mutate_existing_constants() -> None:
    """The new family must be purely additive -- old constants unchanged in value."""
    from hardware.poses import ACTIVE_ORIGIN_Q, LOWER_B_Q

    np.testing.assert_allclose(ACTIVE_ORIGIN_Q, [0.0, -1.5707963267948966, 0.0, -1.5707963267948966, 0.0, 0.0])
    np.testing.assert_allclose(LOWER_B_Q, [0.0, -0.1, -2.4, -0.4, 0.0, 0.0])


def test_hanging_height_alpha_endpoints() -> None:
    from hardware.poses import HANGING_LOWER_Q, HANGING_ORIGIN_Q, q_for_hanging_height_alpha

    np.testing.assert_allclose(q_for_hanging_height_alpha(0.0), HANGING_ORIGIN_Q)
    np.testing.assert_allclose(q_for_hanging_height_alpha(1.0), HANGING_LOWER_Q)


def test_hanging_alpha_0_5_matches_midpoint() -> None:
    from hardware.poses import HANGING_ALPHA_0_5_Q, HANGING_LOWER_Q, HANGING_ORIGIN_Q, q_for_hanging_height_alpha

    np.testing.assert_allclose(HANGING_ALPHA_0_5_Q, 0.5 * HANGING_ORIGIN_Q + 0.5 * HANGING_LOWER_Q)
    np.testing.assert_allclose(q_for_hanging_height_alpha(0.5), HANGING_ALPHA_0_5_Q)


def test_hanging_family_avoids_wrist_2_zero() -> None:
    """The whole point of this family: wrist_2 must never be 0 across its range."""
    from hardware.poses import q_for_hanging_height_alpha

    for alpha in np.linspace(0.0, 1.0, 21):
        q = q_for_hanging_height_alpha(float(alpha))
        assert abs(q[4] - 1.5707963267948966) < 1e-9, f"wrist_2 drifted from +pi/2 at alpha={alpha}"


def test_hanging_height_alpha_rejects_out_of_range() -> None:
    from hardware.poses import q_for_hanging_height_alpha

    with pytest.raises(ValueError):
        q_for_hanging_height_alpha(-0.01)
    with pytest.raises(ValueError):
        q_for_hanging_height_alpha(1.01)
