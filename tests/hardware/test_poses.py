"""Tests for hardware.poses."""

from __future__ import annotations

import numpy as np

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
