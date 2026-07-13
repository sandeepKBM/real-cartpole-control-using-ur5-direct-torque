"""Tests for hardware direct-torque helpers."""

from __future__ import annotations

import numpy as np

from controller_core.kinematics_utils import rotvec_to_quat_wxyz


def test_rotvec_zero_is_identity_quat() -> None:
    q = rotvec_to_quat_wxyz(np.zeros(3))
    np.testing.assert_allclose(q, [1.0, 0.0, 0.0, 0.0], atol=1e-12)
