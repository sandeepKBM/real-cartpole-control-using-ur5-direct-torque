"""The Cartesian control modes (position/servoL, velocity/speedL) must be
REFUSED at a start pose on/near the UR wrist singularity (2026-08-26): those
modes command through PolyScope's Jacobian IK, which faults at wrist_2=0, while
direct_torque/urscript command joint torques and are fine. The guard fails fast
on the host, before any robot connection, with a message pointing at
direct_torque -- so these tests need no robot.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hardware.x_transport import (
    WRIST_SINGULARITY_GUARD_RAD,
    _check_wrist_singularity_for_cartesian_mode,
    run_x_transport,
)

# ARM_Q0 as committed: wrist_2 = 0.004714693 rad (~0.27 deg) -- the singular pose.
_SINGULAR_Q = np.array([-2.3688, -2.1801, -1.8838, -0.7962, 0.004714693, 0.0206])
# Goal-1 pose: same base, wrist_2 = -90 deg -- well-conditioned (cond(J)~7.2).
_NONSINGULAR_Q = np.array([-2.3688, -2.1801, -1.8838, -0.7962, -1.5707963, 0.0206])


@pytest.mark.parametrize("mode", ["position", "velocity"])
def test_cartesian_mode_refused_at_wrist_singularity(mode):
    with pytest.raises(ValueError, match="wrist singularity|direct_torque"):
        _check_wrist_singularity_for_cartesian_mode(mode, _SINGULAR_Q)


@pytest.mark.parametrize("mode", ["direct_torque", "urscript"])
def test_torque_modes_allowed_at_wrist_singularity(mode):
    # Joint-torque modes bypass the IK layer -- the guard must NOT fire.
    _check_wrist_singularity_for_cartesian_mode(mode, _SINGULAR_Q)


@pytest.mark.parametrize("mode", ["position", "velocity"])
def test_cartesian_mode_allowed_at_wellconditioned_pose(mode):
    _check_wrist_singularity_for_cartesian_mode(mode, _NONSINGULAR_Q)


def test_guard_noop_without_start_pose():
    # No target pose specified -> cannot preflight; guard must not fire.
    _check_wrist_singularity_for_cartesian_mode("position", None)


def test_guard_fires_near_pi_too():
    q = _SINGULAR_Q.copy()
    q[4] = np.pi - 0.01  # wrist_2 near +180 deg is also the wrist singularity
    with pytest.raises(ValueError, match="singularity"):
        _check_wrist_singularity_for_cartesian_mode("position", q)


def test_boundary_just_inside_and_outside():
    q = _SINGULAR_Q.copy()
    q[4] = WRIST_SINGULARITY_GUARD_RAD - 1e-4  # just inside -> refuse
    with pytest.raises(ValueError):
        _check_wrist_singularity_for_cartesian_mode("velocity", q)
    q[4] = WRIST_SINGULARITY_GUARD_RAD + 1e-2  # just outside -> allow
    _check_wrist_singularity_for_cartesian_mode("velocity", q)


def test_run_x_transport_refuses_before_connecting():
    """Integration: the dispatcher raises the guard before any RTDE connection,
    so a bogus robot_ip/config never gets touched."""
    with pytest.raises(ValueError, match="direct_torque"):
        run_x_transport(
            control_mode="position",
            robot_ip="0.0.0.0",  # never connected -- guard fires first
            config_path=Path("/nonexistent/config.yaml"),
            target_x_delta_m=0.02,
            move_duration_s=1.0,
            duration_s=2.0,
            output_dir=None,
            motion_opt_in=False,
            start_q_rad=_SINGULAR_Q,
        )
