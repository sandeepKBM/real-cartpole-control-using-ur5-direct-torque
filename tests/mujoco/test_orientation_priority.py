"""MuJoCo-backed tests for CartesianVelocityConfig.orientation_priority --
the parts that need a REAL UR5e Jacobian to mean anything.

The pure-numpy side (the smooth_falloff ramp shape, the off/zero-weight
bit-for-bit exactness guarantees, path-independence, the all-axes-selected
no-op) is covered simulator-free in
tests/unit/test_cartesian_velocity_controller.py. What can only be tested
here is the behaviour the mechanism was actually built for, which depends on
a genuine kinematic coupling no toy Jacobian reproduces honestly:
``hanging_alpha_0_5``'s X-translation dragging rx/ry along through the TASK
(row) space rather than the null space.

Added 2026-08-06 per AGENTS.md sec 5. Deliberately small: single-pose,
single-displacement checks of specific measured properties, not a grid sweep
(the 128-cell before/after grid lives in tools/evaluate_orientation_priority.py
and belongs on ilab, not in a test suite).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from controller_core.cartesian_velocity_controller import (  # noqa: E402
    CartesianVelocityConfig,
    CartesianVelocityController,
)
from controller_core.kinematics_utils import swing_twist_axis_error  # noqa: E402
from hardware.local_dynamics import LocalMujocoDynamics  # noqa: E402
from velocity_gain_tuning.envs.velocity_transport_env import (  # noqa: E402
    VelocityTransportEnv,
    VelocityTransportEnvConfig,
    action_to_gains,
)
from velocity_gain_tuning.optimize import run_episode  # noqa: E402
from velocity_gain_tuning.poses import scenario_by_name  # noqa: E402

pytestmark = pytest.mark.mujoco

# search_result_nullspace_v2_20260806_194402.json -- this lane's reproducible
# fixed-gain best (104/128). Every assertion below is at THESE gains, so the
# comparison is genuinely "same gains, mechanism toggled."
NULLSPACE_V2_ACTION = np.array(
    [
        -0.5452930656195676,
        -0.31201103390079576,
        0.19603435480606923,
        -0.40319481871903273,
        0.6634521666673519,
        -0.29877165734428546,
    ]
)


@pytest.fixture(scope="module")
def dyn():
    return LocalMujocoDynamics()


def _cfg(dyn_gains: dict, **overrides) -> CartesianVelocityConfig:
    base = dict(
        reduced_task_dims=False,
        split_base_wrist_task=False,
        ik_seeded_resolution=True,
        ik_iterations=6,
        task_dim_rx=False,
        task_dim_ry=False,
        task_dim_rz=True,
        kp_x=dyn_gains["kp_x"],
        kp_y=dyn_gains["kp_x"],
        kp_z=dyn_gains["kp_x"],
        kp_rot=dyn_gains["kp_rot"],
        ik_joint_gain=dyn_gains["ik_joint_gain"],
        pinv_damping=dyn_gains["pinv_damping"],
        qp_task_weight=dyn_gains["qp_task_weight"],
        ik_max_joint_deviation_rad=dyn_gains["ik_max_joint_deviation_rad"],
        # Deliberately far above anything these checks command: _q_target
        # recovers q_target by inverting the controller's final joint-space P
        # law, which is only exact if the shared speed clamp did not bite.
        # Leaving the real 0.25 m/s ceiling in place here silently truncates
        # xd_cmd and makes the recovered "q_target" a clamped fiction (found
        # the hard way -- it reported a 0.18 m position residual for a solve
        # that is actually exact). The end-to-end episode test below uses the
        # real, unmodified clamps.
        max_lin_speed_mps=1.0e6,
        max_ang_speed_radps=1.0e6,
    )
    base.update(overrides)
    return CartesianVelocityConfig(**base)


def _q_target(dyn, cfg, dx: float) -> np.ndarray:
    """Recover compute_ik_seeded's q_target for a target dx, by inverting the
    joint-space P law it ends with. Evaluated at q_current == q_rest so the
    Jacobian inversion is exact and unambiguous."""
    q0 = scenario_by_name("hanging_alpha_0_5").q0.copy()
    p0, quat0, jac0 = dyn.fk_and_jacobian(q0)
    target = p0.copy()
    target[0] += dx
    ctrl = CartesianVelocityController(cfg)
    ctrl.reset_from_state(
        {"time": 0.0, "q": q0, "qd": np.zeros(6), "ee_pos": p0, "ee_quat": quat0, "target_x": float(p0[0])}
    )
    xd = ctrl.compute(
        {
            "time": 0.0,
            "q": q0,
            "qd": np.zeros(6),
            "ee_pos": p0,
            "ee_quat": quat0,
            "target_x": float(target[0]),
            "target_ee_pos": target,
            "target_ee_vel": np.zeros(3),
            "fk_jacobian_fn": dyn.fk_and_jacobian,
        }
    )
    return q0 + (np.linalg.pinv(jac0) @ xd) / cfg.ik_joint_gain


def _q_target_quality(dyn, cfg, dx: float) -> tuple[float, float]:
    """(position residual m, swing-twist orientation error rad) of q_target."""
    q0 = scenario_by_name("hanging_alpha_0_5").q0.copy()
    p0, quat0, _ = dyn.fk_and_jacobian(q0)
    target = p0.copy()
    target[0] += dx
    q_t = _q_target(dyn, cfg, dx)
    p_t, quat_t, _ = dyn.fk_and_jacobian(q_t)
    ori = float(np.linalg.norm([swing_twist_axis_error(quat0, quat_t, i) for i in range(3)]))
    return float(np.linalg.norm(p_t - target)), ori


def test_off_is_bit_for_bit_unchanged_on_the_real_pose(dyn):
    """The no-regression guarantee, checked against the real Jacobian rather
    than a toy one: with the flag off, q_target must be IDENTICAL, not close.
    A silent drift here would invalidate every historical result in
    outputs/velocity_gain_tuning/."""
    gains = action_to_gains(NULLSPACE_V2_ACTION)
    for dx in (0.185, -0.185, 0.2405, -0.296):
        a = _q_target(dyn, _cfg(gains), dx)
        b = _q_target(
            dyn,
            _cfg(
                gains,
                orientation_priority=False,
                orientation_priority_weight=5.0,
                orientation_priority_residual_tol_m=1.0,
                orientation_priority_residual_falloff_m=2.0,
            ),
            dx,
        )
        np.testing.assert_array_equal(a, b)


def test_promotion_cuts_orientation_error_where_the_pose_is_reachable(dyn):
    """The mechanism's purpose. At hanging_alpha_0_5 the full 6-DOF pose
    (x0+dx, y0, z0, R0) is genuinely reachable for these displacements, so
    promoting rx/ry costs no position accuracy and must strictly reduce
    q_target's orientation error."""
    gains = action_to_gains(NULLSPACE_V2_ACTION)
    on = _cfg(gains, orientation_priority=True, orientation_priority_weight=1.0)
    off = _cfg(gains)
    for dx in (0.185, 0.2405, -0.185):
        pos_off, ori_off = _q_target_quality(dyn, off, dx)
        pos_on, ori_on = _q_target_quality(dyn, on, dx)
        assert pos_on < 1.0e-3, f"dx={dx}: promotion must not cost position ({pos_on:.5f} m)"
        assert ori_on < ori_off, f"dx={dx}: ori {ori_off:.4f} -> {ori_on:.4f}"
        assert ori_on < 1.0e-3, f"dx={dx}: reachable pose should hold orientation ({ori_on:.4f} rad)"


def test_recovers_the_known_failing_fast_move_cell_without_breaking_neighbours(dyn):
    """End-to-end on the real evaluation environment and the unmodified
    guards: hanging_alpha_0_5 at dx=+0.2035 m / 0.02 s move is a recorded
    orientation_guard failure for this exact gain vector (see
    outputs/velocity_gain_tuning/search_result_nullspace_v2_20260806_194402.json).
    It must pass with the mechanism on -- and the two neighbouring cells that
    already pass must keep passing."""
    scenario = scenario_by_name("hanging_alpha_0_5")
    off_env = VelocityTransportEnv(VelocityTransportEnvConfig(orientation_priority=False), seed=0)
    on_env = VelocityTransportEnv(
        VelocityTransportEnvConfig(orientation_priority=True, orientation_priority_weight=1.0), seed=0
    )

    failing = dict(target_x_delta_m=0.2035, move_duration_s=0.02)
    r_off = run_episode(off_env, NULLSPACE_V2_ACTION, scenario=scenario, **failing)
    r_on = run_episode(on_env, NULLSPACE_V2_ACTION, scenario=scenario, **failing)
    assert r_off.guard_reason is not None and "orientation_guard" in r_off.guard_reason
    assert r_on.guard_reason is None, f"still failing: {r_on.guard_reason}"
    assert abs(r_on.achieved_x_delta_m - 0.2035) < 0.005, "recovered WITHOUT giving up X-tracking"

    for passing in (dict(target_x_delta_m=0.185, move_duration_s=1.0),
                    dict(target_x_delta_m=-0.185, move_duration_s=1.0)):
        assert run_episode(off_env, NULLSPACE_V2_ACTION, scenario=scenario, **passing).guard_reason is None
        assert run_episode(on_env, NULLSPACE_V2_ACTION, scenario=scenario, **passing).guard_reason is None


def test_falls_back_where_the_pose_is_out_of_reach(dyn):
    """The honest scope limit, pinned as a test so it cannot be quietly
    forgotten: at dx=-0.370 m the full 6-DOF pose is genuinely unreachable at
    this pose (a multi-start IK check measured a ~0.047 m position plus
    ~0.32 rad rotation residual floor, and even dropping orientation entirely
    still leaves ~0.034 m unreachable). The mechanism must therefore NOT claim
    to hold orientation there -- the promoted solve's residual is real, and
    what matters is that the fallback keeps position tracking from degrading
    below the position-only solve's."""
    gains = action_to_gains(NULLSPACE_V2_ACTION)
    pos_off, _ = _q_target_quality(dyn, _cfg(gains), -0.370)
    pos_on, _ = _q_target_quality(
        dyn, _cfg(gains, orientation_priority=True, orientation_priority_weight=1.0), -0.370
    )
    assert pos_off > 0.01, "this displacement is supposed to be out of reach"
    assert pos_on <= pos_off + 1.0e-9, f"fallback must not worsen position ({pos_off:.4f} -> {pos_on:.4f} m)"


def test_env_config_defaults_keep_the_mechanism_off(dyn):
    """Every historical result in outputs/velocity_gain_tuning/ was produced
    with this env, so its default must remain the pre-mechanism behaviour."""
    cfg = VelocityTransportEnvConfig()
    assert cfg.orientation_priority is False
    scenario = scenario_by_name("hanging_alpha_0_5")
    env = VelocityTransportEnv(cfg, seed=0)
    env.reset(seed=0, options={"scenario": scenario, "target_x_delta_m": 0.05})
    assert env._controller is not None
    assert env._controller.cfg.orientation_priority is False
