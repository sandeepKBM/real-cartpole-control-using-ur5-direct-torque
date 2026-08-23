"""Closed-loop MuJoCo validation of `split_base_wrist_task_dims` on the REAL UR5e.

``tests/unit/test_split_base_wrist_task_dims.py`` proves the linear algebra of
the combined row+column restriction, but its 1x3 algebra tier uses a synthetic
Jacobian on purpose, and nothing there can see the arm actually move: the
Jacobian changing under the controller, real joint friction
(``assets/ur5e_torque/ur5e_torque.xml``'s ``frictionloss``/``damping``),
gravity, the torque backtracking/clip stage, or the safety guards.

This file drives the SAME adapter pipeline every sim tool in this repo uses
(``simulation/ur5e_mujoco_torque.py``'s ``build_initial_state_and_adapter`` /
``build_mujoco_state`` / ``adapter.step()`` / ``mujoco.mj_step`` on
``assets/ur5e_torque/scene.xml``) at the ACTUAL real-robot start pose this
feature was built for, in BOTH directions (AGENTS.md sec 7). The per-step loop
lives in ``tools/diagnostics/split_base_wrist_task_dims_sim_check.py`` (also
runnable standalone); this file is the reproducible assertion layer over it.

What this locks down
--------------------
1. The kinematic premise, measured from the model rather than quoted: at the
   real pose the ``{shoulder_lift, elbow, wrist_1}`` position sub-Jacobian is
   rank 2 of 3 (so the pre-existing 3-row split task is structurally
   unusable there), while its world-X row alone is perfectly well-posed.
2. The real case converges, both directions, no guard trip -- and beats the
   3-row task it replaces by a wide margin.
3. World Y and Z, the two task rows this config drops, are genuinely
   POSTURE-HELD and not ignored: with the posture spring removed the very same
   run walks Z past the drift guard.
4. No task torque can structurally reach a held joint (shoulder_pan, wrist_2,
   wrist_3) -- exactly zero, every cycle.
5. The validated envelope boundary, asserted so it cannot be quietly assumed
   wider than it was measured.

``gravity_source`` is forced to ``mujoco_qfrc`` and Coriolis feedforward off by
the harness default: the config selects ``pinocchio``, an optional dependency,
and MuJoCo's own ``qfrc_bias`` is parity-checked against it to <1e-8 Nm
(AGENTS.md sec 3).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_check_module():
    path = REPO_ROOT / "tools" / "diagnostics" / "split_base_wrist_task_dims_sim_check.py"
    spec = importlib.util.spec_from_file_location("split_base_wrist_task_dims_sim_check", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec: the module defines @dataclass types, and dataclasses
    # resolves annotations via sys.modules[cls.__module__].
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CHECK = _load_check_module()

MOVE_DURATION_S = 1.5
HOLD_DURATION_S = 2.0
#: The largest displacement validated to pass in BOTH directions at this pose.
#: 0.03 m already trips the Z-drift guard (test_envelope_boundary below).
VALIDATED_DX_M = 0.02

_ROLLOUT_CACHE: dict[tuple, object] = {}


def rollout(variant: str, delta_m: float, *, hold_s: float = HOLD_DURATION_S):
    """Memoized -- each closed-loop rollout costs seconds and several tests
    compare the same pairs against each other."""
    key = (variant, float(delta_m), float(hold_s))
    if key not in _ROLLOUT_CACHE:
        _ROLLOUT_CACHE[key] = CHECK.run_variant(
            variant, CHECK.USER_REAL_POSE_Q,
            target_delta_m=float(delta_m),
            move_duration_s=MOVE_DURATION_S,
            hold_duration_s=float(hold_s),
            pose_label="user_real_pose",
        )
    return _ROLLOUT_CACHE[key]


# --------------------------------------------------------------------------- #
# 1. The kinematic premise, measured from the model.
# --------------------------------------------------------------------------- #
def test_three_row_task_is_structurally_singular_but_the_x_row_is_not():
    q = CHECK.USER_REAL_POSE_Q
    three_row = CHECK.analyze_block(q, (0, 1, 2), CHECK.LIFT_ELBOW_WRIST1)
    x_row = CHECK.analyze_block(q, CHECK.X_ONLY, CHECK.LIFT_ELBOW_WRIST1)

    # Three mutually parallel axes: at most a 2D subspace of 3D linear velocity.
    assert three_row.rank == 2
    assert three_row.cond_or_norm > 1e12
    assert min(three_row.singular_values) < 1e-12
    assert three_row.usable is False

    # World X lies inside that 2D subspace, comfortably.
    assert x_row.rank == 1
    assert x_row.usable is True
    assert x_row.cond_or_norm == pytest.approx(0.2353, abs=5e-4)
    # ...and carries essentially the same world-X authority as (2, 3, 4)
    # = elbow/wrist_1/wrist_2, the full-rank pan-free set
    # config/ur5e_mujoco_torque_osc_tuned_split_base_wrist_pan_fixed.yaml
    # already uses. That equivalence is the whole argument for the row
    # reduction: the ONLY thing (1, 2, 3) lacks is rank this task never needed.
    pan_free_x_row = CHECK.analyze_block(q, CHECK.X_ONLY, (2, 3, 4))
    assert x_row.cond_or_norm == pytest.approx(pan_free_x_row.cond_or_norm, rel=0.01)
    assert CHECK.analyze_block(q, (0, 1, 2), (2, 3, 4)).rank == 3


def test_rank_deficiency_is_structural_not_pose_specific():
    """No pose fixes three parallel axes -- checked at every pose this repo
    keeps, so a future pose change cannot silently make the premise stale."""
    for name, q in CHECK.POSES.items():
        block = CHECK.analyze_block(q, (0, 1, 2), CHECK.LIFT_ELBOW_WRIST1, pose_label=name)
        assert block.rank == 2, f"{name}: expected rank 2, got {block.rank}"


# --------------------------------------------------------------------------- #
# 2. The real case converges, both directions.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("delta_m", [VALIDATED_DX_M, -VALIDATED_DX_M])
def test_x_only_task_converges_at_the_real_pose(delta_m: float):
    r = rollout("x_only_1x3", delta_m)
    assert not r.guard_tripped, f"guard {r.guard_reason} at t={r.guard_time_s}"
    assert r.task_dims_used == (0,)
    assert r.active_joints_used == CHECK.LIFT_ELBOW_WRIST1
    assert np.sign(r.achieved_delta_m) == np.sign(delta_m)
    assert r.tracking_fraction > 0.8, f"tracked only {100 * r.tracking_fraction:.2f}%"
    # Every signal the guards watch, checked explicitly against the config's
    # own safety block rather than assumed.
    assert r.max_abs_y_drift_m < 0.03
    assert r.max_abs_z_drift_m < 0.03
    assert r.max_orientation_error_rad < 0.25
    assert r.max_abs_qd_radps < 3.0
    assert np.isfinite(r.max_abs_tau_nm)
    assert r.steps > 100, "rollout terminated far too early"


@pytest.mark.parametrize("delta_m", [VALIDATED_DX_M, -VALIDATED_DX_M])
def test_x_only_beats_the_three_row_task_it_replaces(delta_m: float):
    """The 3-row task over these columns does not fail loudly -- it just never
    gets there, because the task it is being asked to solve is rank-deficient.
    That is exactly the failure row selection exists to remove."""
    reduced = rollout("x_only_1x3", delta_m)
    three_row = rollout("three_row_3x3", delta_m)
    assert three_row.task_dims_used == (0, 1, 2)
    assert three_row.tracking_fraction < 0.75
    assert reduced.tracking_fraction > three_row.tracking_fraction + 0.15


@pytest.mark.parametrize("delta_m", [VALIDATED_DX_M, -VALIDATED_DX_M])
def test_long_hold_settles_rather_than_creeping(delta_m: float):
    short = rollout("x_only_1x3", delta_m, hold_s=2.0)
    long_hold = rollout("x_only_1x3", delta_m, hold_s=10.0)
    assert not long_hold.guard_tripped
    # Tracking improves or holds with more settle time (no creep away), and the
    # held axes do not keep walking either.
    assert long_hold.tracking_fraction >= short.tracking_fraction - 1e-6
    assert long_hold.max_abs_y_drift_m < 0.03
    assert long_hold.max_abs_z_drift_m < 0.03
    assert long_hold.max_abs_qd_radps < 3.0


# --------------------------------------------------------------------------- #
# 3. The dropped task rows (Y, Z) are posture-held, not ignored.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("delta_m", [VALIDATED_DX_M, -VALIDATED_DX_M])
def test_the_posture_spring_is_what_holds_the_unactuated_axes_and_joints(delta_m: float):
    """Direct evidence, not an argument: the ONLY thing changed between these
    two runs is kp_posture/kd_posture.

    The unambiguous, both-directions signal is in JOINT space, where the
    posture spring acts: the three held joints (shoulder_pan, wrist_2,
    wrist_3, which by construction receive exactly zero task torque) move
    roughly 3x further with the spring removed. In CARTESIAN space the
    picture is more nuanced and is recorded honestly here rather than
    overclaimed: at ``dx = -0.02`` the spring is decisive (Z peaks at 0.0262 m
    with it, and walks past the 0.03 m guard at t~6.4 s without it), while at
    ``dx = +0.02`` the two are within 1% of each other (0.0261 vs 0.0259 m) --
    at that displacement most of the Z excursion is kinematic coupling during
    the move, which the posture spring is not the mechanism that resists.
    """
    held = rollout("x_only_1x3", delta_m, hold_s=6.0)
    unheld = rollout("x_only_no_posture", delta_m, hold_s=6.0)
    assert not held.guard_tripped

    # Joint space: the held joints really are held, in both directions.
    assert held.max_held_joint_excursion_rad < 0.5 * unheld.max_held_joint_excursion_rad

    if delta_m < 0:
        # Cartesian space, the direction where it is decisive: without the
        # spring the SAME move creeps past the Z drift guard during the hold.
        assert unheld.guard_tripped and "Z" in unheld.guard_reason
        assert held.max_abs_z_drift_m < 0.03
    else:
        # ...and the direction where it is roughly neutral, asserted loosely so
        # a regression that made it much worse would still be caught.
        assert held.max_abs_z_drift_m < unheld.max_abs_z_drift_m * 1.05


@pytest.mark.parametrize("delta_m", [VALIDATED_DX_M, -VALIDATED_DX_M])
def test_no_task_torque_ever_reaches_a_held_joint(delta_m: float):
    """shoulder_pan / wrist_2 / wrist_3 keep exactly-zero task-torque columns,
    which is the structural (not gain-dependent) part of this mechanism."""
    r = rollout("x_only_1x3", delta_m)
    assert r.max_held_joint_task_torque_nm == 0.0
    # They still MOVE a little -- the posture spring is finite-stiffness, not a
    # kinematic lock -- but only slightly.
    assert 0.0 < r.max_held_joint_excursion_rad < 0.05


# --------------------------------------------------------------------------- #
# 4. The measured envelope boundary.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("delta_m", [0.03, -0.03])
def test_envelope_boundary_is_where_it_was_measured(delta_m: float):
    """Recorded, not worked around: at 0.03 m the Z-drift guard trips in both
    directions. World X and world Z are not independent inside the 2D subspace
    these three joints span, so an X move necessarily drags Z, and only the
    posture spring resists it. The guard is doing its job -- do not read the
    0.02 m validation above as covering larger displacements."""
    r = rollout("x_only_1x3", delta_m)
    assert r.guard_tripped
    assert "Z" in r.guard_reason
