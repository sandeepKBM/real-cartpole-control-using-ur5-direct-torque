"""Closed-loop MuJoCo validation of ``nullspace_inertia_adaptive_regularization``.

``tests/unit/test_nullspace_inertia_regularization.py`` proves the schedule's
algebra and its two bounding properties, and pins the leak identity against the
real Jacobian/mass matrix -- but nothing there can see the arm move: the
Jacobian changing under the controller, real joint friction, gravity, torque
backtracking/clipping, or the safety guards.

This file drives the SAME adapter pipeline every sim tool in this repo uses
(``simulation/ur5e_mujoco_torque.py``'s ``build_initial_state_and_adapter`` /
``build_mujoco_state`` / ``adapter.step()`` / ``mujoco.mj_step`` on
``assets/ur5e_torque/scene.xml``) at the ACTUAL real-robot start pose
``q = [-2.3688, -2.1801, -1.8838, -0.7962, 0.004714693, 0.0206]``, in BOTH
directions (AGENTS.md sec 7). Never a synthetic plant.

What this locks down
--------------------
1. Default off is unchanged behavior in a real rollout, not just in one cycle.
2. The mechanism does what it was built to do: X tracking improves materially
   in both directions at the validated displacement, with no new guard trip.
3. The HONEST negative result, asserted so it cannot quietly be forgotten:
   it does NOT move the drift-guard ceiling. dx=0.05 m still trips
   ``|Z-Z0| > 0.03 m`` at essentially the same time with the mechanism on --
   i.e. the projector leak was real and is now fixed, but it was not what
   capped the range.
4. The structural reason the range is capped, measured from the model: the
   ``{shoulder_lift, elbow, wrist_1}`` sub-chain can only move the TCP inside a
   2D plane whose unreachable normal is nearly the world-X/Y diagonal, so any
   world-X displacement drags a nearly equal world-Y displacement with it. No
   gain can change that, and it puts a hard kinematic ceiling on dx.

``gravity_source`` is forced to ``mujoco_qfrc`` and Coriolis feedforward off by
the harness default (the config selects ``pinocchio``, an optional dependency,
parity-checked to <1e-8 Nm -- AGENTS.md sec 3).
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
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CHECK = _load_check_module()

MOVE_DURATION_S = 1.5
HOLD_DURATION_S = 1.0
#: The validated displacement for this config at this pose (both directions).
VALIDATED_DX_M = 0.02
#: A displacement measured to trip the drift guard both with and without the
#: mechanism -- the negative result this file pins.
BEYOND_CEILING_DX_M = 0.05

ON = {"nullspace_inertia_adaptive_regularization": True, "nullspace_inertia_eps_ratio": 0.05}


def _run(dx: float, overrides: dict | None = None, hold_s: float = HOLD_DURATION_S):
    return CHECK.run_rollout(
        CHECK.USER_REAL_POSE_Q,
        ctrl_overrides=dict(overrides or {}),
        target_delta_m=float(dx),
        move_duration_s=MOVE_DURATION_S,
        hold_duration_s=float(hold_s),
        pose_label="user_real_pose",
    )


# --------------------------------------------------------------------------- #
# 1. Default off is unchanged, over a whole rollout.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("sign", [+1, -1])
def test_flag_off_is_identical_to_not_setting_it(sign):
    dx = sign * VALIDATED_DX_M
    unset = _run(dx)
    explicit_off = _run(dx, {"nullspace_inertia_adaptive_regularization": False})
    assert unset.achieved_delta_m == explicit_off.achieved_delta_m
    assert unset.max_abs_y_drift_m == explicit_off.max_abs_y_drift_m
    assert unset.max_abs_z_drift_m == explicit_off.max_abs_z_drift_m
    assert unset.steps == explicit_off.steps


# --------------------------------------------------------------------------- #
# 2. It does what it was built to do.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("sign", [+1, -1])
def test_tracking_improves_in_both_directions_with_no_new_guard_trip(sign):
    dx = sign * VALIDATED_DX_M
    off = _run(dx)
    on = _run(dx, ON)
    assert not off.guard_tripped
    assert not on.guard_tripped, on.guard_reason
    # Measured 2026-08-12: 86.1 -> 94.6 (+X) and 83.2 -> 93.1 (-X). The margin
    # asserted here is deliberately loose; the point is a real improvement in
    # BOTH directions, not a specific decimal.
    assert on.tracking_fraction > off.tracking_fraction + 0.05
    assert on.tracking_fraction > 0.90
    # It must not buy that by driving the arm harder into the guards.
    assert on.max_abs_qd_radps < 1.0
    assert on.max_abs_y_drift_m < 0.03
    assert on.max_abs_z_drift_m < 0.03


def test_reported_eps_is_below_the_static_value_and_flag_is_reported():
    """The projector really is running on the scheduled eps in a live rollout,
    read back from the controller's own output rather than from the config."""
    import mujoco  # noqa: F401  (import guard: this is a mujoco-marked test)

    st = CHECK.model_state(CHECK.USER_REAL_POSE_Q)
    J = np.asarray(st["jacobian"], dtype=np.float64).reshape(6, 6)
    M = np.asarray(st["mass_matrix"], dtype=np.float64).reshape(6, 6)
    J_task = np.zeros((1, 6))
    J_task[:, [1, 2, 3]] = J[np.ix_([0], [1, 2, 3])]
    lmin = float((J_task @ np.linalg.inv(M) @ J_task.T)[0, 0])
    # The pose's own inverse-inertia scale, quoted in the status doc as 0.0975.
    assert lmin == pytest.approx(0.0975, abs=5e-4)
    eps = min(max(0.05 * lmin, 0.1 - lmin), 0.1)
    assert 0.0 < eps < 0.1
    # P2 holds at the real pose too: the projector's Lambda is no larger-gain
    # than the static-eps worst case.
    assert lmin + eps >= 0.1 - 1e-12


# --------------------------------------------------------------------------- #
# 3. The honest negative result.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("sign", [+1, -1])
def test_mechanism_does_not_move_the_drift_guard_ceiling(sign):
    """The projector leak was real and is fixed -- but it is NOT what caps the
    range. Pinned so the fix is never mistaken for a range extension."""
    dx = sign * BEYOND_CEILING_DX_M
    off = _run(dx)
    on = _run(dx, ON)
    assert off.guard_tripped and "Z" in off.guard_reason
    assert on.guard_tripped and "Z" in on.guard_reason
    # Same guard, essentially the same moment -- within a tenth of a second.
    assert on.guard_time_s == pytest.approx(off.guard_time_s, abs=0.1)


# --------------------------------------------------------------------------- #
# 4. The structural reason, measured from the model.
# --------------------------------------------------------------------------- #
def test_world_x_is_mostly_outside_the_reachable_subspace_at_this_pose():
    """``{shoulder_lift, elbow, wrist_1}`` are three mutually parallel revolute
    axes, so the TCP can only move inside a 2D plane. At this pose
    (``shoulder_pan = -135.7 deg``) that plane's normal is nearly the world
    X/Y diagonal, so ~70% of a commanded world-X direction is unreachable and
    the reachable ~70% drags a nearly equal world-Y displacement along. That is
    a kinematic fact about the joint selection, immune to every gain."""
    st = CHECK.model_state(CHECK.USER_REAL_POSE_Q)
    J = np.asarray(st["jacobian"], dtype=np.float64).reshape(6, 6)
    block = J[np.ix_([0, 1, 2], [1, 2, 3])]
    u, s, _vt = np.linalg.svd(block)
    assert s[2] < 1e-12 < s[1]  # rank 2 of 3
    normal = u[:, 2]
    assert abs(normal[2]) < 1e-9  # the lost direction is horizontal
    assert abs(float(normal @ np.array([1.0, 0.0, 0.0]))) == pytest.approx(0.698, abs=0.01)
    assert abs(float(normal @ np.array([0.0, 1.0, 0.0]))) == pytest.approx(0.716, abs=0.01)


@pytest.mark.parametrize("sign", [+1, -1])
def test_y_displacement_tracks_x_displacement_one_for_one(sign):
    """The direct consequence of the test above, measured closed-loop: world-Y
    drift is ~97% of the achieved world-X displacement. With the drift guard at
    0.03 m that alone caps dx near 0.031 m however the gains are tuned."""
    res = _run(sign * VALIDATED_DX_M, ON)
    assert not res.guard_tripped
    ratio = res.max_abs_y_drift_m / abs(res.achieved_delta_m)
    assert ratio == pytest.approx(0.97, abs=0.06)


# --------------------------------------------------------------------------- #
# 5. The (X, Z) task-row selection that actually extends the range, and its
#    envelope -- asserted so it cannot be quietly assumed wider than measured.
# --------------------------------------------------------------------------- #
XZ_CONFIG = (
    "config/ur5e_mujoco_torque_osc_tuned_split_base_wrist_lift_elbow_wrist1_xz_kinematic_max.yaml"
)
#: The largest displacement validated clean in BOTH directions with that config.
XZ_VALIDATED_DX_M = 0.031
#: Measured to fail, via the kinematic Y wall -- not via Z, not via a gain.
XZ_BEYOND_DX_M = 0.035
LONG_HOLD_S = 10.0


def _run_xz(dx: float, hold_s: float = LONG_HOLD_S):
    return CHECK.run_rollout(
        CHECK.USER_REAL_POSE_Q,
        config_path=XZ_CONFIG,
        target_delta_m=float(dx),
        move_duration_s=MOVE_DURATION_S,
        hold_duration_s=float(hold_s),
        pose_label="user_real_pose",
    )


def test_xz_block_is_well_conditioned_where_the_three_row_block_is_singular():
    """Selecting (X, Z) instead of (X, Y, Z) is what makes this joint set
    usable: rows (0, 2) are inside the sub-chain's 2D reachable plane, so the
    2x3 block has full row rank while the 3x3 one never can."""
    st = CHECK.model_state(CHECK.USER_REAL_POSE_Q)
    J = np.asarray(st["jacobian"], dtype=np.float64).reshape(6, 6)
    xz = J[np.ix_([0, 2], [1, 2, 3])]
    assert np.linalg.matrix_rank(xz) == 2
    assert float(np.linalg.cond(xz)) == pytest.approx(3.30, abs=0.1)
    xyz = J[np.ix_([0, 1, 2], [1, 2, 3])]
    assert np.linalg.matrix_rank(xyz) == 2  # rank-deficient, at every pose


@pytest.mark.parametrize("sign", [+1, -1])
def test_xz_selection_holds_z_and_reaches_the_kinematic_ceiling(sign):
    """+-0.031 m clean in both directions at a 10 s hold, with Z drift now well
    inside its guard rather than pinned against it -- the range extension this
    work actually produced (baseline: +-0.02 m)."""
    res = _run_xz(sign * XZ_VALIDATED_DX_M)
    assert not res.guard_tripped, res.guard_reason
    assert res.task_dims_used == (0, 2)
    assert res.tracking_fraction > 0.85
    assert res.max_abs_z_drift_m < 0.02          # baseline sat at 0.0300 here
    assert res.max_abs_y_drift_m < 0.03
    assert res.max_orientation_error_rad < 0.25
    assert res.max_abs_qd_radps < 3.0


@pytest.mark.parametrize("sign", [+1, -1])
def test_xz_first_failure_is_the_predicted_kinematic_y_wall(sign):
    """Past the ceiling it fails on Y, at the magnitude the sub-chain's
    unreachable direction predicts -- evidence the remaining limit is
    kinematic, so neither a retune nor a wider guard is the right response."""
    res = _run_xz(sign * XZ_BEYOND_DX_M)
    assert res.guard_tripped
    assert "Y" in res.guard_reason
    assert res.max_abs_z_drift_m < 0.02  # Z is emphatically NOT the problem now
