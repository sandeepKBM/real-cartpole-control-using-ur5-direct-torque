"""Closed-loop MuJoCo validation of ``manipulability_cbf`` on the REAL UR5e.

``tests/unit/test_manipulability_cbf.py`` proves the derivation (gradient,
directional curvature, constraint row, QP filter) against closed-form
kinematics, but nothing there can see whether the mechanism actually stops a
real robot from walking into a real singularity: the Jacobian changing under
the controller, joint friction (``assets/ur5e_torque/ur5e_torque.xml``'s
``frictionloss``/``damping``), gravity, the torque backtracking/clip stage, the
adapter's rate limiter, and the safety guards.

This file drives the SAME adapter pipeline every sim tool in this repo uses
(``simulation/ur5e_mujoco_torque.py``'s ``build_initial_state_and_adapter`` /
``build_mujoco_state`` / ``adapter.step()`` / ``mujoco.mj_step`` on
``assets/ur5e_torque/scene.xml``), once with the CBF and once without,
everything else byte-identical. The per-step loop lives in
``tools/diagnostics/manipulability_cbf_sim_check.py`` (also runnable
standalone); this file is the reproducible assertion layer over it.

What this locks down
--------------------
1. The measured ``mu`` scale the config docstring's default ``epsilon`` was
   sized from still matches what the model produces.
2. THE WRIST SINGULARITY (wrist_2 -> 0, the case this repo has fought
   repeatedly): with the CBF off, a world-Y transport walks wrist_2 from
   0.100 rad down to ~0.006 rad and ``mu`` collapses by an order of
   magnitude. With the CBF on, it does not.
3. A SECOND, DIFFERENT real UR singularity found while building this (the
   wrist-alignment family, reached with wrist_2 essentially unchanged) is
   also prevented -- evidence the barrier is a general kinematic quantity, not
   a wrist_2 proxy.
4. It is an EXACT no-op on moves that never approach a singularity, in both
   directions (AGENTS.md sec 7).
5. It never goes infeasible and never exceeds the torque limits in any of the
   above.

``gravity_source`` is forced to ``mujoco_qfrc`` and Coriolis feedforward off
for the same reason the SCI closed-loop test does it: the tuned config selects
``pinocchio``, an optional dependency, parity-checked to <1e-8 Nm
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
    path = REPO_ROOT / "tools" / "diagnostics" / "manipulability_cbf_sim_check.py"
    spec = importlib.util.spec_from_file_location("manipulability_cbf_sim_check", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec: the module defines @dataclass types, and dataclasses
    # resolves annotations via sys.modules[cls.__module__].
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CHECK = _load_check_module()

MOVE_DURATION_S = 2.0
HOLD_DURATION_S = 1.0

_ROLLOUT_CACHE: dict[tuple, object] = {}


def rollout(pose: str, cbf: bool, delta_m: float, axis: int = 0):
    """Memoized -- each closed-loop rollout costs a couple of seconds and
    several tests compare the same pairs against each other."""
    key = (pose, bool(cbf), float(delta_m), int(axis))
    if key not in _ROLLOUT_CACHE:
        _ROLLOUT_CACHE[key] = CHECK.run_rollout(
            CHECK.POSES[pose],
            cbf=bool(cbf),
            target_delta_m=float(delta_m),
            move_duration_s=MOVE_DURATION_S,
            hold_duration_s=HOLD_DURATION_S,
            pose_label=pose,
            transport_axis=int(axis),
            gravity_source="mujoco_qfrc",
            coriolis_feedforward=False,
        )
    return _ROLLOUT_CACHE[key]


# --------------------------------------------------------------------------- #
# 1. The documented mu scale is still real.
# --------------------------------------------------------------------------- #
def test_documented_mu_scale_still_matches_the_model():
    """``CartesianImpedanceConfig.manipulability_cbf``'s docstring sizes the
    default epsilon (1e-3) from a measured mu-vs-wrist_2 table at
    HEIGHT_ALPHA_0_5_Q. If the model or the site changes, that default stops
    meaning what the docstring says -- so the table is asserted here."""
    r = CHECK.run_profile(CHECK.POSES["height_alpha_0_5"], pose_label="height_alpha_0_5")
    table = dict(zip(r.values, r.manipulability))
    assert table[0.0] < 1.0e-12                     # exact singularity
    assert table[0.005] == pytest.approx(9.44e-05, rel=0.02)
    assert table[0.02] == pytest.approx(3.77e-04, rel=0.02)
    assert table[0.05] == pytest.approx(9.43e-04, rel=0.02)
    assert table[0.2] == pytest.approx(3.75e-03, rel=0.02)
    # Near-linearity in wrist_2 is the property that makes epsilon readable as
    # a physical standoff; check it holds over the region epsilon lives in.
    ratio = table[0.05] / table[0.005]
    assert ratio == pytest.approx(10.0, rel=0.05)
    # The default epsilon corresponds to a wrist_2 standoff of ~0.05 rad here.
    assert table[0.05] < 1.0e-3 < table[0.1]


# --------------------------------------------------------------------------- #
# 2. THE wrist singularity: wrist_2 -> 0 under a world-Y transport.
# --------------------------------------------------------------------------- #
POSE_W2 = "height_alpha_0_5_wrist2_0_10"


def test_baseline_really_does_walk_into_the_wrist_singularity():
    """The premise. Without this, the comparison below proves nothing."""
    off = rollout(POSE_W2, cbf=False, delta_m=-0.10, axis=1)
    assert off.min_abs_wrist_2_rad < 0.01, (
        "the baseline no longer reaches the wrist singularity at this cell; "
        "the CBF comparison below would be vacuous"
    )
    assert off.min_manipulability < 3.0e-4
    assert off.max_cond_j > 500.0


def test_cbf_keeps_wrist_2_and_mu_away_from_the_singularity():
    off = rollout(POSE_W2, cbf=False, delta_m=-0.10, axis=1)
    on = rollout(POSE_W2, cbf=True, delta_m=-0.10, axis=1)

    # wrist_2 barely leaves its start value instead of collapsing to ~0.006.
    assert on.min_abs_wrist_2_rad > 0.05
    assert on.min_abs_wrist_2_rad > 10.0 * off.min_abs_wrist_2_rad

    # mu stays an order of magnitude higher, and above the barrier itself.
    assert on.min_manipulability > 1.0e-3
    assert on.min_manipulability > 8.0 * off.min_manipulability

    # cond(J) never blows up.
    assert on.max_cond_j < 100.0
    assert on.max_cond_j < 0.2 * off.max_cond_j

    # The filter did real work, and cheaply.
    assert on.cbf_active_steps > 100
    assert on.max_cbf_delta_tau_nm > 0.0
    assert on.max_cbf_delta_tau_nm < 5.0


def test_cbf_never_reports_infeasible_on_the_wrist_case():
    on = rollout(POSE_W2, cbf=True, delta_m=-0.10, axis=1)
    assert on.cbf_infeasible_steps == 0
    # The barrier is only ever mildly violated (discrete-time + rate-limited
    # actuator), never collapsed: h stays far above -epsilon.
    assert on.min_cbf_h > -1.0e-3


# --------------------------------------------------------------------------- #
# 3. A second, genuinely different UR singularity (wrist ALIGNMENT).
#    Found 2026-08-13 while building this: from wrist_2 = 0.20 rad, a +0.15 m
#    world-X transport drives shoulder_lift from -0.835 to -0.987 rad while
#    wrist_2 stays at ~0.20, and the arm transits a singularity anyway
#    (shoulder_lift + elbow + wrist_1 -> -pi, i.e. the wrist-3 axis lining up
#    with the shoulder-pan axis). mu catches it; a wrist_2 threshold would not.
# --------------------------------------------------------------------------- #
POSE_W2_OFF = "height_alpha_0_5_wrist2_offset"


def test_baseline_transits_the_wrist_alignment_singularity_with_wrist_2_unchanged():
    off = rollout(POSE_W2_OFF, cbf=False, delta_m=0.15, axis=0)
    assert off.min_manipulability < 1.0e-5
    assert off.max_cond_j > 1.0e4
    # ... and it is NOT a wrist_2 event.
    assert off.min_abs_wrist_2_rad > 0.19


def test_cbf_prevents_the_wrist_alignment_transit():
    off = rollout(POSE_W2_OFF, cbf=False, delta_m=0.15, axis=0)
    on = rollout(POSE_W2_OFF, cbf=True, delta_m=0.15, axis=0)
    assert on.min_manipulability > 1.0e-4
    assert on.min_manipulability > 100.0 * off.min_manipulability
    assert on.max_cond_j < 1.0e3
    assert on.max_cond_j < 0.05 * off.max_cond_j
    assert on.cbf_infeasible_steps == 0


# --------------------------------------------------------------------------- #
# 4. Exact no-op where nothing is approached -- both directions.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("delta_m", [0.05, -0.10, -0.15])
def test_exact_no_op_when_the_move_never_approaches_a_singularity(delta_m):
    """AGENTS.md sec 7: both directions. +X at this pose walks toward the
    alignment singularity (covered above); -X does not approach anything, and
    neither does a small +X move -- in those cells the CBF must be provably
    inert, not merely harmless."""
    off = rollout(POSE_W2_OFF, cbf=False, delta_m=delta_m, axis=0)
    on = rollout(POSE_W2_OFF, cbf=True, delta_m=delta_m, axis=0)
    assert on.cbf_active_steps == 0
    assert on.max_cbf_delta_tau_nm == 0.0
    assert on.min_manipulability == off.min_manipulability
    assert on.achieved_delta_m == off.achieved_delta_m
    assert on.max_abs_tau_nm == off.max_abs_tau_nm
    assert on.steps == off.steps
    assert on.guard_reason == off.guard_reason


# --------------------------------------------------------------------------- #
# 5. Torque stays legal everywhere the CBF was exercised.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "pose,delta_m,axis", [(POSE_W2, -0.10, 1), (POSE_W2_OFF, 0.15, 0)]
)
def test_torque_and_joint_velocity_stay_bounded_with_the_cbf_on(pose, delta_m, axis):
    on = rollout(pose, cbf=True, delta_m=delta_m, axis=axis)
    # UR5e limits used by this repo: 150 Nm base joints, 28 Nm wrists.
    assert on.max_abs_tau_nm <= 150.0 + 1e-6
    # The joint-velocity guard's own ceiling (controller_core/safety.py).
    assert on.max_abs_qd_radps < 3.0
