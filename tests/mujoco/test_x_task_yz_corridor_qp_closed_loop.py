"""Closed-loop MuJoCo validation of the reduced-task (X + orientation) QP with
a Y/Z corridor, on the REAL UR5e model at the real deployment pose ARM_Q0.

``tests/unit/test_x_task_yz_corridor_qp.py`` proves the algebra and, crucially,
the reduced-Jacobian claim (Y/Z cannot touch the QP Hessian). Nothing there can
see whether the resulting controller actually moves a real robot better: the
Jacobian changing under the controller, joint friction
(``assets/ur5e_torque/ur5e_torque.xml``'s ``frictionloss``/``damping``),
gravity, the torque clip/rate-limit stage and the safety guards are all absent
from a synthetic Jacobian.

This file drives the SAME adapter pipeline every sim tool in this repo uses
(``build_initial_state_and_adapter`` / ``build_mujoco_state`` /
``adapter.step()`` / ``mujoco.mj_step``), through the per-step loop in
``tools/diagnostics/x_task_yz_corridor_qp_sim_check.py`` (also runnable
standalone); this file is the reproducible assertion layer over it.

What this locks down
--------------------
1. The reduced task really does track X better than the tuned 6D OSC
   controller at ARM_Q0 -- both directions (AGENTS.md sec 7).
2. Y/Z genuinely move MORE than under the OSC controller (they are supposed
   to: the whole point is that they are free inside a corridor, not held).
3. The corridor is an EXACT no-op when the corridor is never approached --
   byte-identical trajectories, not merely similar.
4. The composition of the two mechanisms is not cosmetic: at a displacement
   large enough to reach a corridor wall, the corridor ALONE walks the arm
   into a near-singularity, and adding the manipulability row prevents it.
5. Nothing ever reports infeasible, and torque stays inside the hard limit.

``gravity_source`` is forced to ``mujoco_qfrc`` and Coriolis feedforward off,
for the same reason the manipulability-CBF and SCI closed-loop tests do it: the
tuned config selects ``pinocchio``, an optional dependency, parity-checked to
<1e-8 Nm (AGENTS.md sec 3).
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
    path = REPO_ROOT / "tools" / "diagnostics" / "x_task_yz_corridor_qp_sim_check.py"
    spec = importlib.util.spec_from_file_location("x_task_yz_corridor_qp_sim_check", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec: the module defines @dataclass types, and dataclasses
    # resolves annotations via sys.modules[cls.__module__].
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CHECK = _load_check_module()

MOVE_DURATION_S = 1.5
HOLD_DURATION_S = 1.0

_CACHE: dict[tuple, object] = {}


def rollout(kind: str, delta_m: float):
    """Memoized -- each closed-loop rollout costs seconds to tens of seconds
    (the QP is expensive; see the timing test below) and several tests compare
    the same pairs.

    ``kind`` selects one of five configurations, all sharing the SAME drift
    guards so a comparison between any two of them is a comparison of the
    mechanisms and nothing else (the shipped default-off config carries the
    tighter 0.03 m guards on purpose, which is why it is not used here).
    """
    key = (kind, float(delta_m))
    if key in _CACHE:
        return _CACHE[key]
    common = dict(
        target_delta_m=float(delta_m),
        move_duration_s=MOVE_DURATION_S,
        hold_duration_s=HOLD_DURATION_S,
        gravity_source="mujoco_qfrc",
        coriolis_feedforward=False,
    )
    if kind == "osc":
        res = CHECK.run_rollout(
            CHECK.ARM_Q0, controller_kind="impedance", config_path=CHECK.OSC_CONFIG,
            label="osc_tuned", **common,
        )
    else:
        corridor, cbf = {
            "off": (False, False),
            "corridor": (True, False),
            "cbf": (False, True),
            "both": (True, True),
        }[kind]
        res = CHECK.run_rollout(
            CHECK.ARM_Q0, config_path=CHECK.NEW_CONFIG_ENABLED,
            corridor=corridor, cbf=cbf, label=f"new_{kind}", **common,
        )
    _CACHE[key] = res
    return res


#: The X+Z variant: world X and Z tracked as task rows, Y alone bounded by the
#: corridor. Same controller, same guards, same task_excluded_joints -- only
#: the row sets and the gains re-derived for them differ.
XZ_CONFIG = REPO_ROOT / "config" / "ur5e_mujoco_torque_x_z_task_y_corridor_qp_enabled.yaml"


def rollout_xz(delta_m: float, move_duration_s: float = MOVE_DURATION_S):
    key = ("xz", float(delta_m), float(move_duration_s))
    if key in _CACHE:
        return _CACHE[key]
    res = CHECK.run_rollout(
        CHECK.ARM_Q0,
        config_path=str(XZ_CONFIG.relative_to(REPO_ROOT)),
        target_delta_m=float(delta_m),
        move_duration_s=float(move_duration_s),
        hold_duration_s=HOLD_DURATION_S,
        gravity_source="mujoco_qfrc",
        coriolis_feedforward=False,
        label="xz",
    )
    _CACHE[key] = res
    return res


# --------------------------------------------------------------------------- #
# 0. The pose is the one everything else assumes.
# --------------------------------------------------------------------------- #
def test_arm_q0_is_the_documented_real_deployment_pose():
    assert np.allclose(
        CHECK.ARM_Q0,
        [-2.3688, -2.1801, -1.8838, -0.7962, 0.004714693, 0.0206],
        atol=0.0,
    )
    prof = CHECK.run_profile(CHECK.ARM_Q0)
    table = dict(zip(prof.values, zip(prof.manipulability, prof.cond_j)))
    mu0, cond0 = table[0.004714693]
    # The pose's own conditioning, as reported by the IK feasibility pre-check
    # (cond(J) ~= 1396). If this drifts, every epsilon and corridor number
    # calibrated below stops meaning what it says.
    assert cond0 == pytest.approx(1396.0, rel=0.02)
    # mu at the start pose, which is what manipulability_cbf_epsilon = 3.0e-4
    # in the shipped config was sized against.
    assert mu0 == pytest.approx(4.33e-4, rel=0.02)
    assert mu0 > 3.0e-4, "epsilon 3.0e-4 must leave h > 0 at the start pose"


# --------------------------------------------------------------------------- #
# 1. The reduced task tracks X better than 6D OSC -- both directions.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("delta_m", [0.02, -0.02])
def test_reduced_task_tracks_x_better_than_tuned_osc(delta_m):
    osc = rollout("osc", delta_m)
    new = rollout("off", delta_m)
    assert new.tracking_fraction > osc.tracking_fraction
    # Measured 2026-08-13: ~0.71/0.70 vs ~0.37/0.36, i.e. roughly double.
    assert new.tracking_fraction > 1.5 * osc.tracking_fraction
    # ... and not by spending more torque. Peak torque at this pose is
    # dominated by the gravity/friction hold (~32 Nm on shoulder_lift), which
    # both controllers pay identically, so this is a "within noise" bound
    # rather than a strict inequality: measured 2026-08-13, +X 32.33 vs 35.28
    # (the new controller is lower) and -X 31.85 vs 31.82 (a 0.09% difference,
    # i.e. the same hold torque).
    assert new.max_abs_tau_nm <= 1.05 * osc.max_abs_tau_nm


@pytest.mark.parametrize("delta_m", [0.06, -0.06])
def test_reduced_task_also_wins_at_the_top_of_the_validated_range(delta_m):
    osc = rollout("osc", delta_m)
    new = rollout("both", delta_m)
    assert new.tracking_fraction > osc.tracking_fraction
    # The tuned OSC controller trips its own 0.03 m drift guard here; the new
    # controller (with the corridor's widened, evidence-scoped guard) does not.
    assert osc.guard_tripped and "Y-Y0" in osc.guard_reason
    assert not new.guard_tripped


@pytest.mark.parametrize("delta_m", [0.02, -0.02, 0.06, -0.06])
def test_orientation_and_joint_velocity_stay_inside_the_guards(delta_m):
    new = rollout("both", delta_m)
    assert new.max_orientation_error_rad < 0.25
    assert new.max_abs_qd_radps < 3.0
    assert new.max_abs_tau_nm <= 150.0 + 1e-6


# --------------------------------------------------------------------------- #
# 2. Y/Z really are freer than under the 6D hold. This is the DESIGN, not a
#    regression -- assert it explicitly so a future change that quietly
#    re-stiffens them is caught.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("delta_m", [0.02, -0.02])
def test_both_y_and_z_move_more_than_under_the_osc_hold(delta_m):
    """UPDATED 2026-08-13 with the shoulder_pan fix, and the reason matters.

    This test used to assert Z freer but Y *tighter* than the 6D OSC hold, on
    the hypothesis that the reduced task's extra X authority would let the arm
    reach the same X with less off-axis excursion. `task_excluded_joints`
    deliberately removes that extra authority -- and shoulder_pan is not just
    any joint here, it is the DOMINANT world-Y actuator at ARM_Q0 (Jacobian
    coefficient -0.578, 2.5x the next joint). So Y is now genuinely freer too:
    measured 0.0120 vs the OSC hold's 0.0069 at dx=+0.02 m, and 0.0112 vs
    0.0066 at -0.02 m, where it used to be lower.

    That is the design working as intended, not a regression -- "Y and Z are
    free inside a corridor rather than held" is the entire premise -- but the
    old assertion encoded the opposite, so it is corrected rather than
    loosened. What must remain true, and is asserted below, is that the
    freedom stays well inside the corridor the HOCBF is defending.
    """
    osc = rollout("osc", delta_m)
    new = rollout("off", delta_m)
    assert new.max_abs_z_drift_m > osc.max_abs_z_drift_m
    assert new.max_abs_y_drift_m > osc.max_abs_y_drift_m
    # ... but still comfortably inside the 0.05 m corridor half-width, which is
    # the claim that actually protects anything.
    assert new.max_abs_y_drift_m < 0.05
    assert new.max_abs_z_drift_m < 0.05


# --------------------------------------------------------------------------- #
# 3. EXACT no-op when the corridor is never approached (both directions).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("delta_m", [0.02, -0.02])
def test_corridor_is_an_exact_no_op_when_its_walls_are_never_reached(delta_m):
    off = rollout("off", delta_m)
    corr = rollout("corridor", delta_m)
    assert corr.corridor_active_steps == 0
    assert corr.corridor_row_active_steps == (0, 0, 0, 0)
    # Byte-identical trajectory, not merely similar: an inactive inequality
    # row must not perturb the solve at all.
    assert corr.achieved_delta_m == off.achieved_delta_m
    assert corr.max_abs_y_drift_m == off.max_abs_y_drift_m
    assert corr.max_abs_z_drift_m == off.max_abs_z_drift_m
    assert corr.max_abs_tau_nm == off.max_abs_tau_nm
    assert corr.steps == off.steps
    assert corr.guard_reason == off.guard_reason
    # ... but the rows WERE built and paid for.
    assert corr.qp_rows == 4
    assert corr.qp_mean_s > off.qp_mean_s


@pytest.mark.parametrize("delta_m", [0.06, -0.06])
def test_the_corridor_now_engages_at_the_top_of_the_validated_range(delta_m):
    """NEW SCOPE 2026-08-13. dx = +-0.06 m used to be an exact corridor no-op
    and no longer is: with shoulder_pan -- the dominant world-Y actuator at
    ARM_Q0 -- held out of the task, Y/Z excursion grows enough to reach the
    walls (measured 649 active cycles at +0.06 m, 728 at -0.06 m, from 0).

    Asserted explicitly rather than silently dropped from the no-op test
    above, because it is a real behavioral change: the corridor stopped being
    decorative at this displacement, which is exactly when its own
    calibration evidence (the 0.0423 m natural transient) starts to bite.
    """
    corr = rollout("corridor", delta_m)
    assert corr.corridor_active_steps > 100
    assert corr.infeasible_steps == 0
    # It engages, and the excursion it is defending against stays inside the
    # wall it is defending -- the corridor is doing its job, not failing it.
    assert corr.max_abs_y_drift_m < 0.06
    assert corr.max_abs_z_drift_m < 0.062


# --------------------------------------------------------------------------- #
# 4. The manipulability row does real work at this pose, and only where needed.
# --------------------------------------------------------------------------- #
def test_cbf_is_inert_on_the_plus_x_move_that_never_approaches_a_singularity():
    off = rollout("off", 0.02)
    cbf = rollout("cbf", 0.02)
    assert cbf.cbf_active_steps == 0
    assert cbf.achieved_delta_m == off.achieved_delta_m
    assert cbf.min_manipulability == off.min_manipulability


def test_cbf_holds_manipulability_up_on_the_minus_x_move_that_does():
    """AGENTS.md sec 7 in action: -X at ARM_Q0 walks mu down, +X does not.
    A one-directional test would have reported "the CBF does nothing"."""
    off = rollout("off", -0.02)
    cbf = rollout("cbf", -0.02)
    assert off.min_manipulability < 3.0e-4, (
        "the baseline no longer dips below the barrier at this cell; the "
        "comparison below would be vacuous"
    )
    assert cbf.cbf_active_steps > 500
    assert cbf.min_manipulability > off.min_manipulability
    assert cbf.min_manipulability > 3.0e-4  # held at/above the barrier
    assert cbf.infeasible_steps == 0


# --------------------------------------------------------------------------- #
# 5. The two mechanisms are complementary, not redundant -- measured.
#    dx = 0.12 m is OUTSIDE the corridor-calibration evidence (which covers
#    dx <= 0.06 m); it is used here precisely because it is the smallest cell
#    that actually drives the end effector into a corridor wall, and the claim
#    being tested is about the interaction of the two rows, not about that
#    displacement being safe.
# --------------------------------------------------------------------------- #
UNVALIDATED_DX = 0.12


def test_the_corridor_actually_engages_at_a_large_displacement():
    corr = rollout("corridor", UNVALIDATED_DX)
    assert corr.corridor_active_steps > 100
    # It is the Y/Z upper walls that fire on this move, not all four.
    assert sum(1 for c in corr.corridor_row_active_steps if c > 0) >= 1


def test_corridor_alone_walks_into_a_near_singularity_that_the_cbf_prevents():
    """The composition claim, with numbers. With only the corridor rows the
    QP buys its Y/Z freedom by driving the arm through a nearly singular
    configuration; adding the manipulability row to the SAME QP holds mu at
    its start value instead.

    UPDATED 2026-08-13. The composition claim itself survives the bug fixes
    intact and is still large: measured mu 1.59e-5 (corridor alone) vs
    4.33e-4 (both) -- a 27x difference, held exactly at the start pose's own
    mu. What did NOT survive is the old "...and the move completes" half: see
    the separate test below. The ratio bound is loosened from 50x to 20x
    against the measured 27x, and the absolute floor -- which is the claim
    that actually means something, mu held at/above the barrier -- is kept
    unchanged.
    """
    corr = rollout("corridor", UNVALIDATED_DX)
    both = rollout("both", UNVALIDATED_DX)
    assert corr.min_manipulability < 1.0e-4
    assert both.min_manipulability > 20.0 * corr.min_manipulability
    assert both.min_manipulability > 3.0e-4
    assert both.infeasible_steps == 0


def test_the_large_move_is_now_outside_the_envelope_for_every_flag_combination():
    """HONEST REPLACEMENT (2026-08-13) for
    ``test_neither_mechanism_alone_completes_the_large_move_but_together_they_do``.

    That test asserted that corridor + manipulability CBF together completed
    dx = 0.12 m where each alone failed. After the shoulder_pan fix that is no
    longer true and the test is not repairable by loosening a threshold:
    holding shoulder_pan out of the task removes the joint with the largest
    world-X AND world-Y Jacobian coefficients at ARM_Q0, so dx = 0.12 m now
    trips the Z-drift guard for every flag combination (X tracking 0.889 ->
    0.294 at this cell).

    This is the single genuine regression from the fix pass and it is
    asserted, not hidden -- if a future change makes dx = 0.12 m reachable
    again, this test fails loudly and
    docs/status/x_task_yz_corridor_qp_2026-08-13.md's practical-floor
    statement must be revisited rather than quietly inherited.

    What the fix DID buy at this cell is asserted too: the orientation error
    that used to sit at 0.2423 rad -- right at its 0.25 rad guard -- is now
    0.0456 rad, and shoulder_pan does not move at all.
    """
    both = rollout("both", UNVALIDATED_DX)
    for kind in ("off", "corridor", "cbf", "both"):
        assert rollout(kind, UNVALIDATED_DX).guard_tripped, kind
    assert "Z-Z0" in both.guard_reason
    assert both.max_orientation_error_rad < 0.10
    assert np.degrees(both.shoulder_pan_range_rad) < MAX_SHOULDER_PAN_RANGE_DEG


# --------------------------------------------------------------------------- #
# 6. Real-time cost: measured here too, and asserted to be the blocker it is.
# --------------------------------------------------------------------------- #
def test_qp_cost_is_measured_and_is_over_the_500hz_budget():
    """Not a performance regression test -- a HONESTY test. The controller is
    sim-only scope precisely because of this number, and if a future change
    ever brings it under budget that fact must be noticed rather than
    inherited from a stale doc."""
    both = rollout("both", -0.02)
    off = rollout("off", -0.02)
    assert off.qp_mean_s < CHECK.DIRECT_TORQUE_BUDGET_S
    assert both.qp_mean_s > 0.0
    assert both.qp_mean_s > 5.0 * off.qp_mean_s
    assert both.qp_mean_s > CHECK.DIRECT_TORQUE_BUDGET_S, (
        "the corridor+CBF QP now fits in the 500 Hz budget -- update "
        "docs/status/x_task_yz_corridor_qp_2026-08-13.md's scope statement "
        "instead of loosening this assertion"
    )


def test_isolated_timing_profile_reports_all_four_row_counts():
    rows = CHECK.run_timing_profile(CHECK.ARM_Q0, n_calls=30)
    by_label = {r.label: r for r in rows}
    assert by_label["no_rows"].n_ineq_rows == 0
    assert by_label["manipulability_only"].n_ineq_rows == 1
    assert by_label["corridor_only"].n_ineq_rows == 4
    assert by_label["corridor_and_manipulability"].n_ineq_rows == 5
    # Monotone in row count, and the zero-row case is the only cheap one.
    assert by_label["no_rows"].mean_total_s < CHECK.DIRECT_TORQUE_BUDGET_S
    assert (
        by_label["corridor_and_manipulability"].mean_total_s
        > by_label["corridor_only"].mean_total_s
        > by_label["manipulability_only"].mean_total_s
        > by_label["no_rows"].mean_total_s
    )


# --------------------------------------------------------------------------- #
# 7. The three 2026-08-13 bug fixes, in closed loop.
#
# These are the assertions the fixes exist to make true. They are deliberately
# stated as bounds on PHYSICAL quantities (how far the base swung, how close
# the orientation got to its guard) rather than as regression baselines, so a
# future change is judged against the requirement rather than against whatever
# this session happened to measure.
# --------------------------------------------------------------------------- #
#: shoulder_pan is held out of the task by BOTH a zeroed Jacobian column and a
#: shut torque box, so its commanded torque is exactly ``tau_hold`` -- a weak
#: spring plus the model's own 5.0 Nm frictionloss. Before the fix it swung
#: 4.32-13.15 deg across this matrix. 0.5 deg is ~9x below the smallest of
#: those and far below any plausible wall-clearance budget, while leaving room
#: for the residual motion a finite spring genuinely allows.
MAX_SHOULDER_PAN_RANGE_DEG = 0.5

#: The orientation guard is 0.25 rad. Before the fix, three of the five matrix
#: cells sat at 0.2405-0.2503 rad -- i.e. AT the guard, with no margin at all,
#: which is why two of them tripped it. 0.10 rad is a 2.5x margin below the
#: guard, below EVERY pre-fix measurement anywhere on the matrix (the smallest
#: was 0.1167), and above the measured post-fix values in this range
#: (0.0798 at dx=-0.06 m, 0.0536 at +0.06 m) with room to spare -- so it is a
#: real improvement bound, not a restatement of the guard and not a
#: regression baseline pinned to this session's exact numbers.
MAX_ORIENTATION_ERROR_RAD = 0.10


@pytest.mark.parametrize("delta_m", [0.02, -0.02, 0.06, -0.06])
def test_shoulder_pan_is_structurally_held_across_the_validated_range(delta_m):
    """Bug 1. ARM_Q0 pins shoulder_pan at -135.7 deg for real wall/base
    clearance; the reduced task's 2 redundant DOF used to spend it."""
    new = rollout("both", delta_m)
    assert np.degrees(new.shoulder_pan_range_rad) < MAX_SHOULDER_PAN_RANGE_DEG
    assert np.degrees(new.max_abs_shoulder_pan_dev_rad) < MAX_SHOULDER_PAN_RANGE_DEG


@pytest.mark.parametrize("delta_m", [0.02, -0.02, 0.06, -0.06])
def test_no_task_torque_reaches_the_excluded_joint(delta_m):
    """The mechanism, not just its effect: the commanded shoulder_pan torque is
    pinned to ``tau_hold`` (posture spring + damping + gravity), and at ARM_Q0
    gravity torque about the vertical base axis is exactly zero, so what is
    left is only the small spring/damper term. Before the fix the QP drove this
    joint with up to 6.66 Nm of task and corridor torque."""
    new = rollout("both", delta_m)
    assert new.max_abs_shoulder_pan_tau_nm < 1.0


@pytest.mark.parametrize("delta_m", [0.02, -0.02, 0.06, -0.06])
def test_orientation_error_keeps_real_margin_below_its_guard(delta_m):
    """Bugs 2 and 3. ``kd_rot`` was never put through the same Lambda unit
    conversion ``kp_x``/``kd_x`` got, and ``kp_rot = 0`` was inherited from a
    Lambda-weighted controller whose instability argument does not apply to
    this plain weighted-least-squares QP -- which also left the three
    orientation rows carrying a 1e-6 weight in the Hessian."""
    new = rollout("both", delta_m)
    assert new.max_orientation_error_rad < MAX_ORIENTATION_ERROR_RAD
    assert not new.guard_tripped


def test_x_tracking_did_not_regress_from_the_bug_fixes():
    """Requirement 6 of the fix brief, asserted rather than asserted-about:
    holding shoulder_pan out of the task removes the joint with the LARGEST
    world-X Jacobian coefficient at this pose, so 'X-tracking is unchanged' is
    a real claim that has to be checked, not an obvious one."""
    for delta_m in (0.02, -0.02, 0.06, -0.06):
        new = rollout("both", delta_m)
        osc = rollout("osc", delta_m)
        # Still comfortably ahead of the 6D OSC controller, which is the
        # comparison this controller exists to win.
        assert new.tracking_fraction > osc.tracking_fraction


@pytest.mark.parametrize("delta_m", [0.02, -0.02, 0.06, -0.06])
def test_the_pin_holds_bit_exactly_on_every_cycle_of_a_real_rollout(delta_m):
    """The guarantee itself, in closed loop rather than against a synthetic
    Jacobian: on every one of the ~1250 cycles, the pre-clip commanded torque
    at the excluded joint equalled ``tau_hold`` bit-for-bit -- through corridor
    walls, an active manipulability row and the velocity-implied bounds."""
    new = rollout("both", delta_m)
    assert new.steps > 100, "vacuous unless the rollout actually ran"
    assert new.excluded_joint_pin_violations == 0


# --------------------------------------------------------------------------- #
# 8. X+Z row set (2026-08-13): world X and Z tracked, Y alone bounded.
#
# These assert the STRUCTURE of the row-set change and the DESIGN CLAIM as a
# relative comparison against the X-only arm at the same cell. They are
# deliberately written not to hardcode absolute tracking/orientation numbers,
# so that re-tuning kp_rot/kd_rot for this row set cannot silently invalidate
# them -- the gains are an input to the comparison, not baked into it.
# --------------------------------------------------------------------------- #
def test_the_xz_config_is_the_same_controller_with_different_rows():
    """Guards, corridor calibration and the shoulder_pan exclusion must be
    identical to the X-only config, or a comparison between the two arms is
    not a comparison of the row sets."""
    import yaml

    def controller_section(name):
        with open(REPO_ROOT / "config" / name, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)["controller"]

    x_only = controller_section("ur5e_mujoco_torque_x_task_yz_corridor_qp_enabled.yaml")
    xz = controller_section("ur5e_mujoco_torque_x_z_task_y_corridor_qp_enabled.yaml")
    assert xz["task_axis_rows"] == [0, 2]
    assert xz["corridor_axis_rows"] == [1]
    assert xz["task_excluded_joints"] == x_only["task_excluded_joints"] == [0]
    assert xz["safety"] == x_only["safety"]
    for field in ("y_corridor_half_width_m", "z_corridor_half_width_m",
                  "yz_corridor_alpha1", "yz_corridor_alpha2",
                  "manipulability_cbf_epsilon", "yz_corridor_enabled",
                  "manipulability_cbf"):
        assert xz[field] == x_only[field], field


@pytest.mark.parametrize("delta_m", [0.06, -0.06])
def test_tracking_z_removes_its_corridor_rows_from_the_qp(delta_m):
    """The QP gets structurally smaller, not bigger, when Z is promoted: Z's
    two HOCBF rows disappear because a tracked axis contributes none. Five
    inequality rows (Y max/min, Z max/min, manipulability) become three."""
    x_only = rollout("both", delta_m)
    xz = rollout_xz(delta_m)
    assert x_only.qp_rows == 5
    assert xz.qp_rows == 3
    assert xz.infeasible_steps == 0


@pytest.mark.parametrize("delta_m", [0.06, -0.06, 0.12])
def test_the_pan_guarantee_survives_the_row_set_change(delta_m):
    """`task_excluded_joints` is orthogonal to which rows are tracked, and the
    bit-exact pin must hold in the X+Z configuration too."""
    xz = rollout_xz(delta_m)
    assert xz.excluded_joint_pin_violations == 0
    assert np.degrees(xz.shoulder_pan_range_rad) < MAX_SHOULDER_PAN_RANGE_DEG
    # Same 1.0 Nm bound the X-only pan test uses: measured <=0.01 Nm, against a
    # pre-fix 5.79-11.90 Nm.
    assert abs(xz.max_abs_shoulder_pan_tau_nm) < 1.0


@pytest.mark.parametrize("delta_m", [0.06, -0.06, -0.12])
def test_tracking_z_holds_z_tighter_than_bounding_it_where_the_move_completes(delta_m):
    """THE design claim of the X+Z row set, as a relative comparison, and
    SCOPED BY MEASUREMENT to the cells where it actually holds.

    A corridor only pushes back at the wall; by the time it engages the arm is
    already at 94-100% of the half-width, which is why every large-move failure
    in the X-only arm is a Z-drift trip. Tracking Z holds it roughly twice as
    tight on every cell that completes -- measured 2026-08-14: 0.0422 -> 0.0227
    at dx=-0.06 m, 0.0516 -> 0.0253 at +0.06 m, 0.0308 -> 0.0236 at -0.12 m.

    The cells that do NOT complete are a separate, explicitly negative result;
    see the test below rather than assuming this generalizes to them.
    """
    x_only = rollout("both", delta_m)
    xz = rollout_xz(delta_m)
    assert xz.max_abs_z_drift_m < x_only.max_abs_z_drift_m


@pytest.mark.parametrize("delta_m", [0.12, 0.20])
def test_tracking_z_does_NOT_rescue_the_large_move_z_drift_trip(delta_m):
    """NEGATIVE RESULT, asserted so it cannot be quietly forgotten.

    dx = +0.12 m is the cell the shoulder_pan fix regressed (design doc §5),
    and promoting Z to a task row was the obvious candidate fix for it. It does
    not work: both arms still trip |Z-Z0| > 0.06 m, at effectively identical
    drift (measured 0.0601 vs 0.0602 at +0.12 m, 0.0601 vs 0.0603 at +0.20 m).

    The interpretation that survives the measurement is that at these
    displacements Z drift is not a matter of insufficient Z authority -- which
    tracking would supply -- but of the arm being driven to a configuration
    where holding both X and Z is not available at all with shoulder_pan
    locked. Halving the drift on the completing cells (test above) and not
    moving it at all here are consistent with that reading.
    """
    x_only = rollout("both", delta_m)
    xz = rollout_xz(delta_m)
    assert x_only.guard_tripped and "Z-Z0" in x_only.guard_reason
    assert xz.guard_tripped and "Z-Z0" in xz.guard_reason
    # Both sit at the guard; neither is meaningfully better than the other.
    assert abs(xz.max_abs_z_drift_m - x_only.max_abs_z_drift_m) < 0.002
