"""The unified pendulum experiment entrypoint.

Ordered by what breaks worst if wrong:

  1. It REFUSES a drive axis the pole cannot feel. This is the whole reason the
     tool exists: a run aimed at a dead or near-dead axis produces logs that all
     look fine and mean nothing. One such run tripped the corridor guard in
     0.134 s with the rod tip 4 mm off the floor.
  2. It dispatches to a backend that can actually EXPRESS the request. The tool-Y
     backend hardcodes pose and asset, so sending it a different pose would run
     something other than what was asked for while every log line still looked
     right -- the most dangerous failure available here.
  3. It refuses to guess. The generic backend's drive axis comes from the config,
     so without a config the checked axis and the run axis can silently differ.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

pytest.importorskip("mujoco")

from tools.diagnostics.pendulum_experiment import (  # noqa: E402
    ASSETS,
    POSES,
    axis_alignment,
    backend_for,
    check_axis_or_die,
    resolve_pose,
)

ARM_Q0 = POSES["arm_q0"]
DEFAULT = ASSETS["default"]


# ============ 1. REFUSE AXES THE POLE CANNOT FEEL ========================

def test_refuses_axis_along_the_hinge():
    """tool Z IS the hinge at ARM_Q0: kappa = 0, nothing this run did would mean
    anything."""
    with pytest.raises(SystemExit, match="ALONG the hinge"):
        check_axis_or_die(DEFAULT, ARM_Q0, "tool-z", force=False)


def test_refuses_the_axis_that_actually_broke_a_run():
    """tool X is near-vertical here. It scores kappa = 1.0 -- indistinguishable
    from the correct axis on that metric alone -- and kappa_hang = 0.127."""
    kap, kh, _, _ = axis_alignment(DEFAULT, ARM_Q0, "tool-x")
    assert kap > 0.999, "guard the premise: kappa alone really does rate this ideal"
    assert kh < 0.2
    with pytest.raises(SystemExit, match="kappa_hang"):
        check_axis_or_die(DEFAULT, ARM_Q0, "tool-x", force=False)


def test_the_weak_axis_refusal_is_overridable():
    """A HOLD task legitimately uses an axis with no pumping authority."""
    check_axis_or_die(DEFAULT, ARM_Q0, "tool-x", force=True)


def test_accepts_the_axis_with_full_authority():
    kap, kh, vec, hinge = axis_alignment(DEFAULT, ARM_Q0, "in-plane-horiz")
    assert kap > 0.999 and kh > 0.999
    assert abs(float(vec @ hinge)) < 1e-2      # perpendicular to the hinge
    assert abs(float(vec[2])) < 1e-9           # and purely horizontal
    check_axis_or_die(DEFAULT, ARM_Q0, "in-plane-horiz", force=False)


def test_world_axes_are_measurably_worse_here():
    """The hinge sits on the -X/+Y diagonal, so NO world axis is a good drive
    direction at this pose -- and world Y is worse than world X."""
    _, kh_x, _, _ = axis_alignment(DEFAULT, ARM_Q0, "world-x")
    _, kh_y, _, _ = axis_alignment(DEFAULT, ARM_Q0, "world-y")
    assert 0.6 < kh_y < kh_x < 0.8


# ============ 2. DISPATCH ONLY WHERE THE BACKEND CAN EXPRESS IT ==========

def test_toolY_backend_only_for_the_pose_and_asset_it_hardcodes():
    assert backend_for("arm_q0", "default", "tool-y") == "toolY"


@pytest.mark.parametrize("pose,asset,drive", [
    ("w2neg90", "realrod", "tool-y"),      # toolY cannot express this pose
    ("arm_q0", "realrod", "tool-y"),       # nor this asset
    ("arm_q0", "default", "in-plane-horiz"),   # nor this axis
])
def test_falls_back_to_generic_rather_than_running_the_wrong_thing(pose, asset, drive):
    assert backend_for(pose, asset, drive) == "generic"


# ============ 3. NEVER GUESS ============================================

def test_named_poses_resolve_and_unknown_ones_are_fatal():
    np.testing.assert_allclose(resolve_pose("arm_q0"), POSES["arm_q0"])
    np.testing.assert_allclose(resolve_pose("w2neg90"), POSES["w2neg90"])
    with pytest.raises(SystemExit, match="unknown --pose"):
        resolve_pose("whatever")


def test_unknown_drive_axis_is_fatal_not_silently_defaulted():
    with pytest.raises(SystemExit, match="unknown --drive-axis"):
        axis_alignment(DEFAULT, ARM_Q0, "diagonal-ish")


def test_the_two_poses_disagree_about_which_asset_is_live():
    """Guards the pairing that AGENTS.md calls the trap: the local-Z-hinge asset
    is live at wrist_2~0 and dead at -90, and the local-X-hinge asset is the
    reverse. Both are 'perpendicular to the hinge' -- only absolute mgr tells
    them apart."""
    import mujoco

    from simulation.ur5e_pendulum_compose import (
        compose_ur5e_pendulum_model,
        derive_pendulum_constants,
    )
    mgr = {}
    for pose in ("arm_q0", "w2neg90"):
        for asset in ("default", "realrod"):
            model = compose_ur5e_pendulum_model(pendulum_xml=str(ASSETS[asset]))
            mgr[(pose, asset)] = derive_pendulum_constants(model, POSES[pose]).mgr_nm
    # Measured, and the two directions are NOT symmetric:
    #   at arm_q0  default/realrod = 0.027870 / 0.003535 = 7.88x  (dead keeps 12.7%)
    #   at w2neg90 realrod/default = 0.027870 / 0.004102 = 6.79x  (dead keeps 14.7%)
    # 6x sits below the smaller of the two with margin, while still being far too
    # tight to pass on a merely-degraded pairing.
    assert mgr[("arm_q0", "default")] > 6 * mgr[("arm_q0", "realrod")]
    assert mgr[("w2neg90", "realrod")] > 6 * mgr[("w2neg90", "default")]


# ============ 4. THE SIGN, WHICH EVERY OTHER CHECK IS BLIND TO ===========

def test_pump_sign_is_pose_dependent_and_goal1_keeps_the_shipped_sign():
    """The energy law's -k_e sign was hardcoded from ONE measurement. It is
    correct at Goal 1's validated configuration and WRONG at ARM_Q0 -- so a
    measured sign leaves the validated flip bit-identical while fixing the pose
    where every swing-up ever run was fighting a damper."""
    from tools.diagnostics.pendulum_experiment import check_pump_sign
    # Each run's ACTUAL drive direction: Goal 1 drives world +X; Goal 2 drives
    # its config's task_rotation row 0, which is the OPPOSITE end of the
    # in-plane line from the render's e_h. Sign matters, so use what runs.
    c_g1 = check_pump_sign(ASSETS["realrod"], POSES["w2neg90"], np.array([1.0, 0.0, 0.0]))
    c_g2 = check_pump_sign(ASSETS["default"], POSES["arm_q0"], np.array([0.716, 0.698, 0.0]))
    assert c_g1 < 0, "Goal 1's validated flip must keep the shipped -k_e sign"
    assert c_g2 > 0, "ARM_Q0 needs the corrected sign"


def test_reversing_the_drive_axis_reverses_the_pump():
    """Sign-blind checks cannot see this; c0 must."""
    from tools.diagnostics.pendulum_experiment import check_pump_sign
    _, _, u, _ = axis_alignment(ASSETS["default"], POSES["arm_q0"], "in-plane-horiz")
    a = check_pump_sign(ASSETS["default"], POSES["arm_q0"], u)
    b = check_pump_sign(ASSETS["default"], POSES["arm_q0"], -np.asarray(u))
    assert a * b < 0 and abs(abs(a) - abs(b)) < 1e-12
