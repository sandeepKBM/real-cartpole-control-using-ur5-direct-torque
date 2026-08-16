"""Task-frame / multi-axis drift checking in ImpedanceSafetyMonitor.

Two properties dominate here and are tested first, because getting either
wrong silently weakens a guard that is shared with the real-hardware lane:

  1. The DEFAULT path (no task_rotation, no tracked_axes) is unchanged.
  2. The guard still FIRES on genuine off-task drift, at the same magnitude.

Everything else is validation of the opt-in arguments.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from controller_core.safety import (  # noqa: E402
    ImpedanceSafetyConfig,
    ImpedanceSafetyMonitor,
    validated_task_rotation,
    validated_tracked_axes,
)

TOL = 0.03  # max_abs_orthogonal_drift_m used throughout


def _cfg() -> ImpedanceSafetyConfig:
    return ImpedanceSafetyConfig(max_abs_orthogonal_drift_m=TOL)


def _state(ee: np.ndarray) -> dict:
    """Minimal state dict; q/qd benign so only the drift term can trip."""
    return {
        "q": np.zeros(6),
        "qd": np.zeros(6),
        "ee_pos": np.asarray(ee, dtype=np.float64),
    }


def _check(mon: ImpedanceSafetyMonitor, ee, **kw):
    return mon.check(state=_state(ee), orientation_error_norm=0.0, **kw)


# --- tool frame at ARM_Q0: columns are the tool axes in world -------------
# Permuted so the PUMP direction (tool Y) sits in column 0 = move_axis.
TOOL_Y = np.array([0.7103, 0.6924, 0.1268])
TOOL_X = np.array([-0.0941, -0.0851, 0.9919])
TOOL_Z = np.array([-0.6976, 0.7164, -0.0047])


def _tool_basis() -> np.ndarray:
    """[pump | up | hinge] with columns re-orthonormalized to float tolerance."""
    b = np.column_stack([TOOL_Y, TOOL_X, TOOL_Z])
    q, r = np.linalg.qr(b)
    return q * np.sign(np.diag(r))  # keep original column directions


# ========================= 1. DEFAULT PATH UNCHANGED =====================

def test_default_path_matches_world_axis_behavior_exactly():
    """No task_rotation, no tracked_axes -> historical world-frame result."""
    mon = ImpedanceSafetyMonitor(_cfg())
    mon.set_initial_position(np.zeros(3), move_axis=0)
    # Motion purely along the move axis: never a drift fault, any magnitude.
    assert _check(mon, [10.0, 0.0, 0.0]).ok
    # Motion on a non-move world axis trips at exactly the threshold.
    assert _check(mon, [0.0, TOL * 0.99, 0.0]).ok
    assert not _check(mon, [0.0, TOL * 1.01, 0.0]).ok
    assert not _check(mon, [0.0, 0.0, TOL * 1.01]).ok


def test_default_reason_strings_are_world_frame():
    mon = ImpedanceSafetyMonitor(_cfg())
    mon.set_initial_position(np.zeros(3), move_axis=0)
    st = _check(mon, [0.0, 0.10, 0.0])
    assert not st.ok
    assert "|Y-Y0|" in st.reason  # world naming, not "task"


# ========================= 2. GUARD STILL FIRES ==========================

def test_genuine_off_task_drift_still_trips_at_same_magnitude():
    """Motion along the hinge axis is off-task and must still be caught."""
    R = _tool_basis()
    mon = ImpedanceSafetyMonitor(_cfg())
    mon.set_initial_position(np.zeros(3), move_axis=0, task_rotation=R,
                             tracked_axes=[0, 1])
    hinge = R[:, 2]
    assert _check(mon, hinge * (TOL * 0.99)).ok
    st = _check(mon, hinge * (TOL * 1.01))
    assert not st.ok, "off-task drift along the hinge axis must still trip"
    assert "task" in st.reason


def test_tracked_axes_does_not_loosen_the_untracked_axis():
    """Exempting more axes must not raise the threshold on what remains."""
    R = _tool_basis()
    one = ImpedanceSafetyMonitor(_cfg())
    one.set_initial_position(np.zeros(3), move_axis=0, task_rotation=R)
    two = ImpedanceSafetyMonitor(_cfg())
    two.set_initial_position(np.zeros(3), move_axis=0, task_rotation=R,
                             tracked_axes=[0, 1])
    off = R[:, 2] * (TOL * 1.01)
    assert not one.check(state=_state(off), orientation_error_norm=0.0).ok
    assert not two.check(state=_state(off), orientation_error_norm=0.0).ok


# ================= 3. THE BUG THIS CHANGE EXISTS TO FIX ==================

def test_second_tracked_axis_is_not_charged_as_drift():
    """A 2-axis task must not have its own commanded motion flagged.

    Single-axis exemption charges intended tool-X travel as drift; declaring
    both tracked axes fixes it, while the hinge axis stays guarded.
    """
    R = _tool_basis()
    big = R[:, 1] * 0.25  # 25 cm of INTENDED "up" travel, >> the 3 cm tol

    single = ImpedanceSafetyMonitor(_cfg())
    single.set_initial_position(np.zeros(3), move_axis=0, task_rotation=R)
    assert not single.check(state=_state(big), orientation_error_norm=0.0).ok

    both = ImpedanceSafetyMonitor(_cfg())
    both.set_initial_position(np.zeros(3), move_axis=0, task_rotation=R,
                              tracked_axes=[0, 1])
    assert both.check(state=_state(big), orientation_error_norm=0.0).ok


def test_world_frame_task_recovers_full_travel_along_tool_y():
    """The concrete motivating case: world-frame guard caps tool-Y travel.

    0.7039 of every metre along tool Y lands in the world-orthogonal
    components, so a pure-tool-Y move trips the world-frame guard well before
    it trips the task-frame one.
    """
    R = _tool_basis()
    travel = R[:, 0] * 0.20  # 20 cm along the pump axis

    world = ImpedanceSafetyMonitor(_cfg())
    world.set_initial_position(np.zeros(3), move_axis=0)
    assert not world.check(state=_state(travel), orientation_error_norm=0.0).ok

    task = ImpedanceSafetyMonitor(_cfg())
    task.set_initial_position(np.zeros(3), move_axis=0, task_rotation=R,
                              tracked_axes=[0, 1])
    assert task.check(state=_state(travel), orientation_error_norm=0.0).ok


# ========================= 4. VALIDATION =================================

def test_tracked_axes_cannot_disable_the_guard():
    with pytest.raises(ValueError, match="silently disables the guard"):
        validated_tracked_axes([0, 1, 2], move_axis=0)


def test_tracked_axes_must_contain_move_axis():
    with pytest.raises(ValueError, match="does not contain move_axis"):
        validated_tracked_axes([1, 2], move_axis=0)


def test_tracked_axes_rejects_out_of_range():
    with pytest.raises(ValueError, match="must be in 0..2"):
        validated_tracked_axes([0, 3], move_axis=0)


def test_tracked_axes_default_is_move_axis_only():
    assert validated_tracked_axes(None, move_axis=2) == frozenset({2})


def test_non_orthonormal_rotation_rejected():
    with pytest.raises(ValueError, match="orthonormal"):
        validated_task_rotation(np.diag([1.0, 1.0, 2.0]))


def test_reflection_accepted_as_valid_basis():
    """A column permutation (det = -1) preserves lengths and is legitimate."""
    R = np.column_stack([[0, 1, 0], [1, 0, 0], [0, 0, 1]]).astype(float)
    assert np.isclose(np.linalg.det(R), -1.0)
    validated_task_rotation(R)  # must not raise


def test_rotation_shape_and_nan_rejected():
    with pytest.raises(ValueError, match="3x3"):
        validated_task_rotation(np.eye(2))
    bad = np.eye(3).copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN/Inf"):
        validated_task_rotation(bad)


def test_reset_clears_task_frame_state():
    R = _tool_basis()
    mon = ImpedanceSafetyMonitor(_cfg())
    mon.set_initial_position(np.zeros(3), move_axis=0, task_rotation=R,
                             tracked_axes=[0, 1])
    assert mon.task_rotation is not None
    mon.reset()
    assert mon.task_rotation is None
    # After reset the default path applies again.
    mon.set_initial_position(np.zeros(3), move_axis=0)
    assert not _check(mon, [0.0, 0.10, 0.0]).ok
