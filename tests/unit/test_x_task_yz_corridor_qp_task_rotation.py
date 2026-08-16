"""Explicit constant task basis (``task_rotation``) on the reduced-task QP.

WHY THIS EXISTS. The drive axis is hardwired to row 0
(``parsing.TRANSPORT_AXIS_ROW``), so whatever basis the task rows live in
decides what row 0 physically IS. Measured at ARM_Q0 2026-08-16, neither named
frame can name the axis this pose needs:

  * ``world`` -- shoulder_pan sits at -135.72 deg, so the arm's reachable
    vertical plane is 44.28 deg off world X. With pan frozen, world-X motion is
    not producible at all; in-plane motion gives dy/dx = 0.9751 by geometry.
  * ``tool``  -- row 0 becomes tool X, which here is near-vertical
    [-0.094 -0.085 +0.992]. Vertical pivot acceleration exerts ZERO hinge
    torque at the hanging equilibrium, so a swing-up driving it has no
    authority over the pendulum: measured, it tripped the Z corridor in 0.134 s
    with the rod tip 4 mm off the floor.

Ordered by what breaks worst if wrong:

  1. Unset is bit-identical to before the field existed. Every shipped config
     leaves it unset and live runs must be unaffected.
  2. The basis is validated by the SAME checker the drift monitor uses. A
     non-orthonormal basis rescales measured drift and therefore silently moves
     the effective guard threshold -- the failure would appear as a guard that
     no longer trips when it should.
  3. It genuinely re-aims row 0, and it is CONSTANT -- the corridor row must
     not rotate underneath a barrier whose bounds were captured at reset.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from controller_core.x_task_yz_corridor_qp import (  # noqa: E402
    XTaskYZCorridorQPConfig,
    XTaskYZCorridorQPController,
)
from tests.unit.test_x_task_yz_corridor_qp import (  # noqa: E402
    make_state,
    make_config,
    _yaml_section,
)


def _rz(deg: float) -> list[list[float]]:
    """Rotation about world Z -- the ARM_Q0 case is exactly this, by 44.28 deg."""
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]


def _run(cfg):
    c = XTaskYZCorridorQPController(cfg)
    c.reset_from_state(make_state())
    return c.compute(make_state())


# ================== 1. UNSET PATH MUST BE UNTOUCHED ======================

def test_defaults_to_unset():
    assert XTaskYZCorridorQPConfig().task_rotation is None


def test_unset_is_bit_identical_to_before_the_field_existed():
    a = _run(make_config())
    b = _run(make_config(task_rotation=None))
    np.testing.assert_array_equal(a.tau, b.tau)


def test_identity_rotation_is_numerically_equal_to_world():
    """Identity must not CHANGE anything -- but it deliberately takes the
    rotated code path, so equality is numerical rather than bit-exact (the
    world path skips the matmul entirely; see _task_frames' docstring)."""
    w = _run(make_config())
    i = _run(make_config(task_rotation=tuple(tuple(r) for r in np.eye(3))))
    np.testing.assert_allclose(i.tau, w.tau, rtol=1e-12, atol=1e-12)


# ================== 2. VALIDATION IS SHARED WITH THE GUARD ===============

def test_parser_accepts_and_round_trips_a_rotation():
    c = XTaskYZCorridorQPConfig.from_controller_yaml_section(
        _yaml_section(task_rotation=_rz(44.28))
    )
    got = np.asarray(c.task_rotation, dtype=float)
    np.testing.assert_allclose(got, np.asarray(_rz(44.28)), rtol=0, atol=1e-12)


@pytest.mark.parametrize("bad", [
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],                      # wrong shape
    [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],     # not orthonormal (scaled)
    [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],     # rank deficient
    [[float("nan"), 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
])
def test_parser_rejects_a_basis_that_would_rescale_drift(bad):
    with pytest.raises(ValueError, match="task_rotation"):
        XTaskYZCorridorQPConfig.from_controller_yaml_section(_yaml_section(task_rotation=bad))


def test_rotation_and_tool_frame_are_mutually_exclusive():
    """Two different answers to 'what basis are the task rows in'. Silently
    letting one win would make the active frame unreadable from the config."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        XTaskYZCorridorQPConfig.from_controller_yaml_section(
            _yaml_section(task_rotation=_rz(44.28), task_frame="tool")
        )


# ================== 3. IT ACTUALLY RE-AIMS ROW 0, CONSTANTLY =============

def test_rotation_changes_the_commanded_torque():
    a = _run(make_config())
    b = _run(make_config(task_rotation=tuple(tuple(r) for r in np.asarray(_rz(44.28)))))
    assert not np.allclose(a.tau, b.tau)


def test_basis_is_constant_across_cycles():
    """Unlike task_frame: 'tool', this basis must NOT follow the tool -- the
    corridor is an HOCBF whose bounds were captured at reset, so a rotating
    barrier direction would move the constraint underneath its own bounds."""
    cfg = make_config(task_rotation=tuple(tuple(r) for r in np.asarray(_rz(44.28))))
    c = XTaskYZCorridorQPController(cfg)
    c.reset_from_state(make_state())
    # A different orientation on a later cycle must not change the basis.
    r_a, c_a = c._task_frames(np.array([1.0, 0.0, 0.0, 0.0]))
    r_b, c_b = c._task_frames(np.array([0.0, 1.0, 0.0, 0.0]))
    np.testing.assert_array_equal(r_a, r_b)
    np.testing.assert_array_equal(c_a, c_b)
    # task rows and corridor row share one basis: nothing rotates, so the
    # frozen-vs-live distinction task_frame_update exists for cannot arise.
    np.testing.assert_array_equal(r_a, c_a)


def test_rotation_wins_over_world_frame_default():
    cfg = make_config(task_rotation=tuple(tuple(r) for r in np.asarray(_rz(44.28))))
    c = XTaskYZCorridorQPController(cfg)
    c.reset_from_state(make_state())
    r_task, _ = c._task_frames(np.array([1.0, 0.0, 0.0, 0.0]))
    assert r_task is not None
    np.testing.assert_allclose(r_task, np.asarray(_rz(44.28)), rtol=0, atol=1e-12)
