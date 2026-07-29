from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from simulation.workspace_guardrails import (
    BoundarySpec,
    _quat_wxyz_to_rot,
    boundary_summary,
    check_point,
    check_trajectory,
    load_guardrail_config,
    overlay_guardrails_on_frame,
)


def test_guardrail_config_loads_and_reports_boundaries() -> None:
    config = load_guardrail_config()
    assert config.frame == "mujoco_world"
    names = {boundary.name for boundary in config.boundaries}
    assert {"floor", "wall", "tools_side_obstacle", "desk_pc_side_obstacle"}.issubset(names)
    summary = boundary_summary(config)
    assert summary["frame"] == "mujoco_world"
    assert summary["units"]["position"] == "m"
    assert summary["units"]["angle"] == "rad"


def test_point_inside_near_boundary_and_outside() -> None:
    config = load_guardrail_config()
    wall = next(boundary for boundary in config.boundaries if boundary.name == "wall")
    normal = _quat_wxyz_to_rot(wall.quaternion())[:, 2]
    inside_point = wall.position() + normal * 0.01
    outside_point = wall.position() - normal * 0.01

    inside = check_point(inside_point, config, frame=config.frame, margin_m=0.0)
    near = check_point(inside_point, config, frame=config.frame, margin_m=0.02)
    outside = check_point(outside_point, config, frame=config.frame, margin_m=0.0)

    assert inside.state == "inside"
    assert near.state == "near_boundary"
    assert near.boundary_name == "wall"
    assert outside.state == "outside"
    assert outside.boundary_name == "wall"


def test_inactive_placeholder_boundary_does_not_block_inside_verdict() -> None:
    """Regression test (2026-07-29 bug audit): a single ``active: false``
    placeholder boundary used to make ``check_point`` return ``"unknown"``
    for every point forever after, even one genuinely inside all the
    *active* boundaries -- because ``_combine_assessments`` treated an
    inactive boundary's per-boundary ``state="unknown"`` the same as a
    genuinely ambiguous (e.g. unimplemented-primitive) one. Verifies both
    that the inactive boundary itself is still reported (state="unknown",
    active=False, in the assessments list) and that it no longer masks the
    "inside" verdict for the boundaries that actually matter.
    """
    config = copy.deepcopy(load_guardrail_config())
    wall = next(boundary for boundary in config.boundaries if boundary.name == "wall")
    normal = _quat_wxyz_to_rot(wall.quaternion())[:, 2]
    inside_point = wall.position() + normal * 0.01

    baseline = check_point(inside_point, config, frame=config.frame, margin_m=0.0)
    assert baseline.state == "inside"

    placeholder = BoundarySpec(
        name="placeholder_unresolved",
        primitive="plane",
        frame=config.frame,
        active=False,
        unresolved_reason="not yet measured",
    )
    config.boundaries.append(placeholder)

    decision = check_point(inside_point, config, frame=config.frame, margin_m=0.0)
    assert decision.state == "inside", decision.message
    placeholder_assessment = next(a for a in decision.assessments if a.name == "placeholder_unresolved")
    assert placeholder_assessment.state == "unknown"
    assert placeholder_assessment.active is False


def test_unknown_frame_is_conservative() -> None:
    config = load_guardrail_config()
    decision = check_point([0.0, 0.0, 0.0], config, frame="not_a_frame")
    assert decision.state == "unknown"
    assert "not compatible" in decision.message


def test_trajectory_violation_is_caught_and_overlay_renders() -> None:
    config = load_guardrail_config()
    wall = next(boundary for boundary in config.boundaries if boundary.name == "wall")
    normal = _quat_wxyz_to_rot(wall.quaternion())[:, 2]
    inside_point = wall.position() + normal * 0.01
    outside_point = wall.position() - normal * 0.01
    trajectory = np.vstack([inside_point, outside_point])

    decision = check_trajectory(trajectory, config, frame=config.frame, margin_m=0.0)
    assert decision.state == "outside"
    assert decision.boundary_name == "wall"

    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    overlay = overlay_guardrails_on_frame(
        frame,
        config,
        trajectory_xyz=trajectory,
        current_xyz=outside_point,
        desired_xyz=inside_point,
        decision=decision,
        show_labels=True,
    )
    assert overlay.shape == frame.shape


def test_guardrail_overlay_corner_changes_render_location() -> None:
    config = load_guardrail_config()
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    overlay_top_left = overlay_guardrails_on_frame(
        frame,
        config,
        current_xyz=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        desired_xyz=np.array([0.1, 0.1, 0.0], dtype=np.float64),
        inset_corner="top-left",
    )
    overlay_bottom_right = overlay_guardrails_on_frame(
        frame,
        config,
        current_xyz=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        desired_xyz=np.array([0.1, 0.1, 0.0], dtype=np.float64),
        inset_corner="bottom-right",
    )
    assert overlay_top_left.shape == frame.shape
    assert overlay_bottom_right.shape == frame.shape
    assert not np.array_equal(overlay_top_left, overlay_bottom_right)
