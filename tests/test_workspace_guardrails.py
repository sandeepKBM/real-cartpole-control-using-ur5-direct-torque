from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from hardware.ros_topics import AsyncGuardrailPublisher, GuardrailStatusSample
from simulation.workspace_guardrails import (
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


def test_guardrail_publisher_queue_is_nonblocking() -> None:
    pub = AsyncGuardrailPublisher(queue_size=1)
    pub._enabled = True  # type: ignore[attr-defined]
    sample = GuardrailStatusSample(
        stamp_ns=123,
        state="inside",
        frame="mujoco_world",
        margin_m=0.02,
        message="ok",
    )
    assert pub.submit(sample) is True
    assert pub.submit(sample) is True
    assert pub.dropped_samples == 1
