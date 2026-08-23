"""Per-joint posture weighting on the corridor-QP controller.

Exists because kp_posture/kd_posture are single scalars, so holding ONE joint
harder previously meant stiffening all six. The alternative -- excluding the
joint from the task -- costs a rank: with shoulder_pan AND wrist_2 both
excluded the 4x4 (X + 3 orientation) task matrix drops to rank 3, leaving a
rotational direction uncontrollable. See the config header for the measurements.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import yaml

from controller_core.x_task_yz_corridor_qp.config import XTaskYZCorridorQPConfig

BALANCE = "config/ur5e_mujoco_torque_x_task_yz_corridor_qp_orientation_cbf_balance.yaml"
WRIST2 = "config/ur5e_mujoco_torque_x_task_yz_corridor_qp_wrist2_posture.yaml"


def ctrl(path):
    return copy.deepcopy(yaml.safe_load(open(path))["controller"])


def test_absent_field_defaults_to_none_not_ones():
    """None must mean 'skip the multiply entirely', so existing configs are
    bit-for-bit unchanged rather than multiplied by an array of ones."""
    cfg = XTaskYZCorridorQPConfig.from_controller_yaml_section(ctrl(BALANCE))
    assert cfg.posture_joint_weights is None


def test_shipped_wrist2_config_declares_the_weight():
    cfg = XTaskYZCorridorQPConfig.from_controller_yaml_section(ctrl(WRIST2))
    assert cfg.posture_joint_weights == (1.0, 1.0, 1.0, 1.0, 60.0, 1.0)


def test_wrist2_config_changes_only_the_weight():
    """A single-variable derivative: every other field must match the parent,
    or a comparison between the two measures more than one thing."""
    a = XTaskYZCorridorQPConfig.from_controller_yaml_section(ctrl(BALANCE))
    b = XTaskYZCorridorQPConfig.from_controller_yaml_section(ctrl(WRIST2))
    differing = [f for f in vars(a) if not np.array_equal(
        np.asarray(getattr(a, f), dtype=object), np.asarray(getattr(b, f), dtype=object))]
    assert differing == ["posture_joint_weights"], differing


def test_rank_is_preserved_unlike_exclusion():
    """The whole point: weighting keeps the joint in the task."""
    b = XTaskYZCorridorQPConfig.from_controller_yaml_section(ctrl(WRIST2))
    assert 4 not in b.task_excluded_joints
    assert b.task_excluded_joints == (0,)


@pytest.mark.parametrize("bad", [[1, 1, 1, 1, 1], [1] * 7, [1, 1, 1, 1, -2, 1],
                                 [1, 1, 1, 1, float("nan"), 1],
                                 [1, 1, 1, 1, float("inf"), 1]])
def test_malformed_weights_are_rejected(bad):
    """A silently-ignored malformed weight would look exactly like a weight
    that had no effect."""
    with pytest.raises(ValueError):
        XTaskYZCorridorQPConfig.from_controller_yaml_section(
            dict(ctrl(BALANCE), posture_joint_weights=bad))


def test_zero_weight_is_allowed():
    """0 means 'no posture hold on this joint' -- a legitimate request, and
    distinct from the joint being excluded from the task."""
    cfg = XTaskYZCorridorQPConfig.from_controller_yaml_section(
        dict(ctrl(BALANCE), posture_joint_weights=[1, 1, 1, 1, 0, 1]))
    assert cfg.posture_joint_weights[4] == 0.0
