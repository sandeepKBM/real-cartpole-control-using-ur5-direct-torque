"""Hard joint-motion enforcement via a high-order CBF row in the QP.

Distinct from `posture_joint_weights`, which is only a stiffer SPRING: it
resists motion but bounds nothing, so a large enough disturbance moves the
joint as far as the torque budget allows. These rows make a breaching torque
INFEASIBLE.

Implementation reuses `_corridor_rows` verbatim -- for joint j the Jacobian row
is the unit vector e_j, because qdot_j = e_j . qdot exactly -- so these tests
target the wiring and the row algebra's joint-space specialisation, not the
barrier math, which the Cartesian corridor tests already cover.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import yaml

from controller_core.x_task_yz_corridor_qp.config import XTaskYZCorridorQPConfig
from controller_core.x_task_yz_corridor_qp.controller import XTaskYZCorridorQPController

BALANCE = "config/ur5e_mujoco_torque_x_task_yz_corridor_qp_orientation_cbf_balance.yaml"


def ctrl(**over):
    c = copy.deepcopy(yaml.safe_load(open(BALANCE))["controller"])
    c.update(over)
    return c


def test_disabled_by_default():
    cfg = XTaskYZCorridorQPConfig.from_controller_yaml_section(ctrl())
    assert cfg.joint_corridor_enabled is False
    assert cfg.joint_corridor_joints == ()


def test_indices_are_deduped_and_sorted():
    """Determinism: row order must not depend on how the YAML happened to list
    the joints, or two identical configs could produce different QPs."""
    cfg = XTaskYZCorridorQPConfig.from_controller_yaml_section(
        ctrl(joint_corridor_enabled=True, joint_corridor_joints=[4, 1, 4]))
    assert cfg.joint_corridor_joints == (1, 4)


@pytest.mark.parametrize("bad", [[6], [-1], [0, 9]])
def test_out_of_range_indices_rejected(bad):
    with pytest.raises(ValueError):
        XTaskYZCorridorQPConfig.from_controller_yaml_section(
            ctrl(joint_corridor_joints=bad))


def test_row_for_joint_j_uses_the_unit_vector():
    """The joint-space specialisation: j_row = e_j, so `lie` is row j of M^-1.
    If this were wrong the barrier would constrain some other combination of
    joints and still look plausible."""
    m_inv = np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    e4 = np.zeros(6); e4[4] = 1.0
    a_max, b_max, a_min, b_min = XTaskYZCorridorQPController._corridor_rows(
        j_row=e4, m_inv=m_inv, bias=np.zeros(6), qd=np.zeros(6),
        value=0.0, lower=-0.02, upper=0.02, alpha1=20.0, alpha2=20.0)
    assert np.allclose(a_max.reshape(-1), m_inv[4])       # row 4 of M^-1
    assert np.allclose(a_min.reshape(-1), -m_inv[4])
    # symmetric bounds about the current value => symmetric slack
    assert b_max == pytest.approx(b_min)


def test_bounds_bracket_the_rest_position_not_zero():
    """The corridor must be centred on q_rest (where the joint STARTED), not on
    zero -- centring on zero would command the joint to a different pose."""
    e4 = np.zeros(6); e4[4] = 1.0
    q0, hw = -1.5707963, 0.02
    a_max, b_max, a_min, b_min = XTaskYZCorridorQPController._corridor_rows(
        j_row=e4, m_inv=np.eye(6), bias=np.zeros(6), qd=np.zeros(6),
        value=q0, lower=q0 - hw, upper=q0 + hw, alpha1=20.0, alpha2=20.0)
    # at the centre both barriers have equal margin, independent of q0's value
    assert b_max == pytest.approx(b_min)
    assert b_max > 0.0


def test_velocity_tightens_the_barrier_it_moves_toward():
    """hdot enters with opposite sign on the two rows; a joint moving toward
    the upper bound must see its upper row tighten."""
    e4 = np.zeros(6); e4[4] = 1.0
    qd = np.zeros(6); qd[4] = 0.5           # moving toward `upper`
    a1, b_max_mov, a2, b_min_mov = XTaskYZCorridorQPController._corridor_rows(
        j_row=e4, m_inv=np.eye(6), bias=np.zeros(6), qd=qd,
        value=0.0, lower=-0.02, upper=0.02, alpha1=20.0, alpha2=20.0)
    _, b_max_still, _, b_min_still = XTaskYZCorridorQPController._corridor_rows(
        j_row=e4, m_inv=np.eye(6), bias=np.zeros(6), qd=np.zeros(6),
        value=0.0, lower=-0.02, upper=0.02, alpha1=20.0, alpha2=20.0)
    assert b_max_mov < b_max_still     # upper barrier tightened
    assert b_min_mov > b_min_still     # lower barrier relaxed
