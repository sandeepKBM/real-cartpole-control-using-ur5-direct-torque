"""Friction feedforward on the corridor QP.

These fields were INHERITED from CartesianImpedanceConfig all along and parsed
by its own parser, but TorqueTaskQPConfig forwards only a fixed subset, so
setting `friction_feedforward: true` in a corridor-QP config produced False --
a silent no-op, the same trap this file's config module already documents for
manipulability_cbf_* and the lambda-shaping trio. So the first test here is that
the flag SURVIVES parsing, and the rest assert the term reaches the torque.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import yaml

from controller_core.x_task_yz_corridor_qp.config import XTaskYZCorridorQPConfig
from controller_core.x_task_yz_corridor_qp.controller import XTaskYZCorridorQPController
from tests.unit.test_x_task_yz_corridor_qp import make_state

PLAIN = "config/ur5e_mujoco_torque_x_task_yz_corridor_qp_inplane_drive.yaml"
FF = "config/ur5e_mujoco_torque_x_task_yz_corridor_qp_inplane_frictionff.yaml"
JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
          "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
QD = [0.3, -0.2, 0.4, 0.1, -0.05, 0.02]


def _cfg(path, **over):
    c = copy.deepcopy(yaml.safe_load(open(path))["controller"])
    c["manipulability_cbf"] = False          # needs a jacobian_fn; not under test
    c.update(over)
    return XTaskYZCorridorQPConfig.from_controller_yaml_section(c)


def _bias(path, qd=QD):
    cfg = _cfg(path)
    k = XTaskYZCorridorQPController(cfg)
    st = make_state(qd=qd)
    k.reset_from_state(st)
    return np.asarray(k.compute(copy.deepcopy(st)).tau_hold, dtype=np.float64)


def test_flag_survives_the_corridor_qp_parser():
    """The silent no-op this whole feature tripped over first."""
    assert _cfg(PLAIN).friction_feedforward is False
    assert _cfg(FF).friction_feedforward is True


def test_shipped_config_uses_the_measured_model_friction_scaled():
    cfg = _cfg(FF)
    measured = np.array([5.0, 6.1, 5.0, 1.0, 1.0, 1.0])
    np.testing.assert_allclose(
        np.asarray(cfg.friction_ff_coulomb_nm, dtype=np.float64), 0.80 * measured, rtol=1e-9)
    # viscous is deliberately zero here, unlike the impedance lane's 0.4/0.15:
    # the ablation that motivated this settled cleanly with the model's viscous
    # damping intact, and viscous damping is passive and stabilising.
    np.testing.assert_allclose(np.asarray(cfg.friction_ff_viscous, dtype=np.float64), 0.0)


def test_shipped_coulomb_matches_the_compiled_model_up_to_the_scale():
    """The base values must come from the plant, not from prose -- an early
    version used 5.0 for shoulder_lift where the model says 6.1."""
    mujoco = pytest.importorskip("mujoco")
    from simulation.ur5e_pendulum_compose import compose_ur5e_pendulum_model

    m = compose_ur5e_pendulum_model(
        pendulum_xml="assets/ur5e_pendulum/pendulum_attachment_realrod.xml")
    np.testing.assert_allclose(
        np.asarray(_cfg(FF).friction_ff_coulomb_nm, dtype=np.float64),
        0.80 * np.asarray(m.dof_frictionloss[:6], dtype=np.float64), rtol=1e-9)


def test_feedforward_enters_the_bias_with_the_expected_value_and_sign():
    delta = _bias(FF) - _bias(PLAIN)
    cfg = _cfg(FF)
    expected = (np.asarray(cfg.friction_ff_coulomb_nm, dtype=np.float64)
                * np.tanh(np.asarray(QD) / cfg.friction_ff_qd_deadband))
    np.testing.assert_allclose(delta, expected, atol=1e-9)
    # Friction opposes motion, so CANCELLING it must push ALONG qd.
    nz = np.abs(delta) > 1e-6
    assert np.all(np.sign(delta[nz]) == np.sign(np.asarray(QD))[nz])


def test_zero_velocity_is_a_no_op():
    """tanh(0) = 0 -- the term must not add a standing offset at rest."""
    np.testing.assert_allclose(_bias(FF, qd=np.zeros(6)), _bias(PLAIN, qd=np.zeros(6)), atol=0.0)


def test_disabled_config_is_bit_identical_to_the_unpatched_path():
    a = _bias(PLAIN)
    b = _bias(PLAIN)
    np.testing.assert_allclose(a, b, atol=0.0)
    assert _cfg(PLAIN).friction_feedforward is False


@pytest.mark.parametrize("bad,match", [
    ({"friction_ff_coulomb_nm": {n: -1.0 for n in JOINTS}}, "must be >= 0"),
    ({"friction_ff_viscous": {n: -0.1 for n in JOINTS}}, "must be >= 0"),
    ({"friction_ff_qd_deadband": 0.0}, "must be > 0"),
    ({"friction_ff_qd_deadband": -0.05}, "must be > 0"),
])
def test_invalid_values_are_refused(bad, match):
    # A negative coulomb entry adds friction-shaped torque ALONG the motion,
    # i.e. negative damping; a zero deadband divides by zero.
    with pytest.raises(ValueError, match=match):
        _cfg(FF, **bad)


def test_array_may_be_given_as_a_joint_keyed_mapping_like_the_impedance_lane():
    cfg = _cfg(FF, friction_ff_coulomb_nm={n: 2.0 for n in JOINTS})
    np.testing.assert_allclose(
        np.asarray(cfg.friction_ff_coulomb_nm, dtype=np.float64), 2.0)
