"""Tests for simulation/ur5e_pendulum_compose.py -- composing the physical
UR5e-mounted pendulum apparatus (assets/ur5e_pendulum/, see that
directory's own docstring for the real-vs-placeholder dimension
provenance) onto the protected centerpiece UR5e model via
mujoco.MjSpec.attach().
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from simulation.ur5e_pendulum_compose import (  # noqa: E402
    DEFAULT_ARM_XML,
    DEFAULT_PENDULUM_XML,
    compose_ur5e_pendulum_model,
    compose_ur5e_pendulum_spec,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_centerpiece_model_untouched():
    """Composing must never modify assets/ur5e_torque/ur5e_torque.xml on
    disk -- the protected centerpiece model (AGENTS.md sec 2)."""
    before = DEFAULT_ARM_XML.read_bytes()
    compose_ur5e_pendulum_model()
    after = DEFAULT_ARM_XML.read_bytes()
    assert before == after


def test_compose_produces_expected_dof_count():
    model = compose_ur5e_pendulum_model()
    # 6 arm joints + 1 pendulum hinge.
    assert model.nq == 7
    assert model.nv == 7
    joint_names = {model.joint(i).name for i in range(model.njnt)}
    assert joint_names == {
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint", "/pendulum_hinge",
    }


def test_compose_raises_on_missing_attachment_site():
    with pytest.raises(ValueError, match="no site named"):
        compose_ur5e_pendulum_spec(attachment_site="not_a_real_site")


def test_compose_raises_on_missing_files():
    with pytest.raises(FileNotFoundError):
        compose_ur5e_pendulum_spec(arm_xml=REPO_ROOT / "assets" / "does_not_exist.xml")
    with pytest.raises(FileNotFoundError):
        compose_ur5e_pendulum_spec(pendulum_xml=REPO_ROOT / "assets" / "does_not_exist.xml")


def test_isolated_pendulum_settles_at_true_hanging_equilibrium():
    """The pendulum fragment alone (no arm, no attachment-site rotation)
    -- joint-local zero IS world-down here, unlike the arm-attached case
    (see test_composed_pendulum_settles_under_gravity's own note). This
    is the test that actually validates "does this behave like a real
    pendulum," decoupled from any arm-pose-dependent rotation offset."""
    spec = mujoco.MjSpec.from_file(str(DEFAULT_PENDULUM_XML))
    model = spec.compile()
    data = mujoco.MjData(model)
    data.qpos[0] = np.pi / 2  # release from horizontal
    mujoco.mj_forward(model, data)

    angle_hist = []
    for _ in range(3000):
        mujoco.mj_step(model, data)
        angle_hist.append(float(data.qpos[0]))
    angle_hist = np.array(angle_hist)

    # Swings down through the bottom (must pass near 0 at some point, not
    # just monotonically crawl toward it) -- confirms real pendulum dynamics,
    # not an over-damped/locked joint.
    assert np.any(np.abs(angle_hist[:500]) < 0.1), "pendulum never swung near 0 rad early in the trace"
    # Settles hanging straight down.
    assert abs(angle_hist[-1]) < 0.05, f"did not settle near 0 rad, got {angle_hist[-1]:.4f}"
    assert angle_hist[-200:].std() < 0.01, "pendulum still oscillating at end of trace"


def test_composed_pendulum_settles_under_gravity():
    """Arm-attached case: the settled angle is pose-dependent (attachment_
    site's fixed rotation relative to wrist_3_link means joint-local zero
    is NOT world-down once attached to the arm at a given configuration --
    see assets/ur5e_pendulum/pendulum_attachment.xml's docstring) so only
    convergence is checked here, not a specific numeric equilibrium."""
    model = compose_ur5e_pendulum_model()
    pendulum_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "/pendulum_hinge")
    assert pendulum_joint_id >= 0
    qpos_adr = model.jnt_qposadr[pendulum_joint_id]

    data = mujoco.MjData(model)
    arm_q0 = np.array([0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0])
    data.qpos[:6] = arm_q0
    data.qpos[qpos_adr] = np.pi / 2
    mujoco.mj_forward(model, data)

    angle_hist = []
    for _ in range(3000):
        data.qpos[:6] = arm_q0
        data.qvel[:6] = 0
        mujoco.mj_step(model, data)
        angle_hist.append(float(data.qpos[qpos_adr]))
    angle_hist = np.array(angle_hist)

    assert abs(angle_hist[0] - angle_hist[-1]) > 0.05, "pendulum barely moved -- joint may be effectively locked"
    assert angle_hist[-200:].std() < 0.01, "pendulum did not converge/settle"


def test_pendulum_angle_sensor_matches_joint_qpos():
    model = compose_ur5e_pendulum_model()
    pendulum_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "/pendulum_hinge")
    qpos_adr = model.jnt_qposadr[pendulum_joint_id]
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "/pendulum_angle_sensor")
    assert sensor_id >= 0
    sensor_adr = model.sensor_adr[sensor_id]

    data = mujoco.MjData(model)
    data.qpos[qpos_adr] = 0.37
    mujoco.mj_forward(model, data)
    assert data.sensordata[sensor_adr] == pytest.approx(0.37, abs=1e-9)


def test_compose_spec_allows_custom_worldbody_additions():
    """compose_ur5e_pendulum_spec returns an uncompiled MjSpec specifically
    so callers can add scene content (matching assets/ur5e_torque/scene.xml's
    pattern) before compiling -- confirm that actually works."""
    spec = compose_ur5e_pendulum_spec()
    geom = spec.worldbody.add_geom()
    geom.name = "custom_test_marker"
    geom.type = mujoco.mjtGeom.mjGEOM_SPHERE
    geom.size = [0.01, 0, 0]
    model = spec.compile()
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "custom_test_marker") >= 0
