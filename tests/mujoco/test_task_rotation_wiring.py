"""controller.task_rotation must reach the SAFETY MONITOR, not just the controller.

Why this test exists. The rotation was wired at exactly one call site
(pendulum_swingup_energy_shaping.py) while three tools build their adapter
through build_initial_state_and_adapter. The two other tools -- the two-phase
swing-up and the LQR cascade -- could therefore only ever drive world X, and
nothing said so: the config parsed, the controller took the rotation, the runs
completed, and the safety monitor quietly kept resolving drift in world axes.

That combination is the expensive one. At ARM_Q0/wrist_2=-90 the hinge lies
44.6 deg off world X, so 71.3% of world-X travel lands in the guarded world-Y
axis; with the monitor unrotated, on-axis travel is charged against the 0.06 m
lateral-drift budget and the guard caps the run for doing its job. Measured:
the LQR catch trips at dX=0.070/dY=0.059 identically for a_max in
{9.603, 14, 20, 30} -- the guard, not the actuation.

So these tests assert the EFFECT (the monitor holds the basis) rather than the
invocation, and they assert the no-rotation path is untouched.
"""

from __future__ import annotations

import numpy as np
import pytest

from tools.diagnostics.pendulum_swingup_energy_shaping import load_config

ROTATED = "config/ur5e_mujoco_torque_x_task_yz_corridor_qp_inplane_drive.yaml"
UNROTATED = "config/ur5e_mujoco_torque_x_task_yz_corridor_qp_wrist2_posture.yaml"
XML = "assets/ur5e_pendulum/pendulum_attachment_realrod.xml"
ARM_Q = [-2.3688, -2.1801, -1.8838, -0.7962, -1.5707963, 0.0206]


def _adapter_for(config_path):
    """Build through the SHARED path all three pendulum tools use."""
    import mujoco

    from simulation.ur5e_mujoco_torque import build_initial_state_and_adapter
    from simulation.ur5e_pendulum_compose import compose_ur5e_pendulum_model

    cfg = load_config(config_path)
    model = compose_ur5e_pendulum_model(pendulum_xml=XML)
    data = mujoco.MjData(model)
    data.qpos[:6] = np.asarray(ARM_Q, dtype=np.float64)
    mujoco.mj_forward(model, data)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    joint_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        for n in cfg["mujoco"]["joint_order"]
    ]
    _state, adapter = build_initial_state_and_adapter(
        model, data, site_id, joint_ids,
        controller_cfg=cfg["controller"],
        transport_axis_index=0,
        target_x_delta=0.0,
        controller_kind="x_task_yz_corridor_qp",
        force_hold_current_pose=True,
        gravity_mode=cfg["mujoco"]["gravity_mode"],
        gravity_source=cfg["mujoco"].get("gravity_source", "mujoco_qfrc"),
        coriolis_feedforward=bool(cfg["mujoco"].get("coriolis_feedforward", False)),
        torque_limit_scale=1.0,
    )
    return cfg, adapter


def test_monitor_holds_the_rotation_from_the_shared_builder():
    cfg, adapter = _adapter_for(ROTATED)
    got = getattr(adapter.safety_monitor, "task_rotation", None)
    assert got is not None, "monitor is still resolving drift in WORLD axes"
    want = np.asarray(cfg["controller"]["task_rotation"], dtype=np.float64)
    np.testing.assert_allclose(np.asarray(got, dtype=np.float64), want, atol=1e-9)


def test_monitor_basis_is_a_real_rotation_and_matches_the_hinge():
    _cfg, adapter = _adapter_for(ROTATED)
    R = np.asarray(adapter.safety_monitor.task_rotation, dtype=np.float64)
    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-9)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-9)  # rotation, not reflection

    # col0 must be perpendicular to the hinge (all authority) and col1 along it
    # (none). This is the whole point of the basis -- a config that merely
    # parsed but pointed somewhere else would pass every other assertion here.
    import mujoco

    from simulation.ur5e_pendulum_compose import compose_ur5e_pendulum_model
    from tools.diagnostics.pendulum_swingup_energy_shaping import resolve_equilibria

    model = compose_ur5e_pendulum_model(pendulum_xml=XML)
    arm_q = np.asarray(ARM_Q, dtype=np.float64)
    hanging, _inv = resolve_equilibria(model, arm_q)
    data = mujoco.MjData(model)
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "/pendulum_hinge")
    data.qpos[:6] = arm_q
    data.qpos[model.jnt_qposadr[jid]] = hanging
    mujoco.mj_forward(model, data)
    hinge = np.asarray(data.xaxis[jid], dtype=np.float64)
    hinge /= np.linalg.norm(hinge)

    assert abs(R[:, 0] @ hinge) < 1e-4, "drive axis is not perpendicular to the hinge"
    assert abs(R[:, 1] @ hinge) > 0.999, "col1 is not the hinge"


def test_rotated_drive_axis_has_more_hinge_authority_than_world_x():
    """The reason for the rotation, stated as a measurement rather than a claim."""
    from tools.diagnostics.pendulum_swingup_energy_shaping import (
        measure_pivot_coupling, resolve_equilibria)
    from simulation.ur5e_pendulum_compose import compose_ur5e_pendulum_model

    _cfg, adapter = _adapter_for(ROTATED)
    R = np.asarray(adapter.safety_monitor.task_rotation, dtype=np.float64)
    model = compose_ur5e_pendulum_model(pendulum_xml=XML)
    arm_q = np.asarray(ARM_Q, dtype=np.float64)
    hanging, _ = resolve_equilibria(model, arm_q)
    c_drive = abs(measure_pivot_coupling(model, arm_q, hanging, R[:, 0]))
    c_worldx = abs(measure_pivot_coupling(model, arm_q, hanging, np.array([1.0, 0.0, 0.0])))
    assert c_drive > 1.4 * c_worldx
    # and col1, the hinge, must have essentially none
    c_hinge = abs(measure_pivot_coupling(model, arm_q, hanging, R[:, 1]))
    assert c_hinge < 0.01 * c_drive


def test_config_without_task_rotation_is_unchanged():
    """The wiring must be a no-op for every existing config."""
    cfg, adapter = _adapter_for(UNROTATED)
    assert "task_rotation" not in cfg["controller"]
    assert getattr(adapter.safety_monitor, "task_rotation", None) is None


def test_builder_raises_if_the_monitor_did_not_take_the_rotation(monkeypatch):
    """A silent no-op here reproduces the unrotated numbers exactly, which is
    indistinguishable from 'the rotation did not help'. It must be loud."""
    import simulation.ur5e_mujoco_torque as mod

    original = mod.MujocoUR5eTorqueAdapter.configure_task_frame
    monkeypatch.setattr(
        mod.MujocoUR5eTorqueAdapter, "configure_task_frame",
        lambda self, **kw: None,  # swallow it, as a mis-ordered call would
    )
    with pytest.raises(RuntimeError, match="WORLD axes"):
        _adapter_for(ROTATED)
    assert original is not None
