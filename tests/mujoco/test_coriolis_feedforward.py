"""Adapter-level tests for the P2 Coriolis feedforward term."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mujoco  # noqa: E402
import yaml  # noqa: E402

from simulation.ur5e_mujoco_torque import (  # noqa: E402
    MujocoUR5eTorqueAdapter,
    MujocoUR5eTorqueAdapterConfig,
    build_controller,
    build_mujoco_state,
    load_model,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENE_XML = REPO_ROOT / "assets" / "ur5e_torque" / "scene.xml"

Q_TEST = np.array([0.0, -1.2, 1.6, -1.9, -1.5, 0.0])
QD_TEST = np.array([0.4, -0.6, 0.5, 0.3, -0.2, 0.1])


def _make_adapter(**cfg_overrides):
    model, data, site_id, joint_ids, _ = load_model(SCENE_XML)
    data.qpos[:6] = Q_TEST
    data.qvel[:6] = QD_TEST
    mujoco.mj_forward(model, data)
    cfg = MujocoUR5eTorqueAdapterConfig(
        controller_kind="impedance",
        gravity_mode="gravity_comp",
        **cfg_overrides,
    )
    controller_cfg = yaml.safe_load((REPO_ROOT / "config" / "ur5e_mujoco_torque.yaml").read_text())["controller"]
    controller = build_controller("impedance", controller_cfg)
    adapter = MujocoUR5eTorqueAdapter(
        model=model, site_id=site_id, joint_ids=joint_ids, controller=controller, config=cfg
    )
    state = build_mujoco_state(
        model,
        data,
        site_id=site_id,
        joint_ids=joint_ids,
        time_s=0.0,
        dt_s=float(model.opt.timestep),
        target_x=float(data.site_xpos[site_id][0]),
        gravity_compensation=False,
    )
    return model, data, adapter, state


def _reference_coriolis(model, q, qd):
    scratch = mujoco.MjData(model)
    scratch.qpos[:6] = q
    scratch.qvel[:6] = qd
    mujoco.mj_forward(model, scratch)
    live = scratch.qfrc_bias[:6].copy()
    scratch.qvel[:] = 0.0
    mujoco.mj_forward(model, scratch)
    static = scratch.qfrc_bias[:6].copy()
    return live - static


def test_coriolis_zero_when_disabled():
    _, _, adapter, state = _make_adapter(coriolis_feedforward=False)
    np.testing.assert_allclose(adapter._coriolis_torque(state), np.zeros(6), atol=0.0)


def test_coriolis_mujoco_source_matches_bias_difference():
    model, _, adapter, state = _make_adapter(coriolis_feedforward=True, gravity_source="mujoco_qfrc")
    expected = _reference_coriolis(model, Q_TEST, QD_TEST)
    got = adapter._coriolis_torque(state)
    assert float(np.max(np.abs(expected))) > 1e-3  # test pose actually excites Coriolis
    np.testing.assert_allclose(got, expected, atol=1e-10)


def test_coriolis_pinocchio_source_matches_mujoco():
    pytest.importorskip("pinocchio")
    model, _, adapter, state = _make_adapter(coriolis_feedforward=True, gravity_source="pinocchio")
    expected = _reference_coriolis(model, Q_TEST, QD_TEST)
    got = adapter._coriolis_torque(state)
    np.testing.assert_allclose(got, expected, atol=1e-6)


def test_coriolis_requires_gravity_comp_mode():
    _, _, adapter, state = _make_adapter(coriolis_feedforward=True, gravity_source="mujoco_qfrc")
    adapter.cfg.gravity_mode = "raw"
    np.testing.assert_allclose(adapter._coriolis_torque(state), np.zeros(6), atol=0.0)
