"""MuJoCo-vs-Pinocchio dynamics parity for the custom torque UR5e MJCF.

Skips cleanly when pinocchio is not installed (it lives in the mujoco_ur5e env).
Tolerances were frozen from the observed parity of the shared MJCF: both engines
read the same inertial blocks, so gravity/bias should agree to numerical noise;
the mass matrix differs exactly by MuJoCo's joint armature on the diagonal.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

pin = pytest.importorskip("pinocchio")
import mujoco  # noqa: E402

from controller_core.model_dynamics import DEFAULT_UR5E_MJCF, PinocchioUR5eDynamics  # noqa: E402
from simulation.ur5e_mujoco_torque import expand_mass_matrix  # noqa: E402

N_SAMPLES = 200
GRAVITY_TOL_NM = 1e-8
BIAS_TOL_NM = 1e-6
MASS_TOL = 1e-8
JACOBIAN_TOL = 1e-6

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENE_XML = REPO_ROOT / "assets" / "ur5e_torque" / "scene.xml"


@pytest.fixture(scope="module")
def engines():
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)
    dyn = PinocchioUR5eDynamics(DEFAULT_UR5E_MJCF)
    return model, data, dyn


def _random_states(model, rng, n):
    lo = model.jnt_range[:, 0].copy()
    hi = model.jnt_range[:, 1].copy()
    unlimited = lo >= hi
    lo[unlimited], hi[unlimited] = -np.pi, np.pi
    for _ in range(n):
        q = rng.uniform(lo, hi)
        qd = rng.uniform(-1.5, 1.5, size=model.nv)
        yield q, qd


def _mujoco_bias(model, data, q, qd):
    data.qpos[:] = q
    data.qvel[:] = qd
    mujoco.mj_forward(model, data)
    return data.qfrc_bias.copy()


def _mujoco_mass(model, data, q):
    data.qpos[:] = q
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    return expand_mass_matrix(model, data)


def _mujoco_site_jacobian(model, data, q, site_id):
    data.qpos[:] = q
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    return np.vstack([jacp, jacr])


def test_joint_order_matches(engines):
    _, _, dyn = engines
    assert dyn.nv == 6  # constructor already validates names/order


def test_gravity_parity(engines):
    model, data, dyn = engines
    rng = np.random.default_rng(0)
    worst = 0.0
    for q, _ in _random_states(model, rng, N_SAMPLES):
        g_mj = _mujoco_bias(model, data, q, np.zeros(model.nv))
        g_pin = dyn.gravity(q)
        worst = max(worst, float(np.max(np.abs(g_mj - g_pin))))
    assert worst < GRAVITY_TOL_NM, f"gravity parity worst |delta| = {worst}"


def test_bias_parity_with_velocity(engines):
    model, data, dyn = engines
    rng = np.random.default_rng(1)
    worst = 0.0
    for q, qd in _random_states(model, rng, N_SAMPLES):
        b_mj = _mujoco_bias(model, data, q, qd)
        b_pin = dyn.bias(q, qd)
        worst = max(worst, float(np.max(np.abs(b_mj - b_pin))))
    assert worst < BIAS_TOL_NM, f"bias parity worst |delta| = {worst}"


def test_mass_matrix_parity(engines):
    # Pinocchio's MJCF loader imports joint armature and crba includes it,
    # so the matrices are directly comparable (verified: both report 0.1 per dof).
    model, data, dyn = engines
    rng = np.random.default_rng(2)
    worst = 0.0
    for q, _ in _random_states(model, rng, N_SAMPLES // 4):
        m_mj = _mujoco_mass(model, data, q)
        m_pin = dyn.mass_matrix(q)
        worst = max(worst, float(np.max(np.abs(m_mj - m_pin))))
    assert worst < MASS_TOL, f"mass-matrix parity worst |delta| = {worst}"


def test_jacobian_parity(engines):
    # Pinocchio's buildModelFromMJCF does not apply this MJCF's root body
    # rotation (base's quat="0 0 0 -1") to the frame tree -- invisible to the
    # joint-space parity tests above (that rotation is about Z, same axis as
    # gravity), but not to a world-frame Cartesian Jacobian. See
    # `controller_core.model_dynamics._root_body_quat_wxyz` and
    # docs/status/local_dynamics_speedup_investigation_2026-07-29.md.
    model, data, dyn = engines
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    rng = np.random.default_rng(4)
    worst = 0.0
    for q, _ in _random_states(model, rng, N_SAMPLES):
        J_mj = _mujoco_site_jacobian(model, data, q, site_id)
        J_pin = dyn.jacobian(q)
        worst = max(worst, float(np.max(np.abs(J_mj - J_pin))))
    assert worst < JACOBIAN_TOL, f"jacobian parity worst |delta| = {worst}"


def test_coriolis_is_bias_minus_gravity(engines):
    model, data, dyn = engines
    rng = np.random.default_rng(3)
    for q, qd in _random_states(model, rng, 10):
        c = dyn.coriolis(q, qd)
        np.testing.assert_allclose(c + dyn.gravity(q), dyn.bias(q, qd), atol=1e-12)
        # Coriolis vanishes at zero velocity.
        np.testing.assert_allclose(dyn.coriolis(q, np.zeros(6)), np.zeros(6), atol=1e-12)
