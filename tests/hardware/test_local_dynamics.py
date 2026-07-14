"""MuJoCo local dynamics parity for hardware fast path."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

mujoco = pytest.importorskip("mujoco")

from hardware.local_dynamics import DEFAULT_SCENE_XML, LocalMujocoDynamics  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENE_XML = REPO_ROOT / "assets" / "ur5e_torque" / "scene.xml"


@pytest.fixture(scope="module")
def engines():
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    local = LocalMujocoDynamics(scene_xml=DEFAULT_SCENE_XML)
    return model, data, site_id, local


def _mujoco_jacobian_mass(model, data, site_id, q):
    data.qpos[:6] = q
    mujoco.mj_forward(model, data)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    jacobian = np.vstack([jacp[:, :6], jacr[:, :6]])
    mass_full = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, data, mass_full)
    return jacobian, mass_full[:6, :6]


@pytest.mark.hardware
def test_local_jacobian_and_mass_match_mujoco(engines) -> None:
    model, data, site_id, local = engines
    rng = np.random.default_rng(0)
    lo = model.jnt_range[:, 0].copy()
    hi = model.jnt_range[:, 1].copy()
    unlimited = lo >= hi
    lo[unlimited], hi[unlimited] = -np.pi, np.pi

    for _ in range(40):
        q = rng.uniform(lo, hi)
        J_mj, M_mj = _mujoco_jacobian_mass(model, data, site_id, q)
        J_loc, M_loc = local.jacobian_and_mass_matrix(q)
        np.testing.assert_allclose(J_loc, J_mj, rtol=0, atol=1e-10)
        np.testing.assert_allclose(M_loc, M_mj, rtol=0, atol=1e-8)


@pytest.mark.hardware
def test_normalize_dynamics_source() -> None:
    from hardware.local_dynamics import normalize_dynamics_source

    assert normalize_dynamics_source("local") == "local"
    assert normalize_dynamics_source("RTDE") == "rtde"
    with pytest.raises(ValueError):
        normalize_dynamics_source("playback")
