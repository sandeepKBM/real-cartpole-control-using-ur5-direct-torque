"""Kinematic regression tests for the "hanging"/elbow-down pose family
(hardware/poses.py::HANGING_ORIGIN_Q/HANGING_LOWER_Q/q_for_hanging_height_alpha, added
2026-08-01), mirroring how tests/unit/test_split_base_wrist_task.py locks down a cond(J)
claim for that fix.

The whole point of this pose family is avoiding the wrist_2=0 kinematic singularity that
ACTIVE_ORIGIN_Q/LOWER_B_Q/q_for_height_alpha sit at across their entire range (cond(full
6x6 J) measured 1e16-2.5e17 -- see docs/status/hanging_pose_transport_family_2026-08-01.md).
These tests assert that claim numerically, directly from the real MJCF, so a future change
to the pose constants or the model can't silently reintroduce the singularity without a
test failure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hardware.poses import HANGING_LOWER_Q, HANGING_ORIGIN_Q, q_for_hanging_height_alpha  # noqa: E402
from simulation.ur5e_mujoco_torque import load_model  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENE_PATH = REPO_ROOT / "assets" / "ur5e_torque" / "scene.xml"

# Comfortably above the measured max (15.41 at HANGING_ORIGIN_Q) but many orders of
# magnitude below the old family's measured floor (1e16) -- a real, meaningful bound, not
# a rubber-stamp threshold.
COND_BOUND = 100.0


def _full_jacobian(model: mujoco.MjModel, data: mujoco.MjData, site_id: int, joint_ids: list[int], q: np.ndarray) -> np.ndarray:
    data.qpos[:] = 0.0
    for dof_id, val in zip(joint_ids, q):
        data.qpos[dof_id] = val
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    return np.vstack([jacp[:, joint_ids], jacr[:, joint_ids]])


def test_hanging_family_cond_j_stays_well_conditioned_across_full_range() -> None:
    model, data, site_id, joint_ids, _ = load_model(SCENE_PATH)
    conds = []
    for alpha in np.linspace(0.0, 1.0, 21):
        q = q_for_hanging_height_alpha(float(alpha))
        J = _full_jacobian(model, data, site_id, joint_ids, q)
        cond = np.linalg.cond(J)
        assert np.isfinite(cond), f"cond(J) not finite at alpha={alpha}"
        assert cond < COND_BOUND, f"cond(J)={cond} exceeds {COND_BOUND} at alpha={alpha}"
        conds.append(cond)
    # Sanity: the whole point of this family is being nowhere near the old family's
    # 1e16-2.5e17 floor.
    assert max(conds) < 1.0e6


def test_hanging_family_endpoints_reach_expected_site_height() -> None:
    """Rough Z-height reachability check: the family should span a range comparable to
    the old family's ~0.54-1.08 m (see docs/status/hanging_pose_transport_family_2026-08-01.md).
    """
    model, data, site_id, joint_ids, _ = load_model(SCENE_PATH)

    def site_z(q: np.ndarray) -> float:
        data.qpos[:] = 0.0
        for dof_id, val in zip(joint_ids, q):
            data.qpos[dof_id] = val
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        return float(data.site_xpos[site_id][2])

    z_high = site_z(HANGING_ORIGIN_Q)
    z_low = site_z(HANGING_LOWER_Q)
    assert z_low < z_high
    assert 0.9 < z_high < 1.2
    assert 0.4 < z_low < 0.7


def test_old_family_is_still_singular_unchanged_baseline() -> None:
    """Confirms the motivating claim on the SAME model this session used, and that the old
    family's constants (checked elsewhere) were never touched: ACTIVE_ORIGIN_Q remains at
    the wrist_2=0 singularity. A sanity anchor, not a claim about the hanging family.
    """
    from hardware.poses import ACTIVE_ORIGIN_Q

    model, data, site_id, joint_ids, _ = load_model(SCENE_PATH)
    J = _full_jacobian(model, data, site_id, joint_ids, ACTIVE_ORIGIN_Q)
    cond = np.linalg.cond(J)
    assert cond > 1.0e10
