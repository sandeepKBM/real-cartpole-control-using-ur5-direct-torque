"""MuJoCo qfrc_bias gravity feedforward for CoppeliaSim RL baseline."""

from __future__ import annotations

from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
MUJOCO_MENAGERIE = REPO_ROOT / "mujoco_menagerie"
MUJOCO_MENAGERIE_VENDOR = REPO_ROOT / "vendor" / "mujoco_menagerie"
MUJOCO_GRAVITY_SCENE_CANDIDATES = (
    MUJOCO_MENAGERIE / "universal_robots_ur5e" / "scene_ur5e_cartpole.xml",
    MUJOCO_MENAGERIE / "universal_robots_ur5e" / "scene.xml",
    MUJOCO_MENAGERIE / "universal_robots_ur5e" / "ur5e.xml",
    MUJOCO_MENAGERIE_VENDOR / "universal_robots_ur5e" / "scene_ur5e_cartpole.xml",
    MUJOCO_MENAGERIE_VENDOR / "universal_robots_ur5e" / "scene.xml",
    MUJOCO_MENAGERIE_VENDOR / "universal_robots_ur5e" / "ur5e.xml",
)

from controller_core.x_axis_cartesian_impedance import JOINT_NAME_ORDER


def build_mujoco_gravity_estimator():
    try:
        import mujoco
    except ImportError:
        return None
    for scene in MUJOCO_GRAVITY_SCENE_CANDIDATES:
        if not scene.exists():
            continue
        try:
            model = mujoco.MjModel.from_xml_path(str(scene))
            if model.nu < 6 or model.nq < 6 or model.nv < 6:
                continue
            return model, mujoco.MjData(model)
        except Exception:
            continue
    return None


def compute_mujoco_gravity_bias(estimator, q: np.ndarray, qd: np.ndarray) -> np.ndarray | None:
    if estimator is None:
        return None
    import mujoco

    model, data = estimator
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    qd = np.asarray(qd, dtype=np.float64).reshape(-1)
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    for idx, joint_name in enumerate(JOINT_NAME_ORDER):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid < 0:
            return None
        qadr = int(model.jnt_qposadr[jid])
        vadr = int(model.jnt_dofadr[jid])
        if qadr < model.nq:
            data.qpos[qadr] = float(q[idx])
        if vadr < model.nv:
            data.qvel[vadr] = float(qd[idx])
    mujoco.mj_forward(model, data)
    return np.asarray(data.qfrc_bias[:6], dtype=np.float64).copy()


def gravity_feedforward(
    estimator,
    q: np.ndarray,
    qd: np.ndarray,
    scale: float = 1.0,
) -> np.ndarray | None:
    bias = compute_mujoco_gravity_bias(estimator, q, qd)
    if bias is None:
        return None
    return -float(scale) * np.asarray(bias, dtype=np.float64)
