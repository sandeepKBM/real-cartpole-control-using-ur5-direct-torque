"""Local J(q) and M(q) for hardware — skips RTDE getJacobian/getMassMatrix.

Uses the same MuJoCo MJCF as the simulation lane so OSC gains tuned in MuJoCo
apply bit-exact on the fast path. Requires ``mujoco`` in the Python env.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE_XML = REPO_ROOT / "assets" / "ur5e_torque" / "scene.xml"
DEFAULT_SITE_NAME = "attachment_site"
DYNAMICS_SOURCES = frozenset({"rtde", "local"})


def normalize_dynamics_source(value: str) -> str:
    source = str(value).strip().lower()
    if source not in DYNAMICS_SOURCES:
        raise ValueError(f"dynamics_source must be one of {sorted(DYNAMICS_SOURCES)}; got {value!r}")
    return source


class LocalMujocoDynamics:
    """Compute 6x6 Jacobian and mass matrix from q via MuJoCo (matches sim lane)."""

    def __init__(
        self,
        scene_xml: str | Path = DEFAULT_SCENE_XML,
        *,
        site_name: str = DEFAULT_SITE_NAME,
        n_joints: int = 6,
    ) -> None:
        try:
            import mujoco
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise ImportError(
                "mujoco is required for dynamics_source=local; install per environment.yml"
            ) from exc

        self._mujoco = mujoco
        scene_xml = Path(scene_xml)
        if not scene_xml.exists():
            raise FileNotFoundError(f"MuJoCo scene not found: {scene_xml}")
        self.model = mujoco.MjModel.from_xml_path(str(scene_xml))
        self.data = mujoco.MjData(self.model)
        self.site_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name))
        if self.site_id < 0:
            raise ValueError(f"site {site_name!r} not found in {scene_xml}")
        self.n_joints = int(n_joints)
        if self.model.nv < self.n_joints:
            raise ValueError(f"model nv={self.model.nv} < n_joints={self.n_joints}")

    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        _, mass_matrix = self.jacobian_and_mass_matrix(q)
        return mass_matrix

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        jacobian, _ = self.jacobian_and_mass_matrix(q)
        return jacobian

    def jacobian_and_mass_matrix(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q_arr = np.asarray(q, dtype=np.float64).reshape(self.n_joints)
        self.data.qpos[: self.n_joints] = q_arr
        self._mujoco.mj_forward(self.model, self.data)

        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        self._mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)
        jacobian = np.vstack([jacp[:, : self.n_joints], jacr[:, : self.n_joints]]).astype(np.float64)

        mass_full = np.zeros((self.model.nv, self.model.nv), dtype=np.float64)
        self._mujoco.mj_fullM(self.model, self.data, mass_full)
        mass_matrix = mass_full[: self.n_joints, : self.n_joints].copy()
        return jacobian, mass_matrix


# Back-compat alias used by transport import.
LocalPinocchioDynamics = LocalMujocoDynamics
