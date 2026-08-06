"""Local J(q) and M(q) for hardware — skips RTDE getJacobian/getMassMatrix.

Uses the same MuJoCo MJCF as the simulation lane so OSC gains tuned in MuJoCo
apply bit-exact on the fast path. Requires ``mujoco`` in the Python env.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE_XML = REPO_ROOT / "assets" / "ur5e_torque" / "scene.xml"
DEFAULT_UR5E_MJCF = REPO_ROOT / "assets" / "ur5e_torque" / "ur5e_torque.xml"
DEFAULT_SITE_NAME = "attachment_site"
DYNAMICS_SOURCES = frozenset({"rtde", "local", "local_pinocchio"})


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

    def fk_and_jacobian(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(ee_pos, ee_quat_wxyz, jacobian) at an ARBITRARY q -- one
        mj_forward pass. Used as CartesianVelocityConfig.ik_seeded_
        resolution's fk_jacobian_fn callback: that mode needs to evaluate
        forward kinematics at intermediate IK-iteration configurations, not
        just the robot's actual current q, which jacobian_and_mass_matrix
        alone doesn't expose (it only returns J/M, no pose)."""
        q_arr = np.asarray(q, dtype=np.float64).reshape(self.n_joints)
        self.data.qpos[: self.n_joints] = q_arr
        self._mujoco.mj_forward(self.model, self.data)
        pos = np.asarray(self.data.site_xpos[self.site_id], dtype=np.float64).copy()
        quat = np.zeros(4, dtype=np.float64)
        self._mujoco.mju_mat2Quat(quat, self.data.site_xmat[self.site_id])

        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        self._mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)
        jacobian = np.vstack([jacp[:, : self.n_joints], jacr[:, : self.n_joints]]).astype(np.float64)
        return pos, quat, jacobian

    def jacobian_and_mass_matrix(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q_arr = np.asarray(q, dtype=np.float64).reshape(self.n_joints)
        self.data.qpos[: self.n_joints] = q_arr
        self._mujoco.mj_forward(self.model, self.data)

        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        self._mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)
        jacobian = np.vstack([jacp[:, : self.n_joints], jacr[:, : self.n_joints]]).astype(np.float64)

        from simulation.ur5e_mujoco_torque import expand_mass_matrix

        mass_full = expand_mass_matrix(self.model, self.data)
        mass_matrix = mass_full[: self.n_joints, : self.n_joints].copy()
        return jacobian, mass_matrix

    def coriolis(self, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        """C(q,qd)@qd via qfrc_bias(q,qd) - qfrc_bias(q,0) -- same MuJoCo-native
        trick as simulation.ur5e_mujoco_torque's non-Pinocchio Coriolis branch
        (kept consistent since this class computes J/M via MuJoCo too, not
        Pinocchio, despite the LocalPinocchioDynamics back-compat alias name).
        """
        jacobian, mass_matrix, coriolis = self.jacobian_mass_and_coriolis(q, qd)
        del jacobian, mass_matrix
        return coriolis

    def jacobian_mass_and_coriolis(
        self, q: np.ndarray, qd: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """J(q), M(q), and C(q,qd)@qd in one call -- two mj_forward passes
        (live qvel, then zeroed qvel) instead of three if J/M and Coriolis
        were each computed via a separate method call."""
        q_arr = np.asarray(q, dtype=np.float64).reshape(self.n_joints)
        qd_arr = np.asarray(qd, dtype=np.float64).reshape(self.n_joints)

        self.data.qpos[: self.n_joints] = q_arr
        self.data.qvel[: self.n_joints] = qd_arr
        self._mujoco.mj_forward(self.model, self.data)

        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        self._mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)
        jacobian = np.vstack([jacp[:, : self.n_joints], jacr[:, : self.n_joints]]).astype(np.float64)

        from simulation.ur5e_mujoco_torque import expand_mass_matrix

        mass_full = expand_mass_matrix(self.model, self.data)
        mass_matrix = mass_full[: self.n_joints, : self.n_joints].copy()

        bias_live = np.asarray(self.data.qfrc_bias, dtype=np.float64)[: self.n_joints].copy()
        self.data.qvel[: self.n_joints] = 0.0
        self._mujoco.mj_forward(self.model, self.data)
        bias_static = np.asarray(self.data.qfrc_bias, dtype=np.float64)[: self.n_joints].copy()
        coriolis = bias_live - bias_static

        return jacobian, mass_matrix, coriolis


# Back-compat alias used by transport import. Despite the name, this is
# MuJoCo-backed (see `coriolis()`'s docstring above) -- kept meaning
# MuJoCo-identical numerics on purpose. Do not repurpose it; if you need a
# real Pinocchio-backed fast path, use `LocalPinocchioFastDynamics` below.
LocalPinocchioDynamics = LocalMujocoDynamics


class LocalPinocchioFastDynamics:
    """J(q)/M(q)/Coriolis via Pinocchio (RNEA/CRBA + frame Jacobian) instead
    of a full MuJoCo ``mj_forward`` pass per cycle.

    This is a genuinely new, explicitly-named, opt-in implementation --
    unlike ``LocalPinocchioDynamics`` above (a legacy alias to
    ``LocalMujocoDynamics`` that intentionally still means MuJoCo-identical
    numerics), this class actually uses Pinocchio for every call. Select it
    via ``dynamics_source="local_pinocchio"`` (see `normalize_dynamics_source`
    / `DYNAMICS_SOURCES`); the default (`dynamics_source="local"`) remains
    unchanged and still resolves to ``LocalPinocchioDynamics`` /
    ``LocalMujocoDynamics``.

    Measured ~10x lower per-call latency than `LocalMujocoDynamics` on this
    hot path (mean ~0.05ms vs ~0.5ms, p99 ~0.08ms vs ~0.78ms, same machine,
    same q samples, 5000 warmed-up calls) with Jacobian/mass-matrix/Coriolis
    parity to <1e-6 against MuJoCo -- see
    ``docs/status/local_dynamics_speedup_investigation_2026-07-29.md`` for
    the full benchmark and the world-frame correction this required
    (Pinocchio's MJCF loader does not apply this model's root body rotation
    to Cartesian outputs; see
    ``controller_core.model_dynamics._root_body_quat_wxyz``).

    Requires the ``pinocchio`` package (see environment.yml).
    """

    def __init__(
        self,
        mjcf_path: str | Path = DEFAULT_UR5E_MJCF,
        *,
        site_name: str = DEFAULT_SITE_NAME,
        n_joints: int = 6,
    ) -> None:
        from controller_core.model_dynamics import PinocchioUR5eDynamics

        self._dyn = PinocchioUR5eDynamics(mjcf_path)
        self._site_name = site_name
        self.n_joints = int(n_joints)
        if self._dyn.nv != self.n_joints:
            raise ValueError(f"Pinocchio model nv={self._dyn.nv} != n_joints={self.n_joints}")

    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        return self._dyn.mass_matrix(q)

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        return self._dyn.jacobian(q, site_name=self._site_name)

    def jacobian_and_mass_matrix(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.jacobian(q), self.mass_matrix(q)

    def coriolis(self, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        return self._dyn.coriolis(q, qd)

    def jacobian_mass_and_coriolis(
        self, q: np.ndarray, qd: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.jacobian(q), self.mass_matrix(q), self.coriolis(q, qd)
