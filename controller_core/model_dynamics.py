"""Model-based rigid-body dynamics provider (Pinocchio-backed) for the UR5e.

`controller_core` stays numpy-pure: pinocchio is a lazy, optional import confined
to this module. Everything else consumes the `DynamicsProvider` protocol —
callers can swap in MuJoCo-derived or dummy providers in tests.

Terms follow the standard manipulator equation
``M(q) qdd + C(q, qd) qd + g(q) = tau``:

- ``gravity(q)``      -> g(q)
- ``coriolis(q, qd)`` -> C(q, qd) @ qd  (velocity-product torques, no gravity)
- ``mass_matrix(q)``  -> M(q) incl. joint armature (Pinocchio's MJCF loader imports
  ``armature`` and ``crba`` includes it, matching MuJoCo's ``mj_fullM``)
- ``bias(q, qd)``     -> C(q, qd) @ qd + g(q)  (equals MuJoCo ``qfrc_bias``)
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

from .x_axis_cartesian_impedance import JOINT_NAME_ORDER

DEFAULT_UR5E_MJCF = Path(__file__).resolve().parents[1] / "assets" / "ur5e_torque" / "ur5e_torque.xml"


@runtime_checkable
class DynamicsProvider(Protocol):
    def gravity(self, q: np.ndarray) -> np.ndarray: ...

    def coriolis(self, q: np.ndarray, qd: np.ndarray) -> np.ndarray: ...

    def mass_matrix(self, q: np.ndarray) -> np.ndarray: ...

    def bias(self, q: np.ndarray, qd: np.ndarray) -> np.ndarray: ...


class PinocchioUR5eDynamics:
    """Pinocchio-backed dynamics for the custom torque-actuated UR5e MJCF.

    Loads the same MJCF the MuJoCo lane simulates (`assets/ur5e_torque/ur5e_torque.xml`),
    so mass/inertia parity with the running sim holds by construction.
    """

    def __init__(
        self,
        mjcf_path: str | Path = DEFAULT_UR5E_MJCF,
        *,
        expected_joint_order: Sequence[str] = JOINT_NAME_ORDER,
    ) -> None:
        try:
            import pinocchio as pin
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise ImportError(
                "pinocchio is required for PinocchioUR5eDynamics; "
                "install it with `pip install pin` (see environment.yml)"
            ) from exc

        self._pin = pin
        mjcf_path = Path(mjcf_path)
        if not mjcf_path.exists():
            raise FileNotFoundError(f"UR5e MJCF not found: {mjcf_path}")
        self.model = pin.buildModelFromMJCF(str(mjcf_path))
        self.data = self.model.createData()
        self.nv = int(self.model.nv)

        # Joint order must match the controller's canonical ordering.
        model_joints = [name for name in self.model.names if name != "universe"]
        expected = list(expected_joint_order)
        if model_joints != expected:
            raise ValueError(
                f"Pinocchio joint order {model_joints} does not match "
                f"expected controller order {expected}"
            )
        if self.nv != len(expected):
            raise ValueError(f"Expected {len(expected)} dofs, model has nv={self.nv}")

    # -- helpers ---------------------------------------------------------------

    def _q(self, q: np.ndarray) -> np.ndarray:
        arr = np.asarray(q, dtype=np.float64).reshape(-1)
        if arr.shape[0] != self.nv:
            raise ValueError(f"q must have length {self.nv}, got {arr.shape[0]}")
        return arr

    # -- DynamicsProvider ------------------------------------------------------

    def gravity(self, q: np.ndarray) -> np.ndarray:
        q = self._q(q)
        return np.asarray(
            self._pin.computeGeneralizedGravity(self.model, self.data, q), dtype=np.float64
        ).copy()

    def coriolis(self, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        q = self._q(q)
        qd = self._q(qd)
        return self.bias(q, qd) - self.gravity(q)

    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        q = self._q(q)
        M = np.asarray(self._pin.crba(self.model, self.data, q), dtype=np.float64).copy()
        # crba fills the upper triangle only; symmetrize.
        return np.triu(M) + np.triu(M, 1).T

    def bias(self, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        q = self._q(q)
        qd = self._q(qd)
        zero_qdd = np.zeros(self.nv, dtype=np.float64)
        return np.asarray(
            self._pin.rnea(self.model, self.data, q, qd, zero_qdd), dtype=np.float64
        ).copy()
