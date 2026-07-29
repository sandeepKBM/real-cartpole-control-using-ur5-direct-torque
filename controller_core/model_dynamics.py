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

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

from .x_axis_cartesian_impedance import JOINT_NAME_ORDER

DEFAULT_UR5E_MJCF = Path(__file__).resolve().parents[1] / "assets" / "ur5e_torque" / "ur5e_torque.xml"
DEFAULT_ATTACHMENT_SITE = "attachment_site"


@runtime_checkable
class DynamicsProvider(Protocol):
    def gravity(self, q: np.ndarray) -> np.ndarray: ...

    def coriolis(self, q: np.ndarray, qd: np.ndarray) -> np.ndarray: ...

    def mass_matrix(self, q: np.ndarray) -> np.ndarray: ...

    def bias(self, q: np.ndarray, qd: np.ndarray) -> np.ndarray: ...


def _root_body_quat_wxyz(mjcf_path: Path) -> tuple[float, float, float, float]:
    """Read the ``quat="w x y z"`` attribute of the first ``<body>`` under
    ``<worldbody>`` in an MJCF file (MJCF's own w-first convention). Defaults
    to identity ``(1, 0, 0, 0)`` if absent.

    This exists to work around a real parity gap in ``pin.buildModelFromMJCF``
    (Pinocchio 4.0.0, this env, verified 2026-07-29): it does not apply this
    root body's own ``quat`` to the loaded frame tree, even though MuJoCo
    does. For this UR5e MJCF the root ``<body name="base" quat="0 0 0 -1">``
    is a 180-degree rotation about Z; Pinocchio's ``base`` frame comes back
    with an identity placement relative to ``universe`` instead. Joint-space
    outputs (gravity/coriolis/mass matrix -- see the parity tests in
    ``tests/mujoco/test_pinocchio_parity.py``) are numerically unaffected
    because this particular root rotation happens to share gravity's axis
    (Z), so it is invisible to a torque comparison; anything expressed in
    *world-frame Cartesian* coordinates (e.g. a site Jacobian) is not, and
    silently comes out mirrored on X/Y. See
    ``docs/status/local_dynamics_speedup_investigation_2026-07-29.md`` for the
    empirical trace (worst-case Jacobian error ~2.0 rad or m before this
    correction, ~3e-15 after, across 200 full-joint-range samples).
    """
    root = ET.parse(mjcf_path).getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        return (1.0, 0.0, 0.0, 0.0)
    first_body = worldbody.find("body")
    if first_body is None:
        return (1.0, 0.0, 0.0, 0.0)
    quat_attr = first_body.get("quat")
    if quat_attr is None:
        return (1.0, 0.0, 0.0, 0.0)
    w, x, y, z = (float(v) for v in quat_attr.split())
    return (w, x, y, z)


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

        # World-frame correction for Cartesian (Jacobian) outputs -- see
        # `_root_body_quat_wxyz`'s docstring. Not needed for any joint-space
        # quantity (gravity/coriolis/mass_matrix/bias), only for `jacobian()`.
        w, x, y, z = _root_body_quat_wxyz(mjcf_path)
        root_R = pin.Quaternion(w, x, y, z).toRotationMatrix()
        self._jacobian_world_correction = np.block(
            [[root_R, np.zeros((3, 3))], [np.zeros((3, 3)), root_R]]
        )
        self._frame_ids: dict[str, int] = {}

    # -- helpers ---------------------------------------------------------------

    def _q(self, q: np.ndarray) -> np.ndarray:
        arr = np.asarray(q, dtype=np.float64).reshape(-1)
        if arr.shape[0] != self.nv:
            raise ValueError(f"q must have length {self.nv}, got {arr.shape[0]}")
        return arr

    def _frame_id(self, site_name: str) -> int:
        frame_id = self._frame_ids.get(site_name)
        if frame_id is None:
            if not self.model.existFrame(site_name):
                raise ValueError(f"frame {site_name!r} not found in Pinocchio model")
            frame_id = int(self.model.getFrameId(site_name))
            self._frame_ids[site_name] = frame_id
        return frame_id

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

    def jacobian(self, q: np.ndarray, *, site_name: str = DEFAULT_ATTACHMENT_SITE) -> np.ndarray:
        """6xnv site Jacobian (translational rows 0:3, rotational rows 3:6),
        expressed in world-aligned coordinates at the site's own origin --
        the same convention MuJoCo's ``mj_jacSite`` uses (parity verified in
        ``tests/mujoco/test_pinocchio_parity.py::test_jacobian_parity``, worst
        case < 1e-6 across 200 full-joint-range samples).

        Uses Pinocchio's ``LOCAL_WORLD_ALIGNED`` reference frame, then applies
        `self._jacobian_world_correction` -- see `_root_body_quat_wxyz` for
        why that correction is required for this MJCF.
        """
        pin = self._pin
        q = self._q(q)
        frame_id = self._frame_id(site_name)
        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        jacobian_raw = pin.getFrameJacobian(self.model, self.data, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        return (self._jacobian_world_correction @ np.asarray(jacobian_raw, dtype=np.float64)).copy()
