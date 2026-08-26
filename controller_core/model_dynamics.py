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

        # Dedicated zero-gravity model/data for `coriolis()` -- see that
        # method's docstring. A second `buildModelFromMJCF` parse is a
        # one-time __init__ cost (not hot-path), and keeping a fully separate
        # Model/Data pair (rather than mutating `self.model.gravity` in place
        # around each call) avoids any risk of a stale/misordered
        # gravity-restore leaking into a `gravity()`/`bias()` call on
        # `self.model`/`self.data` from the same instance.
        self._model_zero_gravity = pin.buildModelFromMJCF(str(mjcf_path))
        self._model_zero_gravity.gravity.linear = np.zeros(3)
        self._model_zero_gravity.gravity.angular = np.zeros(3)
        self._data_zero_gravity = self._model_zero_gravity.createData()

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
        """``C(q, qd) @ qd`` -- velocity-product torques only, no gravity.

        Computed as a single ``rnea`` call against a dedicated zero-gravity
        model/data pair (``self._model_zero_gravity``), not as
        ``bias(q, qd) - gravity(q)`` (two separate rnea/computeGeneralizedGravity
        calls, each redoing its own forward-kinematics pass). This is an
        exact algebraic identity -- ``rnea(q, qd, 0)`` with gravity zeroed
        returns exactly the ``C(q, qd) @ qd`` term, since with no gravity
        contribution the manipulator equation's non-linear-effects term
        reduces to pure Coriolis/centrifugal torque -- verified against the
        previous two-call formula to ~1e-14 abs / ~1e-13 rel across random
        (q, qd) samples (well inside this module's existing 1e-12 parity
        tolerance, see ``tests/mujoco/test_pinocchio_parity.py::
        test_coriolis_is_bias_minus_gravity``) and ~1.5-2x faster on this
        machine (see
        ``docs/status/residual_observer_dynamics_optimization_2026-07-30.md``).
        """
        q = self._q(q)
        qd = self._q(qd)
        zero_qdd = np.zeros(self.nv, dtype=np.float64)
        return np.asarray(
            self._pin.rnea(self._model_zero_gravity, self._data_zero_gravity, q, qd, zero_qdd),
            dtype=np.float64,
        ).copy()

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

    def jacobian_derivative(
        self, q: np.ndarray, *, site_name: str = DEFAULT_ATTACHMENT_SITE
    ) -> np.ndarray:
        """Analytic ``dJ/dq`` tensor, shape ``(nv, 6, nv)``: ``[k][:, i] =
        d J[:, i] / dq_k`` for the SAME Jacobian ``jacobian()`` returns.

        This replaces the manipulability CBF's 2*nv central-difference
        ``jacobian()`` evaluations with a single kinematic pass. It is the
        exact partial derivative of the *geometric* Jacobian's matrix entries,
        matching ``jacobian()``'s frame exactly (``LOCAL_WORLD_ALIGNED`` then
        the ``_jacobian_world_correction`` rotation), so the CBF gradient it
        feeds is byte-for-byte the barrier the finite difference computed, only
        faster (validated to the FD's own O(step^2) noise floor, ~1.6e-10, in
        ``tests/mujoco/test_jacobian_derivative_pinocchio.py``).

        CONVENTION (why not ``getFrameKinematicHessian``): Pinocchio's
        kinematic Hessian is a *spatial* (Lie-bracket) second-order object and
        does NOT equal ``d(J_entries)/dq`` -- measured, it disagrees with the
        finite difference by O(1). The reliable analytic route is the classical
        spatial cross-product identity, which holds EXACTLY only in the fixed
        ``WORLD`` frame::

            dJ_i/dq_j = ad(J_j) J_i     for j < i (j proximal to i);  0 else

        (``ad`` = the 6x6 motion cross-product matrix in Pinocchio's
        (linear, angular) column order). ``jacobian()`` uses
        ``LOCAL_WORLD_ALIGNED``, whose reference point moves with ``q``, so the
        world tensor is transformed through the point-translation
        ``J_lwa = A(q) J_world`` with ``A = [[I, -[p]x], [0, I]]`` and its
        derivative ``dA/dq_k`` carrying ``dp/dq_k = J_lwa[:3, k]`` (the frame
        origin's translational velocity per joint). The constant world
        correction commutes through the derivative.
        """
        pin = self._pin
        q = self._q(q)
        frame_id = self._frame_id(site_name)
        n = self.nv
        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        j_world = np.asarray(
            pin.getFrameJacobian(self.model, self.data, frame_id, pin.ReferenceFrame.WORLD),
            dtype=np.float64,
        )
        j_lwa = np.asarray(
            pin.getFrameJacobian(
                self.model, self.data, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            ),
            dtype=np.float64,
        )
        p = np.asarray(self.data.oMf[frame_id].translation, dtype=np.float64)

        # Column blocks of the WORLD Jacobian (linear v, angular w), (3, n).
        vw, ww = j_world[:3], j_world[3:]

        # Every cross product below is a skew-matrix batched matmul, NOT
        # np.cross or per-element arithmetic: on arrays this small the dominant
        # cost is numpy's per-call Python dispatch (~10-15 us for a single
        # np.cross, measured in this env), so the win comes from doing the whole
        # tensor in a HANDFUL of matmuls rather than O(n^2) scalar ops. This
        # keeps the method well under the finite-difference path it replaces.
        def _skew_stack(vecs: np.ndarray) -> np.ndarray:
            """(3, m) columns -> (m, 3, 3) stack of skew matrices ``[v]x``."""
            m = vecs.shape[1]
            s = np.zeros((m, 3, 3), dtype=np.float64)
            s[:, 0, 1] = -vecs[2]; s[:, 0, 2] = vecs[1]
            s[:, 1, 0] = vecs[2];  s[:, 1, 2] = -vecs[0]
            s[:, 2, 0] = -vecs[1]; s[:, 2, 1] = vecs[0]
            return s

        s_v = _skew_stack(vw)   # (n, 3, 3), s_v[j] = [v_j]x
        s_w = _skew_stack(ww)   # (n, 3, 3), s_w[j] = [w_j]x

        # dJw/dq via the exact spatial cross-product identity:
        #   dJ_i/dq_j = ad(Jw[:,j]) Jw[:,i]  for i > j, else 0, with
        #   ad(a) b = [ w_a x v_b + v_a x w_b ;  w_a x w_b ]  (Featherstone).
        # matmul(s_w, vw) is [w_j]x @ Jw_lin -> (n, 3, n), the (j, :, i) entry
        # being w_j x v_i. d_world[j][:, i] = dJ_i/dq_j.
        lin = s_w @ vw + s_v @ ww    # (n, 3, n): w_j x v_i + v_j x w_i
        ang = s_w @ ww               # (n, 3, n): w_j x w_i
        n_idx = np.arange(n)
        mask = (n_idx[None, :] > n_idx[:, None])[:, None, :]  # (n, 1, n): i > j
        d_world = np.empty((n, 6, n), dtype=np.float64)
        d_world[:, :3, :] = lin * mask
        d_world[:, 3:, :] = ang * mask

        # Transform the tensor from the fixed WORLD frame to LOCAL_WORLD_ALIGNED
        # (moving reference point p): Jlwa = A Jw, A = [[I, -[p]x], [0, I]], so
        #   dJlwa/dq_k = dA/dq_k Jw + A dJw/dq_k,
        # with dA/dq_k carrying dp/dq_k = Jlwa[:3, k]. Then apply the constant
        # world correction (a pure rotation, block-diag(R, R)).
        xk_v = d_world[:, :3, :]     # (n, 3, n): [k] = dJw_lin/dq_k
        xk_w = d_world[:, 3:, :]     # (n, 3, n): [k] = dJw_ang/dq_k
        s_p = _skew_stack(p[:, None])[0]           # [p]x, (3, 3)
        s_dp = _skew_stack(j_lwa[:3, :])           # (n, 3, 3): [dp_k]x
        # top[k] = xk_v[k] - [p]x xk_w[k] - [dp_k]x ww
        top = xk_v - (s_p @ xk_w) - (s_dp @ ww)    # (n, 3, n)
        r_corr = self._jacobian_world_correction[:3, :3]
        d_tensor = np.empty((n, 6, n), dtype=np.float64)
        d_tensor[:, :3, :] = r_corr @ top
        d_tensor[:, 3:, :] = r_corr @ xk_w
        return d_tensor
