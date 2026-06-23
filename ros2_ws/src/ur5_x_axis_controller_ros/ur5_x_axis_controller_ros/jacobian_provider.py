"""
Jacobian provider abstraction.

The controller node needs a world-frame 3x6 position Jacobian at the EE
each cycle. This module defines a small interface with a default
implementation that simply reads whatever the CoppeliaSim bridge has
published on ``/ur5/jacobian``. A fallback ``ConstantIdentityJacobian``
provider is provided for unit testing without any simulator.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class JacobianProvider:
    """Latched Jacobian provider.

    Call ``update(msg_data)`` whenever a new Jacobian message arrives; call
    ``get_pos()`` / ``get_rot()`` each control cycle to retrieve the last
    known values. Returns ``None`` if no message has been seen yet.
    """

    num_joints: int = 6
    _jac_pos: np.ndarray | None = field(default=None, repr=False)
    _jac_rot: np.ndarray | None = field(default=None, repr=False)
    _last_stamp_ns: int | None = field(default=None, repr=False)

    def update(self, flat_data, stamp_ns: int | None = None) -> None:
        arr = np.asarray(flat_data, dtype=np.float64).reshape(-1)
        if arr.size == 3 * self.num_joints:
            self._jac_pos = arr.reshape(3, self.num_joints)
            self._jac_rot = None
        elif arr.size == 6 * self.num_joints:
            self._jac_pos = arr[: 3 * self.num_joints].reshape(3, self.num_joints)
            self._jac_rot = arr[3 * self.num_joints :].reshape(3, self.num_joints)
        else:
            raise ValueError(
                f"Unexpected Jacobian payload size {arr.size}; expected "
                f"{3 * self.num_joints} or {6 * self.num_joints}."
            )
        self._last_stamp_ns = stamp_ns

    def get_pos(self) -> np.ndarray | None:
        return self._jac_pos

    def get_rot(self) -> np.ndarray | None:
        return self._jac_rot

    def get_matrix6(self) -> np.ndarray | None:
        """Return ``(6, 6)`` Jacobian ``[J_pos; J_rot]``, or ``None`` if unset."""
        if self._jac_pos is None:
            return None
        j_rot = (
            self._jac_rot
            if self._jac_rot is not None
            else np.zeros((3, self.num_joints), dtype=np.float64)
        )
        return np.vstack([self._jac_pos, j_rot])

    def age_s(self, now_ns: int) -> float | None:
        if self._last_stamp_ns is None:
            return None
        return max(0.0, (now_ns - self._last_stamp_ns) * 1e-9)


@dataclass
class ConstantIdentityJacobian:
    """Useful for offline tests: maps joint 0 directly to EE X."""

    num_joints: int = 6

    def get_pos(self) -> np.ndarray:
        return np.eye(3, self.num_joints, dtype=np.float64)

    def get_rot(self) -> np.ndarray:
        return np.zeros((3, self.num_joints), dtype=np.float64)

    def update(self, *_args, **_kwargs) -> None:  # pragma: no cover
        pass

    def age_s(self, _now_ns: int) -> float | None:  # pragma: no cover
        return 0.0

    def get_matrix6(self) -> np.ndarray:
        return np.vstack([self.get_pos(), self.get_rot()])
