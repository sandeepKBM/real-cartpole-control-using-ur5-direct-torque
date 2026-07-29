"""Pure-numpy joint-space dynamics-residual math.

``controller_core`` stays simulator-independent (numpy only) -- see
``model_dynamics.py``'s module docstring for the same rule applied to the
dynamics provider this module consumes.

Diagnostic-only. Nothing here feeds any safety trip condition: this module
has no notion of a "limit" or a "decision" at all, it only computes numbers.
See ``hardware/direct_torque_transport.py``'s residual-observer wiring and
``docs/status/direct_torque_residual_observer_2026-07-29.md`` for how the
outputs are used (logged to ``trace_rows`` for post-hoc analysis only).

Manipulator equation (matches ``model_dynamics.py``'s convention):
``M(q) qdd + C(q, qd) qd + g(q) = tau``, i.e. ``M(q) qdd = tau - bias(q, qd)``
where ``bias(q, qd) = C(q, qd) @ qd + g(q)``.
"""

from __future__ import annotations

import numpy as np


def predict_joint_acceleration(
    mass_matrix: np.ndarray,
    tau_total_physical: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    """``qdd_pred = M(q)^-1 @ (tau_total_physical - bias(q, qd))``.

    ``tau_total_physical`` must be the TRUE total torque actually delivered
    to the joints this cycle, not just a controller's own output -- e.g. on
    the real UR5e's ``direct_torque`` control mode, PolyScope's
    ``directTorque()`` call adds gravity compensation automatically on top of
    whatever torque Python sends (see AGENTS.md: "Never add gravity torque in
    Python when using directTorque() -- PolyScope adds it"), so the caller
    must add that gravity term back in before calling this function. ``bias``
    is ``C(q, qd) @ qd + g(q)`` (``PinocchioUR5eDynamics.bias`` / MuJoCo
    ``qfrc_bias``), evaluated at the same ``(q, qd)`` this cycle's state was
    read at.

    Uses ``np.linalg.solve`` rather than an explicit matrix inverse (better
    conditioned, standard practice for a one-off linear solve).
    """
    mass_matrix = np.asarray(mass_matrix, dtype=np.float64).reshape(6, 6)
    tau_total_physical = np.asarray(tau_total_physical, dtype=np.float64).reshape(6)
    bias = np.asarray(bias, dtype=np.float64).reshape(6)
    return np.linalg.solve(mass_matrix, tau_total_physical - bias)


def joint_acceleration_residual(qdd_measured: np.ndarray, qdd_predicted: np.ndarray) -> np.ndarray:
    """Per-joint residual, ``measured - predicted`` (6,).

    Sign convention: positive means the real arm accelerated MORE than known
    dynamics + the commanded torque explain (consistent with the direction a
    real external disturbance -- a collision, a joint fault -- would push
    this signal; see the module docstring in
    ``hardware/direct_torque_transport.py`` for why this is diagnostic-only
    and must never be read as "implausible => ignore").
    """
    qdd_measured = np.asarray(qdd_measured, dtype=np.float64).reshape(6)
    qdd_predicted = np.asarray(qdd_predicted, dtype=np.float64).reshape(6)
    return qdd_measured - qdd_predicted
