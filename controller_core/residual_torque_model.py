"""Deterministic-cost, numpy-only inference for a fitted per-joint residual
torque model (2026-08-01 phase 1 of the offline residual-torque-regression
pipeline -- see ``docs/status/residual_torque_regression_pipeline_2026-08-01.md``
and ``docs/status/nonlinear_controller_research_2026-07-31.md`` section 1 for
the design rationale).

**Not wired into any control path.** Nothing in ``x_axis_cartesian_impedance.py``
imports or calls this module; it exists so the *inference side* of a fitted
residual-torque correction can be prototyped, unit-tested, and cost-profiled
against the real-time budget (docs/status/direct_torque_controller_phase_profiling_2026-07-31.md's
~0.5-0.7 ms headroom) *before* anyone decides whether to actually promote it
into the controller. If it is ever wired in, that must happen the way every
other addition in this file's neighborhood has (``friction_feedforward``,
``wrist_orientation_task``): a new, default-off ``CartesianImpedanceConfig``
flag, zero behavior change for every existing config, and a byte-identical
regression test.

Model form (basis-function regression, per the research doc's explicit
recommendation over a GP or a large NN -- fixed matrix-vector cost, no
data-dependent branching, trivially numpy-only):

    tau_residual_j_pred = w_j . phi_j(q_j, qd_j)

i.e. one small feature vector *per joint*, built only from that joint's own
position/velocity (not the full 12-dim (q, qd) state). This is a deliberate
simplicity choice for a first cut, not a claim that cross-joint coupling is
provably absent -- see the status doc's honesty notes. Fewer features also
matters directly for a small dataset: this phase 1 pipeline was fit on a few
thousand correlated samples from a couple dozen short sim rollouts, not an
independent-sample count anywhere near that -- a high-dimensional feature map
would be very easy to overfit and impossible to validate honestly at this
data volume.

The fixed feature basis (6 features per joint, chosen to resemble the
friction-feedforward term already in ``x_axis_cartesian_impedance.py`` plus a
couple of low-order terms for anything position-dependent left over after
gravity/Coriolis compensation):

    phi_j(q_j, qd_j) = [
        1.0,                          # bias
        qd_j,                         # viscous-like linear term
        tanh(qd_j / deadband),        # Coulomb-like saturating sign term
        qd_j * abs(qd_j),             # quadratic (velocity-squared drag) term
        sin(q_j),                     # low-order position term
        cos(q_j),                     # low-order position term
    ]

Weights are a plain ``(6, 6)`` numpy array (per-joint row, per-feature
column) -- no scipy/sklearn object anywhere near this module, satisfying the
same "controller_core stays numpy-only" rule as everywhere else in this
package. Fitting (least squares, cross-validation, everything that needs
scipy/sklearn-class tooling) lives entirely outside this package, in
``tools/analysis/fit_residual_torque_model.py``, which imports the *same*
feature function from here so the fit and the inference path can never
silently diverge.
"""

from __future__ import annotations

import numpy as np

NUM_JOINTS = 6
NUM_FEATURES_PER_JOINT = 6
DEFAULT_DEADBAND = 0.05  # rad/s; matches friction_feedforward's validated default (see AGENTS.md).


def joint_features(q_j: float, qd_j: float, *, deadband: float = DEFAULT_DEADBAND) -> np.ndarray:
    """Fixed 6-element feature vector for a single joint's own (q_j, qd_j).

    Deterministic cost: one ``tanh``, one ``sin``, one ``cos``, a handful of
    scalar multiplies -- no loops, no data-dependent branching.
    """
    deadband = float(deadband)
    if deadband <= 0.0:
        raise ValueError("deadband must be > 0")
    q_j = float(q_j)
    qd_j = float(qd_j)
    return np.array(
        [
            1.0,
            qd_j,
            np.tanh(qd_j / deadband),
            qd_j * abs(qd_j),
            np.sin(q_j),
            np.cos(q_j),
        ],
        dtype=np.float64,
    )


def all_joint_features(
    q: np.ndarray, qd: np.ndarray, *, deadband: float = DEFAULT_DEADBAND
) -> np.ndarray:
    """``(NUM_JOINTS, NUM_FEATURES_PER_JOINT)`` feature matrix, row j = ``joint_features(q[j], qd[j])``.

    Fixed shape regardless of input values -- the "bounded, data-independent
    worst-case cost" the real-time constraint requires (see module docstring).
    """
    q = np.asarray(q, dtype=np.float64).reshape(NUM_JOINTS)
    qd = np.asarray(qd, dtype=np.float64).reshape(NUM_JOINTS)
    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(qd)):
        raise ValueError("q/qd contain NaN/Inf")
    features = np.empty((NUM_JOINTS, NUM_FEATURES_PER_JOINT), dtype=np.float64)
    for j in range(NUM_JOINTS):
        features[j, :] = joint_features(q[j], qd[j], deadband=deadband)
    return features


def compute_residual_torque(
    weights: np.ndarray,
    q: np.ndarray,
    qd: np.ndarray,
    *,
    deadband: float = DEFAULT_DEADBAND,
) -> np.ndarray:
    """Per-joint residual torque correction, ``(NUM_JOINTS,)``.

    ``weights`` is a plain ``(NUM_JOINTS, NUM_FEATURES_PER_JOINT)`` array (row
    j = joint j's fitted weight vector, as produced by
    ``tools/analysis/fit_residual_torque_model.py`` and saved as a bare
    ``.npy``/``.npz`` array -- never a scipy/sklearn model object). Cost is a
    fixed 6x6 feature build (see :func:`all_joint_features`) plus a per-row
    dot product -- no dynamic-size arrays, no loops over data, so the
    worst-case flop count does not depend on the weights' values, matching
    the deadline-monitor-compatible "bounded, data-independent worst-case
    cost" requirement in the module docstring.
    """
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (NUM_JOINTS, NUM_FEATURES_PER_JOINT):
        raise ValueError(
            f"weights must have shape ({NUM_JOINTS}, {NUM_FEATURES_PER_JOINT}); got {weights.shape}"
        )
    features = all_joint_features(q, qd, deadband=deadband)  # (6, 6)
    # Row-wise dot product: tau_residual_j = weights[j] . features[j]
    return np.einsum("jf,jf->j", weights, features)
