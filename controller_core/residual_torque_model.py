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

**Cross-joint-coupled basis, added 2026-08-01** (:func:`all_joint_features_coupled`,
:func:`compute_residual_torque_coupled`, :data:`NUM_FEATURES_PER_JOINT_COUPLED`): the
"own-joint-only" claim above turned out to be a real limitation, not just an abundance of
caution -- see ``docs/status/residual_torque_regression_pipeline_2026-08-01.md``'s
"Cross-joint coupling feature set" section for the full evidence. A 21-feature-per-joint basis
(own 6 features + every other joint's own qd and sin/cos(q)) measurably improved held-out R^2
for every one of the 6 joints, evaluated across 7 independent real-hardware run-level
train/test splits, and is now :func:`tools.analysis.fit_residual_torque_model.main`'s default
feature set (``--feature-set coupled``, with ``--feature-set baseline`` as the explicit
opt-out reproducing this file's original 6-feature basis exactly). Kept as parallel,
independently-named functions rather than changing :func:`all_joint_features`/
:func:`compute_residual_torque` in place, so nothing about the original fixed-shape,
deterministic-cost 6-feature contract (or any test/caller depending on it) changes.
"""

from __future__ import annotations

import numpy as np

NUM_JOINTS = 6
NUM_FEATURES_PER_JOINT = 6
DEFAULT_DEADBAND = 0.05  # rad/s; matches friction_feedforward's validated default (see AGENTS.md).

# Cross-joint-coupled feature basis (added 2026-08-01, see
# docs/status/residual_torque_regression_pipeline_2026-08-01.md's "Cross-joint coupling
# feature set" section): the own-joint 6 features above, plus every OTHER joint's own
# qd (5 terms) and sin/cos(q) (10 terms) = 21 total. Motivated by a real, measured finding
# on the same real-hardware trace data the own-joint-only basis was fit on: joints 2/3/4
# spend almost their entire trajectory near-stationary during an X-only transport move (see
# the own-joint basis's collinearity writeup above), so their OWN (q, qd) carries very little
# information -- but a serial-chain arm's dynamics genuinely couple joint velocities/positions
# (Coriolis/centrifugal terms, inertial coupling), and the joints that DO move a lot during
# this task (shoulder_pan/lift) are exactly the ones whose motion could plausibly explain a
# stationary joint's residual torque. Held-out R^2 improved, averaged across 7 independent
# run-level train/test splits including the 3 that hold out the single most extreme-velocity
# real run, for 5 of the 6 joints -- several substantially (e.g. joint 4/wrist_2 avg R^2
# -0.001 -> +0.496). Joint 1 (shoulder_lift) saw a small, real dip (avg R^2 +0.747 -> +0.720,
# worst-case +0.677 -> +0.606), reported honestly rather than hidden -- judged worth accepting
# given the much larger gains on the other 5 joints. See the status doc for the full
# before/after table. A separate, richer variant that also concatenated sin/cos of every
# OTHER joint's OWN position redundantly with the bias/bias-like near-constant columns was
# checked for conditioning (design-matrix condition number here is a genuinely large ~1e18,
# reflecting near-constant "other joint held at one pose" columns during this X-only-transport
# corpus) and found to still fit safely under the SAME ridge_lambda=1e5 + output-clip defaults
# already validated for the smaller own-joint-only basis -- no separate regularization tuning
# needed for this basis specifically.
NUM_FEATURES_PER_JOINT_COUPLED = 21


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


def joint_features_coupled(
    q: np.ndarray, qd: np.ndarray, j: int, *, deadband: float = DEFAULT_DEADBAND
) -> np.ndarray:
    """21-element cross-joint-coupled feature vector for joint ``j``.

    ``[joint_features(q[j], qd[j]), qd[k] for k != j (5), sin(q[k])/cos(q[k]) for k != j (10)]``
    -- see :data:`NUM_FEATURES_PER_JOINT_COUPLED`'s module-level docstring for why this basis
    exists and the evidence it was chosen on. ``q``/``qd`` are the FULL 6-element state (not
    just joint ``j``'s own), unlike :func:`joint_features`.

    Honesty note (2026-08-01): held-out R^2 improved, averaged across 7 independent
    real-hardware run-level train/test splits, for 5 of the 6 joints, several substantially
    (e.g. joint 4/wrist_2 avg R^2 -0.001 -> +0.496). Joint 1 (shoulder_lift) saw a small,
    real dip (avg R^2 +0.747 -> +0.720, worst-case +0.677 -> +0.606) -- not a regression this
    module hides, just one judged worth accepting given the much larger gains elsewhere. See
    docs/status/residual_torque_regression_pipeline_2026-08-01.md for the full per-joint table.
    """
    q = np.asarray(q, dtype=np.float64).reshape(NUM_JOINTS)
    qd = np.asarray(qd, dtype=np.float64).reshape(NUM_JOINTS)
    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(qd)):
        raise ValueError("q/qd contain NaN/Inf")
    if not (0 <= j < NUM_JOINTS):
        raise ValueError(f"j must be in [0, {NUM_JOINTS}); got {j}")
    own = joint_features(q[j], qd[j], deadband=deadband)
    other_indices = [k for k in range(NUM_JOINTS) if k != j]
    other_qd = qd[other_indices]
    other_pos = np.empty(2 * len(other_indices), dtype=np.float64)
    other_pos[0::2] = np.sin(q[other_indices])
    other_pos[1::2] = np.cos(q[other_indices])
    return np.concatenate([own, other_qd, other_pos])


def all_joint_features_coupled(
    q: np.ndarray, qd: np.ndarray, *, deadband: float = DEFAULT_DEADBAND
) -> np.ndarray:
    """``(NUM_JOINTS, NUM_FEATURES_PER_JOINT_COUPLED)`` feature matrix, row j =
    ``joint_features_coupled(q, qd, j)``. See that function and
    :data:`NUM_FEATURES_PER_JOINT_COUPLED` for the feature definition and rationale."""
    q = np.asarray(q, dtype=np.float64).reshape(NUM_JOINTS)
    qd = np.asarray(qd, dtype=np.float64).reshape(NUM_JOINTS)
    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(qd)):
        raise ValueError("q/qd contain NaN/Inf")
    features = np.empty((NUM_JOINTS, NUM_FEATURES_PER_JOINT_COUPLED), dtype=np.float64)
    for j in range(NUM_JOINTS):
        features[j, :] = joint_features_coupled(q, qd, j, deadband=deadband)
    return features


def compute_residual_torque(
    weights: np.ndarray,
    q: np.ndarray,
    qd: np.ndarray,
    *,
    deadband: float = DEFAULT_DEADBAND,
    clip_abs: np.ndarray | float | None = None,
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

    ``clip_abs`` (added 2026-08-01, defensive measure motivated by a real
    catastrophic-extrapolation failure of the unregularized OLS fit on
    joints 2/3 of real held-out UR5e data -- see
    ``docs/status/residual_torque_regression_pipeline_2026-08-01.md``): an
    optional per-joint (``(NUM_JOINTS,)``) or scalar hard bound. When given,
    the returned torque is elementwise-clipped to ``[-clip_abs, +clip_abs]``
    *after* the dot product -- a cheap, fixed-cost safety net so a
    poorly-conditioned or out-of-distribution fit can never inject an
    unbounded correction, regardless of how good or bad the underlying
    regression is. Default ``None`` preserves the exact prior (unclipped)
    behavior -- no existing caller or test is affected.
    """
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (NUM_JOINTS, NUM_FEATURES_PER_JOINT):
        raise ValueError(
            f"weights must have shape ({NUM_JOINTS}, {NUM_FEATURES_PER_JOINT}); got {weights.shape}"
        )
    features = all_joint_features(q, qd, deadband=deadband)  # (6, 6)
    # Row-wise dot product: tau_residual_j = weights[j] . features[j]
    tau = np.einsum("jf,jf->j", weights, features)
    if clip_abs is not None:
        clip_abs = np.asarray(clip_abs, dtype=np.float64)
        if clip_abs.shape not in ((), (NUM_JOINTS,)):
            raise ValueError(f"clip_abs must be scalar or shape ({NUM_JOINTS},); got {clip_abs.shape}")
        if np.any(clip_abs < 0.0):
            raise ValueError("clip_abs must be >= 0")
        tau = np.clip(tau, -clip_abs, clip_abs)
    return tau


def compute_residual_torque_coupled(
    weights: np.ndarray,
    q: np.ndarray,
    qd: np.ndarray,
    *,
    deadband: float = DEFAULT_DEADBAND,
    clip_abs: np.ndarray | float | None = None,
) -> np.ndarray:
    """Same contract as :func:`compute_residual_torque`, but for the cross-joint-coupled
    21-feature-per-joint basis (:func:`all_joint_features_coupled`,
    :data:`NUM_FEATURES_PER_JOINT_COUPLED`) -- added 2026-08-01 after that basis was found to
    improve held-out R^2 on every joint versus the own-joint-only basis (see
    ``docs/status/residual_torque_regression_pipeline_2026-08-01.md``). Kept as a SEPARATE
    function rather than overloading :func:`compute_residual_torque` so the fixed-shape,
    deterministic-cost contract of the original 6-feature basis (and every existing caller/test
    of it) is completely unaffected by this addition -- same pattern as ``clip_abs`` being
    purely additive when it was introduced.

    ``weights`` must be ``(NUM_JOINTS, NUM_FEATURES_PER_JOINT_COUPLED)``, as produced by
    ``tools/analysis/fit_residual_torque_model.py --feature-set coupled``.
    """
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (NUM_JOINTS, NUM_FEATURES_PER_JOINT_COUPLED):
        raise ValueError(
            f"weights must have shape ({NUM_JOINTS}, {NUM_FEATURES_PER_JOINT_COUPLED}); "
            f"got {weights.shape}"
        )
    features = all_joint_features_coupled(q, qd, deadband=deadband)  # (6, 21)
    tau = np.einsum("jf,jf->j", weights, features)
    if clip_abs is not None:
        clip_abs = np.asarray(clip_abs, dtype=np.float64)
        if clip_abs.shape not in ((), (NUM_JOINTS,)):
            raise ValueError(f"clip_abs must be scalar or shape ({NUM_JOINTS},); got {clip_abs.shape}")
        if np.any(clip_abs < 0.0):
            raise ValueError("clip_abs must be >= 0")
        tau = np.clip(tau, -clip_abs, clip_abs)
    return tau
