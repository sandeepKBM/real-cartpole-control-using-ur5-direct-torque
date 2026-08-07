"""Pure-numpy unit tests for the qd_estimate_damping fix in
velocity_gain_tuning/envs/velocity_transport_env.py (2026-08-07).

Background: VelocityTransportEnv.step() estimates joint velocity for
joint_velocity_guard (and integrates env._q forward) via
``pinv(jac) @ xd_cmd`` on the FULL 6x6 Jacobian -- a downstream kinematic
reconstruction, NOT the matrix the controller's own IK solve inverts
(that's a separate, already-damped reduced task-space Jacobian, damped via
CartesianVelocityConfig.pinv_damping=0.005). Two documented real cases
(docs/status/nullspace_v2_search_results_2026-08-06.md's 18.14 and 161.57
rad/s spikes) were both traced directly to wrist_2=0 crossings, where the
FULL Jacobian goes ill-conditioned (measured sigma_min ~3.25e-4 against
sigma_max ~2.1 at the actual crossing) even though the controller's own
reduced task-space Jacobian stays well-conditioned (cond~15) throughout --
i.e. the controller never sees an ill-conditioned matrix; only this
reconstruction does.

This file tests ONLY the underlying math (_damped_pinv, already used
elsewhere in this controller, reused rather than reimplemented here) on
synthetic Jacobians standing in for the two real regimes this session
measured directly: (a) a well-conditioned Jacobian matching the sigma_min
this repo's own 96-cell evaluation grid was measured to produce (1st
percentile sigma_min = 0.038, see velocity_transport_env.py's
qd_estimate_damping docstring), and (b) a near-singular Jacobian matching
the real wrist_2=0 crossing's measured singular values. Deliberately
importable with NO mujoco dependency (controller_core is
simulator-independent, numpy-only per this repo's AGENTS.md) -- the actual
VelocityTransportEnvConfig.qd_estimate_damping default/field and the
env-level integration behavior are covered separately by the mujoco-marked
tests in tests/mujoco/test_velocity_gain_tuning.py, since importing that
module pulls in mujoco via hardware.local_dynamics/simulation.ur5e_mujoco_torque.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from controller_core.cartesian_velocity_controller.math_utils import _damped_pinv  # noqa: E402

# Must match VelocityTransportEnvConfig.qd_estimate_damping's default in
# velocity_gain_tuning/envs/velocity_transport_env.py -- kept as a literal
# here (not imported) specifically so this file stays mujoco-import-free;
# test_qd_estimate_damping_field_default_matches_this_module (mujoco-marked,
# tests/mujoco/test_velocity_gain_tuning.py) cross-checks the real field
# against this same literal so the two can't silently drift apart.
QD_ESTIMATE_DAMPING = 1.0e-3


def _jacobian_from_singular_values(sigma: np.ndarray, seed: int) -> np.ndarray:
    """Build a deterministic 6x6 matrix with EXACTLY the given singular
    value spectrum, via a fixed random orthogonal U/V pair -- lets a test
    assert behavior at a precisely chosen conditioning without depending on
    mujoco FK/Jacobian machinery at all."""
    rng = np.random.default_rng(seed)
    u, _ = np.linalg.qr(rng.standard_normal((6, 6)))
    v, _ = np.linalg.qr(rng.standard_normal((6, 6)))
    return u @ np.diag(sigma) @ v.T


# Well-conditioned regime: matches this repo's own measured 1st-percentile
# sigma_min (0.038) across 6063 steps of the real 96-cell evaluation grid's
# non-singular population (see velocity_transport_env.py's
# qd_estimate_damping docstring) -- i.e. this is the WORST well-conditioned
# case actually observed, not an arbitrary number.
_WELL_CONDITIONED_SIGMA = np.array([2.0, 1.5, 1.0, 0.5, 0.2, 0.038])

# Near-singular regime: matches the real measured singular-value spectrum
# at the wrist_2=0 crossing step of the direct-traced neg45_wrist2offset
# spike case (guard temporarily disabled to observe the true trajectory),
# sigma_min ~3.25e-4 against sigma_max ~2.12.
_NEAR_SINGULAR_SIGMA = np.array([2.11796855, 1.42974250, 0.57649937, 0.19733235, 0.16150689, 3.25467221e-04])


def test_damped_pinv_agrees_closely_with_bare_pinv_away_from_singularity():
    """The core "byte-identical off the edge case" requirement: at
    qd_estimate_damping's chosen value, damping must not meaningfully
    change the estimate for a well-conditioned Jacobian (this repo's own
    measured worst-case well-conditioned sigma_min)."""
    jac = _jacobian_from_singular_values(_WELL_CONDITIONED_SIGMA, seed=0)
    rng = np.random.default_rng(1)
    xd_cmd = rng.standard_normal(6)

    bare_qd = np.linalg.pinv(jac) @ xd_cmd
    damped_qd = _damped_pinv(jac, QD_ESTIMATE_DAMPING) @ xd_cmd

    # Measured relative error at this exact sigma_min analytically:
    # (d/sigma_min)^2 = (1e-3/0.038)^2 ~= 6.9e-4 (0.07%). Assert well under
    # 1% with margin, not the exact figure (this is a property test, not a
    # regression lock on floating-point digits).
    rel_err = np.linalg.norm(damped_qd - bare_qd) / np.linalg.norm(bare_qd)
    assert rel_err < 0.01, f"damping perturbs the well-conditioned estimate by {rel_err:.4%}, expected <1%"


def test_damped_pinv_bounds_the_near_singular_blowup():
    """At the real measured near-singularity spectrum, bare pinv must blow
    up (reproducing the mechanism behind the documented 18.14/161.57 rad/s
    false-positive guard trips) while the damped estimate stays bounded to
    a physically plausible order of magnitude."""
    jac = _jacobian_from_singular_values(_NEAR_SINGULAR_SIGMA, seed=2)
    rng = np.random.default_rng(3)
    # A representative task-space command magnitude (order of this
    # controller's typical kp_x*error commands, ~0.01-1 m/s / rad/s per
    # axis -- see CartesianVelocityConfig's own gain docstring).
    xd_cmd = rng.standard_normal(6) * 0.2

    bare_qd = np.linalg.pinv(jac) @ xd_cmd
    damped_qd = _damped_pinv(jac, QD_ESTIMATE_DAMPING) @ xd_cmd

    bare_norm = float(np.max(np.abs(bare_qd)))
    damped_norm = float(np.max(np.abs(damped_qd)))

    # The near-singular direction really is amplifying (this is the failure
    # mode being fixed): bare must be large.
    assert bare_norm > 20.0, "test setup didn't reproduce a real near-singular blowup"
    # Damping meaningfully bounds it -- at least an order of magnitude
    # smaller than the bare blowup.
    assert damped_norm < bare_norm / 10.0


def test_damped_pinv_reduces_to_bare_pinv_at_zero_damping():
    """damping=0.0 must reproduce (up to float tolerance) the exact prior
    undamped behavior -- confirms _damped_pinv's own zero-damping case is a
    true no-op, the property qd_estimate_damping's docstring relies on when
    reasoning about "what changed.\""""
    jac = _jacobian_from_singular_values(_WELL_CONDITIONED_SIGMA, seed=0)
    rng = np.random.default_rng(1)
    xd_cmd = rng.standard_normal(6)
    bare_qd = np.linalg.pinv(jac) @ xd_cmd
    damped_qd_zero = _damped_pinv(jac, 0.0) @ xd_cmd
    np.testing.assert_allclose(damped_qd_zero, bare_qd, atol=1e-8)


def test_qd_estimate_damping_literal_is_positive_and_small():
    """Sanity bound on the chosen constant itself: must be strictly
    positive (damping=0 would silently reproduce the bug this fix closes)
    and small relative to typical well-conditioned sigma_min (0.038,
    measured -- see module docstring), not an accidentally huge value that
    would perturb normal operation."""
    assert 0.0 < QD_ESTIMATE_DAMPING < 0.01
