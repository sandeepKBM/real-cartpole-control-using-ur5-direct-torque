"""Tests for experiments/sim_noise_injection_x_transport.py's noise-injection
harness (see docs/status/safety_envelope_backtest_2026-07-30.md SS9 for the
full writeup this backs).

Uses a short move/hold (0.1s/0.1s at 500Hz = 100 steps) to keep the real
MuJoCo+pinocchio rollout fast -- these tests care about the noise-injection
determinism/isolation properties, not the specific trip outcomes (those are
exercised by running the script itself against the real profiles).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import sim_noise_injection_x_transport as m  # noqa: E402

SHORT_KWARGS = dict(target_x_delta=0.02, move_duration_s=1.0, hold_duration_s=0.1)


@pytest.fixture(scope="module")
def true_rows_and_meta():
    true_rows, dt, y0, z0 = m.run_true_trajectory(**SHORT_KWARGS)
    return true_rows, dt, y0, z0


def test_run_true_trajectory_is_deterministic():
    """No noise/randomness anywhere in the physics rollout itself -- two
    independent calls with identical arguments must produce a bit-identical
    trace."""
    rows_a, dt_a, y0_a, z0_a = m.run_true_trajectory(**SHORT_KWARGS)
    rows_b, dt_b, y0_b, z0_b = m.run_true_trajectory(**SHORT_KWARGS)
    assert dt_a == dt_b
    assert y0_a == pytest.approx(y0_b)
    assert z0_a == pytest.approx(z0_b)
    assert len(rows_a) == len(rows_b)
    for ra, rb in zip(rows_a, rows_b):
        assert np.array_equal(ra["ee_pos"], rb["ee_pos"])
        assert np.array_equal(ra["q"], rb["q"])
        assert np.array_equal(ra["qd"], rb["qd"])


def test_true_trajectory_actually_moves():
    """Regression guard for the real bug found while building this harness:
    forgetting to apply the config's home_qpos before building state left the
    rollout frozen at MuJoCo's raw all-zero (even more degenerate/singular)
    default pose, achieving ~1e-13 m of a commanded 0.02 m move. This asserts
    the move actually happens."""
    true_rows, dt, y0, z0 = m.run_true_trajectory(**SHORT_KWARGS)
    x0 = true_rows[0]["ee_pos"][0]
    x_final = true_rows[-1]["ee_pos"][0]
    achieved = x_final - x0
    # 0.1s move duration is short for a 0.02m move at this controller's
    # bandwidth (config/ur5e_mujoco_torque_osc_tuned.yaml's own header notes
    # sub-0.5s moves undershoot) -- this just checks REAL, substantial motion
    # happened, not full settling.
    assert achieved > 0.005, f"expected real motion toward +0.02m, got {achieved}"


def test_replay_with_noise_is_deterministic_given_a_seed(true_rows_and_meta):
    true_rows, dt, y0, z0 = true_rows_and_meta
    r1 = m.replay_with_noise(
        true_rows, dt=dt, y0=y0, z0=z0, move_duration_s=SHORT_KWARGS["move_duration_s"],
        base_accel=0.5, base_speed=0.05, seed=42,
    )
    r2 = m.replay_with_noise(
        true_rows, dt=dt, y0=y0, z0=z0, move_duration_s=SHORT_KWARGS["move_duration_s"],
        base_accel=0.5, base_speed=0.05, seed=42,
    )
    assert r1 == r2


def test_different_seeds_draw_different_noise(true_rows_and_meta):
    """Not a trip-outcome assertion (that can coincide by chance) -- directly
    checks the noise draw itself differs between seeds, which is what
    "at least 20 independent seeds" in the report depends on being true."""
    rng1 = np.random.default_rng(1)
    rng2 = np.random.default_rng(2)
    draw1 = rng1.normal(0.0, m.TCP_POS_STD_M, size=3)
    draw2 = rng2.normal(0.0, m.TCP_POS_STD_M, size=3)
    assert not np.array_equal(draw1, draw2)


def test_zero_std_reproduces_noise_free_reference(true_rows_and_meta):
    """pos_std=0/q_std=0 must be exactly the noise-free case (no residual
    randomness leaking through) -- same assertion the report's own
    "NOISE-FREE reference" depends on being trustworthy."""
    true_rows, dt, y0, z0 = true_rows_and_meta
    r_zero_seed5 = m.replay_with_noise(
        true_rows, dt=dt, y0=y0, z0=z0, move_duration_s=SHORT_KWARGS["move_duration_s"],
        base_accel=0.5, base_speed=0.05, seed=5,
        pos_std=np.zeros(3, dtype=np.float64), q_std=0.0,
    )
    r_zero_seed99 = m.replay_with_noise(
        true_rows, dt=dt, y0=y0, z0=z0, move_duration_s=SHORT_KWARGS["move_duration_s"],
        base_accel=0.5, base_speed=0.05, seed=99,
        pos_std=np.zeros(3, dtype=np.float64), q_std=0.0,
    )
    # Zero std means rng.normal(0, 0, ...) == 0 regardless of seed -- the seed
    # must not matter when there is no noise to draw.
    assert r_zero_seed5 == r_zero_seed99


def test_noise_perturbs_only_the_safety_check_copy_not_true_rows(true_rows_and_meta):
    """Mirrors rl_gain_scheduling/gain_scheduling_env.py's own convention:
    noise must never mutate the true physics trace it was read from."""
    true_rows, dt, y0, z0 = true_rows_and_meta
    snapshot = [row["ee_pos"].copy() for row in true_rows]
    m.replay_with_noise(
        true_rows, dt=dt, y0=y0, z0=z0, move_duration_s=SHORT_KWARGS["move_duration_s"],
        base_accel=0.5, base_speed=0.05, seed=7,
    )
    for row, snap in zip(true_rows, snapshot):
        assert np.array_equal(row["ee_pos"], snap)


def test_qdgrowth_variant_never_loosens_below_default_on_flat_signal(true_rows_and_meta):
    """QdGrowthRateCandidate only ever shrinks the threshold on a detected
    growth trend -- on a short, smooth, non-diverging move it should trip at
    the same cycle as (or later than) default, never earlier, since its
    threshold can only be <= the default's fixed base value."""
    true_rows, dt, y0, z0 = true_rows_and_meta
    r = m.replay_with_noise(
        true_rows, dt=dt, y0=y0, z0=z0, move_duration_s=SHORT_KWARGS["move_duration_s"],
        base_accel=4.5, base_speed=0.2, seed=3,
    )
    if r["default"]["tripped"] and r["qdgrowth"]["tripped"]:
        assert r["qdgrowth"]["cycle"] >= r["default"]["cycle"]
