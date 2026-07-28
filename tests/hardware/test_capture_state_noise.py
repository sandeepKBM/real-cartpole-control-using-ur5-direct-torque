"""Tests for tools/ur5e_capture_state_noise.py's pure noise-stats function."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from ur5e_capture_state_noise import compute_noise_stats  # noqa: E402


def _row(q, qd, tcp_pose):
    return {"q": list(q), "qd": list(qd), "tcp_pose": list(tcp_pose)}


def test_compute_noise_stats_zero_for_constant_rows():
    q = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6]
    qd = [0.0] * 6
    tcp = [0.1, 0.2, 0.3, 0.0, 0.0, 0.0]
    rows = [_row(q, qd, tcp) for _ in range(20)]
    stats = compute_noise_stats(rows)
    assert stats["q_std_rad_max"] < 1e-12
    assert stats["qd_std_radps_max"] < 1e-12
    assert all(v < 1e-12 for v in stats["tcp_pos_std_m"])
    assert all(v < 1e-12 for v in stats["tcp_rot_std_rad"])


def test_compute_noise_stats_matches_known_std():
    rng = np.random.default_rng(0)
    base_q = np.array([0.0, -0.8, -1.2, -1.0, 0.0, 0.0])
    true_std = 0.001
    rows = []
    for _ in range(2000):
        q = base_q + rng.normal(0.0, true_std, size=6)
        rows.append(_row(q, np.zeros(6), np.zeros(6)))
    stats = compute_noise_stats(rows)
    # Sample std over 2000 draws should be close to the true generating std.
    for v in stats["q_std_rad_per_joint"]:
        assert abs(v - true_std) < true_std * 0.25
