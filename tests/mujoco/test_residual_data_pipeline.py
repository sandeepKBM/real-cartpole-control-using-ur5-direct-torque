"""Integration test for tools/analysis/residual_data.py -- the phase-1
residual-torque-regression pipeline's offline data-loading + target
construction (2026-08-01). See
docs/status/residual_torque_regression_pipeline_2026-08-01.md.

Marked ``mujoco`` per this directory's convention (auto-applied), though the
actual dependency exercised here is Pinocchio (via
``controller_core.model_dynamics.PinocchioUR5eDynamics``), matching the
existing ``tests/mujoco/test_direct_torque_residual_observer.py``'s home for
the same reason.

Uses small, hand-built synthetic trace files (not a real sim rollout) so the
expected residual is analytically known -- a real rollout's residual has no
simple closed form to assert against.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from controller_core.model_dynamics import PinocchioUR5eDynamics
from hardware.poses import HEIGHT_ALPHA_0_5_Q
from tools.analysis.residual_data import build_dataset, build_run_dataset

pytestmark = pytest.mark.mujoco


def _write_trace(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _constant_state_rows(*, q: np.ndarray, n_rows: int, dt_s: float, bias: np.ndarray) -> list[dict]:
    """Rows where qd is exactly constant (zero) and tau == bias exactly.

    Manipulator equation: M(q) qdd = tau - bias = 0, so qdd_pred == 0. qd is
    literally constant across every row, so a finite-difference qdd_measured
    is also exactly 0. Both are zero => tau_residual should be ~0 (up to
    floating point) everywhere once the gap window has filled.
    """
    rows = []
    qd = np.zeros(6)
    for i in range(n_rows):
        rows.append(
            {
                "time_s": i * dt_s,
                "dt_s": dt_s,
                "q": q.tolist(),
                "qd": qd.tolist(),
                "tau": bias.tolist(),
                "qfrc_bias": bias.tolist(),
            }
        )
    return rows


@pytest.fixture(scope="module")
def dynamics() -> PinocchioUR5eDynamics:
    return PinocchioUR5eDynamics()


def test_build_run_dataset_zero_residual_on_static_hold(tmp_path: Path, dynamics: PinocchioUR5eDynamics):
    q = np.asarray(HEIGHT_ALPHA_0_5_Q, dtype=np.float64)
    bias = dynamics.bias(q, np.zeros(6))
    trace_path = tmp_path / "trace.jsonl"
    _write_trace(trace_path, _constant_state_rows(q=q, n_rows=20, dt_s=0.002, bias=bias))

    run = build_run_dataset(trace_path, dynamics=dynamics, gap_cycles=1, lowpass_alpha=1.0)
    assert run is not None
    # At gap_cycles=1, JointAccelEstimator.reset() seeds the gap window with
    # the initial sample itself (matching hardware/direct_torque_transport.py's
    # real init order: reset(state0.qd) once, then update() every cycle
    # starting from that same state), so every row -- including the first --
    # produces a valid measurement here. No warmup rows are dropped.
    assert run.n_rows_valid == 20
    assert run.n_rows_total == 20
    np.testing.assert_allclose(run.tau_residual, 0.0, atol=1e-6)
    np.testing.assert_allclose(run.qdd_residual, 0.0, atol=1e-6)
    assert run.q.shape == (20, 6)
    assert run.qd.shape == (20, 6)


def test_build_run_dataset_nonzero_residual_when_bias_uncompensated(
    tmp_path: Path, dynamics: PinocchioUR5eDynamics
):
    """If the applied torque does NOT include the true bias (as if some
    disturbance/uncompensated dynamics were present), qdd_pred no longer
    matches the (still-zero, qd constant) measured qdd -- a real, nonzero
    residual should appear, and M(q) @ qdd_residual should be O(the missing
    torque), not O(numerical noise)."""
    q = np.asarray(HEIGHT_ALPHA_0_5_Q, dtype=np.float64)
    true_bias = dynamics.bias(q, np.zeros(6))
    missing_torque = np.array([0.0, 5.0, 0.0, 0.0, 0.0, 0.0])  # pretend joint 1 is under-compensated
    applied_tau = true_bias  # controller thinks this is enough...
    rows = _constant_state_rows(q=q, n_rows=20, dt_s=0.002, bias=applied_tau)
    # ...but qfrc_bias (the ground truth used for prediction) is actually higher.
    for row in rows:
        row["qfrc_bias"] = (true_bias + missing_torque).tolist()
    trace_path = tmp_path / "trace.jsonl"
    _write_trace(trace_path, rows)

    run = build_run_dataset(trace_path, dynamics=dynamics, gap_cycles=1, lowpass_alpha=1.0)
    assert run is not None
    # tau_residual = M(q) @ (qdd_measured - qdd_pred). qdd_measured=0 (qd
    # constant); qdd_pred = M^-1 @ (applied_tau - qfrc_bias) = M^-1 @
    # (-missing_torque) != 0, so tau_residual should reconstruct
    # +missing_torque exactly (up to floating point), joint 1 only.
    np.testing.assert_allclose(run.tau_residual[:, 1], missing_torque[1], atol=1e-6)
    for j in (0, 2, 3, 4, 5):
        np.testing.assert_allclose(run.tau_residual[:, j], 0.0, atol=1e-6)


def test_build_run_dataset_returns_none_for_rows_missing_required_fields(
    tmp_path: Path, dynamics: PinocchioUR5eDynamics
):
    # Every row is missing "qfrc_bias" (e.g. an older trace schema, or a sim
    # driver that never wired up this field) -- _required_fields_present
    # filters all of them out, leaving nothing to build a dataset from.
    rows = [
        {"time_s": 0.0, "dt_s": 0.002, "q": [0.0] * 6, "qd": [0.0] * 6, "tau": [0.0] * 6},
        {"time_s": 0.002, "dt_s": 0.002, "q": [0.0] * 6, "qd": [0.0] * 6, "tau": [0.0] * 6},
    ]
    trace_path = tmp_path / "trace.jsonl"
    _write_trace(trace_path, rows)
    run = build_run_dataset(trace_path, dynamics=dynamics, gap_cycles=1, lowpass_alpha=1.0)
    assert run is None


def test_build_run_dataset_returns_none_for_empty_trace(tmp_path: Path, dynamics: PinocchioUR5eDynamics):
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text("", encoding="utf-8")
    run = build_run_dataset(trace_path, dynamics=dynamics, gap_cycles=1, lowpass_alpha=1.0)
    assert run is None


def test_build_dataset_skips_unusable_files_and_keeps_usable_ones(
    tmp_path: Path, dynamics: PinocchioUR5eDynamics
):
    q = np.asarray(HEIGHT_ALPHA_0_5_Q, dtype=np.float64)
    bias = dynamics.bias(q, np.zeros(6))

    good_path = tmp_path / "good" / "trace.jsonl"
    _write_trace(good_path, _constant_state_rows(q=q, n_rows=10, dt_s=0.002, bias=bias))

    empty_path = tmp_path / "empty" / "trace.jsonl"
    empty_path.parent.mkdir(parents=True, exist_ok=True)
    empty_path.write_text("", encoding="utf-8")

    dataset = build_dataset([good_path, empty_path], gap_cycles=1, lowpass_alpha=1.0)
    assert len(dataset.runs) == 1
    assert dataset.runs[0].source_path == str(good_path)
    assert dataset.total_rows() == 10
