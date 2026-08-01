"""Offline data-loading + supervised-target construction for the
residual-torque-regression pipeline (phase 1, 2026-08-01).

See ``docs/status/residual_torque_regression_pipeline_2026-08-01.md`` for the
full writeup. Short version: the async residual observer
(``hardware/residual_observer_worker.py``) and its inline predecessor in
``hardware/direct_torque_transport.py`` already compute
``qdd_residual = qdd_measured - qdd_pred`` every cycle on the real
``direct_torque`` hardware path and log it to ``trace_rows``, but that wiring
does NOT exist in the sim rollout engine
(``tools/ur5e_mujoco_torque_experiments.py``) or its sweep driver
(``tools/ur5e_move_hold_transport.py``) -- confirmed by grepping both for
``qdd_residual``/``residual`` before writing this module. Those sim traces
*do* already log everything needed to reconstruct the same quantity
post-hoc, though: per-step ``q``, ``qd``, ``tau`` (the true final physical
torque delivered to ``data.ctrl`` that cycle -- see
``simulation/ur5e_mujoco_torque.py``'s ``apply_torque_components``, which
writes exactly this into the trace's ``"tau"`` field, not ``tau_controller``
or ``tau_applied`` which are pre-clip/pre-filter), and ``qfrc_bias`` (MuJoCo's
own ``C(q, qd) @ qd + g(q)``, which explicitly excludes joint
friction/damping -- those are ``qfrc_passive`` and are folded into the
residual by construction, which is exactly the "effective uncompensated
torque" this pipeline wants to see). No new trace fields, no new
instrumentation -- this module is pure post-hoc algebra on data that already
exists, mirroring what the real ``direct_torque`` observer computes online.

This module deliberately reuses the exact same building blocks the real
residual observer uses (``controller_core.dynamics_residual``,
``hardware.joint_accel_estimator.JointAccelEstimator``,
``controller_core.model_dynamics.PinocchioUR5eDynamics``) rather than
reimplementing any of that math, so a sim-reconstructed residual and a
real-hardware-logged residual are computed identically once real trace data
(with the observer's fields already populated) is available -- see
``tools/pull_hardware_logs_ssh.sh`` for how that would be pulled from
thinkrobot.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from controller_core.dynamics_residual import joint_acceleration_residual, predict_joint_acceleration
from controller_core.model_dynamics import PinocchioUR5eDynamics
from hardware.joint_accel_estimator import JointAccelEstimator

NUM_JOINTS = 6


def load_trace_rows(path: Path) -> list[dict]:
    """Read a ``trace.jsonl`` file into a list of per-step dicts, in order."""
    path = Path(path)
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


@dataclass
class ResidualDatasetRun:
    """One rollout's worth of reconstructed residual-torque training data."""

    source_path: str
    label: str
    q: np.ndarray  # (n, 6)
    qd: np.ndarray  # (n, 6)
    tau_residual: np.ndarray  # (n, 6) -- M(q) @ qdd_residual, the supervised target
    qdd_residual: np.ndarray  # (n, 6)
    qdd_residual_norm: np.ndarray  # (n,)
    t_s: np.ndarray  # (n,)
    n_rows_total: int  # includes gap-window warmup rows that were dropped
    n_rows_valid: int


def _required_fields_present(row: dict) -> bool:
    return all(key in row for key in ("q", "qd", "tau", "qfrc_bias")) and (
        "dt_s" in row or "time_s" in row
    )


def build_run_dataset(
    trace_path: Path,
    *,
    dynamics: PinocchioUR5eDynamics,
    label: str | None = None,
    gap_cycles: int = 1,
    lowpass_alpha: float = 1.0,
) -> ResidualDatasetRun | None:
    """Reconstruct ``(q, qd) -> tau_residual`` training rows from one trace file.

    Mirrors the real observer's own init/update order exactly
    (``hardware/direct_torque_transport.py``: ``residual_accel_estimator.reset(state0.qd)``
    once before the per-cycle loop begins, then ``update()`` every cycle
    starting from that same first state) -- see
    ``hardware/joint_accel_estimator.py``'s module docstring: at the default
    ``gap_cycles=1``, the reset sample itself already satisfies the gap
    window, so the very first row can produce a (possibly trivial, if qd
    hasn't changed yet) valid measurement. Returns ``None`` only if there is
    nothing usable at all (empty/malformed trace).
    """
    trace_path = Path(trace_path)
    rows = load_trace_rows(trace_path)
    rows = [r for r in rows if _required_fields_present(r)]
    if not rows:
        return None

    q0 = np.asarray(rows[0]["qd"], dtype=np.float64).reshape(NUM_JOINTS)
    estimator = JointAccelEstimator(gap_cycles=gap_cycles, lowpass_alpha=lowpass_alpha)
    estimator.reset(q0)

    q_list: list[np.ndarray] = []
    qd_list: list[np.ndarray] = []
    tau_residual_list: list[np.ndarray] = []
    qdd_residual_list: list[np.ndarray] = []
    t_list: list[float] = []

    prev_t: float | None = None
    for row in rows:
        q = np.asarray(row["q"], dtype=np.float64).reshape(NUM_JOINTS)
        qd = np.asarray(row["qd"], dtype=np.float64).reshape(NUM_JOINTS)
        tau = np.asarray(row["tau"], dtype=np.float64).reshape(NUM_JOINTS)
        qfrc_bias = np.asarray(row["qfrc_bias"], dtype=np.float64).reshape(NUM_JOINTS)
        t_s = float(row.get("time_s", 0.0))

        if "dt_s" in row and row["dt_s"] is not None:
            real_dt_s = float(row["dt_s"])
        elif prev_t is not None:
            real_dt_s = max(t_s - prev_t, 1e-6)
        else:
            real_dt_s = 1e-3  # arbitrary first-row placeholder; estimator drops this sample anyway
        prev_t = t_s

        qdd_measured = estimator.update(qd, real_dt_s)
        if qdd_measured is None:
            continue  # still filling the gap window

        if not (np.all(np.isfinite(q)) and np.all(np.isfinite(qd)) and np.all(np.isfinite(tau))):
            continue

        mass_matrix = dynamics.mass_matrix(q)
        qdd_pred = predict_joint_acceleration(mass_matrix, tau, qfrc_bias)
        qdd_residual = joint_acceleration_residual(qdd_measured, qdd_pred)
        tau_residual = mass_matrix @ qdd_residual

        if not np.all(np.isfinite(tau_residual)):
            continue

        q_list.append(q)
        qd_list.append(qd)
        tau_residual_list.append(tau_residual)
        qdd_residual_list.append(qdd_residual)
        t_list.append(t_s)

    if not q_list:
        return None

    q_arr = np.stack(q_list, axis=0)
    qd_arr = np.stack(qd_list, axis=0)
    tau_residual_arr = np.stack(tau_residual_list, axis=0)
    qdd_residual_arr = np.stack(qdd_residual_list, axis=0)
    t_arr = np.asarray(t_list, dtype=np.float64)

    return ResidualDatasetRun(
        source_path=str(trace_path),
        label=label if label is not None else trace_path.parent.name,
        q=q_arr,
        qd=qd_arr,
        tau_residual=tau_residual_arr,
        qdd_residual=qdd_residual_arr,
        qdd_residual_norm=np.linalg.norm(qdd_residual_arr, axis=1),
        t_s=t_arr,
        n_rows_total=len(rows),
        n_rows_valid=len(q_list),
    )


@dataclass
class ResidualDataset:
    """Multiple runs, kept separate (never mixed into one flat array without
    a run index) so callers can split train/test *by run* -- see
    ``fit_residual_torque_model.py``'s docstring for why row-level splitting
    would leak information across a highly autocorrelated time series."""

    runs: list[ResidualDatasetRun] = field(default_factory=list)

    def total_rows(self) -> int:
        return sum(r.n_rows_valid for r in self.runs)


def build_dataset(
    trace_paths: list[Path],
    *,
    gap_cycles: int = 1,
    lowpass_alpha: float = 1.0,
) -> ResidualDataset:
    """Build a :class:`ResidualDataset` from a list of ``trace.jsonl`` paths.

    One ``PinocchioUR5eDynamics`` instance is constructed once and reused
    across all runs (matches how the real observer amortizes this cost --
    see ``hardware/residual_observer_worker.py``'s module docstring for why
    construction, not steady-state use, is the expensive part).
    """
    dynamics = PinocchioUR5eDynamics()
    dataset = ResidualDataset()
    for trace_path in trace_paths:
        run = build_run_dataset(
            trace_path, dynamics=dynamics, gap_cycles=gap_cycles, lowpass_alpha=lowpass_alpha
        )
        if run is not None:
            dataset.runs.append(run)
    return dataset
