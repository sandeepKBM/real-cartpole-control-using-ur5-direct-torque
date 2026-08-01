"""Offline supervised fit + honest held-out evaluation for the phase-1
residual-torque-regression pipeline (2026-08-01).

See ``docs/status/residual_torque_regression_pipeline_2026-08-01.md`` for the
full writeup, ``docs/status/nonlinear_controller_research_2026-07-31.md``
section 1 for why this direction was chosen over another RL attempt, and
``tools/analysis/residual_data.py`` for how ``tau_residual = M(q) @
qdd_residual`` training targets are reconstructed from existing (sim) trace
data.

What this script does, end to end:

1. Load one or more ``trace.jsonl`` files (glob patterns), reconstruct
   ``(q, qd) -> tau_residual`` rows per run.
2. Split by RUN (not by row) into train/test. Rows within a single 500 Hz
   rollout are extremely autocorrelated (adjacent rows differ by 2 ms of
   physics); a row-level random split would let near-duplicate rows leak
   between train and test and give a falsely optimistic score. Splitting by
   whole run is the honest choice at this data volume, even though it means
   the test set is only a few runs.
3. Featurize with ``controller_core.residual_torque_model.all_joint_features``
   -- the SAME function the deterministic-cost inference path would use, so
   the fit and any future real-time evaluation can never silently diverge.
4. Fit one ordinary-least-squares weight vector per joint
   (``numpy.linalg.lstsq`` -- no scipy/sklearn needed for plain OLS).
5. Evaluate on the held-out runs against a zero-residual (do-nothing)
   baseline, and report basic diagnostics about what the residual itself
   looks like (magnitude by joint, correlation with |qd|) regardless of fit
   quality -- useful even if the fit is bad.

Caveats this script itself prints, not just documents: the fitted weights
and reported R^2 are only as good as the (currently small, currently
sim-only) dataset they were fit on. Nothing here claims real-hardware
readiness.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from controller_core.residual_torque_model import (  # noqa: E402
    NUM_FEATURES_PER_JOINT,
    NUM_JOINTS,
    all_joint_features,
)
from tools.analysis.residual_data import ResidualDataset, ResidualDatasetRun, build_dataset  # noqa: E402


def _find_trace_files(patterns: list[str]) -> list[Path]:
    found: list[Path] = []
    for pattern in patterns:
        p = Path(pattern)
        if p.is_file():
            found.append(p)
            continue
        # Support both a directory (search recursively for trace.jsonl) and a glob.
        if p.is_dir():
            found.extend(sorted(p.rglob("trace.jsonl")))
            continue
        found.extend(sorted(Path().glob(pattern)))
    # De-dup, preserve order.
    seen: set[str] = set()
    unique: list[Path] = []
    for f in found:
        key = str(f.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return unique


def _split_runs(
    runs: list[ResidualDatasetRun], *, test_fraction: float, seed: int
) -> tuple[list[ResidualDatasetRun], list[ResidualDatasetRun]]:
    if len(runs) < 2:
        raise ValueError(f"Need at least 2 runs to hold one out for testing; got {len(runs)}")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(runs))
    n_test = max(1, int(round(len(runs) * test_fraction)))
    n_test = min(n_test, len(runs) - 1)  # always leave >=1 run for training
    test_idx = set(order[:n_test].tolist())
    train_runs = [runs[i] for i in range(len(runs)) if i not in test_idx]
    test_runs = [runs[i] for i in range(len(runs)) if i in test_idx]
    return train_runs, test_runs


def _featurize_runs(
    runs: list[ResidualDatasetRun], *, deadband: float
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (X, y): X shape (N, NUM_JOINTS, NUM_FEATURES_PER_JOINT), y shape (N, NUM_JOINTS)."""
    if not runs:
        return (
            np.zeros((0, NUM_JOINTS, NUM_FEATURES_PER_JOINT)),
            np.zeros((0, NUM_JOINTS)),
        )
    x_chunks = []
    y_chunks = []
    for run in runs:
        n = run.q.shape[0]
        feats = np.empty((n, NUM_JOINTS, NUM_FEATURES_PER_JOINT), dtype=np.float64)
        for i in range(n):
            feats[i] = all_joint_features(run.q[i], run.qd[i], deadband=deadband)
        x_chunks.append(feats)
        y_chunks.append(run.tau_residual)
    return np.concatenate(x_chunks, axis=0), np.concatenate(y_chunks, axis=0)


def fit_ols_weights(x_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    """Ordinary least squares, one joint at a time. Returns (NUM_JOINTS, NUM_FEATURES_PER_JOINT)."""
    weights = np.zeros((NUM_JOINTS, NUM_FEATURES_PER_JOINT), dtype=np.float64)
    for j in range(NUM_JOINTS):
        design = x_train[:, j, :]
        target = y_train[:, j]
        if design.shape[0] < NUM_FEATURES_PER_JOINT:
            # Not enough rows to fit this joint's features; leave weights at zero
            # (falls back to the zero-residual baseline for this joint only).
            continue
        w_j, *_ = np.linalg.lstsq(design, target, rcond=None)
        weights[j, :] = w_j
    return weights


def predict(weights: np.ndarray, x: np.ndarray) -> np.ndarray:
    """x: (N, NUM_JOINTS, NUM_FEATURES_PER_JOINT) -> (N, NUM_JOINTS)."""
    return np.einsum("njf,jf->nj", x, weights)


@dataclass
class JointEvalMetrics:
    joint_index: int
    n_rows: int
    rmse_zero_baseline: float
    rmse_model: float
    mae_zero_baseline: float
    mae_model: float
    r2_vs_zero_baseline: float
    residual_std: float
    residual_mean_abs: float
    corr_abs_residual_vs_abs_qd: float | None


def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def evaluate(
    runs: list[ResidualDatasetRun], weights: np.ndarray, *, deadband: float
) -> list[JointEvalMetrics]:
    x, y = _featurize_runs(runs, deadband=deadband)
    if x.shape[0] == 0:
        return []
    y_pred = predict(weights, x)
    qd_all = np.concatenate([r.qd for r in runs], axis=0) if runs else np.zeros((0, NUM_JOINTS))

    metrics: list[JointEvalMetrics] = []
    for j in range(NUM_JOINTS):
        y_j = y[:, j]
        pred_j = y_pred[:, j]
        err_model = y_j - pred_j
        err_zero = y_j  # zero-baseline prediction is always 0

        sse_model = float(np.sum(err_model**2))
        sse_zero = float(np.sum(err_zero**2))
        r2_vs_zero = float(1.0 - sse_model / sse_zero) if sse_zero > 1e-12 else float("nan")

        corr = _safe_pearson(np.abs(y_j), np.abs(qd_all[:, j])) if qd_all.shape[0] == y_j.shape[0] else None

        metrics.append(
            JointEvalMetrics(
                joint_index=j,
                n_rows=int(y_j.shape[0]),
                rmse_zero_baseline=float(np.sqrt(np.mean(err_zero**2))),
                rmse_model=float(np.sqrt(np.mean(err_model**2))),
                mae_zero_baseline=float(np.mean(np.abs(err_zero))),
                mae_model=float(np.mean(np.abs(err_model))),
                r2_vs_zero_baseline=r2_vs_zero,
                residual_std=float(np.std(y_j)),
                residual_mean_abs=float(np.mean(np.abs(y_j))),
                corr_abs_residual_vs_abs_qd=corr,
            )
        )
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--trace-root",
        nargs="+",
        required=True,
        help="One or more directories (searched recursively for trace.jsonl) or glob patterns.",
    )
    p.add_argument("--test-fraction", type=float, default=0.25, help="Fraction of RUNS held out for eval.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gap-cycles", type=int, default=1, help="Passed to JointAccelEstimator.")
    p.add_argument("--lowpass-alpha", type=float, default=1.0, help="Passed to JointAccelEstimator.")
    p.add_argument("--deadband", type=float, default=0.05, help="tanh deadband, rad/s (feature basis).")
    p.add_argument("--output-json", type=Path, default=None, help="Optional path to write the eval report.")
    p.add_argument("--output-weights", type=Path, default=None, help="Optional path to save fitted weights (.npy).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    trace_files = _find_trace_files(args.trace_root)
    if not trace_files:
        print(f"No trace.jsonl files found under {args.trace_root}", file=sys.stderr)
        return 1
    print(f"Found {len(trace_files)} trace file(s).")

    dataset: ResidualDataset = build_dataset(
        trace_files, gap_cycles=args.gap_cycles, lowpass_alpha=args.lowpass_alpha
    )
    if len(dataset.runs) < 2:
        print(
            f"Only {len(dataset.runs)} usable run(s) after loading -- need >=2 to hold one out.",
            file=sys.stderr,
        )
        return 1
    print(f"Loaded {len(dataset.runs)} usable run(s), {dataset.total_rows()} total rows.")

    train_runs, test_runs = _split_runs(dataset.runs, test_fraction=args.test_fraction, seed=args.seed)
    print(f"Train runs: {len(train_runs)} ({sum(r.n_rows_valid for r in train_runs)} rows)")
    print(f"Test runs:  {len(test_runs)} ({sum(r.n_rows_valid for r in test_runs)} rows)")
    for r in test_runs:
        print(f"  held out: {r.label} ({r.source_path})")

    x_train, y_train = _featurize_runs(train_runs, deadband=args.deadband)
    weights = fit_ols_weights(x_train, y_train)

    train_metrics = evaluate(train_runs, weights, deadband=args.deadband)
    test_metrics = evaluate(test_runs, weights, deadband=args.deadband)

    print("\n=== Train-set fit (in-sample; expect this to look better than test) ===")
    for m in train_metrics:
        print(
            f"  joint {m.joint_index}: rmse zero={m.rmse_zero_baseline:.5f} model={m.rmse_model:.5f} "
            f"R2_vs_zero={m.r2_vs_zero_baseline:+.3f} n={m.n_rows}"
        )

    print("\n=== Held-out test-set evaluation (the honest number) ===")
    for m in test_metrics:
        corr_str = f"{m.corr_abs_residual_vs_abs_qd:+.3f}" if m.corr_abs_residual_vs_abs_qd is not None else "n/a"
        print(
            f"  joint {m.joint_index}: rmse zero={m.rmse_zero_baseline:.5f} model={m.rmse_model:.5f} "
            f"R2_vs_zero={m.r2_vs_zero_baseline:+.3f} mean|residual|={m.residual_mean_abs:.5f} "
            f"corr(|res|,|qd|)={corr_str} n={m.n_rows}"
        )

    report = {
        "trace_root": args.trace_root,
        "n_trace_files_found": len(trace_files),
        "n_runs_loaded": len(dataset.runs),
        "n_train_runs": len(train_runs),
        "n_test_runs": len(test_runs),
        "test_run_labels": [r.label for r in test_runs],
        "test_run_sources": [r.source_path for r in test_runs],
        "gap_cycles": args.gap_cycles,
        "lowpass_alpha": args.lowpass_alpha,
        "deadband": args.deadband,
        "seed": args.seed,
        "train_metrics": [asdict(m) for m in train_metrics],
        "test_metrics": [asdict(m) for m in test_metrics],
        "weights": weights.tolist(),
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWrote report: {args.output_json}")
    if args.output_weights is not None:
        args.output_weights.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output_weights, weights)
        print(f"Wrote weights: {args.output_weights}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
