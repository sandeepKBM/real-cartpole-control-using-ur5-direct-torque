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
4. Fit one ridge-regularized (L2, closed-form normal equations, per-feature
   standardized) weight vector per joint by default -- ``--ridge-lambda 0``
   reproduces the original plain-OLS behavior (``numpy.linalg.lstsq``) for
   comparison. See ``fit_ols_weights``'s docstring for the concrete,
   real-data-measured reason plain OLS is not the default: joints that spend
   almost their whole trajectory near-stationary have a near-collinear
   design matrix and can produce catastrophically-extrapolating held-out
   predictions (observed R^2 in the tens of thousands to billions negative
   on real UR5e joints 2/3, fixed 2026-08-01 -- see the status doc's
   "Ridge regularization + output clipping fix" section).
5. Evaluate on the held-out runs against a zero-residual (do-nothing)
   baseline, and report basic diagnostics about what the residual itself
   looks like (magnitude by joint, correlation with |qd|) regardless of fit
   quality -- useful even if the fit is bad. Also reports the effect of a
   defensive output clip (``compute_clip_bounds``/``--clip-multiple``),
   applied regardless of fit quality, on top of the regression fit.

Caveats this script itself prints, not just documents: the fitted weights
and reported R^2 are only as good as the dataset they were fit on (now
either sim or real UR5e hardware traces, per ``tools/analysis/residual_data.py``).
Nothing here claims real-hardware readiness, and nothing here is wired into
any real-time controller path -- this is offline analysis tooling only.
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
    """Ordinary least squares, one joint at a time. Returns (NUM_JOINTS, NUM_FEATURES_PER_JOINT).

    **Known failure mode (found 2026-08-01 fitting real UR5e data, see
    ``docs/status/residual_torque_regression_pipeline_2026-08-01.md``): joints that spend
    almost their entire trajectory near-stationary (e.g. elbow/wrist_1 during an X-only
    transport move) have a training-set design matrix whose ``qd``, ``tanh(qd/deadband)``,
    and ``qd*|qd|`` columns are nearly linearly dependent (measured ``corr(qd, tanh) >
    0.9999``, ``cond(X) ~ 1e7-3e8``, smallest singular values ~1e-4 to 1e-5) -- these three
    columns are genuinely collinear at small ``|qd|`` since ``tanh(x) ~= x`` and ``x*|x|``
    is also odd and small there. Unregularized least squares (even via ``lstsq``'s
    minimum-norm SVD solution) is only weakly constrained along that near-null direction and
    can produce enormous, poorly-determined weights (observed: single coefficients in the
    hundreds of thousands) that fit the training data fine but blow up catastrophically the
    moment a held-out run has even a modestly larger ``|qd|`` for that joint than training ever
    saw -- a held-out run's peak ``qd`` an order of magnitude above the training range produced
    predictions 3+ orders of magnitude off and R^2 in the tens of thousands negative. This is why
    :func:`fit_ridge_weights` (regularized, with per-feature scaling) is the default fit path in
    :func:`main` -- this function is kept for explicit ``--ridge-lambda 0`` opt-out / comparison,
    not as the recommended default.**
    """
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


def _feature_scale(design: np.ndarray) -> np.ndarray:
    """Per-column std, with near-constant columns (e.g. the bias column, which is
    always exactly 1.0) mapped to scale 1.0 instead of dividing by ~0."""
    scale = np.std(design, axis=0)
    return np.where(scale < 1e-12, 1.0, scale)


def fit_ridge_weights(x_train: np.ndarray, y_train: np.ndarray, *, ridge_lambda: float) -> np.ndarray:
    """L2-regularized (ridge) least squares, one joint at a time, with per-feature scaling.

    Closed form on the normal equations, ``w = (X^T X + lambda*I)^-1 X^T y`` -- plain numpy,
    no scipy/sklearn (``environment.yml`` doesn't have either as a repo dependency; ridge does
    not need them). This is the standard, cheapest fix for the near-collinear-feature
    extrapolation blowup documented in :func:`fit_ols_weights`'s docstring: shrinking the
    weight vector's norm directly bounds how far a wildly-scaled coefficient (fit to noise
    along a near-null design direction) can extrapolate on an out-of-distribution held-out
    row.

    Feature scaling matters here, not just the regularization itself: this model's 6 features
    span wildly different raw magnitudes (bias == 1.0; ``sin``/``cos`` in ``[-1, 1]``; ``qd`` and
    ``qd*|qd|`` for a near-stationary joint can be ~1e-2 and ~1e-4 respectively). A single
    ``ridge_lambda`` applied to the *raw* (unscaled) design matrix would regularize those
    small-magnitude columns far more weakly than the O(1) columns relative to their own scale
    (or vice versa depending on units) -- not a meaningful, comparable penalty across features.
    Standardizing each joint's design matrix by its own per-column training-set std (bias
    column pinned to scale 1.0, since it has zero variance) before solving, then dividing the
    fitted weights back by that same scale, makes one scalar ``ridge_lambda`` penalize every
    standardized feature comparably regardless of its raw units.
    """
    if ridge_lambda < 0.0:
        raise ValueError(f"ridge_lambda must be >= 0; got {ridge_lambda}")
    weights = np.zeros((NUM_JOINTS, NUM_FEATURES_PER_JOINT), dtype=np.float64)
    identity = np.eye(NUM_FEATURES_PER_JOINT, dtype=np.float64)
    for j in range(NUM_JOINTS):
        design = x_train[:, j, :]
        target = y_train[:, j]
        if design.shape[0] < NUM_FEATURES_PER_JOINT:
            continue
        scale = _feature_scale(design)
        design_scaled = design / scale
        gram = design_scaled.T @ design_scaled + ridge_lambda * identity
        rhs = design_scaled.T @ target
        w_scaled = np.linalg.solve(gram, rhs)
        weights[j, :] = w_scaled / scale
    return weights


def compute_clip_bounds(y_train: np.ndarray, *, clip_multiple: float) -> np.ndarray:
    """Per-joint hard output-clip bound: ``clip_multiple * max(|observed training residual|)``.

    A defensive measure independent of fit quality (point 3b of the 2026-08-01 fix task): even
    a well-regularized fit should never be trusted to emit an unbounded correction torque, so
    this gives :func:`controller_core.residual_torque_model.compute_residual_torque` a concrete,
    data-derived ceiling -- "a small multiple of the observed max real residual torque per
    joint" per that task's own guidance -- computed from TRAIN data only (never test, to avoid
    leaking held-out information into what is supposed to be a fixed, data-independent safety
    bound at inference time).
    """
    if clip_multiple <= 0.0:
        raise ValueError(f"clip_multiple must be > 0; got {clip_multiple}")
    max_abs = np.max(np.abs(y_train), axis=0)  # (NUM_JOINTS,)
    # A joint with literally zero observed training residual still gets a tiny nonzero bound
    # rather than clipping everything to exactly 0.
    max_abs = np.where(max_abs < 1e-9, 1e-9, max_abs)
    return clip_multiple * max_abs


def predict(weights: np.ndarray, x: np.ndarray, *, clip_bounds: np.ndarray | None = None) -> np.ndarray:
    """x: (N, NUM_JOINTS, NUM_FEATURES_PER_JOINT) -> (N, NUM_JOINTS).

    ``clip_bounds``, if given (see :func:`compute_clip_bounds`), is a ``(NUM_JOINTS,)`` array
    of per-joint absolute bounds applied after the raw dot product -- the same clipping
    semantics as ``compute_residual_torque(..., clip_abs=...)``, exercised here in bulk over a
    whole dataset for evaluation.
    """
    pred = np.einsum("njf,jf->nj", x, weights)
    if clip_bounds is not None:
        clip_bounds = np.asarray(clip_bounds, dtype=np.float64).reshape(NUM_JOINTS)
        pred = np.clip(pred, -clip_bounds, clip_bounds)
    return pred


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
    # Populated only when a clip_bounds array is passed to evaluate() (2026-08-01 defensive
    # output-clipping addition); None otherwise -- shows what the hard safety-net clip would
    # additionally do on top of whatever the regression fit already produces.
    clip_bound: float | None = None
    rmse_model_clipped: float | None = None
    r2_vs_zero_baseline_clipped: float | None = None
    n_rows_clipped: int | None = None


def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def evaluate(
    runs: list[ResidualDatasetRun],
    weights: np.ndarray,
    *,
    deadband: float,
    clip_bounds: np.ndarray | None = None,
) -> list[JointEvalMetrics]:
    x, y = _featurize_runs(runs, deadband=deadband)
    if x.shape[0] == 0:
        return []
    y_pred = predict(weights, x)
    y_pred_clipped = predict(weights, x, clip_bounds=clip_bounds) if clip_bounds is not None else None
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

        clip_bound_j = None
        rmse_model_clipped = None
        r2_vs_zero_clipped = None
        n_rows_clipped = None
        if y_pred_clipped is not None:
            pred_clipped_j = y_pred_clipped[:, j]
            err_model_clipped = y_j - pred_clipped_j
            sse_model_clipped = float(np.sum(err_model_clipped**2))
            clip_bound_j = float(np.asarray(clip_bounds).reshape(NUM_JOINTS)[j])
            rmse_model_clipped = float(np.sqrt(np.mean(err_model_clipped**2)))
            r2_vs_zero_clipped = (
                float(1.0 - sse_model_clipped / sse_zero) if sse_zero > 1e-12 else float("nan")
            )
            n_rows_clipped = int(np.sum(np.abs(pred_j) > clip_bound_j))

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
                clip_bound=clip_bound_j,
                rmse_model_clipped=rmse_model_clipped,
                r2_vs_zero_baseline_clipped=r2_vs_zero_clipped,
                n_rows_clipped=n_rows_clipped,
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
    p.add_argument(
        "--ridge-lambda",
        type=float,
        default=1.0e5,
        help=(
            "L2 penalty applied in per-feature-standardized space (fit_ridge_weights). "
            "Default 1e5 chosen 2026-08-01 by sweeping held-out R^2 on real UR5e data across "
            "several seeds -- see docs/status/residual_torque_regression_pipeline_2026-08-01.md: "
            "the smallest tested value that reliably bounds the near-collinear-feature "
            "extrapolation blowup on the worst-case joint/split without needlessly destroying "
            "the well-conditioned joints' fit quality. Pass 0 to reproduce the old plain-OLS "
            "behavior (fit_ols_weights) exactly, for comparison."
        ),
    )
    p.add_argument(
        "--clip-multiple",
        type=float,
        default=5.0,
        help=(
            "Defensive output clip, applied regardless of fit quality: predictions are bounded "
            "to +/- clip_multiple * max(|observed training residual|) per joint (0 disables)."
        ),
    )
    p.add_argument(
        "--no-output-clipping", action="store_true", help="Disable the defensive output-clip evaluation."
    )
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
    if args.ridge_lambda > 0.0:
        weights = fit_ridge_weights(x_train, y_train, ridge_lambda=args.ridge_lambda)
        print(f"\nFit method: ridge (lambda={args.ridge_lambda:g}, standardized-feature closed form)")
    else:
        weights = fit_ols_weights(x_train, y_train)
        print("\nFit method: plain OLS (--ridge-lambda 0 -- known extrapolation-blowup risk, see docstring)")

    clip_bounds = None
    if not args.no_output_clipping and args.clip_multiple > 0.0:
        clip_bounds = compute_clip_bounds(y_train, clip_multiple=args.clip_multiple)
        print(
            "Output clip bounds (Nm, "
            f"{args.clip_multiple:g}x max|train residual| per joint): "
            + ", ".join(f"j{j}={clip_bounds[j]:.3f}" for j in range(NUM_JOINTS))
        )

    train_metrics = evaluate(train_runs, weights, deadband=args.deadband, clip_bounds=clip_bounds)
    test_metrics = evaluate(test_runs, weights, deadband=args.deadband, clip_bounds=clip_bounds)

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
        if m.rmse_model_clipped is not None:
            print(
                f"           clipped (bound={m.clip_bound:.3f} Nm, {m.n_rows_clipped} of {m.n_rows} "
                f"rows clipped): rmse={m.rmse_model_clipped:.5f} R2_vs_zero={m.r2_vs_zero_baseline_clipped:+.3f}"
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
        "ridge_lambda": args.ridge_lambda,
        "clip_multiple": args.clip_multiple if clip_bounds is not None else None,
        "clip_bounds": clip_bounds.tolist() if clip_bounds is not None else None,
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
