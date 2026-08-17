"""Pluggable black-box optimizers for the swing-up parameter searches.

One `minimize(...)` with a scipy-like result, so a script picks a backend by
name instead of hardcoding `differential_evolution` -- three of them did.

WHY A SECOND BACKEND. The searches are 6 static continuous parameters,
deterministic rollouts, scalar objective: a black-box optimization, NOT a
sequential decision problem. Differential evolution spends
``popsize * n_params`` individuals across ``maxiter`` generations -- 8,784
evaluations at ~25 s each, i.e. ~1 h per configuration even on 65 workers, and
it spends most of those confirming a basin it found early. TPE (Optuna) usually
locates a comparable optimum in a few hundred evaluations on a smooth 6-D space,
and copes better with our objective's discontinuous guard-trip regions than DE's
population dynamics do.

WHAT A FASTER OPTIMIZER DOES NOT DO, recorded so it is not over-sold: none of
the failures in this lane have been search failures. Measured 2026-08-16,
``k_e = 0`` and ``k_e = 50`` produced bit-identical results, and ``a_max`` swept
21x changed nothing -- flat responses caused by an inverted energy-law sign and
a mis-framed drift guard. A better optimizer searches a broken objective faster
and hides the flatness that made those bugs findable. Prefer a one-variable
sweep when a result looks insensitive; prefer a better optimizer when the
mechanics are known good and you are testing many configurations.

Both backends are given the SAME objective, bounds and seed so their results are
directly comparable; ``backend="de"`` reproduces the previous call exactly.
"""
from __future__ import annotations

import dataclasses
from typing import Callable, Sequence

import numpy as np


@dataclasses.dataclass
class SearchResult:
    """scipy.optimize.OptimizeResult-shaped, so call sites need no changes."""

    x: np.ndarray
    fun: float
    nfev: int
    backend: str


def minimize(
    objective: Callable[[Sequence[float]], float],
    bounds: Sequence[tuple[float, float]],
    *,
    backend: str = "de",
    maxiter: int = 40,
    popsize: int = 16,
    seed: int = 0,
    workers: int = 1,
    n_trials: int | None = None,
) -> SearchResult:
    """Minimize ``objective`` over ``bounds``.

    ``backend="de"``     scipy differential_evolution, unchanged behavior.
    ``backend="optuna"`` TPE. ``n_trials`` defaults to a budget matched to the
                         DE call's own scale but an order of magnitude smaller
                         (see the module docstring); pass it explicitly to
                         compare the two at EQUAL evaluation count, which is the
                         only fair comparison of optimizer quality.
    """
    backend = str(backend).lower()
    if backend == "de":
        from scipy.optimize import differential_evolution

        res = differential_evolution(
            objective, list(bounds), maxiter=maxiter, popsize=popsize, tol=1e-4,
            seed=seed, workers=workers, polish=False)
        n_ind = popsize * len(bounds)
        return SearchResult(np.asarray(res.x, dtype=np.float64), float(res.fun),
                            int(getattr(res, "nfev", n_ind * (maxiter + 1))), "de")

    if backend == "optuna":
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        budget = int(n_trials if n_trials is not None else max(200, 40 * len(bounds)))

        def _obj(trial):
            x = [trial.suggest_float(f"p{i}", lo, hi) for i, (lo, hi) in enumerate(bounds)]
            return float(objective(x))

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=max(20, 4 * len(bounds))),
        )
        # n_jobs uses threads; the objective releases the GIL inside MuJoCo, and
        # each trial recompiles its own model, so this parallelises acceptably
        # without the fork-pool that DE needs.
        study.optimize(_obj, n_trials=budget, n_jobs=max(1, int(workers)), show_progress_bar=False)
        best = study.best_trial
        x = np.array([best.params[f"p{i}"] for i in range(len(bounds))], dtype=np.float64)
        return SearchResult(x, float(best.value), len(study.trials), "optuna")

    raise ValueError(f"unknown search backend {backend!r}; known: de, optuna")
