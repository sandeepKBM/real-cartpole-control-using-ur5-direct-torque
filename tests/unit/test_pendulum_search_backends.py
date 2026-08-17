"""Pluggable search backends.

Ordered by what breaks worst if wrong:

  1. ``backend="de"`` is unchanged. Every existing search defaults to it and a
     silent change would invalidate results that are already recorded.
  2. Both backends receive the SAME objective/bounds/seed and return the same
     result shape, so a comparison between them measures the OPTIMIZER and not
     an accidental difference in problem setup.
  3. An unknown backend is fatal. A typo that silently fell back to the default
     would be indistinguishable from a successful run of the requested one --
     the exact failure mode this lane has hit four times.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.diagnostics.pendulum_search_backends import SearchResult, minimize  # noqa: E402

# Smooth-ish with a shallow ripple: enough structure that a bad optimizer misses
# the basin, cheap enough to run in a unit test.
BOUNDS = [(-3.0, 3.0), (-3.0, 3.0)]


def _f(x):
    return float((x[0] - 1.3) ** 2 + (x[1] + 0.7) ** 2 + 0.1 * np.sin(9.0 * x[0]))


def test_de_is_the_default_and_reports_itself():
    r = minimize(_f, BOUNDS, maxiter=15, popsize=8, seed=0)
    assert r.backend == "de"
    assert isinstance(r, SearchResult)


def test_de_is_deterministic_for_a_fixed_seed():
    """Guards the property every recorded DE result depends on."""
    a = minimize(_f, BOUNDS, backend="de", maxiter=15, popsize=8, seed=3)
    b = minimize(_f, BOUNDS, backend="de", maxiter=15, popsize=8, seed=3)
    np.testing.assert_array_equal(a.x, b.x)
    assert a.fun == b.fun


@pytest.mark.parametrize("backend,kw", [("de", dict(maxiter=25, popsize=10)),
                                        ("optuna", dict(n_trials=200))])
def test_both_backends_find_the_basin(backend, kw):
    r = minimize(_f, BOUNDS, backend=backend, seed=0, **kw)
    assert abs(r.x[0] - 1.3) < 0.35 and abs(r.x[1] + 0.7) < 0.35
    assert r.fun < 0.02
    assert r.nfev > 0


def test_result_shape_is_identical_across_backends():
    """A comparison is only meaningful if the two differ ONLY in the optimizer."""
    a = minimize(_f, BOUNDS, backend="de", maxiter=10, popsize=6, seed=1)
    b = minimize(_f, BOUNDS, backend="optuna", n_trials=80, seed=1)
    for r in (a, b):
        assert isinstance(r.x, np.ndarray) and r.x.shape == (2,)
        assert isinstance(r.fun, float) and isinstance(r.nfev, int)
    assert a.backend != b.backend


def test_unknown_backend_is_fatal_not_a_silent_fallback():
    with pytest.raises(ValueError, match="unknown search backend"):
        minimize(_f, BOUNDS, backend="cma-es-probably")


def test_optuna_respects_an_explicit_budget():
    """Needed for the only fair optimizer comparison: EQUAL evaluation counts."""
    r = minimize(_f, BOUNDS, backend="optuna", n_trials=60, seed=0)
    assert r.nfev == 60


# ============ 4. OPTUNA MUST ACTUALLY PARALLELISE ========================

def _slow(x):
    """Module-level so fork+Pool can pickle it; burns real CPU so a
    thread-based 'parallelism' shows up as no speedup at all."""
    import time
    t0 = time.time()
    while time.time() - t0 < 0.05:
        pass
    return float((x[0] - 1.3) ** 2 + (x[1] + 0.7) ** 2)


def test_optuna_uses_processes_not_threads():
    """study.optimize(n_jobs=...) is THREAD-based, and these objectives are
    Python-level rollout loops holding the GIL -- measured 114% total CPU with
    n_jobs=48, i.e. ~1.1x. That would make TPE SLOWER than DE in wall clock
    despite needing far fewer evaluations, turning the whole point of the
    backend upside down. Guards the ask/tell process pool."""
    import time
    t0 = time.time()
    minimize(_slow, BOUNDS, backend="optuna", n_trials=24, seed=0, workers=1)
    serial = time.time() - t0
    t0 = time.time()
    minimize(_slow, BOUNDS, backend="optuna", n_trials=24, seed=0, workers=6)
    parallel = time.time() - t0
    assert parallel < 0.5 * serial, (
        f"no real parallelism: {serial:.2f}s serial vs {parallel:.2f}s on 6 workers")
