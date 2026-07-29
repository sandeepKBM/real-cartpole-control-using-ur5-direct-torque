"""Noise-robust joint-space qd -> qdd_measured estimator.

Diagnostic-only. This module feeds no safety trip condition anywhere -- see
``hardware/direct_torque_transport.py``'s residual-observer wiring and
``docs/status/direct_torque_residual_observer_2026-07-29.md``.

Ports the *technique* (not the code) from
``hardware.safety.CartesianMoveMonitor``'s TCP acceleration estimate
(2026-07-28 real-hardware noise-floor finding, see that class's docstring and
``CartesianMoveLimits.accel_gap_cycles``/``speed_lowpass_alpha``): a naive
single-cycle finite difference amplifies raw sensor noise by ``~1/dt``, and
using a sample from ``gap_cycles`` cycles back instead of 1 cycle shrinks that
amplification by ``~1/gap_cycles`` -- the noise between any two
approximately-independent samples doesn't grow with the gap, but the time
span in the denominator does.

One difference from the Cartesian case: TCP acceleration there is a *double*
finite difference of raw position (position -> speed -> accel), so
``CartesianMoveMonitor`` applies the gap-widening trick to the intermediate
*speed* signal (the stage that feeds the noise-doubling second difference).
Joint velocity (``qd``, RTDE ``getActualQd()``) is already a single,
directly-measured signal -- there is no intermediate "position" stage to
widen the gap on. ``qdd`` here needs only ONE differentiation, so this class
applies the identical gap-widening trick at that single step (a sample of
``qd`` from ``gap_cycles`` cycles back, divided by the real elapsed time
across that window), then an optional EMA low-pass layer
(``lowpass_alpha``) for further smoothing -- the direct joint-space analog of
``speed_lowpass_alpha``. At the defaults (``gap_cycles=1``,
``lowpass_alpha=1.0``) this reduces to the original single-cycle, unfiltered
finite difference.
"""

from __future__ import annotations

from collections import deque

import numpy as np


class JointAccelEstimator:
    """Gap-windowed, optionally EMA-low-pass-filtered ``qd -> qdd`` estimator.

    Call :meth:`reset` once with the initial ``qd`` sample, then :meth:`update`
    every control cycle with the freshly-read ``qd`` and the real elapsed time
    since the previous call. Returns ``None`` until enough history has
    accumulated to form a gap-windowed sample (exactly ``gap_cycles`` calls
    after ``reset``).
    """

    def __init__(self, *, gap_cycles: int = 1, lowpass_alpha: float = 1.0) -> None:
        gap_cycles = int(gap_cycles)
        if gap_cycles < 1:
            raise ValueError("gap_cycles must be >= 1")
        lowpass_alpha = float(lowpass_alpha)
        if not (0.0 < lowpass_alpha <= 1.0):
            raise ValueError("lowpass_alpha must be in (0.0, 1.0]")
        self.gap_cycles = gap_cycles
        self.lowpass_alpha = lowpass_alpha
        self._gap = gap_cycles
        self._corrected_clock_s = 0.0
        self._qd_history: deque[tuple[np.ndarray, float]] = deque(maxlen=self._gap)
        self._prev_qdd_filtered: np.ndarray | None = None

    def reset(self, qd0: np.ndarray) -> None:
        qd0 = np.asarray(qd0, dtype=np.float64).reshape(6)
        self._corrected_clock_s = 0.0
        self._qd_history.clear()
        self._qd_history.append((qd0.copy(), 0.0))
        self._prev_qdd_filtered = None

    def update(self, qd: np.ndarray, real_dt_s: float) -> np.ndarray | None:
        """Feed one new ``qd`` sample; returns ``qdd_measured`` (6,) or ``None``
        while still filling the initial gap window."""
        if not self._qd_history:
            raise RuntimeError("JointAccelEstimator.reset() must be called before update()")
        qd = np.asarray(qd, dtype=np.float64).reshape(6)
        real_dt_s = max(float(real_dt_s), 1e-9)
        new_clock_s = self._corrected_clock_s + real_dt_s

        qdd_out: np.ndarray | None = None
        if len(self._qd_history) >= self._gap:
            gap_qd, gap_clock_s = self._qd_history[0]
            gap_dt_s = max(new_clock_s - gap_clock_s, 1e-9)
            raw_qdd = (qd - gap_qd) / gap_dt_s

            if self._prev_qdd_filtered is None:
                qdd_filtered = raw_qdd
            else:
                qdd_filtered = (
                    self.lowpass_alpha * raw_qdd + (1.0 - self.lowpass_alpha) * self._prev_qdd_filtered
                )
            self._prev_qdd_filtered = qdd_filtered
            qdd_out = qdd_filtered.copy()

        self._corrected_clock_s = new_clock_s
        self._qd_history.append((qd.copy(), new_clock_s))
        return qdd_out
