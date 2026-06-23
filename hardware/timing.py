"""Loop timing helpers for staged UR5e RTDE bring-up."""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass, field
from typing import Any


def monotonic_ns() -> int:
    return time.monotonic_ns()


def period_from_frequency(frequency_hz: float) -> float:
    frequency_hz = float(frequency_hz)
    if not math.isfinite(frequency_hz) or frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive and finite")
    return 1.0 / frequency_hz


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    values_sorted = sorted(values)
    rank = (pct / 100.0) * (len(values_sorted) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return float(values_sorted[lo])
    frac = rank - lo
    return float(values_sorted[lo] * (1.0 - frac) + values_sorted[hi] * frac)


def compute_stats_ns(values_ns: list[int]) -> dict[str, Any]:
    if not values_ns:
        return {
            "count": 0,
            "mean_ns": None,
            "median_ns": None,
            "p95_ns": None,
            "p99_ns": None,
            "max_ns": None,
            "mean_ms": None,
            "median_ms": None,
            "p95_ms": None,
            "p99_ms": None,
            "max_ms": None,
        }
    values = [float(v) for v in values_ns]
    mean_ns = float(statistics.mean(values))
    median_ns = float(statistics.median(values))
    p95_ns = _percentile(values, 95.0)
    p99_ns = _percentile(values, 99.0)
    max_ns = float(max(values))
    return {
        "count": len(values_ns),
        "mean_ns": mean_ns,
        "median_ns": median_ns,
        "p95_ns": p95_ns,
        "p99_ns": p99_ns,
        "max_ns": max_ns,
        "mean_ms": mean_ns / 1e6,
        "median_ms": median_ns / 1e6,
        "p95_ms": None if p95_ns is None else p95_ns / 1e6,
        "p99_ms": None if p99_ns is None else p99_ns / 1e6,
        "max_ms": max_ns / 1e6,
    }


@dataclass
class TimingSample:
    cycle_index: int
    start_ns: int
    deadline_ns: int
    end_ns: int
    work_ns: int
    sleep_ns: int
    lateness_ns: int
    skipped_periods: int
    interval_ns: int | None = None


@dataclass
class TimingTracker:
    """Collect per-cycle timing metrics and emit a JSON-ready summary."""

    frequency_hz: float
    overrun_threshold_s: float = 0.0025
    samples: list[TimingSample] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.period_s = period_from_frequency(self.frequency_hz)
        self.period_ns = max(1, int(round(self.period_s * 1e9)))
        self.overrun_threshold_s = float(self.overrun_threshold_s)
        if not math.isfinite(self.overrun_threshold_s) or self.overrun_threshold_s <= 0.0:
            self.overrun_threshold_s = 0.0025
        self.overrun_threshold_ns = int(round(self.overrun_threshold_s * 1e9))

    def add_sample(
        self,
        *,
        cycle_index: int,
        start_ns: int,
        deadline_ns: int,
        end_ns: int,
        sleep_ns: int = 0,
        interval_ns: int | None = None,
    ) -> None:
        lateness_ns = max(0, int(start_ns - deadline_ns))
        skipped_periods = int(lateness_ns // self.period_ns) if lateness_ns > 0 else 0
        self.samples.append(
            TimingSample(
                cycle_index=int(cycle_index),
                start_ns=int(start_ns),
                deadline_ns=int(deadline_ns),
                end_ns=int(end_ns),
                work_ns=max(0, int(end_ns - start_ns)),
                sleep_ns=max(0, int(sleep_ns)),
                lateness_ns=int(lateness_ns),
                skipped_periods=skipped_periods,
                interval_ns=None if interval_ns is None else int(interval_ns),
            )
        )

    def summary(self) -> dict[str, Any]:
        cycle_interval_ns = [s.interval_ns for s in self.samples if s.interval_ns is not None]
        work_ns = [s.work_ns for s in self.samples]
        lateness_ns = [s.lateness_ns for s in self.samples]
        late_cycles = sum(1 for v in lateness_ns if v > 0)
        skipped_periods_total = sum(s.skipped_periods for s in self.samples)
        max_consecutive_late_cycles = 0
        consecutive_late_cycles = 0
        for v in lateness_ns:
            if v > 0:
                consecutive_late_cycles += 1
            else:
                consecutive_late_cycles = 0
            max_consecutive_late_cycles = max(max_consecutive_late_cycles, consecutive_late_cycles)
        return {
            "frequency_hz": float(self.frequency_hz),
            "target_period_ns": int(self.period_ns),
            "target_period_s": float(self.period_s),
            "overrun_threshold_ns": int(self.overrun_threshold_ns),
            "overrun_threshold_ms": self.overrun_threshold_ns / 1e6,
            "cycle_count": len(self.samples),
            "late_cycles": int(late_cycles),
            "skipped_periods_total": int(skipped_periods_total),
            "max_consecutive_late_cycles": int(max_consecutive_late_cycles),
            "max_lateness_ns": int(max(lateness_ns) if lateness_ns else 0),
            "max_lateness_ms": (max(lateness_ns) / 1e6) if lateness_ns else 0.0,
            "overrun_count_period_ns": int(sum(1 for v in cycle_interval_ns if v is not None and v > self.period_ns)),
            "overrun_count_threshold_ns": int(sum(1 for v in cycle_interval_ns if v is not None and v > self.overrun_threshold_ns)),
            "cycle_interval": compute_stats_ns([int(v) for v in cycle_interval_ns if v is not None]),
            "work_duration": compute_stats_ns(work_ns),
        }
