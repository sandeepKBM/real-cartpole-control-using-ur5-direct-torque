#!/usr/bin/env python3
"""Safe dry-run timing diagnostic for UR5e RTDE/control-loop readiness.

Default behavior is deliberately non-destructive:
- no robot connection
- no robot motion
- no controller side effects

The script benchmarks a local 500 Hz-style loop, reports timing statistics, and
captures enough metadata to make it obvious whether the host can hold a 2 ms
budget in pure Python. It does not claim that this proves RTDE hardware
readiness; it only measures the local scheduler/loop behavior.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any


DEFAULT_OVERRUN_THRESHOLD_S = 0.0025
REPO_ROOT = Path(__file__).resolve().parents[1]


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


def _stats_ns(values_ns: list[int]) -> dict[str, Any]:
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


def _cpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "cpu_count": os.cpu_count(),
    }
    try:
        cpuinfo_path = Path("/proc/cpuinfo")
        model_name = None
        if cpuinfo_path.exists():
            for line in cpuinfo_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.lower().startswith("model name"):
                    model_name = line.split(":", 1)[1].strip()
                    break
        if model_name:
            info["model_name"] = model_name
    except Exception:
        pass
    return info


def _run_dry_loop(
    *,
    frequency_hz: float,
    duration_s: float,
    overrun_threshold_s: float,
) -> dict[str, Any]:
    frequency_hz = float(frequency_hz)
    if not math.isfinite(frequency_hz) or frequency_hz <= 0.0:
        raise ValueError("frequency must be positive and finite")
    duration_s = float(duration_s)
    if not math.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("duration must be positive and finite")
    overrun_threshold_s = float(overrun_threshold_s)
    if not math.isfinite(overrun_threshold_s) or overrun_threshold_s <= 0.0:
        overrun_threshold_s = DEFAULT_OVERRUN_THRESHOLD_S

    period_ns = max(1, int(round(1e9 / frequency_hz)))
    period_s = period_ns / 1e9
    duration_ns = max(1, int(round(duration_s * 1e9)))
    overrun_threshold_ns = int(round(overrun_threshold_s * 1e9))

    start_ns = time.monotonic_ns()
    end_ns = start_ns + duration_ns
    deadline_ns = start_ns + period_ns

    samples: list[dict[str, Any]] = []
    cycle_intervals_ns: list[int] = []
    work_durations_ns: list[int] = []
    exceptions: list[dict[str, Any]] = []

    late_cycles = 0
    skipped_periods_total = 0
    max_consecutive_late_cycles = 0
    consecutive_late_cycles = 0
    max_lateness_ns = 0

    prev_cycle_start_ns: int | None = None
    cycle_index = 0
    wall_error: str | None = None

    try:
        while True:
            now_ns = time.monotonic_ns()
            if now_ns >= end_ns:
                break

            if now_ns < deadline_ns:
                sleep_s = (deadline_ns - now_ns) / 1e9
                if sleep_s > 0.0:
                    time.sleep(sleep_s)
                continue

            cycle_start_ns = time.monotonic_ns()
            lateness_ns = max(0, cycle_start_ns - deadline_ns)
            skipped_periods_this_cycle = 0
            if lateness_ns > 0:
                late_cycles += 1
                max_lateness_ns = max(max_lateness_ns, lateness_ns)
                consecutive_late_cycles += 1
                skipped_periods_this_cycle = int(lateness_ns // period_ns)
                skipped_periods_total += skipped_periods_this_cycle
            else:
                consecutive_late_cycles = 0
            max_consecutive_late_cycles = max(
                max_consecutive_late_cycles, consecutive_late_cycles
            )

            work_start_ns = time.monotonic_ns()
            # Intentionally minimal work: this benchmarks the loop overhead and
            # scheduler jitter without side effects.
            _ = cycle_index * 0
            work_end_ns = time.monotonic_ns()
            work_ns = work_end_ns - work_start_ns
            work_durations_ns.append(int(work_ns))

            cycle_interval_ns = None
            if prev_cycle_start_ns is not None:
                cycle_interval_ns = int(cycle_start_ns - prev_cycle_start_ns)
                cycle_intervals_ns.append(cycle_interval_ns)
            prev_cycle_start_ns = cycle_start_ns

            samples.append(
                {
                    "cycle_index": cycle_index,
                    "start_ns": int(cycle_start_ns),
                    "deadline_ns": int(deadline_ns),
                    "lateness_ns": int(lateness_ns),
                    "skipped_periods_this_cycle": int(skipped_periods_this_cycle),
                    "cycle_interval_ns": cycle_interval_ns,
                    "work_ns": int(work_ns),
                }
            )

            cycle_index += 1
            deadline_ns += (1 + int(skipped_periods_this_cycle)) * period_ns
    except Exception as exc:  # pragma: no cover - defensive guard for tooling
        wall_error = f"{type(exc).__name__}: {exc}"
        exceptions.append(
            {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )

    end_ns = time.monotonic_ns()
    elapsed_ns = end_ns - start_ns
    expected_cycles = int(math.floor(duration_ns / period_ns))
    completed_cycles = len(samples)
    overrun_count_2ms = sum(1 for d in cycle_intervals_ns if d > period_ns)
    overrun_count_threshold = sum(1 for d in cycle_intervals_ns if d > overrun_threshold_ns)

    report = {
        "mode": "dry_run",
        "requested": {
            "frequency_hz": frequency_hz,
            "duration_s": duration_s,
            "overrun_threshold_s": overrun_threshold_s,
        },
        "run": {
            "start_monotonic_ns": int(start_ns),
            "end_monotonic_ns": int(end_ns),
            "elapsed_ns": int(elapsed_ns),
            "elapsed_s": elapsed_ns / 1e9,
            "target_period_ns": int(period_ns),
            "target_period_s": period_s,
            "expected_cycles": expected_cycles,
            "completed_cycles": completed_cycles,
            "late_cycles": late_cycles,
            "missed_cycles_total": late_cycles,
            "skipped_periods_total": skipped_periods_total,
            "max_consecutive_late_cycles": max_consecutive_late_cycles,
            "max_lateness_ns": int(max_lateness_ns),
            "max_lateness_ms": max_lateness_ns / 1e6,
            "overrun_count_2ms": overrun_count_2ms,
            "overrun_count_threshold": overrun_count_threshold,
            "overrun_threshold_ns": int(overrun_threshold_ns),
            "overrun_threshold_ms": overrun_threshold_ns / 1e6,
            "exception_count": len(exceptions),
            "exception": wall_error,
        },
        "stats": {
            "cycle_interval": _stats_ns(cycle_intervals_ns),
            "work_duration": _stats_ns(work_durations_ns),
        },
        "samples": samples,
        "exceptions": exceptions,
        "cpu": _cpu_info(),
        "robot": {
            "robot_ip": None,
            "receive_only_requested": False,
            "no_motion_requested": True,
            "live_connection_attempted": False,
            "live_motion_attempted": False,
            "note": (
                "This diagnostic intentionally runs without robot connectivity. "
                "It measures local loop timing only."
            ),
        },
    }
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Dry-run UR5e RTDE timing diagnostic.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Run locally without robot I/O. This diagnostic stays in dry-run mode.",
    )
    p.add_argument("--frequency", type=float, default=500.0, help="Target loop frequency in Hz.")
    p.add_argument("--duration", type=float, default=60.0, help="Duration of the benchmark in seconds.")
    p.add_argument(
        "--robot-ip",
        default="",
        help="Reserved metadata for a future receive-only robot probe. Ignored in dry-run.",
    )
    p.add_argument(
        "--receive-only",
        action="store_true",
        help="Reserved metadata for a future robot receive-only probe. Ignored in dry-run.",
    )
    p.add_argument(
        "--no-motion",
        action="store_true",
        default=True,
        help="Never send robot motion commands. Always true in this diagnostic.",
    )
    p.add_argument(
        "--output",
        default="timing_report.json",
        help="Path to the JSON timing report.",
    )
    p.add_argument(
        "--overrun-threshold",
        type=float,
        default=DEFAULT_OVERRUN_THRESHOLD_S,
        help="Secondary overrun threshold in seconds (e.g. 0.0025 = 2.5 ms).",
    )
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    report = _run_dry_loop(
        frequency_hz=args.frequency,
        duration_s=args.duration,
        overrun_threshold_s=args.overrun_threshold,
    )
    report["robot"].update(
        {
            "robot_ip": str(args.robot_ip or ""),
            "receive_only_requested": bool(args.receive_only),
            "no_motion_requested": bool(args.no_motion),
        }
    )
    report["requested_cli"] = {
        "dry_run": bool(args.dry_run),
        "frequency": float(args.frequency),
        "duration": float(args.duration),
        "robot_ip": str(args.robot_ip or ""),
        "receive_only": bool(args.receive_only),
        "no_motion": bool(args.no_motion),
        "output": str(args.output),
        "overrun_threshold": float(args.overrun_threshold),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    run = report["run"]
    stats = report["stats"]
    print(f"wrote: {output_path}")
    print(
        "cycles="
        f"{run['completed_cycles']} expected={run['expected_cycles']} "
        f"late={run['late_cycles']} skipped_periods={run['skipped_periods_total']}"
    )
    print(
        "cycle_interval_ms: "
        f"mean={stats['cycle_interval']['mean_ms']}, "
        f"median={stats['cycle_interval']['median_ms']}, "
        f"p95={stats['cycle_interval']['p95_ms']}, "
        f"p99={stats['cycle_interval']['p99_ms']}, "
        f"max={stats['cycle_interval']['max_ms']}"
    )
    print(
        "work_ms: "
        f"mean={stats['work_duration']['mean_ms']}, "
        f"median={stats['work_duration']['median_ms']}, "
        f"p95={stats['work_duration']['p95_ms']}, "
        f"p99={stats['work_duration']['p99_ms']}, "
        f"max={stats['work_duration']['max_ms']}"
    )
    print(
        "overruns: "
        f">2ms={run['overrun_count_2ms']} "
        f">={run['overrun_threshold_ms']}ms={run['overrun_count_threshold']} "
        f"max_lateness_ms={run['max_lateness_ms']}"
    )
    if report["exceptions"]:
        print(f"exceptions={len(report['exceptions'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
