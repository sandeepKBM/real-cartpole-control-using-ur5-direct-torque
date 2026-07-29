"""Per-phase latency collection for direct-torque control loops."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .timing import compute_stats_ns


@dataclass
class PhaseLatencyRecorder:
    """Accumulate monotonic durations (nanoseconds) for one control loop."""

    read_state_ns: list[int] = field(default_factory=list)
    get_jacobian_ns: list[int] = field(default_factory=list)
    get_mass_matrix_ns: list[int] = field(default_factory=list)
    local_dynamics_ns: list[int] = field(default_factory=list)
    build_state_ns: list[int] = field(default_factory=list)
    controller_ns: list[int] = field(default_factory=list)
    safety_ns: list[int] = field(default_factory=list)
    # Diagnostic-only dynamics residual observer (2026-07-29, direct_torque
    # only) -- see docs/status/direct_torque_residual_observer_2026-07-29.md.
    # Not a safety phase; included here only so its per-cycle cost is visible
    # in the same latency breakdown as every other phase.
    residual_observer_ns: list[int] = field(default_factory=list)
    direct_torque_ns: list[int] = field(default_factory=list)
    sleep_ns: list[int] = field(default_factory=list)
    total_work_ns: list[int] = field(default_factory=list)
    lateness_ns: list[int] = field(default_factory=list)

    def record(self, name: str, duration_ns: int) -> None:
        bucket = getattr(self, name)
        if not isinstance(bucket, list):
            raise ValueError(f"unknown phase bucket {name!r}")
        bucket.append(max(0, int(duration_ns)))

    def summary(self) -> dict[str, Any]:
        phases = {
            "read_state": self.read_state_ns,
            "get_jacobian": self.get_jacobian_ns,
            "get_mass_matrix": self.get_mass_matrix_ns,
            "local_dynamics": self.local_dynamics_ns,
            "build_state": self.build_state_ns,
            "controller": self.controller_ns,
            "safety": self.safety_ns,
            "residual_observer": self.residual_observer_ns,
            "direct_torque": self.direct_torque_ns,
            "sleep": self.sleep_ns,
            "total_work": self.total_work_ns,
            "lateness": self.lateness_ns,
        }
        out: dict[str, Any] = {"phase_count": len(self.total_work_ns)}
        for name, values in phases.items():
            stats = compute_stats_ns(values)
            out[name] = stats
            out[f"{name}_mean_ms"] = stats["mean_ms"]
            out[f"{name}_p95_ms"] = stats["p95_ms"]
            out[f"{name}_max_ms"] = stats["max_ms"]
        if self.total_work_ns:
            total_mean = float(sum(self.total_work_ns)) / len(self.total_work_ns)
            out["dominant_phase"] = max(
                (
                    k
                    for k in (
                        "read_state",
                        "get_jacobian",
                        "get_mass_matrix",
                        "local_dynamics",
                        "build_state",
                        "controller",
                        "safety",
                        "residual_observer",
                        "direct_torque",
                    )
                    if getattr(self, f"{k}_ns")
                ),
                key=lambda k: float(sum(getattr(self, f"{k}_ns"))) / len(self.total_work_ns),
            )
            out["total_work_mean_ms"] = total_mean / 1e6
        return out
