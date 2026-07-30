"""Backtest: rigid CartesianMoveMonitor guard vs. two smooth/state-aware candidates.

Context (see docs/status/safety_envelope_backtest_2026-07-30.md for the full writeup):
a proposal is on the table to replace/augment hardware/safety.py's rigid, flat-threshold
guards (CartesianMoveMonitor's TCP speed/accel ceilings in particular) with a "smooth
funnel"-style envelope that tightens as some risk metric grows, instead of tripping the
instant a fixed number is crossed. The counter-argument already accepted as the standard
to beat: every REAL safety incident found in this project's history was a wrong-number or
missing-check bug, not a rigid-vs-smooth shape problem, and a badly-tuned smooth
interpolation can create a "hole" in the envelope that's worse than a rigid ceiling.

This script is the falsifiable test: replay REAL recorded guard trips from
outputs/hardware_transport/ (first-ever live UR5e motion tests, 2026-07-28, all real
RTDE telemetry -- no synthetic/hand-built data used anywhere in this script) through:
  (baseline) the CURRENT, unmodified hardware.safety.CartesianMoveMonitor -- imported
      directly, its finite-difference speed/accel/EMA math is reused verbatim, never
      reimplemented.
  (candidate A) a CBF-style shrinking bound on max_tcp_accel_mps2 / max_tcp_speed_mps,
      conditioned on move-phase timing (loosest during the commanded min-jerk move,
      smoothly tightening over a settle window once the move ends and the axis target
      goes static -- the risk metric is "how long has this cycle been in a phase where
      the robot is expected to be stationary").
  (candidate B) a cond(J)-scaled threshold, mirroring the exact log(cond(J))-space
      interpolation shape already validated in this codebase for
      controller_core.x_axis_cartesian_impedance.py's lambda_adaptive_regularization
      (see that module's `_scheduled_lambda_regularization`) -- same functional form,
      re-picked breakpoints (see CondJScaledCandidate docstring for why the breakpoints
      are NOT copied verbatim from that module).
  (candidate C) a growth-RATE-aware threshold on |qd| (QdGrowthRateCandidate, added
      2026-07-30 as a follow-up after candidates A and B): conditions on the recent
      per-cycle multiplicative growth trend of |qd| over a short window, not its
      instantaneous magnitude -- see that class's docstring for the full rationale
      and docs/status/safety_envelope_backtest_2026-07-30.md SS8 for the results.

All three candidates are implemented by mutating the SAME real `CartesianMoveMonitor.limits`
object in place before each `check()` call -- `check()` reads `self.limits.max_tcp_*`
fresh every call, so this drives genuinely novel threshold logic through the monitor's
real, unmodified internal state machine (position history, EMA smoothing, gap windowing)
rather than reimplementing any of that math.

IMPORTANT DATA CAVEAT (real, not a bug in this script -- read before trusting the
numbers): production's control loop only appends a trace row for a cycle AFTER that
cycle's `move_monitor.check()` call returns `ok=True`. The cycle whose `check()` call
actually failed and ended the run is therefore never written to trace.jsonl -- confirmed
by cross-referencing every one of the 21 real trip runs used here: `summary["steps"]`
always equals `len(trace_rows)` exactly, and `summary["sim_time_s"]` always equals
`steps * dt_s`, i.e. the tripping cycle is exactly one dt_s past the last logged row.
This script:
  1. Replays every LOGGED row (0..steps-1) through baseline and both candidates. Since
     these rows all passed the real baseline monitor in production, any candidate trip
     found here is either a genuine improvement (catching something earlier that baseline
     also would have flagged) or, if baseline's replay itself disagrees with the recorded
     "always ok" history, a bug in this harness -- the script asserts baseline agreement
     as a self-check and reports if it ever fails.
  2. Reconstructs the missing final (tripping) cycle's state by holding the last logged
     row's joint velocity constant for one more dt_s and mapping it to a Cartesian
     position delta via the real linear Jacobian (from
     controller_core.model_dynamics.PinocchioUR5eDynamics, the same dynamics provider
     already validated elsewhere in this codebase for MuJoCo parity) -- NOT by fabricating
     a value. The reconstructed cycle is then run through baseline (as a cross-check
     against the real recorded termination_reason value) and both candidates, using the
     exact same CartesianMoveMonitor.check() call every other cycle uses.
This is real-trace replay with one bounded, documented, physically-motivated
extrapolation for the one cycle production itself never logged -- not synthetic data.

Run with the mujoco_ur5e conda env (needs pinocchio for cond(J)):
    /common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python3 \
        experiments/safety_envelope_backtest.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from hardware.safety import CartesianMoveLimits, CartesianMoveMonitor  # noqa: E402
from controller_core.model_dynamics import PinocchioUR5eDynamics  # noqa: E402
from simulation.ur5e_mujoco_torque import x_profile_target  # noqa: E402

# Real hardware capture root -- read-only, outside this worktree (gitignored in the
# main repo, not checked into this worktree at all). Every run here is a real live
# UR5e RTDE session from thinkrobot, 2026-07-28 (see
# outputs/hardware_transport in the main repo / hardware_captures/2026-07-28_.../README.md).
HARDWARE_TRANSPORT_ROOT = Path(
    "/common/users/ss5772/real_Cartpole/outputs/hardware_transport"
)

RUN_DIRS = sorted(
    d for d in HARDWARE_TRANSPORT_ROOT.iterdir()
    if d.is_dir() and (d / "trace.jsonl").exists() and (d / "summary.json").exists()
)

_ACCEL_RE = re.compile(r"TCP acceleration ([\d.]+) m/s\^2 > ([\d.]+) m/s\^2")
_SPEED_RE = re.compile(r"TCP speed ([\d.]+) m/s > ([\d.]+) m/s")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def parse_trip(termination_reason: str) -> dict[str, Any] | None:
    """Extract (kind, observed_value, threshold_used) from a real termination_reason
    string, or None if it isn't a CartesianMoveMonitor accel/speed trip (e.g. a
    DeadlineMonitor trip -- out of scope for these two candidates, see report)."""
    m = _ACCEL_RE.search(termination_reason)
    if m:
        return {"kind": "accel", "observed": float(m.group(1)), "threshold": float(m.group(2))}
    m = _SPEED_RE.search(termination_reason)
    if m:
        return {"kind": "speed", "observed": float(m.group(1)), "threshold": float(m.group(2))}
    return None


# ---------------------------------------------------------------------------
# Candidate A: CBF-style shrinking bound, conditioned on move-phase timing.
# ---------------------------------------------------------------------------


class MoveTimingCbfCandidate:
    """Shrinks max_tcp_accel_mps2 / max_tcp_speed_mps smoothly from the baseline
    ceiling down to `floor_fraction` of it, as a function of how long the axis
    target has been static (i.e. how far past the commanded min-jerk move's end
    the current cycle is). Full ceiling throughout the commanded move (real
    intended motion legitimately accelerates there); tightens over
    `settle_window_s` once the move ends, since nothing should be accelerating
    once the target has stopped moving -- a real TCP-accel spike late in hold is
    categorically more suspicious than the same spike mid-move.

    This is a genuinely different risk axis from candidate B (wall-clock/move-phase
    distance-to-expected-quiescence, not kinematic conditioning) -- deliberately
    picked so the two candidates are not the same idea in two costumes.
    """

    name = "cbf_move_timing"

    def __init__(self, *, settle_window_s: float = 0.5, floor_fraction: float = 0.2) -> None:
        self.settle_window_s = float(settle_window_s)
        self.floor_fraction = float(floor_fraction)

    def scale(self, *, t_s: float, move_duration_s: float) -> float:
        if t_s <= move_duration_s:
            u = 0.0
        else:
            dwell = t_s - move_duration_s
            u = min(1.0, dwell / max(self.settle_window_s, 1e-9))
        return 1.0 - u * (1.0 - self.floor_fraction)

    def thresholds(
        self, *, t_s: float, move_duration_s: float, base_accel: float, base_speed: float
    ) -> tuple[float, float]:
        scale = self.scale(t_s=t_s, move_duration_s=move_duration_s)
        return base_accel * scale, base_speed * scale


# ---------------------------------------------------------------------------
# Candidate B: cond(J)-scaled threshold.
# ---------------------------------------------------------------------------


class CondJScaledCandidate:
    """Shrinks max_tcp_accel_mps2 / max_tcp_speed_mps smoothly as cond(J) grows,
    using the exact log(cond(J))-space linear interpolation shape already
    validated in this codebase for
    controller_core.x_axis_cartesian_impedance.py::CartesianImpedanceController.
    _scheduled_lambda_regularization (lambda_adaptive_regularization, landed
    2026-07-25/26): scale = 1 - clip(log_frac, 0, 1) * (1 - floor_fraction),
    log_frac = (log(cond) - log(cond_low)) / (log(cond_high) - log(cond_low)).

    The BREAKPOINTS (cond_low/cond_high) are deliberately NOT copied from that
    module's 1e4/1e8 -- those were tuned for a Tikhonov regularization epsilon
    on the task-space inertia matrix, a different physical quantity entirely.
    This candidate's breakpoints are picked from the empirical cond(J) range
    actually observed in this real dataset (see report): ~1e2 at ordinary
    transport poses, ~6e5 at the wrist_2=0 singularity the real position-mode
    divergence run sat at.
    """

    name = "cond_j_scaled"

    def __init__(
        self,
        dynamics: PinocchioUR5eDynamics,
        *,
        cond_low: float = 2.0e2,
        cond_high: float = 5.0e4,
        floor_fraction: float = 0.2,
    ) -> None:
        self.dynamics = dynamics
        self.cond_low = float(cond_low)
        self.cond_high = float(cond_high)
        self.floor_fraction = float(floor_fraction)

    def cond_of(self, q: np.ndarray) -> float:
        J = self.dynamics.jacobian(np.asarray(q, dtype=np.float64))
        return float(np.linalg.cond(J))

    def scale(self, cond: float) -> float:
        cond = max(float(cond), 1.0)
        cond_low = max(self.cond_low, 1.0)
        cond_high = max(self.cond_high, cond_low * (1.0 + 1e-9))
        log_frac = (np.log(cond) - np.log(cond_low)) / (np.log(cond_high) - np.log(cond_low))
        log_frac = float(np.clip(log_frac, 0.0, 1.0))
        return 1.0 - log_frac * (1.0 - self.floor_fraction)

    def thresholds(self, *, q: np.ndarray, base_accel: float, base_speed: float) -> tuple[float, float, float]:
        cond = self.cond_of(q)
        scale = self.scale(cond)
        return base_accel * scale, base_speed * scale, cond


# ---------------------------------------------------------------------------
# Candidate C: growth-rate-aware threshold (conditions on RATE OF CHANGE of a
# risk quantity, not its static magnitude).
# ---------------------------------------------------------------------------


class QdGrowthRateCandidate:
    """Shrinks max_tcp_accel_mps2 / max_tcp_speed_mps smoothly as recent |qd| shows
    a SUSTAINED multiplicative growth trend over a short window -- not as a function
    of |qd|'s (or cond(J)'s) instantaneous magnitude, which is exactly what made
    Candidate B (cond_j_scaled) nuisance-trip at the real, intentionally-used
    wrist_2=0 transport start pose: cond(J) is high there *always*, whether the run
    is fine or diverging, so a magnitude-only threshold can't tell the two apart.

    Risk metric: |qd| (joint-velocity Euclidean norm), not cond(J). Chosen
    deliberately over d(cond(J))/dt for two reasons, both empirical, not just
    theoretical preference: (1) it is exactly the quantity the real disqualifying
    divergence (position_20260728_150847) is documented as growing in --
    "wrist_1/wrist_3 joint velocities grow near-exponentially step over step (~0.31
    -> ~0.55 -> ~0.84 rad/s, roughly 1.6-1.8x per 8ms step)"
    (hardware_captures/2026-07-28_.../README.md item 4) -- so it is a direct,
    leading indicator of the actual failure mode, not a proxy for it; (2) it needs
    no Jacobian/pinocchio call, so it is available on every real telemetry cycle
    (position mode's `getActualQd()`) at zero extra compute cost, unlike a
    per-cycle cond(J) history which would need a full FK/Jacobian solve every
    cycle just to form the derivative.

    growth_rate() computes the geometric-mean per-cycle multiplicative growth
    factor over the last `window` cycles: (|qd|_now / |qd|_(t-window)) ** (1/window)
    - 1, with both endpoints floored at `qd_floor` (matches the real ~1e-4 rad/s
    stationary noise floor measured in
    hardware_captures/2026-07-28_.../stationary_noise_capture_154018_stats.json --
    without a floor, a genuinely-static pose's own noise ratio (e.g. 2e-4/1e-4) can
    read as "100% growth" and defeat the whole point of this candidate). A flat or
    shrinking |qd| trend (growth_rate <= r_low) gets the full baseline ceiling --
    this is what should let the real, static wrist_2=0 transport pose (Candidate
    B's nuisance-trip case) through untouched. A sustained trend at or above r_high
    (picked below the ~60-80% per-cycle growth documented in the one real
    disqualifying case, so it still tightens before that growth reaches the
    observed cycles) gets the floor ceiling; between the two, smooth linear
    interpolation in growth-rate space (not log space -- growth_rate is already a
    ratio-of-ratios quantity, unlike cond(J)'s raw magnitude, so a second log
    transform isn't motivated by anything measured here).
    """

    name = "qd_growth_rate"

    def __init__(
        self,
        *,
        window: int = 3,
        qd_floor: float = 5.0e-3,
        r_low: float = 0.05,
        r_high: float = 0.5,
        floor_fraction: float = 0.2,
    ) -> None:
        self.window = int(window)
        self.qd_floor = float(qd_floor)
        self.r_low = float(r_low)
        self.r_high = float(r_high)
        self.floor_fraction = float(floor_fraction)
        self._qd_hist: list[float] = []

    def reset(self) -> None:
        self._qd_hist = []

    def growth_rate(self, qd_norm: float) -> float:
        """Feeds one new |qd| sample in and returns the current growth-rate
        estimate. Returns 0.0 (treated as "no growth yet", i.e. full ceiling)
        until `window` samples of history have accumulated -- a fresh run/pose
        should not start out artificially tightened before there's any trend to
        measure."""
        self._qd_hist.append(float(qd_norm))
        if len(self._qd_hist) <= self.window:
            return 0.0
        del self._qd_hist[: -(self.window + 1)]
        oldest = max(self._qd_hist[0], self.qd_floor)
        newest = max(self._qd_hist[-1], self.qd_floor)
        n = len(self._qd_hist) - 1
        return float((newest / oldest) ** (1.0 / n) - 1.0)

    def scale(self, growth_rate: float) -> float:
        r = max(float(growth_rate), 0.0)
        r_low = max(self.r_low, 0.0)
        r_high = max(self.r_high, r_low + 1e-9)
        frac = float(np.clip((r - r_low) / (r_high - r_low), 0.0, 1.0))
        return 1.0 - frac * (1.0 - self.floor_fraction)

    def thresholds(self, *, qd: np.ndarray, base_accel: float, base_speed: float) -> tuple[float, float, float]:
        qd_norm = float(np.linalg.norm(np.asarray(qd, dtype=np.float64)))
        r = self.growth_rate(qd_norm)
        s = self.scale(r)
        return base_accel * s, base_speed * s, r


# ---------------------------------------------------------------------------
# Replay engine
# ---------------------------------------------------------------------------


class RunData:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.summary = json.loads((run_dir / "summary.json").read_text())
        self.rows = _read_jsonl(run_dir / "trace.jsonl")
        self.mode = "position" if run_dir.name.startswith("position_") else "direct_torque"
        self.frequency_hz = float(self.summary.get("frequency_hz", 125.0 if self.mode == "position" else 500.0))
        self.dt_s = 1.0 / self.frequency_hz
        self.move_duration_s = float(self.summary["move_duration_s"])
        self.duration_s = float(self.summary["duration_s"])
        self.target_x_delta = float(self.summary.get("target_x_delta_m", self.summary.get("target_x_delta", 0.0)))
        self.x0 = float(self.summary["initial_ee_pos"][0])
        self.steps = int(self.summary["steps"])
        self.trip = parse_trip(str(self.summary.get("termination_reason", "")))
        # Base CartesianMoveLimits actually enforced in THIS real run (the CLI
        # accel-override sessions used different max_tcp_accel_mps2 values across
        # attempts, e.g. 0.5 / 0.8 / 1.2 -- recovered from the recorded trip string
        # itself so baseline replay reproduces the exact real threshold in force).
        base_accel = 0.5
        base_speed = 0.05
        if self.trip is not None:
            if self.trip["kind"] == "accel":
                base_accel = self.trip["threshold"]
            elif self.trip["kind"] == "speed":
                base_speed = self.trip["threshold"]
        self.base_accel = base_accel
        self.base_speed = base_speed
        # Noise-robust accel filtering (CartesianMoveLimits.accel_gap_cycles /
        # speed_lowpass_alpha) is NOT recorded anywhere in summary.json for any of
        # these 21 runs, even though hardware/direct_torque_transport.py supports
        # CLI overrides for both -- see `calibrate_filtering` below and the report's
        # "data limitation" section for why several of the longer runs need this.
        self.accel_gap_cycles = 1
        self.speed_lowpass_alpha = 1.0
        self.filtering_calibrated = False


def _target_tcp_pose(run: RunData, t_s: float, row0_pose: np.ndarray) -> np.ndarray:
    target_x, _ = x_profile_target(
        "min_jerk_move_hold", run.x0, run.target_x_delta, t_s, run.duration_s,
        move_duration_s=run.move_duration_s,
    )
    target = row0_pose.copy()
    target[0] = target_x
    return target


def _extrapolate_final_row(run: RunData, dynamics: PinocchioUR5eDynamics) -> dict[str, Any]:
    """Reconstruct the one cycle production never logged (see module docstring):
    hold last logged row's qd constant for one more dt_s, map it to a Cartesian
    position delta via the real linear Jacobian at the last logged q."""
    last = run.rows[-1]
    q_last = np.asarray(last["q"], dtype=np.float64)
    qd_last = np.asarray(last["qd"], dtype=np.float64)
    J = dynamics.jacobian(q_last)
    cart_vel = J[:3, :] @ qd_last
    ee_pos_last = np.asarray(last["ee_pos"], dtype=np.float64)
    ee_pos_next = ee_pos_last + cart_vel * run.dt_s
    t_next = float(run.summary["sim_time_s"])
    assert abs(t_next - (last["time_s"] + run.dt_s)) < 1e-6, (
        f"sim_time_s ({t_next}) does not equal last logged row's time_s + dt_s "
        f"({last['time_s'] + run.dt_s}) -- extrapolation assumption violated for "
        f"{run.run_dir.name}"
    )
    tcp_pose_next = np.asarray(last["tcp_pose"], dtype=np.float64).copy()
    tcp_pose_next[:3] = ee_pos_next
    return {
        "time_s": t_next,
        "q": (q_last + qd_last * run.dt_s).tolist(),
        "qd": qd_last.tolist(),
        "tcp_pose": tcp_pose_next.tolist(),
        "orientation_error_norm": float(last.get("orientation_error_norm", 0.0)),
        "ee_pos": ee_pos_next.tolist(),
        "_reconstructed": True,
    }


def replay_baseline(run: RunData, rows: list[dict[str, Any]]) -> dict[str, Any]:
    limits = CartesianMoveLimits(
        max_tcp_accel_mps2=run.base_accel, max_tcp_speed_mps=run.base_speed, max_off_axis_drift_m=0.03,
        accel_gap_cycles=run.accel_gap_cycles, speed_lowpass_alpha=run.speed_lowpass_alpha,
    )
    monitor = CartesianMoveMonitor(limits)
    row0_pose = np.asarray(rows[0]["tcp_pose"], dtype=np.float64)
    monitor.set_start(row0_pose, move_axis_index=0)
    for i, row in enumerate(rows[1:], start=1):
        t_s = float(row["time_s"])
        decision = monitor.check(
            q=row["q"], qd=row["qd"], tcp_pose=row["tcp_pose"],
            target_tcp_pose=_target_tcp_pose(run, t_s, row0_pose),
            orientation_error_rad=float(row.get("orientation_error_norm", 0.0)),
            axis_target_moving=bool(t_s <= run.move_duration_s),
            dt_s=run.dt_s,
        )
        if not decision.ok:
            return {"tripped": True, "cycle": i, "time_s": t_s, "reason": decision.reason}
    return {"tripped": False, "cycle": None, "time_s": None, "reason": None}


def replay_candidate(
    run: RunData, rows: list[dict[str, Any]],
    candidate: MoveTimingCbfCandidate | CondJScaledCandidate | QdGrowthRateCandidate,
    dynamics: PinocchioUR5eDynamics,
) -> dict[str, Any]:
    limits = CartesianMoveLimits(
        max_tcp_accel_mps2=run.base_accel, max_tcp_speed_mps=run.base_speed, max_off_axis_drift_m=0.03,
        accel_gap_cycles=run.accel_gap_cycles, speed_lowpass_alpha=run.speed_lowpass_alpha,
    )
    monitor = CartesianMoveMonitor(limits)
    row0_pose = np.asarray(rows[0]["tcp_pose"], dtype=np.float64)
    monitor.set_start(row0_pose, move_axis_index=0)
    if hasattr(candidate, "reset"):
        # Stateful candidates (e.g. QdGrowthRateCandidate's short-window history)
        # must start fresh for every independent replay of a run -- the same
        # instance is reused across runs/final-cycle re-replays in main().
        candidate.reset()
    cond_trace: list[float] = []
    risk_trace: list[float] = []
    for i, row in enumerate(rows[1:], start=1):
        t_s = float(row["time_s"])
        if isinstance(candidate, MoveTimingCbfCandidate):
            accel_thr, speed_thr = candidate.thresholds(
                t_s=t_s, move_duration_s=run.move_duration_s,
                base_accel=run.base_accel, base_speed=run.base_speed,
            )
        elif isinstance(candidate, CondJScaledCandidate):
            accel_thr, speed_thr, cond = candidate.thresholds(
                q=np.asarray(row["q"], dtype=np.float64), base_accel=run.base_accel, base_speed=run.base_speed,
            )
            cond_trace.append(cond)
        else:
            accel_thr, speed_thr, growth_rate = candidate.thresholds(
                qd=np.asarray(row["qd"], dtype=np.float64), base_accel=run.base_accel, base_speed=run.base_speed,
            )
            risk_trace.append(growth_rate)
        monitor.limits.max_tcp_accel_mps2 = accel_thr
        monitor.limits.max_tcp_speed_mps = speed_thr
        decision = monitor.check(
            q=row["q"], qd=row["qd"], tcp_pose=row["tcp_pose"],
            target_tcp_pose=_target_tcp_pose(run, t_s, row0_pose),
            orientation_error_rad=float(row.get("orientation_error_norm", 0.0)),
            axis_target_moving=bool(t_s <= run.move_duration_s),
            dt_s=run.dt_s,
        )
        if not decision.ok:
            return {
                "tripped": True, "cycle": i, "time_s": t_s, "reason": decision.reason,
                "max_cond_seen": max(cond_trace) if cond_trace else None,
                "max_growth_rate_seen": max(risk_trace) if risk_trace else None,
            }
    return {
        "tripped": False, "cycle": None, "time_s": None, "reason": None,
        "max_cond_seen": max(cond_trace) if cond_trace else None,
        "max_growth_rate_seen": max(risk_trace) if risk_trace else None,
    }


def compare_cycle(baseline_cycle: int | None, candidate_cycle: int | None) -> str:
    if baseline_cycle is None and candidate_cycle is None:
        return "same (neither trips in logged rows)"
    if candidate_cycle is None:
        return "later (never trips in logged rows)"
    if baseline_cycle is None:
        return "earlier (baseline never trips in logged rows, candidate does)"
    if candidate_cycle < baseline_cycle:
        return f"earlier (cycle {candidate_cycle} vs {baseline_cycle})"
    if candidate_cycle > baseline_cycle:
        return f"later (cycle {candidate_cycle} vs {baseline_cycle})"
    return f"same (cycle {candidate_cycle})"


_GAP_GRID = (1, 2, 3, 4, 5, 8, 10, 16, 24, 32, 48, 64, 96, 128)
_ALPHA_GRID = (1.0, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05)


def calibrate_filtering(run: RunData, rows: list[dict[str, Any]]) -> bool:
    """Several of the 21 real runs used CLI --accel-gap-cycles/--speed-lowpass-alpha
    overrides (hardware/direct_torque_transport.py supports both) that are NOT recorded
    anywhere in summary.json -- confirmed by grep, no gap/alpha/lowpass key anywhere in
    any of these summaries. Replaying those runs with CartesianMoveLimits' un-filtered
    defaults (gap=1, alpha=1.0) produces a false self-check failure: real telemetry has
    a genuine several-micron jitter between consecutive samples that a 1/dt^2
    double-difference amplifies into a spurious multi-m/s^2 spike baseline never saw in
    production (production's real run logged hundreds of clean cycles past this point).

    This searches the smallest (gap, alpha) -- i.e. the LEAST filtering -- that makes
    baseline replay agree with the real recorded pass history over every logged row.
    This is a calibration of an unrecorded free parameter to match known-good ground
    truth, NOT a guess at the exact value actually used on the robot that night -- both
    candidates and baseline for this run always use whatever this function picks, so the
    comparison between them stays apples-to-apples regardless of whether the calibrated
    value matches the real forgotten CLI flag exactly. Returns True if the default
    (gap=1, alpha=1.0) already agreed (no calibration needed), False if some filtering
    had to be introduced to reach agreement (flagged in the report per-run).
    """
    for gap in _GAP_GRID:
        for alpha in _ALPHA_GRID:
            run.accel_gap_cycles = gap
            run.speed_lowpass_alpha = alpha
            if not replay_baseline(run, rows)["tripped"]:
                return gap == 1 and alpha == 1.0
    # Nothing in the grid worked -- leave at the loosest filtering tried and let the
    # self-check report the (now much smaller) residual mismatch honestly.
    run.accel_gap_cycles = _GAP_GRID[-1]
    run.speed_lowpass_alpha = _ALPHA_GRID[-1]
    return False


def main() -> None:
    dynamics = PinocchioUR5eDynamics()
    cbf = MoveTimingCbfCandidate()
    condj = CondJScaledCandidate(dynamics)
    qdgrowth = QdGrowthRateCandidate()

    print(f"Loaded {len(RUN_DIRS)} real hardware runs from {HARDWARE_TRANSPORT_ROOT}\n")

    results: list[dict[str, Any]] = []
    for run_dir in RUN_DIRS:
        run = RunData(run_dir)
        rows = run.rows
        if len(rows) < 1:
            print(f"SKIP {run_dir.name}: no logged rows")
            continue

        default_ok = calibrate_filtering(run, rows)
        run.filtering_calibrated = not default_ok

        base_result = replay_baseline(run, rows)
        cbf_result = replay_candidate(run, rows, cbf, dynamics)
        condj_result = replay_candidate(run, rows, condj, dynamics)
        qdgrowth_result = replay_candidate(run, rows, qdgrowth, dynamics)

        # Self-check: every logged row passed baseline in production; if this replay's
        # baseline disagrees, that's a harness bug, not a finding -- surface loudly.
        cal_note = "" if not run.filtering_calibrated else (
            f" [calibrated gap={run.accel_gap_cycles} alpha={run.speed_lowpass_alpha}]"
        )
        self_check = ("OK" + cal_note) if not base_result["tripped"] else (
            f"MISMATCH: baseline replay trips at logged cycle {base_result['cycle']} "
            f"({base_result['reason']}) but ALL logged rows passed in the real run"
            f"{cal_note}"
        )

        # Final (unlogged, reconstructed) tripping cycle.
        final_row = None
        final_baseline_recomputed = None
        final_candidates: dict[str, Any] = {}
        if not base_result["tripped"] and run.trip is not None:
            final_row = _extrapolate_final_row(run, dynamics)
            all_rows = rows + [final_row]

            fb = replay_baseline(run, all_rows)
            final_baseline_recomputed = fb

            for cand_name, cand in (
                ("cbf_move_timing", cbf), ("cond_j_scaled", condj), ("qd_growth_rate", qdgrowth),
            ):
                fc = replay_candidate(run, all_rows, cand, dynamics)
                final_candidates[cand_name] = fc

        results.append({
            "run": run_dir.name,
            "mode": run.mode,
            "steps": run.steps,
            "termination_reason": run.summary.get("termination_reason"),
            "trip": run.trip,
            "self_check": self_check,
            "filtering_calibrated": run.filtering_calibrated,
            "accel_gap_cycles": run.accel_gap_cycles,
            "speed_lowpass_alpha": run.speed_lowpass_alpha,
            "baseline_logged": base_result,
            "cbf_logged": cbf_result,
            "condj_logged": condj_result,
            "qdgrowth_logged": qdgrowth_result,
            "final_row": final_row,
            "final_baseline": final_baseline_recomputed,
            "final_candidates": final_candidates,
        })

    # ---- Report ----
    n_applicable = 0
    n_cbf_avoided = 0
    n_condj_avoided = 0
    n_qdgrowth_avoided = 0
    n_self_check_fail = 0

    print(f"{'run':45s} {'mode':13s} {'trip@steps':10s} {'kind':7s} {'observed':>9s} {'thr':>6s}  "
          f"{'cbf_final':10s} {'condj_final':12s} {'qdgrowth_final':14s} {'cond(J)':>10s} {'growth_r':>9s}  self_check")
    print("-" * 190)
    for r in results:
        if not r["self_check"].startswith("OK"):
            n_self_check_fail += 1
        trip = r["trip"]
        cbf_final_status = "n/a"
        condj_final_status = "n/a"
        qdgrowth_final_status = "n/a"
        cond_val = "n/a"
        growth_val = "n/a"
        if trip is not None and r["final_row"] is not None:
            n_applicable += 1
            fb = r["final_baseline"]
            cbf_fc = r["final_candidates"]["cbf_move_timing"]
            condj_fc = r["final_candidates"]["cond_j_scaled"]
            qdgrowth_fc = r["final_candidates"]["qd_growth_rate"]
            cbf_final_status = "TRIPS" if cbf_fc["tripped"] else "AVOIDED"
            condj_final_status = "TRIPS" if condj_fc["tripped"] else "AVOIDED"
            qdgrowth_final_status = "TRIPS" if qdgrowth_fc["tripped"] else "AVOIDED"
            cond_val = f"{condj_fc.get('max_cond_seen') or 0.0:.3e}"
            growth_val = f"{qdgrowth_fc.get('max_growth_rate_seen') or 0.0:.3f}"
            if not cbf_fc["tripped"]:
                n_cbf_avoided += 1
            if not condj_fc["tripped"]:
                n_condj_avoided += 1
            if not qdgrowth_fc["tripped"]:
                n_qdgrowth_avoided += 1
            # "missed" defined relative to whether baseline's own recomputation on the
            # reconstructed final cycle also trips (cross-check vs. the real documented
            # trip) -- if baseline-recomputed trips but a candidate doesn't, and the
            # underlying event is the genuine singularity divergence run, that's a MISS.
        kind_str = trip["kind"] if trip else "n/a"
        observed_str = f"{trip['observed']:.3f}" if trip else "n/a"
        threshold_str = f"{trip['threshold']:.3f}" if trip else "n/a"
        print(f"{r['run']:45s} {r['mode']:13s} {r['steps']:<10d} "
              f"{kind_str:7s} {observed_str:>9s} {threshold_str:>6s}  "
              f"{cbf_final_status:10s} {condj_final_status:12s} {qdgrowth_final_status:14s} "
              f"{cond_val:>10s} {growth_val:>9s}  {r['self_check']}")

    print("\n" + "=" * 100)
    print("KNOWN-GENUINE CATCH CHECK: position_20260728_150847 (real wrist_2=0 singularity")
    print("divergence -- Z climbing, wrist_1/wrist_3 |qd| growing ~1.6-1.8x per 8ms step,")
    print("documented in hardware_captures/2026-07-28_.../README.md item 4). Any candidate")
    print("that lets this one slide is DISQUALIFIED regardless of other results.")
    singularity = next((r for r in results if r["run"] == "position_20260728_150847"), None)
    if singularity is not None:
        cbf_fc = singularity["final_candidates"]["cbf_move_timing"]
        condj_fc = singularity["final_candidates"]["cond_j_scaled"]
        qdgrowth_fc = singularity["final_candidates"]["qd_growth_rate"]
        print(f"  baseline: TRIPS ({singularity['termination_reason']})")
        print(f"  cbf_move_timing final cycle: {'TRIPS' if cbf_fc['tripped'] else 'AVOIDED -- DISQUALIFYING MISS'}")
        print(f"  cond_j_scaled final cycle:   {'TRIPS' if condj_fc['tripped'] else 'AVOIDED -- DISQUALIFYING MISS'} "
              f"(cond(J) seen up to {condj_fc.get('max_cond_seen'):.3e})")
        print(f"  qd_growth_rate final cycle:  {'TRIPS' if qdgrowth_fc['tripped'] else 'AVOIDED -- DISQUALIFYING MISS'} "
              f"(growth rate seen up to {qdgrowth_fc.get('max_growth_rate_seen'):.3f} per cycle)")
        # Also check every logged row (0..steps-1) for early misses in this run --
        # the divergence was real and growing well before the final cycle.
        print(f"  logged-row replay -- baseline: {singularity['baseline_logged']}")
        print(f"  logged-row replay -- cbf:      {singularity['cbf_logged']}")
        print(f"  logged-row replay -- condj:    {singularity['condj_logged']}")
        print(f"  logged-row replay -- qdgrowth: {singularity['qdgrowth_logged']}")
    else:
        print("  RUN NOT FOUND -- cannot verify.")

    print("\n" + "=" * 100)
    print("CANDIDATE B NUISANCE-TRIP CHECK, RE-RUN AGAINST CANDIDATE C: does qd_growth_rate")
    print("also nuisance-trip on the real, benign position-mode runs at the wrist_2=0 pose")
    print("that Candidate B (cond_j_scaled) tripped on purely from static cond(J), even")
    print("though real qd/drift/orientation stayed bounded there? Checked directly against")
    print("real logged telemetry, not reasoned about.")
    for name in ("position_20260728_145539", "position_20260728_150316"):
        rr = next((r for r in results if r["run"] == name), None)
        if rr is None:
            print(f"  {name}: RUN NOT FOUND")
            continue
        condj_l = rr["condj_logged"]
        qdgrowth_l = rr["qdgrowth_logged"]
        print(f"  {name} (real trip: {rr['termination_reason']}):")
        print(f"    condj_logged (Candidate B):    tripped={condj_l['tripped']} "
              f"cycle={condj_l['cycle']} reason={condj_l['reason']}")
        print(f"    qdgrowth_logged (Candidate C): tripped={qdgrowth_l['tripped']} "
              f"cycle={qdgrowth_l['cycle']} reason={qdgrowth_l['reason']}")

    print("\n" + "=" * 100)
    print("SUMMARY")
    print(f"  Real hardware trip runs replayed: {len(results)}")
    print(f"  Self-check failures (harness disagrees with real recorded pass history): {n_self_check_fail}")
    print(f"  Runs with an accel/speed CartesianMoveMonitor trip (candidates applicable): {n_applicable}")
    print(f"  candidate A (cbf_move_timing):   avoided/graduated {n_cbf_avoided}/{n_applicable} real final trips")
    print(f"  candidate B (cond_j_scaled):     avoided/graduated {n_condj_avoided}/{n_applicable} real final trips")
    print(f"  candidate C (qd_growth_rate):    avoided/graduated {n_qdgrowth_avoided}/{n_applicable} real final trips")
    print(f"  Non-applicable runs (e.g. deadline_overrun -- not a CartesianMoveMonitor")
    print(f"  accel/speed check, out of scope for these candidates): {len(results) - n_applicable}")

    out_path = REPO_ROOT / "experiments" / "safety_envelope_backtest_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nFull per-run results written to {out_path}")


if __name__ == "__main__":
    main()
