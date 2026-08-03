"""Bounded world-X move+hold via PolyScope ``direct_torque()`` and the tuned OSC law."""

from __future__ import annotations

import gc
import json
import time
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from controller_core.dynamics_residual import joint_acceleration_residual, predict_joint_acceleration
from controller_core.model_dynamics import PinocchioUR5eDynamics
from controller_core.safety import ImpedanceSafetyConfig, ImpedanceSafetyMonitor
from controller_core.x_axis_cartesian_impedance import (
    CartesianImpedanceConfig,
    XAxisCartesianImpedanceController,
)
from simulation.ur5e_mujoco_torque import accel_duration_displacement, x_profile_accel, x_profile_target
from transport_metrics import compute_valid_move_hold_metrics, summarize_move_hold_trace

from .direct_torque_link import UR5eDirectTorqueLink
from .joint_accel_estimator import JointAccelEstimator
from .latency import PhaseLatencyRecorder
from .link import RTDEStateError
from .local_dynamics import LocalPinocchioDynamics, LocalPinocchioFastDynamics, normalize_dynamics_source
from .residual_observer_worker import (
    ResidualAsyncSummary,
    ResidualObserverWorkerHandle,
    start_residual_observer_worker,
)
from .safety import (
    CartesianMoveLimits,
    CartesianMoveMonitor,
    DeadlineMonitor,
    EStopLatch,
    StaleStateMonitor,
    UR5eSafetyLimits,
    is_robot_safety_normal,
)
from .telemetry_gap_bridge import TelemetryGapBridge
from .timing import TimingTracker, monotonic_ns
from .transport_common import impedance_safety_config_from_section, max_abs_qd_from_trace


@dataclass
class DirectTorqueTransportResult:
    ok: bool
    reason: str
    summary: dict[str, Any]
    trace_path: Path | None


# Pre-trip diagnostic trend reporting (2026-07-31). Motivated by a real
# incident diagnosing a real-hardware guard trip: the trend of qd/x_error/
# tau/orientation error in the ~60 cycles before the trip required manually
# re-parsing trace.jsonl by hand to see. Capture that trend automatically
# instead. Kept intentionally cheap and simple -- see
# run_x_transport_direct_torque's per-cycle loop for where the bounded
# window is appended (unconditionally, once per cycle, from values already
# computed this cycle) and where this trend is classified (once, only at
# trip time).
PRE_TRIP_TREND_WINDOW_CYCLES = 60


def _classify_trend(values: "list[float] | tuple[float, ...]", deadband_frac: float = 0.05) -> str:
    """Cheap trend heuristic: compare the mean of the first third of the
    window to the mean of the last third, with a small relative deadband for
    "stable". Deliberately not linear regression -- this only ever runs once,
    at trip time, so cost isn't the concern; simplicity/readability is (see
    the module comment above).

    Two deadband conditions, either one is enough to call it "stable": the
    original relative-to-mean check, plus an absolute one against the
    window's own noise (std of all values). The relative-only check
    collapses for a signal that legitimately hovers near zero with no real
    trend (e.g. y_drift_m/z_drift_m during a clean segment) -- found
    2026-08-01 via a Kalman-filtering follow-up investigation: under pure
    noise with a near-zero mean, the relative check misclassified
    rising/falling almost every time, since a tiny absolute noise fluctuation
    is a huge fraction of an already-tiny mean. The absolute check catches
    exactly that case without needing a per-channel hardcoded epsilon."""
    values = list(values)
    n = len(values)
    if n < 2:
        return "insufficient_data"
    third = max(1, n // 3)
    first_mean = float(np.mean(values[:third]))
    last_mean = float(np.mean(values[-third:]))
    change = last_mean - first_mean
    scale = max(abs(first_mean), abs(last_mean), 1e-9)
    rel_change = change / scale
    noise_floor = float(np.std(values))
    if abs(rel_change) < deadband_frac or abs(change) < 2.0 * noise_floor:
        return "stable"
    return "rising" if change > 0.0 else "falling"


def _build_pre_trip_trend(
    window: "deque[tuple[float, float, float, float, float, float, float]]", termination_reason: str
) -> dict[str, Any] | None:
    """Snapshot the rolling per-cycle window into a trend summary -- only
    when a guard actually tripped (``termination_reason`` isn't
    ``"duration_complete"``) and there's at least one cycle recorded. Returns
    None for a clean run, so this is purely additive to summary.json's shape.

    y_drift_m/z_drift_m added 2026-07-31 after diagnosing a real -0.15m
    return-leg trip: the off-axis (Y/Z) components implicated by AGENTS.md's
    documented directional-ceiling/nullspace-projector-asymmetry finding
    weren't previously in this window, only x_error/orientation were --
    manual re-derivation from ee_pos against initial_ee_pos was needed each
    time to check them."""
    if termination_reason == "duration_complete" or len(window) == 0:
        return None
    qd_vals, speed_vals, xerr_vals, tau_vals, orient_vals, ydrift_vals, zdrift_vals = zip(*window)
    return {
        "window_cycles": len(window),
        "qd_max_radps": {"values": list(qd_vals), "trend": _classify_trend(qd_vals)},
        "tcp_speed_mps": {"values": list(speed_vals), "trend": _classify_trend(speed_vals)},
        "x_error_m": {"values": list(xerr_vals), "trend": _classify_trend(xerr_vals)},
        "tau_controller_l1": {"values": list(tau_vals), "trend": _classify_trend(tau_vals)},
        "orientation_error_norm_rad": {"values": list(orient_vals), "trend": _classify_trend(orient_vals)},
        "y_drift_m": {"values": list(ydrift_vals), "trend": _classify_trend(ydrift_vals)},
        "z_drift_m": {"values": list(zdrift_vals), "trend": _classify_trend(zdrift_vals)},
    }


def _load_impedance_bundle(config_path: Path) -> tuple[CartesianImpedanceConfig, ImpedanceSafetyConfig, float]:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    ctrl = cfg.get("controller", {}) or {}
    safety_raw = ctrl.get("safety", {}) or {}
    impedance_cfg = CartesianImpedanceConfig.from_controller_yaml_section(ctrl)
    safety_cfg = impedance_safety_config_from_section(safety_raw)
    frequency_hz = float(cfg.get("hardware", {}).get("rtde_frequency_hz", 500.0))
    return impedance_cfg, safety_cfg, frequency_hz


def run_x_transport_direct_torque(
    link: UR5eDirectTorqueLink,
    *,
    config_path: Path,
    target_x_delta_m: float,
    move_duration_s: float,
    duration_s: float,
    output_dir: Path | None = None,
    motion_opt_in: bool,
    trajectory_profile: str = "min_jerk_move_hold",
    target_accel_mps2: float | None = None,
    record_latency: bool = True,
    dynamics_source: str = "rtde",
    coriolis_feedforward: bool = False,
    gain_overrides: dict[str, float] | None = None,
    max_tcp_accel_mps2_override: float | None = None,
    max_tcp_speed_mps_override: float | None = None,
    accel_gap_cycles_override: int | None = None,
    speed_lowpass_alpha_override: float | None = None,
    speed_limit_gap_cycles_override: int | None = None,
    speed_limit_lowpass_alpha_override: float | None = None,
    accel_max_consecutive_violations_override: int | None = None,
    accel_hard_multiple_override: float | None = None,
    speed_max_consecutive_violations_override: int | None = None,
    speed_hard_multiple_override: float | None = None,
    accel_variable_tolerance_override: bool | None = None,
    speed_variable_tolerance_override: bool | None = None,
    enable_residual_observer: bool = True,
    residual_qdd_gap_cycles: int = 1,
    residual_qdd_lowpass_alpha: float = 1.0,
    residual_observer_async: bool = False,
    telemetry_gap_bridge: bool = False,
    telemetry_gap_bridge_max_cycles: int = 2,
) -> DirectTorqueTransportResult:
    if not motion_opt_in:
        raise ValueError("motion_opt_in must be True for a live direct-torque transport")

    dynamics_source = normalize_dynamics_source(dynamics_source)
    if coriolis_feedforward and dynamics_source not in ("local", "local_pinocchio"):
        # The robot's own firmware compensates gravity inside directTorque()
        # (never add that in Python -- see AGENTS.md), but it does NOT
        # automatically apply Coriolis/centrifugal compensation the way it
        # does gravity; it only exposes the values for retrieval (confirmed
        # against Universal Robots' own Direct Torque Control documentation,
        # 2026-07-26). This flag adds that missing term via the selected
        # local dynamics provider's coriolis(), which needs dynamics_source
        # in {local, local_pinocchio}'s q/qd -> J/M/C pipeline; the rtde
        # dynamics_source path has no equivalent Coriolis getter wired up yet.
        raise ValueError(
            "coriolis_feedforward requires dynamics_source in {'local', 'local_pinocchio'} "
            "(rtde path not implemented)"
        )
    if telemetry_gap_bridge and not enable_residual_observer:
        # The bridge reuses the same dynamics-provider machinery as the
        # residual observer (a dedicated PinocchioUR5eDynamics instance,
        # independent of dynamics_source -- see the residual_dynamics
        # construction below) to forward-predict through a detected RTDE
        # duplicate. See hardware/telemetry_gap_bridge.py's module docstring
        # for the full design and scope.
        raise ValueError("telemetry_gap_bridge requires enable_residual_observer=True")
    if telemetry_gap_bridge and residual_observer_async:
        # The bridge needs a synchronous qdd prediction available within the
        # same cycle it guards; residual_observer_async defers that
        # computation to a worker process and only merges results back after
        # the loop exits, which cannot feed a same-cycle safety check.
        raise ValueError("telemetry_gap_bridge requires residual_observer_async=False")
    if dynamics_source == "local":
        local_dynamics = LocalPinocchioDynamics()
    elif dynamics_source == "local_pinocchio":
        # Opt-in fast path (2026-07-29): a genuinely Pinocchio-backed J(q)/M(q)/
        # Coriolis provider, ~10x lower per-call latency than the MuJoCo-backed
        # LocalPinocchioDynamics/LocalMujocoDynamics default -- see
        # docs/status/local_dynamics_speedup_investigation_2026-07-29.md.
        # Default behavior (dynamics_source="local" or "rtde") is unchanged.
        local_dynamics = LocalPinocchioFastDynamics()
    else:
        local_dynamics = None

    impedance_cfg, safety_cfg, frequency_hz = _load_impedance_bundle(config_path)
    if abs(frequency_hz - 500.0) > 1.0:
        raise ValueError(
            f"direct_torque transport expects ~500 Hz RTDE; config requested {frequency_hz} Hz"
        )
    dt_s = 1.0 / frequency_hz
    move_duration_s = float(move_duration_s)
    duration_s = float(duration_s)
    if move_duration_s <= 0.0 or duration_s <= 0.0:
        raise ValueError("move_duration_s and duration_s must be positive")
    if move_duration_s > duration_s:
        raise ValueError("move_duration_s must not exceed duration_s")
    if trajectory_profile in ("accel_duration_triangular", "accel_duration_scurve"):
        if target_accel_mps2 is None:
            raise ValueError(f"trajectory_profile={trajectory_profile!r} requires target_accel_mps2")
        # target_x_delta_m becomes a derived/reported quantity, not an input,
        # for these profiles -- computed once upfront via the same closed-
        # form used inside x_profile_target(), so every downstream consumer
        # (tolerances, scoring, summary.json) sees a normal target_x_delta_m
        # exactly as with the dx-driven profiles, unaware anything differs.
        target_x_delta_m = accel_duration_displacement(trajectory_profile, target_accel_mps2, move_duration_s)
    elif target_accel_mps2 is not None:
        raise ValueError(f"target_accel_mps2 is only meaningful for accel/duration profiles, not {trajectory_profile!r}")

    controller = XAxisCartesianImpedanceController(impedance_cfg)
    if gain_overrides:
        # Validated (unknown-field / non-finite rejection) and applied before
        # link.connect() -- fail on a bad override dict before ever touching
        # the robot, not mid-run.
        controller.set_gains(gain_overrides)
    safety = ImpedanceSafetyMonitor(safety_cfg)
    estop = EStopLatch()
    tracker = TimingTracker(frequency_hz)
    phases = PhaseLatencyRecorder() if record_latency else None

    # Diagnostic-only dynamics residual observer (2026-07-29, direct_torque
    # only -- see docs/status/direct_torque_residual_observer_2026-07-29.md).
    # Predicts qdd from known rigid-body dynamics + the true total commanded
    # torque and compares it to a noise-robust qdd estimated from consecutive
    # qd samples; logged to trace_rows for post-hoc analysis ONLY. Built
    # unconditionally from a dedicated PinocchioUR5eDynamics instance
    # (independent of `dynamics_source`, which only affects the CONTROLLER's
    # own J/M source) so the residual model is identical regardless of which
    # dynamics_source this run uses. Constructed before link.connect(), same
    # as gain_overrides above, so a bad gap_cycles/lowpass_alpha value fails
    # before ever touching the robot.
    residual_dynamics = None
    residual_accel_estimator = None
    if enable_residual_observer:
        try:
            residual_dynamics = PinocchioUR5eDynamics()
            residual_accel_estimator = JointAccelEstimator(
                gap_cycles=residual_qdd_gap_cycles, lowpass_alpha=residual_qdd_lowpass_alpha
            )
        except Exception as exc:  # noqa: BLE001 - diagnostic-only feature must never block real motion
            print(
                f"[direct_torque_transport] WARNING: residual observer disabled, failed to "
                f"initialize PinocchioUR5eDynamics ({type(exc).__name__}: {exc}). This is a "
                f"diagnostic-only feature (see "
                f"docs/status/direct_torque_residual_observer_2026-07-29.md) -- transport will "
                f"proceed without it rather than block real hardware operation over a missing/"
                f"broken diagnostic dependency.",
                flush=True,
            )
            residual_dynamics = None
            residual_accel_estimator = None

    # Telemetry-gap bridge (2026-08-02, direct_torque only -- see
    # hardware/telemetry_gap_bridge.py's module docstring for the full
    # design/scope). Opt-in, default off. Reuses residual_dynamics above
    # rather than constructing its own provider -- if that construction
    # failed above (diagnostic dependency missing/broken), the bridge
    # degrades to a no-op the same way the residual observer itself does,
    # rather than blocking real motion over it.
    gap_bridge = (
        TelemetryGapBridge(max_bridge_cycles=telemetry_gap_bridge_max_cycles)
        if (telemetry_gap_bridge and residual_dynamics is not None)
        else None
    )

    link.connect()
    state0 = link.read_state()
    x0 = float(state0.tcp_pose[0])
    if local_dynamics is not None:
        J0, M0 = local_dynamics.jacobian_and_mass_matrix(state0.q)
        init_robot_state = link.compose_robot_state(
            state0, jacobian=J0, mass_matrix=M0, time_s=0.0, target_x=x0, target_x_vel=0.0
        )
    else:
        init_robot_state = link.build_robot_state(state0, time_s=0.0, target_x=x0, target_x_vel=0.0)
    controller.reset_from_state(init_robot_state)
    if residual_accel_estimator is not None:
        residual_accel_estimator.reset(state0.qd)

    # Opt-in async residual observer (2026-07-31, see
    # docs/status/direct_torque_residual_observer_async_2026-07-31.md and
    # hardware/residual_observer_worker.py's module docstring). Default
    # (residual_observer_async=False) leaves the inline synchronous path
    # below completely untouched -- residual_worker stays None and the loop
    # takes the exact same branch it always has.
    #
    # residual_dynamics/residual_accel_estimator above are still constructed
    # (or not, on init failure) exactly as before -- that pre-connect probe
    # is what preserves this feature's existing "fail fast on a bad
    # gap_cycles/lowpass_alpha value, and gracefully degrade to disabled on
    # any other init failure, before ever touching the robot" contract for
    # BOTH modes. In async mode those two objects are simply never used for
    # per-cycle computation (the worker builds its own copies, since Pinocchio
    # C++ state can't cross a process boundary) -- only their successful
    # construction here is reused, as the signal that it's safe to start the
    # worker.
    residual_worker: ResidualObserverWorkerHandle | None = None
    residual_async_summary: ResidualAsyncSummary | None = None
    residual_async_active = bool(
        residual_observer_async and residual_dynamics is not None and residual_accel_estimator is not None
    )
    if residual_async_active:
        residual_worker = start_residual_observer_worker(
            qd0=state0.qd,
            gap_cycles=residual_qdd_gap_cycles,
            lowpass_alpha=residual_qdd_lowpass_alpha,
        )

    safety.reset()
    safety.set_initial_position(
        np.asarray(state0.tcp_pose[:3], dtype=np.float64),
        move_axis=0,
    )
    # ImpedanceSafetyMonitor (above) has no TCP speed/acceleration/waypoint-jump
    # ceiling -- only drift/orientation/joint-velocity/axis-growth. Layer
    # CartesianMoveMonitor on top for the checks position mode already has and
    # this (live-torque) mode was missing. Reuse safety_cfg's already-active
    # thresholds for qd/drift/orientation so this doesn't introduce a second,
    # different trip point for a check that already exists.
    move_limits = CartesianMoveLimits.from_impedance_safety_config(safety_cfg)
    if max_tcp_accel_mps2_override is not None:
        # See the identical override in hardware/position_transport.py for
        # the full rationale (real-hardware noise-amplification finding,
        # 2026-07-28). Direct-torque runs at 500 Hz vs position mode's
        # 125 Hz, so the same one-step finite-difference accel estimate's
        # noise amplification (~1/dt^2) is ~16x worse here -- expect this
        # override to matter more, not less, in this mode.
        move_limits = replace(move_limits, max_tcp_accel_mps2=float(max_tcp_accel_mps2_override))
    if max_tcp_speed_mps_override is not None:
        # Explicit, opt-in override of CartesianMoveLimits.max_tcp_speed_mps
        # (class default 0.05 m/s) -- mirrors max_tcp_accel_mps2_override's
        # own pattern above. A real, deliberate, evidence-scoped decision by
        # the caller (e.g. sized with margin above a specific target
        # accel/duration's own computed peak velocity), not a silent
        # threshold change -- default (None) leaves the class default
        # untouched.
        move_limits = replace(move_limits, max_tcp_speed_mps=float(max_tcp_speed_mps_override))
    accel_overrides: dict[str, float] = {}
    if accel_gap_cycles_override is not None:
        accel_overrides["accel_gap_cycles"] = int(accel_gap_cycles_override)
    if speed_lowpass_alpha_override is not None:
        accel_overrides["speed_lowpass_alpha"] = float(speed_lowpass_alpha_override)
    # Noise-robust TCP SPEED-LIMIT overrides (2026-08-01) -- see
    # CartesianMoveLimits.speed_limit_gap_cycles/speed_limit_lowpass_alpha
    # docstring. Independent from accel_gap_cycles/speed_lowpass_alpha above.
    if speed_limit_gap_cycles_override is not None:
        accel_overrides["speed_limit_gap_cycles"] = int(speed_limit_gap_cycles_override)
    if speed_limit_lowpass_alpha_override is not None:
        accel_overrides["speed_limit_lowpass_alpha"] = float(speed_limit_lowpass_alpha_override)
    # DeadlineMonitor-style graduated tolerance overrides (2026-07-30) -- see
    # CartesianMoveLimits.accel_max_consecutive_violations' docstring and
    # NOISE_ROBUST_GUARD_OVERRIDES in hardware/safety.py. Explicit opt-in
    # only; default (None) leaves the class's own no-op defaults untouched.
    if accel_max_consecutive_violations_override is not None:
        accel_overrides["accel_max_consecutive_violations"] = int(accel_max_consecutive_violations_override)
    if accel_hard_multiple_override is not None:
        accel_overrides["accel_hard_multiple"] = float(accel_hard_multiple_override)
    if speed_max_consecutive_violations_override is not None:
        accel_overrides["speed_max_consecutive_violations"] = int(speed_max_consecutive_violations_override)
    if speed_hard_multiple_override is not None:
        accel_overrides["speed_hard_multiple"] = float(speed_hard_multiple_override)
    # Graduated (tiered) tolerance switches (landed 2026-08-02, see
    # CartesianMoveLimits.accel_variable_tolerance's docstring in
    # hardware/safety.py) -- turns on the severity-scaled tier1/tier2/tier3
    # cycle counts (class defaults: 2x-multiple gets 10 cycles, 3.5x gets 5,
    # up to the hard ceiling gets 2), letting a brief, moderate breakaway-
    # transient spike ride through without weakening the untouched hard-
    # ceiling instant-trip path. Tier values themselves are not exposed as
    # CLI flags yet -- only the on/off switch, using the class's own already-
    # documented defaults.
    if accel_variable_tolerance_override is not None:
        accel_overrides["accel_variable_tolerance"] = bool(accel_variable_tolerance_override)
    if speed_variable_tolerance_override is not None:
        accel_overrides["speed_variable_tolerance"] = bool(speed_variable_tolerance_override)
    if accel_overrides:
        move_limits = replace(move_limits, **accel_overrides)
    move_monitor = CartesianMoveMonitor(move_limits)
    move_monitor.set_start(state0.tcp_pose, move_axis_index=0)

    trace_rows: list[dict[str, Any]] = []
    termination_reason = "duration_complete"
    steps = 0
    t_s = 0.0
    last_link_state = state0
    next_deadline_ns = monotonic_ns() + tracker.period_ns
    prev_cycle_start_ns: int | None = None

    # Bounded rolling window feeding pre_trip_trend (see the module-level
    # comment above PRE_TRIP_TREND_WINDOW_CYCLES). Appended once per cycle
    # below, unconditionally, from values already computed this cycle for
    # other purposes -- O(1) amortized, no new per-cycle computation beyond a
    # 3-vector norm/divide for tcp speed (see the loop for why that's derived
    # from tcp_pose deltas, matching CartesianMoveMonitor.check()'s own
    # single-cycle speed_mps formula, rather than from ee_lin_vel, a
    # Jacobian-derived twist and a different real signal).
    pre_trip_window: deque[tuple[float, float, float, float, float, float, float]] = deque(
        maxlen=PRE_TRIP_TREND_WINDOW_CYCLES
    )
    prev_tcp_pos_for_trend = np.asarray(state0.tcp_pose[:3], dtype=np.float64)
    y0_for_trend = float(state0.tcp_pose[1])
    z0_for_trend = float(state0.tcp_pose[2])

    # Enforce the previously-unchecked max_deadline_ms, and catch a
    # frozen-but-non-raising RTDE stream, on every cycle (see hardware/safety.py
    # DeadlineMonitor/StaleStateMonitor for the trip-condition reasoning). This
    # loop already tracks per-cycle start-lateness (lateness_ns below), so the
    # deadline monitor is fed that directly rather than a separate measure.
    safety_limits = getattr(link, "limits", None) or UR5eSafetyLimits()
    # Period-relative deadline cap (2026-07-29, see
    # docs/status/deadline_monitor_period_relative_fix_2026-07-29.md): the flat
    # max_deadline_ms default (3.0 ms) was calibrated for the three 125 Hz
    # loops (8 ms period) and is too loose for this loop's own 500 Hz/2 ms
    # period -- a real ~2 ms overrun on real hardware sat under it undetected.
    # min() makes this a strict no-op for any loop whose period is >= 6 ms
    # (0.5 * period_ms >= max_deadline_ms there), so this only ever tightens
    # the threshold for dt_s well below that -- exactly this loop's 2 ms case.
    effective_deadline_ms = min(
        safety_limits.max_deadline_ms,
        safety_limits.max_deadline_fraction_of_period * dt_s * 1000.0,
    )
    deadline_monitor = DeadlineMonitor(effective_deadline_ms)
    stale_monitor = StaleStateMonitor()

    # Real hardware finding (2026-07-30): this loop's own trace_rows/
    # PhaseLatencyRecorder/TimingTracker.samples grow every cycle and are
    # deliberately kept alive for the whole run, so they're not garbage --
    # but Python's cyclic GC still periodically re-scans every live tracked
    # container, and that scan's cost grows with how many of those
    # containers have accumulated. Combined with real per-cycle garbage
    # (Pinocchio's coriolis() returns a fresh array every call, plus
    # compose_robot_state's temporary dict), this produced a real, observed
    # slowdown starting a few hundred cycles into a run. None of this loop's
    # objects form reference cycles, so refcounting alone still frees true
    # garbage immediately -- disabling the cyclic collector for this
    # bounded-duration loop only turns off the increasingly-expensive
    # periodic re-scan, it does not leak memory within one run.
    gc.disable()
    try:
        while t_s < duration_s - 1e-12:
            if estop.tripped:
                termination_reason = estop.reason or "estop"
                break

            cycle_start_ns = monotonic_ns()
            # Real measured per-cycle interval (this cycle's start minus the
            # previous cycle's start), computed early so it can be threaded
            # into this cycle's RobotState as dt_s below. Same formula as the
            # interval_ns/real_dt_s computed further down this same iteration
            # for tracker.add_sample() and the residual observer -- reusing it
            # there instead of recomputing (prev_cycle_start_ns is not mutated
            # until the end of this iteration, so the value is identical
            # either way).
            interval_ns = None if prev_cycle_start_ns is None else cycle_start_ns - prev_cycle_start_ns
            real_dt_s = dt_s if interval_ns is None else max(dt_s, interval_ns / 1e9)
            lateness_ns = max(0, cycle_start_ns - next_deadline_ns)
            if phases is not None:
                phases.record("lateness_ns", lateness_ns)

            deadline_reason = deadline_monitor.record(lateness_ns)
            if deadline_reason:
                termination_reason = deadline_reason
                estop.trip(deadline_reason)
                break

            t_read = monotonic_ns()
            link_state = link.read_state()
            if phases is not None:
                phases.record("read_state_ns", monotonic_ns() - t_read)
            last_link_state = link_state

            stale_reason = stale_monitor.record(link_state.robot_timestamp_s, link_state.host_stamp_ns)
            if stale_reason:
                termination_reason = stale_reason
                estop.trip(stale_reason)
                break

            target_x, target_x_vel = x_profile_target(
                trajectory_profile,
                x0,
                float(target_x_delta_m),
                t_s,
                duration_s,
                move_duration_s=move_duration_s,
                target_accel_mps2=target_accel_mps2,
            )
            target_x_accel = x_profile_accel(
                trajectory_profile,
                float(target_x_delta_m),
                t_s,
                duration_s,
                move_duration_s=move_duration_s,
                target_accel_mps2=target_accel_mps2,
            )

            t_jac = monotonic_ns()
            tau_coriolis = np.zeros(6, dtype=np.float64)
            if local_dynamics is not None:
                if coriolis_feedforward:
                    jacobian, mass_matrix, tau_coriolis = local_dynamics.jacobian_mass_and_coriolis(
                        link_state.q, link_state.qd
                    )
                else:
                    jacobian, mass_matrix = local_dynamics.jacobian_and_mass_matrix(link_state.q)
                if phases is not None:
                    phases.record("local_dynamics_ns", monotonic_ns() - t_jac)
            else:
                jacobian = link.get_jacobian()
                if phases is not None:
                    phases.record("get_jacobian_ns", monotonic_ns() - t_jac)
                t_mass = monotonic_ns()
                mass_matrix = link.get_mass_matrix()
                if phases is not None:
                    phases.record("get_mass_matrix_ns", monotonic_ns() - t_mass)

            t_build = monotonic_ns()
            robot_state = link.compose_robot_state(
                link_state,
                jacobian=jacobian,
                mass_matrix=mass_matrix,
                time_s=t_s,
                target_x=target_x,
                target_x_vel=target_x_vel,
                dt_s=real_dt_s,
                target_x_accel=target_x_accel,
            )
            if phases is not None:
                phases.record("build_state_ns", monotonic_ns() - t_build)

            t_ctrl = monotonic_ns()
            output = controller.compute(robot_state)
            tau_controller = np.asarray(output.tau, dtype=np.float64).reshape(6)
            # tau_coriolis is zeros(6) unless coriolis_feedforward is on -- see
            # the flag's docstring above. Gravity is deliberately NOT added
            # here; PolyScope's directTorque() already compensates it.
            tau = tau_controller + tau_coriolis
            if phases is not None:
                phases.record("controller_ns", monotonic_ns() - t_ctrl)

            # Pre-trip diagnostic window append (see setup comment above) --
            # placed here, after tau_controller/output are known but BEFORE
            # any guard check below can break the loop, so a trip-causing
            # cycle's own values land in the window too, not just the cycles
            # leading up to it.
            tcp_pos_now = np.asarray(link_state.tcp_pose[:3], dtype=np.float64)
            pre_trip_window.append(
                (
                    float(np.max(np.abs(link_state.qd))),
                    float(np.linalg.norm(tcp_pos_now - prev_tcp_pos_for_trend)) / real_dt_s,
                    float(output.x_error),
                    float(np.sum(np.abs(tau_controller))),
                    float(output.orientation_error_norm),
                    float(tcp_pos_now[1] - y0_for_trend),
                    float(tcp_pos_now[2] - z0_for_trend),
                )
            )
            prev_tcp_pos_for_trend = tcp_pos_now

            t_safe = monotonic_ns()
            safety_decision = safety.check(
                robot_state,
                x_error=float(output.x_error),
                orientation_error_norm=float(output.orientation_error_norm),
                axis_target_moving=bool(t_s <= move_duration_s),
            )
            if phases is not None:
                phases.record("safety_ns", monotonic_ns() - t_safe)
            if not safety_decision.ok:
                termination_reason = safety_decision.reason or "safety_stop"
                estop.trip(termination_reason)
                break

            # Telemetry-gap bridge -- feeds ONLY this guard check, never the
            # controller above (which already computed tau from the raw
            # link_state this cycle, unchanged). See
            # hardware/telemetry_gap_bridge.py's module docstring.
            guard_q, guard_qd, guard_tcp_pose = link_state.q, link_state.qd, link_state.tcp_pose
            telemetry_bridged = False
            if gap_bridge is not None:
                t_bridge = monotonic_ns()
                bridge_coriolis = residual_dynamics.coriolis(link_state.q, link_state.qd)
                bridge_result = gap_bridge.process(
                    q=link_state.q,
                    qd=link_state.qd,
                    tcp_pose=link_state.tcp_pose,
                    robot_timestamp_s=link_state.robot_timestamp_s,
                    tau_applied=tau,
                    mass_matrix=mass_matrix,
                    coriolis_term=bridge_coriolis,
                    jacobian=jacobian,
                    dt_s=real_dt_s,
                )
                guard_q, guard_qd, guard_tcp_pose = bridge_result.q, bridge_result.qd, bridge_result.tcp_pose
                telemetry_bridged = bridge_result.bridged
                if phases is not None:
                    phases.record("telemetry_gap_bridge_ns", monotonic_ns() - t_bridge)

            move_decision = move_monitor.check(
                q=guard_q,
                qd=guard_qd,
                tcp_pose=guard_tcp_pose,
                target_tcp_pose=np.concatenate(([target_x], state0.tcp_pose[1:6])),
                orientation_error_rad=float(output.orientation_error_norm),
                axis_target_moving=bool(t_s <= move_duration_s),
                dt_s=dt_s,
            )
            if not move_decision.ok:
                termination_reason = move_decision.reason or "cartesian_move_stop"
                estop.trip(termination_reason)
                break

            if not is_robot_safety_normal(link_state.safety_status):
                termination_reason = f"robot_safety_status_abnormal: {link_state.safety_status}"
                estop.trip(termination_reason)
                break

            t_torque = monotonic_ns()
            link.direct_torque(tau, friction_comp=True)
            if phases is not None:
                phases.record("direct_torque_ns", monotonic_ns() - t_torque)

            cycle_end_ns = monotonic_ns()
            work_ns = cycle_end_ns - cycle_start_ns
            if phases is not None:
                phases.record("total_work_ns", work_ns)

            sleep_ns = 0
            next_deadline_ns += tracker.period_ns
            sleep_ns = max(0, next_deadline_ns - cycle_end_ns)
            if sleep_ns > 0:
                time.sleep(sleep_ns / 1e9)
            if phases is not None:
                phases.record("sleep_ns", sleep_ns)

            # Diagnostic-only timing (2026-07-31, see
            # docs/status/direct_torque_loop_tail_latency_investigation_2026-07-31.md):
            # this call runs after time.sleep() above, so its real cost was
            # previously invisible to latency_phases and leaked entirely into
            # the NEXT cycle's measured lateness_ns instead. Purely additive
            # -- wraps the existing call in place, does not move it or change
            # what it does.
            t_tracker = monotonic_ns()
            tracker.add_sample(
                cycle_index=steps,
                start_ns=cycle_start_ns,
                deadline_ns=next_deadline_ns - tracker.period_ns,
                end_ns=cycle_end_ns,
                sleep_ns=sleep_ns,
                interval_ns=interval_ns,
            )
            if phases is not None:
                phases.record("tracker_sample_ns", monotonic_ns() - t_tracker)
            prev_cycle_start_ns = cycle_start_ns

            # Diagnostic-only dynamics residual observer (see the setup
            # comment above and docs/status/direct_torque_residual_observer_2026-07-29.md).
            # Reuses this cycle's already-computed mass_matrix (from whichever
            # dynamics_source is active) rather than recomputing it.
            #
            # PolyScope's directTorque() auto-adds gravity compensation that
            # Python never sends (AGENTS.md: never add gravity twice), so the
            # TRUE total physical torque is `tau + gravity(q)`, and the
            # matching bias term is `coriolis(q, qd) + gravity(q)`. The
            # gravity(q) term is IDENTICAL in both (same function, same q,
            # evaluated the same way) and cancels algebraically:
            #   (tau + g(q)) - (C(q,qd)qd + g(q)) = tau - C(q,qd)qd
            # so this only ever needs the raw commanded `tau` and
            # `coriolis(q, qd)` -- one Pinocchio call (a dedicated
            # zero-gravity-model rnea, see
            # controller_core/model_dynamics.py::PinocchioUR5eDynamics.coriolis)
            # instead of the previous two (gravity() + bias()). Verified
            # numerically equivalent to ~1e-14 abs and ~1.5-2x faster on this
            # machine -- see
            # docs/status/residual_observer_dynamics_optimization_2026-07-30.md
            # and dynamics_residual.py::predict_joint_acceleration's docstring
            # for the same derivation.
            # Async mode (2026-07-31, see
            # docs/status/direct_torque_residual_observer_async_2026-07-31.md):
            # replace the inline compute with a strictly non-blocking enqueue
            # to the worker process started before this loop began. qdd_pred/
            # qdd_measured/qdd_residual are NOT available this cycle either
            # way -- they are filled in later by merging the worker's results
            # back into trace_rows by step index, after the loop exits (see
            # the merge block below). This keeps the sync path (below,
            # residual_observer_async=False) exactly as it was before this
            # change -- same objects, same formula, same trace_rows fields
            # populated inline, same phases.record placement.
            qdd_pred = qdd_measured = qdd_residual = None
            if residual_worker is not None:
                t_residual = monotonic_ns()
                real_dt_s = dt_s if interval_ns is None else max(dt_s, interval_ns / 1e9)
                residual_worker.submit(
                    step=steps,
                    q=link_state.q,
                    qd=link_state.qd,
                    tau=tau,
                    mass_matrix=mass_matrix,
                    real_dt_s=real_dt_s,
                )
                if phases is not None:
                    phases.record("residual_observer_ns", monotonic_ns() - t_residual)
            elif residual_dynamics is not None and residual_accel_estimator is not None:
                t_residual = monotonic_ns()
                coriolis_term = residual_dynamics.coriolis(link_state.q, link_state.qd)
                qdd_pred = predict_joint_acceleration(mass_matrix, tau, coriolis_term)
                real_dt_s = dt_s if interval_ns is None else max(dt_s, interval_ns / 1e9)
                qdd_measured = residual_accel_estimator.update(link_state.qd, real_dt_s)
                if qdd_measured is not None:
                    qdd_residual = joint_acceleration_residual(qdd_measured, qdd_pred)
                if phases is not None:
                    phases.record("residual_observer_ns", monotonic_ns() - t_residual)

            # Diagnostic-only timing (2026-07-31, see
            # docs/status/direct_torque_loop_tail_latency_investigation_2026-07-31.md):
            # same rationale as the tracker.add_sample() timer above -- this
            # runs after sleep(), so its cost was previously invisible and
            # leaked into the next cycle's lateness_ns. Purely additive.
            t_trace_append = monotonic_ns()
            trace_rows.append(
                {
                    "time_s": t_s,
                    "q": link_state.q.tolist(),
                    "qd": link_state.qd.tolist(),
                    "ee_pos": robot_state["ee_pos"].tolist(),
                    "ee_quat": robot_state["ee_quat"].tolist(),
                    "tcp_pose": link_state.tcp_pose.tolist(),
                    "target_x": target_x,
                    "x_error": float(output.x_error),
                    "orientation_error_norm": float(output.orientation_error_norm),
                    # Controller-internal diagnostics already computed inside
                    # controller.compute() (see CartesianImpedanceOutput) but
                    # previously discarded every cycle -- found blocking the
                    # 2026-07-28 velocity-overshoot investigation (see
                    # docs/status/clock_timing_late_cycles_2026-07-28.md).
                    "jacobian_cond": float(output.jacobian_cond),
                    "singular_scale": float(output.singular_scale),
                    "task_scale": float(output.task_scale),
                    "task_backtrack_iters": int(output.task_backtrack_iters),
                    "tau_controller": tau_controller.tolist(),
                    "tau_coriolis": tau_coriolis.tolist(),
                    "coriolis_feedforward_active": bool(coriolis_feedforward),
                    "tau_applied": tau.tolist(),
                    # Diagnostic-only, never read by ImpedanceSafetyMonitor or
                    # CartesianMoveMonitor -- see the computation above and
                    # docs/status/direct_torque_residual_observer_2026-07-29.md.
                    # qdd_measured/qdd_residual are None for the first
                    # residual_qdd_gap_cycles cycles (estimator still filling
                    # its gap window) or whenever enable_residual_observer is
                    # False.
                    "qdd_pred": None if qdd_pred is None else qdd_pred.tolist(),
                    "qdd_measured": None if qdd_measured is None else qdd_measured.tolist(),
                    "qdd_residual": None if qdd_residual is None else qdd_residual.tolist(),
                    "qdd_residual_norm": (
                        None if qdd_residual is None else float(np.linalg.norm(qdd_residual))
                    ),
                    "cycle_work_ms": work_ns / 1e6,
                    "lateness_ms": lateness_ns / 1e6,
                }
            )
            if gap_bridge is not None:
                # Only added to the row shape when the bridge is actually
                # enabled -- keeps every existing golden-trace/exact-equality
                # test byte-identical at the (default) telemetry_gap_bridge=False
                # setting. See hardware/telemetry_gap_bridge.py.
                trace_rows[-1]["telemetry_gap_bridged"] = bool(telemetry_bridged)
            if phases is not None:
                phases.record("trace_append_ns", monotonic_ns() - t_trace_append)
            steps += 1
            t_s += dt_s
    except RTDEStateError as exc:
        termination_reason = f"rtde_state_error: {exc}"
        estop.trip(termination_reason)
    finally:
        gc.enable()
        try:
            link.direct_torque(np.zeros(6), friction_comp=True)
        except Exception:
            pass
        link.safe_stop("transport_exit")
        # Worker lifecycle: this MUST run under every exit path (normal
        # completion, estop break, or an exception propagating out of the
        # try block above) so no zombie process can survive a run. Merging
        # results into trace_rows happens separately, below, outside this
        # timed/critical finally block -- see that block's comment for why.
        if residual_worker is not None:
            residual_async_summary = residual_worker.shutdown_and_collect()

    # Merge async residual-observer results back into trace_rows by step
    # index, outside the timed per-cycle section (see the finally block above
    # for why process shutdown itself happens there instead -- that part
    # must run under every exit path, this merge is pure post-hoc
    # bookkeeping and only meaningful on a normal/RTDEStateError exit, since
    # any other exception aborts this function before trace_rows/summary are
    # ever returned to a caller). trace_rows is appended once per loop
    # iteration in step order starting at 0, so `step` is exactly the list
    # index. Steps with no result (dropped request, dropped result, or still
    # in flight when the worker was torn down) keep the qdd_*=None
    # placeholders already written when the row was appended -- same
    # "diagnostic data not yet available" convention the sync path already
    # uses for the gap-window-filling case.
    residual_async_merged_count = 0
    if residual_async_summary is not None:
        if residual_async_summary.worker_init_error:
            print(
                f"[direct_torque_transport] WARNING: async residual observer worker failed to "
                f"initialize ({residual_async_summary.worker_init_error}). This is a "
                f"diagnostic-only feature -- the run proceeded without residual data rather "
                f"than block real hardware operation over a missing/broken diagnostic "
                f"dependency.",
                flush=True,
            )
        for step, payload in residual_async_summary.results.items():
            if "error" in payload:
                continue  # worker-side compute error for this cycle; leave qdd_*=None
            if not (0 <= step < len(trace_rows)):
                continue
            row = trace_rows[step]
            row["qdd_pred"] = payload.get("qdd_pred")
            row["qdd_measured"] = payload.get("qdd_measured")
            row["qdd_residual"] = payload.get("qdd_residual")
            row["qdd_residual_norm"] = payload.get("qdd_residual_norm")
            residual_async_merged_count += 1

    final_state = last_link_state
    achieved_x_delta_m = float(final_state.tcp_pose[0] - x0)
    hold_duration_s = max(duration_s - move_duration_s, 0.0)
    max_abs_qd = max_abs_qd_from_trace(trace_rows)
    pre_trip_trend = _build_pre_trip_trend(pre_trip_window, termination_reason)

    summary = {
        "backend": "ursim_rtde_direct_torque",
        "config_path": str(config_path),
        "gain_overrides": dict(gain_overrides) if gain_overrides else {},
        "target_x_delta": float(target_x_delta_m),
        "target_x_delta_m": float(target_x_delta_m),
        "trajectory_profile": trajectory_profile,
        "target_accel_mps2": None if target_accel_mps2 is None else float(target_accel_mps2),
        "move_duration_s": move_duration_s,
        "hold_duration_s": hold_duration_s,
        "duration_s": duration_s,
        "total_duration_s": duration_s,
        "sim_time_s": min(t_s, duration_s),
        "frequency_hz": frequency_hz,
        "steps": steps,
        "termination_reason": termination_reason,
        "pre_trip_trend": pre_trip_trend,
        "achieved_x_delta_m": achieved_x_delta_m,
        "final_tcp_pose": final_state.tcp_pose.tolist(),
        "initial_ee_pos": state0.tcp_pose[:3].tolist(),
        "success": termination_reason == "duration_complete" and not estop.tripped,
        "velocity_guard_ok": max_abs_qd <= safety_cfg.max_joint_velocity_radps,
        "max_abs_qd_radps": max_abs_qd,
        "joint_limit_guard_ok": True,
        "torque_saturation_percentage": 0.0,
        "dynamics_source": dynamics_source,
        "telemetry_gap_bridge_active": gap_bridge is not None,
        "timing": tracker.summary(),
    }
    if phases is not None:
        summary["latency_phases"] = phases.summary()
    if residual_async_summary is not None:
        # Diagnostic-only lifecycle/coverage bookkeeping for the async
        # residual observer -- see
        # docs/status/direct_torque_residual_observer_async_2026-07-31.md.
        summary["residual_observer_async"] = {
            "enabled": True,
            "steps": steps,
            "dropped_request_count": residual_async_summary.dropped_request_count,
            "dropped_result_count": residual_async_summary.dropped_result_count,
            "merged_step_count": residual_async_merged_count,
            "unmerged_step_count": steps - residual_async_merged_count,
            "worker_init_error": residual_async_summary.worker_init_error,
            "worker_exitcode": residual_async_summary.exitcode,
            "worker_terminated_forcefully": residual_async_summary.terminated_forcefully,
        }
    summary.update(
        summarize_move_hold_trace(
            trace_rows,
            initial_ee_pos=state0.tcp_pose[:3],
            move_duration_s=move_duration_s,
            total_duration_s=duration_s,
            transport_axis_index=0,
        )
    )
    summary.update(compute_valid_move_hold_metrics(summary))
    summary["success"] = bool(
        not estop.tripped
        and summary.get("valid_move_and_hold", False)
        and termination_reason == "duration_complete"
    )

    trace_path = None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        trace_path = output_dir / "trace.jsonl"
        with trace_path.open("w", encoding="utf-8") as fh:
            for row in trace_rows:
                fh.write(json.dumps(row) + "\n")
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    ok = bool(summary["success"])
    reason = "" if ok else str(summary.get("hold_failure_reason") or summary.get("move_failure_reason") or termination_reason)
    return DirectTorqueTransportResult(ok=ok, reason=reason, summary=summary, trace_path=trace_path)
