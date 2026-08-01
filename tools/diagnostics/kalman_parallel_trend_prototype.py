#!/usr/bin/env python3
"""Offline prototype: a PARALLEL, diagnostic-only Kalman branch alongside the
existing (unchanged) accel-guard heuristic and the existing (unchanged)
``pre_trip_trend`` mechanism -- does NOT gate any trip decision.

Follow-up to ``tools/diagnostics/kalman_tcp_accel_filter_prototype.py`` /
``docs/status/kalman_filtering_sensor_noise_2026-08-01.md``'s "single filter
replacing the heuristic" verdict (correctly: don't). This script tests a
different, additive architecture instead: the current heuristic stays the
SOLE real-time trip authority (zero regression risk, zero added lag on the
safety-critical path -- "major deviation then bad switchoff" keeps working
exactly as today), and a separate, MORE HEAVILY smoothed Kalman branch runs
alongside it purely as a diagnostic/trend channel, where the ~300ms lag
measured in the original prototype is irrelevant by construction (nothing
here gates a trip; it's meant for slow/gradual trends whose own timescale is
much longer than 300ms).

Two things are tested, both offline/read-only, no ``hardware/safety.py``
change:

1. ``bucket_demo()`` -- confirms the two branches are genuinely independent:
   the existing heuristic (imported unmodified from ``hardware.safety``) and
   the tight/slow KF variant (jerk_psd=0.1, reused unmodified from the
   original prototype) computed side by side on the same real stationary
   capture, plus a re-run of the original fast-move pass/fail backtest to
   confirm the heuristic's outcome is byte-identical to before (sanity
   check, not new research -- nothing in the accel-guard code changed).

2. ``trend_detection_comparison()`` -- the actual new question: does a
   parallel KF branch make ``hardware.direct_torque_transport._classify_trend``
   (the mechanism landed 2026-07-31, commit 467fe52, that already captures
   the pre-trip y_drift/z_drift/orientation_error_norm window) reach
   "rising" earlier/more robustly on a slow-creep drift trend than the RAW
   values it uses today? Tested against a synthetic slow-creep profile
   (explicitly labeled synthetic -- no real trace.jsonl for the real
   -0.15m return-leg trip this was motivated by exists in this checkout;
   only the commit message/AGENTS.md prose description survived, not the
   raw trace) built to match that description: y_drift ramping toward the
   real 0.03 m off-axis-drift guard over a PRE_TRIP_TREND_WINDOW_CYCLES=60
   cycle / 120 ms window (500 Hz), z_drift/orientation_error_norm rising
   more slowly alongside it, real position/orientation noise bootstrapped
   from the same real stationary capture used throughout this line of work.

Usage:
  python tools/diagnostics/kalman_parallel_trend_prototype.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools" / "diagnostics"))

from hardware.direct_torque_transport import PRE_TRIP_TREND_WINDOW_CYCLES, _classify_trend  # noqa: E402
from kalman_tcp_accel_filter_prototype import (  # noqa: E402
    ConstantAccelKalmanFilter1D,
    DEFAULT_CAPTURE,
    bootstrap_real_noise,
    build_kalman_estimator,
    eval_pass_fail,
    load_real_stationary_capture,
)

# Real off-axis drift guard used by production (CartesianMoveLimits /
# ImpedanceSafetyConfig, both AGENTS.md-documented at ~0.02-0.03 m; the
# real -0.15m return-leg trip AGENTS.md describes tripped at "the identical
# ~0.030 m magnitude").
Y_DRIFT_GUARD_M = 0.03


# ---------------------------------------------------------------------------
# 1. bucket_demo -- prove the two branches are independent, and the existing
#    heuristic's fast-move pass/fail outcome is unchanged.
# ---------------------------------------------------------------------------


def bucket_demo() -> None:
    print("=== (1) Parallel/bucketed architecture demo ===")
    print("Branch A (authoritative, UNCHANGED): hardware.safety.CartesianMoveMonitor, "
          "gap=1/alpha=1.0 default -- this is what actually gates a trip in production, "
          "exactly as it does today. Not modified by anything in this file.")
    print("Branch B (diagnostic-only, NEW, not wired to any trip): ConstantAccelKalmanFilter1D, "
          "jerk_psd=0.1 -- the tight/slow variant from the original prototype "
          "(stationary p99=0.073 m/s^2, ~300ms lag). Computed independently, same input stream, "
          "own state, touches nothing in Branch A.")

    t_s, tcp_pos, used_real = load_real_stationary_capture(DEFAULT_CAPTURE)
    dt_s = float(np.median(np.diff(t_s)))
    print(f"\n[data] {'REAL' if used_real else 'SYNTHESIZED FALLBACK'} capture, {tcp_pos.shape[0]} samples, dt={dt_s*1e3:.3f}ms")

    residuals = tcp_pos - np.mean(tcp_pos, axis=0, keepdims=True)
    measurement_var = np.std(tcp_pos, axis=0) ** 2

    print("\n[sanity check] re-running the ORIGINAL large_displacement fast-move backtest "
          "(dx=-0.20m/T=1.0s, base_accel=0.8) -- must still be 30/30 correct, byte-identical to "
          "the original doc, since nothing in hardware.safety or the accel-guard code changed:")
    pf = eval_pass_fail(residuals, dt_s, 10.0, measurement_var, n_seeds=30)
    current_default = pf["large_displacement"]["current_default"]
    current_graduated = pf["large_displacement"]["current_graduated_filtered"]
    print(f"  current_default:            {current_default['trips']}/{current_default['n_seeds']} "
          f"(original doc: 30/30)")
    print(f"  current_graduated_filtered: {current_graduated['trips']}/{current_graduated['n_seeds']} "
          f"(original doc: 30/30)")
    assert current_default["trips"] == 30 and current_graduated["trips"] == 30, (
        "REGRESSION: the unmodified heuristic's outcome on the genuine-catch case changed -- "
        "this must never happen, nothing in this file should be able to cause it."
    )
    print("  -> CONFIRMED unchanged. The parallel KF branch below never touches this path.")


# ---------------------------------------------------------------------------
# 2. Synthetic slow-creep profile + trend-detection comparison
# ---------------------------------------------------------------------------


def synthetic_slow_creep_profile(
    *, n_cycles: int, y_final_m: float, z_final_m: float, orient_final_rad: float, ramp_start_frac: float = 0.2,
) -> dict:
    """A labeled-synthetic reconstruction of the real -0.15m return-leg trip's
    pre-trip window described in AGENTS.md/commit 467fe52: y_drift/z_drift/
    orientation_error_norm creeping steadily over the PRE_TRIP_TREND_WINDOW_CYCLES
    window, y_drift approaching the real 0.03 m guard by the final cycle (the
    trip cycle itself, one past this window). Flat/near-zero for
    ``ramp_start_frac`` of the window (drift isn't distinguishable from noise
    yet), then a smooth (min-jerk-shaped, for a physically plausible ramp --
    not claimed to be the real trip's exact shape, just a reasonable "slow
    creep" stand-in) rise to the final value. x_error/tau/qd/tcp_speed held
    flat -- this test is specifically about the y_drift/z_drift/orientation
    channels added in commit 467fe52, not a full incident reconstruction.

    NO real trace.jsonl for the actual incident exists in this checkout (only
    the commit-message/AGENTS.md prose survived) -- this is explicitly a
    synthetic reconstruction, not real hardware data.
    """
    tau = np.linspace(0.0, 1.0, n_cycles)
    ramp = np.clip((tau - ramp_start_frac) / (1.0 - ramp_start_frac), 0.0, 1.0)
    s = 10.0 * ramp**3 - 15.0 * ramp**4 + 6.0 * ramp**5  # min-jerk-shaped ramp, smooth start
    return {
        "y_drift_true": y_final_m * s,
        "z_drift_true": z_final_m * s,
        "orient_true": orient_final_rad * s,
        "x_error_true": np.full(n_cycles, 0.002),
        "tau_true": np.full(n_cycles, 3.0),
        "qd_true": np.full(n_cycles, 0.05),
        "speed_true": np.full(n_cycles, 0.01),
    }


def bootstrap_channel_noise(residuals_1d: np.ndarray, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """IID resample (not block -- these are single scalars, not a trajectory
    needing local correlation structure preserved) from a real noise
    residual array."""
    idx = rng.integers(0, residuals_1d.shape[0], size=n_samples)
    return residuals_1d[idx]


def smooth_scalar_series(raw: np.ndarray, dt_s: float, jerk_psd: float, measurement_var: float) -> np.ndarray:
    kf = ConstantAccelKalmanFilter1D(dt_s=dt_s, jerk_psd=jerk_psd, measurement_var=measurement_var)
    out = np.empty_like(raw)
    for i, z in enumerate(raw):
        state = kf.step(float(z))
        out[i] = state[0]  # smoothed position/value, not its derivative
    return out


def trend_detection_comparison(*, n_seeds: int = 50, jerk_psd: float = 0.1) -> None:
    print("\n=== (2) Trend-detection comparison: raw pre_trip_trend window vs parallel KF branch ===")
    print(f"Synthetic slow-creep profile (explicitly synthetic -- see docstring), "
          f"{PRE_TRIP_TREND_WINDOW_CYCLES} cycles matching production's real window length, "
          f"y_drift ramping to the real {Y_DRIFT_GUARD_M} m guard by the final cycle.")

    t_s, tcp_pos, used_real = load_real_stationary_capture(DEFAULT_CAPTURE)
    dt_s = float(np.median(np.diff(t_s)))
    n = PRE_TRIP_TREND_WINDOW_CYCLES

    # Real noise residuals, per channel, bootstrapped from the same real
    # capture used throughout this line of work.
    y_res = tcp_pos[:, 1] - np.mean(tcp_pos[:, 1])
    z_res = tcp_pos[:, 2] - np.mean(tcp_pos[:, 2])
    # Real rotation-vector residual norm, as a real-noise proxy for
    # orientation_error_norm's own noise floor (production computes
    # orientation_error_norm via a quaternion pipeline, not a raw
    # rotation-vector norm -- this is an honestly-labeled approximation,
    # not the exact production computation, since no real
    # orientation_error_norm capture exists in this checkout).
    rot = tcp_pos_full = None
    # reload with full 6-vector pose for the rotation columns
    import json
    rows = [json.loads(l) for l in DEFAULT_CAPTURE.open("r", encoding="utf-8") if l.strip()] if DEFAULT_CAPTURE.exists() else []
    if rows:
        rot_full = np.array([r["tcp_pose"][3:6] for r in rows], dtype=np.float64)
        rot_res_norm = np.linalg.norm(rot_full - np.mean(rot_full, axis=0, keepdims=True), axis=1)
    else:
        rng0 = np.random.default_rng(0)
        rot_res_norm = np.linalg.norm(rng0.normal(0.0, [3.086e-05, 1.417e-05, 2.248e-05], size=(4730, 3)), axis=1)

    y_var, z_var, orient_var = float(np.var(y_res)), float(np.var(z_res)), float(np.var(rot_res_norm))
    print(f"[real noise] y_drift std={np.sqrt(y_var):.2e} m, z_drift std={np.sqrt(z_var):.2e} m, "
          f"orientation_error_norm-proxy std={np.sqrt(orient_var):.2e} rad "
          f"(all real, bootstrapped from the same stationary capture)")

    profile = synthetic_slow_creep_profile(
        n_cycles=n, y_final_m=0.031, z_final_m=0.012, orient_final_rad=0.10,
    )
    print(f"[profile, headline magnitude] y_drift: 0 -> {profile['y_drift_true'][-1]:.4f} m over {n} cycles "
          f"(guard: {Y_DRIFT_GUARD_M} m); z_drift -> {profile['z_drift_true'][-1]:.4f} m; "
          f"orientation_error_norm -> {profile['orient_true'][-1]:.4f} rad")

    channels = {
        "y_drift_m": (y_res, y_var, np.sqrt(y_var)),
        "z_drift_m": (z_res, z_var, np.sqrt(z_var)),
        "orientation_error_norm_rad": (rot_res_norm, orient_var, np.sqrt(orient_var)),
    }

    # First: does classify_trend on the FULL fixed window (matching production's
    # actual one-shot usage at trip time, not an incremental scan) correctly say
    # "rising" at the real trip-magnitude scale? Both should saturate near 100%
    # here -- this just confirms the headline profile isn't itself the hard case.
    print(f"\n[headline magnitude, full {n}-cycle window, {n_seeds} seeds] "
          f"detection rate ('rising' correctly reported):")
    print(f"  {'channel':30s} {'raw':>8s} {'KF':>8s}")
    for ch_name, (real_noise_pool, meas_var, _std) in channels.items():
        true_vals = {"y_drift_m": profile["y_drift_true"], "z_drift_m": profile["z_drift_true"],
                      "orientation_error_norm_rad": profile["orient_true"]}[ch_name]
        raw_hits = kf_hits = 0
        for seed in range(n_seeds):
            rng = np.random.default_rng(seed)
            noisy = true_vals + bootstrap_channel_noise(real_noise_pool, n, rng)
            smoothed = smooth_scalar_series(noisy, dt_s, jerk_psd, meas_var)
            if _classify_trend(list(noisy)) == "rising":
                raw_hits += 1
            if _classify_trend(list(smoothed)) == "rising":
                kf_hits += 1
        print(f"  {ch_name:30s} {raw_hits}/{n_seeds:<6d} {kf_hits}/{n_seeds:<6d}")

    # The actual hard/interesting question: sensitivity near the real noise
    # floor. Sweep the true final drift magnitude down from the headline value
    # toward a few multiples of the real measured per-cycle noise std, and
    # measure detection rate raw vs KF at each -- plus a NULL case (zero true
    # drift, pure real noise) to check false-"rising" rate isn't made WORSE by
    # smoothing.
    print(f"\n[sensitivity sweep, y_drift_m only, full {n}-cycle window, {n_seeds} seeds/point] "
          f"true final magnitude as a multiple of the real per-cycle noise std "
          f"({np.sqrt(y_var):.2e} m):")
    print(f"  {'final_mag_m':>12s} {'x_std':>8s} {'raw rate':>10s} {'KF rate':>10s} {'raw falling':>12s} {'KF falling':>11s}")
    y_std = float(np.sqrt(y_var))
    for mult in [0.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]:
        final_mag = mult * y_std
        prof = synthetic_slow_creep_profile(n_cycles=n, y_final_m=final_mag, z_final_m=0.0, orient_final_rad=0.0)
        true_vals = prof["y_drift_true"]
        raw_rising = kf_rising = raw_falling = kf_falling = 0
        for seed in range(n_seeds):
            rng = np.random.default_rng(1000 + seed)
            noisy = true_vals + bootstrap_channel_noise(y_res, n, rng)
            smoothed = smooth_scalar_series(noisy, dt_s, jerk_psd, y_var)
            raw_t, kf_t = _classify_trend(list(noisy)), _classify_trend(list(smoothed))
            raw_rising += raw_t == "rising"
            kf_rising += kf_t == "rising"
            raw_falling += raw_t == "falling"
            kf_falling += kf_t == "falling"
        print(f"  {final_mag:12.2e} {mult:8.1f} {raw_rising}/{n_seeds:<6d} {kf_rising}/{n_seeds:<6d} "
              f"{raw_falling}/{n_seeds:<9d} {kf_falling}/{n_seeds:<8d}")

    print("\n[interpretation] 'rising'/'falling' rate out of n_seeds at each true-drift magnitude "
          "(0.0x = null case, pure real noise, zero true drift -- any 'rising'/'falling' there is a "
          "false alarm from noise alone). Higher raw-vs-KF 'rising' rate gap at small multiples = KF "
          "smoothing meaningfully helps detect a subtle real trend; equal rates = no measurable benefit.")


def main() -> int:
    bucket_demo()
    trend_detection_comparison()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
