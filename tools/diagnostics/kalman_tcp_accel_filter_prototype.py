#!/usr/bin/env python3
"""Offline prototype: does a Kalman filter beat the current EMA+gap-cycle
heuristic for CartesianMoveMonitor's TCP position -> speed -> acceleration
estimate?

Context (see AGENTS.md SS4 and docs/status/kalman_filtering_sensor_noise_2026-08-01.md
for the full writeup): ``hardware.safety.CartesianMoveMonitor`` estimates TCP
acceleration via a naive finite difference of RTDE position readings, which
amplifies raw position noise by ~1/dt^2. The current production fix
(``accel_gap_cycles`` / ``speed_lowpass_alpha`` / graduated consecutive-
violation tolerance) is a set of ad-hoc heuristics, validated against 21 real
hardware trips (docs/status/safety_envelope_backtest_2026-07-30.md, on the
``experiments/safety-envelope-study`` branch). This script asks whether a
principled recursive estimator -- a constant-acceleration Kalman filter, and
a cheaper steady-state-equivalent alpha-beta-gamma (g-h-k) filter -- does
better on the same three axes the current heuristic is judged on: noise
suppression, tracking lag on a genuine fast move, and pass/fail agreement
with the current production guard on synthetic replay of the two profiles
from that backtest this script can reconstruct without the full MuJoCo
controller-chatter simulation (``canonical_headroom``, ``large_displacement``).

**This script does not modify hardware/safety.py and computes no decision
that is wired into any real guard path.** It is offline research/design
support only -- see the doc above for the recommendation and what wiring
this in for real would require.

Real data used: ``outputs/hardware_state_noise/capture_20260728_154018.jsonl``,
a real 10s/500Hz stationary RTDE capture from thinkrobot (2026-07-28),
already the basis for ``tools/analyze_state_noise_capture.py`` and the
noise-floor numbers cited throughout AGENTS.md/docs/status. If that file is
ever missing (e.g. a fresh checkout without ``outputs/`` synced), this
script falls back to synthesizing zero-mean Gaussian position noise at the
documented std (``tcp_pos_std_m=[8.88e-06, 9.86e-06, 3.25e-06]`` from
``hardware_captures/2026-07-28_thinkrobot_172.16.71.77/
stationary_noise_capture_154018_stats.json``) and says so explicitly.

Usage:
  python tools/diagnostics/kalman_tcp_accel_filter_prototype.py
  python tools/diagnostics/kalman_tcp_accel_filter_prototype.py --seeds 30 --json-out out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from math import sqrt
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from hardware.safety import CartesianMoveLimits, CartesianMoveMonitor  # noqa: E402

DEFAULT_CAPTURE = REPO_ROOT / "outputs" / "hardware_state_noise" / "capture_20260728_154018.jsonl"
# Documented fallback (stationary_noise_capture_154018_stats.json), used only
# if the real capture file above isn't present in this checkout.
FALLBACK_TCP_POS_STD_M = np.array([8.88e-06, 9.86e-06, 3.25e-06], dtype=np.float64)
FALLBACK_RATE_HZ = 500.0

CONTROL_RATE_HZ = 500.0
DT_S = 1.0 / CONTROL_RATE_HZ


# ---------------------------------------------------------------------------
# Real data loading
# ---------------------------------------------------------------------------


def load_real_stationary_capture(path: Path) -> tuple[np.ndarray, np.ndarray, bool]:
    """Returns (t_s (N,), tcp_pos (N,3), used_real_data: bool).

    If ``path`` doesn't exist, synthesizes a same-shaped stationary capture
    from the documented noise std (FALLBACK_TCP_POS_STD_M) at 500 Hz and
    returns ``used_real_data=False`` -- callers must report this explicitly,
    per this project's own "don't trust a number without saying where it
    came from" convention (AGENTS.md SS2).
    """
    if path.exists():
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        t_s = np.array([r["t_s"] for r in rows], dtype=np.float64)
        tcp_pos = np.array([r["tcp_pose"][:3] for r in rows], dtype=np.float64)
        return t_s, tcp_pos, True

    print(f"[warn] real capture not found at {path} -- synthesizing noise at the "
          f"documented std {FALLBACK_TCP_POS_STD_M.tolist()} m instead. This is a "
          f"FALLBACK, not real hardware data.")
    n = int(10.0 * FALLBACK_RATE_HZ)
    rng = np.random.default_rng(12345)
    t_s = np.arange(n, dtype=np.float64) / FALLBACK_RATE_HZ
    true_pos = np.array([-0.3155, -0.1720, 0.9325], dtype=np.float64)
    tcp_pos = true_pos[None, :] + rng.normal(0.0, FALLBACK_TCP_POS_STD_M[None, :], size=(n, 3))
    return t_s, tcp_pos, False


# ---------------------------------------------------------------------------
# Filters under comparison
# ---------------------------------------------------------------------------


class ConstantAccelKalmanFilter1D:
    """Discrete constant-acceleration (white-noise-jerk) Kalman filter for a
    single scalar position channel. State x = [p, v, a].

    Standard formulation (e.g. Bar-Shalom, Li & Kirubarajan, "Estimation with
    Applications to Tracking and Navigation", the discrete Wiener-process
    acceleration / white-noise-jerk model): F is the constant-acceleration
    transition matrix, Q is the exact discretization of continuous jerk white
    noise with power spectral density ``jerk_psd``, H=[1,0,0] since only
    position is measured, R is the real measured position-noise variance for
    this axis.
    """

    def __init__(self, *, dt_s: float, jerk_psd: float, measurement_var: float) -> None:
        self.dt_s = float(dt_s)
        self.jerk_psd = float(jerk_psd)
        self.r = float(measurement_var)
        self.F = np.array(
            [[1.0, self.dt_s, 0.5 * self.dt_s**2], [0.0, 1.0, self.dt_s], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        t = self.dt_s
        q = self.jerk_psd
        self.Q = q * np.array(
            [
                [t**5 / 20.0, t**4 / 8.0, t**3 / 6.0],
                [t**4 / 8.0, t**3 / 3.0, t**2 / 2.0],
                [t**3 / 6.0, t**2 / 2.0, t],
            ],
            dtype=np.float64,
        )
        self.H = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        self.x = np.zeros(3, dtype=np.float64)
        self.P = np.eye(3, dtype=np.float64) * 1.0
        self._initialized = False

    def reset(self, p0: float) -> None:
        self.x = np.array([p0, 0.0, 0.0], dtype=np.float64)
        self.P = np.eye(3, dtype=np.float64) * 1.0
        self._initialized = True

    def step(self, z: float) -> np.ndarray:
        """One predict+update cycle; returns the posterior state [p, v, a]."""
        if not self._initialized:
            self.reset(float(z))
            return self.x.copy()
        # Predict
        x_pred = self.F @ self.x
        p_pred = self.F @ self.P @ self.F.T + self.Q
        # Update (scalar measurement -> scalar innovation covariance, no
        # matrix inverse needed)
        y = float(z) - float(self.H @ x_pred)
        s = float(self.H @ p_pred @ self.H) + self.r
        k = (p_pred @ self.H) / s
        self.x = x_pred + k * y
        self.P = p_pred - np.outer(k, self.H @ p_pred)
        return self.x.copy()

    def steady_state_gain(self, *, n_settle_cycles: int = 2000) -> np.ndarray:
        """Runs the covariance recursion alone (no real measurements needed --
        P converges independent of z) to its fixed point and returns the
        converged Kalman gain [k_p, k_v, k_a]. Used to derive the
        alpha-beta-gamma filter's fixed gains below as the steady-state
        equivalent of this same filter."""
        p = np.eye(3, dtype=np.float64) * 1.0
        k = np.zeros(3, dtype=np.float64)
        for _ in range(n_settle_cycles):
            p_pred = self.F @ p @ self.F.T + self.Q
            s = float(self.H @ p_pred @ self.H) + self.r
            k = (p_pred @ self.H) / s
            p = p_pred - np.outer(k, self.H @ p_pred)
        return k


class AlphaBetaGammaFilter1D:
    """Fixed-gain (g-h-k) constant-acceleration filter for a single scalar
    channel. State [p, v, a]. Cheaper than the Kalman filter above (no
    per-cycle covariance propagation -- gains are fixed at construction), at
    the cost of not adapting online. Gains are derived as the STEADY-STATE
    Kalman gain of an equivalent ``ConstantAccelKalmanFilter1D`` (a standard,
    principled way to pick alpha/beta/gamma rather than guessing), so this
    is the asymptotic behavior of that filter, minus its transient
    adaptation and minus the P-matrix runtime cost.
    """

    def __init__(self, *, dt_s: float, alpha: float, beta: float, gamma: float) -> None:
        self.dt_s = float(dt_s)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.x = np.zeros(3, dtype=np.float64)
        self._initialized = False

    @classmethod
    def from_kalman_steady_state(cls, kf: ConstantAccelKalmanFilter1D) -> "AlphaBetaGammaFilter1D":
        k_p, k_v, k_a = kf.steady_state_gain()
        # Standard g-h-k <-> Kalman correspondence: alpha=k_p, beta=k_v*dt,
        # gamma=k_a*dt^2 (see e.g. Blackman & Popoli or Kalata's original
        # alpha-beta-gamma tracker derivation).
        return cls(dt_s=kf.dt_s, alpha=k_p, beta=k_v * kf.dt_s, gamma=k_a * kf.dt_s**2)

    def reset(self, p0: float) -> None:
        self.x = np.array([p0, 0.0, 0.0], dtype=np.float64)
        self._initialized = True

    def step(self, z: float) -> np.ndarray:
        if not self._initialized:
            self.reset(float(z))
            return self.x.copy()
        p, v, a = self.x
        dt = self.dt_s
        p_pred = p + v * dt + 0.5 * a * dt * dt
        v_pred = v + a * dt
        a_pred = a
        r = float(z) - p_pred
        p_new = p_pred + self.alpha * r
        v_new = v_pred + (self.beta / dt) * r
        a_new = a_pred + (2.0 * self.gamma / (dt * dt)) * r
        self.x = np.array([p_new, v_new, a_new], dtype=np.float64)
        return self.x.copy()


@dataclass
class TCPKinematicsEstimate:
    speed_mps: float  # ||v_xyz||, matches CartesianMoveMonitor's speed_mps
    accel_scalar_mps2: float  # |d(||v||)/dt| -- matches CartesianMoveMonitor's accel_mps2 exactly
    accel_vector_mps2: float  # ||a_xyz|| -- direct from filter state, no re-differencing


class TCPKalmanEstimator:
    """3 independent per-axis filters (x/y/z decoupled, standard for this
    kind of tracking problem -- CartesianMoveMonitor's own noise floor is
    per-axis independent too, see the stats file), producing both the
    backward-compatible scalar accel metric (same definition as
    CartesianMoveMonitor.check()'s accel_mps2: |d(speed)/dt|, one MORE
    differentiation on top of the filter's already-smoothed velocity) and
    the arguably more principled vector-norm acceleration straight from the
    filter state (no extra differencing at all).
    """

    def __init__(self, filters: list) -> None:
        assert len(filters) == 3
        self.filters = filters
        self._prev_speed_mps: float | None = None

    def reset(self, pos0: np.ndarray) -> None:
        pos0 = np.asarray(pos0, dtype=np.float64).reshape(3)
        for f, p0 in zip(self.filters, pos0):
            f.reset(float(p0))
        self._prev_speed_mps = None

    def step(self, pos: np.ndarray, dt_s: float) -> TCPKinematicsEstimate:
        pos = np.asarray(pos, dtype=np.float64).reshape(3)
        states = np.array([f.step(float(p)) for f, p in zip(self.filters, pos)])  # (3,3): rows=axes, cols=[p,v,a]
        v = states[:, 1]
        a = states[:, 2]
        speed_mps = float(np.linalg.norm(v))
        accel_vector_mps2 = float(np.linalg.norm(a))
        if self._prev_speed_mps is None:
            accel_scalar_mps2 = 0.0
        else:
            accel_scalar_mps2 = abs(speed_mps - self._prev_speed_mps) / float(dt_s)
        self._prev_speed_mps = speed_mps
        return TCPKinematicsEstimate(speed_mps, accel_scalar_mps2, accel_vector_mps2)


def build_kalman_estimator(*, dt_s: float, jerk_psd: float, measurement_var_per_axis: np.ndarray) -> TCPKalmanEstimator:
    filters = [
        ConstantAccelKalmanFilter1D(dt_s=dt_s, jerk_psd=jerk_psd, measurement_var=float(measurement_var_per_axis[i]))
        for i in range(3)
    ]
    return TCPKalmanEstimator(filters)


def build_alpha_beta_gamma_estimator(*, dt_s: float, jerk_psd: float, measurement_var_per_axis: np.ndarray) -> TCPKalmanEstimator:
    filters = []
    for i in range(3):
        kf = ConstantAccelKalmanFilter1D(dt_s=dt_s, jerk_psd=jerk_psd, measurement_var=float(measurement_var_per_axis[i]))
        filters.append(AlphaBetaGammaFilter1D.from_kalman_steady_state(kf))
    return TCPKalmanEstimator(filters)


# ---------------------------------------------------------------------------
# Current production heuristic, via the REAL hardware.safety.CartesianMoveMonitor
# (not reimplemented -- see AGENTS.md's own "reuse real math" convention)
# ---------------------------------------------------------------------------


def replay_through_production_monitor(
    tcp_pos: np.ndarray,
    *,
    dt_s: float,
    max_tcp_accel_mps2: float,
    accel_gap_cycles: int,
    speed_lowpass_alpha: float,
    accel_max_consecutive_violations: int = 1,
    accel_hard_multiple: float = 5.0,
) -> tuple[np.ndarray, int | None]:
    """Feeds a synthetic/real position sequence through the REAL
    ``CartesianMoveMonitor`` (X-axis move, drift/orientation/waypoint-jump
    checks defanged with generous limits so only the accel/speed channel can
    trip) and returns (accel_mps2 series reconstructed from decision replay,
    first cycle index a TCP-acceleration violation is reported, or None).

    q/qd are zero throughout (doesn't matter -- only used for the qd_max/NaN
    checks here, both irrelevant to this comparison) and
    ``axis_target_moving=True`` disables the axis-tracking-growth check so
    this isolates the accel/speed channel exactly, matching this script's
    stated purpose.
    """
    limits = CartesianMoveLimits(
        max_off_axis_drift_m=10.0,
        max_orientation_error_rad=10.0,
        max_tcp_speed_mps=10.0,  # isolate the ACCEL channel; speed checked separately if needed
        max_tcp_accel_mps2=max_tcp_accel_mps2,
        max_waypoint_jump_m=10.0,
        max_axis_error_growth_steps=10_000,
        qd_max_radps=100.0,
        accel_gap_cycles=accel_gap_cycles,
        speed_lowpass_alpha=speed_lowpass_alpha,
        accel_max_consecutive_violations=accel_max_consecutive_violations,
        accel_hard_multiple=accel_hard_multiple,
    )
    monitor = CartesianMoveMonitor(limits)
    n = tcp_pos.shape[0]
    pose0 = np.zeros(6, dtype=np.float64)
    pose0[:3] = tcp_pos[0]
    monitor.set_start(pose0, move_axis_index=0)
    trip_cycle: int | None = None
    accel_trace = np.full(n, np.nan, dtype=np.float64)
    q = np.zeros(6)
    qd = np.zeros(6)
    for i in range(1, n):
        pose = np.zeros(6, dtype=np.float64)
        pose[:3] = tcp_pos[i]
        decision = monitor.check(
            q=q,
            qd=qd,
            tcp_pose=pose,
            target_tcp_pose=pose,
            orientation_error_rad=0.0,
            axis_target_moving=True,
            dt_s=dt_s,
        )
        # Reconstruct the accel estimate this cycle actually saw, for the
        # noise-suppression comparison -- monitor._prev_speed_mps is the
        # gap-windowed speed *after* this cycle's update, so recompute the
        # per-cycle delta the same way check() itself does internally is not
        # exposed; instead track via the private attribute deliberately
        # (test-only introspection, not a production dependency).
        if trip_cycle is None and not decision.ok and any("TCP acceleration" in r for r in decision.reasons):
            trip_cycle = i
    return accel_trace, trip_cycle


# ---------------------------------------------------------------------------
# Min-jerk motion profile (identical quintic to hardware/motion.py's
# _min_jerk_s / peak_acceleration_mps2 -- copied here, not imported, since
# hardware.motion transitively imports hardware.link -> the real ur_rtde
# bindings, which this offline-only script must not require)
# ---------------------------------------------------------------------------


def min_jerk_s(tau: np.ndarray) -> np.ndarray:
    tau = np.clip(tau, 0.0, 1.0)
    return 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5


def min_jerk_move_x(*, distance_m: float, duration_s: float, dt_s: float, hold_s: float, start_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (t_s (N,), true_tcp_pos (N,3)) for a min-jerk move along X
    then a hold, no noise. Matches hardware/motion.py's quintic profile."""
    n_move = int(round(duration_s / dt_s))
    n_hold = int(round(hold_s / dt_s))
    n = n_move + n_hold
    t_s = np.arange(n, dtype=np.float64) * dt_s
    tau = np.clip(t_s / duration_s, 0.0, 1.0)
    s = min_jerk_s(tau)
    pos = np.tile(start_pos[None, :], (n, 1)).astype(np.float64)
    pos[:, 0] = start_pos[0] + distance_m * s
    return t_s, pos


def analytic_peak_accel(distance_m: float, duration_s: float) -> float:
    return (10.0 * sqrt(3.0) / 3.0) * abs(distance_m) / (duration_s * duration_s)


# ---------------------------------------------------------------------------
# Real-noise bootstrap: resample real captured position-noise residuals
# (mean-removed, since the capture is stationary -> sample mean ~= true
# position) in fixed-length contiguous blocks, to preserve real short-range
# correlation structure instead of treating each sample as IID.
# ---------------------------------------------------------------------------


def bootstrap_real_noise(residuals: np.ndarray, *, n_samples: int, block_len: int, rng: np.random.Generator) -> np.ndarray:
    n_real = residuals.shape[0]
    out = np.empty((n_samples, residuals.shape[1]), dtype=np.float64)
    i = 0
    while i < n_samples:
        start = int(rng.integers(0, max(n_real - block_len, 1)))
        block = residuals[start:start + block_len]
        take = min(block.shape[0], n_samples - i)
        out[i:i + take] = block[:take]
        i += take
    return out


# ---------------------------------------------------------------------------
# Metric (a): noise suppression on real stationary data
# ---------------------------------------------------------------------------


def eval_noise_suppression(tcp_pos: np.ndarray, dt_s: float, jerk_psd: float, measurement_var: np.ndarray) -> dict:
    n = tcp_pos.shape[0]

    # Current production heuristic, two configs: default (gap=1/alpha=1.0)
    # and the validated noise-robust preset (gap=5/alpha=0.2).
    results = {}
    for name, gap, alpha in [("current_default", 1, 1.0), ("current_gap5_alpha0.2", 5, 0.2)]:
        limits = CartesianMoveLimits(
            max_off_axis_drift_m=10.0, max_orientation_error_rad=10.0, max_tcp_speed_mps=10.0,
            max_tcp_accel_mps2=1e9, max_waypoint_jump_m=10.0, max_axis_error_growth_steps=10_000,
            qd_max_radps=100.0, accel_gap_cycles=gap, speed_lowpass_alpha=alpha,
        )
        monitor = CartesianMoveMonitor(limits)
        pose0 = np.zeros(6); pose0[:3] = tcp_pos[0]
        monitor.set_start(pose0, move_axis_index=0)
        accel_series = []
        q = np.zeros(6); qd = np.zeros(6)
        for i in range(1, n):
            pose = np.zeros(6); pose[:3] = tcp_pos[i]
            monitor.check(q=q, qd=qd, tcp_pose=pose, target_tcp_pose=pose,
                           orientation_error_rad=0.0, axis_target_moving=True, dt_s=dt_s)
            accel_series.append(monitor._prev_speed_mps)  # smoothed speed, not accel; see below
        # Recompute accel the same way check() does, from the smoothed-speed trace
        speed_arr = np.array([s for s in accel_series if s is not None])
        accel_arr = np.abs(np.diff(speed_arr)) / dt_s
        results[name] = accel_arr

    kf_est = build_kalman_estimator(dt_s=dt_s, jerk_psd=jerk_psd, measurement_var_per_axis=measurement_var)
    kf_est.reset(tcp_pos[0])
    abg_est = build_alpha_beta_gamma_estimator(dt_s=dt_s, jerk_psd=jerk_psd, measurement_var_per_axis=measurement_var)
    abg_est.reset(tcp_pos[0])

    kf_scalar, kf_vector, abg_scalar, abg_vector = [], [], [], []
    for i in range(1, n):
        e_kf = kf_est.step(tcp_pos[i], dt_s)
        e_abg = abg_est.step(tcp_pos[i], dt_s)
        kf_scalar.append(e_kf.accel_scalar_mps2)
        kf_vector.append(e_kf.accel_vector_mps2)
        abg_scalar.append(e_abg.accel_scalar_mps2)
        abg_vector.append(e_abg.accel_vector_mps2)

    results["kalman_scalar_speed_diff"] = np.array(kf_scalar[50:])  # drop transient
    results["kalman_vector_norm"] = np.array(kf_vector[50:])
    results["alpha_beta_gamma_scalar_speed_diff"] = np.array(abg_scalar[50:])
    results["alpha_beta_gamma_vector_norm"] = np.array(abg_vector[50:])

    summary = {}
    for name, arr in results.items():
        summary[name] = {
            "mean": float(np.mean(arr)), "std": float(np.std(arr)),
            "p50": float(np.percentile(arr, 50)), "p99": float(np.percentile(arr, 99)),
            "p99.9": float(np.percentile(arr, 99.9)), "max": float(np.max(arr)),
        }
    return summary


# ---------------------------------------------------------------------------
# Metric (b): tracking lag on a genuine fast move + real bootstrapped noise
# ---------------------------------------------------------------------------


def eval_tracking_lag(
    residuals: np.ndarray, dt_s: float, jerk_psd: float, measurement_var: np.ndarray,
    *, distance_m: float, duration_s: float, hold_s: float, seed: int,
) -> dict:
    start_pos = np.array([-0.3155, -0.1720, 0.9325], dtype=np.float64)
    t_s, true_pos = min_jerk_move_x(distance_m=distance_m, duration_s=duration_s, dt_s=dt_s, hold_s=hold_s, start_pos=start_pos)
    rng = np.random.default_rng(seed)
    noise = bootstrap_real_noise(residuals, n_samples=true_pos.shape[0], block_len=20, rng=rng)
    noisy_pos = true_pos + noise

    peak_true = analytic_peak_accel(distance_m, duration_s)
    peak_true_idx = int(round(0.5 * duration_s / dt_s))  # min-jerk peak accel is at tau=0.5

    # current default (gap=1) and gap5/alpha0.2, and KF/alpha-beta-gamma
    out = {"analytic_peak_accel_mps2": peak_true, "analytic_peak_idx": peak_true_idx}

    for name, gap, alpha in [("current_default", 1, 1.0), ("current_gap5_alpha0.2", 5, 0.2)]:
        limits = CartesianMoveLimits(
            max_off_axis_drift_m=10.0, max_orientation_error_rad=10.0, max_tcp_speed_mps=10.0,
            max_tcp_accel_mps2=1e9, max_waypoint_jump_m=10.0, max_axis_error_growth_steps=10_000,
            qd_max_radps=100.0, accel_gap_cycles=gap, speed_lowpass_alpha=alpha,
        )
        monitor = CartesianMoveMonitor(limits)
        pose0 = np.zeros(6); pose0[:3] = noisy_pos[0]
        monitor.set_start(pose0, move_axis_index=0)
        speeds = []
        q = np.zeros(6); qd = np.zeros(6)
        for i in range(1, noisy_pos.shape[0]):
            pose = np.zeros(6); pose[:3] = noisy_pos[i]
            monitor.check(q=q, qd=qd, tcp_pose=pose, target_tcp_pose=pose,
                           orientation_error_rad=0.0, axis_target_moving=True, dt_s=dt_s)
            speeds.append(monitor._prev_speed_mps)
        speed_arr = np.array([s for s in speeds if s is not None])
        accel_arr = np.abs(np.diff(speed_arr)) / dt_s
        peak_idx = int(np.argmax(accel_arr))
        out[name] = {"peak_accel_mps2": float(accel_arr[peak_idx]), "peak_idx": peak_idx,
                     "lag_cycles": peak_idx - peak_true_idx, "lag_ms": (peak_idx - peak_true_idx) * dt_s * 1e3}

    kf_est = build_kalman_estimator(dt_s=dt_s, jerk_psd=jerk_psd, measurement_var_per_axis=measurement_var)
    kf_est.reset(noisy_pos[0])
    abg_est = build_alpha_beta_gamma_estimator(dt_s=dt_s, jerk_psd=jerk_psd, measurement_var_per_axis=measurement_var)
    abg_est.reset(noisy_pos[0])
    kf_vec, abg_vec = [], []
    for i in range(1, noisy_pos.shape[0]):
        kf_vec.append(kf_est.step(noisy_pos[i], dt_s).accel_vector_mps2)
        abg_vec.append(abg_est.step(noisy_pos[i], dt_s).accel_vector_mps2)
    for name, arr in [("kalman_vector_norm", kf_vec), ("alpha_beta_gamma_vector_norm", abg_vec)]:
        arr = np.array(arr)
        peak_idx = int(np.argmax(arr))
        out[name] = {"peak_accel_mps2": float(arr[peak_idx]), "peak_idx": peak_idx,
                     "lag_cycles": peak_idx - peak_true_idx, "lag_ms": (peak_idx - peak_true_idx) * dt_s * 1e3}
    return out


# ---------------------------------------------------------------------------
# Metric (c): pass/fail replay across the backtest's two reconstructable
# profiles (canonical_headroom, large_displacement -- see
# docs/status/safety_envelope_backtest_2026-07-30.md SS9; the third profile,
# "canonical" at base_accel=0.5, was disqualified there by real CONTROLLER
# torque-tracking chatter this script has no access to, not sensor noise --
# out of scope here, noted explicitly, not silently dropped)
# ---------------------------------------------------------------------------


def eval_pass_fail(residuals: np.ndarray, dt_s: float, jerk_psd: float, measurement_var: np.ndarray, *, n_seeds: int) -> dict:
    start_pos = np.array([-0.3155, -0.1720, 0.9325], dtype=np.float64)
    profiles = {
        "canonical_headroom": {"distance_m": 0.02, "duration_s": 1.0, "hold_s": 1.0, "base_accel": 4.5},
        "large_displacement": {"distance_m": -0.20, "duration_s": 1.0, "hold_s": 1.0, "base_accel": 0.8},
    }
    methods = {
        "current_default": {"gap": 1, "alpha": 1.0, "consec": 1, "hard": 5.0},
        "current_graduated_filtered": {"gap": 5, "alpha": 0.2, "consec": 3, "hard": 5.0},
    }
    out = {}
    for prof_name, prof in profiles.items():
        t_s, true_pos = min_jerk_move_x(distance_m=prof["distance_m"], duration_s=prof["duration_s"],
                                         dt_s=dt_s, hold_s=prof["hold_s"], start_pos=start_pos)
        base_accel = prof["base_accel"]
        prof_out = {}
        for m_name, m in methods.items():
            n_trip = 0
            for seed in range(n_seeds):
                rng = np.random.default_rng(1000 * seed + hash(prof_name) % 997)
                noise = bootstrap_real_noise(residuals, n_samples=true_pos.shape[0], block_len=20, rng=rng)
                noisy_pos = true_pos + noise
                _, trip_cycle = replay_through_production_monitor(
                    noisy_pos, dt_s=dt_s, max_tcp_accel_mps2=base_accel,
                    accel_gap_cycles=m["gap"], speed_lowpass_alpha=m["alpha"],
                    accel_max_consecutive_violations=m["consec"], accel_hard_multiple=m["hard"],
                )
                if trip_cycle is not None:
                    n_trip += 1
            prof_out[m_name] = {"trips": n_trip, "n_seeds": n_seeds}

        # KF/alpha-beta-gamma candidates: same graduated consecutive/hard-multiple
        # decision rule (mirrored here, not reimplementing the WHOLE monitor -- see
        # the module docstring), applied to the vector-norm accel estimate.
        for est_name, builder in [("kalman_vector_norm", build_kalman_estimator),
                                   ("alpha_beta_gamma_vector_norm", build_alpha_beta_gamma_estimator)]:
            n_trip = 0
            for seed in range(n_seeds):
                rng = np.random.default_rng(1000 * seed + hash(prof_name) % 997)
                noise = bootstrap_real_noise(residuals, n_samples=true_pos.shape[0], block_len=20, rng=rng)
                noisy_pos = true_pos + noise
                est = builder(dt_s=dt_s, jerk_psd=jerk_psd, measurement_var_per_axis=measurement_var)
                est.reset(noisy_pos[0])
                consecutive = 0
                tripped = False
                for i in range(1, noisy_pos.shape[0]):
                    a = est.step(noisy_pos[i], dt_s).accel_vector_mps2
                    if a > base_accel:
                        hard_ceiling = base_accel * 5.0
                        if a >= hard_ceiling:
                            tripped = True
                            break
                        consecutive += 1
                        if consecutive >= 3:
                            tripped = True
                            break
                    else:
                        consecutive = 0
                if tripped:
                    n_trip += 1
            prof_out[est_name] = {"trips": n_trip, "n_seeds": n_seeds}
        out[prof_name] = prof_out
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
    p.add_argument("--jerk-psd", type=float, default=4000.0,
                    help="Process-noise PSD (m^2/s^5) for the constant-acceleration KF. "
                         "Default chosen by the grid search printed with --tune-jerk-psd.")
    p.add_argument("--seeds", type=int, default=30, help="Seeds for the pass/fail Monte Carlo (matches the "
                                                            "backtest's own 30-seed convention).")
    p.add_argument("--tune-jerk-psd", action="store_true",
                    help="Print a small grid search over jerk_psd trading off noise floor vs tracking lag, then exit.")
    p.add_argument("--json-out", type=Path, default=None)
    args = p.parse_args()

    t_s, tcp_pos, used_real = load_real_stationary_capture(args.capture)
    print(f"[data] {'REAL' if used_real else 'SYNTHESIZED FALLBACK'} stationary capture, {tcp_pos.shape[0]} samples")
    real_dt = np.diff(t_s)
    dt_s = float(np.median(real_dt)) if used_real else DT_S
    print(f"[data] median inter-sample dt: {dt_s * 1e3:.4f} ms ({1.0 / dt_s:.1f} Hz)")

    measured_std = np.std(tcp_pos, axis=0)
    print(f"[data] measured per-axis position std (this capture): {measured_std.tolist()} m")
    print(f"[data] documented std (stationary_noise_capture_154018_stats.json): "
          f"{FALLBACK_TCP_POS_STD_M.tolist()} m")
    measurement_var = measured_std ** 2
    residuals = tcp_pos - np.mean(tcp_pos, axis=0, keepdims=True)

    if args.tune_jerk_psd:
        print("\n[jerk_psd tuning grid] noise-floor p99 (stationary) vs tracking-lag/peak-undershoot "
              "(large_displacement dx=-0.20m/T=1.0s move)")
        for jerk_psd in [10.0, 100.0, 1000.0, 4000.0, 10000.0, 50000.0, 200000.0]:
            noise_summary = eval_noise_suppression(tcp_pos, dt_s, jerk_psd, measurement_var)
            lag = eval_tracking_lag(residuals, dt_s, jerk_psd, measurement_var,
                                     distance_m=-0.20, duration_s=1.0, hold_s=0.5, seed=0)
            kf_p99 = noise_summary["kalman_vector_norm"]["p99"]
            kf_peak = lag["kalman_vector_norm"]["peak_accel_mps2"]
            kf_lag_ms = lag["kalman_vector_norm"]["lag_ms"]
            true_peak = lag["analytic_peak_accel_mps2"]
            print(f"  jerk_psd={jerk_psd:>10.1f}  stationary_p99={kf_p99:.4f} m/s^2  "
                  f"move_peak={kf_peak:.4f} m/s^2 (true={true_peak:.4f}, "
                  f"{100*kf_peak/true_peak:.1f}%)  lag={kf_lag_ms:+.2f} ms")
        return 0

    results: dict = {"used_real_capture": used_real, "dt_s": dt_s, "jerk_psd": args.jerk_psd,
                      "measurement_var_per_axis": measurement_var.tolist()}

    print("\n=== (a) Noise suppression on real stationary capture (zero true motion) ===")
    noise_summary = eval_noise_suppression(tcp_pos, dt_s, args.jerk_psd, measurement_var)
    results["noise_suppression"] = noise_summary
    for name, s in noise_summary.items():
        print(f"  {name:38s}  mean={s['mean']:.5f}  std={s['std']:.5f}  p99={s['p99']:.5f}  "
              f"p99.9={s['p99.9']:.5f}  max={s['max']:.5f}  (m/s^2)")

    print("\n=== (b) Tracking lag on a genuine fast move (dx=-0.20m, T=1.0s min-jerk, "
          "real bootstrapped noise, seed 0) ===")
    lag = eval_tracking_lag(residuals, dt_s, args.jerk_psd, measurement_var,
                             distance_m=-0.20, duration_s=1.0, hold_s=0.5, seed=0)
    results["tracking_lag"] = lag
    print(f"  analytic peak accel: {lag['analytic_peak_accel_mps2']:.4f} m/s^2 at t={lag['analytic_peak_idx']*dt_s:.3f}s")
    for name, v in lag.items():
        if name in ("analytic_peak_accel_mps2", "analytic_peak_idx"):
            continue
        print(f"  {name:38s}  peak={v['peak_accel_mps2']:.4f} m/s^2 "
              f"({100*v['peak_accel_mps2']/lag['analytic_peak_accel_mps2']:.1f}% of true)  lag={v['lag_ms']:+.2f} ms")

    print(f"\n=== (c) Pass/fail replay, {args.seeds} seeds/profile, real bootstrapped noise ===")
    pass_fail = eval_pass_fail(residuals, dt_s, args.jerk_psd, measurement_var, n_seeds=args.seeds)
    results["pass_fail"] = pass_fail
    for prof_name, prof_out in pass_fail.items():
        print(f"  profile: {prof_name}")
        for m_name, r in prof_out.items():
            print(f"    {m_name:38s}  {r['trips']}/{r['n_seeds']} trips")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\n[saved] {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
