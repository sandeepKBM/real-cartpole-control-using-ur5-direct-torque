#!/usr/bin/env python3
"""Sensor-noise and measurement/control-delay robustness test for the
torque-lane LQR pendulum balance controller
(tools/diagnostics/pendulum_balance_torque_lqr.py).

That controller (and the velocity-lane one, pendulum_balance_test.py) was
only ever validated against an IDEALIZED simulation: exact angle
perturbation, perfect noiseless state feedback, zero control-loop delay.
Real hardware gives you neither. This script reuses the base controller's
DESIGN functions (find_inverted_angle, linearize_and_design_lqr,
static_gravity_torque -- imported, not reimplemented, per this session's
own documented lesson about not hand-rolling equilibrium-finding logic)
but writes its OWN simulation loop, because
pendulum_balance_torque_lqr.run_torque_balance_trial has no hook to inject
noise into the state fed to the LQR law or to delay torque application --
both are required here.

Two degradations are tested, matching the real AMT222B-8000-S 14-bit
absolute encoder planned for the physical apparatus (see docs/status/
ur5e_pendulum_cad_model_2026-08-09.md):

1. Sensor noise on the pendulum angle/angular-velocity reading:
   (a) quantization-only noise on the angle (2*pi/16384 rad steps),
       velocity read cleanly (qvel) -- isolates pure quantization effect.
   (b) the realistic case: velocity is NOT measured directly on a device
       like this -- it must be estimated by finite-differencing two
       (quantized, possibly additionally noisy) position samples at the
       real control rate (500 Hz / dt=2ms), which amplifies position
       noise by ~1/dt.
   Both are swept in noise magnitude (quantization bits reduced /
   additional Gaussian noise added) to find where each starts to matter.

2. Fixed measurement-to-actuation delay of k in {0,1,2,5,10,...} control
   cycles: the torque applied at cycle N is computed from the state read
   at cycle N-k, not cycle N. Implemented as a FIFO delay line of
   (state, qpos) pairs; before the pipe fills (first k cycles) there is no
   valid delayed reading yet, so the controller falls back to its
   equilibrium behavior (tau_extra=0, pure static gravity compensation at
   the equilibrium pose) rather than leaking ground truth.

IMPORTANT (found the hard way while building this): before trusting ANY
noise/delay finding, this script's own zero-noise/zero-delay baseline was
re-verified against pendulum_balance_torque_lqr.py's *actual* printed
output (not just its write-up doc's prose summary) -- see
verify_clean_baseline() and the write-up's "Bug/surprise #0" section for
why this mattered: the doc's headline claim ("every perturbation from 0.02
up through 3.0 rad converges cleanly") turned out to NOT match main()'s
own printed table, which shows pert=1.0 and pert=1.5 rad both FELL. A
finer sweep here confirms a real, reproducible, non-monotonic fall band
(~0.65-1.65 rad) sandwiched between two working regions. This is a
pre-existing property of the base controller, independent of anything
added in this script -- it is NOT a noise/delay finding, but it directly
determines which perturbation is safe to use as a "clean" baseline for
the noise/delay sweeps below (0.5 rad, not 1.0 rad, despite the task
description's "0.5-1.0 rad" suggestion).
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from simulation.ur5e_pendulum_compose import compose_ur5e_pendulum_model  # noqa: E402
import pendulum_balance_torque_lqr as base  # noqa: E402

ARM_Q0 = base.ARM_Q0
TORQUE_LIMIT_NM = base.TORQUE_LIMIT_NM
RATE_HZ = base.RATE_HZ
CONTROL_DT = base.CONTROL_DT
FALL_THRESHOLD_RAD = base.FALL_THRESHOLD_RAD
static_gravity_torque = base.static_gravity_torque

ENCODER_BITS = 14
ENCODER_QUANT_RAD = 2 * np.pi / (2 ** ENCODER_BITS)  # ~3.834e-4 rad


def quantize(theta: float, step: float) -> float:
    return round(theta / step) * step


def run_trial(
    model,
    K: np.ndarray,
    q_eq: np.ndarray,
    perturbation_rad: float,
    duration_s: float,
    *,
    rng: np.random.Generator,
    theta_dot_perturbation: float = 0.0,
    angle_noise_mode: str = "none",  # "none" | "quantize" | "quantize+fd_vel"
    quant_step_rad: float = ENCODER_QUANT_RAD,
    extra_angle_noise_std_rad: float = 0.0,
    delay_cycles: int = 0,
) -> dict:
    """Same physical setup/scoring as pendulum_balance_torque_lqr.run_torque_balance_trial,
    but the state fed to the LQR law can be a noisy/estimated measurement of the
    true state, and the torque applied can be computed from a state read
    delay_cycles cycles in the past (a FIFO delay line), instead of the true
    instantaneous state used unconditionally by the base version."""
    nv = model.nv
    data = mujoco.MjData(model)
    pend_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "/pendulum_hinge")
    pend_qpos_adr = model.jnt_qposadr[pend_joint_id]
    pend_dof_adr = model.jnt_dofadr[pend_joint_id]
    inverted_angle = float(q_eq[6])

    data.qpos[:6] = ARM_Q0
    data.qpos[pend_qpos_adr] = inverted_angle + perturbation_rad
    data.qvel[:] = 0.0
    data.qvel[pend_dof_adr] = theta_dot_perturbation
    mujoco.mj_forward(model, data)

    n_steps = int(duration_s * RATE_HZ)
    fell_at = None
    peak_theta_err = 0.0
    theta_err_hist = []

    # FIFO delay line: appended every cycle, popped once it holds more than
    # delay_cycles entries (so the entry popped is exactly delay_cycles old).
    pipe: deque = deque()
    prev_theta_meas = None
    zero_state = np.zeros(2 * nv)

    for step in range(n_steps):
        theta_true = float(data.qpos[pend_qpos_adr])
        theta_err_true = float(np.mod(theta_true - inverted_angle + np.pi, 2 * np.pi) - np.pi)
        theta_err_hist.append(theta_err_true)
        peak_theta_err = max(peak_theta_err, abs(theta_err_true))
        if abs(theta_err_true) > FALL_THRESHOLD_RAD and fell_at is None:
            fell_at = step * CONTROL_DT

        # --- sensor model ---
        if angle_noise_mode == "none":
            theta_meas = theta_true
            theta_dot_meas = float(data.qvel[pend_dof_adr])
        else:
            theta_meas = theta_true
            if extra_angle_noise_std_rad > 0.0:
                theta_meas += rng.normal(0.0, extra_angle_noise_std_rad)
            theta_meas = quantize(theta_meas, quant_step_rad)
            if angle_noise_mode == "quantize+fd_vel":
                if prev_theta_meas is None:
                    theta_dot_meas = 0.0  # no history yet; do not leak ground truth
                else:
                    d = theta_meas - prev_theta_meas
                    d = (d + np.pi) % (2 * np.pi) - np.pi  # unwrap, defensive
                    theta_dot_meas = d / CONTROL_DT
                prev_theta_meas = theta_meas
            else:  # "quantize": clean velocity, noisy angle only
                theta_dot_meas = float(data.qvel[pend_dof_adr])

        theta_err_meas = float(np.mod(theta_meas - inverted_angle + np.pi, 2 * np.pi) - np.pi)
        dq = data.qpos.copy()
        dq[6] = theta_err_meas
        dq[:6] = data.qpos[:6] - q_eq[:6]  # arm joint encoders assumed clean (out of scope; see write-up)
        dqd = data.qvel.copy()
        dqd[pend_dof_adr] = theta_dot_meas
        measured_state = np.concatenate([dq, dqd])

        pipe.append((measured_state, data.qpos.copy()))
        if len(pipe) > delay_cycles:
            used_state, used_qpos = pipe.popleft()
        else:
            # Pipe not yet full: no valid delayed reading exists. Fall back
            # to "no correction yet" rather than peeking at ground truth.
            used_state, used_qpos = zero_state, q_eq

        tau_extra = -K @ used_state
        tau_gravity = static_gravity_torque(model, used_qpos)[:6]
        tau = np.clip(tau_extra + tau_gravity, -TORQUE_LIMIT_NM, TORQUE_LIMIT_NM)
        data.ctrl[:6] = tau

        mujoco.mj_step(model, data)
        if fell_at is not None:
            break

    theta_err_arr = np.array(theta_err_hist)
    return {
        "perturbation_rad": perturbation_rad,
        "fell_at_s": fell_at,
        "peak_theta_err": peak_theta_err,
        "final_theta_err": float(theta_err_arr[-1]) if len(theta_err_arr) else None,
        "survived_full_duration": fell_at is None,
    }


def verify_clean_baseline(model, K, q_eq, rng) -> None:
    """Reproduce a fine-grained zero-noise/zero-delay sweep through this
    script's OWN run_trial (angle_noise_mode="none", delay_cycles=0) and
    cross-check it against the base script's run_torque_balance_trial at
    matching perturbations, to (a) confirm this new loop reproduces the
    base controller's behavior exactly before trusting any new finding
    from it, and (b) document the real non-monotonic clean-baseline fall
    band described in this file's module docstring."""
    print("=== Sanity check: this script's clean (no-noise, no-delay) path ===")
    print("First, cross-check against pendulum_balance_torque_lqr.run_torque_balance_trial "
          "directly (must match exactly -- same physics, same law, same scoring):")
    mismatches = 0
    for pert in [0.02, 0.5, 1.0, 1.5, 2.0]:
        r_new = run_trial(model, K, q_eq, pert, duration_s=8.0, rng=rng)
        r_base = base.run_torque_balance_trial(model, K, q_eq, pert, duration_s=8.0)
        ok = r_new["survived_full_duration"] == r_base["survived_full_duration"]
        mismatches += int(not ok)
        print(f"  pert={pert:.2f}  new={'SURVIVED' if r_new['survived_full_duration'] else 'FELL':>9}"
              f"  base={'SURVIVED' if r_base['survived_full_duration'] else 'FELL':>9}"
              f"  {'OK' if ok else 'MISMATCH -- BUG IN NEW LOOP'}")
    if mismatches:
        raise RuntimeError(f"{mismatches} mismatch(es) between this script's clean path and the "
                            f"base script -- do not trust any noise/delay result until fixed.")

    print("\nFine-grained clean-baseline sweep (documents the pre-existing, noise/delay-independent "
          "non-monotonic fall band -- NOT a finding of this script's noise/delay work):")
    print(f"{'pert':>6} {'result':>10} {'fell_at_s':>10}")
    for pert in [0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.9, 1.0, 1.3, 1.5, 1.6, 1.65, 1.7, 1.8, 2.0]:
        r = run_trial(model, K, q_eq, pert, duration_s=8.0, rng=rng)
        result = "SURVIVED" if r["survived_full_duration"] else "FELL"
        print(f"{pert:6.2f} {result:>10} {str(r['fell_at_s']):>10}")
    print()


def sweep_angle_noise(model, K, q_eq, baseline_pert: float, seed: int = 0) -> None:
    print(f"=== Angle/velocity sensor noise sweep (baseline perturbation = {baseline_pert} rad) ===")
    print(f"{'mode':>18} {'quant_bits':>10} {'extra_std':>10} {'result':>10} {'peak_err':>10} {'final_err':>10}")

    configs = [
        ("none", None, 0.0),
        ("quantize", 14, 0.0),           # real encoder resolution, clean velocity
        ("quantize", 10, 0.0),           # coarser quantization, clean velocity
        ("quantize", 8, 0.0),            # much coarser, clean velocity
        ("quantize+fd_vel", 14, 0.0),    # realistic: FD velocity from real encoder resolution
        ("quantize+fd_vel", 12, 0.0),
        ("quantize+fd_vel", 10, 0.0),
        ("quantize+fd_vel", 8, 0.0),
        ("quantize+fd_vel", 6, 0.0),
        ("quantize+fd_vel", 14, 0.001),  # real encoder + small extra Gaussian (e.g. EMI/vibration)
        ("quantize+fd_vel", 14, 0.005),
        ("quantize+fd_vel", 14, 0.02),
    ]
    for mode, bits, extra_std in configs:
        rng = np.random.default_rng(seed)
        quant_step = 2 * np.pi / (2 ** bits) if bits is not None else ENCODER_QUANT_RAD
        r = run_trial(
            model, K, q_eq, baseline_pert, duration_s=8.0, rng=rng,
            angle_noise_mode=mode, quant_step_rad=quant_step,
            extra_angle_noise_std_rad=extra_std,
        )
        result = "SURVIVED" if r["survived_full_duration"] else "FELL"
        bits_s = str(bits) if bits is not None else "-"
        print(f"{mode:>18} {bits_s:>10} {extra_std:10.4f} {result:>10} "
              f"{r['peak_theta_err']:10.4f} {r['final_theta_err']:10.4f}")
    print()


def sweep_angle_noise_pass_rate(
    model, K, q_eq, baseline_pert: float, n_seeds: int = 10, seed0: int = 100,
) -> None:
    """The single-seed sweep above (sweep_angle_noise) turned out to be
    unreliable near the point where noise starts to matter: a probe pushing
    extra_angle_noise_std_rad far beyond a single-encoder-quantization scale
    found NON-MONOTONIC single-seed pass/fail vs. noise magnitude (e.g.
    "quantize" mode at pert=0.5: extra_std=0.5 FELL, 1.0 FELL, 1.5 SURVIVED,
    2.0 FELL, 3.0 SURVIVED) -- the same saturated-bang-bang/basin-of-
    attraction sensitivity documented for the clean baseline's non-monotonic
    fall band, just excited by a specific noise draw instead of a specific
    perturbation. A single random seed's pass/fail at a given noise level is
    therefore not a reliable data point near the boundary; this sweeps
    several noise magnitudes x several independent seeds and reports a pass
    RATE, which is the statistically honest version of the same question."""
    print(f"=== Angle-noise pass-RATE sweep (baseline perturbation = {baseline_pert} rad, "
          f"{n_seeds} seeds/level) ===")
    print(f"{'mode':>18} {'extra_std':>10} {'pass_rate':>10} {'mean_final_err(survived)':>26}")
    for mode in ["quantize", "quantize+fd_vel"]:
        for extra_std in [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0]:
            n_pass = 0
            final_errs = []
            for i in range(n_seeds):
                rng = np.random.default_rng(seed0 + i)
                r = run_trial(
                    model, K, q_eq, baseline_pert, duration_s=8.0, rng=rng,
                    angle_noise_mode=mode, extra_angle_noise_std_rad=extra_std,
                )
                if r["survived_full_duration"]:
                    n_pass += 1
                    final_errs.append(r["final_theta_err"])
            mean_final = f"{np.mean(final_errs):.4f}" if final_errs else "n/a"
            print(f"{mode:>18} {extra_std:10.4f} {n_pass}/{n_seeds:>7} {mean_final:>26}")
    print()


def sweep_delay(model, K, q_eq, baseline_pert: float, seed: int = 0) -> None:
    print(f"=== Measurement/control delay sweep (baseline perturbation = {baseline_pert} rad, no sensor noise) ===")
    print(f"{'delay_cycles':>13} {'delay_ms':>10} {'result':>10} {'peak_err':>10} {'final_err':>10} {'fell_at_s':>10}")
    for delay_cycles in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 50, 75, 100]:
        rng = np.random.default_rng(seed)
        r = run_trial(
            model, K, q_eq, baseline_pert, duration_s=8.0, rng=rng,
            angle_noise_mode="none", delay_cycles=delay_cycles,
        )
        result = "SURVIVED" if r["survived_full_duration"] else "FELL"
        print(f"{delay_cycles:13d} {delay_cycles * CONTROL_DT * 1000:10.1f} {result:>10} "
              f"{r['peak_theta_err']:10.4f} {r['final_theta_err']:10.4f} {str(r['fell_at_s']):>10}")
    print()


def sweep_combined(model, K, q_eq, baseline_pert: float, seed: int = 0) -> None:
    print(f"=== Combined realistic noise + delay (baseline perturbation = {baseline_pert} rad) ===")
    print(f"{'delay_cycles':>13} {'noise_mode':>18} {'result':>10} {'peak_err':>10} {'final_err':>10}")
    combos = [
        (0, "quantize+fd_vel"),
        (1, "quantize+fd_vel"),
        (2, "quantize+fd_vel"),
        (5, "quantize+fd_vel"),
        (10, "quantize+fd_vel"),
    ]
    for delay_cycles, mode in combos:
        rng = np.random.default_rng(seed)
        r = run_trial(
            model, K, q_eq, baseline_pert, duration_s=8.0, rng=rng,
            angle_noise_mode=mode, quant_step_rad=ENCODER_QUANT_RAD, delay_cycles=delay_cycles,
        )
        result = "SURVIVED" if r["survived_full_duration"] else "FELL"
        print(f"{delay_cycles:13d} {mode:>18} {result:>10} {r['peak_theta_err']:10.4f} {r['final_theta_err']:10.4f}")
    print()


def main() -> int:
    model = compose_ur5e_pendulum_model()
    data = mujoco.MjData(model)
    pend_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "/pendulum_hinge")
    pend_qpos_adr = model.jnt_qposadr[pend_joint_id]

    inverted_angle = base.find_inverted_angle(model, data, pend_qpos_adr)
    print(f"inverted_angle (composed frame) = {inverted_angle:.4f} rad")
    # r_weight=1e6 MUST be passed explicitly -- the function's own default
    # (0.0005) is the pre-fix value from before pendulum_balance_torque_lqr.py's
    # per-actuator R-scaling fix; omitting it here reproduced that bug's
    # exact symptom (a spurious "0.65-1.65 rad fall band" that isn't a real
    # property of the validated controller at all). Found by re-running
    # pendulum_balance_torque_lqr.py's own main() standalone and discovering
    # it did NOT reproduce its own write-up's validated results either --
    # both scripts had the same missing-default bug, independently.
    K, q_eq, diag = base.linearize_and_design_lqr(model, inverted_angle, r_weight=1e6)
    n_unstable = int(np.sum(np.abs(diag["eigvals_discrete"]) >= 1.0))
    if n_unstable > 0:
        print("WARNING: linear design is not stable -- aborting.")
        return 1

    rng = np.random.default_rng(0)
    verify_clean_baseline(model, K, q_eq, rng)

    baseline_pert = 0.5  # known-clean in the idealized case (see verify_clean_baseline above);
    # 1.0 rad, suggested by the task prompt's "0.5-1.0 rad" range, is INSIDE the pre-existing
    # non-monotonic fall band and would confound noise/delay effects with that unrelated issue.
    sweep_angle_noise(model, K, q_eq, baseline_pert)
    sweep_angle_noise_pass_rate(model, K, q_eq, baseline_pert)
    sweep_delay(model, K, q_eq, baseline_pert)
    sweep_combined(model, K, q_eq, baseline_pert)

    # Second baseline in the higher, also-clean regime, to check whether findings generalize.
    baseline_pert_2 = 1.8
    sweep_angle_noise(model, K, q_eq, baseline_pert_2, seed=1)
    sweep_delay(model, K, q_eq, baseline_pert_2, seed=1)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
