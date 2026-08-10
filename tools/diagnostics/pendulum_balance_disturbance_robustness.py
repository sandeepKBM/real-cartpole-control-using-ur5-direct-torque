#!/usr/bin/env python3
"""Sensor-noise and measurement/control-delay robustness test for the
pendulum balance controllers -- BOTH lanes, each using its own
DE-search-validated gains (docs/status/pendulum_balance_gain_search_2026-08-09.md),
not hand-picked guesses.

REWRITTEN 2026-08-10: the original version of this file (see git history) only
covered the torque lane, and used that lane's OLD default-Q, r_weight=1e6 LQR
design -- NOT the DE-search-found gains that are actually this repo's
validated "best" torque-lane controller. It also picked baseline perturbations
(0.5, 1.8 rad) by characterizing that OLD design's own fall band, which is
irrelevant to the NEW gain-search config's (different) envelope. This version:
  1. Uses the real validated gains for BOTH lanes.
  2. Picks baseline perturbations from EACH lane's own already-documented
     genuinely-clean range (torque: 0.05-0.4 rad; velocity: 0.1-0.3 rad --
     the velocity lane's 0.8-1.2 rad "survives" is a real but DIFFERENT,
     weaker claim -- "arrests divergence near the release point", not
     "converges to vertical" -- and deliberately excluded here so a
     noise/delay finding isn't confounded with that pre-existing false-
     positive risk).
  3. Adds a full velocity-lane run_trial (noise + delay), mirroring the
     torque-lane one -- the base pendulum_balance_test.py has no such hook.

Two degradations are tested per lane, matching the real AMT222B-8000-S 14-bit
absolute encoder planned for the physical apparatus (see docs/status/
ur5e_pendulum_cad_model_2026-08-09.md):

1. Sensor noise on the pendulum angle/angular-velocity reading:
   (a) quantization-only noise on the angle (2*pi/16384 rad steps),
       velocity read cleanly -- isolates pure quantization effect.
   (b) the realistic case: velocity is NOT measured directly on a device
       like this -- it must be estimated by finite-differencing two
       (quantized, possibly additionally noisy) position samples at the
       real control rate, which amplifies position noise by ~1/dt.

2. Fixed measurement-to-actuation delay of k control cycles: the command
   applied at cycle N is computed from the state read at cycle N-k. Before
   the pipe fills (first k cycles) there is no valid delayed reading yet, so
   the controller falls back to "no correction yet" rather than leaking
   ground truth.

IMPORTANT: before trusting ANY noise/delay finding for a lane, this script's
own zero-noise/zero-delay path is cross-checked against that lane's base
script's own trial function at the SAME validated gains (verify_clean_baseline_*).
"""

from __future__ import annotations

import sys
from collections import deque
from functools import partial
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pendulum_balance_torque_lqr as torque_base  # noqa: E402
import pendulum_balance_test as vel_base  # noqa: E402

ENCODER_BITS = 14
ENCODER_QUANT_RAD = 2 * np.pi / (2 ** ENCODER_BITS)  # ~3.834e-4 rad

# DE-search-validated gains (docs/status/pendulum_balance_gain_search_2026-08-09.md).
TORQUE_GAIN_KWARGS = dict(
    q_arm_pos=2655.206336272164, q_arm_vel=1.588066109094743,
    q_pend_angle=20.947598183637, q_pend_vel=655.1987898599957,
    r_weight=0.34681947401205965,
)
VELOCITY_GAINS = dict(kp=2.6786107222104634, kd=0.03623843168261678,
                       ik_joint_gain=432.80055226571517)

# Each lane's genuinely-clean (real convergence, not "arrested near release
# point") perturbation range, per the gain-search doc -- baselines must stay
# inside it or a noise/delay finding is confounded with the lane's own
# pre-existing structural envelope.
TORQUE_BASELINE_PERTS = [0.15, 0.35]
# VELOCITY: the gain-search doc's claimed "clean 0.1-0.3 rad" table does NOT
# reproduce today -- direct re-check (2026-08-10) of vel_base.run_balance_trial
# at these exact validated gains found pert=0.15 and 0.20 rad BOTH FELL
# (final_theta_err -1.50/-1.64, fell_at_s 2.85/2.03), while 0.10/0.25/0.30
# survived -- a non-monotonic, fragile pass/fail boundary, not the smooth
# clean range the doc describes (stale doc, or a marginal design right at a
# basin-of-attraction edge -- not root-caused further here, out of scope for
# a disturbance-robustness test). Baselines below are values RE-CONFIRMED to
# survive cleanly just now, not copied from the doc's table.
VELOCITY_BASELINE_PERTS = [0.10, 0.30]


def quantize(theta: float, step: float) -> float:
    return round(theta / step) * step


# --------------------------------------------------------------------------
# Torque lane
# --------------------------------------------------------------------------

def run_trial_torque(
    model, K: np.ndarray, q_eq: np.ndarray,
    perturbation_rad: float, duration_s: float, *,
    rng: np.random.Generator,
    theta_dot_perturbation: float = 0.0,
    angle_noise_mode: str = "none",
    quant_step_rad: float = ENCODER_QUANT_RAD,
    extra_angle_noise_std_rad: float = 0.0,
    delay_cycles: int = 0,
) -> dict:
    """Same physical setup/scoring as pendulum_balance_torque_lqr.run_torque_balance_trial,
    but the state fed to the LQR law can be a noisy/estimated measurement, and
    the torque applied can be delayed by a FIFO delay line."""
    nv = model.nv
    data = mujoco.MjData(model)
    pend_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "/pendulum_hinge")
    pend_qpos_adr = model.jnt_qposadr[pend_joint_id]
    pend_dof_adr = model.jnt_dofadr[pend_joint_id]
    inverted_angle = float(q_eq[6])

    data.qpos[:6] = torque_base.ARM_Q0
    data.qpos[pend_qpos_adr] = inverted_angle + perturbation_rad
    data.qvel[:] = 0.0
    data.qvel[pend_dof_adr] = theta_dot_perturbation
    mujoco.mj_forward(model, data)

    n_steps = int(duration_s * torque_base.RATE_HZ)
    fell_at = None
    peak_theta_err = 0.0
    theta_err_hist = []

    pipe: deque = deque()
    prev_theta_meas = None
    zero_state = np.zeros(2 * nv)

    for step in range(n_steps):
        theta_true = float(data.qpos[pend_qpos_adr])
        theta_err_true = float(np.mod(theta_true - inverted_angle + np.pi, 2 * np.pi) - np.pi)
        theta_err_hist.append(theta_err_true)
        peak_theta_err = max(peak_theta_err, abs(theta_err_true))
        if abs(theta_err_true) > torque_base.FALL_THRESHOLD_RAD and fell_at is None:
            fell_at = step * torque_base.CONTROL_DT

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
                    theta_dot_meas = 0.0
                else:
                    d = theta_meas - prev_theta_meas
                    d = (d + np.pi) % (2 * np.pi) - np.pi
                    theta_dot_meas = d / torque_base.CONTROL_DT
                prev_theta_meas = theta_meas
            else:
                theta_dot_meas = float(data.qvel[pend_dof_adr])

        theta_err_meas = float(np.mod(theta_meas - inverted_angle + np.pi, 2 * np.pi) - np.pi)
        dq = data.qpos.copy()
        dq[6] = theta_err_meas
        dq[:6] = data.qpos[:6] - q_eq[:6]
        dqd = data.qvel.copy()
        dqd[pend_dof_adr] = theta_dot_meas
        measured_state = np.concatenate([dq, dqd])

        pipe.append((measured_state, data.qpos.copy()))
        if len(pipe) > delay_cycles:
            used_state, used_qpos = pipe.popleft()
        else:
            used_state, used_qpos = zero_state, q_eq

        tau_extra = -K @ used_state
        tau_gravity = torque_base.static_gravity_torque(model, used_qpos)[:6]
        tau = np.clip(tau_extra + tau_gravity, -torque_base.TORQUE_LIMIT_NM, torque_base.TORQUE_LIMIT_NM)
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


def verify_clean_baseline_torque(model, K, q_eq, rng) -> None:
    print("=== [TORQUE lane] Sanity check: clean (no-noise, no-delay) path ===")
    print("Cross-check against pendulum_balance_torque_lqr.run_torque_balance_trial directly:")
    mismatches = 0
    for pert in [0.02, 0.15, 0.35, 0.5]:
        r_new = run_trial_torque(model, K, q_eq, pert, duration_s=8.0, rng=rng)
        r_base = torque_base.run_torque_balance_trial(model, K, q_eq, pert, duration_s=8.0)
        ok = r_new["survived_full_duration"] == r_base["survived_full_duration"]
        mismatches += int(not ok)
        print(f"  pert={pert:.2f}  new={'SURVIVED' if r_new['survived_full_duration'] else 'FELL':>9}"
              f"  base={'SURVIVED' if r_base['survived_full_duration'] else 'FELL':>9}"
              f"  {'OK' if ok else 'MISMATCH -- BUG IN NEW LOOP'}")
    if mismatches:
        raise RuntimeError(f"{mismatches} mismatch(es) -- do not trust any torque-lane result until fixed.")
    print(f"Using validated gains {TORQUE_GAIN_KWARGS}, genuinely-clean range 0.05-0.4 rad "
          f"(docs/status/pendulum_balance_gain_search_2026-08-09.md); baselines here: {TORQUE_BASELINE_PERTS}\n")


# --------------------------------------------------------------------------
# Velocity lane
# --------------------------------------------------------------------------

def run_trial_velocity(
    kp: float, kd: float, ik_joint_gain: float,
    perturbation_rad: float, duration_s: float, *,
    rng: np.random.Generator,
    angle_noise_mode: str = "none",
    quant_step_rad: float = ENCODER_QUANT_RAD,
    extra_angle_noise_std_rad: float = 0.0,
    delay_cycles: int = 0,
    max_lin_speed_mps: float = 1000.0,
) -> dict:
    """Same physical setup/scoring as pendulum_balance_test.run_balance_trial,
    but the (theta_err, theta_dot) fed to the outer PD law can be a noisy/
    estimated measurement, and can be delayed by a FIFO delay line -- neither
    hook exists in the base script, which assumes perfect instantaneous
    feedback throughout."""
    model = vel_base.compose_ur5e_pendulum_model()
    model.opt.timestep = vel_base.PHYSICS_DT
    data = mujoco.MjData(model)
    pend_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "/pendulum_hinge")
    pend_qpos_adr = model.jnt_qposadr[pend_joint_id]
    pend_qvel_adr = model.jnt_dofadr[pend_joint_id]
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")

    _, inverted_angle = vel_base.find_inverted_angle(model, data, pend_qpos_adr)

    data.qpos[:6] = vel_base.ARM_Q0
    data.qpos[pend_qpos_adr] = inverted_angle + perturbation_rad
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)

    q_arm = vel_base.ARM_Q0.copy()
    scratch = mujoco.MjData(model)
    jp = np.zeros((3, model.nv))
    jr = np.zeros((3, model.nv))

    def fk_jacobian_fn(q):
        scratch.qpos[:6] = q
        mujoco.mj_forward(model, scratch)
        pos = np.asarray(scratch.site_xpos[site_id], dtype=np.float64).copy()
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, scratch.site_xmat[site_id])
        mujoco.mj_jacSite(model, scratch, jp, jr, site_id)
        jac6 = np.vstack([jp[:, :6], jr[:, :6]])
        return pos, quat, jac6.copy()

    p0, quat0, jac0 = fk_jacobian_fn(q_arm)
    x_pivot = float(p0[0])

    cfg = vel_base.CartesianVelocityConfig(
        reduced_task_dims=False, split_base_wrist_task=False, ik_seeded_resolution=True,
        ik_iterations=6, task_dim_rz=True, task_dim_rx=False, task_dim_ry=False,
        max_lin_speed_mps=max_lin_speed_mps, max_ang_speed_radps=max(max_lin_speed_mps, 0.5),
        ik_joint_gain=ik_joint_gain,
    )
    controller = vel_base.CartesianVelocityController(cfg)
    controller.reset_from_state(
        {"time": 0.0, "q": q_arm, "qd": np.zeros(6), "ee_pos": p0, "ee_quat": quat0, "target_x": x_pivot}
    )

    n_control_steps = int(duration_s * vel_base.RATE_HZ)
    theta_err_hist = []
    fell_at = None

    pipe: deque = deque()
    prev_theta_meas = None

    for step in range(n_control_steps):
        theta_true = float(data.qpos[pend_qpos_adr])
        theta_dot_true = float(data.qvel[pend_qvel_adr])
        theta_err_true = float(np.mod(theta_true - inverted_angle + np.pi, 2 * np.pi) - np.pi)
        theta_err_hist.append(theta_err_true)
        if abs(theta_err_true) > vel_base.FALL_THRESHOLD_RAD and fell_at is None:
            fell_at = step * vel_base.CONTROL_DT

        if angle_noise_mode == "none":
            theta_meas = theta_true
            theta_dot_meas = theta_dot_true
        else:
            theta_meas = theta_true
            if extra_angle_noise_std_rad > 0.0:
                theta_meas += rng.normal(0.0, extra_angle_noise_std_rad)
            theta_meas = quantize(theta_meas, quant_step_rad)
            if angle_noise_mode == "quantize+fd_vel":
                if prev_theta_meas is None:
                    theta_dot_meas = 0.0
                else:
                    d = theta_meas - prev_theta_meas
                    d = (d + np.pi) % (2 * np.pi) - np.pi
                    theta_dot_meas = d / vel_base.CONTROL_DT
                prev_theta_meas = theta_meas
            else:
                theta_dot_meas = theta_dot_true

        theta_err_meas = float(np.mod(theta_meas - inverted_angle + np.pi, 2 * np.pi) - np.pi)
        pipe.append((theta_err_meas, theta_dot_meas))
        if len(pipe) > delay_cycles:
            used_err, used_dot = pipe.popleft()
        else:
            used_err, used_dot = 0.0, 0.0

        p, quat, jac = fk_jacobian_fn(q_arm)
        target_x = x_pivot + kp * used_err + kd * used_dot
        target_ee_pos = p0.copy()
        target_ee_pos[0] = target_x

        robot_state = {
            "time": step * vel_base.CONTROL_DT, "q": q_arm, "qd": np.zeros(6),
            "ee_pos": p, "ee_quat": quat, "target_x": target_x,
            "target_ee_pos": target_ee_pos, "target_ee_vel": np.zeros(3),
            "fk_jacobian_fn": fk_jacobian_fn,
        }
        xd_cmd = controller.compute(robot_state)
        qd = vel_base._damped_pinv(jac, vel_base.QD_ESTIMATE_DAMPING) @ xd_cmd
        qd = np.clip(qd, -vel_base.MAX_JOINT_VELOCITY_RADPS, vel_base.MAX_JOINT_VELOCITY_RADPS)

        q_arm = q_arm + qd * vel_base.CONTROL_DT
        data.qpos[:6] = q_arm
        data.qvel[:6] = qd
        for _ in range(vel_base.SUBSTEPS_PER_CONTROL):
            mujoco.mj_step(model, data)
            data.qpos[:6] = q_arm
            data.qvel[:6] = qd

        if fell_at is not None:
            break

    theta_err_arr = np.array(theta_err_hist)
    return {
        "perturbation_rad": perturbation_rad,
        "fell_at_s": fell_at,
        "peak_theta_err": float(np.max(np.abs(theta_err_arr))) if len(theta_err_arr) else None,
        "final_theta_err": float(theta_err_arr[-1]) if len(theta_err_arr) else None,
        "survived_full_duration": fell_at is None,
    }


def verify_clean_baseline_velocity(rng) -> None:
    print("=== [VELOCITY lane] Sanity check: clean (no-noise, no-delay) path ===")
    print("Cross-check against pendulum_balance_test.run_balance_trial directly "
          f"(base uses its own fixed {vel_base.TEST_DURATION_S}s duration):")
    mismatches = 0
    for pert in [0.10, 0.20, 0.30]:
        r_new = run_trial_velocity(**VELOCITY_GAINS, perturbation_rad=pert,
                                    duration_s=vel_base.TEST_DURATION_S, rng=rng)
        r_base = vel_base.run_balance_trial(VELOCITY_GAINS["kp"], VELOCITY_GAINS["kd"],
                                             perturbation_rad=pert,
                                             ik_joint_gain=VELOCITY_GAINS["ik_joint_gain"])
        ok = r_new["survived_full_duration"] == r_base["survived_full_duration"]
        mismatches += int(not ok)
        print(f"  pert={pert:.2f}  new={'SURVIVED' if r_new['survived_full_duration'] else 'FELL':>9}"
              f"  base={'SURVIVED' if r_base['survived_full_duration'] else 'FELL':>9}"
              f"  {'OK' if ok else 'MISMATCH -- BUG IN NEW LOOP'}")
    if mismatches:
        raise RuntimeError(f"{mismatches} mismatch(es) -- do not trust any velocity-lane result until fixed.")
    print(f"Using validated gains {VELOCITY_GAINS}, genuinely-clean range 0.1-0.3 rad "
          f"(0.8-1.2 rad is a real but WEAKER 'arrests divergence, does not converge' effect, "
          f"deliberately excluded here); baselines here: {VELOCITY_BASELINE_PERTS}\n")


# --------------------------------------------------------------------------
# Generic sweeps (lane-agnostic; take a bound trial function)
# --------------------------------------------------------------------------

def sweep_angle_noise(run_trial_fn, label: str, baseline_pert: float, duration_s: float, seed: int = 0) -> None:
    print(f"=== [{label}] Angle/velocity sensor noise sweep (baseline perturbation = {baseline_pert} rad) ===")
    print(f"{'mode':>18} {'quant_bits':>10} {'extra_std':>10} {'result':>10} {'peak_err':>10} {'final_err':>10}")
    configs = [
        ("none", None, 0.0),
        ("quantize", 14, 0.0),
        ("quantize", 10, 0.0),
        ("quantize", 8, 0.0),
        ("quantize+fd_vel", 14, 0.0),
        ("quantize+fd_vel", 12, 0.0),
        ("quantize+fd_vel", 10, 0.0),
        ("quantize+fd_vel", 8, 0.0),
        ("quantize+fd_vel", 6, 0.0),
        ("quantize+fd_vel", 14, 0.001),
        ("quantize+fd_vel", 14, 0.005),
        ("quantize+fd_vel", 14, 0.02),
    ]
    for mode, bits, extra_std in configs:
        rng = np.random.default_rng(seed)
        quant_step = 2 * np.pi / (2 ** bits) if bits is not None else ENCODER_QUANT_RAD
        r = run_trial_fn(perturbation_rad=baseline_pert, duration_s=duration_s, rng=rng,
                          angle_noise_mode=mode, quant_step_rad=quant_step,
                          extra_angle_noise_std_rad=extra_std)
        result = "SURVIVED" if r["survived_full_duration"] else "FELL"
        bits_s = str(bits) if bits is not None else "-"
        print(f"{mode:>18} {bits_s:>10} {extra_std:10.4f} {result:>10} "
              f"{r['peak_theta_err']:10.4f} {r['final_theta_err']:10.4f}")
    print()


def sweep_angle_noise_pass_rate(run_trial_fn, label: str, baseline_pert: float, duration_s: float,
                                  n_seeds: int = 10, seed0: int = 100) -> None:
    print(f"=== [{label}] Angle-noise pass-RATE sweep (baseline perturbation = {baseline_pert} rad, "
          f"{n_seeds} seeds/level) ===")
    print(f"{'mode':>18} {'extra_std':>10} {'pass_rate':>10} {'mean_final_err(survived)':>26}")
    for mode in ["quantize", "quantize+fd_vel"]:
        for extra_std in [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0]:
            n_pass = 0
            final_errs = []
            for i in range(n_seeds):
                rng = np.random.default_rng(seed0 + i)
                r = run_trial_fn(perturbation_rad=baseline_pert, duration_s=duration_s, rng=rng,
                                  angle_noise_mode=mode, extra_angle_noise_std_rad=extra_std)
                if r["survived_full_duration"]:
                    n_pass += 1
                    final_errs.append(r["final_theta_err"])
            mean_final = f"{np.mean(final_errs):.4f}" if final_errs else "n/a"
            print(f"{mode:>18} {extra_std:10.4f} {n_pass}/{n_seeds:>7} {mean_final:>26}")
    print()


def sweep_delay(run_trial_fn, label: str, baseline_pert: float, duration_s: float, seed: int = 0) -> None:
    print(f"=== [{label}] Measurement/control delay sweep (baseline perturbation = {baseline_pert} rad, "
          f"no sensor noise) ===")
    print(f"{'delay_cycles':>13} {'result':>10} {'peak_err':>10} {'final_err':>10} {'fell_at_s':>10}")
    for delay_cycles in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 50, 75, 100]:
        rng = np.random.default_rng(seed)
        r = run_trial_fn(perturbation_rad=baseline_pert, duration_s=duration_s, rng=rng,
                          angle_noise_mode="none", delay_cycles=delay_cycles)
        result = "SURVIVED" if r["survived_full_duration"] else "FELL"
        print(f"{delay_cycles:13d} {result:>10} {r['peak_theta_err']:10.4f} {r['final_theta_err']:10.4f} "
              f"{str(r['fell_at_s']):>10}")
    print()


def sweep_combined(run_trial_fn, label: str, baseline_pert: float, duration_s: float, seed: int = 0) -> None:
    print(f"=== [{label}] Combined realistic noise + delay (baseline perturbation = {baseline_pert} rad) ===")
    print(f"{'delay_cycles':>13} {'noise_mode':>18} {'result':>10} {'peak_err':>10} {'final_err':>10}")
    combos = [(0, "quantize+fd_vel"), (1, "quantize+fd_vel"), (2, "quantize+fd_vel"),
              (5, "quantize+fd_vel"), (10, "quantize+fd_vel")]
    for delay_cycles, mode in combos:
        rng = np.random.default_rng(seed)
        r = run_trial_fn(perturbation_rad=baseline_pert, duration_s=duration_s, rng=rng,
                          angle_noise_mode=mode, quant_step_rad=ENCODER_QUANT_RAD, delay_cycles=delay_cycles)
        result = "SURVIVED" if r["survived_full_duration"] else "FELL"
        print(f"{delay_cycles:13d} {mode:>18} {result:>10} {r['peak_theta_err']:10.4f} {r['final_theta_err']:10.4f}")
    print()


def run_lane(run_trial_fn, label: str, baseline_perts: list, duration_s: float) -> None:
    for i, baseline_pert in enumerate(baseline_perts):
        sweep_angle_noise(run_trial_fn, label, baseline_pert, duration_s, seed=i)
        sweep_angle_noise_pass_rate(run_trial_fn, label, baseline_pert, duration_s, seed0=100 + i * 100)
        sweep_delay(run_trial_fn, label, baseline_pert, duration_s, seed=i)
        sweep_combined(run_trial_fn, label, baseline_pert, duration_s, seed=i)


def main() -> int:
    # --- Torque lane ---
    model = torque_base.compose_ur5e_pendulum_model()
    data = mujoco.MjData(model)
    pend_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "/pendulum_hinge")
    pend_qpos_adr = model.jnt_qposadr[pend_joint_id]
    inverted_angle = torque_base.find_inverted_angle(model, data, pend_qpos_adr)
    print(f"inverted_angle (composed frame) = {inverted_angle:.4f} rad")
    K, q_eq, diag = torque_base.linearize_and_design_lqr(model, inverted_angle, **TORQUE_GAIN_KWARGS)
    n_unstable = int(np.sum(np.abs(diag["eigvals_discrete"]) >= 1.0))
    if n_unstable > 0:
        print("WARNING: torque-lane linear design is not stable -- aborting.")
        return 1

    rng = np.random.default_rng(0)
    verify_clean_baseline_torque(model, K, q_eq, rng)
    torque_trial_fn = partial(run_trial_torque, model, K, q_eq)
    run_lane(torque_trial_fn, "TORQUE", TORQUE_BASELINE_PERTS, duration_s=8.0)

    # --- Velocity lane ---
    rng2 = np.random.default_rng(1)
    verify_clean_baseline_velocity(rng2)
    velocity_trial_fn = partial(run_trial_velocity, VELOCITY_GAINS["kp"], VELOCITY_GAINS["kd"],
                                 VELOCITY_GAINS["ik_joint_gain"])
    run_lane(velocity_trial_fn, "VELOCITY", VELOCITY_BASELINE_PERTS, duration_s=8.0)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
