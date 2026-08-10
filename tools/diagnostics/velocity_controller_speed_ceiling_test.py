#!/usr/bin/env python3
"""Empirical test of controller_core.cartesian_velocity_controller's
ik_seeded_resolution mode's REAL achievable end-effector speed, and how
much orientation error accumulates as commanded aggressiveness increases --
run at the user's explicit request ("test it," don't just report the
configured max_lin_speed_mps cap).

Kinematic-only simulation, same fidelity level as velocity_gain_tuning/
envs/velocity_transport_env.py (LocalMujocoDynamics for FK/Jacobian at
arbitrary q, no mj_step/no rigid-body dynamics) -- appropriate here because
real speedL is resolved to joint velocities by the robot firmware's own
Jacobian-based IK, not by rigid-body dynamics.

IMPORTANT MECHANISM CORRECTION vs this script's first version: ik_seeded_
resolution does NOT use target_ee_vel/v_ff at all (compute_ik_seeded in
modes.py never reads xd_full -- only p_des, quat0, q_rest, q_current). It
solves a fresh position-IK target q_target from p_des every cycle (seeded
from q_rest, NEVER q_current -- the path-independence property this mode
exists for) and drives qd_joint = ik_joint_gain * (q_target - q_current),
xd_cmd = jac_current @ qd_joint. So "how fast can it move" is governed by
(a) how fast the ABSOLUTE POSITION TARGET itself moves (dx / move_duration_s
via the standard min_jerk_move_hold profile, matching every real transport
driver in this repo) and (b) ik_joint_gain (config default 4.0 /s) turning
the resulting (q_target - q_current) gap into a joint-velocity command,
THEN (c) the outer max_lin_speed_mps/max_ang_speed_radps clamp (controller.
py lines 147-151, confirmed applied unconditionally after mode dispatch,
i.e. NOT bypassed for this mode the way kp_x/kp_rot are).

Two passes:
  1. UNCLAMPED (max_lin_speed_mps=1000.0): isolates the controller's own
     kinematic-following ceiling from the software speed cap -- does
     something else (joint_velocity_guard, orientation guard, orthogonal
     drift) bind first at a lower speed even with the clamp out of the way?
  2. CLAMPED at the real configured default (0.25 m/s): shows what a real
     deployment (with the cap in force) actually delivers for the same
     commanded aggressiveness.

Sweeps dx (target_x_delta_m) and move_duration_s together so peak commanded
position-target speed (v_peak = 1.875*dx/move_duration_s for a min-jerk
profile) ranges from calm transport-scale values up to and past the
swing-up requirement (~5.4 m/s equivalent, from the earlier pendulum
analysis).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from controller_core.cartesian_velocity_controller import (  # noqa: E402
    CartesianVelocityConfig,
    CartesianVelocityController,
)
from controller_core.cartesian_velocity_controller.math_utils import _damped_pinv  # noqa: E402
from controller_core.kinematics_utils import swing_twist_axis_error  # noqa: E402
from hardware.local_dynamics import LocalMujocoDynamics  # noqa: E402
from hardware.poses import HANGING_ALPHA_0_5_Q  # noqa: E402
from simulation.ur5e_mujoco_torque import x_profile_target  # noqa: E402

RATE_HZ = 125.0
DT = 1.0 / RATE_HZ
MAX_JOINT_VELOCITY_RADPS = 3.0
MAX_ORTHOGONAL_DRIFT_M = 0.05
MAX_ORIENTATION_ERROR_RAD = 0.25
QD_ESTIMATE_DAMPING = 1.0e-3
SWINGUP_REQUIREMENT_MPS = 5.42

# (target_x_delta_m, move_duration_s) -- min-jerk peak commanded speed is
# 1.875*dx/move_duration_s. Spans calm transport-scale moves up through
# and past the swing-up requirement.
SWEEP_CASES = [
    (0.06, 1.0),
    (0.06, 0.3),
    (0.06, 0.1),
    (0.06, 0.03),
    (0.06, 0.0104),   # v_peak ~= 10.8 m/s -- past swing-up requirement at this dx
    (0.20, 0.5),
    (0.20, 0.1),
    (0.20, 0.0347),   # v_peak ~= 10.8 m/s
    (0.20, 0.02),     # v_peak ~= 18.75 m/s
]


def run_one(
    dyn: LocalMujocoDynamics,
    target_x_delta_m: float,
    move_duration_s: float,
    max_lin_speed_mps: float,
    profile: str = "min_jerk_move_hold",
    duration_s_override: float | None = None,
    orientation_priority: bool = False,
) -> dict:
    q0 = HANGING_ALPHA_0_5_Q.copy()
    p0, quat0, _ = dyn.fk_and_jacobian(q0)
    x0 = float(p0[0])

    cfg = CartesianVelocityConfig(
        reduced_task_dims=False,
        split_base_wrist_task=False,
        ik_seeded_resolution=True,
        ik_iterations=6,
        orientation_priority=orientation_priority,
        task_dim_rz=True,
        task_dim_rx=False,
        task_dim_ry=False,
        max_lin_speed_mps=max_lin_speed_mps,
        max_ang_speed_radps=max(max_lin_speed_mps, 0.5),
    )
    controller = CartesianVelocityController(cfg)
    controller.reset_from_state(
        {"time": 0.0, "q": q0, "qd": np.zeros(6), "ee_pos": p0, "ee_quat": quat0, "target_x": x0}
    )

    duration_s = duration_s_override if duration_s_override is not None else move_duration_s + 0.2
    n_steps = int(duration_s * RATE_HZ)
    q = q0.copy()
    p_prev = p0.copy()

    peak_achieved_v = 0.0
    peak_max_abs_qd = 0.0
    peak_orientation_error = 0.0
    peak_commanded_v = 0.0
    guard_reason = None
    trip_t = None

    for step in range(n_steps):
        t_s = step * DT
        p, quat, jac = dyn.fk_and_jacobian(q)

        achieved_v = float(np.linalg.norm((p - p_prev) / DT))
        peak_achieved_v = max(peak_achieved_v, achieved_v)
        p_prev = p

        target_x, target_x_vel = x_profile_target(
            profile, x0, target_x_delta_m, t_s, duration_s, move_duration_s=move_duration_s
        )
        peak_commanded_v = max(peak_commanded_v, abs(target_x_vel))
        target_ee_pos = p0.copy()
        target_ee_pos[0] = target_x

        robot_state = {
            "time": t_s,
            "q": q,
            "qd": np.zeros(6),
            "ee_pos": p,
            "ee_quat": quat,
            "target_x": float(target_x),
            "target_ee_pos": target_ee_pos,
            "target_ee_vel": np.array([target_x_vel, 0.0, 0.0]),
            "fk_jacobian_fn": lambda qq: dyn.fk_and_jacobian(qq),
        }
        xd_cmd = controller.compute(robot_state)
        qd = _damped_pinv(jac, QD_ESTIMATE_DAMPING) @ xd_cmd
        max_abs_qd = float(np.max(np.abs(qd)))
        peak_max_abs_qd = max(peak_max_abs_qd, max_abs_qd)

        orientation_error = float(
            np.linalg.norm([swing_twist_axis_error(quat0, quat, i) for i in range(3)])
        )
        peak_orientation_error = max(peak_orientation_error, orientation_error)

        y_drift = abs(float(p[1] - p0[1]))
        z_drift = abs(float(p[2] - p0[2]))
        orthogonal_drift = max(y_drift, z_drift)

        if guard_reason is None:
            if max_abs_qd > MAX_JOINT_VELOCITY_RADPS:
                guard_reason = f"joint_velocity_guard ({max_abs_qd:.3f} > {MAX_JOINT_VELOCITY_RADPS})"
                trip_t = t_s
            elif orthogonal_drift > MAX_ORTHOGONAL_DRIFT_M:
                guard_reason = f"orthogonal_drift_guard ({orthogonal_drift:.4f} > {MAX_ORTHOGONAL_DRIFT_M})"
                trip_t = t_s
            elif orientation_error > MAX_ORIENTATION_ERROR_RAD:
                guard_reason = f"orientation_guard ({orientation_error:.4f} > {MAX_ORIENTATION_ERROR_RAD})"
                trip_t = t_s

        q = q + qd * DT
        if guard_reason is not None:
            break

    return {
        "target_x_delta_m": target_x_delta_m,
        "move_duration_s": move_duration_s,
        "peak_commanded_v_mps": peak_commanded_v,
        "peak_achieved_v_mps": peak_achieved_v,
        "peak_max_abs_qd_radps": peak_max_abs_qd,
        "peak_orientation_error_rad": peak_orientation_error,
        "guard_reason": guard_reason,
        "trip_t": trip_t,
    }


def print_table(rows: list[dict]) -> None:
    print(f"{'dx (m)':>8} {'T (s)':>8} {'cmd v_peak':>11} {'achieved v_peak':>16} "
          f"{'peak |qd|':>10} {'peak orient err':>16} {'guard':>32}")
    for r in rows:
        guard_str = r["guard_reason"] or "none"
        print(f"{r['target_x_delta_m']:8.3f} {r['move_duration_s']:8.4f} "
              f"{r['peak_commanded_v_mps']:11.3f} {r['peak_achieved_v_mps']:16.4f} "
              f"{r['peak_max_abs_qd_radps']:10.3f} {r['peak_orientation_error_rad']:16.4f} {guard_str:>32}")


def main() -> int:
    dyn = LocalMujocoDynamics()

    print("=== PASS 1: max_lin_speed_mps clamp REMOVED (1000.0) -- true controller ceiling ===")
    unclamped = [run_one(dyn, dx, T, max_lin_speed_mps=1000.0) for dx, T in SWEEP_CASES]
    print_table(unclamped)

    print("\n=== PASS 2: max_lin_speed_mps at the REAL configured default (0.25 m/s) ===")
    clamped = [run_one(dyn, dx, T, max_lin_speed_mps=0.25) for dx, T in SWEEP_CASES]
    print_table(clamped)

    print("\n=== PASS 3: STEP profile (instantaneous target jump), dx swept toward reach limit, "
          "clamp REMOVED -- isolates the P-follower's own peak per-step velocity ceiling ===")
    step_dxs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    step_rows = [
        run_one(dyn, dx, move_duration_s=1.0, max_lin_speed_mps=1000.0, profile="step", duration_s_override=1.0)
        for dx in step_dxs
    ]
    print_table(step_rows)

    print(f"\nSwing-up requirement: {SWINGUP_REQUIREMENT_MPS} m/s equivalent peak EE speed.")
    peak_unclamped = max(r["peak_achieved_v_mps"] for r in unclamped)
    peak_clamped = max(r["peak_achieved_v_mps"] for r in clamped)
    print(f"Max achieved EE speed observed, unclamped pass: {peak_unclamped:.4f} m/s")
    print(f"Max achieved EE speed observed, clamped (real default) pass: {peak_clamped:.4f} m/s")

    first_guard_unclamped = next((r for r in unclamped if r["guard_reason"]), None)
    if first_guard_unclamped:
        print(f"\nUnclamped pass: first guard trip at dx={first_guard_unclamped['target_x_delta_m']}, "
              f"T={first_guard_unclamped['move_duration_s']}: {first_guard_unclamped['guard_reason']}, "
              f"achieved v_peak at trip case: {first_guard_unclamped['peak_achieved_v_mps']:.4f} m/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
