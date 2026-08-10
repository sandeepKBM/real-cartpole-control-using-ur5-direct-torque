#!/usr/bin/env python3
"""Empirical test: starting the physical pendulum apparatus (assets/
ur5e_pendulum/, composed onto the real UR5e model via simulation/
ur5e_pendulum_compose.py) AT the inverted (flipped-up) equilibrium, can the
existing ik_seeded_resolution velocity-controller stack -- driven by an
outer PD balance law on pole angle -- actually hold it there?

Unlike the swing-up analysis (a big one-shot impulse) or the speed-ceiling
test (open-loop position tracking), this is a closed-loop STABILIZATION
problem: small, fast corrective cart moves against a genuinely unstable
equilibrium. The relevant question isn't "how fast can the cart move" but
"can the cart react FASTER than the pole falls."

Real dynamics matter here (rigid-body coupling through the arm's base
motion into the pendulum), so this uses the actual composed MuJoCo model
with mj_step, not the kinematic-only LocalMujocoDynamics harness the
speed-ceiling test used. The arm itself is still driven KINEMATICALLY
(qvel[:6] set from the real CartesianVelocityController's output each
cycle, then mj_step lets that velocity couple into the pendulum through
real dynamics) -- this matches the fidelity of every other velocity-control
lane test in this repo (no arm-side torque/inertia realism assumed), while
giving the pendulum itself full, real gravity + coupling dynamics.

Analytical prior (computed before running this): the pendulum's own
inverted-equilibrium instability rate is lambda = sqrt(m*g*l_com/I_pivot) ~=
6.96 rad/s (tau ~= 144 ms), using this session's already-measured I_pivot ~=
0.0036 kg*m^2. The controller's own position-tracking inner loop has
bandwidth ik_joint_gain = 4.0 rad/s (tau = 250 ms) -- SLOWER than the pole's
own fall rate. This is a structural red flag for stabilizability (the
actuator cannot generically "outrun" a plant instability faster than its
own bandwidth), tested here empirically across an outer-loop PD gain grid
rather than asserted from the ratio alone.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from controller_core.cartesian_velocity_controller import (  # noqa: E402
    CartesianVelocityConfig,
    CartesianVelocityController,
)
from controller_core.cartesian_velocity_controller.math_utils import _damped_pinv  # noqa: E402
from simulation.ur5e_pendulum_compose import compose_ur5e_pendulum_model  # noqa: E402

RATE_HZ = 125.0
CONTROL_DT = 1.0 / RATE_HZ
PHYSICS_DT = 0.002  # model.opt.timestep
SUBSTEPS_PER_CONTROL = max(1, round(CONTROL_DT / PHYSICS_DT))
QD_ESTIMATE_DAMPING = 1.0e-3
MAX_JOINT_VELOCITY_RADPS = 3.0
FALL_THRESHOLD_RAD = 1.5  # pole angle error past this = "fell", test ends.
# MUST stay comfortably above any tested perturbation_rad -- 0.5 (the first
# version of this constant) silently coincided with a later sweep's own
# perturbation values (0.5-1.0 rad), so every pert>=0.5 trivially tripped
# "fell" on the very first sample (theta_err starts AT the perturbation,
# before the controller can act at all) regardless of any real dynamics.
# That produced a fake "hard wall at 0.5-0.6 rad" that was actually just
# this threshold, not a controller result -- caught by noticing peak_err/
# final_err were byte-identical to the raw perturbation with fell_at=0.0
# for every such case, at both 125Hz and 500Hz alike (a real dynamic wall
# would not look identical across two different control rates).
TEST_DURATION_S = 3.0

ARM_Q0 = np.array([0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0])


def find_inverted_angle(model, data, pend_qpos_adr: int) -> float:
    """CORRECTED 2026-08-09 (later the same day this file was written): the
    original "release from 0.3 rad and settle for a fixed 4000 steps" never
    actually converged (same slow-drift trap fixed once already in the
    torque-lane script, but never propagated back here) -- verified
    directly: the resulting "hanging"/"inverted" pair (0.32 / -2.82) were
    BOTH unstable under a zero-friction, tiny-perturbation growth test, not
    one stable and one unstable. Replaced with the same direct,
    unambiguous method now used in pendulum_balance_torque_lqr.py: scan the
    pendulum joint's own qfrc_bias for its true zero-crossings (the
    unconditional physical equilibrium condition, no settling time
    needed), then classify each by releasing a TINY (1e-3 rad) perturbation
    with damping/frictionloss temporarily zeroed and checking whether it
    grows (unstable) or stays bounded (stable) -- ground truth, not an
    assumption about which internal MuJoCo quantity's sign means what.
    Confirmed to land at theta=0.0 exactly for this model/pose, matching
    the torque-lane script's independently-fixed result."""
    pend_dof_adr = model.jnt_dofadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "/pendulum_hinge")
    ]

    def qfrc_bias_pend(theta: float) -> float:
        d = mujoco.MjData(model)
        d.qpos[:6] = ARM_Q0
        d.qpos[pend_qpos_adr] = theta
        d.qvel[:] = 0.0
        mujoco.mj_forward(model, d)
        return float(d.qfrc_bias[pend_qpos_adr])

    thetas = np.linspace(-np.pi, np.pi, 361)
    vals = np.array([qfrc_bias_pend(t) for t in thetas])
    crossings = []
    for i in range(len(thetas) - 1):
        if vals[i] == 0.0 or (vals[i] > 0) != (vals[i + 1] > 0):
            lo, hi = thetas[i], thetas[i + 1]
            for _ in range(60):
                mid = (lo + hi) / 2
                if (qfrc_bias_pend(mid) > 0) == (vals[i] > 0):
                    lo = mid
                else:
                    hi = mid
            crossings.append((lo + hi) / 2)
    if len(crossings) != 2:
        raise RuntimeError(f"expected exactly 2 equilibria, found {len(crossings)}: {crossings}")

    saved_damping = float(model.dof_damping[pend_dof_adr])
    saved_frictionloss = float(model.dof_frictionloss[pend_dof_adr])
    model.dof_damping[pend_dof_adr] = 0.0
    model.dof_frictionloss[pend_dof_adr] = 0.0
    classified = []
    for c in crossings:
        d = mujoco.MjData(model)
        d.qpos[:6] = ARM_Q0
        d.qpos[pend_qpos_adr] = c + 1e-3
        d.qvel[:] = 0.0
        mujoco.mj_forward(model, d)
        max_dev = 0.0
        for _ in range(2000):
            theta = float(d.qpos[pend_qpos_adr])
            dev = abs(float(np.mod(theta - c + np.pi, 2 * np.pi) - np.pi))
            max_dev = max(max_dev, dev)
            d.qpos[:6] = ARM_Q0
            d.qvel[:6] = 0.0
            mujoco.mj_step(model, d)
        classified.append((c, "unstable" if max_dev > 0.02 else "stable"))
    model.dof_damping[pend_dof_adr] = saved_damping
    model.dof_frictionloss[pend_dof_adr] = saved_frictionloss

    unstable = [c for c, kind in classified if kind == "unstable"]
    stable = [c for c, kind in classified if kind == "stable"]
    if len(unstable) != 1 or len(stable) != 1:
        raise RuntimeError(f"expected exactly 1 stable + 1 unstable equilibrium: {classified}")
    return float(stable[0]), float(unstable[0])


def run_balance_trial(
    kp: float,
    kd: float,
    perturbation_rad: float = 0.0,
    ik_joint_gain: float = 4.0,
    orientation_priority: bool = False,
    max_lin_speed_mps: float = 1000.0,
) -> dict:
    model = compose_ur5e_pendulum_model()
    model.opt.timestep = PHYSICS_DT
    data = mujoco.MjData(model)
    pend_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "/pendulum_hinge")
    pend_qpos_adr = model.jnt_qposadr[pend_joint_id]
    pend_qvel_adr = model.jnt_dofadr[pend_joint_id]
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")

    _, inverted_angle = find_inverted_angle(model, data, pend_qpos_adr)

    data.qpos[:6] = ARM_Q0
    data.qpos[pend_qpos_adr] = inverted_angle + perturbation_rad
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)

    q_arm = ARM_Q0.copy()
    # Reused scratch MjData for FK/Jacobian queries at arbitrary arm q --
    # the first version of this function allocated a fresh MjData() on
    # EVERY call (up to ~1500/trial x 105 trials), which is what made the
    # first run of this script never finish inside a 10-minute timeout.
    # This scratch buffer only ever needs qpos[:6] set + mj_forward, same
    # as hardware/local_dynamics.py's LocalMujocoDynamics pattern.
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

    cfg = CartesianVelocityConfig(
        reduced_task_dims=False, split_base_wrist_task=False, ik_seeded_resolution=True,
        ik_iterations=6, task_dim_rz=True, task_dim_rx=False, task_dim_ry=False,
        max_lin_speed_mps=max_lin_speed_mps, max_ang_speed_radps=max(max_lin_speed_mps, 0.5),
        ik_joint_gain=ik_joint_gain,
        orientation_priority=orientation_priority,
    )
    controller = CartesianVelocityController(cfg)
    controller.reset_from_state(
        {"time": 0.0, "q": q_arm, "qd": np.zeros(6), "ee_pos": p0, "ee_quat": quat0, "target_x": x_pivot}
    )

    n_control_steps = int(TEST_DURATION_S * RATE_HZ)
    theta_err_hist = []
    fell_at = None

    for step in range(n_control_steps):
        theta = float(data.qpos[pend_qpos_adr])
        theta_dot = float(data.qvel[pend_qvel_adr])
        theta_err = float(np.mod(theta - inverted_angle + np.pi, 2 * np.pi) - np.pi)
        theta_err_hist.append(theta_err)
        if abs(theta_err) > FALL_THRESHOLD_RAD and fell_at is None:
            fell_at = step * CONTROL_DT

        p, quat, jac = fk_jacobian_fn(q_arm)
        target_x = x_pivot + kp * theta_err + kd * theta_dot
        target_ee_pos = p0.copy()
        target_ee_pos[0] = target_x

        robot_state = {
            "time": step * CONTROL_DT, "q": q_arm, "qd": np.zeros(6),
            "ee_pos": p, "ee_quat": quat, "target_x": target_x,
            "target_ee_pos": target_ee_pos, "target_ee_vel": np.zeros(3),
            "fk_jacobian_fn": fk_jacobian_fn,
        }
        xd_cmd = controller.compute(robot_state)
        qd = _damped_pinv(jac, QD_ESTIMATE_DAMPING) @ xd_cmd
        qd = np.clip(qd, -MAX_JOINT_VELOCITY_RADPS, MAX_JOINT_VELOCITY_RADPS)

        q_arm = q_arm + qd * CONTROL_DT
        data.qpos[:6] = q_arm
        data.qvel[:6] = qd
        for _ in range(SUBSTEPS_PER_CONTROL):
            mujoco.mj_step(model, data)
            data.qpos[:6] = q_arm
            data.qvel[:6] = qd

        if fell_at is not None:
            break

    theta_err_arr = np.array(theta_err_hist)
    return {
        "kp": kp, "kd": kd, "fell_at_s": fell_at,
        "final_theta_err": float(theta_err_arr[-1]) if len(theta_err_arr) else None,
        "peak_theta_err": float(np.max(np.abs(theta_err_arr))) if len(theta_err_arr) else None,
        "survived_full_duration": fell_at is None,
    }


def main() -> int:
    # 0.02 rad (first version of this test) is BELOW the joint's own Coulomb
    # breakaway angle (frictionloss=0.01 Nm vs gravity torque -- crosses at
    # ~0.0574 rad, see docs/status/pendulum_balance_test_2026-08-09.md) --
    # the pendulum was stiction-locked, not balanced, and every gain looked
    # identical because none of them were doing anything. Perturbations here
    # are chosen to clear that threshold for real.
    perturbations = [0.0, 0.10, 0.20]
    kp_grid = [0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8]
    kd_grid = [0.0, 0.01, 0.02, 0.05, 0.1]

    for pert in perturbations:
        print(f"\n=== perturbation = {pert} rad ===")
        print(f"{'kp':>8} {'kd':>8} {'fell_at_s':>10} {'peak_err_rad':>13} {'final_err_rad':>14} {'result':>10}")
        best = None
        for kp in kp_grid:
            for kd in kd_grid:
                r = run_balance_trial(kp, kd, perturbation_rad=pert)
                result = "SURVIVED" if r["survived_full_duration"] else "FELL"
                print(f"{kp:8.3f} {kd:8.3f} {str(r['fell_at_s']):>10} "
                      f"{r['peak_theta_err']:13.4f} {r['final_theta_err']:14.4f} {result:>10}")
                if best is None or (r["peak_theta_err"] is not None and r["peak_theta_err"] < best["peak_theta_err"]):
                    best = r
        print(f"Best at perturbation={pert}: kp={best['kp']}, kd={best['kd']}, "
              f"peak_err={best['peak_theta_err']:.4f} rad, survived={best['survived_full_duration']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
