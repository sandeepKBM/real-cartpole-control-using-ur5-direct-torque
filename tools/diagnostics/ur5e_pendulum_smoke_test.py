#!/usr/bin/env python3
"""Smoke test for the UR5e + pendulum composed model (simulation/
ur5e_pendulum_compose.py): compose, compile, release the pendulum from
horizontal, and confirm it swings down and settles near hanging-straight
under gravity -- the simplest possible proof this model is physically
sane, matching AGENTS.md sec 7's "simplest proof first" rule.

See assets/ur5e_pendulum/pendulum_attachment.xml's docstring for which
dimensions in this model are real (extracted from the CAD archive) vs
placeholder (physically reasonable guesses, not measured) -- this smoke
test only proves the model is DYNAMICALLY SANE, not that it matches the
real hardware's actual mass/length.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from simulation.ur5e_pendulum_compose import compose_ur5e_pendulum_model  # noqa: E402


def main() -> int:
    model = compose_ur5e_pendulum_model()
    print(f"Compiled OK: nq={model.nq} nv={model.nv} nbody={model.nbody}")

    pendulum_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "/pendulum_hinge")
    if pendulum_joint_id < 0:
        print("FAIL: could not find /pendulum_hinge joint in the composed model")
        return 1
    qpos_adr = model.jnt_qposadr[pendulum_joint_id]

    data = mujoco.MjData(model)
    # Neutral arm pose, held fixed via qpos reset each step -- this test only
    # checks the pendulum's own dynamics, not full arm+pendulum coupling
    # (a known follow-up, see the accompanying status doc).
    arm_q0 = np.array([0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0])
    data.qpos[:6] = arm_q0
    data.qpos[qpos_adr] = np.pi / 2  # release from horizontal
    mujoco.mj_forward(model, data)

    angle_hist = []
    for _ in range(3000):
        data.qpos[:6] = arm_q0
        data.qvel[:6] = 0
        mujoco.mj_step(model, data)
        angle_hist.append(float(data.qpos[qpos_adr]))
    angle_hist = np.array(angle_hist)

    settled = angle_hist[-200:]
    print(f"start angle: {angle_hist[0]:.4f} rad, final angle: {angle_hist[-1]:.4f} rad")
    print(f"settled-window std: {settled.std():.5f} rad")
    print(
        "note: the settled angle is NOT expected to be 0 rad in this arm-attached "
        "configuration -- attachment_site's fixed rotation relative to wrist_3_link "
        "means the pendulum joint's local zero is not world-down at this arm pose; "
        "only convergence (low settled std) is checked here. See "
        "tests/mujoco/test_ur5e_pendulum_compose.py's isolated-pendulum test for a "
        "check against the true 0-rad hanging equilibrium (no arm rotation involved)."
    )

    ok = True
    if settled.std() > 0.01:
        print(f"FAIL: pendulum did not settle (still oscillating), settled std={settled.std():.5f}")
        ok = False
    if abs(angle_hist[0] - angle_hist[-1]) < 0.05:
        print("FAIL: pendulum barely moved from its release angle -- joint may be effectively locked")
        ok = False
    if ok:
        print("PASS: pendulum released from horizontal swings under gravity and settles (converges).")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
