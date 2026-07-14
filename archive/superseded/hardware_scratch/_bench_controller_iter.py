#!/usr/bin/env python3
"""Benchmark one iteration of the tuned OSC controller."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from controller_core.safety import ImpedanceSafetyConfig, ImpedanceSafetyMonitor  # noqa: E402
from controller_core.x_axis_cartesian_impedance import (  # noqa: E402
    CartesianImpedanceConfig,
    XAxisCartesianImpedanceController,
)

CFG_PATH = REPO / "config" / "ur5e_mujoco_torque_osc_tuned.yaml"
N = 5000


def _mean_us(t0: int, t1: int, n: int) -> float:
    return (t1 - t0) / n / 1e3


def _bench(label: str, fn, n: int = N) -> float:
    for _ in range(200):
        fn()
    t0 = time.perf_counter_ns()
    for _ in range(n):
        fn()
    t1 = time.perf_counter_ns()
    us = _mean_us(t0, t1, n)
    print(f"{label}: {us:.1f} us ({us/1000:.3f} ms)")
    return us


def main() -> int:
    cfg = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))
    ctrl_cfg = CartesianImpedanceConfig.from_controller_yaml_section(cfg["controller"])
    controller = XAxisCartesianImpedanceController(ctrl_cfg)
    safety = ImpedanceSafetyMonitor(ImpedanceSafetyConfig())

    q = np.array([0.0, -1.5707963267948966, 0.0, -1.5707963267948966, 0.0, 0.0], dtype=np.float64)
    qd = np.zeros(6)

    print("config:")
    print(f"  task_space_inertia_shaping={ctrl_cfg.task_space_inertia_shaping}")
    print(f"  nullspace_posture={ctrl_cfg.nullspace_posture}")
    print()

    # --- MuJoCo full state build (sim path) ---
    try:
        import mujoco  # noqa: F401
        from simulation.ur5e_mujoco_torque import build_mujoco_state, load_model

        scene = REPO / cfg["mujoco"]["scene_xml"]
        model, data, site_id, joint_ids, _ = load_model(scene)
        data.qpos[:6] = q
        mujoco.mj_forward(model, data)
        mstate = build_mujoco_state(
            model,
            data,
            site_id=site_id,
            joint_ids=joint_ids,
            time_s=0.0,
            dt_s=0.002,
            target_x=-0.02,
            target_x_vel=0.05,
        )
        robot_state = mstate.as_robot_state()
        controller.reset_from_state(robot_state)
        safety.reset()
        safety.set_initial_position(mstate.ee_pos, move_axis=0)

        _bench("build_mujoco_state (FK + J + M + gravity)", lambda: build_mujoco_state(
            model, data, site_id=site_id, joint_ids=joint_ids, time_s=0.0, dt_s=0.002,
            target_x=-0.02, target_x_vel=0.05,
        ))
        _bench("controller.compute (real J, M from MuJoCo state)", lambda: controller.compute(robot_state))
        _bench("safety.check", lambda: safety.check(robot_state, x_error=0.01, orientation_error_norm=0.05, axis_target_moving=True))
        _bench("controller + safety (sim hot path)", lambda: (
            safety.check(
                robot_state,
                x_error=controller.compute(robot_state).x_error,
                orientation_error_norm=controller.compute(robot_state).orientation_error_norm,
                axis_target_moving=True,
            )
        ))
    except Exception as exc:
        print(f"MuJoCo path skipped: {exc}")
        robot_state = {
            "time": 0.0, "q": q, "qd": qd,
            "ee_pos": np.array([-0.12, -0.234, 0.928]),
            "ee_quat": np.array([0.0079, -0.0032, 0.7071, -0.7071]),
            "ee_lin_vel": np.zeros(3), "ee_ang_vel": np.zeros(3),
            "jacobian": np.eye(6), "mass_matrix": np.eye(6) * 2.0,
            "target_x": -0.02, "target_x_vel": 0.05, "transport_axis_index": 0,
        }
        controller.reset_from_state(robot_state)
        _bench("controller.compute (identity J, M fallback)", lambda: controller.compute(robot_state))

    print()
    print("Budget:")
    print("  500 Hz direct_torque -> 2000 us/cycle (2.0 ms)")
    print("  125 Hz servoL        -> 8000 us/cycle (8.0 ms)")
    print()
    print("Hardware loop (direct_torque_transport) per iteration also pays:")
    print("  RTDE read_state + getJacobian + getMassMatrix + directTorque")
    print("  (typically 0.5-2.0 ms on a lab PC; dominates Python math)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
