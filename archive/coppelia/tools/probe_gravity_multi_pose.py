#!/usr/bin/env python3
"""Probe gravity balance across multiple joint configs along Y-transport sweep.

For each pose: apply MuJoCo-derived gravity at different scales, measure EE drift.
Also measures the Coppelia-actual gravity by binary search for the balance scale.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

COPPELIA_ROOT = Path("/home/kbm/coppelia_runtime/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu22_04")
sys.path.insert(0, str(COPPELIA_ROOT / "programming" / "zmqRemoteApi" / "clients" / "python" / "src"))
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

from simulation.run_coppeliasim_x_axis_headless import (
    build_mujoco_gravity_estimator,
    compute_mujoco_gravity_bias,
)

Q_SEED = np.array([0.0, 0.0443244063, -1.67570517, -1.1888, 0.0, -0.3214])
JOINT_PATHS = [
    "/UR5/joint", "/UR5/link/joint", "/UR5/link/link/joint",
    "/UR5/link/link/link/joint", "/UR5/link/link/link/link/joint",
    "/UR5/link/link/link/link/link/joint",
]

def main():
    import time
    port = int(os.environ.get("PORT", "23000"))
    for attempt in range(10):
        try:
            client = RemoteAPIClient("127.0.0.1", port)
            sim = client.require("sim")
            break
        except Exception as exc:
            print(f"  connect attempt {attempt+1}/10 failed: {exc}")
            time.sleep(3)
    else:
        raise RuntimeError("Failed to connect to CoppeliaSim after 10 attempts")

    for attr_name in ("floatparam_simulation_time_step", "floatparam_physicstimestep"):
        if hasattr(sim, attr_name):
            try:
                sim.setFloatParam(int(getattr(sim, attr_name)), 0.005)
            except Exception:
                pass
    sim.setStepping(True)
    if sim.getSimulationState() == sim.simulation_stopped:
        sim.startSimulation()

    model_path = str(COPPELIA_ROOT / "models" / "robots" / "non-mobile" / "UR5.ttm")
    sim.loadModel(model_path)
    handles = [sim.getObject(p) for p in JOINT_PATHS]
    ee_handle = sim.getObject("/UR5/connection")

    for h, q in zip(handles, Q_SEED):
        sim.setJointPosition(h, float(q))
    sim.step()
    for h in handles:
        sim.setJointMode(h, sim.jointmode_dynamic, 0)
        sim.setObjectInt32Param(h, sim.jointintparam_motor_enabled, 1)
        sim.setObjectInt32Param(h, sim.jointintparam_ctrl_enabled, 0)
    sim.step()

    est = build_mujoco_gravity_estimator()
    assert est is not None, "MuJoCo gravity estimator failed to load"

    def reset_to(q_target):
        for h, q in zip(handles, q_target):
            sim.setJointPosition(h, float(q))
        for h, t in zip(handles, [0.0]*6):
            sim.setJointTargetForce(h, 0.0, True)
        sim.step()

    def apply_torque(tau_vec):
        for h, t in zip(handles, tau_vec):
            sim.setJointTargetForce(h, float(t), True)

    def ee_pos():
        return np.array(sim.getObjectPosition(ee_handle, -1))

    def read_q():
        return np.array([sim.getJointPosition(h) for h in handles])

    # Generate poses along the Y-axis sweep by varying shoulder_lift
    # The Lua sweep varies shoulder_lift from about 0 to pi/2
    poses = []
    for sl_offset in np.linspace(0.0, 1.2, 7):
        q_test = Q_SEED.copy()
        q_test[1] = Q_SEED[1] + sl_offset
        poses.append(("sl+%.2f" % sl_offset, q_test))

    # Also vary elbow
    for el_offset in np.linspace(0.0, -0.8, 5):
        q_test = Q_SEED.copy()
        q_test[1] = Q_SEED[1] + 0.4
        q_test[2] = Q_SEED[2] + el_offset
        poses.append(("sl+0.40_el%.2f" % el_offset, q_test))

    dt = float(sim.getSimulationTimeStep())
    print(f"Sim dt: {dt:.4f}s")
    print(f"{'Pose':25s}  {'EE_y':>8s}  {'EE_z':>8s}  "
          f"{'MJ_bias1':>10s}  {'s=1.0_dz':>10s}  {'s=1.5_dz':>10s}  "
          f"{'s=2.0_dz':>10s}  {'best_s':>8s}  {'best_dz':>10s}")
    print("-" * 120)

    results = []
    for label, q_test in poses:
        reset_to(q_test)
        p0 = ee_pos()
        bias = compute_mujoco_gravity_bias(est, q_test, np.zeros(6))

        best_scale = 1.5
        best_dz = 999.0
        scale_dz = {}
        for scale in [1.0, 1.3, 1.5, 1.7, 2.0, 2.5]:
            reset_to(q_test)
            ps = ee_pos()
            gcomp = -bias * scale
            apply_torque(gcomp.tolist())
            for _ in range(200):
                sim.step()
            pe = ee_pos()
            dz = pe[2] - ps[2]
            scale_dz[scale] = dz
            if abs(dz) < abs(best_dz):
                best_dz = dz
                best_scale = scale

        dz10 = scale_dz.get(1.0, float("nan"))
        dz15 = scale_dz.get(1.5, float("nan"))
        dz20 = scale_dz.get(2.0, float("nan"))

        print(f"{label:25s}  {p0[1]:+8.4f}  {p0[2]:+8.4f}  "
              f"{bias[1]:+10.3f}  {dz10:+10.4f}  {dz15:+10.4f}  "
              f"{dz20:+10.4f}  {best_scale:8.2f}  {best_dz:+10.4f}")
        results.append({
            "label": label, "q": q_test.tolist(),
            "ee_y": float(p0[1]), "ee_z": float(p0[2]),
            "mujoco_bias_sl": float(bias[1]),
            "best_scale": float(best_scale),
            "best_dz_m": float(best_dz),
            "scale_dz": {str(k): float(v) for k, v in scale_dz.items()},
        })

    import json
    out_path = ROOT / "outputs" / "control_runs" / "gravity_multi_pose_probe.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(str(out_path), "w"), indent=2)
    print(f"\nSaved: {out_path}")

    # Summary
    scales = [r["best_scale"] for r in results]
    print(f"\nBest-scale range: {min(scales):.2f} .. {max(scales):.2f}")
    print(f"Mean best scale: {np.mean(scales):.2f}")
    print(f"Std best scale:  {np.std(scales):.2f}")

    sim.stopSimulation()

if __name__ == "__main__":
    main()
