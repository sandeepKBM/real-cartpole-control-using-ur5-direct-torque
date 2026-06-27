#!/usr/bin/env python3
"""
Empirical gravity sign probe:
1. Set UR5 to seed pose.
2. Apply zero torque for 5 steps, log EE z velocity.
3. Apply +tau_shoulder_lift for 5 steps, log velocity.
4. Apply -tau_shoulder_lift for 5 steps, log velocity.
5. Report which direction fights gravity.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

COPPELIA_ROOT = Path("/home/kbm/coppelia_runtime/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu22_04")
sys.path.insert(0, str(COPPELIA_ROOT / "programming" / "zmqRemoteApi" / "clients" / "python" / "src"))

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

Q_SEED = [0.0, 0.0443244063, -1.67570517, 5.09435844, -6.28345754, 5.96178335]

JOINT_PATHS = [
    "/UR5/joint",
    "/UR5/link/joint",
    "/UR5/link/link/joint",
    "/UR5/link/link/link/joint",
    "/UR5/link/link/link/link/joint",
    "/UR5/link/link/link/link/link/joint",
]

def main():
    import os
    port = int(os.environ.get("PORT", "23050"))
    client = RemoteAPIClient("127.0.0.1", port)
    sim = client.require("sim")

    try:
        for attr_name in ("floatparam_simulation_time_step", "floatparam_physicstimestep"):
            if hasattr(sim, attr_name):
                sim.setFloatParam(int(getattr(sim, attr_name)), 0.005)
    except Exception as e:
        print(f"Warning: could not set sim dt: {e}")

    sim.setStepping(True)
    if sim.getSimulationState() == sim.simulation_stopped:
        sim.startSimulation()

    model_path = str(COPPELIA_ROOT / "models" / "robots" / "non-mobile" / "UR5.ttm")
    ur5 = sim.loadModel(model_path)

    handles = []
    for path in JOINT_PATHS:
        handles.append(sim.getObject(path))

    for h, q in zip(handles, Q_SEED):
        sim.setJointPosition(h, float(q))
    sim.step()

    for h in handles:
        sim.setJointMode(h, sim.jointmode_dynamic, 0)
        sim.setObjectInt32Param(h, sim.jointintparam_motor_enabled, 1)
        sim.setObjectInt32Param(h, sim.jointintparam_ctrl_enabled, 0)
    sim.step()

    # Resolve EE
    ee_path = "/UR5/connection"
    ee_handle = sim.getObject(ee_path)
    
    def read_ee_z():
        pos = sim.getObjectPosition(ee_handle, -1)
        return pos[2]
    
    def read_ee_pos():
        return sim.getObjectPosition(ee_handle, -1)
    
    def read_joint_forces():
        forces = []
        for h in handles:
            try:
                forces.append(float(sim.getJointForce(h)))
            except Exception:
                forces.append(float("nan"))
        return forces

    def apply_torque(tau_vec):
        for h, t in zip(handles, tau_vec):
            sim.setJointTargetForce(h, float(t), True)

    z0 = read_ee_z()
    pos0 = read_ee_pos()
    print(f"Start EE: x={pos0[0]:.4f}  y={pos0[1]:.4f}  z={pos0[2]:.4f}")
    
    dt = float(sim.getSimulationTimeStep())
    print(f"Sim dt: {dt:.4f}s")

    # Phase 1: Zero torque — let gravity act
    print("\n--- PHASE 1: Zero torque (gravity only) ---")
    apply_torque([0]*6)
    for i in range(10):
        sim.step()
        z = read_ee_z()
        pos = read_ee_pos()
        forces = read_joint_forces()
        print(f"  step {i}: z={z:.4f}  dz={z-z0:+.4f}  "
              f"joint_forces={['%.2f' % f for f in forces]}")
    
    z_after_freefall = read_ee_z()
    
    # Reset
    for h, q in zip(handles, Q_SEED):
        sim.setJointPosition(h, float(q))
    apply_torque([0]*6)
    sim.step()
    
    # Sweep torques to find the gravity-balancing value
    print("\n--- GRAVITY TORQUE SWEEP (shoulder_lift only) ---")
    for tau_sl_val in [+20, +40, +60, +80, +100, +120, +150, -40, -80]:
        for h, q in zip(handles, Q_SEED):
            sim.setJointPosition(h, float(q))
        apply_torque([0]*6)
        sim.step()
        z_start = read_ee_z()
        
        tau = [0, float(tau_sl_val), 0, 0, 0, 0]
        apply_torque(tau)
        for i in range(100):
            sim.step()
        z_end = read_ee_z()
        print(f"  tau_sl={tau_sl_val:+5.0f} Nm:  dz={z_end - z_start:+.4f} m (z_end={z_end:.4f})")

    # Multi-joint gravity balance test
    print("\n--- MULTI-JOINT GRAVITY BALANCE ---")
    for h, q in zip(handles, Q_SEED):
        sim.setJointPosition(h, float(q))
    apply_torque([0]*6)
    sim.step()
    z_start = read_ee_z()
    
    # Apply negated MuJoCo bias (sign=-1) with scaling
    import numpy as np_
    bias = np_.array([0.0, -37.46, 0.44, -0.44, 0.045, 0.0])
    for scale in [1.0, 1.5, 2.0, 3.0, 4.0]:
        for h, q in zip(handles, Q_SEED):
            sim.setJointPosition(h, float(q))
        apply_torque([0]*6)
        sim.step()
        z_s = read_ee_z()
        
        grav_comp = -bias * scale
        apply_torque(grav_comp.tolist())
        for i in range(200):
            sim.step()
        z_e = read_ee_z()
        print(f"  scale={scale:.1f}x  tau={['%.1f' % t for t in grav_comp]}  "
              f"dz={z_e - z_s:+.4f}  z_end={z_e:.4f}")
    
    # Also check: what does getJointForce report during zero-torque freefall?
    print("\n--- COPPELIA getJointForce DURING FREEFALL ---")
    for h, q in zip(handles, Q_SEED):
        sim.setJointPosition(h, float(q))
    apply_torque([0]*6)
    sim.step()
    sim.step()
    forces = read_joint_forces()
    print(f"  Joint forces (zero cmd): {['%.2f' % f for f in forces]}")
    print(f"  shoulder_lift force: {forces[1]:+.2f} Nm")
    print(f"  (This is the reaction force — gravity comp should CANCEL this)")
    
    sim.stopSimulation()
    print("\nDone.")

if __name__ == "__main__":
    main()
