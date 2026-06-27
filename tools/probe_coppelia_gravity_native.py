#!/usr/bin/env python3
"""
Extract CoppeliaSim UR5 link masses and COM, compute gravity torques natively.
Minimal version — single pose extraction + one empirical validation.
"""
import os, sys, time, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

COPPELIA_ROOT = Path(
    os.environ.get(
        "COPPELIA_ROOT",
        "/home/kbm/coppelia_runtime/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu22_04",
    )
)
sys.path.insert(0, str(COPPELIA_ROOT / "programming" / "zmqRemoteApi" / "clients" / "python" / "src"))
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

Q_SEED = [0.0, 0.0443244063, -1.67570517, 5.09435844, -6.28345754, 5.96178335]
JOINT_PATHS = [
    "/UR5/joint", "/UR5/link/joint", "/UR5/link/link/joint",
    "/UR5/link/link/link/joint", "/UR5/link/link/link/link/joint",
    "/UR5/link/link/link/link/link/joint",
]
G = np.array([0.0, 0.0, -9.81])


def connect(port):
    for attempt in range(20):
        try:
            print(f"  attempt {attempt+1}/20 connecting...", flush=True)
            c = RemoteAPIClient("127.0.0.1", port)
            s = c.require("sim")
            return c, s
        except Exception as e:
            print(f"  attempt {attempt+1}/20 failed: {e}", flush=True)
            time.sleep(5)
    raise RuntimeError("Cannot connect")


def main():
    port = int(os.environ.get("PORT", "23000"))
    print(f"Connecting port {port}...", flush=True)
    client, sim = connect(port)
    print("Connected.", flush=True)

    for attr in ("floatparam_simulation_time_step", "floatparam_physicstimestep"):
        if hasattr(sim, attr):
            try: sim.setFloatParam(int(getattr(sim, attr)), 0.005)
            except: pass
    sim.setStepping(True)
    if sim.getSimulationState() == sim.simulation_stopped:
        sim.startSimulation()
    print("Simulation started.", flush=True)

    model_path = str(COPPELIA_ROOT / "models" / "robots" / "non-mobile" / "UR5.ttm")
    sim.loadModel(model_path)
    handles = [sim.getObject(p) for p in JOINT_PATHS]
    ee_handle = sim.getObject("/UR5/connection")
    print("Model loaded.", flush=True)

    for h, q in zip(handles, Q_SEED):
        sim.setJointPosition(h, float(q))
    sim.step()
    for h in handles:
        sim.setJointMode(h, sim.jointmode_dynamic, 0)
        sim.setObjectInt32Param(h, sim.jointintparam_motor_enabled, 1)
        sim.setObjectInt32Param(h, sim.jointintparam_ctrl_enabled, 0)
    sim.step()
    print("Joints configured.", flush=True)

    # --- Extract link masses and COMs ---
    n = len(handles)
    all_shapes_per_joint = []
    for ji, jh in enumerate(handles):
        shapes = sim.getObjectsInTree(jh, sim.sceneobject_shape, 0)
        all_shapes_per_joint.append(set(shapes))

    link_masses = []
    link_coms_world = []
    print("\n=== LINK PARAMETERS ===", flush=True)
    for ji in range(n):
        if ji < n - 1:
            my_shapes = all_shapes_per_joint[ji] - all_shapes_per_joint[ji + 1]
        else:
            my_shapes = all_shapes_per_joint[ji]

        total_mass = 0.0
        weighted_com = np.zeros(3)
        for sh in my_shapes:
            mass = float(sim.getShapeMass(sh))
            mat = sim.getObjectMatrix(sh, -1)
            R = np.array([[mat[0],mat[1],mat[2]], [mat[4],mat[5],mat[6]], [mat[8],mat[9],mat[10]]])
            t = np.array([mat[3], mat[7], mat[11]])
            _, com_tf = sim.getShapeInertia(sh)
            com_local = np.array([com_tf[3], com_tf[7], com_tf[11]])
            com_world = R @ com_local + t
            total_mass += mass
            weighted_com += mass * com_world

        com = weighted_com / total_mass if total_mass > 1e-9 else np.zeros(3)
        link_masses.append(total_mass)
        link_coms_world.append(com)
        print(f"  Link {ji}: mass={total_mass:.4f} kg  COM=({com[0]:+.4f},{com[1]:+.4f},{com[2]:+.4f})  shapes={len(my_shapes)}", flush=True)

    # --- Compute gravity torques ---
    joint_pos_world = []
    joint_z_world = []
    for jh in handles:
        mat = sim.getObjectMatrix(jh, -1)
        pos = np.array([mat[3], mat[7], mat[11]])
        z = np.array([mat[2], mat[6], mat[10]])
        z = z / np.linalg.norm(z)
        joint_pos_world.append(pos)
        joint_z_world.append(z)

    tau_grav = np.zeros(n)
    for i in range(n):
        for j in range(i, n):
            m = link_masses[j]
            if m < 1e-9:
                continue
            r = link_coms_world[j] - joint_pos_world[i]
            moment = np.cross(r, m * G)
            tau_grav[i] += np.dot(joint_z_world[i], moment)

    print(f"\n=== NATIVE GRAVITY TORQUES ===", flush=True)
    print(f"  tau = {np.array2string(tau_grav, precision=3)}", flush=True)

    # Compare with MuJoCo
    try:
        from simulation.run_coppeliasim_x_axis_headless import (
            build_mujoco_gravity_estimator, compute_mujoco_gravity_bias,
        )
        est = build_mujoco_gravity_estimator()
        if est:
            q_np = np.array(Q_SEED)
            mj = compute_mujoco_gravity_bias(est, q_np, np.zeros(6))
            print(f"  MuJoCo bias = {np.array2string(mj, precision=3)}", flush=True)
            ratio = np.where(np.abs(mj) > 0.1, tau_grav / mj, np.nan)
            print(f"  ratio = {np.array2string(ratio, precision=3)}", flush=True)
    except Exception as e:
        print(f"  MuJoCo skip: {e}", flush=True)

    # --- Recompute gravity dynamically (re-extract COMs each time) ---
    def recompute_gravity_at_current_pose():
        lm, lc = [], []
        for ji in range(n):
            if ji < n - 1:
                my_shapes = all_shapes_per_joint[ji] - all_shapes_per_joint[ji + 1]
            else:
                my_shapes = all_shapes_per_joint[ji]
            tm, wc = 0.0, np.zeros(3)
            for sh in my_shapes:
                mass = float(sim.getShapeMass(sh))
                mat = sim.getObjectMatrix(sh, -1)
                R = np.array([[mat[0],mat[1],mat[2]], [mat[4],mat[5],mat[6]], [mat[8],mat[9],mat[10]]])
                tv = np.array([mat[3], mat[7], mat[11]])
                _, com_tf = sim.getShapeInertia(sh)
                cl = np.array([com_tf[3], com_tf[7], com_tf[11]])
                cw = R @ cl + tv
                tm += mass
                wc += mass * cw
            lm.append(tm)
            lc.append(wc / tm if tm > 1e-9 else np.zeros(3))
        jp, jz = [], []
        for jh in handles:
            mat = sim.getObjectMatrix(jh, -1)
            p = np.array([mat[3], mat[7], mat[11]])
            z = np.array([mat[2], mat[6], mat[10]])
            z = z / np.linalg.norm(z)
            jp.append(p)
            jz.append(z)
        tau = np.zeros(n)
        for i in range(n):
            for j in range(i, n):
                m = lm[j]
                if m < 1e-9: continue
                r = lc[j] - jp[i]
                moment = np.cross(r, m * G)
                tau[i] += np.dot(jz[i], moment)
        return tau

    print(f"\n=== DYNAMIC GRAVITY (re-extracted) ===", flush=True)
    tau_dyn = recompute_gravity_at_current_pose()
    print(f"  tau_dyn = {np.array2string(tau_dyn, precision=3)}", flush=True)

    # --- Empirical: native gravity comp vs freefall ---
    print("\n=== EMPIRICAL VALIDATION ===", flush=True)

    # Freefall
    for h, q in zip(handles, Q_SEED):
        sim.setJointPosition(h, float(q))
    for h in handles:
        sim.setJointTargetForce(h, 0.0, True)
    sim.step()
    z0 = sim.getObjectPosition(ee_handle, -1)[2]
    for _ in range(50):
        sim.step()
    z_free = sim.getObjectPosition(ee_handle, -1)[2]
    print(f"  Zero torque 50 steps: dz={z_free-z0:+.4f} m", flush=True)

    # Native gravity comp
    for h, q in zip(handles, Q_SEED):
        sim.setJointPosition(h, float(q))
    for h in handles:
        sim.setJointTargetForce(h, 0.0, True)
    sim.step()
    tau_fresh = recompute_gravity_at_current_pose()
    z0 = sim.getObjectPosition(ee_handle, -1)[2]
    tau_comp = -tau_fresh
    for h, t in zip(handles, tau_comp):
        sim.setJointTargetForce(h, float(t), True)
    for _ in range(50):
        sim.step()
    z_comp = sim.getObjectPosition(ee_handle, -1)[2]
    print(f"  Native grav comp 50 steps: dz={z_comp-z0:+.4f} m  tau={np.array2string(tau_comp, precision=2)}", flush=True)

    # Sweep scales of native gravity
    print("\n=== NATIVE GRAVITY SCALE SWEEP ===", flush=True)
    for scale in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
        for h, q in zip(handles, Q_SEED):
            sim.setJointPosition(h, float(q))
        for h in handles:
            sim.setJointTargetForce(h, 0.0, True)
        sim.step()
        tau_f = recompute_gravity_at_current_pose()
        z_s = sim.getObjectPosition(ee_handle, -1)[2]
        for h, t in zip(handles, -tau_f * scale):
            sim.setJointTargetForce(h, float(t), True)
        for _ in range(50):
            sim.step()
        z_e = sim.getObjectPosition(ee_handle, -1)[2]
        print(f"  scale={scale:.1f}  dz={z_e-z_s:+.5f} m  tau_sl={-tau_f[1]*scale:+.2f}", flush=True)

    # MuJoCo bias sweep for comparison
    print("\n=== MUJOCO BIAS SCALE SWEEP ===", flush=True)
    try:
        for scale in [1.0, 1.5, 1.8, 2.0, 2.5]:
            for h, q in zip(handles, Q_SEED):
                sim.setJointPosition(h, float(q))
            for h in handles:
                sim.setJointTargetForce(h, 0.0, True)
            sim.step()
            z_s = sim.getObjectPosition(ee_handle, -1)[2]
            mj_ff = -mj * scale
            for h, t in zip(handles, mj_ff):
                sim.setJointTargetForce(h, float(t), True)
            for _ in range(50):
                sim.step()
            z_e = sim.getObjectPosition(ee_handle, -1)[2]
            print(f"  MJ scale={scale:.1f}  dz={z_e-z_s:+.5f} m  tau_sl={mj_ff[1]:+.2f}", flush=True)
    except Exception as e:
        print(f"  MJ sweep skipped: {e}", flush=True)

    # Save results
    result = {
        "q_seed": Q_SEED,
        "link_masses_kg": link_masses,
        "link_coms_world": [c.tolist() for c in link_coms_world],
        "joint_positions_world": [p.tolist() for p in joint_pos_world],
        "joint_axes_world": [z.tolist() for z in joint_z_world],
        "tau_gravity_native": tau_grav.tolist(),
    }
    out_path = ROOT / "outputs" / "control_runs" / "coppelia_ur5_gravity_model.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(str(out_path), "w"), indent=2)
    print(f"\nSaved: {out_path}", flush=True)

    sim.stopSimulation()
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
