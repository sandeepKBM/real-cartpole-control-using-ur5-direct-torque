#!/usr/bin/env python3
"""Deep diagnostic of the torque trace to understand warmup hold failure."""
import json
import sys
import math
from pathlib import Path

path = sys.argv[1] if len(sys.argv) > 1 else (
    "outputs/control_runs/coppeliasim_ur5_x_impedance_headless.jsonl"
)
rows = [json.loads(line) for line in open(path)]
n = len(rows)
z0 = rows[0]["ee_pos"][2]
y0 = rows[0]["ee_pos"][1]
x0 = rows[0]["ee_pos"][0]
q0 = rows[0]["q"]

print(f"=== TRACE DIAGNOSTIC: {n} steps, {rows[-1]['time']:.2f}s ===")
print(f"Start EE: x={x0:.4f}  y={y0:.4f}  z={z0:.4f}")
print(f"Start q:  {['%.3f' % qi for qi in q0]}")
print()

# Phase 1: Characterize the first step
r0 = rows[0]
print("--- STEP 0 (t=0.00) ---")
print(f"  tau_cmd:     {['%.2f' % t for t in r0['tau_cmd']]}")
print(f"  tau_raw:     {['%.2f' % t for t in r0['tau_raw']]}")
tau_preclip = r0.get("tau_preclip", [0]*6)
print(f"  tau_preclip: {['%.2f' % t for t in tau_preclip]}")
tau_task = r0.get("tau_task", [0]*6)
print(f"  tau_task:    {['%.2f' % t for t in tau_task]}")
tau_posture = r0.get("tau_posture", [0]*6)
print(f"  tau_posture: {['%.2f' % t for t in tau_posture]}")
tau_damping = r0.get("tau_damping", [0]*6)
print(f"  tau_damping: {['%.2f' % t for t in tau_damping]}")
print()

# Phase 2: Early warmup dynamics (first 10 steps)
print("--- WARMUP PHASE (step by step) ---")
for i, r in enumerate(rows[:min(20, n)]):
    qd = r["qd"]
    dx = r["ee_pos"][0] - x0
    dy = r["ee_pos"][1] - y0
    dz = r["ee_pos"][2] - z0
    max_tau = max(abs(t) for t in r["tau_cmd"])
    max_qd = max(abs(v) for v in qd)
    # shoulder_lift tau is typically the gravity-dominant joint
    tau_sl = r["tau_cmd"][1]
    tau_el = r["tau_cmd"][2]
    ori = r["orientation_error_norm"]
    safe = r.get("safety_ok")
    print(
        f"  t={r['time']:.2f}  "
        f"dz={dz:+.4f}  dy={dy:+.4f}  dx={dx:+.4f}  "
        f"tau_sl={tau_sl:+.1f}  tau_el={tau_el:+.1f}  "
        f"|tau|={max_tau:.1f}  |qd|={max_qd:.3f}  "
        f"ori={math.degrees(ori):.1f}deg  safe={safe}"
    )

print()

# Phase 3: Find when things go wrong
print("--- KEY TRANSITIONS ---")
first_z_drift_2cm = None
first_z_drift_5cm = None
first_z_drift_8cm = None
first_ori_bad = None
first_qd_fast = None
transport_armed = None

for i, r in enumerate(rows):
    dz = abs(r["ee_pos"][2] - z0)
    ori = r["orientation_error_norm"]
    max_qd = max(abs(v) for v in r["qd"])
    
    if first_z_drift_2cm is None and dz > 0.02:
        first_z_drift_2cm = r["time"]
        print(f"  Z drift > 2cm at t={r['time']:.3f}s (step {i})")
    if first_z_drift_5cm is None and dz > 0.05:
        first_z_drift_5cm = r["time"]
        print(f"  Z drift > 5cm at t={r['time']:.3f}s (step {i})")
    if first_z_drift_8cm is None and dz > 0.08:
        first_z_drift_8cm = r["time"]
        print(f"  Z drift > 8cm at t={r['time']:.3f}s (step {i})")
    if first_ori_bad is None and ori > 0.35:
        first_ori_bad = r["time"]
        print(f"  Orientation > 0.35 rad at t={r['time']:.3f}s (step {i})")
    if first_qd_fast is None and max_qd > 1.0:
        first_qd_fast = r["time"]
        print(f"  Joint speed > 1 rad/s at t={r['time']:.3f}s (step {i})")
    
    ik_diag = r.get("ik_diagnostics", {})
    if ik_diag and transport_armed is None:
        transport_armed = r["time"]
        print(f"  Transport armed at t={r['time']:.3f}s (step {i})")

print()

# Phase 4: Gravity analysis
print("--- GRAVITY ANALYSIS ---")
r0_preclip = rows[0].get("tau_preclip", [0]*6)
r0_task = rows[0].get("tau_task", [0]*6)
print(f"  Step 0 preclip (should be ~gravity comp): {['%.2f' % t for t in r0_preclip]}")
print(f"  Step 0 task (should be ~0 at start pose):  {['%.2f' % t for t in r0_task]}")
print(f"  Step 0 tau_cmd (sent to sim):              {['%.2f' % t for t in r0['tau_cmd']]}")

# Check if shoulder_lift torque direction makes sense for gravity:
# At this pose, shoulder_lift at ~0.044 rad, elbow at ~-1.68 rad
# Gravity should pull the arm down, so we need positive shoulder_lift torque
# to hold the arm up (CoppeliaSim convention)
sl_tau_cmd = r0["tau_cmd"][1]
print(f"\n  Shoulder lift cmd: {sl_tau_cmd:+.2f} Nm")
if sl_tau_cmd > 0:
    print(f"  → Pushing UP (positive shoulder lift in Coppelia frame)")
elif sl_tau_cmd < 0:
    print(f"  → Pushing DOWN (negative shoulder lift in Coppelia frame)")
print(f"  Step 1 shoulder lift EE z velocity: {rows[1]['ee_lin_vel'][2]:.4f} m/s")
if rows[1]['ee_lin_vel'][2] < -0.05:
    print(f"  → EE moving DOWN despite torque — gravity comp may be WRONG SIGN or INSUFFICIENT")
elif rows[1]['ee_lin_vel'][2] > 0.05:
    print(f"  → EE moving UP — gravity comp may be REVERSED (pushing up too hard)")

print()

# Phase 5: Joint-by-joint analysis
print("--- JOINT EXCURSION SUMMARY ---")
for ji in range(6):
    q_vals = [r["q"][ji] for r in rows]
    q_range = max(q_vals) - min(q_vals)
    q_drift = q_vals[-1] - q_vals[0]
    print(f"  Joint {ji}: start={q_vals[0]:+.4f}  end={q_vals[-1]:+.4f}  "
          f"drift={q_drift:+.4f}  range={q_range:.4f} rad")

print()

# Phase 6: Torque saturation analysis
print("--- TORQUE SATURATION ---")
sat_counts = [0]*6
total = len(rows)
for r in rows:
    tau = r["tau_cmd"]
    for ji in range(6):
        limits = [20, 50, 50, 10, 10, 10]
        if abs(tau[ji]) >= limits[ji] - 0.1:
            sat_counts[ji] += 1
for ji in range(6):
    pct = 100.0 * sat_counts[ji] / total
    print(f"  Joint {ji}: saturated {sat_counts[ji]}/{total} steps ({pct:.0f}%)")

print()
print("--- SIMULATION TIMESTEP ---")
if len(rows) >= 2:
    dt = rows[1]["time"] - rows[0]["time"]
    print(f"  sim_dt = {dt:.4f}s ({1/dt:.0f} Hz)")
    print(f"  This is {'good' if dt <= 0.01 else 'SLOW'} for torque control")
    if dt > 0.01:
        print(f"  WARNING: dt={dt}s is very large for impedance control.")
        print(f"  Gains tuned for 100Hz may be unstable at {1/dt:.0f}Hz.")
        print(f"  Consider: reduce gains by factor ~{dt/0.01:.1f}x, or increase sim rate.")
