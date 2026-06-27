#!/usr/bin/env python3
"""Analyze Z-axis drift during transport to understand gravity model mismatch."""
import json, sys
from pathlib import Path

path = sys.argv[1] if len(sys.argv) > 1 else "outputs/control_runs/coppeliasim_ur5_x_impedance_headless.jsonl"
rows = [json.loads(line) for line in open(path)]
n = len(rows)

print(f"=== Z OSCILLATION ANALYSIS: {n} steps, {rows[-1]['time']:.3f}s ===\n")

z0 = rows[0]["ee_pos"][2]
y0 = rows[0]["ee_pos"][1]
q0 = rows[0]["q"]

# Find warmup end / transport start
warmup_end_idx = 0
for i, r in enumerate(rows):
    if r.get("phase", "").startswith("transport") or r.get("transport_active", False):
        warmup_end_idx = i
        break

print(f"Warmup ended at step {warmup_end_idx}, t={rows[warmup_end_idx]['time']:.3f}s")
print(f"Initial EE: y={y0:.4f}  z={z0:.4f}")
print(f"Warmup-end EE: y={rows[warmup_end_idx]['ee_pos'][1]:.4f}  z={rows[warmup_end_idx]['ee_pos'][2]:.4f}")
print()

# During transport: track Z drift, q changes, and gravity torque
print("=== TRANSPORT Z-DRIFT TIMELINE ===")
print(f"{'step':>6}  {'time':>6}  {'ee_y':>8}  {'ee_z':>8}  {'dz':>8}  {'q1':>7}  {'q2':>8}  {'dq1':>7}  {'dq2':>8}  {'tau_cmd[1]':>10}  {'grav[1]':>10}")
print("-" * 110)

transport_rows = rows[warmup_end_idx:]
z_peaks = []
z_troughs = []
prev_dz = 0.0
direction = 0  # +1 rising, -1 falling

for i, r in enumerate(transport_rows):
    idx = warmup_end_idx + i
    z = r["ee_pos"][2]
    y = r["ee_pos"][1]
    dz = z - z0
    q = r["q"]
    dq1 = q[1] - q0[1]
    dq2 = q[2] - q0[2]

    tau_cmd = r.get("tau_cmd", [0]*6)
    grav = r.get("gravity_torque", r.get("gravity_ff", [0]*6))

    new_dir = 1 if dz > prev_dz else (-1 if dz < prev_dz else direction)
    if direction == 1 and new_dir == -1:
        z_peaks.append((idx, r["time"], dz, q[1], q[2]))
    elif direction == -1 and new_dir == 1:
        z_troughs.append((idx, r["time"], dz, q[1], q[2]))
    direction = new_dir
    prev_dz = dz

    if i % 20 == 0 or i < 5 or i == len(transport_rows) - 1:
        tau1 = tau_cmd[1] if len(tau_cmd) > 1 else 0.0
        g1 = grav[1] if grav and len(grav) > 1 else 0.0
        print(f"{idx:6d}  {r['time']:6.3f}  {y:+8.4f}  {z:+8.4f}  {dz:+8.4f}  "
              f"{q[1]:7.3f}  {q[2]:+8.3f}  {dq1:+7.3f}  {dq2:+8.3f}  "
              f"{tau1:+10.2f}  {g1:+10.2f}")

print()
print("=== Z PEAKS (local maxima) ===")
for idx, t, dz, q1, q2 in z_peaks[:15]:
    print(f"  step {idx:5d}  t={t:.3f}  dz={dz:+.5f}  q1={q1:.4f}  q2={q2:.4f}")

print()
print("=== Z TROUGHS (local minima) ===")
for idx, t, dz, q1, q2 in z_troughs[:15]:
    print(f"  step {idx:5d}  t={t:.3f}  dz={dz:+.5f}  q1={q1:.4f}  q2={q2:.4f}")

# Check if gravity_torque field exists and how it varies
print("\n=== GRAVITY TORQUE VARIATION ===")
grav_vals = []
for r in rows:
    g = r.get("gravity_torque", r.get("gravity_ff"))
    if g:
        grav_vals.append(g)
if grav_vals:
    import numpy as np
    ga = np.array(grav_vals)
    print(f"  Gravity torque samples: {len(ga)}")
    for j in range(min(6, ga.shape[1])):
        print(f"  Joint {j}: min={ga[:, j].min():+.3f}  max={ga[:, j].max():+.3f}  "
              f"range={ga[:, j].max() - ga[:, j].min():.3f}  "
              f"start={ga[0, j]:+.3f}  end={ga[-1, j]:+.3f}")
else:
    print("  No gravity_torque field in trace.")

# Check tau_cmd variation for shoulder_lift
print("\n=== TAU_CMD VARIATION (shoulder_lift) ===")
tau1_vals = [r.get("tau_cmd", [0]*6)[1] for r in rows if r.get("tau_cmd")]
if tau1_vals:
    import numpy as np
    ta = np.array(tau1_vals)
    print(f"  min={ta.min():+.2f}  max={ta.max():+.2f}  mean={ta.mean():+.2f}")

# MuJoCo bias variation with q
print("\n=== MUJOCO BIAS VARIATION ACROSS TRACE ===")
try:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from simulation.run_coppeliasim_x_axis_headless import (
        build_mujoco_gravity_estimator, compute_mujoco_gravity_bias,
    )
    import numpy as np
    est = build_mujoco_gravity_estimator()
    if est:
        biases = []
        for r in rows:
            q = np.array(r["q"])
            b = compute_mujoco_gravity_bias(est, q, np.zeros(6))
            biases.append(b)
        ba = np.array(biases)
        print(f"  MuJoCo bias samples: {len(ba)}")
        for j in range(6):
            print(f"  Joint {j}: min={ba[:, j].min():+.3f}  max={ba[:, j].max():+.3f}  "
                  f"range={ba[:, j].max() - ba[:, j].min():.3f}  "
                  f"start={ba[0, j]:+.3f}  end={ba[-1, j]:+.3f}")
        
        # Optimal scale per step
        if grav_vals:
            print("\n=== GRAVITY SCALE NEEDED PER STEP (rough estimate) ===")
            ga_applied = np.array(grav_vals)
            for step_idx in [0, len(rows)//4, len(rows)//2, 3*len(rows)//4, len(rows)-1]:
                q = np.array(rows[step_idx]["q"])
                b = compute_mujoco_gravity_bias(est, q, np.zeros(6))
                applied = ga_applied[step_idx]
                for j in [1, 2, 3]:
                    if abs(b[j]) > 0.1:
                        scale_est = -applied[j] / b[j] if abs(b[j]) > 0.01 else float("nan")
                        print(f"  step {step_idx:5d} joint {j}: bias={b[j]:+.3f}  "
                              f"applied={applied[j]:+.3f}  implied_scale={scale_est:.2f}")
except Exception as exc:
    print(f"  Could not compute MuJoCo bias: {exc}")
