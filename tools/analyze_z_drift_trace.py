#!/usr/bin/env python3
"""Quick trace scan for Z drift during Y transport."""
from __future__ import annotations

import json
import sys
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/control_runs/coppeliasim_ur5_x_impedance_headless.jsonl")
all_rows = [json.loads(line) for line in path.open()]
print(f"trace lines={len(all_rows)}  t=[{all_rows[0]['time']:.2f}, {all_rows[-1]['time']:.2f}]")
print(
    f"ee_z start={all_rows[0]['ee_pos'][2]:.4f}  end={all_rows[-1]['ee_pos'][2]:.4f}  "
    f"last fixed_axis_2_error={all_rows[-1].get('fixed_axis_2_error')}"
)
rows = []
with path.open() as f:
    for line in f:
        d = json.loads(line)
        if float(d.get("time", 0.0)) >= 2.0:
            rows.append(d)

print("all samples:")
for d in rows:
    t = float(d["time"])
    ze = float(d.get("fixed_axis_2_error", d.get("z_error", 0.0)))
    ye = float(d.get("axis_error", d.get("y_error", 0.0)))
    ta = float(d.get("target_axis_accel", 0.0))
    print(
        f"  t={t:.2f}s  y_err={ye:+.4f}  z_err={ze:+.4f}  "
        f"a_cmd={ta:+.4f}  ik_torque_scale={d.get('ik_torque_scale')}  "
        f"ik_speed_scale={d.get('ik_speed_scale')}"
    )

mx = max(rows, key=lambda d: abs(float(d.get("fixed_axis_2_error", 0.0))))
print("\nmax |z_err|:")
print(f"  t={float(mx['time']):.2f}s  z_err={float(mx.get('fixed_axis_2_error',0.0)):+.4f}")
print(f"  ee_z={float(mx['ee_pos'][2]):.4f}  ee_y={float(mx['ee_pos'][1]):.4f}")
