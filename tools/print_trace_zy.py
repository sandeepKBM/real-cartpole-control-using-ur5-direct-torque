#!/usr/bin/env python3
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "outputs/control_runs/coppeliasim_ur5_x_impedance_headless.jsonl"
rows = [json.loads(line) for line in open(path)]
z0 = rows[0]["ee_pos"][2]
y0 = rows[0]["ee_pos"][1]
for r in rows:
    z = r["ee_pos"][2]
    y = r["ee_pos"][1]
    tau_max = max(abs(t) for t in r["tau_cmd"])
    print(
        f"t={r['time']:.2f} dz={z - z0:+.4f} dy={y - y0:+.4f} "
        f"tau_max={tau_max:.2f} safety={r.get('safety_ok')}"
    )
