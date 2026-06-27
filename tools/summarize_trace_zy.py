#!/usr/bin/env python3
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else (
    "outputs/control_runs/coppeliasim_ur5_x_impedance_headless.jsonl"
)
rows = [json.loads(line) for line in open(path)]
z0 = rows[0]["ee_pos"][2]
y0 = rows[0]["ee_pos"][1]
print(f"lines={len(rows)} duration={rows[-1]['time']:.2f}s")
for r in rows[:: max(1, len(rows) // 12)]:
    print(
        f"t={r['time']:.2f} dz={r['ee_pos'][2] - z0:+.3f} "
        f"dy={r['ee_pos'][1] - y0:+.3f} "
        f"tau={max(abs(t) for t in r['tau_cmd']):.1f} "
        f"safe={r.get('safety_ok')}"
    )
print("--- last ---")
r = rows[-1]
print(
    f"t={r['time']:.2f} dz={r['ee_pos'][2] - z0:+.3f} "
    f"dy={r['ee_pos'][1] - y0:+.3f} "
    f"tau={max(abs(t) for t in r['tau_cmd']):.1f}"
)
