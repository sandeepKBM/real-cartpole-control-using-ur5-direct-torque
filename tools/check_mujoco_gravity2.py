#!/usr/bin/env python3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from simulation.run_coppeliasim_x_axis_headless import (
    build_mujoco_gravity_estimator,
    compute_mujoco_gravity_bias,
)

q_wrapped = np.array([0, 0.0443244063, -1.67570517, 5.09435844, -6.28345754, 5.96178335])
q_unwrapped = np.array([0, 0.0443244063, -1.67570517, -1.1888, 0.0, -0.3214])
qd = np.zeros(6)

est = build_mujoco_gravity_estimator()
print("estimator_ok:", est is not None)

g1 = compute_mujoco_gravity_bias(est, q_wrapped, qd)
print("qfrc_bias (wrapped q):", np.array2string(g1, precision=3))

g2 = compute_mujoco_gravity_bias(est, q_unwrapped, qd)
print("qfrc_bias (unwrapped q):", np.array2string(g2, precision=3))

print()
print("shoulder_lift bias (wrapped):", g1[1], "Nm")
print("shoulder_lift bias (unwrapped):", g2[1], "Nm")
print()
print("For Coppelia (where +torque = counterclockwise in joint frame):")
print("  If MuJoCo bias is negative, the sign=-1 negation gives positive torque.")
print("  If MuJoCo bias is positive, the sign=-1 negation gives negative torque.")
print()
print("sign=-1 applied:", -g2)
print("sign=+1 applied:", g2)
