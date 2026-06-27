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

q = np.array([0, 0.0443244063, -1.67570517, 5.09435844, -6.28345754, 5.96178335])
est = build_mujoco_gravity_estimator()
print("estimator_ok", est is not None)
g = compute_mujoco_gravity_bias(est, q, np.zeros(6))
print("gravity_nm", g)
