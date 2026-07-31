"""Named joint poses shared by MuJoCo experiments and hardware bring-up."""

from __future__ import annotations

import numpy as np

# Canonical transport poses (see rl_gain_scheduling/gain_scheduling_env.py).
ACTIVE_ORIGIN_Q = np.array(
    [0.0, -1.5707963267948966, 0.0, -1.5707963267948966, 0.0, 0.0],
    dtype=np.float64,
)
LOWER_B_Q = np.array([0.0, -0.1, -2.4, -0.4, 0.0, 0.0], dtype=np.float64)

# height_alpha=0.5 — matches tools/_run_fixed_gain_test.sh Test 2.
HEIGHT_ALPHA_0_5_Q = (0.5 * ACTIVE_ORIGIN_Q + 0.5 * LOWER_B_Q).astype(np.float64)

# height_alpha=0.5 with a real-lab base-rotation for wall/obstacle clearance
# (shoulder_pan = -45 deg), visually confirmed twice on the real robot
# (2026-07-31) at this exact pose in this exact physical setup. Same
# precedent value used for the alpha=0.1 pose on 2026-07-28. This is now the
# default real-hardware start pose for height_alpha_0_5 (see
# tools/ur5e_move_joints.py / hardware/x_transport.py /
# hardware/urscript_transport.py) -- if the robot or room layout ever
# changes, re-verify clearance visually before trusting this again; override
# back to HEIGHT_ALPHA_0_5_Q (or a fresh --shoulder-pan-override-rad) if so.
HEIGHT_ALPHA_0_5_CLEARANCE_Q = HEIGHT_ALPHA_0_5_Q.copy()
HEIGHT_ALPHA_0_5_CLEARANCE_Q[0] = -0.7853981633974483


def q_for_height_alpha(alpha: float) -> np.ndarray:
    """Interpolate between active-origin (0) and lower-B (1) joint poses."""
    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"height_alpha must be in [0, 1]; got {alpha}")
    return ((1.0 - alpha) * ACTIVE_ORIGIN_Q + alpha * LOWER_B_Q).astype(np.float64)
