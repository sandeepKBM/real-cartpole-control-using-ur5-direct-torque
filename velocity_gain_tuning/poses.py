"""Scenario catalog for velocity-control gain tuning/evaluation.

Reuses hardware/poses.py's named joint-pose constants directly (not
redefined here) -- these are the SAME poses this session's manual sweeps
already characterized for controller_core.cartesian_velocity_controller's
ik_seeded_resolution mode. Each scenario also carries a max_dx_hint (the
approximate safe-range boundary already found by hand, purely to size a
sensible dx search/eval range per pose -- NOT a hard limit, the optimizer
and evaluator are free to test beyond it and will simply see guard trips
reported honestly if they do).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hardware.poses import (
    HANGING_ALPHA_0_5_Q,
    HEIGHT_ALPHA_0_5_CLEARANCE_Q,
    HEIGHT_ALPHA_0_5_WRIST2_OFFSET_Q,
)

# -40deg base-rotation, wrist_2=0.2 offset -- this session's primary test
# pose throughout the ik_seeded_resolution investigation.
NEG40_WRIST2OFFSET_Q = np.array(
    [-0.6981317007977318, -0.8353981633974483, -1.2, -0.9853981633974482, 0.2, 0.0],
    dtype=np.float64,
)

# -45deg base-rotation, wrist_2=0.2 offset -- HEIGHT_ALPHA_0_5_CLEARANCE_Q
# (the real-hardware default start pose) with the wrist_2 singularity
# offset applied, matching this session's neg45 sweep.
NEG45_WRIST2OFFSET_Q = HEIGHT_ALPHA_0_5_CLEARANCE_Q.copy()
NEG45_WRIST2OFFSET_Q[4] = 0.2


@dataclass(frozen=True)
class PoseScenario:
    name: str
    q0: np.ndarray
    max_dx_hint_m: float  # approximate empirically-found safe-range boundary, for sizing searches only


POSE_SCENARIOS: tuple[PoseScenario, ...] = (
    PoseScenario("neg40_wrist2offset", NEG40_WRIST2OFFSET_Q, max_dx_hint_m=0.029),
    PoseScenario("unrotated_wrist2offset", HEIGHT_ALPHA_0_5_WRIST2_OFFSET_Q, max_dx_hint_m=0.049),
    PoseScenario("neg45_wrist2offset", NEG45_WRIST2OFFSET_Q, max_dx_hint_m=0.029),
    PoseScenario("hanging_alpha_0_5", HANGING_ALPHA_0_5_Q, max_dx_hint_m=0.185),
)


def scenario_by_name(name: str) -> PoseScenario:
    for scenario in POSE_SCENARIOS:
        if scenario.name == name:
            return scenario
    raise KeyError(f"Unknown pose scenario {name!r}; known: {[s.name for s in POSE_SCENARIOS]}")
