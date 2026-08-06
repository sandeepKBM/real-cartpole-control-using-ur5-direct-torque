"""
Module-level constants for the X-axis Cartesian impedance controller.

Split out of the former single-file ``x_axis_cartesian_impedance.py`` module
(pure structural refactor; see the package ``__init__.py`` for the original
module docstring).
"""

from __future__ import annotations

import numpy as np


JOINT_NAME_ORDER: tuple[str, ...] = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)

# Fixed per-joint weighting for the optional wrist-orientation task term
# (``wrist_orientation_task``, see CartesianImpedanceConfig below). Shape
# matches ``JOINT_NAME_ORDER``. Values are the ``face_mask``/``orientation
# mask`` ratios from the two pre-torque-lane kinematic controllers that
# originated this task-partition idea (``archive/legacy_mujoco/controller.py``
# -- ``split_forearm_origin_face_controller`` (C1) and
# ``differential_ik_split_controller`` (C2), documented in
# ``docs/archive/SLSQP_CONTROLLER_REFERENCE.md``): both use
# ``face_mask = [0, 0, 0, 1.25, 1.55, 1.25]`` over
# ``[shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3]`` --
# exactly zero on the three proximal joints, heavily weighted on the wrist
# chain with wrist_2 (the joint that sits at 0 at the transport singularity)
# weighted highest. Normalized here to a 1.0 peak; only the *shape* is taken
# from the legacy controllers (an overall scale is absorbed by
# ``kp_rot_wrist``/``kd_rot_wrist``), not the literal legacy PD gains.
WRIST_ORIENTATION_MASK: np.ndarray = np.array(
    [0.0, 0.0, 0.0, 1.25 / 1.55, 1.0, 1.25 / 1.55], dtype=np.float64
)
