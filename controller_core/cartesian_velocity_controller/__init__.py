"""Resolved-rate Cartesian velocity controller for the UR5e's native RTDE
``speedL`` interface. Still no torque/gravity/mass-matrix dynamics -- but
DOES use the kinematic Jacobian (see reduced_task_dims, below), a deliberate
narrowing of the original "no Jacobian at all" design.

Why this exists: UR5e has no native torque interface -- every torque-control
mechanism in this package (x_axis_cartesian_impedance.py, torque_task_qp.py,
hard_constraint_qp.py) exists to fake compliant force behavior on a robot
that is natively position/velocity-controlled, and essentially every
documented Y-drift/orientation bug this repo has fought (see AGENTS.md
section 3) is a consequence of that dynamics modeling, not of the transport
task itself. ``speedL`` resolves a commanded Cartesian velocity to joint
velocities via the Jacobian ON THE ROBOT'S OWN FIRMWARE. The real tradeoff:
this gives zero force compliance, so it is only appropriate for phases
where nothing needs to push back on the end-effector (pure point-to-point
transport / range characterization) -- not for eventual swing-up once a
physical pole is mounted and pole-arm interaction forces matter. See
hardware/velocity_transport.py's module docstring for the real-hardware
wiring.

This package is organized as:
  - ``math_utils.py`` -- ``_damped_pinv``, the Tikhonov-damped pseudoinverse
    shared by the redundancy-resolution modes.
  - ``config.py`` -- ``CartesianVelocityConfig``, the dataclass of gains and
    per-mode flags, plus its YAML loader.
  - ``modes.py`` -- the four end-effector orientation/redundancy resolution
    strategies (``reduced_task_dims`` [default], ``split_base_wrist_task``,
    ``ik_seeded_resolution``, and the trivial full-orientation-hold
    passthrough), each a standalone function. This is where the detailed
    design history lives: what each mode does, what was tried and rejected,
    and the measured evidence (multistability, unbounded divergence, a
    coupling leak, path-dependence) behind the current defaults.
  - ``controller.py`` -- ``CartesianVelocityController``, the orchestrator:
    shared per-cycle setup (position error, swing-twist orientation error),
    dispatch to exactly one mode, then the shared speed clamp.

DO NOT treat reduced_task_dims=True (the current default) or
split_base_wrist_task as validated for real-hardware use across a
displacement range -- both have real, verified failure modes documented in
``modes.py``. ik_seeded_resolution is the one mode with a predictable,
monotonic safety boundary measured in sim; it has NOT yet been tried on real
hardware, and the default is still reduced_task_dims=True pending a
decision to promote ik_seeded_resolution (not done unilaterally here).
"""

from __future__ import annotations

from .config import CartesianVelocityConfig
from .controller import CartesianVelocityController
from .math_utils import _damped_pinv

__all__ = [
    "CartesianVelocityConfig",
    "CartesianVelocityController",
    "_damped_pinv",
]
