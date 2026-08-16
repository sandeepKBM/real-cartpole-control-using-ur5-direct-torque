"""
Reduced-task (X + orientation) torque QP with a Y/Z *corridor* instead of a
Y/Z hold, plus the existing manipulability CBF, composed into ONE QP solve.

Why this exists (and why it is not another set of gains on the existing
operational-space controller): the tuned OSC controller
(``controller_core/x_axis_cartesian_impedance/``) is a full 6-DOF Khatib
operational-space controller. On a NON-redundant 6-DOF arm doing a 6D task,
its dynamically-consistent nullspace projector has essentially nothing to
project into, and every Cartesian axis -- including the two (Y, Z) this task
does not actually care about -- competes for the same torque budget. A
validated finding
(``docs/status/neg45_y_axis_diagnosis_and_fix_2026-08-01.md``) established
that no P, D, or I gain in that architecture can hold Y without breaking
X-tracking at the real transport pose: a structural authority conflict, not
a tuning gap.

This package changes the PROBLEM rather than the gains:

- The task is 4-dimensional (world-X translation + all 3 orientation rows),
  built from ``J_reduced = vstack([J[0:1,:], J[3:6,:]])``. Y and Z are
  excluded BY CONSTRUCTION -- they are not rows with small weights, they are
  not rows at all. On a 6-DOF arm that leaves 2 genuinely redundant DOF, so
  the posture term here is doing real work (unlike in full 6D OSC).
- Y and Z are allowed to move freely inside a bounded corridor around their
  start values. Staying inside that corridor is a HIGH-ORDER CONTROL BARRIER
  FUNCTION constraint (4 inequality rows: y_max, y_min, z_max, z_min) on the
  same QP, not a tracking objective -- so it costs nothing at all until the
  end effector actually approaches a corridor wall.
- A small joint-space PD bias (``tau_yz_soft``) gently recenters Y/Z. It
  enters only the QP's LINEAR term, never the Hessian, so it cannot reshape
  the task.
- The manipulability CBF (``controller_core/manipulability_cbf.py``) is
  reused verbatim as a 5th row of the SAME QP.

See ``docs/status/x_task_yz_corridor_qp_2026-08-13.md`` for the full
derivation, the design forks, and the measured results.

Layout mirrors ``controller_core/x_axis_cartesian_impedance/``:

- ``parsing.py`` -- the one new field validator (plus re-exported reused ones).
- ``config.py``  -- ``XTaskYZCorridorQPConfig``.
- ``output.py``  -- ``XTaskYZCorridorQPOutput``.
- ``controller.py`` -- ``XTaskYZCorridorQPController``.
"""

from __future__ import annotations

from .config import XTaskYZCorridorQPConfig
from .controller import XTaskYZCorridorQPController
from .output import XTaskYZCorridorQPOutput

__all__ = [
    "XTaskYZCorridorQPConfig",
    "XTaskYZCorridorQPController",
    "XTaskYZCorridorQPOutput",
]
