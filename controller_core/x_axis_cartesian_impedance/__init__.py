"""
Constrained Cartesian impedance / PD torque law for UR5 X transport.

Stabilizes X tracking while holding initial Y, Z, tool orientation, and a
rest joint posture. Pure numpy; no simulator imports.

This is a package (split from a single ~1700-line module, pure structural
refactor, zero behavior change) so it can be organized as:

- ``constants.py`` -- ``JOINT_NAME_ORDER``, ``WRIST_ORIENTATION_MASK``.
- ``parsing.py`` -- small permissive YAML-value parsing helpers used by
  ``CartesianImpedanceConfig.from_controller_yaml_section``.
- ``config.py`` -- ``CartesianImpedanceConfig``.
- ``output.py`` -- ``CartesianImpedanceOutput``.
- ``controller.py`` -- ``XAxisCartesianImpedanceController``.

Every name importable from the pre-refactor single-file module remains
importable from this package unchanged.
"""

from __future__ import annotations

from .config import CartesianImpedanceConfig
from .constants import JOINT_NAME_ORDER, WRIST_ORIENTATION_MASK
from .controller import XAxisCartesianImpedanceController
from .output import CartesianImpedanceOutput

__all__ = [
    "JOINT_NAME_ORDER",
    "WRIST_ORIENTATION_MASK",
    "CartesianImpedanceConfig",
    "CartesianImpedanceOutput",
    "XAxisCartesianImpedanceController",
]
