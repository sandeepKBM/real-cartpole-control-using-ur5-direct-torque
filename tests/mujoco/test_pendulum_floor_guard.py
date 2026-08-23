"""Coverage for the DIRECT floor guard in pendulum_two_phase_swingup, added
2026-08-23 with it.

Why this exists. Floor penetration at the singular ARM_Q0 is SILENT: the
pendulum geoms are contype=0/conaffinity=0, so the rod passes through the floor
(world z=0) with no contact force or error. Worse, the shared
ImpedanceSafetyMonitor's per-axis max_abs_z_drift_m is NOT consulted on the
task_rotation path (controller_core/safety.py::check applies only the single
max_abs_orthogonal_drift_m there), so a measured 4.9 cm downward EE drift -- tip
to 0.0144 m, 1.4 cm off the floor -- tripped NO guard. The direct tip-world-z
guard is the fix; this test asserts it actually fires on that exact schedule,
i.e. the effect, not the invocation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from simulation.ur5e_pendulum_compose import (  # noqa: E402
    DEFAULT_ARM_Q,
    DEFAULT_PENDULUM_XML,
    compose_ur5e_pendulum_model,
)
import tools.diagnostics.pendulum_two_phase_swingup as T  # noqa: E402

# resolve_equilibria / derive_pendulum_constants are re-exported through the
# driver module (that is where main() reads them), not from the compose module.
resolve_equilibria = T.resolve_equilibria
derive_pendulum_constants = T.derive_pendulum_constants

pytestmark = pytest.mark.mujoco

_CFG = (
    Path(__file__).resolve().parents[2]
    / "config"
    / "ur5e_mujoco_torque_x_task_yz_corridor_qp_goal2_singular.yaml"
)
# The seed schedule measured to drift the EE down 4.9 cm (tip -> 0.0144 m) with
# NO guard trip before the direct floor guard existed.
_SEED = T.EnergyScheduleParams(
    a_slow=3.0, a_sharp=8.0, e_center=0.7, e_width=0.2,
    db_slow=0.3, db_sharp=0.1, e_target=1.005,
)


def _run(duration_s: float):
    model = compose_ur5e_pendulum_model(pendulum_xml=str(DEFAULT_PENDULUM_XML))
    arm_q = np.asarray(DEFAULT_ARM_Q, dtype=np.float64)
    hanging, inverted = resolve_equilibria(model, arm_q)
    constants = derive_pendulum_constants(model, arm_q)
    rot = np.asarray(
        (T.load_config(_CFG).get("controller") or {}).get("task_rotation"),
        dtype=np.float64,
    ).reshape(3, 3)
    c0 = float(T.measure_pivot_coupling(model, arm_q, hanging, rot[:, 0]))
    return T.run_energy_scheduled_trial(
        model, _SEED, arm_q=arm_q, hanging_angle=hanging, inverted_angle=inverted,
        constants=constants, coupling_c0=c0, config_path=_CFG,
        controller_kind="x_task_yz_corridor_qp", transport_axis_index=0,
        duration_s=duration_s, s_capture=1.2, velocity_swingup=True,
    )


def test_floor_guard_fires_on_downward_drift():
    r = _run(6.0)
    assert r["guard_fired"] is True
    assert "floor" in (r["guard_reason"] or "").lower()
    # It must stop AT the floor margin, never having driven the tip through it.
    # (min is recorded on the same step the guard trips, so it lands just under
    # the 0.03 m margin, but far above the 0.0144 m the un-guarded run reached.)
    assert r["min_tip_world_z_m"] is not None
    assert r["min_tip_world_z_m"] < T.FLOOR_MARGIN_M
    assert r["min_tip_world_z_m"] > 0.0  # never through the floor


def test_floor_margin_is_a_positive_clearance():
    # A sign/units regression tripwire: the margin must be a real clearance
    # above the floor (z=0), comfortably below the hanging tip height 0.063.
    assert 0.0 < T.FLOOR_MARGIN_M < 0.063
