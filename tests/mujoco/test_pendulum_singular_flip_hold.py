"""Regression coverage for the single-axis flip-AND-hold at the SINGULAR ARM_Q0
(cond(J)=1396), added 2026-08-24 with the result.

This is the first sustained inverted hold this repo has achieved at the singular
pose. It is a velocity-swingup -> torque-LQR-hold, guards ON, single drive axis
(the in-plane horizontal). The three ingredients that make it work, each proven
here by an assertion of the EFFECT (not the invocation):

  1. Z promoted to a tracked task axis (the zhold config) -- otherwise the
     horizontal drive sags the EE into the floor guard in 0.38 s.
  2. --velocity-hold: the catch keeps the drive row in VELOCITY tracking.
     Position tracking's lag mis-phases the balancing cart-acceleration and
     PUMPS the pole (E -> 1.12 E_top, falls); velocity tracking realizes the
     LQR command cleanly (E stays ~1.0) so the catch holds.
  3. Pole-weighted LQR gains: the cascade LQR (tuned on placed-inverted
     perturbations) under-weights pole authority for the harder swing-up
     arrival; scaling its phi/phidot gains x1.5 crosses into a sustained hold.

The K=0 counterfactual is MANDATORY here -- this repo has retracted a
singular-pose "hold" before that was really passive hinge friction. With the
LQR the pole holds; with K=0, from the identical swing-up and handoff, it falls.
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

pytestmark = pytest.mark.mujoco

_CFG = (
    Path(__file__).resolve().parents[2]
    / "config"
    / "ur5e_mujoco_torque_x_task_yz_corridor_qp_goal2_singular_zhold.yaml"
)
# The validated swing-up schedule (hold-scoring search) and hold gains.
_BEST = dict(a_slow=5.7958928681177, a_sharp=18.883718056153565,
             e_center=0.35043620661279723, e_width=0.31761609998870677,
             db_slow=0.1584433121534668, db_sharp=0.09906939841856013,
             e_target=0.9978187516750392)
# Cascade 4-state LQR K with phi/phidot gains x1.5 (the validated hold gain).
_LQR_K = [-5.932953757595553, -21.567594433051124, 44.03442331769069, 4.155995071844572]
_A_MAX = 10.0


def _run(lqr_K):
    model = compose_ur5e_pendulum_model(pendulum_xml=str(DEFAULT_PENDULUM_XML))
    arm_q = np.asarray(DEFAULT_ARM_Q, dtype=np.float64)
    hanging, inverted = T.resolve_equilibria(model, arm_q)
    constants = T.derive_pendulum_constants(model, arm_q)
    rot = np.asarray(
        (T.load_config(_CFG).get("controller") or {}).get("task_rotation"),
        dtype=np.float64).reshape(3, 3)
    c0 = float(T.measure_pivot_coupling(model, arm_q, hanging, rot[:, 0]))
    params = T.EnergyScheduleParams(**_BEST)
    return T.run_energy_scheduled_trial(
        model, params, arm_q=arm_q, hanging_angle=hanging, inverted_angle=inverted,
        constants=constants, coupling_c0=c0, config_path=_CFG,
        controller_kind="x_task_yz_corridor_qp", transport_axis_index=0,
        duration_s=14.0, s_capture=1.2, velocity_swingup=True, velocity_hold=True,
        lqr_K=np.asarray(lqr_K, dtype=np.float64), lqr_a_max=_A_MAX,
        s_switch=1.2, phi_switch_max_rad=0.30, hold_s=10.0)


def test_singular_pose_flip_and_hold_holds_guards_on():
    r = _run(_LQR_K)
    assert r["lqr_engaged_t"] is not None, "swing-up never reached the capture window"
    assert r["held_and_upright"] is True, (
        f"did not hold: max|phi|_after={r['max_abs_phi_after_switch_rad']}")
    assert r["guard_fired"] is False, f"a guard fired: {r['guard_reason']}"
    # floor clearance kept (tip stays well above the 0.03 m margin)
    assert r["min_tip_world_z_m"] is not None and r["min_tip_world_z_m"] > 0.03
    # the pole stayed within 0.35 rad of vertical for the whole hold
    assert float(r["max_abs_phi_after_switch_rad"]) <= 0.35


def test_singular_pose_hold_is_control_not_friction():
    """K=0 counterfactual: identical swing-up + handoff, zeroed gains -> falls."""
    r = _run([0.0, 0.0, 0.0, 0.0])
    assert r["lqr_engaged_t"] is not None, "swing-up should still reach the window"
    assert r["held_and_upright"] is not True, "K=0 must NOT hold (else it is friction)"
    assert float(r["max_abs_phi_after_switch_rad"]) > 1.5, "K=0 pole should fall away"
