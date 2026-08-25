"""Regression coverage for the UNIFIED smooth-blend high-level law at the
SINGULAR ARM_Q0 (cond(J)=1396), added 2026-08-25.

Where test_pendulum_singular_flip_hold.py drives the flip->hold with a DISCRETE
energy->LQR source switch, this exercises the switch-free unified law:

    u = (1 - alpha(state)) * u_energy + alpha(state) * u_lqr

with alpha a smooth state function (tanh gates on |s| and dist-from-inverted)
that fades the energy pump out and the LQR in near the top. ONE continuous
high-level controller, no discrete phase change anywhere.

Each ingredient is proven by an assertion of the EFFECT, not the invocation:

  1. It FLIPS AND HOLDS, guards on -- same pose/asset/config/low-level QP as the
     discrete result, only the high-level law differs.
  2. It is genuinely CONTINUOUS, not a disguised switch: the LQR first owns half
     the command at alpha~=0.5 (not a hard 1.0) and the weight climbs smoothly to
     ~1 as the pole settles. A hard switch would jump straight to alpha=1.
  3. The K=0 counterfactual FALLS -- the hold is control, not hinge friction
     (this repo has retracted a passive-friction "hold" at this pose before).
  4. The blend REQUIRES velocity_hold: a switch-free law has no seam at which to
     re-sync a position-tracked catch, so it refuses rather than silently run the
     LQR against a different plant than its K was solved for.
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
# The validated swing-up schedule (hold-scoring search) and hold gains -- shared
# with the discrete-switch test; only the high-level composition differs here.
_BEST = dict(a_slow=5.7958928681177, a_sharp=18.883718056153565,
             e_center=0.35043620661279723, e_width=0.31761609998870677,
             db_slow=0.1584433121534668, db_sharp=0.09906939841856013,
             e_target=0.9978187516750392)
_LQR_K = [-5.932953757595553, -21.567594433051124, 44.03442331769069, 4.155995071844572]
_A_MAX = 10.0
# Blend-weight shape tuned to this pose (dist gate at the phi_switch the discrete
# run uses); alpha reaches ~1 shortly after the pole enters the capture band.
_BLEND = dict(blend_lqr=True, blend_s_center=1.2, blend_s_width=0.4,
              blend_dist_center=0.30, blend_dist_width=0.12, blend_dist_cutoff=0.6)


def _run(lqr_K, *, velocity_hold=True, **over):
    model = compose_ur5e_pendulum_model(pendulum_xml=str(DEFAULT_PENDULUM_XML))
    arm_q = np.asarray(DEFAULT_ARM_Q, dtype=np.float64)
    hanging, inverted = T.resolve_equilibria(model, arm_q)
    constants = T.derive_pendulum_constants(model, arm_q)
    rot = np.asarray(
        (T.load_config(_CFG).get("controller") or {}).get("task_rotation"),
        dtype=np.float64).reshape(3, 3)
    c0 = float(T.measure_pivot_coupling(model, arm_q, hanging, rot[:, 0]))
    params = T.EnergyScheduleParams(**_BEST)
    kw = dict(duration_s=14.0, s_capture=1.2, velocity_swingup=True,
              velocity_hold=velocity_hold,
              lqr_K=np.asarray(lqr_K, dtype=np.float64), lqr_a_max=_A_MAX,
              s_switch=1.2, phi_switch_max_rad=0.30, hold_s=10.0)
    kw.update(_BLEND)
    kw.update(over)
    return T.run_energy_scheduled_trial(
        model, params, arm_q=arm_q, hanging_angle=hanging, inverted_angle=inverted,
        constants=constants, coupling_c0=c0, config_path=_CFG,
        controller_kind="x_task_yz_corridor_qp", transport_axis_index=0, **kw)


def test_blend_flips_and_holds_guards_on():
    r = _run(_LQR_K)
    assert r["blend_lqr"] is True, "run did not use the unified blend law"
    assert r["lqr_engaged_t"] is not None, "swing-up never reached the capture window"
    assert r["held_and_upright"] is True, (
        f"did not hold: max|phi|_after={r['max_abs_phi_after_switch_rad']}")
    assert r["guard_fired"] is False, f"a guard fired: {r['guard_reason']}"
    assert r["min_tip_world_z_m"] is not None and r["min_tip_world_z_m"] > 0.03
    assert float(r["max_abs_phi_after_switch_rad"]) <= 0.35


def test_blend_is_continuous_not_a_disguised_switch():
    """The LQR fades in: it owns ~half the command when engagement is first
    recorded (alpha~=0.5), and the weight climbs smoothly to ~1 -- not the hard
    0->1 jump a discrete switch would produce."""
    r = _run(_LQR_K)
    engage_alpha = float(r["lqr_switch_state"]["alpha"])
    assert 0.40 <= engage_alpha <= 0.70, (
        f"engagement alpha {engage_alpha} looks like a hard switch, not a blend")
    assert float(r["max_alpha"]) > 0.9, (
        f"LQR never took over the cart (max_alpha={r['max_alpha']})")


def test_blend_hold_is_control_not_friction():
    """K=0 counterfactual: identical swing-up + blend handoff, zeroed gains -> falls."""
    r = _run([0.0, 0.0, 0.0, 0.0])
    assert r["lqr_engaged_t"] is not None, "swing-up should still reach the window"
    assert r["held_and_upright"] is not True, "K=0 must NOT hold (else it is friction)"
    assert float(r["max_abs_phi_after_switch_rad"]) > 1.5, "K=0 pole should fall away"


def test_blend_requires_velocity_hold():
    """A switch-free blend has no seam to re-sync a position catch, so it refuses
    velocity_hold=False rather than run the LQR against the wrong plant."""
    with pytest.raises(RuntimeError, match="velocity_hold"):
        _run(_LQR_K, velocity_hold=False)
