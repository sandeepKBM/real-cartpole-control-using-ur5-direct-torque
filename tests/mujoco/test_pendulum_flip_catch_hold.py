"""End-to-end flip -> catch -> hold (Goal 1).

These are `mujoco`-marked because the thing under test IS the closed loop --
a pure-numpy unit test of the switch arithmetic would pass while the cascade
was broken, which is precisely the failure this module exists to catch.

Ordered by what breaks worst if wrong:

  1. The cascade actually flips AND holds, guards ON, on the STOCK drift
     tolerance. This is the headline result; if it regresses, Goal 1 is no
     longer met.
  2. K=0 falls over. This repo has previously retracted an "LQR works" result
     that turned out to be passive hinge friction holding the pendulum up, so
     the counterfactual is not optional -- a hold that survives K=0 is not
     evidence of control.
  3. The handoff does not teleport. One continuous rollout is the entire point
     of the module; a regression that re-zeroed the reference at the switch
     would still "pass" a naive hold check while destroying the property being
     claimed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

mujoco = pytest.importorskip("mujoco")

from simulation.ur5e_pendulum_compose import (  # noqa: E402
    REALROD_PENDULUM_XML,
    compose_ur5e_pendulum_model,
    derive_pendulum_constants,
)
from tools.diagnostics.pendulum_flip_catch_hold import (  # noqa: E402
    ARM_Q_W2NEG90,
    run_flip_catch_hold,
)
from tools.diagnostics.pendulum_swingup_energy_shaping import resolve_equilibria  # noqa: E402

# The validated Goal 1 operating point. Swing-up gains are the guard-clean flip
# recorded 2026-08-14; K/a_max are the LQR from the drift-0.06 cascade search.
# Both are copied here deliberately rather than read from a scratch JSON so the
# test pins the numbers it claims to validate.
SWINGUP = {
    "k_e": 277.74847428550635,
    "a_max": 7.143658540857182,
    "k_pos": 15.796749462795152,
    "k_vel": 8.179516700671257,
    "kick_amplitude_m": 0.1456982428474961,
    "kick_duration_s": 0.5951500761596191,
}
LQR_K = [-28.16284809807168, -100.16678005882562, 937.3280395379477, 53.96090342281156]
LQR_A_MAX = 9.992347443544016
# The STOCK config: max_abs_*_drift_m = 0.03, i.e. no guard loosening anywhere.
STOCK_CONFIG = REPO_ROOT / "config" / "ur5e_mujoco_torque_osc_tuned_friction_ff.yaml"


@pytest.fixture(scope="module")
def rig():
    model = compose_ur5e_pendulum_model(pendulum_xml=str(REALROD_PENDULUM_XML))
    arm_q = np.asarray(ARM_Q_W2NEG90, dtype=np.float64)
    hanging, inverted = resolve_equilibria(model, arm_q)
    return {
        "model": model,
        "hanging_angle": hanging,
        "inverted_angle": inverted,
        "constants": derive_pendulum_constants(model, arm_q),
    }


def _run(rig, *, K, hold_s=4.0, phi_switch_max_rad=0.25, track_history=False):
    return run_flip_catch_hold(
        rig["model"],
        swingup=SWINGUP,
        K=np.asarray(K, dtype=np.float64),
        lqr_a_max=LQR_A_MAX,
        hanging_angle=rig["hanging_angle"],
        inverted_angle=rig["inverted_angle"],
        constants=rig["constants"],
        config_path=STOCK_CONFIG,
        controller_kind="impedance",
        arm_q=ARM_Q_W2NEG90,
        duration_s=10.0 + hold_s,
        hold_s=hold_s,
        phi_switch_max_rad=phi_switch_max_rad,
        track_history=track_history,
    )


@pytest.fixture(scope="module")
def held(rig):
    return _run(rig, K=LQR_K, track_history=True)


# ============ 1. THE HEADLINE: FLIP AND HOLD, GUARDS ON ==================

@pytest.mark.slow
def test_flips_and_holds_with_guards_on_stock_drift_tolerance(held):
    assert held["switched"], "swing-up never reached the capture band"
    assert not held["guard_fired"], f"guard tripped: {held['guard_reason']}"
    assert held["flip_and_hold"], held


@pytest.mark.slow
def test_settles_close_to_vertical(held):
    """Not merely 'did not fall' -- it converges. The hold bar (0.35 rad) is
    loose enough that a slow topple could sneak past a pass/fail check."""
    assert held["final_abs_phi_rad"] < np.radians(2.0), held["final_abs_phi_deg"]


@pytest.mark.slow
def test_pole_and_arm_stay_clear_of_the_floor(held):
    """Pendulum geoms are contype=0/conaffinity=0, so a rod driven through the
    table produces no contact, no warning and no error -- it is only visible
    if the tip site's world z is tracked explicitly."""
    assert held["tip_hit_floor"] is False
    assert held["min_tip_z_m"] > 0.05
    assert held["arm_contact_steps"] == 0


@pytest.mark.slow
def test_does_not_chase_a_singularity(held):
    """A flip bought by walking the arm into a singular Jacobian is not a flip;
    the guards do not reliably catch that on their own."""
    assert held["max_cond_j"] < 50.0, held["max_cond_j"]


# ============ 2. THE COUNTERFACTUAL: K=0 MUST FALL =======================

@pytest.mark.slow
def test_zero_gain_falls_over(rig):
    """If this ever passes, the 'hold' above is passive friction, not control."""
    out = _run(rig, K=np.zeros(4))
    assert out["switched"], "counterfactual must reach the same switch point"
    assert not out["flip_and_hold"]
    # It doesn't just drift a little -- it goes all the way over.
    assert out["max_abs_phi_after_switch_rad"] > np.radians(90.0), out


# ============ 3. THE SEAM: NO TELEPORT, NO REFERENCE STEP ================

@pytest.mark.slow
def test_switch_lands_inside_the_measured_capture_band(held):
    sw = held["switch"]
    assert abs(sw["unstable_mode_s"]) <= 1.2 + 1e-9
    assert abs(sw["phi_from_inverted_rad"]) <= 0.25 + 1e-9
    # phi and thetadot of OPPOSITE sign is the whole content of the band; a
    # same-sign arrival is moving AWAY from vertical and cannot be caught.
    assert sw["phi_from_inverted_rad"] * sw["thetadot_radps"] < 0.0


@pytest.mark.slow
def test_cart_reference_is_continuous_across_the_switch(held):
    """The switch changes which law writes the acceleration -- it must not
    reset the integrated reference the inner loop is tracking. A re-zeroed
    reference shows up as a velocity step at the seam."""
    hist = held["history"]
    idx = next(i for i, r in enumerate(hist) if r["phase"] == "lqr")
    before, after = hist[idx - 1]["target_x_vel"], hist[idx]["target_x_vel"]
    # One control step at the LQR's own a_max is the largest legitimate change.
    assert abs(after - before) <= LQR_A_MAX * (1.0 / 500.0) + 1e-9


@pytest.mark.slow
def test_phase_switches_exactly_once_and_never_reverts(held):
    phases = [r["phase"] for r in held["history"]]
    assert phases[0] == "swingup" and phases[-1] == "lqr"
    assert phases.count("swingup") + phases.count("lqr") == len(phases)
    # Exactly one transition: the handoff is one-way by construction.
    assert sum(1 for a, b in zip(phases, phases[1:]) if a != b) == 1
