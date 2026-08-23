"""mujoco-marked coverage for tools/diagnostics/pendulum_swingup_curved.py --
the CURVED (2-D) pivot-pumping swing-up at Goal 1's pose (AGENTS.md sec 0:
ARM_Q0 with wrist_2=-90 deg, the local-X-hinge realrod asset).

Covers: the pump-basis geometry (orthonormality, hinge-torque coupling sign
and cos/sin functional form, floor clearance at hanging) and a short smoke
run of the full closed-loop trial through the real controller/adapter.
Deliberately SHORT durations -- this is coverage, not a search; the real
search this file's smoke test stands in for is run manually and reported
separately, per this repo's own testing rule (AGENTS.md sec 5: new modules
ship with real tests, not just a manual smoke check, but that does not mean
the test suite re-runs a multi-second differential_evolution search)."""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import tools.diagnostics.pendulum_swingup_curved as pc  # noqa: E402
from tools.diagnostics.pendulum_toolY_common import (  # noqa: E402
    hinge_ids,
    measure_cart_coupling_nm_per_mps2,
)


@pytest.fixture(scope="module")
def ctx():
    return pc.default_context().resolve()


@pytest.fixture(scope="module")
def model(ctx):
    return ctx.build_model()


# ---------------------------------------------------------------------------
# Pump basis geometry.
# ---------------------------------------------------------------------------


def test_pump_basis_is_orthonormal(model, ctx):
    par_hat, vert_hat, n_hat = pc.compute_pump_basis(model, ctx.arm_q_array, ctx.hanging_angle)
    R = np.column_stack([par_hat, vert_hat, n_hat])
    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-9)


def test_vert_hat_points_up_at_goal1_pose(model, ctx):
    """Not a general guarantee (module docstring flags this as a
    pose-dependent fact, not an architectural one) but true, and load-
    bearing for the "bias the loop upward" leash design, at Goal 1's actual
    pose -- pin it as a regression."""
    _par_hat, vert_hat, _n_hat = pc.compute_pump_basis(model, ctx.arm_q_array, ctx.hanging_angle)
    assert vert_hat[2] == pytest.approx(1.0, abs=1e-3)


def test_hinge_torque_coupling_matches_cos_and_sin_of_phi(model, ctx):
    """Re-verifies the module docstring's physics claim directly: hinge
    generalized torque per unit pivot acceleration, swept across phi, must
    match -M*r*cos(phi) along par_hat and -M*r*sin(phi) along vert_hat, with
    EQUAL magnitude (both are exactly perpendicular to the hinge axis and to
    each other by construction)."""
    par_hat, vert_hat, _n_hat = pc.compute_pump_basis(model, ctx.arm_q_array, ctx.hanging_angle)
    phis = np.linspace(-np.pi, np.pi, 13)
    q_par = np.array([
        measure_cart_coupling_nm_per_mps2(model, ctx.arm_q_array, ctx.hanging_angle + phi, par_hat)
        for phi in phis
    ])
    q_vert = np.array([
        measure_cart_coupling_nm_per_mps2(model, ctx.arm_q_array, ctx.hanging_angle + phi, vert_hat)
        for phi in phis
    ])
    m_r = ctx.constants.mgr_nm / ctx.constants.g
    np.testing.assert_allclose(q_par, -m_r * np.cos(phis), atol=1e-6)
    np.testing.assert_allclose(q_vert, -m_r * np.sin(phis), atol=1e-6)
    # Equal peak authority (module docstring: "EQUAL, to 6 sig figs").
    assert abs(np.max(np.abs(q_par)) - np.max(np.abs(q_vert))) < 1e-9


def test_hanging_is_the_true_tip_height_minimum(model, ctx):
    """resolve_equilibria's 'hanging' must be the actual lowest-tip-height
    configuration (a passive pendulum's stable equilibrium), independent of
    any particular claimed floor-clearance figure -- this file's own module
    docstring flags a real disagreement with the task brief's "0.0634 m"
    number and re-derives its own instead; this test pins THAT down as a
    regression rather than trusting either figure blindly."""
    data = mujoco.MjData(model)
    pend_jid, _hub_bid, _site_id = hinge_ids(model)
    qadr = model.jnt_qposadr[pend_jid]
    tip_site = pc._tip_site_id(model)

    def tip_z(theta):
        data.qpos[:6] = ctx.arm_q_array
        data.qpos[qadr] = theta
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        return float(data.site_xpos[tip_site][2])

    z_hanging = tip_z(ctx.hanging_angle)
    z_inverted = tip_z(ctx.inverted_angle)
    assert z_hanging < z_inverted
    thetas = np.linspace(-np.pi, np.pi, 61) + ctx.hanging_angle
    assert z_hanging <= min(tip_z(th) for th in thetas) + 1e-6
    assert z_hanging == pytest.approx(0.2077, abs=0.01)
    assert z_hanging > 0.0  # nowhere near the floor at this pose/asset


# ---------------------------------------------------------------------------
# Closed-loop smoke test.
# ---------------------------------------------------------------------------


def test_curved_trial_runs_and_reports_required_fields(model, ctx):
    result = pc.run_curved_swingup_trial(
        model, k_e=50.0, a_max=5.0, k_pos=2.0, k_vel=1.0,
        duration_s=0.6, hanging_angle=ctx.hanging_angle, inverted_angle=ctx.inverted_angle,
        kick_amplitude_m=0.08, kick_duration_s=0.3,
        config_path=Path(ctx.config_path), arm_q=ctx.arm_q_array, constants=ctx.constants,
    )
    assert result["steps_completed"] > 0
    for key in ("min_theta_dist_from_inverted_rad", "flipped", "guard_fired",
                "min_tip_world_z_m", "floor_struck", "trustworthy",
                "max_cond_j", "par_hat", "vert_hat", "n_hat"):
        assert key in result
    # Floor tracking must be a real (finite, sane) measurement every run,
    # per the task brief's explicit requirement -- not a placeholder.
    assert np.isfinite(result["min_tip_world_z_m"])
    assert result["min_tip_world_z_m"] < 0.5  # sanity ceiling, not a tight bound


def test_disable_vertical_ablation_zeroes_vertical_excursion(model, ctx):
    """enable_vertical=False (added after a real confound was found: the
    first search's win was attributable almost entirely to the seed kick
    along the true par_hat axis, not to the vertical/curved mechanism --
    see run_curved_swingup_trial's own docstring) must force a_vert=0 for
    every post-kick step, i.e. max_abs_vert_dev_m == 0.0 exactly, reducing
    the trial to a genuine single-axis pump."""
    kwargs = dict(
        k_e=50.0, a_max=5.0, k_pos=2.0, k_vel=1.0,
        duration_s=0.6, hanging_angle=ctx.hanging_angle, inverted_angle=ctx.inverted_angle,
        kick_amplitude_m=0.08, kick_duration_s=0.3,
        config_path=Path(ctx.config_path), arm_q=ctx.arm_q_array, constants=ctx.constants,
    )
    with_vertical = pc.run_curved_swingup_trial(model, enable_vertical=True, **kwargs)
    without_vertical = pc.run_curved_swingup_trial(model, enable_vertical=False, **kwargs)
    assert without_vertical["max_abs_vert_dev_m"] == 0.0
    # And the ablation must actually differ from the full law somewhere in
    # this case (a sanity check that enable_vertical is wired through, not
    # silently ignored) -- with k_e=50 the energy term is large enough that
    # SOME vertical drive is expected once phi moves off zero.
    assert with_vertical["max_abs_vert_dev_m"] > 0.0


def test_zero_gain_trial_never_leaves_hanging():
    """k_e=0 (no energy injection at all) must never approach inverted --
    a cheap sanity check that min_theta_dist_from_inverted_rad is measuring
    something real, not trivially near zero for every input."""
    ctx = pc.default_context().resolve()
    model = ctx.build_model()
    result = pc.run_curved_swingup_trial(
        model, k_e=0.0, a_max=1.0, k_pos=1.0, k_vel=1.0,
        duration_s=0.4, hanging_angle=ctx.hanging_angle, inverted_angle=ctx.inverted_angle,
        kick_amplitude_m=0.0, kick_duration_s=0.0,
        config_path=Path(ctx.config_path), arm_q=ctx.arm_q_array, constants=ctx.constants,
    )
    assert result["min_theta_dist_from_inverted_rad"] > 1.0
    assert not result["flipped"]


def test_objective_penalizes_floor_strike_and_guard_trip(monkeypatch):
    """objective() must add its penalty when the trial reports a floor
    strike, mirroring the existing guard-trip/singularity penalties -- a
    silent no-op here would let a DE search converge on a floor-strike
    "solution" undetected (the exact class of bug pendulum_swingup_energy_
    shaping.py's own cond(J) penalty comment warns about for singularities)."""
    ctx = pc.default_context().resolve()

    def fake_trial(*args, **kwargs):
        return {"min_theta_dist_from_inverted_rad": 0.1, "guard_fired": False,
                "floor_struck": True, "cond_j_growth_ratio": 1.0}

    monkeypatch.setattr(pc, "run_curved_swingup_trial", fake_trial)
    cost = pc.objective([1.0, 1.0, 1.0, 1.0, 0.05, 0.2], ctx=ctx, duration_s=0.5)
    assert cost == pytest.approx(0.1 + 5.0)
