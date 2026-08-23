"""Coverage for the pendulum diagnostics' CLI / run-context plumbing, added
2026-08-13 alongside it.

What this exists to catch: the four pendulum diagnostic scripts previously
hardcoded their pendulum asset, their arm pose, their controller config and
(in the phase-locked case) their controller kind, at every call site including
inside the parallel differential_evolution objective. Making those selectable
is only useful if the selection actually reaches the trial -- a flag that is
parsed and then silently dropped looks identical to a working one, since the
default asset still compiles and still runs. So these tests assert the values
arrive, not merely that the flags parse.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from simulation.ur5e_pendulum_compose import (  # noqa: E402
    DEFAULT_ARM_Q,
    DEFAULT_PENDULUM_XML,
    LONGROD_ARM_Q,
    LONGROD_PENDULUM_XML,
    compose_ur5e_pendulum_model,
    derive_pendulum_constants,
)
import tools.diagnostics.pendulum_balance_torque_lqr as lqr  # noqa: E402
import tools.diagnostics.pendulum_swingup_energy_shaping as es  # noqa: E402
import tools.diagnostics.pendulum_swingup_multi_kick as mk  # noqa: E402
import tools.diagnostics.pendulum_swingup_phase_locked as pl  # noqa: E402

SWINGUP_MODULES = pytest.mark.parametrize(
    "module", [es, mk, pl], ids=["energy_shaping", "multi_kick", "phase_locked"])

ASSETS = pytest.mark.parametrize(
    ("xml", "expected_q"),
    [(DEFAULT_PENDULUM_XML, DEFAULT_ARM_Q), (LONGROD_PENDULUM_XML, LONGROD_ARM_Q)],
    ids=["default_0.12m", "longrod_0.30m"])


# --------------------------------------------------------------------------
# argument parsing -> context
# --------------------------------------------------------------------------


@SWINGUP_MODULES
def test_default_args_reproduce_the_default_context(module):
    """Running with no flags must mean exactly what it meant before the CLI
    existed: the default asset, ARM_Q0, the module's own CONFIG_PATH, and the
    impedance controller."""
    args = module.build_parser().parse_args([])
    ctx = es.context_from_args(args)
    default = module.default_context()
    assert Path(ctx.pendulum_xml) == Path(default.pendulum_xml).resolve()
    np.testing.assert_allclose(ctx.arm_q_array, es.ARM_Q0, rtol=0, atol=0)
    assert Path(ctx.config_path) == Path(es.CONFIG_PATH).resolve()
    assert ctx.controller_kind == "impedance"


@SWINGUP_MODULES
@ASSETS
def test_pose_defaults_follow_the_selected_asset(module, xml, expected_q):
    """--pendulum-xml alone must also move the arm pose. The axis/pose pairing
    is the whole reason the registry exists -- a long-rod run left at the
    default pose is silently inoperative, not an error."""
    args = module.build_parser().parse_args(["--pendulum-xml", str(xml)])
    ctx = es.context_from_args(args)
    np.testing.assert_allclose(ctx.arm_q_array, expected_q, rtol=0, atol=0)


@SWINGUP_MODULES
def test_explicit_start_q_overrides_the_registry(module):
    q = [0.1, -1.2, 1.3, -1.4, 1.5, 0.6]
    args = module.build_parser().parse_args(
        ["--pendulum-xml", str(LONGROD_PENDULUM_XML), "--start-q-rad", *map(str, q)])
    np.testing.assert_allclose(es.context_from_args(args).arm_q_array, q)


@SWINGUP_MODULES
def test_controller_kind_and_config_flags_reach_the_context(module):
    args = module.build_parser().parse_args(
        ["--controller-kind", "torque_qp", "--config", str(es.CONFIG_PATH)])
    ctx = es.context_from_args(args)
    assert ctx.controller_kind == "torque_qp"
    assert Path(ctx.config_path) == Path(es.CONFIG_PATH).resolve()


@SWINGUP_MODULES
def test_search_knobs_are_exposed(module):
    args = module.build_parser().parse_args(
        ["--maxiter", "3", "--popsize", "4", "--seed", "7", "--duration-s", "1.5"])
    assert (args.maxiter, args.popsize, args.seed, args.duration_s) == (3, 4, 7, 1.5)


def test_lqr_parser_exposes_asset_pose_and_output():
    args = lqr.build_parser().parse_args(
        ["--pendulum-xml", str(LONGROD_PENDULUM_XML), "--duration-s", "2.0"])
    assert Path(args.pendulum_xml) == LONGROD_PENDULUM_XML
    assert args.start_q_rad is None  # resolved from the registry in main()
    assert args.duration_s == 2.0
    assert args.output_json is None
    # This script runs its own LQR, not the Cartesian impedance controller, so
    # --controller-kind / --config would have nothing to select.
    assert not hasattr(args, "controller_kind")
    assert not hasattr(args, "config")


# --------------------------------------------------------------------------
# context resolution
# --------------------------------------------------------------------------


@ASSETS
def test_resolved_context_carries_the_right_constants_and_equilibria(xml, expected_q):
    ctx = es.PendulumRunContext(
        pendulum_xml=str(xml), arm_q=tuple(float(v) for v in expected_q),
        config_path=str(es.CONFIG_PATH)).resolve()
    model = compose_ur5e_pendulum_model(pendulum_xml=xml)
    expected = derive_pendulum_constants(model, expected_q)
    assert ctx.constants == expected
    hanging, inverted = es.resolve_equilibria(model, expected_q)
    assert ctx.hanging_angle == hanging
    assert ctx.inverted_angle == inverted


def test_resolve_equilibria_uses_the_pose_it_is_given():
    """The regression test for the 2026-08-13 bug: find_inverted_angle reads
    the arm pose out of the MjData it is handed, and every caller used to hand
    it a bare, all-zeros MjData -- so the swing-up target angle came from a
    completely different arm configuration than the one the trial ran at.

    Measured on the default asset, the two disagree by ~1.45 rad and very
    nearly swap 'hanging' with 'inverted'."""
    model = compose_ur5e_pendulum_model()
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "/pendulum_hinge")
    qadr = model.jnt_qposadr[jid]

    unposed = mujoco.MjData(model)  # what every caller used to pass
    hanging_zero, inverted_zero = es.find_hanging_and_inverted_angle(model, unposed, qadr)
    hanging, inverted = es.resolve_equilibria(model, DEFAULT_ARM_Q)

    assert abs(hanging - hanging_zero) > 1.0
    assert abs(inverted - inverted_zero) > 1.0
    # And the posed result is the physically right one: hanging must be the
    # LOWER-potential-energy equilibrium for a passive pendulum.
    tip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "/pendulum_tip_site")

    def tip_z(theta):
        d = mujoco.MjData(model)
        d.qpos[:6] = DEFAULT_ARM_Q
        d.qpos[qadr] = theta
        d.qvel[:] = 0.0
        mujoco.mj_forward(model, d)
        return float(d.site_xpos[tip][2])

    assert tip_z(hanging) < tip_z(inverted)


@ASSETS
def test_context_is_picklable(xml, expected_q):
    """differential_evolution's parallel workers receive the context through
    functools.partial, so it must survive pickling -- which is also why it
    holds a path rather than a compiled MjModel."""
    ctx = es.PendulumRunContext(
        pendulum_xml=str(xml), arm_q=tuple(float(v) for v in expected_q),
        config_path=str(es.CONFIG_PATH)).resolve()
    assert pickle.loads(pickle.dumps(ctx)) == ctx


# --------------------------------------------------------------------------
# the context actually reaching the trials
# --------------------------------------------------------------------------


@SWINGUP_MODULES
def test_objective_uses_the_supplied_context(module, monkeypatch):
    """A flag that is parsed and then dropped is indistinguishable from a
    working one at the top level, so assert the objective forwards its context
    into the trial rather than falling back to module defaults."""
    ctx = es.PendulumRunContext(
        pendulum_xml=str(LONGROD_PENDULUM_XML),
        arm_q=tuple(float(v) for v in LONGROD_ARM_Q),
        config_path=str(es.CONFIG_PATH), controller_kind="impedance").resolve()

    seen = {}
    trial_name = {es: "run_energy_swingup_trial", mk: "run_multi_kick_trial",
                  pl: "run_phase_locked_trial"}[module]

    def fake_trial(model, *args, **kwargs):
        seen.update(kwargs)
        return {"min_theta_dist_from_inverted_rad": 1.25, "guard_fired": False,
                "cond_j_growth_ratio": 1.0}

    monkeypatch.setattr(module, trial_name, fake_trial)
    x = {es: [1.0, 1.0, 1.0, 1.0, 0.05, 0.2], mk: [0.05, 0.2, 0.2],
         pl: [1.0, 0.05, 0.0, 0.2]}[module]
    cost = module.objective(x, ctx=ctx, duration_s=1.0)

    assert cost == pytest.approx(1.25)
    np.testing.assert_allclose(seen["arm_q"], LONGROD_ARM_Q)
    assert seen["constants"] == ctx.constants
    assert seen["hanging_angle"] == ctx.hanging_angle
    assert seen["inverted_angle"] == ctx.inverted_angle
    assert seen["controller_kind"] == "impedance"
    assert Path(seen["config_path"]) == Path(es.CONFIG_PATH).resolve()
    assert seen["duration_s"] == 1.0


def test_phase_locked_controller_kind_is_no_longer_hardcoded():
    """run_phase_locked_trial hardcoded controller_kind="impedance" at its
    build_initial_state_and_adapter call, so this one strategy could not be
    benchmarked against another controller family. If the parameter were still
    ignored, an unsupported kind would run happily instead of raising."""
    model = compose_ur5e_pendulum_model()
    hanging, inverted = es.resolve_equilibria(model, DEFAULT_ARM_Q)
    with pytest.raises(ValueError, match="Unsupported controller kind"):
        pl.run_phase_locked_trial(
            model, k_a=1.0, a_max=0.02, phase_offset_bias=0.0,
            crossing_debounce_s=0.2, duration_s=0.02,
            hanging_angle=hanging, inverted_angle=inverted,
            controller_kind="definitely_not_a_controller")


@SWINGUP_MODULES
def test_trials_default_to_the_default_asset_constants(module):
    """Backwards compatibility: omitting arm_q/constants must reproduce the
    pre-CLI default-asset behavior rather than raising."""
    model = compose_ur5e_pendulum_model()
    hanging, inverted = es.resolve_equilibria(model, DEFAULT_ARM_Q)
    common = dict(duration_s=0.05, hanging_angle=hanging, inverted_angle=inverted)
    if module is es:
        r = es.run_energy_swingup_trial(model, 1.0, 1.0, 1.0, 1.0, **common)
    elif module is mk:
        r = mk.run_multi_kick_trial(model, 0.05, 0.2, 0.2, **common)
    else:
        r = pl.run_phase_locked_trial(model, 1.0, 0.02, 0.0, 0.2, **common)
    assert r["steps_completed"] > 0


# --------------------------------------------------------------------------
# legacy constant names
# --------------------------------------------------------------------------


def test_legacy_module_constants_are_derived_not_stale():
    """M_TOTAL_KG/R_COM_M/I_PIVOT_KGM2/G/E_TOP and T_NATURAL_S survive as
    module attributes (several render scripts import them) but are now derived
    from the default model on first access, so they cannot drift from it."""
    c = derive_pendulum_constants(compose_ur5e_pendulum_model(), DEFAULT_ARM_Q)
    assert es.M_TOTAL_KG == c.m_total_kg
    assert es.R_COM_M == c.r_com_m
    assert es.I_PIVOT_KGM2 == c.i_pivot_kgm2
    assert es.G == c.g
    assert es.E_TOP == c.e_top_j
    assert pl.T_NATURAL_S == c.t_natural_s
    with pytest.raises(AttributeError):
        es.NOT_A_REAL_CONSTANT
    with pytest.raises(AttributeError):
        pl.NOT_A_REAL_CONSTANT


def test_write_output_json_round_trips_constants(tmp_path):
    ctx = es.default_context().resolve()
    out = tmp_path / "nested" / "run.json"
    es.write_output_json(out, {"constants": ctx.constants, "arm_q": list(ctx.arm_q)})
    payload = json.loads(out.read_text())
    assert payload["constants"]["mgr_nm"] == pytest.approx(ctx.constants.mgr_nm)
    assert payload["arm_q"] == pytest.approx(list(ctx.arm_q))
