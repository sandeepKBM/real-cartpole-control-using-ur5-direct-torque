"""Convergence-gated cold fallback for the warm-started corridor QP (2026-08-26).

WHY THIS TEST EXISTS -- and why the existing parity test
(``test_corridor_qp_warm_start_parity.py``) did NOT catch the bug it guards: that
test seeds every warm solve from the CONVERGED reference, i.e. the ideal
slowly-varying regime. The real controller instead seeds each cycle from the
PREVIOUS cycle's own (possibly under-converged) warm solution. At a fast transient
the QP optimum JUMPS, the stale seed lands FARTHER from it than a cold analytic
start, and the reduced warm budget cannot recover -- a multi-Nm tracking-torque
error the converged-seed parity test never produces.

Two tests, both on the REAL corridor QP at ARM_Q0 over a -0.06 m move:

* ``test_headline_transient_defect_and_safeguard`` -- the clean, deterministic
  instance-level demonstration and the strict gate: a real EASY state (cold@80 is
  essentially exact) seeded from a real NEAR-WALL state's solution (an optimum
  jump). WITHOUT the safeguard warm is multi-Nm off and raising max_iters does not
  fix it (the seed dominates); WITH the safeguard the box-projected residual fires
  the gate and the cycle is redone cold, so the error is ~= cold. This FAILS on the
  pre-safeguard path (fallback_tol=None) and PASSES with the gate.

* ``test_closed_loop_gate_removes_large_regressions`` -- production seeding, every
  visited state of the real warm loop: the gate removes every LARGE (>1 Nm) warm
  regression vs cold@80 that the ungated path exhibits, and where the ungated warm
  blows up the gated output equals cold exactly (it fell back). (A bit-tight
  "<= cold at every state" is not asserted because at hard near-wall states cold@80
  itself is ~1-2 Nm under-resolved by the shipped dual budget -- a pre-existing
  property, unrelated to warm-start -- so warm scatters within cold's own error
  band there; the gate's job is to kill the multi-Nm regressions, which it does.)

Marked slow: drives a full move and computes a strong per-state reference.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mujoco  # noqa: E402

import controller_core.constrained_box_qp as cbq  # noqa: E402
import controller_core.x_task_yz_corridor_qp.controller as ctrl_mod  # noqa: E402
from simulation.ur5e_mujoco_torque import (  # noqa: E402
    build_initial_state_and_adapter, build_mujoco_state, load_model, x_profile_target,
)

SCENE_XML = REPO_ROOT / "assets" / "ur5e_torque" / "scene.xml"
HW_CONFIG = REPO_ROOT / "config" / "ur5e_direct_torque_x_task_yz_corridor_qp.yaml"
ARM_Q0 = np.array([-2.3688, -2.1801, -1.8838, -0.7962, 0.004714693, 0.0206])

DX = -0.06
DUR = 1.6            # includes the move phase and the ~step-610 transient
MOVE_DUR = 1.5
GATE_TOL = 1.0e-3    # the shipped qp_warm_fallback_tol
# Strong reference: dual budget richer than the shipped 4/10, 6000 inner iters.
_REF = dict(dual_sweeps=12, dual_root_iters=60, max_iters=6000, tol=1e-13)
_COLD = dict(dual_sweeps=4, dual_root_iters=10, max_iters=80, tol=1e-8)
_WARM = dict(dual_sweeps=4, dual_root_iters=10, max_iters=20, tol=1e-8)


def _seed_model(q):
    model, data, site_id, joint_ids, _ = load_model(SCENE_XML)
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    for idx, jid in enumerate(joint_ids):
        data.qpos[int(model.jnt_qposadr[jid])] = float(q[idx])
    mujoco.mj_forward(model, data)
    return model, data, site_id, joint_ids


def _solve(inst, seed=None, **kw):
    xw = lw = None
    if seed is not None:
        xw, lw = seed
    return cbq.solve_constrained_box_qp(
        inst["h"], inst["f"], inst["lo"], inst["hi"], inst["a"], inst["b"],
        x_warm=xw, lam_warm=lw, **kw)


def _box_residual(inst, x, lam):
    hs = 0.5 * (inst["h"] + inst["h"].T) + 1e-8 * np.eye(inst["h"].shape[0])
    f_sh = inst["f"] + inst["a"].T @ lam
    step = 1.0 / max(float(np.max(np.abs(hs))), 1.0)
    return float(np.max(np.abs(np.clip(x - step * (hs @ x + f_sh), inst["lo"], inst["hi"]) - x)))


def _drive(qp_warm_start, fallback_tol, dur=DUR):
    """Drive the real corridor-QP controller over the move; spy on every solver
    call. Returns (visited, seeds): visited[k] = the (h,f,lo,hi,a,b,x) handed to /
    returned by the solver at visited (constraint-active) states, seeds[k] = the
    (x_warm, lam_warm) the controller passed that cycle (None on cold cycles)."""
    cbq.numba_warmup()
    with open(HW_CONFIG) as fh:
        ctrl_cfg = dict(yaml.safe_load(fh)["controller"])
    assert bool(ctrl_cfg.get("qp_warm_start")), "HW config must ship warm start on"
    ctrl_cfg["qp_warm_start"] = qp_warm_start
    ctrl_cfg["qp_warm_fallback_tol"] = fallback_tol

    model, data, site_id, joint_ids = _seed_model(ARM_Q0)
    state0, adapter = build_initial_state_and_adapter(
        model, data, site_id, joint_ids, controller_cfg=ctrl_cfg,
        transport_axis_index=0, target_x_delta=DX,
        controller_kind="x_task_yz_corridor_qp", force_hold_current_pose=False,
        gravity_mode="gravity_comp", gravity_source="mujoco_qfrc",
        coriolis_feedforward=False, torque_limit_scale=1.0)
    start_ee = np.asarray(state0.ee_pos).copy()
    x0 = float(start_ee[0])
    dt = float(model.opt.timestep)
    steps = int(np.ceil(dur / dt))
    grav = mujoco.MjData(model)

    orig = cbq.solve_constrained_box_qp
    box = {}

    def spy(h, f, lo, hi, a=None, b=None, **kw):
        r = orig(h, f, lo, hi, a, b, **kw)
        box["inst"] = dict(h=np.array(h), f=np.array(f), lo=np.array(lo), hi=np.array(hi),
                           a=None if a is None else np.array(a),
                           b=None if b is None else np.array(b), x=np.array(r[0]))
        box["seed"] = (kw.get("x_warm"), kw.get("lam_warm"))
        return r

    ctrl_mod.solve_constrained_box_qp = spy
    visited, seeds = [], []
    try:
        for k in range(steps):
            t_s = float(data.time)
            tn, tv = x_profile_target("min_jerk_move_hold", x0, DX, t_s, dur, move_duration_s=MOVE_DUR)
            tep = start_ee.copy(); tep[0] = tn
            tev = np.zeros(3); tev[0] = tv
            state = build_mujoco_state(
                model, data, site_id=site_id, joint_ids=joint_ids, time_s=t_s, dt_s=dt,
                target_x=tn, target_x_vel=tv, target_x_accel=0.0,
                target_axis=tn, target_axis_vel=tv, target_ee_pos=tep, target_ee_vel=tev,
                reference_quat=state0.reference_quat, hold_current_pose=False,
                transport_axis_index=0, gravity_compensation=True, gravity_scratch_data=grav)
            tau, diag = adapter.step(state=state)
            inst = box["inst"]
            if inst["a"] is not None:
                inst["step"] = k
                visited.append(inst)
                seeds.append(box["seed"])
            if not bool(diag.get("safety_ok", True)):
                break
            data.ctrl[:6] = np.asarray(tau).reshape(6)
            mujoco.mj_step(model, data)
    finally:
        ctrl_mod.solve_constrained_box_qp = orig
    return visited, seeds


@pytest.fixture(scope="module")
def gated_run():
    cbq.numba_warmup()
    return _drive(qp_warm_start=True, fallback_tol=GATE_TOL)


def test_headline_transient_defect_and_safeguard(gated_run):
    """Instance-level, deterministic. A real EASY state seeded from a real NEAR-WALL
    state's solution: without the gate warm is multi-Nm off (and raising max_iters
    does not fix it); with the gate the residual fires and the result is ~= cold."""
    visited, _ = gated_run
    orig = cbq.solve_constrained_box_qp
    # cold error at every visited state; pick the easiest (cold ~ exact) and the
    # richest-dual (near-wall) instance.
    cold_err, lam_mag = {}, {}
    refs = {}
    for i, inst in enumerate(visited):
        xr, lr, _ = _solve(inst, **_REF)
        xc, _, _ = _solve(inst, **_COLD)
        refs[i] = (xr, lr)
        cold_err[i] = float(np.max(np.abs(xc - xr)))
        lam_mag[i] = float(np.max(np.abs(lr)))
    E = min(cold_err, key=lambda i: cold_err[i])
    W = max(lam_mag, key=lambda i: lam_mag[i])
    assert cold_err[E] < 1e-3, f"need an easy state; easiest cold err {cold_err[E]:.2e}"
    assert E != W

    instE, (xrefE, _) = visited[E], refs[E]
    seedW = refs[W]  # near-wall converged (x, lam) -- a far, wrong-active-set seed

    err_cold = cold_err[E]
    x20, l20, _ = _solve(instE, seed=seedW, **_WARM)
    err_w20 = float(np.max(np.abs(x20 - xrefE)))
    x80, _, _ = _solve(instE, seed=seedW, dual_sweeps=4, dual_root_iters=10, max_iters=80, tol=1e-8)
    err_w80 = float(np.max(np.abs(x80 - xrefE)))
    res = _box_residual(instE, x20, l20)
    xg, _, _ = _solve(instE, seed=seedW, fallback_tol=GATE_TOL, **_WARM)
    err_gated = float(np.max(np.abs(xg - xrefE)))

    # the defect: warm is large, and raising iters does not fix it (seed dominates)
    assert err_w20 > 1.0, f"expected a multi-Nm warm defect, got {err_w20:.3e}"
    assert err_w80 > 0.3, f"raising max_iters should not fix it, got {err_w80:.3e}"
    # the gate fires
    assert res > GATE_TOL, f"warm residual {res:.3e} should exceed gate {GATE_TOL:.1e}"
    # the safeguard: gated ~= cold, i.e. the defect is eliminated
    assert err_gated <= err_cold + 1e-6, f"gated {err_gated:.3e} not ~= cold {err_cold:.3e}"
    # and WITHOUT the gate the same solve fails an "<= cold" assertion by a wide margin
    assert err_w20 > err_cold + 1.0


def test_closed_loop_gate_removes_large_regressions(gated_run):
    """Production seeding, every visited state of the real warm loop. The gate
    removes every LARGE (>1 Nm) warm-vs-cold regression, and where ungated warm
    blows up the gated output equals cold (it fell back). Also reports the fallback
    fraction (kept well below 'always cold', so the warm speedup survives)."""
    visited, seeds = gated_run
    orig = cbq.solve_constrained_box_qp
    fired = 0
    worst_gated_excess = 0.0
    worst_ungated_excess = 0.0
    n_ungated_large = 0
    fellback_matches_cold = True
    for inst, (xw, lw) in zip(visited, seeds):
        xref, _, _ = _solve(inst, **_REF)
        xcold, _, _ = _solve(inst, **_COLD)
        ec = float(np.max(np.abs(xcold - xref)))
        eg = float(np.max(np.abs(inst["x"] - xref)))
        worst_gated_excess = max(worst_gated_excess, eg - ec)
        if xw is not None and lw is not None:
            xu, lu, _ = orig(inst["h"], inst["f"], inst["lo"], inst["hi"], inst["a"], inst["b"],
                             x_warm=xw, lam_warm=lw, **_WARM)
            eu = float(np.max(np.abs(xu - xref)))
            worst_ungated_excess = max(worst_ungated_excess, eu - ec)
            res = _box_residual(inst, xu, lu)
            if res > GATE_TOL:
                fired += 1
                n_ungated_large += int(eu - ec > 1.0)
                # when the gate fired, the controller's output must be the cold solve
                if not np.allclose(inst["x"], xcold, atol=1e-9, rtol=0.0):
                    fellback_matches_cold = False

    active = len(visited)
    # 1) the gate introduces NO large (>1 Nm) regression vs cold anywhere
    assert worst_gated_excess < 1.0, f"gated worst excess over cold {worst_gated_excess:.3e} Nm"
    # 2) the ungated warm path DID have large regressions the gate had to catch
    assert worst_ungated_excess > 1.0, (
        f"expected a large ungated warm regression, got {worst_ungated_excess:.3e}")
    assert n_ungated_large > 0
    # 3) a fired cycle produces exactly the cold solution
    assert fellback_matches_cold
    # 4) the fallback is not firing on (almost) every cycle -> warm speedup survives
    assert 0 < fired < active, f"fallback fired {fired}/{active}"
    assert fired / active < 0.6, f"fallback fraction {fired/active:.2f} too high"
