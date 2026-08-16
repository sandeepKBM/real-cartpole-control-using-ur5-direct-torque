"""Closed-loop MuJoCo smoke test for the orientation HOCBF row
(``XTaskYZCorridorQPConfig.orientation_cbf``) on the real UR5e model.

``tests/unit/test_x_task_yz_corridor_qp.py`` proves the row's algebra against
synthetic Jacobians/mass matrices, including the ``R_ref^T`` rotation
correction the module docstring derives. Nothing there proves the mechanism
survives a genuine simulated rollout: the Jacobian and mass matrix changing
under the controller every cycle, real joint friction/damping
(``assets/ur5e_torque/ur5e_torque.xml``), gravity compensation, the torque
clip/rate-limit stage, and ``ImpedanceSafetyMonitor``'s guards are all absent
from a synthetic fixture.

This drives the SAME adapter pipeline every sim tool in this repo uses
(``build_initial_state_and_adapter`` / ``build_mujoco_state`` /
``adapter.step()`` / ``mujoco.mj_step``), reusing
``tools/diagnostics/x_task_yz_corridor_qp_sim_check.py``'s own model-loading
helpers (``_seed_model``, ``ARM_Q0``, ``NEW_CONFIG_ENABLED``) so this test is
provably driving the same pose/model/config family the rest of that module's
test suite already validates the corridor/manipulability rows against.

``gravity_source`` is forced to ``mujoco_qfrc`` and Coriolis feedforward off,
matching every other closed-loop test in this repo (an optional dependency,
parity-checked elsewhere to <1e-8 Nm -- irrelevant to what this file checks).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_check_module():
    path = REPO_ROOT / "tools" / "diagnostics" / "x_task_yz_corridor_qp_sim_check.py"
    spec = importlib.util.spec_from_file_location("x_task_yz_corridor_qp_sim_check", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CHECK = _load_check_module()

from simulation.ur5e_mujoco_torque import (  # noqa: E402
    build_initial_state_and_adapter,
    build_mujoco_state,
    x_profile_target,
)

import mujoco  # noqa: E402

MOVE_DURATION_S = 1.5
HOLD_DURATION_S = 1.0
DELTA_M = 0.06

_CACHE: dict[bool, dict] = {}


def _rollout(orientation_cbf: bool) -> dict:
    """One closed-loop rollout, memoized -- each costs real wall-clock time
    (the QP is not free; see the timing tests in the sibling closed-loop
    file) and more than one test wants the same run."""
    if orientation_cbf in _CACHE:
        return _CACHE[orientation_cbf]

    cfg_yaml = CHECK.load_controller_config(CHECK.NEW_CONFIG_ENABLED)
    ctrl_yaml = dict(cfg_yaml["controller"])
    ctrl_yaml["orientation_cbf"] = bool(orientation_cbf)
    ctrl_yaml["orientation_cbf_max_error_rad"] = 0.20
    ctrl_yaml["orientation_cbf_alpha1"] = 10.0
    ctrl_yaml["orientation_cbf_alpha2"] = 10.0

    model, data, site_id, joint_ids = CHECK._seed_model(CHECK.ARM_Q0)
    state0, adapter = build_initial_state_and_adapter(
        model, data, site_id, joint_ids,
        controller_cfg=ctrl_yaml,
        transport_axis_index=0,
        target_x_delta=DELTA_M,
        controller_kind="x_task_yz_corridor_qp",
        force_hold_current_pose=False,
        gravity_mode="gravity_comp",
        gravity_source="mujoco_qfrc",
        coriolis_feedforward=False,
        torque_limit_scale=1.0,
    )
    x0 = float(state0.ee_pos[0])
    tau_limit = np.asarray(adapter.torque_limit_nm, dtype=np.float64)
    dt = float(model.opt.timestep)
    total_s = MOVE_DURATION_S + HOLD_DURATION_S
    n_steps = int(total_s / dt)

    max_ori_error = 0.0
    max_abs_tau = 0.0
    active_steps = 0
    feasible_violations = 0
    guard_fired = False
    guard_reason = None
    steps_completed = 0
    for step in range(n_steps):
        t = step * dt
        target_x, target_x_vel = x_profile_target("min_jerk", x0, DELTA_M, t, MOVE_DURATION_S)
        state = build_mujoco_state(
            model, data, site_id=site_id, joint_ids=joint_ids,
            time_s=t, dt_s=dt,
            target_x=target_x, target_x_vel=target_x_vel, target_x_accel=0.0,
            reference_quat=state0.ee_quat, transport_axis_index=0,
            gravity_compensation=True,
        )
        tau, diag = adapter.step(state=state)
        if not bool(diag.get("safety_ok", True)):
            guard_fired = True
            guard_reason = diag.get("safety_reason")
            break
        out = diag.get("controller_output") or {}
        if bool(out.get("orientation_cbf_active", False)):
            active_steps += 1
        if not bool(out.get("orientation_cbf_feasible", True)):
            feasible_violations += 1
        max_ori_error = max(max_ori_error, float(diag.get("orientation_error_norm", 0.0)))
        tau_arr = np.asarray(tau, dtype=np.float64).reshape(6)
        max_abs_tau = max(max_abs_tau, float(np.max(np.abs(tau_arr))))
        assert np.all(np.abs(tau_arr) <= tau_limit + 1.0e-6), "torque exceeded the hard limit"
        data.ctrl[:6] = tau_arr
        mujoco.mj_step(model, data)
        steps_completed = step + 1

    result = {
        "n_steps": n_steps,
        "steps_completed": steps_completed,
        "guard_fired": guard_fired,
        "guard_reason": guard_reason,
        "max_ori_error": max_ori_error,
        "max_abs_tau": max_abs_tau,
        "active_steps": active_steps,
        "feasible_violations": feasible_violations,
    }
    _CACHE[orientation_cbf] = result
    return result


def test_orientation_cbf_closed_loop_runs_to_completion_with_guards_on():
    res = _rollout(orientation_cbf=True)
    assert not res["guard_fired"], f"guard fired: {res['guard_reason']}"
    assert res["steps_completed"] == res["n_steps"]


def test_orientation_cbf_closed_loop_stays_feasible_throughout():
    res = _rollout(orientation_cbf=True)
    assert res["feasible_violations"] == 0


def test_orientation_cbf_closed_loop_row_actually_activates():
    """The row must genuinely engage under real dynamics at least once --
    otherwise this smoke test could not distinguish "the mechanism works" from
    "the mechanism is silently a no-op in a real rollout"."""
    res = _rollout(orientation_cbf=True)
    assert res["active_steps"] > 0


def test_orientation_cbf_off_baseline_also_runs_clean():
    """Sanity check on the comparison baseline itself: with the flag off, the
    same rollout must ALSO complete cleanly (guards on) -- otherwise a
    difference between the two runs could be a pre-existing baseline problem
    rather than anything to do with the new mechanism."""
    res = _rollout(orientation_cbf=False)
    assert not res["guard_fired"], f"guard fired: {res['guard_reason']}"
    assert res["steps_completed"] == res["n_steps"]
    assert res["active_steps"] == 0  # the mechanism is off; must never report active


def test_orientation_cbf_never_pushes_the_error_up_far_from_the_wall():
    """A coarse end-to-end sanity check on the SIGN of the whole mechanism: at
    this pose/move, orientation error never approaches
    ``orientation_cbf_max_error_rad`` (0.20 rad) in the baseline run, so if the
    barrier's sign were flipped (the exact failure mode the module docstring's
    finite-difference verification exists to rule out) it would show up here
    as the CBF-on run's max error being MEANGINGFULLY worse than the baseline,
    not merely noisy-different."""
    off = _rollout(orientation_cbf=False)
    on = _rollout(orientation_cbf=True)
    assert on["max_ori_error"] < 0.20
    # A flipped-sign barrier would actively drive error toward (or past) the
    # wall; a correct one only ever pulls it back, so it cannot make the peak
    # meaningfully WORSE than the uncontrolled baseline.
    assert on["max_ori_error"] <= off["max_ori_error"] + 0.02
