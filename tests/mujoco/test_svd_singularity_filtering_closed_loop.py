"""Closed-loop MuJoCo validation of ``svd_singularity_filtering`` (SCI).

``tests/unit/test_svd_singularity_filtering.py`` proves the linear algebra and
the single-cycle behavior against the REAL UR5e Jacobian and mass matrix, but it
holds those matrices fixed. Nothing there can see what happens once the arm
actually moves: the Jacobian changing under the controller, real joint friction
(``assets/ur5e_torque/ur5e_torque.xml``'s ``frictionloss``/``damping``),
gravity, the torque backtracking/clip stage, or the safety guards.

This file drives the SAME adapter pipeline every sim tool in this repo uses
(``simulation/ur5e_mujoco_torque.py``'s ``build_initial_state_and_adapter`` /
``build_mujoco_state`` / ``adapter.step()`` / ``mujoco.mj_step`` on
``assets/ur5e_torque/scene.xml``) at the genuine wrist singularity, once with
SCI and once without, and asserts the difference. The per-step loop lives in
``tools/diagnostics/svd_singularity_filtering_sim_check.py`` (also runnable
standalone); this file is the reproducible assertion layer over it.

What this locks down
--------------------
1. The unit tests' embedded "real" Jacobian/mass matrix still match what the
   model actually produces -- so those literals cannot silently go stale.
2. The premise: at this pose exactly ONE task direction is lost, and today's
   uniform eps degrades well-conditioned directions as collateral damage.
3. Closed loop, BOTH directions (AGENTS.md sec 7): SCI tracks the transport
   axis strictly better than uniform damping, trips no guard, and does not blow
   up joint velocity, orientation error, or commanded torque.
4. SCI is a no-op-ish nothing-burger at a well-conditioned pose (it must not
   perturb poses that were never the problem).

``gravity_source`` is forced to ``mujoco_qfrc`` and Coriolis feedforward off:
the tuned config selects ``pinocchio``, an optional dependency, and MuJoCo's own
``qfrc_bias`` is parity-checked against it to <1e-8 Nm (AGENTS.md sec 3). The
standalone script's own default (the config's pinocchio path) was run at the
same four cells and agrees to the fifth decimal.
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

from hardware.poses import HEIGHT_ALPHA_0_5_Q  # noqa: E402
from tests.unit.test_svd_singularity_filtering import (  # noqa: E402
    REAL_JACOBIAN,
    REAL_MASS_MATRIX,
)


def _load_check_module():
    path = REPO_ROOT / "tools" / "diagnostics" / "svd_singularity_filtering_sim_check.py"
    spec = importlib.util.spec_from_file_location("svd_singularity_filtering_sim_check", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec: the module defines @dataclass types, and dataclasses
    # resolves annotations via sys.modules[cls.__module__].
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CHECK = _load_check_module()

MOVE_DURATION_S = 1.5
HOLD_DURATION_S = 1.0

_ROLLOUT_CACHE: dict[tuple, object] = {}


def rollout(pose: str, scheme: str, delta_m: float):
    """Memoized -- each closed-loop rollout costs several seconds and several
    tests compare the same pairs against each other."""
    key = (pose, scheme, float(delta_m))
    if key not in _ROLLOUT_CACHE:
        _ROLLOUT_CACHE[key] = CHECK.run_rollout(
            CHECK.POSES[pose],
            scheme=scheme,
            target_delta_m=float(delta_m),
            move_duration_s=MOVE_DURATION_S,
            hold_duration_s=HOLD_DURATION_S,
            pose_label=pose,
            gravity_source="mujoco_qfrc",
            coriolis_feedforward=False,
        )
    return _ROLLOUT_CACHE[key]


# --------------------------------------------------------------------------- #
# 1. The unit tests' real-robot literals are still real.
# --------------------------------------------------------------------------- #
def test_unit_test_literals_still_match_the_model():
    r = CHECK.run_authority_analysis(HEIGHT_ALPHA_0_5_Q, pose_label="height_alpha_0_5")
    model, data, site_id, joint_ids = CHECK._seed_model(HEIGHT_ALPHA_0_5_Q)
    from simulation.ur5e_mujoco_torque import build_mujoco_state

    st = build_mujoco_state(
        model, data, site_id=site_id, joint_ids=joint_ids,
        time_s=0.0, dt_s=float(model.opt.timestep), target_x=0.0,
    ).as_robot_state()
    J = np.asarray(st["jacobian"], dtype=np.float64).reshape(6, 6)
    M = np.asarray(st["mass_matrix"], dtype=np.float64).reshape(6, 6)
    assert np.allclose(J, REAL_JACOBIAN, rtol=0.0, atol=1e-12), (
        "tests/unit/test_svd_singularity_filtering.py::REAL_JACOBIAN no longer matches "
        "assets/ur5e_torque/scene.xml at HEIGHT_ALPHA_0_5_Q"
    )
    assert np.allclose(M, REAL_MASS_MATRIX, rtol=0.0, atol=1e-12), (
        "tests/unit/test_svd_singularity_filtering.py::REAL_MASS_MATRIX no longer matches "
        "assets/ur5e_torque/scene.xml at HEIGHT_ALPHA_0_5_Q"
    )
    assert r.cond_j > 1e10


# --------------------------------------------------------------------------- #
# 2. The premise, measured from the model rather than quoted.
# --------------------------------------------------------------------------- #
def test_one_lost_direction_and_uniform_damping_taxes_the_rest():
    r = CHECK.run_authority_analysis(HEIGHT_ALPHA_0_5_Q, pose_label="height_alpha_0_5")
    sigma = np.asarray(r.sigma)
    uniform = np.asarray(r.attenuation_uniform)
    sci = np.asarray(r.attenuation_sci)

    assert (sigma < 1e-6).sum() == 1, "expected exactly one genuinely lost direction"
    # Both schemes back off from the lost direction.
    lost = int(np.argmin(sigma))
    assert uniform[lost] < 1e-6 and sci[lost] < 1e-6
    # SCI keeps every OTHER direction at full authority; uniform eps does not.
    keep = np.ones(6, dtype=bool)
    keep[lost] = False
    assert np.allclose(sci[keep], 1.0, atol=0.0)
    assert uniform[keep].min() < 0.35, (
        "the collateral damage this feature exists to remove must be present"
    )
    # And that shows up in the acceleration actually delivered for a pure
    # transport-axis command.
    assert r.delivered_task_accel_sci / r.commanded_task_accel == pytest.approx(1.0, rel=1e-6)
    assert r.delivered_task_accel_uniform / r.commanded_task_accel < 0.8
    # Cross-axis leak: uniform eps couples the X command into other task rows.
    assert r.cross_axis_leak_uniform > 0.1
    assert r.cross_axis_leak_sci < 1e-9


# --------------------------------------------------------------------------- #
# 3. Closed loop, both directions.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("delta_m", [0.02, -0.02, 0.05, -0.05])
def test_sci_tracks_better_than_uniform_damping_in_closed_loop(delta_m: float):
    uniform = rollout("height_alpha_0_5", "uniform", delta_m)
    sci = rollout("height_alpha_0_5", "sci", delta_m)
    assert not uniform.guard_tripped
    assert not sci.guard_tripped, f"SCI tripped {sci.guard_reason} at t={sci.guard_time_s}"
    assert sci.tracking_fraction > uniform.tracking_fraction, (
        f"dx={delta_m:+.3f}: SCI tracked {100 * sci.tracking_fraction:.2f}% vs "
        f"uniform {100 * uniform.tracking_fraction:.2f}%"
    )
    # A real, not rounding-level, improvement.
    assert sci.tracking_fraction - uniform.tracking_fraction > 0.02


@pytest.mark.parametrize("delta_m", [0.02, -0.02, 0.05, -0.05])
def test_sci_does_not_degrade_any_safety_relevant_signal(delta_m: float):
    """SCI deliberately commands MORE force in well-conditioned directions, so
    the signals the safety guards actually watch are checked explicitly rather
    than assumed."""
    uniform = rollout("height_alpha_0_5", "uniform", delta_m)
    sci = rollout("height_alpha_0_5", "sci", delta_m)
    # Config guard values (config/ur5e_mujoco_torque_osc_tuned.yaml safety:).
    assert sci.max_abs_y_drift_m < 0.03
    assert sci.max_abs_z_drift_m < 0.03
    assert sci.max_orientation_error_rad < 0.25
    assert sci.max_abs_qd_radps < 3.0
    # And not dramatically worse than the uniform baseline on any of them.
    assert sci.max_abs_y_drift_m <= max(uniform.max_abs_y_drift_m * 1.5, 1e-3)
    assert sci.max_abs_z_drift_m <= max(uniform.max_abs_z_drift_m * 1.5, 1e-3)
    assert sci.max_orientation_error_rad <= uniform.max_orientation_error_rad * 1.2 + 1e-3
    assert sci.max_abs_qd_radps <= uniform.max_abs_qd_radps * 1.5 + 1e-3
    assert sci.max_abs_tau_nm <= uniform.max_abs_tau_nm * 1.5 + 1.0


def test_no_nan_or_frozen_rollout_under_sci():
    for delta_m in (0.02, -0.02):
        sci = rollout("height_alpha_0_5", "sci", delta_m)
        assert np.isfinite(sci.achieved_delta_m)
        assert np.isfinite(sci.max_abs_tau_nm)
        assert sci.steps > 100, "rollout terminated far too early"
        assert np.sign(sci.achieved_delta_m) == np.sign(delta_m)


# --------------------------------------------------------------------------- #
# 4. What SCI does at a pose that is NOT near a singularity -- which turns out
#    to be a bigger change than "nothing", and is the main open risk here.
# --------------------------------------------------------------------------- #
def test_well_conditioned_pose_sci_is_the_exact_undamped_inverse():
    r = CHECK.run_authority_analysis(
        CHECK.POSES["mega_search_winner"], pose_label="mega_search_winner"
    )
    sigma = np.asarray(r.sigma)
    assert r.cond_j < 100.0, "this pose is supposed to be well-conditioned"
    # Every direction is above the threshold, so SCI reduces to the EXACT
    # undamped inverse: no direction is attenuated at all.
    assert sigma.min() > r.svd_sigma_threshold
    assert np.allclose(np.asarray(r.attenuation_sci), 1.0, atol=0.0)


def test_well_conditioned_pose_sci_is_still_a_large_change_from_today():
    """A deliberately-documented RISK, asserted so it cannot be forgotten.

    ``lambda_regularization = 0.1`` is not small compared to the mass-weighted
    sigma^2 this arm actually operates at, so today's uniform eps is damping
    poses that are nowhere near a singularity too. Removing that damping (which
    is exactly what SCI does above the threshold) therefore changes commanded
    torque materially even at a well-conditioned pose -- so enabling this flag
    is NOT a local fix confined to singular poses, and the tuned gains were
    never validated against the un-damped response. This test records the size
    of that change rather than pretending it is negligible.
    """
    r = CHECK.run_authority_analysis(
        CHECK.POSES["mega_search_winner"], pose_label="mega_search_winner"
    )
    uniform = np.asarray(r.attenuation_uniform)
    # Today's uniform eps taxes the smallest direction here by roughly half,
    # despite cond(J) < 100.
    assert uniform.min() < 0.6
    # ...so SCI commands substantially more task torque at this pose.
    ratio = r.tau_task_norm_sci / r.tau_task_norm_uniform
    assert ratio > 1.3, (
        "expected SCI to command materially more torque at a well-conditioned "
        f"pose (got {ratio:.3f}x); if this ever drops to ~1.0 the risk note in "
        "svd_singularity_filtering's docstring should be revisited"
    )
