"""MuJoCo integration test for the diagnostic-only direct_torque dynamics
residual observer (2026-07-29) -- see
docs/status/direct_torque_residual_observer_2026-07-29.md.

Validates the residual math against real MuJoCo physics, not synthetic
data: a scaled-down version of the two checks documented in that status
file --

1. A clean move-hold rollout (no disturbance) must keep the residual small.
2. Injecting a real external force via ``data.xfrc_applied`` on
   ``wrist_3_link`` -- something neither the Pinocchio dynamics model nor
   the commanded torque has any way to predict -- must produce a clearly
   larger residual specifically during (and briefly after, due to the
   gap-windowed qdd estimator's own lag) the disturbed window, and it must
   return to baseline afterward.

This test only exercises the pure math (controller_core.dynamics_residual +
hardware.joint_accel_estimator) against a hand-rolled MuJoCo step loop; it
does not run hardware/direct_torque_transport.py's real-hardware loop (no
RTDE link available in this environment) or feed any safety trip condition.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest
import yaml

from controller_core.dynamics_residual import joint_acceleration_residual, predict_joint_acceleration
from controller_core.model_dynamics import PinocchioUR5eDynamics
from controller_core.x_axis_cartesian_impedance import (
    CartesianImpedanceConfig,
    XAxisCartesianImpedanceController,
)
from hardware.joint_accel_estimator import JointAccelEstimator
from hardware.poses import HEIGHT_ALPHA_0_5_Q
from simulation.ur5e_mujoco_torque import (
    MujocoUR5eTorqueAdapter,
    MujocoUR5eTorqueAdapterConfig,
    build_mujoco_state,
    load_model,
    x_profile_target,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config" / "ur5e_mujoco_torque_osc_tuned.yaml"


def _run_rollout(
    *,
    steps: int,
    move_duration_s: float,
    disturb_window: tuple[int, int] | None,
    disturb_force_n: tuple[float, float, float] = (0.0, 0.0, -30.0),
    gap_cycles: int = 3,
) -> list[float]:
    """Returns qdd_residual_norm per step (None entries skipped -- gap warmup)."""
    model, data, site_id, joint_ids, actuator_ids = load_model("assets/ur5e_torque/scene.xml")
    dt_s = float(model.opt.timestep)
    for idx, jid in enumerate(joint_ids):
        data.qpos[int(model.jnt_qposadr[jid])] = HEIGHT_ALPHA_0_5_Q[idx]
    mujoco.mj_forward(model, data)

    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    ctrl_cfg = cfg.get("controller", {}) or {}
    impedance_cfg = CartesianImpedanceConfig.from_controller_yaml_section(ctrl_cfg)
    controller = XAxisCartesianImpedanceController(impedance_cfg)

    # gravity_mode="gravity_comp" mirrors the real direct_torque control
    # mode's actual physics: PolyScope's directTorque() auto-adds gravity
    # compensation to whatever Python sends, but NOT Coriolis (see AGENTS.md
    # / hardware/direct_torque_transport.py's coriolis_feedforward docstring)
    # -- coriolis_feedforward stays off here to match that default.
    adapter_cfg = MujocoUR5eTorqueAdapterConfig(
        controller_kind="impedance",
        gravity_mode="gravity_comp",
        gravity_source="pinocchio",
        coriolis_feedforward=False,
        transport_axis_index=0,
    )
    adapter = MujocoUR5eTorqueAdapter(
        model=model, site_id=site_id, joint_ids=joint_ids, controller=controller, config=adapter_cfg
    )

    x0 = float(data.site_xpos[site_id][0])
    residual_dynamics = PinocchioUR5eDynamics()
    estimator = JointAccelEstimator(gap_cycles=gap_cycles, lowpass_alpha=1.0)

    body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wrist_3_link"))
    total_duration_s = steps * dt_s

    residual_norms: list[float | None] = []
    for step in range(steps):
        t_s = step * dt_s
        target_x, target_x_vel = x_profile_target(
            "min_jerk_move_hold", x0, 0.02, t_s, total_duration_s, move_duration_s=move_duration_s
        )
        state = build_mujoco_state(
            model, data, site_id=site_id, joint_ids=joint_ids, time_s=t_s, dt_s=dt_s,
            target_x=target_x, target_x_vel=target_x_vel,
        )
        if step == 0:
            controller.reset_from_state(state.as_robot_state())
            adapter.reset(state)
            estimator.reset(state.qd)

        output = controller.compute(state.as_robot_state())
        tau_controller = np.asarray(output.tau, dtype=np.float64).reshape(6)
        tau, _diag = adapter.apply_torque_components(state=state, tau_controller=tau_controller)

        # Diagnostic-only residual observer: predict qdd from known dynamics
        # + the EXACT torque about to be applied to physics (`tau`, the
        # adapter's final post-shaping value -- whatever ends up in
        # data.ctrl IS the true total physical torque here), compare to a
        # qdd estimated from consecutive qd samples. Mirrors
        # hardware/direct_torque_transport.py's per-cycle wiring order:
        # state read -> compute prediction -> feed estimator -> apply torque.
        bias = residual_dynamics.bias(state.q, state.qd)
        qdd_pred = predict_joint_acceleration(state.mass_matrix, tau, bias)
        qdd_measured = estimator.update(state.qd, dt_s)
        if qdd_measured is None:
            residual_norms.append(None)
        else:
            residual = joint_acceleration_residual(qdd_measured, qdd_pred)
            residual_norms.append(float(np.linalg.norm(residual)))

        if disturb_window is not None and disturb_window[0] <= step < disturb_window[1]:
            data.xfrc_applied[body_id, :3] = disturb_force_n
        else:
            data.xfrc_applied[body_id, :3] = 0.0

        data.ctrl[:6] = tau
        mujoco.mj_step(model, data)

    return residual_norms


@pytest.mark.mujoco
def test_residual_stays_small_on_clean_move_hold_rollout():
    # 300 steps (0.6 s) -- long enough to include the real post-move settling
    # transient (see comments below) with a genuinely settled tail.
    residual_norms = _run_rollout(steps=300, move_duration_s=0.1, disturb_window=None)
    populated = [r for r in residual_norms if r is not None]
    assert len(populated) >= 100
    populated_arr = np.asarray(populated)

    # NOTE (2026-07-30): this test's original docstring/assertions claimed the
    # residual "should stay small throughout, including during the move (not
    # just the hold)". That was only true because of a since-fixed bug: the
    # old jacobian_singular_cond_max=1.0e5 default nulled task authority at
    # this pose's wrist singularity, so the commanded move barely executed at
    # all for its first ~half (see config/ur5e_mujoco_torque_osc_tuned.yaml's
    # header comment and docs/status/disable_global_singular_scale_validation_2026-07-30.md).
    # With that freeze gone, the move actually happens, and a real mechanical
    # settling transient follows it -- measured on this exact rollout: residual
    # peaks at 2.816 rad/s^2 around step 40-50 (t=0.08-0.10s, end of the 0.1s
    # move), then decays roughly 3 orders of magnitude, not fully reaching a
    # settled ~0.003-0.004 rad/s^2 floor until around step 190-200 (t=0.38-0.40s).
    # Asserting tight smallness during/immediately after the move would be
    # asserting away correct physics, not testing anything real -- so this test
    # now makes two separate claims instead:
    #
    # NOTE (2026-07-31): assets/ur5e_torque/ur5e_torque.xml gained real joint
    # friction (frictionloss/damping -- see
    # docs/status/ur5e_sim_friction_modeling_2026-07-31.md). This residual
    # observer predicts qdd from PinocchioUR5eDynamics.bias(), which is pure
    # rigid-body dynamics (gravity + Coriolis) and has no friction term by
    # design (it mirrors what the real PolyScope/direct_torque path can
    # actually predict) -- so friction is now a second, permanent, legitimate
    # residual source on top of the settling transient above, not a bug.
    # Both bounds below were re-measured on this exact rollout with friction
    # enabled and rescaled using the *same* margin ratios the 2026-07-30 pass
    # used over its own measured numbers (~3.5x on the loose guard, ~4.9x on
    # the settled median, ~6.7x on the settled max), so the checks stay
    # equally meaningful relative to the new floor rather than being loosened
    # in an ad hoc way.
    #
    # 1. Full-rollout sanity: nothing pathological (NaN/divergence) anywhere,
    #    including during the move and its settling transient. Loose bound
    #    (measured peak with friction: 13.577 rad/s^2, ~3.7x headroom here) --
    #    this is a blowup/NaN guard, not a precision check.
    assert np.all(np.isfinite(populated_arr))
    assert float(np.max(populated_arr)) < 50.0

    # 2. Settled-tail smallness: from step 200 (t=0.4s) onward. With friction,
    #    "settled" no longer means "near the double-precision noise floor" --
    #    frictionloss/damping create a real, permanent gravity-bias-vs-real-
    #    dynamics mismatch even at rest, since the frictionless predictor has
    #    no way to see it. Measured on this exact rollout: median=1.227,
    #    max=1.318 rad/s^2 (vs the frictionless 2026-07-30 values of
    #    median=0.00204, max=0.00300 -- roughly a 600x floor increase, i.e.
    #    the size of the size3-joint frictionloss/damping budget converted
    #    through the mass matrix, not noise).
    settled = [r for r in residual_norms[200:] if r is not None]
    assert len(settled) >= 50
    settled_arr = np.asarray(settled)
    assert float(np.median(settled_arr)) < 6.0
    assert float(np.max(settled_arr)) < 9.0


@pytest.mark.mujoco
def test_residual_detects_and_recovers_from_injected_disturbance():
    # 600 steps (1.2 s) at dt=0.002s: 0.1s move, then hold; a force on
    # wrist_3_link is injected for steps [260, 290) -- t=0.52-0.58 s.
    #
    # NOTE (2026-07-30): the original windows here (baseline [80:140), i.e.
    # t=0.16-0.28s) predate the jacobian_singular_cond_max fix (see
    # config/ur5e_mujoco_torque_osc_tuned.yaml's header comment and
    # test_residual_stays_small_on_clean_move_hold_rollout's comments above)
    # and landed inside the real post-move settling transient that fix
    # legitimately introduced -- baseline_peak measured 0.282 rad/s^2 there
    # (a transient spike around step 110-120), which collapsed the detection
    # margin. All three windows below are placed after the real settling
    # transient completes (measured settled by ~step 200, t=0.4s -- see above).
    #
    # NOTE (2026-07-31): assets/ur5e_torque/ur5e_torque.xml gained real joint
    # friction (docs/status/ur5e_sim_friction_modeling_2026-07-31.md), which
    # (see test_residual_stays_small_on_clean_move_hold_rollout's 2026-07-31
    # note above) raised the settled "clean" baseline residual from ~0.003 to
    # ~1.3 rad/s^2 -- roughly 600x. The original 30 N disturbance force is no
    # longer distinguishable from that floor (measured: baseline_peak=1.318,
    # disturbed_peak=1.610, only a 1.22x bump -- friction noise, not a real
    # detection signal). Rather than accept a near-meaningless ~1.2x margin
    # as this test's new bar, the injected force was raised to 60 N -- a
    # firmer but still plausible bump/snag load (still ~1.2x the UR5e's rated
    # payload weight of ~49 N, not an implausible collision) -- which restores
    # a real, clearly-detectable margin (measured below). This is a scenario
    # change to preserve the test's actual intent (verify the residual
    # observer detects a real disturbance against the *current* noise floor),
    # not a threshold bump against the same underpowered signal.
    steps = 600
    move_duration_s = 0.1
    disturb_window = (260, 290)
    gap_cycles = 3
    residual_norms = _run_rollout(
        steps=steps,
        move_duration_s=move_duration_s,
        disturb_window=disturb_window,
        disturb_force_n=(0.0, 0.0, -60.0),
        gap_cycles=gap_cycles,
    )

    # Baseline: genuinely settled hold-phase residual (t=0.4-0.5s), well
    # after the real move-settling transient and well before the disturbance.
    baseline = [r for r in residual_norms[200:250] if r is not None]
    # Disturbed: allow the estimator's own gap-window lag (gap_cycles) after
    # the disturbance starts before requiring the signal to show it, and
    # stop at the window's end.
    disturbed = [
        r for r in residual_norms[disturb_window[0] + gap_cycles : disturb_window[1]] if r is not None
    ]
    # Recovered: a settling window well after the disturbance ends (t=0.9-1.2s,
    # i.e. ~0.32-0.62s of decay time after the disturbance stops at t=0.58s).
    # Real mechanical recovery from a genuine kinetic-energy injection is a
    # gradual decay, not an instant return to the clean baseline -- with
    # friction now also present, "recovered" settles back toward the ~1.3
    # rad/s^2 friction-noise floor from the note above, not toward zero.
    # Measured on this exact rollout (60 N, friction-enabled model): residual
    # decays from a 16.601 rad/s^2 peak during the disturbance down to 1.441
    # rad/s^2 by step 450-600, ~11.5x below the disturbed peak and back near
    # the settled floor. This window checks that clear, large decay, not a
    # return to a near-zero pristine baseline (no longer physically correct
    # to expect with real friction in the loop).
    recovered = [r for r in residual_norms[450:600] if r is not None]

    assert baseline and disturbed and recovered
    baseline_peak = float(np.max(baseline))
    disturbed_peak = float(np.max(disturbed))
    recovered_peak = float(np.max(recovered))

    # The disturbance must produce a dramatically larger residual than the
    # clean (friction-noise-floor) baseline -- measured on this exact rollout
    # (60 N, friction-enabled model): baseline_peak=1.318, disturbed_peak=
    # 16.601, ratio ~12.6x. Margin set at 10x, comfortably below the measured
    # ratio while still requiring a real, unambiguous detection (not the
    # ~1.2x the original 30 N force produced against this same friction
    # floor -- see the note above).
    assert disturbed_peak > 10.0 * baseline_peak, (
        f"baseline_peak={baseline_peak}, disturbed_peak={disturbed_peak}"
    )
    # And it must clearly decay once the disturbance is removed (measured
    # ratio on this rollout is ~11.5x, disturbed_peak/recovered_peak; margin
    # set at 8x).
    assert recovered_peak < disturbed_peak / 8.0, (
        f"disturbed_peak={disturbed_peak}, recovered_peak={recovered_peak}"
    )
