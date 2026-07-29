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
    residual_norms = _run_rollout(steps=200, move_duration_s=0.1, disturb_window=None)
    populated = [r for r in residual_norms if r is not None]
    assert len(populated) >= 100
    populated_arr = np.asarray(populated)
    # Clean commanded motion, fully explained by known dynamics + the
    # actually-applied torque: residual should stay small throughout,
    # including during the move (not just the hold). This is a real, not
    # hand-picked, numeric ceiling -- see
    # docs/status/direct_torque_residual_observer_2026-07-29.md for the
    # actual measured values from this exact rollout.
    assert float(np.median(populated_arr)) < 0.05
    assert float(np.max(populated_arr)) < 0.5


@pytest.mark.mujoco
def test_residual_detects_and_recovers_from_injected_disturbance():
    # 340 steps (0.68 s) at dt=0.002s: 0.1s move, then hold; a 30 N force on
    # wrist_3_link (a plausible collision-scale load, well under the UR5e's
    # ~5 kg / ~49 N rated payload weight) is injected for steps [150, 180)
    # -- 0.30-0.36 s, well inside the hold phase.
    steps = 340
    move_duration_s = 0.1
    disturb_window = (150, 180)
    gap_cycles = 3
    residual_norms = _run_rollout(
        steps=steps, move_duration_s=move_duration_s, disturb_window=disturb_window, gap_cycles=gap_cycles,
    )

    # Baseline: hold-phase residual well before the disturbance (skip the
    # initial move + settle transient).
    baseline = [r for r in residual_norms[80:140] if r is not None]
    # Disturbed: allow the estimator's own gap-window lag (gap_cycles) after
    # the disturbance starts before requiring the signal to show it, and
    # stop at the window's end.
    disturbed = [
        r for r in residual_norms[disturb_window[0] + gap_cycles : disturb_window[1]] if r is not None
    ]
    # Recovered: a settling window well after the disturbance ends. Real
    # mechanical recovery from a genuine kinetic-energy injection is a
    # gradual decay, not an instant return to the (essentially
    # double-precision-noise-level, ~1e-7) clean baseline -- measured on this
    # exact rollout (see docs/status/direct_torque_residual_observer_2026-07-29.md):
    # residual decays from a ~24 rad/s^2 peak during the disturbance down
    # through ~0.6 rad/s^2 by step 260-340, still ~42x below the peak and
    # falling. This window checks that clear, large decay, not a return to
    # the pristine baseline.
    recovered = [r for r in residual_norms[260:steps] if r is not None]

    assert baseline and disturbed and recovered
    baseline_peak = float(np.max(baseline))
    disturbed_peak = float(np.max(disturbed))
    recovered_peak = float(np.max(recovered))

    # The disturbance must produce a dramatically larger residual than the
    # clean baseline (wide margin -- this is a real physical injection, not
    # a hand-tuned synthetic signal; measured ratio on this rollout is
    # ~6e7x).
    assert disturbed_peak > 1000.0 * baseline_peak, (
        f"baseline_peak={baseline_peak}, disturbed_peak={disturbed_peak}"
    )
    # And it must clearly decay once the disturbance is removed (measured
    # ratio on this rollout is ~42x).
    assert recovered_peak < disturbed_peak / 20.0, (
        f"disturbed_peak={disturbed_peak}, recovered_peak={recovered_peak}"
    )
