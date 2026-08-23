"""Coverage for the swing-up RL environment.

The important tests here are semantic, not structural. An RL env that returns
well-shaped arrays while silently driving the pendulum the wrong way is exactly
the failure mode this repo keeps hitting -- and a reward curve cannot
distinguish it from "has not learned yet". So the central test runs the ANALYTIC
sign rule as an oracle policy and asserts the env reports it doing positive work
~always, which is the Lyapunov property the analytic law provably has.
"""

from __future__ import annotations

import numpy as np
import pytest

from controller_core.config_provenance import ConfigPoseMismatchError
from pendulum_swingup_rl import OBS_DIM, PendulumSwingupEnv
from tools.diagnostics.pendulum_lqr_cascade import wrap_pi

REALROD = "assets/ur5e_pendulum/pendulum_attachment_realrod.xml"
DEFAULT_ASSET = "assets/ur5e_pendulum/pendulum_attachment.xml"
GOAL1_Q = [-2.3688, -2.1801, -1.8838, -0.7962, -1.5707963, 0.0206]
SINGULAR_Q = [-2.3688, -2.1801, -1.8838, -0.7962, 0.004714693, 0.0206]
FRICTION_FF = "config/ur5e_mujoco_torque_osc_tuned_friction_ff.yaml"
BALANCE_CFG = "config/ur5e_mujoco_torque_x_task_yz_corridor_qp_orientation_cbf_balance.yaml"


def make_env(**kw):
    base = dict(
        pendulum_xml=REALROD, arm_q=GOAL1_Q, config_path=FRICTION_FF,
        controller_kind="impedance", episode_s=2.0,
    )
    base.update(kw)
    return PendulumSwingupEnv(**base)


def oracle_action(env, amplitude: float) -> float:
    """The analytic maximum-energy-rate rule: sign(c0 cos(phi) thetadot)."""
    theta = float(env.data.qpos[env._pend_qpos])
    thetadot = float(env.data.qvel[env._pend_dof])
    x = env.c0 * np.cos(wrap_pi(theta - env.hanging_angle)) * thetadot
    s = np.sign(x) if abs(x) > 1e-12 else np.sign(env.c0)   # seed from rest
    return float(s * amplitude)


@pytest.fixture(scope="module")
def env():
    return make_env()


def test_spaces_and_reset(env):
    obs, info = env.reset()
    assert obs.shape == (OBS_DIM,) and obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    assert env.action_space.shape == (1,)


def test_coupling_is_measured_and_nonzero(env):
    assert abs(env.c0) > 1e-4
    assert env.omega == pytest.approx(10.8334, abs=1e-3)


def test_oracle_policy_does_positive_work_almost_always(env):
    """THE test. The analytic rule provably satisfies Edot >= 0, so if the env's
    sign handling or energy accounting were wrong this would not hold."""
    env.reset()
    info = {}
    for _ in range(env.max_steps):
        _, _, term, trunc, info = env.step([oracle_action(env, 0.35)])
        if term or trunc:
            break
    assert info["positive_work_fraction"] > 0.9, info
    assert info["e_peak_over_e_top"] > 0.0


def test_zero_action_leaves_the_pendulum_hanging(env):
    """Guards against a reward that pays out for doing nothing."""
    env.reset()
    info = {}
    for _ in range(env.max_steps):
        _, _, term, trunc, info = env.step([0.0])
        if term or trunc:
            break
    assert info["e_peak_over_e_top"] < 1e-3
    assert not info["guard_fired"]


def test_energy_reward_is_signed_by_energy_progress(env):
    """Return under the oracle must beat return under zero action; otherwise the
    reward is not actually rewarding the thing that has to accumulate."""
    def rollout(fn):
        env.reset()
        total = 0.0
        for _ in range(env.max_steps):
            _, r, term, trunc, _ = env.step([fn()])
            total += r
            if term or trunc:
                break
        return total
    assert rollout(lambda: oracle_action(env, 0.35)) > rollout(lambda: 0.0)


def test_guard_trip_terminates_with_a_penalty():
    """With the leash removed the oracle trips a guard almost immediately --
    the documented reason the leash is part of the env."""
    e = make_env(k_pos=0.0, k_vel=0.0, episode_s=3.0)
    e.reset()
    term = False
    info = {}
    for _ in range(e.max_steps):
        _, r, term, trunc, info = e.step([oracle_action(e, 0.6)])
        if term or trunc:
            break
    assert info["guard_fired"] and term, info
    assert r < 0.0


def test_leash_prevents_that_trip():
    """Same drive, leash restored: the episode must survive to truncation."""
    e = make_env(episode_s=3.0)
    e.reset()
    info = {}
    for _ in range(e.max_steps):
        _, _, term, trunc, info = e.step([oracle_action(e, 0.35)])
        if term or trunc:
            break
    assert not info["guard_fired"], info


def test_env_enforces_the_config_pose_pairing():
    """An env trained against another pose's gains fits a plant nobody runs."""
    with pytest.raises(ConfigPoseMismatchError):
        make_env(pendulum_xml=DEFAULT_ASSET, arm_q=SINGULAR_Q, config_path=BALANCE_CFG,
                 controller_kind="x_task_yz_corridor_qp")


def test_dead_drive_axis_is_refused():
    """World Z has zero authority at hanging (coupling ~ sin(phi)); pumping
    along it does nothing, so constructing such an env must fail loudly."""
    with pytest.raises(ValueError, match="no authority"):
        make_env(transport_axis_index=2)
