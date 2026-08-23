"""Gymnasium env where the policy searches the ENERGY SEQUENCE of a swing-up.

WHAT IS BEING LEARNED, AND WHY IT IS NOT THE THING THAT FAILED BEFORE. This
repo's RL record is 0/20, 0/20, 1/20 across three reward redesigns and ~4.4M
steps against a fixed-gain baseline at 100% -- but that was ``rl_gain_scheduling``,
whose action is a vector of GAIN MULTIPLIERS and whose environment contains no
pendulum at all (verified: zero files in that package mention one). Choosing
static gains is a bandit, and a bandit is strictly worse than the DE/optuna
backends this repo already has. Nothing here reuses it.

What this env asks instead is a genuine sequential-decision problem: given the
pendulum's measured state and energy, how hard and in which direction should the
pivot accelerate RIGHT NOW, so that the pole arrives at the top inside the LQR's
capture band? The hand-written schedule (pendulum_two_phase_swingup.py) answers
it with a tanh of energy fraction; that file is the BASELINE this env must beat,
not a component of it.

THE ACTION IS THE DRIVE ITSELF, NOT A BLEND -- AND THAT IS A BUG FIX, NOT A
PREFERENCE. The first draft of the hand-written law made the drive proportional
to tanh(c0*cos(phi)*thetadot/db), which is IDENTICALLY ZERO at the hanging rest
state. A policy whose action only SCALES such a term could never leave rest: the
gradient is zero everywhere at the start state, and the run would silently be a
no-op (measured: max_blend 0, E_peak/E_top 0, pole never moved, no error
raised). Making the action the commanded acceleration directly means the policy
can bootstrap itself and is free to discover the seed kick the analytic law has
to be handed.

LEGIBILITY IS MANDATORY, NOT OPTIONAL. Energy shaping carries a Lyapunov
guarantee (Edot >= 0 by construction); a learned policy does not, so "pumping
backwards" and "has not learned yet" produce the same falling reward curve. Every
episode therefore reports ``positive_work_fraction`` from EnergyMonitor -- the
fraction of steps where the drive did POSITIVE work on the hinge, ~1.0 for the
analytic law and <0.5 for a drive that removes energy on balance. This is not
decoration: within an hour of writing the hand-tuned law, that number (0.3355)
was what revealed its inherited leash gains were fighting its own pump. Judge a
policy on it before believing any reward curve.

REWARD. Dense energy progress (the thing that must accumulate), a control cost,
a terminal bonus for arriving inside the capture band |s| <= s_capture with
s = thetadot + omega*phi measured from INVERTED, and a hard penalty for a guard
trip. The capture band -- not ``min_theta_dist_from_inverted`` -- is the target
because that metric rewards a fast fly-through, which is the one arrival a catch
cannot use.

GUARDS STAY ON. A guard trip ends the episode with a penalty. A policy that only
works with guards disabled is a negative result, per this repo's standing rule,
and there is deliberately no flag here to turn them off.

POLICY RATE. The inner loop runs at 500 Hz; the policy acts at 500/decimation Hz
(default 50 Hz) with its action held in between. A 10 s episode is then 500
decisions rather than 5000, which is both a saner horizon and closer to the rate
a real high-level controller would run at.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

REPO_ROOT = Path(__file__).resolve().parents[1]

from controller_core.config_provenance import check_config_pose  # noqa: E402
from simulation.ur5e_pendulum_compose import (  # noqa: E402
    compose_ur5e_pendulum_model,
    derive_pendulum_constants,
)
from simulation.ur5e_mujoco_torque import (  # noqa: E402
    build_initial_state_and_adapter,
    build_mujoco_state,
)
from tools.diagnostics.pendulum_energy_monitor import EnergyMonitor  # noqa: E402
from tools.diagnostics.pendulum_lqr_cascade import wrap_pi  # noqa: E402
from tools.diagnostics.pendulum_swingup_energy_shaping import (  # noqa: E402
    CONTROL_DT,
    RATE_HZ,
    load_config,
    measure_pivot_coupling,
    resolve_equilibria,
)

JOINT_NAMES = (
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
)
V_MAX_MPS = 1.00
OBS_DIM = 8


class PendulumSwingupEnv(gym.Env):
    """action: commanded pivot acceleration, normalised to [-1, 1].

    obs: [sin(phi_hang), cos(phi_hang), thetadot_n, E/E_top,
          x_dev_n, xdot_n, sin(phi_inv), cos(phi_inv)]
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        pendulum_xml: str,
        arm_q,
        config_path: str | Path,
        controller_kind: str = "impedance",
        transport_axis_index: int = 0,
        a_max_mps2: float = 12.0,
        # LEASH, inside the env by design. Without it the FIRST energetic action
        # sequence walks the TCP straight out of the drift guard: the analytic
        # sign-rule oracle, which is optimal for energy injection, trips
        # |Y-Y0| > 0.03 m after 0.28 s. Every episode then ends in a fraction of
        # a second with a large negative return and the policy never observes a
        # swing at all -- the same never-move collapse shape this repo already
        # documents for rl_gain_scheduling. Keeping the leash as fixed inner
        # structure means the policy shapes the PUMP (the actual sequential
        # decision) instead of having to rediscover drift regulation first.
        # Set both to 0.0 to hand the policy full authority and see that failure.
        k_pos: float = 15.797,
        k_vel: float = 8.180,
        episode_s: float = 10.0,
        # 20 Hz decisions, not 50. At 50 Hz the policy gets ~29 decisions per
        # pendulum cycle (1.724 Hz) and has to discover phase coherence across
        # all of them before any reward appears. Measured outcome at 50 kHz-steps
        # of PPO: a THRASHING policy -- action std 0.79 spanning +-1, i.e. full
        # scale commands -- that moved the pendulum not at all
        # (E_peak/E_top = 2.7e-5) and never tripped a guard, so the return was
        # flat zero with no gradient. positive_work_fraction = 0.021 showed the
        # drive was incoherent rather than merely weak. The arm's own
        # closed-loop bandwidth is ~0.5 s, so faster commands are not realisable
        # on the plant anyway.
        decimation: int = 25,
        s_capture: float = 1.2,
        phi_capture_rad: float = 0.45,
        w_energy: float = 20.0,
        w_control: float = 0.002,
        r_capture: float = 200.0,
        r_guard: float = 100.0,
        allow_pose_mismatch: bool = False,
        seed: int | None = None,
    ):
        super().__init__()
        self.pendulum_xml = str(pendulum_xml)
        self.arm_q = np.asarray(arm_q, dtype=np.float64).reshape(6)
        self.config_path = Path(config_path)
        self.controller_kind = str(controller_kind)
        self.axis = int(transport_axis_index)
        self.a_max = float(a_max_mps2)
        self.k_pos = float(k_pos)
        self.k_vel = float(k_vel)
        self.decimation = int(decimation)
        self.max_steps = int(episode_s * RATE_HZ / self.decimation)
        self.s_capture = float(s_capture)
        self.phi_capture = float(phi_capture_rad)
        self.w_energy = float(w_energy)
        self.w_control = float(w_control)
        self.r_capture = float(r_capture)
        self.r_guard = float(r_guard)

        self.config = load_config(self.config_path)
        # Same machine-checked config <-> pose pairing every other entrypoint
        # uses. An env that trains against the wrong pose's gains would produce
        # a policy fitted to a plant nobody runs.
        self.provenance = check_config_pose(
            self.config, self.arm_q, self.pendulum_xml,
            config_name=self.config_path.name,
            allow_mismatch=bool(allow_pose_mismatch),
        )

        self.model = compose_ur5e_pendulum_model(pendulum_xml=self.pendulum_xml)
        self.hanging_angle, self.inverted_angle = resolve_equilibria(self.model, self.arm_q)
        self.constants = derive_pendulum_constants(self.model, self.arm_q)
        self.omega = float(self.constants.omega_natural_radps)
        self.e_top = float(self.constants.e_top_j)

        drive_axis = np.zeros(3)
        drive_axis[self.axis] = 1.0
        # Measured, never assumed: this is the coefficient the drive's sign
        # depends on, and it is a property of (pose, asset, axis).
        self.c0 = float(measure_pivot_coupling(
            self.model, self.arm_q, self.hanging_angle, drive_axis))
        if abs(self.c0) < 1e-5:
            raise ValueError(
                f"drive axis {self.axis} has no authority over the hinge at this "
                f"pose (|c0| = {abs(self.c0):.2e}); pumping along it does nothing"
            )

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self._np_random_seed = seed
        self._site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
        self._joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES
        ]
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "/pendulum_hinge")
        self._pend_qpos = self.model.jnt_qposadr[jid]
        self._pend_dof = self.model.jnt_dofadr[jid]

    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:6] = self.arm_q
        self.data.qpos[self._pend_qpos] = self.hanging_angle
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self.state0, self.adapter = build_initial_state_and_adapter(
            self.model, self.data, self._site_id, self._joint_ids,
            controller_cfg=self.config["controller"],
            transport_axis_index=self.axis,
            target_x_delta=0.0,
            controller_kind=self.controller_kind,
            force_hold_current_pose=False,
            gravity_mode=self.config["mujoco"].get("gravity_mode", "gravity_comp"),
            gravity_source=self.config["mujoco"].get("gravity_source", "pinocchio"),
            coriolis_feedforward=bool(self.config["mujoco"].get("coriolis_feedforward", True)),
            torque_limit_scale=1.0,
        )
        self.x_ref = float(self.state0.ee_pos[self.axis])
        self.target_x = self.x_ref
        self.target_x_vel = 0.0
        self._t = 0.0
        self._steps = 0
        self._guard_reason = None
        self._best_abs_s = np.inf
        self._monitor = EnergyMonitor(
            i_pivot_kgm2=float(self.constants.i_pivot_kgm2),
            mgr_nm=float(self.constants.mgr_nm), e_top_j=self.e_top,
            coupling_c0=self.c0,
            hinge_damping=float(self.model.dof_damping[self._pend_dof]),
        )
        self._prev_e_frac = self._energy_frac()
        self._e_frac_peak = self._prev_e_frac
        return self._obs(), {}

    # ------------------------------------------------------------------
    def _pend(self):
        theta = float(self.data.qpos[self._pend_qpos])
        thetadot = float(self.data.qvel[self._pend_dof])
        return theta, thetadot

    def _energy_frac(self) -> float:
        theta, thetadot = self._pend()
        phi_hang = wrap_pi(theta - self.hanging_angle)
        return self._monitor.energy(thetadot, phi_hang) / self.e_top if self.e_top else 0.0

    def _obs(self) -> np.ndarray:
        theta, thetadot = self._pend()
        phi_hang = wrap_pi(theta - self.hanging_angle)
        phi_inv = wrap_pi(theta - self.inverted_angle)
        ee = np.asarray(self.state0.ee_pos, dtype=np.float64)
        x_dev = (self.target_x - self.x_ref)
        return np.array([
            np.sin(phi_hang), np.cos(phi_hang),
            thetadot / max(self.omega, 1e-9),
            self._energy_frac(),
            x_dev / 0.25,
            self.target_x_vel / max(V_MAX_MPS, 1e-9),
            np.sin(phi_inv), np.cos(phi_inv),
        ], dtype=np.float32)

    # ------------------------------------------------------------------
    def step(self, action):
        a_pump = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0)) * self.a_max
        guard_fired = False

        for _ in range(self.decimation):
            # Leash recomputed every INNER step, not once per policy step: it is
            # a regulator on the reference, and holding it fixed across the
            # decimation window would let the reference run away between
            # decisions.
            a_cmd = (a_pump
                     - self.k_pos * (self.target_x - self.x_ref)
                     - self.k_vel * self.target_x_vel)
            self.target_x_vel = float(np.clip(
                self.target_x_vel + a_cmd * CONTROL_DT, -V_MAX_MPS, V_MAX_MPS))
            self.target_x = self.target_x + self.target_x_vel * CONTROL_DT

            state = build_mujoco_state(
                self.model, self.data, site_id=self._site_id, joint_ids=self._joint_ids,
                time_s=self._t, dt_s=CONTROL_DT,
                target_x=self.target_x, target_x_vel=self.target_x_vel,
                target_x_accel=a_cmd, reference_quat=self.state0.ee_quat,
                transport_axis_index=self.axis, gravity_compensation=True,
            )
            tau, diag = self.adapter.step(state=state)
            if not bool(diag.get("safety_ok", True)):
                guard_fired = True
                self._guard_reason = str(diag.get("safety_reason", ""))

            theta, thetadot = self._pend()
            self._monitor.step(thetadot=thetadot, phi=wrap_pi(theta - self.hanging_angle),
                               drive_accel=a_cmd, dt=CONTROL_DT)

            self.data.ctrl[:6] = np.asarray(tau, dtype=np.float64).reshape(6)
            mujoco.mj_step(self.model, self.data)
            self._t += CONTROL_DT
            if guard_fired:
                break

        self._steps += 1
        theta, thetadot = self._pend()
        phi_inv = wrap_pi(theta - self.inverted_angle)
        s_val = thetadot + self.omega * phi_inv
        near_top = abs(phi_inv) < self.phi_capture
        if near_top:
            self._best_abs_s = min(self._best_abs_s, abs(s_val))

        e_frac = self._energy_frac()
        # RATCHET on PEAK energy, not the step-to-step delta. Summing
        # (e - e_prev) telescopes to (e_final - e_0), so a policy that pumps the
        # pendulum up and lets it fall back scores exactly ZERO -- there was no
        # gradient rewarding transient progress at all. Combined with a maximum
        # possible energy return (w_energy * 1.0 = 20) far below the guard
        # penalty (100), that made "do nothing" a strong optimum. Rewarding only
        # NEW peaks makes progress unloseable and sums to w_energy * e_peak,
        # which is the quantity actually wanted.
        reward = self.w_energy * max(0.0, e_frac - self._e_frac_peak)
        self._e_frac_peak = max(self._e_frac_peak, e_frac)
        reward -= self.w_control * (a_pump / max(self.a_max, 1e-9)) ** 2
        self._prev_e_frac = e_frac

        captured = bool(near_top and abs(s_val) <= self.s_capture)
        terminated = False
        if guard_fired:
            reward -= self.r_guard
            terminated = True
        elif captured:
            reward += self.r_capture
            terminated = True
        truncated = bool(self._steps >= self.max_steps)

        budget = self._monitor.budget()
        info = {
            "energy_frac": e_frac,
            "s": s_val,
            "phi_inv_rad": phi_inv,
            "best_abs_s": None if not np.isfinite(self._best_abs_s) else float(self._best_abs_s),
            "captured": captured,
            "guard_fired": guard_fired,
            "guard_reason": self._guard_reason,
            # The legibility number. ~1.0 for the analytic law; below 0.5 the
            # policy is removing energy on balance no matter what reward says.
            "positive_work_fraction": float(budget.positive_work_fraction),
            "e_peak_over_e_top": float(budget.e_peak_j / self.e_top) if self.e_top else None,
        }
        return self._obs(), float(reward), terminated, truncated, info
