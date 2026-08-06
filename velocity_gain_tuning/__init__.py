"""Gain-tuning infrastructure for controller_core.cartesian_velocity_controller
(the native RTDE ``speedL`` velocity-control lane), specifically its
``ik_seeded_resolution`` mode.

Deliberately independent of ``rl_gain_scheduling/`` -- that package trains a
live per-step PPO policy to schedule the TORQUE-control OSC controller's 11
gains, and its documented history (docs/CURRENT_STATUS.md) is four separate
training attempts, none of which beat the fixed-gain baseline, all failing
for RL-specific reasons (a deceptive "sit still" reward optimum, zero
exploration pressure causing policy collapse, a data.time reset bug) that
have nothing to do with whether good gains exist or are easy to find by
other means. Reusing that code or its action-space/reward design here would
import those same failure modes into a genuinely different, much lower-
dimensional, much better-behaved problem.

Design instead follows this session's own literature check: for a small
(~5 parameter), noisy, non-convex, derivative-free tuning problem like this
one, gradient-free global optimization (CMA-ES, Bayesian optimization,
differential evolution) is the standard tool, not RL -- see optimize.py.
The environment (envs/velocity_transport_env.py) is a real gymnasium.Env so
it CAN be driven by an RL algorithm later if ever wanted, but the default
and recommended path (optimize.py) treats each episode as a single-shot
"try these gains for this whole scripted move, see how well it goes"
evaluation -- a contextual-bandit framing, not sequential decision-making --
which is both what differential_evolution needs and what sidesteps every
RL-specific failure mode in rl_gain_scheduling/'s own history (no temporal
credit assignment, no exploration-collapse risk, no reward-shaping-induced
deceptive optimum, since there's no partial-credit-for-sitting-still
mechanism when the whole episode's reward is a small number of
post-hoc-computed terms, not step-summed).
"""
