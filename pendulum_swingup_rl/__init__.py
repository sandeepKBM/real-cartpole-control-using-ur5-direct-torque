"""RL over the swing-up ENERGY SEQUENCE (not over static gains).

See ``env.py``'s module docstring for why this is a different problem from
``rl_gain_scheduling`` (whose action is gain multipliers and whose environment
contains no pendulum), and for the ``positive_work_fraction`` legibility check
any policy trained here must be judged against before its reward curve is
believed.
"""

from pendulum_swingup_rl.env import OBS_DIM, PendulumSwingupEnv

__all__ = ["PendulumSwingupEnv", "OBS_DIM"]
