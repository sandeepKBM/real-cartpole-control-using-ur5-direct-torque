# Archived Legacy MuJoCo Cartpole Lane

Archived 2026-07-03. These are the earlier MuJoCo cartpole/transport diagnostic scripts that
predate the current true-torque lane (`simulation/ur5e_mujoco_torque.py` +
`tools/ur5e_mujoco_torque_experiments.py`). Several have names that read as CoppeliaSim scripts
(`run_fixed_z_x_transport.py`, `run_x_velocity_transport.py`, `run_x_acceleration_transport.py`,
`run_origin_stabilization.py`) but are pure MuJoCo — archiving resolves that naming trap.

They referenced the full `mujoco_menagerie/` checkout, which has been removed from disk
(re-clone SHA recorded in `archive/coppelia/README.md` and AGENTS.md). Not runnable in place.
