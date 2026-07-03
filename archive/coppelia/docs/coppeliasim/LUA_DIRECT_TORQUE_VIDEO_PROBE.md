# Lua direct torque probe

This is a **CoppeliaSim-side Lua** diagnostic. It does **not** validate the
external Python/ZMQ controller.

The goal is narrower:

- load the UR5 model inside CoppeliaSim
- configure the joints for torque-capable dynamic control if the API allows it
- apply a small direct torque to one joint first
- then optionally drive Y-axis motion from acceleration direction
- record whether the joint or Y-axis motion occurs
- capture frames or an MP4 if rendering succeeds
- write an honest JSON summary

This is useful for answering a local simulator question:

> Can this UR5 model move under direct joint torque commands inside CoppeliaSim,
> and can it do so using acceleration direction as the high-level input?

It does **not** prove:

- ZMQ attach works
- Python owns stepping
- the external torque controller is healthy
- the Cartesian impedance controller is correct

The external Python/ZMQ lane remains separately validated by:

1. attach-only
2. zero-torque stepping
3. single-joint torque
4. full Cartesian impedance

The summary from the Lua probe should identify itself clearly as internal Lua
control, for example:

- `controller_family = lua_internal_direct_joint_torque_probe`
- `uses_direct_torque_control = true`
- `external_python_zmq_validated = false`
- `stepping_owner = coppeliasim_lua_or_internal`
- `simulation_started_by = coppeliasim_or_lua`
- `lua_motion_enabled = true`

For the Y-axis torque mode, the summary should explicitly distinguish the
control contract:

- `controller_family = lua_internal_y_axis_accel_direction_direct_torque`
- `lua_torque_mode = y_axis_accel_direction`
- `required_user_inputs = ["ACCEL_DIRECTION"]`
- `accel_magnitude_source = internal_default` or `env_override`
- `travel_distance_source = internal_default` or `env_override`
- `jacobian_source = geometric_from_joint_transforms` when the geometric path is used

The direct torque lane is honest only if it keeps `uses_direct_torque_control = true`
and reports the torque guardrails it actually used.
