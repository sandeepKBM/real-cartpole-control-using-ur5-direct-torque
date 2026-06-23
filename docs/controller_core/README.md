# controller_core

Simulator-independent control for the UR5 CoppeliaSim / ROS port (numpy only).

## Main entry point (Cartesian impedance)

| Module | Purpose |
|--------|---------|
| `x_axis_cartesian_impedance.py` | Full 6D `J^T wrench` + posture + damping + optional gravity; Y/Z/orientation hold. |
| `filters.py` | Torque low-pass + per-joint rate limit. |
| `safety.py` | Drift, orientation, velocity, x-error growth, NaN, joint limits. |
| `state_types.py` | `as_impedance_robot_state` (requires `jacobian` 6×6, EE velocities). |
| `logging_utils.py` | JSON helpers / JSONL writer. |

Legacy X-only PD + `safety_utils.py` remain for older MuJoCo torque demos.

## Tests

```bash
python controller_core/tests/test_core.py
python controller_core/tests/test_impedance.py
```
