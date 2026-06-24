# Cart-pole MPC

Linear receding-horizon MPC on the 4D acceleration-level cart-pole model:

```text
state = [x, x_dot, theta, theta_dot]
input = cart/task-frame X acceleration
```

## MuJoCo validation (pendulum in MJCF)

```bash
cd /common/users/ss5772/real_Cartpole
python simulation/run_mujoco_cartpole_mpc_smoke.py
```

The pendulum hinge is defined in `mujoco_menagerie/universal_robots_ur5e/ur5e.xml`.

## CoppeliaSim transport

Outer loop: `CartPoleMPCController` + `CommandGovernorSafetyFilter`  
Pole state: MuJoCo observer mirrored from Coppelia joint positions (`controller_core/mujoco_cartpole_state.py`)  
Inner loop: existing IK joint PD torque allocator

```bash
bash simulation/launch_coppelia_mpc_transport.sh
```

Or in the Singularity container (same bind pattern as reciprocating transport).

## Key files

| File | Role |
|------|------|
| `controller_core/mpc_controller.py` | Condensed linear MPC with input box constraints |
| `controller_core/mujoco_cartpole_state.py` | Pole observer from MuJoCo UR5e + hinge |
| `simulation/coppelia_mpc_transport.py` | Coppelia outer-loop wrapper |
| `config/controller_coppelia_mpc.yaml` | Torque / safety profile |

## Tuning

- `--mpc-horizon` (default 20)
- `--mpc-q-theta` — raise to stiffen pole regulation during transport
- `--mpc-pole-length-m` — match MJCF pole length (0.4 m)
- `--target-dx` — transport goal for the outer MPC reference

## Tests

```bash
python -m pytest controller_core/tests/test_mpc_controller.py -q
```
