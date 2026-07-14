# URScript inner loop (500 Hz on PolyScope)

**Best latency path** for `direct_torque()` on UR5e: run the control law **on the robot** in
URScript with **V2 friction** (`viscous_scale` / `coulomb_scale` per joint). Python only
generates the script, starts it, and supervises safety at ~125 Hz.

This matches [UR’s recommendation](https://forum.universal-robots.com/t/ur5e-direct-torque-correct-control-equation-gains-and-max-torque-for-vla-rl-inference-not-trajectory-following/42947)
and the [direct_torque manual](https://www.universal-robots.com/manuals/EN/HTML/SW10_12_1/Content/prod-scriptmanual/all_scripts/direct_torque.htm).

## Three hardware lanes (pick one)

| Lane | 500 Hz loop runs on | RTDE calls / step | Friction | Best for |
|------|---------------------|-------------------|----------|----------|
| **URScript** (`ur5e_urscript_x_transport.py`) | **PolyScope** | 0 during loop (Python monitors only) | **V2 per joint** | **Production / minimum latency** |
| Python + `local` dynamics | Your PC | 2 (`read_state`, `directTorque`) | V1 binary | Sim-matched OSC, dev without URScript deploy |
| Python + `rtde` dynamics | Your PC | 4 (+ `getJacobian`, `getMassMatrix`) | V1 binary | Debug vs PolyScope model |

## Quick start (real UR5)

```bash
# 1) Generate script only (inspect before running):
python tools/ur5e_urscript_x_transport.py --robot-ip <IP> --generate-only \
  --target-x-delta 0.02 --move-duration 1.0 --duration 3.0 \
  --output-dir outputs/hardware_urscript/preview

# 2) Live run (moveJ to height 0.5 → URScript OSC transport):
python tools/ur5e_urscript_x_transport.py --robot-ip <IP> \
  --target-x-delta 0.02 --move-duration 1.0 --duration 3.0 \
  --i-understand-this-moves-the-robot --yes
```

**Requirements:** PolyScope **≥ 5.23**, remote control ON, `ur-rtde >= 1.6`, lab PC wired to robot.

## What was converted from Python

Template: `assets/urscript/x_axis_osc_inner.script.template`  
Generator: `hardware/urscript_gen.py` (loads `config/ur5e_mujoco_torque_osc_tuned.yaml`)

| Python (`XAxisCartesianImpedanceController`) | URScript inner loop |
|---------------------------------------------|---------------------|
| min_jerk_move_hold X profile | `min_jerk_move_hold_x()` |
| Task wrench PD (kp_rot=0, kd_rot damping) | Same |
| Λ(q) shaping (`task_space_inertia_shaping`) | `inv(J*inv(M)*J'+eps*I)` when `use_lambda=1` |
| `get_jacobian` / `get_mass_matrix` via RTDE | **On-robot** `get_jacobian()` / `get_mass_matrix()` |
| `direct_torque(..., friction_comp=True)` V1 | **`direct_torque(..., viscous_scale, coulomb_scale)` V2** |
| `nullspace_posture` projector | **Not ported** — posture torque added in joint space |
| Geometric torque backtracking | Simple clamp to limits |
| Python safety monitor | Python **supervisor** trips RTDE input reg 18 |

Validate on hardware with **small dx** before scaling up. Expect minor behavior difference vs
MuJoCo until nullspace/backtracking are ported.

## Stop / e-stop from Python

The script polls `read_input_integer_register(18)`. The supervisor sets this register to `1` on
safety fault. Do not reuse register 18 for other RTDE clients during a run.

## PolyScope manual install (optional)

Instead of `sendCustomScript()` from Python, you can install a permanent program:

1. Copy generated `x_axis_osc_inner.script` to the teach pendant (USB or scp).
2. Create a PolyScope program: **BeforeStart** = ur_rtde init (if needed), **Robot Program** = script.
3. Use `RTDEControlInterface(..., FLAG_CUSTOM_SCRIPT)` from Python for other moves.

See [ur_rtde custom script examples](https://sdurobotics.gitlab.io/ur_rtde/examples/examples.html).

## When *not* to use URScript

- **50 Hz discrete joint targets from a VLA/RL policy** — use joint PD inner loop (forum pattern)
  or `servoJ`, not this Cartesian OSC transport script.
- **URSim** — script may run but will not produce real motion for validation.
