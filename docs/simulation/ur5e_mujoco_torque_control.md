# UR5e MuJoCo Torque Control

This document describes the simulation-only UR5e torque-control experiment
path added in this branch.

It is a MuJoCo-specific path. It does **not** touch hardware, RTDE, ROS 2
hardware bring-up, or CoppeliaSim controller code.

## What Model Source Was Used

The checked-in torque variant is based on the vendored MuJoCo Menagerie UR5e
model:

- base robot XML: `mujoco_menagerie/universal_robots_ur5e/ur5e.xml`
- torque variant: `assets/ur5e_torque/ur5e_torque.xml`
- scene wrapper: `assets/ur5e_torque/scene.xml`

That Menagerie UR5e already contains the link tree, joint names, limits,
inertials, collision geoms, and visual meshes. It was originally built with
position-style `general` actuators. For torque experiments, I replaced the
actuator block with explicit MuJoCo `<motor>` actuators.

An alternate URDF-to-MJCF path is also available for manual sources:

- `scripts/convert_ur5e_urdf_to_mjcf.py`

That script accepts a URDF path or URL and rewrites `package://...` mesh paths
if package roots are provided. In this repo, the vendored Menagerie model was
the practical local starting point, so it is the source actually used for the
checked-in torque scene.

## Phase 2: True Torque Controller Evaluation

Phase 2 is the validation phase for this MuJoCo lane. RL is intentionally
excluded here. The goal is to prove that the UR5e MJCF is truly torque
actuated, that `data.ctrl` maps to joint torque commands, and that the
existing controller-core laws can be evaluated without any hidden
position-servo shortcut.

The torque scene is checked in with six `<motor>` actuators. The verifier
rejects:

- leftover `<position>` or `<velocity>` actuators on the six UR5e joints
- legacy `<general>` actuators on those joints
- equality constraints, nonzero joint stiffness, or suspicious passive hold
  settings in the source tree
- compiled models where the actuator bias / gain / dynamics types are not
  pure torque motors

The anti-cheating probe is:

```bash
python3 tools/ur5e_mujoco_torque_experiments.py --mode zero-torque-gravity --duration 1.0
```

It sets `data.ctrl[:] = 0`, steps the model, and reports whether the arm
moves under gravity. If the arm stays suspiciously static, the summary marks
that as suspicious instead of pretending the test passed.

The controller comparison runner is:

```bash
python3 tools/compare_ur5e_mujoco_controllers.py \
  --controllers torque_qp impedance \
  --target-x-deltas 0.0025 0.005 0.01 0.02 0.04 \
  --durations 1.0 3.0 \
  --torque-limit-scales 0.25 0.5 1.0
```

It writes a comparison directory under:

```text
outputs/ur5e_mujoco_torque/comparison_<timestamp>/
```

The comparison `summary.csv` is the quickest way to read the results. Each
row includes:

- controller name
- target X delta
- duration
- torque-limit scale
- success / failure
- termination reason
- final X error and end-effector error norm
- max / mean torque
- torque saturation percentage
- max joint speed
- joint-limit margin
- trace path

The per-run traces remain compatible with the existing JSONL trace schema and
the run-specific plots still show `q`, `qd`, raw torque, clipped torque, and
the end-effector error over time.

## Phase 3: Gravity-Compensated Envelope Study

Phase 3 stays simulation-only and keeps RL out of scope. The goal is to study
how far the existing non-RL controllers can move in the X frame when we add an
optional gravity-compensation torque term inside the MuJoCo adapter.

There are now two torque application modes:

- `raw`: `data.ctrl` receives only the controller output.
- `gravity_comp`: `data.ctrl` receives controller output plus a gravity torque
  term computed from the current MuJoCo configuration.

Both modes still use the same torque motors. Gravity compensation is still
torque control because the adapter adds torque directly into `data.ctrl`; it
does not switch to position targets, velocity targets, or constraints. For
controller development, `gravity_comp` is now the primary residual-torque
environment. Raw mode stays available only for validation and anti-cheating
sanity checks.

In residual-torque mode the summary explicitly logs:

- `tau_controller`: controller residual torque
- `tau_gravity`: model gravity compensation torque
- `tau_applied`: controller + gravity before clipping
- max / mean absolute values of each
- controller and applied clipping fractions
- gravity/controller torque fractions relative to applied torque

The raw anti-cheating test is still retained:

```bash
python3 tools/ur5e_mujoco_torque_experiments.py --mode zero-torque-gravity --duration 1.0
```

That test keeps `data.ctrl[:] = 0` in raw mode and checks that the arm moves
under gravity. A separate gravity-comp hold probe is also available:

```bash
python3 tools/ur5e_mujoco_torque_experiments.py --mode gravity-comp-hold --duration 0.05
```

That probe uses a zero controller output but adds the current gravity torque
inside the adapter, so it should hold the pose much better if the model and
actuator mapping are correct. Longer horizons can still trip the velocity
guard during the initial transient, so start with a short hold window when
checking the compensation path.

The long-horizon envelope runner is:

```bash
python3 tools/ur5e_x_frame_envelope.py \
  --controllers impedance \
  --gravity-modes gravity_comp \
  --profiles min_jerk \
  --target-x-deltas 0.005 0.01 0.02 \
  --durations 1.0 3.0 \
  --torque-limit-scales 0.5 0.75 1.0
```

It writes an envelope directory under
`outputs/ur5e_mujoco_torque/x_frame_envelope_<timestamp>/` with:

- `summary.csv`
- `summary.json`
- `best_settings.json`
- `per_run_traces/`
- `plots/`

Use `summary.csv` to compare controller, gravity mode, profile, target delta,
and duration. Use `best_settings.json` to see the best stable setting at each
tested duration and the common failure modes. Raw gravity-mode comparisons are
still available if you pass `--gravity-modes raw gravity_comp`, but they should
be treated as validation rather than the default development lane.

The residual-torque tuner is:

```bash
python3 tools/tune_ur5e_residual_impedance_transport.py \
  --config config/ur5e_mujoco_torque_transport.yaml \
  --gravity-mode gravity_comp \
  --profile min_jerk \
  --hold-durations 1.0 3.0 5.0 \
  --small-target-x-deltas 0.005 0.01 0.02 \
  --small-durations 1.0 3.0 \
  --validation-target-x-deltas 0.03 0.04 0.05 0.06 \
  --validation-durations 1.0 3.0 5.0 \
  --torque-limit-scales 0.5 0.75 1.0
```

It writes a tuning directory under
`outputs/ur5e_mujoco_torque_transport/residual_impedance_tuning_<timestamp>/`
and a matching diagnostics directory under
`outputs/ur5e_mujoco_torque_transport/residual_torque_diagnostics_<timestamp>/`.
The tuning output includes `summary.csv`, `summary.json`, `best_settings.json`,
`candidate_configs/`, `per_run_traces/`, `diagnostics/`, and `plots/`.

## Phase 4: Residual Impedance Move-and-Hold Transport

Phase 4 separates move tracking from post-move hold behavior. The goal is to
move to the X target with a `min_jerk_move_hold` profile and then hold the
transported pose for the remainder of the rollout while keeping Y/Z drift and
orientation error inside the same residual-torque guardrails.

The dedicated move-and-hold runner is:

```bash
python3 tools/ur5e_move_hold_transport.py \
  --config config/ur5e_mujoco_torque_transport.yaml \
  --gravity-mode gravity_comp \
  --target-x-deltas 0.01 0.02 0.03 0.04 \
  --move-durations 0.5 1.0 1.5 2.0 \
  --hold-durations 1.0 2.0 4.0 \
  --torque-limit-scales 0.5 0.75 1.0
```

It writes a move-and-hold directory under
`outputs/ur5e_mujoco_torque_transport/move_hold_transport_<timestamp>/` and
records separate move-phase and hold-phase metrics:

- `move_phase_*`
- `hold_phase_*`
- `valid_move_phase`
- `valid_hold_phase`
- `valid_move_and_hold`

The move-and-hold summary is intentionally stricter than raw displacement
ranking. It rejects overshoot cases where the end effector travels far beyond
the commanded X delta, then wanders or drifts during the hold window.

For a transport-biased starting posture, either pass the six joint angles
directly with `--start-q-rad q1 q2 q3 q4 q5 q6` or use the dedicated config:

```bash
python3 tools/ur5e_mujoco_torque_experiments.py \
  --mode controller-rollout \
  --config config/ur5e_mujoco_torque_transport.yaml \
  --target-x-delta 0.01
```

The low-Z seed from the acceleration transport lane is still available via
`--start-q-rad`; the dedicated transport config instead starts from the
canonical active-origin pose used by the existing torque transport lane,
which is the more stable baseline for this controller family.

## Why This Is Torque-Controlled

MuJoCo torque control should use `<motor>` actuators with `gear="1"` and an
explicit torque `ctrlrange`. The torque variant does exactly that:

- shoulder joints / elbow: conservative `±150 Nm`
- wrist joints: conservative `±28 Nm`

Those are simulation control ranges only. They are not a hardware safety case.

The torque actuator mapping is validated by name. The canonical UR5e order is:

1. `shoulder_pan_joint`
2. `shoulder_lift_joint`
3. `elbow_joint`
4. `wrist_1_joint`
5. `wrist_2_joint`
6. `wrist_3_joint`

The MuJoCo adapter refuses a model that does not expose those six joints and
six matching torque motors.

## How The CoppeliaSim Work Was Reused

The CoppeliaSim work is reused as **controller math and guardrail logic**, not
as simulation physics truth.

Reused pieces:

- `controller_core/torque_task_qp.py`
- `controller_core/x_axis_cartesian_impedance.py`
- `controller_core/safety.py`
- `controller_core/filters.py`
- `controller_core/logging_utils.py`

These pieces provide:

- task-space impedance / torque-QP control laws
- safety monitoring and joint-velocity / orientation limits
- torque filtering and slew-rate limiting
- structured JSON / JSONL logging

Not reused:

- CoppeliaSim object or joint semantics
- CoppeliaSim actuator semantics
- CoppeliaSim timing assumptions
- any RTDE / hardware code

## Adapter Structure

The MuJoCo bridge lives in:

- `simulation/ur5e_mujoco_torque.py`

The adapter performs the following steps each cycle:

1. Read `q` and `qd` from MuJoCo `data.qpos` / `data.qvel` using the fixed
   joint mapping.
2. Read the end-effector pose and Jacobian from MuJoCo.
3. Build a canonical `controller_core` state.
4. Call the reused controller (`torque_qp` or impedance).
5. Low-pass filter and slew-rate limit the raw torque.
6. Clip the final torque to the configured limits.
7. Write the six torques to `data.ctrl` in actuator order.
8. Log the raw torque, clipped torque, state, and safety status.

The adapter remains separate from CoppeliaSim-specific APIs.

## How To Run The Validation Experiments

The safe entrypoint is:

```bash
python3 tools/ur5e_mujoco_torque_experiments.py --mode model-load
```

Common rollout examples:

```bash
python3 tools/ur5e_mujoco_torque_experiments.py --mode impedance-hold
python3 tools/ur5e_mujoco_torque_experiments.py --mode controller-rollout --target-x-delta 0.01
python3 tools/ur5e_mujoco_torque_experiments.py --mode single-joint-pulse --joint-index 0 --torque-nm 1.0
python3 tools/ur5e_mujoco_torque_experiments.py --mode safety-clipping --joint-index 0 --torque-nm 1000.0
```

Outputs are written under:

```text
outputs/ur5e_mujoco_torque/
```

Each run writes a timestamped subdirectory containing:

- `summary.json`
- `trace.jsonl`
- `trace_states.png`
- `trace_diagnostics.png`

## What The Logs Mean

Per rollout, the traces record:

- `q` and `qd` over time
- commanded torque before filtering
- filtered torque
- clipped torque
- torque saturation and clipping fractions
- joint-limit margin fraction
- reward terms
- termination reason

The summary also records:

- model dimensions and mapping
- final EE pose and joint state
- max saturation / clip levels
- effort and energy proxies

## Why This Is Not A Hardware Model

This MuJoCo scene is useful for torque-control experimentation, but it is not a
hardware truth model.

Reasons:

- the Menagerie UR5e is a simplified public model
- torque limits are conservative experiment values
- friction, damping, compliance, and controller delays are approximate
- real UR5e RTDE / URScript behavior is not represented here
- CoppeliaSim collision and motor semantics are different again

Treat the MuJoCo path as a controlled torque sandbox, not as proof that a real
UR5e will respond identically.

## Relationship To CoppeliaSim

CoppeliaSim and MuJoCo are both simulation tools, but they are not equivalent:

- CoppeliaSim uses its own object/joint/dynamics representation
- MuJoCo uses MJCF bodies, joints, and actuators
- CoppeliaSim direct-torque work does not automatically transfer to MuJoCo
- MuJoCo `<motor>` actuators are the correct place to test torque policies in
  this branch

This branch keeps the CoppeliaSim work intact and adds a separate MuJoCo torque
experimentation lane for algorithm checks, logging, and small rollouts.
