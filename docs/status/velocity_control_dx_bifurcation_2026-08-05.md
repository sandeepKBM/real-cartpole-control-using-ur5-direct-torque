# Velocity control (speedL): the dx bifurcation, root-caused — 2026-08-05

**Status: sim-only (kinematic), NOT real-hardware validated.** Produced by an offloaded codex
run against `tools/diagnostics/ur5e_velocity_control_kinematic_sim.py`; raw traces were written
to `outputs/velocity_char/` (gitignored, so this file is the durable record).

**Headline:** the previously "not understood" dx-dependent divergence (AGENTS.md section 2b) now
has a mechanism. It is **kinematic branch selection near the wrist_2=0 singular surface**, not a
gain or tuning problem. A small dx does not carry `wrist_2` far enough positive, so during the
hold phase it falls back through zero onto the negative branch and orientation error grows
without bound. A larger dx pushes it clear and it stays bounded. This is why the *smaller*,
apparently easier move was the unstable one — the behavior is genuinely non-monotone in dx.

**Practical envelope at this pose/timing (1 s move, 125 Hz):** only `dx=0.04 m` completes
cleanly. `dx=0.01-0.03` trip the orientation guard; `dx=0.05-0.08` trip the joint-velocity
guard — note this is a *tighter* upper bound than the previously recorded `dx>=0.06`. The
stable/unstable boundary sits near **dx ~= 0.028 m** (bracketed between 0.0275 and 0.02875).

**Two mechanism findings that matter for tuning:** `kp_posture` modulates the basin of
attraction but does not create the boundary (the low-dx cases still fail at `kp_posture=0`),
and `pinv_damping` is strongly coupled to it — raising it 0.005 -> 0.01 collapses the stable
basin and pushes both boundary cases onto the bad branch. That is direct measured support for
the controller docstring's existing warning that these two are not independently tunable.

Reanchor timing was tested and **ruled out** (first reanchor at t=1.072 s in every run, stable
or unstable).

---

# UR5e Velocity-Control Characterization

Run context:

- Requested script: `tools/diagnostics/ur5e_velocity_control_kinematic_sim.py`
- Python: `/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python3`
- Default pose: `[-0.6981317007977318, -0.8353981633974483, -1.2, -0.9853981633974482, 0.2, 0.0]`
- Baseline sweep: `--move-duration 1.0 --duration 7.0 --rate-hz 125`
- Auxiliary probe: `outputs/velocity_char/velocity_boundary_probe.py`
- Saved traces: `outputs/velocity_char/probe_traces/`

## Code-Path Verification

The requested script is kinematic-only. The evidence is in the source:

- `[tools/diagnostics/ur5e_velocity_control_kinematic_sim.py](tools/diagnostics/ur5e_velocity_control_kinematic_sim.py#L2)` explicitly says it is a kinematic-only sim, with no `mj_step`, no torque, and no mass matrix.
- The same file uses MuJoCo only for forward kinematics and Jacobians, then computes `qd = pinv(J) @ xd_cmd` and integrates `q` with plain Euler steps: see `[tools/diagnostics/ur5e_velocity_control_kinematic_sim.py](tools/diagnostics/ur5e_velocity_control_kinematic_sim.py#L7)` and `[tools/diagnostics/ur5e_velocity_control_kinematic_sim.py](tools/diagnostics/ur5e_velocity_control_kinematic_sim.py#L136)`.
- `[hardware/velocity_transport.py](hardware/velocity_transport.py#L1)` shows the live lane is `speedL` velocity streaming, and `[hardware/velocity_transport.py](hardware/velocity_transport.py#L83)` says it is a Cartesian velocity every cycle, not a torque loop.
- `[controller_core/cartesian_velocity_controller.py](controller_core/cartesian_velocity_controller.py#L1)` and `[controller_core/cartesian_velocity_controller.py](controller_core/cartesian_velocity_controller.py#L257)` show the controller is a resolved-rate velocity law with Jacobian + reduced-task nullspace posture, not a torque controller.
- `friction_feedforward` exists only in the torque lane. See `[controller_core/x_axis_cartesian_impedance.py](controller_core/x_axis_cartesian_impedance.py#L283)` and `[controller_core/x_axis_cartesian_impedance.py](controller_core/x_axis_cartesian_impedance.py#L1621)`. The velocity lane files above do not contain that code path.

Implication: MuJoCo joint frictionloss/damping in the model are never exercised in this lane, and the impedance controller's friction feedforward is not part of this control path. That is correct for `speedL`, because the UR5e firmware resolves the Cartesian velocity and handles the joint loop internally.

## Requested Sweep

Default config: `[config/ur5e_velocity_control.yaml](config/ur5e_velocity_control.yaml#L71)`, with `kp_posture=1.0` and `pinv_damping=0.005`.

| dx (m) | Outcome | Achieved / Target | Max orientation error (rad) | Max `|Y|` drift (m) | Max `|qd|` (rad/s) | Guard / stop |
| --- | --- | --- | --- | --- | --- | --- |
| 0.01 | fail | 1.002 | 0.2507 | 1.46e-05 | 0.131 | orientation_guard |
| 0.02 | fail | 1.001 | 0.2507 | 4.77e-05 | 0.269 | orientation_guard |
| 0.03 | fail | 0.998 | 0.2503 | 1.16e-04 | 0.496 | orientation_guard |
| 0.04 | pass | 1.000 | 0.1768 | 6.16e-04 | 1.405 | duration_complete |
| 0.05 | fail | 0.818 | 0.1514 | 4.94e-04 | 3.005 | joint_velocity_guard |
| 0.06 | fail | 0.670 | 0.1362 | 5.06e-04 | 3.079 | joint_velocity_guard |
| 0.08 | fail | 0.482 | 0.1175 | 5.10e-04 | 3.198 | joint_velocity_guard |

What this sweep shows:

- `dx=0.04` is the only point in the requested sweep that completes cleanly.
- `dx=0.01-0.03` all fail on the orientation guard before the full hold phase completes.
- `dx=0.05-0.08` fail on the joint-velocity guard.
- `Y` drift stays tiny in every case, so this is not a lateral drift problem.

## Boundary Probe

To expose the hold-phase behavior, I reran the kinematic loop with guards effectively disabled in the probe and kept the 1 s move, 6 s hold structure. The resulting boundary is not monotone in `dx`.

Representative default-config raw cases:

| Case | Max orientation error (rad) | Hold error at 6 s (rad) | Final `wrist_2` (rad) | Max `cond(J)` | Readout |
| --- | --- | --- | --- | --- | --- |
| `dx=0.025` | 1.902 | 1.698 | -1.117 | 9.01e4 | bad branch |
| `dx=0.0275` | 1.822 | 1.570 | -1.006 | 1.06e6 | bad branch |
| `dx=0.02875` | 0.367 | 0.295 | 0.446 | 1.35e6 | stable branch |
| `dx=0.03` | 0.324 | 0.270 | 0.484 | 2.59e6 | stable branch |
| `dx=0.035` | 0.236 | 0.184 | 0.615 | 3.85e3 | stable branch |
| `dx=0.04` | 0.177 | 0.118 | 0.727 | 4.84e3 | stable branch |

The low end is also bad:

- `dx=0.01`, `0.015`, `0.02` all end on the negative `wrist_2` branch and grow to roughly `0.85-1.08 rad` orientation error by 7 s.
- `dx=0.025` and `0.0275` are the worst cases I checked, with `wrist_2` crossing through zero and then running negative during hold.
- `dx=0.02875` is the first sampled point that stays on the positive `wrist_2` branch and remains bounded.

Boundary estimate from the sampled points: the switch is between `dx=0.0275` and `dx=0.02875`, so roughly `27.5-28.8 mm` at this pose and timing. That is the narrowest defensible boundary from the runs I actually executed.

Two important non-monotonic facts:

- `dx=0.03` is stable even though its peak `cond(J)` is larger than several unstable cases.
- The raw behavior is therefore not a simple "smaller is worse" or "larger is worse" monotone in `dx`.

## Mechanism Tests

I tested three plausible mechanisms.

First, reanchor timing is not the cause. The first reanchor happened at `t=1.072 s` in every run I checked, stable or unstable. That rules out the stale-`q_rest` timing bug as the bifurcation trigger.

Second, the nullspace posture term is a real factor, but it is not the only factor. With `kp_posture=0.0`:

- `dx=0.02` still degrades badly, with `max orientation error = 1.085 rad` and final `wrist_2 = -1.495 rad`.
- `dx=0.04` stays good, with `max orientation error = 0.180 rad` and final `wrist_2 = 0.682 rad`.
- Around the boundary, `dx=0.0275` becomes much milder (`0.462 rad`) and stays on the positive `wrist_2` branch, while `dx=0.02875` is still worse (`0.844 rad`) but also remains positive.

That means the posture term modulates the basin of attraction, but does not on its own create the boundary.

Third, `pinv_damping` is strongly coupled to that basin. With `pinv_damping=0.01`:

- `dx=0.0275` jumps to `1.924 rad` max orientation error and ends at `wrist_2 = -1.171 rad`.
- `dx=0.02875` jumps to `1.920 rad` max orientation error and ends at `wrist_2 = -1.167 rad`.

So heavier damping collapses the local stable basin and pushes both boundary cases onto the bad branch. The default `0.005` damping is doing real work here, and the controller docstring's warning about keeping it small is consistent with the measurements.

## Interpretation

Best current explanation:

- The boundary is a kinematic branch-selection problem in the reduced-task velocity controller near the `wrist_2=0` singular surface.
- The posture term and the damped pseudoinverse together decide whether the arm stays on the positive `wrist_2` branch or falls through to the negative branch during the hold phase.
- Once the bad branch is selected, orientation drift grows during hold and can exceed 1 rad.
- This is not a friction effect, because the entire lane is kinematic-only and never exercises MuJoCo frictionloss/damping or impedance `friction_feedforward`.

What I do not know yet:

- I did not run a dense grid finer than 1.25 mm between `27.5 mm` and `28.75 mm`, so the numeric threshold is approximate.
- I did not sweep a full `kp_posture x pinv_damping` surface, so the exact stability basin shape is still undercharacterized.

Net result:

- `dx=0.04 m` is genuinely stable for this lane.
- `dx=0.02 m` is genuinely unsafe in the raw hold-phase observation.
- The transition is narrow, non-monotone in `dx`, and centered near `dx ≈ 0.028 m` at this pose and timing.
