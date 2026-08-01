# QP vs impedance at the wrist_2=0 transport singularity (2026-08-01)

Sim-only evaluation. Triggered by tonight's real-hardware TCP-acceleration guard trips at
the height_alpha=0.5 pose using `accel_duration_scurve` (±0.02 m/s², 4s move) with
`config/ur5e_mujoco_torque_osc_tuned_wrist_orient_friction_ff.yaml`. Question: does
`controller_core/torque_task_qp.py` (`TorqueTaskQPController`, box-constrained QP torque
allocator) handle the wrist_2=0 `cond(J)` blowup any more principled than the impedance law?

## Mechanistic answer: no

`TorqueTaskQPController.compute()` and `XAxisCartesianImpedanceController.compute()` compute
`cond(J)` with the identical call (`np.linalg.cond(J)`) and apply the **identical** reactive
scaling formula:

```
singular_scale = jacobian_singular_cond_max / cond   if cond > jacobian_singular_cond_max
```

QP does not touch the upstream geometry differently — it inherits the exact same
already-documented (AGENTS.md §4, "Fixed and promoted to default") freeze/defeat behavior.
With `jacobian_singular_cond_max: 1.0e18` (the real-hardware config's value, same as
tonight's), `singular_scale` is provably pinned at `1.0` regardless of `cond(J)` for *both*
controllers — confirmed in every sim trace below (`singular_scale`/`task_scale` == 1.0 at
every step, even with `cond(J)` peaking at 4.36e16 exactly at the wrist_2=0 start pose),
matching tonight's real trace signature exactly (`singular_scale`/`task_scale` pinned at 1.0
throughout the real failure).

QP's only genuinely different mechanism is downstream: an explicit box-constrained QP
(`solve_box_qp`) over final joint torque, with bounds from (a) the per-joint torque limit
headroom and (b) `_velocity_implied_torque_bounds()` — a linear PD map from a max joint
velocity to an implied torque box. In principle this could reactively clamp a torque spike
the impedance law's own backtracking wouldn't catch. In practice, across every sim run in
this evaluation, **the box constraints never bind**: `controller_torque_clip_fraction == 0.0`
at every step in every run, `max |tau_controller|` stayed at 5–25 Nm against a 25.2–135 Nm
headroom, and `max |qd|` stayed at 0.01–0.06 rad/s against a 2.5 rad/s bound. So in the
regime I could test, QP's box constraints are provably inactive — there is no sim evidence
either way that they would reactively catch a real stick-slip/TCP-accel spike, because no
such spike occurred in sim (see caveat below).

One more mechanistic note, unrelated to conditioning: when QP's box constraints don't bind,
`solve_box_qp`'s unconstrained minimizer is exactly `tau_des` (`linear = -hessian @ tau_des`
algebraically inverts to `x* = tau_des`). But `TorqueTaskQPController.compute()` never wires
in `friction_feedforward`, `wrist_orientation_task`, `task_space_inertia_shaping`, or
`nullspace_posture` — the config keys are silently ignored (only `kp_*/kd_*`, `tau_max_nm`,
`jacobian_singular_cond_max`, `torque_headroom`, and the QP-specific fields are read in
`TorqueTaskQPConfig.from_controller_yaml_section`). This produced a large, **singularity-
independent** displacement-tracking gap in every sweep below (QP: ~62–84% of commanded
displacement; impedance: ~93–99.5%, same gap at the singular and off-singular pose) — a
feature-completeness gap of the QP controller class, not evidence about singularity
robustness one way or the other.

## Sim evidence

### Tooling gaps found (both honestly routed around, not silently papered over)

1. `tools/compare_ur5e_mujoco_controllers.py` accepts `--config` but **never forwards it** to
   the per-point `ur5e_mujoco_torque_experiments.py` child subprocess (confirmed by reading
   `_run_child_experiment()`'s `cmd` list — it passes `--mode/--controller-kind/--target-x-
   delta/--duration/--torque-limit-scale/--output-dir/--seed` plus optional
   `--scene/--start-q-rad/--no-plot`, and nothing else). Every compare-tool sweep therefore
   silently uses the child's own default config (`config/ur5e_mujoco_torque.yaml`) regardless
   of what `--config` is passed on the command line. Not fixed (out of scope for an
   evaluation task) — routed around by calling `tools/ur5e_mujoco_torque_experiments.py`
   directly with an explicit `--config` for every run below.
2. `tools/compare_ur5e_mujoco_controllers.py` has **no CLI passthrough** for
   `--trajectory-profile`/`--target-accel`/`--move-duration` at all, even though the child
   script (`ur5e_mujoco_torque_experiments.py`) supports `accel_duration_scurve` and
   `--target-accel` directly. So the compare tool cannot reproduce tonight's real conditions
   even in principle. Worked around the same way: direct calls to the child script.
3. Sim runs must use the `mujoco_ur5e` conda env
   (`/common/users/ss5772/miniforge3/bin/python` lacks `pinocchio`, and this config uses
   `gravity_source: pinocchio`). An initial pass using the wrong interpreter produced
   misleadingly plausible-looking partial trace data before crashing; those results were
   discarded and everything below was regenerated with
   `/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python`.

New eval-only config `config/ur5e_mujoco_torque_osc_tuned_wrist_orient_friction_ff_minjerk_eval.yaml`
was added (copy of the base config with `mujoco.trajectory_profile: min_jerk` set) purely to
give the compare-tool-style displacement sweep a non-shock trajectory once routed through the
child script directly — the base config has no `trajectory_profile` key, so
`_choose_trajectory_profile()` defaults `controller-rollout` mode to an instantaneous `step`,
which trips `|Z-Z0| > 0.03 m` almost immediately for **both** controllers regardless of pose
(a step-input shock artifact, not a singularity-specific finding — confirmed identical
failure at the off-singular pose too).

### Real caveat: sim cannot reproduce tonight's actual failure at all

`tools/ur5e_mujoco_torque_experiments.py` has **no TCP-acceleration guard**
(`hardware/safety.py`'s `CartesianMoveMonitor`, the mechanism that actually tripped on real
hardware tonight, is a `hardware/`-layer-only construct — confirmed by `grep` returning zero
hits for `CartesianMoveMonitor`/`tcp_accel` in the experiment script). Its only safety
checking is `ImpedanceSafetyMonitor` (Y/Z/orthogonal drift, orientation, `|qd|`, axis growth,
NaN/joint-limit). So even a faithful accel/duration reproduction in this sim tool can never
trip the *same* guard that tripped tonight — a hard boundary on what this evaluation can
answer, not a controller-comparison result.

### Runs

All at `q = [0.0, -0.835398, -1.2, -0.985398, wrist_2, 0.0]`: **singular** (wrist_2=0.0, the
real pose) and an off-singular contrast pose (wrist_2=0.4 rad, otherwise identical — every
named pose in `hardware/poses.py`'s height_alpha family has wrist_2=0.0, so a genuinely clean
baseline required a manual offset). Config:
`config/ur5e_mujoco_torque_osc_tuned_wrist_orient_friction_ff.yaml` (tonight's real config).

**Set A — `accel_duration_scurve`, ±0.02 m/s², move=4s, total=6s (matches tonight exactly)**:
all 8 runs (2 controllers × 2 poses × ±accel) completed `duration_complete`, zero guard trips.
`cond(J)` peaks at 4.36e16 at t=0 (exact wrist_2=0) for both controllers at the singular pose,
falling to ~1e4–1e5 within a fraction of a second as the arm moves off the singularity;
`singular_scale`/`task_scale` pinned at 1.0 throughout for both, as predicted. Peak
finite-difference TCP acceleration: impedance 0.359 m/s² (at t=2.47s, cond≈2.7e4, mid-move —
not at the singularity), torque_qp 0.059 m/s² (t=1.91s, cond≈1.2e3) — neither shows a spike
coincident with the cond(J) peak, and both are far below anything that would plausibly trip a
guard. Displacement achieved at accel=+0.02 (target 0.0509m): impedance 0.0507m (99.6%),
torque_qp 0.0430m (84.5%).

**Set B — `min_jerk` displacement sweep, dx=±0.03/±0.06m, duration=4s** (fair proxy for what
the compare tool nominally supports): all 16 runs (2 controllers × 2 poses × 4 deltas)
completed `duration_complete`, zero guard trips, singular and off-singular results
statistically indistinguishable for each controller (confirming `jacobian_singular_cond_max:
1.0e18` makes the exact-singularity case behaviorally inert for both controllers in this
profile family too). Displacement tracking: impedance 93–99% of target; torque_qp 62–73% of
target — QP's `controller_torque_clip_fraction` was 0.0 at every step of every run (box
constraints never bound), so the gap is the missing-feature effect described above (no
`friction_feedforward` against the sim's real per-joint friction, no operational-space Λ
shaping / wrist-orientation task), not saturation.

Full per-run summaries: `outputs/qp_vs_impedance_eval_v3/all_results.json` (24 runs,
`outputs/` is gitignored, not committed).

## Recommendation: do not pursue `torque_task_qp` for this failure

1. It does not change the upstream `cond(J)`-at-singularity computation or scaling logic at
   all — same formula, same already-defeated-at-1e18 clamp as the impedance law. Whatever
   fixes this failure, it isn't "switch to the QP allocator."
2. Its one structurally different mechanism (box-constrained torque/velocity bounds) is
   unexercised in every regime I could test here (0.0 clip fraction throughout) — there is no
   sim evidence it would reactively catch a real stick-slip/TCP-accel spike, because sim
   never produced one to catch. This evaluation cannot confirm or rule that out; it would
   need either a sim-side TCP-accel guard (mirroring `CartesianMoveMonitor`, which doesn't
   exist in the sim tool today) or direct real-hardware testing.
3. **`torque_task_qp` is not real-hardware-reachable today.** Contrary to the task's
   speculation that it might need no URScript work since it's Python-side: I read
   `hardware/direct_torque_transport.py` in full — it hardcodes and unconditionally
   instantiates `XAxisCartesianImpedanceController` (imported directly from
   `controller_core.x_axis_cartesian_impedance`); there is no `controller_kind`/config switch
   anywhere in `hardware/`, and nothing under `hardware/` imports `TorqueTaskQPController` at
   all. Deploying it would need new wiring in `direct_torque_transport.py` *and* either
   reimplementing `friction_feedforward`/`wrist_orientation_task`/`task_space_inertia_shaping`
   for the QP class (none exist today) or accepting the ~20–40 percentage-point tracking
   regression measured above, for a mechanism this evaluation found no evidence helps the
   actual failure being investigated.

Net: real-hardware validation cost for `torque_task_qp` here is not small (new hardware
wiring + feature-parity work), and the mechanistic/sim evidence gives no reason to expect it
would pay off for this specific failure. Better next steps, given sim cannot even reproduce
tonight's trip: (a) add a sim-side TCP-acceleration guard so this failure class becomes
sim-testable at all, or (b) pursue the stick-slip/friction-breakaway hypothesis directly
(AGENTS.md's own LuGre motivation section already documents an analogous real accel spike
from a stick-slip breakaway event on the trajectory-profile work, a mechanism neither
controller here represents).

## Files touched

- Added (new, evaluation-only): `config/ur5e_mujoco_torque_osc_tuned_wrist_orient_friction_ff_minjerk_eval.yaml`.
- No existing config or `controller_core/` file modified.
- Sweep outputs under `outputs/qp_vs_impedance_eval*/` (gitignored, not committed).
