# Does the hanging pose need `split_base_wrist_task`? (2026-08-02)

## Question

`docs/status/session_compilation_2026-08-01_night.md` sec 12 item 8 flagged this as an open
question: `split_base_wrist_task` (`controller_core/x_axis_cartesian_impedance.py`, flag-gated)
was built specifically because the OLD transport pose family sits at the `wrist_2=0` UR5e
kinematic singularity (`docs/status/split_base_wrist_impedance_2026-08-01.md`, cond(full 6x6 J)
~1e16-1e17 throughout). The hanging/elbow-down pose family
(`docs/status/hanging_pose_transport_family_2026-08-01.md`) was designed from scratch to avoid
that singularity structurally (cond(J) 7-15 across its whole range) -- so does it even need
this flag, or does it introduce real cost with no matching benefit away from a singularity?

## Method

Best-validated hanging-pose config,
`config/ur5e_mujoco_torque_osc_hanging_pose_friction_ff.yaml` (36/38 documented), compared
against a new variant with the **single field** `controller.split_base_wrist_task: true` added
and nothing else changed:
`config/ur5e_mujoco_torque_osc_hanging_pose_friction_ff_split_base_wrist.yaml`. Deliberately not
built the same way as `config/ur5e_mujoco_torque_osc_tuned_split_base_wrist.yaml` (which also
turns on `wrist_orientation_task`/`kp_rot_wrist`/`kd_rot_wrist` alongside the split, appropriate
for that config's own real-hardware purpose) -- for a clean single-variable A/B here, only
`split_base_wrist_task` differs between the two hanging-pose configs.

Same 4-category rigor sweep (`canonical_grid`, `long_holds`, `large_displacements`,
`torque_scale_robustness`; identical grids to `tools/ur5e_pose_sweep_transport.py`'s
`CATEGORY_GRIDS`) run via `tools/ur5e_move_hold_transport.py --config ... --no-plot`
(config's own `home_qpos`/`use_home_qpos_as_start: true` supplies the start pose; the
`ur5e_pose_sweep_transport.py` wrapper itself only knows `hardware.poses.q_for_height_alpha`,
the OLD family, so it was not used here) at `HANGING_ALPHA_0_5_Q` (alpha=0.5, the pose both
hanging configs are built around), seed=0. A second, smaller pass (`canonical_grid` +
`large_displacements` only, for time) ran at `q_for_hanging_height_alpha(0.2)` for a broader
check. `controller_core/x_axis_cartesian_impedance.py`, `hardware/poses.py`, and every existing
config were left untouched -- only the one new config file was added.

Note on environment: the first sweep attempt failed immediately with
`ModuleNotFoundError: No module named 'pinocchio'` because the shell's default `python`
resolved to the base conda env, not `mujoco_ur5e` (`environment.yml`) -- reran with
`/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python` explicitly and it worked cleanly.

## Results

### A/B rigor sweep at alpha=0.5 (`HANGING_ALPHA_0_5_Q`)

| category | plain (no split) | + split_base_wrist_task |
|---|---|---|
| canonical_grid | 8/8 | 8/8 |
| long_holds | 8/8 | 8/8 |
| large_displacements | 8/8 | **6/8** |
| torque_scale_robustness | 12/14 | 12/14 (byte-identical failures) |
| **total** | **36/38 (94.7%)** | **34/38 (89.5%)** |

`split_base_wrist_task` **regresses** 2 cases: `dx=0.20m`, `hold={1.0, 2.0}s`. Both fail
identically -- `outcome: failure`, `failure_category: z_drift`,
`termination_reason: "|Z-Z0| > 0.03 m"`, tripped mid-**move** phase at 285/1000 trace steps
(orientation error only 0.116 rad at trip, well under the 0.25 rad guard -- this is not an
orientation problem). The plain config completes the identical move cleanly
(`termination_reason: duration_complete`, orientation error 0.167-0.169 rad, `jacobian_cond`
11.07-11.17). Root cause, read directly from the config/implementation: `split_base_wrist_task`
restricts the *translation* task Jacobian to base-joint columns only (`J_task[:, 3:6] = 0`),
structurally discarding wrist joints' contribution to holding Z (and Y) during a large X move.
Near the old family's `wrist_2=0` singularity those wrist columns are useless anyway (exactly
singular), so zeroing them costs nothing there. Away from any singularity -- which is the
hanging pose's entire point -- those columns are well-conditioned and were genuinely
*contributing* real Z-holding authority; removing them narrows the controller's authority for
no compensating benefit, and at this pose's largest tested displacement (dx=0.20m, already the
most marginal case in the plain config's own envelope, worst orientation 0.175 rad) that lost
authority is enough to tip Z over its 0.03m guard.

### A/B rigor sweep at alpha=0.2 (broader check, `canonical_grid` + `large_displacements` only)

| category | plain | + split_base_wrist_task |
|---|---|---|
| canonical_grid | 8/8 | 8/8 |
| large_displacements | 4/8 | 4/8 (byte-identical: dx=0.05/0.10 pass, dx=0.15/0.20 fail via `move_phase_incomplete` in both) |

**Byte-identical outcome at alpha=0.2** -- no regression, but also no benefit; both configs
fail the same pre-existing reach-limit cases (not investigated further, out of scope for this
task) identically with or without the flag. Consistent with the alpha=0.5 story: the
Z-authority cost is real but only bites when a case is already near its own envelope edge, and
at alpha=0.2 the shared failures happen for an unrelated reason before that cost would matter.

### `jacobian_cond` -- real, quantified, but far smaller than the old family's fix

Aggregated across every trace row in the full alpha=0.5 4-category sweep (~106-108k control
cycles per config):

| config | min | median | max |
|---|---|---|---|
| plain (full 6x6 J) | 8.81 | 8.91 | 11.17 |
| + split_base_wrist_task (reduced 3x3 base-block J) | 4.07 | 4.24 | 4.37 |

At alpha=0.2 (canonical_grid + large_displacements only): plain 11.91-15.63 (median 11.96) vs.
split 6.12-9.17 (median 6.17). **Consistent ~2x conditioning improvement at both alphas** --
real and measurable, but nowhere near the old family's 1e16-to-single-digits fix (12-16 orders
of magnitude), exactly as expected for a pose that was never near-singular to begin with. This
directly confirms the secondary nullspace-projector benefit is real here too (a genuinely
better-conditioned matrix for the nullspace projection to work with), just at a scale where it
doesn't change any pass/fail outcome on its own.

### Orientation-error growth over exposure time

Checked the `long_holds` category's `dx=0.06m` cases (hold = 4/10/20/30s) for both configs --
the same mechanism `docs/status/split_base_wrist_orientation_growth_2026-08-01.md` diagnosed as
a monotonic-with-time orientation-error growth near the old family's singularity:

| config | hold=4s (start&rarr;end) | hold=10s | hold=20s | hold=30s |
|---|---|---|---|---|
| plain | 0.0524&rarr;0.0438 | 0.0524&rarr;0.0430 | 0.0524&rarr;0.0441 | 0.0524&rarr;0.0456 |
| + split_base_wrist_task | 0.0776&rarr;0.0485 | 0.0776&rarr;0.0383 | 0.0776&rarr;0.0341 | 0.0776&rarr;0.0334 |

**No growth-with-time pattern in either config.** Orientation error peaks right at hold-start
(end of the move) and then decays or stays flat for the rest of the hold, out to 30s, in both
configs -- the opposite of the old family's monotonic-growth signature. This is direct evidence
the growth mechanism documented for the old family is specific to holding orientation near the
`wrist_2=0` singularity (where `kp_rot=0` and orientation is held only by a fragile
nullspace-posture/damping path), not a general property of this controller's damping-only
rotational gains. The hanging pose, structurally away from any singularity, doesn't trigger it
at all -- with or without `split_base_wrist_task`.

## Verdict

**`split_base_wrist_task` is not worth keeping for the hanging pose.** It:
- Does **not** fix anything the hanging pose was failing at (jacobian_cond was already
  single-digit/low-double-digit; there is no wrist-singularity failure mode to fix).
- Does **not** stop the orientation-growth mechanism, because that mechanism doesn't occur here
  in the first place -- there is nothing to fix.
- **Does** cause a real, root-caused regression at the pose's own largest validated
  displacement (dx=0.20m, alpha=0.5): 36/38 &rarr; 34/38, via structurally discarding wrist-joint
  Z/Y-holding authority that is genuinely useful away from a singularity.
- Provides a real but small (~2x) conditioning improvement that never translates into a
  pass/fail difference anywhere tested.

**Recommendation: the hanging pose is better served staying simple** -- use
`config/ur5e_mujoco_torque_osc_hanging_pose_friction_ff.yaml` (plain, default flags) as the
hanging family's reference config, not the new split variant. This is also a useful negative
result in the other direction from the split-base-wrist doc's own framing: the flag's value is
specifically tied to fixing an ill-conditioned Jacobian, and forcing it on everywhere (including
poses that never had that problem) has a real, measurable cost, not just a no-op.

## Files changed

- `config/ur5e_mujoco_torque_osc_hanging_pose_friction_ff_split_base_wrist.yaml` (new; not
  recommended for use per the verdict above, kept only as the artifact this comparison was run
  against).
- `docs/status/hanging_pose_split_base_wrist_interaction_2026-08-02.md` (this file).

No existing config, `controller_core/`, or `hardware/poses.py` file modified.

## Tests run

- `python -m pytest -q tests` (full suite, `mujoco_ur5e` conda env): **603 passed, 3 xfailed**
  (pre-existing, unrelated), zero regressions.
- Sim rigor sweeps (not pytest, this task's own validation): alpha=0.5 full 4-category (both
  configs, 76 runs total) + alpha=0.2 partial 2-category (both configs, 32 runs total), all
  completed without crash; results tabulated above.

## Rollback

`git revert <this commit's hash>` -- purely additive (one new config file, one new doc); nothing
else was touched.
