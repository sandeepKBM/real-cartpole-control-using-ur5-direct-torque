# -45deg pose: evidence-scoped drift-tolerance raise, validated

**Status:** Deliberate, human-directed follow-up to
`docs/status/neg45_y_axis_diagnosis_and_fix_2026-08-01.md`'s hypothesis 3. That doc found real
evidence the Y excursion at this pose is a bounded, self-correcting transient (not an
instability) but explicitly declined to act on it, since changing a safety threshold requires a
human decision, not an agent's own judgment call. The user reviewed that evidence and explicitly
directed raising the tolerance for this pose specifically. This doc records the resulting config
and its validation. Sim-only; not yet run on real hardware.

## What changed

New config `config/ur5e_mujoco_torque_osc_tuned_wrist_orient_fixed_neg45_pose.yaml` (based on
`..._wrist_orient_fixed.yaml`, the validated directional-ceiling fix): `controller.safety.
max_abs_orthogonal_drift_m` raised from the class default `0.03` to `0.05` -- the field actually
enforced by `ImpedanceSafetyMonitor.check()`'s `set_initial_position()` code path (confirmed by
reading `controller_core/safety.py` directly; `max_abs_y_drift_m`/`max_abs_z_drift_m` are not
used on this path). `controller_core/safety.py`'s own class default is unchanged; this override
lives only in this one named config, applied only when a caller also selects the -45deg pose.

0.05m was chosen with real margin (~18%) above the largest VALIDATED natural peak
(0.0423m at dx=0.06m, from the diagnosis doc's guard-disabled measurement), not an arbitrary
round number, and deliberately not raised further -- see that doc's own dose-response table.

## Validation: 4-category rigor sweep, -45deg pose, same methodology as the diagnosis doc

`tools/ur5e_move_hold_transport.py --start-q-rad <HEIGHT_ALPHA_0_5_CLEARANCE_Q>
--gain-overrides-json <config's own gains>`, same grids as
`docs/status/nullspace_envelope_search_2026-08-01.md` / the diagnosis doc, seed 0.

| category | this config | wrist_orient_fixed (no tolerance raise) | plain baseline |
|---|---|---|---|
| canonical_grid (dx 0.01-0.04m) | **8/8** | 0/8 | 0/8 |
| long_holds (dx 0.03/0.06m, hold 4-30s) | **8/8** | 0/8 | 0/8 |
| large_displacements (dx 0.05-0.20m) | **2/8** | 0/8 | 0/8 |
| torque_scale_robustness (dx 0.03/0.06m) | **14/14** | 0/14 | 6/14 |
| **total** | **32/38** | 0/38 | 18/38 (pre-friction baseline) |

`large_displacements`' 2 passes are both `dx=0.05m` (the two hold durations); `dx=0.10/0.15/0.20m`
still fail via `failure_category: y_drift`, exactly as intended -- these displacements were never
validated by the diagnosis doc's dose-response measurement (which only went to dx=0.06m), so the
guard correctly still catches them rather than the tolerance raise silently covering territory
with no real evidence behind it. This is the deliberate design goal, not a partial failure: an
evidence-scoped increase, not a blanket loosening.

## What this does NOT establish

- **No real-hardware validation.** The diagnosis doc already flagged that this exact pose's real
  vs. sim dose-response has an unexplained gap (real trip at dx=0.20m historically vs. sim onset
  dx=0.05-0.06m) -- this config needs its own real-hardware check, starting small, before being
  trusted there, per this repo's standing real-motion discipline.
- **No claim about dx > 0.06m being safe.** `large_displacements`' remaining 6 failures are
  correct, expected behavior, not a residual bug.
- **The underlying X-Y authority trade-off is unchanged.** This is not a controller fix -- the
  diagnosis doc's finding stands: raising Y-axis PID authority far enough to suppress the drift
  itself breaks X-tracking. This config accepts the natural transient as safe (per the human
  decision above) rather than suppressing it.

## Files changed

- `config/ur5e_mujoco_torque_osc_tuned_wrist_orient_fixed_neg45_pose.yaml` (new).
- This doc.
- No `controller_core/` or `hardware/` files touched. `controller_core/safety.py`'s class default
  unchanged.

## Tests

Config-only change; no new code path. Existing suite unaffected (no controller_core/hardware
files modified in this step).

## Rollback

`git rm config/ur5e_mujoco_torque_osc_tuned_wrist_orient_fixed_neg45_pose.yaml
docs/status/neg45_drift_tolerance_validation_2026-08-01.md`, or revert the commit noted in the
final report.
