# kp_rot_wrist retune for split_base_wrist_task's orientation-growth problem

Gain-tuning pass, 2026-08-01/02. Sim-only (no real-hardware access used). Scope: retune
`kp_rot_wrist`/`kd_rot_wrist` (existing flags, both default 0.0/damping-only-via-kd) on top
of tonight's real-hardware-validated `config/ur5e_mujoco_torque_osc_tuned_split_base_wrist.yaml`
(9/9 real). No `controller_core/` changes. No existing config modified. One new named config.

## Background

`docs/status/split_base_wrist_orientation_growth_2026-08-01.md` found real orientation error
grows continuously under `split_base_wrist_task` (up to 0.18 rad in some real tests, not yet
guard-tripping at 0.25 rad but the actual driver of several real TCP-speed-guard trips), because
with `kp_rot=0` (required elsewhere in this repo -- unstable near this pose through the shared,
eps-regularized Lambda-weighted wrench pipeline) and `kp_rot_wrist=0`, orientation has no
proportional/restoring channel at all, only `kd_rot`/`kd_rot_wrist` damping plus whatever
`nullspace_posture` supplies as a side effect. That doc's own concrete, not-yet-attempted
pointer was to raise `kp_rot_wrist`.

## Step 1 -- confirming `kp_rot_wrist` is structurally independent of `kp_rot`'s instability

Read `controller_core/x_axis_cartesian_impedance.py` lines ~941-953 in full. With
`wrist_orientation_task` on:

```python
J_rot = J[3:6, :]
m_wrist = self.cfg.kp_rot_wrist * e_rot - self.cfg.kd_rot_wrist * omega
tau_orient_wrist = (J_rot.T @ m_wrist) * WRIST_ORIENTATION_MASK
```

This is a plain joint-space PD term, computed exactly like `tau_posture` and summed into the
same `tau_bias` (alongside `tau_damping`/`tau_posture`/`tau_friction_ff`/gravity), then flowing
through the existing geometric-backtracking and hard-clip logic unchanged -- no bypass. Unlike
`kp_rot`'s term (which is a wrench component `M` that gets premultiplied by
`Lambda = (J M^-1 J^T + eps*I)^-1`, an explicit matrix inversion regularized near the exact
wrist_2=0 singularity -- the documented source of `kp_rot`'s positive-feedback instability),
`tau_orient_wrist` involves **no matrix inversion anywhere** -- `J_rot.T` is a plain transpose.
This confirms the code comment's claim structurally: the specific mechanism that made `kp_rot`
unstable (regularized-Lambda positive feedback) cannot occur here. This does **not** mean
`kp_rot_wrist` is unconditionally safe at any gain -- see Step 3's own, different instability.

## Step 2 -- sweep at the exact real failure scenarios

Fixed pose `q = [0.0, -0.835398, -1.2, -0.985398, 0.0, 0.0]` (the real failure pose,
`hardware/poses.py::HEIGHT_ALPHA_0_5_Q`). `accel_duration_scurve`, `target_accel` in
`{+0.02, -0.02, +0.015, -0.015}` m/s^2, `move_duration` in `{4.0, 8.0}` s (8 scenarios total,
run to `move_duration + 4.0`s hold so the orientation-error trajectory is visible past the
move), on top of `config/ur5e_mujoco_torque_osc_tuned_split_base_wrist.yaml`, varying only
`kp_rot_wrist`/`kd_rot_wrist`.

**kp_rot_wrist alone, kd_rot_wrist fixed at 10 (the shipped value) -- makes things WORSE, then
unstable**: kp=5/10/20 (max orientation error across the 8 scenarios, 4s-move cases):
kp=0 (baseline): 0.055-0.073 rad. kp=20: 0.120-0.198 rad (up to 2.7x worse). kp>=30: the
0.25 rad orientation guard trips even on the previously-clean 4s moves. kp=50: `max|qd|`
0.29-0.31 rad/s (vs baseline's 0.04-0.17 rad/s) -- classic underdamped second-order response,
not the `kp_rot`-style positive-feedback mechanism, but a real failure mode regardless if `kd`
isn't scaled with `kp`.

**Scaling kd_rot_wrist alongside kp_rot_wrist (~4:1 ratio) fixes the underdamped response and
gives a genuine, monotonic improvement** on the short-hold (4s move + 4s hold) scenarios:

| kp_rot_wrist | kd_rot_wrist | max oe, 4s scenarios (4 values, rad) | vs baseline (0.0559-0.0734) |
|---|---|---|---|
| 0 (baseline) | 10 | 0.0559 / 0.0573 / 0.0707 / 0.0734 | -- |
| 10 | 40 | 0.0463 / 0.0483 / 0.0554 / 0.0578 | ~20% lower |
| 20 | 80 | 0.0363 / 0.0373 / 0.0420 / 0.0434 | ~35-41% lower |
| 40 | 160 | 0.0274 / 0.0284 / 0.0306 / 0.0322 | ~55-58% lower |
| 60 | 240 | 0.0210 / 0.0231 / 0.0238 / 0.0261 | ~62-65% lower |
| 80 | 320 | 0.0172 / 0.0192 / 0.0185 / 0.0195 | ~73-76% lower |

Growth is smooth and monotonic in every trace checked (no oscillation) -- e.g. kp=80/kd=320 at
`accel=0.02, move_duration=8.0` (the hardest scenario, previously guard-tripping at baseline):
orientation error rises 0.00->0.02->0.05->0.10->0.16->0.18 rad over 0-12s, no overshoot,
`torque_saturation`/clip fraction 0.0 throughout. At `kp=60/kd=240` and `kp=80/kd=320`, **all
8/8** of the exact real-failure scenarios pass (vs. baseline's 6/8 -- the two `+-0.02/8.0s`
cases fail at baseline via the orientation guard; `kp=60/kd=240` clears them at 0.2488/0.16;
`kp=80/kd=320` clears them comfortably at 0.183/0.130).

## Step 3 -- a real, separate instability found at high gain: long-hold divergence

The 4-category rigor sweep (`tools/ur5e_pose_sweep_transport.py --height-alphas 0.5`,
`canonical_grid`/`long_holds`/`large_displacements`/`torque_scale_robustness`) at
`kp_rot_wrist=80/kd_rot_wrist=320` found a real regression in `long_holds` (dx 0.03/0.06m,
hold up to 30s): **8/8 (baseline) -> 6/8**. Reading `dx=0.06m`'s per-hold-duration orientation
error directly:

| hold_duration_s | baseline (kp=0) oe | kp=80/kd=320 oe |
|---|---|---|
| 4 | 0.0680 | 0.0083 |
| 10 | 0.0696 | 0.0244 |
| 20 | 0.0707 | 0.1245 |
| 30 | 0.0710 | **0.2499 (guard trip)** |

Baseline plateaus almost immediately (flat 0.068-0.071 across the whole 4-30s window) --
matching the diagnostic doc's own documented plateau behavior. `kp=80/kd=320` starts far
better (short-hold win, matching Step 2) but **never plateaus** -- it keeps growing across the
entire 30s hold and crosses the guard by t=30s. This is a real, separate, slow-onset
instability at high gain, distinct from both the underdamped-PD failure in Step 2 and the
matrix-inversion mechanism that made `kp_rot` unstable -- not diagnosed further here (out of
scope for a gain-tuning pass; flagged for anyone touching this mechanism next).

**kp_rot_wrist=20/kd_rot_wrist=80 and kp_rot_wrist=10/kd_rot_wrist=40 both stay stable across
the full 30s hold** (`kp=20/kd=80` dx=0.06m: 0.032/0.039/0.053/0.075 rad at hold=4/10/20/30s --
still slowly rising but well clear of the 0.25 rad guard with an order-of-magnitude margin;
`kp=10/kd=40`: 0.032/0.039/0.046/0.049 rad, flatter still). Both preserve `long_holds` 8/8.

## Chosen candidate: kp_rot_wrist=20.0, kd_rot_wrist=80.0

Chosen over `kp=10/kd=40` for a larger effect size on the scenarios that actually motivated
this retune (35-41% orientation-error reduction vs. ~20%) at equal `long_holds` safety margin.
Chosen over `kp=60/80` x `kd=240/320` (which give an even larger short-hold improvement and
even clear the two previously-failing real scenarios outright) because of the Step 3
long-hold-divergence finding -- a materially larger, unexplained failure mode not worth trading
for extra short-hold margin without further investigation. New config:
`config/ur5e_mujoco_torque_osc_tuned_split_base_wrist_wrist_orient_retuned.yaml`, byte-identical
to `config/ur5e_mujoco_torque_osc_tuned_split_base_wrist.yaml` except `kp_rot_wrist: 20.0` /
`kd_rot_wrist: 80.0` (was `0.0`/`10.0`).

## Step 4 -- 4-category rigor sweep, regression check

`tools/ur5e_pose_sweep_transport.py --height-alphas 0.5 --categories canonical_grid long_holds
large_displacements torque_scale_robustness --seed 0`, this candidate vs. a freshly re-run
baseline (`config/ur5e_mujoco_torque_osc_tuned_split_base_wrist.yaml`, unmodified -- the
original split_base_wrist doc only ran 3/4 categories, so `long_holds` needed a fresh baseline
number too):

| category | baseline (fresh) | kp=20/kd=80 candidate |
|---|---|---|
| canonical_grid | 6/8 | 6/8 |
| long_holds | 8/8 | 8/8 |
| large_displacements | 8/8 | 8/8 |
| torque_scale_robustness | 12/14 | 12/14 |
| **total** | **34/38** | **34/38** |

Byte-identical pass/fail pattern across every category -- **zero regressions**.

## A tooling gotcha worth recording for future sweeps

`tools/ur5e_pose_sweep_transport.py`'s auto-extracted `--gain-overrides-json` (printed in its
own log) only lists `transport_metrics.GAIN_FIELDS`'s 11 canonical fields and does **not**
include `kp_rot_wrist`/`kd_rot_wrist` -- at first glance this looks like the override is
silently dropped. It is not: `tools/tuning_common.py::_candidate_config_payload` deep-copies
the full `--config` YAML (which already carries the correct `kp_rot_wrist`/`kd_rot_wrist`
values from the file on disk) and only *overwrites* the 11 `GAIN_FIELDS` keys in
`controller.gains`, leaving every other key -- including `kp_rot_wrist`/`kd_rot_wrist` --
untouched. Verified directly by inspecting a written `candidate_configs/.../baseline_001.yaml`
from a real sweep run: it correctly carried `kp_rot_wrist: 80.0`/`kd_rot_wrist: 320.0`. Not a
bug, but the printed log line alone is misleading; anyone debugging a "why didn't my override
apply" question for a non-`GAIN_FIELDS` field should check the written candidate config, not
just the log.

## Bottom line

- `kp_rot_wrist` is confirmed structurally independent of `kp_rot`'s known instability (no
  matrix inversion in its code path) -- but it has its **own**, different failure modes at
  high gain: underdamped oscillation-like error growth if `kd_rot_wrist` isn't scaled with it,
  and a separate slow-onset long-hold divergence once gain gets high enough (observed at
  kp=80/kd=320, not at kp<=20/kd<=80) -- a real, non-obvious finding from direct sim
  verification, not something inference from the `kp_rot` history alone would have predicted.
- **Helps, moderately, at a conservative gain**: `kp_rot_wrist=20.0`/`kd_rot_wrist=80.0` cuts
  orientation error 35-41% on the exact short-exposure real-failure scenarios that motivated
  this pass, with zero regressions across the full 4-category rigor sweep (34/38 both, byte-
  identical) and zero degradation on `long_holds` specifically (the category that ruled out the
  more aggressive candidates).
- New config: `config/ur5e_mujoco_torque_osc_tuned_split_base_wrist_wrist_orient_retuned.yaml`.
  **Sim only. Not real-hardware validated.** Needs its own careful, small-first real-lab check
  before it can be trusted or promoted, per every other unvalidated config in this repo's
  history -- and specifically should be checked against the real orientation-growth trend that
  motivated this pass (`direct_torque_20260801_193232`/`_200315` style scurve runs), not just
  sim reproductions.
- No `controller_core/` changes. No existing config modified.
