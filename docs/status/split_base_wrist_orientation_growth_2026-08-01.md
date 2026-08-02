# Why orientation error grows faster at higher accel with `split_base_wrist_task`

Diagnostic investigation, 2026-08-01. Sim-only (no real-hardware access used). Scope: root
cause only, per this repo's own rule not to combine controller changes/retunes with a
diagnosis pass. No `controller_core/`, config, or production files modified. No new named
configs added. One diagnostic script kept (see bottom).

## Background

Tonight's real-hardware validation of `config/ur5e_mujoco_torque_osc_tuned_split_base_wrist.yaml`
(`accel_duration_scurve`, height_alpha=0.5 zero-degree pose) found:
- `target_accel=0.005, move_duration=8.0s`: clean full pass, `jacobian_cond` 7.8-11.8
  throughout, orientation error negligible (~0.00038 rad) during hold.
- `target_accel=0.02, move_duration=4.0s`: TCP-speed guard trip at ~43% of target
  displacement, driven by real, monotonic orientation-error growth (~0.028 rad by trip),
  not sensor noise (confirmed by the same-night speed-smoothing check).

Two hypotheses were posed: (1) `split_base_wrist_task`'s reduced-Jacobian nullspace
projector has genuinely less orientation-restoring authority than the old full-J one; (2) a
real dynamics/friction effect independent of tonight's fix, that any config would show when
pushed this hard.

## Step 1 — full real trace analysis (not just the last 60 cycles)

Real runs identified from `outputs/hardware_transport_remote/hardware_transport/summary.json`
(`config_path` == `config/ur5e_mujoco_torque_osc_tuned_split_base_wrist.yaml`):
- `direct_torque_20260801_193036` — accel=0.005/dur=8.0, `termination_reason:
  duration_complete`, 5000 cycles.
- `direct_torque_20260801_193232` — accel=0.02/dur=4.0, `termination_reason: "TCP speed
  0.0539 m/s > 0.05 m/s for 3 consecutive cycles"`, 1039 cycles (the original trip).
- `direct_torque_20260801_200315` — same scenario, retested after the speed-guard smoothing
  fix, tripped at 0.0507 m/s, 1036 cycles (confirms the smoothing only cleaned up the
  reported value, per tonight's own note).

Full-trace `orientation_error_norm` and `q[4]` (wrist_2):

| run | t | oe (rad) | wrist_2 (rad) | jacobian_cond |
|---|---|---|---|---|
| clean (8s) | 0.000 | 0.00008 | 0.000011 | 7.83 |
| clean (8s) | 2.856 | 0.00837 | -0.000084 | 8.24 |
| clean (8s) | 5.712 | 0.04630 | -0.000076 | 11.01 |
| clean (8s) | 9.998 (end) | **0.05392** | -0.000060 | 11.81 |
| failing (4s) | 0.888 | 0.00030 | -0.000016 | 7.84 |
| failing (4s) | 1.630 | 0.01824 | 0.000002 | 8.83 |
| failing (4s) | 2.076 (trip) | **0.02854** | -0.000024 | 9.53 |

Key finding, not anticipated going in: **the "clean" 8s run also shows orientation error
growing essentially monotonically, reaching 0.054 rad by the end — nearly double the
failing run's trip-point value of 0.028 rad.** It never trips only because the run ends
(`duration_complete`) before the trend crosses the speed-guard's derived threshold, and
because growth is decelerating late (last ~1.4s: +0.0003 rad only) — an asymptotic
approach to a nonzero plateau, not a return to zero. `jacobian_cond` stays bounded 7.8-11.8
throughout both runs — the split-Jacobian singularity fix itself holds completely; this is
a separate, secondary phenomenon. `wrist_2` stays pinned within ±0.00013 rad of exactly 0 in
both real traces — the real arm essentially never leaves the singular configuration, unlike
sim (see Step 2).

**Conclusion from real data alone**: growth-vs-accel isn't really the story — growth vs.
*elapsed/exposure time* is closer to what's observed (comparable orientation error at
comparable fractional move progress between the two runs, e.g. ~0.018-0.021 rad at ~45-52%
of each move). The higher-accel run trips sooner mainly because a real, always-present
growth trend gets less time to matter in the slower run before `duration_complete` cuts it
off — not because higher accel produces qualitatively faster growth. This reframes the
original question: the real puzzle isn't "why does higher accel grow orientation error
faster" so much as "why does this config's orientation error grow at all, continuously,
without correcting" — which is what Steps 2-4 investigate.

## Step 2 — sim reproduction

Ran `tools/ur5e_mujoco_torque_experiments.py --mode controller-rollout --controller-kind
impedance` with `config/ur5e_mujoco_torque_osc_tuned_split_base_wrist.yaml`, start pose
`[0.0, -0.835398, -1.2, -0.985398, 0.0, 0.0]`, `accel_duration_scurve`, at both real
parameter points (needed the `mujoco_ur5e` conda env's Python — the base env lacks
`pinocchio`, required by this config's `gravity_source: pinocchio`/
`coriolis_feedforward: true`).

Sim **does** qualitatively reproduce a rise-then-plateau orientation-error pattern (not a
full return to zero) for both scenarios — reading the raw per-cycle `orientation_error_norm`
directly, not the `hold_phase_max_abs_orientation_error_rad` summary field, which measures
drift *since hold start*, not absolute error, and is misleadingly small (0.002-0.006 rad)
once the error has already plateaued by hold start:

| scenario | sim oe at move end | sim oe plateau (t=6s) |
|---|---|---|
| accel=0.005/dur=8 (clean) | 0.0795 (peak, mid-move) | 0.00063 *(this one — the real
  8s run's hold is long enough that oe genuinely settles low; see caveat below)* |
| accel=0.02/dur=4 (failing), split config | 0.0685 (move end) | 0.0707 |

Caveat/honest gap: the accel=0.005/8s sim run's orientation error *does* decay close to
zero by end of its longer hold — genuinely different from the real run's non-decaying
0.054 rad plateau at the same scenario. Sim's friction/damping model apparently lets this
particular (gentle) case fully recover where real hardware doesn't; the accel=0.02/4s case,
with less hold time, plateaus at a nonzero value in both sim and real. So: **sim reproduces
the qualitative failure-scenario pattern (rise-then-plateau-nonzero) but not the clean
scenario's real non-decaying residual** — a genuine, partial sim/real gap, consistent with
this repo's documented history of sim under-representing some real friction/drift
mechanisms (e.g. the `friction_feedforward`/asymmetric-Coulomb work in AGENTS.md sec 3).

## Step 3 — direct hypothesis-2 test: split vs baseline at the identical scenario

Ran the identical accel=0.02/dur=4.0 scenario in sim with
`config/ur5e_mujoco_torque_osc_tuned_wrist_orient_friction_ff.yaml` (`split_base_wrist_task:
false`, otherwise the base this config builds on) instead of the split config:

| config | move-phase max oe (rad) | oe plateau at t=6s (rad) | max `\|tau_posture[wrist_2]\|` (Nm) | `jacobian_cond` range |
|---|---|---|---|---|
| split (tonight's fix) | 0.0685 | 0.0707 | 0.00087 | 7.8-13.3 (bounded) |
| baseline (no split) | 0.0647 | 0.0668 | **0.0115 (13x larger)** | 4.36e16 → 1.04e4 (starts singular, decays as wrist_2 drifts) |

**The orientation-error outcome is nearly identical between split and baseline (~6% higher
plateau for split) — this does not support hypothesis 1 as the dominant cause.** Both
configs show essentially the same rise-then-plateau curve at this scenario in sim.

However, the `tau_posture[wrist_2]` breakdown (real controller output, not synthetic) shows
a large, real, and directly-measured mechanism difference: under the baseline (full-J,
near-singular) config, the nullspace-posture projector routes **13x more restoring torque**
through wrist_2 than the split config does, despite wrist_2's own raw PD term being tiny in
both cases (wrist_2 barely deviates from `q_rest`, sim: -0.000089 rad split vs +0.000358 rad
baseline). This is cross-coupling: near the exact singularity, the projector's eps=0.1
regularization redirects some of the *base joints'* larger posture-correction torque through
wrist_2, which the exact/hard base-wrist split (by construction) cannot do — wrist_2's
projector row is an exact identity pass-through of its own raw PD term when
`split_base_wrist_task` is on (see Step 4).

## Step 4 — the nullspace-projector mechanism, isolated

Read `controller_core/x_axis_cartesian_impedance.py` lines ~789-870 directly (not modified).
When `split_base_wrist_task` is on:
```python
J_task = np.zeros((3, 6))
J_task[:, 0:3] = J[0:3, 0:3]   # position rows, base-joint columns only; wrist cols exactly 0
...
j_bar = m_inv @ J_task.T @ lambda_mat_nullspace
nullspace_proj = np.eye(6) - J_task.T @ j_bar.T
tau_posture = nullspace_proj @ tau_posture
```
Because `J_task`'s wrist columns are structurally zero, `J_task.T`'s wrist *rows* are zero,
so `nullspace_proj`'s wrist-joint rows reduce algebraically to an exact 3x3 identity block —
verified both analytically and numerically (`tools/../nullspace_authority_probe.py` below):
`||proj[wrist_rows]||_F == sqrt(3)` exactly, and `tau_posture_out[wrist] ==
tau_posture_in[wrist]` bit-for-bit, at every q tested. Wrist joints get their *raw,
unprojected* `kp_posture*(q_rest-q) - kd_posture*qd` term — no more, no less — regardless of
how the base-joint task consumes the rest of the projector. Meanwhile base joints, whose
`J_task` submatrix `J[0:3,0:3]` is well-conditioned (cond ~7.8-13 throughout both real and
sim traces), retain substantial (not zero, contrary to the initial hypothesis) nullspace
authority of their own — a static probe at the transport pose found base-row
`||proj||_F` actually slightly *higher* for split (1.35) than baseline (1.47 at the
near-singular pose, comparable order), so base-joint posture authority is not meaningfully
reduced by the split either.

**Net effect of the split, precisely stated**: it does not remove authority from base
joints. It removes a specific *cross-coupling leak* that the old near-singular full-J
projector had — extra, amplified (here 13x, and sign-differing in a synthetic static
example) restoring torque arriving at wrist_2 as a side effect of the regularized
pseudo-inverse near `cond(J)~1e16`, not as a designed mechanism. `split_base_wrist_task`
closes that leak by construction (that's the point: task torque and, now demonstrably,
*some* posture-projector cross-talk can no longer route through the near-singular wrist
sub-block). The magnitude difference is real and directly measured (13x on `tau_posture[w2]`
in this exact reproduction), but its effect on the actual orientation-error outcome in sim
is modest (~6% plateau increase) — evidence it's a real, secondary contributing factor, not
the dominant driver of the observed growth.

## Conclusion — neither hypothesis cleanly wins; here's what the evidence actually shows

- **Not primarily hypothesis 1** as originally framed ("the reduced Jacobian removes
  restoring authority"): base-joint authority is not reduced, and split vs baseline produce
  nearly identical orientation-error trajectories in sim at the exact failing scenario.
- **A real, verified, secondary mechanism from `split_base_wrist_task` does exist**: it
  closes a near-singularity cross-coupling leak that used to route extra (13x) restoring
  torque through wrist_2 via the nullspace posture projector. This is a genuine, quantified
  authority reduction specific to wrist_2 — just not the dominant driver of the outcome
  measured here.
- **Mostly hypothesis 2, with a real gap**: orientation error grows similarly whether or not
  `split_base_wrist_task` is on, in both configs, at this pose — a property of holding
  orientation at this pose with `kp_rot=0`/`kp_rot_wrist=0` (damping-only rotational
  authority; the *only* proportional/restoring channel is `tau_posture`, whose overall
  magnitude here is small relative to what's needed) more than a regression introduced
  tonight. The real-hardware clean (8s) run's *non-decaying* 0.054 rad plateau — which sim
  does NOT reproduce for that specific gentle scenario — is the biggest honest open gap: it
  suggests a real friction/stiction-like effect (in the same family as this repo's
  documented `friction_feedforward`/asymmetric-Coulomb findings) that the current sim
  friction model under-represents specifically for orientation-holding, separate from
  anything `split_base_wrist_task` changed.

## Pointer for a future retune (not attempted here)

If a fix is scoped later, the evidence above points at **giving orientation-holding at this
pose a real, non-damping-only proportional channel that doesn't depend on the near-singular
nullspace-projector cross-coupling `split_base_wrist_task` correctly removed** — most
directly, raising `kp_rot_wrist` (currently 0.0, damping-only, in
`config/ur5e_mujoco_torque_osc_tuned_split_base_wrist.yaml`) so `wrist_orientation_task`
supplies a genuine P-term through the wrist chain, independent of `tau_posture`'s now-honest
(un-leaked) authority. This needs its own gain search and 4-category rigor sweep per this
repo's established validation pattern — not attempted here per this task's diagnostic-only
scope. A second, lower-confidence angle worth checking first: whether the real clean-run's
non-decaying plateau (which sim didn't reproduce) is itself a friction-model gap rather than
a controller-authority gap — if so, a retune alone won't close it.

## Files

- Diagnostic script (kept, ad hoc, not a named permanent config):
  `/common/home/ss5772/.tmp/claude-1905239669/-common-users-ss5772-real-Cartpole/0ae9bccf-1721-4928-b3ae-540681a11bc1/scratchpad/nullspace_authority_probe.py`
  and `compare_sim_traces.py` — both read-only probes against `controller_core`/sim traces,
  live in the session scratchpad (not the repo) since they were purely diagnostic.
- Real traces read in full: `outputs/hardware_transport_remote/hardware_transport/
  direct_torque_20260801_193036/{summary.json,trace.jsonl}`,
  `direct_torque_20260801_193232/{summary.json,trace.jsonl}`,
  `direct_torque_20260801_200315/{summary.json,trace.jsonl}` (gitignored, not committed).
- Sim reproduction runs (not committed, `outputs/`-equivalent scratch data): three
  `controller-rollout` runs via `tools/ur5e_mujoco_torque_experiments.py`, `mujoco_ur5e`
  conda env, `MUJOCO_GL=egl`.
- No `controller_core/`, config, or other production files modified.
