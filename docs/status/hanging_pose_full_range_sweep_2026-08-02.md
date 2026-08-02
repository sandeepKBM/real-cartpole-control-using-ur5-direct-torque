# Hanging-pose family: full `height_alpha` range validation vs the old family (2026-08-02)

## Why this exists

`docs/status/hanging_pose_transport_family_2026-08-01.md` built and validated the "hanging"/
elbow-down pose family (`hardware/poses.py::HANGING_ORIGIN_Q`/`HANGING_LOWER_Q`/
`q_for_hanging_height_alpha`) at ONE point only: `height_alpha=0.5`, reaching 36/38 (94.7%)
with `config/ur5e_mujoco_torque_osc_hanging_pose_friction_ff.yaml`. The old pose family
(`q_for_height_alpha`) has historically been characterized at `height_alpha ∈ {0.1, 0.2, 0.3,
0.5}` (e.g. the `singular_scale` promotion validation, AGENTS.md sec 4). This doc runs the
identical 4-category rigor sweep for the hanging family at those same four alphas, checks that
`cond(J)` stays well-conditioned dynamically at each one (not just statically, and not just at
0.5), and reports an honest before/after against the old family's own documented numbers —
filling in one gap in the old family's own record (height_alpha=0.1 with friction feedforward
had never been run) so the comparison is complete.

**Bottom line up front**: the hanging family is ahead in aggregate (146/152 = 96.1% vs the old
family's 144/152 = 94.7%, both with `friction_feedforward` against the current friction-modeled
plant) and the singularity-avoidance property (`cond(J)` in the single-to-low-double digits)
holds dynamically at every alpha tested. But the advantage is **not uniform**: at
`height_alpha=0.2` the hanging family is clearly *worse* than the old family in one specific
category (`large_displacements`, 4/8 vs 8/8) via a genuine `|Z-Z0| > 0.03 m` guard trip that is
unrelated to conditioning (`cond(J)` measured 11.9-15.3 throughout that same failing run). This
does not generalize from the strong `alpha=0.5` result and should be treated as a real,
alpha-specific weak point, not noise.

## 1. Method

- Tooling: the shipped `tools/ur5e_pose_sweep_transport.py` hardcodes `hardware.poses.
  q_for_height_alpha` (the OLD family only) with no pose-family flag, so it cannot run the
  hanging family directly. Rather than modify that script (out of scope — "validation, not
  further construction"), a validation-only driver was used that mirrors it exactly (same
  `CATEGORY_GRIDS`, same per-category subprocess-to-`tools/ur5e_move_hold_transport.py`
  structure, same gain-extraction-from-config logic) but calls `q_for_hanging_height_alpha`
  instead and passes the result via `--start-q-rad`. This driver was **not added to the repo**
  (it lives outside the worktree, in the session scratchpad) — it is a thin, disposable wrapper
  around already-existing, unmodified project tooling (`tools/ur5e_move_hold_transport.py`),
  not a new capability. No repo file was changed to produce these results.
- Config: `config/ur5e_mujoco_torque_osc_hanging_pose_friction_ff.yaml` for all four alphas
  (its own `home_qpos` is the alpha=0.5 midpoint, but `--start-q-rad` overrides it per run,
  identical to how `ur5e_pose_sweep_transport.py` overrides the old family's config `home_qpos`
  for each swept alpha).
- Categories, seed=0, gains taken verbatim from the config (`--gain-overrides-json`, no
  retuning) — identical to sec 3 of the 2026-08-01 build doc and to every other 4-category
  sweep in this repo's history.
- `uptime`/`nproc` checked before launch (load average ~2-9 on 72 cores); the four alphas were
  run in parallel (`OPENBLAS_NUM_THREADS=1` etc. set per AGENTS.md sec 8 guidance) since
  headroom was ample; total wall time ~14 minutes.
- Old-family comparison numbers were taken from already-existing docs (cited per row below), not
  re-derived from memory or estimated. One gap was found and filled: no doc anywhere runs the
  old family's `friction_ff` config at `height_alpha=0.1` post-friction-model. That single cell
  was filled by running the existing, unmodified `tools/ur5e_pose_sweep_transport.py
  --height-alphas 0.1 --config config/ur5e_mujoco_torque_osc_tuned_friction_ff.yaml` (no new
  script, no config change) so the comparison set is complete across all four alphas.

## 2. Results — 4-category rigor sweep, `friction_feedforward` config, both families

| alpha | family  | canonical_grid | long_holds | large_disp | torque_scale | **total** | source |
|---|---|---|---|---|---|---|---|
| 0.1 | hanging | 8/8 | 8/8 | 8/8 | 14/14 | **38/38 (100%)** | this sweep |
| 0.1 | old     | 8/8 | 8/8 | 8/8 | 14/14 | **38/38 (100%)** | this sweep (gap-fill; not previously documented anywhere) |
| 0.2 | hanging | 8/8 | 8/8 | **4/8** | 14/14 | **34/38 (89.5%)** | this sweep |
| 0.2 | old     | 8/8 | 8/8 | 8/8 | 14/14 | **38/38 (100%)** | `docs/status/friction_ff_alpha_0.2_0.3_sweep_2026-07-31.md` |
| 0.3 | hanging | 8/8 | 8/8 | 8/8 | 14/14 | **38/38 (100%)** | this sweep |
| 0.3 | old     | 7/8 | 8/8 | 8/8 | 12/14 | **35/38 (92.1%)** | same doc |
| 0.5 | hanging | 8/8 | 8/8 | 8/8 | 12/14 | **36/38 (94.7%)** | 2026-08-01 build doc (reproduced identically this run) |
| 0.5 | old     | 7/8 | 8/8 | 6/8 | 12/14 | **33/38 (86.8%)** | `docs/status/ur5e_sim_friction_modeling_2026-07-31.md` sec 3.1; also AGENTS.md sec 3 |
| **sum** | hanging | 32/32 | 32/32 | 28/32 | 54/56 | **146/152 (96.1%)** | |
| **sum** | old     | 30/32 | 32/32 | 30/32 | 52/56 | **144/152 (94.7%)** | |

Aggregate: hanging family ahead by 2/152 points (96.1% vs 94.7%). **Not a landslide, and not
uniform** — see honest breakdown below.

## 3. Per-alpha verdict — honest, not just the aggregate

- **alpha=0.1**: tie, both families clean (38/38). No advantage either direction here.
- **alpha=0.2**: **hanging family is worse**, by a full category. `large_displacements`:
  `dx=0.05` and `dx=0.10` pass at both hold durations; `dx=0.15` and `dx=0.20` fail at both hold
  durations via `failure_reason: "|Z-Z0| > 0.03 m"` — a genuine `CartesianMoveMonitor`/safety
  guard trip, not a tracking-tolerance miss. Checked whether this is a conditioning problem:
  it is not — `jacobian_cond` in the failing `dx=0.20` trace stays 11.91-15.27 throughout (312
  steps), same healthy range as every passing run. This is a real, alpha-specific reach/Z-hold
  limitation of the hanging pose at this height, independent of the singularity-avoidance
  property the family was built for.
- **alpha=0.3**: hanging family ahead (38/38 vs 35/38); old family's shortfall here
  (`canonical_grid` 7/8, `torque_scale_robustness` 12/14) is the same friction-undershoot
  signature documented for the old family generally, unrelated to this comparison.
- **alpha=0.5**: hanging family ahead (36/38 vs 33/38), reproducing the 2026-08-01 build doc's
  headline result exactly (same category breakdown: canonical 8/8, long_holds 8/8, large_disp
  8/8, torque_scale 12/14).

**Conclusion**: the hanging family's strong alpha=0.5 result does **not** generalize uniformly
across the range. It wins clearly at 0.5, wins clearly at 0.3, ties at 0.1, and **loses** at 0.2
via a distinct, guard-trip-confirmed (not conditioning-related) large-displacement Z-drift
failure. Anyone picking this family for a specific `height_alpha` should check that alpha's own
number, not assume the 0.5 result transfers.

## 4. `cond(J)` dynamic confirmation across all four alphas — the core design claim

The 2026-08-01 build doc's dynamic confirmation (closed-loop `jacobian_cond` trace, not just
static FK) was previously done only at alpha=0.5 (8.91-11.17 across a dx=0.20m run). Extracted
the same field from this sweep's own `large_displacements` dx=0.20 trace at every alpha (no
extra runs needed — reused the rigor-sweep traces already produced):

| alpha | jacobian_cond range (dynamic, dx=0.20m run) | steps | run outcome |
|---|---|---|---|
| 0.1 | 13.38 – 37.20 | 1500 | pass |
| 0.2 | 11.91 – 15.27 | 312 | **fail** (Z-drift guard, unrelated to cond(J)) |
| 0.3 | 10.77 – 15.83 | 1500 | pass |
| 0.5 | 8.91 – 11.17 | 1000 | pass |

(alpha=0.5's range reproduces the build doc's own number exactly, a useful sanity check on
methodology consistency.) **The singularity-avoidance property holds dynamically at every alpha
checked** — `cond(J)` never leaves the single-to-low-double-digit range anywhere, including
during the one failing run (alpha=0.2, dx=0.20m), confirming that failure is a genuine
reach/Z-hold limit of that specific pose+displacement, not a re-emergence of the wrist
singularity the family was designed to avoid. This is 14-16 orders of magnitude better than the
old family's documented 1e16-2.5e17 range at the identical model, at every alpha, not just 0.5.

## 5. What this changes about the 2026-08-01 build doc's verdict

Section 4 of that doc ("Honest verdict") explicitly flagged "validation across the full
`hanging_alpha` range rather than just the midpoint" as not yet done. That gap is now closed:
the core `cond(J)` claim holds everywhere tested, and the aggregate pass-rate advantage survives
(146/152 vs 144/152). But the practical takeaway is now more nuanced than "36/38, at or above
every historical number this repo has for the old family" (true only at alpha=0.5) — at
alpha=0.2 specifically, the old family is still the better choice for large displacements. Sim
only; still no real-hardware validation of the hanging family at any alpha (unchanged from the
2026-08-01 doc's own explicit real-hardware-readiness section — nothing here changes that).

## 6. Files changed / not changed

- No `hardware/poses.py`, `rl_gain_scheduling/gain_scheduling_env.py`, or existing config
  modified, per task constraints.
- No new config added — all four alphas reused
  `config/ur5e_mujoco_torque_osc_hanging_pose_friction_ff.yaml` via `--start-q-rad` overrides,
  exactly mirroring how the old family's own `ur5e_pose_sweep_transport.py` sweeps a single
  config across multiple alphas.
- No new file added to `tools/`; the sweep wrapper used to drive the hanging family across
  alphas lived only in the session scratchpad, not the repo.
- This doc (new).
- `outputs/ur5e_mujoco_torque_transport/hanging_pose_full_range_sweep_2026-08-02/` (gitignored,
  not tracked) holds all `summary.json`/`trace.jsonl`/`alpha_summary.json` artifacts backing
  every number above, including the old-family alpha=0.1 gap-fill run.

## Tests run

- `python -m pytest -q -k hanging` — 8 passed (pre-existing hanging-pose-family tests,
  unaffected by this validation-only pass).
- `python -m pytest -q` (full suite) — 603 passed, 3 xfailed (pre-existing, unrelated), zero
  regressions.

## Tests not run

- No real-hardware or URSim tests (sim-only task, no hardware access this session).

## Rollback

This change is additive (one new status doc only; no repo pose/config/tool file touched).
`git revert <this commit's hash>` removes the doc with no effect on any other file.
