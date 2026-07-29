# Bloat/trim audit — 2026-07-29 (post-hardware-rewrite / RL gain-scheduling growth)

Context: the prior cleanup pass (`docs/archive/AUDIT_REPORT.md`, `docs/archive/BLOAT_REPORT.md`,
2026-07-03) is archived and mostly executed; AGENTS.md §6 lists what's already correctly
archived (`archive/coppelia/`, `archive/legacy_mujoco/`, `archive/superseded/`). This pass
covers what accumulated since, especially 2026-07-25 through 2026-07-29 (hardware lane rewrite,
RL gain-scheduling ~12-config family, new `docs/status/` dated investigations, a new skill).
Method: `git log --follow` per candidate file, cross-repo grep for references (docs/tests/tools),
and spot-diffs between near-duplicate configs. Report-only — nothing deleted or moved, per
AGENTS.md's rule against touching checkpoints/datasets/logs/generated artifacts without explicit
request.

**Live-repo caveat**: another process was actively editing `AGENTS.md`, `controller_core/
torque_task_qp.py`, `controller_core/x_axis_cartesian_impedance.py`, `docs/CURRENT_STATUS.md`,
and `docs/hardware/AUTO_TUNING_PLAN.md` while this audit ran (confirmed via repeated `git
status` — the working tree changed mid-session). This audit did not touch any of those files
and does not evaluate their in-flight content; treat any findings below that reference them as
based on the version seen at read time, which may already be stale.

## Priority 1 — safe to archive, low risk

- **`rl_gain_scheduling/_scratch_accel_profile_demo.py`** (243 lines, added in `084440c`,
  "Add out-of-distribution stress-test script for the gain-scheduling policy"). This is not an
  accidentally-committed scratch file — it's a deliberate one-off diagnostic, and its own
  docstring says so explicitly: "this is a one-off stress test of an out-of-distribution target
  shape, not a permanent feature." It has served its documented purpose (used once to validate
  `reward_v3_2M`, narrated in `docs/CURRENT_STATUS.md` lines 143-156) and has no test coverage
  and no other caller. Candidate for `archive/superseded/` alongside the other one-off tuning
  scripts already there — would preserve git history and match the project's existing pattern
  for "real experiment, done being active."

## Priority 2 — borderline, needs a human call

- **The `config/rl_gain_scheduling_alpha05_bidirectional*.yaml` chain (5 files, one committed
  today: `..._safety_fix.yaml`)**. On the surface these look like near-duplicate configs
  (pairwise diffs are 61-156 lines each), but git history + `docs/status/
  rl_gain_scheduling_alpha05_directional_fix_2026-07-29.md` confirm each is a genuinely distinct,
  real training attempt in a documented supersession chain, not redundant copies:
  `bidirectional.yaml` (base, orientation-tolerance mismatched, not one of the "real" attempts)
  → `_adaptive_lambda.yaml` (fixed to match the documented bug, attempt 1, 0/8)
  → `_adaptive_lambda_residual.yaml` (residual-torque action mode, attempt 2, 2/8)
  → `_adaptive_lambda_residual_penalized.yaml` (+ magnitude penalty, attempt 3, 5/8, "best
  performing prior config")
  → `_adaptive_lambda_residual_penalized_safety_fix.yaml` (orientation-threshold bug fixed,
  attempt 4, 0/8 — regression, documented as "do not adopt"). This matches AGENTS.md §7's
  explicit rule ("preserve old configs — add new named configs instead of mutating shared
  ones"), so keeping all five is the intended behavior, not bloat. **The human call**: none of
  the four real trained policies beat the fixed-gain baseline (7/8), and the doc's own
  recommendation is "do not adopt this policy" / "further blind iteration is not recommended
  without first designing a stronger positive incentive." If gain-scheduling RL is shelved for
  now (as `docs/CURRENT_STATUS.md` suggests), it would be reasonable to add a one-line
  "historical, superseded — see docs/status/rl_gain_scheduling_alpha05_directional_fix_2026-07-29.md"
  banner comment at the top of the four now-dead-end configs, the same convention used for
  archived reports — but that's a documentation-style call, not a bloat problem, and I did not
  make this change since it touches files outside the explicitly-approved additive-doc-fix scope.
- **`docs/CURRENT_STATUS.md` staleness**: its header claims "Last updated: 2026-07-29" but large
  parts of the body (the "Next" section in particular) were, at read time, still asserting facts
  contradicted by later-dated material already in the repo — e.g. "Hardware lane is code-complete
  ... has never touched a real robot — first physical contact is the next real-world milestone,"
  which is stale given `AGENTS.md`'s own 2026-07-28 section documents real hardware runs
  (`hardware_captures/2026-07-28_thinkrobot_172.16.71.77/`). This file was being actively edited
  by another process during this audit (see caveat above), so it may already be corrected by the
  time this is read — recommend a human re-check rather than assuming this finding still holds.
- **`hardware/local_dynamics.py` naming collision**: `LocalPinocchioDynamics = LocalMujocoDynamics`
  is a deliberate, documented back-compat alias (per `docs/status/
  local_dynamics_speedup_investigation_2026-07-29.md`) — the name says Pinocchio but the code is
  MuJoCo. A *different*, newly-added, genuinely Pinocchio-backed path is activated via
  `dynamics_source="local_pinocchio"` (a string, not the class name). Both are real, both are
  tested (`tests/hardware/test_local_dynamics.py`, `tests/hardware/test_local_pinocchio_dynamics.py`),
  neither is dead code — but the same word "Pinocchio" now means two different things in the same
  file, one real, one a MuJoCo alias kept for compatibility. Not bloat, but a readability landmine
  worth a maintainer's attention next time this file is touched.

## Checked and found to be legitimate (false positives, noted so this doesn't get re-flagged)

- **`config/rl_gain_scheduling_reward_v2.yaml` / `_v3.yaml` / `_v4_height0.5.yaml`**: look
  superseded-in-place by the later `_alpha05_bidirectional*` family, but they cover a different,
  earlier phase of the RL work (documented in `docs/CURRENT_STATUS.md` lines 117-215) and are
  still referenced by that doc plus `rl_gain_scheduling/_scratch_accel_profile_demo.py`. Kept
  per the same config-preservation convention — legitimate history, not orphaned.
- **`tools/_rtde_control_probe.py`, `_check_ursim_remote.py`, `_clear_dashboard_mode_lock.py`,
  `_direct_torque_pulse_test.py`, `_ursim_wait_and_power_on.py`**: the leading-underscore naming
  looks like scratch/private scripts, but all five are actively referenced from
  `docs/hardware/{HARDWARE_GUIDE,README,URSIM_REMOTE_CONTROL,POLYSCOPE_PANEL_CHECKLIST}.md` as
  documented bring-up/debug utilities. Legitimate, not scratch.
- **`.agents/skills/remote-compute-background-jobs/`**: reads as generic advice at a glance, but
  it's specific, hard-won operational lore (cgroup memory caps, `systemd-logind` Linger
  reverting, `pkill` self-matching) tied to this exact cluster (`westeros`, `ilab1-4`). Legitimate,
  keep.
- **Logging duplication** (flagged in the old `BLOAT_REPORT.md` §3): already resolved.
  `hardware/logging.py` is now a thin re-export shim (`from controller_core.logging_utils import
  ...`), not a parallel implementation. No action needed.
- **Tuning-script duplication** (also flagged in the old report): already resolved. The
  predecessor `tune_ur5e_impedance_transport.py` lives in `archive/superseded/`; only
  `tools/tune_ur5e_residual_impedance_transport.py` remains active, matching AGENTS.md §2.
- **`environment.yml` vs. Pinocchio usage**: `pin>=3.1` is declared and matches the heavy new
  Pinocchio usage across `controller_core/model_dynamics.py`, `hardware/local_dynamics.py`, and
  the new residual observer. No manifest drift found.
- **`hardware_captures/2026-07-28_thinkrobot_172.16.71.77/`**: only 7 tracked files, 88K total —
  real hardware capture data with its own README, not a stray dump. Legitimate.
- **`hardware/direct_torque_transport.py`'s residual observer** (`enable_residual_observer`,
  diagnostic-only, default `True`): has real test coverage across
  `tests/mujoco/test_direct_torque_residual_observer.py`,
  `tests/hardware/test_direct_torque_residual_observer_trace.py`,
  `tests/unit/test_dynamics_residual.py`, `tests/hardware/test_joint_accel_estimator.py`. Not
  dead code.
- **Doc cross-reference spot-check** (regex-extracted backtick-quoted filenames from all six
  `docs/status/*.md` files, checked for existence): every genuine hit resolved to a real file
  once given its correct directory prefix (e.g. `direct_torque_transport.py` →
  `hardware/direct_torque_transport.py`, `eval_gain_scheduler.py` →
  `rl_gain_scheduling/eval_gain_scheduler.py`). No broken path references found in this sample.
  Caveat: this was a basename-level regex check, not exhaustive — treat as a spot-check, not a
  full audit.
- **No stray tracked binaries/artifacts**: `git ls-files` under `outputs/`/`reports/` is empty
  (both correctly gitignored); the only tracked image files repo-wide are the two vendored
  `mujoco_menagerie` robot preview PNGs, which are legitimate vendored assets.

## Not evaluated (out of scope / too volatile to assess mid-session)

- `AGENTS.md`, `controller_core/torque_task_qp.py`, `controller_core/x_axis_cartesian_
  impedance.py`, `docs/hardware/AUTO_TUNING_PLAN.md` — actively being edited by another process
  during this audit; no findings recorded against their current content.
- `outputs/` disk usage (gitignored, previously flagged in `docs/CURRENT_STATUS.md`'s own "Next"
  section as needing a regenerated purge list) — out of scope for a git-tracked-bloat audit, and
  explicitly a human/`rm -rf` decision per that doc and per AGENTS.md's own restriction on
  touching generated artifacts.

## Summary

One clear, low-risk archive candidate (`rl_gain_scheduling/_scratch_accel_profile_demo.py`).
One borderline documentation-convention question (whether to banner the four dead-end
`alpha05_bidirectional*` RL configs as historical) that a human should decide, not a bloat
problem to fix. One doc-staleness flag on `docs/CURRENT_STATUS.md` that may already be
self-resolving given concurrent edits observed during this audit. Everything else checked —
the RL config family sprawl, underscore-prefixed hardware probe scripts, the new skill, the
Pinocchio naming, prior-flagged logging/tuning duplication, hardware capture data, and the new
residual observer — turned out to be legitimate, already-resolved, or working as intended.
