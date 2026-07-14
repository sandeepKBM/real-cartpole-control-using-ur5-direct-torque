# Current Status

Last updated: 2026-07-07 (hardware lane rewrite + RL gain-scheduling training/eval).

## Active Objective

**MuJoCo UR5e gravity-compensated residual-torque transport** in simulation, using a
Pinocchio-backed model-based controller (operational-space, tuned). Active lane:

```text
tools/ur5e_mujoco_torque_experiments.py    # rollout engine (owns the step loop)
tools/audit_ur5e_mujoco_gravity_torque.py  # gravity-sign / hold-quality audit
tools/ur5e_move_hold_transport.py          # move+hold sweep driver
tools/ur5e_x_frame_envelope.py             # transport envelope sweep
tools/tune_ur5e_residual_impedance_transport.py
tools/compare_ur5e_mujoco_controllers.py
simulation/ur5e_mujoco_torque.py           # MuJoCo adapter
controller_core/model_dynamics.py          # Pinocchio DynamicsProvider
```

Residual torque means `tau_applied = tau_controller + tau_gravity (+ tau_coriolis)`.
`tau_gravity` defaults to MuJoCo `qfrc_bias` but can source from Pinocchio
(`mujoco.gravity_source: pinocchio`); Coriolis feedforward is a separate opt-in flag. Raw
mode is validation/anti-cheating only.

## Current diagnosis: resolved, tuned, and validated

The 2026-07-01 diagnosis (`docs/archive/DIAGNOSTIC_real_cartpole_torque_control_questions.md`)
found the impedance controller structurally too simple — no mass matrix, no Coriolis
compensation, no inverse dynamics — combined with tight safety envelopes that tripped before
it settled. That diagnostic's recommended staged fix has since been fully implemented and its
outcome tuned well past parity:

- **P0-P2 (Pinocchio dynamics, flag-gated, default = legacy behavior)**: gravity source swap
  and Coriolis feedforward, both validated against MuJoCo to tight tolerances.
- **P3 (operational-space upgrade, flag-gated)**: task-space inertia shaping (Λ(q)) +
  dynamically consistent nullspace posture projection.
- **Tuned OSC config** (`config/ur5e_mujoco_torque_osc_tuned.yaml`): ~250+ evaluation runs
  found and fixed two root causes at the transport start pose's wrist singularity — a
  task-unactuatable drift direction (fixed by a joint-space posture anchor + re-anchoring)
  and a genuine positive-feedback instability in the task rotation PD (fixed by `kp_rot=0`,
  damping-only). Validated envelope, zero safety-guard trips throughout: canonical grid 8/8,
  long holds to 30s 8/8, displacements to 20cm 16/16, torque-scale robustness to 10% 14/14.
  Full evidence and mechanism write-up: `AGENTS.md` §3.
- **Bug found and fixed along the way**: `transport_metrics.py`'s move-hold pass-check
  `abs()`'d the target displacement before comparing it to the signed achieved value, so
  every negative-direction transport run was silently reported invalid regardless of actual
  tracking quality. Fixed 2026-07-03 (`transport_metrics.py`, see `AGENTS.md` §4).

Untested/known-out-of-scope boundaries (physical or bandwidth limits, not defects): dx=0.25m
fails via Z-drift (workspace/reach limit); moves faster than ~0.5s undershoot (closed-loop
bandwidth limit at the tuned `kp_x`, not saturation).

## Done

**2026-07-03 consolidation:**
- CoppeliaSim lane archived to `archive/coppelia/` (code, Lua, launchers, ZMQ probes, WSL
  bring-up, RL PPO stack, ROS2 controller/bridge nodes, docs, configs, tests). Legacy MuJoCo
  cartpole diagnostics archived to `archive/legacy_mujoco/`.
- Full `mujoco_menagerie/` zoo checkout removed (re-clone SHA in `archive/coppelia/README.md`
  and `AGENTS.md`); the active model's meshes come from the tracked `vendor/mujoco_menagerie/`.
- Test suite consolidated: `tests/{unit,mujoco,hardware}` with pytest markers.
- Observability layer landed: `observability/run_logger.py` writes per-run `run_record.json`
  + sweep `run_log.jsonl`/`run_log.csv` from both audit and move-hold entrypoints.
- Dependency manifest fixed (unused pkgs dropped; `ur_rtde`, `pin` added).
- AGENTS.md rewritten as a structured playbook; old log at `docs/archive/AGENTS_HISTORY.md`.

**2026-07-03 Pinocchio + OSC tuning (later the same day):**
- Pinocchio dynamics provider (P0), gravity source swap (P1), Coriolis feedforward (P2) —
  all landed, flag-gated, parity-validated against MuJoCo.
- Operational-space controller upgrade (P3) — Λ(q) inertia shaping + nullspace posture
  projection, flag-gated.
- Posture re-anchoring (`controller.posture_reanchor_on_settle`) — flag-gated controller
  feature, added specifically to fix long-hold drift.
- Tuned OSC config landed and exhaustively validated (see diagnosis section above).
- `transport_metrics.py` sign-bug fix (negative-direction transport validation).
- Test suite: 116 passing (full suite, including hardware-mocked tests).

**Documentation pass (2026-07-03, this update):**
- Archived three root-level point-in-time reports (`AUDIT_REPORT.md`, `BLOAT_REPORT.md`,
  `DIAGNOSTIC_real_cartpole_torque_control_questions.md`) to `docs/archive/` with
  superseded-banners, since their recommendations are now executed.
- Archived the CoppeliaSim-era `docs/PROJECT_PLAN.md` to
  `docs/archive/PROJECT_PLAN_coppeliasim_era.md`.
- Refreshed `docs/controller_core/`, `docs/simulation/`, `docs/ros2/` subsystem docs to match
  the current (post-archival, post-Pinocchio/OSC) code layout.

## 2026-07-14: hardware three-mode lane + docs refresh

Hardware now has three modes via `hardware/x_transport.py` (`--control-mode`):
`position` (default, `servoL` + shadow OSC), `direct_torque` (Python OSC @ 500 Hz),
`urscript` (on-robot). Learning map: `docs/hardware/README.md`. Scratch RTDE probes
moved to `archive/superseded/hardware_scratch/`. `AGENTS.md` §4 updated (direct torque
is **in** scope with PolyScope ≥5.23 / `ur_rtde>=1.6`).

## 2026-07-07: hardware lane rewrite

`hardware/` was rewritten from scratch (audited for architecture bugs first — see
`AGENTS.md` §4 for the real stale-state bug found in the old ROS2 pipeline node, and the
missing e-stop-latch-implementation claim that turned out to be false). Initial rewrite:
`hardware/{safety,link,motion}.py` + `tools/ur5e_{connect,move}.py`. Old lane archived to
`archive/superseded/hardware_rtde_v1/`. Direct-torque path landed later (see 2026-07-14).
Full detail: `docs/hardware/README.md`.

Additionally validated against the *real* Universal Robots `URControl` binary (via a local
URSim instance, not a mock — see session notes): confirmed `RTDEReceiveInterface` really has
`getSafetyStatusBits` (not `getSafetyStatus`) and `RTDEControlInterface.servoL`'s real
signature is exactly `(pose, speed, acceleration, time, lookahead_time, gain)` — both match
this code's assumptions exactly. `hardware/link.py`'s `read_state()` was also confirmed to
correctly reject a degenerate (not-fully-powered-on) real-server response rather than
accepting bad data.

Not verifiable without the real robot: actual `servoL` motion, 125Hz streaming under real
network conditions, real TCP orientation-vector convention, and whether the physical mount
makes the chosen `--axis` argument correspond to true left/right.

## 2026-07-06/07: RL gain-scheduling — unresolved, do not treat any checkpoint as "the good one"

`rl_gain_scheduling/` trains a PPO policy to schedule the tuned OSC controller's gains live
(vs. the fixed-gain baseline config). Four training runs so far, **none has beaten the fixed
baseline on the full comparative eval grid** (`rl_gain_scheduling/eval_gain_scheduler.py`,
5 heights × 4 displacements = 20 cells, `valid_move_and_hold` + quality score):

- `run1_200k` / `run2_continued_2.2M` (original reward config): collapsed to "never move" —
  0/20 valid, near-zero X displacement in 19/20 cells (diagnosed via commit `c52043a`).
- `reward_v2_2M` (`config/rl_gain_scheduling_reward_v2.yaml`, alive-bonus cut 25x,
  terminal-quality-weight raised 4x to fix the above): stopped the "do nothing" collapse but
  converged to a different pathological corner instead — evaluated 2026-07-07, **0/20 valid**,
  quality 0.013-0.111 across the whole grid vs. baseline's 0.14-0.66. See
  `outputs/rl_gain_scheduling/eval/reward_v2_2M_first_eval/`.
- `reward_v3_2M` (`config/rl_gain_scheduling_reward_v3.yaml`, tightened `kp_rot` upper bound
  and raised damping-gain lower bounds to fix the max-stiffness/zero-damping instability
  diagnosed in v2): evaluated 2026-07-07, **1/20 valid** (only alpha=0, dx=0.05), quality
  0.014-0.195 in the 19 failing cells. See
  `outputs/rl_gain_scheduling/eval/reward_v3_2M_first_eval/`.

Baseline comparison (fixed-gain tuned OSC config, same grid, both eval runs): 100% valid at
height 0.0/0.25/0.5, 75% at 0.75, 50% at 1.0 — this is the actually-working controller.
**No RL checkpoint currently in this repo should be presented as a working/trained model.**
The gain-scheduling approach itself is not disproven — the reward shaping and/or training
budget (2M steps) may simply be insufficient — but that's an open problem, not a solved one.

**Important nuance, checked 2026-07-07 after the grid eval above**: the accel-then-reverse
stress test that originally motivated the v3 gain-bounds fix
(`rl_gain_scheduling/_scratch_accel_profile_demo.py`, mid-height, 1s forward + 4s reverse
constant-acceleration target, no hold phase) was re-run against `reward_v3_2M` and **passes
cleanly** — full 5s, no safety-guard trip, max position error 7.7mm, final error 2.1mm, max
orientation error 0.062 rad (well under the 0.35 rad guard that killed `reward_v2` on this same
test at ~1.2s). Gains settle to `kp_x=800, kd_x=5.0, kp_rot=0.2, kd_rot=11.3` — a stable
non-degenerate point, not a guard-avoidance trick. See
`outputs/rl_gain_scheduling/eval/reward_v3_2M_accel_reverse_stress_test/`. **So the v3
gain-bounds fix did work for the specific instability it targeted** — this narrows the open
problem from "the policy is broken" to "the policy tracks a smoothly-varying mid-height target
well but does not reliably settle-and-hold at a static target and/or does not generalize across
height," which is what the grid eval actually exercises and the stress test does not (no hold
phase, single height only).

Next step if this is picked back up: inspect `reward_v3_2M`'s per-step gain traces from the
grid eval itself (`outputs/rl_gain_scheduling/eval/reward_v3_2M_first_eval/learned/runs/*/trace.jsonl`)
to see specifically what breaks during the hold phase and/or at other heights, now that the
mid-height/no-hold tracking behavior is confirmed sound — before trying another reward/bound
iteration blind.

**2026-07-07, later the same day -- user narrowed scope to height_alpha=0.5 only, found and
fixed a real environment bug, root problem still unresolved:**

New config `config/rl_gain_scheduling_reward_v4_height0.5.yaml` (height_alpha_range pinned to
`[0.5, 0.5]`, everything else identical to v3) trained fresh (2M steps, n_envs=8, ~34 min
wall-clock) to isolate whether height generalization was the obstacle. **It was not** --
evaluated at alpha=0.5 only, still 0/4 valid, with the identical "frozen during move phase,
violent correction during hold phase, guard trip" signature seen in the v3 grid eval (per-cell
`move_phase_achieved_x_delta_m` ~0, `move_phase_max_abs_tau_controller_nm` ~1e-10, then
`hold_phase_max_abs_qd_radps` 2.4-3.2 rad/s tripping the velocity or Z-drift guard).

Root-caused from there, not guessed: `GainSchedulingEnv.step()` computes
`t_s = float(self.data.time)` and feeds it into the `min_jerk_move_hold` target generator as
"elapsed time since episode start." But `GainSchedulingEnv.reset()` calls `apply_start_q()`
(which resets qpos/qvel/qacc/ctrl/qfrc_applied/xfrc_applied) and never reset `data.time` itself
-- every other current caller of `apply_start_q` (the sim's rollout tools, the accel-reverse
stress test script) only runs one episode per process, so this was never triggered before. A
live training/eval loop resets thousands of times per process. Confirmed directly: before the
fix, a second `reset()` in the same process left `data.time` at whatever the first episode
ended on (e.g. 1.2s); after, it correctly reads 0.0. Practical effect: from the *second*
episode onward in every one of the `n_envs` parallel workers -- i.e. nearly all of training and
of every eval grid cell after the first -- `t_s` was already past `move_duration_s` (often past
`max_episode_seconds` entirely) at episode start, so the target generator returned the
already-settled final value from step 0. The policy never saw a real ramping move-phase target
during nearly all of training. Fixed in `rl_gain_scheduling/gain_scheduling_env.py` (`reset()`
now sets `self.data.time = 0.0` after `apply_start_q()`). Full test suite unaffected: 167
passed (there was no existing coverage of this reset path, which is why the bug went unnoticed).

Retrained from scratch with the fix (`reward_v5_height0.5_timefix_2M`, same config otherwise).
The violent hold-phase guard-trips are **gone** -- every eval episode now runs to
`termination_reason: "duration_complete"`, confirming the target-timing bug was real and the
fix works. But **the original "never move" collapse from `run1`/`run2` (commit `c52043a`) is
back**: `achieved_x_delta_m` ~0 (float noise) in 3 of 4 cells, a small partial move (0.046 of a
0.2m target) in the 4th, `explained_variance=1.0` at end of training (the value function easily
fits a very low-variance "sit still" policy -- the same signature `c52043a` originally
diagnosed). Still 0/4 valid at height=0.5. See
`outputs/rl_gain_scheduling/eval/reward_v5_height0.5_timefix_first_eval/`.

**Conclusion:** the `alive_bonus`/`terminal_quality_weight` rebalance from `c52043a` (carried
unchanged through v2/v3/v4/v5) was never actually a robust fix for the never-move collapse --
it appears to have changed the *symptom* while training was still running on the broken,
already-settled target signal from the `data.time` bug. Now that the environment is correct,
the same degenerate "survive by doing nothing" optimum reappears. This is a genuine
reward-shaping/exploration problem, not a quick config tweak, and **two full training cycles
with different structural fixes have not resolved it** -- further blind iteration is not
recommended without first designing a stronger positive incentive for the move phase
specifically (e.g. a progress-based reward term rather than pure accumulated `abs(x_error)`
penalty, which does not obviously reward attempting a fast move over sitting still) or
revisiting exploration/entropy settings. **Recommendation: use the fixed-gain baseline
controller (`config/ur5e_mujoco_torque_osc_tuned.yaml`) for real-world testing now** -- it
already validates 100% at height_alpha=0.5 across the full displacement range on this same eval
grid. Treat RL gain-scheduling as a longer-term improvement track, not a near-term dependency.

## Next

Nothing is currently blocking or in-progress on the MuJoCo controller/tuning side (OSC config
above). Open items, none urgent:
- `outputs/` still holds the pre-purge `outputs/PURGE_LIST.md` DELETE-list directories plus a
  large volume of OSC-tuning-campaign sweep output generated the same day — regenerate the
  purge list to cover them, then the user needs to run the `rm -rf` themselves (blocked from
  automated execution by the environment's destructive-command guard).
- Known, deliberately-not-chased envelope edges (0.25m Z-drift limit, sub-0.5s move bandwidth
  limit) are documented, not bugs — no action needed unless a real use case needs them.
- Gain retuning for `y`/`z`/reanchor-tolerance dimensions was evaluated and found to have no
  meaningful headroom to improve (150-4000x safety margin already) — don't re-open without a
  concrete reason.
- RL gain-scheduling policy is unresolved (see section above) — the `data.time` reset bug is
  fixed and confirmed real, but the underlying never-move reward collapse persists even with
  the fix and a single-height-only task. Next step needs actual reward redesign (a
  progress-based move-phase term, most likely) or exploration-setting changes, not another
  blind config nudge. Not currently blocking real-world testing — use the fixed-gain baseline.
- Hardware lane is code-complete and unit-tested but has never touched a real robot — first
  physical contact is the next real-world milestone, not further code changes.

## Historical status

The CoppeliaSim bring-up era (headless video smoke, RPC/ZMQ controller, ROS2 bridge probes,
RL Y-transport) is documented in `docs/archive/AGENTS_HISTORY.md`,
`docs/archive/PROJECT_PLAN_coppeliasim_era.md`, and `archive/coppelia/docs/`. The hardware
lane described there is fully superseded by the 2026-07-07 rewrite above.
