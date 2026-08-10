# Disturbance robustness, re-tested against the real validated balance gains

**Date:** 2026-08-10
**Context:** re-run of sensor-noise/control-delay robustness testing
(`tools/diagnostics/pendulum_balance_disturbance_robustness.py`), against the actual
DE-search-validated gains for BOTH control lanes (docs/status/pendulum_balance_gain_search_2026-08-09.md)
instead of the torque lane's old default-Q/r_weight=1e6 design, which is what the first version of
this script tested. Run overnight on ilab3 (idle machine).

## Headline finding: the velocity lane is machine-precision chaotic -- its numbers here are not trustworthy as stated

While reviewing the overnight log, the velocity lane's own **zero-noise, zero-delay** control
condition ("none" row, baseline_pert=0.10 rad) showed FELL -- despite this exact perturbation
being independently reconfirmed as a clean SURVIVE earlier the same session. Investigated
directly rather than assumed:

- Running the identical trial (same gains, same perturbation, no rng calls on this code path)
  4x in a row on the SAME machine (westeros) gives byte-identical results every time --
  **fully deterministic on a given machine.**
- Running the SAME trial on **ilab3 gives a different, also-deterministic result**: westeros
  reproducibly SURVIVES (final_theta_err=0.328), ilab3 reproducibly FALLS
  (final_theta_err=1.556, exactly matching the overnight log). Confirmed on ilab1 too for the
  torque lane (see below) to rule out a single-host quirk.
- This is **not a code bug** -- it is genuine sensitive dependence on machine-level
  floating-point differences (CPU microarchitecture / BLAS build / etc., not anything this
  script controls), amplified over a few seconds of simulation by the closed loop's own
  dynamics into a completely different macroscopic outcome (balance vs. fall).

**Cross-check against the torque lane, to see if this is a general property of the simulation or
specific to one lane**: the torque lane's identical trials (same gains, pert=0.15/0.35 rad) match
across westeros/ilab1/ilab3 to ~1e-8 -- normal floating-point roundoff, **not** amplified into a
different outcome. So this is a real, structural difference between the two lanes' controllers,
not a property of MuJoCo or this repo's simulation stack in general.

**What this means**: the velocity lane's balance behavior at these DE-search-validated gains is
right at (or past) the edge of stability -- so close that a machine-precision-level perturbation,
with no modeled noise or delay at all, is enough to flip the outcome. The **specific pass/fail
numbers and thresholds in the velocity-lane tables below are one sample from what is apparently a
wide, chaotically-sensitive outcome distribution** -- they would very plausibly look different
again on yet another machine, a different BLAS build, or even this same machine with a different
process/thread layout. They should NOT be read as precise, reproducible robustness boundaries the
way the torque-lane numbers can be. The qualitative conclusion -- the velocity lane is far more
fragile than the torque lane, and single-trial testing of it is fundamentally unreliable -- is
itself the real, solid finding here, and this experiment is direct evidence for it.

## Torque lane: real, solid, cross-machine-verified envelope

Gains: `q_arm_pos=2655.2, q_arm_vel=1.59, q_pend_angle=20.9, q_pend_vel=655.2, r_weight=0.347`
(the DE-search "best" config). Baselines: 0.15 and 0.35 rad, both inside the already-documented
genuinely-clean 0.05-0.4 rad range. Clean-path cross-check against
`pendulum_balance_torque_lqr.run_torque_balance_trial` matched exactly before trusting anything.

- **Sensor noise**: robust to realistic encoder quantization alone (14/10/8-bit, clean velocity
  or FD-estimated velocity) at both baselines. The FD-velocity path (the realistic case for this
  encoder, which has no direct velocity output) tolerates *additional* Gaussian jitter up to
  ~0.005 rad extra std (10/10 seeds survive) before degrading sharply by 0.01 rad (2/10 survive
  at pert=0.15) and failing completely by 0.05 rad+. Real 14-bit-encoder-only noise (no extra
  jitter) is comfortably inside the robust region.
- **Delay**: a clean, sharp, reproducible boundary -- **survives up to 3 control cycles (6ms at
  500Hz), fails at 4+ cycles (8ms+)** -- consistent at both baseline perturbations. This is a
  real, actionable real-time budget number for this controller.
- **Combined realistic noise + delay**: survives up to 2 cycles with FD-velocity noise, fails at
  5+ -- consistent with delay being the dominant constraint over noise in this lane.

## Velocity lane: fragile, and (per the headline finding) numerically unreliable as measured

Gains: `kp=2.68, kd=0.036, ik_joint_gain=432.8` (DE-search best). The gain-search doc's claimed
"clean 0.1-0.3 rad" range does not hold up under direct re-check either (a separate, smaller
issue found before the chaos finding above): pert=0.15 and 0.20 rad both FELL on direct
re-verification, while 0.10/0.25/0.30 survived -- baselines here were re-picked to values
reconfirmed to survive at the time of the check (0.10, 0.30 rad), though per the headline finding
above even that reconfirmation is machine-dependent.

With that caveat firmly in mind, the overnight numbers (ilab3) show: near-total intolerance of
any delay (fails at 1+ cycles in most conditions, survives only sporadically), and noise pass-rates
that degrade far more steeply and less predictably than the torque lane's. Read this as "the
velocity lane is qualitatively much more fragile," not as "delay cycle N is the precise failure
boundary" -- the quantitative boundary itself is not reproducible.

## Recommendation

- Trust and use the torque-lane numbers above as real characterization.
- Do not use the velocity-lane table's specific numbers for anything requiring precision (e.g.
  setting a real-hardware delay budget). If the velocity lane's robustness needs a real answer,
  it would require many-seed AND many-machine/many-build statistics, not a single-machine sweep
  like this one -- a materially bigger undertaking than what was run here, not attempted.
- The chaos finding is itself worth keeping in mind for any FUTURE single-trial velocity-lane
  claim in this repo, not just this specific sweep -- a "survived" or "fell" result from one
  trial of this lane's balance controller cannot be trusted without cross-machine or multi-seed
  confirmation.

## Files

| path | what |
|---|---|
| `tools/diagnostics/pendulum_balance_disturbance_robustness.py` | rewritten: both lanes, real validated gains, noise+delay hooks for both |
| this file | write-up |

**Tests run:** full overnight sweep (both lanes, both baselines each, noise/pass-rate/delay/combined)
on ilab3; clean-baseline cross-checks against each lane's own base script; cross-machine
determinism check (westeros/ilab1/ilab3) for both lanes, which is what surfaced the chaos finding.
**Tests not run:** no pytest coverage (diagnostic script); no multi-seed/multi-machine
characterization of the velocity lane's true robustness (identified as needed, not attempted here);
real-hardware validation.

**Rollback:**
```bash
git checkout -- tools/diagnostics/pendulum_balance_disturbance_robustness.py
rm -f docs/status/pendulum_balance_disturbance_robustness_2026-08-10.md
```
Nothing committed.
