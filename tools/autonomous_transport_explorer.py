#!/usr/bin/env python3
"""Semi-autonomous real-hardware X-transport envelope explorer.

Replaces this session's manual loop (run one real trial -> hand-write a
Python snippet to probe the result -> pick the next command by eye -> repeat)
with a driver that proposes the next trial itself, using the accumulated
history of real trials on disk, and
:mod:`tools.diagnostics.transport_trial_diagnostics` to score every result
automatically instead of re-deriving it by hand each time.

**This does not remove the human from the loop.** Every trial is printed
with its full command and a one-line rationale, then this script BLOCKS on
a typed confirmation before it will touch the robot. That is a deliberate
design choice, not a placeholder: this repo's own hardware guardrails
(AGENTS.md sec 4, item 2 -- "Motion requires explicit opt-in") apply to
every individual move, not just the first one in a session. Nothing here
loosens that. Approve, skip, or stop at each step:
    y / yes / <enter>  -> run the proposed trial on the real robot
    s / skip           -> discard this proposal, generate a different one
    q / quit / stop     -> end the session (state is saved; re-run to resume)

**Hard displacement guardrail**: dx is clamped to +/-0.20m
(X_ABS_GUARDRAIL_M below) everywhere a proposal is constructed -- this
mirrors the physical travel limit the user has independently configured on
the real arm's own controller (a second, independent layer; this script
does not know about or rely on that one). This script will refuse to
propose or execute anything outside that bound; it is not a CLI flag,
change it in code deliberately if the validated envelope genuinely grows.

**Scope, deliberately**: this loop explores TRAJECTORY parameters only --
displacement, move duration, direction, and (once a baseline min-jerk
envelope is covered) commanded peak acceleration via the
accel_duration_scurve profile, to push toward faster/harder moves as
requested. It does NOT touch controller gains (kp_x/kd_x/friction_ff_*/
deadband/etc.) -- this session found those tradeoffs are real and
non-orthogonal (2026-08-02: widening friction_ff_qd_deadband to damp the
~2Hz oscillation also gave back most of the hold-phase accuracy benefit),
and this repo's own working rule is not to mix controller-gain changes with
other work in the same pass. Pick a --config explicitly; gain retuning
stays a deliberate, separate, human-directed step.

State is a single append-only JSONL log (default
outputs/hardware_transport/autonomous_explorer_log.jsonl) -- re-running this
script re-reads it and resumes with full memory of every past trial, so a
session can be stopped and picked back up at any time.

Run this ON the machine with real robot access (thinkrobot), not on the
analysis machine -- it shells out to tools/ur5e_direct_torque_x_transport.py
directly.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.diagnostics.transport_trial_diagnostics import diagnose  # noqa: E402

# Hard guardrail. Not a CLI flag on purpose -- see module docstring.
X_ABS_GUARDRAIL_M = 0.20

WRIST2_OFFSET_START_Q = [0.0, -0.835398, -1.200000, -0.985398, 0.200000, 0.0]
DEFAULT_CONFIG = "config/ur5e_mujoco_torque_osc_tuned_wrist_orient_friction_ff_calibrated.yaml"
DEFAULT_LOG_PATH = REPO_ROOT / "outputs" / "hardware_transport" / "autonomous_explorer_log.jsonl"
RUNS_GLOB = str(REPO_ROOT / "outputs" / "hardware_transport" / "direct_torque_*")

# Trajectory-parameter grid for the baseline min-jerk sweep.
DX_CANDIDATES = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20]
MOVE_DURATION_CANDIDATES = [3.0, 4.5, 6.0, 8.0]
SIGNS = [1, -1]

# Acceleration ladder for the "faster / harder" escalation axis
# (accel_duration_scurve, held at a fixed move_duration to isolate the accel
# variable). Starts well below anything tested so far (real hardware has
# only cleanly validated accel_duration_scurve at 0.02 m/s^2 / 4.0s; sim
# found 0.15 m/s^2/2s clean but 0.3 m/s^2/2s trips orientation) and climbs
# one rung at a time, never jumping. HARD ceiling below is deliberately
# conservative relative to that sim finding, not a promise the real arm is
# safe all the way there -- the ladder structure exists precisely so a real
# failure at rung N stops escalation past N, not to justify starting high.
ACCEL_LADDER_MPS2 = [0.02, 0.04, 0.06, 0.09, 0.13, 0.18, 0.25]
ACCEL_LADDER_MOVE_DURATION_S = 4.0

COMMON_FLAGS = [
    "--dynamics-source",
    "local",
    "--noise-robust-guards",
    "--accel-variable-tolerance",
    "--speed-variable-tolerance",
    "--i-understand-this-moves-the-robot",
    "--yes",
]


@dataclass
class Sample:
    success: bool
    guard_category: str
    achieved_fraction: float | None
    resonance_band_power_fraction: float | None


@dataclass
class Cell:
    profile: str  # "min_jerk" or "scurve"
    dx: float | None  # None for scurve cells (accel-driven, not displacement-driven)
    move_duration: float
    sign: int
    target_accel: float | None  # None for min_jerk cells
    samples: list[Sample] = field(default_factory=list)

    @property
    def key(self) -> tuple:
        return (self.profile, self.dx, round(self.move_duration, 2), self.sign, self.target_accel)

    @property
    def n(self) -> int:
        return len(self.samples)

    @property
    def pass_rate(self) -> float:
        if not self.samples:
            return float("nan")
        return sum(1 for s in self.samples if s.success) / len(self.samples)

    @property
    def is_mixed(self) -> bool:
        if self.n < 2:
            return False
        return len({s.success for s in self.samples}) > 1

    @property
    def all_clean_pass(self) -> bool:
        return self.n > 0 and all(s.success for s in self.samples)


def load_history(log_path: Path) -> dict[tuple, Cell]:
    cells: dict[tuple, Cell] = {}
    if not log_path.is_file():
        return cells
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("record_type") != "result":
                continue
            profile = rec.get("profile", "min_jerk")
            dx = rec.get("dx")
            move_duration = rec.get("move_duration")
            sign = rec.get("sign")
            target_accel = rec.get("target_accel")
            if move_duration is None or sign is None:
                continue
            key = (profile, dx, round(float(move_duration), 2), sign, target_accel)
            cell = cells.setdefault(
                key, Cell(profile=profile, dx=dx, move_duration=float(move_duration), sign=sign, target_accel=target_accel)
            )
            cell.samples.append(
                Sample(
                    success=bool(rec.get("success")),
                    guard_category=rec.get("guard_category", "unknown"),
                    achieved_fraction=rec.get("achieved_fraction"),
                    resonance_band_power_fraction=rec.get("resonance_band_power_fraction"),
                )
            )
    return cells


def seed_log_from_existing_runs(log_path: Path, runs_dir: Path) -> int:
    """Backfill the explorer's log from real trials that already exist on
    disk but predate this tool (tonight's manual session, or any other
    direct_torque_* run directory). Idempotent -- tracks which run_dir each
    record came from via 'source_run_dir' and skips ones already present,
    so it's safe to call every time the explorer starts. Real trial history
    lives in outputs/hardware_transport/ on the hardware machine itself
    (gitignored, per AGENTS.md -- it never reaches this repo via git), so
    this has to run locally against that directory, not ship as data.
    Returns the number of newly-seeded records."""
    already_seeded: set[str] = set()
    if log_path.is_file():
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                src = rec.get("source_run_dir")
                if src:
                    already_seeded.add(src)

    candidates = sorted(d for d in glob.glob(str(runs_dir / "direct_torque_*")) if os.path.isdir(d))
    seeded = 0
    for run_dir_str in candidates:
        if run_dir_str in already_seeded:
            continue
        run_dir = Path(run_dir_str)
        if not (run_dir / "summary.json").is_file():
            continue
        try:
            diag = diagnose(run_dir)
        except Exception as exc:  # noqa: BLE001 - best-effort backfill, one bad run shouldn't block the rest
            print(f"  skipping {run_dir.name}: {exc}")
            continue
        with open(run_dir / "summary.json") as f:
            summary = json.load(f)
        move_duration = summary.get("move_duration_s")
        if move_duration is None:
            continue
        trajectory_profile = summary.get("trajectory_profile") or "min_jerk_move_hold"
        if trajectory_profile.startswith("accel_duration"):
            profile = "scurve"
            target_accel_signed = summary.get("target_accel_mps2")
            if target_accel_signed is None:
                continue
            sign = 1 if target_accel_signed >= 0 else -1
            dx = None
            target_accel = abs(target_accel_signed)
        else:
            profile = "min_jerk"
            dx_signed = summary.get("target_x_delta_m")
            if not dx_signed:
                continue
            sign = 1 if dx_signed > 0 else -1
            dx = round(abs(dx_signed), 3)
            target_accel = None
        record = {
            "record_type": "result",
            "profile": profile,
            "dx": dx,
            "move_duration": round(float(move_duration), 2),
            "sign": sign,
            "target_accel": target_accel,
            "source_run_dir": run_dir_str,
            **diag,
        }
        append_log(log_path, record)
        seeded += 1
    return seeded


def _score_followup(cell: Cell) -> float:
    score = 0.0
    if cell.is_mixed:
        score += 10.0
    if cell.n < 2:
        score += 5.0 * (2 - cell.n)
    return score


def propose_next(cells: dict[tuple, Cell]) -> dict:
    """Returns a proposal dict: {profile, dx, move_duration, sign,
    target_accel, rationale}. Priority: (1) mixed-outcome cells worth
    re-confirming, (2) undersampled min-jerk cells within the validated
    grid, (3) escalate the accel ladder one rung if the current rung passed
    cleanly on both signs, (4) start the accel ladder if untried, (5)
    fallback to any untested min-jerk cell."""
    # Restrict follow-up scoring to cells that actually belong to the current
    # grid/ladder -- the seeded log can contain cells from earlier, unrelated
    # campaigns (e.g. long-hold tests at move_duration=20s), and those
    # shouldn't drive this sweep's proposals even though they're preserved
    # in the log for reference.
    in_grid_min_jerk = [
        c
        for c in cells.values()
        if c.profile == "min_jerk" and c.dx in DX_CANDIDATES and round(c.move_duration, 2) in MOVE_DURATION_CANDIDATES
        and c.sign in SIGNS
    ]

    followups = sorted(in_grid_min_jerk, key=_score_followup, reverse=True)
    if followups and _score_followup(followups[0]) > 0:
        c = followups[0]
        tag = "mixed pass/fail" if c.is_mixed else f"only {c.n} sample(s)"
        return {
            "profile": "min_jerk",
            "dx": c.dx,
            "move_duration": c.move_duration,
            "sign": c.sign,
            "target_accel": None,
            "rationale": f"re-test: existing cell has {tag} -- {[s.guard_category for s in c.samples]}",
        }

    for dx in DX_CANDIDATES:
        for move_dur in MOVE_DURATION_CANDIDATES:
            for sign in SIGNS:
                key = ("min_jerk", dx, round(move_dur, 2), sign, None)
                if key not in cells:
                    return {
                        "profile": "min_jerk",
                        "dx": dx,
                        "move_duration": move_dur,
                        "sign": sign,
                        "target_accel": None,
                        "rationale": "unexplored cell within the baseline dx/duration/sign grid",
                    }

    # Baseline grid fully covered -- move to the acceleration ladder.
    for rung_idx, accel in enumerate(ACCEL_LADDER_MPS2):
        for sign in SIGNS:
            key = ("scurve", None, ACCEL_LADDER_MOVE_DURATION_S, sign, accel)
            cell = cells.get(key)
            if cell is None:
                if rung_idx == 0:
                    return {
                        "profile": "scurve",
                        "dx": None,
                        "move_duration": ACCEL_LADDER_MOVE_DURATION_S,
                        "sign": sign,
                        "target_accel": accel,
                        "rationale": "starting the acceleration ladder (rung 0, most conservative)",
                    }
                prev_key = ("scurve", None, ACCEL_LADDER_MOVE_DURATION_S, sign, ACCEL_LADDER_MPS2[rung_idx - 1])
                prev_cell = cells.get(prev_key)
                if prev_cell is not None and prev_cell.all_clean_pass:
                    return {
                        "profile": "scurve",
                        "dx": None,
                        "move_duration": ACCEL_LADDER_MOVE_DURATION_S,
                        "sign": sign,
                        "target_accel": accel,
                        "rationale": (
                            f"escalating accel ladder: rung {rung_idx - 1} "
                            f"(accel={ACCEL_LADDER_MPS2[rung_idx - 1]}) passed cleanly, trying rung {rung_idx}"
                        ),
                    }
                # Previous rung not yet clean or not yet tried on this
                # sign -- don't jump ahead of it.
                continue

    return {
        "profile": "done",
        "dx": None,
        "move_duration": None,
        "sign": None,
        "target_accel": None,
        "rationale": "baseline grid + acceleration ladder both fully explored (or ladder stalled on a failed rung)",
    }


def build_command(proposal: dict, *, config: str, robot_ip: str) -> list[str]:
    profile = proposal["profile"]
    sign = proposal["sign"]
    move_duration = proposal["move_duration"]
    start_q = " ".join(f"{v:.6f}" for v in WRIST2_OFFSET_START_Q)

    argv = [
        sys.executable,
        "tools/ur5e_direct_torque_x_transport.py",
        "--robot-ip",
        robot_ip,
        "--control-mode",
        "direct_torque",
        "--config",
        config,
        "--start-q-rad",
    ] + [f"{v:.6f}" for v in WRIST2_OFFSET_START_Q]

    if profile == "min_jerk":
        dx = proposal["dx"]
        assert abs(dx) <= X_ABS_GUARDRAIL_M + 1e-9, f"dx={dx} exceeds hard guardrail {X_ABS_GUARDRAIL_M}"
        signed_dx = sign * dx
        peak_speed = 1.875 * dx / move_duration
        speed_override = round(peak_speed * 1.3 + 0.02, 3)
        argv += [
            "--target-x-delta",
            f"{signed_dx:.4f}",
            "--move-duration",
            f"{move_duration:.2f}",
            "--duration",
            f"{move_duration + 2.0:.2f}",
            "--max-tcp-speed-mps",
            f"{speed_override:.3f}",
        ]
    elif profile == "scurve":
        accel = sign * proposal["target_accel"]
        # Peak velocity for accel_duration_scurve = accel*move_duration/pi
        # (per tools/ur5e_direct_torque_x_transport.py's own --max-tcp-speed-mps
        # docstring). Margin matches the min-jerk sizing above.
        import math

        peak_speed = abs(accel) * move_duration / math.pi
        speed_override = round(peak_speed * 1.3 + 0.02, 3)
        argv += [
            "--trajectory-profile",
            "accel_duration_scurve",
            "--target-accel",
            f"{accel:.4f}",
            "--move-duration",
            f"{move_duration:.2f}",
            "--duration",
            f"{move_duration + 2.0:.2f}",
            "--max-tcp-speed-mps",
            f"{speed_override:.3f}",
        ]
    else:
        raise ValueError(f"unknown profile {profile!r}")

    argv += COMMON_FLAGS
    return argv


def find_latest_run_dir(after_mtime: float) -> Path | None:
    candidates = [d for d in glob.glob(RUNS_GLOB) if os.path.isdir(d)]
    fresh = [d for d in candidates if os.path.getmtime(d) >= after_mtime]
    if not fresh:
        return None
    fresh.sort(key=os.path.getmtime, reverse=True)
    return Path(fresh[0])


def append_log(log_path: Path, record: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def print_proposal(proposal: dict, cmd: list[str]) -> None:
    print("\n" + "=" * 78)
    print("PROPOSED NEXT TRIAL")
    print(f"  profile:        {proposal['profile']}")
    if proposal["profile"] == "min_jerk":
        print(f"  dx:             {proposal['sign'] * proposal['dx']:+.3f} m  (guardrail: +/-{X_ABS_GUARDRAIL_M} m)")
    else:
        print(f"  target_accel:   {proposal['sign'] * proposal['target_accel']:+.3f} m/s^2")
    print(f"  move_duration:  {proposal['move_duration']} s")
    print(f"  rationale:      {proposal['rationale']}")
    print("  command:")
    print("    " + " ".join(cmd))
    print("=" * 78)


def print_result_summary(diag: dict) -> None:
    print("\n--- result ---")
    print(f"  success:              {diag.get('success')}")
    print(f"  guard_category:       {diag.get('guard_category')}")
    print(f"  termination_reason:   {diag.get('termination_reason')}")
    af = diag.get("achieved_fraction")
    if af is not None:
        print(f"  achieved_fraction:    {af:.3f}")
    rbp = diag.get("resonance_band_power_fraction")
    if rbp is not None:
        print(f"  resonance_band_power: {rbp:.3f}  (baseline tight-deadband ~0.81, suppressed ~0.07)")
    hold = diag.get("hold_phase_error")
    if hold:
        print(f"  hold_phase_error:     mean={hold['mean_abs_mm']:.2f}mm final={hold['final_abs_mm']:.2f}mm")
    trend = diag.get("pre_trip_trend")
    if trend:
        orient = trend.get("orientation_error_norm_rad", {})
        print(f"  pre_trip orientation trend: {orient.get('trend')}")
    print("--------------\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot-ip", required=True)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--max-trials", type=int, default=20)
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print proposals and commands without touching the robot or writing result records.",
    )
    parser.add_argument(
        "--seed-from-dir",
        type=Path,
        default=None,
        help=(
            "Backfill the log from existing direct_torque_* run directories under this path "
            "(e.g. outputs/hardware_transport) before proposing anything -- idempotent, safe to "
            "pass on every invocation. Use this so real trials already run before this tool "
            "existed aren't re-proposed from scratch."
        ),
    )
    args = parser.parse_args()

    if args.seed_from_dir is not None:
        n = seed_log_from_existing_runs(args.log_path, args.seed_from_dir)
        print(f"Seeded {n} new record(s) from {args.seed_from_dir} into {args.log_path}.")

    cells = load_history(args.log_path)
    print(f"Loaded {len(cells)} cells from {args.log_path} ({sum(c.n for c in cells.values())} past trials).")

    trials_run = 0
    while trials_run < args.max_trials:
        proposal = propose_next(cells)
        if proposal["profile"] == "done":
            print(f"\n{proposal['rationale']}")
            print("Nothing left to propose within the current grid/ladder. Extend the constants at the top of this")
            print("file deliberately if you want to widen the search -- this is not auto-extending itself.")
            break

        import time

        cmd = build_command(proposal, config=args.config, robot_ip=args.robot_ip)
        print_proposal(proposal, cmd)

        if args.dry_run:
            print("[--dry-run] not executing.")
            break

        resp = input("Approve this trial? [y]es / [s]kip / [q]uit: ").strip().lower()
        if resp in ("q", "quit", "stop"):
            print("Stopping. State is saved -- re-run this script to resume.")
            break
        if resp in ("s", "skip"):
            append_log(
                args.log_path,
                {
                    "record_type": "result",
                    "profile": proposal["profile"],
                    "dx": proposal["dx"],
                    "move_duration": proposal["move_duration"],
                    "sign": proposal["sign"],
                    "target_accel": proposal["target_accel"],
                    "success": False,
                    "guard_category": "skipped_by_operator",
                    "achieved_fraction": None,
                    "resonance_band_power_fraction": None,
                },
            )
            cells = load_history(args.log_path)
            continue
        if resp not in ("y", "yes", ""):
            print("Unrecognized response, treating as skip.")
            continue

        start_mtime = time.time()
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
        run_dir = find_latest_run_dir(after_mtime=start_mtime - 5.0)
        if run_dir is None:
            print("WARNING: could not locate the run directory this trial produced. Not logging a result.")
            continue

        diag = diagnose(run_dir)
        print_result_summary(diag)

        record = {
            "record_type": "result",
            "profile": proposal["profile"],
            "dx": proposal["dx"],
            "move_duration": proposal["move_duration"],
            "sign": proposal["sign"],
            "target_accel": proposal["target_accel"],
            "exit_code": proc.returncode,
            **diag,
        }
        append_log(args.log_path, record)
        cells = load_history(args.log_path)
        trials_run += 1

    print(f"\nSession complete: {trials_run} trial(s) run this invocation. Log: {args.log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
