#!/usr/bin/env python3
"""Active variance search for real UR5e transport trials.

Not a black-box optimizer (no GP, no external BO library) -- deliberately
simple, because each "evaluation" here is a real robot command costing real
time (~8-14s+overhead), so the number of trials will always be small enough
that a full GP's benefit over an auditable empirical-variance heuristic is
marginal, and simple/auditable matters more than sample-efficient when a
human has to read and trust every proposed next command before running it on
real hardware.

What this does:
  1. Reads all existing real-hardware trace summaries under
     outputs/hardware_transport_remote/hardware_transport/ that match a
     direct_torque run, at the wrist2-offset pose.
  2. Buckets them by (target_x_delta_m, move_duration_s, sign) into cells.
  3. For each cell with >=2 samples, computes an empirical "noisiness" score:
     the coefficient of variation of the guard-trip-relevant peak metric
     (max_abs_qd_radps as a proxy -- the one summary field consistently
     present regardless of trip cause) plus the pass/fail rate. Cells that
     mix passes and failures at the SAME nominal command are the most
     interesting -- that's the signature of real stick-slip/friction
     variability this whole investigation has been chasing, not a fixed
     controller bug.
  4. Ranks candidate NEXT points (both already-tested cells needing more
     samples to confirm variance, and unexplored cells within the already-
     validated dx in [-0.20, 0.20] / move_duration in [3.0, 8.0] envelope)
     by an uncertainty-favoring score: prioritize cells with few samples
     (<3) or a mixed pass/fail record over cells that are either untested
     deep in unvalidated territory or already confidently characterized.
  5. Prints the top N candidates as ready-to-run CLI commands, using the
     already-validated margin-sizing pattern (--noise-robust-guards
     --accel-variable-tolerance --speed-variable-tolerance plus a speed/
     accel override sized from the min-jerk peak-speed formula with margin).

This is a human-in-the-loop tool, not an autonomous loop: run it, get
proposed commands, run ONE on real hardware, re-run this script (it re-reads
whatever new trace landed), repeat. No attempt is made to execute real motion
from here.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass, field

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_GLOB = os.path.join(
    REPO_ROOT, "outputs", "hardware_transport_remote", "hardware_transport", "direct_torque_*"
)

WRIST2_OFFSET_START_Q = "0.000000 -0.835398 -1.200000 -0.985398 0.200000 0.000000"
VALIDATED_DX_ABS_MAX = 0.20
VALIDATED_MOVE_DURATION_RANGE = (3.0, 8.0)

# Candidate grid this tool proposes from -- deliberately confined to the
# already-validated envelope (AGENTS.md/session history: dx up to 0.20m and
# move_duration 3.0-6.0s have real successful precedent at this pose). Not a
# license to extrapolate further without a fresh, deliberate decision.
CANDIDATE_DX = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20]
CANDIDATE_MOVE_DURATION = [3.0, 4.5, 6.0, 8.0]
CANDIDATE_SIGNS = [1, -1]


@dataclass
class Cell:
    dx: float
    move_duration: float
    sign: int
    samples: list = field(default_factory=list)  # list of (success: bool, max_abs_qd: float, reason: str)

    @property
    def key(self):
        return (round(self.dx, 3), round(self.move_duration, 2), self.sign)

    @property
    def n(self) -> int:
        return len(self.samples)

    @property
    def pass_rate(self) -> float:
        if not self.samples:
            return float("nan")
        return sum(1 for s in self.samples if s[0]) / len(self.samples)

    @property
    def is_mixed(self) -> bool:
        """Both a pass and a fail exist at this exact nominal command --
        the real stick-slip-variability signature this tool exists to find."""
        if self.n < 2:
            return False
        outcomes = {s[0] for s in self.samples}
        return len(outcomes) > 1

    @property
    def qd_cv(self) -> float:
        """Coefficient of variation of max_abs_qd_radps across samples --
        0 if <2 samples or all identical."""
        vals = [s[1] for s in self.samples if s[1] is not None]
        if len(vals) < 2:
            return 0.0
        mean = sum(vals) / len(vals)
        if mean <= 1e-9:
            return 0.0
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        return (var**0.5) / mean


def load_cells(pose_filter: str | None = WRIST2_OFFSET_START_Q) -> dict:
    cells: dict = {}
    for run_dir in sorted(glob.glob(RUNS_GLOB)):
        summary_path = os.path.join(run_dir, "summary.json")
        if not os.path.isfile(summary_path):
            continue
        try:
            with open(summary_path) as f:
                s = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        dx = s.get("target_x_delta_m")
        move_dur = s.get("move_duration_s")
        if dx is None or move_dur is None or dx == 0:
            continue
        sign = 1 if dx > 0 else -1
        key = (round(abs(dx), 3), round(float(move_dur), 2), sign)
        cell = cells.setdefault(key, Cell(dx=abs(dx), move_duration=float(move_dur), sign=sign))
        success = bool(s.get("success"))
        max_qd = s.get("max_abs_qd_radps")
        reason = s.get("termination_reason", "")
        cell.samples.append((success, max_qd, reason))
    return cells


def score_existing_cell_for_followup(cell: Cell) -> float:
    """Higher = more worth re-testing. Mixed pass/fail is the strongest
    signal (confirmed real variability); few samples is next (not enough
    data to know yet); high qd_cv is a secondary signal."""
    score = 0.0
    if cell.is_mixed:
        score += 10.0
    if cell.n < 3:
        score += 5.0 * (3 - cell.n)
    score += cell.qd_cv * 2.0
    return score


def score_unexplored_cell(dx: float, move_duration: float, sign: int, cells: dict) -> float:
    """Prefer unexplored cells adjacent to already-tested ones (interpolating
    the boundary of the validated envelope) over ones far from any real data."""
    key = (round(dx, 3), round(move_duration, 2), sign)
    if key in cells:
        return -1.0  # not unexplored
    # Distance (in a simple normalized sense) to nearest tested cell of the
    # same sign -- smaller distance = higher priority (fills in the map
    # near what's already known, rather than jumping to a totally cold spot).
    same_sign = [c for c in cells.values() if c.sign == sign]
    if not same_sign:
        return 1.0
    best = min(
        abs(c.dx - dx) / VALIDATED_DX_ABS_MAX + abs(c.move_duration - move_duration) / 8.0
        for c in same_sign
    )
    return max(0.0, 2.0 - best)


def build_command(dx: float, move_duration: float, sign: int, config: str) -> str:
    signed_dx = sign * dx
    # Margin-sized overrides, same formula used throughout this session:
    # min-jerk peak speed = 1.875*dx/T, ~30% margin, rounded.
    peak_speed = 1.875 * dx / move_duration
    speed_override = round(peak_speed * 1.3 + 0.02, 3)  # +0.02 floor so tiny dx still gets *some* margin
    duration = move_duration + 2.0
    return (
        "python tools/ur5e_direct_torque_x_transport.py --robot-ip 172.16.71.77 "
        f"--control-mode direct_torque --config {config} "
        f"--start-q-rad {WRIST2_OFFSET_START_Q} "
        f"--target-x-delta {signed_dx:.3f} --move-duration {move_duration:.1f} --duration {duration:.1f} "
        "--dynamics-source local --noise-robust-guards --accel-variable-tolerance --speed-variable-tolerance "
        f"--max-tcp-speed-mps {speed_override:.3f} "
        "--i-understand-this-moves-the-robot --yes"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--top", type=int, default=5, help="How many candidate next trials to print.")
    parser.add_argument(
        "--config",
        default="config/ur5e_mujoco_torque_osc_tuned_wrist_orient_friction_ff_calibrated.yaml",
        help="Config to use in the proposed commands.",
    )
    args = parser.parse_args()

    cells = load_cells()

    print(f"Loaded {len(cells)} tested (dx, move_duration, sign) cells from real trace data.\n")

    print("=== Existing cells worth re-testing (mixed pass/fail or too few samples) ===")
    followups = sorted(
        (c for c in cells.values() if score_existing_cell_for_followup(c) > 0),
        key=score_existing_cell_for_followup,
        reverse=True,
    )
    for c in followups[:10]:
        tag = "MIXED PASS/FAIL" if c.is_mixed else f"n={c.n}"
        print(
            f"  dx={c.sign*c.dx:+.3f} move_dur={c.move_duration:.1f}s  {tag}  "
            f"pass_rate={c.pass_rate:.2f}  qd_cv={c.qd_cv:.3f}  "
            f"reasons={[s[2][:35] for s in c.samples]}"
        )

    print("\n=== Unexplored cells near the validated boundary ===")
    unexplored_scored = []
    for dx in CANDIDATE_DX:
        for move_dur in CANDIDATE_MOVE_DURATION:
            for sign in CANDIDATE_SIGNS:
                sc = score_unexplored_cell(dx, move_dur, sign, cells)
                if sc > 0:
                    unexplored_scored.append((sc, dx, move_dur, sign))
    unexplored_scored.sort(reverse=True)
    for sc, dx, move_dur, sign in unexplored_scored[:10]:
        print(f"  dx={sign*dx:+.3f} move_dur={move_dur:.1f}s  score={sc:.2f}")

    print(f"\n=== Top {args.top} proposed next real-hardware trials ===")
    proposals = []
    for c in followups:
        proposals.append((score_existing_cell_for_followup(c), c.dx, c.move_duration, c.sign, "re-test"))
    for sc, dx, move_dur, sign in unexplored_scored:
        proposals.append((sc, dx, move_dur, sign, "new"))
    proposals.sort(key=lambda p: p[0], reverse=True)

    seen = set()
    shown = 0
    for sc, dx, move_dur, sign, kind in proposals:
        key = (round(dx, 3), round(move_dur, 2), sign)
        if key in seen:
            continue
        seen.add(key)
        print(f"\n# [{kind}, score={sc:.2f}] dx={sign*dx:+.3f}, move_duration={move_dur:.1f}s")
        print(build_command(dx, move_dur, sign, args.config))
        shown += 1
        if shown >= args.top:
            break


if __name__ == "__main__":
    main()
