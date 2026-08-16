#!/usr/bin/env python3
"""Kick-parameter search for a GENTLE swing-up arrival, i.e. the pendulum's
FIRST pass near the inverted equilibrium happens at low angular velocity --
the arrival condition the LQR cascade (pendulum_lqr_cascade.py) can actually
catch.

Why this differs from pendulum_swingup_multi_kick.py's own search: that
script's objective minimizes min_theta_dist_from_inverted OVER THE WHOLE
TRIAL, which a kick well past the guard-clean energy ceiling satisfies too
-- e.g. the validated single 0.1534 m kick reaches dist=5.6e-7 rad at
t=12s, but only after SAILING PAST inverted on its first pass at
thetadot=14.6 rad/s (E/E_top=1.42, a genuine multi-revolution windmill, not
a swing that arrives and stops) and asymptotically bleeding the excess via
friction over several further revolutions. That is a real, guard-clean flip
-- and a legitimate result in its own right -- but 14.6 rad/s at first
approach is far outside any physically deliverable LQR capture envelope
(the cart authority ceiling is ~1.06 m/s / mild rad/s-scale, not tens of
rad/s), so it cannot be the trial an LQR handoff switches on.

This script instead scores the FIRST time phi (from inverted) enters
--phi-band-deg, by |thetadot| at that crossing (want near zero) plus a
mild centering term -- the classical "arrive near the top gently" swing-up
target, not "eventually visits the top." Reuses
pendulum_swingup_multi_kick.run_multi_kick_trial as the sim engine (same
event-triggered pumping law, same OSC path) with only the SCORE changed.
"""

from __future__ import annotations

import argparse
import functools
import os
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.diagnostics.pendulum_swingup_multi_kick import run_multi_kick_trial  # noqa: E402
from tools.diagnostics.pendulum_swingup_energy_shaping import (  # noqa: E402
    PendulumRunContext, add_common_pendulum_args, context_from_args,
    describe_context, write_output_json,
)
from tools.diagnostics.pendulum_lqr_cascade import DEFAULT_CONFIG, wrap_pi  # noqa: E402


def _de_workers() -> int:
    import multiprocessing as mp
    try:
        mp.set_start_method("fork")
    except RuntimeError:
        pass
    env = os.environ.get("DE_WORKERS")
    if env:
        return max(1, int(env))
    return max(1, int((os.cpu_count() or 2) * 0.9))


def first_crossing(history, phi_band_rad, min_energy_frac: float = 0.0):
    """history entries carry phi_deg measured FROM HANGING (see
    run_multi_kick_trial); convert to phi-from-inverted (exactly pi away)
    and return the first (t, phi_inv_rad, thetadot) with |phi_inv| < band
    AND E/E_top >= min_energy_frac, or None.

    min_energy_frac guards against the event-triggered pumping law's own
    repeated re-triggering: it keeps firing a kick on every bottom-pass
    regardless of energy already banked, so a WIDE phi_band can register a
    real but low-energy, still-mid-pump pass through the band as if it were
    a genuine near-top arrival. Requiring the crossing also be
    near-fully-pumped (default caller uses ~0.85) is what makes this
    "the arrival," not "some swing that happened to clip the band.\""""
    for row in history:
        phi_inv = wrap_pi(np.radians(row["phi_deg"]) - np.pi)
        if abs(phi_inv) < phi_band_rad and row["E_over_Etop"] >= min_energy_frac:
            return row["t"], phi_inv, row["thetadot"], row["E_over_Etop"]
    return None


def gentle_arrival_trial(model, kick_amplitude_m, kick_duration_s, phi_trigger_rad, *,
                          duration_s, phi_band_rad, ctx: PendulumRunContext,
                          min_energy_frac: float = 0.85) -> dict:
    res = run_multi_kick_trial(
        model, kick_amplitude_m, kick_duration_s, phi_trigger_rad, duration_s=duration_s,
        hanging_angle=ctx.hanging_angle, inverted_angle=ctx.inverted_angle,
        enforce_guard=True, track_history=True, config_path=Path(ctx.config_path),
        controller_kind=ctx.controller_kind, arm_q=ctx.arm_q_array, constants=ctx.constants,
    )
    crossing = None if res["guard_fired"] else first_crossing(res["history"], phi_band_rad, min_energy_frac)
    res["crossing"] = crossing
    return res


def objective(x, ctx: PendulumRunContext, duration_s: float, phi_band_rad: float,
              min_energy_frac: float = 0.85) -> float:
    kick_amplitude_m, kick_duration_s, phi_trigger_rad = x
    model = ctx.build_model()
    res = gentle_arrival_trial(model, kick_amplitude_m, kick_duration_s, phi_trigger_rad,
                                duration_s=duration_s, phi_band_rad=phi_band_rad, ctx=ctx,
                                min_energy_frac=min_energy_frac)
    if res["guard_fired"]:
        return 60.0
    if res["crossing"] is None:
        hist = res["history"]
        max_E = max((r["E_over_Etop"] for r in hist), default=0.0)
        return 20.0 + (1.0 - min(max_E, 1.0)) * 10.0
    _, phi_inv, thetadot, _e = res["crossing"]
    return abs(thetadot) + 2.0 * abs(phi_inv)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_pendulum_args(p, default_config=DEFAULT_CONFIG)
    p.add_argument("--maxiter", type=int, default=25)
    p.add_argument("--popsize", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--duration-s", type=float, default=2.5)
    p.add_argument("--phi-band-deg", type=float, default=35.0)
    p.add_argument("--final-duration-s", type=float, default=6.0)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    ctx = context_from_args(args).resolve()
    print(describe_context(ctx))
    phi_band_rad = np.radians(args.phi_band_deg)
    bounds = [(0.02, 0.28), (0.1, 0.5), (np.radians(3.0), np.radians(40.0))]
    print("=== searching (kick_amplitude_m, kick_duration_s, phi_trigger_rad) for GENTLE arrival ===")
    res = differential_evolution(
        functools.partial(objective, ctx=ctx, duration_s=args.duration_s, phi_band_rad=phi_band_rad),
        bounds, maxiter=args.maxiter, popsize=args.popsize, tol=1e-4, seed=args.seed,
        workers=_de_workers(), polish=False,
    )
    print("Best params:", res.x, "cost", res.fun)
    model = ctx.build_model()
    best = gentle_arrival_trial(model, res.x[0], res.x[1], res.x[2],
                                 duration_s=args.final_duration_s, phi_band_rad=phi_band_rad, ctx=ctx)
    print("Re-validated:", {k: v for k, v in best.items() if k not in ("history",)})
    if args.output_json:
        write_output_json(args.output_json, {
            "context": {"pendulum_xml": ctx.pendulum_xml, "arm_q": list(ctx.arm_q),
                        "config_path": ctx.config_path, "controller_kind": ctx.controller_kind,
                        "hanging_angle": ctx.hanging_angle, "inverted_angle": ctx.inverted_angle,
                        "constants": ctx.constants},
            "best_params": {"kick_amplitude_m": res.x[0], "kick_duration_s": res.x[1],
                            "phi_trigger_rad": res.x[2]},
            "best_cost": float(res.fun),
            "best_trial": {k: v for k, v in best.items() if k != "history"},
        })
        print("wrote", args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
