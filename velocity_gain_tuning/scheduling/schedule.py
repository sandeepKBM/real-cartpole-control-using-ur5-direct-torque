"""The gain schedule itself: a smooth map (pose, target_x_delta_m) -> gains.

Pure numpy/scipy, no MuJoCo, no gym -- deliberately, so the whole
interpolation layer is unit-testable without a simulator (AGENTS.md sec 5)
and so it can be imported by the real-hardware lane later without dragging
in the tuning stack.

THREE DESIGN CHOICES THAT MATTER, AND WHY
-----------------------------------------
1. **Interpolate in ACTION space ([-1,1]^ACTION_DIM), not in physical gain
   space.** ``velocity_transport_env.action_to_gains`` already defines the
   canonical remap, including LOG-scaling for ``pinv_damping``,
   ``qp_task_weight`` and ``ik_max_joint_deviation_rad`` (each spans 5-10
   orders of magnitude). Interpolating linearly between, say,
   qp_task_weight=1e2 and 1e9 in physical space would put the midpoint at
   5e8 -- i.e. essentially the upper knot -- whereas the action-space
   midpoint is the geometric mean 1e5.5, which is what "halfway between
   these two settings" actually means for a log-scaled gain. Interpolating
   in action space also makes staying inside the validated physical bounds
   automatic (a clip to [-1,1] is all that is required), instead of
   something the interpolant could silently violate.

2. **PCHIP (shape-preserving piecewise cubic Hermite), not a natural cubic
   spline.** A natural cubic spline through unevenly-spaced, noisy knots
   OVERSHOOTS -- it can produce interpolated values outside the hull of
   the surrounding data. For gains that means the schedule could command,
   between two searched-and-validated knots, a gain more extreme than any
   gain that was ever evaluated. PCHIP is monotonicity- and
   local-extremum-preserving by construction: an interpolated value always
   lies between its two bracketing knot values. Every point the schedule
   ever emits is therefore inside the convex hull of two validated
   settings. That property is worth more here than the extra smoothness
   (C2 vs C1) a natural spline would buy.

3. **Clamp, never extrapolate.** Queries outside the knot range return the
   nearest end knot's action. Extrapolating a cubic past the last searched
   displacement is exactly how a schedule would hand the real robot a gain
   vector that nothing ever evaluated. The knot grid spans +/-2.0x each
   scenario's ``max_dx_hint_m``, well past its known safe boundary, so
   clamping only ever engages beyond the range where the controller is
   already known to fail.

Poses are handled as DISCRETE lookups, not interpolated. The four
scenarios differ in several joints at once (base rotation, shoulder,
wrist_2 offset), so there is no single scalar to interpolate along, and
inventing one would be a fabricated axis. This is a real limitation: the
schedule can only be queried at a pose it was fitted at. See the handoff
doc for the "fit against a continuous pose parameter" follow-on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.interpolate import PchipInterpolator


def _action_to_gains(action: np.ndarray) -> dict[str, float]:
    """Canonical action -> physical gain remap, imported LAZILY.

    ``velocity_transport_env`` pulls in gymnasium, controller_core and
    MuJoCo-backed local dynamics. Keeping that import inside the two
    functions that actually need the remap is what makes this module
    importable (and unit-testable) with nothing but numpy/scipy -- which
    matters because the schedule itself is the artifact a hardware lane
    would eventually consume, and it should not have to load a simulator
    to look up a gain vector.
    """
    from ..envs.velocity_transport_env import action_to_gains

    return action_to_gains(action)


def _infer_action_dim(actions: Iterable[np.ndarray]) -> int:
    """Action dimension taken from the knots themselves rather than
    imported, so a schedule JSON stays readable even if ACTION_FIELDS
    later gains or loses a field (this repo has already added and removed
    three redundancy-resolution fields in a single day). A mismatch
    against the live controller surfaces at ``gains_for`` time with a
    clear error, instead of silently reshaping here."""
    dims = {int(np.asarray(a, dtype=np.float64).ravel().shape[0]) for a in actions}
    if len(dims) > 1:
        raise ValueError(f"knot actions have inconsistent dimensions: {sorted(dims)}")
    return dims.pop() if dims else 0


@dataclass(frozen=True)
class ScheduleKnot:
    """One searched (pose, displacement) cell and the gains found for it."""

    scenario: str
    dx_fraction: float
    target_x_delta_m: float
    action: np.ndarray
    fitness: float = 0.0
    # True iff the best action found for this cell completed BOTH its slow
    # and fast episode without tripping a guard. False means "no gain
    # vector this search could find makes this cell safe" -- kept (not
    # discarded) because that is a first-class result, but see
    # GainSchedule(drop_failed_knots=...) for why it may be excluded from
    # the fit.
    passed: bool = True
    guard_reasons: tuple[str | None, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        d = {
            "scenario": self.scenario,
            "dx_fraction": self.dx_fraction,
            "target_x_delta_m": self.target_x_delta_m,
            "action": np.asarray(self.action, dtype=np.float64).tolist(),
            "fitness": float(self.fitness),
            "passed": bool(self.passed),
            "guard_reasons": list(self.guard_reasons),
        }
        # Physical gains are written for HUMAN READABILITY only -- the
        # action vector is the authoritative record, and from_dict never
        # reads this field back. Emitting it requires the gym/MuJoCo-backed
        # env module, so it is skipped rather than fatal when this module
        # is used somewhere that stack isn't installed (see
        # _action_to_gains).
        try:
            d["gains"] = _action_to_gains(np.asarray(self.action, dtype=np.float64))
        except ImportError:  # pragma: no cover - depends on install layout
            pass
        return d

    @staticmethod
    def from_dict(d: dict) -> "ScheduleKnot":
        return ScheduleKnot(
            scenario=str(d["scenario"]),
            dx_fraction=float(d["dx_fraction"]),
            target_x_delta_m=float(d["target_x_delta_m"]),
            action=np.asarray(d["action"], dtype=np.float64),
            fitness=float(d.get("fitness", 0.0)),
            passed=bool(d.get("passed", True)),
            guard_reasons=tuple(d.get("guard_reasons", ())),
        )


def smooth_actions(actions: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average of shape (n_knots, ACTION_DIM) along the
    knot axis, with edge-clamped padding.

    Why this exists: each knot is searched INDEPENDENTLY, and the
    single-cell objective has genuinely flat directions (e.g. once
    ``qp_task_weight`` is large enough, further increases change nothing
    measurable), so neighbouring knots can land at very different points
    that score identically. Interpolating between them produces a schedule
    that swings hard between two settings for no measured benefit. A small
    moving average trades a little per-knot optimality for a schedule that
    varies gently -- worth testing, not worth assuming, which is why it is
    an option compared empirically rather than a hardcoded step.

    ``window <= 1`` is a no-op (returns a copy).
    """
    actions = np.asarray(actions, dtype=np.float64)
    if window <= 1 or actions.shape[0] == 0:
        return actions.copy()
    if window % 2 == 0:
        raise ValueError(f"smoothing window must be odd, got {window}")
    half = window // 2
    padded = np.pad(actions, ((half, half), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    out = np.empty_like(actions)
    for k in range(actions.shape[1]):
        out[:, k] = np.convolve(padded[:, k], kernel, mode="valid")
    return out


class GainSchedule:
    """Smooth (pose, target_x_delta_m) -> action/gains map fitted to knots."""

    def __init__(
        self,
        knots: Iterable[ScheduleKnot],
        *,
        drop_failed_knots: bool = False,
        smoothing_window: int = 1,
    ) -> None:
        """``drop_failed_knots``: exclude knots whose own best-found action
        still tripped a guard. Rationale: at a cell where NO gain vector
        found by the search is safe, the DE result is essentially an
        arbitrary point on a uniformly-bad landscape -- letting it anchor
        the interpolant drags the schedule around for its two neighbouring
        (feasible) cells too. Dropping it means those neighbours' gains are
        carried across the infeasible region instead. This cannot rescue
        the infeasible cell (nothing can), it only stops it from damaging
        the cells around it. Off by default because it discards real
        measured data; enabled explicitly when the comparison shows it
        helps. If EVERY knot for a scenario failed there is nothing left to
        drop, so all of that scenario's knots are kept.
        """
        self._smoothing_window = int(smoothing_window)
        self._drop_failed_knots = bool(drop_failed_knots)
        self.knots: list[ScheduleKnot] = list(knots)
        if not self.knots:
            raise ValueError("GainSchedule needs at least one knot")
        self.action_dim = _infer_action_dim(k.action for k in self.knots)

        by_scenario: dict[str, list[ScheduleKnot]] = {}
        for knot in self.knots:
            by_scenario.setdefault(knot.scenario, []).append(knot)

        self._interpolators: dict[str, list] = {}
        self._dx_bounds: dict[str, tuple[float, float]] = {}
        self._fit_knots: dict[str, list[ScheduleKnot]] = {}

        for scenario, scenario_knots in by_scenario.items():
            used = scenario_knots
            if drop_failed_knots:
                kept = [k for k in scenario_knots if k.passed]
                if kept:
                    used = kept
            used = sorted(used, key=lambda k: k.target_x_delta_m)
            # Guard against duplicate displacements (PchipInterpolator
            # requires strictly increasing x): keep the first occurrence.
            deduped: list[ScheduleKnot] = []
            for knot in used:
                if deduped and abs(knot.target_x_delta_m - deduped[-1].target_x_delta_m) < 1e-12:
                    continue
                deduped.append(knot)
            used = deduped
            self._fit_knots[scenario] = used

            xs = np.array([k.target_x_delta_m for k in used], dtype=np.float64)
            actions = np.array(
                [np.asarray(k.action, dtype=np.float64).reshape(self.action_dim) for k in used],
                dtype=np.float64,
            )
            actions = smooth_actions(actions, self._smoothing_window)
            self._dx_bounds[scenario] = (float(xs[0]), float(xs[-1]))
            if len(used) == 1:
                # Degenerate but legal: a single knot means a constant
                # schedule for that pose.
                const = actions[0].copy()
                self._interpolators[scenario] = [
                    (lambda _x, v=float(const[k]): np.float64(v)) for k in range(self.action_dim)
                ]
            else:
                self._interpolators[scenario] = [
                    PchipInterpolator(xs, actions[:, k], extrapolate=False)
                    for k in range(self.action_dim)
                ]

    @property
    def scenarios(self) -> tuple[str, ...]:
        return tuple(sorted(self._interpolators))

    def fitted_knots(self, scenario: str) -> list[ScheduleKnot]:
        """Knots actually used for the fit (post drop_failed_knots/dedupe)."""
        return list(self._fit_knots[scenario])

    def action_for(self, scenario: str, target_x_delta_m: float) -> np.ndarray:
        """Interpolated action vector, clipped to the legal [-1,1] box.

        Queries outside the fitted displacement range are CLAMPED to the
        nearest end knot (never extrapolated) -- see the module docstring.
        """
        if scenario not in self._interpolators:
            raise KeyError(
                f"schedule has no knots for scenario {scenario!r}; known: {self.scenarios}"
            )
        lo, hi = self._dx_bounds[scenario]
        x = float(np.clip(float(target_x_delta_m), lo, hi))
        action = np.array(
            [float(interp(x)) for interp in self._interpolators[scenario]],
            dtype=np.float64,
        )
        return np.clip(action, -1.0, 1.0)

    def gains_for(self, scenario: str, target_x_delta_m: float) -> dict[str, float]:
        """Physical gain dict for a (pose, displacement), via the canonical
        ``action_to_gains`` remap -- so a scheduled gain set is always a
        point in the exact same parameter space the global search used."""
        return _action_to_gains(self.action_for(scenario, target_x_delta_m))

    # ---------------- serialization ----------------

    def to_dict(self) -> dict:
        return {
            "format": "velocity_gain_tuning.scheduling.GainSchedule/v1",
            "action_dim": self.action_dim,
            "drop_failed_knots": self._drop_failed_knots,
            "smoothing_window": self._smoothing_window,
            "knots": [k.to_dict() for k in self.knots],
        }

    @staticmethod
    def from_dict(d: dict, **overrides) -> "GainSchedule":
        kwargs = {
            "drop_failed_knots": bool(d.get("drop_failed_knots", False)),
            "smoothing_window": int(d.get("smoothing_window", 1)),
        }
        kwargs.update(overrides)
        return GainSchedule([ScheduleKnot.from_dict(k) for k in d["knots"]], **kwargs)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @staticmethod
    def load(path: str | Path, **overrides) -> "GainSchedule":
        return GainSchedule.from_dict(
            json.loads(Path(path).read_text(encoding="utf-8")), **overrides
        )
