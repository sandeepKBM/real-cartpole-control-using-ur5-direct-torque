"""Does a swing-up drive ADD energy? The check that makes a learned policy legible.

THE RL-SHAPED QUESTION. Replacing the energy-shaping law with a state-dependent
learned policy u = pi(theta, thetadot, x, xdot) is a real proposal -- unlike
using RL to pick static gains, which is a bandit and strictly worse than a
proper optimizer (see pendulum_search_backends). It is worth taking seriously
because the phasing of a 2-D curved pump is a genuine sequential-decision
problem that no closed form obviously solves.

THE OBJECTION, AND WHY THIS MODULE EXISTS. Energy shaping carries a Lyapunov
guarantee:

    Edot = c0 * a * thetadot,   a = sign(c0) * k_e * thetadot * cos(phi) * (E_top - E)
    =>  Edot proportional to (E_top - E) * (thetadot cos phi)^2  >=  0

Energy can only increase, and the drive self-limits to zero at the top. That
guarantee is not decoration -- it is why the sign inversion found 2026-08-16 was
DETECTABLE: a law that provably cannot remove energy, removing energy, is a
contradiction you can see in one trace. A learned policy has no such property,
so a policy that pumps backwards is indistinguishable from a policy that simply
has not learned yet, and both look like "reward went down".

So before any policy is trained, the same quantity must be observable
empirically. EnergyMonitor accumulates the MEASURED energy budget of a rollout:

    E(t)            the pendulum's actual mechanical energy
    W_drive         work done on the hinge by the commanded pivot acceleration
    dissipated      what friction/damping removed

and reports the fraction of steps where the drive did POSITIVE work. For the
analytic law that fraction is ~1 by construction; for a learned policy it is the
headline diagnostic, and a policy below ~0.5 is removing energy on balance no
matter what its reward curve says.

Deliberately NOT a training loop. This repo's RL record on a strictly easier
problem is 0/20, 0/20, 1/20 across three reward redesigns and ~4.4M steps versus
100% for fixed gains (AGENTS.md), so the sequencing that makes sense is:
instrument first, establish that the analytic baseline scores ~1.0, and only
then judge a policy against a number that means something.
"""
from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass
class EnergyBudget:
    """Closed energy budget for one rollout."""

    e_initial_j: float
    e_final_j: float
    e_peak_j: float
    e_top_j: float
    work_by_drive_j: float
    dissipated_j: float
    positive_work_fraction: float
    steps: int

    @property
    def closes(self) -> float:
        """|dE - (W - D)| / E_top: how well the budget balances.

        A large residual means the model of who-does-what is wrong -- the same
        class of error as the sign inversion, and visible here before it can be
        mistaken for a control failure.
        """
        residual = (self.e_final_j - self.e_initial_j) - (self.work_by_drive_j - self.dissipated_j)
        return abs(residual) / max(self.e_top_j, 1e-12)

    @property
    def net_pump(self) -> bool:
        """Did the drive add energy on balance?"""
        return self.work_by_drive_j > 0.0

    def summary(self) -> str:
        return (
            f"E {self.e_initial_j:.5f} -> {self.e_final_j:.5f} J "
            f"(peak {self.e_peak_j:.5f}, E_top {self.e_top_j:.5f})\n"
            f"  work by drive   {self.work_by_drive_j:+.5f} J   "
            f"{'PUMPS' if self.net_pump else '*** DAMPS ***'}\n"
            f"  dissipated      {self.dissipated_j:.5f} J\n"
            f"  positive-work steps {100 * self.positive_work_fraction:.1f}%   "
            f"(analytic law ~100%; < 50% means the drive removes energy)\n"
            f"  budget closes to {100 * self.closes:.2f}% of E_top"
        )


class EnergyMonitor:
    """Accumulate the energy budget of a swing-up rollout, step by step.

    ``coupling_c0`` is Q/a at the hanging state along the drive direction, from
    ``measure_pivot_coupling``. The instantaneous hinge torque from the pivot is
    ``Q = c0 * cos(phi) * a`` (the cos(phi) shape is measured, see AGENTS.md's
    a_par/a_z table), so the work increment is ``Q * thetadot * dt``.
    """

    def __init__(self, *, i_pivot_kgm2: float, mgr_nm: float, e_top_j: float,
                 coupling_c0: float, hinge_damping: float = 0.0):
        self.i_pivot = float(i_pivot_kgm2)
        self.mgr = float(mgr_nm)
        self.e_top = float(e_top_j)
        self.c0 = float(coupling_c0)
        self.b = float(hinge_damping)
        self._e0 = None
        self._e = 0.0
        self._peak = -np.inf
        self._work = 0.0
        self._diss = 0.0
        self._pos = 0
        self._n = 0

    def energy(self, thetadot: float, phi: float) -> float:
        return 0.5 * self.i_pivot * thetadot * thetadot + self.mgr * (1.0 - np.cos(phi))

    def step(self, *, thetadot: float, phi: float, drive_accel: float, dt: float) -> None:
        e = self.energy(thetadot, phi)
        if self._e0 is None:
            self._e0 = e
        self._e = e
        self._peak = max(self._peak, e)
        dw = self.c0 * np.cos(phi) * float(drive_accel) * float(thetadot) * float(dt)
        self._work += dw
        self._diss += self.b * thetadot * thetadot * float(dt)
        if dw > 0.0:
            self._pos += 1
        self._n += 1

    def budget(self) -> EnergyBudget:
        if self._n == 0:
            raise ValueError("EnergyMonitor.step was never called")
        return EnergyBudget(
            e_initial_j=float(self._e0), e_final_j=float(self._e),
            e_peak_j=float(self._peak), e_top_j=self.e_top,
            work_by_drive_j=float(self._work), dissipated_j=float(self._diss),
            positive_work_fraction=self._pos / self._n, steps=self._n,
        )
