"""Energy budget instrumentation -- the prerequisite for judging a learned policy.

Energy shaping is safe to trust partly because it carries a Lyapunov guarantee:
Edot >= 0 by construction. A learned policy has no such property, so a policy
that pumps BACKWARDS is indistinguishable from one that has not learned yet --
both just show a disappointing reward curve. This module makes the guarantee an
empirical measurement, so the same failure is one number instead of a mystery.

Ordered by what breaks worst if wrong:

  1. It separates a pumping drive from a damping one. If it cannot do that, it
     cannot serve its purpose. Calibrated against the exact failure it exists
     for: the -k_e sign at a pose whose coupling c0 is POSITIVE.
  2. The budget closes. dE == W - D within tolerance, or the model of who does
     what is wrong -- the same class of error as the sign inversion, and one
     that would otherwise be mistaken for a control failure.
  3. It refuses to report on a rollout it never saw.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.diagnostics.pendulum_energy_monitor import EnergyMonitor  # noqa: E402

# The real ARM_Q0 constants, so the calibration is not against invented numbers.
I_PIVOT, MGR, E_TOP = 2.3746980581744643e-04, 0.027870131648938572, 0.055740263297877145
C0_ARM_Q0 = +0.002841          # measured; the shipped -k_e law DAMPS against this
B_HINGE = 1.0e-4


def _roll(sign: float, steps: int = 4000, k_e: float = 60.0, dt: float = 0.002):
    """Closed-loop swing driven by the analytic law at the given sign."""
    m = EnergyMonitor(i_pivot_kgm2=I_PIVOT, mgr_nm=MGR, e_top_j=E_TOP,
                      coupling_c0=C0_ARM_Q0, hinge_damping=B_HINGE)
    phi, thd = 0.0, 0.4
    for _ in range(steps):
        e = m.energy(thd, phi)
        a = sign * k_e * thd * np.cos(phi) * (E_TOP - e)
        m.step(thetadot=thd, phi=phi, drive_accel=a, dt=dt)
        thdd = (-MGR * np.sin(phi) + C0_ARM_Q0 * np.cos(phi) * a - B_HINGE * thd) / I_PIVOT
        thd += thdd * dt
        phi += thd * dt
    return m.budget()


# ============ 1. IT SEPARATES A PUMP FROM A DAMPER =======================

def test_sign_matched_to_the_coupling_pumps():
    b = _roll(+1.0)
    assert b.net_pump
    assert b.positive_work_fraction > 0.99
    assert b.e_peak_j > 0.9 * E_TOP, "a working pump should approach E_top"


def test_the_shipped_sign_at_this_pose_damps():
    """c0 is POSITIVE at ARM_Q0, so the shipped -k_e removes energy. This is the
    bug that cost a day: k_e=0 and k_e=50 gave bit-identical results because
    both ended at rest."""
    b = _roll(-1.0)
    assert not b.net_pump
    assert b.positive_work_fraction < 0.01
    assert b.e_peak_j < 0.01 * E_TOP


def test_the_two_are_unambiguous_not_marginal():
    """A diagnostic that only just separates them would not survive noise."""
    good, bad = _roll(+1.0), _roll(-1.0)
    assert good.positive_work_fraction - bad.positive_work_fraction > 0.9
    assert good.work_by_drive_j > 0 > bad.work_by_drive_j


# ============ 2. THE BUDGET CLOSES ======================================

def test_budget_closes_for_the_pumping_case():
    """dE == W - D. A large residual means the accounting is wrong, which would
    otherwise be misread as a control failure."""
    assert _roll(+1.0).closes < 0.05


def test_budget_closes_for_the_damping_case():
    assert _roll(-1.0).closes < 0.05


# ============ 3. IT WILL NOT INVENT A REPORT ============================

def test_reports_nothing_without_a_rollout():
    m = EnergyMonitor(i_pivot_kgm2=I_PIVOT, mgr_nm=MGR, e_top_j=E_TOP, coupling_c0=C0_ARM_Q0)
    with pytest.raises(ValueError, match="never called"):
        m.budget()


def test_zero_drive_does_no_work_and_only_dissipates():
    m = EnergyMonitor(i_pivot_kgm2=I_PIVOT, mgr_nm=MGR, e_top_j=E_TOP,
                      coupling_c0=C0_ARM_Q0, hinge_damping=B_HINGE)
    phi, thd, dt = 0.0, 0.4, 0.002
    for _ in range(2000):
        m.step(thetadot=thd, phi=phi, drive_accel=0.0, dt=dt)
        thdd = (-MGR * np.sin(phi) - B_HINGE * thd) / I_PIVOT
        thd += thdd * dt
        phi += thd * dt
    b = m.budget()
    assert b.work_by_drive_j == 0.0
    assert b.positive_work_fraction == 0.0
    assert b.dissipated_j > 0.0
