"""Pure-numpy unit coverage for tools/diagnostics/pendulum_swingup_curved.py's
``curved_pump_accel``/``pendulum_edot`` -- the CURVED (2-D) pivot-pumping
swing-up law itself, with no mujoco/model dependency.

What these pin down (see that module's own docstring for the derivation):
  1. a_vec = -k_e*thetadot*(E_top-E)*(cos phi, sin phi): the pump component
     scales as cos(phi), the vertical component as sin(phi), sharing the
     SAME scalar prefactor -- i.e. this is a genuine vector generalization
     of the 1-axis law (pendulum_swingup_energy_shaping.py's
     ``a_energy = -k_e*thetadot*cos(phi)*(E_top-E)``), not a different law
     that happens to agree at phi=0.
  2. Edot = -M*r*thetadot*(a_vec . n_hat) is >= 0 for k_e >= 0 and E <= E_top
     -- the Lyapunov property the whole design rests on: this law can only
     ever ADD energy, never remove it, regardless of phi or thetadot's sign.
  3. The law is identically zero at thetadot=0 (both components) -- the
     bootstrap problem the trial loop's seed kick exists to solve.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.diagnostics.pendulum_swingup_curved import (  # noqa: E402
    curved_pump_accel,
    pendulum_edot,
)

PHIS = np.linspace(-np.pi, np.pi, 25)
THETADOTS = [-3.0, -0.7, -1e-6, 0.0, 1e-6, 0.7, 3.0]


@pytest.mark.parametrize("thetadot", THETADOTS)
@pytest.mark.parametrize("phi", PHIS)
def test_components_scale_as_cos_and_sin_of_phi(thetadot, phi):
    """a_pump/common == cos(phi) and a_vert/common == sin(phi) EXACTLY (this
    is a closed-form formula, not a fit) for every phi and thetadot, with
    the SAME scalar prefactor `common` for both components."""
    k_e, e_top, E = 37.0, 0.5, 0.1
    common = -k_e * thetadot * (e_top - E)
    a_pump, a_vert = curved_pump_accel(thetadot, phi, E, e_top, k_e)
    assert a_pump == pytest.approx(common * np.cos(phi), abs=1e-12)
    assert a_vert == pytest.approx(common * np.sin(phi), abs=1e-12)


def test_reduces_to_the_1_axis_law_at_phi_zero():
    """At phi=0 (hanging), cos(phi)=1 and sin(phi)=0: a_pump must equal
    EXACTLY pendulum_swingup_energy_shaping.py's scalar law,
    -k_e*thetadot*cos(phi)*(E_top-E), and a_vert must be exactly zero --
    i.e. the curved law is a strict generalization, not a different
    formula that happens to coincide numerically."""
    k_e, thetadot, e_top, E = 12.3, 0.42, 0.9, 0.2
    a_pump, a_vert = curved_pump_accel(thetadot, 0.0, E, e_top, k_e)
    expected_1d = -k_e * thetadot * np.cos(0.0) * (e_top - E)
    assert a_pump == pytest.approx(expected_1d, abs=1e-12)
    assert a_vert == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("phi", PHIS)
def test_zero_at_rest(phi):
    """thetadot=0 must make BOTH components identically zero -- the
    bootstrap problem a seed kick exists to solve (module docstring)."""
    a_pump, a_vert = curved_pump_accel(0.0, phi, 0.1, 0.5, 100.0)
    assert a_pump == 0.0
    assert a_vert == 0.0


@pytest.mark.parametrize("phi", PHIS)
def test_zero_at_energy_ceiling(phi):
    """E == E_top must make BOTH components identically zero -- the law
    self-limits at the top of the energy budget regardless of thetadot."""
    a_pump, a_vert = curved_pump_accel(2.5, phi, 0.5, 0.5, 100.0)
    assert a_pump == 0.0
    assert a_vert == 0.0


@pytest.mark.parametrize("thetadot", [-3.0, -0.5, 0.5, 3.0])
@pytest.mark.parametrize("phi", PHIS)
@pytest.mark.parametrize("E", [0.0, 0.02, 0.04])
def test_edot_is_nonnegative_below_the_energy_ceiling(thetadot, phi, E):
    """The Lyapunov property: for k_e>=0 and E<=E_top, applying the law's
    OWN output back through Edot = -M*r*thetadot*(a.n_hat) must never
    decrease energy, for any (phi, thetadot) and any E below the ceiling."""
    k_e, e_top, mgr_nm, g = 40.0, 0.05574, 0.0278701, 9.81
    a_pump, a_vert = curved_pump_accel(thetadot, phi, E, e_top, k_e)
    edot = pendulum_edot(thetadot, mgr_nm, g, a_pump, a_vert, phi)
    assert edot >= -1e-9


def test_edot_scales_with_k_e_and_energy_gap():
    """Edot = M*r*k_e*thetadot^2*(E_top-E) in closed form (substitute the
    law into the Edot formula and simplify) -- check that closed form
    directly, not just its sign, at one representative point."""
    k_e, thetadot, phi, e_top, E = 25.0, 1.3, 0.7, 0.05574, 0.01
    mgr_nm, g = 0.0278701, 9.81
    m_r = mgr_nm / g
    a_pump, a_vert = curved_pump_accel(thetadot, phi, E, e_top, k_e)
    edot = pendulum_edot(thetadot, mgr_nm, g, a_pump, a_vert, phi)
    expected = m_r * k_e * thetadot ** 2 * (e_top - E)
    assert edot == pytest.approx(expected, rel=1e-9)


def test_negative_k_e_can_remove_energy():
    """Sanity check on the sign convention: with k_e<0 (a damper, not a
    pump -- the sign pendulum_swingup_energy_shaping.py's own history
    documents getting wrong once, see that module's docstring), Edot must
    be <= 0, the mirror image of the k_e>=0 case."""
    thetadot, phi, e_top, E = 1.1, 0.9, 0.05574, 0.01
    mgr_nm, g = 0.0278701, 9.81
    a_pump, a_vert = curved_pump_accel(thetadot, phi, E, e_top, -30.0)
    edot = pendulum_edot(thetadot, mgr_nm, g, a_pump, a_vert, phi)
    assert edot <= 1e-9
