"""The residue-theorem solution must agree with direct integration over real frequency."""

import numpy as np
import pytest

from magnetar_qpo import ModelParams
from magnetar_qpo.direct import F_direct
from magnetar_qpo.eigen.solver import solve_table
from magnetar_qpo.initial_conditions import InitialCondition
from magnetar_qpo.residues import build_modes


@pytest.fixture(scope="module")
def strongly_coupled():
    p = ModelParams.make(2, 1)  # broad spectral peaks: direct quadrature is feasible
    return p, solve_table(p, lmax=3, nmax=60)


@pytest.mark.parametrize("l", [2, 3])
@pytest.mark.parametrize("r", [1.0, 1.5])
def test_residue_sum_matches_direct_integral(strongly_coupled, l, r):
    p, tab = strongly_coupled
    ic = InitialCondition(0.7)
    t = np.array([1.0, 2.0, 3.5, 6.0])
    modes = build_modes(ic, l, tab)
    F = modes.F(r, t)
    Fd = F_direct(ic, l, r, t, p, w_max=200, dw=0.01)
    assert np.abs(F.imag).max() < 1e-12  # mirrored poles make the field real
    # both sums are truncated in frequency; agreement improves with nmax / w_max
    assert np.abs(F.real - Fd).max() < 3e-5 * np.abs(Fd).max() + 1e-7


def test_mirror_and_type2_poles_matter(strongly_coupled):
    """Dropping poles must visibly break agreement (guards against vacuous tests)."""
    p, tab = strongly_coupled
    ic = InitialCondition(0.7)
    t = np.array([1.0, 2.0])
    Fd = F_direct(ic, 2, 1.0, t, p, w_max=200, dw=0.01)
    no_mirror = build_modes(ic, 2, tab, include_mirror=False).F(1.0, t)
    assert np.abs(no_mirror.real - Fd).max() > 0.1 * np.abs(Fd).max()


def test_quadrature_convergence(strongly_coupled):
    p, tab = strongly_coupled
    ic = InitialCondition(0.7)
    a = build_modes(ic, 2, tab, n_nodes=400).F(1.0, [2.0])
    b = build_modes(ic, 2, tab, n_nodes=1200).F(1.0, [2.0])
    assert abs(a - b).max() < 1e-9


@pytest.mark.slow
def test_luminosity_matches_direct_derivatives(strongly_coupled):
    """Power through r = R* from the residue sum vs. the same formula built from the direct
    frequency integral with accurate finite-difference derivatives."""
    from magnetar_qpo.luminosity import alfven_luminosity
    from magnetar_qpo.solution import solve_ivp

    p, tab = strongly_coupled
    ic = InitialCondition(0.7)
    sol = solve_ivp(ic, tab, lmax=3, lmin=2, nmax=60)
    t0 = np.array([2.0, 4.0])
    P_res = alfven_luminosity(sol, 1.0, t0)

    from magnetar_qpo.luminosity import angular_weights

    S = angular_weights(sol)
    ht, hr = 1e-3, 1e-3
    kw = dict(w_max=200, dw=0.01)
    P_dir = np.zeros_like(t0)
    for l in (2, 3):
        Ft = (F_direct(ic, l, 1.0, t0 + ht, p, **kw) - F_direct(ic, l, 1.0, t0 - ht, p, **kw)) / (
            2 * ht
        )
        f0 = F_direct(ic, l, 1.0, t0, p, **kw)
        f1 = F_direct(ic, l, 1.0 + hr, t0, p, **kw)
        f2 = F_direct(ic, l, 1.0 + 2 * hr, t0, p, **kw)
        Fr = (-3 * f0 + 4 * f1 - f2) / (2 * hr)
        P_dir -= float(p.mu2) * l * (l + 1) * S[l] * Ft * (Fr - f0)
    assert np.abs(P_res - P_dir).max() < 2e-3 * np.abs(P_dir).max()
