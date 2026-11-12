"""General-time (pole-subtracted) solution: checks against independent references."""

from pathlib import Path

import numpy as np
import pytest

from magnetar_qpo import ModelParams
from magnetar_qpo.direct import F_direct
from magnetar_qpo.eigen import EigenTable
from magnetar_qpo.eigen.solver import solve_table
from magnetar_qpo.initial_conditions import InitialCondition
from magnetar_qpo.residues import build_modes
from magnetar_qpo.transient import GeneralTimeSolver

TABLE370 = Path(__file__).parent.parent / "data" / "eigenvalues" / "c370_mu2_1o11"


@pytest.fixture(scope="module")
def coupled():
    p = ModelParams.make(2, 1)  # broad peaks: direct integration is a valid all-time reference
    return p, solve_table(p, lmax=3, nmax=90)


@pytest.mark.slow
def test_matches_direct_integral_at_early_and_late_times(coupled):
    p, tab = coupled
    ic = InitialCondition(0.7)
    t = np.array([0.05, 0.3, 0.8, 2.0, 8.0])
    ref = F_direct(ic, 2, 1.5, t, p, w_max=300, dw=0.005)
    late = build_modes(ic, 2, tab).F(1.5, t).real
    assert np.abs(late - ref).max() > 1e-2  # the late-time formula alone is badly wrong early on
    for gamma_s in (0.0, 0.5):  # the subtracted pole set must not matter (exact identity)
        F = GeneralTimeSolver(ic, tab, 2, omega_cut=150, gamma_s=gamma_s).F(1.5, t)
        assert np.abs(F - ref).max() < 2e-6


def test_initial_profile_is_reproduced_inside_and_outside_the_star(coupled):
    """l(l+1) F_l(r, 0+) = f_r(r), dF/dt(0+) = 0: completeness of the pole+transient solution."""
    p, tab = coupled
    ic = InitialCondition(1.6)  # exterior deposit: late-time formula is wrong by O(10) here
    sv = GeneralTimeSolver(ic, tab, 2, omega_cut=200, gamma_s=0.5)
    rs = np.array([0.5, 0.9, 1.2, 1.6, 1.8])
    got = np.array([sv.F(r, [1e-6])[0] for r in rs]) * 6
    assert np.abs(got - ic.fr(rs)).max() < 1e-4
    ic2 = InitialCondition(0.7)
    sv2 = GeneralTimeSolver(ic2, tab, 3, omega_cut=200, gamma_s=0.5)
    got2 = np.array([sv2.F(r, [1e-6])[0] for r in (0.3, 0.7, 1.0, 1.4)]) * 12
    assert np.abs(got2 - ic2.fr(np.array([0.3, 0.7, 1.0, 1.4]))).max() < 1e-4
    Ft = sv.F(1.2, [1e-6], derivatives=True)[2]
    assert abs(Ft[0]) < 1e-6


def test_spatial_and_time_derivatives_match_finite_differences(coupled):
    _, tab = coupled
    sv = GeneralTimeSolver(InitialCondition(0.7), tab, 2, omega_cut=150, gamma_s=0.5)
    t, r, h = np.array([0.4, 1.5, 5.0]), 1.3, 1e-4
    F, Fr, Ft = sv.F(r, t, derivatives=True)
    assert np.allclose(Fr, (sv.F(r + h, t) - sv.F(r - h, t)) / (2 * h), atol=1e-6)
    assert np.allclose(Ft, (sv.F(r, t + h) - sv.F(r, t - h)) / (2 * h), atol=1e-6)
    assert np.allclose(F, sv.F(r, t))


def test_hand_off_to_late_time_sum(coupled):
    _, tab = coupled
    sv_full = GeneralTimeSolver(InitialCondition(0.7), tab, 2, omega_cut=150, gamma_s=0.5)
    sv_cut = GeneralTimeSolver(InitialCondition(0.7), tab, 2, omega_cut=150, gamma_s=0.5,
                               t_cut=6.0)
    t = np.array([1.0, 5.0, 9.0, 20.0])
    assert np.abs(sv_full.F(1.2, t) - sv_cut.F(1.2, t)).max() < 1e-6


@pytest.mark.slow
@pytest.mark.skipif(not TABLE370.with_suffix(".csv").exists(), reason="c370 table missing")
def test_weak_coupling_transient_is_negligible_and_converges():
    """At c2 = 370 the narrow poles are subtracted analytically: result is independent of the
    grid spacing and agrees with the late-time sum (which is already accurate there)."""
    tab = EigenTable.load(TABLE370)
    ic = InitialCondition(0.7)
    modes = build_modes(ic, 6, tab)
    t = np.array([0.1, 0.8, 3.0, 12.0])
    a = GeneralTimeSolver(ic, tab, 6, omega_cut=200, d_omega=0.02, modes=modes).F(1.5, t)
    b = GeneralTimeSolver(ic, tab, 6, omega_cut=200, d_omega=0.01, modes=modes).F(1.5, t)
    assert np.abs(a - b).max() < 1e-10 * np.abs(a).max()
    assert np.abs(a - modes.F(1.5, t).real).max() < 1e-4 * np.abs(a).max()
