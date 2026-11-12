from pathlib import Path

import numpy as np
import pytest

from magnetar_qpo import PhysicalScales
from magnetar_qpo.eigen import EigenTable
from magnetar_qpo.initial_conditions import InitialCondition
from magnetar_qpo.solution import solve_ivp
from magnetar_qpo.spectrum import lorentzian_mode_amplitude, mode_spectrum, windowed_dft

TABLE = Path(__file__).parent.parent / "data" / "eigenvalues" / "c370_mu2_1o11"


@pytest.fixture(scope="module")
def sol():
    if not TABLE.with_suffix(".csv").exists():
        pytest.skip("run scripts/make_eigen_table.py to create the c370 table")
    tab = EigenTable.load(TABLE)
    return solve_ivp(InitialCondition(0.7), tab, lmax=6, nmax=6).with_energy(1e46, 1e47)


def test_residue_amplitude_equals_paper_lorentzian_formula(sol):
    spec = mode_spectrum(sol)
    for l, n in [(3, 1), (5, 2), (6, 3)]:
        i = np.nonzero((spec.l == l) & (spec.n == n))[0][0]
        assert abs(lorentzian_mode_amplitude(sol, l, n) / spec.amplitude[i] - 1) < 1e-4


def test_l2_fundamental_is_28hz_and_no_l1_modes_excited(sol):
    spec = mode_spectrum(sol)
    i = np.nonzero((spec.l == 2) & (spec.n == 1))[0][0]
    assert 28.0 < spec.freq_hz[i] < 28.8
    assert spec.amplitude[spec.l == 1].max() < 1e-12


def test_time_series_oscillates_at_mode_frequencies(sol):
    t = np.arange(5.0, 105.0, 0.05)
    u = sol.probe_displacement(t)
    f, a = windowed_dft(t, u, PhysicalScales(), n_windows=1, f_max_hz=120)
    spec = mode_spectrum(sol).select(0, 120)
    f_peak = f[np.argmax(a)]
    strongest = spec.freq_hz[np.argmax(spec.amplitude)]
    assert abs(f_peak - strongest) < 1.5  # DFT bin width ~0.6 Hz here
