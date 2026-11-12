import math
import os
from pathlib import Path

import mpmath as mp
import pytest

from magnetar_qpo import ModelParams, PhysicalScales
from magnetar_qpo.dispersion import D
from magnetar_qpo.eigen import EigenTable, read_evalues
from magnetar_qpo.eigen.seeds import reverse_bessel_zeros
from magnetar_qpo.eigen.solver import (
    refine_root,
    required_dps,
    solve_table,
    solve_type1,
    solve_type2,
)
from magnetar_qpo.special import sph_h2

DATA = Path(__file__).parent / "data"
P370 = ModelParams.make(370, "1/11")


@pytest.fixture(scope="module")
def legacy():
    return EigenTable.load(DATA / "legacy_c370_l5_n20")


def test_type1_matches_legacy_subset(legacy):
    for l in (1, 2, 3, 5):
        roots = solve_type1(l, P370, nmax=20)
        assert len(roots) == 20
        mp.mp.dps = 120
        for n, w in enumerate(roots, start=1):
            ref = legacy[(l, n)]
            assert abs(w.real - ref.real) < mp.mpf(10) ** -55, (l, n)
            assert abs(w.imag - ref.imag) < mp.mpf(10) ** -15 * abs(ref.imag), (l, n)


def test_fundamental_frequencies():
    scales = PhysicalScales()
    w = solve_type1(2, P370, nmax=1)[0]
    assert 28.0 < scales.omega_to_hz(complex(w)) < 28.8  # paper: l=2 fundamental ~ 28 Hz
    assert abs(complex(solve_type1(1, P370, nmax=1)[0]).real - 1.1452) < 1e-4


def test_type2_count_and_imaginary_modes(legacy):
    for l in range(1, 9):
        t2 = solve_type2(l, P370)
        assert len(t2) == math.ceil(l / 2)
        assert all(w.imag > 0 for w in t2)
        if l % 2:
            assert t2[0].real == 0
            if (l, 0) in legacy:
                assert abs(t2[0].imag - legacy[(l, 0)].imag) < 1e-50 * abs(t2[0].imag)


def test_type2_seeds_are_hankel_zeros():
    for l in (3, 6):
        for x in reverse_bessel_zeros(l):
            assert abs(sph_h2(l, mp.mpc(x))) < 1e-9 * abs(sph_h2(l, mp.mpc(x) + 0.1))


def test_small_c2_imaginary_root_from_surprisepoles():
    """SurprisePoles.nb: l=3, mu2=0.001, c2=0.1 has an imaginary root near 0.23233424."""
    p = ModelParams.make("0.1", "0.001")
    w = refine_root(3, p, 0.23j, dps=50, pole_free=True)
    assert abs(w.real) < 1e-30 and abs(w.imag - 0.23233424025340868) < 1e-12
    assert abs(D(3, w, p)) < 1e-30


def test_solve_table_roundtrip(tmp_path):
    tab = solve_table(P370, lmax=3, nmax=5)
    assert (3, 0) in tab and tab.type2[(2, 1)].real > 300
    tab.save(tmp_path / "t")
    back = EigenTable.load(tmp_path / "t")
    assert back.params == P370 and set(back.omega) == set(tab.omega)
    assert set(back.type2) == set(tab.type2)
    assert abs(back[(2, 3)] - tab[(2, 3)]) < 1e-60
    assert required_dps(25, P370) > 150


@pytest.mark.slow
def test_full_legacy_table_agrees():
    root = os.environ.get("QPO_LEGACY_DIR")
    if not root:
        pytest.skip("set QPO_LEGACY_DIR to a QPO-Project clone's 'Spherical Eigenvalues' dir")
    tab = read_evalues(Path(root) / "evaluesc370", dps=300)
    # Known defects of the Mathematica table (scripts/audit_legacy_eigenvalues.py):
    #  - for even l >= 6 the first type-II root (Re ~ 321, Im > 1e3) sits in the type-I sequence
    #    (the table is sorted by Re), shifting later entries by one -> compare after dropping it;
    #  - (25, 100) is off by 1.7e-4.
    for l in tab.ls():
        legacy_t1 = [
            tab[(l, n)] for n in tab.ns(l) if tab[(l, n)].imag < 1e-3
        ]
        mine = solve_type1(l, tab.params, nmax=100)
        for n, ref in enumerate(legacy_t1, start=1):
            if (l, n) == (25, 100):
                continue
            with mp.workdps(300):
                assert abs(mine[n - 1].real - ref.real) < mp.mpf(10) ** -50, (l, n)
