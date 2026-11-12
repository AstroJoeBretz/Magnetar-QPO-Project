import mpmath as mp
import pytest

from magnetar_qpo.special import dsph_h, dsph_j, sph_h1, sph_h2, sph_j

mp.mp.dps = 40
ZS = [mp.mpf("0.37"), mp.mpf("3.1"), mp.mpc("2.2", "1.5"), mp.mpc("0.4", "-2.7"), mp.mpf(25)]


@pytest.mark.parametrize("l", [0, 1, 2, 5, 12])
@pytest.mark.parametrize("z", ZS)
def test_matches_mpmath_bessel(l, z):
    ref_j = mp.sqrt(mp.pi / (2 * z)) * mp.besselj(l + 0.5, z)
    ref_y = mp.sqrt(mp.pi / (2 * z)) * mp.bessely(l + 0.5, z)
    scale = 1 + abs(ref_j) + abs(ref_y)
    assert abs(sph_j(l, z) - ref_j) < 1e-28 * scale
    assert abs(sph_h1(l, z) - (ref_j + 1j * ref_y)) < 1e-28 * scale
    assert abs(sph_h2(l, z) - (ref_j - 1j * ref_y)) < 1e-28 * scale


@pytest.mark.parametrize("l", [1, 2, 7])
@pytest.mark.parametrize("z", ZS)
def test_wronskian_and_parity(l, z):
    w = sph_h1(l - 1, z) * sph_h2(l, z) - sph_h2(l - 1, z) * sph_h1(l, z)
    assert abs(w - 2j / z**2) < 1e-25 * abs(1 / z**2)
    assert abs(sph_h1(l, -z) - (-1) ** l * sph_h2(l, z)) < 1e-25 * (1 + abs(sph_h2(l, z)))
    assert abs(sph_j(l, -z) - (-1) ** l * sph_j(l, z)) < 1e-25 * (1 + abs(sph_j(l, z)))


@pytest.mark.parametrize("l", [0, 1, 4])
def test_derivatives(l):
    z = mp.mpc("1.3", "0.7")
    assert abs(dsph_j(l, z) - mp.diff(lambda t: sph_j(l, t), z)) < 1e-25
    for kind, fn in ((1, sph_h1), (2, sph_h2)):
        num = mp.diff(lambda t, fn=fn: fn(l, t), z)
        assert abs(dsph_h(l, z, kind) - num) < 1e-25
