import mpmath as mp
import pytest

from magnetar_qpo import ModelParams
from magnetar_qpo.dispersion import A, B, B_at_root, D, F_and_dF, dA, dA_at_root
from magnetar_qpo.special import sph_h1, sph_h2, sph_j

P = ModelParams.make(370, "1/11")
mp.mp.dps = 50


def test_param_names_and_refl():
    from fractions import Fraction

    assert P.mu_ratio == Fraction(1, 11) and P.gamma == Fraction(1, 370)
    assert P.refl == (370 - Fraction(1, 11)) / (370 + Fraction(1, 11))


@pytest.mark.parametrize("l", [1, 2, 3, 6])
def test_equal_media_limit_gives_unity(l):
    eq = ModelParams.make(1, 1)
    w = mp.mpc("3.3", "0.8")
    assert abs(A(l, w, eq) - 1) < 1e-30 and abs(B(l, w, eq) - 1) < 1e-30


@pytest.mark.parametrize("l", [1, 2, 3, 6])
def test_symmetries_and_factorisation(l):
    w = mp.mpc("7.3", "0.4")
    assert abs(B(l, w, P) - A(l, -w, P)) < 1e-30 * abs(B(l, w, P))
    assert abs(B(l, w, P) - mp.conj(A(l, mp.conj(w), P))) < 1e-30 * abs(B(l, w, P))
    x, z = w / 370, w
    fact = 1j * x * sph_j(l, z) * sph_h2(l, x) * D(l, w, P)
    assert abs(A(l, w, P) - fact) < 1e-30 * abs(A(l, w, P))


@pytest.mark.parametrize("l", [1, 2, 3, 6])
def test_derivative_matches_numerical(l):
    w0 = mp.mpc("7.3", "0.4")
    num = mp.diff(lambda w: A(l, w, P), w0)
    assert abs(num - dA(l, w0, P)) < 1e-25 * abs(num)
    _, dF = F_and_dF(l, w0, P)
    assert abs(dF - mp.diff(lambda w: F_and_dF(l, w, P)[0], w0)) < 1e-25 * abs(dF)


def test_boundary_conditions_satisfied_by_A_B():
    """Displacement and traction continuity with w_> = (A h1 + B h2)/2 at complex omega."""
    l, w = 3, mp.mpc("4.1", "0.3")
    x, z = w / 370, w
    a, b = A(l, w, P), B(l, w, P)
    j = sph_j(l, z)
    w_out = (a * sph_h1(l, x) + b * sph_h2(l, x)) / 2
    assert abs(j - w_out) < 1e-30 * abs(j)
    Pj = z * sph_j(l - 1, z) - (l + 2) * j

    def Ph(h):
        return x * h(l - 1, x) - (l + 2) * h(l, x)

    Q = P.mp()
    lhs = Q.mu1 * Pj
    rhs = Q.mu2 * (a * Ph(sph_h1) + b * Ph(sph_h2)) / 2
    assert abs(lhs - rhs) < 1e-28 * abs(lhs)


def test_root_identities():
    from magnetar_qpo.eigen.solver import refine_root

    w = refine_root(2, P, 2.861, dps=60)
    mp.mp.dps = 60
    assert abs(A(2, w, P)) < 1e-40
    assert abs(B(2, w, P) / B_at_root(2, w, P) - 1) < 1e-40
    assert abs(dA(2, w, P) / dA_at_root(2, w, P) - 1) < 1e-40
