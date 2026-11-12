r"""Boundary-condition (dispersion) functions A_l(omega), B_l(omega) and their derivatives.

With ``z = Rs*omega/c1`` (interior argument) and ``x = Rs*omega/c2`` (exterior argument)
the continuity of displacement and traction at ``r = Rs`` for the radial functions

    w_<(r) = j_l(k1 r),     w_>(r) = (A h1_l(k2 r) + B h2_l(k2 r)) / 2

gives (paper Appendix C, eqs. A and B; derived here from the 2x2 linear system and the
Wronskian ``h1_{l-1} h2_l - h2_{l-1} h1_l = 2i/x^2``)::

    P_j(z) = z j_{l-1}(z) - (l+2) j_l(z)
    P_h(x) = x h_{l-1}(x) - (l+2) h_l(x)
    A = i x [ j_l(z) P_h2(x) - (mu1/mu2) P_j(z) h2_l(x) ]
    B = i x [ (mu1/mu2) P_j(z) h1_l(x) - j_l(z) P_h1(x) ]

Factoring ``j_l(z) h2_l(x)`` out of ``A`` leaves the meromorphic *reduced* function

    D_l(omega) = [x h2_{l-1}/h2_l - (l+2)] - (mu1/mu2) [z j_{l-1}/j_l - (l+2)],
    A_l = i x j_l(z) h2_l(x) D_l(omega),

whose zeros are the eigenfrequencies (paper eq. ShiftBoundaryCondition).  At a zero,
``B_l(omega_0) = 2 j_l(z)/h2_l(x)`` and ``A_l'(omega_0) = i x j_l h2_l D_l'(omega_0)``.

Note on normalisation: the Mathematica notebooks' ``A2``/``A1`` equal ``(1-refl) A``/``(1-refl) B``
with eigenfunction interior ``(1-refl) j_l``; this package uses the paper's normalisation
(interior ``j_l``) throughout.
"""

from __future__ import annotations

import mpmath as mp

from .params import ModelParams
from .special import dsph_h, dsph_j, riccati_q, sph_h1, sph_h2, sph_j


def _args(w, P):
    w = mp.mpmathify(w)
    return w, P.Rs * w / P.c1, P.Rs * w / P.c2


def D(l: int, w, params: ModelParams):
    """Reduced dispersion function D_l(omega) (zeros = eigenfrequencies)."""
    P = params.mp()
    w, z, x = _args(w, P)
    q, _ = riccati_q(l, sph_h2(l - 1, x) / sph_h2(l, x), x)
    p, _ = riccati_q(l, sph_j(l - 1, z) / sph_j(l, z), z)
    return (q - (l + 2)) - P.m * (p - (l + 2))


def D_and_dD(l: int, w, params: ModelParams):
    """Return (D_l(omega), dD_l/domega) using the Riccati identity (no extra Bessel calls)."""
    P = params.mp()
    w, z, x = _args(w, P)
    q, dq = riccati_q(l, sph_h2(l - 1, x) / sph_h2(l, x), x)
    p, dp = riccati_q(l, sph_j(l - 1, z) / sph_j(l, z), z)
    Dv = (q - (l + 2)) - P.m * (p - (l + 2))
    dD = P.Rs * (dq / P.c2 - P.m * dp / P.c1)
    return Dv, dD


def F_and_dF(l: int, w, params: ModelParams):
    """Return (F, dF/domega) with ``F = h2_l(x) * D_l(omega)``.

    ``F`` has the same zeros as ``D`` but no poles at the zeros of ``h2_l`` (which are the
    seeds of the type-II modes), so Newton iteration is safe near them.  Poles of ``D`` at
    zeros of ``j_l`` remain; those are far from the type-II roots (large |Im omega|).
    """
    P = params.mp()
    w, z, x = _args(w, P)
    h = sph_h2(l, x)
    hm = sph_h2(l - 1, x)
    p, dp = riccati_q(l, sph_j(l - 1, z) / sph_j(l, z), z)
    Ph = x * hm - (l + 2) * h
    dPh = -2 * hm + ((l + 1) * (l + 2) / x - x) * h  # d/dx of x h_{l-1} - (l+2) h_l
    dh = hm - (l + 1) / x * h  # h2'(x)
    F = Ph - P.m * (p - (l + 2)) * h
    dF = (P.Rs / P.c2) * dPh - P.m * ((P.Rs / P.c1) * dp * h + (p - (l + 2)) * (P.Rs / P.c2) * dh)
    return F, dF


def A(l: int, w, params: ModelParams):
    """A_l(omega), coefficient of the outgoing wave."""
    P = params.mp()
    w, z, x = _args(w, P)
    jl = sph_j(l, z)
    h2 = sph_h2(l, x)
    Pj = z * sph_j(l - 1, z) - (l + 2) * jl
    Ph2 = x * sph_h2(l - 1, x) - (l + 2) * h2
    return 1j * x * (jl * Ph2 - P.m * Pj * h2)


def B(l: int, w, params: ModelParams):
    """B_l(omega), coefficient of the incoming wave."""
    P = params.mp()
    w, z, x = _args(w, P)
    jl = sph_j(l, z)
    h1 = sph_h1(l, x)
    Pj = z * sph_j(l - 1, z) - (l + 2) * jl
    Ph1 = x * sph_h1(l - 1, x) - (l + 2) * h1
    return 1j * x * (P.m * Pj * h1 - jl * Ph1)


def B_at_root(l: int, w, params: ModelParams):
    """B_l at a zero of A_l:  B = 2 j_l(z) / h2_l(x)."""
    P = params.mp()
    w, z, x = _args(w, P)
    return 2 * sph_j(l, z) / sph_h2(l, x)


def dA(l: int, w, params: ModelParams):
    """Analytic dA_l/domega at arbitrary omega."""
    P = params.mp()
    w, z, x = _args(w, P)
    jl, h2 = sph_j(l, z), sph_h2(l, x)
    Dv, dD = D_and_dD(l, w, params)
    dj = dsph_j(l, z) * P.Rs / P.c1
    dh2 = dsph_h(l, x, 2) * P.Rs / P.c2
    dx = P.Rs / P.c2
    # A = i x j h2 D
    return 1j * (dx * jl * h2 * Dv + x * dj * h2 * Dv + x * jl * dh2 * Dv + x * jl * h2 * dD)


def dA_at_root(l: int, w, params: ModelParams):
    """dA_l/domega at a zero of A_l (only the D' term survives)."""
    P = params.mp()
    w, z, x = _args(w, P)
    _, dD = D_and_dD(l, w, params)
    return 1j * x * sph_j(l, z) * sph_h2(l, x) * dD
