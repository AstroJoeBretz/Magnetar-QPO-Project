"""Initial guesses for eigenfrequency searches (double precision, vectorised)."""

from __future__ import annotations

from functools import cache

import mpmath as mp
import numpy as np
from scipy.special import spherical_jn, spherical_yn

from ..params import ModelParams


def real_axis_function(l: int, w: np.ndarray, params: ModelParams) -> np.ndarray:
    """Real entire function g(omega) whose sign changes bracket the type-I modes.

    ``g = Re[-j_l(z) D_l(omega)] = (mu1/mu2) P_j(z) - j_l(z) Re[q(x) - (l+2)]`` with
    ``q = x h2_{l-1}(x)/h2_l(x)``.  Multiplying D by j_l removes the poles of D at the zeros
    of j_l, so the zeros of ``g`` sit next to the (nearly real) type-I roots.
    """
    c1, c2, mu1, mu2, Rs = (float(v) for v in (params.c1, params.c2, params.mu1, params.mu2,
                                               params.Rs))
    w = np.asarray(w, dtype=float)
    z = Rs * w / c1
    x = Rs * w / c2
    jl = spherical_jn(l, z)
    jlm = spherical_jn(l - 1, z)
    Pj = z * jlm - (l + 2) * jl
    # h2 = j - i y ; Re[q] = x Re[h2_{l-1}/h2_l]
    hl = spherical_jn(l, x) - 1j * spherical_yn(l, x)
    hlm = spherical_jn(l - 1, x) - 1j * spherical_yn(l - 1, x)
    with np.errstate(all="ignore"):
        q = x * hlm / hl
    return (mu1 / mu2) * Pj - jl * (np.real(q) - (l + 2))


def bracket_real_roots(
    l: int, params: ModelParams, w_lo: float, w_hi: float, step: float = 0.02
) -> list[tuple[float, float]]:
    """Sign-change brackets of :func:`real_axis_function` on ``[w_lo, w_hi]``."""
    n = int(np.ceil((w_hi - w_lo) / step)) + 1
    w = np.linspace(w_lo, w_hi, n)
    g = real_axis_function(l, w, params)
    s = np.sign(g)
    idx = np.nonzero(s[:-1] * s[1:] < 0)[0]
    return [(float(w[i]), float(w[i + 1])) for i in idx]


@cache
def reverse_bessel_zeros(l: int) -> np.ndarray:
    """Zeros x_k of h2_l(x) with Re x >= 0 (upper half plane), sorted by Re x.

    h2_l(x) is proportional to ``Sum_k (l+k)!/(k!(l-k)!) (-i/(2x))^k``; with ``y = 1/x`` this is
    a degree-l polynomial in y.  These are the exact type-II eigenvalues (in units of c2/Rs)
    in the limit mu1/mu2 -> infinity.  There are ceil(l/2) of them with Re x >= 0.
    """
    with mp.workdps(max(60, 4 * l)):
        coeff = [
            mp.factorial(l + k) / (mp.factorial(k) * mp.factorial(l - k)) * (-0.5j) ** k
            for k in range(l + 1)
        ]
        y = mp.polyroots(coeff[::-1], maxsteps=500, extraprec=20 * l + 100)
        x = [1 / yy for yy in y]
        tol = mp.mpf(10) ** -20
        keep = []
        for xx in x:
            if abs(xx.real) < tol * abs(xx):  # exactly on the imaginary axis
                keep.append(complex(0.0, float(xx.imag)))
            elif xx.real > 0:
                keep.append(complex(xx))
    out = np.array(sorted(keep, key=lambda z: z.real))
    return out
