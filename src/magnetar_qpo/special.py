"""Spherical Bessel/Hankel functions at arbitrary precision (mpmath backend).

All functions use the *current* ``mpmath.mp.dps``.  Integer orders only.

Conventions (as in Mathematica):
``h1 = SphericalHankelH1 = j + i y`` and ``h2 = SphericalHankelH2 = j - i y``.
"""

from __future__ import annotations

import mpmath as mp


def _hankel_poly(l: int, z, sign: int):
    """Sum_{k=0}^{l} (l+k)!/(k!(l-k)!) (sign*i/(2z))^k, the Bessel-polynomial factor."""
    z = mp.mpmathify(z)
    q = sign * 1j / (2 * z)
    term = mp.mpf(1)
    total = mp.mpf(1)
    for k in range(1, l + 1):
        # ratio of consecutive coefficients: (l+k)(l-k+1)/k
        term = term * (l + k) * (l - k + 1) / k * q
        total += term
    return total


def sph_h1(l: int, z):
    """Spherical Hankel function of the first kind, h_l^(1)(z), exact finite form."""
    z = mp.mpmathify(z)
    return (-1j) ** (l + 1) * mp.exp(1j * z) / z * _hankel_poly(l, z, +1)


def sph_h2(l: int, z):
    """Spherical Hankel function of the second kind, h_l^(2)(z), exact finite form."""
    z = mp.mpmathify(z)
    return (1j) ** (l + 1) * mp.exp(-1j * z) / z * _hankel_poly(l, z, -1)


def sph_j(l: int, z):
    """Spherical Bessel function j_l(z) (entire, parity (-1)^l)."""
    z = mp.mpmathify(z)
    if z == 0:
        return mp.mpf(1) if l == 0 else mp.mpf(0)
    if mp.re(z) < 0:
        return (-1) ** l * sph_j(l, -z)
    return mp.sqrt(mp.pi / (2 * z)) * mp.besselj(l + mp.mpf(1) / 2, z)


def sph_y(l: int, z):
    """Spherical Neumann function y_l(z) = (h1 - h2)/(2i)."""
    return (sph_h1(l, z) - sph_h2(l, z)) / 2j


def dsph_j(l: int, z):
    """d j_l / dz."""
    z = mp.mpmathify(z)
    return sph_j(l - 1, z) - (l + 1) / z * sph_j(l, z) if l >= 1 else -sph_j(1, z)


def dsph_h(l: int, z, kind: int):
    """d h_l^(kind) / dz."""
    z = mp.mpmathify(z)
    h = sph_h1 if kind == 1 else sph_h2
    return h(l - 1, z) - (l + 1) / z * h(l, z) if l >= 1 else -h(1, z)


def ratio_prev_over(fn, l: int, z):
    """f_{l-1}(z) / f_l(z) for f = fn."""
    return fn(l - 1, z) / fn(l, z)


def riccati_q(l: int, v, x):
    """q(x) = x f_{l-1}/f_l  given v = f_{l-1}/f_l, and its derivative dq/dx.

    For any spherical Bessel-type f: q' = (2l+1) v - x - x v^2.
    Returns (q, dq/dx).
    """
    q = x * v
    dq = (2 * l + 1) * v - x - x * v * v
    return q, dq
