"""Vectorised double-precision spherical Bessel/Hankel helpers (numpy/scipy backend).

Used for time-series synthesis and quadrature.  Quantities that would overflow at complex
frequencies are expressed as *ratios* (e.g. ``j_l(w r)/j_l(w R)``) that stay bounded.
"""

from __future__ import annotations

from math import factorial

import numpy as np
from scipy.special import jve, spherical_jn, spherical_yn


def _coeffs(l: int) -> np.ndarray:
    return np.array([factorial(l + k) / (factorial(k) * factorial(l - k)) for k in range(l + 1)])


def _poly(l: int, x, sign: int):
    """Sum_k (l+k)!/(k!(l-k)!) (sign*i/(2x))^k evaluated by Horner in q = sign*i/(2x)."""
    x = np.asarray(x, dtype=complex)
    q = sign * 1j / (2 * x)
    c = _coeffs(l)
    acc = np.zeros_like(x) + c[l]
    for k in range(l - 1, -1, -1):
        acc = acc * q + c[k]
    return acc


def sph_h1(l: int, x):
    x = np.asarray(x, dtype=complex)
    return (-1j) ** (l + 1) * np.exp(1j * x) / x * _poly(l, x, +1)


def sph_h2(l: int, x):
    x = np.asarray(x, dtype=complex)
    return (1j) ** (l + 1) * np.exp(-1j * x) / x * _poly(l, x, -1)


def sph_j(l: int, z):
    return spherical_jn(l, z)


def h2_ratio(l: int, w, r, Rs: float, c2: float):
    """h2_l(w r/c2) / h2_l(w Rs/c2) for complex ``w`` (array) and radius ``r``.

    Uses the exact finite form, so ``= (Rs/r) exp(-i w (r-Rs)/c2) P(w r/c2) / P(w Rs/c2)`` with
    the Bessel-polynomial factor ``P``.  Broadcasts ``w`` against ``r``.
    """
    w = np.asarray(w, dtype=complex)
    r = np.asarray(r, dtype=float)
    xs = w * Rs / c2
    x = w * r / c2
    return (Rs / r) * np.exp(-1j * (x - xs)) * _poly(l, x, -1) / _poly(l, xs, -1)


def h2_ratio_dr(l: int, w, r, Rs: float, c2: float):
    """d/dr of ``h2_l(w r/c2)/h2_l(w Rs/c2)`` (same overflow-safe construction)."""
    w = np.asarray(w, dtype=complex)
    r = np.asarray(r, dtype=float)
    xs = w * Rs / c2
    x = w * r / c2
    decay = np.exp(-1j * (x - xs))
    low = -1j * (Rs / r) * decay * _poly(l - 1, x, -1) / _poly(l, xs, -1)  # h2_{l-1}(x)/h2_l(x*)
    return (w / c2) * (low - (l + 1) / x * h2_ratio(l, w, r, Rs, c2))


def j_ratio(l: int, w, r, Rs: float, c1: float):
    """j_l(w r/c1) / j_l(w Rs/c1) for complex ``w`` and ``r <= Rs`` (overflow-safe).

    Parity makes the ratio even in ``w``, so frequencies with negative real part are mapped to
    ``-w``.  The exponential growth ``exp(|Im z|)`` is factored out via ``jve``.
    """
    w = np.asarray(w, dtype=complex)
    w = np.where(w.real < 0, -w, w)
    r = np.asarray(r, dtype=float)
    zs = w * Rs / c1
    z = w * r / c1
    nu = l + 0.5
    scaled = np.sqrt(zs / z) * jve(nu, z) / jve(nu, zs)
    return scaled * np.exp(np.abs(zs.imag) * (r / Rs - 1.0) * Rs / Rs)


def alpha_beta_real(l: int, w, params):
    """Real coefficients with ``w_>(r) = alpha j_l(w r/c2) + beta y_l(w r/c2)`` at real ``w``.

    Obtained by solving the two boundary conditions in the real basis ``(j_l, y_l)``: value
    ``j_l(z*)`` and traction ``(mu1/mu2) P_j(z*)`` at the surface.  Here ``alpha = Re A`` and
    ``beta = -Im A``, so ``|A|^2 = alpha^2 + beta^2``; unlike ``Re[A h1]`` this has no
    catastrophic cancellation at small ``w``.
    """
    c1, c2, mu1, mu2, Rs = (float(v) for v in (params.c1, params.c2, params.mu1, params.mu2,
                                               params.Rs))
    w = np.asarray(w, dtype=float)
    z, x = Rs * w / c1, Rs * w / c2
    jz, jzm = spherical_jn(l, z), spherical_jn(l - 1, z)
    Pz = z * jzm - (l + 2) * jz
    jx, jxm = spherical_jn(l, x), spherical_jn(l - 1, x)
    yx, yxm = spherical_yn(l, x), spherical_yn(l - 1, x)
    Pjx = x * jxm - (l + 2) * jx
    Pyx = x * yxm - (l + 2) * yx
    m = mu1 / mu2
    det = jx * Pyx - yx * Pjx
    alpha = (jz * Pyx - yx * m * Pz) / det
    beta = (jx * m * Pz - Pjx * jz) / det
    return alpha, beta


def abs_A2_real(l: int, w, params) -> np.ndarray:
    """``|A_l(omega)|^2`` at real ``omega`` (double precision, ``alpha^2 + beta^2``)."""
    alpha, beta = alpha_beta_real(l, w, params)
    return alpha**2 + beta**2


def exterior_standing_wave(l: int, w, r, params, derivative: bool = False) -> np.ndarray:
    """Exterior radial function ``w_>(r) = Re[A h1(w r/c2)]`` at real ``w`` (broadcasting w, r).

    With ``derivative=True`` returns ``d w_>/dr`` instead.
    """
    c2 = float(params.c2)
    w = np.asarray(w, dtype=float)
    alpha, beta = alpha_beta_real(l, w, params)
    r = np.asarray(r, dtype=float)
    xr = np.asarray(w)[..., None] * r / c2 if np.ndim(r) else w * r / c2
    a = alpha[..., None] if np.ndim(r) else alpha
    b = beta[..., None] if np.ndim(r) else beta
    if not derivative:
        return a * spherical_jn(l, xr) + b * spherical_yn(l, xr)
    k = (np.asarray(w)[..., None] if np.ndim(r) else w) / c2
    return k * (a * spherical_jn(l, xr, derivative=True) + b * spherical_yn(l, xr, derivative=True))


def A_real_axis(l: int, w, params) -> np.ndarray:
    """A_l(omega) for real omega in double precision (used by the direct-integration check)."""
    c1, c2, mu1, mu2, Rs = (float(v) for v in (params.c1, params.c2, params.mu1, params.mu2,
                                               params.Rs))
    w = np.asarray(w, dtype=float)
    z, x = Rs * w / c1, Rs * w / c2
    jl, jlm = spherical_jn(l, z), spherical_jn(l - 1, z)
    hl = spherical_jn(l, x) - 1j * spherical_yn(l, x)
    hlm = spherical_jn(l - 1, x) - 1j * spherical_yn(l - 1, x)
    Pj = z * jlm - (l + 2) * jl
    Ph2 = x * hlm - (l + 2) * hl
    return 1j * x * (jl * Ph2 - (mu1 / mu2) * Pj * hl)
