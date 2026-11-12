r"""Reference solution by direct quadrature over real frequency (paper eq. Fl).

    F_l(r, t) = 2/(pi c2 mu2 l(l+1)) int_0^inf omega^2 abar(omega)/|A|^2 Re[A h1(omega r/c2)]
                cos(omega t) d omega,

    abar(omega) = rho1 int_0^R f_r j_l(omega r/c1) r^2 dr + rho2 int_R^inf f_r Re[A h1] r^2 dr.

This is expensive when the spectral peaks are narrow (Im omega << spacing), which is exactly why
the residue sum is used for the production runs, but for moderate speed ratios (c2 ~ 2-10) it is
an independent check of the residue machinery.
"""

from __future__ import annotations

import numpy as np

from .initial_conditions import InitialCondition, _gl
from .params import ModelParams
from .special_np import A_real_axis, exterior_standing_wave, sph_j


def abar_real(ic: InitialCondition, l: int, w: np.ndarray, params: ModelParams,
              r_max: float | None = None, n_nodes: int = 800) -> np.ndarray:
    Rs, c1 = float(params.Rs), float(params.c1)
    rho1, rho2 = float(params.mu1 / params.c1**2), float(params.mu2 / params.c2**2)
    lo = max(1e-6, ic.r0 - 12 * ic.sigma_r)
    hi = ic.r0 + 12 * ic.sigma_r if r_max is None else r_max
    out = np.zeros(len(w))
    if lo < Rs:
        r, wt = _gl(lo, min(Rs, hi), n_nodes)
        f = ic.fr(r) * r**2 * wt
        out += rho1 * (sph_j(l, w[:, None] * r[None, :] / c1).real * f[None, :]).sum(axis=1)
    if hi > Rs:
        r, wt = _gl(max(lo, Rs), hi, n_nodes)
        f = ic.fr(r) * r**2 * wt
        out += rho2 * (exterior_standing_wave(l, w, r, params) * f[None, :]).sum(axis=1)
    return out


def F_direct(ic: InitialCondition, l: int, r: float, t, params: ModelParams,
             w_max: float = 120.0, dw: float = 0.01) -> np.ndarray:
    """Direct frequency-integral solution at radius ``r >= R*`` for an array of times."""
    c2, mu2 = float(params.c2), float(params.mu2)
    w = np.arange(dw / 2, w_max, dw)  # midpoint rule
    A = A_real_axis(l, w, params)
    ab = abar_real(ic, l, w, params)
    weight = w**2 * ab / np.abs(A) ** 2 * exterior_standing_wave(l, w, r, params)
    t = np.atleast_1d(np.asarray(t, dtype=float))
    pref = 2.0 / (np.pi * c2 * mu2 * l * (l + 1))
    return pref * dw * (np.cos(np.outer(t, w)) @ weight)
