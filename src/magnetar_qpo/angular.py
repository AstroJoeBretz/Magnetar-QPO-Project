r"""Angular projection of the initial displacement onto toroidal vector harmonics.

The tangential field ``V(theta, phi) = (V_theta, V_phi) = (-C E, A B)`` (radial profile
stripped) is expanded in toroidal harmonics ``Phi_lm = (-d_phi Y/sin t, d_theta Y)``, for which
``int Phi_lm . Phi*_l'm' dOmega = l(l+1) delta delta``.  The coefficient integral is the angular
factor of paper eq. (almFull),

    a_lm = int [V_phi d_theta Y*_lm - V_theta d_phi Y*_lm / sin t] dOmega
         = T1[l,m] Phi1[m] + T2[l,m] Phi2[m],

    T1 = int sin t A(t) y'_lm(t) dt,        Phi1[m] = int B(phi) e^{-i m phi} dphi
    T2 = int C(t) y_lm(t) dt,               Phi2[m] = -i m int E(phi) e^{-i m phi} dphi

(``Y_lm = y_lm(t) e^{i m phi}``).  These are the notebooks' ``TH1, TH2, PH1, PH2``.
The l-th angular pattern of the field is ``G_l = sum_m a_lm d_theta Y_lm`` (for u_phi) and
``H_l = -sum_m a_lm d_phi Y_lm / sin t`` (for u_theta); the field is
``V = sum_l (1/(l(l+1))) (H_l, G_l)``.
"""

from __future__ import annotations

import numpy as np
from scipy.special import sph_harm_y

from .initial_conditions import InitialCondition, _gl


def projection_coefficients(
    ic: InitialCondition, lmax: int, lmin: int = 1, n_theta: int = 900, n_phi: int | None = None
) -> dict[tuple[int, int], complex]:
    """``a_lm`` for ``lmin <= l <= lmax`` and ``-l <= m <= l`` (angular factor only)."""
    th, wth = _gl(0.0, np.pi, n_theta)
    n_phi = n_phi if n_phi is not None else max(600, int(80 * np.pi / ic.sigma_phi))
    ph, wph = _gl(0.0, 2 * np.pi, n_phi)
    Tf = ic.theta_functions(th)
    Pf = ic.phi_functions(ph)
    s = np.sin(th)
    out: dict[tuple[int, int], complex] = {}
    for l in range(lmin, lmax + 1):
        for m in range(-l, l + 1):
            Y, dY = sph_harm_y(l, m, th, 0.0, diff_n=1)
            # y_lm(t) = Y(t, phi=0) is real; d_theta y = dY[..., 0]
            y = Y.real
            dy = dY[..., 0].real
            T1 = np.sum(wth * s * Tf["A"] * dy)
            Phi1 = np.sum(wph * Pf["B"] * np.exp(-1j * m * ph))
            a = T1 * Phi1
            if not ic.axisymmetric:
                T2 = np.sum(wth * Tf["C"] * y)
                Phi2 = -1j * m * np.sum(wph * Pf["E"] * np.exp(-1j * m * ph))
                a = a + T2 * Phi2
            out[(l, m)] = complex(a)
    return out


def G_H(coeffs: dict[tuple[int, int], complex], l: int, theta, phi):
    """Angular patterns ``(G_l, H_l)`` evaluated at (theta, phi) (broadcastable arrays)."""
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    G = np.zeros(np.broadcast(theta, phi).shape, dtype=complex)
    H = np.zeros_like(G)
    s = np.sin(theta)
    for m in range(-l, l + 1):
        a = coeffs[(l, m)]
        if a == 0:
            continue
        Y, dY = sph_harm_y(l, m, theta, phi, diff_n=1)
        G = G + a * dY[..., 0]
        H = H - a * (1j * m * Y) / s
    return G.real, H.real


def reconstruct(coeffs, lmax: int, theta, phi, lmin: int = 1):
    """Sum of ``(1/(l(l+1))) (H_l, G_l)`` over l: returns (V_theta, V_phi)."""
    Vt = 0.0
    Vp = 0.0
    for l in range(lmin, lmax + 1):
        G, H = G_H(coeffs, l, theta, phi)
        Vp = Vp + G / (l * (l + 1))
        Vt = Vt + H / (l * (l + 1))
    return Vt, Vp


def power_per_l(coeffs: dict[tuple[int, int], complex], lmax: int) -> np.ndarray:
    """``sum_m |a_lm|^2 / (l(l+1))`` -- the angular energy/momentum content of each l."""
    return np.array(
        [sum(abs(coeffs[(l, m)]) ** 2 for m in range(-l, l + 1)) / (l * (l + 1))
         for l in range(1, lmax + 1)]
    )
