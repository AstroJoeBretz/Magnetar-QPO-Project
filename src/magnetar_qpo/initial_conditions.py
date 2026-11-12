r"""Divergence-free, zero-angular-momentum initial displacement fields (paper Sec. 2.3, App. A).

All fields are purely tangential, ``u = u_theta theta_hat + u_phi phi_hat`` with

    u_phi   =  f_r(r) A(theta) B(phi)
    u_theta = -f_r(r) C(theta) E(phi),     E = dB/dphi,

which follows from ``u = -curl(Psi r)`` with ``Psi = f_r g_theta(theta) g_phi(phi)``:
``A = g_theta'``, ``C = g_theta / sin(theta)``, ``B = g_phi``.  Any smooth ``g`` gives
``div u = 0`` identically.

Two polar profiles are available (``profile=``):

``"paper"`` (default)
    ``g_theta = d/dtheta [sin^2(theta) G(theta)] = f_theta'`` and ``g_phi = G'(phi)``, i.e. the
    paper's eq. (IC): ``u_phi = f_r f_theta'' f_phi'`` and
    ``u_theta = -f_r f_theta' f_phi''/sin(theta)``, with ``f_theta = sin^2(theta) G(theta)`` and
    ``f_phi = G(phi)``.
``"notebook"``
    ``g_theta = sin^2(theta) G'(theta)``: what ``DisplacmentVsFrequency.nb`` evaluates.  It drops
    the ``sin(2 theta) G`` term of the product rule.  After normalising to the same energy the
    mode spectra agree to within 3%.

The radial profile is ``f_r = r exp(-(r - rbar0)^2 / 2 sigma_r^2)`` with
``rbar0 = (r0^2 - sigma_r^2)/r0`` so that ``f_r`` peaks exactly at ``r0``.

The axisymmetric field (paper eq. IC2) is ``u_phi = f_r d f_theta/d theta``,
``u_theta = 0`` with ``f_theta = sin^2(theta) G(theta)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import numpy as np


def _gauss(x, x0, s):
    d = x - x0
    G = np.exp(-(d**2) / (2 * s**2))
    return (
        G,
        -d / s**2 * G,
        (d**2 / s**4 - 1 / s**2) * G,
        (3 * d / s**4 - d**3 / s**6) * G,
    )


@dataclass(frozen=True)
class InitialCondition:
    r0: float
    sigma_r: float = 1 / 12
    sigma_theta: float = 1 / 3
    sigma_phi: float = 1 / 3
    theta0: float = np.pi / 2
    phi0: float = np.pi
    axisymmetric: bool = False
    profile: Literal["notebook", "paper"] = "paper"

    # ---- radial profile -------------------------------------------------------------
    @property
    def r_shift(self) -> float:
        return (self.r0**2 - self.sigma_r**2) / self.r0

    def fr(self, r):
        r = np.asarray(r, dtype=float)
        return r * np.exp(-((r - self.r_shift) ** 2) / (2 * self.sigma_r**2))

    def dfr(self, r):
        r = np.asarray(r, dtype=float)
        return self.fr(r) * (1 / r - (r - self.r_shift) / self.sigma_r**2)

    # ---- polar profile: A = g', C = g/sin, plus derivatives ------------------------
    def theta_functions(self, th):
        """Return dict with A, dA, C, dC, and sin(theta)*(A/sin)' for the polar profile."""
        th = np.asarray(th, dtype=float)
        G, G1, G2, G3 = _gauss(th, self.theta0, self.sigma_theta)
        s, c = np.sin(th), np.cos(th)
        s2, c2 = np.sin(2 * th), np.cos(2 * th)
        if self.axisymmetric or self.profile == "paper":
            # g = (sin^2 G)' = sin2 G + sin^2 G'
            g = s2 * G + s**2 * G1
            g1 = 2 * c2 * G + 2 * s2 * G1 + s**2 * G2
            g2 = -4 * s2 * G + 6 * c2 * G1 + 3 * s2 * G2 + s**2 * G3
        else:
            # g = sin^2 G'
            g = s**2 * G1
            g1 = s2 * G1 + s**2 * G2
            g2 = 2 * c2 * G1 + 2 * s2 * G2 + s**2 * G3
        if self.axisymmetric:
            # u_phi = f_r * g (g = f_theta'); no u_theta.  Here A = g, dA = g1.
            A, dA = g, g1
            zero = np.zeros_like(th)
            return {"A": A, "dA": dA, "C": zero, "dC": zero,
                    "sin_dAoverSin": s * (dA / s - A * c / s**2)}
        A, dA = g1, g2
        C = g / s
        dC = g1 / s - g * c / s**2
        return {"A": A, "dA": dA, "C": C, "dC": dC,
                "sin_dAoverSin": s * (dA / s - A * c / s**2)}

    # ---- azimuthal profile: B = g_phi, E = B', E' = B'' ---------------------------
    def phi_functions(self, ph):
        ph = np.asarray(ph, dtype=float)
        if self.axisymmetric:
            one, zero = np.ones_like(ph), np.zeros_like(ph)
            return {"B": one, "E": zero, "dE": zero}
        _, G1, G2, G3 = _gauss(ph, self.phi0, self.sigma_phi)
        return {"B": G1, "E": G2, "dE": G3}

    # ---- full field ----------------------------------------------------------------
    def u_phi(self, r, th, ph):
        return self.fr(r) * self.theta_functions(th)["A"] * self.phi_functions(ph)["B"]

    def u_theta(self, r, th, ph):
        if self.axisymmetric:
            return np.zeros(np.broadcast(r, th, ph).shape)
        return -self.fr(r) * self.theta_functions(th)["C"] * self.phi_functions(ph)["E"]

    def divergence_numeric(self, r, th, ph, h=1e-5):
        """div u by central differences (u_r = 0); should vanish identically."""
        sin = np.sin
        t_p, t_m = th + h, th - h
        term_th = (sin(t_p) * self.u_theta(r, t_p, ph) - sin(t_m) * self.u_theta(r, t_m, ph)) / (
            2 * h
        )
        term_ph = (self.u_phi(r, th, ph + h) - self.u_phi(r, th, ph - h)) / (2 * h)
        return (term_th + term_ph) / (r * sin(th))

    def peak(self, n: int = 241) -> float:
        """max |u| over the sphere at r = r0 (used to normalise the amplitude to unity)."""
        th = np.linspace(1e-3, np.pi - 1e-3, n)
        ph = np.linspace(0, 2 * np.pi, 2 * n)
        T, P = np.meshgrid(th, ph, indexing="ij")
        mag = np.hypot(self.u_theta(self.r0, T, P), self.u_phi(self.r0, T, P))
        return float(mag.max())

    def probe_point(self) -> tuple[float, float]:
        """Surface point (theta_p, phi_p) of maximal |u_phi| with u_theta = 0 (paper Sec. 4)."""
        if self.axisymmetric:
            th = np.linspace(1e-3, np.pi - 1e-3, 4001)
            a = np.abs(self.theta_functions(th)["A"])
            return float(th[np.argmax(a)]), float(self.phi0)
        return float(self.theta0), float(self.phi0 + self.sigma_phi)


def angular_momentum(ic: InitialCondition, rho1: float, rho2: float, Rs: float = 1.0,
                     r_max: float | None = None, n: int = 400) -> np.ndarray:
    """Total angular momentum vector J = integral of rho(r) r x u dV (should be zero).

    ``r x u = r (u_theta phi_hat - u_phi theta_hat)``; the integral is done on a tensor
    Gauss-Legendre grid.
    """
    r_max = r_max if r_max is not None else max(Rs, ic.r0 + 12 * ic.sigma_r) + 1.0
    xr, wr = _gl(0.0, Rs, n // 2)
    xr2, wr2 = _gl(Rs, r_max, n // 2)
    r = np.concatenate([xr, xr2])
    wr = np.concatenate([wr * rho1, wr2 * rho2])
    th, wth = _gl(0.0, np.pi, n)
    ph, wph = _gl(0.0, 2 * np.pi, n)
    R, T, P = np.meshgrid(r, th, ph, indexing="ij")
    uth, uph = ic.u_theta(R, T, P), ic.u_phi(R, T, P)
    that = np.stack([np.cos(T) * np.cos(P), np.cos(T) * np.sin(P), -np.sin(T)])
    phat = np.stack([-np.sin(P), np.cos(P), np.zeros_like(P)])
    integrand = R * (uth * phat - uph * that) * R**2 * np.sin(T)
    w = wr[:, None, None] * wth[None, :, None] * wph[None, None, :]
    return (integrand * w).sum(axis=(1, 2, 3))


@lru_cache(maxsize=64)
def _gl_std(n: int):
    from scipy.special import roots_legendre

    return roots_legendre(n)


def _gl(a: float, b: float, n: int):
    """Gauss-Legendre nodes and weights on ``[a, b]``."""
    x, w = _gl_std(int(n))
    return 0.5 * (b - a) * x + 0.5 * (b + a), 0.5 * (b - a) * w
