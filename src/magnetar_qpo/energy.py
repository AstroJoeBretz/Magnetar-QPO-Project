r"""Strain energy of an initial displacement field (paper App. B, eq. StrainEnergy).

For tangential displacements the strain tensor components are (``u`` in units of R*)

    e_tt = (1/r) d_theta u_theta
    e_pp = (1/(r sin t)) d_phi u_phi + u_theta cot(t) / r
    e_rt = (r/2) d_r (u_theta/r),   e_rp = (r/2) d_r (u_phi/r)
    e_tp = sin t/(2r) d_theta (u_phi / sin t) + (1/(2 r sin t)) d_phi u_theta

and ``E = int mu(r) (e_tt^2 + e_pp^2 + 2 e_rt^2 + 2 e_rp^2 + 2 e_tp^2) dV`` with ``mu = mu1``
inside the star and ``mu2`` outside.  For the separable fields of
:mod:`magnetar_qpo.initial_conditions` every component is a sum of products
``R(r) T(theta) P(phi)``, so each squared integral reduces to 1-D quadratures (this is the
decomposition used in the Mathematica notebooks, but with exterior weighting by ``mu2``).

The returned energy is dimensionless: ``E_phys = mu1 R*^3 a^2 * strain_energy(...)`` for a field
of amplitude ``a`` (in units of R*) -- see :func:`energy_scale`.
"""

from __future__ import annotations

import numpy as np

from .initial_conditions import InitialCondition, _gl


def _terms(ic: InitialCondition, r, th, ph):
    fr = ic.fr(r)
    dfr = ic.dfr(r)
    rad = {"fr/r": fr / r, "dfr": dfr - fr / r}
    T = ic.theta_functions(th)
    P = ic.phi_functions(ph)
    s = np.sin(th)
    cot = np.cos(th) / s
    A, C, dC = T["A"], T["C"], T["dC"]
    E, dE, B = P["E"], P["dE"], P["B"]
    # each component: list of (R, T, P) terms
    comps = {
        "tt": ([(-rad["fr/r"], dC, E)], 1.0),
        "pp": ([(rad["fr/r"], A / s - cot * C, E)], 1.0),
        "rt": ([(-0.5 * rad["dfr"], C, E)], 2.0),
        "rp": ([(0.5 * rad["dfr"], A, B)], 2.0),
        "tp": ([(0.5 * rad["fr/r"], T["sin_dAoverSin"], B),
                (-0.5 * rad["fr/r"], C / s, dE)], 2.0),
    }
    return comps


def strain_energy(
    ic: InitialCondition,
    mu_ratio: float,
    Rs: float = 1.0,
    r_max: float | None = None,
    n_r: int = 400,
    n_theta: int = 600,
    n_phi: int | None = None,
    amplitude: float = 1.0,
    return_components: bool = False,
):
    """Dimensionless strain energy ``int (mu/mu1) e_ij e_ij dV`` (in units mu1 R*^3 a^2/R*^2*R*^2).

    ``amplitude`` multiplies the field (energy scales as amplitude^2).  The field is used as
    defined (not peak-normalised); divide by ``ic.peak()**2`` for unit peak amplitude.
    """
    r_max = r_max if r_max is not None else max(Rs, ic.r0 + 12 * ic.sigma_r) + 1.0
    # radial grid: split at the surface, where mu jumps
    rin, win = _gl(0.0, Rs, n_r)
    rout, wout = _gl(Rs, r_max, n_r)
    r = np.concatenate([rin, rout])
    w_r = np.concatenate([win, wout * mu_ratio]) * r**2
    th, w_th = _gl(0.0, np.pi, n_theta)
    w_th = w_th * np.sin(th)
    n_phi = n_phi if n_phi is not None else max(400, int(60 * 2 * np.pi * (1 / ic.sigma_phi)))
    ph, w_ph = _gl(0.0, 2 * np.pi, n_phi)

    comps = _terms(ic, r, th, ph)
    total = 0.0
    parts = {}
    for name, (terms, mult) in comps.items():
        val = 0.0
        for R1, T1, P1 in terms:
            for R2, T2, P2 in terms:
                val += (R1 * R2 * w_r).sum() * (T1 * T2 * w_th).sum() * (P1 * P2 * w_ph).sum()
        parts[name] = mult * val * amplitude**2
        total += parts[name]
    return (total, parts) if return_components else total


def strain_energy_brute_force(ic: InitialCondition, mu_ratio: float, Rs: float = 1.0,
                              r_max: float | None = None, n: int = 120, h: float = 1e-5):
    """Independent check: 3-D quadrature of eq. StrainEnergy using finite-difference derivatives."""
    r_max = r_max if r_max is not None else max(Rs, ic.r0 + 12 * ic.sigma_r) + 1.0
    rin, win = _gl(0.0, Rs, n)
    rout, wout = _gl(Rs, r_max, n)
    r = np.concatenate([rin, rout])
    wr = np.concatenate([win, wout * mu_ratio])
    th, wth = _gl(0.0, np.pi, n)
    ph, wph = _gl(0.0, 2 * np.pi, 2 * n)
    R, T, P = np.meshgrid(r, th, ph, indexing="ij")
    uth = ic.u_theta
    uph = ic.u_phi
    d = 2 * h
    e_tt = (uth(R, T + h, P) - uth(R, T - h, P)) / d / R
    e_pp = (uph(R, T, P + h) - uph(R, T, P - h)) / d / (R * np.sin(T))
    e_pp = e_pp + uth(R, T, P) / np.tan(T) / R
    e_rt = R / 2 * ((uth(R + h, T, P) / (R + h)) - (uth(R - h, T, P) / (R - h))) / d
    e_rp = R / 2 * ((uph(R + h, T, P) / (R + h)) - (uph(R - h, T, P) / (R - h))) / d
    e_tp = (
        np.sin(T) / (2 * R)
        * (uph(R, T + h, P) / np.sin(T + h) - uph(R, T - h, P) / np.sin(T - h)) / d
        + (uth(R, T, P + h) - uth(R, T, P - h)) / d / (2 * R * np.sin(T))
    )
    dens = e_tt**2 + e_pp**2 + 2 * e_rt**2 + 2 * e_rp**2 + 2 * e_tp**2
    w = (wr * r**2)[:, None, None] * (wth * np.sin(th))[None, :, None] * wph[None, None, :]
    return float((dens * w).sum())


def energy_scale(mu1_cgs: float, Rs_cm: float) -> float:
    """Energy unit mu1 R*^3 in erg (the factor multiplying ``a^2 * strain_energy``)."""
    return mu1_cgs * Rs_cm**3


def amplitude_for_energy(E_erg: float, strain: float, E_unit_erg: float) -> float:
    """Amplitude a (in units of R*) such that ``E_unit * a^2 * strain = E_erg``."""
    return float(np.sqrt(E_erg / (E_unit_erg * strain)))
