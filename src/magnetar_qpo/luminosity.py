r"""Alfven-wave luminosity through a sphere and accumulated escaped energy (paper App. B).

Energy conservation ``d/dt (rho u_dot^2/2 + mu e^2) = 2 mu div(u_dot . e)`` gives the outward
power through a sphere of radius ``r``:

    P(r, t) = -2 mu r^2 oint (u_dot_theta e_rtheta + u_dot_phi e_rphi) dOmega
            = -mu r sum_l l(l+1) S_l  Fdot_l (r F_l' - F_l),     S_l = sum_m |a_lm|^2,

using ``e_rj = (1/2) r d_r(u_j/r)`` and the orthogonality of toroidal harmonics, with
``u = amplitude * sum_l F_l (H_l, G_l)``.  (The Mathematica notebook squares ``a_lm`` instead of
using ``|a_lm|^2``; for the real displacement the modulus is the one orthogonality gives.)
All quantities are in code units (``mu1 = R* = c1 = 1``); the escaped energy is returned as a
fraction of the initial strain energy.
"""

from __future__ import annotations

import numpy as np

from .energy import strain_energy
from .solution import IVPSolution


def angular_weights(sol: IVPSolution) -> dict[int, float]:
    """``S_l = sum_m |a_lm|^2``."""
    return {
        l: sum(abs(sol.coeffs[(l, m)]) ** 2 for m in range(-l, l + 1)) for l in sol.ls
    }


def alfven_luminosity(sol: IVPSolution, r: float, t, ls: list[int] | None = None,
                      general_time: bool = False, **solver_kw) -> np.ndarray:
    """Outward power P(r, t) in units of ``mu1 R*^2 / t_unit * amplitude^2`` (unit-field form)."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    # the radial derivative is taken on the exterior side for r >= R*, so mu2 applies at r = R*
    mu = float(sol.table.params.mu2 if r >= float(sol.table.params.Rs) else sol.table.params.mu1)
    S = angular_weights(sol)
    P = np.zeros_like(t)
    for l in ls if ls is not None else sol.ls:
        F, dF, dt = sol.radial(l, general_time, **solver_kw).F_with_derivatives(r, t)
        P -= mu * r * l * (l + 1) * S[l] * dt * (r * dF - F)
    return P


def escaped_energy_fraction(sol: IVPSolution, r: float, t, ls: list[int] | None = None,
                            general_time: bool = False, **solver_kw) -> np.ndarray:
    """``E(t)/E0``: time integral of the luminosity through radius ``r`` over the initial energy."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    P = alfven_luminosity(sol, r, t, ls, general_time, **solver_kw)
    E0 = strain_energy(sol.ic, float(sol.table.params.mu_ratio))
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (P[1:] + P[:-1]) * np.diff(t))])
    return cum / E0
