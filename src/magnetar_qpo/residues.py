r"""Residue coefficients of the initial value solution (paper App. D, eq. FullContourSolution).

For late times (after the initial pulse has crossed the surface) the radial function of the
exterior solution is a sum over the poles ``omega_k`` (zeros of ``A_l`` in the upper half plane:
type-I and type-II modes and their mirror images ``-conj(omega)``):

    F_l(r, t) = i/(c2 mu2 l(l+1)) sum_k omega_k^2 abar_l(omega_k) / A_l'(omega_k)
                * h2_l(omega_k r/c2) e^{i omega_k t}.

At a pole ``A = 0`` so ``B = 2 j_l(z*)/h2_l(x*)`` and ``A' = i x* j_l(z*) h2_l(x*) D'``, and the
overlap of the initial radial profile with the eigenfunction is

    abar = rho1 int_0^R f_r j_l(omega r/c1) r^2 dr
         + rho2 j_l(z*)/h2_l(x*) int_R^inf f_r h2_l(omega r/c2) r^2 dr.

Factoring out ``j_l(z*)`` removes all exponentially large quantities:

    F_l(r, t) = sum_k K_k  [h2_l(omega_k r/c2) / h2_l(omega_k R/c2)]  e^{i omega_k t},
    K_k = omega_k^2 Q_k / (c2 mu2 l(l+1) x*_k D'(omega_k)),
    Q_k = rho1 int_0^R f_r [j_l(omega r/c1)/j_l(z*)] r^2 dr
        + rho2 int_R^inf f_r [h2_l(omega r/c2)/h2_l(x*)] r^2 dr.
"""

from __future__ import annotations

from dataclasses import dataclass

import mpmath as mp
import numpy as np

from .dispersion import D_and_dD
from .eigen.table import EigenTable
from .initial_conditions import InitialCondition, _gl
from .params import ModelParams
from .special_np import h2_ratio, h2_ratio_dr, j_ratio


@dataclass
class RadialModes:
    """Poles and residue weights for one angular order ``l``."""

    l: int
    omega: np.ndarray  # complex128, all poles used
    K: np.ndarray  # complex128, residue weights (see module docstring)
    params: ModelParams
    kind: np.ndarray  # "I" / "II" label per pole
    re_lo: np.ndarray | None = None  # Re(omega) - float(Re(omega)), for double-double accuracy

    def profile(self, r: float) -> np.ndarray:
        """Eigenfunction of each pole at ``r`` relative to its surface value ``j_l(z*)``.

        ``h2_l(w r/c2)/h2_l(w R/c2)`` outside the star and ``j_l(w r/c1)/j_l(w R/c1)`` inside
        (the interior profile is entire, so there is no divergence at the origin).
        """
        Rs = float(self.params.Rs)
        if r >= Rs:
            return h2_ratio(self.l, self.omega, r, Rs, float(self.params.c2))
        return j_ratio(self.l, self.omega, r, Rs, float(self.params.c1))

    def F(self, r: float, t) -> np.ndarray:
        """Late-time radial function ``F_l(r, t)`` for an array of times."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        coeff = self.K * self.profile(r)
        return np.exp(1j * np.outer(t, self.omega)) @ coeff

    def F_with_derivatives(self, r: float, t) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(F, dF/dr, dF/dt)`` (real parts) at radius ``r`` for an array of times."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        c2, Rs = float(self.params.c2), float(self.params.Rs)
        a = self.K * h2_ratio(self.l, self.omega, r, Rs, c2)
        da = self.K * h2_ratio_dr(self.l, self.omega, r, Rs, c2)
        E = np.exp(1j * np.outer(t, self.omega))
        return (E @ a).real, (E @ da).real, (E @ (1j * self.omega * a)).real

    def coefficients_at(self, r: float) -> np.ndarray:
        """Per-pole amplitudes at radius ``r``: ``K_k`` times the relative eigenfunction."""
        return self.K * self.profile(r)


def radial_overlap_Q(
    ic: InitialCondition,
    l: int,
    omega: np.ndarray,
    params: ModelParams,
    r_max: float | None = None,
    n_nodes: int | None = None,
) -> np.ndarray:
    """``Q_k`` of the module docstring for an array of complex poles (double precision)."""
    Rs, c1, c2 = float(params.Rs), float(params.c1), float(params.c2)
    rho1, rho2 = float(params.mu1 / params.c1**2), float(params.mu2 / params.c2**2)
    lo = max(1e-6, ic.r0 - 12 * ic.sigma_r)
    hi = ic.r0 + 12 * ic.sigma_r if r_max is None else r_max
    # Resolution is set by the type-I (nearly real) frequencies.  Type-II poles (|Im| >~ 1) only
    # overlap the profile in a boundary layer at the surface, which Gauss-Legendre endpoint
    # clustering resolves.
    slow = omega[np.abs(omega.imag) < 1.0]
    wmax = float(np.max(np.abs(slow.real))) if len(slow) else float(np.max(np.abs(omega.real)))
    q = np.zeros(len(omega), dtype=complex)
    if lo < Rs:
        top = min(Rs, hi)
        n = n_nodes or int(np.clip(150 + 3 * wmax / min(c1, c2) * (top - lo), 200, 3000))
        r, w = _gl(lo, top, n)
        f = ic.fr(r) * r**2 * w
        q += rho1 * (j_ratio(l, omega[:, None], r[None, :], Rs, c1) * f[None, :]).sum(axis=1)
    if hi > Rs:
        bot = max(lo, Rs)
        n = n_nodes or int(np.clip(150 + 3 * wmax / c2 * (hi - bot), 200, 3000))
        r, w = _gl(bot, hi, n)
        f = ic.fr(r) * r**2 * w
        q += rho2 * (h2_ratio(l, omega[:, None], r[None, :], Rs, c2) * f[None, :]).sum(axis=1)
    return q


def build_modes(
    ic: InitialCondition,
    l: int,
    table: EigenTable,
    nmax: int | None = None,
    include_type2: bool = True,
    include_mirror: bool = True,
    n_nodes: int | None = None,
) -> RadialModes:
    """Assemble all poles and residue weights for angular order ``l``."""
    params = table.params
    P = params.mp()
    poles: list[mp.mpc] = list(table.type1(l, nmax))
    kinds = ["I"] * len(poles)
    if include_type2:
        t2 = [v for (ll, _), v in sorted(table.type2.items()) if ll == l]
        poles += t2
        kinds += ["II"] * len(t2)
    if include_mirror:
        # -conj(omega) for each pole off the imaginary axis (purely imaginary poles are their own)
        mirrors = [(-mp.conj(w), k) for w, k in zip(poles, kinds, strict=True) if w.real != 0]
        poles += [m for m, _ in mirrors]
        kinds += [k for _, k in mirrors]
    omega = np.array([complex(w) for w in poles])
    re_lo = np.array([float(w.real - float(w.real)) for w in poles])
    # D'(omega_k) in multiprecision, then cast
    with mp.workdps(40):
        dD = np.array([complex(D_and_dD(l, w, params)[1]) for w in poles])
    Q = radial_overlap_Q(ic, l, omega, params, n_nodes=n_nodes)
    xs = omega * float(params.Rs / params.c2)
    c2mu2 = float(P.c2 * P.mu2)
    K = omega**2 * Q / (c2mu2 * l * (l + 1) * xs * dD)
    return RadialModes(l, omega, K, params, np.array(kinds), re_lo)
