r"""General-time solution by pole subtraction.

The radial function is a full-line frequency integral,

    F_l(r,t) = (pref/2) int_{-inf}^{inf} g(w) e^{i w t} dw,
    g(w) = w^2 abar(w) W(w, r) / |A(w)|^2,      pref = 2/(pi c2 mu2 l(l+1)),

with ``g`` real and even on the real axis (``W`` is the real standing wave: ``j_l(w r/c1)`` inside,
``alpha j_l + beta y_l`` outside).  Its continuation to complex ``w`` is meromorphic with simple
poles at the zeros ``w_k`` of ``A`` (upper half plane, residue ``rho_k``) and, by reality, at
``conj(w_k)`` (residue ``conj(rho_k)``).  The late-time residue formula is exactly the pole part
``sum_k C_k e^{i w_k t}`` with ``C_k = i pi pref rho_k``; it is only valid once every signal delay
has elapsed.

Let ``S`` be the set of *narrow* poles (``Im w_k < gamma_s``).  Since for ``t > 0``

    int rho/(w - z) e^{i w t} dw = 2 pi i rho e^{i z t}  (Im z > 0),   0  (Im z < 0),

the identity

    F = sum_{k in S} C_k e^{i w_k t}
        + (pref/2) [ int_{-a}^{a} (g - L_S) e^{i w t} dw  -  int_{|w|>a} L_S e^{i w t} dw ],
    L_S(w) = sum_{k in S} 2 Re[ rho_k/(w - w_k) ],

holds for every ``t > 0`` when ``g`` is negligible beyond ``|w| = a``.  The bracket removes the
narrow Lorentzian spikes (widths down to 1e-69) analytically, so ``g - L_S`` is smooth and a uniform
midpoint grid resolves it spectrally; it is the entire transient and decays after a few crossing
times.  The tails use ``int_a^inf e^{iwt}/(w-z) dw = e^{izt} E1(-i(a-z)t)`` and
``int_{-inf}^{-a} e^{iwt}/(w-z) dw = -e^{izt} E1(i(a+z)t)``.  With ``S`` empty this reduces to the
direct real-frequency integral of :mod:`magnetar_qpo.direct`.
"""

from __future__ import annotations

import mpmath as mp
import numpy as np
from scipy.special import exp1, spherical_jn

from .direct import abar_real
from .dispersion import A as A_mp
from .eigen.table import EigenTable
from .initial_conditions import InitialCondition
from .residues import RadialModes, build_modes
from .special_np import abs_A2_real, alpha_beta_real, exterior_standing_wave, h2_ratio_dr


class GeneralTimeSolver:
    """Valid-for-all-``t > 0`` radial solution ``F_l(r, t)`` for one angular order."""

    def __init__(
        self,
        ic: InitialCondition,
        table: EigenTable,
        l: int,
        d_omega: float = 0.02,
        omega_cut: float | None = None,
        gamma_s: float = 0.25,
        modes: RadialModes | None = None,
        near: float = 0.05,
        mp_dps: int = 50,
        t_cut: float | None = None,
    ):
        self.ic, self.l, self.params = ic, l, table.params
        self.t_cut = t_cut
        p = self.params
        self.c1, self.c2 = float(p.c1), float(p.c2)
        self.Rs = float(p.Rs)
        self.pref = 2.0 / (np.pi * self.c2 * float(p.mu2) * l * (l + 1))
        omega_cut = omega_cut if omega_cut is not None else 9.0 / ic.sigma_r
        self.dw = d_omega
        self.N = int(np.ceil(omega_cut / d_omega))
        self.a = self.N * d_omega
        self.w = (np.arange(self.N) + 0.5) * d_omega
        self.modes = modes if modes is not None else build_modes(ic, l, table)
        m = self.modes
        sel = (m.omega.imag < gamma_s) & (np.abs(m.omega.real) < self.a)
        self.sel = np.nonzero(sel)[0]
        self.om = m.omega[self.sel]
        self.re_hi = self.om.real.copy()
        self.re_lo = m.re_lo[self.sel] if m.re_lo is not None else np.zeros(len(self.sel))
        self.im = np.maximum(self.om.imag, 0.0)

        self.ab = abar_real(ic, l, self.w, p)
        self.A2 = abs_A2_real(l, self.w, p)
        self._refine_A2_near_poles(near, mp_dps)

    # ------------------------------------------------------------------------------------
    def _refine_A2_near_poles(self, near: float, dps: int) -> None:
        """Recompute ``|A|^2`` in multiprecision near narrow poles, where it nearly vanishes."""
        if not len(self.sel):
            return
        pos = np.unique(self.re_hi[self.re_hi > 0])
        idx = np.searchsorted(pos, self.w)
        lo = pos[np.clip(idx - 1, 0, len(pos) - 1)]
        hi = pos[np.clip(idx, 0, len(pos) - 1)]
        dmin = np.minimum(np.abs(self.w - lo), np.abs(self.w - hi))
        self.near_idx = np.nonzero(dmin < near)[0]
        with mp.workdps(dps):
            for i in self.near_idx:
                self.A2[i] = float(abs(A_mp(self.l, mp.mpf(float(self.w[i])), self.params)) ** 2)

    def _wave(self, r: float, derivative: bool = False) -> np.ndarray:
        if r >= self.Rs:
            return exterior_standing_wave(self.l, self.w, r, self.params, derivative=derivative)
        z = self.w * r / self.c1
        if not derivative:
            return spherical_jn(self.l, z)
        return (self.w / self.c1) * spherical_jn(self.l, z, derivative=True)

    # ------------------------------------------------------------------------------------
    def _assemble(self, W: np.ndarray, C: np.ndarray, t: np.ndarray, time_derivative: bool):
        """Evaluate the identity for wave values ``W`` on the grid and pole amplitudes ``C``.

        The expression is linear in ``(W, C)``, so ``d/dr`` is obtained by passing derivatives.
        """
        l = self.l
        rho = C * self.c2 * float(self.params.mu2) * l * (l + 1) / 2j
        g = self.w**2 * self.ab * W / self.A2
        d = (self.w[:, None] - self.re_hi[None, :]) - self.re_lo[None, :]
        L = 2.0 * np.real(rho[None, :] / (d - 1j * self.im[None, :])).sum(axis=1)
        R = g - L
        om = self.om
        z_all = np.concatenate([om, np.conj(om)])
        rho_all = np.concatenate([rho, np.conj(rho)])
        a = self.a
        out = np.empty(len(t), dtype=complex)
        for s in range(0, len(t), 512):
            ts = t[s:s + 512]
            if time_derivative:
                integral = -2 * self.dw * (np.sin(np.outer(ts, self.w)) * self.w) @ R
                poles = (np.exp(1j * np.outer(ts, om)) * (1j * om)) @ C
            else:
                integral = 2 * self.dw * np.cos(np.outer(ts, self.w)) @ R
                poles = np.exp(1j * np.outer(ts, om)) @ C
            tt = ts[:, None]
            zz = z_all[None, :]
            eiz = np.exp(1j * zz * tt)
            TR = eiz * exp1(-1j * (a - zz) * tt)
            TL = -eiz * exp1(1j * (a + zz) * tt)
            if time_derivative:
                TR = 1j * zz * TR - np.exp(1j * a * tt) / tt
                TL = 1j * zz * TL + np.exp(-1j * a * tt) / tt
            tails = ((TR + TL) * rho_all[None, :]).sum(axis=1)
            out[s:s + 512] = poles + 0.5 * self.pref * (integral - tails)
        return out

    def F(self, r: float, t, derivatives: bool = False):
        """``F_l(r, t)`` (real part) for times ``t > 0``; optionally ``(F, dF/dr, dF/dt)``.

        If ``t_cut`` was given, times beyond it use the full late-time pole sum (all poles,
        including broad ones), which is exact once the transient has died out.
        """
        t = np.maximum(np.atleast_1d(np.asarray(t, dtype=float)), 1e-9)
        early = t <= self.t_cut if self.t_cut is not None else np.ones(len(t), dtype=bool)
        te = t[early]
        outs = [np.zeros(len(t))] * (3 if derivatives else 1)
        outs = [np.zeros(len(t)) for _ in outs]
        if te.size:
            C = self.modes.coefficients_at(r)[self.sel]
            outs[0][early] = self._assemble(self._wave(r), C, te, False).real
            if derivatives:
                if r < self.Rs:
                    raise NotImplementedError("d/dr inside the star is not implemented")
                dC = (self.modes.K * h2_ratio_dr(self.l, self.modes.omega, r, self.Rs,
                                                 self.c2))[self.sel]
                outs[1][early] = self._assemble(self._wave(r, derivative=True), dC, te, False).real
                outs[2][early] = self._assemble(self._wave(r), C, te, True).real
        if (~early).any():
            tl = t[~early]
            if derivatives:
                late = self.modes.F_with_derivatives(r, tl)
                for o, v in zip(outs, late, strict=True):
                    o[~early] = v
            else:
                outs[0][~early] = self.modes.F(r, tl).real
        return tuple(outs) if derivatives else outs[0]

    def F_with_derivatives(self, r: float, t):
        """Same interface as :meth:`RadialModes.F_with_derivatives`."""
        return self.F(r, t, derivatives=True)

    def diagnostics(self) -> dict:
        """Cheap sanity numbers: spectral cutoff leakage and nearest approach to a narrow pole."""
        g = self.w**2 * self.ab * self._wave(self.Rs) / self.A2
        return {
            "n_grid": self.N,
            "omega_cut": self.a,
            "g_edge_over_max": float(abs(g[-1]) / max(np.abs(g).max(), 1e-300)),
            "n_subtracted_poles": int(len(self.sel)),
            "n_refined_points": int(len(getattr(self, "near_idx", []))),
        }


__all__ = ["GeneralTimeSolver", "alpha_beta_real"]
