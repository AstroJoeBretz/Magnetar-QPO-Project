r"""Assembled initial value solution: displacement field, mode amplitudes, time series.

The displacement for a field of amplitude ``amplitude`` (in units of R*) is

    u_phi(r, th, ph, t)   = amplitude sum_l F_l(r, t) G_l(th, ph)
    u_theta(r, th, ph, t) = amplitude sum_l F_l(r, t) H_l(th, ph)

with the radial functions ``F_l`` from :mod:`magnetar_qpo.residues` (late-time residue sum, valid
for ``r >= R*`` after the initial pulse has crossed the surface) and the angular patterns
``G_l, H_l`` from :mod:`magnetar_qpo.angular`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .angular import G_H, projection_coefficients
from .eigen.table import EigenTable
from .energy import amplitude_for_energy, strain_energy
from .initial_conditions import InitialCondition
from .params import PhysicalScales
from .residues import RadialModes, build_modes


@dataclass
class IVPSolution:
    ic: InitialCondition
    table: EigenTable
    modes: dict[int, RadialModes]
    coeffs: dict[tuple[int, int], complex]
    amplitude: float = 1.0
    scales: PhysicalScales = field(default_factory=PhysicalScales)
    general_cache: dict = field(default_factory=dict, repr=False)

    @property
    def ls(self) -> list[int]:
        return sorted(self.modes)

    def with_energy(self, E_erg: float, E_unit_erg: float) -> IVPSolution:
        """Rescale so that the initial strain energy is ``E_erg`` (``E_unit = mu1 R*^3``)."""
        eps = strain_energy(self.ic, float(self.table.params.mu_ratio))
        a = amplitude_for_energy(E_erg, eps, E_unit_erg)
        return IVPSolution(self.ic, self.table, self.modes, self.coeffs, a, self.scales,
                           self.general_cache)

    def radial(self, l: int, general_time: bool = False, **solver_kw):
        """Radial function object for order ``l`` (``.F`` and ``.F_with_derivatives``).

        ``general_time=False`` is the late-time residue sum; ``True`` is the pole-subtracted
        solution valid at all ``t > 0`` (built once per ``l`` and cached; ``solver_kw`` are
        passed to :class:`magnetar_qpo.transient.GeneralTimeSolver`).
        """
        if not general_time:
            return self.modes[l]
        key = (l, tuple(sorted(solver_kw.items())))
        if key not in self.general_cache:
            from .transient import GeneralTimeSolver

            self.general_cache[key] = GeneralTimeSolver(
                self.ic, self.table, l, modes=self.modes[l], **solver_kw)
        return self.general_cache[key]

    # ---- fields ---------------------------------------------------------------------
    def displacement(self, r: float, theta: float, phi: float, t, general_time: bool = False,
                     **solver_kw) -> tuple[np.ndarray, np.ndarray]:
        """(u_theta, u_phi) in units of R* at one point for an array of times."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        ut = np.zeros_like(t)
        up = np.zeros_like(t)
        for l in self.ls:
            G, H = G_H(self.coeffs, l, np.array([theta]), np.array([phi]))
            F = self.radial(l, general_time, **solver_kw).F(r, t).real
            up += F * G[0]
            ut += F * H[0]
        return self.amplitude * ut, self.amplitude * up

    def probe_displacement(self, t, general_time: bool = False, **solver_kw) -> np.ndarray:
        """u_phi at the surface probe point (paper Sec. 4) as a function of time."""
        th, ph = self.ic.probe_point()
        return self.displacement(float(self.table.params.Rs), th, ph, t, general_time,
                                 **solver_kw)[1]


def solve_ivp(
    ic: InitialCondition,
    table: EigenTable,
    lmax: int,
    lmin: int = 1,
    nmax: int | None = None,
    include_type2: bool = True,
    amplitude: float = 1.0,
    scales: PhysicalScales | None = None,
) -> IVPSolution:
    """Build the residue solution for orders ``lmin..lmax`` (table must cover them)."""
    coeffs = projection_coefficients(ic, lmax, lmin=lmin)
    modes = {
        l: build_modes(ic, l, table, nmax=nmax, include_type2=include_type2)
        for l in range(lmin, lmax + 1)
    }
    return IVPSolution(ic, table, modes, coeffs, amplitude, scales or PhysicalScales())
