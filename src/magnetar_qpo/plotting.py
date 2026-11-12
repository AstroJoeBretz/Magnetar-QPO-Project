"""Paper-style figures (matplotlib).  Every function returns the ``Axes`` it drew on."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .eigen.table import EigenTable
from .initial_conditions import InitialCondition
from .observations import SGR1900
from .params import PhysicalScales
from .spectrum import ModeSpectrum


def plot_eigenfrequencies(table: EigenTable, scales: PhysicalScales | None = None, lmax: int = 5,
                          nmax: int = 30, ax=None):
    """Decay rate vs frequency of the type-I modes, coloured by ``l`` (paper Fig. 2, left)."""
    scales = scales or PhysicalScales()
    ax = ax or plt.subplots(figsize=(6, 4))[1]
    for l in table.ls():
        if l > lmax:
            break
        w = table.to_complex128(l)[:nmax]
        # decay rates can underflow double precision; use log10 of the multiprecision values
        im = np.array([float(v.imag) for v in table.type1(l, nmax)])
        ax.semilogy(scales.omega_to_hz(w), scales.zshift * im / scales.tau, "-D", ms=3,
                    label=f"l={l}")
    ax.axhline(1 / 300.0, color="gray", lw=0.8)
    ax.set_xlabel("Frequency [Hz]   Re[$\\omega/2\\pi$]")
    ax.set_ylabel("Decay rate [s$^{-1}$]   Im[$\\omega$]")
    ax.legend(ncol=2, fontsize=8)
    return ax


def plot_mode_spectrum(spec: ModeSpectrum, observed=SGR1900, threshold: float | None = 0.01,
                       f_max: float = 200.0, ax=None):
    """Surface-displacement amplitude of each mode vs frequency (paper Figs. 5-9)."""
    ax = ax or plt.subplots(figsize=(8, 3.5))[1]
    m = spec.freq_hz <= f_max
    ax.vlines(spec.freq_hz[m], 0, spec.amplitude[m], color="k", lw=1.2)
    ymax = 1.15 * max(spec.amplitude[m].max(), threshold or 0)
    for f, w in observed:
        if w > 0:
            ax.axvspan(f - w, f + w, color="tab:red", alpha=0.25, lw=0)
        ax.annotate("", xy=(f, 0.86 * ymax), xytext=(f, 0.98 * ymax),
                    arrowprops=dict(arrowstyle="-|>", color="k"))
    if threshold:
        ax.axhline(threshold, ls="--", color="gray", lw=0.8)
    ax.set_xlim(0, f_max)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel(r"$|\tilde u_s|/R_*$")
    return ax


def plot_initial_condition(ic: InitialCondition, ax=None, n: int = 40):
    """Vector plot of the angular dependence of the initial displacement (paper Fig. 3)."""
    ax = ax or plt.subplots(figsize=(7, 3.5))[1]
    th = np.linspace(0.05, np.pi - 0.05, n // 2)
    ph = np.linspace(0, 2 * np.pi, n)
    T, P = np.meshgrid(th, ph, indexing="ij")
    uth, uph = ic.u_theta(ic.r0, T, P), ic.u_phi(ic.r0, T, P)
    ax.quiver(P, T, uph, -uth, scale=None, width=0.003)
    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"$\theta$")
    ax.invert_yaxis()
    for x in (ic.phi0 - ic.sigma_phi, ic.phi0 + ic.sigma_phi):
        ax.axvline(x, color="gray", lw=0.5)
    for y in (ic.theta0 - ic.sigma_theta, ic.theta0 + ic.sigma_theta):
        ax.axhline(y, color="gray", lw=0.5)
    return ax


def plot_escaped_energy(t, fractions: dict[str, np.ndarray], scales: PhysicalScales | None = None,
                        ax=None):
    """Accumulated escaped energy E(t)/E0 through spheres of several radii (paper Fig. 6)."""
    scales = scales or PhysicalScales()
    ax = ax or plt.subplots(figsize=(7, 3))[1]
    for label, frac in fractions.items():
        ax.semilogy(np.asarray(t) * scales.tau, frac, label=label)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("$E/E_0$")
    ax.legend()
    return ax
