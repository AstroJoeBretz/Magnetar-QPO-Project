r"""Mode-amplitude spectra and windowed Fourier spectra (paper Sec. 4, Figs 5-9).

Each pole pair ``(omega_k, -conj(omega_k))`` contributes an oscillation
``2 |C_k| e^{-Im(omega_k) t} cos(Re(omega_k) t + arg C_k)`` to ``F_l(R*, t)``; multiplying by
the angular pattern ``G_l`` at the probe point and by the field amplitude gives the surface
displacement amplitude of mode ``(l, n)``.  This is identical (to the accuracy of the Lorentzian
approximation) to the paper's ``pi Im(omega) N_l a_lm(Re omega) w(r_p, omega)``, which
:func:`lorentzian_mode_amplitude` evaluates directly as a cross-check.
"""

from __future__ import annotations

from dataclasses import dataclass

import mpmath as mp
import numpy as np

from .angular import G_H
from .dispersion import A, B
from .params import PhysicalScales
from .solution import IVPSolution
from .special import sph_h1


@dataclass
class ModeSpectrum:
    """Surface displacement amplitude of each type-I mode (units of R*)."""

    l: np.ndarray
    n: np.ndarray
    freq_hz: np.ndarray
    decay_rate: np.ndarray  # 1/s
    amplitude: np.ndarray

    def select(self, f_lo: float, f_hi: float) -> ModeSpectrum:
        m = (self.freq_hz >= f_lo) & (self.freq_hz <= f_hi)
        return ModeSpectrum(*(getattr(self, k)[m] for k in ("l", "n", "freq_hz", "decay_rate",
                                                            "amplitude")))


def mode_spectrum(sol: IVPSolution, r_probe: float | None = None,
                  probe: tuple[float, float] | None = None) -> ModeSpectrum:
    """Amplitudes of all type-I modes at the probe point (default: surface probe of the IC)."""
    Rs = float(sol.table.params.Rs)
    r_probe = Rs if r_probe is None else r_probe
    th, ph = probe if probe is not None else sol.ic.probe_point()
    sc: PhysicalScales = sol.scales
    L, N, FR, DR, AM = [], [], [], [], []
    for l in sol.ls:
        G, _ = G_H(sol.coeffs, l, np.array([th]), np.array([ph]))
        modes = sol.modes[l]
        C = modes.coefficients_at(r_probe)
        sel = np.nonzero((modes.kind == "I") & (modes.omega.real > 0))[0]
        order = sel[np.argsort(modes.omega.real[sel])]
        for n, k in enumerate(order, start=1):
            L.append(l)
            N.append(n)
            FR.append(sc.omega_to_hz(modes.omega[k]))
            DR.append(sc.decay_rate(modes.omega[k]))
            AM.append(sol.amplitude * 2 * abs(C[k]) * abs(G[0]))
    return ModeSpectrum(np.array(L), np.array(N), np.array(FR), np.array(DR), np.array(AM))


def lorentzian_mode_amplitude(sol: IVPSolution, l: int, n: int, r_probe: float | None = None,
                              dps: int | None = None) -> float:
    """Paper's formula ``pi Im(w) N_l(Re w) abar(Re w) G_l w_>(r_p)`` evaluated in multiprecision.

    ``abar`` is the overlap of the initial profile with the standing-wave eigenfunction at real
    frequency ``Re(omega_ln)`` and ``N_l = [l(l+1) pi c2 mu2 |A|^2 / (2 omega^2)]^-1``.
    """
    params = sol.table.params
    P = params.mp()
    w0 = sol.table[(l, n)]
    th, ph = sol.ic.probe_point()
    Rs = float(params.Rs)
    r_probe = Rs if r_probe is None else r_probe
    dps = dps or 120
    with mp.workdps(dps):
        w = mp.mpf(w0.real)
        Aw = A(l, w, params)
        absA2 = abs(Aw) ** 2
        norm = 1 / (l * (l + 1) * mp.pi * P.c2 * P.mu2 * absA2 / (2 * w**2))
        # overlap at real omega with the standing wave (double precision quadrature is enough)
        from .direct import abar_real

        ab = float(abar_real(sol.ic, l, np.array([float(w)]), params)[0])
        G, _ = G_H(sol.coeffs, l, np.array([sol.ic.probe_point()[0]]),
                   np.array([sol.ic.probe_point()[1]]))
        wr = (Aw * sph_h1(l, w * r_probe / P.c2)).real  # Re[A h1] = w_>(r_p)
        amp = mp.pi * w0.imag * norm * ab * float(G[0]) * wr
        _ = (B, th, ph)  # B = conj(A) on the real axis (kept for clarity)
    return float(abs(amp)) * sol.amplitude


def surface_timeseries(sol: IVPSolution, t_start: float, t_stop: float, dt: float) -> tuple[
        np.ndarray, np.ndarray]:
    """Probe displacement u_phi(t) in units of R* on a uniform grid (code time units)."""
    t = np.arange(t_start, t_stop, dt)
    return t, sol.probe_displacement(t)


def windowed_dft(t: np.ndarray, u: np.ndarray, scales: PhysicalScales, n_windows: int = 10,
                 drop: int = 10, f_max_hz: float = 200.0) -> tuple[np.ndarray, np.ndarray]:
    """Overlapping-window DFT amplitude spectrum, as in ``AnalyzeSimulationData.nb``.

    ``n_windows`` windows, each ending ``drop`` samples before the previous one, are Fourier
    transformed (amplitude ``2/sqrt(N) |FFT|``) and their (frequency, amplitude) points merged
    and sorted, which traces out the peaks more smoothly than a single transform.
    """
    dt = t[1] - t[0]
    pts: list[tuple[float, float]] = []
    N_full = len(t)
    for k in range(n_windows):
        N = N_full - k * drop
        if N < 8:
            break
        fft = np.fft.fft(u[:N], norm="ortho")
        span = N * dt
        f_code = np.arange(N // 2) / span  # cycles per code time unit
        f_hz = scales.zshift * f_code / scales.tau
        amp = 2.0 * np.abs(fft[: N // 2]) / np.sqrt(N)
        keep = f_hz <= f_max_hz
        pts.extend(zip(f_hz[keep], amp[keep], strict=True))
    pts.sort()
    arr = np.array(pts)
    return arr[:, 0], arr[:, 1]
