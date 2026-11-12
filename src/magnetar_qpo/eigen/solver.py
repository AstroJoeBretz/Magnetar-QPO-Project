"""High-precision solver for the eigenfrequencies omega_ln (zeros of A_l).

Type-I modes (interior-dominated, Re(omega) > 0, tiny Im(omega) ~ gamma^(2l+1)):
    real-axis sign scan of an entire real function -> Brent -> complex Newton in mpmath.
Type-II modes (magnetosphere-dominated, Im(omega) ~ c2/Rs):
    seeds from the zeros of h2_l (reverse Bessel polynomial) -> Newton, with continuation
    in mu1/mu2 as a fall-back.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor

import mpmath as mp
import numpy as np
from scipy.optimize import brentq

from ..dispersion import D_and_dD, F_and_dF
from ..params import ModelParams
from .seeds import bracket_real_roots, real_axis_function, reverse_bessel_zeros
from .table import EigenTable

SOLVER_VERSION = "0.1"


def required_dps(l: int, params: ModelParams, margin: int = 60) -> int:
    """Working precision needed to resolve Im(omega) ~ gamma^(2l+1) to many digits."""
    gam = float(params.gamma)
    extra = max(0.0, (2 * l + 1) * math.log10(1.0 / gam)) if gam < 1 else 0.0
    return int(margin + math.ceil(extra))


def newton_root(
    l: int,
    params: ModelParams,
    w0,
    dps: int,
    maxit: int = 80,
    tol_digits: int | None = None,
    pole_free: bool = False,
) -> mp.mpc:
    """Complex Newton iteration on D_l(omega) at ``dps`` digits; raises on non-convergence.

    ``pole_free=True`` iterates on ``h2_l(x) D_l`` instead (needed near zeros of ``h2_l``).
    """
    func = F_and_dF if pole_free else D_and_dD
    tol_digits = dps - 20 if tol_digits is None else tol_digits
    with mp.workdps(dps):
        w = mp.mpc(w0)
        tol = mp.mpf(10) ** (-tol_digits) * max(abs(w), 1)
        for _ in range(maxit):
            d, dd = func(l, w, params)
            step = d / dd
            w = w - step
            if abs(step) < tol:
                return +w
    raise RuntimeError(f"Newton did not converge for l={l}, seed={w0}")


def refine_root(
    l: int, params: ModelParams, w0, dps: int, pole_free: bool = False
) -> mp.mpc:
    """Two-stage refinement: cheap Newton at 40 digits, then at the full precision."""
    stage1 = newton_root(l, params, w0, dps=min(40, dps), pole_free=pole_free)
    return newton_root(l, params, stage1, dps=dps, pole_free=pole_free)


def solve_type1(
    l: int,
    params: ModelParams,
    nmax: int | None = 100,
    omega_max: float | None = None,
    dps: int | None = None,
    w_min: float = 0.01,
    step: float = 0.02,
) -> list[mp.mpc]:
    """Type-I eigenfrequencies for angular order ``l``, ordered by increasing Re(omega).

    Either ``nmax`` roots are returned (the scan range is extended as needed) or, if
    ``omega_max`` is given, all roots with Re(omega) < omega_max.
    """
    dps = required_dps(l, params) if dps is None else dps
    roots: list[mp.mpc] = []
    lo = w_min
    chunk = 60.0 if omega_max is None else omega_max - w_min
    hi_cap = omega_max if omega_max is not None else 50.0 * (nmax or 100) + 1000.0
    while True:
        hi = min(lo + chunk, hi_cap)
        for a, b in bracket_real_roots(l, params, lo, hi, step):
            seed = brentq(lambda w: real_axis_function(l, np.array([w]), params)[0], a, b,
                          xtol=1e-14)
            roots.append(refine_root(l, params, seed, dps))
        if omega_max is not None or (nmax is not None and len(roots) >= nmax) or hi >= hi_cap:
            break
        lo = hi
    roots = _dedupe_sorted(roots)
    return roots[:nmax] if (omega_max is None and nmax is not None) else roots


def _dedupe_sorted(roots: list[mp.mpc]) -> list[mp.mpc]:
    roots = sorted(roots, key=lambda r: (float(r.real), float(r.imag)))
    out: list[mp.mpc] = []
    for r in roots:
        if out and abs(r - out[-1]) < mp.mpf(10) ** -12 * max(1, abs(r)):
            continue
        out.append(r)
    return out


def solve_type2(l: int, params: ModelParams, dps: int | None = None) -> list[mp.mpc]:
    """Type-II (overdamped) eigenfrequencies with Re(omega) >= 0, sorted by Re(omega).

    There are ceil(l/2) of them; for odd l the first is purely imaginary.
    """
    dps = max(required_dps(l, params), 60) if dps is None else dps
    seeds = reverse_bessel_zeros(l) * float(params.c2 / params.Rs)
    found: list[mp.mpc] = []
    for s in seeds:
        try:
            w = refine_root(l, params, complex(s), dps, pole_free=True)
        except (RuntimeError, ZeroDivisionError):
            w = _continue_in_stiffness(l, params, complex(s), dps)
        if abs(w.real) < mp.mpf(10) ** (-(dps - 25)) * abs(w):
            w = mp.mpc(0, w.imag)
        if w.imag > 0:
            found.append(w)
    found = _dedupe_sorted(found)
    if len(found) != math.ceil(l / 2):
        raise RuntimeError(
            f"type-II count for l={l}: found {len(found)}, expected {math.ceil(l / 2)}"
        )
    return found


def _continue_in_stiffness(l: int, params: ModelParams, w_seed: complex, dps: int) -> mp.mpc:
    """Follow a root from large mu1/mu2 (exactly the h2 zero) down to the target value."""
    from dataclasses import replace
    from fractions import Fraction

    m_target = params.mu1 / params.mu2
    w = w_seed
    # geometric ladder of stiffness contrasts
    ladder = [Fraction(10) ** k for k in range(8, 0, -1)]
    ladder = [m for m in ladder if m > m_target] + [m_target]
    for m in ladder:
        p = replace(params, mu2=params.mu1 / m)
        w = newton_root(l, p, w, dps=min(50, dps), pole_free=True)
    return refine_root(l, params, w, dps, pole_free=True)


def _solve_l(args) -> tuple[int, list[mp.mpc], list[mp.mpc]]:
    l, params, nmax, omega_max, with_type2 = args
    t1 = solve_type1(l, params, nmax=nmax, omega_max=omega_max)
    t2 = solve_type2(l, params) if with_type2 else []
    return l, t1, t2


def solve_table(
    params: ModelParams,
    lmax: int,
    nmax: int | None = 100,
    lmin: int = 1,
    omega_max: float | None = None,
    with_type2: bool = True,
    workers: int | None = 1,
    progress: Callable[[int], None] | None = None,
) -> EigenTable:
    """Compute the full eigenfrequency table for ``lmin <= l <= lmax``.

    ``workers=None`` uses all cores (processes), ``workers=1`` runs serially.
    """
    jobs = [(l, params, nmax, omega_max, with_type2) for l in range(lmin, lmax + 1)]
    if workers == 1:
        results = []
        for j in jobs:
            results.append(_solve_l(j))
            if progress:
                progress(j[0])
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_solve_l, jobs))
    tab = EigenTable(params, meta={"solver": SOLVER_VERSION, "nmax": nmax, "lmax": lmax,
                                   "dps": max(required_dps(lmax, params), 80)})
    for l, t1, t2 in results:
        for n, w in enumerate(t1, start=1):
            tab.omega[(l, n)] = w
        for k, w in enumerate(t2, start=1):
            tab.type2[(l, k)] = w
        if l % 2 == 1 and t2:
            tab.omega[(l, 0)] = t2[0]  # legacy alias: ww[0, l] is the imaginary mode
    return tab
