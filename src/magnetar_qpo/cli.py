"""Command-line interface: ``qpo eigen`` and ``qpo spectrum``."""

from __future__ import annotations

import argparse
from pathlib import Path

from .params import ModelParams


def _cmd_eigen(a: argparse.Namespace) -> None:
    from .eigen.solver import solve_table

    tab = solve_table(ModelParams.make(a.c2, a.mu2), lmax=a.lmax, nmax=a.nmax, workers=a.workers)
    tab.save(a.out)
    print(f"wrote {len(tab.omega)} type-I and {len(tab.type2)} type-II modes to {a.out}.csv")


def _cmd_spectrum(a: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .eigen import EigenTable
    from .initial_conditions import InitialCondition
    from .plotting import plot_mode_spectrum
    from .solution import solve_ivp
    from .spectrum import mode_spectrum

    tab = EigenTable.load(a.table)
    ic = InitialCondition(
        a.r0, a.sigma_r, a.sigma_theta, a.sigma_phi, axisymmetric=a.axisymmetric, profile=a.profile
    )
    sol = solve_ivp(ic, tab, lmax=a.lmax, nmax=a.nmax).with_energy(a.energy, a.energy_unit)
    spec = mode_spectrum(sol)
    ax = plot_mode_spectrum(spec, f_max=a.fmax)
    ax.figure.tight_layout()
    ax.figure.savefig(a.out, dpi=200)
    plt.close(ax.figure)
    print(f"{len(spec.l)} modes; max amplitude {spec.amplitude.max():.4g} R*; figure -> {a.out}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="qpo", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("eigen", help="compute an eigenfrequency table")
    e.add_argument("--c2", default="370", help="exterior/interior wave-speed ratio")
    e.add_argument("--mu2", default="1/11", help="exterior/interior shear-modulus ratio")
    e.add_argument("--lmax", type=int, default=10)
    e.add_argument("--nmax", type=int, default=50)
    e.add_argument("--workers", type=int, default=None, help="processes (default: all cores)")
    e.add_argument("--out", type=Path, default=Path("eigenvalues"))
    e.set_defaults(fn=_cmd_eigen)

    s = sub.add_parser("spectrum", help="mode-amplitude spectrum for an initial condition")
    s.add_argument("--table", type=Path, required=True, help="eigen table stem (no extension)")
    s.add_argument("--r0", type=float, default=0.7)
    s.add_argument("--sigma-r", type=float, default=1 / 12)
    s.add_argument("--sigma-theta", type=float, default=1 / 3)
    s.add_argument("--sigma-phi", type=float, default=1 / 3)
    s.add_argument("--axisymmetric", action="store_true")
    s.add_argument("--profile", choices=["notebook", "paper"], default="paper")
    s.add_argument("--lmax", type=int, default=10)
    s.add_argument("--nmax", type=int, default=25)
    s.add_argument("--energy", type=float, default=1e46, help="initial strain energy [erg]")
    s.add_argument("--energy-unit", type=float, default=1e47,
                   help="mu1 R*^3 in erg; 1e47 is the rounded value used for the paper's figures")
    s.add_argument("--fmax", type=float, default=200.0)
    s.add_argument("--out", type=Path, default=Path("spectrum.png"))
    s.set_defaults(fn=_cmd_spectrum)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
