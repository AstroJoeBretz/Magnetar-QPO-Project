"""Reproduce paper Fig. 6: accumulated energy through spheres of radius 1, 1.5 and 2 R*.

Interior deposit (r0 = 0.7 R*, sigma_r = 1/12, sigma_theta = sigma_phi = 1/3), 0.5 s, fiducial
model (c2 = 370, mu2 = 1/11).  Uses the general-time solution (pole subtraction) from t = 0 and
hands over to the late-time pole sum after 3 crossing times.  Takes about 10 minutes.

Usage: python scripts/make_fig6.py [output.png]
"""

import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from magnetar_qpo.eigen import EigenTable  # noqa: E402
from magnetar_qpo.initial_conditions import InitialCondition  # noqa: E402
from magnetar_qpo.luminosity import escaped_energy_fraction  # noqa: E402
from magnetar_qpo.params import PhysicalScales  # noqa: E402
from magnetar_qpo.solution import solve_ivp  # noqa: E402


def main() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else "escaped_energy.png"
    sc = PhysicalScales()
    table = EigenTable.load("data/eigenvalues/c370_mu2_1o11")
    ic = InitialCondition(0.7, 1 / 12, 1 / 3, 1 / 3)
    sol = solve_ivp(ic, table, lmax=14, lmin=2)
    t = np.arange(0.0, 0.5 / sc.tau, 0.004)
    fig, ax = plt.subplots(figsize=(10, 2.8))
    for r, color in ((1.0, "k"), (1.5, "tab:orange"), (2.0, "tab:blue")):
        frac = escaped_energy_fraction(sol, r, t, general_time=True, omega_cut=200, t_cut=3.0)
        print(f"r = {r} R*: E(0.5 s)/E0 = {frac[-1]:.2e}, max = {frac.max():.2e}")
        ax.semilogy(t * sc.tau, np.clip(frac, 1e-9, None), color=color, lw=1, label=f"{r:g} $R_*$")
    ax.set_xlim(0, 0.5)
    ax.set_ylim(1e-7, 1)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"$\int \dot E/E_0\,dt$")
    ax.legend(ncol=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=160)


if __name__ == "__main__":
    main()
