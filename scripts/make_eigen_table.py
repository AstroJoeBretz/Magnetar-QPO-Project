"""Compute an eigenfrequency table and save it under data/eigenvalues/.

Usage: python scripts/make_eigen_table.py <c2> <mu2> <lmax> <nmax> <output stem>
Paper fiducial model:
    python scripts/make_eigen_table.py 370 1/11 25 100 data/eigenvalues/c370_mu2_1o11
"""

import sys
import time

from magnetar_qpo import ModelParams
from magnetar_qpo.eigen.solver import solve_table


def main() -> None:
    c2, mu2, lmax, nmax, out = sys.argv[1:6]
    t0 = time.time()
    tab = solve_table(ModelParams.make(c2, mu2), lmax=int(lmax), nmax=int(nmax), workers=None)
    tab.save(out)
    print(
        f"{len(tab.omega)} type-I/legacy-alias + {len(tab.type2)} type-II modes "
        f"in {time.time() - t0:.0f}s -> {out}.csv"
    )


if __name__ == "__main__":
    main()
