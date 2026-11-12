"""Audit legacy Mathematica eigenvalue tables against the Python dispersion function.

For each stored root report the Newton correction |D/D'| (how far the stored value is from
a true zero of D).  Usage: python audit_legacy_eigenvalues.py <dir containing evalues*> [dps]
"""
import sys
import time
from pathlib import Path

import mpmath as mp

from magnetar_qpo.dispersion import D_and_dD
from magnetar_qpo.eigen import read_evalues

root = Path(sys.argv[1])
dps = int(sys.argv[2]) if len(sys.argv) > 2 else 260
tol = mp.mpf(10) ** -50
for f in sorted(root.glob("evalues*")):
    t0 = time.time()
    tab = read_evalues(f, dps=dps)
    bad = []
    worst = mp.mpf(0)
    with mp.workdps(dps):
        for (l, n), w in sorted(tab.omega.items()):
            d, dd = D_and_dD(l, w, tab.params)
            step = abs(d / dd)
            worst = max(worst, step) if step < tol else worst
            if step > tol:
                bad.append((l, n, mp.nstr(step, 3)))
    print(f"{f.name}: {len(tab.omega)} entries, {len(bad)} outliers (>1e-50), "
          f"worst good step {mp.nstr(worst,3)}, {time.time()-t0:.0f}s")
    for b in bad[:40]:
        print("   l={} n={} step={}".format(*b))
    if len(bad) > 40:
        print("   ...", len(bad) - 40, "more")
