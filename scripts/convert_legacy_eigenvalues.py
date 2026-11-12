"""Convert Mathematica ``evalues*`` files to the package's CSV+JSON format.

Usage:
    python scripts/convert_legacy_eigenvalues.py <evalues file> <output stem> [lmax] [nmax]

``lmax``/``nmax`` optionally truncate the table (used to build the small CI test fixture).
Note: a few legacy entries are inaccurate (see scripts/audit_legacy_eigenvalues.py); use the
solver in ``magnetar_qpo.eigen.solver`` to regenerate tables.
"""

import sys

from magnetar_qpo.eigen import read_evalues

src, out = sys.argv[1], sys.argv[2]
lmax = int(sys.argv[3]) if len(sys.argv) > 3 else None
nmax = int(sys.argv[4]) if len(sys.argv) > 4 else None
tab = read_evalues(src)
if lmax is not None:
    tab.omega = {k: v for k, v in tab.omega.items() if k[0] <= lmax}
if nmax is not None:
    tab.omega = {k: v for k, v in tab.omega.items() if k[1] <= nmax}
tab.save(out, digits=70)
print(f"wrote {len(tab.omega)} entries to {out}.csv")
