"""Container for eigenfrequency tables with exact (decimal-string) persistence."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import mpmath as mp
import numpy as np

from ..params import ModelParams


def _fmt(x, digits: int) -> str:
    return mp.nstr(x, digits, min_fixed=-10**9, max_fixed=10**9) if x != 0 else "0"


@dataclass
class EigenTable:
    """Complex eigenfrequencies ``omega[l, n]`` in code units.

    ``n = 0`` holds the purely imaginary (overdamped) type-II mode for odd ``l``;
    ``n >= 1`` are the type-I overtones (n = 1 is the fundamental), matching the
    Mathematica ``ww[n, l]`` convention.  All type-II modes (``ceil(l/2)`` per ``l``, sorted by
    increasing Re(omega); for odd ``l`` the first is the purely imaginary one) are stored in
    ``type2[(l, k)]``, ``k = 1, 2, ...``; these are absent from the legacy files.
    """

    params: ModelParams
    omega: dict[tuple[int, int], mp.mpc] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    type2: dict[tuple[int, int], mp.mpc] = field(default_factory=dict)

    # ---- access ---------------------------------------------------------------------
    def __getitem__(self, key: tuple[int, int]) -> mp.mpc:
        return self.omega[key]

    def __contains__(self, key) -> bool:
        return key in self.omega

    def ls(self) -> list[int]:
        return sorted({l for l, _ in self.omega})

    def ns(self, l: int, include_zero: bool = False) -> list[int]:
        lo = 0 if include_zero else 1
        return sorted(n for ll, n in self.omega if ll == l and n >= lo)

    def type1(self, l: int, nmax: int | None = None) -> list[mp.mpc]:
        ns = self.ns(l)
        if nmax is not None:
            ns = [n for n in ns if n <= nmax]
        return [self.omega[(l, n)] for n in ns]

    def to_complex128(self, l: int, include_zero: bool = False) -> np.ndarray:
        ns = self.ns(l, include_zero)
        return np.array([complex(self.omega[(l, n)]) for n in ns])

    # ---- persistence ----------------------------------------------------------------
    def save(self, path: str | Path, digits: int = 70) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path.with_suffix(".csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["kind", "l", "n", "re", "im"])
            for (l, n), v in sorted(self.omega.items()):
                w.writerow(["I", l, n, _fmt(v.real, digits), _fmt(v.imag, digits)])
            for (l, k), v in sorted(self.type2.items()):
                w.writerow(["II", l, k, _fmt(v.real, digits), _fmt(v.imag, digits)])
        meta = {
            "c1": str(self.params.c1),
            "c2": str(self.params.c2),
            "mu1": str(self.params.mu1),
            "mu2": str(self.params.mu2),
            "Rs": str(self.params.Rs),
            **self.meta,
        }
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> EigenTable:
        path = Path(path)
        meta = json.loads(path.with_suffix(".json").read_text())
        params = ModelParams.make(
            meta["c2"], meta["mu2"], meta["c1"], meta["mu1"], meta["Rs"]
        )
        tab = cls(params, meta={k: v for k, v in meta.items() if k not in
                               ("c1", "c2", "mu1", "mu2", "Rs")})
        dps = int(meta.get("dps", 80))
        with mp.workdps(max(dps, 80)):
            with open(path.with_suffix(".csv"), newline="") as f:
                for row in csv.DictReader(f):
                    val = mp.mpc(mp.mpf(row["re"]), mp.mpf(row["im"]))
                    target = tab.omega if row.get("kind", "I") == "I" else tab.type2
                    target[(int(row["l"]), int(row["n"]))] = val
        return tab
