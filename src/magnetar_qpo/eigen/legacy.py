"""Reader for the Mathematica ``evalues*`` eigenvalue files (``ww[n,l]=...;`` assignments)."""

from __future__ import annotations

import re
from fractions import Fraction
from pathlib import Path

import mpmath as mp

from ..params import ModelParams
from .table import EigenTable

_ASSIGN = re.compile(r"^\s*(\w+)\s*=\s*([^;]+);\s*$")
_WW = re.compile(r"^\s*ww\[\s*(-?\d+)\s*,\s*(\d+)\s*\]\s*=\s*(.+?)\s*;\s*$")
# one real term: [sign] digits[.digits] [`prec] [*^exp] [*I]
_TERM = re.compile(
    r"([+-]?)\s*(\d+\.?\d*)(?:`[\d.]*)?(?:\*\^(-?\d+))?(\*I)?"
)


def _parse_complex(expr: str) -> mp.mpc:
    expr = expr.replace("\\\n", "").replace("\n", "")
    re_part = mp.mpf(0)
    im_part = mp.mpf(0)
    pos = 0
    for m in _TERM.finditer(expr):
        sign, digits, exp, imag = m.groups()
        val = mp.mpf(digits)
        if exp:
            val = val * mp.mpf(10) ** int(exp)
        if sign == "-":
            val = -val
        if imag:
            im_part += val
        else:
            re_part += val
        pos = m.end()
    if expr[pos:].strip():
        raise ValueError(f"unparsed trailing text in {expr!r}")
    return mp.mpc(re_part, im_part)


def read_evalues(path: str | Path, dps: int = 90) -> EigenTable:
    """Parse a legacy ``evaluesc*`` file into an :class:`EigenTable`.

    The header assignments (``c2=370; mu2=1/11; ...``) give the model parameters.
    """
    text = Path(path).read_text().replace("\\\r\n", "").replace("\\\n", "")
    hdr: dict[str, str] = {}
    omega: dict[tuple[int, int], mp.mpc] = {}
    with mp.workdps(dps):
        for line in text.splitlines():
            if not line.strip():
                continue
            m = _WW.match(line)
            if m:
                n, l, expr = int(m.group(1)), int(m.group(2)), m.group(3)
                omega[(l, n)] = _parse_complex(expr)
                continue
            m = _ASSIGN.match(line)
            if m:
                hdr[m.group(1)] = m.group(2).strip()
    params = ModelParams.make(
        hdr["c2"], hdr.get("mu2", "1"), hdr.get("c1", "1"), hdr.get("mu1", "1"), hdr.get("Rs", "1")
    )
    meta = {"source": Path(path).name, "dps": dps, "legacy_prec": hdr.get("prec")}
    return EigenTable(params, omega, meta)


__all__ = ["read_evalues", "Fraction"]
