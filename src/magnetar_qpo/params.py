"""Model parameters in code units.

Code units follow the paper's Mathematica implementation: the stellar radius is
``Rs = 1`` and the interior wave speed is ``c1 = 1``.  The exterior (magnetosphere)
wave speed is ``c2`` (about 370 for SGR 1900+14, i.e. ``gamma = c1/c2 = 0.0027``) and
the ratio of exterior to interior shear modulus is ``mu2/mu1`` (the paper's ``alpha``,
about 1/11 for the fiducial tangle strength).

Naming note: the Mathematica notebooks call ``(c2 mu1 - c1 mu2)/(c2 mu1 + c1 mu2)``
``alpha`` (a reflection coefficient).  That quantity is :attr:`ModelParams.refl` here;
the paper's ``alpha = mu_> / mu_<`` is :attr:`ModelParams.mu_ratio`.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import mpmath as mp

Number = int | float | str | Fraction


def _frac(x: Number) -> Fraction:
    if isinstance(x, Fraction):
        return x
    if isinstance(x, float):
        return Fraction(str(x))
    return Fraction(x)


@dataclass(frozen=True)
class ModelParams:
    """Two-region (star + magnetosphere) shear-wave model in code units.

    Values are stored as exact fractions so that high-precision work is not polluted by
    binary floating-point representation (``mu2="1/11"`` is exactly 1/11).
    """

    c2: Fraction
    mu2: Fraction = Fraction(1)
    c1: Fraction = Fraction(1)
    mu1: Fraction = Fraction(1)
    Rs: Fraction = Fraction(1)

    @classmethod
    def make(
        cls,
        c2: Number,
        mu2: Number = 1,
        c1: Number = 1,
        mu1: Number = 1,
        Rs: Number = 1,
    ) -> ModelParams:
        return cls(_frac(c2), _frac(mu2), _frac(c1), _frac(mu1), _frac(Rs))

    # -- derived dimensionless quantities (exact) ---------------------------------------
    @property
    def gamma(self) -> Fraction:
        """Speed ratio c_< / c_> (paper eq. SpeedRatio)."""
        return self.c1 / self.c2

    @property
    def mu_ratio(self) -> Fraction:
        """The paper's alpha = mu_> / mu_<."""
        return self.mu2 / self.mu1

    @property
    def refl(self) -> Fraction:
        """The notebooks' ``alpha``: (c2 mu1 - c1 mu2)/(c2 mu1 + c1 mu2)."""
        return (self.c2 * self.mu1 - self.c1 * self.mu2) / (self.c2 * self.mu1 + self.c1 * self.mu2)

    # -- mpmath views at the current working precision ----------------------------------
    def mp(self) -> MPParams:
        def f(q: Fraction):
            return mp.mpf(q.numerator) / mp.mpf(q.denominator)

        return MPParams(f(self.c1), f(self.c2), f(self.mu1), f(self.mu2), f(self.Rs))

    def label(self) -> str:
        return f"c2={self.c2}_mu2={self.mu2}".replace("/", "o")


@dataclass(frozen=True)
class MPParams:
    c1: mp.mpf
    c2: mp.mpf
    mu1: mp.mpf
    mu2: mp.mpf
    Rs: mp.mpf

    @property
    def rho1(self):
        """Interior effective density mu1/c1^2."""
        return self.mu1 / self.c1**2

    @property
    def rho2(self):
        """Exterior effective density mu2/c2^2."""
        return self.mu2 / self.c2**2

    @property
    def m(self):
        """mu1/mu2, the stiffness contrast appearing in the boundary condition."""
        return self.mu1 / self.mu2


@dataclass(frozen=True)
class PhysicalScales:
    """Conversion of code-unit frequencies/times to physical ones.

    ``tau`` is the interior crossing time R*/c_< in seconds (0.0123 s in the paper's
    fiducial model) and ``zshift`` the gravitational redshift factor (0.766).
    Physical frequency [Hz] = zshift * Re(omega) / (2 pi tau).
    """

    tau: float = 0.0123
    zshift: float = 0.766

    def omega_to_hz(self, omega):
        import numpy as np

        return self.zshift * np.real(omega) / (2 * np.pi * self.tau)

    def decay_rate(self, omega):
        """Decay rate [1/s] of a mode: zshift * Im(omega) / tau."""
        import numpy as np

        return self.zshift * np.imag(omega) / self.tau
