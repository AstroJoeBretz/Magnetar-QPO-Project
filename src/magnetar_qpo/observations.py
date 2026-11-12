"""Observed magnetar QPO frequencies quoted in the paper (Hz, 1-sigma width).

A width of 0 means the width was not measured.
"""

from __future__ import annotations

#: SGR 1900+14 giant flare (Strohmayer & Watts 2005)
SGR1900 = [(28.0, 2.0), (53.0, 5.0), (84.0, 0.0), (155.0, 6.0)]

#: SGR 1806-20 giant flare (Israel et al. 2005; Strohmayer & Watts 2006; Watts & Strohmayer 2006;
#: Hambaryan et al. 2011)
SGR1806 = [(18.0, 2.0), (26.0, 3.0), (30.0, 4.0), (93.0, 2.0), (150.0, 17.0), (626.0, 2.0),
           (1837.0, 5.0)]

#: Additional SGR 1806-20 QPOs from the reanalysis of Miller et al. (2019)
SGR1806_MILLER19 = [(51.0, 0.52), (97.0, 5.94), (157.0, 1.14)]

#: Averaged short bursts of SGR J1550-5418 (Huppenkothen et al. 2014b)
SGR1550 = [(93.0, 0.0), (127.0, 0.0)]

CATALOGS = {
    "SGR1900": SGR1900,
    "SGR1806": SGR1806 + SGR1806_MILLER19,
    "SGR1550": SGR1550,
}
