# magnetar-qpo

Python code for the tangled-magnetic-field model of magnetar quasi-periodic oscillations
(QPOs): shear waves in a uniformly tangled stellar interior coupled to a relativistic
Alfvén-wave magnetosphere, solved as an initial value problem.

1. Find the complex eigenfrequencies ω_ℓn (zeros of the boundary-condition function A_ℓ(ω)) at
   arbitrary precision (`magnetar_qpo.eigen`).
2. Close the frequency integral of the initial value solution in the complex plane and sum
   residues at those poles (`magnetar_qpo.residues`, `solution`).
3. Derive mode amplitudes, power spectra, time series and Alfvén-wave energy flux
   (`spectrum`, `luminosity`).

This package replaces the original Mathematica notebooks, archived at
<https://github.com/AstroJoeBretz/QPO-Project>. See [`docs/theory.md`](docs/theory.md) for the
equation-to-code map, the discrepancies found between notebooks, and what is not yet ported.

## Reference

Bretz, J., van Eysden, C. A., & Link, B., *Tangled magnetic field model of QPOs*, Monthly Notices
of the Royal Astronomical Society (`CITATION.cff`; the DOI still needs to be filled in).

## Install and test

```bash
pip install -e ".[dev]"
pytest              # fast tests
pytest -m slow      # full-precision checks (set QPO_LEGACY_DIR to compare with the old tables)
```

## Quick start

```bash
# eigenfrequencies for the paper's fiducial model (c2 = 370, mu2 = 1/11), l <= 25, 100 overtones
python scripts/make_eigen_table.py 370 1/11 25 100 data/eigenvalues/c370_mu2_1o11

# mode-amplitude spectrum for an interior energy deposit of 1e46 erg (paper Fig. 5)
qpo spectrum --table data/eigenvalues/c370_mu2_1o11 --r0 0.7 --out spectrum.png
```

```python
from magnetar_qpo.eigen import EigenTable
from magnetar_qpo.initial_conditions import InitialCondition
from magnetar_qpo.solution import solve_ivp
from magnetar_qpo.spectrum import mode_spectrum

table = EigenTable.load("data/eigenvalues/c370_mu2_1o11")
ic = InitialCondition(r0=0.7, sigma_r=1/12, sigma_theta=1/3, sigma_phi=1/3)
sol = solve_ivp(ic, table, lmax=10, nmax=25).with_energy(1e46, E_unit_erg=1e47)
spec = mode_spectrum(sol)          # frequency [Hz], decay rate [1/s], amplitude [R*] per mode
```

## Layout

- `src/magnetar_qpo/` – the package (`dispersion`, `eigen/`, `initial_conditions`, `angular`,
  `energy`, `residues`, `solution`, `direct`, `spectrum`, `luminosity`, `plotting`, `cli`).
- `data/eigenvalues/` – eigenfrequencies computed by this package (decimal strings + JSON metadata).
- `scripts/` – table generation, legacy-table converter and audit.
- `tests/` – unit tests and cross-checks against independent calculations.

## `archived_code/`

Read-only legacy material, not used by the package:

- `ExtraModes.nb` (2020) – finder for the complex type-II modes using hand-fitted seeds.
- `SurprisePoles.nb` (2019) – pole scans for exterior speed ratios c2 < 1 and small μ2.
- `Power Spectra Data/` (2016) – ℓ × n amplitude matrices from an earlier version of the model.
