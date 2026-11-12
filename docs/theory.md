# Theory-to-code map

Paper: Bretz, van Eysden & Link, *Tangled magnetic field model of QPOs* (MNRAS). Equation labels
below are the LaTeX labels in the paper source.

## Units

Code units: `R* = 1`, interior wave speed `c1 = 1`, interior shear modulus `mu1 = 1`.
Exterior: wave speed `c2 = c1/gamma` (370 for the fiducial model), shear modulus `mu2` (1/11).
Densities are `rho = mu/c^2`. Time unit `tau = R*/c1 = 0.0123 s`; frequencies in Hz are
`zshift * Re(omega) / (2 pi tau)` with `zshift = 0.766` (`PhysicalScales`).

Naming: the notebooks' `alpha = (c2 mu1 - c1 mu2)/(c2 mu1 + c1 mu2)` is `ModelParams.refl`;
the paper's `alpha = mu_>/mu_<` is `ModelParams.mu_ratio`.

## Equation map

| Paper | Code |
|---|---|
| A_l, B_l (App. C, `eq:A`, `eq:B`) | `dispersion.A`, `dispersion.B` |
| eq. `ShiftBoundaryCondition` (eigenvalue condition) | `dispersion.D` (`A = i x j_l h2_l D`) |
| `eq:ComplexFrequencies` (perturbative) | cross-check only; roots from `eigen.solver` |
| `A'` in the residues | `dispersion.dA`, `dA_at_root`, `D_and_dD` |
| eq. IC, IC2 (initial conditions), App. A | `initial_conditions.InitialCondition`, `angular_momentum` |
| App. B strain energy | `energy.strain_energy` (separable), `strain_energy_brute_force` |
| `a_lm` angular factor (eq. `almFull`) | `angular.projection_coefficients` (`TH1,TH2,PH1,PH2`) |
| `abar_l`, App. D | `residues.radial_overlap_Q` |
| `F_l(r,t)`, eq. `FullContourSolution` | `residues.build_modes`, `RadialModes.F` |
| eq. `Fl` (real-frequency integral) | `direct.F_direct` (reference only) |
| Sec. 4 mode amplitudes | `spectrum.mode_spectrum`, `lorentzian_mode_amplitude` |
| eq. `AlfvenLuminosity` | `luminosity.alfven_luminosity` |

## Eigenvalues

With `z = omega R/c1`, `x = omega R/c2`, `m = mu1/mu2`,

    A_l = i x j_l(z) h2_l(x) D_l(omega),
    D_l = [x h2_{l-1}/h2_l - (l+2)] - m [z j_{l-1}/j_l - (l+2)].

`D'` follows from the Riccati identity `q' = (2l+1) v - x - x v^2` (`q = x v`,
`v = f_{l-1}/f_l`). At a zero, `B = 2 j_l(z)/h2_l(x)` and `A' = i x j_l h2_l D'`.

* **Type I** (interior modes, `Im omega ~ gamma^(2l+1)`): sign changes of the real entire function
  `g = Re[-j_l D_l]` bracket the roots, then complex Newton in mpmath. Precision is set per `l`
  from `gamma^(2l+1)` (`solver.required_dps`; about 190 digits for `l = 25`, `c2 = 370`).
* **Type II** (`|Im omega| ~ c2`): seeds are the zeros of `h2_l` (reverse Bessel polynomial),
  exact for `mu1/mu2 -> infinity`; Newton runs on `h2_l D_l`, which has no pole at the seeds.
  There are `ceil(l/2)` per `l`.

## Residue solution

    F_l(r,t) = sum_k K_k [h2_l(w_k r/c2)/h2_l(w_k R/c2)] e^{i w_k t},
    K_k = w_k^2 Q_k / (c2 mu2 l(l+1) x_k D'(w_k)),

summed over all zeros `w_k` of `A_l` in the upper half plane (type I, type II and the mirror images
`-conj(w)`). `Q_k` is the overlap of the initial radial profile with the pole's eigenfunction
divided by `j_l(z*)`; writing it this way avoids exponentially large intermediate numbers.
This is the late-time form (`FullContourSolution`): valid for `r >= R*` once the initial pulse has
crossed the surface (for an interior deposition at `r0 = 0.7`, `sigma_r = 1/12`: `t >~ 0.8 tau`).
The general-time form with Heaviside conditions from App. D is **not** implemented.

Validated against direct integration over real frequency (`tests/test_residue_solution.py`, agreement
`~1e-5` relative, limited by the truncation of both sums), and against the paper's Lorentzian
mode-amplitude formula (agreement to 6 digits).

## Points where the original notebooks disagree (resolved here)

1. **Mirror-pole factor.** One notebook cell uses `a_{-n} = conj(a_n)`, another `(-1)^l conj(a_n)`.
   Here every pole, including `-conj(w)`, is evaluated directly; the field comes out real
   (`|Im F| ~ 1e-18`), so no symmetry factor is assumed.
2. **Factor 1/2 on the exterior overlap.** The notebook `ab` drops the `1/2` of
   `w_> = (A h1 + B h2)/2`. The derivation above (and agreement with direct integration) fixes it.
3. **`c_<` vs `c_>`** in the paper's exterior overlap integral: the exterior argument uses `c_>`.
4. **Polar profile of the initial condition.** Any field `u = -curl(Psi r)` with
   `Psi = f_r(r) Theta(theta) Phi(phi)` has `u_phi = f_r Theta' Phi`,
   `u_theta = -f_r Theta Phi'/sin(theta)` and is divergence-free. The paper's eq. (IC) corresponds to
   `Theta = f_theta'`, `Phi = f_phi'`, with `f_theta = sin^2(theta) G(theta)`, i.e.
   `Theta_paper = sin(2 theta) G + sin^2(theta) G'`. The figure-producing notebooks
   (`DisplacmentVsFrequency.nb`) use `Theta_nb = sin^2(theta) G'`. They differ by
   `Theta_paper - Theta_nb = sin(2 theta) G(theta)`, about `2 sigma_theta^2` of `Theta_nb` near the
   equator. Both are odd about `theta = pi/2`, so both have zero angular momentum. After normalising
   to the same strain energy the mode spectra agree to within 3% (identical frequencies and
   strongest modes), so the choice is immaterial. `InitialCondition(profile="paper")` is the default
   (it is the paper's eq. IC, term by term); `profile="notebook"` reproduces the recovered notebook.
   The notebooks were recovered from a failed laptop and may be incomplete, which is another reason
   to follow the paper. The axisymmetric field uses the paper's form.
5. **Energy normalisation (a deliberate rounding).** The field of amplitude `a` (in units of R*) has
   strain energy `E = E_unit a^2 E_strain`, where `E_strain` is the dimensionless separable strain
   energy (`energy.strain_energy`) and `E_unit = mu1 R*^3`. The original notebooks use the rounded
   value `E_unit = 1e47 erg`; the literal `mu1 = <B_t^2>/4 pi` with `B_t = 2.3e15 G`, `R* = 10 km`
   would give `4.2e47 erg`. The difference is intentional rounding within the physical
   uncertainty of `B_t` and `R*`: amplitudes scale as `E_unit^(-1/2)` (a factor ~2), which is
   immaterial because 0.01 R* is an order-of-magnitude detectability criterion, not a sharp
   threshold, and the conclusions (which modes are excited, frequencies, energy having to straddle
   the surface) do not depend on it. `1e47` is the default and reproduces the published figures;
   pass `E_unit_erg` / `--energy-unit` to use another value.
6. **Alfven luminosity angular weight.** The notebook squares `a_lm`; orthogonality of the toroidal
   harmonics gives `|a_lm|^2`, which is used here.
7. The strain energy of exterior deposits is weighted by `mu2` (the notebook weights by `mu1`).

## Defects found in the legacy eigenvalue tables

`scripts/audit_legacy_eigenvalues.py` checks every stored root against `D_l`:

* All seven `evalues*` tables agree with this package to the stored 64 digits, **except**:
* `evaluesc370`: for even `l >= 6` the first type-II root (`Re ~ 321`, `Im > 1500`) is stored inside
  the type-I sequence (the table is sorted by `Re`), shifting later entries by one, i.e. entries
  `(6,100), (8,99), ..., (24,91)` are not type-I modes; and `ww[100,25]` is off by `1.7e-4`.
* `evaluesc2`: the odd-`l` `n = 0` entries are not roots of `A_l`, and many `l >= 20` entries are
  accurate to only 47-49 digits.
* `evaluesc100`: `(20, 75)` is accurate to 49 digits.

## Validation status

* Eigenvalues: all 25,000 stored legacy roots verified; solver reproduces them (`pytest -m slow`).
* Residue solution vs direct frequency integration: agree to ~1e-5 (c2 = 2, l = 2, 3, 8, 10).
* Residue luminosity vs direct-integral luminosity (accurate derivatives): agree to 2e-3
  (`tests/test_residue_solution.py`, slow). Integrating the direct-integral luminosity from t = 0
  gives 96% of the initial energy radiated by t = 30 (c2 = 2), still rising.
* Mode spectrum of the fiducial interior deposit (r0 = 0.7 R*, sigma_r = 1/12, sigma_theta =
  sigma_phi = 1/3, 1e46 erg; either polar profile) reproduces the peak positions and heights of paper Fig. 5 (bottom).
* **Not reproduced:** the escaped-energy numbers of paper Fig. 6 (1e-4 at 1.5 R*, 1e-6 at 2 R*
  after 0.5 s). The late-time formula gives -1.4e-5 and -1.6e-6 (and 3.5e-3 at R*) because it
  omits the first ~1 tau, when the early transient crosses these radii; the general-time form is
  needed.

## Not (yet) ported

* General-time residue solution (40 Heaviside-conditioned terms), needed for exterior energy
  deposition at early times and for interior fields near the origin (`ContourOriginDivergence.nb`).
* Beaming maps (`AnalyzeSimulationData.nb`, 3-D mean-amplitude plots).
* The 2016 power-spectrum data (kept locally, not in this repository) came from an earlier model
  without a coupled magnetosphere; no generating code survives, so they are not used.
