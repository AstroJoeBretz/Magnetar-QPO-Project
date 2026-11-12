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
