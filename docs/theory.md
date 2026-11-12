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

## General-time solution (pole subtraction)

The radial function is `F = (pref/2) int_{-inf}^{inf} g(w) e^{iwt} dw` with `g` real and even on the
real axis and `pref = 2/(pi c2 mu2 l(l+1))`. Its continuation has simple poles at the zeros `w_k` of
`A` (upper half plane) and `conj(w_k)`. The late-time formula is the pole part `sum_k C_k e^{i w_k t}`
and is only valid once every signal delay has elapsed. For a set `S` of narrow poles
(`Im w_k < gamma_s`) with residues `rho_k` (`C_k = i pi pref rho_k`),

    F = sum_{k in S} C_k e^{i w_k t}
        + (pref/2) [ int_{-a}^{a} (g - L_S) e^{iwt} dw - int_{|w|>a} L_S e^{iwt} dw ],
    L_S = sum_{k in S} 2 Re[ rho_k/(w - w_k) ],

exactly, for all `t > 0`. The bracket is smooth (the Lorentzian spikes are removed analytically), so a
uniform grid of spacing 0.02 resolves it; the tails of `L_S` beyond the cutoff `a` are closed-form
(`E1`). Inside the star the eigenfunction is `j_l(w r/c1)` (entire), so there is no divergence at
the origin. With `S` empty this is the direct frequency integral. Practical notes:

* `omega_cut` should cover the spectral content of the initial profile. The pole amplitudes fall
  off as a Gaussian only to ~1e-7 of their peak and then as a slow power law (the density jump at
  the surface puts a kink in every eigenfunction), so results converge slowly (~1e-5 relative at
  `omega_cut` = 108 vs 340 for the fiducial interior deposit). An *exterior* deposit has bandwidth
  ~`c2/sigma_r` (~4000 for the fiducial model), far beyond the 100-overtone tables.
* `t_cut` hands over to the full late-time pole sum once the transient has died out (a few
  crossing times); this makes long series cheap.
* `d/dr` is available outside the star only.
* Near-pole accuracy: `|A|^2` is recomputed in multiprecision within 0.05 of a narrow pole.

## Validation status

* Eigenvalues: all 25,000 stored legacy roots verified; solver reproduces them (`pytest -m slow`).
* Residue solution vs direct frequency integration: agree to ~1e-5 (c2 = 2, l = 2, 3, 8, 10).
* Residue luminosity vs direct-integral luminosity (accurate derivatives): agree to 2e-3
  (`tests/test_residue_solution.py`, slow). Integrating the direct-integral luminosity from t = 0
  gives 96% of the initial energy radiated by t = 30 (c2 = 2), still rising.
* Mode spectrum of the fiducial interior deposit (r0 = 0.7 R*, sigma_r = 1/12, sigma_theta =
  sigma_phi = 1/3, 1e46 erg; either polar profile) reproduces the peak positions and heights of paper Fig. 5 (bottom).
* General-time solution (`transient.GeneralTimeSolver`): agrees with the direct frequency integral
  to ~1e-7 at all t for c2 = 2 (where the late-time formula is wrong by more than the field itself
  at early times); reproduces the initial profile, `l(l+1) F_l(r, 0) = f_r(r)`, and `dF/dt(0) = 0`
  to 1e-6 (interior deposit), 3e-5 (deposit straddling the surface) and 5e-10 (exterior deposit),
  at radii from 0.3 to 1.8 R*; matches the late-time sum to ~1e-8 at late times for c2 = 370; and
  its result is independent of the grid spacing to 1e-14 at c2 = 370.
* Paper Fig. 6 (escaped energy through 1, 1.5, 2 R* in 0.5 s) is reproduced by
  `scripts/make_fig6.py`: ~1e-2 (peaks 2e-2), ~1e-4 (peaks 3e-4) and ~1e-5 (peaks 2e-5), including the
  early dip near 0.03 s (`docs/figures/escaped_energy.png`). Integrating from t = 0 is essential: the
  cumulative flux measures the near-field energy built up in the first crossing time.

## Not (yet) ported

* The paper's 40-term Heaviside-conditioned form (App. D). It is replaced by the equivalent
  pole-subtracted formulation below, which needs no new asymptotic algebra.
* Beaming maps (`AnalyzeSimulationData.nb`, 3-D mean-amplitude plots).
