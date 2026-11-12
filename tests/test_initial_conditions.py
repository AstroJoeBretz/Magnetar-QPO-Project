import numpy as np
import pytest

from magnetar_qpo.angular import G_H, power_per_l, projection_coefficients, reconstruct
from magnetar_qpo.energy import amplitude_for_energy, strain_energy, strain_energy_brute_force
from magnetar_qpo.initial_conditions import InitialCondition, angular_momentum

CASES = {
    "notebook": InitialCondition(0.7, profile="notebook"),
    "paper": InitialCondition(0.7),
    "axisym": InitialCondition(0.7, axisymmetric=True),
    "exterior": InitialCondition(1.3),
}


@pytest.mark.parametrize("name", list(CASES))
def test_divergence_free_and_zero_angular_momentum(name):
    ic = CASES[name]
    rng = np.random.default_rng(1)
    th, ph, r = rng.uniform(0.9, 2.2, 6), rng.uniform(2.5, 4.0, 6), rng.uniform(0.6, 0.8, 6)
    scale = np.abs(ic.u_phi(r, th, ph)).max()
    assert np.abs(ic.divergence_numeric(r, th, ph)).max() < 1e-7 * scale
    J = angular_momentum(ic, 1.0, 1 / 11, n=160)
    assert np.abs(J).max() < 1e-12 * ic.peak()


@pytest.mark.parametrize("name", list(CASES))
def test_analytic_profile_derivatives(name):
    ic, h = CASES[name], 1e-6
    t = np.linspace(0.5, 2.6, 7)
    f = ic.theta_functions
    assert np.allclose((f(t + h)["A"] - f(t - h)["A"]) / (2 * h), f(t)["dA"], atol=1e-6)
    assert np.allclose((f(t + h)["C"] - f(t - h)["C"]) / (2 * h), f(t)["dC"], atol=1e-6)
    p = np.linspace(2.5, 3.8, 5)
    g = ic.phi_functions
    assert np.allclose((g(p + h)["B"] - g(p - h)["B"]) / (2 * h), g(p)["E"], atol=1e-6)
    assert np.allclose((g(p + h)["E"] - g(p - h)["E"]) / (2 * h), g(p)["dE"], atol=1e-6)


def test_radial_profile_peaks_at_r0():
    ic = InitialCondition(0.7)
    r = np.linspace(0.3, 1.1, 40001)
    assert abs(r[np.argmax(ic.fr(r))] - 0.7) < 1e-4
    assert np.allclose(ic.dfr(r[::5000]), np.gradient(ic.fr(r), r)[::5000], atol=1e-4)


@pytest.mark.parametrize("name", ["notebook", "paper", "axisym", "exterior"])
def test_separable_strain_energy_matches_brute_force(name):
    ic = CASES[name]
    E = strain_energy(ic, 1 / 11)
    assert abs(E / strain_energy_brute_force(ic, 1 / 11, n=60) - 1) < 1e-4


def test_exterior_energy_is_weighted_by_mu2_and_scales_quadratically():
    ic = InitialCondition(2.0, sigma_r=1 / 6)  # entirely outside the star
    assert abs(strain_energy(ic, 1 / 11) * 11 / strain_energy(ic, 1.0) - 1) < 1e-9
    assert abs(strain_energy(ic, 1.0, amplitude=3.0) / strain_energy(ic, 1.0) - 9) < 1e-12
    a = amplitude_for_energy(1e46, 100.0, 4.2e47)
    assert abs(4.2e47 * a**2 * 100.0 / 1e46 - 1) < 1e-12


def test_angular_projection_reconstructs_field_and_has_no_l1_power():
    ic = InitialCondition(0.7)
    co = projection_coefficients(ic, 26)
    th, ph = np.array([1.3, 1.6, 2.0]), np.array([3.0, 3.6, 4.2])
    Vt, Vp = reconstruct(co, 26, th, ph)
    Tf, Pf = ic.theta_functions(th), ic.phi_functions(ph)
    assert np.allclose(Vp, Tf["A"] * Pf["B"], atol=1e-3 * np.abs(Vp).max())
    assert np.allclose(Vt, -Tf["C"] * Pf["E"], atol=1e-3 * np.abs(Vt).max())
    assert power_per_l(co, 3)[0] < 1e-12  # zero angular momentum -> nothing in l=1


def test_axisymmetric_has_only_m0_and_even_parity_content():
    co = projection_coefficients(CASES["axisym"], 10)
    assert all(abs(a) < 1e-12 for (l, m), a in co.items() if m != 0)
    G, _ = G_H(co, 2, np.array([1.0]), np.array([0.3]))
    G2, _ = G_H(co, 2, np.array([1.0]), np.array([2.1]))
    assert np.allclose(G, G2)  # phi-independent
