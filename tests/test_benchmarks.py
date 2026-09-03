"""Test the corrected rate formula against known benchmarks."""
import sys
sys.path.insert(0, '.')

from tools.nuclear import (
    dRdER_inelastic, dRdER_xenon, expected_events_lz,
    helm_form_factor_squared, v_min_inelastic, spectral_shape,
    R_0_total, sigma_0_SI, higgsino_sigma_n, E_R_peak, reduced_mass,
    AMU_GEV
)
import numpy as np

print("=" * 70)
print("BENCHMARK TESTS FOR CORRECTED RATE FORMULA")
print("=" * 70)

# Test 1: Elastic SI, m_chi=100 GeV, sigma_n=1e-45 cm^2
print("\n--- Test 1: Elastic SI benchmark ---")
m_chi = 100.0
sigma_n = 1e-45
A, Z = 131, 54

sig0 = sigma_0_SI(sigma_n, m_chi, A, Z, f_p_over_f_n=1.0)
R0 = R_0_total(m_chi, sig0, A)
print(f"sigma_0 = {sig0:.3e} cm^2")
print(f"R_0 = {R0:.4e} evts/kg/day")
print(f"Expected R_0 ~ 1.8e-3 (standard benchmark): {'PASS' if 1e-3 < R0 < 3e-3 else 'FAIL'}")

# Differential rate at several energies
for E_R in [0.1, 10, 30, 50]:
    rate = dRdER_inelastic(E_R, m_chi, 0.0, sigma_n, A, Z, is_higgsino=False)
    print(f"  dR/dER({E_R:4.0f} keV) = {rate:.3e} evts/keV/kg/day")

# The rate should decrease with E_R (exponential-like)
r10 = dRdER_inelastic(10.0, m_chi, 0.0, sigma_n, A, Z, is_higgsino=False)
r50 = dRdER_inelastic(50.0, m_chi, 0.0, sigma_n, A, Z, is_higgsino=False)
print(f"Rate ratio dR(10)/dR(50) = {r10/r50:.1f} (should be >> 1)")

# Test 2: Higgsino cross section
print("\n--- Test 2: Higgsino cross section ---")
sig_hig = higgsino_sigma_n(1000.0)
print(f"sigma_n(higgsino, 1 TeV) = {sig_hig:.2e} cm^2")
print(f"Expected ~7.4e-39: {'PASS' if 6e-39 < sig_hig < 9e-39 else 'FAIL'}")

# Test 3: Inelastic kinematics
print("\n--- Test 3: Inelastic kinematics ---")
m_chi_h = 1100.0
delta = 350.0  # keV
m_N_xe = 131 * AMU_GEV
E_peak = E_R_peak(m_chi_h, delta)
vm_peak = v_min_inelastic(E_peak, m_chi_h, m_N_xe, delta)
vm_248 = v_min_inelastic(248.0, m_chi_h, m_N_xe, delta)
print(f"E_R_peak(1.1 TeV, 350 keV) = {E_peak:.0f} keV")
print(f"v_min at peak = {vm_peak:.0f} km/s")
print(f"v_min at 248 keV = {vm_248:.0f} km/s")
print(f"v_esc + v_E = 776 km/s")
print(f"248 keV event accessible: {vm_248 < 776}")

# What delta gives E_peak = 248 keV?
# E_peak = delta * mu_N / m_N => delta = E_peak * m_N / mu_N
mu_Nh = reduced_mass(m_chi_h, m_N_xe)
delta_for_248 = 248.0 * m_N_xe / mu_Nh
print(f"delta for E_peak=248 keV: {delta_for_248:.0f} keV")

# Test 4: Inelastic rate for higgsino at LZ event energy
print("\n--- Test 4: Higgsino inelastic rate ---")
for delta_test in [200, 250, 300, 350]:
    rate = dRdER_inelastic(248.0, m_chi_h, delta_test, is_higgsino=True)
    # Also compute expected events
    N_ev = expected_events_lz(m_chi_h, delta_test, is_higgsino=True)
    print(f"  delta={delta_test:3d} keV: dR/dER(248)={rate:.3e}, N_LZ={N_ev:.3f}")

# Test 5: Form factor
print("\n--- Test 5: Helm form factor ---")
for E in [0.1, 10, 50, 100, 200, 300]:
    F2 = helm_form_factor_squared(E, 131)
    print(f"  F^2({E:4.0f} keV, Xe-131) = {F2:.4f}")
# F^2 should be ~1 at low E, decrease, and show diffraction minima

print("\n" + "=" * 70)
print("All benchmarks complete.")
