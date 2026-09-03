"""Test cosmological constraints."""
import sys
sys.path.insert(0, '.')
from tools.cosmology import (
    relic_density, higgsino_relic_density, higgsino_sigma_v,
    higgsino_indirect, cmb_constraint, v_earth
)

print("=" * 70)
print("COSMOLOGY MODULE TESTS")
print("=" * 70)

# Test 1: Generic relic density
print("\n--- Test 1: Standard freeze-out ---")
# Canonical thermal relic: sigma_v = 3e-26 cm^3/s should give Omega ~ 0.1
result = relic_density(100.0, 3.0e-26)
print(f"m_chi=100 GeV, sigma_v=3e-26: Omega h^2 = {result['omega_h2']:.4f}")
print(f"x_f = {result['x_f']:.1f}, T_f = {result['T_f_GeV']:.2f} GeV")
print(f"Tension with Planck: {result['tension_sigma']:.1f} sigma")
ok = 0.05 < result['omega_h2'] < 0.25
print(f"In expected range: {'PASS' if ok else 'FAIL'}")

# Test 2: Higgsino relic density
print("\n--- Test 2: Higgsino relic density ---")
for mu in [500, 800, 1000, 1100, 1200, 1500, 2000]:
    r = higgsino_relic_density(mu)
    planck_match = "***MATCH***" if abs(r['omega_h2'] - 0.12) < 0.03 else ""
    print(f"  mu={mu:5d} GeV: Omega h^2 = {r['omega_h2']:.4f}, "
          f"sigma_v = {r['sigma_v_cm3_s']:.2e} cm^3/s {planck_match}")

# The thermal higgsino mass should be ~1.1 TeV
r_1100 = higgsino_relic_density(1100)
print(f"\nmu=1100 GeV: Omega h^2 = {r_1100['omega_h2']:.4f}")
print(f"Expected ~0.12: {'PASS' if 0.08 < r_1100['omega_h2'] < 0.16 else 'FAIL'}")

# Test 3: Indirect detection
print("\n--- Test 3: Indirect detection ---")
# Elastic higgsino at 1.1 TeV
ind_elastic = higgsino_indirect(1100.0, 0.0)
print(f"Elastic higgsino (1.1 TeV):")
print(f"  sigma_v_today = {ind_elastic['sigma_v_today_cm3_s']:.2e} cm^3/s")
print(f"  Fermi limit = {ind_elastic['fermi_limit_cm3_s']:.2e} cm^3/s")
print(f"  Excluded: {ind_elastic['excluded']}")
print(f"  Note: {ind_elastic['note']}")

# Inelastic higgsino at 1.1 TeV with delta = 350 keV
ind_inelastic = higgsino_indirect(1100.0, 350.0)
print(f"\nInelastic higgsino (1.1 TeV, delta=350 keV):")
print(f"  sigma_v_today = {ind_inelastic['sigma_v_today_cm3_s']:.2e} cm^3/s")
print(f"  Fermi limit = {ind_inelastic['fermi_limit_cm3_s']:.2e} cm^3/s")
print(f"  Excluded: {ind_inelastic['excluded']}")
print(f"  Note: {ind_inelastic['note']}")
print(f"  KEY: Inelastic suppresses indirect detection!")

# Test 4: CMB constraint
print("\n--- Test 4: CMB energy injection ---")
cmb = cmb_constraint(1100.0, ind_elastic['sigma_v_today_cm3_s'], "WW")
print(f"Elastic higgsino CMB: p_ann = {cmb['p_ann']:.2e}, limit = {cmb['p_ann_limit']:.2e}")
print(f"Excluded: {cmb['excluded']}")

cmb_inel = cmb_constraint(1100.0, ind_inelastic['sigma_v_today_cm3_s'], "WW")
print(f"Inelastic higgsino CMB: p_ann = {cmb_inel['p_ann']:.2e}")
print(f"Excluded: {cmb_inel['excluded']}")

# Test 5: Annual modulation
print("\n--- Test 5: Annual modulation ---")
for day in [1, 80, 153, 244, 335]:
    v = v_earth(day)
    month_names = {1: "Jan 1", 80: "Mar 21", 153: "Jun 2", 244: "Sep 1", 335: "Dec 1"}
    print(f"  {month_names.get(day, f'Day {day}'):8s}: v_E = {v:.1f} km/s")
print(f"Maximum modulation amplitude: {v_earth(153) - v_earth(335):.1f} km/s")

print("\n" + "=" * 70)
print("All cosmology tests complete.")
