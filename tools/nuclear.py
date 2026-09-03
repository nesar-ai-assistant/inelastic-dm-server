"""
Nuclear form factors and differential recoil rates.

ALL RATES use the Lewin & Smith (1996) R_0 normalization to avoid
unit conversion errors between halo integrals and event rates.

References:
- Helm, Phys. Rev. 104 (1956) 1466
- Lewin & Smith, Astropart. Phys. 6 (1996) 87, Eq. 3.3-3.16, 4.7-4.11
- Fan & Reece, arXiv:2609.01504, Eq. 1-9
- Engel, Phys. Lett. B 264 (1991) 114 (Helm parameterization)
"""

import numpy as np
from scipy.special import erf

# Physical constants
GEV_FM = 0.197327        # hbar*c [GeV*fm]
HBARC_CM = 1.9732698e-14 # hbar*c [GeV*cm]
G_F = 1.1663788e-5       # Fermi constant [GeV^-2]
M_PROTON = 0.938272088   # [GeV]
M_NEUTRON = 0.939565420  # [GeV]
AMU_GEV = 0.931494       # atomic mass unit [GeV]
SIN2_THETA_W = 0.23122
AVOGADRO = 6.02214076e23
C_KM_S = 2.99792458e5    # speed of light [km/s]
RHO_0 = 0.3              # local DM density [GeV/cm^3]

# Xenon isotope data (IUPAC)
XENON_ISOTOPES = {
    128: (54, 128 * AMU_GEV, 0.01910, 0.0),
    129: (54, 129 * AMU_GEV, 0.26401, 0.5),
    130: (54, 130 * AMU_GEV, 0.04071, 0.0),
    131: (54, 131 * AMU_GEV, 0.21232, 1.5),
    132: (54, 132 * AMU_GEV, 0.26909, 0.0),
    134: (54, 134 * AMU_GEV, 0.10436, 0.0),
    136: (54, 136 * AMU_GEV, 0.08857, 0.0),
}

# LZ parameters
LZ_EXPOSURE_TY = 2.84   # tonne-year
LZ_ER_MIN = 40.0        # keV, extended NR ROI
LZ_ER_MAX = 270.0       # keV


def reduced_mass(m1, m2):
    """Reduced mass [same units as inputs]."""
    return m1 * m2 / (m1 + m2)


def helm_form_factor_squared(E_R_keV, A):
    """Helm nuclear form factor squared.

    Lewin & Smith (1996) Eq. 4.7-4.11, Engel (1991):
        F(qr_n) = 3 * j_1(q*r_n) / (q*r_n) * exp(-(q*s)^2 / 2)

    Parameters: c = 1.23*A^(1/3) - 0.60 fm, a = 0.52 fm, s = 0.9 fm,
                r_n = sqrt(c^2 + 7/3*pi^2*a^2 - 5*s^2)
    """
    E_R = np.atleast_1d(np.float64(E_R_keV))
    m_N = A * AMU_GEV
    q_GeV = np.sqrt(2.0 * m_N * E_R * 1.0e-6)
    q_fm = q_GeV / GEV_FM

    c = 1.23 * A**(1.0/3.0) - 0.60
    s = 0.9
    r_n = np.sqrt(c**2 + 7.0/3.0 * np.pi**2 * 0.52**2 - 5.0 * s**2)
    qr = q_fm * r_n

    F2 = np.ones_like(q_fm)
    mask = qr > 1.0e-6
    if np.any(mask):
        j1 = np.sin(qr[mask]) / qr[mask]**2 - np.cos(qr[mask]) / qr[mask]
        F = 3.0 * j1 / qr[mask] * np.exp(-0.5 * (q_fm[mask] * s)**2)
        F2[mask] = F**2

    return float(F2[0]) if F2.size == 1 else F2


def v_min_inelastic(E_R_keV, m_chi_GeV, m_N_GeV, delta_keV=0.0):
    """Minimum WIMP velocity [km/s] for inelastic scattering.

    Fan & Reece arXiv:2609.01504 Eq. 1:
        v_min = (1/sqrt(2 m_N E_R)) * (m_N E_R / mu_N + delta)
    """
    E_R = np.atleast_1d(np.float64(E_R_keV))
    E_R_GeV = E_R * 1.0e-6
    delta_GeV = delta_keV * 1.0e-6
    mu_N = reduced_mass(m_chi_GeV, m_N_GeV)
    v_nat = (m_N_GeV * E_R_GeV / mu_N + delta_GeV) / np.sqrt(2.0 * m_N_GeV * E_R_GeV)
    v_km_s = v_nat * C_KM_S
    result = np.where(E_R_GeV > 0, v_km_s, np.inf)
    return float(result[0]) if result.size == 1 else result


def spectral_shape(v_min_km_s, v_0=220.0, v_E=232.0, v_esc=544.0):
    """Lewin & Smith spectral shape function T(v_min).

    This encodes the velocity integral in a form that directly gives
    the differential rate when multiplied by R_0/(E_0*r).

    Lewin & Smith (1996) Eq. 3.8-3.11:
        For v_min < v_esc - v_E:
            T = sqrt(pi)/(4*y)*[erf(x+y) - erf(x-y)] - exp(-z^2)
        For v_esc - v_E <= v_min < v_esc + v_E:
            T = sqrt(pi)/(4*y)*[erf(z) - erf(x-y)] - (z+y-x)/(2*y)*exp(-z^2)
        For v_min >= v_esc + v_E:
            T = 0

    where x = v_min/v_0, y = v_E/v_0, z = v_esc/v_0.
    """
    vm = np.atleast_1d(np.float64(v_min_km_s))
    x = vm / v_0
    y = v_E / v_0
    z = v_esc / v_0
    ez2 = np.exp(-z**2)

    T = np.zeros_like(vm)

    mask1 = vm < (v_esc - v_E)
    if np.any(mask1):
        x1 = x[mask1]
        T[mask1] = (np.sqrt(np.pi) / (4.0 * y)
                    * (erf(x1 + y) - erf(x1 - y)) - ez2)

    mask2 = (vm >= (v_esc - v_E)) & (vm < (v_esc + v_E))
    if np.any(mask2):
        x2 = x[mask2]
        T[mask2] = (np.sqrt(np.pi) / (4.0 * y)
                    * (erf(z) - erf(x2 - y))
                    - (z + y - x2) / (2.0 * y) * ez2)

    T = np.maximum(T, 0.0)
    return float(T[0]) if T.size == 1 else T


def R_0_total(m_chi_GeV, sigma_0_cm2, A=131):
    """Total event rate per kg per day (no form factor, no threshold).

    Lewin & Smith (1996) Eq. 3.3:
        R_0 = (2/sqrt(pi)) * N_T * (rho_0/m_chi) * sigma_0 * v_0

    Parameters
    ----------
    m_chi_GeV : float
        WIMP mass [GeV].
    sigma_0_cm2 : float
        DM-nucleus cross section at zero momentum transfer [cm^2].
    A : int
        Target nucleus mass number.

    Returns
    -------
    R0 : float
        Total rate [events/kg/day].
    """
    N_T = AVOGADRO * 1.0e3 / A
    n_chi = RHO_0 / m_chi_GeV   # number density [cm^-3]
    v_0_cms = 220.0 * 1.0e5     # cm/s
    return (2.0 / np.sqrt(np.pi)) * N_T * n_chi * sigma_0_cm2 * v_0_cms * 86400.0


def sigma_0_SI(sigma_n_cm2, m_chi_GeV, A, Z, f_p_over_f_n=1.0):
    """Zero-momentum-transfer DM-nucleus cross section for SI scattering.

    sigma_0 = sigma_n * (mu_N/mu_n)^2 * [Z*f_p/f_n + (A-Z)]^2

    For isospin-symmetric (f_p = f_n): coherent factor = A^2.
    For Z-exchange (higgsino): f_p/f_n = -(1 - 4*sin^2 theta_W) ~ -0.075.
    """
    m_N = A * AMU_GEV
    mu_N = reduced_mass(m_chi_GeV, m_N)
    mu_n = reduced_mass(m_chi_GeV, M_NEUTRON)
    coherent = (Z * f_p_over_f_n + (A - Z))**2
    return sigma_n_cm2 * (mu_N / mu_n)**2 * coherent


def higgsino_sigma_n(m_chi_GeV):
    """Higgsino-nucleon cross section via Z exchange [cm^2].

    sigma_n = G_F^2 * mu_n^2 / (2*pi)

    Fan & Reece arXiv:2609.01504 Eq. 8-9.
    ~7.4e-39 cm^2 for m_chi >> m_n.
    """
    mu_n = reduced_mass(m_chi_GeV, M_NEUTRON)
    sigma_nat = G_F**2 * mu_n**2 / (2.0 * np.pi)  # GeV^-2
    return sigma_nat * HBARC_CM**2                   # cm^2


def dRdER_inelastic(E_R_keV, m_chi_GeV, delta_keV=0.0, sigma_n_cm2=None,
                     A=131, Z=54, v_E_km_s=232.0, is_higgsino=True):
    """Differential recoil rate for (in)elastic SI scattering [evts/keV/kg/day].

    Uses R_0 normalization (Lewin & Smith 1996) to avoid unit conversion errors.

    dR/dER = (k_0/k_1) * R_0/(E_0*r) * F^2(E_R) * T(v_min(E_R))

    For inelastic scattering, v_min includes the mass splitting delta.
    """
    E_R = np.atleast_1d(np.float64(E_R_keV))
    m_N = A * AMU_GEV

    # Cross section
    if sigma_n_cm2 is None:
        sigma_n_cm2 = higgsino_sigma_n(m_chi_GeV)

    if is_higgsino:
        f_ratio = -(1.0 - 4.0 * SIN2_THETA_W)
    else:
        f_ratio = 1.0  # isospin-symmetric

    sig_0 = sigma_0_SI(sigma_n_cm2, m_chi_GeV, A, Z, f_ratio)

    # R_0 and kinematic parameters
    R0 = R_0_total(m_chi_GeV, sig_0, A)
    v_0_nat = 220.0 / C_KM_S
    E_0_keV = 0.5 * m_chi_GeV * 1e6 * v_0_nat**2
    r_kin = 4.0 * m_chi_GeV * m_N / (m_chi_GeV + m_N)**2
    E_0r = E_0_keV * r_kin

    # Truncation correction
    z = 544.0 / 220.0
    N_esc = erf(z) - 2.0 * z * np.exp(-z**2) / np.sqrt(np.pi)
    k0_k1 = 1.0 / N_esc

    # v_min (inelastic)
    vm = v_min_inelastic(E_R, m_chi_GeV, m_N, delta_keV)

    # Spectral shape and form factor
    T = spectral_shape(vm, v_E=v_E_km_s)
    F2 = helm_form_factor_squared(E_R, A)

    rate = k0_k1 * R0 / E_0r * F2 * T
    rate = np.atleast_1d(rate)
    return float(rate[0]) if rate.size == 1 else rate


def dRdER_xenon(E_R_keV, m_chi_GeV, delta_keV=0.0, sigma_n_cm2=None,
                v_E_km_s=232.0, is_higgsino=True):
    """Differential rate summed over natural xenon isotopes [evts/keV/kg/day]."""
    E_R = np.atleast_1d(np.float64(E_R_keV))
    total = np.zeros_like(E_R)
    for A, (Z, m_iso, frac, J) in XENON_ISOTOPES.items():
        rate = dRdER_inelastic(E_R, m_chi_GeV, delta_keV, sigma_n_cm2,
                                A=A, Z=Z, v_E_km_s=v_E_km_s,
                                is_higgsino=is_higgsino)
        total += frac * np.atleast_1d(rate)
    return float(total[0]) if total.size == 1 else total


def lz_efficiency(E_R_keV):
    """Approximate LZ NR detection efficiency.

    Piecewise linear model based on extended NR analysis (arXiv:2609.02823).
    NOTE: Approximate. Use official LZ efficiency for precision work.
    """
    E = np.atleast_1d(np.float64(E_R_keV))
    eff = np.zeros_like(E)
    mask1 = (E >= 5.0) & (E < 40.0)
    eff[mask1] = 0.85 * (E[mask1] - 5.0) / 35.0
    mask2 = (E >= 40.0) & (E < 250.0)
    eff[mask2] = 0.85
    mask3 = (E >= 250.0) & (E < 280.0)
    eff[mask3] = 0.85 * (280.0 - E[mask3]) / 30.0
    return float(eff[0]) if eff.size == 1 else eff


def expected_events_lz(m_chi_GeV, delta_keV, sigma_n_cm2=None,
                       E_min=40.0, E_max=270.0, n_points=500,
                       v_E_km_s=232.0, is_higgsino=True):
    """Expected signal events in LZ extended NR search."""
    exposure_kg_day = LZ_EXPOSURE_TY * 1000.0 * 365.25
    E_R_arr = np.linspace(E_min, E_max, n_points)
    eff = lz_efficiency(E_R_arr)
    rate = dRdER_xenon(E_R_arr, m_chi_GeV, delta_keV, sigma_n_cm2,
                       v_E_km_s=v_E_km_s, is_higgsino=is_higgsino)
    from scipy.integrate import trapezoid
    return exposure_kg_day * trapezoid(rate * eff, E_R_arr)


def E_R_peak(m_chi_GeV, delta_keV, A=131):
    """Recoil energy where v_min is minimized (spectrum peak).

    Fan & Reece Eq. 2: E_R_peak = delta * mu_N / m_N.
    """
    m_N = A * AMU_GEV
    mu_N = reduced_mass(m_chi_GeV, m_N)
    return delta_keV * mu_N / m_N
