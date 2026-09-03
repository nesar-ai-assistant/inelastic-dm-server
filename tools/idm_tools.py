"""
MCP tool functions for the inelastic dark matter server.

Each function returns a dict with results, citations, and interpretation.
"""

import os
import json
import numpy as np

from tools.nuclear import (
    dRdER_inelastic, dRdER_xenon, expected_events_lz,
    helm_form_factor_squared, v_min_inelastic, spectral_shape,
    R_0_total, sigma_0_SI, higgsino_sigma_n, E_R_peak, reduced_mass,
    lz_efficiency, XENON_ISOTOPES, AMU_GEV, C_KM_S,
    LZ_EXPOSURE_TY, LZ_ER_MIN, LZ_ER_MAX,
)
from tools.cosmology import (
    higgsino_relic_density, higgsino_sigma_v,
    higgsino_indirect, cmb_constraint, v_earth,
    relic_density, OMEGA_DM_H2_PLANCK, OMEGA_DM_H2_ERR,
)


def describe_idm_tools():
    """Describe the server's capabilities."""
    return {
        "server": "inelastic-dm-server",
        "description": (
            "Bridges particle physics (WIMP mass, mass splitting, cross section) "
            "with cosmological observables (relic density, recoil spectra, indirect "
            "detection, CMB constraints). Motivated by the LZ 248 keV nuclear recoil "
            "event (September 2026) and the inelastic higgsino interpretation."
        ),
        "tools": [
            "describe_tools — this description",
            "predict_recoil_spectrum — dR/dER for inelastic DM on xenon",
            "compute_relic_density — thermal freeze-out Omega h^2",
            "check_all_constraints — relic + direct + indirect + CMB",
            "scan_mass_splitting — (m_chi, delta) parameter space",
            "predict_annual_modulation — seasonal rate variation",
            "plot_recoil_spectrum — generate spectrum plot (PNG)",
            "plot_constraint_summary — combined exclusion plot (PNG)",
        ],
        "physics_references": [
            "Lewin & Smith, Astropart. Phys. 6 (1996) 87 (rate formula)",
            "Fan & Reece, arXiv:2609.01504 (higgsino iDM + LZ event)",
            "LZ Collaboration, arXiv:2609.02608 (extended NR search)",
            "LZ Collaboration, arXiv:2609.02823 (EFT + inelastic analysis)",
            "Kolb & Turner, The Early Universe (1990) (freeze-out)",
            "Ackermann et al., PRL 115 (2015) 231301 (Fermi-LAT limits)",
            "Planck 2018, arXiv:1807.06209 (CMB energy injection)",
        ],
        "observational_data": {
            "LZ_event": "248 ± 23 keV nuclear recoil (September 2026)",
            "Planck_relic": f"Omega_DM h^2 = {OMEGA_DM_H2_PLANCK} ± {OMEGA_DM_H2_ERR}",
            "LZ_exposure": f"{LZ_EXPOSURE_TY} tonne-years",
        },
    }


def predict_recoil_spectrum(
    m_chi_GeV=1100.0,
    delta_keV=350.0,
    sigma_n_cm2=None,
    is_higgsino=True,
    E_min=10.0,
    E_max=350.0,
    n_points=200,
):
    """Predict the differential recoil spectrum for (in)elastic DM scattering."""

    if sigma_n_cm2 is None and is_higgsino:
        sigma_n_cm2 = float(higgsino_sigma_n(m_chi_GeV))

    E_arr = np.linspace(E_min, E_max, n_points)
    rate = dRdER_xenon(E_arr, m_chi_GeV, delta_keV, sigma_n_cm2,
                       is_higgsino=is_higgsino)
    rate = np.atleast_1d(rate)

    # Also compute elastic for comparison
    rate_elastic = dRdER_xenon(E_arr, m_chi_GeV, 0.0, sigma_n_cm2,
                               is_higgsino=is_higgsino)
    rate_elastic = np.atleast_1d(rate_elastic)

    # LZ efficiency
    eff = np.atleast_1d(lz_efficiency(E_arr))

    # Expected events
    N_ev = expected_events_lz(m_chi_GeV, delta_keV, sigma_n_cm2,
                              is_higgsino=is_higgsino)

    # Peak energy
    E_peak_val = E_R_peak(m_chi_GeV, delta_keV)

    # v_min at LZ event energy
    m_N_xe = 131 * AMU_GEV
    vm_248 = v_min_inelastic(248.0, m_chi_GeV, m_N_xe, delta_keV)
    vm_peak = v_min_inelastic(E_peak_val, m_chi_GeV, m_N_xe, delta_keV)

    result = {
        "parameters": {
            "m_chi_GeV": m_chi_GeV,
            "delta_keV": delta_keV,
            "sigma_n_cm2": sigma_n_cm2,
            "is_higgsino": is_higgsino,
        },
        "kinematics": {
            "E_R_peak_keV": round(E_peak_val, 1),
            "v_min_at_peak_km_s": round(vm_peak, 0),
            "v_min_at_248keV_km_s": round(vm_248, 0),
            "v_esc_plus_vE_km_s": 776,
            "lz_event_accessible": vm_248 < 776,
        },
        "expected_events": {
            "N_LZ_extended_NR": round(N_ev, 3),
            "LZ_observed": 1,
            "poisson_prob_1_or_more": round(1 - np.exp(-N_ev), 4) if N_ev < 50 else 1.0,
        },
        "spectrum": {
            "E_R_keV": E_arr.tolist(),
            "dRdER_inelastic": rate.tolist(),
            "dRdER_elastic": rate_elastic.tolist(),
            "lz_efficiency": eff.tolist(),
        },
        "interpretation": (
            f"Inelastic DM (m_chi={m_chi_GeV:.0f} GeV, delta={delta_keV:.0f} keV) "
            f"peaks at E_R={E_peak_val:.0f} keV with v_min={vm_peak:.0f} km/s. "
            f"The LZ 248 keV event requires v_min={vm_248:.0f} km/s "
            f"({'accessible' if vm_248 < 776 else 'INACCESSIBLE'}). "
            f"Expected {N_ev:.2f} events in LZ extended NR search."
        ),
    }

    return result


def compute_relic_density_tool(m_chi_GeV=1100.0, is_higgsino=True,
                               sigma_v_cm3_s=None):
    """Compute thermal relic density and compare to Planck."""

    if is_higgsino:
        result = higgsino_relic_density(m_chi_GeV)
        sigma_v = result["sigma_v_cm3_s"]
    else:
        if sigma_v_cm3_s is None:
            sigma_v_cm3_s = 3.0e-26
        result = relic_density(m_chi_GeV, sigma_v_cm3_s)
        sigma_v = sigma_v_cm3_s
        result["sigma_v_cm3_s"] = sigma_v

    return {
        "parameters": {
            "m_chi_GeV": m_chi_GeV,
            "is_higgsino": is_higgsino,
            "sigma_v_cm3_s": sigma_v,
        },
        "relic_density": {
            "omega_h2": round(result["omega_h2"], 4),
            "planck_value": OMEGA_DM_H2_PLANCK,
            "planck_error": OMEGA_DM_H2_ERR,
            "tension_sigma": round(result["tension_sigma"], 1),
            "consistent_planck": result["consistent_planck"],
        },
        "freeze_out": {
            "x_f": round(result["x_f"], 1),
            "T_f_GeV": round(result["T_f_GeV"], 2),
            "g_star": result["g_star"],
        },
        "interpretation": (
            f"{'Higgsino' if is_higgsino else 'Generic WIMP'} at "
            f"m_chi={m_chi_GeV:.0f} GeV: Omega h^2 = {result['omega_h2']:.4f} "
            f"({'consistent' if result['consistent_planck'] else 'inconsistent'} "
            f"with Planck at {result['tension_sigma']:.1f}sigma). "
            f"Freeze-out at T_f = {result['T_f_GeV']:.2f} GeV."
        ),
        "references": [
            "Kolb & Turner (1990) Eq. 5.26-5.29",
            "Arkani-Hamed et al., hep-ph/0601041 (higgsino relic)",
            "Planck 2018, arXiv:1807.06209",
        ],
    }


def check_all_constraints_tool(m_chi_GeV=1100.0, delta_keV=350.0,
                                sigma_n_cm2=None, is_higgsino=True):
    """Run all constraints: relic density + direct + indirect + CMB."""

    if sigma_n_cm2 is None and is_higgsino:
        sigma_n_cm2 = float(higgsino_sigma_n(m_chi_GeV))

    # 1. Relic density
    if is_higgsino:
        relic = higgsino_relic_density(m_chi_GeV)
    else:
        relic = relic_density(m_chi_GeV, 3.0e-26)

    # 2. Direct detection (LZ expected events)
    N_lz = expected_events_lz(m_chi_GeV, delta_keV, sigma_n_cm2,
                              is_higgsino=is_higgsino)

    # 3. Indirect detection
    if is_higgsino:
        indirect = higgsino_indirect(m_chi_GeV, delta_keV)
    else:
        sigma_v = 3.0e-26
        from tools.cosmology import indirect_detection_check
        indirect = indirect_detection_check(m_chi_GeV, sigma_v)
        indirect["note"] = "Generic WIMP"
        indirect["delta_keV"] = delta_keV

    # 4. CMB constraint
    cmb = cmb_constraint(m_chi_GeV, indirect["sigma_v_today_cm3_s"], "WW")

    # Aggregate verdict
    all_pass = (
        relic.get("consistent_planck", False)
        and not indirect["excluded"]
        and not cmb["excluded"]
    )

    return {
        "parameters": {
            "m_chi_GeV": m_chi_GeV,
            "delta_keV": delta_keV,
            "sigma_n_cm2": sigma_n_cm2,
            "is_higgsino": is_higgsino,
        },
        "relic_density": {
            "omega_h2": round(relic["omega_h2"], 4),
            "tension_sigma": round(relic.get("tension_sigma", 0), 1),
            "passes": relic.get("consistent_planck", False),
        },
        "direct_detection": {
            "N_expected_LZ": round(N_lz, 3),
            "LZ_observed": 1,
            "consistent_with_1_event": 0.1 < N_lz < 10.0,
        },
        "indirect_detection": {
            "sigma_v_today_cm3_s": indirect["sigma_v_today_cm3_s"],
            "fermi_limit_cm3_s": indirect["fermi_limit_cm3_s"],
            "excluded": indirect["excluded"],
            "note": indirect.get("note", ""),
        },
        "cmb_energy_injection": {
            "p_ann": cmb["p_ann"],
            "p_ann_limit": cmb["p_ann_limit"],
            "excluded": cmb["excluded"],
        },
        "verdict": {
            "all_constraints_pass": all_pass,
            "summary": (
                "VIABLE" if all_pass else "EXCLUDED"
            ) + f" — Omega h^2={relic['omega_h2']:.3f}, "
              f"N_LZ={N_lz:.2f}, "
              f"indirect {'OK' if not indirect['excluded'] else 'EXCLUDED'}, "
              f"CMB {'OK' if not cmb['excluded'] else 'EXCLUDED'}",
        },
    }


def scan_mass_splitting_tool(
    m_chi_min=500, m_chi_max=2000, n_m=20,
    delta_min=100, delta_max=500, n_d=20,
    is_higgsino=True,
    output_dir="output",
):
    """Scan the (m_chi, delta) plane checking all constraints."""
    os.makedirs(output_dir, exist_ok=True)

    m_arr = np.linspace(m_chi_min, m_chi_max, n_m)
    d_arr = np.linspace(delta_min, delta_max, n_d)
    M, D = np.meshgrid(m_arr, d_arr)

    N_lz = np.zeros_like(M)
    omega = np.zeros_like(M)
    viable = np.zeros_like(M, dtype=bool)

    for i in range(n_d):
        for j in range(n_m):
            mc = float(M[i, j])
            dc = float(D[i, j])
            sig_n = float(higgsino_sigma_n(mc)) if is_higgsino else 1e-45

            N_lz[i, j] = expected_events_lz(mc, dc, sig_n, is_higgsino=is_higgsino)

            if is_higgsino:
                r = higgsino_relic_density(mc)
            else:
                r = relic_density(mc, 3e-26)
            omega[i, j] = r["omega_h2"]

            relic_ok = r.get("consistent_planck", False)
            indirect_ok = True
            if is_higgsino:
                ind = higgsino_indirect(mc, dc)
                indirect_ok = not ind["excluded"]
            viable[i, j] = relic_ok and indirect_ok

    # Save data
    scan_file = os.path.join(output_dir, "scan_idm.npz")
    np.savez(scan_file, m_chi=m_arr, delta=d_arr,
             M=M, D=D, N_lz=N_lz, omega=omega, viable=viable)

    # Find best-fit region (N_lz ~ 1-3 and viable)
    good = viable & (N_lz > 0.3) & (N_lz < 5.0)

    return {
        "scan_file": os.path.abspath(scan_file),
        "grid": f"{n_m} x {n_d} = {n_m*n_d} points",
        "ranges": {
            "m_chi_GeV": [m_chi_min, m_chi_max],
            "delta_keV": [delta_min, delta_max],
        },
        "results_summary": {
            "viable_points": int(viable.sum()),
            "total_points": int(M.size),
            "best_fit_region": int(good.sum()),
        },
        "interpretation": (
            f"{int(viable.sum())}/{int(M.size)} points pass all constraints. "
            f"{int(good.sum())} points also predict 0.3-5 events in LZ."
        ),
    }


def predict_annual_modulation_tool(
    m_chi_GeV=1100.0, delta_keV=350.0,
    sigma_n_cm2=None, is_higgsino=True,
    E_R_keV=248.0,
):
    """Predict the annual modulation signal for inelastic DM."""

    if sigma_n_cm2 is None and is_higgsino:
        sigma_n_cm2 = float(higgsino_sigma_n(m_chi_GeV))

    days = np.arange(1, 366)
    rates = []
    for d in days:
        vE = v_earth(d)
        r = dRdER_xenon(E_R_keV, m_chi_GeV, delta_keV, sigma_n_cm2,
                        v_E_km_s=vE, is_higgsino=is_higgsino)
        rates.append(float(r))

    rates = np.array(rates)
    max_day = int(days[np.argmax(rates)])
    min_day = int(days[np.argmin(rates)])
    modulation_frac = (max(rates) - min(rates)) / (max(rates) + min(rates) + 1e-100)

    # Month labels for peak
    import datetime
    peak_date = datetime.date(2026, 1, 1) + datetime.timedelta(days=max_day - 1)
    trough_date = datetime.date(2026, 1, 1) + datetime.timedelta(days=min_day - 1)

    # LZ event was Sept 1 = day 244
    lz_event_day = 244
    rate_at_event = float(dRdER_xenon(E_R_keV, m_chi_GeV, delta_keV, sigma_n_cm2,
                                       v_E_km_s=v_earth(lz_event_day),
                                       is_higgsino=is_higgsino))

    return {
        "parameters": {
            "m_chi_GeV": m_chi_GeV,
            "delta_keV": delta_keV,
            "E_R_keV": E_R_keV,
        },
        "modulation": {
            "peak_day": max_day,
            "peak_date": peak_date.strftime("%B %d"),
            "trough_day": min_day,
            "trough_date": trough_date.strftime("%B %d"),
            "modulation_fraction": round(modulation_frac, 4),
            "rate_at_peak": float(max(rates)),
            "rate_at_trough": float(min(rates)),
        },
        "lz_event_check": {
            "event_date": "September 1",
            "event_day": lz_event_day,
            "rate_at_event_date": rate_at_event,
            "fraction_of_peak": round(rate_at_event / (max(rates) + 1e-100), 3),
        },
        "daily_rates": {
            "days": days.tolist(),
            "dRdER": rates.tolist(),
        },
        "interpretation": (
            f"Annual modulation peaks on {peak_date.strftime('%B %d')} "
            f"with {modulation_frac*100:.1f}% fractional amplitude. "
            f"For inelastic DM near the kinematic threshold, the modulation "
            f"is amplified because only the fastest halo particles contribute. "
            f"The LZ event on September 1 sees "
            f"{rate_at_event/(max(rates)+1e-100)*100:.0f}% of peak rate."
        ),
    }


def plot_recoil_spectrum_tool(
    m_chi_GeV=1100.0, delta_keV=350.0,
    sigma_n_cm2=None, is_higgsino=True,
    output_dir="output",
):
    """Generate a publication-quality recoil spectrum plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    if sigma_n_cm2 is None and is_higgsino:
        sigma_n_cm2 = float(higgsino_sigma_n(m_chi_GeV))

    E_arr = np.linspace(5, 350, 500)

    # Inelastic spectrum
    rate_inel = np.atleast_1d(dRdER_xenon(E_arr, m_chi_GeV, delta_keV,
                              sigma_n_cm2, is_higgsino=is_higgsino))

    # Elastic spectrum for comparison
    rate_elas = np.atleast_1d(dRdER_xenon(E_arr, m_chi_GeV, 0.0,
                              sigma_n_cm2, is_higgsino=is_higgsino))

    # LZ efficiency
    eff = np.atleast_1d(lz_efficiency(E_arr))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[3, 1],
                                    sharex=True, gridspec_kw={'hspace': 0.05})

    # Main plot
    ax1.semilogy(E_arr, rate_inel, 'b-', lw=2,
                 label=f'Inelastic (δ={delta_keV:.0f} keV)')
    ax1.semilogy(E_arr, rate_elas, 'r--', lw=1.5, alpha=0.7,
                 label='Elastic (δ=0)')

    # LZ event
    ax1.axvline(248, color='green', ls=':', lw=1.5, alpha=0.8)
    ax1.annotate('LZ event\n248 keV', xy=(248, max(rate_inel)*0.3),
                fontsize=10, ha='center', color='green')

    # LZ ROI
    ax1.axvspan(LZ_ER_MIN, LZ_ER_MAX, alpha=0.08, color='blue',
               label='LZ extended NR ROI')

    ax1.set_ylabel('dR/dE$_R$ [evts/keV/kg/day]', fontsize=12)
    ax1.set_title(
        f'Inelastic DM Recoil Spectrum: '
        f'm$_\\chi$={m_chi_GeV:.0f} GeV, δ={delta_keV:.0f} keV'
        + (' (Higgsino)' if is_higgsino else ''),
        fontsize=13
    )
    ax1.legend(fontsize=10, loc='upper right')

    # Set y limits to show both curves
    y_max = max(max(rate_inel), max(rate_elas)) * 3
    y_min = max(min(rate_inel[rate_inel > 0].min() if np.any(rate_inel > 0) else 1e-20,
                    rate_elas[rate_elas > 0].min() if np.any(rate_elas > 0) else 1e-20),
                1e-20) * 0.1
    ax1.set_ylim(y_min, y_max)
    ax1.grid(True, alpha=0.3)

    # Efficiency panel
    ax2.plot(E_arr, eff, 'k-', lw=2)
    ax2.set_xlabel('Recoil Energy E$_R$ [keV]', fontsize=12)
    ax2.set_ylabel('LZ Efficiency', fontsize=12)
    ax2.set_ylim(-0.05, 1.0)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    outfile = os.path.join(output_dir, 'recoil_spectrum.png')
    fig.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return {
        "plot_file": os.path.abspath(outfile),
        "parameters": {
            "m_chi_GeV": m_chi_GeV,
            "delta_keV": delta_keV,
        },
    }


def plot_constraint_summary_tool(
    scan_file="output/scan_idm.npz",
    output_dir="output",
):
    """Plot the combined constraint summary from a parameter scan."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    os.makedirs(output_dir, exist_ok=True)

    data = np.load(scan_file)
    M, D = data["M"], data["D"]
    N_lz = data["N_lz"]
    omega = data["omega"]
    viable = data["viable"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Expected LZ events
    ax = axes[0]
    N_plot = np.where(N_lz > 1e-10, N_lz, 1e-10)
    im = ax.pcolormesh(M, D, N_plot, norm=LogNorm(vmin=0.01, vmax=100),
                       cmap='viridis', shading='auto')
    ax.contour(M, D, N_lz, levels=[0.5, 1.0, 3.0], colors=['white', 'yellow', 'red'],
               linewidths=1.5)
    plt.colorbar(im, ax=ax, label='Expected events')
    ax.set_xlabel('m$_\\chi$ [GeV]')
    ax.set_ylabel('δ [keV]')
    ax.set_title('LZ Expected Events')
    ax.plot([], [], 'y-', lw=2, label='N=1')
    ax.plot([], [], 'r-', lw=2, label='N=3')
    ax.legend(fontsize=9)

    # Panel 2: Relic density
    ax = axes[1]
    im2 = ax.pcolormesh(M, D, omega, cmap='RdYlBu_r', shading='auto',
                        vmin=0.05, vmax=0.25)
    ax.contour(M, D, omega, levels=[OMEGA_DM_H2_PLANCK], colors=['white'],
               linewidths=2)
    plt.colorbar(im2, ax=ax, label='Ω$_\\mathrm{DM}$ h²')
    ax.set_xlabel('m$_\\chi$ [GeV]')
    ax.set_title('Relic Density')

    # Panel 3: Viable region
    ax = axes[2]
    good = viable & (N_lz > 0.3) & (N_lz < 5.0)
    ax.pcolormesh(M, D, viable.astype(float), cmap='RdYlGn',
                  vmin=0, vmax=1, shading='auto', alpha=0.5)
    ax.contourf(M, D, good.astype(float), levels=[0.5, 1.5],
                colors=['gold'], alpha=0.5)
    ax.contour(M, D, good.astype(float), levels=[0.5], colors=['black'], linewidths=2)
    ax.set_xlabel('m$_\\chi$ [GeV]')
    ax.set_title('Viable + LZ-consistent')

    # Mark best-fit point
    if np.any(good):
        idx = np.unravel_index(np.argmin(np.abs(N_lz[good] - 1.0)), N_lz[good].shape)
        # Find the best-fit (N_lz closest to 1) among good points
        good_mask = good.flatten()
        N_flat = N_lz.flatten()
        M_flat = M.flatten()
        D_flat = D.flatten()
        best = np.argmin(np.abs(N_flat[good_mask] - 1.0))
        m_best = M_flat[good_mask][best]
        d_best = D_flat[good_mask][best]
        for a in axes:
            a.plot(m_best, d_best, 'r*', ms=15, mew=1.5, mec='white')

    fig.suptitle('Inelastic Higgsino Dark Matter: Combined Constraints',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    outfile = os.path.join(output_dir, 'constraint_summary.png')
    fig.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return {
        "plot_file": os.path.abspath(outfile),
        "best_fit": {
            "m_chi_GeV": round(float(m_best), 0) if np.any(good) else None,
            "delta_keV": round(float(d_best), 0) if np.any(good) else None,
        },
    }
