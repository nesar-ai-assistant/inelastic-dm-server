"""MCP server entry point for inelastic-dm-server (MCP SDK v2)."""
from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from tools.idm_tools import (
    describe_idm_tools,
    predict_recoil_spectrum,
    compute_relic_density_tool,
    check_all_constraints_tool,
    scan_mass_splitting_tool,
    predict_annual_modulation_tool,
    plot_recoil_spectrum_tool,
    plot_constraint_summary_tool,
)

mcp = MCPServer(
    name="inelastic-dm-server",
    description=(
        "Bridges particle physics (WIMP mass, mass splitting, cross section) "
        "with cosmological observables. Motivated by the LZ 248 keV event "
        "(September 2026) and the inelastic higgsino interpretation."
    ),
)


@mcp.tool()
def describe_tools() -> str:
    """Describe this server's capabilities, the physics, and the observational data it uses. Call this first."""
    import json
    return json.dumps(describe_idm_tools(), indent=2, default=str)


@mcp.tool()
def predict_recoil(
    m_chi_GeV: float = 1100.0,
    delta_keV: float = 350.0,
    sigma_n_cm2: float = 0.0,
    is_higgsino: bool = True,
) -> str:
    """Predict the differential recoil spectrum for (in)elastic DM on xenon.

    Parameters:
        m_chi_GeV: WIMP mass [GeV] (default: 1100 for thermal higgsino)
        delta_keV: Mass splitting [keV] (default: 350, set 0 for elastic)
        sigma_n_cm2: Per-nucleon cross section [cm^2] (0 = auto for higgsino)
        is_higgsino: If true, use Z-exchange couplings and auto sigma_n
    """
    import json
    sig = sigma_n_cm2 if sigma_n_cm2 > 0 else None
    result = predict_recoil_spectrum(m_chi_GeV, delta_keV, sig, is_higgsino)
    # Remove the large array data from JSON response (keep it compact)
    result.pop("spectrum", None)
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def compute_relic_density(
    m_chi_GeV: float = 1100.0,
    is_higgsino: bool = True,
    sigma_v_cm3_s: float = 0.0,
) -> str:
    """Compute thermal relic density and compare to Planck measurement.

    Parameters:
        m_chi_GeV: DM mass [GeV]
        is_higgsino: If true, use higgsino annihilation channels
        sigma_v_cm3_s: Annihilation cross section [cm^3/s] (0 = auto)
    """
    import json
    sv = sigma_v_cm3_s if sigma_v_cm3_s > 0 else None
    return json.dumps(compute_relic_density_tool(m_chi_GeV, is_higgsino, sv),
                      indent=2, default=str)


@mcp.tool()
def check_all_constraints(
    m_chi_GeV: float = 1100.0,
    delta_keV: float = 350.0,
    is_higgsino: bool = True,
) -> str:
    """Run all constraints simultaneously: relic density + LZ + Fermi-LAT + Planck CMB.

    Parameters:
        m_chi_GeV: DM mass [GeV]
        delta_keV: Mass splitting [keV]
        is_higgsino: If true, use higgsino model
    """
    import json
    return json.dumps(check_all_constraints_tool(m_chi_GeV, delta_keV,
                      is_higgsino=is_higgsino), indent=2, default=str)


@mcp.tool()
def scan_mass_splitting(
    m_chi_min: float = 500.0,
    m_chi_max: float = 2000.0,
    delta_min: float = 100.0,
    delta_max: float = 500.0,
    n_points: int = 15,
) -> str:
    """Scan the (m_chi, delta) parameter plane checking all constraints.

    Parameters:
        m_chi_min/max: WIMP mass range [GeV]
        delta_min/max: Mass splitting range [keV]
        n_points: Grid points per axis (n_points x n_points grid)
    """
    import json
    return json.dumps(scan_mass_splitting_tool(
        m_chi_min, m_chi_max, n_points,
        delta_min, delta_max, n_points,
    ), indent=2, default=str)


@mcp.tool()
def predict_annual_modulation(
    m_chi_GeV: float = 1100.0,
    delta_keV: float = 350.0,
    E_R_keV: float = 248.0,
) -> str:
    """Predict annual modulation signal for inelastic DM at a given recoil energy.

    Parameters:
        m_chi_GeV: WIMP mass [GeV]
        delta_keV: Mass splitting [keV]
        E_R_keV: Recoil energy [keV] (default: 248 for LZ event)
    """
    import json
    result = predict_annual_modulation_tool(m_chi_GeV, delta_keV, E_R_keV=E_R_keV)
    result.pop("daily_rates", None)
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def plot_recoil_spectrum(
    m_chi_GeV: float = 1100.0,
    delta_keV: float = 350.0,
) -> str:
    """Generate a recoil spectrum plot comparing inelastic vs elastic. Returns PNG path.

    Parameters:
        m_chi_GeV: WIMP mass [GeV]
        delta_keV: Mass splitting [keV]
    """
    import json
    return json.dumps(plot_recoil_spectrum_tool(m_chi_GeV, delta_keV),
                      indent=2, default=str)


@mcp.tool()
def plot_constraint_summary(scan_file: str = "output/scan_idm.npz") -> str:
    """Plot the combined constraint summary from a parameter space scan. Returns PNG path.

    Call scan_mass_splitting first to generate the scan data.
    """
    import json
    return json.dumps(plot_constraint_summary_tool(scan_file),
                      indent=2, default=str)


if __name__ == "__main__":
    mcp.run(transport="stdio")
