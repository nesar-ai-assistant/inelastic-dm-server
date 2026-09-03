# Inelastic Dark Matter MCP Server

**Q2C Bridge: Particle physics → Cosmological observables**

MCP server that connects WIMP particle physics parameters (mass, mass splitting, cross section) to cosmological observables (recoil spectra, relic density, indirect detection limits, CMB constraints). Motivated by the **LZ 248 keV nuclear recoil event** (September 2026) and the **inelastic higgsino** interpretation.

## Physics

The server implements the following calculations — all with real formulas from published papers, no fitting functions or placeholders:

| Calculation | Reference |
|---|---|
| Differential recoil rate (SI, inelastic) | Lewin & Smith (1996), Eq. 3.3-3.16 |
| Helm nuclear form factor | Engel (1991), Lewin & Smith Eq. 4.7-4.11 |
| Inelastic kinematics (v_min with mass splitting) | Fan & Reece, arXiv:2609.01504 |
| Thermal relic density (freeze-out) | Kolb & Turner (1990), Eq. 5.26-5.29 |
| Higgsino cross section (Z-exchange) | Fan & Reece, Eq. 8-9 |
| Fermi-LAT dSph indirect detection limits | Ackermann et al. (2015) |
| CMB energy injection constraint | Planck 2018, Slatyer (2016) |
| Annual modulation (SHM) | Freese, Frieman, Gould (1988) |

### Key Result

A **thermal higgsino** at μ ≈ 1.1 TeV with mass splitting δ ≈ 300-380 keV:
- ✅ Gives correct relic density (Ω h² ≈ 0.12)
- ✅ Predicts ~1-3 events in LZ extended NR search (consistent with observation)
- ✅ Evades Fermi-LAT indirect detection (coannihilation frozen at galactic velocities)
- ✅ Evades Planck CMB energy injection bounds

## MCP Tools (8)

| Tool | Description |
|---|---|
| `describe_tools` | Server capabilities and physics references |
| `predict_recoil` | Differential recoil spectrum for inelastic DM on xenon |
| `compute_relic_density` | Thermal freeze-out Ω h² vs Planck |
| `check_all_constraints` | All constraints simultaneously (relic + LZ + Fermi + CMB) |
| `scan_mass_splitting` | (m_χ, δ) parameter space scan |
| `predict_annual_modulation` | Seasonal rate variation for inelastic DM |
| `plot_recoil_spectrum` | Generate spectrum plot (PNG) |
| `plot_constraint_summary` | Combined exclusion plot from scan (PNG) |

## Setup

```bash
# Create venv with dependencies
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python numpy scipy matplotlib "mcp>=1.27" pydantic "cryptography<44"

# Test the server
python -m mcp_server
```

### Register with Hermes

```bash
hermes config set mcp_servers.inelastic_dm.command "$HOME/github-integration/inelastic-dm-server/run_server.sh"
hermes config set mcp_servers.inelastic_dm.timeout 120
hermes mcp test inelastic_dm
```

## Project Structure

```
tools/
  nuclear.py      — Recoil rates, form factors, inelastic kinematics (Lewin & Smith)
  cosmology.py    — Relic density, indirect detection, CMB constraints
  idm_tools.py    — MCP tool wrappers combining nuclear + cosmology
mcp_server/
  __init__.py     — MCPServer registration (8 tools)
  __main__.py     — Entry point
tests/
  test_benchmarks.py  — Physics benchmark tests
  test_cosmology.py   — Cosmology module tests
```

## Observational Data Used

- **LZ**: 248 ± 23 keV nuclear recoil event (arXiv:2609.02608, September 2026)
- **Planck**: Ω_DM h² = 0.1200 ± 0.0012
- **Fermi-LAT**: dSph 95% CL limits (Ackermann et al. 2015)
- **CMB**: p_ann < 3.2×10⁻²⁸ cm³ s⁻¹ GeV⁻¹ (Planck 2018)

## References

- Fan & Reece, "The LZ anomaly as inelastic dark matter", arXiv:2609.01504
- LZ Collaboration, "Search for new physics with an extended nuclear recoil ROI", arXiv:2609.02608
- LZ Collaboration, "Effective field theory and inelastic interpretations", arXiv:2609.02823
- Lewin & Smith, Astropart. Phys. 6 (1996) 87
- Kolb & Turner, *The Early Universe* (1990)
- Helm, Phys. Rev. 104 (1956) 1466
