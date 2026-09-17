"""export_dispatch_pack: the client-side HPC handoff contract.

The pack must be stageable by any hep-genesis client that receives it:
sanitized relative paths, engine size caps, and a manifest that names the
kernel entry point and the node-side pip requirements.
"""

import importlib
import json
import sys

# Import mcp_server.dispatch directly without triggering mcp_server.__init__
# (which pulls in the MCP SDK that may not be installed in the test env).
_spec = importlib.util.spec_from_file_location(
    "mcp_server.dispatch",
    str(__import__("pathlib").Path(__file__).resolve().parents[1] / "mcp_server" / "dispatch.py"),
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["mcp_server.dispatch"] = _mod
_spec.loader.exec_module(_mod)

_PACK_MAX_BYTES = _mod._PACK_MAX_BYTES
_PACK_MAX_FILES = _mod._PACK_MAX_FILES
export_dispatch_pack = _mod.export_dispatch_pack


def test_pack_shape_and_caps():
    pack = export_dispatch_pack()["dispatch_pack"]
    assert pack["name"] == "tools"
    assert pack["server"] == "inelastic-dm-server"
    assert 0 < pack["file_count"] <= _PACK_MAX_FILES
    assert 0 < pack["bytes"] <= _PACK_MAX_BYTES
    assert pack["file_count"] == len(pack["files"])
    # Serializable as one JSON tool result.
    json.dumps(pack)


def test_paths_are_safe_and_rooted():
    files = export_dispatch_pack()["dispatch_pack"]["files"]
    for rel in files:
        assert rel.startswith("tools/"), rel
        assert not rel.startswith("/"), rel
        parts = rel.split("/")
        assert all(p and p != ".." and not p.startswith(".") for p in parts), rel
        assert "__pycache__" not in rel, rel
    # The core tool modules that dispatch uses must ship.
    assert "tools/__init__.py" in files
    assert "tools/idm_tools.py" in files
    assert "tools/nuclear.py" in files
    assert "tools/cosmology.py" in files


def test_manifest_matches_kernels():
    kernels = export_dispatch_pack()["dispatch_pack"]["kernels"]
    # All expected kernel entries exist
    assert "recoil_spectrum" in kernels
    assert "relic_density" in kernels
    assert "constraints" in kernels
    assert "scan" in kernels
    assert "annual_modulation" in kernels
    # Check function paths match the actual tools/ layout
    assert kernels["recoil_spectrum"]["function"] == "idm_tools.predict_recoil_spectrum"
    assert kernels["relic_density"]["function"] == "idm_tools.compute_relic_density_tool"
    assert kernels["constraints"]["function"] == "idm_tools.check_all_constraints_tool"
    assert kernels["scan"]["function"] == "idm_tools.scan_mass_splitting_tool"
    assert kernels["annual_modulation"]["function"] == "idm_tools.predict_annual_modulation_tool"
    # All kernels require numpy + scipy
    for name, kernel in kernels.items():
        assert "numpy" in kernel["base_pip_deps"], f"{name} missing numpy"
        assert "scipy" in kernel["base_pip_deps"], f"{name} missing scipy"
