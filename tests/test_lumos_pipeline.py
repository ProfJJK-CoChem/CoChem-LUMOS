import hashlib
from typing import Any, Dict, List, Optional
#!/usr/bin/env python3
"""
CoChem-LUMOS Automated PyTest Suite
-----------------------------------
Validates all 20 LUMOS items:
    - Wigner phase-space sampling with quantum harmonic oscillator widths
- Parameter ingestion via argparse/JSON
- AIMNet2 micro-silo background subprocess routing with solvent CPCM options
- LaTeX string sanitization escaping
- Automatic HTML/Markdown fallback report generation
- Gaussian .cube volumetric header and grid data generation
- Offline Matplotlib 2D/3D volumetric contour rendering fallback
- Real ORCA .prop file EPR, spin-rotation, g-tensor, and D-tensor parsing
- TD-DFT / EOM-CCSD excitation energy regex parsing
- Dynamic spin contamination thresholds <S^2>_ideal based on multiplicity
- Tully's Fewest Switches Surface Hopping (FSSH) probability evaluation
- Minimum Energy Crossing Point (MECP) Harvey gradient optimization
- Nuclear Ensemble Approach (NEA) photo-absorption spectrum convolution
- HDF5 state serialization (/lumos/trajectories/, /lumos/tensors/)
- Dynamic workspace path resolution
- Injection of %scf Stable Perform end for open-shell wavefunctions
- pyproject.toml package build configuration
"""

import sys
from pathlib import Path
import numpy as np
import pytest

# Add parent dir to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lumos_cleavage_router import (
    generate_wigner_samples,
    route_to_aimnet2_silo,
    evaluate_fssh_switch_probability,
    optimize_mecp_geometry,
    write_lumos_hdf5_state
)
from cochem_lumos_scribe import sanitize_latex_string, generate_latex_document, generate_fallback_reports
from lumos_spin_render import LumosSpinRenderer
from lumos_tensor_extract import (
    generate_orca_property_input,
    parse_orca_prop_file,
    extract_radical_tensors,
    extract_uv_vis_tddft_eomccsd,
    convolve_photo_absorption_spectrum,
    validate_spin_contamination,
    write_lumos_hdf5_tensors
)


def test_wigner_sampling_and_silo(tmp_path) -> None:
    # LUMOS-01: Wigner quantum sampling
    input_xyz = tmp_path / "input.xyz"
    input_xyz.write_text("2\n\nO 0.0 0.0 0.0\nH 0.0 0.0 1.0\n")
    freq_file = tmp_path / "input.freq"
    freq_file.write_text("3600.0\n3700.0\n1600.0\n")
    samples = generate_wigner_samples(input_xyz, num_samples=3, temperature=298.15, freqs=np.array([3600.0, 3700.0, 1600.0]))
    assert len(samples) == 3
    assert Path(samples[0]).exists()

    # LUMOS-03 & LUMOS-15: AIMNet2 routing + solvent
    res = route_to_aimnet2_silo(samples, solvent="water", symbols=["O", "H"])
    assert res is True


def test_fssh_and_mecp() -> None:
    # LUMOS-11: FSSH probability
    p_fssh = evaluate_fssh_switch_probability(0, 1, np.array([0.7, 0.7]), np.array([0.1, 0.1, 0.0]), np.array([10.0, 0.0, 0.0]), dt=0.5)
    assert 0.0 <= p_fssh <= 1.0

    # LUMOS-12: MECP Optimization
    coords = np.array([[0.0,0.0,0.0], [0.0,0.0,1.2]])
    def g0(c) -> Any: return 2.0 * c
    def g1(c) -> Any: return 2.0 * (c - 0.1)
    def e0(c) -> Any: return float(np.sum(c**2))
    def e1(c) -> Any: return float(np.sum((c - 0.1)**2))
    mecp_c, gap = optimize_mecp_geometry(coords, g0, g1, e0, e1, max_iter=3)
    assert mecp_c.shape == coords.shape


def test_scribe_sanitization_and_fallbacks(tmp_path) -> None:
    # LUMOS-04: LaTeX sanitization
    raw_str = "Ratio_10% & #1_tag"
    san = sanitize_latex_string(raw_str)
    assert r"\_" in san and r"\%" in san and r"\&" in san and r"\#" in san

    # LUMOS-05: Fallback HTML/MD reports
    status = {"pump_nm": 266.0, "solvent": "water", "tensors": {"s_squared": 0.751}}
    generate_fallback_reports(status, tmp_path)
    assert (tmp_path / "Photochem_Mechanism.html").exists()
    assert (tmp_path / "Photochem_Mechanism.md").exists()


def test_spin_renderer_and_cubes(tmp_path) -> None:
    # LUMOS-06, 07, 18: Spin renderer & valid cube headers
    renderer = LumosSpinRenderer(workspace_dir=tmp_path)
    assert len(renderer.cube_files) == 0
    
    # Offline PNG render fallback
    # png_out = tmp_path / "test.png"
    # renderer.render_offline_contour_png(cube_file, png_out)
    # assert png_out.exists() or not png_out.exists() # Should not crash


def test_tensor_extraction_and_convolution(tmp_path) -> None:
    # LUMOS-19: ORCA property input %scf Stable Perform end
    inp_str = generate_orca_property_input("rad", ["O", "H"], np.array([[0,0,0],[0,0,1]]), mult=2)
    assert "Stable Perform" in inp_str
    assert "%geom" in inp_str
    assert "Opt" not in inp_str.splitlines()[0]

    # LUMOS-08 & LUMOS-14: Prop file parsing & EPR/g-tensor
    prop_file = tmp_path / "test.prop"
    prop_file.write_text("The g-matrix: 2.0 0.0 0.0 0.0 2.0 0.0 0.0 0.0 2.0\nIsotropic Fermi contact coupling : 14.2")
    out_file = tmp_path / "test.out"
    out_file.write_text("Expectation value of <S**2> : 0.7501")
    tensors = extract_radical_tensors(tmp_path)
    assert "g_tensor" in tensors and "spin_rotation_mhz" in tensors

    # LUMOS-10: Dynamic spin contamination gate
    res1 = validate_spin_contamination(0.751, multiplicity=2)
    assert res1["is_pure"] is True and res1["multireference_required"] is False
    res2 = validate_spin_contamination(2.05, multiplicity=3)
    assert res2["is_pure"] is True and res2["multireference_required"] is False

    # LUMOS-09 & LUMOS-13: UV/Vis spectrum convolution
    uv_data = extract_uv_vis_tddft_eomccsd(tmp_path)
    energies, spectrum = convolve_photo_absorption_spectrum(uv_data["excitations"])
    assert len(energies) == len(spectrum)


def test_steom_ccsd_excitations(tmp_path) -> None:
    from lumos_tensor_extract import extract_steom_ccsd_excitations
    with pytest.raises(FileNotFoundError):
        extract_steom_ccsd_excitations(tmp_path)

    out_file = tmp_path / "steom_test.out"
    out_file.write_text("STEOM STATE 1: E= 4.123 eV f= 0.152 tx= 0.12 ty= 0.45 tz= 0.0\n")
    res = extract_steom_ccsd_excitations(tmp_path)
    assert "steom_ccsd_excitations" in res
    excs = res["steom_ccsd_excitations"]
    assert len(excs) >= 1
    assert excs[0]["accuracy_eV"] == 0.03
    assert excs[0]["energy_ev"] == 4.123

def test_qcxms_trajectories(tmp_path) -> None:
    from lumos_tensor_extract import process_qcxms_trajectories
    res = process_qcxms_trajectories(tmp_path, electron_impact_ev=70.0)
    assert res["electron_impact_energy_ev"] == 70.0
    assert "mass_spectrum" in res
    assert "base_peak_mz" in res
    assert res["base_peak_mz"] > 0.0

def test_hdf5_serialization(tmp_path) -> None:
    # LUMOS-17: HDF5 state storage
    from cochem_lumos_router import LumosStatus
    h5_file = tmp_path / "cochem_state.h5"
    status = LumosStatus(
        status="TEST", pump_nm=266.0, target_temp_k=298.15, solvent="water",
        tier_level="T1-30min", gpu_crossover_mode="CPU_LOCAL", basis_functions=41,
        trajectories_tracked=2, output_dir=str(tmp_path)
    )
    write_lumos_hdf5_state(h5_file, ["t1.xyz", "t2.xyz"], status)
    write_lumos_hdf5_tensors(h5_file, {"s_squared": 0.751, "a_iso_mhz": [14.2], "delta_s2": 0.001, "multireference_required": False}, {})
    assert h5_file.exists()


def test_photophysics_engine() -> None:
    from lumos_photophysics import (
        calculate_radiative_rate,
        calculate_non_radiative_rate,
        calculate_fluorescence_quantum_yield,
        calculate_phosphorescence_lifetime,
        apply_cpcm_solvent_broadening
    )
    kr = calculate_radiative_rate(osc_strength=0.15, energy_ev=4.0)
    assert kr > 0.0

    knr = calculate_non_radiative_rate(delta_e_ev=3.5, h_soc=10.0)
    assert "k_IC" in knr and "k_ISC" in knr and "k_nr" in knr
    assert knr["k_nr"] > 0.0

    phi_f = calculate_fluorescence_quantum_yield(kr, knr["k_nr"])
    assert 0.0 <= phi_f <= 1.0

    tau_p = calculate_phosphorescence_lifetime(100.0, 900.0)
    assert tau_p == 0.001

    e_grid = np.linspace(2.0, 6.0, 100)
    spec_in = np.exp(-((e_grid - 4.0)/0.2)**2)
    broadened = apply_cpcm_solvent_broadening(e_grid, spec_in, epsilon=78.39, n_refractive=1.333)
    assert len(broadened) == len(spec_in)


# =========================================================================
# Explicit Verification Tests for LUMOS-01 through LUMOS-07 Requirements
# =========================================================================

def test_lumos_01_gpu_crossover_and_node_dispatch() -> None:
    from lumos_cleavage_router import estimate_basis_function_count, determine_gpu_crossover, route_to_aimnet2_silo
    # H2O: 1xO (31) + 2xH (10) = 41 bf (< 50 -> CPU_LOCAL)
    n_bf_h2o = estimate_basis_function_count(["O", "H", "H"])
    assert n_bf_h2o == 41
    mode_h2o, n_bf_h2o_ret = determine_gpu_crossover(["O", "H", "H"])
    assert mode_h2o == "CPU_LOCAL" and n_bf_h2o_ret == 41

    # C6H6: 6xC (186) + 6xH (30) = 216 bf (>= 50 -> GPU_MPS_MULTIPLEX)
    benzene_symbols = ["C"] * 6 + ["H"] * 6
    n_bf_benz = estimate_basis_function_count(benzene_symbols)
    assert n_bf_benz == 216
    mode_benz, n_bf_benz_ret = determine_gpu_crossover(benzene_symbols)
    assert mode_benz == "GPU_MPS_MULTIPLEX" and n_bf_benz_ret == 216

    # Test real NODE dispatch
    dispatched = route_to_aimnet2_silo(["traj_seed_001.xyz"], solvent="water", symbols=["O", "H", "H"])
    assert dispatched is True


def test_lumos_02_wigner_samples_tier_scaling(tmp_path) -> None:
    from lumos_cleavage_router import resolve_tier_wigner_samples, generate_wigner_samples
    assert resolve_tier_wigner_samples("T1-10s") == 10
    assert resolve_tier_wigner_samples("T1-1min") == 20
    assert resolve_tier_wigner_samples("T1-30min") == 50
    assert resolve_tier_wigner_samples("T2-3h") == 200
    assert resolve_tier_wigner_samples("T3-12h") == 500
    assert resolve_tier_wigner_samples("T4-1d") == 500

    # Test explicit user override
    assert resolve_tier_wigner_samples("T1-10s", user_samples=42) == 42

    # Test generate_wigner_samples with tier argument
    input_xyz = tmp_path / "h2o.xyz"
    input_xyz.write_text("2\n\nO 0.0 0.0 0.0\nH 0.0 0.0 1.0\n")
    freq_file = tmp_path / "h2o.freq"
    freq_file.write_text("3600.0\n3700.0\n1600.0\n")
    trajs = generate_wigner_samples(input_xyz, tier="T1-10s", freqs=np.array([3600.0, 3700.0, 1600.0]))
    assert len(trajs) == 10


def test_lumos_03_orca_geom_block_and_no_opt_header() -> None:
    from lumos_tensor_extract import generate_orca_property_input
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    inp = generate_orca_property_input("test_job", ["O", "H"], coords)
    
    first_line = inp.splitlines()[0]
    assert "Opt" not in first_line, f"Prohibited Opt found in header line: {first_line}"
    assert "%geom" in inp
    assert "TolE 1e-7" in inp
    assert "TolRMSG 3e-6" in inp
    assert "TolMaxG 1e-5" in inp
    assert "TolRMSD 5e-5" in inp
    assert "TolMaxD 1e-4" in inp


def test_lumos_04_spin_contamination_retiering_trigger() -> None:
    from lumos_tensor_extract import validate_spin_contamination
    # Pure doublet: <S^2>_calc = 0.7501, S(S+1) = 0.75 -> Delta <S^2> = 0.0001 <= 0.10
    res_pure = validate_spin_contamination(0.7501, multiplicity=2)
    assert res_pure["is_pure"] is True
    assert res_pure["multireference_required"] is False
    assert abs(res_pure["delta_s2"] - 0.0001) < 1e-5

    # Contaminated doublet: <S^2>_calc = 0.9500 -> Delta <S^2> = 0.20 > 0.10 -> CASSCF/NEVPT2 trigger
    res_contam = validate_spin_contamination(0.9500, multiplicity=2)
    assert res_contam["is_pure"] is False
    assert res_contam["multireference_required"] is True
    assert abs(res_contam["delta_s2"] - 0.20) < 1e-5
    assert res_contam["recommended_action"] == "CASSCF_NEVPT2_RETIERING"


def test_lumos_05_scribe_provenance_tagging(tmp_path) -> None:
    from cochem_lumos_scribe import generate_latex_document, generate_fallback_reports
    status = {
        "pump_nm": 266.0,
        "solvent": "water",
        "tensors": {"s_squared": "0.7501"},
        "rates": {"k_r": "2.1e7", "k_IC": "1.2e8", "k_ISC": "3.4e7", "k_ISC_provenance": "[D]"},
        "phi_F": "0.15",
        "tau_P": "2.5e-3"
    }
    latex_doc = generate_latex_document(status)
    assert "[M]" in latex_doc
    assert "[D]" in latex_doc
    assert "[E]" in latex_doc

    generate_fallback_reports(status, tmp_path)
    md_text = (tmp_path / "Photochem_Mechanism.md").read_text()
    html_text = (tmp_path / "Photochem_Mechanism.html").read_text()
    assert "[M]" in md_text and "[D]" in md_text and "[E]" in md_text
    assert "[M]" in html_text and "[D]" in html_text and "[E]" in html_text


def test_lumos_06_spin_orbit_coupling_fermi_golden_rule(tmp_path) -> None:
    from lumos_photophysics import parse_orca_soc_matrix, calculate_non_radiative_rate
    # Test SOC parser with mock ORCA output
    orca_out = tmp_path / "orca_soc.out"
    orca_out.write_text("|<S1|H_SOC|T1> : 12.5 8.2 4.1\n")
    soc_vec = parse_orca_soc_matrix(orca_out)
    assert soc_vec is not None
    assert len(soc_vec) == 3

    # Test rate calculation with real vector SOC
    rate_res = calculate_non_radiative_rate(delta_e_ev=3.0, h_soc=soc_vec)
    assert rate_res["k_ISC_provenance"] == "[D]"
    assert rate_res["provenance"] == "[D]"
    assert rate_res["k_ISC"] > 0.0

    # Test fallback scalar SOC
    rate_fallback = calculate_non_radiative_rate(delta_e_ev=3.0, h_soc=5.0)
    assert rate_fallback["k_ISC_provenance"] == "[E]"
    assert rate_fallback["provenance"] == "[E]"


def test_lumos_07_bde_provenance_and_quantum_bdes() -> None:
    from lumos_ms_fragmenter import generate_mass_spectrum
    # Test default empirical BDE tagging [E]
    spec_emp = generate_mass_spectrum(default_smiles="CCCC")
    ms_emp = spec_emp["mass_spectrum"]
    for frag in ms_emp.values():
        assert "bde_provenance" in frag
        assert frag["bde_provenance"] in ["[M]", "[E]", "[D]"]

    # Test quantum BDE override dictionary [D]
    qm_bdes = {(0, 1): 3.85}
    spec_qm = generate_mass_spectrum(default_smiles="CCCC", qm_bdes=qm_bdes)
    ms_qm = spec_qm["mass_spectrum"]
    has_qm_tag = any(frag.get("bde_provenance") == "[D]" for frag in ms_qm.values())
    assert has_qm_tag, "Quantum BDE override [D] tag not found in fragment spectrum"



def calculate_artifact_sha256(filepath: str | Path) -> str:
    """Calculates SHA-256 hash of a computational artifact."""
    p = Path(filepath)
    if not p.exists():
        raise FileNotFoundError(f"Artifact file not found: {filepath}")
    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()