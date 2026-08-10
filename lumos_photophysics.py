#!/usr/bin/env python3
"""
CoChem-LUMOS: Photophysics & Lifetime Simulation Engine
-------------------------------------------------------
Implements non-radiative decay rates (k_IC, k_ISC via Energy Gap Law), radiative decay rates (k_r),
fluorescence quantum yields (Phi_F = k_r / (k_r + k_nr)), phosphorescence lifetimes (tau_P),
CPCM solvent dielectric broadening, and serialization into cochem_state.h5.
"""

import re
import math
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union, List
import numpy as np

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    h5py = None
    H5PY_AVAILABLE = False


def parse_orca_soc_matrix(out_file: Path) -> Optional[np.ndarray]:
    """
    Parses real Spin-Orbit Coupling (SOC) matrix elements H_SOC (in cm^-1) from ORCA output files.
    Extracts H_SOC matrix elements between S1 and T1 states.
    """
    if not out_file.exists():
        return None
    
    try:
        with open(out_file, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except Exception:
        return None

    soc_matches = re.findall(r"(?:\|<S1\|H_SOC\|T1>\|?|SOC\s+matrix)\s*:\s*([-\d\.\+eE]+)\s+([-\d\.\+eE]+)\s+([-\d\.\+eE]+)", content, re.IGNORECASE)
    if not soc_matches:
        soc_matches = re.findall(r"SOC\s+matrix\s+element\s+.*?([-\d\.\+eE]+)\s+cm\^-1", content, re.IGNORECASE)
        if soc_matches:
            try:
                vals = [float(x) for x in soc_matches[:3]]
                return np.array(vals, dtype=np.float64)
            except Exception:
                pass
    else:
        try:
            vals = [float(x) for x in soc_matches[0]]
            return np.array(vals, dtype=np.float64)
        except Exception:
            pass

    return None


def calculate_radiative_rate(osc_strength: float, energy_ev: float) -> float:
    """
    Calculates radiative decay rate constant k_r (s^-1) from oscillator strength f and transition energy E (eV):
    k_r = (2 * e^2 * omega^2 / (3 * m_e * c^3 * eps_0)) * f
        ~ 4.341e7 * f * (E_eV^2)  [in s^-1] [D]
    """
    if osc_strength <= 0 or energy_ev <= 0:
        return 0.0
    k_r = 4.341e7 * osc_strength * (energy_ev ** 2)
    return float(k_r)


def calculate_non_radiative_rate(delta_e_ev: float, h_soc: Union[float, np.ndarray, List[float]] = 5.0, hw_max_ev: float = 0.18) -> Dict[str, Any]:
    """
    Calculates non-radiative rate constants k_IC (Internal Conversion) and k_ISC (Intersystem Crossing)
    using Fermi's Golden Rule and Energy Gap Law with real or estimated SOC matrix elements.
    """
    if delta_e_ev <= 0:
        return {
            "k_IC": 1e12, "k_IC_provenance": "[E]",
            "k_ISC": 1e11, "k_ISC_provenance": "[E]",
            "k_nr": 1e12, "k_nr_provenance": "[E]",
            "h_soc_cm1": 5.0, "provenance": "[E]"
        }
    if hw_max_ev <= 0.0:
        return {
            "k_IC": 0.0, "k_IC_provenance": "[E]",
            "k_ISC": 0.0, "k_ISC_provenance": "[E]",
            "k_nr": 0.0, "k_nr_provenance": "[E]",
            "h_soc_cm1": 0.0, "provenance": "[E]"
        }

    if isinstance(h_soc, (list, tuple, np.ndarray)):
        h_soc_arr = np.asarray(h_soc, dtype=np.float64)
        if h_soc_arr.size == 3:
            h_soc_cm1 = float(np.sqrt(np.mean(np.abs(h_soc_arr)**2)))
        else:
            h_soc_cm1 = float(np.linalg.norm(h_soc_arr))
        soc_provenance = "[D]"
    else:
        h_soc_cm1 = float(h_soc)
        soc_provenance = "[E]"

    # 1. Internal Conversion rate k_IC (Energy Gap Law [E])
    gamma = 1.5
    exponent = gamma * (delta_e_ev / hw_max_ev)
    k_IC = 1e11 * math.exp(-min(exponent, 50.0))

    # 2. Intersystem Crossing rate k_ISC (Fermi's Golden Rule via |H_SOC|^2)
    soc_factor = (h_soc_cm1 / 5.0) ** 2
    delta_e_st_ev = max(0.05, delta_e_ev * 0.2) # S1-T1 energy gap
    k_ISC = 1e8 * soc_factor * math.exp(-min(delta_e_st_ev / 0.1, 50.0))

    k_nr = k_IC + k_ISC
    provenance = "[D]" if soc_provenance == "[D]" else "[E]"

    return {
        "k_IC": float(k_IC),
        "k_IC_provenance": "[E]",
        "k_ISC": float(k_ISC),
        "k_ISC_provenance": soc_provenance,
        "k_nr": float(k_nr),
        "k_nr_provenance": provenance,
        "h_soc_cm1": h_soc_cm1,
        "provenance": provenance
    }


def calculate_fluorescence_quantum_yield(k_r: float, k_nr: float) -> float:
    """
    Computes fluorescence quantum yield Phi_F [D]:
    Phi_F = k_r / (k_r + k_nr)
    """
    total_rate = k_r + k_nr
    if total_rate <= 0:
        return 0.0
    phi_f = k_r / total_rate
    return float(min(1.0, max(0.0, phi_f)))


def calculate_phosphorescence_lifetime(k_r_triplet: float, k_nr_triplet: float) -> float:
    """
    Computes phosphorescence lifetime tau_P (seconds) [D]:
    tau_P = 1 / (k_r^P + k_nr^T)
    """
    total_rate = k_r_triplet + k_nr_triplet
    if total_rate <= 0:
        return 1.0 # 1 second default
    return float(1.0 / total_rate)


def apply_cpcm_solvent_broadening(spectrum_energies: np.ndarray, spectrum_intensities: np.ndarray, 
                                  epsilon: float = 78.39, n_refractive: float = 1.333, 
                                  temperature_k: float = 298.15) -> np.ndarray:
    """
    Applies CPCM implicit solvent dielectric broadening to cross-section spectrum:
    sigma_solvent = sqrt(sigma_0^2 + lambda_solv * k_B * T)
    where lambda_solv ~ (1/n^2 - 1/epsilon)
    """
    kb_ev = 8.617333262145e-5 # eV / K
    kb_t = kb_ev * temperature_k
    
    # Solvent dielectric factor f_solv
    if n_refractive <= 0.0 or epsilon <= 0.0:
        f_solv = 0.0
    else:
        f_solv = max(0.0, (1.0 / (n_refractive**2)) - (1.0 / epsilon))
    sigma_0 = 0.05 # eV intrinsic width
    lambda_solv = 0.20 * f_solv # eV reorganization energy
    sigma_total = math.sqrt(sigma_0**2 + lambda_solv * kb_t)
    
    # Convolve spectrum with Gaussian kernel of width sigma_total
    broadened = np.zeros_like(spectrum_intensities)
    for i, e in enumerate(spectrum_energies):
        gaussian = (1.0 / (math.sqrt(2.0 * math.pi) * sigma_total)) * np.exp(-0.5 * ((spectrum_energies - e) / sigma_total)**2)
        broadened += spectrum_intensities[i] * gaussian
        
    return broadened


def write_lumos_hdf5_photophysics(h5_path: Path, photophysics_data: Dict[str, Any]):
    """
    Serializes photophysics rate constants, quantum yields, and solvent properties into cochem_state.h5
    under /lumos/rates/ and /lumos/excitations/ with mandatory provenance tags [M], [D], [E].
    """
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    if not H5PY_AVAILABLE:
        logging.warning("h5py not available for photophysics serialization.")
        return

    with h5py.File(h5_path, "a") as f:
        grp = f.require_group("lumos/rates")
        grp.attrs["k_r_provenance"] = "[D]"
        grp.attrs["k_IC_provenance"] = "[E]"
        grp.attrs["k_ISC_provenance"] = photophysics_data.get("k_ISC_provenance", "[E]")
        grp.attrs["phi_F_provenance"] = "[D]"
        grp.attrs["tau_P_provenance"] = "[D]"
        
        for k, v in photophysics_data.items():
            if isinstance(v, (int, float, str)):
                grp.attrs[k] = v
            elif isinstance(v, dict):
                subgrp = grp.require_group(k)
                for sk, sv in v.items():
                    if isinstance(sv, (int, float, str)):
                        subgrp.attrs[sk] = sv
    logging.info(f"Serialized photophysics rates to {h5_path.name} (/lumos/rates/).")
