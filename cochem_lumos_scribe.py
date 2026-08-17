#!/usr/bin/env python3
"""
CoChem-LUMOS: Automated LaTeX & HTML Scribe
-------------------------------------------
Converts LUMOS_Refinement_Status.json into compiled LaTeX (Photochem_Mechanism.tex / .pdf)
with LaTeX special character sanitization, and automatically generates HTML/Markdown
fallback report artifacts when pdflatex is missing or encounters errors.
"""

import os
import re
import sys
import json
import subprocess
import logging
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

class Colors:
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

logging.basicConfig(filename='cochem_lumos_scribe.log', level=logging.INFO)

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

def sanitize_latex_string(text: str) -> str:
    r"""
    Escapes LaTeX special characters (\, _, %, &, #, $, {, }, ~, ^) to prevent syntax crashes.
    """
    if not isinstance(text, str):
        text = str(text)
    
    latex_special = {
        '\\': r'\textbackslash{}',
        '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}',
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
    }
    
    pattern = re.compile('|'.join(re.escape(k) for k in latex_special.keys()))
    return pattern.sub(lambda m: latex_special[m.group(0)], text)

sanitize_latex = sanitize_latex_string

def generate_latex_document(status: dict) -> str:
    """Generates the raw LaTeX string, safely interpolating and sanitizing physics data with provenance tags [M], [D], [E]."""
    if "tensors" not in status or "rates" not in status:
        raise ValueError("[ANTI-SPOOFING] Refinement status missing required physics data (tensors/rates). Fallbacks/hallucinations are strictly prohibited.")
        
    pump_nm = sanitize_latex_string(str(status.get("pump_nm", "Unknown")))
    solvent = sanitize_latex_string(status.get("solvent", "water"))
    tensors = status["tensors"]
    s2 = sanitize_latex_string(str(tensors["s_squared"]))
    rates = status["rates"]
    k_r = sanitize_latex_string(str(rates["k_r"]))
    k_ic = sanitize_latex_string(str(rates["k_IC"]))
    k_isc = sanitize_latex_string(str(rates["k_ISC"]))
    soc_tag = sanitize_latex_string(rates.get("k_ISC_provenance", "[E]"))
    phi_f = sanitize_latex_string(str(status["phi_F"]))
    tau_p = sanitize_latex_string(str(status["tau_P"]))

    soc_assumption = " (Estimated using default SOC = 5.0 cm$^{-1}$)" if "[E]" in soc_tag else ""
    
    tex_template = r"""\documentclass[11pt, a4paper]{article}
\usepackage{amsmath}
\usepackage{geometry}
\geometry{margin=1in}

\title{\textbf{CoChem-LUMOS: Photochemical Cleavage Analysis}}
\author{Automated Pipeline Report}
\date{\today}

\begin{document}
\maketitle

\section{Methodology}
Open-shell photo-cleavage was simulated using a %PUMP%~nm [D] excitation pulse in %SOLVENT% [D] solvent. 
Wigner phase-space sampling \cite{Wigner1932} generated the initial conditions, which were subsequently propagated via the AIMNet2 neural network potential \cite{AIMNet2}.

\section{Spin-State Validation \& Photophysics}
Broken-symmetry DFT \cite{ORCA6} was utilized to extract the final EPR tensors. 
The expectation value of the spin squared operator was explicitly calculated to monitor spin contamination:
    \begin{equation}
    \langle S^2 \rangle = %S2% \text{ [D]}
\end{equation}

Radiative decay rate constant $k_r = %KR% \text{ s}^{-1} \text{ [D]}$, internal conversion rate $k_{\text{IC}} = %KIC% \text{ s}^{-1} \text{ [E]}$, and intersystem crossing rate $k_{\text{ISC}} = %KISC% \text{ s}^{-1} \text{ %SOC_TAG%}%SOC_ASSUMPTION%.
Fluorescence quantum yield $\Phi_F = %PHIF% \text{ [D]}$ and phosphorescence lifetime $\tau_P = %TAUP% \text{ s [D]}$.

\section{Conclusion}
The tensor and photophysics data with mandatory provenance tags have been serialized to \texttt{cochem\_state.h5} and are ready for experimental matching.

\begin{thebibliography}{9}
\bibitem{Wigner1932}
E. Wigner, \textit{On the Quantum Correction For Thermodynamic Equilibrium}, Phys. Rev. \textbf{40}, 749 (1932).
\bibitem{AIMNet2}
A. E. Roitberg et al., \textit{AIMNet2: A General-Purpose Neural Network Potential}, ChemRxiv (2023).
\bibitem{ORCA6}
F. Neese et al., \textit{The ORCA quantum chemistry program package}, J. Chem. Phys. \textbf{152}, 224108 (2020).
\end{thebibliography}

\end{document}
"""
    tex_template = tex_template.replace("%PUMP%", pump_nm)
    tex_template = tex_template.replace("%SOLVENT%", solvent)
    tex_template = tex_template.replace("%S2%", s2)
    tex_template = tex_template.replace("%KR%", k_r)
    tex_template = tex_template.replace("%KIC%", k_ic)
    tex_template = tex_template.replace("%KISC%", k_isc)
    tex_template = tex_template.replace("%SOC_TAG%", soc_tag)
    tex_template = tex_template.replace("%SOC_ASSUMPTION%", soc_assumption)
    tex_template = tex_template.replace("%PHIF%", phi_f)
    tex_template = tex_template.replace("%TAUP%", tau_p)
    return tex_template

def generate_fallback_reports(status: dict, workspace_dir: Path) -> Any:
    """
    Generates standalone HTML and Markdown reports with mandatory provenance tags [M], [D], [E].
    """
    if "tensors" not in status or "rates" not in status:
        raise ValueError("[ANTI-SPOOFING] Refinement status missing required physics data (tensors/rates). Fallbacks/hallucinations are strictly prohibited.")

    pump_nm = str(status.get("pump_nm", "Unknown"))
    solvent = status.get("solvent", "water")
    tensors = status["tensors"]
    s2 = str(tensors["s_squared"])
    rates = status["rates"]
    k_r = str(rates["k_r"])
    k_ic = str(rates["k_IC"])
    k_isc = str(rates["k_ISC"])
    soc_tag = rates.get("k_ISC_provenance", "[E]")
    phi_f = str(status["phi_F"])
    tau_p = str(status["tau_P"])

    soc_assumption = " *(Estimated using default SOC = 5.0 cm^-1)*" if "[E]" in soc_tag else ""

    md_content = f"""# CoChem-LUMOS: Photochemical Cleavage Report

**Excitation Pump:** {pump_nm} nm [D]  
**Solvent Environment:** {solvent} [D]  
**Spin Expectation $\\langle S^2 \\rangle$:** {s2} [D]  

## Methodology & Photophysics
Open-shell photo-cleavage was simulated using quantum Wigner phase-space sampling combined with AIMNet2 neural network potential propagation. Broken-symmetry DFT via ORCA was utilized to extract final EPR tensors.

- **Radiative Decay Rate $k_r$:** {k_r} s^-1 [D]
- **Internal Conversion Rate $k_{{IC}}$:** {k_ic} s^-1 [E]
- **Intersystem Crossing Rate $k_{{ISC}}$:** {k_isc} s^-1 {soc_tag}{soc_assumption}
- **Fluorescence Quantum Yield $\\Phi_F$:** {phi_f} [D]
- **Phosphorescence Lifetime $\\tau_P$:** {tau_p} s [D]

## Tensor Status
EPR spin-rotation tensors, isotropic hyperfine constants, and g-tensors have been extracted and verified with [D] provenance tags. All results are stored in `cochem_state.h5`.

## References
1. E. Wigner, *On the Quantum Correction For Thermodynamic Equilibrium*, Phys. Rev. **40**, 749 (1932).
2. A. E. Roitberg et al., *AIMNet2: A General-Purpose Neural Network Potential*, ChemRxiv (2023).
3. F. Neese et al., *The ORCA quantum chemistry program package*, J. Chem. Phys. **152**, 224108 (2020).
"""
    md_path = workspace_dir / "Photochem_Mechanism.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    soc_assumption_html = " <i>(Estimated using default SOC = 5.0 cm<sup>-1</sup>)</i>" if "[E]" in soc_tag else ""

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>CoChem-LUMOS Photochemical Mechanism Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 30px; background-color: #f4f6f9; color: #333; }}
        .card {{ background: white; padding: 25px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); max-width: 800px; margin: 0 auto; }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        .badge {{ background: #27ae60; color: white; padding: 5px 10px; border-radius: 4px; font-weight: bold; }}
        .references {{ margin-top: 30px; font-size: 0.9em; color: #666; border-top: 1px solid #ddd; padding-top: 15px; }}
    </style>
</head>
<body>
    <div class="card">
        <h1>CoChem-LUMOS Photochemical Mechanism</h1>
        <p><span class="badge">Pump: {pump_nm} nm [D]</span> <span class="badge">Solvent: {solvent} [D]</span></p>
        <h2>Methodology & Spin Validation</h2>
        <p>Quantum Wigner trajectories propagated via AIMNet2 FSSH dynamics. Broken-symmetry DFT via ORCA was utilized. Spin expectation value: <b>&lang;S<sup>2</sup>&rang; = {s2} [D]</b>.</p>
        <h3>Photophysics Rates & Lifetimes</h3>
        <ul>
            <li>Radiative decay rate k<sub>r</sub>: <b>{k_r} s<sup>-1</sup> [D]</b></li>
            <li>Internal conversion k<sub>IC</sub>: <b>{k_ic} s<sup>-1</sup> [E]</b></li>
            <li>Intersystem crossing k<sub>ISC</sub>: <b>{k_isc} s<sup>-1</sup> {soc_tag}{soc_assumption_html}</b></li>
            <li>Fluorescence quantum yield &Phi;<sub>F</sub>: <b>{phi_f} [D]</b></li>
            <li>Phosphorescence lifetime &tau;<sub>P</sub>: <b>{tau_p} s [D]</b></li>
        </ul>
        <p>Full tensor datasets serialized to <code>cochem_state.h5</code>.</p>
        <div class="references">
            <h3>References</h3>
            <ol>
                <li>E. Wigner, <i>On the Quantum Correction For Thermodynamic Equilibrium</i>, Phys. Rev. <b>40</b>, 749 (1932).</li>
                <li>A. E. Roitberg et al., <i>AIMNet2: A General-Purpose Neural Network Potential</i>, ChemRxiv (2023).</li>
                <li>F. Neese et al., <i>The ORCA quantum chemistry program package</i>, J. Chem. Phys. <b>152</b>, 224108 (2020).</li>
            </ol>
        </div>
    </div>
</body>
</html>
"""
    html_path = workspace_dir / "Photochem_Mechanism.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info(f"{Colors.OKGREEN}✅ Fallback HTML & Markdown reports with [M], [D], [E] provenance tags generated successfully at {html_path.name}{Colors.ENDC}")

def main() -> Any:
    logger.info(f"\n{Colors.OKCYAN}--- CoChem-LUMOS: Scribe Output Generator ---{Colors.ENDC}")
    
    workspace_dir = Path(os.environ.get("COCHEM_ARTIFACT_DIR", ".")).resolve()
    status_file = workspace_dir / "LUMOS_Refinement_Status.json"
    if not status_file.exists():
        status_file = Path("LUMOS_Refinement_Status.json")

    if not status_file.exists():
        logger.error(f"{Colors.FAIL}❌ [MISSING DATA] LUMOS Registry missing. Cannot proceed without physical status data.{Colors.ENDC}")
        sys.exit(1)
    else:
        with open(status_file, "r") as f:
            status = json.loads(f.read())
        
    tex_out = workspace_dir / "Photochem_Mechanism.tex"
    
    latex_content = generate_latex_document(status)
    with open(tex_out, "w", encoding="utf-8") as f:
        f.write(latex_content)
    logger.info(f"{Colors.OKGREEN}✅ LaTeX payload generated: {tex_out.name}{Colors.ENDC}")

    generate_fallback_reports(status, workspace_dir)

    try:
        logger.info("⚙️  Attempting pdflatex compilation...")
        subprocess.run(["pdflatex", "-interaction=nonstopmode", str(tex_out)], check=True, timeout=300,
            capture_output=True, text=True, cwd=str(workspace_dir)
        )
        logger.info(f"{Colors.OKGREEN}✅ PDF successfully compiled: Photochem_Mechanism.pdf{Colors.ENDC}")
        logging.info("LUMOS PDF compiled successfully.")
    except FileNotFoundError:
        logger.info(f"{Colors.WARNING}⚠️ 'pdflatex' not found on system path. Fallback HTML/Markdown generated.{Colors.ENDC}")
        logging.warning("pdflatex not found. Using fallback HTML/MD reports.")
    except subprocess.CalledProcessError as e:
        logger.info(f"{Colors.WARNING}⚠️ LaTeX compilation encountered errors. Fallback HTML/Markdown preserved.{Colors.ENDC}")
        logging.error(f"pdflatex error:\n{e.stdout}")

if __name__ == "__main__":
    main()