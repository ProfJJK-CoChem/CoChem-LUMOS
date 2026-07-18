#!/usr/bin/env python3
"""
CoChem-LUMOS: Automated LaTeX Scribe
Converts the LUMOS_Refinement_Status.json into a compiled Photochem_Mechanism.tex 
document via subprocess, handing it off to the final reporting tier.
"""

import os
import json
import subprocess
import logging
from pathlib import Path

class Colors:
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

logging.basicConfig(filename='cochem_lumos_scribe.log', level=logging.INFO)

def generate_latex_document(status: dict) -> str:
    """Generates the raw LaTeX string, safely interpolating physics data."""
    pump_nm = status.get("pump_nm", "Unknown")
    tensors = status.get("tensors", {})
    s2 = tensors.get("s_squared", "N/A")
    
    # Use raw string r""" to prevent Python from parsing \n or \t
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
Open-shell photo-cleavage was simulated using a %PUMP% nm excitation pulse. 
Wigner phase-space sampling generated the initial conditions, which were subsequently propagated via the AIMNet2 neural network potential.

\section{Spin-State Validation}
Broken-symmetry DFT was utilized to extract the final EPR tensors. 
The expectation value of the spin squared operator was explicitly calculated to monitor spin contamination:
\begin{equation}
    \langle S^2 \rangle = %S2%
\end{equation}

\section{Conclusion}
The tensor data has been serialized to \texttt{lumos\_tensors.npz} and is ready for experimental matching.

\end{document}
"""
    # Safely inject the variables
    tex_template = tex_template.replace("%PUMP%", str(pump_nm))
    tex_template = tex_template.replace("%S2%", str(s2))
    return tex_template

def main():
    print(f"\n{Colors.OKCYAN}--- CoChem-LUMOS: Scribe Output Generator ---{Colors.ENDC}")
    
    status_file = Path("LUMOS_Refinement_Status.json")
    if not status_file.exists():
        print(f"{Colors.FAIL}❌ LUMOS Registry missing. Cannot generate report.{Colors.ENDC}")
        return
        
    with open(status_file, "r") as f:
        status = json.load(f)
        
    tex_out = Path("Photochem_Mechanism.tex")
    
    try:
        latex_content = generate_latex_document(status)
        with open(tex_out, "w") as f:
            f.write(latex_content)
        print(f"{Colors.OKGREEN}✅ LaTeX payload generated: {tex_out.name}{Colors.ENDC}")
        
        # Attempt compilation
        print("⚙️  Attempting pdflatex compilation...")
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", str(tex_out)],
            capture_output=True, text=True, check=True
        )
        print(f"{Colors.OKGREEN}✅ PDF successfully compiled: Photochem_Mechanism.pdf{Colors.ENDC}")
        logging.info("LUMOS PDF compiled successfully.")
        
    except FileNotFoundError:
        print(f"{Colors.WARNING}⚠️ 'pdflatex' not found on system path. Compilation skipped.{Colors.ENDC}")
        print(f"   The LaTeX source is preserved at {tex_out.absolute()} for manual compilation.")
        logging.warning("pdflatex not found. Handing off raw .tex file.")
    except subprocess.CalledProcessError as e:
        print(f"{Colors.FAIL}❌ LaTeX compilation encountered errors (see logs).{Colors.ENDC}")
        logging.error(f"pdflatex error:\n{e.stdout}")

if __name__ == "__main__":
    main()