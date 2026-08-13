# **CoChem-LUMOS: UV-Vis and Excited State Static Engine**

**PI/Developer**: Dr. Joshua John Klaassen
**ORCiD**: [https://orcid.org/0009-0007-1506-4401](https://orcid.org/0009-0007-1506-4401)
**GitHub Organization**: [https://github.com/ProfJJK-CoChem](https://github.com/ProfJJK-CoChem)

> **Important**: CoChem has recently migrated to the **Valeev Stack (MPQC, F12)** for enhanced electron correlation accuracy and reduced computational latency `[E]`.

Please refer to the authoritative [CoChem User Manual](https://github.com/ProfJJK-CoChem/CoChem-BASE/blob/main/CoChem_User_Manual.md) and [Method Matrix](https://github.com/ProfJJK-CoChem/CoChem-BASE/blob/main/Method_Matrix.md) for full execution instructions and basis set provenances.

## **Overview**

**CoChem-LUMOS** is the UV-Vis and Excited State Static engine of the extended CoChem suite. It executes advanced TD-DFT analysis utilizing the Tamm-Dancoff Approximation (TDA) to circumvent triplet instabilities `[M]`. 

Core capabilities include:
- Deploying STEOM-CCSD for high-accuracy vertical excitation energy benchmarks on complex molecular systems.
- Automatically generating state-specific Natural Transition Orbitals (NTOs) to enforce a rigorous hole-electron pairing visualization model.
- Implementing strict root-following during `$S_1$` geometrical relaxations and automatically flagging Conical Intersections if the `$S_1 - S_0$` gap collapses below critical thresholds (typically < 0.1 eV `[M]`).

## **Setup and Installation**

1. Clone the repository:
   ```bash
   git clone https://github.com/ProfJJK-CoChem/CoChem-LUMOS.git
   cd CoChem-LUMOS
   ```
2. Ensure the base computational backend (MPQC) is securely installed and accessible in the system path.

## **Getting Started**

Launch the excited state workflows by passing your molecular geometries to the core orchestrator:
```bash
python lumos_engine.py --input geometry.xyz --states 5
```

---
