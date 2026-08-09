# CoChem-LUMOS: Architectural Changes (2026-08-07)

## 1. Tamm-Dancoff & STEOM-CCSD
**Target File:** `lumos_core/excited_states.py`
**Required Architectural Change:**
- Default TD-DFT MUST employ the Tamm-Dancoff Approximation (TDA) to prevent triplet instabilities.
- For Time-Tiers 8-10, LUMOS must execute STEOM-CCSD to guarantee $\sim 0.03$ eV accuracy for vertical excitations.

## 2. NTO & Solvatochromism Tracking
**Target File:** `lumos_core/analysis.py`
**Required Architectural Change:**
- Natural Transition Orbitals (NTOs) must be calculated universally to enforce strict hole-electron pair visualization.
- State-specific implicit solvation (SMD) must be dynamically triggered to accurately predict Solvatochromic shifts.

## 3. Conical Intersection Gating
**Target File:** `lumos_core/optimization.py`
**Required Architectural Change:**
- During analytical $S_1$ relaxations, LUMOS must employ strict root-following. If the $S_1 - S_0$ gap collapses below 0.2 eV, the code must instantly flag a Conical Intersection, halt the optimization, and ping CoChem-PULSE for non-adiabatic analysis.
